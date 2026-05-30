"""
Phase 4A: patch model_card.json with cold-tuned BLOCK/WARN thresholds.

Идея: модель уже обучена и зашиплена в `app/src/main/assets/spam_model.tflite`.
Веса не меняем. Но текущий `model_card.json` содержит ОДИН набор порогов,
тюненный на «тёплой» val (со всеми metadata-фичами). На cold-start (нет
интернета, репутация/whitelist/blacklist недоступны) этот порог режет BLOCK
recall в ~3x. Patch‑скрипт:

1. Загружает существующую TFLite-модель + dataset + model_card.json.
2. Восстанавливает val/test split с тем же seed/val_frac, что в kd-train.
3. Считает proba на cold-view val (`metadata=0, noMetadata=1`).
4. Тюнит отдельные `block_threshold` / `warn_threshold` под precision-floor.
5. Дописывает в model_card.json блок `cold_thresholds`.
6. Печатает before/after метрики на cold-view test (warm-on-cold vs cold-on-cold).

Всё это БЕЗ ретрейна сети — только пороги. Нужен `tflite_runtime` (или
`tensorflow`) и `numpy`. Запуск:

  python scripts/patch_cold_thresholds.py
  python scripts/patch_cold_thresholds.py --min-cold-block-precision 0.85

Результат: `app/src/main/assets/model_card.json` получает поле
`cold_thresholds`, которое Android (`SpamModel.kt`) выбирает на устройстве,
когда `noMetadata=1 AND inAllowlist=0 AND inBlacklist=0`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from ru_metadata_features import COMPACT_FEATURES, ID_TO_LABEL  # noqa: E402

# Не вызываем tensorflow на верхнем уровне train_kd_distillation: ни load_csv,
# ни stratified_split, ни tune_thresholds, ни make_cold_view, ни
# feature_mask_indices не требуют tf.
from train_kd_distillation import (  # noqa: E402
    COLD_START_MASK_FEATURES,
    feature_mask_indices,
    load_csv,
    make_cold_view,
    stratified_split,
    tune_thresholds,
)

NUM_CLASSES = 3
DEFAULT_DATA = os.path.join(THIS_DIR, '..', 'datasets', 'ru', 'processed', 'ru_tflite_features.csv')
DEFAULT_TFLITE = os.path.join(THIS_DIR, '..', 'app', 'src', 'main', 'assets', 'spam_model.tflite')
DEFAULT_MODEL_CARD = os.path.join(THIS_DIR, '..', 'app', 'src', 'main', 'assets', 'model_card.json')


def _load_interpreter(tflite_path: str):
    """Prefer the lightweight tflite_runtime, fall back to full tensorflow.lite.

    Both expose the same Interpreter API (allocate_tensors, set_tensor, invoke,
    get_tensor). Patcher works in both environments.
    """
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        return Interpreter(model_path=tflite_path)
    except Exception:
        import tensorflow as tf  # type: ignore
        return tf.lite.Interpreter(model_path=tflite_path)


def tflite_predict(tflite_path: str, X: np.ndarray) -> np.ndarray:
    """Run TFLite inference row-by-row and return [N, 3] probabilities.

    Note: spam_model.tflite is exported with a `tf.nn.softmax` head
    (see build_export_model / _make_serving_fn in train_kd_distillation.py),
    so output is already a probability vector summing to 1. Do NOT softmax
    again — that would compress everything toward uniform.
    """
    interp = _load_interpreter(tflite_path)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    out = np.zeros((len(X), NUM_CLASSES), dtype=np.float32)
    for i, row in enumerate(X.astype(np.float32)):
        interp.set_tensor(in_d['index'], row.reshape(1, -1))
        interp.invoke()
        out[i] = interp.get_tensor(out_d['index']).reshape(-1)
    return out


def _per_class_prf(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Bare-bones precision/recall/F1 per class (no sklearn dependency)."""
    out: Dict[str, Dict[str, float]] = {}
    for cls in range(NUM_CLASSES):
        tp = int(np.sum((y_pred == cls) & (y_true == cls)))
        fp = int(np.sum((y_pred == cls) & (y_true != cls)))
        fn = int(np.sum((y_pred != cls) & (y_true == cls)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        out[ID_TO_LABEL[cls]] = {'precision': float(prec), 'recall': float(rec), 'f1': float(f1)}
    return out


def evaluate(y_true: np.ndarray, proba: np.ndarray, thresholds: Optional[Dict] = None) -> Dict:
    if thresholds:
        bt = float(thresholds.get('block_threshold', 0.5))
        wt = float(thresholds.get('warn_threshold', 0.3))
        pred = np.zeros(len(y_true), dtype=np.int64)
        block_mask = proba[:, 2] >= bt
        warn_mask = (~block_mask) & (proba[:, 1] >= wt)
        pred[block_mask] = 2
        pred[warn_mask] = 1
    else:
        pred = np.argmax(proba, axis=1).astype(np.int64)
    cls_metrics = _per_class_prf(y_true, pred)
    macro = float(np.mean([cls_metrics[ID_TO_LABEL[c]]['f1'] for c in range(NUM_CLASSES)]))
    return {
        **cls_metrics,
        'macro_f1': macro,
    }


def operating_points(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, Dict[str, float]]:
    """For each precision floor, find best (threshold, recall) for BLOCK class."""
    pts: Dict[str, Dict[str, float]] = {}
    for floor in (0.95, 0.90, 0.80):
        best_t, best_p, best_r = 1.0, 0.0, 0.0
        for t in np.linspace(0.10, 0.99, 90):
            pb = proba[:, 2] >= t
            tp = int(np.sum(pb & (y_true == 2)))
            fp = int(np.sum(pb & (y_true != 2)))
            fn = int(np.sum(~pb & (y_true == 2)))
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            if p >= floor and r > best_r:
                best_t, best_p, best_r = float(t), float(p), float(r)
        pts[f'P>={floor}'] = {'threshold': best_t, 'precision': best_p, 'recall': best_r}
    return pts


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=DEFAULT_DATA, help='Path to ru_tflite_features.csv')
    ap.add_argument('--tflite', default=DEFAULT_TFLITE, help='Path to spam_model.tflite')
    ap.add_argument('--model-card', default=DEFAULT_MODEL_CARD,
                    help='Path to model_card.json (will be modified in-place)')
    ap.add_argument('--seed', type=int, default=42, help='Same seed used by kd-train (for split parity).')
    ap.add_argument('--val-frac', type=float, default=0.10)
    ap.add_argument('--test-frac', type=float, default=0.10)
    ap.add_argument('--min-cold-block-precision', type=float, default=0.85,
                    help='Floor BLOCK precision when tuning cold thresholds.')
    ap.add_argument('--also-retune-warm', action='store_true',
                    help='\u0422\u0430\u043a\u0436\u0435 \u043f\u0435\u0440\u0435\u0442\u044e\u043d\u0438\u0442\u044c warm thresholds (\u0438\u043d\u0430\u0447\u0435 \u043e\u0441\u0442\u0430\u044e\u0442\u0441\u044f \u0438\u0437 \u043a\u0430\u0440\u0442\u043e\u0447\u043a\u0438). '
                         '\u041f\u043e \u0443\u043c\u043e\u043b\u0447\u0430\u043d\u0438\u044e \u041d\u0415 \u0442\u0440\u043e\u0433\u0430\u0435\u043c warm \u2014 \u0447\u0442\u043e\u0431\u044b \u043d\u0435 \u0440\u0438\u0441\u043a\u043e\u0432\u0430\u0442\u044c \u043d\u0438 \u043e\u0434\u043d\u0438\u043c \u0431\u0430\u0439\u0442\u043e\u043c warm-\u043f\u0440\u043e\u0438\u0437\u0432\u043e\u0434\u0438\u0442\u0435\u043b\u044c\u043d\u043e\u0441\u0442\u0438.')
    ap.add_argument('--dry-run', action='store_true', help='\u041d\u0435 \u043f\u0438\u0441\u0430\u0442\u044c model_card.json, \u0442\u043e\u043b\u044c\u043a\u043e \u043f\u0435\u0447\u0430\u0442\u0430\u0442\u044c.')
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    print(f'[patch] data:        {args.data}')
    print(f'[patch] tflite:      {args.tflite}')
    print(f'[patch] model_card:  {args.model_card}')

    if not os.path.exists(args.data):
        print(f'ERROR: dataset not found: {args.data}', file=sys.stderr)
        return 2
    if not os.path.exists(args.tflite):
        print(f'ERROR: tflite not found: {args.tflite}', file=sys.stderr)
        return 2
    if not os.path.exists(args.model_card):
        print(f'ERROR: model_card not found: {args.model_card}', file=sys.stderr)
        return 2

    print('\n[1/5] Loading CSV ...')
    X, y = load_csv(args.data)
    print(f'  rows={len(y)}, features={X.shape[1]}')
    if X.shape[1] != len(COMPACT_FEATURES):
        print(
            f'ERROR: feature count mismatch (csv={X.shape[1]} vs '
            f'COMPACT_FEATURES={len(COMPACT_FEATURES)})',
            file=sys.stderr,
        )
        return 2

    print('\n[2/5] Reproducing train/val/test split ...')
    train_size = 1.0 - args.val_frac - args.test_frac
    if train_size <= 0:
        print('ERROR: val_frac + test_frac must be < 1.0', file=sys.stderr)
        return 2
    train_idx, val_idx, test_idx = stratified_split(
        X, y, sizes=(train_size, args.val_frac, args.test_frac), seed=args.seed,
    )
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f'  train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}')

    print('\n[3/5] Building cold-view (val + test) ...')
    cold_mask_idx = feature_mask_indices(list(COLD_START_MASK_FEATURES))
    no_meta_idx = (
        COMPACT_FEATURES.index('noMetadata') if 'noMetadata' in COMPACT_FEATURES else -1
    )
    print(f'  mask_features={list(COLD_START_MASK_FEATURES)}')
    print(f'  noMetadata_idx={no_meta_idx}')
    X_val_cold = make_cold_view(X_val, cold_mask_idx, no_meta_idx)
    X_test_cold = make_cold_view(X_test, cold_mask_idx, no_meta_idx)

    print('\n[4/5] Running TFLite inference (val cold-view + test cold-view) ...')
    proba_val_cold = tflite_predict(args.tflite, X_val_cold)
    proba_test_cold = tflite_predict(args.tflite, X_test_cold)
    proba_test_warm = tflite_predict(args.tflite, X_test)
    proba_val_warm = tflite_predict(args.tflite, X_val)

    print('\n[5/5] Tuning thresholds ...')
    cold_thr = tune_thresholds(
        y_val, proba_val_cold, min_block_precision=args.min_cold_block_precision,
    )
    print(
        f'  cold thresholds (val cold-view): '
        f'block={cold_thr["block_threshold"]:.3f} warn={cold_thr["warn_threshold"]:.3f} '
        f'(BLOCK_P={cold_thr["block_precision"]:.3f}, F1={cold_thr["block_f1"]:.3f})'
    )

    # Load existing card to read warm thresholds (so we can show before/after).
    with open(args.model_card, 'r', encoding='utf-8') as f:
        card: Dict[str, Any] = json.load(f)
    warm_thr_card: Optional[Dict] = card.get('thresholds')
    if not warm_thr_card:
        print(
            '  WARN: model_card has no `thresholds` block \u2014 cannot compute warm-on-cold baseline.',
            file=sys.stderr,
        )

    if args.also_retune_warm:
        warm_thr_new = tune_thresholds(
            y_val, proba_val_warm, min_block_precision=args.min_cold_block_precision,
        )
        print(
            f'  warm thresholds re-tuned: '
            f'block={warm_thr_new["block_threshold"]:.3f} '
            f'warn={warm_thr_new["warn_threshold"]:.3f}'
        )
    else:
        warm_thr_new = None

    # === Reporting ===
    print('\n--- Cold-start eval slice (test, metadata zeroed, noMetadata=1) ---')
    metrics_argmax = evaluate(y_test, proba_test_cold)
    metrics_warm_on_cold = (
        evaluate(y_test, proba_test_cold, thresholds=warm_thr_card)
        if warm_thr_card else None
    )
    metrics_cold_on_cold = evaluate(y_test, proba_test_cold, thresholds=cold_thr)
    metrics_warm_on_warm = evaluate(y_test, proba_test_warm, thresholds=warm_thr_card or {})

    def _row(label: str, m: Optional[Dict]) -> None:
        if m is None:
            return
        block = m.get('BLOCK', {})
        warn = m.get('WARN', {})
        print(
            f'  {label:<32s} macroF1={m["macro_f1"]:.4f} '
            f'BLOCK P={block.get("precision", 0):.3f} '
            f'R={block.get("recall", 0):.3f} '
            f'F1={block.get("f1", 0):.3f} '
            f'WARN F1={warn.get("f1", 0):.3f}'
        )

    _row('argmax (no thresholds)', metrics_argmax)
    _row('warm thresholds on cold', metrics_warm_on_cold)
    _row('COLD thresholds on cold', metrics_cold_on_cold)
    _row('warm thresholds on warm', metrics_warm_on_warm)

    print('\n--- Cold-start BLOCK operating points (precision floor \u2192 recall) ---')
    op = operating_points(y_test, proba_test_cold)
    for k, v in op.items():
        print(f'  {k:<10s} t={v["threshold"]:.2f} R={v["recall"]:.3f} (P={v["precision"]:.3f})')

    if args.dry_run:
        print('\n[dry-run] model_card.json NOT modified.')
        return 0

    # === Patch model_card.json ===
    card['cold_thresholds'] = {
        'block_threshold': float(cold_thr['block_threshold']),
        'warn_threshold': float(cold_thr['warn_threshold']),
        'block_precision': float(cold_thr['block_precision']),
        'block_recall': float(cold_thr['block_recall']),
        'block_f1': float(cold_thr['block_f1']),
        'warn_f1': float(cold_thr['warn_f1']),
        'tuning_info': {
            'mask_features': list(COLD_START_MASK_FEATURES),
            'no_meta_set_to_1': no_meta_idx >= 0,
            'min_cold_block_precision': float(args.min_cold_block_precision),
            'val_rows': int(len(y_val)),
            'patched_at': datetime.utcnow().isoformat() + 'Z',
            'patched_by': 'scripts/patch_cold_thresholds.py',
        },
        'eval_metrics_on_cold_test': {
            'macro_f1': float(metrics_cold_on_cold['macro_f1']),
            'BLOCK': metrics_cold_on_cold['BLOCK'],
            'WARN': metrics_cold_on_cold['WARN'],
            'eval_rows': int(len(y_test)),
        },
    }
    if warm_thr_new is not None:
        card['thresholds'] = {
            'block_threshold': float(warm_thr_new['block_threshold']),
            'warn_threshold': float(warm_thr_new['warn_threshold']),
            'block_precision': float(warm_thr_new['block_precision']),
            'block_recall': float(warm_thr_new['block_recall']),
            'block_f1': float(warm_thr_new['block_f1']),
            'warn_f1': float(warm_thr_new['warn_f1']),
        }
    if 'cold_start' in card and isinstance(card['cold_start'], dict):
        card['cold_start']['tflite_metrics_cold_thresholded'] = {
            'macro_f1': float(metrics_cold_on_cold['macro_f1']),
            'BLOCK': metrics_cold_on_cold['BLOCK'],
            'WARN': metrics_cold_on_cold['WARN'],
            'ALLOW': metrics_cold_on_cold['ALLOW'],
        }

    notes = card.get('notes') or ''
    if 'Phase 4A' not in notes:
        card['notes'] = (notes + ' | Phase 4A: cold thresholds patched in by scripts/patch_cold_thresholds.py').strip(' |')

    with open(args.model_card, 'w', encoding='utf-8') as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    print(f'\n[done] model_card.json patched: {args.model_card}')
    print(f'  cold_thresholds.block_threshold = {cold_thr["block_threshold"]:.3f}')
    print(f'  cold_thresholds.warn_threshold  = {cold_thr["warn_threshold"]:.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
