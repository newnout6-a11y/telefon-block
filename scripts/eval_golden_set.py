#!/usr/bin/env python3
"""PR-6: Golden-set evaluation harness.

Прогоняет модель (TFLite) через hold-out eval CSV и проверяет quality gates.
Если модель не проходит — exit code = 1, CI блокирует деплой.

Использование:
    python3 scripts/eval_golden_set.py \
        --model app/src/main/assets/spam_model.tflite \
        --card  app/src/main/assets/model_card.json \
        --golden datasets/ru/eval/cold_eval_600.csv \
        --min-block-precision 0.90 \
        --min-block-recall 0.60 \
        --max-allow-fp-rate 0.10

    # Или с бинарной моделью:
    python3 scripts/eval_golden_set.py \
        --model app/src/main/assets/experimental/spam_model_binary.tflite \
        --card  app/src/main/assets/experimental/model_card_binary.json \
        --golden datasets/ru/eval/cold_eval_600.csv

Exit codes:
    0 — все gates passed
    1 — хотя бы один gate failed
    2 — ошибка I/O / parse

Eval CSV формат (минимум):
    normalized_number, label   (label = ALLOW|WARN|BLOCK)
    или
    normalized_number, expected_label

Совместим с cold_eval_600.csv из datasets/ru/eval/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from ru_metadata_features import COMPACT_FEATURES, FIELD_TO_RU  # noqa: E402

LABEL_MAP = {'ALLOW': 0, 'WARN': 1, 'BLOCK': 2}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_card(card_path: str) -> Dict:
    with open(card_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def tflite_predict_batch(model_path: str, X: np.ndarray) -> np.ndarray:
    """Run TFLite on batch. Returns raw output (Nx1 or Nx3)."""
    # Prefer the modern ai_edge_litert package (TF 2.20+) which is API-stable.
    # Fall back to legacy tf.lite.Interpreter for older environments.
    try:
        from ai_edge_litert.interpreter import Interpreter as _Interp
        interp = _Interp(model_path=model_path)
    except ImportError:
        import tensorflow as tf
        try:
            interp = tf.lite.Interpreter(model_path=model_path)
        except TypeError:
            with open(model_path, 'rb') as fh:
                interp = tf.lite.Interpreter(model_content=fh.read())
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    out_shape = out_d['shape']
    out_size = int(out_shape[-1]) if len(out_shape) > 1 else 1
    results = np.zeros((len(X), out_size), dtype=np.float32)
    for i, row in enumerate(X.astype(np.float32)):
        interp.set_tensor(in_d['index'], row.reshape(1, -1))
        interp.invoke()
        results[i] = interp.get_tensor(out_d['index']).reshape(-1)[:out_size]
    return results


# ---------------------------------------------------------------------------
# Eval CSV loading
# ---------------------------------------------------------------------------

def load_eval_csv(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load eval CSV. Returns (X, y_true, numbers).

    Supports two schemas:
      1. Full features CSV (52 columns + label) — used directly.
      2. Minimal CSV (normalized_number + label/expected_label) — features
         are all zeros (cold-start simulation, model sees only prefix lookups).
         In this case caller should use --cold flag or ensure model handles it.
    """
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f'Empty eval CSV: {path}')

    headers = list(rows[0].keys())

    # Detect label column
    label_key = None
    for candidate in ('label', 'expected_label', 'метка'):
        if candidate in headers:
            label_key = candidate
            break
    if label_key is None:
        raise SystemExit(f'Eval CSV missing label column. Headers: {headers[:8]}')

    # Detect if full features are present
    has_features = all(name in headers or FIELD_TO_RU.get(name, name) in headers
                       for name in COMPACT_FEATURES)

    numbers: List[str] = []
    y_list: List[int] = []
    X_rows: List[List[float]] = []

    for row in rows:
        # Label
        lbl = str(row[label_key]).strip().upper()
        if lbl not in LABEL_MAP:
            continue  # skip unknown labels
        y_list.append(LABEL_MAP[lbl])

        # Number (for reporting)
        num = row.get('normalized_number', row.get('number', ''))
        numbers.append(str(num).strip())

        # Features
        if has_features:
            feat_row = []
            for name in COMPACT_FEATURES:
                col = name if name in headers else FIELD_TO_RU.get(name, name)
                feat_row.append(float(row.get(col, 0.0) or 0.0))
            X_rows.append(feat_row)
        else:
            # Cold-start: all features zero, noMetadata=1
            feat_row = [0.0] * len(COMPACT_FEATURES)
            no_meta_idx = COMPACT_FEATURES.index('noMetadata') if 'noMetadata' in COMPACT_FEATURES else -1
            if no_meta_idx >= 0:
                feat_row[no_meta_idx] = 1.0
            X_rows.append(feat_row)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, numbers


# ---------------------------------------------------------------------------
# Verdict assignment
# ---------------------------------------------------------------------------

def assign_verdicts(
    raw_output: np.ndarray, card: Dict, output_format: str,
) -> np.ndarray:
    """Map raw model output to verdict indices (0=ALLOW, 1=WARN, 2=BLOCK).

    Handles both 3-class softmax and binary sigmoid formats.
    """
    thr = card.get('thresholds', {})
    block_thr = float(thr.get('block_threshold', 0.5))
    warn_thr = float(thr.get('warn_threshold', 0.3))

    n = len(raw_output)
    verdicts = np.zeros(n, dtype=np.int64)

    if output_format == 'binary_sigmoid':
        # raw_output shape: (N, 1) — p_spam
        p_spam = raw_output[:, 0]
        verdicts[p_spam >= block_thr] = 2  # BLOCK
        verdicts[(p_spam >= warn_thr) & (p_spam < block_thr)] = 1  # WARN
    else:
        # 3-class softmax: (N, 3) — [allow, warn, block]
        block_p = raw_output[:, 2] if raw_output.shape[1] >= 3 else raw_output[:, 0]
        warn_p = raw_output[:, 1] if raw_output.shape[1] >= 3 else np.zeros(n)
        verdicts[block_p >= block_thr] = 2
        verdicts[(warn_p >= warn_thr) & (block_p < block_thr)] = 1

    return verdicts


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute binary-style metrics treating BLOCK+WARN as positive, ALLOW as negative."""
    # For gate purposes: "spam" = BLOCK (class 2), "safe" = ALLOW (class 0)
    # WARN in ground truth treated as spam (it's a reputation-flagged number).
    true_spam = (y_true >= 1)  # WARN or BLOCK = spam
    pred_spam = (y_pred >= 2)  # model says BLOCK (the hard-action threshold)
    pred_any_flag = (y_pred >= 1)  # model says WARN or BLOCK

    # BLOCK precision/recall (the thing that actually rejects calls)
    tp_block = int((pred_spam & true_spam).sum())
    fp_block = int((pred_spam & ~true_spam).sum())
    fn_block = int((~pred_spam & true_spam).sum())
    tn_block = int((~pred_spam & ~true_spam).sum())

    block_precision = tp_block / max(tp_block + fp_block, 1)
    block_recall = tp_block / max(tp_block + fn_block, 1)
    block_f1 = 2 * block_precision * block_recall / max(block_precision + block_recall, 1e-9)

    # ALLOW false-positive rate: fraction of true-ALLOW that model flags as spam
    true_allow = (y_true == 0)
    allow_flagged = int((pred_any_flag & true_allow).sum())
    allow_total = max(int(true_allow.sum()), 1)
    allow_fp_rate = allow_flagged / allow_total

    # Per-class accuracy
    accuracy = int((y_pred == y_true).sum()) / max(len(y_true), 1)

    return {
        'block_precision': float(block_precision),
        'block_recall': float(block_recall),
        'block_f1': float(block_f1),
        'allow_fp_rate': float(allow_fp_rate),
        'accuracy': float(accuracy),
        'tp_block': tp_block,
        'fp_block': fp_block,
        'fn_block': fn_block,
        'tn_block': tn_block,
        'total': int(len(y_true)),
        'true_spam_count': int(true_spam.sum()),
        'true_allow_count': int(true_allow.sum()),
    }


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def check_gates(metrics: Dict, args) -> List[str]:
    """Return list of failed gate descriptions. Empty = all pass."""
    failures = []
    if metrics['block_precision'] < args.min_block_precision:
        failures.append(
            f"BLOCK precision {metrics['block_precision']:.4f} < "
            f"min {args.min_block_precision:.2f}"
        )
    if metrics['block_recall'] < args.min_block_recall:
        failures.append(
            f"BLOCK recall {metrics['block_recall']:.4f} < "
            f"min {args.min_block_recall:.2f}"
        )
    if metrics['allow_fp_rate'] > args.max_allow_fp_rate:
        failures.append(
            f"ALLOW FP rate {metrics['allow_fp_rate']:.4f} > "
            f"max {args.max_allow_fp_rate:.2f}"
        )
    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description='PR-6: Golden-set eval gate')
    ap.add_argument('--model', required=True, help='Path to .tflite model')
    ap.add_argument('--card', required=True, help='Path to model_card.json')
    ap.add_argument('--golden', required=True, help='Path to eval CSV (golden set)')
    ap.add_argument('--min-block-precision', type=float, default=0.85,
                    help='Gate: minimum BLOCK precision (default 0.85)')
    ap.add_argument('--min-block-recall', type=float, default=0.55,
                    help='Gate: minimum BLOCK recall (default 0.55)')
    ap.add_argument('--max-allow-fp-rate', type=float, default=0.15,
                    help='Gate: max fraction of true-ALLOW flagged as spam (default 0.15)')
    ap.add_argument('--output-json', default=None,
                    help='Write detailed results to JSON file (optional)')
    ap.add_argument('--cold', action='store_true', default=False,
                    help='Force cold-start masking on features before inference '
                         '(zero out 9 metadata features, set noMetadata=1)')
    args = ap.parse_args()

    # Validate paths
    for path, name in [(args.model, 'model'), (args.card, 'card'), (args.golden, 'golden')]:
        if not os.path.isfile(path):
            print(f'ERROR: {name} file not found: {path}')
            return 2

    print(f'Golden-set eval')
    print(f'  model:  {args.model}')
    print(f'  card:   {args.card}')
    print(f'  golden: {args.golden}')

    # Load
    card = load_model_card(args.card)
    output_format = card.get('output_format', '3class_softmax')
    print(f'  output_format: {output_format}')

    X, y_true, numbers = load_eval_csv(args.golden)
    print(f'  eval rows: {len(y_true)} (ALLOW={int((y_true==0).sum())}, '
          f'WARN={int((y_true==1).sum())}, BLOCK={int((y_true==2).sum())})')

    # Cold-start masking (optional)
    if args.cold:
        from train_kd_distillation import COLD_START_MASK_FEATURES, feature_mask_indices, make_cold_view
        cold_idx = feature_mask_indices(list(COLD_START_MASK_FEATURES))
        no_meta_idx = COMPACT_FEATURES.index('noMetadata') if 'noMetadata' in COMPACT_FEATURES else -1
        X = make_cold_view(X, cold_idx, no_meta_idx)
        print(f'  [cold mode] zeroed 9 metadata features, forced noMetadata=1')

    # Predict
    print(f'  running inference...')
    raw_output = tflite_predict_batch(args.model, X)
    print(f'  output shape: {raw_output.shape}')

    # Assign verdicts
    y_pred = assign_verdicts(raw_output, card, output_format)

    # Metrics
    metrics = compute_metrics(y_true, y_pred)
    print(f'\n=== Metrics ===')
    print(f'  BLOCK precision: {metrics["block_precision"]:.4f}')
    print(f'  BLOCK recall:    {metrics["block_recall"]:.4f}')
    print(f'  BLOCK F1:        {metrics["block_f1"]:.4f}')
    print(f'  ALLOW FP rate:   {metrics["allow_fp_rate"]:.4f}')
    print(f'  Accuracy:        {metrics["accuracy"]:.4f}')
    print(f'  (TP={metrics["tp_block"]} FP={metrics["fp_block"]} '
          f'FN={metrics["fn_block"]} TN={metrics["tn_block"]})')

    # Gates
    failures = check_gates(metrics, args)
    print(f'\n=== Quality Gates ===')
    print(f'  min_block_precision: {args.min_block_precision:.2f}')
    print(f'  min_block_recall:    {args.min_block_recall:.2f}')
    print(f'  max_allow_fp_rate:   {args.max_allow_fp_rate:.2f}')

    if failures:
        print(f'\n  FAILED ({len(failures)} gate(s)):')
        for f in failures:
            print(f'    - {f}')
        status = 'FAIL'
    else:
        print(f'\n  ALL GATES PASSED')
        status = 'PASS'

    # Output JSON
    result = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'model': args.model,
        'card': args.card,
        'golden_set': args.golden,
        'cold_mode': args.cold,
        'output_format': output_format,
        'eval_rows': int(len(y_true)),
        'metrics': metrics,
        'gates': {
            'min_block_precision': args.min_block_precision,
            'min_block_recall': args.min_block_recall,
            'max_allow_fp_rate': args.max_allow_fp_rate,
        },
        'failures': failures,
        'status': status,
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f'\n  results written to: {args.output_json}')

    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
