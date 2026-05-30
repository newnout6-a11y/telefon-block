"""PR-2: Binary spam classifier + Platt calibration.

Альтернатива `scripts/train_kd_distillation.py`. Решает три проблемы
3-class пайплайна:

  1. WARN F1 = 0 в warm режиме / 0.30 в cold — класс не работает.
  2. Class imbalance ALLOW:WARN:BLOCK ≈ 1:0.8:7.7 → требует костылей
     с class_weights и threshold = 0.10.
  3. Probability shape кривая, отсюда ALLOW precision = 0.42 на cold
     thresholded eval (массовые false-blocks).

Что делает этот скрипт:

  * Объединяет WARN + BLOCK в class 1 (spam=1), ALLOW = 0. Это «честная»
    бинарка: всё, что подозрительно или фрод, считается spam, и
    пользователь увидит этот разный уровень подозрения через два
    порога над одной вероятностью (warn_threshold, block_threshold).
    Выбор стратегии WARN — флаг `--binary-warn-strategy`:
       merge_block (default)     → WARN добавляется к BLOCK (=1).
       merge_allow               → WARN добавляется к ALLOW (=0).
       drop                      → WARN-строки выбрасываются.
  * Обучает простой MLP с binary cross-entropy + class-balanced sampling.
  * Делает Platt-калибровку (sigmoid(a*z + b)) на hold-out части val.
    a/b встраиваются в экспортируемую модель — TFLite возвращает уже
    калиброванную вероятность.
  * Экспортирует TFLite с одним sigmoid-выходом [1, 1] (формат
    `binary_sigmoid`).
  * Пишет model_card с полями:
       output_format: "binary_sigmoid"
       calibration: { method: "platt", a, b, applied_in_model: true,
                      ece_before, ece_after, brier_before, brier_after }
       thresholds: { block_threshold, warn_threshold } — в шкале
       одной p_spam ∈ [0,1].

  * По умолчанию пишет в `app/src/main/assets/experimental/`, прод
    (`spam_model.tflite`) не подменяется.

Vendor-совместимость:

  * Android-сторона (PR-2) читает `output_format` и работает в обоих
    режимах. Если поля нет в карте — дефолт `3class_softmax` для
    обратной совместимости с production.

Этот скрипт намеренно проще `train_kd_distillation.py`: нет KD,
ансамбля, Optuna. Цель — корректно поставленная бинарная задача с
калибровкой; гипероптимизация при необходимости делается отдельно.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from ru_metadata_features import COMPACT_FEATURES, FIELD_TO_RU  # noqa: E402

LABEL_TO_ID = {'ALLOW': 0, 'WARN': 1, 'BLOCK': 2}

BASE_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, '..'))
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
PROCESSED_DIR = os.path.join(DATASETS_DIR, 'ru', 'processed')
DEFAULT_DATA = os.path.join(PROCESSED_DIR, 'ru_tflite_features.csv')
ASSETS_DIR = os.path.join(BASE_DIR, 'app', 'src', 'main', 'assets')
EXPERIMENTAL_DIR = os.path.join(ASSETS_DIR, 'experimental')
DEFAULT_TFLITE = os.path.join(EXPERIMENTAL_DIR, 'spam_model_binary.tflite')
DEFAULT_MODEL_CARD = os.path.join(EXPERIMENTAL_DIR, 'model_card_binary.json')


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load CSV with COMPACT_FEATURES + label. Принимает и en/ru заголовки."""
    if path.endswith('.npz'):
        data = np.load(path)
        return np.asarray(data['X'], dtype=np.float32), np.asarray(data['y'], dtype=np.int64)

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f'No rows in {path}')

    headers = list(rows[0].keys())
    has_english = all(name in headers for name in COMPACT_FEATURES)
    if has_english:
        feat_keys = {name: name for name in COMPACT_FEATURES}
        label_key = 'label' if 'label' in headers else 'метка'
    else:
        feat_keys = {name: FIELD_TO_RU.get(name, name) for name in COMPACT_FEATURES}
        label_key = FIELD_TO_RU.get('label', 'метка')

    missing = [n for n, k in feat_keys.items() if k not in headers]
    if missing:
        raise SystemExit(f'CSV missing features: {missing[:5]}{"..." if len(missing) > 5 else ""}')
    if label_key not in headers:
        raise SystemExit(f"CSV missing label column ('{label_key}')")

    X = np.array(
        [[float(r[feat_keys[n]]) for n in COMPACT_FEATURES] for r in rows],
        dtype=np.float32,
    )
    y_list: List[int] = []
    for r in rows:
        v = str(r[label_key]).strip()
        if v in LABEL_TO_ID:
            y_list.append(LABEL_TO_ID[v])
        else:
            y_list.append(int(float(v)))
    return X, np.array(y_list, dtype=np.int64)


# ---------------------------------------------------------------------------
# Splits / labels
# ---------------------------------------------------------------------------

def stratified_split(
    y: np.ndarray, sizes: Tuple[float, float, float], seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split into train/val/test by class label."""
    assert abs(sum(sizes) - 1.0) < 1e-6, f'sizes must sum to 1.0, got {sizes}'
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * sizes[0]))
        n_val = int(round(n * sizes[1]))
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def to_binary_labels(y: np.ndarray, warn_strategy: str) -> Tuple[np.ndarray, np.ndarray]:
    """Convert {ALLOW, WARN, BLOCK} → {0, 1}. Returns (y_bin, keep_mask).

    keep_mask пометит строки, которые нужно отбросить (warn_strategy='drop'
    выбрасывает WARN).
    """
    keep = np.ones_like(y, dtype=bool)
    y_bin = np.zeros_like(y, dtype=np.int64)

    block_mask = y == LABEL_TO_ID['BLOCK']
    warn_mask = y == LABEL_TO_ID['WARN']

    y_bin[block_mask] = 1
    if warn_strategy == 'merge_block':
        y_bin[warn_mask] = 1
    elif warn_strategy == 'merge_allow':
        y_bin[warn_mask] = 0
    elif warn_strategy == 'drop':
        keep[warn_mask] = False
    else:
        raise ValueError(f'Unknown warn_strategy: {warn_strategy!r}')
    return y_bin, keep


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_binary_mlp(
    n_features: int, hidden_sizes: Sequence[int], dropout: float, l2: float,
):
    import tensorflow as tf

    reg = tf.keras.regularizers.l2(l2) if l2 > 0 else None
    inp = tf.keras.Input(shape=(n_features,), name='features')
    x = inp
    for h in hidden_sizes:
        x = tf.keras.layers.Dense(h, activation='relu', kernel_regularizer=reg)(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
    # Single logit (uncalibrated). Sigmoid + Platt применяются в build_export_model.
    logit = tf.keras.layers.Dense(1, name='spam_logit')(x)
    return tf.keras.Model(inp, logit, name='spam_binary_backbone')


def build_export_model(backbone, platt_a: float, platt_b: float):
    """Backbone → calibrated probability. P = sigmoid(a * logit + b)."""
    import tensorflow as tf
    inp = tf.keras.Input(shape=(len(COMPACT_FEATURES),), name='features')
    logit = backbone(inp, training=False)
    # Platt: sigmoid(a*z + b). a и b замораживаются как константы внутри графа.
    a = tf.constant(float(platt_a), dtype=tf.float32, name='platt_a')
    b = tf.constant(float(platt_b), dtype=tf.float32, name='platt_b')
    calibrated = tf.keras.layers.Activation(
        'sigmoid', name='p_spam'
    )(a * logit + b)
    return tf.keras.Model(inp, calibrated, name='spam_binary_export')


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def class_balanced_weights(y_bin: np.ndarray) -> Dict[int, float]:
    """Inverse-frequency class weights for binary y. {0: w0, 1: w1}, mean ~ 1."""
    n0 = max(int((y_bin == 0).sum()), 1)
    n1 = max(int((y_bin == 1).sum()), 1)
    total = n0 + n1
    w0 = total / (2.0 * n0)
    w1 = total / (2.0 * n1)
    return {0: float(w0), 1: float(w1)}


def fit_platt(logits_val: np.ndarray, y_val_bin: np.ndarray, max_iter: int = 200) -> Tuple[float, float]:
    """Platt scaling: solve P = sigmoid(a*z + b) on (logits, labels) via lbfgs.

    Реализация без scipy: используем gradient descent по NLL. Для
    несбалансированных классов даём label smoothing 1/(N_+ + 2) / 1/(N_- + 2)
    (Niculescu-Mizil 2005), чтобы избежать overfit'а на маленьком hold-out.
    """
    z = logits_val.astype(np.float64)
    y = y_val_bin.astype(np.float64)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        # Hold-out с одним классом — калибровка тривиальна, возвращаем identity-mapping.
        return 1.0, 0.0
    # Smoothed targets per Platt's original paper.
    t_pos = (n_pos + 1.0) / (n_pos + 2.0)
    t_neg = 1.0 / (n_neg + 2.0)
    t = np.where(y == 1, t_pos, t_neg)

    a, b = 1.0, 0.0
    lr = 0.05
    for _ in range(max_iter):
        u = a * z + b
        p = 1.0 / (1.0 + np.exp(-u))
        # NLL gradient w.r.t. a, b
        diff = p - t
        ga = float(np.mean(diff * z))
        gb = float(np.mean(diff))
        a -= lr * ga
        b -= lr * gb
        # adaptive: cool down if overshoot
        if abs(ga) < 1e-7 and abs(gb) < 1e-7:
            break
    return float(a), float(b)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Standard ECE: weighted absolute gap between confidence and accuracy."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(len(p), 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not mask.any():
            continue
        bin_conf = float(p[mask].mean())
        bin_acc = float(y[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def evaluate_binary(p: np.ndarray, y: np.ndarray, threshold: float) -> Dict:
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    spec = tn / max(tn + fp, 1)
    return {
        'threshold': float(threshold),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': float(prec),
        'recall': float(rec),
        'specificity': float(spec),
        'f1': float(f1),
        'accuracy': float((tp + tn) / max(tp + fp + fn + tn, 1)),
    }


def tune_thresholds_binary(
    p: np.ndarray, y: np.ndarray, *, min_block_precision: float,
) -> Tuple[float, float]:
    """Two thresholds: block_threshold (high precision) и warn_threshold < block.

    block_threshold maximizes BLOCK F1 with precision floor.
    warn_threshold выбирается так, чтобы BLOCK recall на интервале
    [warn, block] восстановился хотя бы до 95% от BLOCK recall на
    block_threshold (это «зона неопределённости» — то, что мы готовы
    показать пользователю как WARN).
    """
    best_block = 0.5
    best_f1 = 0.0
    best_recall_at_block = 0.0
    for bt in np.linspace(0.10, 0.95, 86):
        m = evaluate_binary(p, y, float(bt))
        if m['precision'] >= min_block_precision and m['f1'] > best_f1:
            best_f1 = m['f1']
            best_block = float(bt)
            best_recall_at_block = m['recall']

    # warn_threshold < block_threshold: чтобы расширить «подозрительную» зону.
    # Берём минимальный wt, при котором precision на интервале [wt, block] ≥ 0.5
    # (компромисс: WARN не должен быть тотально ALLOW-смещённым).
    best_warn = max(0.10, best_block - 0.20)
    for wt in np.linspace(0.10, best_block - 0.05, 30):
        mask_warn = (p >= float(wt)) & (p < best_block)
        if mask_warn.sum() == 0:
            continue
        warn_precision = float(y[mask_warn].mean())
        if warn_precision >= 0.5:
            best_warn = float(wt)
            break
    return best_block, float(min(best_warn, best_block - 0.01))


# ---------------------------------------------------------------------------
# TFLite export
# ---------------------------------------------------------------------------

def export_tflite(export_model, out_path: str) -> Dict:
    import tensorflow as tf
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = len(COMPACT_FEATURES)

    @tf.function(input_signature=[tf.TensorSpec([1, n], tf.float32, name='features')])
    def serving_fn(x):
        return export_model(x, training=False)

    converter = tf.lite.TFLiteConverter.from_concrete_functions([serving_fn.get_concrete_function()])
    tflite_bytes = converter.convert()
    with open(out_path, 'wb') as f:
        f.write(tflite_bytes)
    return {'path': out_path, 'bytes': len(tflite_bytes)}


def tflite_predict(tflite_path: str, X: np.ndarray) -> np.ndarray:
    import tensorflow as tf
    try:
        interp = tf.lite.Interpreter(model_path=tflite_path)
    except TypeError:
        # TF 2.21 broke the model_path ctor; fall back to model_content.
        with open(tflite_path, 'rb') as fh:
            interp = tf.lite.Interpreter(model_content=fh.read())
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    out = np.zeros(len(X), dtype=np.float32)
    for i, row in enumerate(X.astype(np.float32)):
        interp.set_tensor(in_d['index'], row.reshape(1, -1))
        interp.invoke()
        out[i] = float(interp.get_tensor(out_d['index']).reshape(-1)[0])
    return out


def sanity_check(export_model, tflite_path: str, X: np.ndarray, atol: float = 1e-4) -> Dict:
    n = min(200, len(X))
    sample = X[:n].astype(np.float32)
    keras_p = export_model.predict(sample, verbose=0).reshape(-1)
    tflite_p = tflite_predict(tflite_path, sample)
    diff = float(np.max(np.abs(keras_p - tflite_p)))
    return {'samples': int(n), 'max_abs_diff': diff, 'pass': bool(diff < atol)}


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

def write_card(
    out_path: str, *, version: str, dataset_path: str, dataset_hash: str,
    feature_count: int, rows: int, class_counts_orig: Dict[str, int],
    binary_class_counts: Dict[str, int], warn_strategy: str,
    platt_a: float, platt_b: float,
    calib_metrics: Dict, thresholds: Tuple[float, float],
    test_metrics: Dict, hidden_sizes: Sequence[int],
) -> None:
    block_thr, warn_thr = thresholds
    card = {
        'version': version,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'feature_count': feature_count,
        'features': COMPACT_FEATURES,
        'rows': int(rows),
        'class_counts': class_counts_orig,
        'binary_class_counts': binary_class_counts,
        'dataset': dataset_path,
        'dataset_hash': dataset_hash,
        # === Output format ===
        # Android-сторона (PR-2 SpamModel.kt) читает это поле и переключает
        # парсинг выхода: 1 sigmoid (binary) vs 3 softmax (legacy 3-class).
        'output_format': 'binary_sigmoid',
        'output_meaning': 'p_spam in [0,1]; verdict via two thresholds',
        # === Calibration ===
        # Platt scaling уже встроена в граф TFLite (sigmoid(a*z+b)),
        # поэтому интерпретатор сразу возвращает калиброванную вероятность.
        # ECE/Brier до/после калибровки — для аудита.
        'calibration': {
            'method': 'platt',
            'a': float(platt_a),
            'b': float(platt_b),
            'applied_in_model': True,
            'ece_before': float(calib_metrics.get('ece_before', 0.0)),
            'ece_after': float(calib_metrics.get('ece_after', 0.0)),
            'brier_before': float(calib_metrics.get('brier_before', 0.0)),
            'brier_after': float(calib_metrics.get('brier_after', 0.0)),
        },
        'binary_target': {
            'warn_strategy': warn_strategy,
            'spam_class_includes': (
                ['BLOCK', 'WARN'] if warn_strategy == 'merge_block'
                else (['BLOCK'] if warn_strategy == 'drop'
                      else ['BLOCK'])
            ),
        },
        # Пороги — над одной шкалой p_spam ∈ [0, 1].
        'thresholds': {
            'block_threshold': float(block_thr),
            'warn_threshold': float(warn_thr),
            'block_precision': float(test_metrics.get('precision', 0.0)),
            'block_recall': float(test_metrics.get('recall', 0.0)),
            'block_f1': float(test_metrics.get('f1', 0.0)),
        },
        'test_metrics': test_metrics,
        'training_config': {
            'hidden_sizes': list(hidden_sizes),
        },
        'notes': (
            'Generated by scripts/train_binary_model.py (PR-2). Single-output '
            'binary spam classifier with Platt calibration baked into the '
            'exported TFLite graph. WARN is no longer a separate class — '
            'the warn_threshold/block_threshold pair defines a calibrated '
            'uncertainty zone over p_spam.'
        ),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(card, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def class_counts_str(y: np.ndarray) -> Dict[str, int]:
    return {
        'ALLOW': int((y == LABEL_TO_ID['ALLOW']).sum()),
        'WARN': int((y == LABEL_TO_ID['WARN']).sum()),
        'BLOCK': int((y == LABEL_TO_ID['BLOCK']).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='PR-2: Binary spam classifier + Platt calibration')
    ap.add_argument('--data', default=DEFAULT_DATA)
    ap.add_argument('--tflite-output', default=DEFAULT_TFLITE)
    ap.add_argument('--model-card-output', default=DEFAULT_MODEL_CARD)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val-frac', type=float, default=0.10)
    ap.add_argument('--test-frac', type=float, default=0.10)
    ap.add_argument('--calib-frac', type=float, default=0.50,
                    help='Доля val, идущая на Platt калибровку (а не на early-stopping). '
                         '0.5 — компромисс.')
    ap.add_argument('--binary-warn-strategy', type=str, default='merge_block',
                    choices=['merge_block', 'merge_allow', 'drop'],
                    help='Что делать с WARN-классом при бинаризации. По умолчанию '
                         'WARN сливается в spam=1 (т.к. WARN-эвристика срабатывает на '
                         'reputation/категории — те же сигналы, что и BLOCK).')
    ap.add_argument('--hidden-sizes', type=str, default='96,64,32')
    ap.add_argument('--dropout', type=float, default=0.10)
    ap.add_argument('--l2', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--min-block-precision', type=float, default=0.85)
    ap.add_argument('--sanity-atol', type=float, default=1e-4)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(args.seed)
        tf.keras.utils.set_random_seed(args.seed)
    except ImportError:
        raise SystemExit('tensorflow is required for training.')

    print(f'Loading {args.data}...')
    X, y = load_csv(args.data)
    if X.shape[1] != len(COMPACT_FEATURES):
        raise SystemExit(f'Feature mismatch: {X.shape[1]} vs {len(COMPACT_FEATURES)}')
    orig_counts = class_counts_str(y)
    print(f'  rows={len(y)}, class counts: {orig_counts}')

    # --- Split ---
    train_size = 1.0 - args.val_frac - args.test_frac
    if train_size <= 0:
        raise SystemExit('val_frac + test_frac must be < 1.0')
    train_idx, val_idx, test_idx = stratified_split(
        y, sizes=(train_size, args.val_frac, args.test_frac), seed=args.seed,
    )
    print(f'  split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

    # --- Binary labels ---
    y_bin, keep = to_binary_labels(y, args.binary_warn_strategy)
    train_idx = np.array([i for i in train_idx if keep[i]])
    val_idx = np.array([i for i in val_idx if keep[i]])
    test_idx = np.array([i for i in test_idx if keep[i]])
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise SystemExit('Empty split after WARN handling — check --binary-warn-strategy')

    X_train, y_train = X[train_idx], y_bin[train_idx]
    X_val, y_val = X[val_idx], y_bin[val_idx]
    X_test, y_test = X[test_idx], y_bin[test_idx]

    bin_counts = {
        'allow': int((np.concatenate([y_train, y_val, y_test]) == 0).sum()),
        'spam':  int((np.concatenate([y_train, y_val, y_test]) == 1).sum()),
    }
    print(f'  binary (warn_strategy={args.binary_warn_strategy}): {bin_counts}')

    # Calibration hold-out: разрезаем val на (val_train, val_calib).
    rng = np.random.default_rng(args.seed + 1)
    val_perm = rng.permutation(len(X_val))
    n_calib = int(round(len(X_val) * args.calib_frac))
    calib_idx = val_perm[:n_calib]
    early_idx = val_perm[n_calib:]
    X_val_early, y_val_early = X_val[early_idx], y_val[early_idx]
    X_val_calib, y_val_calib = X_val[calib_idx], y_val[calib_idx]
    print(f'  val split: early-stop={len(X_val_early)}, calibration={len(X_val_calib)}')

    # --- Model ---
    hidden_sizes = [int(s) for s in args.hidden_sizes.split(',') if s.strip()]
    print(f'  building MLP, hidden_sizes={hidden_sizes}, dropout={args.dropout}, l2={args.l2}')
    backbone = build_binary_mlp(
        n_features=X.shape[1], hidden_sizes=hidden_sizes,
        dropout=args.dropout, l2=args.l2,
    )

    cls_w = class_balanced_weights(y_train)
    print(f'  class weights (binary): {cls_w}')

    # Compile with from_logits BCE — последний слой backbone это logit.
    backbone.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)],  # sign-based accuracy
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=args.patience, restore_best_weights=True,
    )
    print(f'  training {args.epochs} epochs (patience={args.patience})...')
    backbone.fit(
        X_train, y_train.astype(np.float32),
        validation_data=(X_val_early, y_val_early.astype(np.float32)),
        epochs=args.epochs, batch_size=args.batch,
        class_weight=cls_w,
        callbacks=[es], verbose=2,
    )

    # --- Platt calibration on hold-out ---
    logits_calib = backbone.predict(X_val_calib, verbose=0).reshape(-1)
    p_calib_pre = 1.0 / (1.0 + np.exp(-logits_calib))
    ece_before = expected_calibration_error(p_calib_pre, y_val_calib)
    brier_before = brier_score(p_calib_pre, y_val_calib.astype(np.float64))
    a, b = fit_platt(logits_calib, y_val_calib)
    p_calib_post = 1.0 / (1.0 + np.exp(-(a * logits_calib + b)))
    ece_after = expected_calibration_error(p_calib_post, y_val_calib)
    brier_after = brier_score(p_calib_post, y_val_calib.astype(np.float64))
    print(
        f'  Platt: a={a:.4f} b={b:.4f}\n'
        f'    ECE   before={ece_before:.4f} after={ece_after:.4f}\n'
        f'    Brier before={brier_before:.4f} after={brier_after:.4f}'
    )

    # --- Build export model with calibration baked in ---
    export_model = build_export_model(backbone, a, b)

    # --- Threshold tuning on val_calib (after calibration) ---
    block_thr, warn_thr = tune_thresholds_binary(
        p_calib_post, y_val_calib, min_block_precision=args.min_block_precision,
    )
    print(f'  thresholds: block={block_thr:.3f}, warn={warn_thr:.3f}')

    # --- TFLite export ---
    print(f'\nExporting TFLite -> {args.tflite_output}')
    info = export_tflite(export_model, args.tflite_output)
    print(f'  wrote {info["bytes"]} bytes')
    sanity = sanity_check(export_model, args.tflite_output, X_test, atol=args.sanity_atol)
    print(f'  sanity: max_abs_diff={sanity["max_abs_diff"]:.6f} pass={sanity["pass"]}')
    if not sanity['pass']:
        print('  !! sanity check FAILED — exiting non-zero')
        return 2

    # --- Final test metrics from TFLite ---
    p_test = tflite_predict(args.tflite_output, X_test)
    test_metrics_block = evaluate_binary(p_test, y_test, block_thr)
    test_metrics_warn = evaluate_binary(p_test, y_test, warn_thr)
    test_metrics = {
        'at_block_threshold': test_metrics_block,
        'at_warn_threshold': test_metrics_warn,
        # WARN-zone — что попадает в [warn_thr, block_thr).
        'in_warn_zone': int(((p_test >= warn_thr) & (p_test < block_thr)).sum()),
        'rows': int(len(y_test)),
        # Распределение test:
        'rows_allow': int((y_test == 0).sum()),
        'rows_spam':  int((y_test == 1).sum()),
        # ECE на test для аудита.
        'ece_test_calibrated': float(expected_calibration_error(p_test, y_test)),
        'brier_test_calibrated': float(brier_score(p_test, y_test.astype(np.float64))),
    }
    print(
        f'\nTest metrics:\n'
        f'  block_thr={block_thr:.3f}: P={test_metrics_block["precision"]:.4f} '
        f'R={test_metrics_block["recall"]:.4f} F1={test_metrics_block["f1"]:.4f}\n'
        f'  warn_thr={warn_thr:.3f}:  P={test_metrics_warn["precision"]:.4f} '
        f'R={test_metrics_warn["recall"]:.4f} F1={test_metrics_warn["f1"]:.4f}\n'
        f'  warn_zone_size={test_metrics["in_warn_zone"]}\n'
        f'  ECE(test, calibrated)={test_metrics["ece_test_calibrated"]:.4f}'
    )

    # --- Card ---
    version = f"binary-mlp-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    write_card(
        args.model_card_output,
        version=version,
        dataset_path=args.data,
        dataset_hash=file_sha256(args.data) if os.path.isfile(args.data) else 'no-hash',
        feature_count=X.shape[1],
        rows=int(len(y)),
        class_counts_orig=orig_counts,
        binary_class_counts=bin_counts,
        warn_strategy=args.binary_warn_strategy,
        platt_a=a, platt_b=b,
        calib_metrics={
            'ece_before': ece_before, 'ece_after': ece_after,
            'brier_before': brier_before, 'brier_after': brier_after,
        },
        thresholds=(block_thr, warn_thr),
        test_metrics=test_metrics,
        hidden_sizes=hidden_sizes,
    )
    print(f'\n[done] tflite: {args.tflite_output}')
    print(f'       card:   {args.model_card_output}')
    print(f'       version: {version}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
