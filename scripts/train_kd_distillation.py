"""
Knowledge Distillation: CatBoost teacher → Keras MLP student → TFLite

Отдельный пайплайн, не трогает train_ru_metadata_models.py.
Конечная цель: компактный TFLite [1, 32] -> [1, 3] (ALLOW/WARN/BLOCK), совместимый
с существующим Android SpamModel.kt (FloatArray вход, 3 Float выхода).

Решения по результатам ревью плана:
  - Без feature embeddings: большинство COMPACT_FEATURES бинарные.
  - Без Platt/Isotonic-калибровки teacher: T-scaling уже даёт smoothing.
  - Soft targets берутся ТОЛЬКО с реальных train-точек (не с SMOTE-синтетики),
    чтобы student не учился на интерполированных predictions.
  - Hyperparam search в два этапа: (a) grid (T, alpha) — 9 trials,
    (b) Optuna (lr, dropout, hidden) на лучших T, alpha.
  - Sanity check: |p_keras - p_tflite| < 1e-4 на 200 семплах.
  - Пороги BLOCK / WARN тюнятся на val, пишутся в model_card.json
    в стандартную секцию `thresholds` (поле уже читается ModelCard.kt).
  - Бэкап старого .tflite и model_card.json в reports/<run>/before/.

Sample sizes (по умолчанию):
  --teacher-train-per-class 6000  (6k legit + 6k spam = 12k для обучения teacher)
  --student-train-per-class 4000  (4k legit + 4k spam = 8k подвыборка teacher train)
  Если в данных меньше указанного — печатается warning и берётся реальный максимум,
  если не передан --pad-with-smote.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ru_metadata_features import COMPACT_FEATURES, FIELD_TO_RU, ID_TO_LABEL, LABEL_TO_ID

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'ru')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
DEFAULT_DATA = os.path.join(PROCESSED_DIR, 'ru_tflite_features.csv')
ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'app', 'src', 'main', 'assets')
DEFAULT_TFLITE = os.path.join(ASSETS_DIR, 'spam_model.tflite')
DEFAULT_MODEL_CARD = os.path.join(ASSETS_DIR, 'model_card.json')

LEGIT_LABELS = (LABEL_TO_ID['ALLOW'],)
SPAM_LABELS = (LABEL_TO_ID['WARN'], LABEL_TO_ID['BLOCK'])
NUM_CLASSES = 3

# Phase 4A: фичи, недоступные на устройстве без интернета. На cold-start eval и
# при cold-threshold tuning они обнуляются, noMetadata→1. Источник истины:
# FeatureExtractor.kt (где reviewsLog/negativeRatio/searchVolumeLog/categories
# всегда 0f, и whitelist/blacklist считаются только если репо доступно).
COLD_START_MASK_FEATURES: Tuple[str, ...] = (
    'inAllowlist', 'inBlacklist',
    'reputationScore', 'sourceConfidence',
    'reviewsLog', 'negativeRatio', 'searchVolumeLog',
    'hasFraudCategory', 'hasTelemarketingCategory',
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Pin seed for numpy / random / torch (if loaded) / tf (if loaded) / sklearn."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load CSV with COMPACT_FEATURES + label.

    Принимает оба варианта заголовков:
      - английские: 'isContact', ..., 'label'
      - русские:    'в_контактах', ..., 'метка' (формат builder-а)
    """
    if path.endswith('.npz'):
        data = np.load(path)
        X = np.asarray(data['X'], dtype=np.float32)
        y = np.asarray(data['y'], dtype=np.int64)
        return X, y

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f'No rows in {path}')

    headers = list(rows[0].keys())
    has_english = all(name in headers for name in COMPACT_FEATURES)
    if has_english:
        feature_keys = {name: name for name in COMPACT_FEATURES}
        label_key = 'label' if 'label' in headers else 'метка'
    else:
        # russian headers — translate via FIELD_TO_RU
        feature_keys = {name: FIELD_TO_RU.get(name, name) for name in COMPACT_FEATURES}
        label_key = FIELD_TO_RU.get('label', 'метка')

    missing = [eng for eng, ru in feature_keys.items() if ru not in headers]
    if missing:
        raise SystemExit(
            f'CSV missing features ({len(missing)}): {missing[:5]}{"..." if len(missing) > 5 else ""}'
            f'\nHeaders sample: {headers[:5]}'
        )
    if label_key not in headers:
        raise SystemExit(f"CSV missing label column ('label' or 'метка'). Headers: {headers[:8]}...")

    X = np.array(
        [[float(row[feature_keys[name]]) for name in COMPACT_FEATURES] for row in rows],
        dtype=np.float32,
    )
    raw_labels = [row[label_key] for row in rows]
    # label may be int code (0/1/2) or string ('ALLOW'/'WARN'/'BLOCK')
    y_list: List[int] = []
    for v in raw_labels:
        s = str(v).strip()
        if s in LABEL_TO_ID:
            y_list.append(LABEL_TO_ID[s])
        else:
            y_list.append(int(float(s)))
    y = np.array(y_list, dtype=np.int64)
    return X, y


def class_counts(y: np.ndarray) -> Dict[str, int]:
    return {ID_TO_LABEL.get(i, str(i)): int(np.sum(y == i)) for i in range(NUM_CLASSES)}


# ---------------------------------------------------------------------------
# Sampling: stratified pools + teacher/student train subsets
# ---------------------------------------------------------------------------

def stratified_split(
    X: np.ndarray, y: np.ndarray, sizes: Tuple[float, float, float], seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return three np.ndarray index arrays (train, val, test) preserving per-class ratios."""
    assert abs(sum(sizes) - 1.0) < 1e-6, sizes
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in range(NUM_CLASSES):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(round(n * sizes[0]))
        n_val = int(round(n * sizes[1]))
        train_idx.extend(cls_idx[:n_train].tolist())
        val_idx.extend(cls_idx[n_train:n_train + n_val].tolist())
        test_idx.extend(cls_idx[n_train + n_val:].tolist())
    return (
        np.array(sorted(train_idx), dtype=np.int64),
        np.array(sorted(val_idx), dtype=np.int64),
        np.array(sorted(test_idx), dtype=np.int64),
    )


def sample_teacher_train(
    train_idx: np.ndarray,
    y: np.ndarray,
    legit_target: int,
    spam_target: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """Sample N legit (ALLOW) + N spam (WARN+BLOCK) from train pool.

    Spam sampling preserves WARN/BLOCK ratio inside the train pool.
    Returns (sampled_indices, actual_counts, warnings_list).
    """
    rng = np.random.default_rng(seed)
    warnings: List[str] = []

    # Legit bucket
    legit_pool = train_idx[np.isin(y[train_idx], LEGIT_LABELS)]
    if len(legit_pool) < legit_target:
        warnings.append(
            f'requested {legit_target} legit (ALLOW), only {len(legit_pool)} available in train pool — using all'
        )
        legit_pick = legit_pool.copy()
    else:
        legit_pick = rng.choice(legit_pool, size=legit_target, replace=False)

    # Spam bucket — keep WARN/BLOCK ratio
    spam_pool = train_idx[np.isin(y[train_idx], SPAM_LABELS)]
    if len(spam_pool) < spam_target:
        warnings.append(
            f'requested {spam_target} spam (WARN+BLOCK), only {len(spam_pool)} available in train pool — using all'
        )
        spam_pick = spam_pool.copy()
    else:
        warn_pool = train_idx[y[train_idx] == LABEL_TO_ID['WARN']]
        block_pool = train_idx[y[train_idx] == LABEL_TO_ID['BLOCK']]
        total_spam = len(warn_pool) + len(block_pool)
        warn_share = len(warn_pool) / max(total_spam, 1)
        n_warn = min(int(round(spam_target * warn_share)), len(warn_pool))
        n_block = min(spam_target - n_warn, len(block_pool))
        spam_pick = np.concatenate([
            rng.choice(warn_pool, size=n_warn, replace=False) if n_warn else np.array([], dtype=np.int64),
            rng.choice(block_pool, size=n_block, replace=False) if n_block else np.array([], dtype=np.int64),
        ])

    sampled = np.concatenate([legit_pick, spam_pick]).astype(np.int64)
    sampled.sort()
    return sampled, class_counts(y[sampled]), warnings


def sample_student_train_subset(
    teacher_idx: np.ndarray,
    y: np.ndarray,
    legit_target: int,
    spam_target: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """Sample student train as a subset of teacher train (smaller distillation set)."""
    rng = np.random.default_rng(seed + 1)
    warnings: List[str] = []

    legit_pool = teacher_idx[np.isin(y[teacher_idx], LEGIT_LABELS)]
    spam_pool = teacher_idx[np.isin(y[teacher_idx], SPAM_LABELS)]

    if len(legit_pool) < legit_target:
        warnings.append(
            f'student: requested {legit_target} legit, only {len(legit_pool)} in teacher train — using all'
        )
        legit_pick = legit_pool.copy()
    else:
        legit_pick = rng.choice(legit_pool, size=legit_target, replace=False)

    if len(spam_pool) < spam_target:
        warnings.append(
            f'student: requested {spam_target} spam, only {len(spam_pool)} in teacher train — using all'
        )
        spam_pick = spam_pool.copy()
    else:
        warn_pool = teacher_idx[y[teacher_idx] == LABEL_TO_ID['WARN']]
        block_pool = teacher_idx[y[teacher_idx] == LABEL_TO_ID['BLOCK']]
        total_spam = len(warn_pool) + len(block_pool)
        warn_share = len(warn_pool) / max(total_spam, 1)
        n_warn = min(int(round(spam_target * warn_share)), len(warn_pool))
        n_block = min(spam_target - n_warn, len(block_pool))
        spam_pick = np.concatenate([
            rng.choice(warn_pool, size=n_warn, replace=False) if n_warn else np.array([], dtype=np.int64),
            rng.choice(block_pool, size=n_block, replace=False) if n_block else np.array([], dtype=np.int64),
        ])

    sampled = np.concatenate([legit_pick, spam_pick]).astype(np.int64)
    sampled.sort()
    return sampled, class_counts(y[sampled]), warnings


def maybe_pad_with_smote(
    X: np.ndarray, y: np.ndarray, target_per_class: Dict[int, int], seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Pad up to target counts using SMOTE. Returns padded X, y, and info dict."""
    info = {'before': class_counts(y), 'enabled': True}
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        info.update({'enabled': False, 'reason': 'imblearn not installed'})
        return X, y, info
    counts = np.bincount(y, minlength=NUM_CLASSES)
    sampling_strategy = {
        cls: max(int(counts[cls]), int(target_per_class.get(cls, 0)))
        for cls in range(NUM_CLASSES)
        if counts[cls] > 1
    }
    if all(v <= counts[k] for k, v in sampling_strategy.items()):
        info['after'] = info['before']
        info['noop'] = True
        return X, y, info
    min_class = min(int(counts[c]) for c in sampling_strategy if counts[c] > 1)
    k_neighbors = max(1, min(5, min_class - 1))
    try:
        sm = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=seed)
        X_res, y_res = sm.fit_resample(X, y)
    except Exception as e:
        info.update({'enabled': False, 'reason': f'SMOTE failed: {e}'})
        return X, y, info
    info['after'] = class_counts(y_res)
    return X_res.astype(np.float32), y_res.astype(np.int64), info


# ---------------------------------------------------------------------------
# CatBoost teacher
# ---------------------------------------------------------------------------

def train_catboost_teacher(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    seed: int,
    iterations: int = 500,
    depth: int = 6,
    learning_rate: float = 0.05,
    sample_weight: Optional[np.ndarray] = None,
):
    from catboost import CatBoostClassifier
    # CatBoost не любит пустые eval_set sample_weight, поэтому передаём только train.
    teacher = CatBoostClassifier(
        loss_function='MultiClass',
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        auto_class_weights='Balanced' if sample_weight is None else None,
        random_seed=seed,
        verbose=False,
        eval_metric='TotalF1',
        early_stopping_rounds=30,
    )
    fit_kwargs = dict(eval_set=(X_val, y_val), use_best_model=True, verbose=False)
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight
    teacher.fit(X_train, y_train, **fit_kwargs)
    return teacher


def train_lightgbm_teacher(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    seed: int,
    n_estimators: int = 600,
    num_leaves: int = 63,
    learning_rate: float = 0.05,
    sample_weight: Optional[np.ndarray] = None,
):
    """Phase 3: второй teacher для ensemble (CatBoost + LightGBM averaging).

    Делает базовую классификацию multiclass с balanced class_weight (если веса
    не переданы) или предоставленными per-sample весами (для WARN re-weighting).
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return None
    params = dict(
        objective='multiclass',
        num_class=NUM_CLASSES,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        class_weight='balanced' if sample_weight is None else None,
    )
    model = __import__('lightgbm').LGBMClassifier(**params)
    fit_kwargs = dict(
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[__import__('lightgbm').early_stopping(40, verbose=False)],
    )
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight
    model.fit(X_train, y_train, **fit_kwargs)
    return model


class TeacherEnsemble:
    """Ансамбль teacher-моделей: усредняет predict_proba от CatBoost + LightGBM.

    Phase 3: помогает когда у CatBoost'а есть систематический bias по WARN/ALLOW —
    LightGBM с другой древесной схемой даёт независимый сигнал.
    """

    def __init__(self, models, weights=None):
        self.models = [m for m in models if m is not None]
        if not self.models:
            raise ValueError('TeacherEnsemble: no models provided')
        if weights is None:
            weights = [1.0] * len(self.models)
        self.weights = np.array(weights[:len(self.models)], dtype=np.float64)
        self.weights = self.weights / max(self.weights.sum(), 1e-9)

    def predict_proba(self, X):
        probs = None
        for m, w in zip(self.models, self.weights):
            p = np.asarray(m.predict_proba(X), dtype=np.float64)
            if probs is None:
                probs = w * p
            else:
                probs = probs + w * p
        return probs.astype(np.float32)


def teacher_soft_targets(teacher, X: np.ndarray, T: float) -> np.ndarray:
    """Get teacher soft targets at temperature T.

    p_T = softmax(log(p) / T). For T=1 returns original probabilities.
    """
    proba = teacher.predict_proba(X).astype(np.float64)
    proba = np.clip(proba, 1e-9, 1.0)
    logits = np.log(proba)
    scaled = logits / max(T, 1e-6)
    # numerical-stable softmax
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    e = np.exp(scaled)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------------------
# Keras student (subclass with KD train_step)
# ---------------------------------------------------------------------------

def feature_mask_indices(mask_features: Sequence[str]) -> List[int]:
    """Превращает имена фич в индексы в векторе COMPACT_FEATURES (для маскировки во время обучения)."""
    name_to_idx = {n: i for i, n in enumerate(COMPACT_FEATURES)}
    out = []
    for name in mask_features:
        if name in name_to_idx:
            out.append(name_to_idx[name])
    return out


def make_masked_view(X: np.ndarray, mask_indices: Sequence[int]) -> np.ndarray:
    """Возвращает копию X, где колонки из mask_indices обнулены (детерминированно для всех строк).

    Это «вид без подсказок»: имитация unknown-номера, у которого нет ни whitelist-,
    ни blacklist-сигнала. Используется для teacher-aware feature-masking augmentation.

    Phase 3: если в COMPACT_FEATURES есть `noMetadata` И мы маскируем хотя бы одну
    из metadata-фич (reputation/reviews/categories/prefix histogram) — устанавливаем
    noMetadata=1, чтобы тренировочные масочные ряды совпадали по распределению
    с runtime cold-start (когда metadata реально отсутствуют).
    """
    if not mask_indices:
        return X
    Xm = X.copy()
    Xm[:, list(mask_indices)] = 0.0
    # noMetadata=1 для cold-start consistency: триггерим только когда маскируются
    # ОНЛАЙН-метаданные (reputation/reviews/categories/lists). Шипимые JSON-лукапы
    # (prefix histogram, def_code_risk) НЕ триггерят noMetadata, т.к. на устройстве
    # они доступны независимо от наличия metadata.
    if 'noMetadata' in COMPACT_FEATURES:
        no_meta_idx = COMPACT_FEATURES.index('noMetadata')
        meta_trigger_names = {
            'reputationScore', 'sourceConfidence',
            'reviewsLog', 'negativeRatio', 'searchVolumeLog',
            'hasFraudCategory', 'hasTelemarketingCategory',
            'inAllowlist', 'inBlacklist',
        }
        meta_trigger_idx = {COMPACT_FEATURES.index(n) for n in meta_trigger_names if n in COMPACT_FEATURES}
        if any(i in meta_trigger_idx for i in mask_indices):
            Xm[:, no_meta_idx] = 1.0
    return Xm


def concat_masked_aug(
    X: np.ndarray, y: np.ndarray,
    mask_indices: Sequence[int], mask_prob: float, seed: int,
    weight_multiplier: float = 1.0,
    stratified: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Расширяет (X, y) дополнительными «masked-копиями» строк.

    Конкатенирует к (X, y) случайно выбранную долю mask_prob строк, в которых
    обнулены колонки mask_indices. mask_prob=0.5 → +50% rows (1.5*N total).
    mask_prob=1.0 → +100% rows (full doubling). mask_prob=0 → no-op.

    Цель — учить teacher и student на согласованном множестве (unmasked + masked),
    чтобы soft-targets, которые student видит при KD, отражали тот же mask-режим,
    что и сам student-вход.

    Phase 4C параметры:
      - `weight_multiplier` (default 1.0 = Phase 3 поведение, 1.5 для Phase 4C):
        per-row множитель weight на masked рядах (orig=1.0, masked=mult).
      - `stratified` (default True): при `mask_prob<1`, выборка осуществляется
        пропорционально классам, чтобы masked subset сохранял class balance
        исходного train-set. Это критично для Phase 4C — без stratified, с
        weight_mult>1, дисбаланс классов в masked subset искажает effective
        loss и валит модель.

    Возвращает (X_aug, y_aug, mask_extra_weights, info).
    """
    info = {'orig_rows': int(len(X)), 'masked_added': 0, 'total': int(len(X)),
            'mask_prob': float(mask_prob), 'mask_indices': list(mask_indices),
            'mode': 'random_full' if mask_prob >= 1.0 else 'random_partial',
            'weight_multiplier': float(weight_multiplier),
            'stratified': bool(stratified)}
    extra_w = np.ones(len(X), dtype=np.float32)
    if not mask_indices or mask_prob <= 0.0 or len(X) == 0:
        return X, y, extra_w, info
    rng = np.random.default_rng(seed)
    n = len(X)
    n_add = int(round(n * float(mask_prob)))
    if n_add <= 0:
        return X, y, extra_w, info
    if n_add >= n:
        # Полное удвоение: все ряды дублируются с masked.
        idx = np.arange(n)
    elif stratified:
        # Stratified subsample: каждый класс пропорционально mask_prob.
        idx_chunks: List[np.ndarray] = []
        for c in np.unique(y):
            class_idx = np.where(y == c)[0]
            n_class = int(round(len(class_idx) * float(mask_prob)))
            if n_class <= 0:
                continue
            n_class = min(n_class, len(class_idx))
            picked = rng.choice(class_idx, size=n_class, replace=False)
            idx_chunks.append(picked)
        idx = np.concatenate(idx_chunks) if idx_chunks else np.array([], dtype=np.int64)
    else:
        idx = rng.choice(n, size=n_add, replace=False)
    if len(idx) == 0:
        return X, y, extra_w, info
    X_masked = make_masked_view(X[idx], mask_indices)
    X_aug = np.concatenate([X, X_masked.astype(X.dtype)], axis=0)
    y_aug = np.concatenate([y, y[idx]], axis=0)
    extra_w_aug = np.concatenate([
        extra_w,
        np.full(len(idx), float(weight_multiplier), dtype=np.float32),
    ], axis=0)
    info.update({'masked_added': int(len(idx)), 'total': int(len(X_aug))})
    return X_aug, y_aug, extra_w_aug, info


def concat_masked_aug_balanced(
    X: np.ndarray, y: np.ndarray,
    mask_indices: Sequence[int],
    target_classes: Sequence[int],
    weight_multiplier: float,
    seed: int,
    warn_oversample_cap: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Phase 4C: усиленный cold-start aug — selective spam masking + weight bump.

    Отличия от `concat_masked_aug`:
      1. **Selective masking**: маскируются только ряды target классов (BLOCK + WARN
         по умолчанию). ALLOW в датасете уже на 99.3% имеет noMetadata=1
         (shortcut, см. cold_start_baseline.md), поэтому добавлять masked-ALLOW
         только укрепляет связку «noMetadata=1 → ALLOW». Для cold-start нам
         нужно ровно противоположное: научить модель распознавать BLOCK/WARN
         БЕЗ метаданных.
      2. **mask_prob=1.0 на каждом spam-классе**: каждая исходная BLOCK/WARN
         строка получает masked-копию (или умножается на cap для редких классов).
      3. **WARN oversample cap**: множитель для редких классов (по умолчанию 1.0
         = без oversampling, только одна masked-копия на каждую WARN-строку).
         Балансировка отдаётся `class_weights_for_y` (WARN×3, см. CLI
         `--warn-class-weight`). Cap=4 — компромисс между усилением WARN
         и риском over-fitting на 3.5k WARN-prototypes (~14k synthetic).
      4. **Sample weight bump**: каждая masked строка получает множитель
         `weight_multiplier` (×1.5 по умолчанию) поверх class-weight. Это
         сдвигает effective loss в сторону «учись на cold-start даже больше,
         чем на обычных рядах».

    Возвращает (X_aug, y_aug, mask_extra_weights, info).
    """
    info = {
        'orig_rows': int(len(X)), 'masked_added': 0, 'total': int(len(X)),
        'mask_indices': list(mask_indices),
        'mode': 'balanced_spam',
        'target_classes': [int(c) for c in target_classes],
        'weight_multiplier': float(weight_multiplier),
        'warn_oversample_cap': float(warn_oversample_cap),
        'per_class_added': {},
    }
    if not mask_indices or len(X) == 0 or not target_classes:
        return X, y, np.ones(len(X), dtype=np.float32), info

    rng = np.random.default_rng(seed)

    class_idxs = {int(c): np.where(y == int(c))[0] for c in target_classes}
    sizes = {c: len(idxs) for c, idxs in class_idxs.items() if len(idxs) > 0}
    if not sizes:
        return X, y, np.ones(len(X), dtype=np.float32), info

    # Default: target_per_class = original count (mask_prob=1.0, no row duplication
    # beyond cap). cap > 1.0 — oversample minorities up to cap × max(other classes).
    max_size = max(sizes.values())
    masked_X_chunks: List[np.ndarray] = []
    masked_y_chunks: List[np.ndarray] = []
    for c, idxs in class_idxs.items():
        if len(idxs) == 0:
            continue
        target_size = len(idxs)
        if warn_oversample_cap > 1.0:
            # Допускаем oversample до cap × original. Но не больше max_size.
            cap_target = int(min(max_size, len(idxs) * warn_oversample_cap))
            target_size = max(target_size, cap_target)
        if target_size <= len(idxs):
            picked = rng.choice(idxs, size=target_size, replace=False)
        else:
            picked = rng.choice(idxs, size=target_size, replace=True)
        masked_X_chunks.append(make_masked_view(X[picked], mask_indices))
        masked_y_chunks.append(np.full(target_size, c, dtype=y.dtype))
        info['per_class_added'][int(c)] = int(target_size)

    if not masked_X_chunks:
        return X, y, np.ones(len(X), dtype=np.float32), info

    X_masked = np.concatenate(masked_X_chunks, axis=0).astype(X.dtype)
    y_masked = np.concatenate(masked_y_chunks, axis=0)
    n_added = int(len(X_masked))

    X_aug = np.concatenate([X, X_masked], axis=0)
    y_aug = np.concatenate([y, y_masked], axis=0)
    extra_w_aug = np.concatenate([
        np.ones(len(X), dtype=np.float32),
        np.full(n_added, float(weight_multiplier), dtype=np.float32),
    ], axis=0)

    info.update({
        'masked_added': n_added,
        'total': int(len(X_aug)),
    })
    return X_aug, y_aug, extra_w_aug, info


def build_student_model(
    hidden_sizes: Tuple[int, ...], dropout: float, T: float, alpha: float, lr: float,
    *, weight_decay: float = 0.0, label_smoothing: float = 0.0,
    mask_indices: Optional[List[int]] = None, mask_prob: float = 0.0,
):
    """Construct KD-Keras model (training mode produces logits + KL/CE loss).

    Architecture:
        Input(32) -> Linear(h0) -> ReLU -> Dropout
                  -> Linear(h1) -> ReLU -> Dropout
                  -> Linear(h2) -> ReLU
                  -> Linear(3 logits)

    No BatchNorm: avoids fold-time errors during TFLite export and is unnecessary
    at this scale.

    Optional regularizers (все без BatchNorm и без изменения архитектуры весов —
    точно такая же [1, 32] -> [1, 3] экспортная модель для Android, это только при обучении):
      - weight_decay: L2-регуляризация на ядра всех Dense (kernel_regularizer).
      - label_smoothing: мягче хард-CE в KD-лоссе (0.05 → 5% массы размазывается по остальным классам).
      - mask_indices/mask_prob: больше НЕ используются внутри KDModel (оставлены
        в сигнатуре только для совместимости). Маскирование теперь baked-in в датасет
        (см. concat_masked_aug в main): teacher и student обучаются на одинаковом
        concat(X, masked_X), и soft-targets берутся именно с этого расширенного X.
    """
    import tensorflow as tf

    reg = tf.keras.regularizers.l2(weight_decay) if weight_decay and weight_decay > 0 else None
    inputs = tf.keras.Input(shape=(len(COMPACT_FEATURES),), name='features')
    h = inputs
    for i, units in enumerate(hidden_sizes[:-1]):
        h = tf.keras.layers.Dense(units, activation='relu', kernel_regularizer=reg, name=f'dense_{i}')(h)
        h = tf.keras.layers.Dropout(dropout, name=f'drop_{i}')(h)
    h = tf.keras.layers.Dense(
        hidden_sizes[-1], activation='relu', kernel_regularizer=reg,
        name=f'dense_{len(hidden_sizes) - 1}',
    )(h)
    logits = tf.keras.layers.Dense(NUM_CLASSES, activation=None, kernel_regularizer=reg, name='logits')(h)
    backbone = tf.keras.Model(inputs, logits, name='student_backbone')

    # Маскинг теперь baked-in в сам датасет (concat_masked_aug),
    # KDModel не делает per-batch random masking — soft-targets и x согласованы
    # по построению (teacher был обучен на тех же masked/unmasked парах).
    _ = mask_indices  # noqa: kept for backward compat in signature
    _ = mask_prob

    class KDModel(tf.keras.Model):
        def __init__(self, backbone, T, alpha, label_smoothing):
            super().__init__()
            self.backbone = backbone
            self.T = float(T)
            self.alpha = float(alpha)
            self.label_smoothing = float(label_smoothing)
            self.ce_metric = tf.keras.metrics.Mean(name='ce')
            self.kd_metric = tf.keras.metrics.Mean(name='kd')
            self.acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='acc')

        def call(self, x, training=False):
            return self.backbone(x, training=training)

        def train_step(self, data):
            # Поддерживаем два формата:
            #  ((x, y_hard), soft)        — старый вариант без sample_weight
            #  ((x, y_hard, w), soft)     — Phase 3: взвешивание по классу/свежести
            inputs, soft = data
            if len(inputs) == 3:
                x, y_hard, w = inputs
                w = tf.cast(w, tf.float32)
            else:
                x, y_hard = inputs
                w = tf.ones((tf.shape(x)[0],), dtype=tf.float32)
            with tf.GradientTape() as tape:
                student_logits = self.backbone(x, training=True)
                # Hard CE (с label smoothing если > 0).
                if self.label_smoothing > 0.0:
                    y_onehot = tf.one_hot(tf.cast(y_hard, tf.int32), NUM_CLASSES, dtype=tf.float32)
                    y_smooth = y_onehot * (1.0 - self.label_smoothing) + self.label_smoothing / NUM_CLASSES
                    log_p = tf.nn.log_softmax(student_logits, axis=-1)
                    ce_per_sample = -tf.reduce_sum(y_smooth * log_p, axis=-1)
                else:
                    ce_per_sample = tf.keras.losses.sparse_categorical_crossentropy(
                        y_hard, student_logits, from_logits=True,
                    )
                w_sum = tf.reduce_sum(w) + 1e-9
                ce = tf.reduce_sum(w * ce_per_sample) / w_sum
                # KD: KL(student/T || teacher_soft@T)
                T = self.T
                student_log_soft = tf.nn.log_softmax(student_logits / T, axis=-1)
                teacher_soft = tf.cast(soft, tf.float32)
                eps = 1e-9
                teacher_log = tf.math.log(teacher_soft + eps)
                kl_per_sample = tf.reduce_sum(teacher_soft * (teacher_log - student_log_soft), axis=-1)
                kl = tf.reduce_sum(w * kl_per_sample) / w_sum
                kd_loss = self.alpha * ce + (1.0 - self.alpha) * (T * T) * kl
                # Активируем L2-регуляризаторы Dense'ов.
                reg_loss = tf.add_n(self.backbone.losses) if self.backbone.losses else 0.0
                loss = kd_loss + reg_loss
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.ce_metric.update_state(ce)
            self.kd_metric.update_state(kl)
            self.acc_metric.update_state(y_hard, student_logits)
            return {
                'loss': loss,
                'ce': self.ce_metric.result(),
                'kd': self.kd_metric.result(),
                'acc': self.acc_metric.result(),
            }

        def test_step(self, data):
            x, y_hard = data
            logits = self.backbone(x, training=False)
            ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                y_hard, logits, from_logits=True,
            ))
            self.acc_metric.update_state(y_hard, logits)
            return {'loss': ce, 'acc': self.acc_metric.result()}

    model = KDModel(backbone, T=T, alpha=alpha, label_smoothing=label_smoothing)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    return model, backbone


def class_weights_for_y(
    y: np.ndarray,
    *,
    warn_weight: float = 1.0,
    block_weight: float = 1.0,
    allow_weight: float = 1.0,
) -> np.ndarray:
    """Per-sample weights for re-weighting WARN/BLOCK during training.

    Phase 3: WARN класс получает 3x вес по умолчанию (~3.5k WARN против ~48k BLOCK).
    Без этого модель схлопывает WARN в ALLOW.
    """
    w = np.ones_like(y, dtype=np.float32)
    w[y == LABEL_TO_ID['ALLOW']] = float(allow_weight)
    w[y == LABEL_TO_ID['WARN']] = float(warn_weight)
    w[y == LABEL_TO_ID['BLOCK']] = float(block_weight)
    return w


def train_student(
    X_train: np.ndarray, y_train: np.ndarray, soft_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *, T: float, alpha: float, hidden_sizes: Tuple[int, ...], dropout: float,
    lr: float, epochs: int, batch_size: int, patience: int, seed: int, verbose: int = 0,
    weight_decay: float = 0.0, label_smoothing: float = 0.0,
    mask_indices: Optional[List[int]] = None, mask_prob: float = 0.0,
    sample_weight: Optional[np.ndarray] = None,
):
    """Train one student with given hyperparams. Returns (backbone, history, best_val_acc).

    Phase 3: поддерживает sample_weight (per-row) — это веса для WARN-ревейтинга.
    При sample_weight=None ведёт себя как раньше (веса по умолчанию = 1).
    """
    import tensorflow as tf
    set_global_seed(seed)
    model, backbone = build_student_model(
        hidden_sizes, dropout, T, alpha, lr,
        weight_decay=weight_decay, label_smoothing=label_smoothing,
        mask_indices=mask_indices, mask_prob=mask_prob,
    )

    if sample_weight is None:
        train_ds = tf.data.Dataset.from_tensor_slices(
            ((X_train.astype(np.float32), y_train.astype(np.int32)), soft_train.astype(np.float32))
        ).shuffle(buffer_size=min(len(X_train), 8192), seed=seed).batch(batch_size)
    else:
        if len(sample_weight) != len(X_train):
            raise ValueError(f'sample_weight length {len(sample_weight)} != X_train {len(X_train)}')
        train_ds = tf.data.Dataset.from_tensor_slices(
            ((X_train.astype(np.float32), y_train.astype(np.int32), sample_weight.astype(np.float32)),
             soft_train.astype(np.float32))
        ).shuffle(buffer_size=min(len(X_train), 8192), seed=seed).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (X_val.astype(np.float32), y_val.astype(np.int32))
    ).batch(batch_size)

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_acc', mode='max', patience=patience, restore_best_weights=True,
    )
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=[es], verbose=verbose,
    )
    best_val_acc = float(max(history.history.get('val_acc', [0.0])))
    return backbone, history.history, best_val_acc


# ---------------------------------------------------------------------------
# Metrics + threshold tuning
# ---------------------------------------------------------------------------

def proba_from_backbone(backbone, X: np.ndarray) -> np.ndarray:
    import tensorflow as tf
    logits = backbone.predict(X.astype(np.float32), verbose=0)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    out = {
        ID_TO_LABEL[i]: {'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f[i])}
        for i in range(NUM_CLASSES)
    }
    out['macro_f1'] = float(np.mean(f))
    out['confusion_matrix'] = cm
    return out


def evaluate_proba(y_true: np.ndarray, proba: np.ndarray, thresholds: Optional[Dict] = None) -> Dict:
    if thresholds:
        bt = float(thresholds.get('block_threshold', 0.5))
        wt = float(thresholds.get('warn_threshold', 0.3))
        pred = np.zeros(len(y_true), dtype=np.int64)
        block_mask = proba[:, 2] >= bt
        warn_mask = (~block_mask) & (proba[:, 1] >= wt)
        pred[block_mask] = 2
        pred[warn_mask] = 1
        # else stays ALLOW=0
    else:
        pred = np.argmax(proba, axis=1)
    metrics = per_class_metrics(y_true, pred)
    try:
        from sklearn.metrics import roc_auc_score
        metrics['roc_auc_ovr'] = float(roc_auc_score(y_true, proba, multi_class='ovr', labels=[0, 1, 2]))
    except Exception:
        metrics['roc_auc_ovr'] = None
    return metrics


def make_cold_view(X: np.ndarray, mask_indices: Sequence[int], no_meta_idx: int) -> np.ndarray:
    """Phase 4A: cold-start view of feature matrix.

    Возвращает копию X с обнулёнными metadata-колонками (mask_indices) и
    noMetadata=1 (если у нас есть такая колонка). Используется и для
    cold-start eval slice, и для tune_thresholds на cold view val.
    """
    Xc = X.copy()
    if mask_indices:
        Xc[:, list(mask_indices)] = 0.0
    if no_meta_idx >= 0:
        Xc[:, no_meta_idx] = 1.0
    return Xc


def tune_thresholds(y_true: np.ndarray, proba: np.ndarray, min_block_precision: float) -> Dict:
    """Find block_threshold maximizing BLOCK F1 subject to BLOCK precision >= floor.

    Then find warn_threshold maximizing WARN F1 with the picked block threshold.
    """
    best = {
        'block_threshold': 0.50,
        'warn_threshold': 0.30,
        'block_precision': 0.0,
        'block_recall': 0.0,
        'block_f1': 0.0,
    }
    for bt in np.linspace(0.10, 0.95, 86):
        pred_block = proba[:, 2] >= bt
        tp = int(np.sum(pred_block & (y_true == 2)))
        fp = int(np.sum(pred_block & (y_true != 2)))
        fn = int(np.sum(~pred_block & (y_true == 2)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if prec >= min_block_precision and f1 > best['block_f1']:
            best.update({
                'block_threshold': float(bt),
                'block_precision': float(prec),
                'block_recall': float(rec),
                'block_f1': float(f1),
            })

    bt = best['block_threshold']
    best_warn = {'warn_threshold': 0.30, 'warn_f1': 0.0}
    for wt in np.linspace(0.10, 0.85, 76):
        block_mask = proba[:, 2] >= bt
        warn_mask = (~block_mask) & (proba[:, 1] >= wt)
        tp = int(np.sum(warn_mask & (y_true == 1)))
        fp = int(np.sum(warn_mask & (y_true != 1)))
        fn = int(np.sum(~warn_mask & (y_true == 1)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_warn['warn_f1']:
            best_warn = {'warn_threshold': float(wt), 'warn_f1': float(f1)}
    best['warn_threshold'] = best_warn['warn_threshold']
    best['warn_f1'] = best_warn['warn_f1']
    return best


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

def kd_objective_score(metrics: Dict, min_block_precision: float) -> float:
    """Macro F1 with soft penalty on BLOCK precision below floor."""
    macro = metrics.get('macro_f1', 0.0)
    block_p = metrics.get('BLOCK', {}).get('precision', 0.0)
    penalty = 0.3 * max(0.0, min_block_precision - block_p)
    return macro - penalty


def stage1_grid_search(
    teacher, X_st: np.ndarray, y_st: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *, hidden_sizes: Tuple[int, ...], dropout: float, lr: float,
    epochs: int, batch_size: int, patience: int, seed: int,
    Ts: List[float], alphas: List[float], min_block_precision: float, verbose: int = 0,
    weight_decay: float = 0.0, label_smoothing: float = 0.0,
    mask_indices: Optional[List[int]] = None, mask_prob: float = 0.0,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict:
    """Grid search over (T, alpha). Returns best config + all results."""
    results = []
    best_score = -1e9
    best_cfg = None
    print(f'  [stage1] grid: T={Ts} × alpha={alphas} = {len(Ts) * len(alphas)} runs')
    for T in Ts:
        soft = teacher_soft_targets(teacher, X_st, T)
        for alpha in alphas:
            t0 = time.time()
            backbone, _, val_acc = train_student(
                X_st, y_st, soft, X_val, y_val,
                T=T, alpha=alpha, hidden_sizes=hidden_sizes, dropout=dropout,
                lr=lr, epochs=epochs, batch_size=batch_size, patience=patience,
                seed=seed, verbose=verbose,
                weight_decay=weight_decay, label_smoothing=label_smoothing,
                mask_indices=mask_indices, mask_prob=mask_prob,
                sample_weight=sample_weight,
            )
            proba_val = proba_from_backbone(backbone, X_val)
            metrics = evaluate_proba(y_val, proba_val)
            score = kd_objective_score(metrics, min_block_precision)
            elapsed = time.time() - t0
            print(
                f'    T={T:>4.2f} alpha={alpha:>4.2f} | macroF1={metrics["macro_f1"]:.4f} '
                f'BLOCK_P={metrics["BLOCK"]["precision"]:.3f} score={score:.4f} ({elapsed:.1f}s)'
            )
            results.append({
                'T': float(T), 'alpha': float(alpha), 'val_acc': val_acc,
                'metrics': metrics, 'score': score, 'elapsed_s': elapsed,
            })
            if score > best_score:
                best_score = score
                best_cfg = {'T': float(T), 'alpha': float(alpha), 'score': score, 'metrics': metrics}
    return {'all': results, 'best': best_cfg}


def stage2_optuna_search(
    teacher, X_st: np.ndarray, y_st: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *, T: float, alpha: float,
    n_trials: int, epochs: int, batch_size: int, patience: int, seed: int,
    min_block_precision: float, verbose: int = 0,
    weight_decay: float = 0.0, label_smoothing: float = 0.0,
    mask_indices: Optional[List[int]] = None, mask_prob: float = 0.0,
    sample_weight: Optional[np.ndarray] = None,
    hidden_search_space: Optional[Dict[str, List[int]]] = None,
) -> Dict:
    """Optuna over (lr, dropout, hidden_sizes) at fixed T, alpha. Returns best config."""
    try:
        import optuna
        from optuna.pruners import MedianPruner
    except ImportError:
        return {'skipped': 'optuna not installed'}

    print(f'  [stage2] optuna: {n_trials} trials at T={T} alpha={alpha}')
    soft = teacher_soft_targets(teacher, X_st, T)
    trials_log = []
    space = hidden_search_space or {
        'h0': [64, 96, 128, 160],
        'h1': [48, 64, 96],
        'h2': [24, 32, 48],
    }

    def objective(trial: 'optuna.Trial') -> float:
        lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        dropout = trial.suggest_float('dropout', 0.05, 0.4)
        h0 = trial.suggest_categorical('h0', space['h0'])
        h1 = trial.suggest_categorical('h1', space['h1'])
        h2 = trial.suggest_categorical('h2', space['h2'])
        hidden_sizes = (h0, h1, h2)
        backbone, _, val_acc = train_student(
            X_st, y_st, soft, X_val, y_val,
            T=T, alpha=alpha, hidden_sizes=hidden_sizes, dropout=dropout,
            lr=lr, epochs=epochs, batch_size=batch_size, patience=patience,
            seed=seed, verbose=verbose,
            weight_decay=weight_decay, label_smoothing=label_smoothing,
            sample_weight=sample_weight,
            mask_indices=mask_indices, mask_prob=mask_prob,
        )
        proba_val = proba_from_backbone(backbone, X_val)
        metrics = evaluate_proba(y_val, proba_val)
        score = kd_objective_score(metrics, min_block_precision)
        trials_log.append({
            'params': {'lr': lr, 'dropout': dropout, 'h0': h0, 'h1': h1, 'h2': h2},
            'val_acc': val_acc, 'macro_f1': metrics['macro_f1'],
            'block_precision': metrics['BLOCK']['precision'], 'score': score,
        })
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        'best_params': study.best_params,
        'best_value': float(study.best_value),
        'trials': trials_log,
    }


# ---------------------------------------------------------------------------
# Plain MLP baseline (no KD) for comparison
# ---------------------------------------------------------------------------

def train_plain_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *, hidden_sizes: Tuple[int, ...], dropout: float, lr: float,
    epochs: int, batch_size: int, patience: int, seed: int, verbose: int = 0,
    weight_decay: float = 0.0,
    mask_indices: Optional[List[int]] = None, mask_prob: float = 0.0,  # kept for sig compat
    sample_weight: Optional[np.ndarray] = None,
):
    import tensorflow as tf
    set_global_seed(seed)
    reg = tf.keras.regularizers.l2(weight_decay) if weight_decay and weight_decay > 0 else None
    inputs = tf.keras.Input(shape=(len(COMPACT_FEATURES),), name='features')
    h = inputs
    for i, units in enumerate(hidden_sizes[:-1]):
        h = tf.keras.layers.Dense(units, activation='relu', kernel_regularizer=reg, name=f'dense_{i}')(h)
        h = tf.keras.layers.Dropout(dropout, name=f'drop_{i}')(h)
    h = tf.keras.layers.Dense(
        hidden_sizes[-1], activation='relu', kernel_regularizer=reg,
        name=f'dense_{len(hidden_sizes) - 1}',
    )(h)
    logits = tf.keras.layers.Dense(NUM_CLASSES, activation=None, kernel_regularizer=reg, name='logits')(h)
    model = tf.keras.Model(inputs, logits)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc')],
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_acc', mode='max', patience=patience, restore_best_weights=True,
    )
    # Маскинг теперь делает caller через concat_masked_aug; здесь — обычный fit на (X_train, y_train).
    _ = (mask_indices, mask_prob)
    fit_kwargs = dict(
        x=X_train.astype(np.float32), y=y_train.astype(np.int32),
        validation_data=(X_val.astype(np.float32), y_val.astype(np.int32)),
        epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=verbose,
    )
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight.astype(np.float32)
    model.fit(**fit_kwargs)
    return model


# ---------------------------------------------------------------------------
# Export to TFLite + sanity check
# ---------------------------------------------------------------------------

def build_export_model(backbone) -> 'tf.keras.Model':
    """Wrap backbone with Softmax so Android receives probabilities directly."""
    import tensorflow as tf
    inputs = tf.keras.Input(shape=(len(COMPACT_FEATURES),), name='features')
    logits = backbone(inputs, training=False)
    probs = tf.keras.layers.Softmax(name='probabilities')(logits)
    return tf.keras.Model(inputs, probs, name='spam_model_export')


def _make_serving_fn(backbone):
    """tf.function with fixed [1, N] input signature returning softmax probs.

    Avoids the TFLiteConverter.from_keras_model bug under TF 2.16+/Keras 3
    ('NoneType is not callable' from tflite_keras_util._wrapped_model).
    """
    import tensorflow as tf
    n_features = len(COMPACT_FEATURES)

    @tf.function(input_signature=[tf.TensorSpec([1, n_features], tf.float32, name='features')])
    def serving_fn(x):
        logits = backbone(x, training=False)
        return tf.nn.softmax(logits, axis=-1)

    return serving_fn


def export_tflite(backbone, out_path: str, *, quantize: bool = False) -> Dict:
    """Export backbone to a TFLite file with [1, N] input and [1, 3] softmax output.

    Tries three converter paths in order — first that succeeds wins:
      1. from_concrete_functions(serving_fn) — most stable on Keras 3.
      2. from_keras_model(export_model)      — classic path.
      3. from_saved_model(SavedModel dir)    — last resort.

    By default produces a pure FP32 model (no optimizations). Pass quantize=True
    to enable dynamic-range quantization (weights -> int8, compute in float),
    which reduces .tflite size ~4x but introduces ~1e-3 numerical drift and
    will fail the FP32 sanity check.
    """
    import tensorflow as tf
    import tempfile

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    export_model = build_export_model(backbone)

    def apply_opts(conv):
        if quantize:
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
        return conv

    errors: List[str] = []
    tflite_bytes: Optional[bytes] = None

    # Path 1: concrete-function signature
    try:
        serving_fn = _make_serving_fn(backbone)
        concrete = serving_fn.get_concrete_function()
        converter = apply_opts(tf.lite.TFLiteConverter.from_concrete_functions([concrete]))
        tflite_bytes = converter.convert()
    except Exception as e:
        errors.append(f'from_concrete_functions: {type(e).__name__}: {e}')

    # Path 2: keras model directly
    if tflite_bytes is None:
        try:
            converter = apply_opts(tf.lite.TFLiteConverter.from_keras_model(export_model))
            tflite_bytes = converter.convert()
        except Exception as e:
            errors.append(f'from_keras_model: {type(e).__name__}: {e}')

    # Path 3: SavedModel dir round-trip
    if tflite_bytes is None:
        try:
            with tempfile.TemporaryDirectory(prefix='kd_savedmodel_') as tmpdir:
                serving_fn = _make_serving_fn(backbone)
                tf.saved_model.save(
                    backbone, tmpdir,
                    signatures={'serving_default': serving_fn.get_concrete_function()},
                )
                converter = apply_opts(tf.lite.TFLiteConverter.from_saved_model(tmpdir))
                tflite_bytes = converter.convert()
        except Exception as e:
            errors.append(f'from_saved_model: {type(e).__name__}: {e}')

    if tflite_bytes is None:
        raise SystemExit('TFLite export failed via all paths:\n  - ' + '\n  - '.join(errors))

    with open(out_path, 'wb') as f:
        f.write(tflite_bytes)
    print(f'  TFLite converter path used: {"concrete_functions" if not errors else ("keras_model" if len(errors) == 1 else "saved_model")}')
    return {'path': out_path, 'bytes': len(tflite_bytes), 'export_model': export_model}


def tflite_predict(tflite_path: str, X: np.ndarray) -> np.ndarray:
    """Run TFLite inference. Tries the modern ai_edge_litert package first
    (TF 2.20+), then falls back to legacy tf.lite.Interpreter.

    The legacy `tf.lite.Interpreter(model_path=...)` ctor signature broke in
    TF 2.21 (the underlying CreateWrapperFromFile pybind grew a new arg).
    """
    # Path 1: ai-edge-litert (recommended by TF for 2.20+).
    try:
        from ai_edge_litert.interpreter import Interpreter as _LiteRTInterpreter
        interp = _LiteRTInterpreter(model_path=tflite_path)
    except Exception:
        # Path 2: legacy tf.lite.Interpreter, robust to ctor signature drift.
        import tensorflow as tf
        try:
            interp = tf.lite.Interpreter(model_path=tflite_path)
        except TypeError:
            # TF 2.21 changed CreateWrapperFromFile bindings — load via
            # model_content from disk to bypass the broken model_path branch.
            with open(tflite_path, 'rb') as f:
                tflite_bytes = f.read()
            interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    out = np.zeros((len(X), NUM_CLASSES), dtype=np.float32)
    for i, row in enumerate(X.astype(np.float32)):
        interp.set_tensor(in_d['index'], row.reshape(1, -1))
        interp.invoke()
        out[i] = interp.get_tensor(out_d['index']).reshape(-1)
    return out


def sanity_check_export(export_model, tflite_path: str, X: np.ndarray, atol: float = 1e-4) -> Dict:
    import tensorflow as tf
    n = min(200, len(X))
    sample = X[:n].astype(np.float32)
    keras_p = export_model.predict(sample, verbose=0)
    tflite_p = tflite_predict(tflite_path, sample)
    diff = float(np.max(np.abs(keras_p - tflite_p)))
    return {
        'samples': n,
        'max_abs_diff': diff,
        'pass': bool(diff < atol),
        'atol': atol,
    }


# ---------------------------------------------------------------------------
# Backup + reports
# ---------------------------------------------------------------------------

def backup_existing_assets(run_dir: str, paths: List[str]) -> List[str]:
    backup_dir = os.path.join(run_dir, 'before')
    os.makedirs(backup_dir, exist_ok=True)
    saved = []
    for p in paths:
        if os.path.exists(p):
            dest = os.path.join(backup_dir, os.path.basename(p))
            shutil.copy2(p, dest)
            saved.append(dest)
    return saved


def write_kd_model_card(
    report: Dict, best_metrics: Dict, thresholds: Dict, out_path: str,
    *, best_model_name: str = 'kd_student', best_of_info: Optional[Dict] = None,
    cold_thresholds: Optional[Dict] = None,
    cold_threshold_info: Optional[Dict] = None,
    leak_free: bool = False,
) -> None:
    version_prefix = 'kd-mlp' if best_model_name == 'kd_student' else 'plain-mlp'
    if leak_free:
        version_prefix = f'{version_prefix}-leakfree'
    card = {
        'version': f"{version_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        'created_at': report['created_at'],
        'feature_count': report['feature_count'],
        'features': report['features'],
        'rows': report['rows'],
        'class_counts': report['class_counts'],
        'dataset_hash': report['dataset_hash'],
        'best_model': best_model_name,
        'best_of': best_of_info or {},
        'block_precision': float(best_metrics.get('BLOCK', {}).get('precision', 0.0)),
        'block_recall': float(best_metrics.get('BLOCK', {}).get('recall', 0.0)),
        'roc_auc_ovr': best_metrics.get('roc_auc_ovr'),
        'thresholds': {
            'block_threshold': float(thresholds.get('block_threshold', 0.5)),
            'warn_threshold': float(thresholds.get('warn_threshold', 0.3)),
            'block_precision': float(thresholds.get('block_precision', 0.0)),
            'block_recall': float(thresholds.get('block_recall', 0.0)),
            'block_f1': float(thresholds.get('block_f1', 0.0)),
            'warn_f1': float(thresholds.get('warn_f1', 0.0)),
        },
        'smote_applied': report.get('smote_applied', False),
        'kd': {
            'T': float(report.get('kd', {}).get('T', 0.0)),
            'alpha': float(report.get('kd', {}).get('alpha', 0.0)),
            'teacher': '+'.join(report.get('training_config', {}).get('teacher_components', ['catboost'])) or 'catboost_multiclass',
            'teacher_train_per_class': report.get('teacher_train_per_class'),
            'student_train_per_class': report.get('student_train_per_class'),
        },
        'cold_start': {
            'eval_size': report.get('cold_start_slice_size', 0),
            'tflite_metrics': report.get('test_metrics_cold_start_slice', {}).get('tflite_winner', {}),
            'tflite_metrics_cold_thresholded': report.get('test_metrics_cold_start_slice', {}).get(
                'tflite_winner_cold_thresholded', {}
            ),
        },
        'class_weights': {
            'allow': report.get('training_config', {}).get('allow_class_weight', 1.0),
            'warn': report.get('training_config', {}).get('warn_class_weight', 1.0),
            'block': report.get('training_config', {}).get('block_class_weight', 1.0),
        },
        'notes': f'Generated by scripts/train_kd_distillation.py (Phase 4B — {len(COMPACT_FEATURES)} features incl. multi-resolution prefix histograms + def_code×operator cross, cold-start aug, WARN re-weight, two-mode thresholds)',
    }
    if cold_thresholds is not None:
        # Phase 4A: cold thresholds picked on cold-view val. Android consumes this
        # block when noMetadata=1 AND there are no list-based hints (allow/blacklist).
        card['cold_thresholds'] = {
            'block_threshold': float(cold_thresholds.get('block_threshold', 0.5)),
            'warn_threshold': float(cold_thresholds.get('warn_threshold', 0.3)),
            'block_precision': float(cold_thresholds.get('block_precision', 0.0)),
            'block_recall': float(cold_thresholds.get('block_recall', 0.0)),
            'block_f1': float(cold_thresholds.get('block_f1', 0.0)),
            'warn_f1': float(cold_thresholds.get('warn_f1', 0.0)),
        }
        if cold_threshold_info:
            card['cold_thresholds']['tuning_info'] = cold_threshold_info
    if leak_free:
        # PR-1: Leak-free training. 9 metadata-фич всегда =0, noMetadata всегда =1
        # во всех сплитах (train/val/test). Делает warm-режим тождественным cold,
        # убирает train-test mismatch, отключает аугментацию (она тут вырождается).
        card['leak_free'] = {
            'enabled': True,
            'zeroed_features': list(COLD_START_MASK_FEATURES),
            'forced_no_metadata': True,
            'rationale': (
                'These 9 features are not available on-device without internet (whitelist/'
                'blacklist DB hits and online metadata). Training with them creates train-'
                'test mismatch, manifests as ALLOW precision collapse on cold-thresholded '
                'eval (~0.42 in baseline). Zeroing them at training time produces a model '
                'that behaves identically warm and cold, eliminating threshold drift.'
            ),
        }
        # В leak-free режиме cold_thresholds не пишутся (warm == cold).
        card.pop('cold_thresholds', None)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(card, f, ensure_ascii=False, indent=2)


def write_kd_report(report: Dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'kd_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    md_lines = [
        '# Knowledge Distillation report',
        '',
        f'- **Created**: {report["created_at"]}',
        f'- **Dataset**: {report["data"]} (sha256={report["dataset_hash"][:12]}…)',
        f'- **Rows**: {report["rows"]}',
        f'- **Class counts**: {report["class_counts"]}',
        f'- **Teacher train/class**: {report.get("teacher_train_per_class")}',
        f'- **Student train/class**: {report.get("student_train_per_class")}',
        f'- **Best T**: {report.get("kd", {}).get("T")}, α: {report.get("kd", {}).get("alpha")}',
        '',
        '## Comparison (test set)',
        '',
        '| Model | macro F1 | BLOCK P | BLOCK R | BLOCK F1 | WARN F1 | ROC-AUC OVR |',
        '|---|---|---|---|---|---|---|',
    ]
    for name, m in report.get('test_metrics', {}).items():
        md_lines.append(
            f'| {name} | {m.get("macro_f1", 0):.4f} '
            f'| {m.get("BLOCK", {}).get("precision", 0):.4f} '
            f'| {m.get("BLOCK", {}).get("recall", 0):.4f} '
            f'| {m.get("BLOCK", {}).get("f1", 0):.4f} '
            f'| {m.get("WARN", {}).get("f1", 0):.4f} '
            f'| {m.get("roc_auc_ovr") if m.get("roc_auc_ovr") is not None else "n/a"} |'
        )
    cold_metrics = report.get('test_metrics_cold_start_slice') or {}
    if cold_metrics:
        md_lines += [
            '',
            '## TRUE cold-start slice (all metadata zeroed, noMetadata=1)',
            '',
            'Phase 3: имитация совершенно нового номера — все reputation/reviews/categories/'
            'prefix-histogram/list-flags обнулены. Модель видит только operator/def_code/'
            'digit-entropy/prefixRisk. Это и есть «невидимый номер в проде».',
            '',
            '| Model | macro F1 | BLOCK P | BLOCK R | BLOCK F1 | WARN F1 |',
            '|---|---|---|---|---|---|',
        ]
        for name, m in cold_metrics.items():
            md_lines.append(
                f'| {name} | {m.get("macro_f1", 0):.4f} '
                f'| {m.get("BLOCK", {}).get("precision", 0):.4f} '
                f'| {m.get("BLOCK", {}).get("recall", 0):.4f} '
                f'| {m.get("BLOCK", {}).get("f1", 0):.4f} '
                f'| {m.get("WARN", {}).get("f1", 0):.4f} |'
            )
        op_points = report.get('cold_start_operating_points') or {}
        if op_points:
            md_lines += [
                '',
                '### Cold-start BLOCK operating points (precision floor → recall)',
                '',
                'Acceptance Phase 3: BLOCK precision ≥ 0.95 при recall ≥ 0.40.',
                '',
                '| Model | P≥0.95 (t / R / actualP) | P≥0.90 (t / R / actualP) | P≥0.80 (t / R / actualP) |',
                '|---|---|---|---|',
            ]
            for name, points in op_points.items():
                cells = []
                for floor in (0.95, 0.90, 0.80):
                    pt = points.get(f'P>={floor}', {})
                    cells.append(
                        f't={pt.get("threshold", 0):.2f} '
                        f'R={pt.get("recall", 0):.3f} '
                        f'(P={pt.get("precision", 0):.3f})'
                    )
                md_lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    md_lines += [
        '',
        '## Export',
        f'- **TFLite path**: {report.get("tflite", {}).get("path")}',
        f'- **TFLite bytes**: {report.get("tflite", {}).get("bytes")}',
        f'- **Sanity max|p_keras - p_tflite|**: {report.get("sanity", {}).get("max_abs_diff")}',
        f'- **Sanity passed**: {report.get("sanity", {}).get("pass")}',
        '',
        '## Thresholds (val-tuned, written to model_card.json)',
        f'- **warm** block_threshold = {report.get("thresholds", {}).get("block_threshold")}',
        f'- **warm** warn_threshold  = {report.get("thresholds", {}).get("warn_threshold")}',
    ]
    cold_thr = report.get('thresholds_cold')
    if cold_thr:
        md_lines += [
            f'- **cold** block_threshold = {cold_thr.get("block_threshold")} '
            f'(BLOCK_P={cold_thr.get("block_precision", 0):.3f}, '
            f'F1={cold_thr.get("block_f1", 0):.3f})',
            f'- **cold** warn_threshold  = {cold_thr.get("warn_threshold")} '
            f'(WARN_F1={cold_thr.get("warn_f1", 0):.3f})',
            '- Phase 4A: cold thresholds applied on-device when '
            '`noMetadata=1 AND inAllowlist=0 AND inBlacklist=0`.',
        ]
    md_lines += [
        '',
        '## Warnings',
    ]
    for w in report.get('warnings', []) or ['(none)']:
        md_lines.append(f'- {w}')
    with open(os.path.join(out_dir, 'kd_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    def _rows(metrics_dict: Dict) -> str:
        rows = ''
        for name, m in metrics_dict.items():
            rows += (
                f'<tr><td>{name}</td>'
                f'<td>{m.get("macro_f1", 0):.4f}</td>'
                f'<td>{m.get("BLOCK", {}).get("precision", 0):.4f}</td>'
                f'<td>{m.get("BLOCK", {}).get("recall", 0):.4f}</td>'
                f'<td>{m.get("BLOCK", {}).get("f1", 0):.4f}</td>'
                f'<td>{m.get("WARN", {}).get("f1", 0):.4f}</td>'
                f'<td>{m.get("roc_auc_ovr") if m.get("roc_auc_ovr") is not None else "—"}</td></tr>'
            )
        return rows

    html_rows = _rows(report.get('test_metrics', {}))
    unknown_rows = _rows(report.get('test_metrics_unknown_slice') or {})
    n_unknown = report.get('unknown_slice_size', 0)
    n_known = report.get('known_slice_size', 0)
    train_cfg = report.get('training_config', {})
    train_cfg_html = (
        f'<p><b>Training config</b>: '
        f'use_full_train={train_cfg.get("use_full_train")}, '
        f'mask_features={train_cfg.get("mask_features")}, '
        f'feature_mask_prob={train_cfg.get("feature_mask_prob")}, '
        f'weight_decay={train_cfg.get("weight_decay")}, '
        f'label_smoothing={train_cfg.get("label_smoothing")}</p>'
        if train_cfg else ''
    )
    unknown_section = ''
    if unknown_rows:
        unknown_section = (
            f'<h2>Comparison on UNKNOWN slice ({n_unknown}/{n_unknown + n_known} rows; '
            'inAllowlist=0 AND inBlacklist=0)</h2>'
            '<p>Это «честный» срез: только номера, у которых нет ни whitelist-, ни blacklist-подсказок. '
            'Так модель работает на свежих, ранее невиданных номерах в проде.</p>'
            '<table><tr><th>Model</th><th>macro F1</th><th>BLOCK P</th><th>BLOCK R</th>'
            '<th>BLOCK F1</th><th>WARN F1</th><th>ROC-AUC OVR</th></tr>'
            f'{unknown_rows}</table>'
        )
    with open(os.path.join(out_dir, 'kd_report.html'), 'w', encoding='utf-8') as f:
        f.write(
            '<!doctype html><html><head><meta charset="utf-8"><title>KD report</title>'
            '<style>body{font-family:system-ui;padding:24px;max-width:980px;margin:auto}'
            'table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px 10px}'
            'th{background:#f3f3f3}</style></head><body>'
            f'<h1>Knowledge Distillation</h1>'
            f'<p><b>Created</b>: {report["created_at"]}</p>'
            f'<p><b>Dataset</b>: {report["data"]}</p>'
            f'<p><b>Rows</b>: {report["rows"]} | <b>Class counts</b>: {report["class_counts"]}</p>'
            f'<p><b>Best T</b>: {report.get("kd", {}).get("T")}, <b>α</b>: {report.get("kd", {}).get("alpha")}</p>'
            f'{train_cfg_html}'
            '<h2>Comparison (full test set)</h2>'
            '<table><tr><th>Model</th><th>macro F1</th><th>BLOCK P</th><th>BLOCK R</th>'
            '<th>BLOCK F1</th><th>WARN F1</th><th>ROC-AUC OVR</th></tr>'
            f'{html_rows}</table>'
            f'{unknown_section}'
            f'<h2>Export</h2><p>TFLite: {report.get("tflite", {}).get("path")} '
            f'({report.get("tflite", {}).get("bytes")} bytes)</p>'
            f'<p>Sanity max|p_keras - p_tflite| = {report.get("sanity", {}).get("max_abs_diff")}'
            f' (pass={report.get("sanity", {}).get("pass")})</p>'
            '</body></html>'
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description='Knowledge Distillation: CatBoost → Keras MLP → TFLite')
    ap.add_argument('--data', default=DEFAULT_DATA)
    ap.add_argument('--tflite-output', default=DEFAULT_TFLITE)
    ap.add_argument('--model-card-output', default=DEFAULT_MODEL_CARD)
    ap.add_argument('--reports-dir', default=REPORTS_DIR)
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--teacher-train-per-class', type=int, default=6000,
                    help='Кол-во примеров на класс для teacher (CatBoost): 6k legit + 6k spam.')
    ap.add_argument('--student-train-per-class', type=int, default=4000,
                    help='Кол-во примеров на класс для student (KD-MLP): 4k legit + 4k spam (subset of teacher train).')
    ap.add_argument('--val-frac', type=float, default=0.10, help='Доля данных в val (стратифицированно).')
    ap.add_argument('--test-frac', type=float, default=0.10, help='Доля данных в test (стратифицированно).')
    ap.add_argument('--pad-with-smote', action='store_true',
                    help='Добивать teacher train SMOTE-ом до целевых цифр, если данных не хватает.')

    ap.add_argument('--teacher-iterations', type=int, default=500)
    ap.add_argument('--teacher-depth', type=int, default=6)
    ap.add_argument('--teacher-lr', type=float, default=0.05)

    ap.add_argument('--student-epochs', type=int, default=80)
    ap.add_argument('--student-batch', type=int, default=64)
    ap.add_argument('--student-patience', type=int, default=10)

    ap.add_argument('--T-grid', type=float, nargs='+', default=[2.0, 4.0, 8.0, 12.0],
                    help='Сетка температур для stage1. T=12 — мягче soft-targets, лучше для маленьких классов.')
    ap.add_argument('--alpha-grid', type=float, nargs='+', default=[0.3, 0.5, 0.7],
                    help='Сетка alpha для stage1.')
    ap.add_argument('--optuna-trials', type=int, default=30,
                    help='Кол-во Optuna trials в stage2 (lr/dropout/hidden). 0=пропустить.')

    ap.add_argument('--min-block-precision', type=float, default=0.85,
                    help='Минимально допустимая BLOCK precision при threshold tuning (warm).')
    ap.add_argument('--min-cold-block-precision', type=float, default=None,
                    help='Phase 4A: floor BLOCK precision для cold-thresholds. По умолчанию '
                         'наследуется от --min-block-precision (одинаковое требование к качеству '
                         'BLOCK для warm и cold). Можно понизить, если cold recall важнее точности.')
    ap.add_argument('--tune-cold-thresholds', action='store_true', default=True,
                    help='Phase 4A: тюнить отдельные block/warn thresholds на cold view '
                         'val (metadata зануляются, noMetadata=1). Записываются в '
                         'model_card.json в секцию `cold_thresholds`. Android-сторона '
                         'выбирает их на устройстве, когда noMetadata=1 и нет list-подсказок.')
    ap.add_argument('--no-tune-cold-thresholds', dest='tune_cold_thresholds', action='store_false',
                    help='Не тюнить cold thresholds (cold-сторона будет fallback на warm).')
    ap.add_argument('--allow-unsafe-export', action='store_true',
                    help='Экспортировать .tflite даже если sanity check проваливается.')
    ap.add_argument('--quantize', action='store_true',
                    help='Dynamic-range int8 quantization для весов (~4x меньше .tflite, ~1e-3 numerical drift). По умолчанию выключено: Android ждёт чистый FP32.')
    ap.add_argument('--sanity-atol', type=float, default=1e-4,
                    help='Допустимое расхождение между Keras FP32 и TFLite. Для quantize=True имеет смысл 5e-3.')
    ap.add_argument('--verbose', type=int, default=0, help='Keras verbose (0/1/2).')

    # Новые флаги: использовать весь трейн + feature masking + регуляризация.
    ap.add_argument('--use-full-train', action='store_true', default=True,
                    help='Использовать все доступные train-строки для teacher и student (по умолчанию включено). '
                         'Альтернатива — --no-use-full-train и стратифицированный sample по --teacher/student-train-per-class.')
    ap.add_argument('--no-use-full-train', dest='use_full_train', action='store_false',
                    help='Отключить «весь трейн», вернуться к старой логике 6k+6k / 4k+4k.')
    ap.add_argument('--mask-features', type=str,
                    default=('inAllowlist,inBlacklist,reputationScore,sourceConfidence,'
                             'reviewsLog,negativeRatio,searchVolumeLog,'
                             'hasFraudCategory,hasTelemarketingCategory'),
                    help='Список фич (через запятую), которые в augmentation-копиях будут обнуляться. '
                         'Phase 3: по умолчанию — online metadata фичи (репутация, отзывы, категории, '
                         'allow/blacklist). Шипимые JSON-лукапы (prefix histogram, def_code_risk, '
                         'operator) ОСТАЮТСЯ — они и есть offline-сигнал. Это «true cold-start aug»: '
                         'модель обучается распознавать BLOCK даже когда онлайн-метаданных нет, опираясь '
                         'на shipped lookups + operator/'
                         'def_code/digit-entropy. inAllowlist/inBlacklist маскируются, чтобы '
                         'модель не выучивала список как self-fulfilling prophecy. '
                         'noMetadata=1 при этом сохраняется как явный cold-start индикатор.')
    ap.add_argument('--feature-mask-prob', type=float, default=1.0,
                    help='Доля строк (0..1), для которых добавляется masked-копия в обучающий набор. '
                         'Phase 4C: 1.0 (default) → +100%% rows (полное удвоение). Phase 3 default '
                         'был 0.5. Stratified=True (см. --no-stratified-aug) сохраняет class balance.')
    ap.add_argument('--cold-aug-mode', type=str, default='legacy', choices=['balanced', 'legacy'],
                    help='Phase 4C: режим cold-start augmentation. '
                         '  "legacy" (default) — расширяем все классы пропорционально '
                         '(stratified subsample при mask_prob<1, full doubling при 1.0), с '
                         'опциональным weight_mult на masked рядах. Сохраняет class proportions. '
                         '  "balanced" — экспериментальный режим: маскируем только spam-классы '
                         '(BLOCK + WARN) с опциональным WARN oversampling. Может ломать модель '
                         'если веса BLOCK значительно превышают ALLOW в effective loss '
                         '(см. retrain notes Phase 4C).')
    ap.add_argument('--cold-aug-weight-multiplier', type=float, default=1.5,
                    help='Phase 4C: per-row множитель weight на masked рядах (>=1). '
                         '1.0=без буста (Phase 3), 1.5 (default)=масked рядам ×1.5 effective loss. '
                         'Применяется в обоих режимах (legacy и balanced).')
    ap.add_argument('--cold-aug-warn-oversample-cap', type=float, default=1.0,
                    help='Phase 4C: cap на oversampling меньшинств (WARN) в balanced режиме. '
                         '1.0 (default)=без oversampling. Не применяется в legacy режиме.')
    ap.add_argument('--no-stratified-aug', dest='stratified_aug', action='store_false', default=True,
                    help='Phase 4C: отключить stratified subsample при mask_prob<1 в legacy режиме. '
                         'По умолчанию выборка идёт пропорционально классам.')
    ap.add_argument('--weight-decay', type=float, default=1e-4,
                    help='L2 весовых ядер (kernel_regularizer). 0 = отключено.')
    ap.add_argument('--label-smoothing', type=float, default=0.05,
                    help='Label smoothing в хард-CE части KD-лосса (0 = отключено).')

    # Phase 3: новые флаги для re-weighting / ensemble teacher / cold-start eval / архитектуры.
    ap.add_argument('--warn-class-weight', type=float, default=3.0,
                    help='Per-sample weight для WARN класса (3.0 → 3x). Без этого WARN '
                         'схлопывается в ALLOW из-за дисбаланса (3.5k vs 80k).')
    ap.add_argument('--block-class-weight', type=float, default=1.0,
                    help='Per-sample weight для BLOCK (по умолчанию 1.0).')
    ap.add_argument('--allow-class-weight', type=float, default=1.0,
                    help='Per-sample weight для ALLOW.')
    ap.add_argument('--use-lightgbm-teacher', action='store_true',
                    help='Phase 3: добавить LightGBM teacher в ансамбль (CatBoost + LightGBM, '
                         'усреднение proba). Помогает на WARN/ALLOW границе.')
    ap.add_argument('--lightgbm-iterations', type=int, default=600,
                    help='LightGBM n_estimators (если --use-lightgbm-teacher).')
    ap.add_argument('--cold-start-eval', action='store_true', default=True,
                    help='Включить true-cold-start eval slice: на test set обнуляем все '
                         'metadata-фичи и считаем BLOCK precision/recall. По умолчанию on.')
    ap.add_argument('--no-cold-start-eval', dest='cold_start_eval', action='store_false',
                    help='Выключить true-cold-start eval slice.')
    ap.add_argument('--hidden-sizes', type=str, default='96,64,32',
                    help='Стартовые hidden sizes для plain MLP / stage1 grid (3 числа через запятую). '
                         'Phase 3: по умолчанию 96,64,32 (вместо старых 64,48,24) — больше capacity '
                         'для 47 фич.')

    # PR-1: leak-free режим. Все 9 metadata-фич COLD_START_MASK_FEATURES физически
    # обнуляются во ВСЕХ split'ах (train/val/test) ДО любого сэмплирования и
    # аугментации. noMetadata форсится в 1. Эффект:
    #   * модель никогда не видит leakage-фичи (которые на устройстве недоступны),
    #     поэтому warm/cold режимы вырождаются в один.
    #   * cold-aug-mode игнорируется (уже всё cold).
    #   * --tune-cold-thresholds игнорируется (warm thresholds = cold thresholds).
    # Output по умолчанию идёт в app/src/main/assets/experimental/, прод не трогаем.
    ap.add_argument('--leak-free', action='store_true', default=False,
                    help='PR-1: physically zero out 9 metadata leakage features '
                         '(reputationScore, sourceConfidence, reviewsLog, negativeRatio, '
                         'searchVolumeLog, hasFraudCategory, hasTelemarketingCategory, '
                         'inAllowlist, inBlacklist) in ALL splits BEFORE training. '
                         'Forces noMetadata=1. By default writes to '
                         'app/src/main/assets/experimental/spam_model_leak_free.tflite '
                         'so production is not touched.')

    args = ap.parse_args()

    set_global_seed(args.seed)

    # --- Data ---
    print(f'Loading {args.data}...')
    X, y = load_csv(args.data)
    if X.shape[1] != len(COMPACT_FEATURES):
        raise SystemExit(f'Feature mismatch: {X.shape[1]} vs {len(COMPACT_FEATURES)}')
    counts = class_counts(y)
    print(f'  rows={len(y)}, class counts: {counts}')

    # --- PR-1: leak-free mode ---
    # Применяем cold-view ко ВСЕМУ X сразу (train+val+test), чтобы обучение и
    # все eval-метрики были полностью «honest cold». Это устраняет train-test
    # mismatch: warm-фичи модель не видит вообще. Также форсим safe пути по
    # умолчанию для tflite/model_card в experimental/, чтобы прод не подменялся
    # случайно.
    if args.leak_free:
        leak_free_features = list(COLD_START_MASK_FEATURES)
        leak_free_indices = feature_mask_indices(leak_free_features)
        no_meta_idx = COMPACT_FEATURES.index('noMetadata') if 'noMetadata' in COMPACT_FEATURES else -1
        X = make_cold_view(X, leak_free_indices, no_meta_idx)
        # Sanity-check: эти колонки теперь все нули (no_meta — все единицы).
        for idx, name in zip(leak_free_indices, leak_free_features):
            col_max = float(np.abs(X[:, idx]).max()) if len(X) > 0 else 0.0
            assert col_max == 0.0, f'leak-free: {name} (idx={idx}) not zeroed (max={col_max})'
        if no_meta_idx >= 0:
            assert float(X[:, no_meta_idx].min()) == 1.0, 'leak-free: noMetadata not forced to 1'
        print(f'  [PR-1 leak-free] zeroed {len(leak_free_indices)} features '
              f'({leak_free_features}); forced noMetadata=1.')
        # В leak-free режиме cold==warm, augmentation вырождается, cold-tuning не нужен.
        if args.cold_aug_mode != 'legacy' or args.feature_mask_prob > 0:
            print(f'  [PR-1 leak-free] disabling masked augmentation (X is already cold).')
        args.feature_mask_prob = 0.0
        args.cold_aug_mode = 'legacy'
        args.tune_cold_thresholds = False
        # Безопасные дефолтные пути в experimental/, если пользователь не задал явно.
        experimental_dir = os.path.join(ASSETS_DIR, 'experimental')
        if args.tflite_output == DEFAULT_TFLITE:
            os.makedirs(experimental_dir, exist_ok=True)
            args.tflite_output = os.path.join(experimental_dir, 'spam_model_leak_free.tflite')
            print(f'  [PR-1 leak-free] tflite output → {args.tflite_output}')
        if args.model_card_output == DEFAULT_MODEL_CARD:
            os.makedirs(experimental_dir, exist_ok=True)
            args.model_card_output = os.path.join(experimental_dir, 'model_card_leak_free.json')
            print(f'  [PR-1 leak-free] model card → {args.model_card_output}')

    train_size = 1.0 - args.val_frac - args.test_frac
    if train_size <= 0:
        raise SystemExit('val_frac + test_frac must be < 1.0')
    train_idx, val_idx, test_idx = stratified_split(
        X, y, sizes=(train_size, args.val_frac, args.test_frac), seed=args.seed,
    )
    print(f'  split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

    # --- Teacher / student train sampling ---
    if args.use_full_train:
        teacher_idx = train_idx
        student_idx = train_idx
        teacher_counts = class_counts(y[teacher_idx])
        student_counts = class_counts(y[student_idx])
        warns_t: List[str] = []
        warns_s: List[str] = []
        print(f'  teacher train: ALL {len(teacher_idx)} rows {teacher_counts} («full-train» mode)')
        print(f'  student train: ALL {len(student_idx)} rows {student_counts} («full-train» mode)')
    else:
        teacher_idx, teacher_counts, warns_t = sample_teacher_train(
            train_idx, y,
            legit_target=args.teacher_train_per_class,
            spam_target=args.teacher_train_per_class,
            seed=args.seed,
        )
        print(f'  teacher train sample: {len(teacher_idx)} rows {teacher_counts}')
        for w in warns_t:
            print(f'    WARN: {w}')

        student_idx, student_counts, warns_s = sample_student_train_subset(
            teacher_idx, y,
            legit_target=args.student_train_per_class,
            spam_target=args.student_train_per_class,
            seed=args.seed,
        )
        print(f'  student train sample: {len(student_idx)} rows {student_counts}')
        for w in warns_s:
            print(f'    WARN: {w}')

    X_teacher, y_teacher = X[teacher_idx], y[teacher_idx]
    X_student, y_student = X[student_idx], y[student_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # --- Feature masking config (вынуждённое «забывание» подсказок) ---
    mask_feature_names: List[str] = [s.strip() for s in args.mask_features.split(',') if s.strip()]
    mask_indices = feature_mask_indices(mask_feature_names)
    mask_prob = float(args.feature_mask_prob) if mask_indices else 0.0
    if mask_indices and mask_prob > 0:
        print(f'  feature masking: prob={mask_prob:.2f} on '
              f'{[COMPACT_FEATURES[i] for i in mask_indices]} (training only)')
    else:
        print(f'  feature masking: OFF (mask_features={mask_feature_names}, mask_prob={mask_prob})')
    if args.weight_decay > 0 or args.label_smoothing > 0:
        print(f'  regularization: weight_decay={args.weight_decay:g} label_smoothing={args.label_smoothing:g}')

    smote_info: Dict = {}
    if args.pad_with_smote:
        target = {
            LABEL_TO_ID['ALLOW']: args.teacher_train_per_class,
            LABEL_TO_ID['WARN']: int(round(args.teacher_train_per_class * 0.22)),
            LABEL_TO_ID['BLOCK']: int(round(args.teacher_train_per_class * 0.78)),
        }
        X_teacher, y_teacher, smote_info = maybe_pad_with_smote(
            X_teacher, y_teacher, target, seed=args.seed,
        )
        print(f'  SMOTE: {smote_info}')

    # --- Teacher-aware feature masking augmentation ---
    # Дублируем часть строк с обнулёнными list-фичами и в teacher train, и в student train.
    # Teacher теперь учится на согласованном множестве (X, masked_X), поэтому soft-targets,
    # которые student видит при KD, отражают тот же mask-режим, что и сам student-вход.
    cold_aug_mode = str(args.cold_aug_mode).lower()
    if cold_aug_mode == 'balanced':
        # Phase 4C: selective spam-classes masking + weight bump on masked rows.
        spam_target_classes = (LABEL_TO_ID['BLOCK'], LABEL_TO_ID['WARN'])
        X_teacher_aug, y_teacher_aug, teacher_mask_extra, teacher_aug_info = concat_masked_aug_balanced(
            X_teacher, y_teacher, mask_indices,
            target_classes=spam_target_classes,
            weight_multiplier=args.cold_aug_weight_multiplier,
            seed=args.seed,
            warn_oversample_cap=args.cold_aug_warn_oversample_cap,
        )
        X_student_aug, y_student_aug, student_mask_extra, student_aug_info = concat_masked_aug_balanced(
            X_student, y_student, mask_indices,
            target_classes=spam_target_classes,
            weight_multiplier=args.cold_aug_weight_multiplier,
            seed=args.seed + 1,
            warn_oversample_cap=args.cold_aug_warn_oversample_cap,
        )
    elif cold_aug_mode == 'legacy':
        X_teacher_aug, y_teacher_aug, teacher_mask_extra, teacher_aug_info = concat_masked_aug(
            X_teacher, y_teacher, mask_indices, mask_prob, seed=args.seed,
            weight_multiplier=args.cold_aug_weight_multiplier,
            stratified=bool(args.stratified_aug),
        )
        X_student_aug, y_student_aug, student_mask_extra, student_aug_info = concat_masked_aug(
            X_student, y_student, mask_indices, mask_prob, seed=args.seed + 1,
            weight_multiplier=args.cold_aug_weight_multiplier,
            stratified=bool(args.stratified_aug),
        )
    else:
        raise SystemExit(f'--cold-aug-mode unknown: {cold_aug_mode!r} (expected legacy|balanced)')
    if mask_indices:
        print(f'  cold-aug mode={cold_aug_mode} '
              f'teacher {teacher_aug_info["orig_rows"]} → '
              f'{teacher_aug_info["total"]} (+{teacher_aug_info["masked_added"]} masked); '
              f'student {student_aug_info["orig_rows"]} → {student_aug_info["total"]} '
              f'(+{student_aug_info["masked_added"]} masked)')
        if cold_aug_mode == 'balanced':
            print(f'  cold-aug per-class added: {teacher_aug_info.get("per_class_added")} '
                  f'weight_mult={args.cold_aug_weight_multiplier}')

    # --- Phase 3: per-sample weights (WARN re-weight) ---
    # Phase 4C: масштабируем class-weights на mask_extra множитель (для masked рядов >1.0).
    teacher_sw = class_weights_for_y(
        y_teacher_aug,
        warn_weight=args.warn_class_weight,
        block_weight=args.block_class_weight,
        allow_weight=args.allow_class_weight,
    ) * teacher_mask_extra
    student_sw = class_weights_for_y(
        y_student_aug,
        warn_weight=args.warn_class_weight,
        block_weight=args.block_class_weight,
        allow_weight=args.allow_class_weight,
    ) * student_mask_extra
    print(f'  class weights: WARN={args.warn_class_weight} BLOCK={args.block_class_weight} '
          f'ALLOW={args.allow_class_weight}')

    # Phase 3: парсинг hidden_sizes из CLI.
    try:
        baseline_hidden = tuple(int(x.strip()) for x in args.hidden_sizes.split(',') if x.strip())
        if len(baseline_hidden) < 2:
            raise ValueError('need at least 2 hidden sizes')
    except Exception as exc:
        raise SystemExit(f'--hidden-sizes parse error: {exc}; got {args.hidden_sizes!r}')
    print(f'  baseline hidden_sizes: {baseline_hidden}')

    # --- Teacher (на concat(X, masked_X)) ---
    teacher_kind = 'catboost'
    if args.use_lightgbm_teacher:
        teacher_kind = 'catboost+lightgbm ensemble'
    print(f'\n[1/5] Training teacher ({teacher_kind}) on unmasked + masked-augmented set...')
    t0 = time.time()
    catboost_teacher = train_catboost_teacher(
        X_teacher_aug, y_teacher_aug, X_val, y_val, seed=args.seed,
        iterations=args.teacher_iterations, depth=args.teacher_depth,
        learning_rate=args.teacher_lr,
        sample_weight=teacher_sw,
    )
    teacher_components = {'catboost': catboost_teacher}
    if args.use_lightgbm_teacher:
        lgb_teacher = train_lightgbm_teacher(
            X_teacher_aug, y_teacher_aug, X_val, y_val, seed=args.seed,
            n_estimators=args.lightgbm_iterations,
            sample_weight=teacher_sw,
        )
        if lgb_teacher is None:
            print('  WARN: lightgbm not installed — falling back to single CatBoost teacher')
        else:
            teacher_components['lightgbm'] = lgb_teacher
    if len(teacher_components) > 1:
        teacher = TeacherEnsemble(list(teacher_components.values()))
    else:
        teacher = catboost_teacher
    print(f'  done in {time.time() - t0:.1f}s ({len(teacher_components)} teacher(s))')
    teacher_proba_test = teacher.predict_proba(X_test)
    teacher_proba_val = teacher.predict_proba(X_val)
    teacher_metrics_test = evaluate_proba(y_test, teacher_proba_test)
    teacher_metrics_val = evaluate_proba(y_val, teacher_proba_val)
    print(f'  teacher val macroF1={teacher_metrics_val["macro_f1"]:.4f} '
          f'BLOCK_P={teacher_metrics_val["BLOCK"]["precision"]:.3f}')

    # --- Plain MLP baseline (no KD) — обучаем на тех же augmented данных, что и student ---
    print('\n[2/5] Training plain MLP baseline (no KD, on augmented set)...')
    t0 = time.time()
    plain = train_plain_mlp(
        X_student_aug, y_student_aug, X_val, y_val,
        hidden_sizes=baseline_hidden, dropout=0.2, lr=1e-3,
        epochs=args.student_epochs, batch_size=args.student_batch,
        patience=args.student_patience, seed=args.seed, verbose=args.verbose,
        weight_decay=args.weight_decay,
        sample_weight=student_sw,
    )
    print(f'  done in {time.time() - t0:.1f}s')

    def plain_proba(X):
        logits = plain.predict(X.astype(np.float32), verbose=0)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

    plain_metrics_test = evaluate_proba(y_test, plain_proba(X_test))
    plain_metrics_val = evaluate_proba(y_val, plain_proba(X_val))
    print(f'  plain MLP val macroF1={plain_metrics_val["macro_f1"]:.4f} '
          f'BLOCK_P={plain_metrics_val["BLOCK"]["precision"]:.3f}')

    # --- Stage 1: grid (T, alpha) — student обучается на augmented set, soft-targets — оттуда же ---
    print('\n[3/5] KD stage1 grid search (T, alpha)...')
    t0 = time.time()
    stage1 = stage1_grid_search(
        teacher, X_student_aug, y_student_aug, X_val, y_val,
        hidden_sizes=baseline_hidden, dropout=0.2, lr=1e-3,
        epochs=args.student_epochs, batch_size=args.student_batch,
        patience=args.student_patience, seed=args.seed,
        Ts=args.T_grid, alphas=args.alpha_grid,
        min_block_precision=args.min_block_precision, verbose=args.verbose,
        weight_decay=args.weight_decay, label_smoothing=args.label_smoothing,
        sample_weight=student_sw,
    )
    best_T = stage1['best']['T']
    best_alpha = stage1['best']['alpha']
    print(f'  stage1 best T={best_T} α={best_alpha} in {time.time() - t0:.1f}s')

    # --- Stage 2: Optuna over lr/dropout/hidden ---
    stage2 = {}
    best_lr = 1e-3
    best_dropout = 0.2
    best_hidden = baseline_hidden if len(baseline_hidden) >= 3 else (96, 64, 32)
    if args.optuna_trials > 0:
        print('\n[4/5] KD stage2 Optuna (lr, dropout, hidden)...')
        t0 = time.time()
        stage2 = stage2_optuna_search(
            teacher, X_student_aug, y_student_aug, X_val, y_val,
            T=best_T, alpha=best_alpha,
            n_trials=args.optuna_trials,
            epochs=args.student_epochs, batch_size=args.student_batch,
            patience=args.student_patience, seed=args.seed,
            min_block_precision=args.min_block_precision, verbose=args.verbose,
            weight_decay=args.weight_decay, label_smoothing=args.label_smoothing,
            sample_weight=student_sw,
        )
        bp = stage2.get('best_params', {})
        if bp:
            best_lr = float(bp['lr'])
            best_dropout = float(bp['dropout'])
            best_hidden = (int(bp['h0']), int(bp['h1']), int(bp['h2']))
        print(f'  stage2 done in {time.time() - t0:.1f}s, best_params={bp}')
    else:
        print('\n[4/5] Optuna stage2 disabled (--optuna-trials 0)')

    # --- Final student with best config ---
    print('\n[5/5] Training final student with best config (on augmented set)...')
    soft_final = teacher_soft_targets(teacher, X_student_aug, best_T)
    final_backbone, _, _ = train_student(
        X_student_aug, y_student_aug, soft_final, X_val, y_val,
        T=best_T, alpha=best_alpha, hidden_sizes=best_hidden, dropout=best_dropout,
        lr=best_lr, epochs=args.student_epochs, batch_size=args.student_batch,
        patience=args.student_patience, seed=args.seed, verbose=args.verbose,
        weight_decay=args.weight_decay, label_smoothing=args.label_smoothing,
        sample_weight=student_sw,
    )
    proba_val = proba_from_backbone(final_backbone, X_val)
    proba_test = proba_from_backbone(final_backbone, X_test)
    student_metrics_test_argmax = evaluate_proba(y_test, proba_test)

    # --- Threshold tuning on val (для kd_student) ---
    thresholds_kd = tune_thresholds(y_val, proba_val, min_block_precision=args.min_block_precision)
    print(f'  kd_student thresholds: block={thresholds_kd["block_threshold"]:.3f} '
          f'warn={thresholds_kd["warn_threshold"]:.3f} '
          f'(BLOCK_P={thresholds_kd["block_precision"]:.3f}, '
          f'BLOCK_F1={thresholds_kd["block_f1"]:.3f})')
    student_metrics_test = evaluate_proba(y_test, proba_test, thresholds=thresholds_kd)

    # --- Tune thresholds for plain MLP too — для honest сравнения ---
    plain_proba_val_arr = plain_proba(X_val)
    thresholds_plain = tune_thresholds(y_val, plain_proba_val_arr, min_block_precision=args.min_block_precision)
    print(f'  plain_mlp thresholds: block={thresholds_plain["block_threshold"]:.3f} '
          f'warn={thresholds_plain["warn_threshold"]:.3f} '
          f'(BLOCK_P={thresholds_plain["block_precision"]:.3f}, '
          f'BLOCK_F1={thresholds_plain["block_f1"]:.3f})')

    # --- Phase 4A: cold threshold tuning (на cold-view val) ---
    # Считаем proba на cold-view val (metadata→0, noMetadata→1) и тюним отдельные
    # block/warn пороги. Это решает threshold-mismatch: warm-tuned порог BLOCK режет
    # recall на cold-start с argmax 84% до thresholded 30%. Cold-tuned порог
    # ориентируется на распределение proba именно при отсутствии metadata.
    thresholds_kd_cold: Optional[Dict] = None
    thresholds_plain_cold: Optional[Dict] = None
    cold_threshold_info: Dict = {}
    if args.tune_cold_thresholds:
        cold_mask_idx = feature_mask_indices(list(COLD_START_MASK_FEATURES))
        no_meta_idx = COMPACT_FEATURES.index('noMetadata') if 'noMetadata' in COMPACT_FEATURES else -1
        X_val_cold = make_cold_view(X_val, cold_mask_idx, no_meta_idx)
        proba_val_kd_cold = proba_from_backbone(final_backbone, X_val_cold)
        proba_val_plain_cold = plain_proba(X_val_cold)
        min_cold_p = (
            args.min_cold_block_precision if args.min_cold_block_precision is not None
            else args.min_block_precision
        )
        thresholds_kd_cold = tune_thresholds(y_val, proba_val_kd_cold, min_block_precision=min_cold_p)
        thresholds_plain_cold = tune_thresholds(y_val, proba_val_plain_cold, min_block_precision=min_cold_p)
        cold_threshold_info = {
            'mask_features': list(COLD_START_MASK_FEATURES),
            'no_meta_set_to_1': no_meta_idx >= 0,
            'min_cold_block_precision': float(min_cold_p),
            'val_rows': int(len(y_val)),
        }
        print(
            f'  [Phase 4A] cold thresholds (val cold-view, n={len(y_val)}, '
            f'floor BLOCK_P>={min_cold_p:.2f}):\n'
            f'    kd_student_cold:  block={thresholds_kd_cold["block_threshold"]:.3f} '
            f'warn={thresholds_kd_cold["warn_threshold"]:.3f} '
            f'(BLOCK_P={thresholds_kd_cold["block_precision"]:.3f}, '
            f'F1={thresholds_kd_cold["block_f1"]:.3f})\n'
            f'    plain_mlp_cold:   block={thresholds_plain_cold["block_threshold"]:.3f} '
            f'warn={thresholds_plain_cold["warn_threshold"]:.3f} '
            f'(BLOCK_P={thresholds_plain_cold["block_precision"]:.3f}, '
            f'F1={thresholds_plain_cold["block_f1"]:.3f})'
        )

    # --- Best-of selection: plain_mlp vs kd_student по unknown-slice macroF1 (val) ---
    # Это «прод-метрика»: качество на номерах без list-подсказок (inAllowlist=0 AND inBlacklist=0).
    in_allow_idx = COMPACT_FEATURES.index('inAllowlist')
    in_block_idx = COMPACT_FEATURES.index('inBlacklist')
    val_unknown_mask = (X_val[:, in_allow_idx] == 0.0) & (X_val[:, in_block_idx] == 0.0)
    n_val_unknown = int(val_unknown_mask.sum())
    if n_val_unknown == 0:
        # Нет unknown-номеров на val — фолбэк на val argmax macroF1
        kd_val_score = evaluate_proba(y_val, proba_val).get('macro_f1', 0.0)
        plain_val_score = evaluate_proba(y_val, plain_proba_val_arr).get('macro_f1', 0.0)
        selection_basis = 'val_full_argmax_macroF1 (no unknown rows in val)'
    else:
        kd_val_score = evaluate_proba(
            y_val[val_unknown_mask], proba_val[val_unknown_mask], thresholds=thresholds_kd,
        ).get('macro_f1', 0.0)
        plain_val_score = evaluate_proba(
            y_val[val_unknown_mask], plain_proba_val_arr[val_unknown_mask], thresholds=thresholds_plain,
        ).get('macro_f1', 0.0)
        selection_basis = f'val_unknown_macroF1 (n={n_val_unknown})'
    if kd_val_score >= plain_val_score:
        best_model_name = 'kd_student'
        thresholds = thresholds_kd
        thresholds_cold = thresholds_kd_cold
        winner_proba_test = proba_test
        winner_keras = final_backbone
    else:
        best_model_name = 'plain_mlp'
        thresholds = thresholds_plain
        thresholds_cold = thresholds_plain_cold
        winner_proba_test = plain_proba(X_test)
        winner_keras = plain
    print(
        f'\n=== Best-of selection ({selection_basis}) ===\n'
        f'  plain_mlp:  {plain_val_score:.4f}\n'
        f'  kd_student: {kd_val_score:.4f}\n'
        f'  >> exporting: {best_model_name} (thresholds: '
        f'block={thresholds["block_threshold"]:.3f}, warn={thresholds["warn_threshold"]:.3f})'
    )

    # --- Run dir + backup ---
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(args.reports_dir, f'kd_{run_id}')
    os.makedirs(run_dir, exist_ok=True)
    backed = backup_existing_assets(run_dir, [args.tflite_output, args.model_card_output])
    if backed:
        print(f'  backed up old assets: {backed}')

    # --- Export winner to TFLite ---
    print(f'\nExporting TFLite (winner = {best_model_name})...')
    export_info = export_tflite(winner_keras, args.tflite_output, quantize=args.quantize)
    print(f'  wrote {export_info["bytes"]} bytes -> {export_info["path"]}')

    # --- Sanity check ---
    sanity = sanity_check_export(
        export_info['export_model'], args.tflite_output, X_test, atol=args.sanity_atol,
    )
    print(f'  sanity: max_abs_diff={sanity["max_abs_diff"]:.6f} pass={sanity["pass"]}')
    if not sanity['pass'] and not args.allow_unsafe_export:
        print('  !! sanity check FAILED — restoring backup and exiting non-zero')
        for src in backed:
            dst = os.path.join(ASSETS_DIR, os.path.basename(src))
            shutil.copy2(src, dst)
        return 2

    # --- TFLite test metrics (winner) ---
    tflite_proba_test = tflite_predict(args.tflite_output, X_test)
    tflite_metrics_test = evaluate_proba(y_test, tflite_proba_test, thresholds=thresholds)
    plain_metrics_test_thresholded = evaluate_proba(y_test, plain_proba(X_test), thresholds=thresholds_plain)

    # --- Unknown-numbers slice: rows where both list-flags = 0 ---
    # Это «честное» число: метрики на номерах, у которых нет подсказок ни от whitelist, ни от blacklist.
    # Именно так модель работает в проде на свежих, неизвестных номерах.
    unknown_mask = (X_test[:, in_allow_idx] == 0.0) & (X_test[:, in_block_idx] == 0.0)
    n_unknown = int(unknown_mask.sum())
    test_metrics_unknown: Dict = {}
    test_metrics_known: Dict = {}
    if n_unknown > 0:
        unknown_counts = class_counts(y_test[unknown_mask])
        print(f'\n=== Unknown-numbers slice (inAllowlist=0 AND inBlacklist=0) ===')
        print(f'  n={n_unknown}/{len(y_test)} rows, classes={unknown_counts}')
        plain_proba_test_full = plain_proba(X_test)
        test_metrics_unknown = {
            'catboost_teacher': evaluate_proba(y_test[unknown_mask], teacher_proba_test[unknown_mask]),
            'plain_mlp_argmax': evaluate_proba(y_test[unknown_mask], plain_proba_test_full[unknown_mask]),
            'plain_mlp_thresholded': evaluate_proba(
                y_test[unknown_mask], plain_proba_test_full[unknown_mask], thresholds=thresholds_plain,
            ),
            'kd_student_argmax': evaluate_proba(y_test[unknown_mask], proba_test[unknown_mask]),
            'kd_student_thresholded': evaluate_proba(
                y_test[unknown_mask], proba_test[unknown_mask], thresholds=thresholds_kd,
            ),
            'tflite_winner': evaluate_proba(
                y_test[unknown_mask], tflite_proba_test[unknown_mask], thresholds=thresholds,
            ),
        }
    known_mask = ~unknown_mask
    n_known = int(known_mask.sum())
    if n_known > 0 and n_unknown > 0:
        test_metrics_known = {
            'tflite_winner': evaluate_proba(
                y_test[known_mask], tflite_proba_test[known_mask], thresholds=thresholds,
            ),
        }

    # --- Phase 3: TRUE cold-start eval slice ---
    # Берём весь test set, обнуляем все metadata-фичи (всё, что есть в --mask-features
    # + reputationScore + sourceConfidence + reviewsLog + negative/search + categories +
    # prefix histogram + inAllowlist/inBlacklist) и поднимаем noMetadata=1.
    # Это имитация «совершенно нового номера»: модель должна увидеть только operator,
    # def_code, digit-entropy, prefixRisk (он остаётся, т.к. это пер-DEF-кода фолбэк), и
    # сделать предсказание. Старая модель здесь падала в ALLOW почти на всём — отсюда
    # cold-start проблема.
    cold_metrics: Dict = {}
    cold_op_points: Dict = {}
    if args.cold_start_eval:
        print('\n=== TRUE cold-start eval (online metadata zeroed, noMetadata=1) ===')
        # Phase 3/4: список фич, которые НА РУНТАЙМЕ недоступны без интернета —
        # репутация, отзывы, категории, белый/чёрный списки. Шипимые JSON-лукапы
        # (prefix histogram, def_code_risk, operator bucket) ОСТАЮТСЯ — они и есть
        # «cold-start сигнал», на который мы опираемся offline. defCodeRisk и
        # prefixRisk тоже остаются (это per-DEF-кода и легаси-фолбэки).
        cold_mask_names = list(COLD_START_MASK_FEATURES)
        cold_mask_idx = feature_mask_indices(cold_mask_names)
        no_meta_idx = COMPACT_FEATURES.index('noMetadata') if 'noMetadata' in COMPACT_FEATURES else -1

        X_test_cold = make_cold_view(X_test, cold_mask_idx, no_meta_idx)

        # Прогоняем через все три модели (CatBoost teacher, plain MLP, KD student) + tflite.
        teacher_cold_proba = teacher.predict_proba(X_test_cold)
        plain_cold_proba = plain_proba(X_test_cold)
        kd_cold_proba = proba_from_backbone(final_backbone, X_test_cold)
        tflite_cold_proba = tflite_predict(args.tflite_output, X_test_cold)

        cold_counts = class_counts(y_test)
        print(f'  rows={len(y_test)} class_counts={cold_counts} '
              f'mask_features={cold_mask_names}, noMetadata→1')
        cold_metrics = {
            'catboost_teacher': evaluate_proba(y_test, teacher_cold_proba),
            'plain_mlp_argmax': evaluate_proba(y_test, plain_cold_proba),
            'plain_mlp_thresholded': evaluate_proba(y_test, plain_cold_proba, thresholds=thresholds_plain),
            'kd_student_argmax': evaluate_proba(y_test, kd_cold_proba),
            'kd_student_thresholded': evaluate_proba(y_test, kd_cold_proba, thresholds=thresholds_kd),
            'tflite_winner': evaluate_proba(y_test, tflite_cold_proba, thresholds=thresholds),
        }
        # Phase 4A: side-by-side warm-on-cold vs cold-on-cold для победителя.
        if thresholds_cold is not None:
            cold_metrics['plain_mlp_cold_thresholded'] = evaluate_proba(
                y_test, plain_cold_proba, thresholds=thresholds_plain_cold,
            )
            cold_metrics['kd_student_cold_thresholded'] = evaluate_proba(
                y_test, kd_cold_proba, thresholds=thresholds_kd_cold,
            )
            cold_metrics['tflite_winner_cold_thresholded'] = evaluate_proba(
                y_test, tflite_cold_proba, thresholds=thresholds_cold,
            )
        for name, m in cold_metrics.items():
            print(
                f'  {name:<28s} macroF1={m["macro_f1"]:.4f} '
                f'BLOCK P={m["BLOCK"]["precision"]:.3f} '
                f'R={m["BLOCK"]["recall"]:.3f} '
                f'F1={m["BLOCK"]["f1"]:.3f} '
                f'WARN F1={m["WARN"]["f1"]:.3f}'
            )

        # --- Cold-start operating points: для каждой модели ищем порог BLOCK,
        # дающий BLOCK precision ≥ {0.95, 0.90, 0.80}, и репортим recall.
        # Это явно проверяет фейс-3 acceptance: P≥0.95 при R≥0.40.
        print('\n--- Cold-start BLOCK operating points (precision floor → recall) ---')
        cold_op_points: Dict = {}
        for name, proba in (
            ('catboost_teacher', teacher_cold_proba),
            ('plain_mlp', plain_cold_proba),
            ('kd_student', kd_cold_proba),
            ('tflite_winner', tflite_cold_proba),
        ):
            row_metrics: Dict = {}
            for floor in (0.95, 0.90, 0.80):
                best_t, best_p, best_r = 1.0, 0.0, 0.0
                for t in np.linspace(0.10, 0.99, 90):
                    pb = proba[:, 2] >= t
                    tp = int(np.sum(pb & (y_test == 2)))
                    fp = int(np.sum(pb & (y_test != 2)))
                    fn = int(np.sum(~pb & (y_test == 2)))
                    p = tp / max(tp + fp, 1)
                    r = tp / max(tp + fn, 1)
                    if p >= floor and r > best_r:
                        best_t, best_p, best_r = float(t), float(p), float(r)
                row_metrics[f'P>={floor}'] = {
                    'threshold': best_t,
                    'precision': best_p,
                    'recall': best_r,
                }
            cold_op_points[name] = row_metrics
            print(
                f'  {name:<20s} '
                + ' | '.join(
                    f'P>={floor:.2f}: t={row_metrics[f"P>={floor}"]["threshold"]:.2f} '
                    f'R={row_metrics[f"P>={floor}"]["recall"]:.3f} '
                    f'(P={row_metrics[f"P>={floor}"]["precision"]:.3f})'
                    for floor in (0.95, 0.90, 0.80)
                )
            )

    # --- Compose final report ---
    report = {
        'created_at': datetime.now().isoformat(),
        'data': args.data,
        'dataset_hash': file_sha256(args.data),
        'rows': int(len(y)),
        'feature_count': int(X.shape[1]),
        'features': COMPACT_FEATURES,
        'class_counts': counts,
        'split': {
            'train': int(len(train_idx)),
            'val': int(len(val_idx)),
            'test': int(len(test_idx)),
        },
        'teacher_train_per_class': args.teacher_train_per_class,
        'student_train_per_class': args.student_train_per_class,
        'teacher_counts_actual': teacher_counts,
        'student_counts_actual': student_counts,
        'smote_applied': bool(smote_info),
        'smote_info': smote_info,
        'kd': {'T': best_T, 'alpha': best_alpha,
               'lr': best_lr, 'dropout': best_dropout, 'hidden': list(best_hidden)},
        'stage1_grid': stage1,
        'stage2_optuna': stage2,
        'thresholds': thresholds,
        'thresholds_kd': thresholds_kd,
        'thresholds_plain': thresholds_plain,
        'thresholds_cold': thresholds_cold,
        'thresholds_kd_cold': thresholds_kd_cold,
        'thresholds_plain_cold': thresholds_plain_cold,
        'cold_threshold_info': cold_threshold_info,
        'best_of': {
            'winner': best_model_name,
            'selection_basis': selection_basis,
            'kd_val_score': float(kd_val_score),
            'plain_val_score': float(plain_val_score),
        },
        'val_metrics': {
            'catboost_teacher': teacher_metrics_val,
            'plain_mlp_argmax': plain_metrics_val,
            'plain_mlp_thresholded': evaluate_proba(y_val, plain_proba_val_arr, thresholds=thresholds_plain),
            'kd_student_argmax': evaluate_proba(y_val, proba_val),
            'kd_student_thresholded': evaluate_proba(y_val, proba_val, thresholds=thresholds_kd),
        },
        'test_metrics': {
            'catboost_teacher': teacher_metrics_test,
            'plain_mlp_argmax': plain_metrics_test,
            'plain_mlp_thresholded': plain_metrics_test_thresholded,
            'kd_student_argmax': student_metrics_test_argmax,
            'kd_student_thresholded': student_metrics_test,
            'tflite_winner': tflite_metrics_test,
        },
        'test_metrics_unknown_slice': test_metrics_unknown,
        'test_metrics_known_slice': test_metrics_known,
        'unknown_slice_size': n_unknown,
        'known_slice_size': len(y_test) - n_unknown,
        'test_metrics_cold_start_slice': cold_metrics,
        'cold_start_slice_size': len(y_test) if args.cold_start_eval else 0,
        'cold_start_operating_points': cold_op_points,
        'training_config': {
            'use_full_train': bool(args.use_full_train),
            'mask_features': mask_feature_names,
            'mask_indices': mask_indices,
            'feature_mask_prob': float(mask_prob),
            'weight_decay': float(args.weight_decay),
            'label_smoothing': float(args.label_smoothing),
            'teacher_aug_info': teacher_aug_info,
            'student_aug_info': student_aug_info,
            # Phase 3 config:
            'warn_class_weight': float(args.warn_class_weight),
            'block_class_weight': float(args.block_class_weight),
            'allow_class_weight': float(args.allow_class_weight),
            'use_lightgbm_teacher': bool(args.use_lightgbm_teacher),
            'teacher_components': list(teacher_components.keys()),
            'baseline_hidden_sizes': list(baseline_hidden),
            'cold_start_eval': bool(args.cold_start_eval),
            # Phase 4C config:
            'cold_aug_mode': str(args.cold_aug_mode),
            'cold_aug_weight_multiplier': float(args.cold_aug_weight_multiplier),
            'cold_aug_warn_oversample_cap': float(args.cold_aug_warn_oversample_cap),
        },
        'tflite': {'path': args.tflite_output, 'bytes': export_info['bytes']},
        'sanity': sanity,
        'warnings': warns_t + warns_s + ([smote_info.get('reason')] if smote_info.get('reason') else []),
        'run_dir': run_dir,
    }

    # Best-of картка: какую модель экспортируем + её test-метрики.
    winner_metrics_for_card = (
        student_metrics_test if best_model_name == 'kd_student'
        else plain_metrics_test_thresholded
    )
    write_kd_report(report, run_dir)
    write_kd_model_card(report, winner_metrics_for_card, thresholds, args.model_card_output,
                        best_model_name=best_model_name,
                        best_of_info=report['best_of'],
                        cold_thresholds=thresholds_cold,
                        cold_threshold_info=cold_threshold_info,
                        leak_free=bool(args.leak_free))
    print(f'\n[done] reports: {run_dir}')
    print(f'       tflite:  {args.tflite_output}')
    print(f'       card:    {args.model_card_output}')

    print('\n=== Test metrics summary (full test) ===')
    for name, m in report['test_metrics'].items():
        print(
            f'  {name:<28s} macroF1={m["macro_f1"]:.4f} '
            f'BLOCK P={m["BLOCK"]["precision"]:.3f} '
            f'R={m["BLOCK"]["recall"]:.3f} '
            f'F1={m["BLOCK"]["f1"]:.3f} '
            f'WARN F1={m["WARN"]["f1"]:.3f}'
        )
    if test_metrics_unknown:
        print(f'\n=== Test metrics on UNKNOWN slice ({n_unknown}/{len(y_test)} rows; '
              f'inAllowlist=0 AND inBlacklist=0) ===')
        for name, m in test_metrics_unknown.items():
            print(
                f'  {name:<28s} macroF1={m["macro_f1"]:.4f} '
                f'BLOCK P={m["BLOCK"]["precision"]:.3f} '
                f'R={m["BLOCK"]["recall"]:.3f} '
                f'F1={m["BLOCK"]["f1"]:.3f} '
                f'WARN F1={m["WARN"]["f1"]:.3f}'
            )
    return 0


if __name__ == '__main__':
    sys.exit(main())
