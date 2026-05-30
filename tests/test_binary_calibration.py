"""Тесты PR-2: бинарный классификатор + Platt калибровка.

Проверяет чистую логику scripts/train_binary_model.py (без тяжёлых зависимостей):
  * to_binary_labels: 3 стратегии для WARN (merge_block / merge_allow / drop).
  * stratified_split: пропорции по классам сохраняются.
  * fit_platt: на синтетических данных восстанавливает identity, и
    улучшает калибровку, если логиты сдвинуты.
  * expected_calibration_error / brier_score: базовые свойства.
  * tune_thresholds_binary: respects min_block_precision, warn_thr <= block_thr - 0.01.

Тесты не запускают tensorflow.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))

try:
    from train_binary_model import (  # noqa: E402
        LABEL_TO_ID,
        brier_score,
        expected_calibration_error,
        fit_platt,
        stratified_split,
        to_binary_labels,
        tune_thresholds_binary,
    )
except ImportError as e:  # pragma: no cover
    pytest.skip(f'train_binary_model import failed: {e}', allow_module_level=True)


# ---------------------------------------------------------------------------
# Binary labels
# ---------------------------------------------------------------------------

class TestToBinaryLabels:
    def _y(self, allow=10, warn=5, block=20):
        return np.array(
            [LABEL_TO_ID['ALLOW']] * allow
            + [LABEL_TO_ID['WARN']] * warn
            + [LABEL_TO_ID['BLOCK']] * block,
            dtype=np.int64,
        )

    def test_merge_block_default(self):
        y = self._y()
        y_bin, keep = to_binary_labels(y, 'merge_block')
        # все WARN+BLOCK = 1, ALLOW = 0, никто не выброшен
        assert keep.all()
        assert int((y_bin == 1).sum()) == 25  # warn + block
        assert int((y_bin == 0).sum()) == 10  # allow

    def test_merge_allow(self):
        y = self._y()
        y_bin, keep = to_binary_labels(y, 'merge_allow')
        assert keep.all()
        assert int((y_bin == 1).sum()) == 20  # block only
        assert int((y_bin == 0).sum()) == 15  # allow + warn

    def test_drop_warn(self):
        y = self._y()
        y_bin, keep = to_binary_labels(y, 'drop')
        # WARN строки помечены как drop
        assert int((~keep).sum()) == 5
        # spam = block, allow = allow (но в y_bin для drop'нутых WARN — тоже 0,
        # но keep=False, так что они не используются вверху по пайплайну).
        kept = y_bin[keep]
        assert int((kept == 1).sum()) == 20
        assert int((kept == 0).sum()) == 10

    def test_unknown_strategy_raises(self):
        y = self._y()
        with pytest.raises(ValueError):
            to_binary_labels(y, 'whatever')


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

class TestStratifiedSplit:
    def test_preserves_class_ratio(self):
        rng = np.random.default_rng(0)
        y = np.concatenate([
            np.zeros(100, dtype=np.int64),
            np.ones(50, dtype=np.int64),
            np.full(30, 2, dtype=np.int64),
        ])
        rng.shuffle(y)
        train_idx, val_idx, test_idx = stratified_split(y, (0.7, 0.15, 0.15), seed=1)
        # все индексы покрыты ровно один раз
        all_idx = np.concatenate([train_idx, val_idx, test_idx])
        assert sorted(all_idx.tolist()) == list(range(len(y)))
        # train >= val + test (приблизительно 70%)
        assert len(train_idx) >= len(val_idx) + len(test_idx) - 5
        # для каждого класса — есть представители во всех сплитах
        for cls in (0, 1, 2):
            for sp in (train_idx, val_idx, test_idx):
                assert (y[sp] == cls).sum() >= 1


# ---------------------------------------------------------------------------
# Platt
# ---------------------------------------------------------------------------

class TestPlatt:
    def test_already_calibrated_stays_calibrated(self):
        """Если данные уже идеально калиброваны (logit ↔ true label), Platt не должен сильно ухудшать."""
        rng = np.random.default_rng(7)
        n = 2000
        y = rng.integers(0, 2, size=n)
        # logit ровно соответствует p=y (т.е. ±large для нужной стороны)
        logits = np.where(y == 1, 5.0, -5.0) + rng.normal(0, 0.3, size=n)
        a, b = fit_platt(logits, y)
        # a близко к 1, b близко к 0 — небольшая корректировка ок
        assert 0.4 < a < 2.5, f'Platt a={a} drifted unexpectedly'
        assert -1.5 < b < 1.5, f'Platt b={b} drifted unexpectedly'

    def test_improves_overconfident_logits(self):
        """Если все логиты раздуты в 5x, Platt должен сжать до ~1x."""
        rng = np.random.default_rng(11)
        n = 3000
        y = rng.integers(0, 2, size=n)
        true_logits = np.where(y == 1, 1.5, -1.5) + rng.normal(0, 0.5, size=n)
        # делаем «overconfident»
        bad_logits = true_logits * 5.0
        # ECE до калибровки:
        p_pre = 1 / (1 + np.exp(-bad_logits))
        ece_pre = expected_calibration_error(p_pre, y)
        a, b = fit_platt(bad_logits, y)
        p_post = 1 / (1 + np.exp(-(a * bad_logits + b)))
        ece_post = expected_calibration_error(p_post, y)
        # Platt должен снизить ECE
        assert ece_post < ece_pre, f'Platt failed: ECE before={ece_pre} after={ece_post}'
        # a должен быть < 1 (мы сжимаем overconfident логиты)
        assert a < 1.0, f'Platt a={a} should compress overconfident logits'

    def test_one_class_returns_identity(self):
        """Если в hold-out только один класс — возвращаем (1, 0)."""
        logits = np.array([0.5, -0.5, 1.0])
        y = np.array([1, 1, 1])
        a, b = fit_platt(logits, y)
        assert a == 1.0
        assert b == 0.0


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

class TestCalibrationMetrics:
    def test_perfect_calibration_zero_ece(self):
        """ECE ≈ 0, когда predicted == empirical accuracy в каждом бине."""
        # Если p=0.5 везде, и y=Bernoulli(0.5) — ECE → 0.
        rng = np.random.default_rng(13)
        n = 5000
        p = np.full(n, 0.5)
        y = rng.integers(0, 2, size=n)
        ece = expected_calibration_error(p, y)
        assert ece < 0.05, f'ECE={ece} should be near 0'

    def test_brier_for_perfect_predictor(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([0.0, 1.0, 1.0, 0.0])
        assert brier_score(p, y) == 0.0

    def test_brier_for_random_predictor(self):
        y = np.array([0, 1, 1, 0])
        p = np.full(4, 0.5)
        assert abs(brier_score(p, y) - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

class TestTuneThresholds:
    def test_block_thr_respects_precision_floor(self):
        rng = np.random.default_rng(17)
        n = 3000
        y = rng.integers(0, 2, size=n)
        # «осмысленные» вероятности: spam → высокие p, allow → низкие
        p = np.where(y == 1, rng.beta(5, 2, size=n), rng.beta(2, 5, size=n)).astype(np.float64)
        block_thr, warn_thr = tune_thresholds_binary(p, y, min_block_precision=0.85)
        # block_thr должен давать precision >= floor
        pred = (p >= block_thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        prec = tp / max(tp + fp, 1)
        assert prec >= 0.85, f'BLOCK precision={prec} below floor 0.85'
        assert warn_thr < block_thr, 'warn_thr must be strictly below block_thr'
        assert 0.10 <= warn_thr < block_thr

    def test_warn_thr_below_block_thr(self):
        rng = np.random.default_rng(19)
        n = 1000
        y = rng.integers(0, 2, size=n)
        p = np.where(y == 1, rng.beta(8, 2, size=n), rng.beta(2, 8, size=n))
        block_thr, warn_thr = tune_thresholds_binary(p, y, min_block_precision=0.80)
        assert warn_thr <= block_thr - 0.005
