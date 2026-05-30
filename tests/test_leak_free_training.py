"""Тесты PR-1: leak-free training режим.

Проверяем, что:
* `make_cold_view` физически зануляет указанные индексы и форсит noMetadata=1.
* `feature_mask_indices` корректно резолвит COLD_START_MASK_FEATURES в индексы.
* После применения cold-view ни одна leakage-фича не имеет ненулевого значения.
* `write_kd_model_card(..., leak_free=True)` добавляет блок `leak_free` и не
  оставляет блок `cold_thresholds` (warm == cold).

Тесты не требуют tensorflow / catboost (они optional в train_kd_distillation
до момента вызова конкретных функций).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

# Add scripts/ to path
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))

from ru_metadata_features import COMPACT_FEATURES  # noqa: E402

# train_kd_distillation импортирует tensorflow при загрузке некоторых функций,
# но top-level импорт самого модуля от этого не зависит. Если tf нет — тесты
# по-прежнему могут проверить чистую логику (make_cold_view, feature_mask_indices,
# write_kd_model_card работают без tf).
try:
    from train_kd_distillation import (  # noqa: E402
        COLD_START_MASK_FEATURES,
        feature_mask_indices,
        make_cold_view,
        write_kd_model_card,
    )
except ImportError as e:  # pragma: no cover
    pytest.skip(f'train_kd_distillation import failed: {e}', allow_module_level=True)


def _fake_X(n_rows: int = 50, seed: int = 7) -> np.ndarray:
    """Синтетический X той же ширины, что COMPACT_FEATURES, со случайными значениями."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n_rows, len(COMPACT_FEATURES))).astype(np.float32)


class TestColdStartMaskFeatures:
    """Sanity-check: список COLD_START_MASK_FEATURES стабилен и совпадает с проектной спецификацией."""

    def test_contains_expected_9_features(self):
        expected = {
            'inAllowlist',
            'inBlacklist',
            'reputationScore',
            'sourceConfidence',
            'reviewsLog',
            'negativeRatio',
            'searchVolumeLog',
            'hasFraudCategory',
            'hasTelemarketingCategory',
        }
        assert set(COLD_START_MASK_FEATURES) == expected, (
            f'COLD_START_MASK_FEATURES drifted from the canonical 9: {COLD_START_MASK_FEATURES}'
        )

    def test_all_resolve_to_valid_indices(self):
        idx = feature_mask_indices(list(COLD_START_MASK_FEATURES))
        assert len(idx) == len(COLD_START_MASK_FEATURES)
        for i in idx:
            assert 0 <= i < len(COMPACT_FEATURES)


class TestMakeColdView:
    def test_zeroes_target_columns(self):
        X = _fake_X(20)
        idx = feature_mask_indices(list(COLD_START_MASK_FEATURES))
        no_meta_idx = COMPACT_FEATURES.index('noMetadata')
        Xc = make_cold_view(X, idx, no_meta_idx)
        for i in idx:
            col = Xc[:, i]
            assert float(col.max()) == 0.0
            assert float(col.min()) == 0.0

    def test_forces_no_metadata_to_one(self):
        X = _fake_X(20)
        idx = feature_mask_indices(list(COLD_START_MASK_FEATURES))
        no_meta_idx = COMPACT_FEATURES.index('noMetadata')
        Xc = make_cold_view(X, idx, no_meta_idx)
        assert float(Xc[:, no_meta_idx].min()) == 1.0
        assert float(Xc[:, no_meta_idx].max()) == 1.0

    def test_does_not_modify_other_columns(self):
        X = _fake_X(30)
        idx = feature_mask_indices(list(COLD_START_MASK_FEATURES))
        no_meta_idx = COMPACT_FEATURES.index('noMetadata')
        Xc = make_cold_view(X, idx, no_meta_idx)
        untouched = [
            j for j in range(X.shape[1])
            if j not in idx and j != no_meta_idx
        ]
        # Все остальные колонки сохранены без изменений.
        np.testing.assert_array_equal(Xc[:, untouched], X[:, untouched])

    def test_does_not_modify_input_in_place(self):
        X = _fake_X(10)
        before = X.copy()
        idx = feature_mask_indices(list(COLD_START_MASK_FEATURES))
        no_meta_idx = COMPACT_FEATURES.index('noMetadata')
        _ = make_cold_view(X, idx, no_meta_idx)
        np.testing.assert_array_equal(X, before)

    def test_idempotent(self):
        X = _fake_X(15)
        idx = feature_mask_indices(list(COLD_START_MASK_FEATURES))
        no_meta_idx = COMPACT_FEATURES.index('noMetadata')
        Xc1 = make_cold_view(X, idx, no_meta_idx)
        Xc2 = make_cold_view(Xc1, idx, no_meta_idx)
        np.testing.assert_array_equal(Xc1, Xc2)


class TestModelCardLeakFreeFlag:
    """write_kd_model_card записывает блок `leak_free` и убирает `cold_thresholds`."""

    def _minimal_report(self) -> dict:
        return {
            'created_at': '2026-05-15T00:00:00Z',
            'feature_count': len(COMPACT_FEATURES),
            'features': COMPACT_FEATURES,
            'rows': 100,
            'class_counts': {'ALLOW': 30, 'WARN': 20, 'BLOCK': 50},
            'dataset_hash': 'fake-hash',
            'best_of': {'winner': 'kd_student'},
            'training_config': {
                'teacher_components': ['catboost'],
                'allow_class_weight': 1.0,
                'warn_class_weight': 1.0,
                'block_class_weight': 1.0,
            },
            'kd': {'T': 4.0, 'alpha': 0.5},
            'teacher_train_per_class': 100,
            'student_train_per_class': 100,
            'cold_start_slice_size': 0,
            'test_metrics_cold_start_slice': {},
            'smote_applied': False,
        }

    def _minimal_metrics(self) -> dict:
        return {
            'BLOCK': {'precision': 0.95, 'recall': 0.80, 'f1': 0.87},
            'WARN': {'precision': 0.5, 'recall': 0.4, 'f1': 0.44},
            'ALLOW': {'precision': 0.92, 'recall': 0.99, 'f1': 0.95},
            'macro_f1': 0.75,
            'roc_auc_ovr': 0.97,
        }

    def _thresholds(self) -> dict:
        return {
            'block_threshold': 0.55,
            'warn_threshold': 0.30,
            'block_precision': 0.95,
            'block_recall': 0.80,
            'block_f1': 0.87,
            'warn_f1': 0.44,
        }

    def test_leak_free_block_written(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'card.json')
            write_kd_model_card(
                self._minimal_report(),
                self._minimal_metrics(),
                self._thresholds(),
                out,
                best_model_name='kd_student',
                best_of_info={'winner': 'kd_student'},
                cold_thresholds=None,
                cold_threshold_info=None,
                leak_free=True,
            )
            with open(out) as f:
                card = json.load(f)
        assert 'leak_free' in card
        assert card['leak_free']['enabled'] is True
        assert card['leak_free']['forced_no_metadata'] is True
        assert set(card['leak_free']['zeroed_features']) == set(COLD_START_MASK_FEATURES)
        assert card['version'].startswith('kd-mlp-leakfree-')

    def test_cold_thresholds_dropped_when_leak_free(self):
        cold_thr = {
            'block_threshold': 0.77,
            'warn_threshold': 0.31,
            'block_precision': 0.95,
            'block_recall': 0.69,
            'block_f1': 0.80,
            'warn_f1': 0.30,
        }
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'card.json')
            write_kd_model_card(
                self._minimal_report(),
                self._minimal_metrics(),
                self._thresholds(),
                out,
                best_model_name='kd_student',
                best_of_info={'winner': 'kd_student'},
                cold_thresholds=cold_thr,
                cold_threshold_info={'mask_features': list(COLD_START_MASK_FEATURES)},
                leak_free=True,
            )
            with open(out) as f:
                card = json.load(f)
        # leak-free => warm == cold, поэтому отдельный cold_thresholds не имеет смысла.
        assert 'cold_thresholds' not in card

    def test_legacy_path_unchanged(self):
        """Проверяем, что без leak_free карта остаётся такой же, как раньше."""
        cold_thr = {
            'block_threshold': 0.77,
            'warn_threshold': 0.31,
            'block_precision': 0.95,
            'block_recall': 0.69,
            'block_f1': 0.80,
            'warn_f1': 0.30,
        }
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'card.json')
            write_kd_model_card(
                self._minimal_report(),
                self._minimal_metrics(),
                self._thresholds(),
                out,
                best_model_name='kd_student',
                best_of_info={'winner': 'kd_student'},
                cold_thresholds=cold_thr,
                cold_threshold_info={'mask_features': list(COLD_START_MASK_FEATURES)},
                leak_free=False,
            )
            with open(out) as f:
                card = json.load(f)
        assert 'leak_free' not in card
        assert 'cold_thresholds' in card
        assert card['version'].startswith('kd-mlp-') and 'leakfree' not in card['version']
