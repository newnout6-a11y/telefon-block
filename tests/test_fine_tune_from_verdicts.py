"""Offline tests for scripts/fine_tune_from_verdicts.py.

The bridge script converts a spam_predict verdicts CSV into a feedback CSV
that ``online_fine_tune.py`` can consume. These tests cover:

  * row selection by ``verdict`` column,
  * row skipping when ``error`` is set,
  * cold-start feature recomputation (every COMPACT_FEATURES column present),
  * ``user_action`` propagation,
  * ``verdict_override`` semantics,
  * CSV schema (header order + all 52 features),
  * ``--limit`` plumbing,
  * tolerance to Cyrillic / alternative ``number`` column names.
"""
from __future__ import annotations

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import fine_tune_from_verdicts as ftv  # noqa: E402
from ru_metadata_features import COMPACT_FEATURES  # noqa: E402


def _rows(*verdict_pairs):
    """Build minimal verdicts rows quickly."""
    out = []
    for i, (num, verdict) in enumerate(verdict_pairs):
        out.append({
            'normalized_number': num,
            'expected_label': 'ALLOW',
            'verdict': verdict,
            'model_verdict': verdict,
            'risk_score': '50',
            'probs_ALLOW': '0.1',
            'probs_WARN': '0.1',
            'probs_BLOCK': '0.8',
            'feature_source': 'cold',
            'rule_overrides': '',
            'disagreement': 'True' if verdict != 'ALLOW' else 'False',
            'error': '',
        })
    return out


class TestSelectRows:
    def test_keeps_only_target_verdict(self):
        rows = _rows(
            ('+79991111111', 'BLOCK'),
            ('+79992222222', 'ALLOW'),
            ('+79993333333', 'WARN'),
            ('+79994444444', 'BLOCK'),
        )
        out = ftv.select_rows(rows, target_verdict='BLOCK')
        nums = [r['normalized_number'] for r in out]
        assert nums == ['+79991111111', '+79994444444']

    def test_any_keeps_all_non_error(self):
        rows = _rows(('+79991111111', 'BLOCK'), ('+79992222222', 'ALLOW'))
        out = ftv.select_rows(rows, target_verdict='ANY')
        assert len(out) == 2

    def test_skips_error_rows(self):
        rows = _rows(('+79991111111', 'BLOCK'))
        rows[0]['error'] = 'invalid_number'
        out = ftv.select_rows(rows, target_verdict='BLOCK')
        assert out == []

    def test_skips_empty_verdict(self):
        rows = _rows(('+79991111111', ''))
        rows[0]['model_verdict'] = ''
        out = ftv.select_rows(rows, target_verdict='ANY')
        assert out == []

    def test_falls_back_to_model_verdict(self):
        rows = _rows(('+79991111111', ''))
        rows[0]['verdict'] = ''
        rows[0]['model_verdict'] = 'BLOCK'
        out = ftv.select_rows(rows, target_verdict='BLOCK')
        assert len(out) == 1


class TestBuildFeedbackRecords:
    def test_every_record_has_all_compact_features(self):
        rows = _rows(('+79991111111', 'BLOCK'))
        records = ftv.build_feedback_records(rows, user_action='not_spam')
        assert len(records) == 1
        rec = records[0]
        for name in COMPACT_FEATURES:
            assert name in rec, f'missing feature {name}'
            assert isinstance(rec[name], float)
        assert rec['verdict'] == 'BLOCK'
        assert rec['user_action'] == 'not_spam'
        assert rec['normalized_number'] == '+79991111111'

    def test_verdict_override_replaces_per_row(self):
        rows = _rows(('+79991111111', 'WARN'), ('+79992222222', 'BLOCK'))
        records = ftv.build_feedback_records(
            rows, user_action='not_spam', verdict_override='BLOCK'
        )
        assert [r['verdict'] for r in records] == ['BLOCK', 'BLOCK']

    def test_skips_rows_without_number(self):
        rows = [{'normalized_number': '', 'verdict': 'BLOCK', 'error': ''}]
        records = ftv.build_feedback_records(rows, user_action='not_spam')
        assert records == []

    def test_tolerates_cyrillic_number_column(self):
        rows = [{'номер': '+79991111111', 'verdict': 'BLOCK', 'error': ''}]
        records = ftv.build_feedback_records(rows, user_action='not_spam')
        assert len(records) == 1
        assert records[0]['normalized_number'] == '+79991111111'


class TestWriteFeedbackCSV:
    def test_writes_full_schema(self, tmp_path):
        rows = _rows(('+79991111111', 'BLOCK'))
        records = ftv.build_feedback_records(rows, user_action='not_spam')
        p = tmp_path / 'feedback.csv'
        ftv.write_feedback_csv(str(p), records)

        with open(p, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            header = list(reader.fieldnames)
            written = list(reader)

        # Header has the three meta cols first, then all 52 features.
        assert header[:3] == ['normalized_number', 'verdict', 'user_action']
        for name in COMPACT_FEATURES:
            assert name in header

        assert len(written) == 1
        assert written[0]['normalized_number'] == '+79991111111'
        assert written[0]['user_action'] == 'not_spam'
        # Spot-check that a feature was serialised as numeric text.
        first_feat = COMPACT_FEATURES[0]
        float(written[0][first_feat])  # raises if not parseable


class TestReadVerdicts:
    def test_round_trip_via_csv(self, tmp_path):
        rows = _rows(('+79991111111', 'BLOCK'), ('+79992222222', 'ALLOW'))
        p = tmp_path / 'verdicts.csv'
        with open(p, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        loaded = ftv.read_verdicts(str(p))
        assert [r['normalized_number'] for r in loaded] == [
            '+79991111111', '+79992222222'
        ]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ftv.read_verdicts('/nonexistent/verdicts.csv')


class TestEndToEnd:
    def test_main_writes_expected_rows(self, tmp_path, monkeypatch, capsys):
        rows = _rows(
            ('+79991111111', 'BLOCK'),
            ('+79992222222', 'ALLOW'),
            ('+79993333333', 'BLOCK'),
        )
        verdicts = tmp_path / 'verdicts.csv'
        with open(verdicts, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        out = tmp_path / 'feedback.csv'
        argv = [
            'fine_tune_from_verdicts.py',
            '--verdicts', str(verdicts),
            '--target-verdict', 'BLOCK',
            '--user-action', 'not_spam',
            '--out', str(out),
        ]
        monkeypatch.setattr(sys, 'argv', argv)
        rc = ftv.main()
        assert rc == 0

        with open(out, 'r', encoding='utf-8') as f:
            written = list(csv.DictReader(f))
        assert [r['normalized_number'] for r in written] == [
            '+79991111111', '+79993333333',
        ]
        assert all(r['user_action'] == 'not_spam' for r in written)
        assert all(r['verdict'] == 'BLOCK' for r in written)

    def test_limit_caps_output(self, tmp_path, monkeypatch):
        rows = _rows(
            ('+79991111111', 'BLOCK'),
            ('+79992222222', 'BLOCK'),
            ('+79993333333', 'BLOCK'),
        )
        verdicts = tmp_path / 'verdicts.csv'
        with open(verdicts, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        out = tmp_path / 'feedback.csv'
        argv = [
            'fine_tune_from_verdicts.py',
            '--verdicts', str(verdicts),
            '--target-verdict', 'BLOCK',
            '--user-action', 'not_spam',
            '--out', str(out),
            '--limit', '2',
        ]
        monkeypatch.setattr(sys, 'argv', argv)
        ftv.main()
        with open(out, 'r', encoding='utf-8') as f:
            assert sum(1 for _ in csv.DictReader(f)) == 2
