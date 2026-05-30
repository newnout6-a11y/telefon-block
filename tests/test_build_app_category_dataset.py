"""Pytest stub for the Dataset_Builder.

Real tests for label normalization, source-merge dedup, counters, CSV/report
formatting, and corpus-size validation come in tasks 2.x and 3.x of
``app-category-ml-classifier/tasks.md``. This stub exists so the test file
is on disk from task 1.1 and ``pytest tests/`` keeps discovering it.
"""
from __future__ import annotations

import importlib
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


def test_build_app_category_dataset_module_importable() -> None:
    """Skeleton sanity check: the orchestrator module loads."""
    module = importlib.import_module("build_app_category_dataset")
    assert hasattr(module, "main"), "build_app_category_dataset.main is missing"

from pathlib import Path

import pytest


# ── Build report writer (task 3.2, Requirement 1.11) ──────────────────────


class TestWriteBuildReport:
    """Tests for :func:`write_build_report` (Requirement 1.11)."""

    def _make_counters(self):
        """Create a Counters instance with realistic values for testing."""
        from build_app_category_dataset import Counters

        c = Counters()
        c.total_input_rows = 350000
        c.dropped_rows = 1234
        c.unknown_category_rows = 5678
        c.corpus_rows = 240000
        from build_app_category_dataset import KOTLIN_APP_CATEGORY_ORDER

        for i, name in enumerate(KOTLIN_APP_CATEGORY_ORDER):
            c.per_category_counts[name] = 7000 + i * 100
        return c

    def test_report_has_all_required_fields(self, tmp_path: Path) -> None:
        """The JSON report contains every field from Requirement 1.11."""
        import json
        from datetime import datetime, timezone

        from build_app_category_dataset import write_build_report

        counters = self._make_counters()
        report_path = tmp_path / "build_report.json"
        fixed_time = datetime(2025, 6, 15, 12, 30, 45, tzinfo=timezone.utc)

        write_build_report(counters, seed=42, path=report_path, built_at=fixed_time)

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["schema_version"] == 1
        assert data["total_input_rows"] == 350000
        assert data["dropped_rows"] == 1234
        assert data["unknown_category_rows"] == 5678
        assert data["corpus_rows"] == 240000
        assert data["seed"] == 42
        assert data["built_at"] == "2025-06-15T12:30:45Z"
        assert isinstance(data["per_category_counts"], dict)

    def test_per_category_counts_has_20_keys(self, tmp_path: Path) -> None:
        """``per_category_counts`` always has exactly 20 keys."""
        import json
        from datetime import datetime, timezone

        from build_app_category_dataset import (
            KOTLIN_APP_CATEGORY_ORDER,
            write_build_report,
        )

        counters = self._make_counters()
        report_path = tmp_path / "build_report.json"
        fixed_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        write_build_report(counters, seed=7, path=report_path, built_at=fixed_time)

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pcc = data["per_category_counts"]
        assert len(pcc) == 20
        assert set(pcc.keys()) == set(KOTLIN_APP_CATEGORY_ORDER)
        assert "OTHER" in pcc

    def test_per_category_counts_order_matches_kotlin_enum(
        self, tmp_path: Path
    ) -> None:
        """Keys in ``per_category_counts`` are in KOTLIN_APP_CATEGORY_ORDER order."""
        import json
        from datetime import datetime, timezone

        from build_app_category_dataset import (
            KOTLIN_APP_CATEGORY_ORDER,
            write_build_report,
        )

        counters = self._make_counters()
        report_path = tmp_path / "build_report.json"
        fixed_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        write_build_report(counters, seed=42, path=report_path, built_at=fixed_time)

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # json.load preserves insertion order in Python 3.7+
        actual_keys = list(data["per_category_counts"].keys())
        assert actual_keys == KOTLIN_APP_CATEGORY_ORDER

    def test_built_at_is_iso8601_utc_with_z_suffix(self, tmp_path: Path) -> None:
        """``built_at`` is formatted as ISO-8601 UTC with 'Z' suffix."""
        import json
        from datetime import datetime, timezone

        from build_app_category_dataset import write_build_report

        counters = self._make_counters()
        report_path = tmp_path / "build_report.json"
        fixed_time = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        write_build_report(counters, seed=1, path=report_path, built_at=fixed_time)

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["built_at"] == "2025-12-31T23:59:59Z"
        assert data["built_at"].endswith("Z")

    def test_file_is_utf8_no_bom_lf_trailing_newline(self, tmp_path: Path) -> None:
        """Report file is UTF-8 without BOM, LF line endings, trailing newline."""
        from datetime import datetime, timezone

        from build_app_category_dataset import write_build_report

        counters = self._make_counters()
        report_path = tmp_path / "build_report.json"
        fixed_time = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

        write_build_report(counters, seed=42, path=report_path, built_at=fixed_time)

        raw_bytes = report_path.read_bytes()

        # No BOM
        assert not raw_bytes.startswith(b"\xef\xbb\xbf"), "File has UTF-8 BOM"
        # No CRLF
        assert b"\r\n" not in raw_bytes, "File contains CRLF line endings"
        # Trailing newline
        assert raw_bytes.endswith(b"\n"), "File does not end with trailing newline"
        # Valid UTF-8
        raw_bytes.decode("utf-8")

    def test_atomic_write_no_partial_on_failure(self, tmp_path: Path) -> None:
        """If writing fails, no partial file is left at the target path."""
        from datetime import datetime, timezone
        from unittest.mock import patch

        from build_app_category_dataset import write_build_report

        counters = self._make_counters()
        report_path = tmp_path / "build_report.json"
        fixed_time = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

        with patch(
            "build_app_category_dataset.json.dump", side_effect=IOError("disk full")
        ):
            with pytest.raises(IOError, match="disk full"):
                write_build_report(
                    counters, seed=42, path=report_path, built_at=fixed_time
                )

        assert not report_path.exists()
        assert not report_path.with_name(report_path.name + ".tmp").exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Parent directories are created if they do not exist."""
        from datetime import datetime, timezone

        from build_app_category_dataset import write_build_report

        counters = self._make_counters()
        report_path = tmp_path / "nested" / "deep" / "build_report.json"
        fixed_time = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

        write_build_report(counters, seed=42, path=report_path, built_at=fixed_time)

        assert report_path.exists()

    def test_uses_current_utc_when_built_at_not_provided(
        self, tmp_path: Path
    ) -> None:
        """When ``built_at`` is None, the report uses the current UTC time."""
        import json
        from datetime import datetime

        from build_app_category_dataset import write_build_report

        counters = self._make_counters()
        report_path = tmp_path / "build_report.json"

        write_build_report(counters, seed=42, path=report_path)

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        built_at = data["built_at"]
        assert built_at.endswith("Z")
        # Parse it back to verify format
        parsed = datetime.strptime(built_at, "%Y-%m-%dT%H:%M:%SZ")
        assert parsed is not None

    def test_counters_values_match_report(self, tmp_path: Path) -> None:
        """Counter values are faithfully transcribed into the JSON report."""
        import json
        from datetime import datetime, timezone

        from build_app_category_dataset import Counters, write_build_report

        counters = Counters()
        counters.total_input_rows = 999
        counters.dropped_rows = 10
        counters.unknown_category_rows = 5
        counters.corpus_rows = 984
        counters.per_category_counts["BANK"] = 100
        counters.per_category_counts["EMAIL"] = 200

        report_path = tmp_path / "build_report.json"
        fixed_time = datetime(2025, 3, 14, 9, 26, 53, tzinfo=timezone.utc)

        write_build_report(counters, seed=123, path=report_path, built_at=fixed_time)

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["total_input_rows"] == 999
        assert data["dropped_rows"] == 10
        assert data["unknown_category_rows"] == 5
        assert data["corpus_rows"] == 984
        assert data["seed"] == 123
        assert data["per_category_counts"]["BANK"] == 100
        assert data["per_category_counts"]["EMAIL"] == 200
        # Categories not explicitly set should be 0
        assert data["per_category_counts"]["GAMES"] == 0
