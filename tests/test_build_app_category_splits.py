"""Tests for the stratified splits builder.

Covers:
  - Module importability (sanity check from task 1.1).
  - load_labeled_csv: correct parsing, error handling.
  - stratified_split: disjoint invariant, total coverage, proportions,
    determinism (same seed → same output), stratification.
  - write_split_csv: UTF-8/LF/no-BOM format, trailing newline, atomic write.
  - main() end-to-end: reads labeled.csv, writes train/val/test.csv.

Requirements validated: 1.7, 1.8.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from build_app_category_splits import (
    LABELED_CSV_HEADER,
    load_labeled_csv,
    main,
    stratified_split,
    write_split_csv,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_corpus(num_per_category: int = 20) -> list[tuple[str, str, str]]:
    """Generate a synthetic corpus with balanced categories.

    Creates ``num_per_category`` rows for each of the 18 trainable
    categories (BANK..PRODUCTIVITY). This gives enough rows for
    stratified splitting to work (sklearn needs at least 2 per class
    in each split).
    """
    categories = [
        "BANK", "INVESTMENTS", "GOVERNMENT", "MARKETPLACE", "DELIVERY",
        "TRANSPORT", "TRAVEL", "HEALTH", "MESSENGER", "SOCIAL",
        "EMAIL", "NEWS", "MEDIA", "GAMES", "DATING",
        "EDUCATION", "BROWSER", "VPN", "PRODUCTIVITY",
    ]
    rows: list[tuple[str, str, str]] = []
    for cat in categories:
        for i in range(num_per_category):
            pkg = f"com.example.{cat.lower()}.app{i}"
            label = f"{cat.title()} App {i}"
            rows.append((pkg, label, cat))
    return rows


def _write_corpus_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """Write a labeled corpus CSV for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(LABELED_CSV_HEADER)
        for row in rows:
            writer.writerow(row)


# ── Tests: load_labeled_csv ───────────────────────────────────────────────


def test_load_labeled_csv_basic(tmp_path: Path) -> None:
    """Loads a well-formed CSV and returns correct tuples."""
    rows = [("com.example.bank", "Bank App", "BANK")]
    csv_path = tmp_path / "labeled.csv"
    _write_corpus_csv(csv_path, rows)

    loaded = load_labeled_csv(csv_path)
    assert loaded == rows


def test_load_labeled_csv_missing_file(tmp_path: Path) -> None:
    """Raises FileNotFoundError for a non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_labeled_csv(tmp_path / "nonexistent.csv")


def test_load_labeled_csv_bad_header(tmp_path: Path) -> None:
    """Raises ValueError when the header doesn't match."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("col1,col2,col3\na,b,c\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected CSV header"):
        load_labeled_csv(csv_path)


def test_load_labeled_csv_empty_file(tmp_path: Path) -> None:
    """Raises ValueError for an empty file."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty CSV file"):
        load_labeled_csv(csv_path)


# ── Tests: stratified_split ───────────────────────────────────────────────


def test_stratified_split_disjoint() -> None:
    """No packageName appears in more than one split (Requirement 1.7)."""
    rows = _make_corpus(num_per_category=50)
    train, val, test = stratified_split(rows, seed=42)

    train_pkgs = {r[0] for r in train}
    val_pkgs = {r[0] for r in val}
    test_pkgs = {r[0] for r in test}

    assert train_pkgs & val_pkgs == set()
    assert train_pkgs & test_pkgs == set()
    assert val_pkgs & test_pkgs == set()


def test_stratified_split_total() -> None:
    """All rows from the input appear in exactly one split."""
    rows = _make_corpus(num_per_category=50)
    train, val, test = stratified_split(rows, seed=42)

    assert len(train) + len(val) + len(test) == len(rows)

    # Every original row is present in exactly one split.
    all_split = set(train) | set(val) | set(test)
    assert all_split == set(rows)


def test_stratified_split_proportions() -> None:
    """Split proportions are approximately 80/10/10."""
    rows = _make_corpus(num_per_category=100)
    train, val, test = stratified_split(rows, seed=42)

    total = len(rows)
    # Allow ±3% tolerance for rounding.
    assert abs(len(train) / total - 0.80) < 0.03
    assert abs(len(val) / total - 0.10) < 0.03
    assert abs(len(test) / total - 0.10) < 0.03


def test_stratified_split_determinism() -> None:
    """Same seed + same input → identical splits (Requirement 1.8)."""
    rows = _make_corpus(num_per_category=50)

    train1, val1, test1 = stratified_split(rows, seed=42)
    train2, val2, test2 = stratified_split(rows, seed=42)

    assert train1 == train2
    assert val1 == val2
    assert test1 == test2


def test_stratified_split_different_seed() -> None:
    """Different seeds produce different splits."""
    rows = _make_corpus(num_per_category=50)

    train1, _, _ = stratified_split(rows, seed=42)
    train2, _, _ = stratified_split(rows, seed=123)

    # With different seeds, the train sets should differ.
    assert train1 != train2


def test_stratified_split_stratification() -> None:
    """Each split preserves the category distribution (stratified)."""
    rows = _make_corpus(num_per_category=100)
    train, val, test = stratified_split(rows, seed=42)

    # Count categories in each split.
    def cat_counts(split):
        counts: dict[str, int] = {}
        for _, _, cat in split:
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    train_counts = cat_counts(train)
    val_counts = cat_counts(val)
    test_counts = cat_counts(test)

    # Every category present in the input should appear in all splits.
    input_cats = {r[2] for r in rows}
    assert set(train_counts.keys()) == input_cats
    assert set(val_counts.keys()) == input_cats
    assert set(test_counts.keys()) == input_cats


def test_stratified_split_empty() -> None:
    """Empty input returns three empty lists."""
    train, val, test = stratified_split([], seed=42)
    assert train == []
    assert val == []
    assert test == []


# ── Tests: write_split_csv ────────────────────────────────────────────────


def test_write_split_csv_format(tmp_path: Path) -> None:
    """Output CSV has UTF-8/LF/no-BOM format with trailing newline."""
    rows = [
        ("com.example.bank", "Bank App", "BANK"),
        ("com.example.news", "News App", "NEWS"),
    ]
    out_path = tmp_path / "split.csv"
    write_split_csv(rows, out_path)

    raw_bytes = out_path.read_bytes()

    # No BOM.
    assert not raw_bytes.startswith(b"\xef\xbb\xbf")

    # LF line endings only (no CR).
    assert b"\r" not in raw_bytes

    # Trailing newline.
    assert raw_bytes.endswith(b"\n")

    # Correct header.
    lines = raw_bytes.decode("utf-8").split("\n")
    assert lines[0] == "packageName,label,category"

    # Correct data rows.
    assert lines[1] == "com.example.bank,Bank App,BANK"
    assert lines[2] == "com.example.news,News App,NEWS"

    # Last line is empty (trailing newline splits into empty string).
    assert lines[-1] == ""


def test_write_split_csv_creates_parent_dirs(tmp_path: Path) -> None:
    """Parent directories are created if they don't exist."""
    out_path = tmp_path / "sub" / "dir" / "split.csv"
    write_split_csv([("pkg", "lbl", "CAT")], out_path)
    assert out_path.exists()


def test_write_split_csv_atomic(tmp_path: Path) -> None:
    """No .tmp file remains after successful write."""
    out_path = tmp_path / "split.csv"
    write_split_csv([("pkg", "lbl", "CAT")], out_path)
    tmp_file = tmp_path / "split.csv.tmp"
    assert not tmp_file.exists()


# ── Tests: main() end-to-end ──────────────────────────────────────────────


def test_main_end_to_end(tmp_path: Path) -> None:
    """main() reads labeled.csv and writes train/val/test.csv."""
    corpus = _make_corpus(num_per_category=50)
    input_path = tmp_path / "labeled.csv"
    _write_corpus_csv(input_path, corpus)

    # --output-dir defaults to datasets/categories/ per task 4.1; tests
    # pin it explicitly to a tmp_path subdir to stay hermetic.
    output_dir = tmp_path / "out"
    exit_code = main([
        "--input", str(input_path),
        "--output-dir", str(output_dir),
        "--seed", "42",
    ])
    assert exit_code == 0

    # All three output files exist.
    for name in ("train.csv", "val.csv", "test.csv"):
        assert (output_dir / name).exists()

    # Load and verify disjoint + total.
    train = load_labeled_csv(output_dir / "train.csv")
    val = load_labeled_csv(output_dir / "val.csv")
    test = load_labeled_csv(output_dir / "test.csv")

    assert len(train) + len(val) + len(test) == len(corpus)

    train_pkgs = {r[0] for r in train}
    val_pkgs = {r[0] for r in val}
    test_pkgs = {r[0] for r in test}
    assert train_pkgs & val_pkgs == set()
    assert train_pkgs & test_pkgs == set()
    assert val_pkgs & test_pkgs == set()


def test_main_byte_determinism(tmp_path: Path) -> None:
    """Same seed → byte-identical output files (Requirement 1.8)."""
    corpus = _make_corpus(num_per_category=50)
    input_path = tmp_path / "labeled.csv"
    _write_corpus_csv(input_path, corpus)

    # First run.
    dir1 = tmp_path / "run1"
    main(["--input", str(input_path), "--output-dir", str(dir1), "--seed", "42"])

    # Second run.
    dir2 = tmp_path / "run2"
    main(["--input", str(input_path), "--output-dir", str(dir2), "--seed", "42"])

    for name in ("train.csv", "val.csv", "test.csv"):
        bytes1 = (dir1 / name).read_bytes()
        bytes2 = (dir2 / name).read_bytes()
        assert bytes1 == bytes2, f"{name} differs between runs"


def test_main_missing_input(tmp_path: Path) -> None:
    """main() returns 1 when the input file doesn't exist."""
    exit_code = main(["--input", str(tmp_path / "nonexistent.csv")])
    assert exit_code == 1


def test_main_custom_output_dir(tmp_path: Path) -> None:
    """main() respects --output-dir."""
    corpus = _make_corpus(num_per_category=20)
    input_path = tmp_path / "input" / "labeled.csv"
    _write_corpus_csv(input_path, corpus)

    output_dir = tmp_path / "output"
    exit_code = main([
        "--input", str(input_path),
        "--output-dir", str(output_dir),
        "--seed", "7",
    ])
    assert exit_code == 0

    for name in ("train.csv", "val.csv", "test.csv"):
        assert (output_dir / name).exists()
