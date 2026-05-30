"""Stratified 80/10/10 splitter for App Category ML Classifier.

Reads ``datasets/categories/labeled.csv`` (or a custom path via
``--input``) and writes ``train.csv``, ``val.csv``, ``test.csv`` next
to it. The split is stratified by ``category`` and deterministic for a
given ``--seed`` (Requirements 1.7, 1.8).

Algorithm:
  1. Load the labeled corpus CSV.
  2. Dedup by ``packageName`` preserving the first occurrence — this
     guarantees the unique-package invariant the splitter relies on
     even if a caller hands us a corpus that has not been through
     :func:`build_app_category_dataset.merge_sources` (Requirement 1.3).
  3. Extract the ``category`` column from the deduplicated rows as the
     stratification target.
  4. First split: 80% train / 20% holdout (sklearn ``train_test_split``
     with ``stratify=y`` and ``random_state=seed``).
  5. Second split: 50/50 on the 20% holdout → 10% val + 10% test
     (same call shape, ``stratify=y_holdout`` and ``random_state=seed``).
  6. Write each split to its own CSV via
     :func:`build_app_category_dataset.write_labeled_csv` — which
     already enforces UTF-8/no-BOM, LF line endings, trailing newline,
     and atomic ``.tmp``→rename semantics (Requirement 1.5, shared
     format with ``labeled.csv``).

Disjoint invariant (Requirement 1.7): the splitter operates on row
indices of the *deduplicated* corpus, so each ``packageName`` appears
in exactly one of train/val/test by construction. The dedup step in
(2) hardens this invariant against duplicate rows that might slip
through a hand-edited input file.

Determinism (Requirement 1.8): for the same ``--seed`` and the same
input corpus, the output files are byte-identical. This is guaranteed
by sklearn's ``train_test_split(random_state=seed)`` which uses a
deterministic PRNG seeded from the integer argument.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

from sklearn.model_selection import train_test_split

# Reuse the canonical, atomic CSV writer and the matching :class:`Row`
# / :class:`Source` types from the Dataset_Builder module so train.csv,
# val.csv, and test.csv share the exact same on-disk format guarantees
# as labeled.csv (Requirement 1.5).
from build_app_category_dataset import (
    LABELED_CSV_HEADER,
    Row,
    Source,
    write_labeled_csv,
)

# ``Source`` is irrelevant once a row is on disk (the column is not
# emitted by :func:`write_labeled_csv`), but :class:`Row` requires it
# to construct an instance. We pick a fixed placeholder so split rows
# round-trip cleanly through ``Row(...)`` without callers having to
# reason about provenance — splits don't track it.
_SPLIT_PLACEHOLDER_SOURCE: Source = Source.PLAY


# ── Core split logic ──────────────────────────────────────────────────────


def load_labeled_csv(path: Path) -> list[tuple[str, str, str]]:
    """Load a labeled corpus CSV into a list of (packageName, label, category) tuples.

    The file must have a header row ``packageName,label,category`` as
    its first line. Subsequent rows are returned in file order.

    Parameters
    ----------
    path:
        Path to the labeled CSV file.

    Returns
    -------
    list[tuple[str, str, str]]
        Each element is ``(packageName, label, category)`` in the order
        they appear in the file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the header row does not match the expected columns.
    """
    path = Path(path)
    rows: list[tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV file: {path}")
        if tuple(header) != LABELED_CSV_HEADER:
            raise ValueError(
                f"Unexpected CSV header in {path}: {header!r}; "
                f"expected {LABELED_CSV_HEADER!r}"
            )
        for row in reader:
            if len(row) != 3:
                # Skip malformed rows silently — the upstream
                # build_app_category_dataset.py guarantees well-formed
                # output, but defensive coding never hurts.
                continue
            rows.append((row[0], row[1], row[2]))
    return rows


def dedup_by_package(
    rows: Sequence[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Return ``rows`` deduplicated by ``packageName``, first-wins.

    Implements the dedup half of the disjoint invariant (Requirement
    1.7): when the splitter operates on the deduplicated row list, each
    distinct ``packageName`` lives at exactly one index, so partitioning
    by index automatically partitions by package.

    Comparison is case-sensitive (Requirement 1.3, mirrors
    :func:`build_app_category_dataset.merge_sources`). The first
    occurrence wins — its ``label`` and ``category`` are kept verbatim,
    and any later row carrying the same ``packageName`` is dropped
    silently. Output preserves the input order of survivors so the
    split is stable for a given input file.

    Parameters
    ----------
    rows:
        Sequence of ``(packageName, label, category)`` tuples in
        ``labeled.csv`` order.

    Returns
    -------
    list[tuple[str, str, str]]
        Deduplicated rows, one per unique ``packageName``, in
        first-appearance order.
    """
    seen: set[str] = set()
    unique: list[tuple[str, str, str]] = []
    for row in rows:
        package = row[0]
        if package in seen:
            continue
        seen.add(package)
        unique.append(row)
    return unique


def stratified_split(
    rows: list[tuple[str, str, str]],
    seed: int = 42,
) -> tuple[
    list[tuple[str, str, str]],
    list[tuple[str, str, str]],
    list[tuple[str, str, str]],
]:
    """Split rows into train/val/test (80/10/10) stratified by category.

    Calls sklearn ``train_test_split`` twice (Requirement 1.7):
      1. 80% train / 20% holdout, ``stratify=categories``,
         ``random_state=seed``.
      2. 50/50 on the 20% holdout → val + test, stratified by the
         holdout's categories with the **same** ``random_state=seed``.

    The split operates on the ``rows`` list directly, so the disjoint
    invariant follows from the input being deduplicated by
    ``packageName`` (see :func:`dedup_by_package`). Each row — and
    therefore each unique ``packageName`` — appears in exactly one
    split (Requirement 1.7).

    Parameters
    ----------
    rows:
        List of ``(packageName, label, category)`` tuples.  Should be
        deduplicated by ``packageName`` upstream so the disjoint
        invariant holds.  Must have at least one row per category for
        sklearn's stratification to succeed.
    seed:
        Random state for both ``train_test_split`` calls.  For the
        same ``seed`` and the same ``rows`` the splits are byte-stable
        (Requirement 1.8).

    Returns
    -------
    tuple[list, list, list]
        ``(train_rows, val_rows, test_rows)`` — each is a list of
        ``(packageName, label, category)`` tuples.
    """
    if not rows:
        return [], [], []

    # Stratify on the category column — the third tuple element of each
    # row.  sklearn copies the stratify array internally, so the list
    # comprehension does not need to be passed by reference.
    categories = [row[2] for row in rows]

    # 1) 80% train / 20% holdout.
    train_rows, holdout_rows, _, holdout_categories = train_test_split(
        rows,
        categories,
        test_size=0.2,
        stratify=categories,
        random_state=seed,
    )

    # 2) 50/50 on the holdout → 10% val + 10% test.
    val_rows, test_rows = train_test_split(
        holdout_rows,
        test_size=0.5,
        stratify=holdout_categories,
        random_state=seed,
    )

    return train_rows, val_rows, test_rows


def write_split_csv(rows: Sequence[tuple[str, str, str]], path: Path) -> None:
    """Atomically write a split CSV to ``path`` via :func:`write_labeled_csv`.

    Thin adapter that wraps each ``(packageName, label, category)``
    tuple in a :class:`Row` and delegates to
    :func:`build_app_category_dataset.write_labeled_csv` so train,
    val, and test files share the exact same on-disk format and atomic
    write semantics as ``labeled.csv`` (Requirement 1.5):

    * UTF-8 encoding without BOM.
    * LF line terminator, including a trailing newline.
    * Header line ``packageName,label,category`` first.
    * ``.tmp``-then-rename atomic write with best-effort cleanup of
      the partial file on failure.

    The placeholder :data:`_SPLIT_PLACEHOLDER_SOURCE` populates the
    in-memory ``source`` tag on :class:`Row` — this column is never
    written to disk, so the placeholder choice is invisible to callers.

    Parameters
    ----------
    rows:
        Sequence of ``(packageName, label, category)`` tuples.
    path:
        Destination path for the CSV file. Parent directories are
        created if they do not exist.
    """
    materialised = [
        Row(
            packageName=row[0],
            label=row[1],
            category=row[2],
            source=_SPLIT_PLACEHOLDER_SOURCE,
        )
        for row in rows
    ]
    write_labeled_csv(materialised, path)


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the splits builder."""
    parser = argparse.ArgumentParser(
        description=(
            "Split labeled.csv into train/val/test (80/10/10) "
            "stratified by category."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/categories/labeled.csv"),
        help=(
            "Path to the labeled corpus CSV "
            "(default: datasets/categories/labeled.csv)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/categories/"),
        help=(
            "Directory to write train.csv, val.csv, test.csv "
            "(default: datasets/categories/)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the stratified splits builder.

    Reads the labeled corpus, deduplicates it by ``packageName``,
    splits it 80/10/10 stratified by category, and writes train.csv,
    val.csv, test.csv into ``--output-dir`` (default
    ``datasets/categories/``).

    Returns
    -------
    int
        ``0`` on success, ``1`` on any failure (missing input,
        malformed header, empty corpus, sklearn stratification error).
    """
    args = parse_args(argv)

    input_path: Path = args.input
    output_dir: Path = args.output_dir
    seed: int = args.seed

    # Load the labeled corpus.
    try:
        rows = load_labeled_csv(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print("ERROR: labeled corpus is empty", file=sys.stderr)
        return 1

    # Dedup by packageName — first occurrence wins (Requirement 1.3).
    # This is what makes the index-based split disjoint by package
    # (Requirement 1.7) regardless of duplicates in the input file.
    unique_rows = dedup_by_package(rows)

    # Stratified 80/10/10 split via two sklearn train_test_split calls.
    try:
        train_rows, val_rows, test_rows = stratified_split(
            unique_rows, seed=seed
        )
    except ValueError as exc:
        print(f"ERROR: stratified split failed: {exc}", file=sys.stderr)
        return 1

    # Write all three splits via the shared atomic CSV writer.
    write_split_csv(train_rows, output_dir / "train.csv")
    write_split_csv(val_rows, output_dir / "val.csv")
    write_split_csv(test_rows, output_dir / "test.csv")

    # Summary to stdout — useful for CI logs and the operator running
    # the pipeline by hand.  Includes the dedup delta so a corpus with
    # accidental duplicates is visible at a glance.
    duplicates_dropped = len(rows) - len(unique_rows)
    print(
        f"Splits written (seed={seed}): "
        f"input={len(rows)} unique={len(unique_rows)} "
        f"dropped_duplicates={duplicates_dropped} "
        f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
