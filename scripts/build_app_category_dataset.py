"""Dataset_Builder for App Category ML Classifier.

Merges raw CSV outputs of ``scripts/crawlers/*`` and the bootstrap seed
derived from ``RuleBasedAppCategoryClassifier`` into a single
``datasets/categories/labeled.csv`` plus ``build_report.json``. Pure
helper functions (``normalize_label``, ``merge_sources``,
``map_category``, ``Counters``, …) and the orchestrator's main() come
in tasks 2.x and 3.4 of ``app-category-ml-classifier/tasks.md``.

This module currently provides:

* :class:`Source` — priority enum for the corpus origins
  (Bootstrap_Seed > Google Play Store > RuStore > Huawei AppGallery).
* :class:`Row` — minimal record shape
  ``(packageName, label, category, source)`` shared between the helper
  pipeline and the CSV writer.
* :func:`merge_sources` — task 2.3: priority-aware dedup of crawler
  rows by ``packageName`` (Requirements 1.2, 1.3).
* :func:`map_category` — task 2.5: normalize a raw category string to
  one of the 20 :data:`KOTLIN_APP_CATEGORY_ORDER` names or to
  ``"OTHER"`` with an unknown flag (Requirement 1.10).
* :func:`is_blank_package` — task 2.5: classify rows whose
  ``packageName`` is empty after :func:`str.strip` (Requirement 1.9).
* :class:`Counters` — task 2.5: per-run row counters that the build
  report (Requirement 1.11) is rendered from.
* :func:`write_labeled_csv` — task 3.1: atomic, UTF-8/no-BOM, LF-only
  CSV emitter for the labeled corpus (Requirement 1.5).
* :func:`validate_corpus_size` — task 3.3: post-dedup validation that
  the corpus has ≥ 200 000 unique packages and ≥ 5 000 packages in
  every :data:`AppCategory` except ``OTHER`` (Requirement 1.6). Pure
  (no ``sys.exit``) — returns one human-readable failure message per
  breached floor; the orchestrator's main() (task 3.4) prints each
  line to stderr and exits with code 1 when the list is non-empty.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Iterable

# Single source of truth for the 20-element :class:`AppCategory` order
# lives in the training pipeline module and is shared with both the
# Model_Card and the Kotlin enum (see Requirements 2.9, 2.10, 6.1).
# Importing it here keeps :func:`map_category` and :class:`Counters`
# tied to the same canonical list rather than duplicating it.
# ``train_app_category_classifier`` only does stdlib imports at module
# scope, so this does not pull in TensorFlow.
from train_app_category_classifier import KOTLIN_APP_CATEGORY_ORDER

LABELED_CSV_HEADER: tuple[str, str, str] = ("packageName", "label", "category")

#: Set of valid :data:`AppCategory` names, derived once from the
#: canonical ``KOTLIN_APP_CATEGORY_ORDER`` list. Used by
#: :func:`map_category` to decide whether a raw category string
#: matches a known enum value (Requirement 1.10).
_VALID_CATEGORY_NAMES: frozenset[str] = frozenset(KOTLIN_APP_CATEGORY_ORDER)


def normalize_label(s: str) -> str:
    """Normalise a raw app label per Requirement 1.4.

    Rules:

    * Apply Unicode NFC composition (``unicodedata.normalize("NFC", s)``).
    * Strip leading and trailing whitespace.
    * If the resulting string is empty *or* its Unicode-character length
      exceeds 200, return ``""``.

    Returning ``""`` for both fall-through cases means callers do not
    need to distinguish "missing label" from "label too long" — the
    on-disk corpus and the runtime tokenizer treat both identically
    (the encoder concatenates ``packageName + " " + label`` only when
    the label is non-empty).
    """
    if not isinstance(s, str):
        raise TypeError("normalize_label expects a str")
    nfc = unicodedata.normalize("NFC", s).strip()
    if not nfc or len(nfc) > 200:
        return ""
    return nfc


class Source(IntEnum):
    """Priority ranking of corpus origins.

    The integer value encodes priority directly: **lower value = higher
    priority**. ``BOOTSTRAP`` wins over every crawler so any package
    that ``RuleBasedAppCategoryClassifier`` already classifies with
    high confidence keeps its rule-based category in the training
    corpus (Requirement 1.2). Among crawlers, Google Play Store beats
    RuStore, which beats Huawei AppGallery (Requirement 1.3).
    """

    BOOTSTRAP = 0
    PLAY = 1
    RUSTORE = 2
    APPGALLERY = 3


@dataclass(frozen=True)
class Row:
    """A single ``(packageName, label, category, source)`` record.

    The first three fields mirror the CSV column order from Requirement
    1.5; ``source`` is an in-memory tag used by :func:`merge_sources`
    to pick a winner per ``packageName`` and is **not** written to the
    final ``labeled.csv``. CSV writers therefore project explicitly
    onto ``(packageName, label, category)`` rather than relying on
    :func:`dataclasses.astuple`.
    """

    packageName: str
    label: str
    category: str
    source: Source


def merge_sources(rows: Iterable[Row]) -> list[Row]:
    """Dedup ``rows`` by ``packageName`` using :class:`Source` priority.

    For every distinct ``packageName`` (compared **case-sensitively**,
    Requirement 1.3) the row with the lowest ``source.value`` wins:
    ``BOOTSTRAP > PLAY > RUSTORE > APPGALLERY``. The winning row is
    kept in full, so both ``label`` and ``category`` come from the
    same prioritised record — we never mix a label from one source
    with a category from another.

    Iteration is stable in two senses:

    * **Output order** follows the order in which each ``packageName``
      first appears in ``rows``. A later, higher-priority record for
      the same package replaces the value at the original slot
      without moving it (relies on Python ``dict`` insertion-order
      semantics on ``__setitem__`` for an existing key).
    * **Tie-breaking** within the same ``Source`` is "first wins":
      among rows with equal ``source.value`` we keep the earliest
      occurrence in input order.

    The function does not mutate its input and consumes the iterable
    exactly once.

    Parameters
    ----------
    rows:
        Iterable of :class:`Row` records, typically the concatenation
        of all crawler outputs and the Bootstrap_Seed.

    Returns
    -------
    list[Row]
        Deduplicated rows, one per unique ``packageName``, in
        first-appearance order.
    """
    chosen: dict[str, Row] = {}
    for row in rows:
        existing = chosen.get(row.packageName)
        if existing is None:
            # First time we see this packageName — record it. The
            # insertion slot is now anchored for the whole pass.
            chosen[row.packageName] = row
        elif row.source.value < existing.source.value:
            # Strictly higher priority (lower int) replaces the value
            # in place. dict preserves insertion order on
            # __setitem__ for an existing key, so the slot doesn't
            # move. Equal priority falls through (first-wins).
            chosen[row.packageName] = row
    return list(chosen.values())


def map_category(raw: str) -> tuple[str, bool]:
    """Map a raw category string onto an :data:`AppCategory` name.

    Implements Requirement 1.10: a row's ``category`` field, after
    ``str.strip().upper()``, must either match one of the 20 enum
    values in :data:`KOTLIN_APP_CATEGORY_ORDER` exactly (returned
    as-is, with ``unknown=False``), or fall through to ``"OTHER"``
    with ``unknown=True`` so the orchestrator can bump the
    ``unknown_category_rows`` counter for the build report
    (Requirement 1.11).

    The returned tuple is ``(normalized_name, unknown)``:

    * ``normalized_name`` is always one of the 20 canonical strings
      (i.e. an element of :data:`KOTLIN_APP_CATEGORY_ORDER`).
    * ``unknown`` is ``True`` only when ``raw`` did not match any
      known category and was therefore coerced to ``"OTHER"``.
      A literal ``"other"`` / ``"OTHER"`` from the source is
      considered known and returns ``("OTHER", False)``.

    Parameters
    ----------
    raw:
        Category string as obtained from a crawler row. Leading and
        trailing whitespace is tolerated; case is normalized via
        :meth:`str.upper`.

    Returns
    -------
    tuple[str, bool]
        ``(normalized_category, unknown_flag)``.
    """
    normalized = raw.strip().upper()
    if normalized in _VALID_CATEGORY_NAMES:
        return normalized, False
    return "OTHER", True


def is_blank_package(row: Row) -> bool:
    """Return ``True`` iff ``row.packageName`` is empty after strip.

    Implements Requirement 1.9's blank-package check: the
    orchestrator drops rows for which ``packageName`` is missing or
    whitespace-only, atomically incrementing ``dropped_rows`` in the
    build report. Centralising the predicate keeps the orchestrator
    readable and gives the Property 13 PBT (task 2.6) one clear
    function to assert against.

    The check is intentionally *only* over ``packageName`` — labels
    and categories are normalised separately
    (:func:`map_category` for the latter, label normalization in task
    2.1) and a row with an empty label but a real package is *not*
    considered blank.
    """
    return row.packageName.strip() == ""


@dataclass
class Counters:
    """Per-run row counters that feed into the build report.

    The five fields mirror the exact JSON keys the build report needs
    to expose under Requirement 1.11. ``per_category_counts`` is
    pre-populated with all 20 :data:`KOTLIN_APP_CATEGORY_ORDER` names
    set to zero so the report always carries the full key set,
    including categories that produced no rows in this run.

    Field semantics (Requirement 1.11):

    * ``total_input_rows`` — every row read from any source
      (crawler outputs + Bootstrap_Seed), before any filtering. The
      orchestrator increments this once per raw row, regardless of
      what happens next.
    * ``dropped_rows`` — rows dropped because of a blank
      ``packageName`` (see :func:`is_blank_package`, Requirement 1.9).
      Bumped atomically before the row is discarded so the counter
      stays consistent even if the subsequent skip is short-circuited.
    * ``unknown_category_rows`` — rows whose raw category did not
      match any of the 20 known names and were therefore coerced to
      ``"OTHER"`` by :func:`map_category` (Requirement 1.10). A row
      can be both ``unknown_category`` and counted in
      ``per_category_counts["OTHER"]``.
    * ``corpus_rows`` — final number of unique-package rows that
      survived blank-package filtering and source-merge dedup
      (Requirement 1.3). This is the size of the on-disk
      ``labeled.csv`` minus its header.
    * ``per_category_counts`` — number of rows per category in the
      final corpus. Sum of values equals ``corpus_rows``. Always
      contains exactly 20 keys so the JSON shape is invariant to
      which categories are populated in a given build.
    """

    total_input_rows: int = 0
    dropped_rows: int = 0
    unknown_category_rows: int = 0
    corpus_rows: int = 0
    per_category_counts: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in KOTLIN_APP_CATEGORY_ORDER}
    )


def write_labeled_csv(rows: Iterable[Row], path: Path) -> None:
    """Atomically write the labeled corpus to ``path``.

    Format guarantees (Requirement 1.5):

    * UTF-8 encoding **without** BOM.
    * LF (``\\n``) line terminator on every row, including the last one
      (i.e. trailing newline).
    * Header line ``packageName,label,category`` as the first row.
    * Data rows in the iteration order of ``rows``, with columns in the
      order ``packageName,label,category``. The in-memory ``source``
      tag on each :class:`Row` is intentionally not emitted.

    Atomicity: the file is first written to ``<path>.tmp`` and then
    renamed onto ``path`` via :func:`os.replace`, which is atomic on
    POSIX and on NTFS. On any failure during the write phase the
    temporary file is best-effort cleaned up and the original
    exception is re-raised so callers can decide how to recover.

    Parameters
    ----------
    rows:
        Iterable of :class:`Row` records. Consumed exactly once.
    path:
        Destination path of the final ``labeled.csv``. Parent
        directories are created if they do not yet exist.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # ``newline=""`` keeps Python from translating "\n" to the OS
        # native line separator on Windows; ``encoding="utf-8"`` writes
        # bytes without a BOM (use "utf-8-sig" to opt into a BOM, which
        # we deliberately do not).
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(LABELED_CSV_HEADER)
            for row in rows:
                writer.writerow((row.packageName, row.label, row.category))
        os.replace(tmp, path)
    except BaseException:
        # Best-effort cleanup of the partial temp file; never mask the
        # original exception with a cleanup failure.
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        raise


def write_build_report(
    counters: Counters,
    seed: int,
    path: Path,
    *,
    built_at: datetime | None = None,
) -> None:
    """Atomically write the build report to ``path``.

    Implements Requirement 1.11: the report captures pipeline statistics
    so downstream consumers (CI, data scientists, the Property 13 PBT)
    can verify the corpus composition without re-running the full build.

    Format guarantees:

    * JSON object with keys: ``schema_version``, ``total_input_rows``,
      ``dropped_rows``, ``unknown_category_rows``, ``corpus_rows``,
      ``per_category_counts`` (object with exactly 20 keys — all
      :data:`KOTLIN_APP_CATEGORY_ORDER` names including ``OTHER``),
      ``seed``, ``built_at`` (ISO-8601 UTC string).
    * UTF-8 encoding without BOM.
    * LF line endings (``json.dump`` with ``ensure_ascii=False``).
    * Trailing newline after the closing ``}``.
    * Keys in ``per_category_counts`` are emitted in
      :data:`KOTLIN_APP_CATEGORY_ORDER` order for deterministic output.

    Atomicity: the file is first written to ``<path>.tmp`` and then
    renamed onto ``path`` via :func:`os.replace`. On any failure during
    the write phase the temporary file is best-effort cleaned up and the
    original exception is re-raised.

    Parameters
    ----------
    counters:
        A :class:`Counters` instance populated by the orchestrator after
        processing all rows.
    seed:
        The ``--seed`` value used for this build (Requirement 1.8).
    path:
        Destination path of the final ``build_report.json``. Parent
        directories are created if they do not yet exist.
    built_at:
        Optional override for the ``built_at`` timestamp. When ``None``
        (the default), the current UTC time is used. Exposed for
        deterministic testing — callers can inject a fixed timestamp to
        produce byte-identical reports across runs.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)

    if built_at is None:
        built_at = datetime.now(timezone.utc)

    # Format as ISO-8601 UTC with 'Z' suffix (no +00:00 form).
    built_at_str = built_at.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Ensure per_category_counts is emitted in canonical order for
    # deterministic output (Requirement 1.8 / Property 12).
    ordered_counts = {
        name: counters.per_category_counts.get(name, 0)
        for name in KOTLIN_APP_CATEGORY_ORDER
    }

    report = {
        "schema_version": 1,
        "total_input_rows": counters.total_input_rows,
        "dropped_rows": counters.dropped_rows,
        "unknown_category_rows": counters.unknown_category_rows,
        "corpus_rows": counters.corpus_rows,
        "per_category_counts": ordered_counts,
        "seed": seed,
        "built_at": built_at_str,
    }

    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")  # trailing newline
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        raise


# ── Corpus-size validation (Requirement 1.6) ──────────────────────────────
#
# After source-merge dedup but before the orchestrator declares the build
# successful, two minimum-size invariants must hold on the final
# Labelled_Corpus:
#
#   * len(unique_packages)         ≥ 200 000  — total corpus floor
#   * per_category_counts[cat]     ≥   5 000  for every cat ∈ AppCategory
#                                              \ {OTHER}
#
# ``OTHER`` is intentionally excluded because (a) it is the rule-based
# fallback bucket and is not predicted by the 18-class softmax head
# (Requirement 2.2), and (b) a bloated ``OTHER`` count would dwarf the
# 18 trainable classes during stratified split.
#
# :func:`validate_corpus_size` is pure: it returns a list of
# human-readable failure messages — empty iff every floor is cleared —
# so callers (the orchestrator's main() in task 3.4 and any future
# property test) can act on the result without :func:`sys.exit` /
# :data:`sys.stderr` side effects bleeding into test output. The
# orchestrator surfaces a non-empty list as exit code 1 with one stderr
# line per message (Requirement 1.6).

#: Minimum number of unique packages the final corpus must contain
#: (Requirement 1.6).  Hard-coded into the failure message so log
#: scrapers looking for the exact integer keep working.
CORPUS_MIN_UNIQUE_PACKAGES: int = 200_000

#: Minimum number of packages every non-OTHER :data:`AppCategory` must
#: have in the final corpus (Requirement 1.6).
CORPUS_MIN_PER_CATEGORY: int = 5_000


def validate_corpus_size(rows: list[Row], counters: Counters) -> list[str]:
    """Return human-readable failure messages for any breached corpus floor.

    Implements Requirement 1.6 as a **pure** function: no I/O, no
    :func:`sys.exit`, no :data:`sys.stderr` writes. The orchestrator's
    main() (task 3.4) is responsible for surfacing a non-empty result
    as exit code 1 with one stderr line per message; keeping this
    function side-effect-free lets unit tests assert against the
    returned list directly without capturing process I/O.

    The two floors checked, after source-merge dedup, are:

        len(unique_packages)        ≥ CORPUS_MIN_UNIQUE_PACKAGES (200 000)
        per_category_counts[cat]    ≥ CORPUS_MIN_PER_CATEGORY    (  5 000)
                                       for every cat in
                                       KOTLIN_APP_CATEGORY_ORDER \\ {OTHER}

    The unique-package count is taken from ``rows`` rather than from
    ``counters.corpus_rows`` so the function is robust against a
    miscounting orchestrator: the contract is that ``rows`` is the
    post-dedup corpus that will be written to ``labeled.csv``.
    Counting distinct ``packageName`` values via :class:`set` is O(n)
    and never overcounts even if the caller accidentally hands in a
    list with duplicates.

    Failure ordering is deterministic so the orchestrator's stderr
    summary is reproducible across runs:

      1. The unique-packages floor (if breached) is reported first.
      2. Per-category floors follow in :data:`KOTLIN_APP_CATEGORY_ORDER`
         order (BANK first, PRODUCTIVITY last). ``OTHER`` is skipped.

    A category that is missing from ``counters.per_category_counts`` is
    treated as ``0`` (i.e. a failure against the 5 000 floor) rather
    than raising :class:`KeyError` — :class:`Counters` always
    pre-populates all 20 keys, but this keeps the function safe when
    called from tests with a partially populated dict.

    Message format (frozen, see task 3.3):

      * ``"corpus has <N> unique packages < 200000 minimum"``
      * ``"category <NAME> has <N> rows < 5000 minimum"``

    The integers are inlined verbatim (no thousands separator) so a
    grep against the canonical floors stays trivial.

    Parameters
    ----------
    rows:
        Post-dedup corpus rows (output of :func:`merge_sources`). Used
        for the unique-package count via ``len({r.packageName for r in
        rows})``.
    counters:
        Run-level :class:`Counters` aggregated by the orchestrator.
        Only ``per_category_counts`` is consulted here; the other
        fields are scoped to the build report (Requirement 1.11).

    Returns
    -------
    list[str]
        One failure message per breached floor, in the deterministic
        order described above. Empty iff every floor is cleared.
    """
    failures: list[str] = []

    # Recompute the unique-package count from ``rows`` rather than
    # trusting ``counters.corpus_rows`` — the latter is the
    # orchestrator's bookkeeping and not authoritative for this gate.
    unique_packages = len({row.packageName for row in rows})
    if unique_packages < CORPUS_MIN_UNIQUE_PACKAGES:
        failures.append(
            f"corpus has {unique_packages} unique packages "
            f"< {CORPUS_MIN_UNIQUE_PACKAGES} minimum"
        )

    per_category = counters.per_category_counts
    for category in KOTLIN_APP_CATEGORY_ORDER:
        if category == "OTHER":
            # OTHER is the rule-based fallback bucket (Requirement 2.2);
            # the 5 000-per-category floor explicitly excludes it.
            continue
        count = int(per_category.get(category, 0))
        if count < CORPUS_MIN_PER_CATEGORY:
            failures.append(
                f"category {category} has {count} rows "
                f"< {CORPUS_MIN_PER_CATEGORY} minimum"
            )

    return failures


def _load_csv_rows(path: Path, source: Source) -> list[Row]:
    """Load a raw CSV file into a list of :class:`Row` records.

    Expects a CSV with header ``packageName,label,category`` (the same
    format produced by the crawler scripts and the bootstrap seed
    generator). Each data row is wrapped into a :class:`Row` with the
    given ``source`` tag for priority-aware merge.

    The file is read with ``encoding="utf-8-sig"`` to tolerate an
    optional BOM (some Windows tools emit one); the BOM is stripped
    transparently by the codec. Missing ``label`` fields default to
    ``""`` so downstream :func:`normalize_label` always receives a
    string.

    Parameters
    ----------
    path:
        Path to the CSV file. Must exist and be readable.
    source:
        The :class:`Source` priority tag to attach to every row.

    Returns
    -------
    list[Row]
        One :class:`Row` per data line in the CSV (header excluded).
    """
    rows: list[Row] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for record in reader:
            package_name = record.get("packageName", "") or ""
            label = record.get("label", "") or ""
            category = record.get("category", "") or ""
            rows.append(Row(
                packageName=package_name,
                label=label,
                category=category,
                source=source,
            ))
    return rows


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the dataset builder.

    Arguments
    ---------
    --seed : int
        Random seed passed through to the build report and downstream
        split builder. Default 42 (Requirement 1.8).
    --play-store : path
        Path to the Google Play Store raw CSV
        (default ``datasets/categories/raw/play_store.csv``).
    --rustore : path
        Path to the RuStore raw CSV
        (default ``datasets/categories/raw/rustore.csv``).
    --appgallery : path
        Path to the Huawei AppGallery raw CSV
        (default ``datasets/categories/raw/appgallery.csv``).
    --bootstrap-seed : path
        Path to the Bootstrap_Seed CSV generated from
        ``RuleBasedAppCategoryClassifier`` known packages
        (default ``datasets/categories/raw/bootstrap_seed.csv``).
    --output-dir : path
        Directory for output files (``labeled.csv``,
        ``build_report.json``). Default ``datasets/categories/``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the App Category labeled corpus from raw crawler "
            "outputs and the RuleBasedAppCategoryClassifier bootstrap seed."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--play-store",
        type=Path,
        default=Path("datasets/categories/raw/play_store.csv"),
        help="Path to Google Play Store raw CSV.",
    )
    parser.add_argument(
        "--rustore",
        type=Path,
        default=Path("datasets/categories/raw/rustore.csv"),
        help="Path to RuStore raw CSV.",
    )
    parser.add_argument(
        "--appgallery",
        type=Path,
        default=Path("datasets/categories/raw/appgallery.csv"),
        help="Path to Huawei AppGallery raw CSV.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=Path,
        default=Path("datasets/categories/raw/bootstrap_seed.csv"),
        help="Path to Bootstrap_Seed CSV from RuleBasedAppCategoryClassifier.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/categories"),
        help="Output directory for labeled.csv and build_report.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Orchestrate the full dataset build pipeline.

    Sequence (Requirements 1.1–1.6, 1.9–1.11):

      1. Parse CLI arguments (``--seed``, source paths, output dir).
      2. Load raw CSV sources (Play Store, RuStore, AppGallery).
      3. Load Bootstrap_Seed CSV (from ``RuleBasedAppCategoryClassifier``
         known packages — highest priority).
      4. Concatenate all rows, tracking ``total_input_rows``.
      5. For each row:
         a. Drop if ``packageName`` is blank (Req 1.9) → bump
            ``dropped_rows``.
         b. Normalize ``label`` via :func:`normalize_label` (Req 1.4).
         c. Map ``category`` via :func:`map_category` (Req 1.10) →
            bump ``unknown_category_rows`` if unknown.
      6. Merge sources with priority dedup via :func:`merge_sources`
         (Req 1.2, 1.3).
      7. Compute ``per_category_counts`` and ``corpus_rows`` from the
         merged result.
      8. Write ``labeled.csv`` via :func:`write_labeled_csv` (Req 1.5).
      9. Write ``build_report.json`` via :func:`write_build_report`
         (Req 1.11).
     10. Validate corpus size via :func:`validate_corpus_size`
         (Req 1.6). On failure → exit 1 with stderr messages.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on corpus-size validation failure.
    """
    args = _parse_args(argv)

    # ── Step 1–3: Load all sources ────────────────────────────────────
    all_raw_rows: list[Row] = []

    # Bootstrap_Seed has highest priority (Source.BOOTSTRAP = 0).
    if args.bootstrap_seed.exists():
        all_raw_rows.extend(_load_csv_rows(args.bootstrap_seed, Source.BOOTSTRAP))

    # Crawlers in priority order: Play > RuStore > AppGallery.
    if args.play_store.exists():
        all_raw_rows.extend(_load_csv_rows(args.play_store, Source.PLAY))

    if args.rustore.exists():
        all_raw_rows.extend(_load_csv_rows(args.rustore, Source.RUSTORE))

    if args.appgallery.exists():
        all_raw_rows.extend(_load_csv_rows(args.appgallery, Source.APPGALLERY))

    # ── Step 4–5: Process rows (drop blanks, normalize, map) ──────────
    counters = Counters()
    counters.total_input_rows = len(all_raw_rows)

    processed_rows: list[Row] = []
    for row in all_raw_rows:
        # 5a. Drop blank packageName (Req 1.9).
        # Counter is bumped atomically before the row is discarded.
        if is_blank_package(row):
            counters.dropped_rows += 1
            continue

        # 5b. Normalize label (Req 1.4).
        normalized_label = normalize_label(row.label)

        # 5c. Map category (Req 1.10).
        mapped_category, is_unknown = map_category(row.category)
        if is_unknown:
            counters.unknown_category_rows += 1

        processed_rows.append(Row(
            packageName=row.packageName,
            label=normalized_label,
            category=mapped_category,
            source=row.source,
        ))

    # ── Step 6: Merge sources with priority dedup (Req 1.2, 1.3) ─────
    merged_rows = merge_sources(processed_rows)

    # ── Step 7: Compute final counters ────────────────────────────────
    counters.corpus_rows = len(merged_rows)
    for row in merged_rows:
        counters.per_category_counts[row.category] = (
            counters.per_category_counts.get(row.category, 0) + 1
        )

    # ── Step 8: Write labeled.csv (Req 1.5) ──────────────────────────
    output_dir = args.output_dir
    labeled_path = output_dir / "labeled.csv"
    write_labeled_csv(merged_rows, labeled_path)

    # ── Step 9: Write build_report.json (Req 1.11) ───────────────────
    report_path = output_dir / "build_report.json"
    write_build_report(counters, seed=args.seed, path=report_path)

    # ── Step 10: Validate corpus size (Req 1.6) ──────────────────────
    failures = validate_corpus_size(merged_rows, counters)
    if failures:
        for msg in failures:
            print(msg, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
