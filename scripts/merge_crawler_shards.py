"""Merge per-shard crawler outputs into the master ru_reputation_raw.csv.

GitHub Actions runs the crawler as 3 parallel shards (a/b/c), each writing
its own CSV under datasets/ru/raw/shards/<shard>/raw.csv. This script
appends new rows from those shards into the canonical master file.

Important: the master CSV intentionally contains multiple snapshot rows
for the same (number, source) pair (different detail_date / review_count
values across crawl runs). Downstream `ru_metadata_dataset_builder.py`
aggregates them into per-number stats. So this script does NOT dedupe
the master — it only **adds** shard rows that are not already present
verbatim in master.
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from typing import Dict, Set, Tuple

# Add scripts/ to path so we can reuse translate_row from ru_metadata_features.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from ru_metadata_features import translate_row, translate_headers  # type: ignore
except Exception:  # pragma: no cover
    translate_row = None  # type: ignore
    translate_headers = None  # type: ignore

from ru_number_normalizer import is_valid_ru_phone


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(BASE_DIR, 'datasets', 'ru', 'raw')
MASTER = os.path.join(RAW_DIR, 'ru_reputation_raw.csv')
EVIDENCE_MASTER = os.path.join(RAW_DIR, 'ru_reputation_evidence.csv')
SHARDS_GLOB = os.path.join(RAW_DIR, 'shards', '*', 'raw.csv')
EVIDENCE_GLOB = os.path.join(RAW_DIR, 'shards', '*', 'evidence.csv')


# Russian -> English column-name aliases used by ru_reputation_crawler.py
# when it writes shard CSVs (the master CSV uses English headers).
_RU_ALIASES = {
    'номер': 'normalized_number',
    'источник': 'source',
    'негативных': 'negative_count',
    'позитивных': 'positive_count',
    'нейтральных': 'neutral_count',
    'отзывов': 'review_count',
    'объём_поиска': 'search_volume',
    'категории': 'categories',
    'последний_отзыв': 'last_review_at',
    'первое_появление': 'first_seen_at',
    'надёжность_источника': 'source_reliability',
    'уверенность_источника': 'source_confidence',
    'дата_страницы': 'detail_date',
    'просмотров': 'view_count',
    'связанных': 'related_count',
    'заголовок': 'page_title',
    'url': 'url',
}


def _normalize_row(row: Dict[str, str]) -> Dict[str, str]:
    """Map Russian column names back to canonical English ones."""
    out: Dict[str, str] = {}
    for k, v in row.items():
        if k is None:
            continue
        canonical = _RU_ALIASES.get(k, k)
        out[canonical] = v
    return out


def _row_signature(row: Dict[str, str]) -> Tuple[str, ...]:
    """Tuple of fields that uniquely identifies a snapshot row.

    Two rows differing only on review_count or detail_date are considered
    distinct snapshots and both kept.
    """
    return (
        row.get('normalized_number', ''),
        row.get('source', ''),
        row.get('detail_date', ''),
        row.get('review_count', ''),
        row.get('negative_count', ''),
        row.get('positive_count', ''),
        row.get('last_review_at', ''),
    )


def _merge_append(master_path: str, shard_paths) -> int:
    if not os.path.exists(master_path):
        with open(master_path, 'w', encoding='utf-8') as f:
            pass

    fieldnames = None
    seen: Set[Tuple[str, ...]] = set()
    existing_rows = 0

    if os.path.getsize(master_path) > 0:
        with open(master_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            master_fields = reader.fieldnames or []
            # Master uses English headers; preserve them as the canonical schema.
            fieldnames = [_RU_ALIASES.get(h, h) for h in master_fields]
            for row in reader:
                normalized = _normalize_row(row)
                seen.add(_row_signature(normalized))
                existing_rows += 1

    new_rows = []
    dropped_invalid = 0
    for shard_path in shard_paths:
        if not os.path.exists(shard_path):
            continue
        with open(shard_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            shard_fields = reader.fieldnames or []
            if fieldnames is None:
                fieldnames = [_RU_ALIASES.get(h, h) for h in shard_fields]
            for row in reader:
                normalized = _normalize_row(row)
                num = normalized.get('normalized_number')
                if not num or not normalized.get('source'):
                    continue
                # Drop non-RU (Kazakhstan +77XX, invalid def-codes +70XX/+71XX/+72XX),
                # and round-number placeholders. См. is_valid_ru_phone().
                if not is_valid_ru_phone(num):
                    dropped_invalid += 1
                    continue
                sig = _row_signature(normalized)
                if sig in seen:
                    continue
                seen.add(sig)
                new_rows.append(normalized)
    if dropped_invalid:
        print(f'merge_crawler_shards: dropped {dropped_invalid} invalid rows '
              f'(non-RU/placeholder) from {master_path}')

    if fieldnames is None:
        print('no shards and no master — nothing to do', file=sys.stderr)
        return 0

    if not new_rows:
        print(f'master {master_path}: {existing_rows} rows, no new shard rows')
        return 0

    write_header = existing_rows == 0
    with open(master_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in new_rows:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    print(f'master {master_path}: {existing_rows} -> {existing_rows + len(new_rows)} '
          f'(+{len(new_rows)} new snapshot rows from {len(shard_paths)} shards)')
    return len(new_rows)


def main() -> int:
    shards = sorted(glob.glob(SHARDS_GLOB))
    if not shards:
        print('no shard CSVs found under', SHARDS_GLOB)
        return 0

    _merge_append(MASTER, shards)
    evidence_shards = sorted(glob.glob(EVIDENCE_GLOB))
    if evidence_shards:
        _merge_append(EVIDENCE_MASTER, evidence_shards)
    return 0


if __name__ == '__main__':
    sys.exit(main())
