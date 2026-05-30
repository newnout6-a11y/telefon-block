"""Merge per-shard ALLOW collector outputs into the master legitimate_numbers.csv.

The crawl-allow-keepalive matrix runs 5 shards (org-fed, org-civic, catalogs,
geo, weak), each writing to its own
``datasets/ru/raw/legitimate_shards/<shard>/legitimate.csv``. The
crawl-allow-osm-wikidata workflow adds two more shards (osm, wikidata).
This script concatenates everything into the canonical master CSV at
``datasets/ru/raw/legitimate_numbers.csv``.

Dedup policy: one entry per ``normalized_number``. When the same number
appears in multiple shards we keep the row with the **highest
source_confidence** (tiebreaker: first encountered). This biases the master
toward authoritative sources (banks/gov/Wikidata) over weaker ones
(rusprofile listings, classifieds) for the same phone.

Schema expected (matches ru_legitimate_collector.py output):
    normalized_number, name, category, source, city, url, source_confidence
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from typing import Dict, Iterable, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ru_number_normalizer import is_valid_ru_phone  # type: ignore

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(BASE_DIR, 'datasets', 'ru', 'raw')
MASTER = os.path.join(RAW_DIR, 'legitimate_numbers.csv')
SHARDS_GLOB = os.path.join(RAW_DIR, 'legitimate_shards', '*', 'legitimate.csv')

FIELDNAMES = [
    'normalized_number', 'name', 'category', 'source', 'city', 'url',
    'source_confidence',
]


def _parse_conf(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.70


def _iter_csv_rows(path: str) -> Iterable[Dict[str, str]]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _ingest(rows: Iterable[Dict[str, str]],
            store: Dict[str, Tuple[float, Dict[str, str]]]) -> Tuple[int, int]:
    added = 0
    upgraded = 0
    for row in rows:
        num = (row.get('normalized_number') or '').strip()
        if not num or not is_valid_ru_phone(num):
            continue
        conf = _parse_conf(row.get('source_confidence', ''))
        prev = store.get(num)
        if prev is None:
            store[num] = (conf, row)
            added += 1
        elif conf > prev[0]:
            store[num] = (conf, row)
            upgraded += 1
    return added, upgraded


def main() -> int:
    shards = sorted(glob.glob(SHARDS_GLOB))
    if not shards and not os.path.exists(MASTER):
        print(f'no shards under {SHARDS_GLOB} and no master — nothing to do',
              file=sys.stderr)
        return 0

    store: Dict[str, Tuple[float, Dict[str, str]]] = {}

    if os.path.exists(MASTER):
        master_added, _ = _ingest(_iter_csv_rows(MASTER), store)
        print(f'  master: {master_added} existing rows loaded')

    for shard_path in shards:
        added, upgraded = _ingest(_iter_csv_rows(shard_path), store)
        rel = os.path.relpath(shard_path, BASE_DIR)
        print(f'  {rel}: +{added} new, ↑{upgraded} upgraded')

    # Write the master atomically.
    os.makedirs(os.path.dirname(MASTER), exist_ok=True)
    tmp = MASTER + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        # Stable order: by category, then number — same as
        # ru_legitimate_collector.save_results so diffs stay readable.
        for num in sorted(store.keys(),
                          key=lambda n: (store[n][1].get('category', ''), n)):
            row = store[num][1]
            writer.writerow({k: row.get(k, '') for k in FIELDNAMES})
    os.replace(tmp, MASTER)

    print(f'master {MASTER}: {len(store)} unique numbers '
          f'from {len(shards)} shards')
    return 0


if __name__ == '__main__':
    sys.exit(main())
