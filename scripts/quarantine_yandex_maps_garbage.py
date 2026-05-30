#!/usr/bin/env python3
"""One-shot cleanup for the yandex_maps garbage incident (#4).

Splits ``datasets/ru/raw/legitimate_numbers.csv`` into two files:

  * ``legitimate_numbers.csv`` (rewritten in-place) — keep only rows that
    are not from the broken yandex_maps fallback path.  A row is kept iff
    EITHER its source is not ``yandex_maps``, OR its source is
    ``yandex_maps`` but the row carries a real business name (``name``
    column ≠ ``"Yandex Maps"`` placeholder, indicating the JSON-state
    extractor actually produced something useful).

  * ``datasets/ru/raw/quarantine/yandex_maps_unverified.csv`` — created
    fresh; receives every dropped row so we keep an audit trail and can
    inspect / re-import after the collector is rewritten.

Additionally, applies the tightened ``is_russian_number()`` DEF-code
allowlist as a sanity filter on the kept rows (drops anything that
doesn't pass the new check from any source — defensive cleanup).

Run once on the default branch:

    python3 scripts/quarantine_yandex_maps_garbage.py

The script is idempotent: if no garbage rows are found it just rewrites
the file unchanged.
"""

from __future__ import annotations

import csv
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from ru_number_normalizer import is_russian_number  # noqa: E402

ALLOW_CSV = os.path.join(ROOT, 'datasets', 'ru', 'raw', 'legitimate_numbers.csv')
QUARANTINE_DIR = os.path.join(ROOT, 'datasets', 'ru', 'raw', 'quarantine')
QUARANTINE_CSV = os.path.join(QUARANTINE_DIR, 'yandex_maps_unverified.csv')


def main() -> int:
    if not os.path.isfile(ALLOW_CSV):
        print(f'no such file: {ALLOW_CSV}')
        return 1

    with open(ALLOW_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    keep, drop = [], []
    for r in rows:
        src = (r.get('source') or '').strip()
        name = (r.get('name') or '').strip()
        normalized = (r.get('normalized_number') or '').strip()

        # Reason 1: yandex_maps with the placeholder name = fallback regex
        # garbage. Drop unconditionally.
        if src == 'yandex_maps' and name == 'Yandex Maps':
            drop.append(r)
            continue

        # Reason 2: number doesn't pass the new strict DEF-code allowlist.
        # This catches cross-source pollution (e.g. orgpage scrape that
        # picked up a Yandex footer phone). Defensive — should be a small
        # number of rows.
        if normalized and not is_russian_number(normalized):
            drop.append(r)
            continue

        keep.append(r)

    os.makedirs(QUARANTINE_DIR, exist_ok=True)

    # Write quarantine first so we never lose data.
    with open(QUARANTINE_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in drop:
            writer.writerow(r)

    # Rewrite the main allow CSV with only kept rows.
    with open(ALLOW_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in keep:
            writer.writerow(r)

    print(f'input rows:       {len(rows)}')
    print(f'kept (rewritten): {len(keep)}')
    print(f'quarantined:      {len(drop)}  → {os.path.relpath(QUARANTINE_CSV, ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
