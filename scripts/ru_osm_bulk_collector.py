"""
OSM bulk PBF collector for legitimate Russian organization phones.

Phase-1 ALLOW-side leverage: replaces the incremental Overpass API approach in
``ru_osm_collector.py`` with a single-pass scan of the Geofabrik
``russia-latest.osm.pbf`` extract. The Overpass collector is rate-limited to
~60 sec per category query and times out on broad scans, so it caps out at
~80k unique RU phones per run regardless of budget. The bulk PBF parses the
entire Russia extract (~3.5 GB) in one streamed pass, which yields **every**
element with ``phone`` / ``contact:phone`` / ``phone:mobile`` /
``contact:mobile`` tags — typically 200-400k unique RU phones.

The collector is an append-with-dedup citizen alongside the existing OSM
Overpass collector (``ru_osm_collector.py``) and the other ALLOW shards
(``ru_legitimate_collector.py`` / ``ru_wikidata_collector.py``); it writes to
the same ``legitimate.csv`` schema and is picked up by
``scripts/merge_legitimate_shards.py`` like any other shard.

Usage:
    python scripts/ru_osm_bulk_collector.py
    python scripts/ru_osm_bulk_collector.py --output /path/to/legitimate.csv
    python scripts/ru_osm_bulk_collector.py --pbf-cache /tmp/russia.osm.pbf
    python scripts/ru_osm_bulk_collector.py --dry-run

Environment:
    PBF_URL  optional override (default: https://download.geofabrik.de/russia-latest.osm.pbf)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

try:
    import requests  # type: ignore
except ImportError:
    requests = None  # type: ignore

try:
    import osmium  # type: ignore
except ImportError:
    osmium = None  # type: ignore


def _require_runtime_deps() -> None:
    """Raise a helpful error if required runtime deps are missing.

    Defer this until the bulk collector is actually invoked so unit tests
    that only exercise pure-Python helpers (e.g. _categorise) don't need
    pyosmium installed.
    """
    missing = []
    if requests is None:
        missing.append('requests')
    if osmium is None:
        missing.append('osmium')
    if missing:
        raise SystemExit(f"Required: pip install {' '.join(missing)}")

sys.path.insert(0, os.path.dirname(__file__))
from ru_number_normalizer import normalize_ru_phone, is_russian_number  # noqa: E402
from ru_osm_collector import (  # noqa: E402
    CATEGORIES,
    _split_phones,
    _city_from_tags,
    _name_from_tags,
)


OUTPUT_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'datasets', 'ru', 'raw', 'legitimate_numbers.csv'
))

PBF_URL = os.environ.get(
    'PBF_URL', 'https://download.geofabrik.de/russia-latest.osm.pbf'
)


# Build a fast (osm_tag, osm_value) → (category, feature, confidence) lookup
# from the existing CATEGORIES registry. Wildcard values (``None``) are
# stored under the sentinel ``'*'`` key.
TAG_LOOKUP: Dict[Tuple[str, str], Tuple[str, str, float]] = {}
TAG_KEYS_WILDCARD: Set[str] = set()
for _cat_name, _sub_specs in CATEGORIES.items():
    for _osm_tag, _osm_value, _confidence, _feature in _sub_specs:
        if _osm_value is None:
            TAG_LOOKUP[(_osm_tag, '*')] = (_cat_name, _feature, _confidence)
            TAG_KEYS_WILDCARD.add(_osm_tag)
        else:
            TAG_LOOKUP[(_osm_tag, _osm_value)] = (_cat_name, _feature, _confidence)


# Tags that, on their own, are strong "this is a real organization" signals
# even without a name tag. Used by the fallback path so we don't drop a
# phone-tagged element just because OSM contributors didn't fill in a name.
ORG_HINT_TAGS = (
    'amenity', 'shop', 'office', 'tourism', 'leisure', 'craft', 'healthcare',
    'industrial', 'building', 'public_transport', 'aeroway', 'historic',
    'man_made', 'club', 'emergency', 'landuse',
)

# Address tags — when present alongside a phone, indicate a real organisation
# at a real location even when both name and explicit org-hint tags are missing.
ADDR_HINT_TAGS = (
    'addr:street', 'addr:housenumber', 'addr:city',
    'addr:full', 'addr:place',
)

# Tag values that almost always indicate a non-organisation (e.g. an
# individual's phone, a generic POI, etc.). Even with phone+addr we should
# skip these to avoid contaminating the ALLOW pool.
NEGATIVE_VALUES: Set[Tuple[str, str]] = {
    ('amenity', 'bench'),
    ('amenity', 'waste_basket'),
    ('amenity', 'recycling'),
    ('amenity', 'parking_space'),
    ('amenity', 'vending_machine'),
    ('highway', 'street_lamp'),
    ('emergency', 'fire_hydrant'),
    ('barrier', '*'),
}


@dataclass
class BulkEntry:
    number: str
    name: str
    category: str
    source: str
    city: str
    url: str
    confidence: float


def _is_negative(tags: Dict[str, str]) -> bool:
    """Reject elements whose tags identify them as non-organisations
    (street furniture, generic POIs, etc.)."""
    for (key, val) in NEGATIVE_VALUES:
        if val == '*':
            if key in tags:
                return True
        else:
            if tags.get(key) == val:
                return True
    return False


def _categorise(tags: Dict[str, str]) -> Optional[Tuple[str, str, str, float]]:
    """Map an element's tags to (category, feature, source_label, confidence).

    Tries exact (key, value) matches first against the CATEGORIES registry,
    then wildcard (key, '*') matches, then progressively looser fallbacks:
      - phone + org-hint tag + readable name  → osm_other_business / 0.72
      - phone + org-hint tag (no name)        → osm_org_hint / 0.66
      - phone + address tags + name           → osm_addressed_business / 0.62
    Returns ``None`` if no signal of organisation-ness.
    """
    # Reject obvious non-organisations even before trying exact matches —
    # `phone` on a `amenity=bench` is just somebody's number scrawled on a
    # bench, not a business.
    if _is_negative(tags):
        return None

    # Exact matches
    for key, val in tags.items():
        hit = TAG_LOOKUP.get((key, val))
        if hit:
            cat, feat, conf = hit
            src = f'osm_{cat}_{val}'
            return cat, feat, src, conf

    # Wildcard matches (e.g. healthcare=*)
    for key in TAG_KEYS_WILDCARD:
        if key in tags:
            cat, feat, conf = TAG_LOOKUP[(key, '*')]
            src = f'osm_{cat}_{key}'
            return cat, feat, src, conf

    has_org_hint = any(k in tags for k in ORG_HINT_TAGS)
    has_name = any(k in tags for k in ('name', 'name:ru', 'official_name'))
    has_addr = any(k in tags for k in ADDR_HINT_TAGS)

    # Tier-1 fallback: org-hint + name → 'osm_other_business'
    # (was the only fallback before — kept as the highest-confidence path).
    if has_org_hint and has_name:
        return 'other', 'business', 'osm_other_business', 0.72

    # Tier-2 fallback: org-hint without a name. Phones on tagged
    # amenity/shop/office elements are real org phones even when the
    # contributor didn't fill in name:ru — this used to be silently
    # dropped, costing ~50% of the bulk yield.
    if has_org_hint:
        return 'other', 'business', 'osm_org_hint', 0.66

    # Tier-3 fallback: phone + address + readable name. Some OSM elements
    # are tagged only with addr:* + name (no amenity/shop/office at all),
    # but the address+name combination is a strong RU-business signal.
    if has_addr and has_name:
        return 'other', 'business', 'osm_addressed_business', 0.62

    return None


# Phone-bearing OSM tag keys, sorted by frequency. The first four are the
# core phone tags; the rest are RU-specific or department-specific tags
# that the upstream Geofabrik PBF carries on a smaller but non-trivial
# fraction of organisations (especially gov, healthcare, and transit hubs).
PHONE_TAG_KEYS: Tuple[str, ...] = (
    'phone', 'contact:phone', 'phone:mobile', 'contact:mobile',
    'phone:landline', 'contact:landline',
    'phone:reception', 'phone:emergency', 'phone:helpline',
    'phone:department', 'phone:office', 'phone:fax',
    'contact:fax', 'fax',
    'phone:dispatcher',
)


def _phone_tags(tags: Dict[str, str]) -> List[str]:
    out: List[str] = []
    for k in PHONE_TAG_KEYS:
        v = tags.get(k)
        if v:
            out.extend(_split_phones(v))
    return out


# `_PhoneHandler` extends `osmium.SimpleHandler`. To keep this module
# importable in environments without pyosmium (e.g. unit tests for
# `_categorise`), we use a tiny stub base class when osmium is unavailable.
# The collector main() asserts pyosmium presence before instantiating.
if osmium is not None:
    _SimpleHandlerBase = osmium.SimpleHandler  # type: ignore[attr-defined]
else:
    class _SimpleHandlerBase:  # type: ignore[no-redef]
        """Stub base used only when pyosmium isn't installed."""
        def __init__(self) -> None:
            pass


class _PhoneHandler(_SimpleHandlerBase):  # type: ignore[misc]
    """SimpleHandler that streams nodes/ways/relations and harvests phones."""

    def __init__(self) -> None:
        super().__init__()
        self.entries: List[BulkEntry] = []
        self.seen: Set[str] = set()
        self._scanned = 0
        self._phoned = 0
        self._t0 = time.time()

    # pyosmium TagList → plain dict for downstream helpers
    @staticmethod
    def _tags_dict(tags) -> Dict[str, str]:  # type: ignore[no-untyped-def]
        return {tag.k: tag.v for tag in tags}

    def _consume(self, elem_type: str, elem_id: int, tagslist) -> None:  # type: ignore[no-untyped-def]
        if not tagslist:
            return
        self._scanned += 1
        if self._scanned % 5_000_000 == 0:
            elapsed = time.time() - self._t0
            logging.info(
                "scanned %s elements (%.1fM/s), kept %s phones so far",
                f'{self._scanned:,}',
                self._scanned / max(elapsed, 0.01) / 1_000_000,
                f'{len(self.entries):,}',
            )
        # Cheap pre-filter: phone tags very rarely appear; bail on the common
        # case where neither phone-related key is present.
        if not any(t.k in ('phone', 'contact:phone', 'phone:mobile', 'contact:mobile') for t in tagslist):
            return
        self._phoned += 1
        tags = self._tags_dict(tagslist)
        raw_phones = _phone_tags(tags)
        if not raw_phones:
            return
        cat_info = _categorise(tags)
        if cat_info is None:
            return
        category, feature, source_label, confidence = cat_info
        name = _name_from_tags(tags)
        city = _city_from_tags(tags)
        url = f'https://www.openstreetmap.org/{elem_type}/{elem_id}'
        for raw in raw_phones:
            norm = normalize_ru_phone(raw)
            if not norm or not is_russian_number(norm):
                continue
            if norm in self.seen:
                continue
            self.seen.add(norm)
            self.entries.append(BulkEntry(
                number=norm,
                name=name,
                category=feature,
                source=source_label,
                city=city,
                url=url,
                confidence=confidence,
            ))

    # SimpleHandler callbacks
    def node(self, n) -> None:  # type: ignore[no-untyped-def]
        self._consume('node', n.id, n.tags)

    def way(self, w) -> None:  # type: ignore[no-untyped-def]
        self._consume('way', w.id, w.tags)

    def relation(self, r) -> None:  # type: ignore[no-untyped-def]
        self._consume('relation', r.id, r.tags)


def download_pbf(url: str, dest: str, force: bool = False) -> str:
    """Download the Russia PBF if not already cached locally.

    Geofabrik refreshes the file once a day. We redownload only when the
    cache is missing or older than 24h to avoid hammering the mirror.
    """
    if not force and os.path.isfile(dest):
        age = time.time() - os.path.getmtime(dest)
        size = os.path.getsize(dest)
        if size > 100 * 1024 * 1024 and age < 36 * 3600:
            logging.info(
                "PBF cache hit: %s (%.1f MB, %.1fh old)",
                dest, size / 1024 / 1024, age / 3600,
            )
            return dest
        logging.info("PBF cache stale (size=%s age=%.1fh) — redownloading", size, age / 3600)

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + '.partial'
    logging.info("downloading %s → %s", url, dest)
    t0 = time.time()
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', '0'))
        wrote = 0
        last_log = t0
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                wrote += len(chunk)
                now = time.time()
                if now - last_log >= 10:
                    pct = (wrote / total * 100) if total else 0
                    rate = wrote / max(now - t0, 0.01) / 1024 / 1024
                    logging.info(
                        "  …%.1f%% (%.1f / %.1f MB, %.1f MB/s)",
                        pct, wrote / 1024 / 1024, total / 1024 / 1024, rate,
                    )
                    last_log = now
    os.replace(tmp, dest)
    logging.info(
        "downloaded %.1f MB in %.1fs",
        wrote / 1024 / 1024, time.time() - t0,
    )
    return dest


def load_existing_numbers(csv_path: str) -> Set[str]:
    """Read existing legitimate*.csv to skip duplicates on append."""
    if not os.path.isfile(csv_path):
        return set()
    out: Set[str] = set()
    try:
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                num = (row.get('normalized_number') or '').strip()
                if num:
                    out.add(num)
    except Exception as exc:
        logging.warning("Failed to load existing %s: %s", csv_path, exc)
    return out


def append_entries(csv_path: str, entries: List[BulkEntry]) -> Tuple[int, int]:
    existing = load_existing_numbers(csv_path)
    new_count = 0
    skipped = 0
    file_existed = os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if not file_existed:
            writer.writerow([
                'normalized_number', 'name', 'category', 'source',
                'city', 'url', 'source_confidence',
            ])
        for e in entries:
            if e.number in existing:
                skipped += 1
                continue
            existing.add(e.number)
            writer.writerow([
                e.number, e.name, e.category, e.source,
                e.city, e.url, f'{e.confidence:.2f}',
            ])
            new_count += 1
    return new_count, skipped


def main(argv: Optional[List[str]] = None) -> int:
    _require_runtime_deps()
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--output', default=OUTPUT_PATH,
        help=f'path to legitimate.csv (default: {OUTPUT_PATH})',
    )
    parser.add_argument(
        '--pbf-url', default=PBF_URL,
        help=f'Geofabrik PBF URL (default: {PBF_URL})',
    )
    parser.add_argument(
        '--pbf-cache',
        default=os.path.join('/tmp', 'russia-latest.osm.pbf'),
        help='local path to cache the PBF download (default: /tmp/russia-latest.osm.pbf)',
    )
    parser.add_argument(
        '--force-download', action='store_true',
        help='ignore on-disk cache and redownload',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='do not write CSV, just print counts',
    )
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    pbf_path = download_pbf(args.pbf_url, args.pbf_cache, force=args.force_download)
    if not os.path.isfile(pbf_path):
        logging.error("PBF download failed: %s missing", pbf_path)
        return 2

    handler = _PhoneHandler()
    logging.info("streaming PBF: %s", pbf_path)
    t0 = time.time()
    handler.apply_file(pbf_path, locations=False)  # we don't need geometry
    logging.info(
        "scanned %s elements in %.1fs (%s with phone tags, %s unique RU phones)",
        f'{handler._scanned:,}',
        time.time() - t0,
        f'{handler._phoned:,}',
        f'{len(handler.entries):,}',
    )

    if not handler.entries:
        logging.warning("no entries harvested — likely a PBF parsing problem")
        return 0

    if args.dry_run:
        # Print top categories for quick visibility
        from collections import Counter
        c = Counter(e.source for e in handler.entries)
        print(f'Total unique RU phones: {len(handler.entries)}')
        for src, n in c.most_common(20):
            print(f'  {src:40s} {n:>8d}')
        return 0

    new_count, skipped = append_entries(args.output, handler.entries)
    logging.info(
        "wrote %s new rows to %s (%s skipped as duplicates)",
        f'{new_count:,}', args.output, f'{skipped:,}',
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
