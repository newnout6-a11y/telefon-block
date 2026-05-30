"""
OSM Overpass API collector for legitimate Russian organization phones.

Queries OpenStreetMap for organizations in Russia with `phone` or
`contact:phone` tags and writes them to ``datasets/ru/raw/legitimate_numbers.csv``
in the same format used by ``ru_legitimate_collector.py``.

Why this matters: spam aggregators systematically mislabel many landline
numbers (especially Moscow +7495/+7499, regional polyclinics, government
offices) as "marketing/spam" because subscribers complain about appointment
reminders and unsolicited calls. This causes false-positive BLOCK verdicts
on prefixes like +74953 (88% block_share in current data) for legitimate
medical/government numbers (e.g. +74953387144 — a confirmed FP polyclinic
in the 2026-05-02 spot-check).

OSM has community-curated `phone` tags on hundreds of thousands of Russian
businesses, hospitals, schools, government offices, and so on. Pulling
these as a dedicated ALLOW source dramatically improves the ALLOW class
representation (currently ~50k vs ~410k BLOCK in the active corpus).

Output format: ``normalized_number,name,category,source,city,url,source_confidence``
appended (with deduplication against the existing file) to the canonical
``datasets/ru/raw/legitimate_numbers.csv``.

Usage:
    python scripts/ru_osm_collector.py                    # default: all categories
    python scripts/ru_osm_collector.py --categories medical,gov
    python scripts/ru_osm_collector.py --dry-run          # don't write CSV
    python scripts/ru_osm_collector.py --max-per-category 50000

Environment:
    OVERPASS_URL  optional override (default: https://overpass-api.de/api/interpreter)
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
    import requests
except ImportError:
    print("Требуется requests:  pip install requests", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, os.path.dirname(__file__))
from ru_number_normalizer import normalize_ru_phone, is_russian_number

OUTPUT_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'datasets', 'ru', 'raw', 'legitimate_numbers.csv'
))

OVERPASS_URL = os.environ.get(
    'OVERPASS_URL', 'https://overpass-api.de/api/interpreter'
)

# Per-query timeout. Overpass enforces 180s server-side; we mirror it locally.
QUERY_TIMEOUT = 180

# How long to wait between successive Overpass queries to stay under the
# public endpoint's fair-use quota.
INTER_QUERY_PAUSE = 5.0

# Each category is a list of (osm_tag, osm_value, source_confidence, feature_label)
# tuples. Each tuple becomes a separate Overpass sub-query. Splitting per-value
# keeps individual queries fast (regex over ~30M Russian elements times out
# on the public Overpass endpoint at 60s, but a single amenity=value query
# typically runs in 5-15s).
#
# When osm_value is None, the filter is just `["osm_tag"]` (any value).
# Confidence is per-row source_confidence used downstream by the dataset
# builder. Categories prioritised toward the top fix the most-impactful
# false-positive types first (medical / gov / banks).
CATEGORIES: Dict[str, List[Tuple[str, Optional[str], float, str]]] = {
    # MEDICAL — direct fix for the +74953387144 polyclinic false-positive.
    # Polyclinics in OSM are mostly tagged amenity=hospital or amenity=clinic.
    'medical': [
        ('amenity', 'hospital',     0.95, 'medical'),
        ('amenity', 'clinic',       0.95, 'medical'),
        ('amenity', 'doctors',      0.93, 'medical'),
        ('amenity', 'dentist',      0.92, 'medical'),
        ('amenity', 'pharmacy',     0.90, 'medical'),
        ('amenity', 'veterinary',   0.88, 'medical'),
        ('healthcare', None,        0.93, 'medical'),
    ],
    # EDUCATION — schools, universities, kindergartens, libraries
    'education': [
        ('amenity', 'school',           0.92, 'education'),
        ('amenity', 'university',       0.93, 'education'),
        ('amenity', 'college',          0.91, 'education'),
        ('amenity', 'kindergarten',     0.92, 'education'),
        ('amenity', 'library',          0.90, 'education'),
        ('amenity', 'childcare',        0.88, 'education'),
        ('amenity', 'driving_school',   0.85, 'education'),
        ('amenity', 'language_school',  0.85, 'education'),
    ],
    # GOV / PUBLIC SERVICES — police, post offices, town halls, courts
    'gov': [
        ('amenity', 'police',           0.95, 'gov'),
        ('amenity', 'post_office',      0.93, 'gov'),
        ('amenity', 'townhall',         0.95, 'gov'),
        ('amenity', 'courthouse',       0.95, 'gov'),
        ('amenity', 'fire_station',     0.95, 'gov'),
        ('amenity', 'embassy',          0.93, 'gov'),
        ('office',  'government',       0.92, 'gov'),
        ('office',  'lawyer',           0.85, 'gov'),
        ('office',  'notary',           0.88, 'gov'),
        ('office',  'tax_advisor',      0.85, 'gov'),
        ('office',  'advocate',         0.85, 'gov'),
        ('office',  'ngo',              0.83, 'gov'),
    ],
    # FINANCE — banks, ATMs, money transfer offices
    'finance': [
        ('amenity', 'bank',                 0.93, 'bank'),
        ('amenity', 'bureau_de_change',     0.85, 'bank'),
        ('amenity', 'money_transfer',       0.85, 'bank'),
        ('office',  'insurance',            0.83, 'insurance'),
        ('office',  'financial',            0.85, 'bank'),
    ],
    # TOURISM — hotels, hostels, museums
    'tourism': [
        ('tourism', 'hotel',                0.85, 'tourism'),
        ('tourism', 'hostel',               0.83, 'tourism'),
        ('tourism', 'guest_house',          0.82, 'tourism'),
        ('tourism', 'motel',                0.82, 'tourism'),
        ('tourism', 'museum',               0.90, 'tourism'),
        ('tourism', 'attraction',           0.80, 'tourism'),
        ('tourism', 'information',          0.85, 'tourism'),
    ],
    # RETAIL — large shop categories (medium confidence)
    'retail': [
        ('shop', 'supermarket',         0.82, 'retail'),
        ('shop', 'mall',                0.80, 'retail'),
        ('shop', 'department_store',    0.80, 'retail'),
        ('shop', 'car',                 0.78, 'retail'),
        ('shop', 'car_repair',          0.78, 'autoservice'),
        ('shop', 'electronics',         0.78, 'retail'),
        ('shop', 'furniture',           0.78, 'retail'),
        ('shop', 'hardware',            0.78, 'retail'),
        ('shop', 'optician',            0.80, 'retail'),
        ('shop', 'jewelry',             0.78, 'retail'),
        ('shop', 'bicycle',             0.78, 'retail'),
    ],
    # FOOD — restaurants, cafes (medium confidence)
    'food': [
        ('amenity', 'restaurant',           0.80, 'restaurant'),
        ('amenity', 'cafe',                 0.78, 'restaurant'),
        ('amenity', 'fast_food',            0.78, 'restaurant'),
        ('amenity', 'food_court',           0.78, 'restaurant'),
        ('amenity', 'bar',                  0.75, 'restaurant'),
        ('amenity', 'pub',                  0.75, 'restaurant'),
    ],
    # AUTO — fuel stations, car_wash, parking
    'auto': [
        ('amenity', 'fuel',                 0.88, 'autoservice'),
        ('amenity', 'car_wash',             0.83, 'autoservice'),
        ('amenity', 'charging_station',     0.85, 'autoservice'),
        ('amenity', 'car_rental',           0.83, 'autoservice'),
        ('amenity', 'car_sharing',          0.83, 'autoservice'),
    ],
    # SPORT / FITNESS
    'sport': [
        ('leisure', 'fitness_centre',       0.82, 'sport'),
        ('leisure', 'sports_centre',        0.82, 'sport'),
        ('leisure', 'swimming_pool',        0.82, 'sport'),
        ('leisure', 'stadium',              0.85, 'sport'),
        ('leisure', 'sports_hall',          0.82, 'sport'),
    ],
    # OFFICE — generic office types
    'office_other': [
        ('office', 'company',               0.83, 'office'),
        ('office', 'coworking',             0.80, 'office'),
        ('office', 'consulting',            0.83, 'office'),
        ('office', 'estate_agent',          0.78, 'realestate'),
        ('office', 'it',                    0.85, 'office'),
        ('office', 'telecommunication',     0.88, 'office'),
        ('office', 'employment_agency',     0.78, 'office'),
        ('office', 'accountant',            0.83, 'office'),
        ('office', 'architect',             0.83, 'office'),
        ('office', 'educational_institution', 0.88, 'education'),
        ('office', 'logistics',             0.83, 'office'),
        ('office', 'research',              0.85, 'office'),
        ('office', 'travel_agent',          0.78, 'office'),
        ('office', 'private_postal_service', 0.80, 'office'),
        ('office', 'newspaper',             0.82, 'office'),
        ('office', 'tutoring',              0.78, 'education'),
        ('office', 'water_utility',         0.85, 'gov'),
        ('office', 'visa',                  0.80, 'gov'),
        ('office', 'guide',                 0.78, 'tourism'),
        ('office', 'union',                 0.78, 'office'),
        ('office', 'yes',                   0.72, 'office'),
    ],
    # CRAFT — small workshops / artisans (often phone-tagged in RU)
    'craft': [
        ('craft', None,                     0.78, 'craft'),
    ],
    # PUBLIC TRANSPORT — major transit hubs
    'public_transport': [
        ('public_transport', 'station',     0.85, 'official'),
        ('railway',          'station',     0.85, 'official'),
        ('aeroway',          'aerodrome',   0.90, 'official'),
        ('aeroway',          'terminal',    0.90, 'official'),
        ('amenity',          'taxi',        0.78, 'official'),
        ('amenity',          'bus_station', 0.85, 'official'),
        ('amenity',          'ferry_terminal', 0.85, 'official'),
    ],
    # SHOP — extended subcategories (clothes, beauty, food, services, etc.)
    'shop_extended': [
        ('shop', 'clothes',                 0.78, 'retail'),
        ('shop', 'shoes',                   0.78, 'retail'),
        ('shop', 'beauty',                  0.78, 'beauty'),
        ('shop', 'hairdresser',             0.78, 'beauty'),
        ('shop', 'cosmetics',               0.78, 'beauty'),
        ('shop', 'florist',                 0.78, 'retail'),
        ('shop', 'gift',                    0.78, 'retail'),
        ('shop', 'sports',                  0.78, 'retail'),
        ('shop', 'toys',                    0.78, 'retail'),
        ('shop', 'books',                   0.78, 'retail'),
        ('shop', 'kiosk',                   0.74, 'retail'),
        ('shop', 'convenience',             0.74, 'retail'),
        ('shop', 'butcher',                 0.78, 'retail'),
        ('shop', 'bakery',                  0.78, 'retail'),
        ('shop', 'beverages',               0.78, 'retail'),
        ('shop', 'wine',                    0.78, 'retail'),
        ('shop', 'alcohol',                 0.74, 'retail'),
        ('shop', 'tobacco',                 0.74, 'retail'),
        ('shop', 'travel_agency',           0.80, 'tourism'),
        ('shop', 'mobile_phone',            0.78, 'retail'),
        ('shop', 'computer',                0.78, 'retail'),
        ('shop', 'tea',                     0.78, 'retail'),
        ('shop', 'coffee',                  0.78, 'retail'),
        ('shop', 'pastry',                  0.78, 'retail'),
        ('shop', 'baby_goods',              0.78, 'retail'),
        ('shop', 'pet',                     0.78, 'retail'),
        ('shop', 'paint',                   0.78, 'retail'),
        ('shop', 'tyres',                   0.78, 'autoservice'),
        ('shop', 'motorcycle',              0.78, 'retail'),
        ('shop', 'massage',                 0.78, 'beauty'),
        ('shop', 'tattoo',                  0.74, 'beauty'),
        ('shop', 'photo',                   0.78, 'retail'),
        ('shop', 'art',                     0.78, 'retail'),
        ('shop', 'antiques',                0.78, 'retail'),
        ('shop', 'musical_instrument',      0.78, 'retail'),
        ('shop', 'video_games',             0.78, 'retail'),
        ('shop', 'bag',                     0.78, 'retail'),
    ],
    # AMENITY — extended civic/social
    'amenity_extended': [
        ('amenity', 'community_centre',     0.85, 'gov'),
        ('amenity', 'social_facility',      0.85, 'gov'),
        ('amenity', 'social_centre',        0.85, 'gov'),
        ('amenity', 'place_of_worship',     0.78, 'religious'),
        ('amenity', 'cinema',               0.82, 'tourism'),
        ('amenity', 'theatre',              0.85, 'tourism'),
        ('amenity', 'arts_centre',          0.82, 'tourism'),
        ('amenity', 'nightclub',            0.74, 'restaurant'),
        ('amenity', 'events_venue',         0.80, 'tourism'),
        ('amenity', 'studio',               0.78, 'office'),
        ('amenity', 'spa',                  0.78, 'beauty'),
    ],
}


# ── Helpers ─────────────────────────────────────────────────────────────────


@dataclass
class OsmEntry:
    """A single OSM element with one normalised phone."""
    number: str
    name: str
    category: str
    source: str
    city: str
    url: str
    confidence: float


def _split_phones(raw: str) -> List[str]:
    """OSM phone tag often contains multiple numbers separated by ``;`` or ``,``."""
    if not raw:
        return []
    parts: List[str] = []
    for chunk in raw.replace(';', ',').replace('|', ',').split(','):
        s = chunk.strip()
        if s:
            parts.append(s)
    return parts


def _city_from_tags(tags: Dict[str, str]) -> str:
    """Best-effort city extraction from OSM tags."""
    for key in ('addr:city', 'is_in:city', 'addr:state', 'is_in:region', 'addr:region'):
        v = tags.get(key)
        if v:
            return v.strip()
    return ''


def _name_from_tags(tags: Dict[str, str]) -> str:
    """Pick the most readable name."""
    for key in ('name:ru', 'name', 'official_name', 'alt_name', 'description'):
        v = tags.get(key)
        if v:
            return v.strip()
    return ''


def _osm_url(elem_type: str, elem_id: int) -> str:
    return f'https://www.openstreetmap.org/{elem_type}/{elem_id}'


def _build_subquery(osm_tag: str, osm_value: Optional[str]) -> str:
    """Build a single-tag Overpass query (one tag/value pair) for RU.

    Splits node-only and way-only into separate union members. Nodes alone
    cover ~95% of useful business POIs and are 5-10× faster to scan than ways.
    """
    if osm_value is None:
        tag_filter = f'["{osm_tag}"]'
    else:
        tag_filter = f'["{osm_tag}"="{osm_value}"]'
    return f"""
[out:json][timeout:{QUERY_TIMEOUT}];
area["ISO3166-1"="RU"][admin_level=2]->.ru;
(
  node{tag_filter}["phone"](area.ru);
  node{tag_filter}["contact:phone"](area.ru);
);
out tags;
""".strip()


def _query_overpass(query: str, max_retries: int = 3) -> Optional[List[Dict]]:
    """POST to Overpass and return ``elements`` list, or None on error."""
    headers = {
        'User-Agent': 'SpamBlocker-OSM-Collector/1.0 (defensive-research; respects fair-use)',
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={'data': query},
                headers=headers,
                timeout=QUERY_TIMEOUT + 30,
            )
            if resp.status_code == 200:
                payload = resp.json()
                # Detect server-side timeout in the response body
                if 'remark' in payload and 'timed out' in str(payload.get('remark', '')):
                    logging.warning("Overpass server-side timeout: %s", payload['remark'])
                    return None
                return payload.get('elements', [])
            if resp.status_code in (429, 503, 504):
                wait = (attempt + 1) * 30
                logging.warning(
                    "Overpass returned %s, backing off %ss (attempt %s/%s)",
                    resp.status_code, wait, attempt + 1, max_retries,
                )
                time.sleep(wait)
                continue
            logging.error("Overpass returned %s: %s", resp.status_code, resp.text[:200])
            return None
        except requests.RequestException as exc:
            logging.warning("Overpass request error (attempt %s/%s): %s",
                            attempt + 1, max_retries, exc)
            time.sleep((attempt + 1) * 10)
    return None


def fetch_subcategory(osm_tag: str, osm_value: Optional[str],
                      confidence: float, feature: str,
                      cat_name: str) -> List[OsmEntry]:
    """Fetch one (osm_tag, osm_value) pair from Overpass."""
    query = _build_subquery(osm_tag, osm_value)
    label = f'{osm_tag}={osm_value or "*"}'
    logging.info("[%s/%s] Querying Overpass…", cat_name, label)
    elements = _query_overpass(query)
    if elements is None:
        logging.error("[%s/%s] Overpass query failed, skipping", cat_name, label)
        return []
    logging.info("[%s/%s] Got %s elements", cat_name, label, len(elements))

    entries: List[OsmEntry] = []
    seen_numbers: Set[str] = set()
    source_label = f'osm_{cat_name}_{osm_value or osm_tag}'

    for elem in elements:
        tags = elem.get('tags') or {}
        if not tags:
            continue
        elem_type = elem.get('type', 'node')
        elem_id = elem.get('id', 0)
        raw_phones = []
        for key in ('phone', 'contact:phone', 'phone:mobile', 'contact:mobile'):
            v = tags.get(key)
            if v:
                raw_phones.extend(_split_phones(v))
        if not raw_phones:
            continue

        name_str = _name_from_tags(tags)
        city = _city_from_tags(tags)
        url = _osm_url(elem_type, elem_id)

        for raw in raw_phones:
            norm = normalize_ru_phone(raw)
            if not norm or not is_russian_number(norm):
                continue
            if norm in seen_numbers:
                continue
            seen_numbers.add(norm)
            entries.append(OsmEntry(
                number=norm,
                name=name_str,
                category=feature,
                source=source_label,
                city=city,
                url=url,
                confidence=confidence,
            ))

    logging.info("[%s/%s] %s unique RU phones", cat_name, label, len(entries))
    return entries


def fetch_category(name: str, sub_specs: List[Tuple[str, Optional[str], float, str]],
                   max_per_category: int = 100_000) -> List[OsmEntry]:
    """Fetch one category by running each sub-query and merging results."""
    out: List[OsmEntry] = []
    seen: Set[str] = set()
    for i, (osm_tag, osm_value, confidence, feature) in enumerate(sub_specs):
        if i > 0:
            time.sleep(INTER_QUERY_PAUSE)
        if len(out) >= max_per_category:
            logging.info("[%s] reached cap (%s), stopping", name, max_per_category)
            break
        sub_entries = fetch_subcategory(osm_tag, osm_value, confidence, feature, name)
        for e in sub_entries:
            if e.number in seen:
                continue
            seen.add(e.number)
            out.append(e)
            if len(out) >= max_per_category:
                break
    logging.info("[%s] total unique: %s", name, len(out))
    return out


def load_existing_numbers(csv_path: str) -> Set[str]:
    """Read existing legitimate_numbers.csv to skip duplicates on append."""
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


def append_entries(csv_path: str, entries: List[OsmEntry]) -> Tuple[int, int]:
    """Append new entries (skipping numbers already present). Returns (new, skipped)."""
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


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--categories', default='all',
        help='comma-separated category names (default: all). '
             f'Available: {",".join(CATEGORIES.keys())}',
    )
    parser.add_argument(
        '--max-per-category', type=int, default=100_000,
        help='cap per category (default: 100k)',
    )
    parser.add_argument(
        '--output', default=OUTPUT_PATH,
        help=f'path to legitimate_numbers.csv (default: {OUTPUT_PATH})',
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='do not write CSV, just print counts')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    if args.categories == 'all':
        wanted = list(CATEGORIES.keys())
    else:
        wanted = [c.strip() for c in args.categories.split(',') if c.strip()]
        unknown = [c for c in wanted if c not in CATEGORIES]
        if unknown:
            print(f'Unknown categories: {unknown}. '
                  f'Available: {list(CATEGORIES.keys())}', file=sys.stderr)
            return 2

    all_entries: List[OsmEntry] = []
    for i, cat_name in enumerate(wanted):
        if i > 0:
            time.sleep(INTER_QUERY_PAUSE)
        spec = CATEGORIES[cat_name]
        cat_entries = fetch_category(cat_name, spec, args.max_per_category)
        all_entries.extend(cat_entries)

    # Dedup across categories (same number can be tagged in multiple)
    seen: Set[str] = set()
    unique: List[OsmEntry] = []
    for e in all_entries:
        if e.number in seen:
            continue
        seen.add(e.number)
        unique.append(e)

    print(f'\n=== OSM collection summary ===')
    print(f'Total raw entries fetched: {len(all_entries)}')
    print(f'Unique RU numbers:         {len(unique)}')

    if args.dry_run:
        print('(dry-run: not writing CSV)')
        return 0

    new_count, skipped = append_entries(args.output, unique)
    print(f'Appended new:              {new_count}')
    print(f'Skipped duplicates:        {skipped}')
    print(f'Output:                    {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
