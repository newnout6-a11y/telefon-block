"""
Wikidata SPARQL collector for legitimate Russian organization phones.

Queries the Wikidata Query Service for entities with:
  * P17 (country) = Q159 (Russia), OR
  * P131 (located in admin entity) within Russia
And the property:
  * P1329 (phone number)

Extracts ``(label, phone, instance_of, headquarters_label)`` tuples and
appends them to ``datasets/ru/raw/legitimate_numbers.csv`` in the same
format used by ``ru_legitimate_collector.py``. Skips emergency short
codes (101/102/103/112/etc.) which fail RU number validation.

Why this is useful: Wikidata is community-curated and disproportionately
covers high-importance legitimate entities — government agencies,
universities, hospitals, large companies, museums, banks. These are
exactly the categories where aggregator-driven block_share
overestimates spam likelihood (e.g. Moscow polyclinic landlines on
+74953/+74957 prefixes).

Usage:
    python scripts/ru_wikidata_collector.py                    # default fetch
    python scripts/ru_wikidata_collector.py --dry-run          # don't write
    python scripts/ru_wikidata_collector.py --max-results 50000

Environment:
    WIKIDATA_SPARQL_URL  optional override (default: https://query.wikidata.org/sparql)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

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

WIKIDATA_SPARQL_URL = os.environ.get(
    'WIKIDATA_SPARQL_URL', 'https://query.wikidata.org/sparql'
)

QUERY_TIMEOUT = 60
MAX_RETRIES = 3


@dataclass
class WikidataEntry:
    number: str
    name: str
    category: str
    source: str
    city: str
    url: str
    confidence: float


# Phase-1 ALLOW ×10: split the Wikidata fetch into category-specific queries
# instead of one broad SPARQL. The single-query approach times out on the
# public 60-sec slot (Wikidata's query service kills it after seeing the
# OPTIONAL P31 join blow up), so the previous run pulled only ~450 entries
# against a 50k budget. Each per-category query below pins ?item to a
# narrow P31/P279* subclass set and reliably returns 1-15k bindings under
# 60s.
#
# Each tuple is (category_name, [QIDs of instance-of values to walk]).
# QIDs are walked transitively via wdt:P31/wdt:P279* so subclasses are
# included automatically (e.g. Q16917 hospital pulls Q21024710 children's
# hospital, Q31855 polyclinic pulls Q1185356 medical clinic).
CATEGORICAL_QIDS: List[Tuple[str, List[str]]] = [
    ('medical',     ['Q16917', 'Q3270632', 'Q31855', 'Q4287745', 'Q1185356']),
    ('education',   ['Q3914', 'Q3918', 'Q120560', 'Q1664720', 'Q23002054', 'Q9842']),
    ('gov',         ['Q15640612', 'Q35657', 'Q294440', 'Q4671277', 'Q3917681',
                     'Q327333', 'Q1255921', 'Q35535']),
    ('bank',        ['Q22687', 'Q806463']),
    ('insurance',   ['Q35666']),
    ('hotel',       ['Q27686']),
    ('museum',      ['Q33506']),
    ('library',     ['Q7075']),
    ('theatre',     ['Q24354', 'Q41253']),
    ('airport',     ['Q1248784', 'Q62447', 'Q644371']),
    ('railway',     ['Q55488', 'Q928830']),
    ('restaurant',  ['Q11707', 'Q56042', 'Q325053']),
    ('pharmacy',    ['Q124365']),
    ('telecom',     ['Q1668024']),
    ('media',       ['Q15265344', 'Q11032', 'Q1110794']),
    ('religious',   ['Q4671277', 'Q44613', 'Q16970']),
    # Broad business catch-all — last query, picks up anything not in a
    # named bucket above. Keep LIMIT modest because the result set is huge.
    ('business',    ['Q4830453', 'Q43229']),
]


def _build_categorical_query(qids: List[str], limit: int = 15_000) -> str:
    """Build a SPARQL query for items with given P31 subclass set in Russia."""
    values = ' '.join(f'wd:{q}' for q in qids)
    return f"""
SELECT DISTINCT ?item ?itemLabel ?phone ?typeLabel ?cityLabel WHERE {{
  ?item wdt:P31/wdt:P279* ?type .
  VALUES ?type {{ {values} }}
  ?item wdt:P17 wd:Q159 .
  ?item wdt:P1329 ?phone .
  OPTIONAL {{ ?item wdt:P31 ?typeRaw . }}
  OPTIONAL {{ ?item wdt:P131 ?city . }}
  BIND(?typeRaw AS ?typeForLabel)
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "ru,en". 
    ?item rdfs:label ?itemLabel .
    ?typeForLabel rdfs:label ?typeLabel .
    ?city rdfs:label ?cityLabel .
  }}
}}
LIMIT {limit}
""".strip()


# Legacy single-query fallback. Kept only as a safety net for entries that
# don't carry a P31 in any of the curated buckets above; runs last with a
# small LIMIT to stay under the 60-sec slot.
SPARQL_QUERY = """
SELECT DISTINCT ?item ?itemLabel ?phone ?typeLabel ?cityLabel WHERE {
  ?item wdt:P17 wd:Q159 .
  ?item wdt:P1329 ?phone .
  OPTIONAL { ?item wdt:P31 ?type . }
  OPTIONAL { ?item wdt:P131 ?city . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "ru,en". }
}
LIMIT 8000
""".strip()


# Map Wikidata "instance of" labels to feature labels and confidences.
# Specific labels first, then fallback patterns.
TYPE_TO_FEATURE: List[Tuple[str, str, float]] = [
    # (substring-match in label, feature, confidence)
    ('hospital',         'medical',    0.95),
    ('больниц',          'medical',    0.95),
    ('clinic',           'medical',    0.94),
    ('поликлиник',       'medical',    0.95),
    ('university',       'education',  0.93),
    ('институт',         'education',  0.92),
    ('университет',      'education',  0.93),
    ('школ',             'education',  0.92),
    ('school',           'education',  0.92),
    ('library',          'education',  0.90),
    ('библиотек',        'education',  0.90),
    ('museum',           'tourism',    0.92),
    ('музей',            'tourism',    0.92),
    ('government',       'gov',        0.95),
    ('министерств',      'gov',        0.95),
    ('федеральн',        'gov',        0.95),
    ('агентство',        'gov',        0.92),
    ('embassy',          'gov',        0.93),
    ('посольств',        'gov',        0.93),
    ('police',           'gov',        0.95),
    ('court',            'gov',        0.93),
    ('суд',              'gov',        0.93),
    ('bank',             'bank',       0.93),
    ('банк',             'bank',       0.93),
    ('insurance',        'insurance',  0.85),
    ('страхов',          'insurance',  0.85),
    ('hotel',            'tourism',    0.85),
    ('отел',             'tourism',    0.85),
    ('гостиниц',         'tourism',    0.85),
    ('theatre',          'tourism',    0.90),
    ('театр',            'tourism',    0.90),
    ('airline',          'official',   0.92),
    ('авиакомпан',       'official',   0.92),
    ('railway',          'official',   0.92),
    ('телеком',          'official',   0.92),
    ('telecom',          'official',   0.92),
    ('стадион',          'sport',      0.85),
    ('stadium',          'sport',      0.85),
    ('agency',           'gov',        0.85),
]


def _classify(type_label: str) -> Tuple[str, float]:
    """Map a Wikidata 'instance of' label to (feature, confidence)."""
    if not type_label:
        return 'other', 0.78
    low = type_label.lower()
    for keyword, feature, confidence in TYPE_TO_FEATURE:
        if keyword.lower() in low:
            return feature, confidence
    # Fallback for unclassified orgs in Russia: still high-confidence ALLOW
    # (these are notable enough to be in Wikidata) but lower than typed.
    return 'other', 0.80


def _query_wikidata(query: str) -> Optional[List[dict]]:
    """Run a SPARQL query against Wikidata and return result bindings."""
    headers = {
        'Accept': 'application/sparql-results+json',
        'User-Agent': 'SpamBlocker-Wikidata-Collector/1.0',
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                WIKIDATA_SPARQL_URL,
                params={'query': query},
                headers=headers,
                timeout=QUERY_TIMEOUT,
            )
            if resp.status_code == 200:
                payload = resp.json()
                return payload.get('results', {}).get('bindings', [])
            if resp.status_code in (429, 503, 504):
                wait = (attempt + 1) * 30
                logging.warning(
                    "Wikidata returned %s, backing off %ss (attempt %s/%s)",
                    resp.status_code, wait, attempt + 1, MAX_RETRIES,
                )
                time.sleep(wait)
                continue
            logging.error("Wikidata returned %s: %s", resp.status_code, resp.text[:200])
            return None
        except requests.RequestException as exc:
            logging.warning("Wikidata request error (attempt %s/%s): %s",
                            attempt + 1, MAX_RETRIES, exc)
            time.sleep((attempt + 1) * 10)
    return None


def _bindings_to_entries(
    bindings: List[dict],
    seen_numbers: Set[str],
    entries: List[WikidataEntry],
    *,
    bucket_hint: Optional[str],
    max_results: int,
) -> int:
    """Convert SPARQL bindings to WikidataEntry, deduping against seen_numbers.

    Returns the number of new entries added. ``bucket_hint`` overrides the
    feature classification when the per-row P31 typeLabel is missing or
    unrecognised — the bucket name itself is a strong category signal.
    """
    added = 0
    for b in bindings:
        if len(entries) >= max_results:
            break
        item_uri = b.get('item', {}).get('value', '')
        item_label = b.get('itemLabel', {}).get('value', '').strip()
        phone_raw = b.get('phone', {}).get('value', '').strip()
        type_label = b.get('typeLabel', {}).get('value', '').strip()
        city_label = b.get('cityLabel', {}).get('value', '').strip()

        if not phone_raw:
            continue

        norm = normalize_ru_phone(phone_raw)
        if not norm or not is_russian_number(norm):
            continue
        if norm in seen_numbers:
            continue
        seen_numbers.add(norm)

        feature, confidence = _classify(type_label)
        # bucket_hint promotes generic 'other' classifications to the
        # bucket's own feature when a P31 hit landed us here.
        if bucket_hint and feature == 'other':
            feature = bucket_hint
            confidence = max(confidence, 0.85)

        entries.append(WikidataEntry(
            number=norm,
            name=item_label,
            category=feature,
            source=f'wikidata_{feature}',
            city=city_label,
            url=item_uri,
            confidence=confidence,
        ))
        added += 1
    return added


def fetch_entries(max_results: int = 200_000) -> List[WikidataEntry]:
    """Fetch organisations across categorical SPARQL queries + broad fallback."""
    entries: List[WikidataEntry] = []
    seen_numbers: Set[str] = set()

    for bucket_name, qids in CATEGORICAL_QIDS:
        if len(entries) >= max_results:
            break
        logging.info("Wikidata bucket %s: %s QIDs…", bucket_name, len(qids))
        query = _build_categorical_query(qids, limit=15_000)
        bindings = _query_wikidata(query)
        if bindings is None:
            logging.warning("Wikidata bucket %s query failed, skipping", bucket_name)
            continue
        logging.info("  bucket %s: %s raw bindings", bucket_name, len(bindings))
        added = _bindings_to_entries(
            bindings, seen_numbers, entries,
            bucket_hint=bucket_name, max_results=max_results,
        )
        logging.info("  bucket %s: +%s new entries (total=%s)",
                     bucket_name, added, len(entries))
        # Be polite: the public WDQS slot is 60 sec total per IP
        time.sleep(2.0)

    # Broad fallback — picks up entries that don't carry any of our
    # bucketed P31 subclass values but are still RU orgs with a phone.
    if len(entries) < max_results:
        logging.info("Wikidata broad fallback query (everything else)…")
        bindings = _query_wikidata(SPARQL_QUERY)
        if bindings is not None:
            logging.info("  fallback: %s raw bindings", len(bindings))
            added = _bindings_to_entries(
                bindings, seen_numbers, entries,
                bucket_hint=None, max_results=max_results,
            )
            logging.info("  fallback: +%s new entries (total=%s)",
                         added, len(entries))

    logging.info("%s unique RU phones extracted from Wikidata across %s queries",
                 len(entries), len(CATEGORICAL_QIDS) + 1)
    return entries


def load_existing_numbers(csv_path: str) -> Set[str]:
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


def append_entries(csv_path: str, entries: List[WikidataEntry]) -> Tuple[int, int]:
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
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--max-results', type=int, default=200_000)
    parser.add_argument('--output', default=OUTPUT_PATH)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    entries = fetch_entries(args.max_results)
    print(f'\n=== Wikidata collection summary ===')
    print(f'Unique RU numbers: {len(entries)}')

    if args.dry_run:
        print('(dry-run: not writing CSV)')
        return 0

    new_count, skipped = append_entries(args.output, entries)
    print(f'Appended new:      {new_count}')
    print(f'Skipped duplicates: {skipped}')
    print(f'Output:            {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
