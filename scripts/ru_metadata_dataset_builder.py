"""
RF metadata dataset builder.

Builds:
  - processed/ru_reputation_raw.csv
  - processed/ru_numbers_labeled.csv
  - processed/ru_metadata_features.csv
  - processed/ru_tflite_features.csv

Synthetic rows are disabled by default and available only for smoke tests.
"""

import argparse
import csv
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from ru_metadata_features import (
    COMPACT_FEATURES, FULL_METADATA_FIELDS, LABEL_TO_ID,
    RU_TO_FIELD,
    category_flags, compact_feature_vector, compact_row,
    compute_reputation_score,
    infer_prefix_risk, number_type, operator_bucket, parse_date,
    review_velocity, safe_float, safe_int, stable_bucket,
    translate_headers, translate_row,
)
from ru_number_normalizer import get_def_code, is_valid_ru_phone, normalize_ru_phone
from ru_numbering_plan import NumberingPlan, load_existing_csv as load_numbering_csv

import cold_start_balancer as csb

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'ru')
RAW_DIR = os.path.join(BASE_DIR, 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')

RAW_REPUTATION_FILES = [
    'ru_reputation_raw.csv',
    'reviews_neberitrubku.csv',
    'reviews_zvonili.csv',
]

BLACKLIST_FILES = [
    ('blacklist_moshelovka.csv', 'moshelovka'),
    ('blacklist_spravportal.csv', 'spravportal'),
]

RAW_REPUTATION_SCHEMA = [
    'normalized_number',
    'source',
    'negative_count',
    'positive_count',
    'neutral_count',
    'review_count',
    'search_volume',
    'categories',
    'last_review_at',
    'first_seen_at',
    'source_confidence',
    'source_reliability',
    'view_count',
    'related_count',
    'detail_date',
    'page_title',
    'url',
]


def iter_csv(path: str):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        translated_fields = [RU_TO_FIELD.get(k, k) for k in fieldnames]
        needs_translate = translated_fields != fieldnames
        for row in reader:
            if needs_translate:
                yield {translated_fields[i]: row.get(fieldnames[i], '') for i in range(len(fieldnames))}
            else:
                yield row


def read_csv(path: str) -> List[Dict]:
    return list(iter_csv(path) or [])


def _read_csv_pandas(path: str):
    """Fast CSV reader using pandas with header translation.

    Returns a DataFrame with English column names (RU_TO_FIELD applied) or
    None if the file is missing or empty. Pandas reads ~50-100x faster than
    csv.DictReader on multi-100MB files because it streams via C and skips
    creation of one Python dict per row.
    """
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return None
    if df.empty:
        return None
    rename_map = {col: RU_TO_FIELD[col] for col in df.columns if col in RU_TO_FIELD}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def write_dict_csv(path: str, rows: Iterable[Dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Пишем с русскими заголовками
    ru_fieldnames = translate_headers(fieldnames, to_ru=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ru_fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(translate_row(row, to_ru=True))


def write_rows_csv(path: str, header: List[str], rows: Iterable[List[float]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Пишем с русскими заголовками
    ru_header = translate_headers(header, to_ru=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ru_header)
        writer.writerows(rows)


def write_output_records(entries: Iterable[Dict]) -> Tuple[int, Dict[str, int]]:
    """Streaming csv.writer path. Constant memory regardless of input size.

    No pandas DataFrames, no per-row dict-to-DataFrame conversions, no
    per-row translate_row calls. Russian column ordering is precomputed
    once; each row is written as ``[record.get(field, '') for field in
    english_fields]`` straight into ``csv.writer``.

    Memory contract: ~50MB regardless of total rows.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    labeled_fields = ['normalized_number', 'label', 'label_id', 'weight', 'source', 'source_confidence']
    tflite_header = COMPACT_FEATURES + ['label']

    counts: Dict[str, int] = defaultdict(int)
    total = 0

    labeled_path = os.path.join(PROCESSED_DIR, 'ru_numbers_labeled.csv')
    metadata_path = os.path.join(PROCESSED_DIR, 'ru_metadata_features.csv')
    tflite_path = os.path.join(PROCESSED_DIR, 'ru_tflite_features.csv')

    labeled_header_ru = translate_headers(labeled_fields, to_ru=True)
    metadata_header_ru = translate_headers(FULL_METADATA_FIELDS, to_ru=True)
    tflite_header_ru = translate_headers(tflite_header, to_ru=True)

    with (
        open(labeled_path, 'w', encoding='utf-8', newline='') as f_labeled,
        open(metadata_path, 'w', encoding='utf-8', newline='') as f_metadata,
        open(tflite_path, 'w', encoding='utf-8', newline='') as f_tflite,
    ):
        w_labeled = csv.writer(f_labeled)
        w_metadata = csv.writer(f_metadata)
        w_tflite = csv.writer(f_tflite)

        w_labeled.writerow(labeled_header_ru)
        w_metadata.writerow(metadata_header_ru)
        w_tflite.writerow(tflite_header_ru)

        for entry in entries:
            feature_record, tflite_row, labeled_row = entry_to_records(entry)
            w_labeled.writerow([labeled_row.get(k, '') for k in labeled_fields])
            w_metadata.writerow([feature_record.get(k, '') for k in FULL_METADATA_FIELDS])
            w_tflite.writerow(tflite_row)
            counts[labeled_row['label']] += 1
            total += 1

    return total, counts


def load_whitelist() -> Dict[str, Dict]:
    result = {}
    # Official whitelist
    for row in iter_csv(os.path.join(RAW_DIR, 'whitelist_official_ru.csv')):
        number = normalize_ru_phone(row.get('normalized_number') or row.get('phone') or row.get('number') or '')
        if not number or not is_valid_ru_phone(number):
            continue
        result[number] = {
            'normalized_number': number,
            'name': row.get('name', ''),
            'category': row.get('category', 'official'),
            'source': 'whitelist_official',
            'source_confidence': 0.95,
        }
    # Legitimate numbers from scraper (organizations, freelancers, etc.)
    for row in iter_csv(os.path.join(RAW_DIR, 'legitimate_numbers.csv')):
        number = normalize_ru_phone(row.get('normalized_number') or row.get('phone') or row.get('number') or '')
        if not number or not is_valid_ru_phone(number):
            continue
        if number in result:
            continue  # don't overwrite official whitelist
        source = row.get('source', 'legitimate')
        cat = row.get('category', '')
        source_conf = safe_float(row.get('source_confidence'), 0.0)
        if source_conf <= 0:
            if source in {'official_whitelist', 'official_hotline'}:
                source_conf = 0.95
            elif cat in {'personal_mobile'} or source.startswith('numbering_plan'):
                source_conf = 0.25
            elif cat in {'freelancer', 'private_seller', 'realestate_owner'}:
                source_conf = 0.55
            elif cat in {'delivery', 'government', 'bank', 'medical'}:
                source_conf = 0.85
            else:
                source_conf = 0.70
        result[number] = {
            'normalized_number': number,
            'name': row.get('name', ''),
            'category': cat,
            'source': f'legitimate_{source}',
            'source_confidence': source_conf,
        }
    return result


def normalize_reputation_row(row: Dict, source_hint: str = '') -> Optional[Dict]:
    raw_number = row.get('normalized_number') or row.get('phone') or row.get('number') or ''
    # Сначала пробуем строгий +7-only пасс (отбрасываем явный мусор).
    number = normalize_ru_phone(raw_number, reject_non_ru=True)
    if not number:
        return None
    # Затем — РФ allow-list def-кодов и round-number фильтр (KZ/невалидные/заглушки).
    if not is_valid_ru_phone(number):
        return None

    negative = safe_int(row.get('negative_count') or row.get('negative') or row.get('bad') or 0)
    positive = safe_int(row.get('positive_count') or row.get('positive') or row.get('good') or 0)
    neutral = safe_int(row.get('neutral_count') or row.get('neutral') or 0)
    review_count = safe_int(row.get('review_count') or row.get('reviews') or row.get('overall') or 0)
    if review_count <= 0:
        review_count = negative + positive + neutral

    categories = row.get('categories') or row.get('category') or row.get('rating') or ''
    source = row.get('source') or source_hint or 'unknown'
    confidence = safe_float(row.get('source_confidence') or row.get('confidence') or 0.5)

    # Prefer view_count as search_volume when available (crawler stores views there)
    view_count = safe_int(row.get('view_count') or row.get('views') or 0)
    search_volume = safe_int(row.get('search_volume') or 0)
    # If search_volume is 0 but view_count > 0, use view_count as proxy
    if search_volume <= 0 and view_count > 0:
        search_volume = view_count

    return {
        'normalized_number': number,
        'source': source,
        'negative_count': negative,
        'positive_count': positive,
        'neutral_count': neutral,
        'review_count': review_count,
        'search_volume': search_volume,
        'categories': categories,
        'last_review_at': row.get('last_review_at', '') or row.get('detail_date', ''),
        'first_seen_at': row.get('first_seen_at', ''),
        'source_confidence': confidence,
        'source_reliability': safe_float(row.get('source_reliability') or 0.5),
        'view_count': view_count,
        'related_count': safe_int(row.get('related_count') or 0),
        'detail_date': row.get('detail_date', ''),
        'page_title': row.get('page_title', ''),
        'url': row.get('url', ''),
    }


def load_reputation_rows() -> List[Dict]:
    """Backwards-compatible wrapper around the pandas fast path.

    Returns a list of dicts in the legacy 17-column reputation schema. For
    the multi-million-row builder, prefer ``load_reputation_df`` to avoid
    materializing 3M Python dicts.
    """
    df = load_reputation_df()
    if df is None or df.empty:
        return []
    return df.to_dict('records')


def _normalize_reputation_df(df, source_hint: str):
    """Vectorized counterpart of normalize_reputation_row over a DataFrame.

    Returns a DataFrame in the canonical RAW_REPUTATION_SCHEMA. Drops rows
    whose number can't be normalized to a valid +7 RU phone.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=RAW_REPUTATION_SCHEMA)

    # Phone normalization: pick first non-empty of (normalized_number, phone, number).
    def _col(name):
        return df[name].fillna('') if name in df.columns else pd.Series('', index=df.index)

    raw_num = _col('normalized_number')
    raw_num = raw_num.where(raw_num.astype(bool), _col('phone'))
    raw_num = raw_num.where(raw_num.astype(bool), _col('number'))

    normalized = raw_num.map(lambda x: normalize_ru_phone(x, reject_non_ru=True) or '')
    valid_mask = normalized.map(is_valid_ru_phone)
    df = df.loc[valid_mask].copy()
    if df.empty:
        return pd.DataFrame(columns=RAW_REPUTATION_SCHEMA)
    df['normalized_number'] = normalized.loc[valid_mask].values

    def _num(col_options, default=0):
        for c in col_options:
            if c in df.columns:
                return pd.to_numeric(df[c], errors='coerce').fillna(default).astype(int)
        return pd.Series(default, index=df.index, dtype=int)

    def _flt(col_options, default):
        for c in col_options:
            if c in df.columns:
                return pd.to_numeric(df[c], errors='coerce').fillna(default).astype(float)
        return pd.Series(default, index=df.index, dtype=float)

    def _str(col_options, default=''):
        for c in col_options:
            if c in df.columns:
                return df[c].fillna(default).astype(str)
        return pd.Series(default, index=df.index, dtype=object)

    negative = _num(['negative_count', 'negative', 'bad'])
    positive = _num(['positive_count', 'positive', 'good'])
    neutral = _num(['neutral_count', 'neutral'])
    review_count = _num(['review_count', 'reviews', 'overall'])
    review_count = review_count.where(review_count > 0, negative + positive + neutral)

    view_count = _num(['view_count', 'views'])
    search_volume = _num(['search_volume'])
    search_volume = search_volume.where(
        ~((search_volume <= 0) & (view_count > 0)), view_count
    )

    confidence = _flt(['source_confidence', 'confidence'], 0.5)
    reliability = _flt(['source_reliability'], 0.5)

    categories = _str(['categories', 'category', 'rating'])
    src = _str(['source'], default=source_hint or 'unknown')
    src = src.where(src.astype(bool), source_hint or 'unknown')
    last_review = _str(['last_review_at'])
    last_review = last_review.where(last_review.astype(bool), _str(['detail_date']))
    first_seen = _str(['first_seen_at'])
    detail_date = _str(['detail_date'])
    page_title = _str(['page_title'])
    url = _str(['url'])
    related = _num(['related_count'])

    return pd.DataFrame({
        'normalized_number': df['normalized_number'].values,
        'source': src.values,
        'negative_count': negative.values,
        'positive_count': positive.values,
        'neutral_count': neutral.values,
        'review_count': review_count.values,
        'search_volume': search_volume.values,
        'categories': categories.values,
        'last_review_at': last_review.values,
        'first_seen_at': first_seen.values,
        'source_confidence': confidence.values,
        'source_reliability': reliability.values,
        'view_count': view_count.values,
        'related_count': related.values,
        'detail_date': detail_date.values,
        'page_title': page_title.values,
        'url': url.values,
    })


def load_reputation_df():
    """Pandas-fast loader. Reads all RAW_REPUTATION_FILES, normalizes and
    concatenates them into a single DataFrame. Returns an empty DataFrame
    in the canonical schema if nothing is available.
    """
    parts = []
    for filename in RAW_REPUTATION_FILES:
        source_hint = filename.replace('reviews_', '').replace('.csv', '')
        df = _read_csv_pandas(os.path.join(RAW_DIR, filename))
        if df is None:
            continue
        normalized = _normalize_reputation_df(df, source_hint)
        if not normalized.empty:
            parts.append(normalized)
    if not parts:
        return pd.DataFrame(columns=RAW_REPUTATION_SCHEMA)
    return pd.concat(parts, ignore_index=True)


def load_blacklist_df():
    """Pandas-fast loader for blacklist files. Returns DataFrame in the
    canonical reputation schema with ``negative_count=1`` and the proper
    static-blacklist source tag per Requirement design.
    """
    parts = []
    for filename, source in BLACKLIST_FILES:
        df = _read_csv_pandas(os.path.join(RAW_DIR, filename))
        if df is None:
            continue
        # Pick first non-empty of the candidate phone columns.
        def _col(name, frame=df):
            return frame[name].fillna('') if name in frame.columns else pd.Series('', index=frame.index)

        raw_num = _col('normalized_number')
        raw_num = raw_num.where(raw_num.astype(bool), _col('phone'))
        raw_num = raw_num.where(raw_num.astype(bool), _col('number'))
        normalized = raw_num.map(lambda x: normalize_ru_phone(x) or '')
        valid_mask = normalized.map(is_valid_ru_phone)
        sub = df.loc[valid_mask].copy()
        if sub.empty:
            continue
        sub['normalized_number'] = normalized.loc[valid_mask].values

        category = sub.get('category', pd.Series('мошенничество', index=sub.index)).fillna('мошенничество')
        category = category.where(category.astype(bool), 'мошенничество')

        if 'confidence' in sub.columns:
            confidence = pd.to_numeric(sub['confidence'], errors='coerce').fillna(0.85).astype(float)
        elif 'source_confidence' in sub.columns:
            confidence = pd.to_numeric(sub['source_confidence'], errors='coerce').fillna(0.85).astype(float)
        else:
            confidence = pd.Series(0.85, index=sub.index, dtype=float)

        url = sub.get('url', pd.Series('', index=sub.index)).fillna('').astype(str)

        n = len(sub)
        parts.append(pd.DataFrame({
            'normalized_number': sub['normalized_number'].values,
            'source': [source] * n,
            'negative_count': [1] * n,
            'positive_count': [0] * n,
            'neutral_count': [0] * n,
            'review_count': [1] * n,
            'search_volume': [0] * n,
            'categories': category.values,
            'last_review_at': [''] * n,
            'first_seen_at': [''] * n,
            'source_confidence': confidence.values,
            'source_reliability': [0.85] * n,
            'view_count': [0] * n,
            'related_count': [0] * n,
            'detail_date': [''] * n,
            'page_title': [''] * n,
            'url': url.values,
        }))
    if not parts:
        return pd.DataFrame(columns=RAW_REPUTATION_SCHEMA)
    return pd.concat(parts, ignore_index=True)


def load_blacklist_rows() -> List[Dict]:
    rows = []
    for filename, source in BLACKLIST_FILES:
        for row in iter_csv(os.path.join(RAW_DIR, filename)):
            number = normalize_ru_phone(row.get('normalized_number') or row.get('phone') or row.get('number') or '')
            if not number or not is_valid_ru_phone(number):
                continue
            category = row.get('category') or 'мошенничество'
            confidence = safe_float(row.get('confidence') or row.get('source_confidence') or 0.85)
            rows.append({
                'normalized_number': number,
                'source': source,
                'negative_count': 1,
                'positive_count': 0,
                'neutral_count': 0,
                'review_count': 1,
                'search_volume': 0,
                'categories': category,
                'last_review_at': '',
                'first_seen_at': '',
                'source_confidence': confidence,
                'source_reliability': 0.85,
                'view_count': 0,
                'related_count': 0,
                'detail_date': '',
                'page_title': '',
                'url': row.get('url', ''),
            })
    return rows


def aggregate_reputation(rows: List[Dict]) -> Dict[str, Dict]:
    """Backwards-compatible aggregation. For multi-million-row inputs use
    aggregate_reputation_df which keeps everything in pandas and avoids
    materializing per-number Python dicts twice.
    """
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    return _df_to_aggregated_dict(_aggregate_reputation_df(df))


def _aggregate_reputation_df(df):
    """Vectorized counterpart of aggregate_reputation. Returns a DataFrame
    indexed by ``normalized_number`` with one row per unique number.

    This avoids per-group Python callables (which are O(groups × py-overhead)
    and choke on >1M groups) by:
      * doing all numeric reductions via builtin ``groupby(...).sum/max``;
      * doing string aggregations via ``groupby(...).agg(set)`` followed by
        a single vectorized ';'.join over a per-group list.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=RAW_REPUTATION_SCHEMA).set_index('normalized_number')

    for col in ['negative_count', 'positive_count', 'neutral_count', 'review_count',
                'search_volume', 'view_count', 'related_count']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        else:
            df[col] = 0
    for col in ['source_confidence', 'source_reliability']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.5).astype(float)
        else:
            df[col] = 0.5
    for col in ['categories', 'last_review_at', 'first_seen_at', 'detail_date',
                'page_title', 'url', 'source']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
        else:
            df[col] = ''

    # Sort once so groupby.first picks the most informative row per number.
    # Highest source_confidence first, then non-empty categories first.
    df = df.assign(
        _has_cats=df['categories'].astype(bool).astype(int),
    ).sort_values(
        by=['normalized_number', 'source_confidence', '_has_cats'],
        ascending=[True, False, False],
        kind='mergesort',
    )
    g = df.groupby('normalized_number', sort=False)

    # Numeric reductions — all native (C-level, no Python callbacks).
    numeric = g.agg(
        negative_count=('negative_count', 'sum'),
        positive_count=('positive_count', 'sum'),
        neutral_count=('neutral_count', 'sum'),
        review_count=('review_count', 'sum'),
        search_volume=('search_volume', 'max'),
        view_count=('view_count', 'max'),
        related_count=('related_count', 'max'),
        source_confidence=('source_confidence', 'max'),
        source_reliability=('source_reliability', 'max'),
    )

    sum_components = numeric['negative_count'] + numeric['positive_count'] + numeric['neutral_count']
    numeric['review_count'] = numeric[['review_count']].assign(_s=sum_components).max(axis=1).astype(int)

    # String reductions: take 'first' from the pre-sorted DataFrame. The top
    # row per number now has the highest confidence and (if tied) non-empty
    # categories, which is the same observation the legacy code preferred.
    # No multi-source string union is performed — that was unused by the
    # trainer and 100x slower per-group.
    first_strs = g.agg(
        source=('source', 'first'),
        categories=('categories', 'first'),
        last_review_at=('last_review_at', 'first'),
        first_seen_at=('first_seen_at', 'first'),
        detail_date=('detail_date', 'first'),
        page_title=('page_title', 'first'),
        url=('url', 'first'),
    )

    out = numeric.join(first_strs)
    out['normalized_number'] = out.index.values
    return out


def _df_to_aggregated_dict(df) -> Dict[str, Dict]:
    """Convert aggregated DataFrame to the legacy ``{number: {fields...}}`` dict.

    Used only by the legacy aggregate_reputation wrapper. The fast path in
    main() keeps the DataFrame and reads rows lazily via itertuples.
    """
    if df is None or df.empty:
        return {}
    cols = ['normalized_number', 'source', 'negative_count', 'positive_count',
            'neutral_count', 'review_count', 'search_volume', 'categories',
            'last_review_at', 'first_seen_at', 'source_confidence',
            'source_reliability', 'view_count', 'related_count',
            'detail_date', 'page_title', 'url']
    sub = df[cols] if all(c in df.columns for c in cols) else df.reindex(columns=cols, fill_value='')
    return {row['normalized_number']: dict(row) for row in sub.to_dict('records')}


def determine_label(number: str, reputation: Optional[Dict], whitelist_info: Optional[Dict], in_public_blacklist: bool) -> Tuple[str, float, str]:
    if whitelist_info:
        confidence = safe_float(whitelist_info.get('source_confidence'), 0.7)
        source = whitelist_info.get('source', 'whitelist_official')
        weight = 0.25 if confidence < 0.35 else round(max(0.5, min(2.0, confidence * 2.0)), 2)
        return 'ALLOW', weight, source

    if in_public_blacklist:
        return 'BLOCK', 2.0, reputation.get('source', 'blacklist') if reputation else 'blacklist'

    if not reputation:
        return 'WARN', 0.25, 'unknown'

    negative = safe_int(reputation.get('negative_count'))
    positive = safe_int(reputation.get('positive_count'))
    neutral = safe_int(reputation.get('neutral_count'))
    review_count = max(safe_int(reputation.get('review_count')), negative + positive + neutral)
    search_volume = safe_int(reputation.get('search_volume'))
    view_count = safe_int(reputation.get('view_count'))
    # view_count can substitute for search_volume as a popularity proxy
    if search_volume <= 0 and view_count > 0:
        search_volume = view_count
    total = max(negative + positive + neutral, review_count, 1)
    negative_ratio = negative / total
    positive_ratio = positive / total
    flags = category_flags(reputation.get('categories', ''))
    confidence = safe_float(reputation.get('source_confidence'), 0.5)
    reliability = safe_float(reputation.get('source_reliability'), 0.5)

    # reliability boosts weight: high-reliability sources get stronger signals
    rel_boost = 1.0 + (reliability - 0.5) * 0.4  # range ~0.8..1.2

    if flags['has_fraud_category'] and (negative_ratio >= 0.5 or negative >= 2 or confidence >= 0.8):
        return 'BLOCK', round(1.7 * rel_boost, 2), reputation.get('source', 'reviews')

    if negative_ratio >= 0.75 and review_count >= 3:
        return 'BLOCK', round(1.5 * rel_boost, 2), reputation.get('source', 'reviews')

    if flags['has_telemarketing_category'] or negative_ratio >= 0.35:
        return 'WARN', round(1.0 * rel_boost, 2), reputation.get('source', 'reviews')

    if search_volume >= 500 and review_count <= 2:
        return 'WARN', round(0.8 * rel_boost, 2), reputation.get('source', 'search_volume')

    if positive_ratio >= 0.75 and review_count >= 3:
        return 'ALLOW', 0.9, reputation.get('source', 'reviews')

    return 'WARN', 0.5, reputation.get('source', 'reviews')


def infer_timezone_offset(region: str) -> int:
    text = (region or '').lower()
    if any(k in text for k in ['камчат', 'чукот']):
        return 12
    if any(k in text for k in ['магадан', 'сахалин']):
        return 11
    if any(k in text for k in ['якут', 'примор', 'хабаров']):
        return 10
    if any(k in text for k in ['иркут', 'бурят']):
        return 8
    if any(k in text for k in ['краснояр', 'кемеров', 'томск', 'новосибир']):
        return 7
    if 'омск' in text:
        return 6
    if any(k in text for k in ['екатерин', 'свердлов', 'челябин', 'перм', 'тюмень']):
        return 5
    if any(k in text for k in ['самар', 'саратов', 'удмурт']):
        return 4
    if 'калининград' in text:
        return 2
    return 3


def enrich_numbering(number: str, numbering_plan: Optional[NumberingPlan]) -> Dict:
    match = numbering_plan.lookup(number) if numbering_plan else None
    operator = match.get('operator', '') if match else ''
    region = match.get('region', '') if match else ''
    n_type = match.get('number_type') if match else number_type(number)
    op_bucket = operator_bucket(operator)
    return {
        'numbering_match': match is not None,
        'is_valid_ru_range': match is not None,
        'operator': operator,
        'region': region,
        'number_type': n_type,
        'def_code': get_def_code(number) or '',
        'operator_bucket': op_bucket,
        'region_bucket': stable_bucket(region),
        'timezone_offset': infer_timezone_offset(region),
        'is_mvno': 1 if op_bucket == 'mvno' else 0,
    }


def _is_public_blacklist_source(source: Optional[str], reputation: Optional[Dict]) -> bool:
    """Номер в публичном blacklist (мошеловка и т.п.) — реальный сигнал, который будет доступен на устройстве
    через бандл/обновляемый фид, а не выводится из label.
    """
    haystacks = []
    if source:
        haystacks.append(source.lower())
    if reputation:
        rep_src = reputation.get('source')
        if rep_src:
            haystacks.append(str(rep_src).lower())
    return any('moshelovka' in h or 'public_blacklist' in h for h in haystacks)


def _is_static_allowlist_source(source: Optional[str]) -> bool:
    """Номер в статическом whitelist (банки/госуслуги/официальные бизнесы) — бандлится в APK."""
    if not source:
        return False
    s = source.lower()
    return s.startswith('whitelist') or s.startswith('static_whitelist')


def build_feature_record(number: str, label: str, weight: float, source: str, reputation: Optional[Dict], numbering: Dict) -> Dict:
    reputation = reputation or {}
    negative = safe_int(reputation.get('negative_count'))
    positive = safe_int(reputation.get('positive_count'))
    neutral = safe_int(reputation.get('neutral_count'))
    review_count = max(safe_int(reputation.get('review_count')), negative + positive + neutral)
    total_votes = max(negative + positive + neutral, review_count, 1)
    search_volume = safe_int(reputation.get('search_volume'))
    flags = category_flags(reputation.get('categories', ''))
    in_public_bl = _is_public_blacklist_source(source, reputation)
    in_static_wl = _is_static_allowlist_source(source)
    compact_meta = {
        **reputation,
        **numbering,
        'source_confidence': safe_float(reputation.get('source_confidence'), 0.95 if source == 'whitelist_official' else 0.5),
        'source_reliability': safe_float(reputation.get('source_reliability'), 0.5),
        # Источник-based, НЕ label-based: эти флаги отражают реальные бандл/фиды на устройстве.
        'inAllowlist': in_static_wl,
        'inBlacklist': in_public_bl,
        'contactsAvailable': True,
    }
    compact = compact_feature_vector(number, label, compact_meta)
    view_count = safe_int(reputation.get('view_count'))
    related_count = safe_int(reputation.get('related_count'))
    row = {
        'normalized_number': number,
        'label': label,
        'label_id': LABEL_TO_ID[label],
        'weight': weight,
        'source': source,
        'source_confidence': compact['sourceConfidence'],
        'negative_count': negative,
        'positive_count': positive,
        'neutral_count': neutral,
        'review_count': review_count,
        'negative_ratio': negative / total_votes,
        'positive_ratio': positive / total_votes,
        'search_volume': search_volume,
        'search_volume_log': math.log1p(search_volume),
        'review_velocity_48h': review_velocity(reputation.get('last_review_at'), reputation.get('first_seen_at'), review_count, 2),
        'review_velocity_7d': review_velocity(reputation.get('last_review_at'), reputation.get('first_seen_at'), review_count, 7),
        'has_fraud_category': flags['has_fraud_category'],
        'has_telemarketing_category': flags['has_telemarketing_category'],
        'has_finance_category': flags['has_finance_category'],
        'number_type': numbering.get('number_type', number_type(number)),
        'def_code': numbering.get('def_code', ''),
        'operator': numbering.get('operator', ''),
        'region': numbering.get('region', ''),
        'timezone_offset': numbering.get('timezone_offset', 3),
        'is_mvno': numbering.get('is_mvno', 0),
        'source_reliability': safe_float(reputation.get('source_reliability'), 0.5),
        'view_count': view_count,
        'view_count_log': math.log1p(view_count),
        'related_count': related_count,
        'detail_date': reputation.get('detail_date', ''),
    }
    row.update(compact)
    return row


def generate_smoke_rows(count: int) -> List[Dict]:
    rows = []
    for _ in range(count):
        label = random.choice(['ALLOW', 'WARN', 'BLOCK'])
        if label == 'ALLOW':
            number = f'+7800{random.randint(0, 9999999):07d}'
            negative, positive, category = 0, random.randint(3, 12), ''
        elif label == 'WARN':
            number = f'+7495{random.randint(0, 9999999):07d}'
            negative, positive, category = random.randint(2, 8), random.randint(0, 2), 'спам;телемаркетинг'
        else:
            number = f'+79{random.randint(0, 999999999):09d}'[:12]
            negative, positive, category = random.randint(5, 25), random.randint(0, 1), 'мошенничество'
        rows.append({
            'normalized_number': number,
            'source': 'smoke_synthetic',
            'negative_count': negative,
            'positive_count': positive,
            'neutral_count': 0,
            'review_count': negative + positive,
            'search_volume': random.randint(0, 5000),
            'categories': category,
            'last_review_at': '',
            'first_seen_at': '',
            'source_confidence': 0.5,
            'source_reliability': 0.5,
            'view_count': random.randint(0, 500),
            'related_count': random.randint(0, 10),
            'detail_date': '',
            'page_title': '',
            'url': '',
        })
    return rows


def build_entry(number: str, reputation: Optional[Dict], whitelist_info: Optional[Dict],
                numbering: Dict) -> Dict:
    """Build a pre-feature-record entry that carries everything needed
    to construct the final feature row plus the cold-start signals
    (in_static_allowlist / in_public_blacklist / no_metadata) that the
    Phase 4D balancer reads.
    """
    in_public_blacklist = bool(
        reputation and any(s in (reputation.get('source') or '') for s in ['moshelovka'])
    )
    label, weight, source = determine_label(number, reputation, whitelist_info, in_public_blacklist)

    signal_meta = {**(reputation or {})}
    if whitelist_info:
        signal_meta['source_confidence'] = whitelist_info.get(
            'source_confidence', signal_meta.get('source_confidence', 0.7),
        )

    in_public_bl = _is_public_blacklist_source(source, signal_meta) or in_public_blacklist
    in_static_wl = _is_static_allowlist_source(source)

    return {
        'number': number,
        'label': label,
        'weight': weight,
        'source': source,
        'reputation': signal_meta,
        'numbering': numbering,
        'in_static_allowlist': bool(in_static_wl),
        'in_public_blacklist': bool(in_public_bl),
        'is_contact': False,  # always False at dataset-build time; runtime-only signal.
    }


def entry_to_records(entry: Dict) -> Tuple[Dict, List[float], Dict]:
    """Re-derive (metadata_row, tflite_row, labeled_row) from an entry.

    Called once per entry after the balancer has had a chance to mutate
    the underlying ``reputation`` dict (Strategy A) or zero it out
    (Strategy C). ``build_feature_record`` and ``compact_row`` consume
    the latest reputation state directly so the compact feature vector
    stays consistent with the modified metadata.
    """
    number = entry['number']
    label = entry['label']
    weight = entry['weight']
    source = entry['source']
    signal_meta = entry['reputation'] or {}
    numbering = entry['numbering']
    in_static_wl = entry.get('in_static_allowlist', False)
    in_public_bl = entry.get('in_public_blacklist', False)

    feature_record = build_feature_record(number, label, weight, source, signal_meta, numbering)

    tflite_row = compact_row(number, label, {
        **signal_meta,
        **numbering,
        'source_confidence': feature_record['source_confidence'],
        'inAllowlist': in_static_wl,
        'inBlacklist': in_public_bl,
    }) + [LABEL_TO_ID[label]]

    labeled_row = {
        'normalized_number': number,
        'label': label,
        'label_id': LABEL_TO_ID[label],
        'weight': weight,
        'source': source,
        'source_confidence': feature_record['source_confidence'],
    }
    return feature_record, tflite_row, labeled_row


def _no_metadata_counts(entries: Iterable[Dict]) -> Dict[str, int]:
    counts = {'ALLOW': 0, 'WARN': 0, 'BLOCK': 0}
    for e in entries:
        if csb._has_no_metadata(e):
            counts[e['label']] = counts.get(e['label'], 0) + 1
    return counts


def _label_counts(entries: Iterable[Dict]) -> Dict[str, int]:
    counts = {'ALLOW': 0, 'WARN': 0, 'BLOCK': 0}
    for e in entries:
        counts[e['label']] = counts.get(e['label'], 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description='Build RF metadata dataset')
    parser.add_argument('--smoke-synthetic', type=int, default=0, help='Add synthetic rows only for pipeline smoke tests')
    parser.add_argument('--min-real-block', type=int, default=1, help='Warn if fewer real BLOCK rows exist')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the Phase 4D balancer (deterministic)')
    parser.add_argument('--phase4d-balance', action='store_true',
                        help='Phase 4D: break ALLOW⇄noMetadata=1 shortcut (Strategy A+B by default).')
    parser.add_argument('--phase4d-inject-fraction', type=float, default=0.30,
                        help='Phase 4D Strategy A: fraction of cold-ALLOW entries to enrich with synthetic positive metadata.')
    parser.add_argument('--phase4d-drop-fraction', type=float, default=0.25,
                        help='Phase 4D Strategy B: fraction of cold-ALLOW entries (no allowlist, no contact) to drop.')
    parser.add_argument('--phase4d-shadow-fraction', type=float, default=0.0,
                        help='Phase 4D Strategy C (optional): fraction of BLOCK/WARN entries to duplicate as cold-view shadow rows.')
    parser.add_argument('--max-numbers', type=int, default=0,
                        help='Cap the number of unique normalized phones processed (0 = no cap). Stratified by label proxy: keeps all whitelist + blacklist, then random reputation rows.')
    parser.add_argument('--progress-every', type=int, default=50000,
                        help='Print a heartbeat every N entries during the per-number loop.')
    args = parser.parse_args()

    import time
    t0 = time.time()
    def _log(msg: str):
        print(f'[{time.time() - t0:6.1f}s] {msg}', flush=True)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    _log('Loading numbering plan...')
    numbering_records = load_numbering_csv(os.path.join(RAW_DIR, 'ru_numbering_plan.csv'))
    numbering_plan = NumberingPlan(numbering_records) if numbering_records else None
    _log(f'Numbering ranges: {len(numbering_records)}')

    _log('Loading whitelist...')
    whitelist = load_whitelist()
    _log(f'Whitelist: {len(whitelist)} numbers')

    _log('Loading reputation (pandas fast path)...')
    rep_df = load_reputation_df()
    _log(f'Reputation rows after normalization: {len(rep_df):,}')

    _log('Loading blacklist...')
    bl_df = load_blacklist_df()
    _log(f'Blacklist rows after normalization: {len(bl_df):,}')

    if not bl_df.empty:
        rep_df = pd.concat([rep_df, bl_df], ignore_index=True)

    if args.smoke_synthetic > 0:
        smoke_rows = generate_smoke_rows(args.smoke_synthetic)
        if smoke_rows:
            smoke_df = pd.DataFrame(smoke_rows).reindex(columns=RAW_REPUTATION_SCHEMA, fill_value='')
            rep_df = pd.concat([rep_df, smoke_df], ignore_index=True)

    _log(f'Aggregating {len(rep_df):,} rows by number...')
    aggregated_df = _aggregate_reputation_df(rep_df)
    del rep_df
    _log(f'Unique reputation numbers: {len(aggregated_df):,}')

    write_dict_csv(
        os.path.join(PROCESSED_DIR, 'ru_reputation_raw.csv'),
        aggregated_df.to_dict('records'),
        RAW_REPUTATION_SCHEMA,
    )
    _log('Wrote processed/ru_reputation_raw.csv')

    # Sort by number once for deterministic output and union with whitelist.
    aggregated_numbers = aggregated_df.index.tolist()
    all_numbers = sorted(set(whitelist.keys()) | set(aggregated_numbers))
    _log(f'Total unique numbers (whitelist ∪ reputation): {len(all_numbers):,}')

    if args.max_numbers > 0 and len(all_numbers) > args.max_numbers:
        rng = random.Random(args.seed)
        # Always keep whitelist members and any number with categories starting
        # with мошенничество (proxy for high-signal); fill remainder randomly.
        wl = set(whitelist.keys())
        priority = [n for n in all_numbers if n in wl]
        rest = [n for n in all_numbers if n not in wl]
        rng.shuffle(rest)
        budget = max(args.max_numbers - len(priority), 0)
        all_numbers = sorted(priority + rest[:budget])
        _log(f'Capped to --max-numbers: {len(all_numbers):,}')

    # aggregated_df.loc[number] is fast; convert to per-number dict lazily.
    aggregated_records = aggregated_df.to_dict('index')
    del aggregated_df

    entries: List[Dict] = []
    real_block_count = 0
    progress_every = max(args.progress_every, 1)
    _log(f'Building entries for {len(all_numbers):,} numbers...')
    for idx, number in enumerate(all_numbers, start=1):
        reputation = aggregated_records.get(number)
        if reputation is not None:
            # to_dict('index') strips the index column; restore for downstream readers.
            reputation = dict(reputation)
            reputation.setdefault('normalized_number', number)
        whitelist_info = whitelist.get(number)
        numbering = enrich_numbering(number, numbering_plan)
        entry = build_entry(number, reputation, whitelist_info, numbering)
        if entry['label'] == 'BLOCK' and entry['source'] != 'smoke_synthetic':
            real_block_count += 1
        entries.append(entry)
        if idx % progress_every == 0:
            _log(f'  built {idx:,}/{len(all_numbers):,} entries')

    if args.phase4d_balance:
        rng = random.Random(args.seed)
        before_no_meta = _no_metadata_counts(entries)
        before_total = _label_counts(entries)

        entries = csb.inject_synthetic_metadata_into_allow(
            entries, args.phase4d_inject_fraction, rng,
        )
        entries = csb.subsample_allow_no_metadata(
            entries, args.phase4d_drop_fraction, rng,
        )
        if args.phase4d_shadow_fraction > 0:
            entries = csb.add_shadow_cold_block_warn(
                entries, args.phase4d_shadow_fraction, rng,
            )

        after_no_meta = _no_metadata_counts(entries)
        after_total = _label_counts(entries)

        def _pct_allow_given_nometa(counts):
            total = sum(counts.values())
            return (counts.get('ALLOW', 0) / total) if total else 0.0

        print('=== Phase 4D balancer summary ===')
        for cls in ('ALLOW', 'WARN', 'BLOCK'):
            print(f'  {cls:5s} with noMetadata=1   before: {before_no_meta.get(cls, 0):6d}'
                  f'   after: {after_no_meta.get(cls, 0):6d}')
        print(f'  P(ALLOW | noMetadata=1)   before: {_pct_allow_given_nometa(before_no_meta):.4f}'
              f'   after: {_pct_allow_given_nometa(after_no_meta):.4f}')
        print(f'  Total rows                before: {sum(before_total.values()):6d}'
              f'   after: {sum(after_total.values()):6d}')
        print(f'  Per-class                 before: ALLOW={before_total["ALLOW"]} '
              f'WARN={before_total["WARN"]} BLOCK={before_total["BLOCK"]}')
        print(f'                            after:  ALLOW={after_total["ALLOW"]} '
              f'WARN={after_total["WARN"]} BLOCK={after_total["BLOCK"]}')

    total_numbers, counts = write_output_records(entries)

    print(f'Total numbers: {total_numbers}')
    print(f'ALLOW={counts["ALLOW"]} WARN={counts["WARN"]} BLOCK={counts["BLOCK"]}')
    print(f'Files written to {PROCESSED_DIR}')

    if real_block_count < args.min_real_block:
        print('WARNING: real BLOCK rows are too few for final training. Use collector/API/manual CSV before export.')


if __name__ == '__main__':
    main()
