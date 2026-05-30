"""
Safe offline collector для РФ reputation sources.

Скрипт принимает список candidate-номеров и сохраняет raw reputation CSV.
Он не делает агрессивный обход сайтов и не используется внутри Android-приложения.
"""

import argparse
import csv
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html import unescape
from typing import Dict, Iterable, List, Optional

import sys
sys.path.insert(0, os.path.dirname(__file__))
from ru_number_normalizer import normalize_ru_phone
from ru_metadata_features import category_flags, safe_int

RAW_SCHEMA = [
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

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'ru')
RAW_DIR = os.path.join(BASE_DIR, 'raw')
DEFAULT_OUTPUT = os.path.join(RAW_DIR, 'ru_reputation_raw.csv')
CACHE_DIR = os.path.join(RAW_DIR, 'cache')

USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/124.0 Safari/537.36'
)


def read_candidates(path: str) -> List[str]:
    numbers = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            value = line.strip()
            if not value or value.startswith('#'):
                continue
            if ',' in value:
                value = value.split(',')[0].strip()
            normalized = normalize_ru_phone(value, reject_non_ru=False)
            if normalized:
                numbers.append(normalized)
    return sorted(set(numbers))


def append_rows(path: str, rows: Iterable[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RAW_SCHEMA)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in RAW_SCHEMA})


def safe_get(url: str, timeout: int = 25, retries: int = 2) -> Optional[str]:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, 'Accept-Language': 'ru,en;q=0.8'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                charset = resp.headers.get_content_charset() or 'utf-8'
                try:
                    return raw.decode(charset, errors='replace')
                except LookupError:
                    return raw.decode('utf-8', errors='replace')
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt >= retries:
                print(f'GET failed: {url} ({e})')
                return None
            time.sleep(2 + attempt * 2)
    return None


def clean_text(value: str) -> str:
    value = re.sub(r'<script.*?</script>', ' ', value, flags=re.I | re.S)
    value = re.sub(r'<style.*?</style>', ' ', value, flags=re.I | re.S)
    value = re.sub(r'<[^>]+>', ' ', value)
    value = unescape(value)
    return re.sub(r'\s+', ' ', value).strip()


def count_keywords(text: str, keywords: Iterable[str]) -> int:
    lower = text.lower()
    return sum(lower.count(k.lower()) for k in keywords)


def extract_first_int(patterns: Iterable[str], text: str) -> int:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return safe_int(match.group(1))
    return 0


def parse_categories(text: str) -> str:
    candidates = []
    lowered = text.lower()
    known = [
        'мошенничество', 'мошенники', 'спам', 'телемаркетинг', 'реклама',
        'коллекторы', 'опрос', 'банк', 'финансовые услуги', 'робот',
        'безопасность банка', 'нежелательный звонок', 'другое',
    ]
    for item in known:
        if item in lowered:
            candidates.append(item)
    return ';'.join(sorted(set(candidates)))


def parse_neberitrubku(number: str, html: str, url: str) -> Dict:
    text = clean_text(html)
    negative = count_keywords(text, ['отрицатель', 'опасн', 'мошен', 'спам'])
    positive = count_keywords(text, ['положитель', 'безопасн', 'полезн'])
    neutral = count_keywords(text, ['нейтраль', 'неизвест'])
    review_count = extract_first_int([
        r'(\d+)\s+отзыв',
        r'отзывов\s*[:\-]?\s*(\d+)',
        r'комментариев\s*[:\-]?\s*(\d+)',
    ], text)
    search_volume = extract_first_int([
        r'(\d+)\s+просмотр',
        r'просмотров\s*[:\-]?\s*(\d+)',
        r'запросов\s*[:\-]?\s*(\d+)',
    ], text)
    categories = parse_categories(text)
    flags = category_flags(categories)
    if review_count == 0:
        review_count = max(negative + positive + neutral, 0)
    confidence = 0.35
    if review_count >= 5:
        confidence += 0.25
    if flags['has_fraud_category']:
        confidence += 0.25
    if search_volume >= 100:
        confidence += 0.15
    return {
        'normalized_number': number,
        'source': 'neberitrubku',
        'negative_count': negative,
        'positive_count': positive,
        'neutral_count': neutral,
        'review_count': review_count,
        'search_volume': search_volume,
        'categories': categories,
        'source_confidence': min(confidence, 1.0),
        'url': url,
    }


def parse_zvonili(number: str, html: str, url: str) -> Dict:
    text = clean_text(html)
    negative = count_keywords(text, ['мошен', 'спам', 'отрицатель', 'опасн', 'реклама'])
    positive = count_keywords(text, ['полезн', 'безопасн', 'положитель'])
    neutral = count_keywords(text, ['неизвест', 'нейтраль'])
    review_count = extract_first_int([r'(\d+)\s+отзыв', r'(\d+)\s+комментар'], text)
    search_volume = extract_first_int([r'(\d+)\s+просмотр', r'(\d+)\s+запрос'], text)
    categories = parse_categories(text)
    flags = category_flags(categories)
    if review_count == 0:
        review_count = max(negative + positive + neutral, 0)
    confidence = 0.3 + min(review_count / 20, 0.35) + (0.25 if flags['has_fraud_category'] else 0.0)
    return {
        'normalized_number': number,
        'source': 'zvonili',
        'negative_count': negative,
        'positive_count': positive,
        'neutral_count': neutral,
        'review_count': review_count,
        'search_volume': search_volume,
        'categories': categories,
        'source_confidence': min(confidence, 1.0),
        'url': url,
    }


def parse_moshelovka(number: str, html: str, url: str) -> Dict:
    text = clean_text(html)
    found = number.replace('+', '') in re.sub(r'\D', '', text)
    categories = parse_categories(text) or 'мошенничество'
    confidence = 0.9 if found else 0.65
    return {
        'normalized_number': number,
        'source': 'moshelovka',
        'negative_count': 1 if found else 0,
        'positive_count': 0,
        'neutral_count': 0,
        'review_count': 1 if found else 0,
        'search_volume': 0,
        'categories': categories,
        'source_confidence': confidence,
        'url': url,
    }


def build_url(source: str, number: str) -> str:
    query = urllib.parse.quote(number)
    digits = re.sub(r'\D', '', number)
    if source == 'neberitrubku':
        return f'https://www.neberitrubku.ru/search?q={query}'
    if source == 'zvonili':
        return f'https://zvonili.com/phone/{digits}'
    if source == 'moshelovka':
        return f'https://moshelovka.onf.ru/blacklist/?search={query}'
    raise ValueError(f'Unsupported source: {source}')


def parse_source(source: str, number: str, html: str, url: str) -> Dict:
    if source == 'neberitrubku':
        return parse_neberitrubku(number, html, url)
    if source == 'zvonili':
        return parse_zvonili(number, html, url)
    if source == 'moshelovka':
        return parse_moshelovka(number, html, url)
    raise ValueError(source)


def cache_path(source: str, number: str) -> str:
    safe = re.sub(r'\W+', '_', number).strip('_')
    return os.path.join(CACHE_DIR, source, f'{safe}.html')


def get_with_cache(source: str, number: str, url: str, use_cache: bool) -> Optional[str]:
    path = cache_path(source, number)
    if use_cache and os.path.exists(path):
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    html = safe_get(url)
    if html:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
    return html


def collect(source: str, candidates: List[str], output: str, limit: int, delay_min: float, delay_max: float, use_cache: bool):
    rows = []
    processed = 0
    for number in candidates[:limit if limit > 0 else None]:
        url = build_url(source, number)
        print(f'[{source}] {number} -> {url}')
        html = get_with_cache(source, number, url, use_cache)
        if not html:
            continue
        row = parse_source(source, number, html, url)
        rows.append(row)
        append_rows(output, [row])
        processed += 1
        time.sleep(random.uniform(delay_min, delay_max))
    print(f'Collected {processed} rows to {output}')


def import_csv_adapter(source: str, input_path: str, output: str):
    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_number = row.get('normalized_number') or row.get('phone') or row.get('number') or row.get('Телефон') or ''
            number = normalize_ru_phone(raw_number, reject_non_ru=False)
            if not number:
                continue
            negative = safe_int(row.get('negative_count') or row.get('negative') or row.get('bad') or 0)
            positive = safe_int(row.get('positive_count') or row.get('positive') or row.get('good') or 0)
            neutral = safe_int(row.get('neutral_count') or row.get('neutral') or 0)
            review_count = safe_int(row.get('review_count') or row.get('reviews') or negative + positive + neutral)
            categories = row.get('categories') or row.get('category') or row.get('tags') or ''
            rows.append({
                'normalized_number': number,
                'source': source,
                'negative_count': negative,
                'positive_count': positive,
                'neutral_count': neutral,
                'review_count': review_count,
                'search_volume': safe_int(row.get('search_volume') or row.get('views') or 0),
                'categories': categories,
                'last_review_at': row.get('last_review_at', ''),
                'first_seen_at': row.get('first_seen_at', ''),
                'source_confidence': row.get('source_confidence') or row.get('confidence') or 0.7,
                'url': row.get('url', ''),
            })
    append_rows(output, rows)
    print(f'Imported {len(rows)} rows from {input_path} to {output}')


def main():
    parser = argparse.ArgumentParser(description='Collect RF phone reputation metadata')
    parser.add_argument('--source', choices=['neberitrubku', 'zvonili', 'moshelovka'], required=True)
    parser.add_argument('--candidates', type=str, help='Text/CSV file with candidate phone numbers')
    parser.add_argument('--import-csv', type=str, help='Import already downloaded CSV into unified raw schema')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--delay-min', type=float, default=2.0)
    parser.add_argument('--delay-max', type=float, default=6.0)
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()

    if args.import_csv:
        import_csv_adapter(args.source, args.import_csv, args.output)
        return

    if not args.candidates:
        raise SystemExit('--candidates is required unless --import-csv is used')

    candidates = read_candidates(args.candidates)
    if not candidates:
        raise SystemExit('No candidate numbers found')

    collect(
        source=args.source,
        candidates=candidates,
        output=args.output,
        limit=args.limit,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        use_cache=not args.no_cache,
    )


if __name__ == '__main__':
    main()
