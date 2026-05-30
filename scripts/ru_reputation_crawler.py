"""
Multi-source reputation crawler for Russian phone-call classification datasets.

The crawler does not generate random ALLOW rows. It only saves numbers that have
public evidence: categories, negative ratings, review text, or blacklist entries.
"""

import argparse
import csv
import json
import os
import random
import threading
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from datetime import datetime
from html import unescape
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from ru_collect_sources import RAW_SCHEMA, USER_AGENT
from ru_metadata_features import safe_int, translate_headers, translate_row, RU_TO_FIELD
from ru_number_normalizer import normalize_ru_phone

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'ru')
RAW_DIR = os.path.join(BASE_DIR, 'raw')
DEFAULT_OUTPUT = os.path.join(RAW_DIR, 'ru_reputation_raw.csv')
DEFAULT_EVIDENCE = os.path.join(RAW_DIR, 'ru_reputation_evidence.csv')
DEFAULT_STATE = os.path.join(RAW_DIR, 'crawler_state.json')
DEFAULT_CACHE_DIR = os.path.join(RAW_DIR, 'cache', 'reputation_crawler')

EVIDENCE_SCHEMA = [
    'normalized_number',
    'source',
    'label_hint',
    'evidence_type',
    'evidence_text',
    'full_text',
    'page_title',
    'negative_count',
    'positive_count',
    'neutral_count',
    'review_count',
    'view_count',
    'related_count',
    'categories',
    'source_confidence',
    'source_reliability',
    'detail_date',
    'fraud_hits',
    'warn_hits',
    'url',
    'collected_at',
]

SOURCES = {
    'spravportal', 'callfilter', 'zvonili', 'moshelovka', 'bloha', 'getscam',
    'znum', 'prozvonok', 'netrubi', 'zvonkoff', 'ktozvonil', 'znomer',
    'phoneregion',
    # Phase 2 additions (callfilter.info is distinct from callfilter.app):
    'callfilter_info', 'scamcall',
    # Phase 2.5 additions: more reputation aggregators discovered via
    # second-wave scout (web search + sitemap probing). Each chosen for a
    # distinct angle:
    #   * kto_zvonil_tel (kto.zvonil.tel) — clean Bootstrap detail pages with
    #     blockquote-shaped reviews + "Новые отзывы" / "Случайные отзывы"
    #     fan-out lists for cheap discovery.
    #   * abonentik (abonentik.ru) — Vue/Nuxt SSR with rich /category/<slug>
    #     links + numeric "Оценка" rating + ~32k phone URLs in sitemap_1.xml.
    #   * badcall (badcall.ru) — Bootstrap detail pages where review verdict
    #     is encoded in <li class="list-group-item-{danger,warning,success}">.
    'kto_zvonil_tel', 'abonentik', 'badcall',
}

FRAUD_PATTERNS = [
    'телефонное мошенничество', 'мошен', 'развод', 'обман', 'фишинг', 'scam', 'fraud',
    'служба безопасности', 'безопасности банка', 'cvv', 'код из смс', 'код смс',
    'данные карты', 'карта', 'банк мошен', 'госуслуг', 'персональные данные',
    'требуют деньги', 'вымог', 'украли', 'списали', 'деньги',
]

WARN_PATTERNS = [
    'реклама', 'спам', 'нежелательный звонок', 'нежелательное сообщение', 'телемаркетинг',
    'опрос', 'коллектор', 'хулиганство', 'угрозы', 'робот', 'робозвон', 'автоответчик',
    'немой звонок', 'молчок', 'звонят и сбрасывают', 'сбрасывают', 'навязывают',
    'навязывание', 'кредит', 'займ', 'мфо', 'страховк', 'предложения', 'названивает',
]

POSITIVE_PATTERNS = [
    'организация', 'компания', 'доставка', 'поддержка', 'официальный', 'безопасный',
]

CATEGORY_LABELS = [
    'Телефонное мошенничество',
    'Реклама, спам',
    'Нежелательный звонок, сообщение',
    'Телефонное хулиганство, угрозы',
    'Опросы',
    'Коллекторы',
    'Другое',
    'Организация',
]

SEED_URLS = {
    'spravportal': [
        'https://www.spravportal.ru/services/who-calls',
        # Public sitemap-index with ~150k phone-page URLs across 3 sub-sitemaps.
        'https://www.spravportal.ru/sitemap.xml',
        'https://www.spravportal.ru/sitemap-whocalls-0.xml',
        'https://www.spravportal.ru/sitemap-whocalls-1.xml',
        'https://www.spravportal.ru/sitemap-whocalls-2.xml',
    ],
    'callfilter': [
        'https://callfilter.app/ru',
        # Numeric prefixes the site responds to (country code 7).
        'https://callfilter.app/74950000000',
        'https://callfilter.app/79000000000',
        'https://callfilter.app/79050000000',
    ],
    'zvonili': [
        'https://zvonili.com/phone/4994441730',
        'https://zvonili.com/phone/9000000000',
        'https://zvonili.com/phone/79031234567',
        'https://zvonili.com/phone/79150000001',
        'https://zvonili.com/phone/79290000001',
        'https://zvonili.com/phone/74990000001',
    ],
    'moshelovka': [
        'https://moshelovka.onf.ru/blacklist/',
        'https://moshelovka.onf.ru/sitemap.xml',
        'https://moshelovka.onf.ru/blacklist-sitemap.xml',
        'https://moshelovka.onf.ru/blacklist-sitemap2.xml',
        'https://moshelovka.onf.ru/blacklist-sitemap3.xml',
        'https://moshelovka.onf.ru/blacklist-sitemap4.xml',
        'https://moshelovka.onf.ru/blacklist-sitemap5.xml',
    ],
    'bloha': [
        'https://bloha.ru/news/chernyy-spisok-telefonnykh-nomerov/',
        # The wp-sitemap on bloha.ru is for a different site. The black-list
        # article above is the real source of phone numbers for this domain.
    ],
    'getscam': [
        'https://getscam.com/',
        # Sitemap-index with phone sub-sitemaps.
        'https://getscam.com/sitemap.xml',
        'https://getscam.com/sitemap_phone.xml',
        'https://getscam.com/sitemap_index_phone_programmatic.xml',
        'https://getscam.com/sitemap_index_phone_programmatic_reviews.xml',
    ],
    'znum': [
        'https://znum.ru/',
        # gzip-compressed; sitemap-index points to it.
        'https://znum.ru/sitemap.xml',
        'https://znum.ru/sitemap-1.xml.gz',
    ],
    'prozvonok': [
        'https://prozvonok.ru/otzyvy',
        # Numeric enumeration starts.
        'https://prozvonok.ru/nomer/9000000000',
        'https://prozvonok.ru/nomer/4951000000',
        'https://prozvonok.ru/nomer/8000000000',
    ],
    'netrubi': [
        # Homepage exposes ~60 most-recently-reviewed /nomer/<X> links — the
        # cheapest fan-out we have for this source.
        'https://netrubi.ru/',
        'https://netrubi.ru/nomer/8001000800',
        'https://netrubi.ru/nomer/9000000000',
        'https://netrubi.ru/nomer/4951000000',
    ],
    'zvonkoff': [
        'https://zvonkoff.net/ru/number/74993806399',
        # Has a small (~50 entry) sitemap with category landing pages.
        'https://zvonkoff.net/sitemap.xml',
        'https://zvonkoff.net/ru/number/79000000000',
    ],
    'ktozvonil': [
        # Detail pages
        'https://ktozvonil.net/nomer/79998094989',
        'https://ktozvonil.net/nomer/79050000000',
        'https://ktozvonil.net/nomer/74950000000',
        # Listing pagination — the homepage exposes thousands of phone-pages
        # via numbered /page/N/ slices. We seed a wide range so the
        # discovery walk has plenty of fan-out without re-scraping the
        # homepage on every iteration.
        'https://ktozvonil.net/',
        'https://ktozvonil.net/page/2/',
        'https://ktozvonil.net/page/3/',
        'https://ktozvonil.net/page/4/',
        'https://ktozvonil.net/page/5/',
        'https://ktozvonil.net/page/10/',
        'https://ktozvonil.net/page/15/',
        'https://ktozvonil.net/page/20/',
        'https://ktozvonil.net/page/30/',
        'https://ktozvonil.net/page/50/',
        'https://ktozvonil.net/page/75/',
        'https://ktozvonil.net/page/100/',
        'https://ktozvonil.net/page/150/',
        'https://ktozvonil.net/page/200/',
        'https://ktozvonil.net/page/250/',
        'https://ktozvonil.net/page/300/',
        # Sitemap (if available)
        'https://ktozvonil.net/sitemap.xml',
    ],
    'phoneregion': [
        # phoneregion.ru — Russian phone reviews; detail URL: /number/<10digits>
        # (10 digits, no country code prefix).
        'https://www.phoneregion.ru/',
        'https://www.phoneregion.ru/number/9361341951',
        'https://www.phoneregion.ru/number/9021125263',
        'https://www.phoneregion.ru/number/8005515185',
        'https://www.phoneregion.ru/number/9853426699',
        'https://www.phoneregion.ru/number/9851127060',
        'https://www.phoneregion.ru/number/4957389881',
    ],
    'znomer': [
        'https://znomer.ru/',
        # ~34k phone-page URLs in this sitemap.
        'https://znomer.ru/znomer-sitemap.xml',
        'https://znomer.ru/79150611881',
        'https://znomer.ru/74951271424',
        'https://znomer.ru/79684937322',
        'https://znomer.ru/79310091251',
        'https://znomer.ru/79587639148',
        'https://znomer.ru/74952291499',
        'https://znomer.ru/79291965082',
    ],
    'callfilter_info': [
        # callfilter.info publishes a flat sitemap with ~50k /number/<11-digit>
        # URLs (status-label + comments). Single-fetch fan-out is huge.
        'https://callfilter.info/',
        'https://callfilter.info/sitemap.xml',
        'https://callfilter.info/number/74957730527',
        'https://callfilter.info/number/79867257983',
        'https://callfilter.info/number/78005500500',
    ],
    'scamcall': [
        # scamcall.ru detail URL: /phone/<10digits> (no country code).
        # Sitemap-index points to gz-sitemaps with phone URLs.
        'https://scamcall.ru/',
        'https://scamcall.ru/sitemap.xml',
        'https://scamcall.ru/phone/9584069694',
        'https://scamcall.ru/phone/4957730527',
        'https://scamcall.ru/phone/8005500500',
    ],
    'kto_zvonil_tel': [
        # kto.zvonil.tel detail URL: /+<11-digit, leading 7>.
        # Root page exposes ~60 most-recent /+<phone> links plus a "Случайные
        # отзывы" block of older fan-out URLs.
        'https://kto.zvonil.tel/',
        'https://kto.zvonil.tel/+79867257983',
        'https://kto.zvonil.tel/+79584069694',
        'https://kto.zvonil.tel/+74957730527',
        'https://kto.zvonil.tel/+78005500500',
    ],
    'abonentik': [
        # abonentik.ru detail URL: /nomer/<11-digit, leading 7>.
        # Sitemap index points to /sitemap_1.xml with ~32k phone URLs and a
        # separate /abonentik_category.xml with category landing pages.
        'https://abonentik.ru/',
        'https://abonentik.ru/sitemap.xml',
        'https://abonentik.ru/sitemap_index_phone.xml',
        'https://abonentik.ru/sitemap_1.xml',
        'https://abonentik.ru/abonentik_category.xml',
        'https://abonentik.ru/nomer/79867257983',
        'https://abonentik.ru/nomer/79688770249',
        'https://abonentik.ru/nomer/74957730527',
    ],
    'badcall': [
        # badcall.ru detail URL: /phones/<10digits, NO country code>.
        # No public sitemap; root page exposes the most-recently-reviewed
        # phones via /phones/<10> hrefs.
        'https://badcall.ru/',
        'https://badcall.ru/phones/9867257983',
        'https://badcall.ru/phones/9584069694',
        'https://badcall.ru/phones/4957730527',
        'https://badcall.ru/phones/8005500500',
    ],
}


# Mobile + special prefixes for the numeric-enumeration fallback. Each entry is
# the leading 5 digits (country code + DEF code). The fallback generator picks
# from this list and appends 6 random digits, for sources with the URL pattern
# <host>/<11-digit-number>.
ENUMERATION_PREFIXES = [
    # 8800/8804 — toll-free (extremely high spam concentration)
    '78001', '78002', '78003', '78004', '78005', '78006', '78007', '78008',
    '78009', '78010', '78050', '78080', '78840', '78048',
    # 7495/7499 — Moscow landline (calls from there are heavy spam)
    '74951', '74952', '74953', '74954', '74955', '74956', '74957', '74958',
    '74959', '74991', '74992', '74993', '74994', '74995', '74996',
    # 7812/7813 — St. Petersburg landline
    '78121', '78122', '78123',
    # Mobile DEF codes (most spammers operate from mobile prefixes nowadays)
    '79001', '79002', '79003', '79041', '79051', '79061', '79071', '79081',
    '79091', '79101', '79111', '79121', '79141', '79161', '79171', '79181',
    '79201', '79211', '79221', '79231', '79241', '79251', '79261', '79271',
    '79281', '79291', '79301', '79311', '79321', '79331', '79341', '79351',
    '79361', '79371', '79381', '79391', '79401', '79411', '79421', '79431',
    '79501', '79521', '79531', '79611', '79621', '79631', '79661', '79671',
    '79681', '79691', '79801', '79811', '79821', '79831', '79841', '79851',
    '79861', '79871', '79881', '79911', '79961', '79991',
]


def synthetic_enum_urls(source: str, count: int = 200) -> List[str]:
    """Build synthetic detail URLs by enumerating mobile/landline prefixes."""
    if source not in {'spravportal', 'callfilter', 'zvonili', 'getscam',
                      'prozvonok', 'netrubi', 'zvonkoff', 'ktozvonil', 'znomer',
                      'phoneregion', 'callfilter_info', 'scamcall',
                      'kto_zvonil_tel', 'abonentik', 'badcall'}:
        return []
    out: List[str] = []
    rng = random.Random()
    for _ in range(count):
        prefix = rng.choice(ENUMERATION_PREFIXES)
        suffix = ''.join(str(rng.randint(0, 9)) for _ in range(6))
        digits = prefix + suffix  # 11-digit
        url = detail_url(source, '+' + digits)
        if url:
            out.append(url)
    return out


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


def normalize_number(value: str) -> Optional[str]:
    value = unescape(value or '').strip()
    number = normalize_ru_phone(value, reject_non_ru=False)
    if not number:
        return None
    digits = re.sub(r'\D', '', number)
    if len(digits) < 10 or len(digits) > 12:
        return None
    return number


def normalize_path_digits(value: str) -> Optional[str]:
    digits = re.sub(r'\D', '', value or '')
    if len(digits) == 11 and digits[0] in ('7', '8'):
        return normalize_number(digits)
    if len(digits) == 10 and digits[0] in ('9', '8', '4', '3'):
        return normalize_number(digits)
    return None


def numbers_from_text(text: str) -> List[str]:
    if not text:
        return []
    patterns = [
        r'\+\s*7[\s\-()\u00a0]*\d[\d\s\-()\u00a0]{8,}\d',
        r'8[\s\-()\u00a0]*\d{3}[\s\-()\u00a0]*\d[\d\s\-()\u00a0]{6,}\d',
        r'(?<!\d)7\d{10}(?!\d)',
        r'(?<!\d)8\d{10}(?!\d)',
    ]
    result: List[str] = []
    seen: Set[str] = set()
    for pattern in patterns:
        for raw in re.findall(pattern, text):
            normalized = normalize_number(raw)
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
    return result


def meta_description(html: str) -> str:
    match = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)',
        html or '',
        flags=re.I | re.S,
    )
    return unescape(match.group(1)).strip() if match else ''


def clean_html_fragment(fragment: str) -> str:
    value = re.sub(r'<script.*?</script>', ' ', fragment or '', flags=re.I | re.S)
    value = re.sub(r'<style.*?</style>', ' ', value, flags=re.I | re.S)
    value = re.sub(r'<[^>]+>', ' ', value)
    return re.sub(r'\s+', ' ', unescape(value)).strip()


def class_blocks(html: str, class_name: str) -> List[str]:
    pattern = (
        r'<(?:div|li|article)[^>]+class=["\'][^"\']*'
        + re.escape(class_name)
        + r'[^"\']*["\'][^>]*>.*?</(?:div|li|article)>'
    )
    return re.findall(pattern, html or '', flags=re.I | re.S)


def html_class_attribute_contains(html: str, *needles: str) -> bool:
    """Return True iff a rendered HTML ``class="..."`` contains a needle.

    Parsers use this to gate fallback heuristics on actual CSS markers
    (e.g. ``rating-negative`` / ``label-danger``) rather than free-text
    substrings, which fire on JS string literals, HTML comments, hidden
    data-* attributes, and other non-rendered noise. ``<!-- ... -->``,
    ``<script>...</script>`` and ``<style>...</style>`` blocks are stripped
    before scanning so class-shaped fragments inside them never trigger.
    """
    if not html or not needles:
        return False
    needles_l = [n.lower() for n in needles if n]
    if not needles_l:
        return False
    cleaned = re.sub(r'<!--.*?-->', ' ', html, flags=re.S)
    cleaned = re.sub(r'<script\b[^>]*>.*?</script>', ' ', cleaned, flags=re.I | re.S)
    cleaned = re.sub(r'<style\b[^>]*>.*?</style>', ' ', cleaned, flags=re.I | re.S)
    for match in re.finditer(r'class=["\']([^"\']+)["\']', cleaned, flags=re.I):
        cls = match.group(1).lower()
        for needle in needles_l:
            if needle in cls:
                return True
    return False


def review_blocks_text(html: str, class_names: Sequence[str]) -> str:
    pieces: List[str] = []
    for class_name in class_names:
        for block in class_blocks(html, class_name):
            text = clean_html_fragment(block)
            if text:
                pieces.append(text)
    return ' '.join(pieces)


def phone_digits_for_path(number: str, drop_country: bool = False) -> str:
    digits = re.sub(r'\D', '', number)
    if drop_country and digits.startswith('7') and len(digits) == 11:
        return digits[1:]
    return digits


def html_to_text(html: str) -> str:
    value = re.sub(r'<script.*?</script>', ' ', html or '', flags=re.I | re.S)
    value = re.sub(r'<style.*?</style>', ' ', value, flags=re.I | re.S)
    value = re.sub(r'<[^>]+>', ' ', value)
    value = unescape(value)
    return re.sub(r'\s+', ' ', value).strip()


def html_to_lines(html: str) -> List[str]:
    value = re.sub(r'<script.*?</script>', ' ', html or '', flags=re.I | re.S)
    value = re.sub(r'<style.*?</style>', ' ', value, flags=re.I | re.S)
    value = re.sub(r'(?i)<br\s*/?>|</p>|</li>|</div>|</h\d>', '\n', value)
    value = re.sub(r'<[^>]+>', ' ', value)
    value = unescape(value)
    return [re.sub(r'\s+', ' ', line).strip() for line in value.splitlines() if line.strip()]


def extract_section(text: str, starts: Sequence[str], ends: Sequence[str]) -> str:
    lower = text.lower()
    start_idx = -1
    for marker in starts:
        idx = lower.find(marker.lower())
        if idx >= 0 and (start_idx < 0 or idx < start_idx):
            start_idx = idx
    if start_idx < 0:
        return ''
    end_idx = len(text)
    for marker in ends:
        idx = lower.find(marker.lower(), start_idx + 1)
        if idx >= 0 and idx < end_idx:
            end_idx = idx
    return text[start_idx:end_idx].strip()


def count_patterns(text: str, patterns: Sequence[str]) -> int:
    lower = re.sub(r'\bне\s+мошен\w*|\bне\s+спам\w*|\bне\s+реклам\w*|\bне\s+опасн\w*', ' ', (text or '').lower())
    return sum(lower.count(pattern.lower()) for pattern in patterns)


def categories_from_text(text: str) -> List[str]:
    lower = re.sub(r'\bне\s+мошен\w*|\bне\s+спам\w*|\bне\s+реклам\w*|\bне\s+опасн\w*', ' ', (text or '').lower())
    categories: List[str] = []
    for label in CATEGORY_LABELS:
        if label.lower() in lower:
            categories.append(label)
    if any(pattern in lower for pattern in FRAUD_PATTERNS):
        categories.append('мошенничество')
    if any(pattern in lower for pattern in WARN_PATTERNS):
        categories.append('спам')
    return sorted(set(categories))


NEUTRAL_ONLY_CATEGORIES = {'Другое', 'Организация'}


def has_reputation_signal(text: str, categories: Optional[Sequence[str]] = None) -> bool:
    cats = list(categories if categories is not None else categories_from_text(text))
    if any(cat not in NEUTRAL_ONLY_CATEGORIES for cat in cats):
        return True
    return count_patterns(text, FRAUD_PATTERNS) > 0 or count_patterns(text, WARN_PATTERNS) > 0


def view_count_from_text(text: str) -> int:
    matches = re.findall(r'(?<!\d)(\d{1,7})\s+(?:просмотр|просмотров|запрос|запросов|поиск|поисков)', text, flags=re.I)
    values = [safe_int(match) for match in matches if safe_int(match) <= 5_000_000]
    return max(values) if values else 0


def extract_page_title(html: str) -> str:
    match = re.search(r'<title[^>]*>(.*?)</title>', html, flags=re.I | re.S)
    if match:
        return unescape(match.group(1)).strip()[:200]
    return ''


def extract_date_from_text(text: str) -> str:
    for pattern in [
        r'(\d{1,2})[./](\d{1,2})[./](\d{4})',
        r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',
    ]:
        match = re.search(pattern, text)
        if match:
            g = match.groups()
            if len(g[0]) == 4:
                return f'{g[0]}-{g[1].zfill(2)}-{g[2].zfill(2)}'
            return f'{g[2]}-{g[1].zfill(2)}-{g[0].zfill(2)}'
    return ''


def count_related_numbers(html: str, source: str) -> int:
    links = extract_number_links(source, html)
    return len(links)


def review_count_from_text(text: str) -> int:
    matches = re.findall(r'(?<!\d)(\d{1,5})\s+(?:отзыв|отзыва|отзывов|оценк|оценки|оценок)', text, flags=re.I)
    values = [safe_int(match) for match in matches if safe_int(match) <= 5000]
    return max(values) if values else 0


def build_row(
    number: str,
    source: str,
    url: str,
    text: str,
    categories: Optional[Sequence[str]] = None,
    negative_count: int = 0,
    positive_count: int = 0,
    neutral_count: int = 0,
    review_count: int = 0,
    confidence: Optional[float] = None,
    evidence_type: str = 'review',
    full_text: str = '',
    page_title: str = '',
    view_count: int = 0,
    related_count: int = 0,
    detail_date: str = '',
    source_reliability: Optional[float] = None,
) -> Optional[Tuple[Dict, Dict]]:
    number = normalize_number(number) or ''
    if not number:
        return None

    text = re.sub(r'\s+', ' ', (text or '')).strip()
    cats = list(categories or []) + categories_from_text(text)
    cats = sorted({c.strip() for c in cats if c and c.strip()})

    fraud_hits = count_patterns(text + ' ' + ';'.join(cats), FRAUD_PATTERNS)
    warn_hits = count_patterns(text + ' ' + ';'.join(cats), WARN_PATTERNS)
    positive_hits = count_patterns(text + ' ' + ';'.join(cats), POSITIVE_PATTERNS)

    if source == 'moshelovka':
        cats = sorted(set(cats + ['мошенничество', 'blacklist']))
        negative_count = max(negative_count, 3)
        review_count = max(review_count, 1)
        confidence = confidence if confidence is not None else 0.95
        label_hint = 'BLOCK'
        evidence_type = 'blacklist'
    elif fraud_hits > 0:
        negative_count = max(negative_count, max(2, fraud_hits))
        review_count = max(review_count, negative_count + positive_count + neutral_count, 1)
        confidence = confidence if confidence is not None else 0.85
        label_hint = 'BLOCK'
    elif warn_hits > 0 or negative_count > positive_count:
        negative_count = max(negative_count, max(1, warn_hits))
        review_count = max(review_count, negative_count + positive_count + neutral_count, 1)
        confidence = confidence if confidence is not None else 0.65
        label_hint = 'WARN'
    elif positive_hits > 0 and positive_count >= 3:
        return None
    else:
        return None

    categories_str = ';'.join(cats)
    if not categories_str and label_hint == 'WARN':
        categories_str = 'нежелательный звонок'
    if not categories_str and label_hint == 'BLOCK':
        categories_str = 'мошенничество'

    # source_reliability: per-source baseline trust (0..1)
    reliability = source_reliability if source_reliability is not None else {
        'spravportal': 0.86, 'callfilter': 0.70, 'zvonili': 0.62,
        'moshelovka': 0.95, 'bloha': 0.75, 'getscam': 0.55,
        'znomer': 0.55,
    }.get(source, 0.5)

    row = {
        'normalized_number': number,
        'source': source,
        'negative_count': int(negative_count),
        'positive_count': int(positive_count),
        'neutral_count': int(neutral_count),
        'review_count': int(max(review_count, negative_count + positive_count + neutral_count)),
        'search_volume': view_count,
        'categories': categories_str,
        'last_review_at': detail_date,
        'first_seen_at': '',
        'source_confidence': f'{float(confidence or 0.5):.2f}',
        'source_reliability': f'{float(reliability):.2f}',
        'view_count': int(view_count),
        'related_count': int(related_count),
        'detail_date': detail_date,
        'page_title': page_title[:200],
        'url': url,
    }
    evidence = {
        'normalized_number': number,
        'source': source,
        'label_hint': label_hint,
        'evidence_type': evidence_type,
        'evidence_text': text[:500],
        'full_text': (full_text or text)[:2000],
        'page_title': page_title[:200],
        'negative_count': row['negative_count'],
        'positive_count': row['positive_count'],
        'neutral_count': row['neutral_count'],
        'review_count': row['review_count'],
        'view_count': int(view_count),
        'related_count': int(related_count),
        'categories': categories_str,
        'source_confidence': row['source_confidence'],
        'source_reliability': row['source_reliability'],
        'detail_date': detail_date,
        'fraud_hits': int(fraud_hits),
        'warn_hits': int(warn_hits),
        'url': url,
        'collected_at': now_iso(),
    }
    return row, evidence


def read_existing_keys(path: str) -> Set[Tuple[str, str, str]]:
    keys: Set[Tuple[str, str, str]] = set()
    if not os.path.exists(path):
        return keys
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            # Поддержка русских и английских заголовков
            n = row.get('normalized_number', '') or row.get('номер', '')
            s = row.get('source', '') or row.get('источник', '')
            u = row.get('url', '') or row.get('ссылка', '')
            keys.add((n, s, u))
    return keys


def append_dict_rows(path: str, fieldnames: Sequence[str], rows: Iterable[Dict]):
    rows = list(rows)
    if not rows:
        return
    ensure_dir(path)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    # Пишем с русскими заголовками
    ru_fieldnames = translate_headers(fieldnames, to_ru=True)
    ru_rows = [translate_row(r, to_ru=True) for r in rows]
    with open(path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ru_fieldnames, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        for row in ru_rows:
            writer.writerow(row)


def cache_path(cache_dir: str, url: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', url).strip('_')[:180]
    return os.path.join(cache_dir, f'{safe}.html')


def fetch_url(url: str, cache_dir: str, timeout: int, retries: int, use_cache: bool) -> Optional[str]:
    path = cache_path(cache_dir, url)
    if use_cache and os.path.exists(path):
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': USER_AGENT,
                'Accept-Language': 'ru,en;q=0.8',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                # Auto-decompress gzip sitemaps (e.g. znum.ru/sitemap-1.xml.gz).
                if url.endswith('.gz') or raw[:2] == b'\x1f\x8b':
                    try:
                        import gzip as _gz
                        raw = _gz.decompress(raw)
                    except OSError:
                        pass
                charset = resp.headers.get_content_charset() or 'utf-8'
                try:
                    text = raw.decode(charset, errors='replace')
                except LookupError:
                    text = raw.decode('utf-8', errors='replace')
                os.makedirs(cache_dir, exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(text)
                return text
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt >= retries:
                print(f'GET failed: {url} ({e})')
                return None
            time.sleep(1.0 + attempt * 1.5)
    return None


def source_from_url(url: str) -> str:
    host = urllib.parse.urlparse(url).netloc.lower()
    if 'spravportal.ru' in host:
        return 'spravportal'
    if 'callfilter.app' in host:
        return 'callfilter'
    if 'zvonili.com' in host:
        return 'zvonili'
    if 'moshelovka.onf.ru' in host:
        return 'moshelovka'
    if 'bloha.ru' in host:
        return 'bloha'
    if 'getscam.com' in host:
        return 'getscam'
    if 'znum.ru' in host:
        return 'znum'
    if 'prozvonok.ru' in host:
        return 'prozvonok'
    if 'netrubi.ru' in host:
        return 'netrubi'
    if 'zvonkoff.net' in host:
        return 'zvonkoff'
    if 'ktozvonil.net' in host:
        return 'ktozvonil'
    if 'znomer.ru' in host:
        return 'znomer'
    if 'phoneregion.ru' in host:
        return 'phoneregion'
    if 'callfilter.info' in host:
        return 'callfilter_info'
    if 'scamcall.ru' in host:
        return 'scamcall'
    if 'kto.zvonil.tel' in host or 'zvonili.tel' in host:
        return 'kto_zvonil_tel'
    if 'abonentik.ru' in host:
        return 'abonentik'
    if 'badcall.ru' in host:
        return 'badcall'
    return 'unknown'


def detail_url(source: str, number: str) -> Optional[str]:
    digits = phone_digits_for_path(number)
    if source == 'spravportal':
        return f'https://www.spravportal.ru/services/who-calls/num/{digits}'
    if source == 'callfilter':
        return f'https://callfilter.app/{digits}'
    if source == 'getscam':
        return f'https://getscam.com/{digits}'
    if source == 'zvonili':
        return f'https://zvonili.com/phone/{phone_digits_for_path(number, drop_country=True)}'
    if source == 'moshelovka':
        return f'https://moshelovka.onf.ru/blacklist/?search={urllib.parse.quote(number)}'
    if source == 'znum':
        return f'https://znum.ru/z-{phone_digits_for_path(number, drop_country=True)}'
    if source == 'prozvonok':
        return f'https://prozvonok.ru/nomer/{phone_digits_for_path(number, drop_country=True)}'
    if source == 'netrubi':
        return f'https://netrubi.ru/nomer/{phone_digits_for_path(number, drop_country=True)}'
    if source == 'zvonkoff':
        return f'https://zvonkoff.net/ru/number/{digits}'
    if source == 'ktozvonil':
        return f'https://ktozvonil.net/nomer/{digits}'
    if source == 'znomer':
        return f'https://znomer.ru/{digits}'
    if source == 'phoneregion':
        # phoneregion.ru detail URL uses 10 digits without country prefix.
        local = digits[1:] if digits.startswith('7') and len(digits) == 11 else digits
        return f'https://www.phoneregion.ru/number/{local}'
    if source == 'callfilter_info':
        # callfilter.info detail URL: /number/<11-digit, leading 7>.
        local = digits if digits.startswith('7') and len(digits) == 11 else (
            f'7{digits[-10:]}' if len(digits) >= 10 else digits
        )
        return f'https://callfilter.info/number/{local}'
    if source == 'scamcall':
        # scamcall.ru detail URL: /phone/<10-digit, NO country code>.
        local = digits[1:] if digits.startswith('7') and len(digits) == 11 else digits
        return f'https://scamcall.ru/phone/{local}'
    if source == 'kto_zvonil_tel':
        # kto.zvonil.tel detail URL: /+<11-digit, leading 7>.
        local = digits if digits.startswith('7') and len(digits) == 11 else (
            f'7{digits[-10:]}' if len(digits) >= 10 else digits
        )
        return f'https://kto.zvonil.tel/+{local}'
    if source == 'abonentik':
        # abonentik.ru detail URL: /nomer/<11-digit, leading 7>.
        local = digits if digits.startswith('7') and len(digits) == 11 else (
            f'7{digits[-10:]}' if len(digits) >= 10 else digits
        )
        return f'https://abonentik.ru/nomer/{local}'
    if source == 'badcall':
        # badcall.ru detail URL: /phones/<10-digit, NO country code>.
        local = digits[1:] if digits.startswith('7') and len(digits) == 11 else digits
        return f'https://badcall.ru/phones/{local}'
    return None


def extract_number_links(source: str, html: str) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    patterns: List[str] = []
    if source == 'spravportal':
        patterns = [r'/services/who-calls/num/(\d{10,12})']
    elif source == 'callfilter':
        patterns = [r'callfilter\.app/(\d{10,12})(?:["/?#])', r'href=["\']/(\d{10,12})(?:["/?#])']
    elif source == 'zvonili':
        patterns = [r'zvonili\.com/phone/(\d{9,12})(?:["/?#])', r'href=["\']/phone/(\d{9,12})(?:["/?#])']
    elif source == 'getscam':
        patterns = [r'getscam\.com/(\d{10,12})(?:["/?#])', r'href=["\']/(\d{10,12})(?:["/?#])']
    elif source == 'znum':
        patterns = [r'href=["\']/z-(\d{10,11})(?:["/?#])', r'znum\.ru/z-(\d{10,11})(?:["/?#])']
    elif source == 'prozvonok':
        patterns = [r'href=["\']/nomer/(\d{10,12})(?:["/?#])', r'prozvonok\.ru/nomer/(\d{10,12})(?:["/?#])']
    elif source == 'netrubi':
        patterns = [r'href=["\']/nomer/(\d{10,12})(?:["/?#])', r'netrubi\.ru/nomer/(\d{10,12})(?:["/?#])']
    elif source == 'zvonkoff':
        patterns = [r'zvonkoff\.net/ru/number/(\d{10,12})(?:["/?#])', r'href=["\']/ru/number/(\d{10,12})(?:["/?#])']
    elif source == 'ktozvonil':
        patterns = [r'ktozvonil\.net/nomer/(\d{10,12})(?:["/?#])', r'href=["\']/nomer/(\d{10,12})(?:["/?#])']
    elif source == 'znomer':
        patterns = [r'znomer\.ru/(\d{10,12})(?:["/?#])', r'href=["\']/(\d{10,12})(?:["/?#])']
    elif source == 'phoneregion':
        patterns = [r'phoneregion\.ru/number/(\d{10,12})', r'href=["\']/number/(\d{10,12})']
    elif source == 'callfilter_info':
        patterns = [
            r'callfilter\.info/number/(\d{10,12})(?:["/?#])',
            r'href=["\']/number/(\d{10,12})(?:["/?#])',
        ]
    elif source == 'scamcall':
        patterns = [
            r'scamcall\.ru/phone/(\d{9,11})(?:["/?#])',
            r'href=["\']/phone/(\d{9,11})(?:["/?#])',
        ]
    elif source == 'kto_zvonil_tel':
        # /+<11-digit> — the literal '+' character is part of the path.
        patterns = [
            r'kto\.zvonil\.tel/\+(\d{10,12})(?:["/?#])',
            r'href=["\']/\+(\d{10,12})(?:["/?#])',
        ]
    elif source == 'abonentik':
        patterns = [
            r'abonentik\.ru/nomer/(\d{10,12})(?:["/?#])',
            r'href=["\']/nomer/(\d{10,12})(?:["/?#])',
        ]
    elif source == 'badcall':
        # /phones/<10-digit, no country code> — handle both absolute and
        # relative hrefs. The 9-13 lower bound catches the rare 9-digit case
        # where the leading '8' is dropped on the canonical URL.
        patterns = [
            r'badcall\.ru/phones/(\d{9,12})(?:["/?#])',
            r'href=["\']/phones/(\d{9,12})(?:["/?#])',
        ]

    for pattern in patterns:
        for digits in re.findall(pattern, html, flags=re.I):
            number = normalize_path_digits(digits)
            if number and number not in seen:
                seen.add(number)
                result.append(number)

    for number in numbers_from_text(html_to_text(html)):
        if number not in seen:
            seen.add(number)
            result.append(number)
    return result


def extract_moshelovka_links(html: str) -> List[str]:
    links: List[str] = []
    seen: Set[str] = set()
    # Sub-sitemap links from sitemap index
    for loc in re.findall(r'<loc>(https://moshelovka\.onf\.ru/blacklist-sitemap\d*\.xml)</loc>', html, flags=re.I):
        if loc not in seen:
            seen.add(loc)
            links.append(loc)
    # Detail page links from href attributes
    for href in re.findall(r'href=["\']([^"\']+/blacklist/[^"\']*)["\']', html, flags=re.I):
        url = urllib.parse.urljoin('https://moshelovka.onf.ru/blacklist/', href)
        parsed = urllib.parse.urlparse(url)
        if parsed.path.rstrip('/') == '/blacklist':
            continue
        if url not in seen:
            seen.add(url)
            links.append(url)
    # Detail page links from <loc> in sub-sitemaps
    for loc in re.findall(r'<loc>(https://moshelovka\.onf\.ru/blacklist/[^<]+)</loc>', html, flags=re.I):
        if loc not in seen:
            seen.add(loc)
            links.append(loc)
    return links


def parse_spravportal(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    text = html_to_text(html)
    page_title = extract_page_title(html)
    rows: List[Tuple[Dict, Dict]] = []
    new_urls: List[str] = []

    for number in extract_number_links('spravportal', html):
        page_url = detail_url('spravportal', number)
        if page_url:
            new_urls.append(page_url)

    number_match = re.search(r'/num/(\d{10,12})', url)
    if number_match:
        number = normalize_number(number_match.group(1))
        section = extract_section(text, ['сводка по отзывам', 'отзывы по номеру', 'что пишут пользователи'], ['оставьте отзыв', 'телефонный код', 'информация по оператору']) or text
        categories = categories_from_text(section)
        review_count = review_count_from_text(text)
        views = view_count_from_text(text)
        detail_date = extract_date_from_text(section)
        related = len(extract_number_links('spravportal', html))
        if number:
            result = build_row(number, 'spravportal', url, section, categories=categories,
                               review_count=review_count, confidence=0.86,
                               full_text=text, page_title=page_title,
                               view_count=views, related_count=related,
                               detail_date=detail_date)
            if result:
                rows.append(result)

    return rows, new_urls


def parse_callfilter(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    text = html_to_text(html)
    page_title = extract_page_title(html)
    rows: List[Tuple[Dict, Dict]] = []
    new_urls: List[str] = []

    for number in extract_number_links('callfilter', html):
        page_url = detail_url('callfilter', number)
        if page_url:
            new_urls.append(page_url)

    path_digits = re.search(r'callfilter\.app/(\d{10,12})(?:$|[/?#])', url)
    if path_digits:
        number = normalize_number(path_digits.group(1))
        negative = max([safe_int(x) for x in re.findall(r'(\d+)\s*x\s*отрицатель', text, flags=re.I)] or [0])
        positive = max([safe_int(x) for x in re.findall(r'(\d+)\s*x\s*положитель', text, flags=re.I)] or [0])
        neutral = max([safe_int(x) for x in re.findall(r'(\d+)\s*x\s*нейтраль', text, flags=re.I)] or [0])
        section = extract_section(text, ['оценки', 'категории', 'отзывы'], ['похожие телефонные номера', 'добавить отзыв']) or text
        categories = categories_from_text(section)
        if negative > 0 and not categories:
            categories = ['нежелательный звонок']
        views = view_count_from_text(text)
        detail_date = extract_date_from_text(section)
        related = len(extract_number_links('callfilter', html))
        if number:
            result = build_row(number, 'callfilter', url, section, categories=categories,
                               negative_count=negative, positive_count=positive, neutral_count=neutral,
                               review_count=negative + positive + neutral, confidence=0.7,
                               full_text=text, page_title=page_title,
                               view_count=views, related_count=related,
                               detail_date=detail_date)
            if result:
                rows.append(result)

    return rows, new_urls


def parse_zvonili(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    text = html_to_text(html)
    page_title = extract_page_title(html)
    rows: List[Tuple[Dict, Dict]] = []
    new_urls: List[str] = []

    for number in extract_number_links('zvonili', html):
        page_url = detail_url('zvonili', number)
        if page_url:
            new_urls.append(page_url)

    path_digits = re.search(r'/phone/(\d{9,12})', url)
    if path_digits:
        raw = path_digits.group(1)
        number = normalize_number(raw if raw.startswith('7') or raw.startswith('8') else f'7{raw}')
        section = extract_section(text, ['отзывы по номеру'], ['похожие номера с отзывами', 'похожие номера', 'новые отзывы'])
        if section and number:
            categories = categories_from_text(section)
            review_count = review_count_from_text(text)
            views = view_count_from_text(text)
            detail_date = extract_date_from_text(section)
            related = len(extract_number_links('zvonili', html))
            result = build_row(number, 'zvonili', url, section, categories=categories,
                               review_count=review_count, confidence=0.62,
                               full_text=text, page_title=page_title,
                               view_count=views, related_count=related,
                               detail_date=detail_date)
            if result:
                rows.append(result)

    return rows, new_urls


def parse_moshelovka(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    text = html_to_text(html)
    page_title = extract_page_title(html)
    rows: List[Tuple[Dict, Dict]] = []
    new_urls = extract_moshelovka_links(html)

    # Sub-sitemaps and listing pages: only enqueue links, don't extract numbers
    is_sitemap = url.endswith('.xml') or 'sitemap' in url
    is_listing = url.rstrip('/').endswith('/blacklist') and not is_sitemap

    if is_sitemap or is_listing:
        return rows, new_urls

    # Detail page: extract numbers and category info
    for number in numbers_from_text(text + ' ' + url):
        context = text[:800]
        detail_date = extract_date_from_text(text)
        # Try to extract specific category from detail page
        cats = categories_from_text(text)
        if not cats:
            cats = ['мошенничество']
        result = build_row(number, 'moshelovka', url, context, categories=cats,
                           confidence=0.95, evidence_type='blacklist',
                           full_text=text, page_title=page_title,
                           detail_date=detail_date)
        if result:
            rows.append(result)

    return rows, new_urls


def parse_bloha(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    text = html_to_text(html)
    page_title = extract_page_title(html)
    rows: List[Tuple[Dict, Dict]] = []
    lines = html_to_lines(html)
    for line in lines:
        nums = numbers_from_text(line)
        if not nums:
            continue
        lower = line.lower()
        if any(word in lower for word in ['мошен', 'требуют деньги', 'обман', 'деньги']):
            categories = ['мошенничество']
            confidence = 0.8
        elif any(word in lower for word in ['спам', 'сбрасывают', 'звонят', 'реклама']):
            categories = ['спам', 'нежелательный звонок']
            confidence = 0.65
        else:
            categories = ['мошенничество']
            confidence = 0.72
        for number in nums:
            result = build_row(number, 'bloha', url, line, categories=categories,
                               confidence=confidence, evidence_type='blacklist_article',
                               full_text=text, page_title=page_title)
            if result:
                rows.append(result)
    return rows, []


def parse_getscam(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    text = html_to_text(html)
    page_title = extract_page_title(html)
    rows: List[Tuple[Dict, Dict]] = []
    new_urls: List[str] = []

    for number in extract_number_links('getscam', html):
        page_url = detail_url('getscam', number)
        if page_url:
            new_urls.append(page_url)

    path_digits = re.search(r'getscam\.com/(\d{10,12})(?:$|[/?#])', url)
    if path_digits:
        number = normalize_number(path_digits.group(1))
        section = extract_section(text, ['отзывы номера телефона', 'важная информация'], ['последние проверенные отзывы', 'о проекте']) or text
        views = view_count_from_text(text)
        detail_date = extract_date_from_text(section)
        related = len(extract_number_links('getscam', html))
        if number:
            result = build_row(number, 'getscam', url, section, confidence=0.55,
                               full_text=text, page_title=page_title,
                               view_count=views, related_count=related,
                               detail_date=detail_date)
            if result:
                rows.append(result)

    return rows, new_urls


def number_from_url(source: str, url: str) -> Optional[str]:
    path = urllib.parse.urlparse(url).path
    patterns = {
        'znum': r'/z-(\d{10,11})',
        'prozvonok': r'/nomer/(\d{10,12})',
        'netrubi': r'/nomer/(\d{10,12})',
        'zvonkoff': r'/ru/number/(\d{10,12})',
        'ktozvonil': r'/nomer/(\d{10,12})',
        'znomer': r'/(\d{10,12})',
        'phoneregion': r'/number/(\d{10,12})',
        'callfilter_info': r'/number/(\d{10,12})',
        'scamcall': r'/phone/(\d{9,11})',
        'kto_zvonil_tel': r'/\+(\d{10,12})',
        'abonentik': r'/nomer/(\d{10,12})',
        'badcall': r'/phones/(\d{9,12})',
    }
    pattern = patterns.get(source)
    if not pattern:
        return None
    match = re.search(pattern, path, flags=re.I)
    return normalize_path_digits(match.group(1)) if match else None


def parse_structured_review_site(
    source: str,
    url: str,
    html: str,
    block_classes: Sequence[str],
    confidence: float,
    reliability: float,
) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    text = html_to_text(html)
    page_title = extract_page_title(html)
    rows: List[Tuple[Dict, Dict]] = []
    new_urls: List[str] = []

    for number in extract_number_links(source, html):
        page_url = detail_url(source, number)
        if page_url:
            new_urls.append(page_url)

    number = number_from_url(source, url)
    blocks_text = review_blocks_text(html, block_classes)
    description = meta_description(html)
    section = blocks_text or description or text[:1200]

    if number and section:
        categories = categories_from_text(section)
        negative = count_patterns(section, FRAUD_PATTERNS) + count_patterns(section, WARN_PATTERNS)
        positive = count_patterns(section, POSITIVE_PATTERNS)
        review_count = review_count_from_text(text) or max(negative + positive, 1)
        if has_reputation_signal(section, categories):
            result = build_row(
                number,
                source,
                url,
                section[:1600],
                categories=categories,
                negative_count=negative,
                positive_count=positive,
                review_count=review_count,
                confidence=confidence,
                full_text=section[:2200],
                page_title=page_title,
                view_count=view_count_from_text(text),
                related_count=len(new_urls),
                detail_date=extract_date_from_text(section),
                source_reliability=reliability,
            )
            if result:
                rows.append(result)

    # Listing pages like /otzyvy or home pages: each review block can contain its own number.
    if not number:
        for class_name in block_classes:
            for block in class_blocks(html, class_name):
                block_text = clean_html_fragment(block)
                if not block_text:
                    continue
                block_categories = categories_from_text(block_text)
                block_hits = count_patterns(block_text, FRAUD_PATTERNS) + count_patterns(block_text, WARN_PATTERNS)
                if not has_reputation_signal(block_text, block_categories):
                    continue
                for block_number in numbers_from_text(block_text):
                    result = build_row(
                        block_number,
                        source,
                        url,
                        block_text[:1200],
                        categories=block_categories,
                        negative_count=max(block_hits, 1),
                        review_count=1,
                        confidence=confidence,
                        evidence_type='listing_review',
                        full_text=block_text[:1800],
                        page_title=page_title,
                        source_reliability=reliability,
                    )
                    if result:
                        rows.append(result)

    return rows, new_urls


def parse_znum(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    return parse_structured_review_site('znum', url, html, ['review_block', 'review_text'], 0.62, 0.58)


def parse_prozvonok(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    text = html_to_text(html)
    page_title = extract_page_title(html)
    rows: List[Tuple[Dict, Dict]] = []
    new_urls: List[str] = []

    for number in extract_number_links('prozvonok', html):
        page_url = detail_url('prozvonok', number)
        if page_url:
            new_urls.append(page_url)

    number = number_from_url('prozvonok', url)
    if number:
        type_match = re.search(r'Тип звонка:.*?<span[^>]*>(.*?)</span>', html, flags=re.I | re.S)
        call_type = clean_html_fragment(type_match.group(1)) if type_match else ''
        top_section = extract_section(text, ['Рейтинг номера:', 'Тип звонка:'], ['какой регион номера', 'Последние комментарии']) or meta_description(html)
        section = f'{top_section} {call_type}'.strip()
        categories = categories_from_text(section)
        hits = count_patterns(section, FRAUD_PATTERNS) + count_patterns(section, WARN_PATTERNS)
        if has_reputation_signal(section, categories):
            review_count = review_count_from_text(top_section) or 1
            views = view_count_from_text(top_section)
            result = build_row(
                number,
                'prozvonok',
                url,
                section,
                categories=categories,
                negative_count=max(hits, 1),
                review_count=review_count,
                confidence=0.70,
                full_text=section,
                page_title=page_title,
                view_count=views,
                related_count=len(new_urls),
                detail_date=extract_date_from_text(section),
                source_reliability=0.65,
            )
            if result:
                rows.append(result)

    # /otzyvy cards: number, call type, comment live in the card. On detail pages
    # similar-number blocks look like cards too, so don't treat them as evidence.
    if '/otzyvy' in urllib.parse.urlparse(url).path:
        for card in re.findall(r'<div class="card w-100 mb-3">.*?(?=<div class="card w-100 mb-3">|</main>|$)', html, flags=re.I | re.S):
            number_match = re.search(r'href=["\']/nomer/(\d{10,12})', card, flags=re.I)
            if not number_match:
                continue
            card_number = normalize_path_digits(number_match.group(1))
            if not card_number:
                continue
            card_text = clean_html_fragment(card)
            card_categories = categories_from_text(card_text)
            card_hits = count_patterns(card_text, FRAUD_PATTERNS) + count_patterns(card_text, WARN_PATTERNS)
            if not has_reputation_signal(card_text, card_categories):
                continue
            result = build_row(
                card_number,
                'prozvonok',
                url,
                card_text,
                categories=card_categories,
                negative_count=max(card_hits, 1),
                review_count=1,
                confidence=0.70,
                evidence_type='listing_review',
                full_text=card_text,
                page_title=page_title,
                source_reliability=0.65,
            )
            if result:
                rows.append(result)

    return rows, new_urls


def parse_netrubi(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    return parse_structured_review_site('netrubi', url, html, ['comment-item'], 0.66, 0.58)


def parse_zvonkoff(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    rows: List[Tuple[Dict, Dict]] = []
    new_urls: List[str] = []
    for number_link in extract_number_links('zvonkoff', html):
        page_url = detail_url('zvonkoff', number_link)
        if page_url:
            new_urls.append(page_url)

    number = number_from_url('zvonkoff', url)
    description = meta_description(html)
    tags = review_blocks_text(html, ['sectionInfo__tags'])
    current_section = f'{description} {tags}'.strip()
    current_categories = categories_from_text(current_section)
    current_hits = count_patterns(current_section, FRAUD_PATTERNS) + count_patterns(current_section, WARN_PATTERNS)
    if number and (
        has_reputation_signal(current_section, current_categories)
        or 'нежелательный' in current_section.lower()
        or html_class_attribute_contains(html, 'negative')
    ):
        result = build_row(
            number,
            'zvonkoff',
            url,
            current_section,
            categories=current_categories or ['нежелательный звонок'],
            negative_count=max(current_hits, 1),
            review_count=review_count_from_text(html_to_text(html)) or 1,
            confidence=0.68,
            full_text=current_section,
            page_title=extract_page_title(html),
            source_reliability=0.62,
        )
        if result:
            rows.append(result)

    # The "newReviews" feed on any page contains independent recent number reviews.
    for block in class_blocks(html, 'newReviews__item'):
        block_text = clean_html_fragment(block)
        block_categories = categories_from_text(block_text)
        block_hits = count_patterns(block_text, FRAUD_PATTERNS) + count_patterns(block_text, WARN_PATTERNS)
        if not has_reputation_signal(block_text, block_categories):
            continue
        for block_number in numbers_from_text(block_text):
            result = build_row(
                block_number,
                'zvonkoff',
                url,
                block_text,
                categories=block_categories,
                negative_count=max(block_hits, 1),
                review_count=1,
                confidence=0.68,
                evidence_type='recent_review',
                full_text=block_text,
                page_title=extract_page_title(html),
                source_reliability=0.62,
            )
            if result:
                rows.append(result)
    return rows, new_urls


def parse_ktozvonil(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    rows, new_urls = parse_structured_review_site('ktozvonil', url, html, ['post-comments', 'blog-comment'], 0.66, 0.58)
    if rows:
        return rows, new_urls

    number = number_from_url('ktozvonil', url)
    description = meta_description(html)
    title = extract_page_title(html)
    section = f'{title} {description}'
    section_categories = categories_from_text(section)
    section_hits = count_patterns(section, FRAUD_PATTERNS) + count_patterns(section, WARN_PATTERNS)
    if number and (
        has_reputation_signal(section, section_categories)
        or html_class_attribute_contains(html, 'label-danger')
    ):
        result = build_row(
            number,
            'ktozvonil',
            url,
            section,
            categories=section_categories,
            negative_count=max(section_hits, 1),
            review_count=review_count_from_text(title) or review_count_from_text(html_to_text(html)) or 1,
            confidence=0.66,
            full_text=section,
            page_title=title,
            source_reliability=0.58,
        )
        if result:
            rows.append(result)
    return rows, new_urls


def parse_znomer(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    return parse_structured_review_site('znomer', url, html, ['review', 'comment'], 0.55, 0.45)


def parse_phoneregion(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    return parse_structured_review_site('phoneregion', url, html, ['item-review', 'item-review-header', 'review', 'comment'], 0.62, 0.58)


# ---------------------------------------------------------------------------
# callfilter.info — distinct from callfilter.app, ~50k phone-page sitemap.
# Detail pages embed <div class="number-status-label status-{scam,spam,ad,…}">
# plus a list of <div class="user-comment"> blocks (author, date, "Тип звонка").
# ---------------------------------------------------------------------------

# Map status-suffix -> coarse category we already understand.
CALLFILTER_INFO_STATUS = {
    'scam': 'мошенничество',
    'spam': 'спам',
    'ad': 'реклама',
    'advertising': 'реклама',
    'collector': 'коллекторы',
    'survey': 'опрос',
    'hooligan': 'хулиганство',
    'unknown': '',
    'allow': '',
    'safe': '',
    'good': '',
}


def parse_callfilter_info(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    text = html_to_text(html)
    page_title = extract_page_title(html)
    rows: List[Tuple[Dict, Dict]] = []
    new_urls: List[str] = []

    for number in extract_number_links('callfilter_info', html):
        page_url = detail_url('callfilter_info', number)
        if page_url:
            new_urls.append(page_url)

    number = number_from_url('callfilter_info', url)
    if not number:
        return rows, new_urls

    status_match = re.search(
        r'<div[^>]*class=["\']number-status-label\s+status-([\w\-]+)["\'][^>]*>([^<]*)',
        html, flags=re.I,
    )
    status_key = ''
    status_label = ''
    if status_match:
        status_key = status_match.group(1).lower().strip()
        status_label = clean_html_fragment(status_match.group(2))

    info_match = re.search(
        r'<div[^>]*class=["\']result-info-list[^"\']*["\'][^>]*>(.*?)<div[^>]*id=["\']complaint-form["\']',
        html, flags=re.I | re.S,
    )
    info_section = clean_html_fragment(info_match.group(1)) if info_match else ''

    review_count = 0
    rc_match = re.search(r'Отзывов\s*:?\s*(\d+)', info_section or text, flags=re.I)
    if rc_match:
        review_count = safe_int(rc_match.group(1))

    views = 0
    v_match = re.search(r'Просмотров\s*:?\s*(\d+)', info_section or text, flags=re.I)
    if v_match:
        views = safe_int(v_match.group(1))

    comments = re.findall(
        r'<div[^>]*class=["\']user-comment["\'][^>]*>(.*?)</div>\s*</div>\s*</div>',
        html, flags=re.I | re.S,
    )
    comments_text = ' '.join(clean_html_fragment(c) for c in comments)

    section_parts = [p for p in (status_label, info_section, comments_text) if p]
    section = ' '.join(section_parts)[:2400] or text[:1600]

    categories = categories_from_text(section)
    mapped = CALLFILTER_INFO_STATUS.get(status_key, '')
    if mapped and mapped not in categories:
        categories.append(mapped)

    negative = count_patterns(section, FRAUD_PATTERNS) + count_patterns(section, WARN_PATTERNS)
    positive = count_patterns(section, POSITIVE_PATTERNS)

    is_scam_status = status_key in {'scam', 'spam', 'ad', 'advertising', 'collector', 'survey', 'hooligan'}
    has_signal = is_scam_status or has_reputation_signal(section, categories) or review_count > 0
    if not has_signal:
        return rows, new_urls

    confidence = 0.85 if status_key == 'scam' else 0.74 if is_scam_status else 0.65
    reliability = 0.78 if is_scam_status else 0.62
    evidence_type = 'blacklist' if status_key == 'scam' else 'review'

    result = build_row(
        number,
        'callfilter_info',
        url,
        section[:1600],
        categories=categories,
        negative_count=max(negative, len(comments) if is_scam_status else 0),
        positive_count=positive,
        review_count=review_count or len(comments),
        confidence=confidence,
        evidence_type=evidence_type,
        full_text=section[:2400],
        page_title=page_title,
        view_count=views,
        related_count=len(new_urls),
        detail_date=extract_date_from_text(section),
        source_reliability=reliability,
    )
    if result:
        rows.append(result)

    return rows, new_urls


# ---------------------------------------------------------------------------
# scamcall.ru — Vue/Nuxt SPA but the static HTML retains:
#   * commentary_item: 1 main comment with category, date, author, text
#   * sidebar swiper with 25-30 phone+verdict pairs (related complaints)
# Verdict classes: review_item_{scammers, advertising, dumb, dumb-call,
#   positively, unknown}.
# ---------------------------------------------------------------------------

SCAMCALL_VERDICT_MAP = {
    'scammers': ('мошенничество', 'BLOCK', 0.84, 0.78),
    'advertising': ('реклама', 'WARN', 0.72, 0.65),
    'dumb': ('хулиганство', 'WARN', 0.68, 0.60),
    'dumb-call': ('немой звонок', 'WARN', 0.66, 0.58),
    # 'positively' / 'unknown' deliberately omitted: a single carousel-card
    # vouching for a number is too weak a signal compared to our
    # legitimate_numbers.csv pipeline (orgpage / spravker / zoon),
    # and counting them here would otherwise inflate ALLOW rows with
    # questionable provenance.
}


def parse_scamcall(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    text = html_to_text(html)
    page_title = extract_page_title(html)
    rows: List[Tuple[Dict, Dict]] = []
    new_urls: List[str] = []

    for number in extract_number_links('scamcall', html):
        page_url = detail_url('scamcall', number)
        if page_url:
            new_urls.append(page_url)

    # Swiper-carousel pairs: (phone, verdict_key, label).
    # These appear on detail and listing pages alike, so always extract them.
    pair_re = re.compile(
        r'<a[^>]+href=["\']/phone/(\d{9,11})[^"\']*["\'][^>]+class=["\']otziv_tel["\'][^>]*>'
        r'[\d\s]+</a>\s*'
        r'<p[^>]+class=["\']review_item review_item_([\w\-]+)["\'][^>]*>([^<]+)</p>',
        flags=re.I | re.S,
    )
    person_desc_re = re.compile(
        r'<p[^>]+class=["\']otziv_person["\'][^>]*>([^<]+)</p>'
        r'\s*<p[^>]+class=["\']otziv_desc["\'][^>]*>([^<]+)</p>',
        flags=re.I | re.S,
    )

    # The swiper renders blocks of 4 elements per number: (tel, verdict, person,
    # desc). Slice the document into otziv_slide_item chunks and pair them.
    slides = re.split(r'<div class=["\']otziv_slide_item["\'][^>]*>', html)
    for chunk in slides[1:]:
        m = pair_re.search(chunk)
        if not m:
            continue
        digits, verdict_key, label = m.group(1), m.group(2).lower(), m.group(3)
        normalized = normalize_path_digits(digits)
        if not normalized:
            continue
        meta = SCAMCALL_VERDICT_MAP.get(verdict_key)
        if not meta:
            continue
        category, _verdict, confidence, reliability = meta
        if not category:
            continue  # Skip 'unknown' / unmapped — no real signal.

        person_match = person_desc_re.search(chunk)
        person = clean_html_fragment(person_match.group(1)) if person_match else ''
        desc = clean_html_fragment(person_match.group(2)) if person_match else ''
        section_parts = [label, category, person, desc]
        section = ' '.join(p for p in section_parts if p)[:1200] or category

        categories = categories_from_text(section)
        if category not in categories:
            categories.append(category)
        hits = count_patterns(section, FRAUD_PATTERNS) + count_patterns(section, WARN_PATTERNS)
        result = build_row(
            normalized,
            'scamcall',
            url,
            section[:800],
            categories=categories,
            negative_count=max(hits, 1),
            review_count=1,
            confidence=confidence,
            evidence_type='listing_review',
            full_text=section[:1500],
            page_title=page_title,
            source_reliability=reliability,
        )
        if result:
            rows.append(result)

    # Main commentary_item: parse the page's central comment (1 per detail page).
    main_number = number_from_url('scamcall', url)
    if main_number:
        commentary = re.findall(
            r'<div class=["\']commentary_item["\'][^>]*>(.*?)</div>\s*</div>\s*</div>',
            html, flags=re.I | re.S,
        )
        # Fallback: grab the chunk between the first commentary marker and the
        # next major section if the regex above misses (Vue SSR sometimes
        # nests divs unevenly).
        if not commentary:
            idx = html.find('class="commentary_item"')
            if idx >= 0:
                commentary = [html[idx:idx + 3000]]

        if commentary:
            body = clean_html_fragment(commentary[0])
            verdict_match = re.search(
                r'review_item review_item_([\w\-]+)["\'][^>]*>([^<]+)',
                commentary[0], flags=re.I,
            )
            verdict_key = verdict_match.group(1).lower() if verdict_match else 'unknown'
            meta = SCAMCALL_VERDICT_MAP.get(verdict_key)
            if meta and meta[0]:
                category, _verdict, confidence, reliability = meta
                categories = categories_from_text(body)
                if category not in categories:
                    categories.append(category)
                hits = count_patterns(body, FRAUD_PATTERNS) + count_patterns(body, WARN_PATTERNS)
                result = build_row(
                    main_number,
                    'scamcall',
                    url,
                    body[:1200],
                    categories=categories,
                    negative_count=max(hits, 1),
                    review_count=1,
                    confidence=confidence + 0.05,
                    full_text=body[:1800],
                    page_title=page_title,
                    detail_date=extract_date_from_text(body),
                    source_reliability=reliability,
                )
                if result:
                    rows.append(result)

    return rows, new_urls


# ---------------------------------------------------------------------------
# kto.zvonil.tel — Bootstrap-styled phone-review aggregator. Detail page lives
# at /+<11-digit, leading 7>. Each review is shaped like:
#
#   <div class="card p-2 mb-4">
#     <div class="card-body">
#       <div class="card-title">
#         <i …></i> <a href="/+79867257983">+79867257983</a>: от <b>Author</b>
#       </div>
#       <blockquote class="blockquote">
#         <free-form review text>
#         <category label / verdict on its own line>
#       </blockquote>
#       <div class="text-secondary">Добавлен: DD.MM.YYYY</div>
#     </div>
#   </div>
#
# The page also exposes a flat list of /+<phone> links under the labels
# "Новые отзывы:" and "Случайные отзывы:" — a cheap way to fan out into 10-30
# fresh detail URLs per crawl. Title encodes the total review count
# ("Всего N отзывов в базе"), which we use as a soft lower bound when no
# blockquotes match (e.g. cold pages with views-only metadata).
# ---------------------------------------------------------------------------

def parse_kto_zvonil_tel(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    text = html_to_text(html)
    page_title = extract_page_title(html)
    rows: List[Tuple[Dict, Dict]] = []
    new_urls: List[str] = []

    for number in extract_number_links('kto_zvonil_tel', html):
        page_url = detail_url('kto_zvonil_tel', number)
        if page_url:
            new_urls.append(page_url)

    number = number_from_url('kto_zvonil_tel', url)
    if not number:
        return rows, new_urls

    blockquotes = re.findall(
        r'<blockquote[^>]*class=["\']blockquote["\'][^>]*>(.*?)</blockquote>',
        html, flags=re.I | re.S,
    )
    review_text = ' '.join(clean_html_fragment(b) for b in blockquotes)

    # Title pattern: "Всего N отзывов в базе." — strict integer extraction.
    title_count = 0
    title_match = re.search(r'Всего\s+(\d+)\s+отзыв', page_title, flags=re.I)
    if title_match:
        title_count = safe_int(title_match.group(1))

    views = view_count_from_text(text)
    detail_date = extract_date_from_text(review_text or text)
    section_parts = [page_title, review_text]
    section = ' '.join(p for p in section_parts if p)[:2400] or text[:1600]

    categories = categories_from_text(section)
    has_signal = (
        title_count > 0
        or has_reputation_signal(section, categories)
        or bool(blockquotes)
    )
    if not has_signal:
        return rows, new_urls

    negative = count_patterns(section, FRAUD_PATTERNS) + count_patterns(section, WARN_PATTERNS)
    positive = count_patterns(section, POSITIVE_PATTERNS)
    review_count = max(title_count, len(blockquotes))

    fraud_signal = count_patterns(section, FRAUD_PATTERNS)
    confidence = 0.78 if fraud_signal > 0 else 0.66
    reliability = 0.62

    result = build_row(
        number,
        'kto_zvonil_tel',
        url,
        section[:1600],
        categories=categories,
        negative_count=max(negative, len(blockquotes) if blockquotes else 0),
        positive_count=positive,
        review_count=review_count,
        confidence=confidence,
        full_text=section[:2200],
        page_title=page_title,
        view_count=views,
        related_count=len(new_urls),
        detail_date=detail_date,
        source_reliability=reliability,
    )
    if result:
        rows.append(result)

    return rows, new_urls


# ---------------------------------------------------------------------------
# abonentik.ru — Vue/Nuxt SSR aggregator with a rich category vocabulary.
# Each phone page (/nomer/<11-digit>) advertises:
#   * /category/<slug> links (multi-tag: spam, moshennichestvo-telefonnoe,
#     fishing, kollektory, ugrozy, hooliganstvo, …) — the strongest signal.
#   * "Оценка номера: X.X из 5" rating (lower = worse).
#   * <article class="flex flex-col-reverse …"> review cards with author /
#     ISO date / body. Most pages have 0-3 reviews even when categories are
#     populated, so we treat categories as a first-class verdict source.
#   * Sitemap index → /sitemap_1.xml with ~32k phone URLs (very high
#     fan-out; fetched separately by parse_sitemap_generic).
# Some category slugs are intentionally NOT mapped (e.g. 'neizvestny',
# 'tishina') so we don't inflate WARN/BLOCK from no-information slugs.
# ---------------------------------------------------------------------------

ABONENTIK_CATEGORY_MAP = {
    # BLOCK (high-confidence fraud)
    'moshennichestvo-telefonnoe': ('телефонное мошенничество', 'BLOCK', 0.86),
    'fishing': ('фишинг', 'BLOCK', 0.86),
    'predoplata': ('мошенничество', 'BLOCK', 0.82),
    'vymogatelstvo': ('вымогательство', 'BLOCK', 0.84),
    'forex': ('мошенничество', 'BLOCK', 0.78),
    'crypto': ('мошенничество', 'BLOCK', 0.78),
    'ugrozy': ('угрозы', 'BLOCK', 0.80),
    'virusy': ('фишинг', 'BLOCK', 0.74),
    # WARN (annoyance / soft-spam categories)
    'spam': ('спам', 'WARN', 0.74),
    'reklama': ('реклама', 'WARN', 0.70),
    'avtoobzvon': ('автоответчик', 'WARN', 0.66),
    'socopros': ('опросы', 'WARN', 0.66),
    'kollektory': ('коллекторы', 'WARN', 0.72),
    'navazchivye': ('нежелательный звонок', 'WARN', 0.68),
    'hooliganstvo': ('хулиганство', 'WARN', 0.70),
    'podozritelny': ('нежелательный звонок', 'WARN', 0.62),
    'telemarketing': ('реклама', 'WARN', 0.64),
    'strakhovki': ('реклама', 'WARN', 0.62),
    # Intentionally NOT mapped: 'neizvestny', 'tishina', 'golos',
    # 'spravka' — these are weak/neutral categories that would dilute the
    # signal if treated as evidence.
}


def parse_abonentik(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    text = html_to_text(html)
    page_title = extract_page_title(html)
    rows: List[Tuple[Dict, Dict]] = []
    new_urls: List[str] = []

    for number in extract_number_links('abonentik', html):
        page_url = detail_url('abonentik', number)
        if page_url:
            new_urls.append(page_url)

    number = number_from_url('abonentik', url)
    if not number:
        return rows, new_urls

    # Category slug links: <a href="/category/<slug>">…</a>
    cat_slugs = re.findall(r'href=["\']/category/([\w-]+)["\']', html, flags=re.I)
    mapped = []
    best_confidence = 0.0
    has_block_cat = False
    for slug in cat_slugs:
        meta = ABONENTIK_CATEGORY_MAP.get(slug.lower())
        if meta:
            mapped.append(meta)
            if meta[1] == 'BLOCK':
                has_block_cat = True
            best_confidence = max(best_confidence, meta[2])

    # Rating (lower = worse).
    rating = 0.0
    rating_match = re.search(r'(\d(?:\.\d)?)\s*из\s*5', html)
    if rating_match:
        rating = float(rating_match.group(1))

    # Reviews — author + date + body.
    review_articles = re.findall(
        r'<article[^>]*class="[^"]*flex flex-col-reverse[^"]*"[^>]*>(.*?)</article>',
        html, flags=re.I | re.S,
    )
    review_text = ' '.join(clean_html_fragment(a) for a in review_articles)

    section_parts = [
        ' '.join(meta[0] for meta in mapped),
        review_text,
    ]
    section = ' '.join(p for p in section_parts if p)[:2400] or text[:1600]

    has_signal = bool(mapped) or has_reputation_signal(
        section, [meta[0] for meta in mapped]
    )
    if not has_signal:
        return rows, new_urls

    categories = sorted({meta[0] for meta in mapped}) or categories_from_text(section)
    negative = count_patterns(section, FRAUD_PATTERNS) + count_patterns(section, WARN_PATTERNS)
    positive = count_patterns(section, POSITIVE_PATTERNS)
    review_count = max(len(review_articles), 1)

    # Confidence: best category baseline, slightly boosted when rating ≤ 2.0
    # confirms a negative tag, and when ≥2 BLOCK-category slugs co-occur.
    confidence = best_confidence or 0.62
    if has_block_cat and rating and rating <= 2.0:
        confidence = min(0.92, confidence + 0.05)
    if sum(1 for meta in mapped if meta[1] == 'BLOCK') >= 2:
        confidence = min(0.92, confidence + 0.04)

    reliability = 0.74 if has_block_cat else 0.62

    result = build_row(
        number,
        'abonentik',
        url,
        section[:1600],
        categories=categories,
        negative_count=max(negative, len(mapped)),
        positive_count=positive,
        review_count=review_count,
        confidence=confidence,
        evidence_type='blacklist' if has_block_cat else 'review',
        full_text=section[:2200],
        page_title=page_title,
        related_count=len(new_urls),
        detail_date=extract_date_from_text(review_text or text),
        source_reliability=reliability,
    )
    if result:
        rows.append(result)

    return rows, new_urls


# ---------------------------------------------------------------------------
# badcall.ru — Bootstrap-themed aggregator where the verdict is encoded
# directly in the Bootstrap colour class on each <li>:
#   * list-group-item-danger  → negative (BLOCK / WARN signal)
#   * list-group-item-warning → neutral (WARN signal)
#   * list-group-item-success → positive (we IGNORE these — see ALLOW
#     pipeline in legitimate_numbers.csv)
# Detail URL: /phones/<10-digit, NO country code>. Pages with zero reviews
# render an "Неизвестный номер!" stub which we deliberately reject (no
# signal, no provenance) so they don't pollute the dataset.
# ---------------------------------------------------------------------------

def parse_badcall(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    text = html_to_text(html)
    page_title = extract_page_title(html)
    rows: List[Tuple[Dict, Dict]] = []
    new_urls: List[str] = []

    for number in extract_number_links('badcall', html):
        page_url = detail_url('badcall', number)
        if page_url:
            new_urls.append(page_url)

    number = number_from_url('badcall', url)
    if not number:
        return rows, new_urls

    danger_items = re.findall(
        r'<li[^>]*class=["\']list-group-item\s+list-group-item-danger["\'][^>]*>(.*?)</li>',
        html, flags=re.I | re.S,
    )
    warning_items = re.findall(
        r'<li[^>]*class=["\']list-group-item\s+list-group-item-warning["\'][^>]*>(.*?)</li>',
        html, flags=re.I | re.S,
    )

    # No negative/neutral reviews → reject the stub. We deliberately count
    # ONLY danger + warning items: every page renders a site-wide
    # "Неизвестный номер!" footer card, so checking for that string is
    # unreliable. The empty-review case is the absence of any list-group
    # items in our verdict-classes.
    if not (danger_items or warning_items):
        return rows, new_urls

    review_text = ' '.join(
        clean_html_fragment(item) for item in (danger_items + warning_items)
    )

    title_count = 0
    title_match = re.search(r'(\d+)\s+отзыв', page_title, flags=re.I)
    if title_match:
        title_count = safe_int(title_match.group(1))

    section = review_text[:2400] or text[:1600]
    categories = categories_from_text(section)

    fraud_signal = count_patterns(section, FRAUD_PATTERNS)
    if danger_items and not fraud_signal:
        # Bootstrap "danger" colour class is itself a negative tag — when
        # the reviewer flags a call as Negative (smiley 😡) but doesn't use
        # any FRAUD vocabulary, surface this as a soft WARN by injecting a
        # generic negative category.
        if 'нежелательный звонок' not in categories:
            categories.append('нежелательный звонок')

    negative = count_patterns(section, FRAUD_PATTERNS) + count_patterns(section, WARN_PATTERNS)
    positive = count_patterns(section, POSITIVE_PATTERNS)

    confidence = 0.78 if fraud_signal > 0 else 0.62
    reliability = 0.55

    review_count = max(title_count, len(danger_items) + len(warning_items))
    result = build_row(
        number,
        'badcall',
        url,
        section[:1600],
        categories=categories,
        negative_count=max(negative, len(danger_items)),
        positive_count=positive,
        review_count=review_count,
        confidence=confidence,
        full_text=section[:2200],
        page_title=page_title,
        related_count=len(new_urls),
        detail_date=extract_date_from_text(section),
        source_reliability=reliability,
    )
    if result:
        rows.append(result)

    return rows, new_urls


def looks_like_sitemap(url: str, html: str) -> bool:
    if html is None:
        return False
    head = html.lstrip()[:512].lower()
    if '<urlset' in head or '<sitemapindex' in head:
        return True
    if url.endswith('.xml') or url.endswith('.xml.gz'):
        return True
    if 'sitemap' in url.lower() and html.lstrip().startswith('<?xml'):
        return True
    return False


def parse_sitemap_generic(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    """Generic sitemap / sitemap-index parser.

    Reads <loc>…</loc> entries, returns them all as new URLs to enqueue.
    Limits to same-host links (so we don't accidentally follow 3rd-party
    sitemap entries). For listing pages that contain no extractable phone
    numbers — sitemaps don't yield rows directly, only candidate URLs.
    """
    host = urllib.parse.urlparse(url).netloc.lower()
    new_urls: List[str] = []
    seen: Set[str] = set()
    for loc in re.findall(r'<loc>\s*([^<\s]+)\s*</loc>', html, flags=re.I):
        if not loc.startswith('http'):
            continue
        loc_host = urllib.parse.urlparse(loc).netloc.lower()
        if loc_host and loc_host != host and not loc_host.endswith('.' + host) \
                and not host.endswith('.' + loc_host):
            continue
        if loc in seen:
            continue
        seen.add(loc)
        new_urls.append(loc)
    return [], new_urls


def is_antibot_challenge(html: str) -> bool:
    """Detect anti-bot interstitial pages so we don't waste bandwidth/queue on them.

    spravportal.ru serves a 2-3 KB ``Проверка браузера`` page to non-JS clients,
    which has no number signal but used to inflate our queue with derived URLs.
    """
    if not html:
        return False
    head = html[:4000].lower()
    if len(html) < 5000 and 'проверка браузера' in head:
        return True
    if 'cloudflare' in head and ('attention required' in head or 'just a moment' in head):
        return True
    return False


def parse_page(url: str, html: str) -> Tuple[List[Tuple[Dict, Dict]], List[str]]:
    if looks_like_sitemap(url, html):
        return parse_sitemap_generic(url, html)
    if is_antibot_challenge(html):
        return [], []
    source = source_from_url(url)
    if source == 'spravportal':
        return parse_spravportal(url, html)
    if source == 'callfilter':
        return parse_callfilter(url, html)
    if source == 'zvonili':
        return parse_zvonili(url, html)
    if source == 'moshelovka':
        return parse_moshelovka(url, html)
    if source == 'bloha':
        return parse_bloha(url, html)
    if source == 'getscam':
        return parse_getscam(url, html)
    if source == 'znum':
        return parse_znum(url, html)
    if source == 'prozvonok':
        return parse_prozvonok(url, html)
    if source == 'netrubi':
        return parse_netrubi(url, html)
    if source == 'zvonkoff':
        return parse_zvonkoff(url, html)
    if source == 'ktozvonil':
        return parse_ktozvonil(url, html)
    if source == 'znomer':
        return parse_znomer(url, html)
    if source == 'phoneregion':
        return parse_phoneregion(url, html)
    if source == 'callfilter_info':
        return parse_callfilter_info(url, html)
    if source == 'scamcall':
        return parse_scamcall(url, html)
    if source == 'kto_zvonil_tel':
        return parse_kto_zvonil_tel(url, html)
    if source == 'abonentik':
        return parse_abonentik(url, html)
    if source == 'badcall':
        return parse_badcall(url, html)
    return [], []


def load_state(path: str) -> Dict:
    if not os.path.exists(path):
        return {'queue': [], 'visited': [], 'saved': 0, 'started_at': now_iso()}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_state(path: str, state: Dict):
    ensure_dir(path)
    tmp = f'{path}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def add_unique_url(queue: List[str], queued: Set[str], visited: Set[str], url: Optional[str]):
    if not url:
        return
    if url in queued or url in visited:
        return
    queue.append(url)
    queued.add(url)


def initial_urls(sources: Set[str], seed_file: Optional[str]) -> List[str]:
    urls: List[str] = []
    for source in sources:
        urls.extend(SEED_URLS.get(source, []))
    if seed_file:
        with open(seed_file, 'r', encoding='utf-8') as f:
            for line in f:
                value = line.strip()
                if not value or value.startswith('#'):
                    continue
                if value.startswith('http://') or value.startswith('https://'):
                    urls.append(value)
                    continue
                number = normalize_number(value)
                if not number:
                    continue
                for source in sources:
                    url = detail_url(source, number)
                    if url:
                        urls.append(url)
    result: List[str] = []
    seen: Set[str] = set()
    for url in urls:
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def parse_sources(value: str) -> Set[str]:
    if value.strip().lower() == 'all':
        return set(SOURCES)
    result = {part.strip().lower() for part in value.split(',') if part.strip()}
    unknown = result - SOURCES
    if unknown:
        raise SystemExit(f'Unknown sources: {sorted(unknown)}')
    return result


def run(args) -> int:
    sources = parse_sources(args.sources)
    state = load_state(args.state)
    if args.reset_state:
        state = {'queue': [], 'visited': [], 'saved': 0, 'started_at': now_iso()}

    queue: List[str] = list(state.get('queue') or [])
    visited: Set[str] = set(state.get('visited') or [])
    queued: Set[str] = set(queue)
    for url in initial_urls(sources, args.seed_file):
        add_unique_url(queue, queued, visited, url)

    # If the persisted queue is small, top it up with synthetic enumeration URLs
    # so a long-running crawl-keepalive job never starves between sitemap ticks.
    if len(queue) < 200:
        for source in sorted(sources):
            for url in synthetic_enum_urls(source, count=300):
                add_unique_url(queue, queued, visited, url)

    existing = read_existing_keys(args.output)
    fetched = 0
    saved = safe_int(state.get('saved'), 0)
    # Счётчик последовательных ошибок по источнику
    source_errors: Dict[str, int] = {}
    SOURCE_ERROR_LIMIT = 5  # после 5 ошибок подряд — пропускать URL этого источника

    print(f'Sources: {", ".join(sorted(sources))}')
    print(f'Queue: {len(queue)} urls, visited: {len(visited)}, existing rows: {len(existing)}')
    print(f'Output: {args.output}')
    print(f'Evidence: {args.evidence}')
    print(f'Workers: {args.workers}')

    def current_state() -> Dict:
        return {
            'queue': queue,
            'visited': sorted(visited),
            'saved': saved,
            'started_at': state.get('started_at') or now_iso(),
            'updated_at': now_iso(),
        }

    def next_fetch_item() -> Optional[Tuple[int, str, str]]:
        nonlocal fetched
        while queue and fetched < args.max_urls:
            url = queue.pop(0)
            queued.discard(url)
            source = source_from_url(url)
            if source not in sources:
                continue
            if url in visited:
                continue
            if source_errors.get(source, 0) >= SOURCE_ERROR_LIMIT:
                continue
            visited.add(url)
            fetched += 1
            print(f'[{fetched}/{args.max_urls}] {source}: {url}')
            return fetched, url, source
        return None

    def fetch_with_delay(url: str) -> Optional[str]:
        html = fetch_url(url, args.cache_dir, args.timeout, args.retries, not args.no_cache)
        if args.delay_max > 0:
            time.sleep(random.uniform(args.delay_min, args.delay_max))
        return html

    def process_result(idx: int, url: str, source: str, html: Optional[str]):
        nonlocal saved
        if not html:
            source_errors[source] = source_errors.get(source, 0) + 1
            errs = source_errors[source]
            if errs >= SOURCE_ERROR_LIMIT:
                print(f'  ⚠ {source}: {errs} errors in a row, skipping remaining URLs for this source')
            return

        source_errors[source] = 0
        rows, new_urls = parse_page(url, html)
        raw_to_write: List[Dict] = []
        evidence_to_write: List[Dict] = []
        for row, evidence in rows:
            key = (row.get('normalized_number', ''), row.get('source', ''), row.get('url', ''))
            if key in existing:
                continue
            existing.add(key)
            raw_to_write.append(row)
            evidence_to_write.append(evidence)

        append_dict_rows(args.output, RAW_SCHEMA, raw_to_write)
        append_dict_rows(args.evidence, EVIDENCE_SCHEMA, evidence_to_write)
        saved += len(raw_to_write)
        if raw_to_write:
            hints = ', '.join(f'{r["normalized_number"]}:{r["categories"]}' for r in raw_to_write[:3])
            print(f'  saved={len(raw_to_write)} ({hints})')

        for new_url in new_urls:
            add_unique_url(queue, queued, visited, new_url)

        if idx % args.save_every == 0:
            save_state(args.state, current_state())
            print(f'  checkpoint: queue={len(queue)} visited={len(visited)} saved={saved}')

    if int(args.workers) > 1:
        import signal
        _stop = threading.Event()
        def _sigint(sig, frame):
            print('\n⚠ Interrupted — saving state...')
            _stop.set()
        old_handler = signal.signal(signal.SIGINT, _sigint)

        try:
            with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
                in_flight = {}
                while (queue or in_flight) and not _stop.is_set():
                    while len(in_flight) < int(args.workers) and not _stop.is_set():
                        item = next_fetch_item()
                        if not item:
                            break
                        idx, url, source = item
                        in_flight[executor.submit(fetch_with_delay, url)] = (idx, url, source)
                    if not in_flight:
                        break
                    # Wait for any future with timeout so we can check _stop
                    done_futures = []
                    for future in list(in_flight):
                        try:
                            future.result(timeout=0.5)
                            done_futures.append(future)
                        except FuturesTimeoutError:
                            pass
                        except Exception:
                            done_futures.append(future)
                        if _stop.is_set():
                            break
                    if not done_futures:
                        continue
                    for future in done_futures:
                        if future not in in_flight:
                            continue
                        idx, url, source = in_flight.pop(future)
                        try:
                            html = future.result(timeout=0)
                        except Exception as e:
                            print(f'GET failed: {url} ({e})')
                            html = None
                        if not _stop.is_set():
                            process_result(idx, url, source, html)
        finally:
            signal.signal(signal.SIGINT, old_handler)
            save_state(args.state, current_state())

        print(f'Done. Fetched={fetched}, saved_new_rows={saved}, remaining_queue={len(queue)}')
        return 0

    try:
        while queue and fetched < args.max_urls:
            url = queue.pop(0)
            queued.discard(url)
            source = source_from_url(url)
            if source not in sources:
                continue
            if url in visited:
                continue
            # Пропуск источника с слишком многими ошибками подряд
            if source_errors.get(source, 0) >= SOURCE_ERROR_LIMIT:
                continue
            visited.add(url)
            fetched += 1

            print(f'[{fetched}/{args.max_urls}] {source}: {url}')
            html = fetch_url(url, args.cache_dir, args.timeout, args.retries, not args.no_cache)
            if not html:
                source_errors[source] = source_errors.get(source, 0) + 1
                errs = source_errors[source]
                if errs >= SOURCE_ERROR_LIMIT:
                    print(f'  ⚠ {source}: {errs} ошибок подряд — пропускаю оставшиеся URL этого источника')
                continue
            # Успешный запрос — сброс счётчика ошибок
            source_errors[source] = 0

            rows, new_urls = parse_page(url, html)
            raw_to_write: List[Dict] = []
            evidence_to_write: List[Dict] = []
            for row, evidence in rows:
                key = (row.get('normalized_number', ''), row.get('source', ''), row.get('url', ''))
                if key in existing:
                    continue
                existing.add(key)
                raw_to_write.append(row)
                evidence_to_write.append(evidence)

            append_dict_rows(args.output, RAW_SCHEMA, raw_to_write)
            append_dict_rows(args.evidence, EVIDENCE_SCHEMA, evidence_to_write)
            saved += len(raw_to_write)
            if raw_to_write:
                hints = ', '.join(f'{r["normalized_number"]}:{r["categories"]}' for r in raw_to_write[:3])
                print(f'  saved={len(raw_to_write)} ({hints})')

            for new_url in new_urls:
                add_unique_url(queue, queued, visited, new_url)

            state = {'queue': queue, 'visited': sorted(visited), 'saved': saved, 'started_at': state.get('started_at') or now_iso(), 'updated_at': now_iso()}
            if fetched % args.save_every == 0:
                save_state(args.state, state)
                print(f'  checkpoint: queue={len(queue)} visited={len(visited)} saved={saved}')

            time.sleep(random.uniform(args.delay_min, args.delay_max))
    finally:
        state = {'queue': queue, 'visited': sorted(visited), 'saved': saved, 'started_at': state.get('started_at') or now_iso(), 'updated_at': now_iso()}
        save_state(args.state, state)

    print(f'Done. Fetched={fetched}, saved_new_rows={saved}, remaining_queue={len(queue)}')
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Multi-source reputation crawler for SpamBlocker datasets')
    parser.add_argument('--sources', default='all', help='Comma-separated sources or all')
    parser.add_argument('--seed-file', help='Optional file with seed numbers or URLs')
    parser.add_argument('--max-urls', type=int, default=2000, help='Max URLs to fetch in this run')
    parser.add_argument('--delay-min', type=float, default=1.0)
    parser.add_argument('--delay-max', type=float, default=3.0)
    parser.add_argument('--timeout', type=int, default=12)
    parser.add_argument('--retries', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8, help='Parallel fetch workers (1 keeps old sequential mode)')
    parser.add_argument('--save-every', type=int, default=25)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--evidence', default=DEFAULT_EVIDENCE)
    parser.add_argument('--state', default=DEFAULT_STATE)
    parser.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR)
    parser.add_argument('--reset-state', action='store_true')
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args(argv)
    if args.delay_max < args.delay_min:
        raise SystemExit('--delay-max must be >= --delay-min')
    if args.workers < 1:
        raise SystemExit('--workers must be >= 1')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())

# restart-trigger 2026-04-29T20:45 — keep-alive workflows died at 19:22, push to retrigger
