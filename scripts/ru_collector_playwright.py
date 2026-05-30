"""
Playwright-based collector for sources that block aiohttp / urllib.

Two modes:

* ``--mode block`` — uses headless Chromium to fetch phone-review pages on
  ``spravportal.ru`` and ``getscam.com`` (anti-bot protected). Each page
  is parsed with the existing :mod:`ru_reputation_crawler` parsers and
  written into ``datasets/ru/raw/shards/js/raw.csv`` (+ ``evidence.csv``)
  so the existing :mod:`merge_crawler_shards` pipeline picks it up
  automatically.

* ``--mode allow`` — drives ``yandex.ru/maps`` SSR pages to harvest
  business listings (name + phone) and append rows to
  ``datasets/ru/raw/legitimate_numbers.csv`` with
  ``source=yandex_maps`` / ``source_confidence=0.85``.

The script is intentionally a *single iteration*: it scrapes a small,
randomized batch and exits. The accompanying
``crawl-keepalive-js.yml`` workflow loops it 66 × ~5 min over 5.5 h to
match the existing keep-alive pattern.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote, urlparse

sys.path.insert(0, os.path.dirname(__file__))

try:
    from playwright.async_api import (
        Browser, BrowserContext, Page, TimeoutError as PWTimeout,
        async_playwright,
    )
except ImportError:
    print(
        'Playwright is not installed. Install with:\n'
        '    pip install playwright\n'
        '    python -m playwright install --with-deps chromium',
        file=sys.stderr,
    )
    sys.exit(0)  # exit clean so keepalive loop doesn't crash

from ru_collect_sources import RAW_SCHEMA
from ru_number_normalizer import is_russian_number, normalize_ru_phone
from ru_reputation_crawler import (
    EVIDENCE_SCHEMA,
    append_dict_rows,
    ensure_dir,
    initial_urls,
    parse_page,
    read_existing_keys,
    synthetic_enum_urls,
)


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(BASE_DIR, 'datasets', 'ru', 'raw')
JS_SHARD_DIR = os.path.join(RAW_DIR, 'shards', 'js')
JS_RAW_OUTPUT = os.path.join(JS_SHARD_DIR, 'raw.csv')
JS_EVIDENCE_OUTPUT = os.path.join(JS_SHARD_DIR, 'evidence.csv')
JS_STATE_FILE = os.path.join(JS_SHARD_DIR, 'state.json')

ALLOW_OUTPUT = os.path.join(RAW_DIR, 'legitimate_numbers.csv')

LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger('ru_collector_playwright')


# ── BLOCK side: Playwright over spravportal + getscam ──────────────────────

JS_BLOCK_SOURCES = {'spravportal', 'getscam'}


async def fetch_with_browser(
    page: Page, url: str, *, timeout_ms: int = 25_000,
    wait_selector: Optional[str] = None,
) -> Optional[str]:
    try:
        await page.goto(url, wait_until='domcontentloaded', timeout=timeout_ms)
        if wait_selector:
            try:
                await page.wait_for_selector(wait_selector, timeout=4000)
            except PWTimeout:
                pass
        # Settle a beat so any 'just a moment' challenge can resolve.
        await page.wait_for_timeout(random.randint(900, 2100))
        html = await page.content()
        return html
    except PWTimeout:
        log.warning(f'  timeout on {url}')
        return None
    except Exception as exc:  # pragma: no cover — wide net is intentional
        log.warning(f'  error on {url}: {exc}')
        return None


def _seed_urls_for(sources: Sequence[str], extra_count: int = 80) -> List[str]:
    sources_set = {s.strip() for s in sources if s.strip()}
    urls = list(initial_urls(sources_set, None))
    for src in sources_set:
        urls.extend(synthetic_enum_urls(src, count=extra_count))
    seen: Set[str] = set()
    deduped: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


async def run_block_mode(
    sources: Sequence[str],
    max_urls: int,
    workers: int,
    headful: bool,
) -> int:
    sources = [s for s in sources if s in JS_BLOCK_SOURCES]
    if not sources:
        log.warning('No JS-block sources selected — exiting')
        return 0

    ensure_dir(JS_RAW_OUTPUT)
    ensure_dir(JS_EVIDENCE_OUTPUT)

    existing_keys = read_existing_keys(JS_RAW_OUTPUT)
    seed = _seed_urls_for(sources, extra_count=120)
    random.shuffle(seed)
    seed = seed[:max_urls]
    log.info(f'block mode: sources={sources} url_budget={max_urls} seeds={len(seed)}')

    rows_buffer: List[Dict[str, object]] = []
    evidence_buffer: List[Dict[str, object]] = []
    fetched = 0

    async with async_playwright() as pw:
        browser: Browser = await pw.chromium.launch(headless=not headful)
        try:
            ctx: BrowserContext = await browser.new_context(
                locale='ru-RU',
                user_agent=(
                    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                    '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
                ),
                viewport={'width': 1366, 'height': 900},
            )
            page = await ctx.new_page()
            for url in seed:
                fetched += 1
                html = await fetch_with_browser(page, url, wait_selector='body')
                if not html:
                    continue
                rows, _new_urls = parse_page(url, html)
                for row, evidence in rows:
                    key = (
                        row.get('normalized_number', ''),
                        row.get('source', ''),
                        row.get('url', ''),
                    )
                    if key in existing_keys:
                        continue
                    existing_keys.add(key)
                    rows_buffer.append(row)
                    evidence_buffer.append(evidence)
                if fetched % 25 == 0:
                    log.info(f'  fetched={fetched} new_rows={len(rows_buffer)}')
        finally:
            await browser.close()

    if rows_buffer:
        append_dict_rows(JS_RAW_OUTPUT, RAW_SCHEMA, rows_buffer)
        append_dict_rows(JS_EVIDENCE_OUTPUT, EVIDENCE_SCHEMA, evidence_buffer)
        log.info(f'block mode: wrote {len(rows_buffer)} rows to {JS_RAW_OUTPUT}')
    else:
        log.info('block mode: no new rows this run')
    return len(rows_buffer)


# ── ALLOW side: Playwright over yandex.ru/maps ─────────────────────────────

YANDEX_MAPS_CITIES: List[Tuple[str, str]] = [
    # Tier 1: million+ population (kept first — highest yield per SERP)
    ('Москва', 'msk'),
    ('Санкт-Петербург', 'spb'),
    ('Екатеринбург', 'ekb'),
    ('Казань', 'kzn'),
    ('Новосибирск', 'nsk'),
    ('Нижний Новгород', 'nnov'),
    ('Ростов-на-Дону', 'rnd'),
    ('Уфа', 'ufa'),
    ('Краснодар', 'krd'),
    ('Самара', 'sam'),
    ('Челябинск', 'chl'),
    ('Воронеж', 'vrn'),
    ('Пермь', 'prm'),
    ('Волгоград', 'vlg'),
    ('Тюмень', 'tmn'),
    # Phase-1 ALLOW ×10 expansion: 35 more major regional centers.
    # Each pair is (Yandex search city name, slug). The slug is unused
    # in the SERP URL itself but kept for logging consistency.
    ('Омск', 'omk'),
    ('Красноярск', 'krk'),
    ('Владивосток', 'vvo'),
    ('Новокузнецк', 'nkz'),
    ('Тольятти', 'tlt'),
    ('Ижевск', 'izh'),
    ('Саратов', 'srt'),
    ('Сочи', 'soc'),
    ('Калининград', 'kgd'),
    ('Ярославль', 'yar'),
    ('Хабаровск', 'khb'),
    ('Иркутск', 'irk'),
    ('Томск', 'tmk'),
    ('Тула', 'tul'),
    ('Кемерово', 'kem'),
    ('Оренбург', 'orb'),
    ('Рязань', 'rzn'),
    ('Набережные Челны', 'nch'),
    ('Пенза', 'pnz'),
    ('Липецк', 'lpk'),
    ('Киров', 'kvr'),
    ('Ставрополь', 'stv'),
    ('Астрахань', 'ast'),
    ('Барнаул', 'brn'),
    ('Махачкала', 'mhk'),
    ('Белгород', 'blg'),
    ('Чебоксары', 'cbk'),
    ('Курск', 'kur'),
    ('Смоленск', 'sml'),
    ('Калуга', 'klg'),
    ('Тверь', 'tvr'),
    ('Архангельск', 'ahn'),
    ('Мурманск', 'mmk'),
    ('Владимир', 'vlm'),
    ('Брянск', 'brk'),
    ('Новороссийск', 'nvr'),
    ('Симферополь', 'sip'),
    ('Севастополь', 'svp'),
    ('Сургут', 'sgt'),
    ('Нижний Тагил', 'ntg'),
    ('Магнитогорск', 'mgn'),
    ('Петрозаводск', 'ptz'),
    ('Тамбов', 'tmb'),
    ('Иваново', 'ivn'),
    ('Саранск', 'srn'),
    ('Ульяновск', 'uln'),
]

YANDEX_MAPS_QUERIES = [
    # Food
    'кафе', 'ресторан', 'столовая', 'пиццерия', 'кофейня',
    'бар', 'суши', 'кондитерская', 'булочная',
    # Medical
    'аптека', 'клиника', 'стоматология', 'медицинский центр',
    'поликлиника', 'больница', 'роддом',
    'лаборатория', 'оптика',
    # Finance & legal
    'банк', 'страховая', 'нотариус', 'адвокат',
    'банкомат', 'юридические услуги', 'бухгалтерия',
    # Auto
    'автосервис', 'автомойка', 'шиномонтаж', 'автосалон',
    'АЗС', 'автозапчасти', 'эвакуатор',
    # Beauty / fitness
    'парикмахерская', 'салон красоты', 'фитнес',
    'массаж', 'бассейн', 'йога', 'барбершоп',
    # Retail
    'магазин продуктов', 'супермаркет',
    'торговый центр', 'цветы', 'электроника',
    # Education
    'детский сад', 'школа', 'университет',
    'колледж', 'языковые курсы', 'музыкальная школа',
    # Tourism / hospitality
    'гостиница', 'хостел', 'туристическое агентство',
    'музей', 'театр', 'кинотеатр',
    # Logistics / services
    'доставка еды', 'такси', 'грузоперевозки',
    'курьерская служба', 'почта', 'баня',
    # Home / repair
    'ремонт квартир', 'окна', 'мебель',
    'химчистка', 'ремонт техники', 'ремонт обуви',
    # Pets / vet
    'ветеринарная клиника', 'зоомагазин', 'груминг',
    # Government
    'МФЦ', 'налоговая', 'ПФР', 'суд', 'Почта России',
]

PHONE_TEXT_RE = re.compile(
    r'(?:\+?7|8)[\s\u00a0\-()]*\d{3}[\s\u00a0\-()]*\d{3}'
    r'[\s\u00a0\-()]*\d{2}[\s\u00a0\-()]*\d{2}'
)


def _read_legit_existing() -> Set[str]:
    seen: Set[str] = set()
    if not os.path.exists(ALLOW_OUTPUT):
        return seen
    with open(ALLOW_OUTPUT, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            n = (row.get('normalized_number') or '').strip()
            if n:
                seen.add(n)
    return seen


def _append_legit_rows(rows: List[Dict[str, object]]) -> int:
    if not rows:
        return 0
    ensure_dir(ALLOW_OUTPUT)
    fields = [
        'normalized_number', 'name', 'category', 'source',
        'city', 'url', 'source_confidence',
    ]
    file_exists = os.path.exists(ALLOW_OUTPUT) and os.path.getsize(ALLOW_OUTPUT) > 0
    with open(ALLOW_OUTPUT, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fields})
    return len(rows)


_ORG_LINK_RE = re.compile(r'/maps/org/([^/?#]+)/(\d+)/')


async def _serp_collect_org_links(page: Page, city: str, query: str) -> List[str]:
    """Open a SERP tab, scroll to load cards, return canonical org URLs.

    Replaces the legacy `__INITIAL_STATE__` walk: that JS global is no
    longer published by Yandex, so we pull the visible card hrefs and
    fetch each org's detail page in a second step.
    """
    url = f'https://yandex.ru/maps/?text={quote(f"{query} {city}")}'
    try:
        await page.goto(url, wait_until='domcontentloaded', timeout=30_000)
    except Exception as exc:
        log.warning(f'  serp goto fail {url}: {exc}')
        return []
    for _ in range(6):
        try:
            await page.mouse.wheel(0, 1500)
            await page.wait_for_timeout(350)
        except Exception:
            break
    try:
        hrefs = await page.evaluate(
            '''() => Array.from(document.querySelectorAll('a[href*="/maps/org/"]'))
                .map(a => a.getAttribute('href') || '').filter(h => h)'''
        )
    except Exception:
        return []
    seen_ids: Set[str] = set()
    out: List[str] = []
    for h in hrefs:
        if not isinstance(h, str):
            continue
        if not h.startswith('http'):
            h = 'https://yandex.ru' + h
        h = h.split('?')[0].split('#')[0]
        m = _ORG_LINK_RE.search(h)
        if not m:
            continue
        oid = m.group(2)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        out.append(f'https://yandex.ru/maps/org/{m.group(1)}/{m.group(2)}/')
    return out


async def _org_page_extract(
    ctx: BrowserContext, url: str, city: str, query: str,
) -> List[Dict[str, object]]:
    """Open an org detail page, wait for SPA hydration, return rows for
    every valid RU phone in the card panel.

    Only DOM-rendered phones are kept (XHR responses sometimes leak
    cross-promotion / "similar businesses" numbers — those are excluded).
    """
    rows: List[Dict[str, object]] = []
    page: Page = await ctx.new_page()
    try:
        try:
            await page.goto(url, wait_until='domcontentloaded', timeout=40_000)
        except Exception as exc:
            log.debug(f'  org goto fail {url}: {exc}')
            return rows
        try:
            await page.wait_for_function(
                '''() => {
                    const t = document.title || '';
                    return t && !/^Яндекс\\s*$/.test(t) && !/Найдётся всё/.test(t);
                }''',
                timeout=15_000,
            )
        except Exception:
            return rows
        await page.wait_for_timeout(1200)
        for sel in (
            'button:has-text("Показать номер")',
            'a:has-text("Показать")',
            '[aria-label*="телефон"]',
            '.card-phones-view__more',
        ):
            try:
                el = await page.query_selector(sel)
                if el:
                    await el.click()
                    await page.wait_for_timeout(250)
            except Exception:
                pass
        try:
            for _ in range(3):
                await page.mouse.wheel(0, 700)
                await page.wait_for_timeout(200)
        except Exception:
            pass
        try:
            dom_text = await page.evaluate('() => document.body.innerText || ""')
            name = await page.evaluate(
                '''() => {
                    const og = document.querySelector('meta[property="og:title"]');
                    if (og && og.content) return og.content;
                    const ip = document.querySelector('[itemprop="name"]');
                    if (ip && ip.textContent) return ip.textContent.trim();
                    return (document.title || '').replace(/ — Яндекс.*/, '').trim();
                }'''
            )
        except Exception:
            return rows
        # Skip pages that didn't actually hydrate with org data.
        if not name or 'Найдётся всё' in name or name.strip() == 'Яндекс':
            return rows
        phones: List[str] = []
        for m in PHONE_TEXT_RE.findall(dom_text or ''):
            n = normalize_ru_phone(m, reject_non_ru=True)
            if n and is_russian_number(n) and n not in phones:
                phones.append(n)
        # Trim org-card-name suffix once for readability.
        name = re.sub(r'\s*—\s*Яндекс[\s\u00a0]*Карты\s*$', '', name).strip()
        for ph in phones:
            rows.append({
                'normalized_number': ph,
                'name': name[:200],
                'category': 'business',
                'source': 'yandex_maps',
                'city': city,
                'url': url,
                'source_confidence': '0.85',
            })
        return rows
    finally:
        try:
            await page.close()
        except Exception:
            pass


async def _harvest_yandex_org_card(page: Page, base_url: str) -> List[Tuple[str, List[str]]]:
    """Scroll the SERP, then read business cards' name + phone via the
    on-page JSON state. Returns [(name, [phones])].
    """
    out: List[Tuple[str, List[str]]] = []
    try:
        for _ in range(8):
            await page.mouse.wheel(0, 1500)
            await page.wait_for_timeout(450)
    except Exception:  # pragma: no cover
        pass

    # Yandex Maps inlines a `window.__INITIAL_STATE__` blob on SSR pages
    # which is more reliable than DOM-scraping the virtualised list.
    try:
        state_json = await page.evaluate(
            'window.__INITIAL_STATE__ ? JSON.stringify(window.__INITIAL_STATE__) : ""'
        )
    except Exception:  # pragma: no cover
        state_json = ''

    if state_json:
        try:
            state = json.loads(state_json)
        except (TypeError, ValueError):
            state = None
        if state:
            for org in _walk_state_orgs(state):
                name = org.get('name') or org.get('title') or 'Yandex Maps'
                phones = [p for p in org.get('phones') or [] if p]
                if phones:
                    out.append((str(name)[:200], list(dict.fromkeys(phones))))

    # NOTE: previously there was a regex-over-rendered-HTML fallback here that
    # ran when `_walk_state_orgs(state)` returned no orgs.  In practice the
    # fallback fired on EVERY page (Yandex changed `__INITIAL_STATE__` schema
    # at some point) and harvested any phone-shaped digit sequence on the
    # page — Yandex footers, ad widgets, partner help-line stubs, etc.  This
    # poured ~55k garbage rows into legitimate_numbers.csv (#4).  Until a
    # proper extractor is wired up (Yandex Geo API or DOM-card click + parse),
    # we'd rather emit zero rows than poison the ALLOW dataset.
    return out


def _walk_state_orgs(state: object) -> List[Dict[str, object]]:
    """Walk the __INITIAL_STATE__ tree, collecting dicts that look like
    business cards (have a name + a phones array)."""
    found: List[Dict[str, object]] = []
    stack: List[object] = [state]
    seen_ids: Set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen_ids:
            continue
        seen_ids.add(id(node))
        if isinstance(node, dict):
            phones_raw = node.get('phones') or node.get('phone') or node.get('contactPhones')
            phones_norm: List[str] = []
            if isinstance(phones_raw, (list, tuple)):
                for entry in phones_raw:
                    if isinstance(entry, dict):
                        for k in ('formatted', 'number', 'value', 'text'):
                            v = entry.get(k)
                            if v:
                                norm = normalize_ru_phone(str(v), reject_non_ru=True)
                                if norm and is_russian_number(norm):
                                    phones_norm.append(norm)
                                    break
                    elif isinstance(entry, str):
                        norm = normalize_ru_phone(entry, reject_non_ru=True)
                        if norm and is_russian_number(norm):
                            phones_norm.append(norm)
            elif isinstance(phones_raw, str):
                norm = normalize_ru_phone(phones_raw, reject_non_ru=True)
                if norm and is_russian_number(norm):
                    phones_norm.append(norm)
            name = node.get('name') or node.get('title') or node.get('seoname')
            if phones_norm and isinstance(name, str):
                found.append({'name': name, 'phones': list(dict.fromkeys(phones_norm))})
            for v in node.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(node, list):
            for v in node:
                if isinstance(v, (dict, list)):
                    stack.append(v)
    return found


async def run_allow_mode(
    max_pairs: int,
    headful: bool,
    api_key: str = '',
    org_concurrency: int = 4,
) -> int:
    """Two-stage Yandex Maps harvest:

      1. Visit ``max_pairs`` SERP pages sequentially (single tab) and pull
         each result's canonical ``/maps/org/<slug>/<id>/`` URL.
      2. Fan the dedup'd org URLs out across ``org_concurrency``
         parallel tabs, each navigating to the org page and extracting
         (name, address-implicitly-from-name, phones) from the rendered
         DOM after SPA hydration.

    Old strategy (pre-#5) read ``window.__INITIAL_STATE__`` directly off
    the SERP — that global is no longer published by Yandex, so the old
    code fell into a regex-over-rendered-HTML fallback that pulled in
    Yandex's own footer/widget phones (#4).
    """
    pairs: List[Tuple[str, str]] = []
    cities = list(YANDEX_MAPS_CITIES)
    queries = list(YANDEX_MAPS_QUERIES)
    random.shuffle(cities)
    random.shuffle(queries)
    # Phase-1 ALLOW ×10: was cities[:6] × queries[:8] = 48 pairs cap.
    # We have 50 cities and 80 queries now, so widen to a much larger
    # pool and let max_pairs (workflow-controlled) gate actual SERP
    # work per iter. The shuffle keeps every iter sampling a different
    # slice instead of repeatedly hitting the same Moscow + cafe SERP.
    for city, _ in cities[:25]:
        for q in queries[:20]:
            pairs.append((city, q))
    random.shuffle(pairs)
    pairs = pairs[:max_pairs]
    log.info(f'allow mode: {len(pairs)} (city,query) SERP pairs')

    seen = _read_legit_existing()
    new_rows: List[Dict[str, object]] = []

    async with async_playwright() as pw:
        browser: Browser = await pw.chromium.launch(headless=not headful)
        try:
            ctx: BrowserContext = await browser.new_context(
                locale='ru-RU',
                user_agent=(
                    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                    '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
                ),
                viewport={'width': 1366, 'height': 900},
            )

            # Stage 1: SERP → list of (org_url, city, query).
            serp_page = await ctx.new_page()
            org_jobs: List[Tuple[str, str, str]] = []
            seen_org_urls: Set[str] = set()
            for city, query in pairs:
                links = await _serp_collect_org_links(serp_page, city, query)
                added = 0
                for link in links:
                    if link in seen_org_urls:
                        continue
                    seen_org_urls.add(link)
                    org_jobs.append((link, city, query))
                    added += 1
                log.info(f'  serp {city!r}/{query!r}: {added} new org links')
            try:
                await serp_page.close()
            except Exception:
                pass

            log.info(
                f'allow mode: {len(org_jobs)} unique org pages to fetch '
                f'(concurrency={org_concurrency})'
            )

            # Stage 2: process org pages in parallel.
            sem = asyncio.Semaphore(max(1, org_concurrency))

            async def worker(url: str, city: str, query: str) -> List[Dict[str, object]]:
                async with sem:
                    try:
                        return await _org_page_extract(ctx, url, city, query)
                    except Exception as exc:
                        log.warning(f'  org extract fail {url}: {exc}')
                        return []

            results = await asyncio.gather(
                *(worker(u, c, q) for (u, c, q) in org_jobs),
                return_exceptions=False,
            )
            for batch in results:
                for row in batch:
                    phone = str(row.get('normalized_number', ''))
                    if not phone or phone in seen:
                        continue
                    seen.add(phone)
                    new_rows.append(row)
        finally:
            await browser.close()

    written = _append_legit_rows(new_rows)
    log.info(f'allow mode: appended {written} new rows to {ALLOW_OUTPUT}')
    return written


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Playwright-based collector (block + allow modes).',
    )
    parser.add_argument('--mode', choices=['block', 'allow'], required=True)
    parser.add_argument('--sources', default='spravportal,getscam',
                        help='block-mode sources (comma-sep)')
    parser.add_argument('--max-urls', type=int, default=80,
                        help='block mode: number of phone-pages per iteration')
    parser.add_argument('--max-pairs', type=int, default=20,
                        help='allow mode: number of (city,query) SERP pairs')
    parser.add_argument('--org-concurrency', type=int, default=4,
                        help='allow mode: parallel org-page tabs')
    parser.add_argument('--workers', type=int, default=1,
                        help='reserved (Playwright run is single-threaded)')
    parser.add_argument('--headful', action='store_true',
                        help='show the browser window (debug)')
    return parser.parse_args(argv)


async def _amain(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == 'block':
        sources = [s.strip() for s in args.sources.split(',') if s.strip()]
        await run_block_mode(
            sources, max_urls=args.max_urls,
            workers=args.workers, headful=args.headful,
        )
    else:
        await run_allow_mode(
            max_pairs=args.max_pairs,
            headful=args.headful,
            org_concurrency=args.org_concurrency,
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return asyncio.run(_amain(argv))
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())

# restart-trigger 2026-04-29T20:45 — keep-alive workflows died at 19:22, push to retrigger
