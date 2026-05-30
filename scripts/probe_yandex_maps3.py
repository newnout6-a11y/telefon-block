#!/usr/bin/env python3
"""v3 probe: harvest org IDs from SERP, then visit org pages directly.

The v2 probe established that:
 - `window.__INITIAL_STATE__` no longer exists.
 - The SERP listing renders 5-10 cards per page but NEVER includes
   phone numbers in the DOM or in any XHR response — phones load only
   when the user clicks an individual card.

So the new flow is:
 1. Open SERP `https://yandex.ru/maps/?text={q}+{city}` → grab `<a>`
    elements pointing to `/maps/org/<id>/`.
 2. For each org URL, navigate the page and inspect both the rendered
    DOM and any JSON XHR responses for phone fields.

This is more requests per (city, query) pair, but produces VERIFIED
business data: real org name + real phone + real address.

Usage:
    python3 scripts/probe_yandex_maps3.py --pair "Москва,аптека" --max-orgs 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from playwright.async_api import async_playwright, Page, Response

from ru_number_normalizer import is_russian_number, normalize_ru_phone

PROBE_DIR = '/tmp/yandex_probe3'

PHONE_TEXT_RE = re.compile(
    r'(?:\+?7|8)[\s\u00a0\-()]*\d{3}[\s\u00a0\-()]*\d{3}'
    r'[\s\u00a0\-()]*\d{2}[\s\u00a0\-()]*\d{2}'
)


async def harvest_org_links(page: Page, city: str, query: str) -> List[str]:
    url = f'https://yandex.ru/maps/?text={query}+{city}'
    print(f'[serp] {url}')
    try:
        await page.goto(url, wait_until='domcontentloaded', timeout=30000)
    except Exception as exc:
        print(f'  [err] goto: {exc}')
        return []
    for _ in range(6):
        try:
            await page.mouse.wheel(0, 1500)
            await page.wait_for_timeout(400)
        except Exception:
            pass
    hrefs = await page.evaluate(
        '''() => Array.from(document.querySelectorAll('a[href*="/maps/org/"]'))
            .map(a => a.getAttribute('href') || '')
            .filter(h => h)'''
    )
    # dedupe by org id (canonical URL: /maps/org/<slug>/<id>/), drop sub-pages
    # like /gallery/, /reviews/, etc.
    org_re = re.compile(r'/maps/org/([^/?#]+)/(\d+)/')
    seen_ids: set = set()
    seen: List[str] = []
    for h in hrefs:
        if not h.startswith('http'):
            h = 'https://yandex.ru' + h
        h = h.split('?')[0].split('#')[0]
        m = org_re.search(h)
        if not m:
            continue
        oid = m.group(2)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        # Reduce to canonical /maps/org/<slug>/<id>/.
        canon = f'https://yandex.ru/maps/org/{m.group(1)}/{m.group(2)}/'
        seen.append(canon)
    print(f'  [serp] unique orgs: {len(seen)}')
    return seen


async def harvest_org_page(page: Page, url: str) -> Optional[Dict[str, Any]]:
    json_blobs: List[Dict[str, Any]] = []

    async def on_response(resp: Response) -> None:
        try:
            ct = (resp.headers or {}).get('content-type', '').lower()
            if 'json' not in ct:
                return
            ru = resp.url
            if 'yandex' not in ru:
                return
            try:
                body = await resp.text()
            except Exception:
                return
            if not body or len(body) > 800_000:
                return
            if 'phone' not in body.lower() and 'тел' not in body.lower() and not PHONE_TEXT_RE.search(body):
                return
            json_blobs.append({'url': ru, 'body': body})
        except Exception:
            pass

    page.on('response', lambda r: asyncio.create_task(on_response(r)))

    try:
        await page.goto(url, wait_until='domcontentloaded', timeout=40000)
    except Exception as exc:
        print(f'  [err] goto org: {exc}')
        return None
    # Wait for SPA hydration — the page first renders a generic Yandex
    # shell ("Яндекс / Найдётся всё") and then populates org content.
    try:
        await page.wait_for_function(
            '''() => {
                const t = document.title || '';
                return t && !/^Яндекс\\s*$/.test(t) && !/Найдётся всё/.test(t);
            }''',
            timeout=15000,
        )
    except Exception:
        pass
    await page.wait_for_timeout(1500)

    # Click "show phone" if such a button exists, then wait briefly.
    try:
        for sel in [
            'button:has-text("Показать номер")',
            'a:has-text("Показать")',
            '[aria-label*="телефон"]',
            '.card-phones-view__more',
        ]:
            try:
                el = await page.query_selector(sel)
                if el:
                    await el.click()
                    await page.wait_for_timeout(300)
            except Exception:
                pass
    except Exception:
        pass

    for _ in range(4):
        try:
            await page.mouse.wheel(0, 800)
            await page.wait_for_timeout(300)
        except Exception:
            pass

    # 1) Phone from DOM text.
    dom_text = await page.evaluate('() => document.body.innerText || ""')
    dom_phones: List[str] = []
    for m in PHONE_TEXT_RE.findall(dom_text):
        n = normalize_ru_phone(m, reject_non_ru=True)
        if n and is_russian_number(n) and n not in dom_phones:
            dom_phones.append(n)

    # 2) Name from <title> / og:title / itemprop.
    name = await page.evaluate(
        '''() => {
            const og = document.querySelector('meta[property="og:title"]');
            if (og && og.content) return og.content;
            const ip = document.querySelector('[itemprop="name"]');
            if (ip && ip.textContent) return ip.textContent.trim();
            return (document.title || '').replace(/ — Яндекс.*/, '').trim();
        }'''
    )

    # 3) Address.
    address = await page.evaluate(
        '''() => {
            const og = document.querySelector('meta[property="og:description"]');
            if (og && og.content) return og.content;
            const ip = document.querySelector('[itemprop="address"]');
            if (ip && ip.textContent) return ip.textContent.trim();
            return '';
        }'''
    )

    # 4) Phones from any XHR JSON that mentioned phones/тел.
    xhr_phones: List[str] = []
    for blob in json_blobs:
        for m in PHONE_TEXT_RE.findall(blob['body']):
            n = normalize_ru_phone(m, reject_non_ru=True)
            if n and is_russian_number(n) and n not in xhr_phones:
                xhr_phones.append(n)

    # Combine unique.
    all_phones = list(dict.fromkeys(dom_phones + xhr_phones))

    return {
        'url': url,
        'name': name,
        'address': address,
        'dom_phones': dom_phones,
        'xhr_phones': xhr_phones,
        'all_phones': all_phones,
        'xhr_count': len(json_blobs),
    }


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--pair', default='Москва,аптека')
    p.add_argument('--max-orgs', type=int, default=5)
    p.add_argument('--headful', action='store_true')
    args = p.parse_args()

    city, query = args.pair.split(',', 1)
    os.makedirs(PROBE_DIR, exist_ok=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not args.headful)
        ctx = await browser.new_context(
            locale='ru-RU',
            user_agent=(
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
            ),
            viewport={'width': 1366, 'height': 900},
        )
        page = await ctx.new_page()
        org_urls = await harvest_org_links(page, city, query)
        org_urls = org_urls[: args.max_orgs]

        org_page = await ctx.new_page()
        results: List[Dict[str, Any]] = []
        for i, url in enumerate(org_urls):
            print(f'\n[org {i+1}/{len(org_urls)}] {url[:100]}')
            r = await harvest_org_page(org_page, url)
            if r:
                print(f'  name={r["name"][:60]!r}')
                print(f'  addr={r["address"][:60]!r}')
                print(f'  dom_phones={r["dom_phones"]}')
                print(f'  xhr_phones={r["xhr_phones"]}')
                results.append(r)

        out_path = os.path.join(PROBE_DIR, f'{city[:6]}_{query[:6]}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'\n=== summary: {len(results)} orgs, '
              f'{sum(1 for r in results if r["all_phones"])} with phones, '
              f'saved to {out_path}')

        await browser.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
