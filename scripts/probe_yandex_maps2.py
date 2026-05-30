#!/usr/bin/env python3
"""Deeper probe: discover where Yandex Maps now stores org data.

`window.__INITIAL_STATE__` is gone, so the v1 probe found nothing.
This v2 probe instruments the page to inspect:

  1. All `window.__*__` globals (Yandex-style stash names).
  2. All `<script>` tags whose body looks like JSON.
  3. Outgoing network responses with content-type `application/json`
     (the SPA likely fetches `/api/...?text=...&...` and renders cards).
  4. DOM cards via attribute selectors (`[data-business-id]`, `.card-link`,
     `[itemtype*=Organization]`, etc.).

Usage:
    python3 scripts/probe_yandex_maps2.py --pair "Москва,кафе"
    python3 scripts/probe_yandex_maps2.py --pair "Санкт-Петербург,аптека"

Writes a self-contained debug bundle to `/tmp/yandex_probe2/` per pair so
the JSON dumps can be grepped offline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from playwright.async_api import async_playwright, Response


PROBE_DIR = '/tmp/yandex_probe2'

JSON_RESPONSE_TYPES = ('application/json', 'text/json', 'application/x-json')


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--pair', default='Москва,кафе', help='comma-separated "city,query"')
    p.add_argument('--headful', action='store_true')
    args = p.parse_args()

    city, query = args.pair.split(',', 1)
    out_dir = os.path.join(PROBE_DIR, f'{city[:6]}_{query[:6]}')
    os.makedirs(out_dir, exist_ok=True)

    captured_responses: List[Dict[str, Any]] = []

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

        async def on_response(resp: Response) -> None:
            try:
                ct = (resp.headers or {}).get('content-type', '').lower()
                if not any(t in ct for t in JSON_RESPONSE_TYPES):
                    return
                url = resp.url
                if 'yandex' not in url:
                    return
                try:
                    body = await resp.text()
                except Exception:
                    return
                if not body:
                    return
                # Save first 200KB only — keeps disk usage sane.
                trimmed = body[:200_000]
                idx = len(captured_responses)
                fp = os.path.join(out_dir, f'resp_{idx:03d}.json')
                with open(fp, 'w', encoding='utf-8') as f:
                    f.write(trimmed)
                captured_responses.append({
                    'idx': idx,
                    'url': url,
                    'content_type': ct,
                    'len_total': len(body),
                    'len_saved': len(trimmed),
                    'has_phone': 'phone' in body.lower() or 'тел' in body.lower(),
                    'file': fp,
                })
            except Exception as exc:
                print(f'  on_response error: {exc}')

        page.on('response', lambda r: asyncio.create_task(on_response(r)))

        url = f'https://yandex.ru/maps/?text={query}+{city}'
        print(f'probing: {url}')
        await page.goto(url, wait_until='domcontentloaded', timeout=30000)

        # Trigger card rendering by scrolling.
        for _ in range(8):
            try:
                await page.mouse.wheel(0, 1500)
                await page.wait_for_timeout(500)
            except Exception:
                pass

        # 1) Window globals.
        globals_info = await page.evaluate(
            '''() => {
                const out = {};
                const interesting = Object.getOwnPropertyNames(window)
                    .filter(k => /^(__|nk|backendData|REDUX|MAP|YANDEX)/i.test(k));
                for (const k of interesting) {
                    try {
                        const v = window[k];
                        out[k] = {
                            type: typeof v,
                            preview: v && typeof v === 'object'
                                ? JSON.stringify(v).slice(0, 400)
                                : String(v).slice(0, 200),
                        };
                    } catch (e) {
                        out[k] = { type: 'error', error: e.message };
                    }
                }
                return out;
            }'''
        )

        # 2) Script tags with JSON-ish content.
        scripts_info = await page.evaluate(
            '''() => {
                const out = [];
                for (const s of document.querySelectorAll('script')) {
                    const src = s.getAttribute('src') || '';
                    const type = s.getAttribute('type') || '';
                    const id = s.id || '';
                    const text = s.textContent || '';
                    if (!text || text.length < 200) continue;
                    out.push({
                        id, type, src,
                        len: text.length,
                        head: text.slice(0, 200),
                        looks_json: /^[\\s\\n]*[{\\[]/.test(text),
                    });
                }
                return out;
            }'''
        )

        # 3) DOM card-attribute hints.
        dom_info = await page.evaluate(
            '''() => {
                const sel = [
                    '[itemtype*="Organization"]',
                    '[data-business-id]',
                    '[data-coordinates]',
                    '.search-snippet-view',
                    '.search-business-snippet-view',
                    '.business-card-view',
                    '.card-info-view',
                    'a[href*="org/"]',
                    'a[href*="phone"]',
                ];
                const out = {};
                for (const s of sel) {
                    try {
                        out[s] = document.querySelectorAll(s).length;
                    } catch (e) { out[s] = -1; }
                }
                return out;
            }'''
        )

        # 4) Phone-shaped strings already in DOM.
        phones_in_dom = await page.evaluate(
            '''() => {
                const re = /(?:\\+?7|8)[\\s\\u00a0\\-()]*\\d{3}[\\s\\u00a0\\-()]*\\d{3}[\\s\\u00a0\\-()]*\\d{2}[\\s\\u00a0\\-()]*\\d{2}/g;
                const txt = document.body.innerText || '';
                const m = txt.match(re) || [];
                return { sample: m.slice(0, 20), total: m.length };
            }'''
        )

        bundle = {
            'url': url,
            'globals': globals_info,
            'scripts': scripts_info,
            'dom_selectors': dom_info,
            'phones_in_dom': phones_in_dom,
            'captured_responses': captured_responses,
        }
        with open(os.path.join(out_dir, 'bundle.json'), 'w', encoding='utf-8') as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)

        print(json.dumps({
            'globals_count': len(globals_info),
            'globals_keys': list(globals_info.keys()),
            'scripts_count': len(scripts_info),
            'json_responses_captured': len(captured_responses),
            'dom_card_counts': dom_info,
            'phones_in_dom_total': phones_in_dom.get('total'),
            'phones_in_dom_sample': phones_in_dom.get('sample', [])[:10],
            'bundle_dir': out_dir,
        }, ensure_ascii=False, indent=2))

        await browser.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
