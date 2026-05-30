#!/usr/bin/env python3
"""Dry-run probe for the Yandex Maps SERP collector.

Goal: figure out *where in `window.__INITIAL_STATE__`* the per-org name +
phones live now (since the previous walker stopped finding them, see
incident #4).

This script does NOT write to ``legitimate_numbers.csv``.  It opens
N (city, query) pairs through Playwright, dumps the state JSON to
``/tmp/yandex_state_<i>.json``, and prints to stdout:

  * total state size in bytes
  * top-level keys
  * for each candidate path that yields ≥1 (name, phones[]) pair, a
    summary count and 3 example rows
  * a final cumulative tally

Usage:

    python3 scripts/probe_yandex_maps.py --pairs 5
    python3 scripts/probe_yandex_maps.py --pairs 10 --headful   # debug

The output is meant to be human-reviewed; the next step is to bake the
winning extraction path into ``ru_collector_playwright.py`` proper.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from ru_number_normalizer import (  # noqa: E402
    is_russian_number,
    normalize_ru_phone,
)

try:
    from playwright.async_api import async_playwright
except ImportError:
    print('error: playwright not installed; run `pip install playwright && playwright install chromium`')
    sys.exit(2)


# --- inputs ------------------------------------------------------------------

PROBE_CITIES = [
    'Москва',
    'Санкт-Петербург',
    'Новосибирск',
    'Екатеринбург',
    'Казань',
]
PROBE_QUERIES = [
    'кафе',
    'аптека',
    'парикмахерская',
    'школа',
    'больница',
]


# --- candidate extraction strategies -----------------------------------------

PHONE_KEYS = ('phones', 'phone', 'contactPhones', 'phoneNumbers')
NAME_KEYS = ('name', 'title', 'seoname', 'displayName', 'shortName')


def _walk(state: Any, path: str = '') -> Iterable[Tuple[str, Dict]]:
    """Yield (path, dict) for every dict in state."""
    if isinstance(state, dict):
        yield path, state
        for k, v in state.items():
            yield from _walk(v, f'{path}.{k}' if path else k)
    elif isinstance(state, list):
        for i, v in enumerate(state):
            yield from _walk(v, f'{path}[{i}]')


def _extract_phones(node: Dict) -> List[str]:
    """Pull phone-shaped values from a candidate org dict, normalize, drop non-RU."""
    out: List[str] = []
    for k in PHONE_KEYS:
        v = node.get(k)
        if v is None:
            continue
        if isinstance(v, str):
            n = normalize_ru_phone(v, reject_non_ru=True)
            if n and is_russian_number(n):
                out.append(n)
        elif isinstance(v, list):
            for entry in v:
                if isinstance(entry, str):
                    n = normalize_ru_phone(entry, reject_non_ru=True)
                    if n and is_russian_number(n):
                        out.append(n)
                elif isinstance(entry, dict):
                    for kk in ('formatted', 'number', 'value', 'text', 'raw'):
                        vv = entry.get(kk)
                        if isinstance(vv, str):
                            n = normalize_ru_phone(vv, reject_non_ru=True)
                            if n and is_russian_number(n):
                                out.append(n)
                                break
    # dedupe preserving order
    return list(dict.fromkeys(out))


def _extract_name(node: Dict) -> Optional[str]:
    for k in NAME_KEYS:
        v = node.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _harvest_candidates(state: Any) -> List[Dict[str, Any]]:
    """Walk the entire state tree, return every dict that has BOTH a name
    AND ≥1 valid RU phone."""
    out: List[Dict[str, Any]] = []
    for path, node in _walk(state):
        if not isinstance(node, dict):
            continue
        phones = _extract_phones(node)
        if not phones:
            continue
        name = _extract_name(node)
        if not name:
            continue
        # Heuristic: skip Yandex-internal entries (e.g. their own help line).
        # Real org cards usually have at least one of these supporting fields:
        has_addr = any(k in node for k in ('address', 'fullAddress', 'addr', 'oid'))
        out.append({
            'path': path,
            'name': name,
            'phones': phones,
            'has_addr_hint': has_addr,
            'extra_keys': sorted(set(node.keys()) - set(PHONE_KEYS) - set(NAME_KEYS))[:8],
        })
    return out


# --- runner ------------------------------------------------------------------

async def probe_one(page, city: str, query: str, dump_path: str) -> Dict[str, Any]:
    url = f'https://yandex.ru/maps/?text={query}+{city}'
    print(f'\n--- probing {city!r} / {query!r}\n    {url}')
    try:
        await page.goto(url, wait_until='domcontentloaded', timeout=30000)
    except Exception as exc:
        return {'city': city, 'query': query, 'url': url, 'error': f'goto: {exc}'}

    # Give SPA a moment to populate __INITIAL_STATE__ + render cards.
    try:
        for _ in range(6):
            await page.mouse.wheel(0, 1500)
            await page.wait_for_timeout(400)
    except Exception:
        pass

    try:
        state_json = await page.evaluate(
            'window.__INITIAL_STATE__ ? JSON.stringify(window.__INITIAL_STATE__) : ""'
        )
    except Exception as exc:
        return {'city': city, 'query': query, 'url': url, 'error': f'eval: {exc}'}

    if not state_json:
        return {'city': city, 'query': query, 'url': url, 'state_size': 0, 'note': 'no __INITIAL_STATE__'}

    with open(dump_path, 'w', encoding='utf-8') as f:
        f.write(state_json)

    try:
        state = json.loads(state_json)
    except (TypeError, ValueError) as exc:
        return {'city': city, 'query': query, 'url': url, 'state_size': len(state_json), 'error': f'json: {exc}'}

    candidates = _harvest_candidates(state)
    top_keys = list(state.keys()) if isinstance(state, dict) else []

    # Group candidates by path-prefix (strip the [N] indexes) so we can spot
    # the ONE path Yandex now uses for orgs.
    grouped: Dict[str, int] = {}
    for c in candidates:
        key = re.sub(r'\[\d+\]', '[*]', c['path'])
        grouped[key] = grouped.get(key, 0) + 1

    return {
        'city': city,
        'query': query,
        'url': url,
        'state_size': len(state_json),
        'top_keys': top_keys[:20],
        'candidate_count': len(candidates),
        'paths_summary': sorted(grouped.items(), key=lambda kv: -kv[1])[:10],
        'samples': candidates[:5],
        'dump': dump_path,
    }


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', type=int, default=5)
    p.add_argument('--headful', action='store_true')
    args = p.parse_args()

    pairs: List[Tuple[str, str]] = []
    for c in PROBE_CITIES:
        for q in PROBE_QUERIES:
            pairs.append((c, q))
    pairs = pairs[: args.pairs]
    print(f'probing {len(pairs)} (city, query) pairs')

    os.makedirs('/tmp/yandex_probe', exist_ok=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not args.headful)
        try:
            ctx = await browser.new_context(
                locale='ru-RU',
                user_agent=(
                    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                    '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
                ),
                viewport={'width': 1366, 'height': 900},
            )
            page = await ctx.new_page()

            results: List[Dict[str, Any]] = []
            for i, (city, query) in enumerate(pairs):
                dump = f'/tmp/yandex_probe/state_{i}_{city[:6]}_{query[:6]}.json'
                r = await probe_one(page, city, query, dump)
                results.append(r)
                print(json.dumps({
                    k: v for k, v in r.items() if k != 'samples'
                }, ensure_ascii=False, indent=2))
                if r.get('samples'):
                    print('  samples:')
                    for s in r['samples'][:3]:
                        print(f'    path={s["path"]}')
                        print(f'      name={s["name"]!r}')
                        print(f'      phones={s["phones"]}')
                        print(f'      has_addr_hint={s["has_addr_hint"]}  extra_keys={s["extra_keys"]}')

            print('\n=== SUMMARY ===')
            total_orgs = sum(r.get('candidate_count', 0) for r in results)
            print(f'total candidates harvested: {total_orgs} across {len(results)} pages')
            agg_paths: Dict[str, int] = {}
            for r in results:
                for path, n in r.get('paths_summary', []):
                    agg_paths[path] = agg_paths.get(path, 0) + n
            print('top paths across all pages:')
            for path, n in sorted(agg_paths.items(), key=lambda kv: -kv[1])[:10]:
                print(f'  {n:5d}  {path}')

        finally:
            await browser.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
