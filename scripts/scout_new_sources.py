"""Скаут-скрипт для разведки новых источников реп-данных по РФ-номерам.

Используется в Phase 1 expand-sources плана: тестим кандидатов БЕЗ интеграции,
смотрим что вообще доступно/живо, какая семантика отзывов, нужен ли JS,
есть ли тех. барьеры (auth-wall, robots, Cloudflare).

Запуск:
    python3 scripts/scout_new_sources.py
    python3 scripts/scout_new_sources.py --json --out scout.md

Не коммит-сорсы — это разведка. Решение об интеграции принимается после
scout-отчёта на ревью.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import ssl
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

PHONE_RE = re.compile(r'(?:\+7|\b8)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-()]*\d{2}[\s\-()]*\d{2}')
TEL_LINK_RE = re.compile(r'tel:([0-9+\-()\s]+)')
COMPLAINT_KEYWORDS = re.compile(
    r'(мошенник|мошенничеств|спам|развод|телемаркет|коллектор|обман|фишинг|жалоб'
    r'|реклам|опрос|робозвон|неж\w*\s*звон|сбрасыва|поддельн)',
    re.I,
)
JS_HINTS = re.compile(
    r'(window\.__INITIAL_STATE__|__NEXT_DATA__|<script[^>]*src="[^"]*react|<noscript[^>]*>'
    r'|build/static/js|nuxt|<div[^>]*id="root"[^>]*></div>|<div[^>]*id="app"[^>]*></div>)',
    re.I,
)
CF_CHALLENGE = re.compile(r'cloudflare|cf-ray|attention required|just a moment', re.I)
USER_AGENT = (
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
)


@dataclass
class ProbeResult:
    name: str
    url: str
    http_status: Optional[int] = None
    html_length: int = 0
    phones_found: int = 0
    sample_phones: List[str] = field(default_factory=list)
    tel_links: int = 0
    complaint_keywords: int = 0
    js_required: bool = False
    cloudflare_blocked: bool = False
    robots_disallow: Optional[bool] = None
    error: Optional[str] = None
    elapsed_ms: int = 0
    viability: str = ''  # VIABLE / NEEDS_PLAYWRIGHT / NO_FRAUD_SIGNAL / CLOSED / ERROR
    priority: int = 0  # 1=high, 2=medium, 3=low, 0=skip
    notes: str = ''


# Кандидаты на интеграцию. Уже подключённые к ru_reputation_crawler.py НЕ включаем.
# Категория "personal" — отзывы по конкретным номерам; "directory" — каталоги
# организаций (источник ALLOW); "forum" — обсуждения, нужен поиск; "blacklist" —
# готовые чёрные списки.
CANDIDATES: List[Tuple[str, str, str]] = [
    # ── Top-priority: подтверждённо живые RU-агрегаторы отзывов ──
    ('who_calls_ru_root', 'https://who-calls.ru/', 'personal'),
    ('who_calls_ru_phone', 'https://who-calls.ru/number/74951234567', 'personal'),
    ('callfilter_info_root', 'https://callfilter.info/', 'personal'),
    ('callfilter_info_phone', 'https://callfilter.info/number/74951234567', 'personal'),
    ('scamcall_ru_root', 'https://scamcall.ru/', 'personal'),
    ('scamcall_ru_phone', 'https://scamcall.ru/phone/9584069694', 'personal'),
    ('vsezvonki_com', 'https://vsezvonki.com/', 'personal'),
    # ── Существующие в моих знаниях, но проверим живость ──
    ('tellows_ru_root', 'https://www.tellows.ru/', 'personal'),
    ('tellows_ru_phone', 'https://www.tellows.ru/num/74951234567', 'personal'),
    ('shouldianswer_ru', 'https://www.shouldianswer.com/ru/phone-number/', 'personal'),
    ('cleverdialer_ru', 'https://cleverdialer.ru/', 'personal'),
    # ── Сайты-кандидаты, ранее упомянутые в плане Phase 1 (могут быть мертвы) ──
    ('kto_zvonit_ua', 'https://kto-zvonit.com.ua/', 'personal'),
    ('phone_check_online', 'https://nomercheckonline.ru/', 'personal'),
    ('callsite_ru', 'https://callsite.ru/', 'personal'),
    ('kolokoltchik_ru', 'https://kolokoltchik.ru/', 'personal'),
    ('antikoll_ru', 'https://antikoll.ru/', 'blacklist'),
    ('ne_zvonite_ru', 'https://ne-zvonite.ru/', 'blacklist'),
    ('antispam_express', 'https://antispam.express/', 'blacklist'),
    ('antispam_club', 'https://anti-spam.club/', 'blacklist'),
    ('spam_num_ru', 'https://spam-num.ru/', 'blacklist'),
    ('nomer_org_ru', 'https://nomer.org.ru/', 'personal'),
    ('kinotele_ru', 'https://kinotele.ru/', 'personal'),
    # ── Forums / community Q&A ──
    ('yandex_q', 'https://yandex.ru/q/loves/phone/', 'forum'),
    ('pikabu_search', 'https://pikabu.ru/search?q=%D1%81%D0%BF%D0%B0%D0%BC+%D1%82%D0%B5%D0%BB%D0%B5%D1%84%D0%BE%D0%BD', 'forum'),
    ('otzovik_search', 'https://otzovik.com/category/0/?type=&order=&page=1&search=телефон', 'forum'),
    ('forum_4pda', 'https://4pda.to/forum/', 'forum'),
    ('mail_otvet', 'https://otvet.mail.ru/search?q=мошенник%20звонит', 'forum'),
    # ── Business directories (источник ALLOW) ──
    ('list_org', 'https://www.list-org.com/', 'directory'),
    ('rusprofile_search', 'https://www.rusprofile.ru/search?query=Сбербанк', 'directory'),
    ('cdek_complaints', 'https://comments.cdek.ru/', 'directory'),
    ('yell_ru', 'https://www.yell.ru/moscow/', 'directory'),
    # ── "Soft" sources: blocklists/warnings ──
    ('cbr_blacklist', 'https://www.cbr.ru/inside/warning-list/', 'blacklist'),
    ('sudact_search', 'https://sudact.ru/', 'directory'),
]


async def fetch(session, url: str, timeout: int = 12) -> Tuple[Optional[int], str]:
    import aiohttp
    headers = {'User-Agent': USER_AGENT, 'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.5'}
    try:
        async with session.get(url, headers=headers, ssl=False,
                               timeout=aiohttp.ClientTimeout(total=timeout),
                               allow_redirects=True) as r:
            text = await r.text(errors='replace')
            return r.status, text
    except asyncio.TimeoutError:
        return None, '__TIMEOUT__'
    except Exception as e:
        return None, f'__ERROR__:{type(e).__name__}:{e}'


async def fetch_robots(session, base_url: str) -> Optional[bool]:
    """Возвращает True если есть Disallow для всех (User-agent: *)."""
    parsed = urlparse(base_url)
    robots_url = f'{parsed.scheme}://{parsed.netloc}/robots.txt'
    status, text = await fetch(session, robots_url, timeout=6)
    if status != 200 or text.startswith('__'):
        return None
    in_star = False
    for line in text.splitlines():
        s = line.strip().lower()
        if s.startswith('user-agent:'):
            in_star = s.split(':', 1)[1].strip() == '*'
        elif in_star and s.startswith('disallow:'):
            path = s.split(':', 1)[1].strip()
            if path == '/':
                return True
    return False


def classify(p: ProbeResult, category: str) -> Tuple[str, int, str]:
    """Эвристически решаем: годен / нужен JS / нет fraud-семантики / закрыт."""
    if p.error and p.http_status is None:
        return 'CLOSED', 0, p.error
    if p.http_status is not None and p.http_status >= 400:
        return 'CLOSED', 0, f'HTTP {p.http_status}'
    if p.cloudflare_blocked:
        return 'CLOSED', 0, 'Cloudflare challenge'
    if p.robots_disallow:
        return 'CLOSED', 0, 'robots.txt Disallow: /'

    has_phones = p.phones_found >= 1 or p.tel_links >= 1
    has_complaints = p.complaint_keywords >= 2
    has_html_body = p.html_length >= 1500

    if not has_html_body and p.js_required:
        # SPA, статический HTML пустой — без Playwright не достать
        return 'NEEDS_PLAYWRIGHT', 2, 'SPA/JS-rendering'
    if not has_html_body:
        return 'CLOSED', 0, f'тело страницы пустое ({p.html_length} байт)'

    if category == 'directory':
        # Каталоги — источник ALLOW. Главное — телефоны и валидный HTML.
        if has_phones:
            return 'VIABLE', 1, 'каталог орг-номеров (ALLOW source)'
        return 'NO_FRAUD_SIGNAL', 3, 'нет телефонов на главной'

    if category in ('personal', 'blacklist'):
        if has_phones and has_complaints:
            return 'VIABLE', 1, 'есть номера + жалобная семантика'
        if has_phones:
            return 'VIABLE', 2, 'есть номера, но мало жалоб-семантики (надо проверить detail-страницы)'
        if has_complaints:
            return 'NEEDS_PLAYWRIGHT', 2, 'есть жалоб-текст, но телефонов нет (вероятно, поиск по номеру)'
        return 'NO_FRAUD_SIGNAL', 3, 'ни телефонов, ни жалоб'

    if category == 'forum':
        if has_complaints:
            return 'VIABLE', 2, 'форум с жалобной семантикой — нужен поиск/индекс'
        return 'NO_FRAUD_SIGNAL', 3, 'нет жалобных ключей'

    return 'UNKNOWN', 3, ''


async def probe_one(session, name: str, url: str, category: str) -> ProbeResult:
    p = ProbeResult(name=name, url=url)
    t0 = time.time()
    status, html = await fetch(session, url)
    p.elapsed_ms = int((time.time() - t0) * 1000)
    if html.startswith('__TIMEOUT__'):
        p.error = 'timeout'
    elif html.startswith('__ERROR__'):
        p.error = html.replace('__ERROR__:', '').strip()
    p.http_status = status
    if html and not html.startswith('__'):
        p.html_length = len(html)
        phones = PHONE_RE.findall(html)
        p.phones_found = len(set(phones))
        p.sample_phones = list(dict.fromkeys(phones))[:5]
        p.tel_links = len(set(TEL_LINK_RE.findall(html)))
        p.complaint_keywords = len(COMPLAINT_KEYWORDS.findall(html))
        p.js_required = bool(JS_HINTS.search(html)) and p.html_length < 5000
        p.cloudflare_blocked = bool(CF_CHALLENGE.search(html)) and p.html_length < 5000
    p.robots_disallow = await fetch_robots(session, url)
    viability, priority, notes = classify(p, category)
    p.viability = viability
    p.priority = priority
    p.notes = notes
    return p


async def main_async(out_path: Optional[str], emit_json: bool) -> int:
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=15)
    connector = aiohttp.TCPConnector(ssl=ssl.create_default_context(), limit=4)
    results: List[ProbeResult] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        sem = asyncio.Semaphore(4)

        async def worker(name: str, url: str, category: str):
            async with sem:
                r = await probe_one(session, name, url, category)
                print(
                    f'[{r.viability:18s} prio={r.priority}] {r.name:24s} '
                    f'http={r.http_status} html={r.html_length:>6} '
                    f'phones={r.phones_found:>3} compl={r.complaint_keywords:>3} '
                    f'{r.notes}'
                )
                results.append(r)

        await asyncio.gather(*[worker(n, u, c) for n, u, c in CANDIDATES])

    results.sort(key=lambda x: (x.priority if x.priority else 99, x.name))
    if emit_json:
        print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
    if out_path:
        write_markdown(out_path, results)
        print(f'\nReport written to {out_path}')
    return 0


def write_markdown(path: str, results: List[ProbeResult]) -> None:
    lines = [
        '# Scout report: кандидаты на интеграцию reputation-источников',
        '',
        f'_Сгенерировано {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())} '
        f'через `scripts/scout_new_sources.py`._',
        '',
        'Цель — посмотреть какие новые сайты живы, отдают HTML с телефонами и '
        'жалобной семантикой, прежде чем вкладываться в полноценные парсеры. '
        'Уже подключённые 13 источников (`spravportal`, `callfilter`, `zvonili`, '
        '`moshelovka`, `bloha`, `getscam`, `znum`, `prozvonok`, `netrubi`, '
        '`zvonkoff`, `ktozvonil`, `znomer`, `phoneregion`) в скоп не входят.',
        '',
        '## Сводка',
        '',
        '| Source | URL | HTTP | HTML, B | Phones | Compl | JS? | CF? | Robots | Viability | Prio | Notes |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for r in results:
        robots = '?' if r.robots_disallow is None else ('block' if r.robots_disallow else 'OK')
        lines.append(
            f'| `{r.name}` | <{r.url}> | {r.http_status or "—"} | {r.html_length} | '
            f'{r.phones_found} | {r.complaint_keywords} | {"yes" if r.js_required else "no"} | '
            f'{"yes" if r.cloudflare_blocked else "no"} | {robots} | '
            f'**{r.viability}** | {r.priority or "—"} | {r.notes} |'
        )
    lines.extend([
        '',
        '## Виды viability',
        '',
        '* **VIABLE** — статический HTML, есть номера + либо жалобная семантика, либо это каталог организаций (для ALLOW). Готов к написанию парсера.',
        '* **NEEDS_PLAYWRIGHT** — контент рендерится JS, требует браузерного шарда (`crawl-keepalive-js`).',
        '* **NO_FRAUD_SIGNAL** — главная страница не содержит ни телефонов, ни жалоб; вероятно, поиск только по конкретному номеру.',
        '* **CLOSED** — HTTP 4xx/5xx, Cloudflare-challenge, robots.txt полностью запрещает или таймаут.',
        '',
        '## Приоритеты',
        '',
        '* **prio=1** — берём первыми в Phase 2.',
        '* **prio=2** — берём вторым раундом, после 1.',
        '* **prio=3** — на потом / watch-mode.',
        '* **prio=0** — пропускаем.',
        '',
    ])
    sample_section = ['## Примеры найденных номеров (top-3)\n']
    for r in results:
        if r.sample_phones:
            sample_section.append(f'* `{r.name}`: ' + ', '.join(f'`{p}`' for p in r.sample_phones[:3]))
    sample_section.append('')
    lines.extend(sample_section)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', help='Markdown-файл для отчёта.')
    ap.add_argument('--json', action='store_true', help='Распечатать JSON-сводку в stdout.')
    args = ap.parse_args()
    return asyncio.run(main_async(args.out, args.json))


if __name__ == '__main__':
    sys.exit(main())
