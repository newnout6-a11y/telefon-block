"""VK Service Token collector for cold-start ALLOW eval candidates.

Что собираем
============
Публичные посты пользователей в RU-классифайд-группах ВКонтакте + комменты к
ним. Из текстов вытаскиваем телефонные номера, нормализуем, дедупим, фильтруем
по существующему blacklist-у (``ru_reputation_raw.csv``) — и пишем в отдельный
``datasets/ru/eval/vk_candidates.csv`` с дефолтным ``expected_label=ALLOW``.

Дальше эти кандидаты прогоняются через :mod:`spam_predict` в режиме
``--from-csv``, чтобы понять, где cold-start модель не согласна с дефолтным
ALLOW. Те, где модель говорит BLOCK на «нормального» юзера, — кандидаты на
fine-tune через :mod:`online_fine_tune`.

Источники (через VK API service-token)
======================================
* ``wall.search`` с query=``+7`` по seed-списку публичных групп — это самый
  плотный источник: VK сам предварительно отфильтровывает посты, где есть
  телефонная подстрока ``+7``.
* ``wall.get`` по тем же группам с пагинацией — для широкого обхода (если
  нужно больше, чем выдаёт wall.search).
* ``wall.getComments`` на найденные посты — комменты, где люди дублируют
  свой телефон / просят перезвонить.

Что **не работает** с service-token (проверено эмпирически на live API):

* ``market.search`` / ``market.get`` → ``error 28: method is unavailable
  with service token`` — VK закрыл marketplace для service-токенов.
* ``users.get`` с полями ``mobile_phone`` / ``home_phone`` / ``contacts``
  — поля молча выпиливаются из ответа.
* ``groups.search`` → ``Access denied``.

Поэтому источники сужены до wall.* — этого достаточно для cold-start eval.

Токен
=====
Создаётся за минуту на https://vk.com/apps?act=manage как «Standalone-
приложение». Без user-логина, без SMS, без 2FA. Кладём в env как
``VK_SERVICE_TOKEN`` (или legacy ``vk``). В код не попадает.

Usage
=====
.. code-block:: shell

    python scripts/vk_token_collector.py \
        --output datasets/ru/eval/vk_candidates.csv \
        --max-calls 200 --max-groups 20 --concurrency 3

    # Smoke run (мало вызовов, быстро):
    python scripts/vk_token_collector.py --max-calls 20 --max-groups 3

    # Затем прогон через модель:
    python scripts/spam_predict.py --from-csv datasets/ru/eval/vk_candidates.csv \
        --out-csv datasets/ru/eval/vk_verdicts.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as _dt
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote, urlencode

sys.path.insert(0, os.path.dirname(__file__))
from ru_legitimate_collector import extract_phones, is_plausible_phone  # noqa: E402

VK_API_VERSION = '5.131'
VK_API_BASE = 'https://api.vk.com/method'
DEFAULT_RATE_LIMIT_PER_SEC = 3.0  # VK docs say ≤3 req/sec for service tokens
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__), '..', 'datasets', 'ru', 'eval', 'vk_candidates.csv'
)
DEFAULT_BLACKLIST = os.path.join(
    os.path.dirname(__file__), '..', 'datasets', 'ru', 'processed', 'ru_reputation_raw.csv'
)
SAMPLE_TEXT_MAX_LEN = 140
ALLOW_LABEL = 'ALLOW'

log = logging.getLogger('vk_token_collector')


# ── Seed groups (verified live on 2026-05-13) ─────────────────────────────
#
# Real RU classifieds / "барахолка" public communities discovered via direct
# probes of ``wall.search owner_id=-GID query="+7"``. Phone hit-rate per
# 20-post sample is shown below; groups with hit-rate ≥ 25% are included.
#
# This list is NOT exhaustive — add new groups via the ``--seed-file`` CLI
# flag or by editing this list. Closed/blocked-wall groups are excluded
# (they return ``Access denied: wall is disabled`` / ``... only for community
# members``).

SEED_GROUPS: List[Tuple[int, str, str]] = [
    # group_id (positive), screen_name, human label
    # — top yielders (>= 75% phone hit-rate per 20-post wall.search sample) —
    (57963800,  'baraholka_chelyabinsk', 'Барахолка | Доска объявлений | Челябинск'),
    (1259638,   'baraholka_ekb',         'СТЕНА ОБЪЯВЛЕНИЙ Екатеринбурга'),
    (133313689, 'baraholka_kazan',       'Барахолка Казань'),
    (221554081, 'baraholka_khabarovsk',  'Барахолка Объявления Хабаровск'),
    (43878638,  'baraholka_tomsk',       'Барахолка Томск'),
    # — region barahоlка discovered live with measurable phone yield —
    (43176132,  'baraholka_pskov',       'Псков и область! Барахолка'),
    (33319630,  'baraholka_irkutsk',     'Барахолка Иркутск'),
    (232000618, 'baraholka_penza',       'Барахолка Пенза'),
    (52960603,  'baraholka_lipetsk',     'Объявления Липецка | Барахолка'),
    (27046159,  'baraholka_tula',        'Барахолка в Туле'),
    (54466005,  'baraholka_yaroslavl',   'Барахолка Ярославль'),
    (51609015,  'baraholka_kostroma',    'Костромская барахолка'),
    (136969894, 'baraholka_vladimir',    'Барахолка во Владимире'),
    (105751597, 'baraholka_volgograd',   'Барахолка Волгограда и Волжского'),
    (79712504,  'baraholka_smolensk',    'Смоленская барахолка'),
    (49126994,  'baraholka_arkhangelsk', 'Барахолка All Inclusive (Архангельск)'),
    (54950300,  'baraholka_petrozavodsk','Барахолка Петрозаводск'),
    (43865052,  'baraholka_tver',        'Барахолка Тверь'),
    (30733145,  'baraholka_stavropol',   'Не подошло | Барахолка Ставрополь'),
    (25968599,  'baraholka_saratov',     'Саратов барахолка'),
    (53017050,  'baraholka_kursk',       'Барахолка Курск'),
    (80411044,  'baraholka_orel',        'Это Орёл. Продажа/дар/обмен'),
    # — capital-region groups with smaller but live wall —
    (65530700,  'market_spb',            'market_spb (Маркет Санкт-Петербург)'),
    (26715478,  'baraholka_msk',         'Барахолка Москва'),
    # — generic fallback that still surfaces user content —
    (102222710, 'club102222710',         'локальный паблик с user-постами'),
]


@dataclass
class TokenHolder:
    """Wraps the access token so it never prints by accident.

    ``repr()`` returns a redacted form. The real value is in ``.value``.
    """

    value: str = field(repr=False)

    def __post_init__(self) -> None:
        if not self.value or not isinstance(self.value, str):
            raise ValueError('token must be a non-empty string')

    @property
    def masked(self) -> str:
        if len(self.value) <= 8:
            return '*' * len(self.value)
        return f'{self.value[:4]}…{self.value[-2:]}'

    def __repr__(self) -> str:
        return f'TokenHolder(masked={self.masked!r})'

    def __str__(self) -> str:
        return self.masked


@dataclass
class Candidate:
    normalized_number: str
    vk_source: str           # 'vk_wall_search' | 'vk_wall_get' | 'vk_comments'
    vk_object_id: str        # e.g. 'wall-57963800_12345' or 'wall_comment-...'
    vk_object_url: str       # 'https://vk.com/wall-57963800_12345'
    sample_text: str
    collected_at: str        # ISO-8601 UTC
    vk_author_id: str = ''   # str(from_id) of the post/comment author, '' if unknown
    vk_author_url: str = ''  # 'https://vk.com/id<from_id>'
    expected_label: str = ALLOW_LABEL

    def to_row(self) -> Dict[str, str]:
        return {
            'normalized_number': self.normalized_number,
            'vk_source': self.vk_source,
            'vk_object_id': self.vk_object_id,
            'vk_object_url': self.vk_object_url,
            'vk_author_id': self.vk_author_id,
            'vk_author_url': self.vk_author_url,
            'sample_text': self.sample_text,
            'expected_label': self.expected_label,
            'collected_at': self.collected_at,
        }


CSV_FIELDS: Tuple[str, ...] = (
    'normalized_number',
    'vk_source',
    'vk_object_id',
    'vk_object_url',
    'vk_author_id',
    'vk_author_url',
    'sample_text',
    'expected_label',
    'collected_at',
)


def _author_url_for(from_id: int) -> str:
    """VK profile URL for a positive user from_id.

    Negative from_ids (community-posts) are filtered upstream — we still
    accept them defensively here and return an empty string.
    """
    if not isinstance(from_id, int) or from_id <= 0:
        return ''
    return f'https://vk.com/id{from_id}'


# ── HTTP / API plumbing ────────────────────────────────────────────────────

class RateLimiter:
    """Простой token-bucket для VK API (≤ N req/sec из любого числа корутин)."""

    def __init__(self, rate_per_sec: float) -> None:
        self._interval = 1.0 / float(rate_per_sec)
        self._lock = asyncio.Lock()
        self._last_ts = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._last_ts + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._last_ts = now


class VKAPIError(RuntimeError):
    def __init__(self, code: int, msg: str, method: str):
        super().__init__(f'VK API {method} error {code}: {msg}')
        self.code = code
        self.method = method


async def call_vk(
    session: Any,
    method: str,
    params: Dict[str, Any],
    token: TokenHolder,
    limiter: RateLimiter,
    *,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Один вызов VK API с rate-limit + ретраями."""
    params = dict(params)
    params['access_token'] = token.value
    params['v'] = VK_API_VERSION
    url = f'{VK_API_BASE}/{method}?{urlencode(params)}'
    for attempt in range(1, max_retries + 1):
        await limiter.acquire()
        try:
            async with session.get(url, timeout=20) as resp:
                data = await resp.json(content_type=None)
        except Exception as exc:
            log.warning('vk.%s attempt %d/%d transport error: %s',
                        method, attempt, max_retries, exc)
            if attempt == max_retries:
                raise
            await asyncio.sleep(0.5 * attempt)
            continue
        if 'error' in data:
            code = int(data['error'].get('error_code', -1))
            msg = data['error'].get('error_msg', 'unknown')
            # 6: too many requests / 9: flood control — retry
            if code in (6, 9) and attempt < max_retries:
                log.warning('vk.%s rate-limit (%d), retrying', method, code)
                await asyncio.sleep(0.5 * attempt)
                continue
            raise VKAPIError(code, msg, method)
        return data.get('response', {})
    raise VKAPIError(-1, 'exhausted retries', method)


# ── Blacklist loading ─────────────────────────────────────────────────────

def load_blacklist(path: str) -> Set[str]:
    """Загружаем нормализованные +7-номера из ``ru_reputation_raw.csv``.

    Любой номер, лежащий тут, исключается из ALLOW-кандидатов — даже если
    его кто-то постит на стене VK. Это безопасный default: cold-start не
    должен поднимать known-фрод как «нормального».
    """
    if not os.path.isfile(path):
        log.warning('blacklist file not found: %s — proceeding without filter', path)
        return set()
    blacklist: Set[str] = set()
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            num = (row.get('номер') or row.get('normalized_number')
                   or row.get('number') or '').strip()
            if num and num.startswith('+7') and len(num) == 12:
                blacklist.add(num)
    log.info('blacklist loaded: %d numbers from %s', len(blacklist), path)
    return blacklist


# ── Phone extraction from VK post text ────────────────────────────────────

def extract_phones_from_text(text: str) -> List[str]:
    """Реюзаем ``extract_phones`` из ru_legitimate_collector, но передаём текст
    как HTML (он работает и с plain text — все regex'ы устойчивы к этому)."""
    if not text:
        return []
    # extract_phones() уже делает unescape + regex + normalize + plausibility
    return extract_phones(text)


def _truncate(text: str, max_len: int = SAMPLE_TEXT_MAX_LEN) -> str:
    if not text:
        return ''
    text = ' '.join(text.split())  # collapse whitespace
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + '…'


# ── Per-post / per-comment → Candidate conversion ─────────────────────────

def candidates_from_wall_item(
    item: Dict[str, Any],
    group_id: int,
    source: str,
    now_iso: str,
    blacklist: Set[str],
    seen: Set[str],
) -> List[Candidate]:
    """Конвертируем один post-объект из wall.* в Candidate(s).

    Фильтры:
      * ``from_id > 0`` — только посты юзеров. ``from_id < 0`` — это посты
        от лица сообщества (паблики/админы), такие пропускаем.
      * Все номера, попадающие в ``blacklist`` (известный фрод из
        ``ru_reputation_raw.csv``), исключаются.
      * Глобальный ``seen``-set гарантирует, что один и тот же номер не
        попадёт в CSV дважды.
    """
    from_id = item.get('from_id', 0)
    if not isinstance(from_id, int) or from_id <= 0:
        return []
    text = item.get('text') or ''
    if not text:
        return []
    phones = extract_phones_from_text(text)
    if not phones:
        return []
    out: List[Candidate] = []
    post_id = item.get('id')
    object_id = f'wall-{group_id}_{post_id}' if post_id is not None else f'wall-{group_id}'
    object_url = f'https://vk.com/wall-{group_id}_{post_id}' if post_id is not None else f'https://vk.com/club{group_id}'
    sample = _truncate(text)
    author_id_str = str(from_id)
    author_url = _author_url_for(from_id)
    for num in phones:
        if num in blacklist:
            log.debug('skip blacklisted number %s in %s', num, object_url)
            continue
        if num in seen:
            continue
        seen.add(num)
        out.append(Candidate(
            normalized_number=num,
            vk_source=source,
            vk_object_id=object_id,
            vk_object_url=object_url,
            vk_author_id=author_id_str,
            vk_author_url=author_url,
            sample_text=sample,
            collected_at=now_iso,
        ))
    return out


def candidates_from_comment(
    comment: Dict[str, Any],
    group_id: int,
    post_id: int,
    now_iso: str,
    blacklist: Set[str],
    seen: Set[str],
) -> List[Candidate]:
    from_id = comment.get('from_id', 0)
    if not isinstance(from_id, int) or from_id <= 0:
        return []
    text = comment.get('text') or ''
    if not text:
        return []
    phones = extract_phones_from_text(text)
    if not phones:
        return []
    out: List[Candidate] = []
    cid = comment.get('id')
    object_id = f'wall_comment-{group_id}_{post_id}_{cid}'
    object_url = f'https://vk.com/wall-{group_id}_{post_id}?reply={cid}'
    sample = _truncate(text)
    author_id_str = str(from_id)
    author_url = _author_url_for(from_id)
    for num in phones:
        if num in blacklist:
            continue
        if num in seen:
            continue
        seen.add(num)
        out.append(Candidate(
            normalized_number=num,
            vk_source='vk_comments',
            vk_object_id=object_id,
            vk_object_url=object_url,
            vk_author_id=author_id_str,
            vk_author_url=author_url,
            sample_text=sample,
            collected_at=now_iso,
        ))
    return out


# ── Per-group collectors ──────────────────────────────────────────────────

@dataclass
class CollectStats:
    api_calls: int = 0
    posts_seen: int = 0
    user_posts_seen: int = 0
    candidates_added: int = 0
    errors: int = 0


async def collect_wall_search(
    session: Any,
    token: TokenHolder,
    limiter: RateLimiter,
    group_id: int,
    *,
    query: str,
    count: int,
    blacklist: Set[str],
    seen: Set[str],
    stats: CollectStats,
    now_iso: str,
    out: List[Candidate],
) -> None:
    try:
        resp = await call_vk(session, 'wall.search',
                             {'owner_id': -group_id, 'query': query, 'count': count},
                             token, limiter)
        stats.api_calls += 1
    except VKAPIError as exc:
        stats.errors += 1
        log.info('club%d wall.search skipped: %s', group_id, exc)
        return
    items = resp.get('items', []) or []
    stats.posts_seen += len(items)
    for item in items:
        if item.get('from_id', 0) > 0:
            stats.user_posts_seen += 1
        cands = candidates_from_wall_item(item, group_id, 'vk_wall_search',
                                          now_iso, blacklist, seen)
        out.extend(cands)
        stats.candidates_added += len(cands)


async def collect_wall_get(
    session: Any,
    token: TokenHolder,
    limiter: RateLimiter,
    group_id: int,
    *,
    count: int,
    offset: int,
    blacklist: Set[str],
    seen: Set[str],
    stats: CollectStats,
    now_iso: str,
    out: List[Candidate],
) -> List[int]:
    """Returns post_ids that look classifieds-like for downstream comment scan."""
    try:
        resp = await call_vk(session, 'wall.get',
                             {'owner_id': -group_id, 'count': count, 'offset': offset},
                             token, limiter)
        stats.api_calls += 1
    except VKAPIError as exc:
        stats.errors += 1
        log.info('club%d wall.get skipped: %s', group_id, exc)
        return []
    items = resp.get('items', []) or []
    stats.posts_seen += len(items)
    post_ids_for_comments: List[int] = []
    for item in items:
        if item.get('from_id', 0) > 0:
            stats.user_posts_seen += 1
            pid = item.get('id')
            if pid is not None:
                post_ids_for_comments.append(pid)
        cands = candidates_from_wall_item(item, group_id, 'vk_wall_get',
                                          now_iso, blacklist, seen)
        out.extend(cands)
        stats.candidates_added += len(cands)
    return post_ids_for_comments


async def collect_post_comments(
    session: Any,
    token: TokenHolder,
    limiter: RateLimiter,
    group_id: int,
    post_id: int,
    *,
    count: int,
    blacklist: Set[str],
    seen: Set[str],
    stats: CollectStats,
    now_iso: str,
    out: List[Candidate],
) -> None:
    try:
        resp = await call_vk(session, 'wall.getComments',
                             {'owner_id': -group_id, 'post_id': post_id,
                              'count': count, 'need_likes': 0, 'extended': 0},
                             token, limiter)
        stats.api_calls += 1
    except VKAPIError as exc:
        stats.errors += 1
        log.debug('comments club%d post %d skipped: %s', group_id, post_id, exc)
        return
    items = resp.get('items', []) or []
    for c in items:
        cands = candidates_from_comment(c, group_id, post_id, now_iso, blacklist, seen)
        out.extend(cands)
        stats.candidates_added += len(cands)


# ── Top-level driver ──────────────────────────────────────────────────────

async def collect_for_group(
    session: Any,
    token: TokenHolder,
    limiter: RateLimiter,
    group: Tuple[int, str, str],
    *,
    posts_per_group: int,
    do_wall_get: bool,
    do_comments: bool,
    comments_top_n: int,
    blacklist: Set[str],
    seen: Set[str],
    out: List[Candidate],
    now_iso: str,
) -> CollectStats:
    gid, alias, label = group
    stats = CollectStats()
    log.info('club%d (%s) — %s — start', gid, alias, label)
    # 1) wall.search query="+7" — самый плотный источник
    await collect_wall_search(session, token, limiter, gid,
                              query='+7', count=posts_per_group,
                              blacklist=blacklist, seen=seen, stats=stats,
                              now_iso=now_iso, out=out)
    # 2) wall.get для широкого охвата
    post_ids: List[int] = []
    if do_wall_get:
        post_ids = await collect_wall_get(session, token, limiter, gid,
                                          count=posts_per_group, offset=0,
                                          blacklist=blacklist, seen=seen,
                                          stats=stats, now_iso=now_iso, out=out)
    # 3) комменты к топ-N постам (опционально)
    if do_comments and post_ids:
        for pid in post_ids[:comments_top_n]:
            await collect_post_comments(session, token, limiter, gid, pid,
                                        count=100, blacklist=blacklist,
                                        seen=seen, stats=stats,
                                        now_iso=now_iso, out=out)
    log.info('club%d done: api_calls=%d posts=%d user_posts=%d new_phones=%d errors=%d',
             gid, stats.api_calls, stats.posts_seen, stats.user_posts_seen,
             stats.candidates_added, stats.errors)
    return stats


async def collect_all(
    token: TokenHolder,
    seeds: Sequence[Tuple[int, str, str]],
    *,
    max_calls: int,
    posts_per_group: int,
    do_wall_get: bool,
    do_comments: bool,
    comments_top_n: int,
    rate_per_sec: float,
    blacklist: Set[str],
) -> Tuple[List[Candidate], Dict[str, Any]]:
    import aiohttp  # local import — keeps module importable if aiohttp missing in CI
    seen: Set[str] = set()
    out: List[Candidate] = []
    limiter = RateLimiter(rate_per_sec)
    now_iso = _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    total_calls = 0
    per_group_stats: List[Dict[str, Any]] = []
    async with aiohttp.ClientSession(headers={'User-Agent': 'vk-eval-collector/1.0'}) as session:
        for group in seeds:
            if total_calls >= max_calls:
                log.info('budget exhausted (%d ≥ %d), stopping', total_calls, max_calls)
                break
            s = await collect_for_group(session, token, limiter, group,
                                        posts_per_group=posts_per_group,
                                        do_wall_get=do_wall_get,
                                        do_comments=do_comments,
                                        comments_top_n=comments_top_n,
                                        blacklist=blacklist,
                                        seen=seen, out=out, now_iso=now_iso)
            total_calls += s.api_calls
            per_group_stats.append({
                'group_id': group[0],
                'screen_name': group[1],
                'api_calls': s.api_calls,
                'posts_seen': s.posts_seen,
                'user_posts_seen': s.user_posts_seen,
                'new_phones': s.candidates_added,
                'errors': s.errors,
            })
    summary = {
        'total_api_calls': total_calls,
        'total_groups_processed': len(per_group_stats),
        'total_candidates': len(out),
        'unique_numbers': len({c.normalized_number for c in out}),
        'per_group': per_group_stats,
    }
    return out, summary


# ── CSV output ────────────────────────────────────────────────────────────

def write_candidates_csv(path: str, candidates: Sequence[Candidate]) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for c in candidates:
            writer.writerow(c.to_row())


def load_seed_file(path: str) -> List[Tuple[int, str, str]]:
    """Load extra seed groups from a CSV file with columns ``group_id, screen_name, label``."""
    out: List[Tuple[int, str, str]] = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid_raw = row.get('group_id') or row.get('id') or ''
            try:
                gid = int(gid_raw)
            except ValueError:
                continue
            alias = row.get('screen_name') or row.get('alias') or ''
            label = row.get('label') or row.get('name') or ''
            out.append((gid, alias, label))
    return out


def resolve_token(cli_token: Optional[str]) -> TokenHolder:
    val = cli_token or os.environ.get('VK_SERVICE_TOKEN') or os.environ.get('vk')
    if not val:
        raise SystemExit(
            'ERROR: no VK service token. Provide --token, or set $VK_SERVICE_TOKEN, '
            'or $vk env var. Create one at https://vk.com/apps?act=manage as a '
            'Standalone application — no user login required.'
        )
    return TokenHolder(val.strip())


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', default=DEFAULT_OUTPUT,
                    help=f'output CSV path (default: {DEFAULT_OUTPUT})')
    ap.add_argument('--token', default=None,
                    help='VK service token (env $VK_SERVICE_TOKEN or $vk used by default)')
    ap.add_argument('--blacklist', default=DEFAULT_BLACKLIST,
                    help=f'ru_reputation_raw.csv path (default: {DEFAULT_BLACKLIST})')
    ap.add_argument('--seed-file', default=None,
                    help='Extra seed groups CSV (cols: group_id,screen_name,label)')
    ap.add_argument('--max-groups', type=int, default=None,
                    help='cap on number of seed groups to process')
    ap.add_argument('--max-calls', type=int, default=200,
                    help='hard cap on total API calls (default 200)')
    ap.add_argument('--posts-per-group', type=int, default=100,
                    help='posts to fetch per wall.search/wall.get call (default 100)')
    ap.add_argument('--no-wall-get', action='store_true',
                    help='disable wall.get (use only wall.search for highest precision)')
    ap.add_argument('--with-comments', action='store_true',
                    help='also fetch wall.getComments on top-N user posts')
    ap.add_argument('--comments-top-n', type=int, default=3,
                    help='how many posts per group to fetch comments for (default 3)')
    ap.add_argument('--rate', type=float, default=DEFAULT_RATE_LIMIT_PER_SEC,
                    help=f'API rate limit per second (default {DEFAULT_RATE_LIMIT_PER_SEC})')
    ap.add_argument('--summary-json', default=None,
                    help='Optional path to write run summary JSON')
    ap.add_argument('--verbose', action='store_true', help='DEBUG logging')
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )

    token = resolve_token(args.token)
    log.info('using VK service token %s', token)  # masked by repr

    seeds: List[Tuple[int, str, str]] = list(SEED_GROUPS)
    if args.seed_file:
        seeds.extend(load_seed_file(args.seed_file))
    if args.max_groups is not None:
        seeds = seeds[: args.max_groups]
    log.info('seed groups: %d', len(seeds))

    blacklist = load_blacklist(args.blacklist)

    candidates, summary = asyncio.run(collect_all(
        token, seeds,
        max_calls=args.max_calls,
        posts_per_group=args.posts_per_group,
        do_wall_get=not args.no_wall_get,
        do_comments=args.with_comments,
        comments_top_n=args.comments_top_n,
        rate_per_sec=args.rate,
        blacklist=blacklist,
    ))

    write_candidates_csv(args.output, candidates)
    log.info('wrote %d rows (%d unique RU numbers) to %s',
             len(candidates), summary['unique_numbers'], args.output)

    if args.summary_json:
        with open(args.summary_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log.info('summary saved to %s', args.summary_json)

    return 0


if __name__ == '__main__':
    sys.exit(main())
