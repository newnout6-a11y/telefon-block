"""Offline tests for scripts.vk_token_collector.

We avoid real VK API traffic by feeding hand-crafted JSON fixtures through
the module's parsing helpers. The HTTP layer (``call_vk``) is mocked with
a tiny async session that returns canned bodies.
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import vk_token_collector as vtc  # noqa: E402

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'vk_token_collector')


def _load_fixture(name: str) -> Dict[str, Any]:
    with open(os.path.join(FIXTURE_DIR, name), 'r', encoding='utf-8') as f:
        return json.load(f)


# ── TokenHolder ───────────────────────────────────────────────────────────

class TestTokenHolder:
    def test_value_accessible(self):
        t = vtc.TokenHolder('abc12345678901234567890')
        assert t.value == 'abc12345678901234567890'

    def test_repr_masks_token(self):
        t = vtc.TokenHolder('abc12345678901234567890')
        r = repr(t)
        assert 'abc12345678901234567890' not in r
        assert 'abc1' in r  # first 4 chars visible
        assert 'TokenHolder' in r

    def test_str_masks_token(self):
        t = vtc.TokenHolder('abc12345678901234567890')
        assert 'abc12345678901234567890' not in str(t)

    def test_short_token_fully_masked(self):
        t = vtc.TokenHolder('short')
        assert '*' * 5 == t.masked

    def test_empty_token_rejected(self):
        with pytest.raises(ValueError):
            vtc.TokenHolder('')


# ── candidates_from_wall_item ─────────────────────────────────────────────

class TestCandidatesFromWallItem:
    def test_user_post_with_phone(self):
        resp = _load_fixture('wall_search_response.json')
        item = resp['response']['items'][0]  # from_id=9683502 (user), phone in text
        seen: set = set()
        cands = vtc.candidates_from_wall_item(item, group_id=57963800,
                                              source='vk_wall_search',
                                              now_iso='2026-05-13T00:00:00Z',
                                              blacklist=set(), seen=seen)
        assert len(cands) == 1
        c = cands[0]
        assert c.normalized_number == '+79822753595'
        assert c.vk_source == 'vk_wall_search'
        assert c.vk_object_id == 'wall-57963800_9001'
        assert c.vk_object_url == 'https://vk.com/wall-57963800_9001'
        assert c.vk_author_id == '9683502'
        assert c.vk_author_url == 'https://vk.com/id9683502'
        assert c.expected_label == 'ALLOW'
        assert 'Продаю' in c.sample_text
        assert seen == {'+79822753595'}

    def test_compact_phone_form_extracted(self):
        resp = _load_fixture('wall_search_response.json')
        item = resp['response']['items'][1]  # text contains "89823295413"
        seen: set = set()
        cands = vtc.candidates_from_wall_item(item, group_id=57963800,
                                              source='vk_wall_search',
                                              now_iso='2026-05-13T00:00:00Z',
                                              blacklist=set(), seen=seen)
        assert len(cands) == 1
        assert cands[0].normalized_number == '+79823295413'

    def test_admin_post_skipped_via_from_id(self):
        resp = _load_fixture('wall_search_response.json')
        item = resp['response']['items'][2]  # from_id=-57963800 (admin/group)
        seen: set = set()
        cands = vtc.candidates_from_wall_item(item, group_id=57963800,
                                              source='vk_wall_search',
                                              now_iso='2026-05-13T00:00:00Z',
                                              blacklist=set(), seen=seen)
        assert cands == []
        assert seen == set()

    def test_blacklist_filter(self):
        resp = _load_fixture('wall_search_response.json')
        item = resp['response']['items'][3]  # user post but number is blacklisted
        seen: set = set()
        blacklist = {'+79031234567'}
        cands = vtc.candidates_from_wall_item(item, group_id=57963800,
                                              source='vk_wall_search',
                                              now_iso='2026-05-13T00:00:00Z',
                                              blacklist=blacklist, seen=seen)
        assert cands == []
        # seen is not polluted by blacklisted numbers
        assert seen == set()

    def test_dedup_within_run(self):
        resp = _load_fixture('wall_search_response.json')
        item = resp['response']['items'][0]
        seen: set = {'+79822753595'}  # already seen
        cands = vtc.candidates_from_wall_item(item, group_id=57963800,
                                              source='vk_wall_search',
                                              now_iso='2026-05-13T00:00:00Z',
                                              blacklist=set(), seen=seen)
        assert cands == []

    def test_empty_text_skipped(self):
        item = {'id': 1, 'from_id': 100, 'text': ''}
        cands = vtc.candidates_from_wall_item(item, group_id=1,
                                              source='vk_wall_get',
                                              now_iso='2026-05-13T00:00:00Z',
                                              blacklist=set(), seen=set())
        assert cands == []

    def test_missing_from_id_skipped(self):
        item = {'id': 1, 'text': '+79991234567 sell something'}
        cands = vtc.candidates_from_wall_item(item, group_id=1,
                                              source='vk_wall_get',
                                              now_iso='2026-05-13T00:00:00Z',
                                              blacklist=set(), seen=set())
        assert cands == []


# ── candidates_from_comment ───────────────────────────────────────────────

class TestCandidatesFromComment:
    def test_user_comment_with_phone(self):
        resp = _load_fixture('wall_comments_response.json')
        comment = resp['response']['items'][0]
        seen: set = set()
        cands = vtc.candidates_from_comment(comment, group_id=1259638,
                                            post_id=8001,
                                            now_iso='2026-05-13T00:00:00Z',
                                            blacklist=set(), seen=seen)
        assert len(cands) == 1
        c = cands[0]
        assert c.normalized_number == '+79175554433'
        assert c.vk_source == 'vk_comments'
        assert c.vk_object_id == 'wall_comment-1259638_8001_100'
        assert '?reply=100' in c.vk_object_url
        assert c.vk_author_id == '111222333'
        assert c.vk_author_url == 'https://vk.com/id111222333'

    def test_admin_comment_skipped(self):
        resp = _load_fixture('wall_comments_response.json')
        comment = resp['response']['items'][1]  # from_id < 0
        cands = vtc.candidates_from_comment(comment, group_id=1259638,
                                            post_id=8001,
                                            now_iso='2026-05-13T00:00:00Z',
                                            blacklist=set(), seen=set())
        assert cands == []

    def test_comment_no_phone(self):
        resp = _load_fixture('wall_comments_response.json')
        comment = resp['response']['items'][2]
        cands = vtc.candidates_from_comment(comment, group_id=1259638,
                                            post_id=8001,
                                            now_iso='2026-05-13T00:00:00Z',
                                            blacklist=set(), seen=set())
        assert cands == []


# ── Helpers ───────────────────────────────────────────────────────────────

class TestHelpers:
    def test_extract_phones_from_text_normalizes(self):
        text = "перезвоните на 8 (982) 275-35-95, спасибо"
        phones = vtc.extract_phones_from_text(text)
        assert phones == ['+79822753595']

    def test_extract_phones_drops_non_russian(self):
        text = "Call us at +1 (415) 555-1234 or +44 7700 900000"
        phones = vtc.extract_phones_from_text(text)
        assert phones == []

    def test_truncate_long_text(self):
        text = 'x' * 500
        result = vtc._truncate(text, max_len=140)
        assert len(result) == 140
        assert result.endswith('…')

    def test_truncate_collapses_whitespace(self):
        text = 'hello    \n\n  world'
        assert vtc._truncate(text) == 'hello world'


# ── Mock async HTTP session ───────────────────────────────────────────────

class FakeResponse:
    def __init__(self, body: Dict[str, Any]):
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def json(self, **_: Any):
        return self._body


class FakeSession:
    """Replays URL → fixture mapping. Records requests for assertions."""

    def __init__(self, mapping: Dict[str, Dict[str, Any]]):
        self._mapping = mapping
        self.calls: List[str] = []

    def get(self, url: str, **_: Any):
        self.calls.append(url)
        # Match by VK method name substring
        for needle, body in self._mapping.items():
            if needle in url:
                return FakeResponse(body)
        # Unknown URL → error response
        return FakeResponse({'error': {'error_code': -1, 'error_msg': 'unmocked URL'}})


# ── call_vk ───────────────────────────────────────────────────────────────

class TestCallVK:
    def test_unwraps_response_key(self):
        body = _load_fixture('wall_search_response.json')
        session = FakeSession({'wall.search': body})
        token = vtc.TokenHolder('test_token_abcdef0123')
        limiter = vtc.RateLimiter(rate_per_sec=100)
        result = asyncio.run(vtc.call_vk(
            session, 'wall.search',
            {'owner_id': -57963800, 'query': '+7', 'count': 5},
            token, limiter,
        ))
        assert 'items' in result
        assert result['count'] == 4

    def test_raises_on_error_response(self):
        body = _load_fixture('error_response.json')
        session = FakeSession({'wall.get': body})
        token = vtc.TokenHolder('test_token_abcdef0123')
        limiter = vtc.RateLimiter(rate_per_sec=100)
        with pytest.raises(vtc.VKAPIError) as exc_info:
            asyncio.run(vtc.call_vk(
                session, 'wall.get', {'owner_id': -1},
                token, limiter, max_retries=1,
            ))
        assert exc_info.value.code == 15
        assert 'wall is disabled' in str(exc_info.value)

    def test_token_in_url_not_logged(self, caplog):
        """The raw token may appear in the URL, but our log messages mask it."""
        body = _load_fixture('error_response.json')
        session = FakeSession({'wall.get': body})
        token = vtc.TokenHolder('super_secret_token_must_not_leak_0123')
        limiter = vtc.RateLimiter(rate_per_sec=100)
        with caplog.at_level('DEBUG', logger='vk_token_collector'):
            with pytest.raises(vtc.VKAPIError):
                asyncio.run(vtc.call_vk(
                    session, 'wall.get', {'owner_id': -1},
                    token, limiter, max_retries=1,
                ))
        for rec in caplog.records:
            assert 'super_secret_token_must_not_leak_0123' not in rec.getMessage()


# ── Blacklist loading ────────────────────────────────────────────────────

class TestBlacklist:
    def test_loads_normalized_numbers(self, tmp_path):
        p = tmp_path / 'bl.csv'
        p.write_text(
            'номер,источник\n'
            '+79991234567,test\n'
            '+79998887766,test\n'
            'not_a_number,test\n',
            encoding='utf-8',
        )
        bl = vtc.load_blacklist(str(p))
        assert bl == {'+79991234567', '+79998887766'}

    def test_missing_file_returns_empty(self):
        assert vtc.load_blacklist('/nonexistent/path.csv') == set()


# ── CSV output ───────────────────────────────────────────────────────────

class TestWriteCSV:
    def test_writes_full_schema(self, tmp_path):
        c = vtc.Candidate(
            normalized_number='+79991234567',
            vk_source='vk_wall_search',
            vk_object_id='wall-1_1',
            vk_object_url='https://vk.com/wall-1_1',
            vk_author_id='42',
            vk_author_url='https://vk.com/id42',
            sample_text='hello',
            collected_at='2026-05-13T00:00:00Z',
        )
        p = tmp_path / 'out.csv'
        vtc.write_candidates_csv(str(p), [c])
        rows = list(csv.DictReader(open(p, 'r', encoding='utf-8')))
        assert len(rows) == 1
        r = rows[0]
        assert r['normalized_number'] == '+79991234567'
        assert r['vk_source'] == 'vk_wall_search'
        assert r['expected_label'] == 'ALLOW'
        assert r['sample_text'] == 'hello'
        assert r['vk_author_id'] == '42'
        assert r['vk_author_url'] == 'https://vk.com/id42'
        # CSV schema must include the new author columns explicitly.
        with open(p, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
        assert 'vk_author_id' in header
        assert 'vk_author_url' in header

    def test_author_url_helper(self):
        assert vtc._author_url_for(9683502) == 'https://vk.com/id9683502'
        assert vtc._author_url_for(0) == ''
        assert vtc._author_url_for(-100) == ''
        # malformed inputs are tolerated and return ''
        assert vtc._author_url_for('not-an-int') == ''  # type: ignore[arg-type]


# ── End-to-end with fake session ─────────────────────────────────────────

class TestEndToEnd:
    def test_collect_all_minimal(self):
        body = _load_fixture('wall_search_response.json')
        session_map = {'wall.search': body}

        async def runner():
            import aiohttp  # noqa: F401  (force import for module presence)
            limiter = vtc.RateLimiter(100)
            token = vtc.TokenHolder('test_token_abcdef0123')
            session = FakeSession(session_map)
            out: List[vtc.Candidate] = []
            seen: set = set()
            stats = vtc.CollectStats()
            await vtc.collect_wall_search(
                session, token, limiter, group_id=57963800,
                query='+7', count=5,
                blacklist={'+79031234567'},
                seen=seen, stats=stats, out=out,
                now_iso='2026-05-13T00:00:00Z',
            )
            return out, stats

        out, stats = asyncio.run(runner())
        # 4 items total, 1 admin (skipped), 1 blacklisted, 2 valid user posts
        # → 2 candidates
        assert len(out) == 2
        nums = {c.normalized_number for c in out}
        assert nums == {'+79822753595', '+79823295413'}
        assert all(c.expected_label == 'ALLOW' for c in out)
        assert stats.api_calls == 1
        assert stats.user_posts_seen == 3  # 3 user posts seen (incl. blacklisted)
        assert stats.candidates_added == 2
