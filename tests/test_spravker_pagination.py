"""Tests for the deeper spravker.ru crawl introduced in the ALLOW 10x PR.

The pre-existing scrape_spravker_category() only fetched listing page 1
and the first 10 org pages — discarding ~80% of the available phones
because spravker categories typically paginate to 10-30 pages with 10-20
org cards each.

The new behaviour:
  * detects the highest pagination page from `?page=N` anchors,
  * iterates pages 2..min(found, SPRAVKER_MAX_LISTING_PAGES=8),
  * dedups org URLs across pages, then visits up to
    SPRAVKER_MAX_ORG_PAGES_PER_CATEGORY=30 org pages.

These tests exercise the parsing path with offline HTML fixtures so the
crawler doesn't need network or a live spravker subdomain.
"""
import asyncio
import os
import sys
from typing import Dict, List, Optional

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from ru_legitimate_collector import (  # noqa: E402
    AsyncScraper,
    SPRAVKER_PAGINATION_RE,
    SPRAVKER_ORG_RE,
    SPRAVKER_MAX_LISTING_PAGES,
    SPRAVKER_MAX_ORG_PAGES_PER_CATEGORY,
    _spravker_max_page,
    scrape_spravker_category,
)


FIX_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'allow_10x')


def _read(name: str) -> str:
    with open(os.path.join(FIX_DIR, name), encoding='utf-8') as f:
        return f.read()


class FakeScraper:
    """Stub of `AsyncScraper.fetch` returning preloaded fixtures by URL."""

    def __init__(self, page_map: Dict[str, str]):
        self.page_map = page_map
        self.fetches: List[str] = []
        self.added: List[tuple] = []
        self.results: List = []
        self.seen: set = set()
        self.blacklist: set = set()
        self.stats = {'fetched': 0, 'failed': 0, 'phones_found': 0, 'skipped': 0}

    async def fetch(self, url: str, allow_status=None) -> Optional[str]:
        self.fetches.append(url)
        return self.page_map.get(url)

    def add_phones(self, phones, name, category, source, city, url, source_confidence=0.70):
        for ph in phones:
            self.added.append((ph, name, category, source, city, url))
            self.results.append((ph, name, category, source, city, url))
        return len(phones)

    def add(self, *args, **kwargs):
        return True


# ── pagination regex ──────────────────────────────────────────────────────


class TestPaginationRegex:
    def test_finds_all_page_numbers(self):
        html = _read('spravker_listing_page1.html')
        # Fixture lists pages 1..5 in pagination-list anchors. The highest
        # number is 5; page 1 itself isn't required by the helper (it's
        # always implicit).
        assert _spravker_max_page(html) == 5

    def test_no_pagination_returns_one(self):
        html = _read('spravker_listing_no_pagination.html')
        assert _spravker_max_page(html) == 1

    def test_ignores_non_pagination_anchors(self):
        # Random ?page=N in a non-pagination-list class shouldn't match.
        html = '''
            <a href="/bolnicy/?page=99" class="some-other-class">spurious</a>
            <a href="/bolnicy/?page=2" class="pagination-list__link">2</a>
        '''
        assert _spravker_max_page(html) == 2

    def test_works_with_class_before_href(self):
        # Order: class first, then href.
        html = (
            '<a class="pagination-list__link" '
            'href="https://x.spravker.ru/cat/?page=7">7</a>'
        )
        assert _spravker_max_page(html) == 7

    def test_works_with_href_before_class(self):
        # Order: href first, then class.
        html = (
            '<a href="https://x.spravker.ru/cat/?page=8" '
            'class="pagination-list__link">8</a>'
        )
        assert _spravker_max_page(html) == 8


# ── org-page regex stays intact ───────────────────────────────────────────


class TestOrgPageRegex:
    def test_finds_htm_org_links(self):
        html = _read('spravker_listing_page1.html')
        links = SPRAVKER_ORG_RE.findall(html)
        assert '/bolnicy/pervaia-klinika.htm' in links
        assert '/bolnicy/ekomed.htm' in links
        assert '/bolnicy/doremi-clinic.htm' in links

    def test_dedup_across_pages(self):
        # Page 1 has /pervaia-klinika.htm; page 2 has different orgs. Total
        # URLs after dedup should equal sum of unique URLs.
        p1 = SPRAVKER_ORG_RE.findall(_read('spravker_listing_page1.html'))
        p2 = SPRAVKER_ORG_RE.findall(_read('spravker_listing_page2.html'))
        all_unique = set(p1) | set(p2)
        assert len(all_unique) == len(p1) + len(p2)  # no overlap in fixtures


# ── async scrape_spravker_category integration ────────────────────────────


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestScrapeSpravkerCategoryPagination:
    """Verify the new scrape_spravker_category iterates through paginated
    listings and dedups org URLs."""

    def _build_scraper_with_pages(self) -> FakeScraper:
        host = 'msk.spravker.ru'
        return FakeScraper({
            f'https://{host}/bolnicy/':            _read('spravker_listing_page1.html'),
            f'https://{host}/bolnicy/?page=2':     _read('spravker_listing_page2.html'),
            f'https://{host}/bolnicy/?page=3':     '',  # empty triggers stop
            f'https://{host}/bolnicy/pervaia-klinika.htm':   _read('spravker_org_page.html'),
            f'https://{host}/bolnicy/ekomed.htm':            _read('spravker_org_page.html'),
            f'https://{host}/bolnicy/doremi-clinic.htm':     _read('spravker_org_page.html'),
            f'https://{host}/bolnicy/clinic-page2-a.htm':    _read('spravker_org_page.html'),
            f'https://{host}/bolnicy/clinic-page2-b.htm':    _read('spravker_org_page.html'),
        })

    def test_paginates_through_discovered_pages(self):
        scraper = self._build_scraper_with_pages()
        loop = asyncio.new_event_loop()
        try:
            count = loop.run_until_complete(
                scrape_spravker_category(scraper, 'msk', 'msk.spravker.ru', 'bolnicy')
            )
        finally:
            loop.close()
        # Should have visited:
        #   - listing page 1
        #   - listing page 2 (page=2)
        #   - listing page 3 (page=3) which returns '' — stops there
        #   - 5 unique org pages
        listing_fetches = [u for u in scraper.fetches if '/bolnicy/' in u and '.htm' not in u]
        assert any('?page=2' in u for u in listing_fetches), 'expected page=2 fetch'
        # We added phones from both listings + org pages.
        assert count > 0

    def test_handles_no_pagination(self):
        host = 'tinytown.spravker.ru'
        scraper = FakeScraper({
            f'https://{host}/bolnicy/': _read('spravker_listing_no_pagination.html'),
            f'https://{host}/bolnicy/single-clinic.htm': _read('spravker_org_page.html'),
        })
        loop = asyncio.new_event_loop()
        try:
            count = loop.run_until_complete(
                scrape_spravker_category(scraper, 'tinytown', host, 'bolnicy')
            )
        finally:
            loop.close()
        # No pagination → only fetches listing page 1 + org page(s).
        listing_fetches = [u for u in scraper.fetches if '/bolnicy/' in u]
        # exactly one listing fetch, no ?page=N
        assert sum(1 for u in listing_fetches if '?page=' in u) == 0
        assert count > 0

    def test_caps_at_max_listing_pages(self):
        # Build a fixture that claims pages 1..50; the scraper must cap
        # at SPRAVKER_MAX_LISTING_PAGES regardless.
        host = 'msk.spravker.ru'
        page1_with_50 = '''<!DOCTYPE html><html><body>
        <div class="org-card"><a href="/bolnicy/x.htm">X</a><a href="tel:+74950000001">x</a></div>
        ''' + ''.join(
            f'<a href="https://{host}/bolnicy/?page={n}" class="pagination-list__link">{n}</a>'
            for n in range(2, 51)
        ) + '</body></html>'
        page_map = {f'https://{host}/bolnicy/': page1_with_50}
        # also satisfy fetches up to MAX_LISTING_PAGES so we don't break
        # early on missing fixture
        for n in range(2, SPRAVKER_MAX_LISTING_PAGES + 5):
            page_map[f'https://{host}/bolnicy/?page={n}'] = (
                '<a href="/bolnicy/y.htm">y</a>'
                f'<a href="tel:+74950{n:06d}">y</a>'
            )
        page_map[f'https://{host}/bolnicy/x.htm'] = _read('spravker_org_page.html')
        page_map[f'https://{host}/bolnicy/y.htm'] = _read('spravker_org_page.html')

        scraper = FakeScraper(page_map)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                scrape_spravker_category(scraper, 'msk', host, 'bolnicy')
            )
        finally:
            loop.close()
        # The scraper should NOT fetch beyond SPRAVKER_MAX_LISTING_PAGES.
        page_n_fetched = [
            int(u.split('?page=')[1].split('&')[0])
            for u in scraper.fetches if '?page=' in u
        ]
        assert page_n_fetched, 'expected at least some paginated fetches'
        assert max(page_n_fetched) <= SPRAVKER_MAX_LISTING_PAGES, (
            f'cap broken: fetched up to page {max(page_n_fetched)}, '
            f'should be ≤ {SPRAVKER_MAX_LISTING_PAGES}'
        )

    def test_caps_at_max_org_pages_per_category(self):
        # Listing with 100 unique org URLs and 1 page; scraper must visit
        # at most SPRAVKER_MAX_ORG_PAGES_PER_CATEGORY.
        host = 'msk.spravker.ru'
        org_links_html = ''.join(
            f'<a href="/bolnicy/clinic-{i}.htm">x</a>' for i in range(100)
        )
        page1 = f'''<!DOCTYPE html><html><body>
        {org_links_html}
        <a href="tel:+74959999999">stub phone</a>
        </body></html>'''
        page_map = {f'https://{host}/bolnicy/': page1}
        for i in range(100):
            page_map[f'https://{host}/bolnicy/clinic-{i}.htm'] = _read('spravker_org_page.html')

        scraper = FakeScraper(page_map)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                scrape_spravker_category(scraper, 'msk', host, 'bolnicy')
            )
        finally:
            loop.close()
        org_fetches = [u for u in scraper.fetches if u.endswith('.htm')]
        assert len(org_fetches) <= SPRAVKER_MAX_ORG_PAGES_PER_CATEGORY, (
            f'org-page cap broken: visited {len(org_fetches)}, '
            f'should be ≤ {SPRAVKER_MAX_ORG_PAGES_PER_CATEGORY}'
        )

    def test_stops_on_empty_page(self):
        # If page 3 returns '' (empty), should not attempt page 4.
        host = 'msk.spravker.ru'
        scraper = FakeScraper({
            f'https://{host}/bolnicy/': _read('spravker_listing_page1.html'),
            f'https://{host}/bolnicy/?page=2': _read('spravker_listing_page2.html'),
            f'https://{host}/bolnicy/?page=3': '',
            # We deliberately don't put 4 / 5 in the map even though page1
            # advertises them — the scraper should bail out at page 3.
            f'https://{host}/bolnicy/pervaia-klinika.htm': _read('spravker_org_page.html'),
            f'https://{host}/bolnicy/ekomed.htm': _read('spravker_org_page.html'),
            f'https://{host}/bolnicy/doremi-clinic.htm': _read('spravker_org_page.html'),
            f'https://{host}/bolnicy/clinic-page2-a.htm': _read('spravker_org_page.html'),
            f'https://{host}/bolnicy/clinic-page2-b.htm': _read('spravker_org_page.html'),
        })
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                scrape_spravker_category(scraper, 'msk', host, 'bolnicy')
            )
        finally:
            loop.close()
        # No page=4 or page=5 fetch.
        page_qs = [u for u in scraper.fetches if '?page=' in u]
        page_nums = [int(u.split('?page=')[1].split('&')[0]) for u in page_qs]
        assert 4 not in page_nums and 5 not in page_nums

    def test_dedups_org_urls_across_pages(self):
        # Build 2 pages where one org URL is shared. Scraper should fetch
        # the shared org page exactly once.
        host = 'msk.spravker.ru'
        shared = '/bolnicy/shared-clinic.htm'
        p1 = f'''<a href="{shared}">A</a>
            <a href="/bolnicy/?page=2" class="pagination-list__link">2</a>
            <a href="tel:+74957654321">x</a>'''
        p2 = f'''<a href="{shared}">B</a><a href="/bolnicy/p2-only.htm">C</a>
            <a href="tel:+74951112233">y</a>'''
        scraper = FakeScraper({
            f'https://{host}/bolnicy/': p1,
            f'https://{host}/bolnicy/?page=2': p2,
            f'https://{host}{shared}': _read('spravker_org_page.html'),
            f'https://{host}/bolnicy/p2-only.htm': _read('spravker_org_page.html'),
        })
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                scrape_spravker_category(scraper, 'msk', host, 'bolnicy')
            )
        finally:
            loop.close()
        shared_fetches = [u for u in scraper.fetches if u.endswith(shared)]
        assert len(shared_fetches) == 1, (
            f'shared org URL fetched {len(shared_fetches)} times, expected 1 '
            '(dedup broken)'
        )


class TestConfigDefaults:
    """Ensure the public caps are positive and match what callers expect."""

    def test_max_listing_pages_positive(self):
        assert SPRAVKER_MAX_LISTING_PAGES >= 5

    def test_max_org_pages_per_category_positive(self):
        assert SPRAVKER_MAX_ORG_PAGES_PER_CATEGORY >= 20

    def test_caps_are_a_strict_increase_over_baseline(self):
        # The pre-PR scraper hardcoded 10 org pages and no pagination.
        # If anyone ever shrinks these below baseline, fail loudly.
        assert SPRAVKER_MAX_ORG_PAGES_PER_CATEGORY > 10
        assert SPRAVKER_MAX_LISTING_PAGES > 1
