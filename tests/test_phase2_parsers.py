"""Unit tests for Phase 2 reputation-source parsers.

Covers:
* callfilter.info  (rich detail page with status-label + comments)
* scamcall.ru      (Vue/Nuxt SPA with main commentary + sidebar swiper)

Each parser is exercised against a small, hand-trimmed fixture so the test
runs offline and stays stable when the live site changes its inessential
markup. Fixtures live under ``tests/fixtures/phase2_sources/``.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import ru_reputation_crawler as crawler  # noqa: E402

FIXTURES = os.path.join(REPO_ROOT, "tests", "fixtures", "phase2_sources")


def _load(name: str) -> str:
    with open(os.path.join(FIXTURES, name), "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Source registration sanity
# ---------------------------------------------------------------------------

class TestSourceRegistration:
    def test_callfilter_info_in_sources(self):
        assert "callfilter_info" in crawler.SOURCES

    def test_scamcall_in_sources(self):
        assert "scamcall" in crawler.SOURCES

    def test_callfilter_info_distinct_from_callfilter(self):
        # Phase 1 already had `callfilter` (callfilter.app); Phase 2 adds
        # `callfilter_info` for callfilter.info — make sure the existing
        # source survives.
        assert "callfilter" in crawler.SOURCES
        assert "callfilter_info" in crawler.SOURCES
        assert (
            crawler.source_from_url("https://callfilter.app/74957730527")
            == "callfilter"
        )
        assert (
            crawler.source_from_url("https://callfilter.info/number/74957730527")
            == "callfilter_info"
        )

    def test_scamcall_source_from_url(self):
        assert (
            crawler.source_from_url("https://scamcall.ru/phone/9584069694")
            == "scamcall"
        )

    def test_callfilter_info_detail_url(self):
        assert (
            crawler.detail_url("callfilter_info", "+74957730527")
            == "https://callfilter.info/number/74957730527"
        )

    def test_scamcall_detail_url_strips_country_code(self):
        # scamcall.ru uses a 10-digit path (no leading 7).
        assert (
            crawler.detail_url("scamcall", "+74957730527")
            == "https://scamcall.ru/phone/4957730527"
        )

    def test_seed_urls_present(self):
        assert crawler.SEED_URLS["callfilter_info"]
        assert crawler.SEED_URLS["scamcall"]


# ---------------------------------------------------------------------------
# callfilter.info
# ---------------------------------------------------------------------------

class TestCallfilterInfoParser:
    @pytest.fixture
    def scam_page(self):
        return _load("callfilter_info_scam.html")

    @pytest.fixture
    def safe_page(self):
        return _load("callfilter_info_safe.html")

    def test_scam_page_extracts_one_block_row(self, scam_page):
        url = "https://callfilter.info/number/74957730527"
        rows, new_urls = crawler.parse_page(url, scam_page)
        assert len(rows) == 1
        row, evidence = rows[0]
        assert row["normalized_number"] == "+74957730527"
        assert row["source"] == "callfilter_info"
        assert evidence["label_hint"] == "BLOCK"
        assert evidence["evidence_type"] == "blacklist"
        assert "мошенничество" in row["categories"]

    def test_scam_page_yields_related_number_urls(self, scam_page):
        url = "https://callfilter.info/number/74957730527"
        _, new_urls = crawler.parse_page(url, scam_page)
        # Sidebar contains links to 79676567890 / 79016288450 / 74995305690.
        assert any("79016288450" in u for u in new_urls)
        assert any("74995305690" in u for u in new_urls)
        assert all("callfilter.info/number/" in u for u in new_urls)

    def test_scam_page_keeps_review_count_and_views(self, scam_page):
        url = "https://callfilter.info/number/74957730527"
        rows, _ = crawler.parse_page(url, scam_page)
        row, _ = rows[0]
        assert row["review_count"] >= 766
        assert row["view_count"] == 2124

    def test_scam_page_confidence_high(self, scam_page):
        url = "https://callfilter.info/number/74957730527"
        rows, _ = crawler.parse_page(url, scam_page)
        row, _ = rows[0]
        # status-scam → 0.85 confidence.
        assert float(row["source_confidence"]) >= 0.80

    def test_safe_page_yields_no_rows(self, safe_page):
        url = "https://callfilter.info/number/78001000600"
        rows, _ = crawler.parse_page(url, safe_page)
        # status-allow + no fraud signals → must not promote to BLOCK/WARN.
        assert rows == []

    def test_sitemap_yields_only_urls(self):
        url = "https://callfilter.info/sitemap.xml"
        sm = _load("callfilter_info_sitemap.xml")
        rows, new_urls = crawler.parse_page(url, sm)
        assert rows == []
        assert len(new_urls) == 3
        assert all("/number/" in u for u in new_urls)


# ---------------------------------------------------------------------------
# scamcall.ru
# ---------------------------------------------------------------------------

class TestScamcallParser:
    @pytest.fixture
    def detail_page(self):
        return _load("scamcall_phone.html")

    def test_main_commentary_extracts_block(self, detail_page):
        url = "https://scamcall.ru/phone/9584069694"
        rows, _ = crawler.parse_page(url, detail_page)
        # The main commentary_item is for 9584069694 (scammers).
        main_rows = [r for r, _ in rows if r["normalized_number"] == "+79584069694"]
        assert main_rows, "main number row missing"
        assert "мошенничество" in main_rows[0]["categories"]

    def test_swiper_extracts_block_and_warn_rows(self, detail_page):
        url = "https://scamcall.ru/phone/9584069694"
        rows, _ = crawler.parse_page(url, detail_page)
        rows_by_number = {r["normalized_number"]: (r, ev) for r, ev in rows}
        assert "+79046200203" in rows_by_number  # scammers
        assert "+78125663179" in rows_by_number  # scammers
        assert "+74951098498" in rows_by_number  # advertising
        assert rows_by_number["+79046200203"][1]["label_hint"] == "BLOCK"
        assert rows_by_number["+74951098498"][1]["label_hint"] == "WARN"

    def test_swiper_filters_positively_and_unknown(self, detail_page):
        # The fixture has a "positively" entry (4994554827) and an "unknown"
        # entry (8005337732). Neither must produce a row — they're too weak
        # signals for a reputation-only feed.
        url = "https://scamcall.ru/phone/9584069694"
        rows, _ = crawler.parse_page(url, detail_page)
        numbers = {r["normalized_number"] for r, _ in rows}
        assert "+74994554827" not in numbers  # positively
        assert "+78005337732" not in numbers  # unknown

    def test_swiper_yields_followup_urls(self, detail_page):
        url = "https://scamcall.ru/phone/9584069694"
        _, new_urls = crawler.parse_page(url, detail_page)
        assert any("/phone/9046200203" in u for u in new_urls)
        assert any("/phone/4951098498" in u for u in new_urls)
        # 10-digit, no country code in scamcall.ru paths.
        for u in new_urls:
            tail = u.rsplit("/phone/", 1)[-1].rstrip("/")
            assert tail.isdigit() and 9 <= len(tail) <= 11

    def test_sitemap_index_yields_subsitemaps(self):
        url = "https://scamcall.ru/sitemap.xml"
        sm = _load("scamcall_sitemap.xml")
        rows, new_urls = crawler.parse_page(url, sm)
        assert rows == []
        assert any(u.endswith("sitemap1.xml.gz") for u in new_urls)
        assert any(u.endswith("sitemap2.xml.gz") for u in new_urls)


# ---------------------------------------------------------------------------
# Cross-cutting safety — never produce ALLOW rows from these reputation feeds.
# ---------------------------------------------------------------------------

class TestNoAllowFromReputationFeeds:
    def test_callfilter_info_never_produces_allow(self):
        for name in ("callfilter_info_scam.html", "callfilter_info_safe.html"):
            html = _load(name)
            url = "https://callfilter.info/number/74957730527"
            rows, _ = crawler.parse_page(url, html)
            for _, ev in rows:
                assert ev["label_hint"] in {"BLOCK", "WARN"}

    def test_scamcall_never_produces_allow(self):
        url = "https://scamcall.ru/phone/9584069694"
        rows, _ = crawler.parse_page(url, _load("scamcall_phone.html"))
        for _, ev in rows:
            assert ev["label_hint"] in {"BLOCK", "WARN"}
