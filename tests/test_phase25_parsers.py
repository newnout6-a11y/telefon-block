"""Unit tests for Phase 2.5 reputation-source parsers.

Covers:
* kto.zvonil.tel — Bootstrap detail page with blockquote-shaped reviews +
  flat list of /+<phone> fan-out links.
* abonentik.ru   — Vue/Nuxt SSR detail page where the verdict is encoded in
  /category/<slug> links + numeric "X.X из 5" rating.
* badcall.ru     — Bootstrap detail page where the verdict is encoded in
  the Bootstrap colour class on each <li>: list-group-item-danger /
  list-group-item-warning / list-group-item-success.

Each parser is exercised against a small, hand-trimmed fixture so the test
runs offline and stays stable when the live site changes its inessential
markup. Fixtures live under ``tests/fixtures/phase25_sources/``.
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

FIXTURES = os.path.join(REPO_ROOT, "tests", "fixtures", "phase25_sources")


def _load(name: str) -> str:
    with open(os.path.join(FIXTURES, name), "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Source registration sanity
# ---------------------------------------------------------------------------

class TestSourceRegistration:
    def test_kto_zvonil_tel_in_sources(self):
        assert "kto_zvonil_tel" in crawler.SOURCES

    def test_abonentik_in_sources(self):
        assert "abonentik" in crawler.SOURCES

    def test_badcall_in_sources(self):
        assert "badcall" in crawler.SOURCES

    def test_phase2_sources_still_registered(self):
        # Don't accidentally remove Phase 2 entries when adding Phase 2.5.
        assert "callfilter_info" in crawler.SOURCES
        assert "scamcall" in crawler.SOURCES

    def test_kto_zvonil_tel_source_from_url(self):
        assert (
            crawler.source_from_url("https://kto.zvonil.tel/+79867257983")
            == "kto_zvonil_tel"
        )

    def test_abonentik_source_from_url(self):
        assert (
            crawler.source_from_url("https://abonentik.ru/nomer/79867257983")
            == "abonentik"
        )

    def test_badcall_source_from_url(self):
        assert (
            crawler.source_from_url("https://badcall.ru/phones/9867257983")
            == "badcall"
        )

    def test_kto_zvonil_tel_detail_url_keeps_leading_seven(self):
        # /+<11-digit, leading 7> — country code stays in the path.
        assert (
            crawler.detail_url("kto_zvonil_tel", "+74957730527")
            == "https://kto.zvonil.tel/+74957730527"
        )

    def test_kto_zvonil_tel_detail_url_pads_country_code(self):
        # Bare 10-digit input must be promoted to leading 7 to match the URL.
        assert (
            crawler.detail_url("kto_zvonil_tel", "9867257983")
            == "https://kto.zvonil.tel/+79867257983"
        )

    def test_abonentik_detail_url_keeps_leading_seven(self):
        assert (
            crawler.detail_url("abonentik", "+74957730527")
            == "https://abonentik.ru/nomer/74957730527"
        )

    def test_badcall_detail_url_strips_country_code(self):
        # badcall.ru uses 10-digit path (no leading 7).
        assert (
            crawler.detail_url("badcall", "+74957730527")
            == "https://badcall.ru/phones/4957730527"
        )

    def test_seed_urls_present(self):
        assert crawler.SEED_URLS["kto_zvonil_tel"]
        assert crawler.SEED_URLS["abonentik"]
        assert crawler.SEED_URLS["badcall"]


# ---------------------------------------------------------------------------
# kto.zvonil.tel
# ---------------------------------------------------------------------------

class TestKtoZvonilTelParser:
    @pytest.fixture
    def scam_page(self):
        return _load("kto_zvonil_tel_scam.html")

    @pytest.fixture
    def empty_page(self):
        return _load("kto_zvonil_tel_empty.html")

    def test_scam_page_extracts_one_block_row(self, scam_page):
        url = "https://kto.zvonil.tel/+79867257983"
        rows, _ = crawler.parse_page(url, scam_page)
        assert len(rows) == 1
        row, evidence = rows[0]
        assert row["normalized_number"] == "+79867257983"
        assert row["source"] == "kto_zvonil_tel"
        assert evidence["label_hint"] == "BLOCK"
        assert "мошенничество" in row["categories"]

    def test_scam_page_review_count_matches_title(self, scam_page):
        url = "https://kto.zvonil.tel/+79867257983"
        rows, _ = crawler.parse_page(url, scam_page)
        row, _ = rows[0]
        # Title says "Всего 4 отзывов" and there are 4 blockquotes — review
        # count must be at least 4.
        assert row["review_count"] >= 4

    def test_scam_page_yields_related_number_urls(self, scam_page):
        url = "https://kto.zvonil.tel/+79867257983"
        _, new_urls = crawler.parse_page(url, scam_page)
        assert any("/+74957730527" in u for u in new_urls)
        assert any("/+79615741094" in u for u in new_urls)
        assert all("kto.zvonil.tel/" in u for u in new_urls)

    def test_empty_page_yields_no_rows_but_keeps_fanout(self, empty_page):
        url = "https://kto.zvonil.tel/+79024209530"
        rows, new_urls = crawler.parse_page(url, empty_page)
        # No reviews and no fraud vocabulary → must not produce a row.
        assert rows == []
        # But the "Новые отзывы" fan-out URLs must still be discovered.
        assert any("/+79867257983" in u for u in new_urls)

    def test_sitemap_yields_only_urls(self):
        sitemap = _load("kto_zvonil_tel_sitemap.xml")
        url = "https://kto.zvonil.tel/sitemap.xml"
        rows, new_urls = crawler.parse_page(url, sitemap)
        assert rows == []
        assert any("/+79867257983" in u for u in new_urls)
        assert any("/+74957730527" in u for u in new_urls)


# ---------------------------------------------------------------------------
# abonentik.ru
# ---------------------------------------------------------------------------

class TestAbonentikParser:
    @pytest.fixture
    def scam_page(self):
        return _load("abonentik_phone_scam.html")

    @pytest.fixture
    def warn_page(self):
        return _load("abonentik_phone_warn.html")

    @pytest.fixture
    def unknown_page(self):
        return _load("abonentik_phone_unknown.html")

    def test_scam_page_extracts_block_row(self, scam_page):
        url = "https://abonentik.ru/nomer/79867257983"
        rows, _ = crawler.parse_page(url, scam_page)
        assert len(rows) == 1
        row, evidence = rows[0]
        assert row["normalized_number"] == "+79867257983"
        assert row["source"] == "abonentik"
        assert evidence["label_hint"] == "BLOCK"
        assert evidence["evidence_type"] == "blacklist"

    def test_scam_page_categories_mapped_from_slugs(self, scam_page):
        url = "https://abonentik.ru/nomer/79867257983"
        rows, _ = crawler.parse_page(url, scam_page)
        row, _ = rows[0]
        # Three BLOCK-categories on the page: moshennichestvo-telefonnoe /
        # fishing / vymogatelstvo. Each must surface in the row's
        # categories field via the slug→label map.
        cats = row["categories"]
        assert "телефонное мошенничество" in cats
        assert "фишинг" in cats
        assert "вымогательство" in cats

    def test_scam_page_confidence_boosted_by_low_rating(self, scam_page):
        url = "https://abonentik.ru/nomer/79867257983"
        rows, _ = crawler.parse_page(url, scam_page)
        row, _ = rows[0]
        # Multiple BLOCK categories + "1.5 из 5" rating → confidence ≥ 0.86.
        assert float(row["source_confidence"]) >= 0.86

    def test_warn_page_yields_warn_row(self, warn_page):
        url = "https://abonentik.ru/nomer/79688770249"
        rows, _ = crawler.parse_page(url, warn_page)
        assert len(rows) == 1
        row, evidence = rows[0]
        assert evidence["label_hint"] == "WARN"
        assert "спам" in row["categories"] or "реклама" in row["categories"]

    def test_unknown_page_yields_no_rows(self, unknown_page):
        url = "https://abonentik.ru/nomer/74957730527"
        rows, _ = crawler.parse_page(url, unknown_page)
        # /category/neizvestny is intentionally NOT mapped — must reject.
        assert rows == []

    def test_category_slug_map_covers_known_block_terms(self):
        # Belt-and-braces sanity check: the parser's category map MUST
        # contain at least these slugs (otherwise high-value spam pages
        # would be silently downgraded).
        for slug in (
            "moshennichestvo-telefonnoe",
            "fishing",
            "vymogatelstvo",
            "ugrozy",
        ):
            assert slug in crawler.ABONENTIK_CATEGORY_MAP
            assert crawler.ABONENTIK_CATEGORY_MAP[slug][1] == "BLOCK"

    def test_sitemap_yields_only_urls(self):
        sitemap = _load("abonentik_sitemap.xml")
        url = "https://abonentik.ru/sitemap.xml"
        rows, new_urls = crawler.parse_page(url, sitemap)
        assert rows == []
        assert any("/nomer/79867257983" in u for u in new_urls)


# ---------------------------------------------------------------------------
# badcall.ru
# ---------------------------------------------------------------------------

class TestBadcallParser:
    @pytest.fixture
    def reviewed_page(self):
        return _load("badcall_phone_reviewed.html")

    @pytest.fixture
    def unknown_page(self):
        return _load("badcall_phone_unknown.html")

    def test_reviewed_page_extracts_block_row(self, reviewed_page):
        url = "https://badcall.ru/phones/9867257983"
        rows, _ = crawler.parse_page(url, reviewed_page)
        assert len(rows) == 1
        row, evidence = rows[0]
        assert row["normalized_number"] == "+79867257983"
        assert row["source"] == "badcall"
        assert evidence["label_hint"] == "BLOCK"

    def test_reviewed_page_counts_danger_and_warning_items(self, reviewed_page):
        url = "https://badcall.ru/phones/9867257983"
        rows, _ = crawler.parse_page(url, reviewed_page)
        row, _ = rows[0]
        # Fixture has 2 danger + 1 warning items → review_count ≥ 3.
        assert row["review_count"] >= 3
        assert row["negative_count"] >= 2

    def test_reviewed_page_categories_include_fraud(self, reviewed_page):
        url = "https://badcall.ru/phones/9867257983"
        rows, _ = crawler.parse_page(url, reviewed_page)
        row, _ = rows[0]
        # FRAUD_PATTERNS hits ('мошенник', 'обман', etc.) must promote the
        # categories list with 'мошенничество'.
        assert "мошенничество" in row["categories"]

    def test_reviewed_page_yields_related_number_urls(self, reviewed_page):
        url = "https://badcall.ru/phones/9867257983"
        _, new_urls = crawler.parse_page(url, reviewed_page)
        assert any("/phones/9867257984" in u for u in new_urls)
        assert any("/phones/4957730527" in u for u in new_urls)

    def test_unknown_page_yields_no_rows(self, unknown_page):
        url = "https://badcall.ru/phones/4957730527"
        rows, _ = crawler.parse_page(url, unknown_page)
        # "Неизвестный номер!" stub → must reject (no provenance).
        assert rows == []

    def test_unknown_page_still_yields_fanout(self, unknown_page):
        url = "https://badcall.ru/phones/4957730527"
        _, new_urls = crawler.parse_page(url, unknown_page)
        # Even when rejecting the row, the "Соседние номера" fan-out URLs
        # are still useful for discovery.
        assert any("/phones/4957730528" in u for u in new_urls)

    def test_reviewed_page_not_rejected_by_sitewide_footer(self, reviewed_page):
        # Regression: the site renders an "Неизвестный номер!" footer card
        # on EVERY page, including pages with hundreds of reviews. A naive
        # `if 'Неизвестный номер' in html: return []` short-circuit would
        # silently throw away all rich pages. The parser must instead key
        # off the absence of list-group-item-{danger,warning} <li> tags.
        assert "Неизвестный номер!" in reviewed_page
        url = "https://badcall.ru/phones/9867257983"
        rows, _ = crawler.parse_page(url, reviewed_page)
        assert len(rows) == 1, (
            "Site-wide 'Неизвестный номер!' footer card must not cause "
            "rich pages to be rejected."
        )


# ---------------------------------------------------------------------------
# Cross-source invariants
# ---------------------------------------------------------------------------

class TestNoSpuriousAllow:
    """Reputation feeds must NEVER produce ALLOW rows.

    All three new sources catalogue complaints, not legitimate businesses.
    A leak of ALLOW from these feeds into the consolidated dataset would
    poison the negative-set for future model retraining.
    """

    def test_kto_zvonil_tel_never_yields_allow(self):
        scam = _load("kto_zvonil_tel_scam.html")
        empty = _load("kto_zvonil_tel_empty.html")
        for url, html in [
            ("https://kto.zvonil.tel/+79867257983", scam),
            ("https://kto.zvonil.tel/+79024209530", empty),
        ]:
            rows, _ = crawler.parse_page(url, html)
            for _, ev in rows:
                assert ev["label_hint"] in {"BLOCK", "WARN"}, (
                    f"Spurious ALLOW from kto_zvonil_tel: {ev}"
                )

    def test_abonentik_never_yields_allow(self):
        for fname, url in [
            ("abonentik_phone_scam.html", "https://abonentik.ru/nomer/79867257983"),
            ("abonentik_phone_warn.html", "https://abonentik.ru/nomer/79688770249"),
            ("abonentik_phone_unknown.html", "https://abonentik.ru/nomer/74957730527"),
        ]:
            rows, _ = crawler.parse_page(url, _load(fname))
            for _, ev in rows:
                assert ev["label_hint"] in {"BLOCK", "WARN"}, (
                    f"Spurious ALLOW from abonentik: {ev}"
                )

    def test_badcall_never_yields_allow(self):
        for fname, url in [
            ("badcall_phone_reviewed.html", "https://badcall.ru/phones/9867257983"),
            ("badcall_phone_unknown.html", "https://badcall.ru/phones/4957730527"),
        ]:
            rows, _ = crawler.parse_page(url, _load(fname))
            for _, ev in rows:
                assert ev["label_hint"] in {"BLOCK", "WARN"}, (
                    f"Spurious ALLOW from badcall: {ev}"
                )
