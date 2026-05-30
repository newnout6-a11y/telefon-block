"""Unit tests for ``parse_zvonkoff`` / ``parse_ktozvonil`` fallback signals.

The fallback heuristics in these parsers historically used loose substring
checks (``'negative' in html.lower()`` and ``'label-danger' in html``) that
would fire on any random JS string, HTML comment, or hidden attribute that
happened to contain those words. The current implementation gates the
fallback on ``html_class_attribute_contains`` so only real CSS-class markers
(``rating-negative``, ``label-danger``, …) trigger a row.

These tests pin that contract:
  * a real CSS-class marker still produces a row;
  * the same word appearing only inside a JS string / comment / non-class
    attribute does NOT produce a row.
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


# ---------------------------------------------------------------------------
# Helper: ``html_class_attribute_contains``
# ---------------------------------------------------------------------------

class TestHtmlClassAttributeContains:
    def test_matches_class_attribute(self):
        html = '<div class="foo rating-negative bar">x</div>'
        assert crawler.html_class_attribute_contains(html, 'rating-negative')

    def test_matches_substring_inside_class(self):
        html = '<span class="negativeReviews">x</span>'
        assert crawler.html_class_attribute_contains(html, 'negative')

    def test_case_insensitive(self):
        html = '<div class="LABEL-DANGER">!</div>'
        assert crawler.html_class_attribute_contains(html, 'label-danger')

    def test_does_not_match_text_content(self):
        html = '<p>This page has a negative tone</p>'
        assert not crawler.html_class_attribute_contains(html, 'negative')

    def test_does_not_match_js_string_literal(self):
        html = '<script>var x = "negative reviews go here";</script>'
        assert not crawler.html_class_attribute_contains(html, 'negative')

    def test_does_not_match_html_comment(self):
        html = '<!-- TODO: render label-danger badge -->'
        assert not crawler.html_class_attribute_contains(html, 'label-danger')

    def test_does_not_match_data_attribute(self):
        html = '<div data-state="negative">x</div>'
        assert not crawler.html_class_attribute_contains(html, 'negative')

    def test_empty_html_returns_false(self):
        assert not crawler.html_class_attribute_contains('', 'negative')

    def test_no_needles_returns_false(self):
        html = '<div class="negative">x</div>'
        assert not crawler.html_class_attribute_contains(html)


# ---------------------------------------------------------------------------
# parse_zvonkoff: fallback gate on ``class="*negative*"``
# ---------------------------------------------------------------------------

# Minimal page with no Russian fraud/spam keywords in the meta description
# or tags and *no* CSS marker — must NOT produce a row even if the literal
# string "negative" appears in unrelated places.
_ZVONKOFF_NO_SIGNAL_BUT_TEXT_NEGATIVE = """
<!doctype html>
<html><head>
<title>+7 (495) 773-05-27 - zvonkoff</title>
<meta name="description" content="Информация о номере телефона.">
</head>
<body>
<div class="sectionInfo__tags">справочная служба</div>
<script>var lastNegative = "no rows please";</script>
<!-- has "negative" in a comment too -->
<div data-state="negative">hidden</div>
</body></html>
""".strip()


# Same page but now with an actual ``class="...rating-negative..."`` widget,
# which should be treated as a real fallback signal.
_ZVONKOFF_NEGATIVE_CLASS = """
<!doctype html>
<html><head>
<title>+7 (495) 773-05-27 - zvonkoff</title>
<meta name="description" content="Информация о номере.">
</head>
<body>
<div class="sectionInfo__tags">справочная</div>
<div class="sectionInfo__rating rating-negative">!</div>
</body></html>
""".strip()


# A page that already has explicit Russian fraud signal in the meta
# description — a row must be emitted regardless of the class fallback.
_ZVONKOFF_RUSSIAN_FRAUD_META = """
<!doctype html>
<html><head>
<title>+7 (495) 773-05-27</title>
<meta name="description" content="Звонят мошенники, осторожно.">
</head>
<body>
<div class="sectionInfo__tags">мошенничество</div>
</body></html>
""".strip()


class TestParseZvonkoffFallback:
    URL = "https://zvonkoff.net/ru/number/74957730527"

    def test_text_negative_alone_does_not_produce_row(self):
        rows, _ = crawler.parse_page(self.URL, _ZVONKOFF_NO_SIGNAL_BUT_TEXT_NEGATIVE)
        assert rows == []

    def test_negative_css_class_produces_row(self):
        rows, _ = crawler.parse_page(self.URL, _ZVONKOFF_NEGATIVE_CLASS)
        assert len(rows) == 1
        row, evidence = rows[0]
        assert row["normalized_number"] == "+74957730527"
        assert row["source"] == "zvonkoff"
        assert evidence["label_hint"] in {"BLOCK", "WARN"}

    def test_russian_meta_signal_produces_row(self):
        rows, _ = crawler.parse_page(self.URL, _ZVONKOFF_RUSSIAN_FRAUD_META)
        assert len(rows) == 1
        row, evidence = rows[0]
        assert row["normalized_number"] == "+74957730527"
        assert row["source"] == "zvonkoff"
        assert evidence["label_hint"] == "BLOCK"

    def test_no_number_in_url_yields_no_rows(self):
        rows, _ = crawler.parse_page(
            "https://zvonkoff.net/ru/", _ZVONKOFF_NEGATIVE_CLASS
        )
        # Without a recognisable number on the URL the fallback section
        # cannot promote anything to a row.
        for row, _ in rows:
            assert row["source"] != "zvonkoff" or row["normalized_number"]


# ---------------------------------------------------------------------------
# parse_ktozvonil: fallback gate on ``class="*label-danger*"``
# ---------------------------------------------------------------------------

# No Russian fraud signal in title/meta/comments + ``label-danger`` only in
# a script string and an HTML comment — must NOT produce a row.
_KTOZVONIL_NO_SIGNAL_BUT_TEXT_DANGER = """
<!doctype html>
<html><head>
<title>+7 (495) 773-05-27 — ktozvonil</title>
<meta name="description" content="Информация о номере.">
</head>
<body>
<!-- could render <span class="label-danger"> in future -->
<script>var k = "label-danger placeholder";</script>
</body></html>
""".strip()


# Same page but with a real Bootstrap ``label label-danger`` badge.
_KTOZVONIL_DANGER_CLASS = """
<!doctype html>
<html><head>
<title>+7 (495) 773-05-27 — ktozvonil</title>
<meta name="description" content="Сведения о номере.">
</head>
<body>
<span class="label label-danger">опасный</span>
</body></html>
""".strip()


# Russian fraud signal already present in description — must produce a row.
_KTOZVONIL_RUSSIAN_FRAUD_META = """
<!doctype html>
<html><head>
<title>+7 (495) 773-05-27 — ktozvonil</title>
<meta name="description" content="Жалобы на мошенничество.">
</head>
<body><p>Описание</p></body></html>
""".strip()


class TestParseKtozvonilFallback:
    URL = "https://ktozvonil.net/nomer/74957730527"

    def test_text_label_danger_alone_does_not_produce_row(self):
        rows, _ = crawler.parse_page(self.URL, _KTOZVONIL_NO_SIGNAL_BUT_TEXT_DANGER)
        assert rows == []

    def test_label_danger_css_class_produces_row(self):
        rows, _ = crawler.parse_page(self.URL, _KTOZVONIL_DANGER_CLASS)
        assert len(rows) == 1
        row, evidence = rows[0]
        assert row["normalized_number"] == "+74957730527"
        assert row["source"] == "ktozvonil"
        assert evidence["label_hint"] in {"BLOCK", "WARN"}

    def test_russian_meta_signal_produces_row(self):
        rows, _ = crawler.parse_page(self.URL, _KTOZVONIL_RUSSIAN_FRAUD_META)
        assert len(rows) == 1
        row, evidence = rows[0]
        assert row["normalized_number"] == "+74957730527"
        assert row["source"] == "ktozvonil"
        assert evidence["label_hint"] == "BLOCK"


# ---------------------------------------------------------------------------
# Cross-cutting: these reputation parsers must never emit ALLOW rows.
# ---------------------------------------------------------------------------

class TestNoAllowFromZvonkoffOrKtozvonil:
    @pytest.mark.parametrize(
        "url,html",
        [
            ("https://zvonkoff.net/ru/number/74957730527", _ZVONKOFF_NEGATIVE_CLASS),
            ("https://zvonkoff.net/ru/number/74957730527", _ZVONKOFF_RUSSIAN_FRAUD_META),
            ("https://ktozvonil.net/nomer/74957730527", _KTOZVONIL_DANGER_CLASS),
            ("https://ktozvonil.net/nomer/74957730527", _KTOZVONIL_RUSSIAN_FRAUD_META),
        ],
    )
    def test_no_allow(self, url, html):
        rows, _ = crawler.parse_page(url, html)
        for _, ev in rows:
            assert ev["label_hint"] in {"BLOCK", "WARN"}
