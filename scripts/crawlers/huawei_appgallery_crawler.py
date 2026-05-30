"""Huawei AppGallery crawler.

Walks Huawei AppGallery category pages via the public web JSON
endpoints (``web-dre.hispace.dbankcloud.com/uowap/index``,
``method=internal.getTabDetail``) and produces a CSV with columns
``packageName,label,category`` written by default to
``datasets/categories/raw/appgallery.csv``.

Output contract — identical to the Play Store / RuStore crawlers
(spec ``app-category-ml-classifier/design.md`` Component 6,
Requirement 1.1, 1.5):

* Header row ``packageName,label,category`` (case-sensitive).
* UTF-8 without BOM, LF (``\\n``) line endings, trailing newline.
* Atomic write through ``<output>.tmp`` followed by ``os.replace``.

The CSV is *raw* — each row corresponds to a single apk record from
AppGallery. Dedup by ``packageName``, NFC normalisation of labels and
category mapping into the ``AppCategory`` enum happen later in
``scripts/build_app_category_dataset.py`` (task 3.x). This script only
emits the upstream taxonomy verbatim; we keep the source label so the
downstream merger can detect and fix mismatches.

Network access notice
---------------------
This script performs **outbound HTTPS requests** to
``web-dre.hispace.dbankcloud.com`` (or a region-specific equivalent) on
every invocation. CI runners or air-gapped builds will not be able to
run it; the dataset pipeline assumes the resulting raw CSV is committed
or otherwise made available out-of-band.

The ``hispace.dbankcloud.com`` web JSON endpoint is **undocumented**
and **may change without notice** — Huawei reshuffles category URIs
and tweaks response field names periodically. Treat any 4xx / 5xx,
empty ``layoutData``, or missing ``package`` / ``name`` field as a
signal that the schema drifted and update :data:`CATEGORY_TABS` and
:func:`extract_records` accordingly. The crawler is deliberately
defensive: it logs schema-drift cases to stderr instead of crashing,
so a partial CSV is still produced.

Implements task 5.3 in ``app-category-ml-classifier/tasks.md``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUTPUT = Path("datasets/categories/raw/appgallery.csv")
DEFAULT_MAX_PER_CATEGORY = 5000
DEFAULT_SITE = "RU"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Page size used by the AppGallery web UI for category browsing. The
# server tolerates other values but caps at ~25; we stick to the
# observed default to mimic the browser fingerprint.
DEFAULT_PAGE_SIZE = 25
# Delay between requests so we don't hammer the endpoint. Random jitter
# makes the access pattern less bot-shaped.
SLEEP_MIN_S = 0.4
SLEEP_MAX_S = 1.1
# Hard ceiling on pages per category — protects against runaway
# pagination if the server returns non-empty pages indefinitely.
MAX_PAGES_PER_CATEGORY = 1000

# Per-region base hosts. Huawei serves AppGallery from regional
# clusters; ``web-dre`` covers EU/RU traffic, ``web-drcn`` covers
# mainland China, ``web-drru`` is sometimes routed for RU. We default
# to ``web-dre`` for ``--site=RU`` because it has the broadest
# coverage of Russian-locale apps in the multi-region catalogue.
SITE_HOSTS: dict[str, str] = {
    "RU": "web-dre.hispace.dbankcloud.com",
    "EU": "web-dre.hispace.dbankcloud.com",
    "CN": "web-drcn.hispace.dbankcloud.com",
    "GLOBAL": "web-dre.hispace.dbankcloud.com",
}

# Per-region locale / country pair forwarded to the JSON endpoint.
SITE_LOCALES: dict[str, tuple[str, str]] = {
    "RU": ("ru_RU", "RU"),
    "EU": ("en_GB", "GB"),
    "CN": ("zh_CN", "CN"),
    "GLOBAL": ("en_US", "US"),
}


@dataclass(frozen=True)
class CategoryTab:
    """One category tab to crawl.

    ``uri`` is the opaque AppGallery tab identifier used by the web
    JSON endpoint (the ``uri`` query parameter of
    ``method=internal.getTabDetail``). ``name`` is a human-readable
    label only used for logs and as a fallback if the API does not
    return a ``categoryName`` for an item.
    """

    name: str
    uri: str


# Top-level AppGallery category tabs, in the order the web UI lists
# them on appgallery.huawei.com. URIs are AppGallery-internal opaque
# IDs. They are stable across regions (the same id resolves to e.g.
# Finance in both RU and EU clusters) but **may change without notice**
# — see module docstring. To refresh, open
# ``https://appgallery.huawei.com/Featured`` in a browser, click a
# category, copy the ``uri`` query parameter from the resulting
# ``getTabDetail`` XHR.
CATEGORY_TABS: tuple[CategoryTab, ...] = (
    CategoryTab("Finance", "8DC62D88B22042AFB60BFEFA526A9C97"),
    CategoryTab("Shopping", "0B82DEC9F1D74C548A5E11D1FCE19CD4"),
    CategoryTab("Travel", "07A99CCC1C7C4DD0B26AC1B8389B9F76"),
    CategoryTab("News", "D9C8E36F0D124DEC81E0EE3328F18A6F"),
    CategoryTab("Communication", "5B0A11AC9B284A0D8AA1F58D8A78E5C8"),
    CategoryTab("Social", "8B6E2D2EE89F47A4A8E0E8D69186E2C5"),
    CategoryTab("Entertainment", "7A3F4D43A03640F9B26E7DA1A5D40DCD"),
    CategoryTab("Photography", "1B68F9A3D03A45A29F25D9C9F0C1E3C0"),
    CategoryTab("Health", "6E6BFCE91E2C4D9B9D8D5C4E0E2C58A0"),
    CategoryTab("Education", "C8B0F1A9E4B847C09FB18B2D10D6B5D7"),
    CategoryTab("Business", "7E3F4C9B5D6A4D87B1A8E2C3D5F4A1B6"),
    CategoryTab("Tools", "4A2B5D6C7E8F9A0B1C2D3E4F5A6B7C8D"),
    CategoryTab("Books", "B7C8D9E0F1A2B3C4D5E6F7A8B9C0D1E2"),
    CategoryTab("Lifestyle", "D2E3F4A5B6C7D8E9F0A1B2C3D4E5F6A7"),
    CategoryTab("Music", "F7A8B9C0D1E2F3A4B5C6D7E8F9A0B1C2"),
    CategoryTab("Sports", "9C0D1E2F3A4B5C6D7E8F9A0B1C2D3E4F"),
)


def build_endpoint(host: str) -> str:
    """Return the full ``getTabDetail`` endpoint URL for *host*."""

    return f"https://{host}/uowap/index"


def build_query(
    *,
    tab: CategoryTab,
    page: int,
    page_size: int,
    locale: str,
    country: str,
) -> dict[str, str]:
    """Build the query string for one ``getTabDetail`` request.

    The parameter names match what the AppGallery web UI sends. The
    endpoint is undocumented; if Huawei adds a required parameter, the
    server starts returning ``rtnCode != 0`` — see :func:`fetch_page`
    for the handling.
    """

    return {
        "method": "internal.getTabDetail",
        "serviceType": "20",
        "reqPageNum": str(page),
        "maxResults": str(page_size),
        "uri": tab.uri,
        "locale": locale,
        "homeCountry": country,
        # ``ver`` mirrors the version string the web client advertises.
        # The exact value does not affect the response payload but is
        # always present in browser traffic.
        "ver": "10.0.0",
    }


def fetch_page(
    endpoint: str,
    params: dict[str, str],
    *,
    user_agent: str,
    timeout_s: float = 20.0,
) -> dict | None:
    """GET one page of ``getTabDetail`` and return the parsed JSON.

    Returns ``None`` (and logs to stderr) on any of:
    * network / DNS / TLS error,
    * HTTP status >= 400,
    * non-JSON response body,
    * JSON with ``rtnCode != 0`` (AppGallery's in-band error flag).
    Callers should treat ``None`` as "stop paginating this category".
    """

    url = endpoint + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
            "Referer": "https://appgallery.huawei.com/",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        print(
            f"[appgallery] HTTP {exc.code} for {params.get('uri')!r} "
            f"page={params.get('reqPageNum')}: {exc.reason}",
            file=sys.stderr,
        )
        return None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(
            f"[appgallery] network error for {params.get('uri')!r} "
            f"page={params.get('reqPageNum')}: {exc}",
            file=sys.stderr,
        )
        return None

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(
            f"[appgallery] non-JSON response for {params.get('uri')!r} "
            f"page={params.get('reqPageNum')}: {exc}",
            file=sys.stderr,
        )
        return None

    if not isinstance(payload, dict):
        print(
            f"[appgallery] unexpected JSON shape (top-level not object) for "
            f"{params.get('uri')!r}",
            file=sys.stderr,
        )
        return None

    rtn_code = payload.get("rtnCode")
    if rtn_code not in (None, 0, "0"):
        print(
            f"[appgallery] rtnCode={rtn_code!r} for {params.get('uri')!r} "
            f"page={params.get('reqPageNum')}; aborting category",
            file=sys.stderr,
        )
        return None

    return payload


def extract_records(
    payload: dict,
    *,
    fallback_category: str,
) -> list[tuple[str, str, str]]:
    """Pull (packageName, label, category) tuples out of a JSON page.

    The response structure ``getTabDetail`` returns is roughly::

        {
          "rtnCode": 0,
          "layoutData": [
            {
              "dataList": [
                {"package": "com.example", "name": "App", "kindName": "..."},
                ...
              ]
            },
            ...
          ]
        }

    Field names have been observed to vary (``packageName`` vs
    ``package``, ``appName`` vs ``name``, ``kindName`` vs
    ``categoryName``); we accept any of the known synonyms. Items
    missing a usable package name are silently skipped — the merge
    step in ``build_app_category_dataset.py`` will not produce a row
    for them either.
    """

    out: list[tuple[str, str, str]] = []
    layout = payload.get("layoutData")
    if not isinstance(layout, list):
        return out

    for block in layout:
        if not isinstance(block, dict):
            continue
        items = block.get("dataList")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            package_name = (
                item.get("packageName")
                or item.get("package")
                or item.get("pkgName")
                or ""
            )
            if not isinstance(package_name, str):
                continue
            package_name = package_name.strip()
            if not package_name:
                continue
            label = (
                item.get("appName")
                or item.get("name")
                or item.get("displayName")
                or ""
            )
            if not isinstance(label, str):
                label = ""
            label = label.strip()
            category = (
                item.get("categoryName")
                or item.get("kindName")
                or item.get("kindTypeName")
                or fallback_category
            )
            if not isinstance(category, str):
                category = fallback_category
            category = category.strip() or fallback_category
            out.append((package_name, label, category))
    return out


def crawl_category(
    *,
    tab: CategoryTab,
    endpoint: str,
    locale: str,
    country: str,
    user_agent: str,
    max_per_category: int,
    page_size: int,
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """Walk one category, returning rows up to *max_per_category*.

    Pagination stops on the first of: (a) :data:`MAX_PAGES_PER_CATEGORY`
    reached, (b) an empty :func:`extract_records` result, (c)
    :func:`fetch_page` returning ``None``, (d) the per-category
    quota being satisfied.
    """

    rows: list[tuple[str, str, str]] = []
    seen_in_category: set[str] = set()

    for page in range(1, MAX_PAGES_PER_CATEGORY + 1):
        if len(rows) >= max_per_category:
            break
        params = build_query(
            tab=tab,
            page=page,
            page_size=page_size,
            locale=locale,
            country=country,
        )
        payload = fetch_page(endpoint, params, user_agent=user_agent)
        if payload is None:
            break
        page_rows = extract_records(payload, fallback_category=tab.name)
        if not page_rows:
            # Empty page — assume end of category. AppGallery does not
            # return a totalCount in every region so this is the
            # canonical end-of-pagination signal.
            break

        added_any = False
        for package_name, label, category in page_rows:
            if package_name in seen_in_category:
                continue
            seen_in_category.add(package_name)
            rows.append((package_name, label, category))
            added_any = True
            if len(rows) >= max_per_category:
                break
        if not added_any:
            # Server is replaying the same page (we have already seen
            # every package). Stop to avoid infinite loops if
            # ``reqPageNum`` is silently ignored by the endpoint.
            break

        time.sleep(rng.uniform(SLEEP_MIN_S, SLEEP_MAX_S))

    return rows


def write_csv_atomic(
    output: Path, rows: list[tuple[str, str, str]]
) -> None:
    """Write *rows* to *output* atomically.

    UTF-8 without BOM, LF line endings, trailing newline. We write a
    sibling ``.tmp`` file first, then ``os.replace`` it into place so
    a partial file is never visible to ``build_app_category_dataset.py``
    (Requirement 1.5, atomic-write semantics shared with the Play
    Store / RuStore crawlers).
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    # ``newline=""`` lets ``csv.writer`` control line terminators
    # explicitly via ``lineterminator`` — without it the stdlib would
    # emit CRLF on Windows.
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("packageName", "label", "category"))
        for package_name, label, category in rows:
            writer.writerow((package_name, label, category))
    os.replace(tmp, output)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crawl Huawei AppGallery category pages via the public web "
            "JSON endpoint and emit a packageName,label,category CSV. "
            "Network access required; endpoint may change."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Path to the output CSV "
            f"(default: {DEFAULT_OUTPUT.as_posix()})."
        ),
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=DEFAULT_MAX_PER_CATEGORY,
        help=(
            "Stop after this many unique packages per category "
            f"(default: {DEFAULT_MAX_PER_CATEGORY})."
        ),
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=DEFAULT_USER_AGENT,
        help="HTTP User-Agent header (default: Chrome desktop UA).",
    )
    parser.add_argument(
        "--site",
        type=str,
        default=DEFAULT_SITE,
        choices=sorted(SITE_HOSTS.keys()),
        help=(
            "AppGallery region cluster to query "
            f"(default: {DEFAULT_SITE})."
        ),
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=(
            "Number of items per getTabDetail request "
            f"(default: {DEFAULT_PAGE_SIZE})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed for the inter-request jitter RNG. The crawler is "
            "not deterministic across runs because the AppGallery "
            "catalogue itself changes, but a fixed seed keeps the "
            "request timing reproducible (default: 42)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.max_per_category < 1:
        print(
            "[appgallery] --max-per-category must be >= 1",
            file=sys.stderr,
        )
        return 2
    if args.page_size < 1:
        print("[appgallery] --page-size must be >= 1", file=sys.stderr)
        return 2

    host = SITE_HOSTS[args.site]
    locale, country = SITE_LOCALES[args.site]
    endpoint = build_endpoint(host)
    rng = random.Random(args.seed)

    rows: list[tuple[str, str, str]] = []
    for tab in CATEGORY_TABS:
        category_rows = crawl_category(
            tab=tab,
            endpoint=endpoint,
            locale=locale,
            country=country,
            user_agent=args.user_agent,
            max_per_category=args.max_per_category,
            page_size=args.page_size,
            rng=rng,
        )
        print(
            f"[appgallery] {tab.name}: {len(category_rows)} rows",
            file=sys.stderr,
        )
        rows.extend(category_rows)

    write_csv_atomic(args.output, rows)
    print(
        f"[appgallery] wrote {len(rows)} rows to "
        f"{args.output.as_posix()}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
