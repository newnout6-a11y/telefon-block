"""RuStore crawler.

Walks RuStore category pages via the public web API
(``backapi.rustore.ru/applicationByCategory``) and produces a CSV with
columns ``packageName,label,category`` written by default to
``datasets/categories/raw/rustore.csv``.

Output contract — identical to the Play Store / Huawei AppGallery
crawlers (spec ``app-category-ml-classifier/design.md`` Component 6,
Requirement 1.1, 1.5):

* Header row ``packageName,label,category`` (case-sensitive).
* UTF-8 without BOM, LF (``\\n``) line endings, trailing newline.
* Atomic write through ``<output>.tmp`` followed by ``os.replace``.

The CSV is *raw* — each row corresponds to a single app record from
RuStore. Dedup by ``packageName``, NFC normalisation of labels and
category mapping into the ``AppCategory`` enum happen later in
``scripts/build_app_category_dataset.py`` (task 3.x). This script only
emits the upstream taxonomy verbatim; we keep the source label so the
downstream merger can detect and fix mismatches.

Network access notice
---------------------
This script performs **outbound HTTPS requests** to
``backapi.rustore.ru`` on every invocation. CI runners or air-gapped
builds will not be able to run it; the dataset pipeline assumes the
resulting raw CSV is committed or otherwise made available out-of-band.

The RuStore web API is **undocumented** and **may change without
notice** — RuStore periodically updates the API structure and category
identifiers. Treat any 4xx / 5xx, empty response, or missing expected
JSON fields as a signal that the schema drifted. The crawler is
deliberately defensive: it logs schema-drift cases to stderr instead
of crashing, so a partial CSV is still produced.

Implements task 5.2 in ``app-category-ml-classifier/tasks.md``.
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

DEFAULT_OUTPUT = Path("datasets/categories/raw/rustore.csv")
DEFAULT_MAX_PER_CATEGORY = 5000
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Delay between requests to avoid rate-limiting. Random jitter makes
# the access pattern less bot-shaped.
SLEEP_MIN_S = 0.5
SLEEP_MAX_S = 1.5
# Maximum retries per HTTP request on transient failures (5xx, timeout).
MAX_RETRIES = 3
# Exponential backoff base for retries (seconds).
RETRY_BACKOFF_BASE_S = 2.0
# Hard ceiling on pages per category — protects against runaway
# pagination if the server returns non-empty pages indefinitely.
MAX_PAGES_PER_CATEGORY = 500
# Default page size for RuStore API requests.
DEFAULT_PAGE_SIZE = 30

# RuStore API base URL. The public web frontend at apps.rustore.ru
# fetches app listings from this backend.
API_BASE = "https://backapi.rustore.ru"


@dataclass(frozen=True)
class RuStoreCategory:
    """One RuStore category to crawl.

    ``category_id`` is the numeric or string identifier used by the
    RuStore API to filter apps by category. ``name`` is a
    human-readable label used for logging and as the category value in
    the output CSV.
    """

    category_id: str
    name: str


# RuStore top-level categories. The ``category_id`` values correspond
# to the identifiers used by the RuStore backend API. These are
# observed from the web frontend at apps.rustore.ru and may change
# without notice.
#
# This list covers the main app categories relevant to the 18
# AppCategory enum values.
RUSTORE_CATEGORIES: tuple[RuStoreCategory, ...] = (
    # Finance / Banking
    RuStoreCategory("finance", "Финансы"),
    # Shopping / Marketplace
    RuStoreCategory("shopping", "Покупки"),
    # Travel
    RuStoreCategory("travel", "Путешествия"),
    # News
    RuStoreCategory("news", "Новости"),
    # Communication / Messenger
    RuStoreCategory("communication", "Общение"),
    # Social
    RuStoreCategory("social", "Социальные сети"),
    # Entertainment / Media
    RuStoreCategory("entertainment", "Развлечения"),
    # Music & Audio
    RuStoreCategory("music_and_audio", "Музыка и аудио"),
    # Video Players
    RuStoreCategory("video_players", "Видеоплееры"),
    # Health & Fitness
    RuStoreCategory("health_and_fitness", "Здоровье и фитнес"),
    # Medical
    RuStoreCategory("medical", "Медицина"),
    # Education
    RuStoreCategory("education", "Образование"),
    # Business
    RuStoreCategory("business", "Бизнес"),
    # Productivity
    RuStoreCategory("productivity", "Продуктивность"),
    # Tools
    RuStoreCategory("tools", "Инструменты"),
    # Maps & Navigation (Transport)
    RuStoreCategory("maps_and_navigation", "Карты и навигация"),
    # Food & Drink (Delivery)
    RuStoreCategory("food_and_drink", "Еда и напитки"),
    # Dating
    RuStoreCategory("dating", "Знакомства"),
    # Sports
    RuStoreCategory("sports", "Спорт"),
    # Lifestyle
    RuStoreCategory("lifestyle", "Стиль жизни"),
    # Books & Reference
    RuStoreCategory("books_and_reference", "Книги"),
    # Photography
    RuStoreCategory("photography", "Фотография"),
    # Auto & Vehicles
    RuStoreCategory("auto_and_vehicles", "Авто и транспорт"),
    # House & Home
    RuStoreCategory("house_and_home", "Дом"),
    # Government services
    RuStoreCategory("government", "Госуслуги"),
    # Games — all mapped to a single "Игры" output label
    RuStoreCategory("games_action", "Игры"),
    RuStoreCategory("games_adventure", "Игры"),
    RuStoreCategory("games_arcade", "Игры"),
    RuStoreCategory("games_board", "Игры"),
    RuStoreCategory("games_card", "Игры"),
    RuStoreCategory("games_casual", "Игры"),
    RuStoreCategory("games_puzzle", "Игры"),
    RuStoreCategory("games_racing", "Игры"),
    RuStoreCategory("games_role_playing", "Игры"),
    RuStoreCategory("games_simulation", "Игры"),
    RuStoreCategory("games_sports", "Игры"),
    RuStoreCategory("games_strategy", "Игры"),
)


def _build_category_url(
    category_id: str,
    *,
    page: int = 0,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> str:
    """Build the URL for a RuStore category listing API request."""
    params = urllib.parse.urlencode({
        "categoryId": category_id,
        "page": str(page),
        "pageSize": str(page_size),
    })
    return f"{API_BASE}/applicationByCategory?{params}"


def _fetch_with_retries(
    url: str,
    *,
    user_agent: str,
    max_retries: int = MAX_RETRIES,
    rng: random.Random,
    timeout_s: float = 30.0,
) -> dict | None:
    """Fetch a URL with exponential-backoff retries on transient errors.

    Returns the parsed JSON response on success, or ``None`` on
    permanent failure (4xx other than 429, non-recoverable errors).
    Retries on: 429 (rate limit), 5xx, timeouts, and network errors.
    """
    for attempt in range(max_retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": user_agent,
                    "Accept": "application/json,text/plain,*/*",
                    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
                    "Referer": "https://apps.rustore.ru/",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print(
                    f"[rustore] HTTP 404 for {url}; skipping",
                    file=sys.stderr,
                )
                return None
            if exc.code == 429 or exc.code >= 500:
                if attempt < max_retries:
                    backoff = RETRY_BACKOFF_BASE_S * (2 ** attempt)
                    jitter = rng.uniform(0, backoff * 0.5)
                    wait = backoff + jitter
                    print(
                        f"[rustore] HTTP {exc.code} for {url}; "
                        f"retry {attempt + 1}/{max_retries} after "
                        f"{wait:.1f}s",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    continue
                print(
                    f"[rustore] HTTP {exc.code} for {url}; "
                    f"exhausted retries",
                    file=sys.stderr,
                )
                return None
            print(
                f"[rustore] HTTP {exc.code} for {url}: "
                f"{exc.reason}; skipping",
                file=sys.stderr,
            )
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < max_retries:
                backoff = RETRY_BACKOFF_BASE_S * (2 ** attempt)
                jitter = rng.uniform(0, backoff * 0.5)
                wait = backoff + jitter
                print(
                    f"[rustore] network error for {url}: {exc}; "
                    f"retry {attempt + 1}/{max_retries} after "
                    f"{wait:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            print(
                f"[rustore] network error for {url}: {exc}; "
                f"exhausted retries",
                file=sys.stderr,
            )
            return None

        # Parse JSON response.
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            print(
                f"[rustore] non-JSON response for {url}: {exc}",
                file=sys.stderr,
            )
            return None

        if not isinstance(payload, dict):
            print(
                f"[rustore] unexpected JSON shape (top-level not object) "
                f"for {url}",
                file=sys.stderr,
            )
            return None

        # RuStore API uses a ``code`` field for in-band error signaling.
        # A successful response has ``code == 0`` or ``code`` absent.
        code = payload.get("code")
        if code is not None and code != 0 and str(code) != "0":
            print(
                f"[rustore] API error code={code!r} for {url}; "
                f"aborting page",
                file=sys.stderr,
            )
            return None

        return payload

    return None


def extract_records(
    payload: dict,
    *,
    fallback_category: str,
) -> list[tuple[str, str, str]]:
    """Pull (packageName, label, category) tuples out of a JSON response.

    The RuStore API response structure is roughly::

        {
          "code": 0,
          "body": {
            "apps": [
              {
                "packageName": "com.example",
                "appName": "App Name",
                "categoryName": "Финансы"
              },
              ...
            ]
          }
        }

    Field names have been observed to vary (``packageName`` vs
    ``package``, ``appName`` vs ``name`` vs ``appTitle``,
    ``categoryName`` vs ``category``); we accept any of the known
    synonyms. Items missing a usable package name are silently
    skipped — the merge step in ``build_app_category_dataset.py``
    will not produce a row for them either.
    """
    out: list[tuple[str, str, str]] = []

    # Navigate to the apps list. RuStore wraps the list in a ``body``
    # object, but we also handle the case where apps are at the top
    # level or under ``data``.
    body = payload.get("body") or payload.get("data") or payload
    if isinstance(body, dict):
        items = (
            body.get("apps")
            or body.get("applications")
            or body.get("content")
            or body.get("list")
        )
    elif isinstance(body, list):
        items = body
    else:
        items = None

    if not isinstance(items, list):
        return out

    for item in items:
        if not isinstance(item, dict):
            continue

        package_name = (
            item.get("packageName")
            or item.get("package")
            or item.get("appId")
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
            or item.get("appTitle")
            or item.get("title")
            or ""
        )
        if not isinstance(label, str):
            label = ""
        label = label.strip()

        category = (
            item.get("categoryName")
            or item.get("category")
            or item.get("categoryTitle")
            or fallback_category
        )
        if not isinstance(category, str):
            category = fallback_category
        category = category.strip() or fallback_category

        out.append((package_name, label, category))

    return out


def crawl_category(
    *,
    category: RuStoreCategory,
    user_agent: str,
    max_per_category: int,
    page_size: int,
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """Crawl one RuStore category, returning (packageName, label, category) rows.

    Iterates through paginated API responses until the per-category
    quota is met or pages are exhausted.

    Pagination stops on the first of:
    (a) ``max_per_category`` unique packages collected,
    (b) ``MAX_PAGES_PER_CATEGORY`` pages fetched,
    (c) an empty extraction result (no new apps found),
    (d) a permanent fetch failure.
    """
    rows: list[tuple[str, str, str]] = []
    seen_in_category: set[str] = set()

    for page in range(MAX_PAGES_PER_CATEGORY):
        if len(rows) >= max_per_category:
            break

        url = _build_category_url(
            category.category_id,
            page=page,
            page_size=page_size,
        )

        payload = _fetch_with_retries(url, user_agent=user_agent, rng=rng)
        if payload is None:
            break

        page_rows = extract_records(
            payload, fallback_category=category.name
        )
        if not page_rows:
            # Empty page — assume end of category.
            break

        added_any = False
        for package_name, label, cat in page_rows:
            if package_name in seen_in_category:
                continue
            seen_in_category.add(package_name)
            rows.append((package_name, label, cat))
            added_any = True
            if len(rows) >= max_per_category:
                break

        if not added_any:
            # All apps on this page were already seen — stop to
            # avoid infinite loops.
            break

        # Throttle between page requests.
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
    Store / Huawei AppGallery crawlers).
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
            "Crawl RuStore category pages via the public web API "
            "and emit a packageName,label,category CSV. Network "
            "access required; API may change."
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
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=(
            "Number of items per API request "
            f"(default: {DEFAULT_PAGE_SIZE})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed for the inter-request jitter RNG. The crawler is "
            "not deterministic across runs because the RuStore "
            "catalogue itself changes, but a fixed seed keeps the "
            "request timing reproducible (default: 42)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the RuStore crawler.

    Iterates over all configured RuStore categories, extracts app
    records, and writes the result to a CSV file.
    """
    args = parse_args(argv)

    if args.max_per_category < 1:
        print(
            "[rustore] --max-per-category must be >= 1",
            file=sys.stderr,
        )
        return 2
    if args.page_size < 1:
        print("[rustore] --page-size must be >= 1", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)

    rows: list[tuple[str, str, str]] = []
    global_seen: set[str] = set()

    for category in RUSTORE_CATEGORIES:
        category_rows = crawl_category(
            category=category,
            user_agent=args.user_agent,
            max_per_category=args.max_per_category,
            page_size=args.page_size,
            rng=rng,
        )

        # Deduplicate across categories — a package may appear in
        # multiple categories (e.g., a banking app in both finance
        # and business). We keep the first occurrence.
        new_rows: list[tuple[str, str, str]] = []
        for package_name, label, cat in category_rows:
            if package_name not in global_seen:
                global_seen.add(package_name)
                new_rows.append((package_name, label, cat))

        print(
            f"[rustore] {category.name} ({category.category_id}): "
            f"{len(category_rows)} found, {len(new_rows)} new",
            file=sys.stderr,
        )
        rows.extend(new_rows)

    write_csv_atomic(args.output, rows)
    print(
        f"[rustore] wrote {len(rows)} rows to "
        f"{args.output.as_posix()}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
