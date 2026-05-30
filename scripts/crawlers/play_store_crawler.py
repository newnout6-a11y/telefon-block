"""Google Play Store crawler.

Walks Google Play Store category pages and produces a CSV with columns
``packageName,label,category`` written by default to
``datasets/categories/raw/play_store.csv``.

Output contract — identical to the RuStore / Huawei AppGallery crawlers
(spec ``app-category-ml-classifier/design.md`` Component 6,
Requirement 1.1, 1.5):

* Header row ``packageName,label,category`` (case-sensitive).
* UTF-8 without BOM, LF (``\\n``) line endings, trailing newline.
* Atomic write through ``<output>.tmp`` followed by ``os.replace``.

The CSV is *raw* — each row corresponds to a single app record from
Google Play. Dedup by ``packageName``, NFC normalisation of labels and
category mapping into the ``AppCategory`` enum happen later in
``scripts/build_app_category_dataset.py`` (task 3.x). This script only
emits the upstream taxonomy verbatim; we keep the source label so the
downstream merger can detect and fix mismatches.

Network access notice
---------------------
This script performs **outbound HTTPS requests** to
``play.google.com`` on every invocation. CI runners or air-gapped
builds will not be able to run it; the dataset pipeline assumes the
resulting raw CSV is committed or otherwise made available out-of-band.

The Google Play Store web interface is **undocumented** and **may
change without notice** — Google periodically updates the HTML
structure and category URLs. Treat any 4xx / 5xx, empty response, or
missing expected HTML elements as a signal that the schema drifted.
The crawler is deliberately defensive: it logs schema-drift cases to
stderr instead of crashing, so a partial CSV is still produced.

Implements task 5.1 in ``app-category-ml-classifier/tasks.md``.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUTPUT = Path("datasets/categories/raw/play_store.csv")
DEFAULT_MAX_PER_CATEGORY = 5000
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Delay between requests to avoid rate-limiting. Random jitter makes
# the access pattern less bot-shaped.
SLEEP_MIN_S = 1.0
SLEEP_MAX_S = 2.5
# Maximum retries per HTTP request on transient failures (5xx, timeout).
MAX_RETRIES = 3
# Exponential backoff base for retries (seconds).
RETRY_BACKOFF_BASE_S = 2.0
# Hard ceiling on collection pages per category — protects against
# runaway pagination.
MAX_PAGES_PER_CATEGORY = 200
# Google Play uses a token-based pagination; each page returns ~50 apps.
# We request up to this many items per page via the ``num`` parameter.
DEFAULT_PAGE_SIZE = 50


@dataclass(frozen=True)
class PlayCategory:
    """One Google Play Store category to crawl.

    ``category_id`` is the URL path segment used by Google Play
    (e.g. ``FINANCE``, ``GAME_ACTION``). ``display_name`` is a
    human-readable label used for logging and as a fallback category
    name in the output CSV.
    """

    category_id: str
    display_name: str


# Google Play Store top-level categories. The ``category_id`` values
# correspond to the URL path used by Google Play:
# ``https://play.google.com/store/apps/category/<CATEGORY_ID>``
#
# This list covers the main app categories relevant to the 18
# AppCategory enum values. Game sub-categories are grouped under a
# single ``GAMES`` output label.
PLAY_CATEGORIES: tuple[PlayCategory, ...] = (
    # Finance / Banking
    PlayCategory("FINANCE", "FINANCE"),
    # Shopping / Marketplace
    PlayCategory("SHOPPING", "SHOPPING"),
    # Travel & Local
    PlayCategory("TRAVEL_AND_LOCAL", "TRAVEL_AND_LOCAL"),
    # News & Magazines
    PlayCategory("NEWS_AND_MAGAZINES", "NEWS_AND_MAGAZINES"),
    # Communication (Messenger)
    PlayCategory("COMMUNICATION", "COMMUNICATION"),
    # Social
    PlayCategory("SOCIAL", "SOCIAL"),
    # Entertainment / Media
    PlayCategory("ENTERTAINMENT", "ENTERTAINMENT"),
    # Video Players & Editors
    PlayCategory("VIDEO_PLAYERS", "VIDEO_PLAYERS"),
    # Music & Audio
    PlayCategory("MUSIC_AND_AUDIO", "MUSIC_AND_AUDIO"),
    # Health & Fitness
    PlayCategory("HEALTH_AND_FITNESS", "HEALTH_AND_FITNESS"),
    # Medical
    PlayCategory("MEDICAL", "MEDICAL"),
    # Education
    PlayCategory("EDUCATION", "EDUCATION"),
    # Business
    PlayCategory("BUSINESS", "BUSINESS"),
    # Productivity
    PlayCategory("PRODUCTIVITY", "PRODUCTIVITY"),
    # Tools
    PlayCategory("TOOLS", "TOOLS"),
    # Maps & Navigation (Transport)
    PlayCategory("MAPS_AND_NAVIGATION", "MAPS_AND_NAVIGATION"),
    # Food & Drink (Delivery)
    PlayCategory("FOOD_AND_DRINK", "FOOD_AND_DRINK"),
    # Dating
    PlayCategory("DATING", "DATING"),
    # Sports
    PlayCategory("SPORTS", "SPORTS"),
    # Lifestyle
    PlayCategory("LIFESTYLE", "LIFESTYLE"),
    # Books & Reference
    PlayCategory("BOOKS_AND_REFERENCE", "BOOKS_AND_REFERENCE"),
    # Weather
    PlayCategory("WEATHER", "WEATHER"),
    # Photography
    PlayCategory("PHOTOGRAPHY", "PHOTOGRAPHY"),
    # Auto & Vehicles
    PlayCategory("AUTO_AND_VEHICLES", "AUTO_AND_VEHICLES"),
    # House & Home
    PlayCategory("HOUSE_AND_HOME", "HOUSE_AND_HOME"),
    # Parenting
    PlayCategory("PARENTING", "PARENTING"),
    # Events
    PlayCategory("EVENTS", "EVENTS"),
    # Art & Design
    PlayCategory("ART_AND_DESIGN", "ART_AND_DESIGN"),
    # Beauty
    PlayCategory("BEAUTY", "BEAUTY"),
    # Comics
    PlayCategory("COMICS", "COMICS"),
    # Libraries & Demo
    PlayCategory("LIBRARIES_AND_DEMO", "LIBRARIES_AND_DEMO"),
    # Game categories — all mapped to "GAME" in output
    PlayCategory("GAME_ACTION", "GAME"),
    PlayCategory("GAME_ADVENTURE", "GAME"),
    PlayCategory("GAME_ARCADE", "GAME"),
    PlayCategory("GAME_BOARD", "GAME"),
    PlayCategory("GAME_CARD", "GAME"),
    PlayCategory("GAME_CASINO", "GAME"),
    PlayCategory("GAME_CASUAL", "GAME"),
    PlayCategory("GAME_EDUCATIONAL", "GAME"),
    PlayCategory("GAME_MUSIC", "GAME"),
    PlayCategory("GAME_PUZZLE", "GAME"),
    PlayCategory("GAME_RACING", "GAME"),
    PlayCategory("GAME_ROLE_PLAYING", "GAME"),
    PlayCategory("GAME_SIMULATION", "GAME"),
    PlayCategory("GAME_SPORTS", "GAME"),
    PlayCategory("GAME_STRATEGY", "GAME"),
    PlayCategory("GAME_TRIVIA", "GAME"),
    PlayCategory("GAME_WORD", "GAME"),
)


def _build_category_url(category_id: str, hl: str = "en", gl: str = "us") -> str:
    """Build the URL for a Google Play category listing page."""
    return (
        f"https://play.google.com/store/apps/category/{category_id}"
        f"?hl={hl}&gl={gl}"
    )


def _build_collection_url(
    category_id: str,
    collection: str = "topselling_free",
    *,
    hl: str = "en",
    gl: str = "us",
    num: int = DEFAULT_PAGE_SIZE,
    start: int = 0,
) -> str:
    """Build the URL for a Google Play collection page within a category.

    Google Play exposes several collections per category:
    - ``topselling_free`` — top free apps
    - ``topselling_paid`` — top paid apps
    - ``topgrossing`` — top grossing apps
    - ``movers_shakers`` — trending apps

    We primarily crawl ``topselling_free`` for volume, with
    ``topselling_paid`` as a secondary source.
    """
    params = urllib.parse.urlencode({
        "hl": hl,
        "gl": gl,
        "num": str(num),
        "start": str(start),
    })
    return (
        f"https://play.google.com/store/apps/collection/{collection}"
        f"?cat={category_id}&{params}"
    )


def _fetch_with_retries(
    url: str,
    *,
    user_agent: str,
    max_retries: int = MAX_RETRIES,
    rng: random.Random,
    timeout_s: float = 30.0,
) -> str | None:
    """Fetch a URL with exponential-backoff retries on transient errors.

    Returns the response body as a string on success, or ``None`` on
    permanent failure (4xx other than 429, non-recoverable errors).
    Retries on: 429 (rate limit), 5xx, timeouts, and network errors.
    """
    for attempt in range(max_retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": user_agent,
                    "Accept": "text/html,application/xhtml+xml,*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                # Category page does not exist — not retryable.
                print(
                    f"[play_store] HTTP 404 for {url}; skipping",
                    file=sys.stderr,
                )
                return None
            if exc.code == 429 or exc.code >= 500:
                # Rate-limited or server error — retryable.
                if attempt < max_retries:
                    backoff = RETRY_BACKOFF_BASE_S * (2 ** attempt)
                    jitter = rng.uniform(0, backoff * 0.5)
                    wait = backoff + jitter
                    print(
                        f"[play_store] HTTP {exc.code} for {url}; "
                        f"retry {attempt + 1}/{max_retries} after "
                        f"{wait:.1f}s",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    continue
                print(
                    f"[play_store] HTTP {exc.code} for {url}; "
                    f"exhausted retries",
                    file=sys.stderr,
                )
                return None
            # Other 4xx — not retryable.
            print(
                f"[play_store] HTTP {exc.code} for {url}: "
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
                    f"[play_store] network error for {url}: {exc}; "
                    f"retry {attempt + 1}/{max_retries} after "
                    f"{wait:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            print(
                f"[play_store] network error for {url}: {exc}; "
                f"exhausted retries",
                file=sys.stderr,
            )
            return None
    return None


def extract_apps_from_html(html: str) -> list[tuple[str, str]]:
    """Extract (packageName, displayName) pairs from a Play Store HTML page.

    Google Play renders app listings as links with the pattern:
    ``/store/apps/details?id=<packageName>``

    Display names are typically in nearby elements. We use regex-based
    extraction since the HTML structure is not stable enough for a
    proper DOM parser dependency, and we want to keep the crawler
    dependency-free (stdlib only).

    Returns a list of (packageName, displayName) tuples. Display names
    may be empty strings if extraction fails for a particular app.
    """
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    # Pattern 1: Extract package names from detail links.
    # Matches: /store/apps/details?id=com.example.app
    package_pattern = re.compile(
        r'/store/apps/details\?id=([a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)'
    )

    # Pattern 2: Extract display names from title attributes or aria-labels
    # near the package link. We look for patterns like:
    #   <a href="/store/apps/details?id=..." aria-label="App Name">
    #   <span ...>App Name</span> near the link
    name_pattern = re.compile(
        r'href="/store/apps/details\?id=([^"&]+)"[^>]*?'
        r'(?:aria-label="([^"]*?)"|title="([^"]*?)")',
        re.DOTALL,
    )

    # First pass: extract names from links with aria-label or title.
    for match in name_pattern.finditer(html):
        package_name = match.group(1).strip()
        display_name = (match.group(2) or match.group(3) or "").strip()
        if package_name and package_name not in seen:
            seen.add(package_name)
            results.append((package_name, display_name))

    # Second pass: extract any remaining package names without names.
    for match in package_pattern.finditer(html):
        package_name = match.group(1).strip()
        if package_name and package_name not in seen:
            seen.add(package_name)
            # Try to find a nearby display name in the surrounding context.
            start = max(0, match.start() - 500)
            end = min(len(html), match.end() + 500)
            context = html[start:end]
            # Look for text content in spans/divs near the link.
            name_in_context = _extract_nearby_name(context, package_name)
            results.append((package_name, name_in_context))

    return results


def _extract_nearby_name(context: str, package_name: str) -> str:
    """Try to extract a display name from HTML context near a package link.

    Looks for common patterns where Google Play renders app names:
    - ``<span class="...">App Name</span>``
    - ``<div class="...">App Name</div>``
    - Text content in elements with title-like classes.

    Returns an empty string if no plausible name is found.
    """
    # Look for span/div text content that looks like an app name
    # (not too long, not HTML, not a URL).
    text_pattern = re.compile(
        r'<(?:span|div)[^>]*>([^<]{2,80})</(?:span|div)>'
    )
    candidates: list[str] = []
    for m in text_pattern.finditer(context):
        text = m.group(1).strip()
        # Filter out things that are clearly not app names.
        if not text:
            continue
        if text.startswith("http") or text.startswith("/"):
            continue
        if package_name in text:
            continue
        if len(text) < 2 or len(text) > 80:
            continue
        # Prefer shorter, cleaner strings.
        candidates.append(text)

    if candidates:
        # Return the first reasonable candidate (closest to the link).
        return candidates[0]
    return ""


def crawl_category(
    *,
    category: PlayCategory,
    user_agent: str,
    max_per_category: int,
    rng: random.Random,
    hl: str = "en",
    gl: str = "us",
) -> list[tuple[str, str, str]]:
    """Crawl one Google Play category, returning (packageName, label, category) rows.

    Iterates through collection pages (topselling_free, topselling_paid)
    until the per-category quota is met or pages are exhausted.

    Pagination stops on the first of:
    (a) ``max_per_category`` unique packages collected,
    (b) ``MAX_PAGES_PER_CATEGORY`` pages fetched,
    (c) an empty extraction result (no new apps found),
    (d) a permanent fetch failure.
    """
    rows: list[tuple[str, str, str]] = []
    seen_in_category: set[str] = set()

    # Crawl multiple collections for better coverage.
    collections = ["topselling_free", "topselling_paid", "topgrossing"]

    for collection in collections:
        if len(rows) >= max_per_category:
            break

        for page_idx in range(MAX_PAGES_PER_CATEGORY):
            if len(rows) >= max_per_category:
                break

            start = page_idx * DEFAULT_PAGE_SIZE
            url = _build_collection_url(
                category.category_id,
                collection,
                hl=hl,
                gl=gl,
                num=DEFAULT_PAGE_SIZE,
                start=start,
            )

            html = _fetch_with_retries(url, user_agent=user_agent, rng=rng)
            if html is None:
                break

            apps = extract_apps_from_html(html)
            if not apps:
                # No apps found — end of this collection's pagination.
                break

            added_any = False
            for package_name, display_name in apps:
                if package_name in seen_in_category:
                    continue
                seen_in_category.add(package_name)
                rows.append((package_name, display_name, category.display_name))
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
    (Requirement 1.5, atomic-write semantics shared with the RuStore /
    Huawei AppGallery crawlers).
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
            "Crawl Google Play Store category pages and emit a "
            "packageName,label,category CSV. Network access required; "
            "page structure may change."
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
        "--hl",
        type=str,
        default="en",
        help="Language code for Play Store pages (default: en).",
    )
    parser.add_argument(
        "--gl",
        type=str,
        default="us",
        help="Country code for Play Store pages (default: us).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed for the inter-request jitter RNG. The crawler is "
            "not deterministic across runs because the Play Store "
            "catalogue itself changes, but a fixed seed keeps the "
            "request timing reproducible (default: 42)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the Google Play Store crawler.

    Iterates over all configured Play Store categories, extracts app
    records, and writes the result to a CSV file.
    """
    args = parse_args(argv)

    if args.max_per_category < 1:
        print(
            "[play_store] --max-per-category must be >= 1",
            file=sys.stderr,
        )
        return 2

    rng = random.Random(args.seed)

    rows: list[tuple[str, str, str]] = []
    global_seen: set[str] = set()

    for category in PLAY_CATEGORIES:
        category_rows = crawl_category(
            category=category,
            user_agent=args.user_agent,
            max_per_category=args.max_per_category,
            rng=rng,
            hl=args.hl,
            gl=args.gl,
        )

        # Deduplicate across categories — a package may appear in
        # multiple categories (e.g., a banking app in both FINANCE
        # and BUSINESS). We keep the first occurrence.
        new_rows: list[tuple[str, str, str]] = []
        for package_name, label, cat in category_rows:
            if package_name not in global_seen:
                global_seen.add(package_name)
                new_rows.append((package_name, label, cat))

        print(
            f"[play_store] {category.category_id}: "
            f"{len(category_rows)} found, {len(new_rows)} new",
            file=sys.stderr,
        )
        rows.extend(new_rows)

    write_csv_atomic(args.output, rows)
    print(
        f"[play_store] wrote {len(rows)} rows to "
        f"{args.output.as_posix()}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
