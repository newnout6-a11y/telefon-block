"""Bulk Google Play Store scraper using google-play-scraper package.

Collects (packageName, label, category) triples from Google Play Store
using the google-play-scraper library's search API.

Strategy:
1. For each target category, run multiple search queries (keywords).
2. For each result, extract appId (packageName), title (label), and genre (category).
3. Deduplicate by packageName globally.
4. Write results to datasets/categories/raw/play_store.csv.

This script replaces the HTML-based play_store_crawler.py which no
longer works due to Google Play requiring JavaScript rendering.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

try:
    from google_play_scraper import search as gps_search
except ImportError:
    print(
        "[gplay_bulk] google-play-scraper not installed. "
        "Run: pip install google-play-scraper",
        file=sys.stderr,
    )
    sys.exit(1)

DEFAULT_OUTPUT = Path("datasets/categories/raw/play_store.csv")

# Search queries grouped by target category.
# Each query will be run in multiple locales to maximize coverage.
# The output category is the Play Store genre name (raw), which will
# be mapped to AppCategory later by build_app_category_dataset.py.
SEARCH_QUERIES: dict[str, list[str]] = {
    "Finance": [
        "bank", "banking", "mobile bank", "payment", "wallet",
        "money transfer", "credit card", "debit card", "fintech",
        "банк", "мобильный банк", "оплата", "кошелёк", "перевод денег",
        "кредит", "дебетовая карта", "финансы", "платёж",
        "neobank", "digital bank", "online banking", "pay bills",
        "savings", "checking account", "loan", "mortgage",
        "банковское приложение", "вклад", "ипотека", "займ",
    ],
    "Shopping": [
        "marketplace", "online shopping", "shop", "buy",
        "маркетплейс", "интернет магазин", "покупки", "товары",
        "wildberries", "ozon", "aliexpress", "amazon", "ebay",
        "shopping app", "deals", "discount", "coupon",
        "скидки", "распродажа", "купон", "доставка товаров",
        "online store", "ecommerce", "retail",
    ],
    "Communication": [
        "messenger", "chat", "messaging", "video call",
        "мессенджер", "чат", "видеозвонок", "общение",
        "telegram", "whatsapp", "viber", "signal",
        "voice call", "sms", "instant messaging",
        "голосовой звонок", "сообщения",
    ],
    "Social": [
        "social network", "social media", "friends",
        "социальная сеть", "друзья", "подписчики",
        "instagram", "tiktok", "facebook", "twitter",
        "community", "forum", "blog", "vlog",
        "сообщество", "форум", "блог",
    ],
    "Entertainment": [
        "video streaming", "movies", "tv shows", "series",
        "видео", "фильмы", "сериалы", "стриминг",
        "netflix", "youtube", "twitch", "anime",
        "кино", "аниме", "развлечения",
    ],
    "Music & Audio": [
        "music player", "music streaming", "podcast",
        "музыка", "плеер", "подкаст", "радио",
        "spotify", "soundcloud", "audio",
        "музыкальный плеер", "слушать музыку",
    ],
    "Health & Fitness": [
        "health", "fitness", "workout", "exercise",
        "здоровье", "фитнес", "тренировка", "упражнения",
        "doctor", "medical", "clinic", "hospital",
        "доктор", "медицина", "клиника", "больница",
        "yoga", "meditation", "diet", "nutrition",
        "йога", "медитация", "диета", "питание",
    ],
    "Education": [
        "education", "learning", "courses", "study",
        "образование", "обучение", "курсы", "учёба",
        "english learning", "language", "math",
        "английский", "язык", "математика",
        "university", "school", "tutor", "exam",
        "университет", "школа", "репетитор", "экзамен",
    ],
    "Productivity": [
        "productivity", "task manager", "notes", "calendar",
        "продуктивность", "задачи", "заметки", "календарь",
        "office", "document", "spreadsheet", "pdf",
        "офис", "документ", "таблица",
        "todo", "planner", "organizer", "reminder",
        "планировщик", "органайзер", "напоминание",
    ],
    "Maps & Navigation": [
        "taxi", "navigation", "maps", "gps",
        "такси", "навигация", "карты", "маршрут",
        "ride sharing", "uber", "yandex taxi",
        "scooter", "transport", "bus", "metro",
        "самокат", "транспорт", "автобус", "метро",
    ],
    "Food & Drink": [
        "food delivery", "restaurant", "pizza",
        "доставка еды", "ресторан", "пицца",
        "grocery delivery", "cooking", "recipe",
        "доставка продуктов", "готовка", "рецепт",
        "cafe", "coffee", "food order",
        "кафе", "кофе", "заказ еды",
    ],
    "Travel & Local": [
        "travel", "flights", "hotel", "booking",
        "путешествия", "авиабилеты", "отель", "бронирование",
        "vacation", "trip planner", "tourism",
        "отпуск", "планировщик поездок", "туризм",
        "airbnb", "hostel", "car rental",
        "хостел", "аренда авто",
    ],
    "News & Magazines": [
        "news", "newspaper", "magazine", "journalism",
        "новости", "газета", "журнал",
        "breaking news", "world news", "local news",
        "rbc", "lenta", "tass", "reuters",
    ],
    "Dating": [
        "dating", "love", "relationship", "match",
        "знакомства", "любовь", "отношения",
        "tinder", "bumble", "badoo", "mamba",
        "свидания", "пара", "романтика",
    ],
    "Games": [
        "game", "puzzle", "action game", "strategy",
        "игра", "головоломка", "экшн", "стратегия",
        "arcade", "racing", "rpg", "adventure",
        "аркада", "гонки", "рпг", "приключения",
        "multiplayer", "online game", "casual game",
        "мультиплеер", "онлайн игра",
    ],
    "Tools": [
        "vpn", "browser", "file manager", "cleaner",
        "впн", "браузер", "файловый менеджер", "очистка",
        "antivirus", "security", "password manager",
        "антивирус", "безопасность", "менеджер паролей",
        "flashlight", "calculator", "scanner",
        "фонарик", "калькулятор", "сканер",
    ],
    "Business": [
        "business", "crm", "invoice", "accounting",
        "бизнес", "бухгалтерия", "счёт", "учёт",
        "project management", "team", "collaboration",
        "управление проектами", "команда",
    ],
    "Sports": [
        "sports", "football", "basketball", "soccer",
        "спорт", "футбол", "баскетбол",
        "live score", "sports news", "betting",
        "счёт матча", "спортивные новости",
    ],
    "Lifestyle": [
        "lifestyle", "fashion", "beauty", "home",
        "стиль жизни", "мода", "красота", "дом",
        "interior design", "garden", "pets",
        "дизайн интерьера", "сад", "питомцы",
    ],
    "Books & Reference": [
        "books", "ebook", "reading", "library",
        "книги", "электронная книга", "чтение", "библиотека",
        "audiobook", "dictionary", "encyclopedia",
        "аудиокнига", "словарь", "энциклопедия",
    ],
    "Photography": [
        "photo editor", "camera", "filter",
        "фоторедактор", "камера", "фильтр",
        "selfie", "collage", "photo",
        "селфи", "коллаж", "фото",
    ],
    "Auto & Vehicles": [
        "car", "auto", "vehicle", "driving",
        "авто", "машина", "вождение",
        "car insurance", "fuel", "parking",
        "автострахование", "бензин", "парковка",
    ],
    "Weather": [
        "weather", "forecast", "temperature",
        "погода", "прогноз", "температура",
    ],
    "Parenting": [
        "parenting", "baby", "kids", "pregnancy",
        "родители", "ребёнок", "дети", "беременность",
    ],
}

# Locales to search in
LOCALES: list[tuple[str, str]] = [
    ("en", "us"),
    ("ru", "ru"),
    ("en", "gb"),
    ("en", "in"),
]


def write_csv_atomic(output: Path, rows: list[tuple[str, str, str]]) -> None:
    """Write rows to output atomically (UTF-8, LF, trailing newline)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("packageName", "label", "category"))
        for package_name, label, category in rows:
            writer.writerow((package_name, label, category))
    os.replace(tmp, output)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bulk scrape Google Play Store using google-play-scraper search API"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--n-hits", type=int, default=30,
        help="Max results per search query (default: 30)",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    all_rows: list[tuple[str, str, str]] = []
    global_seen: set[str] = set()
    total_queries = sum(
        len(queries) * len(LOCALES)
        for queries in SEARCH_QUERIES.values()
    )
    done_queries = 0

    for category_name, queries in SEARCH_QUERIES.items():
        cat_new = 0
        for query in queries:
            for lang, country in LOCALES:
                done_queries += 1
                try:
                    results = gps_search(
                        query,
                        lang=lang,
                        country=country,
                        n_hits=args.n_hits,
                    )
                except Exception as exc:
                    print(
                        f"[gplay_bulk] search '{query}' ({lang}/{country}) "
                        f"failed: {exc}",
                        file=sys.stderr,
                    )
                    time.sleep(args.delay)
                    continue

                new = 0
                for app in results:
                    if not app:
                        continue
                    pkg = (app.get("appId") or "").strip()
                    title = (app.get("title") or "").strip()
                    genre = (app.get("genre") or "").strip()
                    if not pkg:
                        continue
                    if pkg not in global_seen:
                        global_seen.add(pkg)
                        # Use the Play Store genre as category (raw)
                        out_cat = genre if genre else category_name
                        all_rows.append((pkg, title, out_cat))
                        new += 1
                        cat_new += 1

                time.sleep(args.delay)

                # Progress report every 20 queries
                if done_queries % 20 == 0:
                    print(
                        f"[gplay_bulk] progress: {done_queries}/{total_queries} "
                        f"queries, {len(all_rows)} unique apps",
                        file=sys.stderr,
                    )

        print(
            f"[gplay_bulk] {category_name}: +{cat_new} new "
            f"(total: {len(all_rows)})",
            file=sys.stderr,
        )

    write_csv_atomic(args.output, all_rows)
    print(
        f"\n[gplay_bulk] DONE: wrote {len(all_rows)} unique rows to "
        f"{args.output.as_posix()}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
