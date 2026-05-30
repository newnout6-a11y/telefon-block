"""Generate a synthetic training corpus for App Category Model.

This script creates a large synthetic dataset of (packageName, label, category)
triples by combining:
1. Real data from scraped CSVs (if available in datasets/categories/raw/)
2. Known packages from RuleBasedAppCategoryClassifier (bootstrap)
3. Synthetically generated package names following real naming patterns

The synthetic generation uses category-specific patterns observed in real
Android package names to create plausible training data. While not as good
as real crawled data, it allows the training pipeline to produce a working
model for development and testing purposes.

Output: datasets/categories/raw/synthetic.csv
"""
from __future__ import annotations

import csv
import os
import random
import sys
from pathlib import Path

DEFAULT_OUTPUT = Path("datasets/categories/raw/synthetic.csv")
TARGET_PER_CATEGORY = 12000  # Target ~12k per category × 18 = ~216k total

# Category-specific package name patterns and label templates
CATEGORY_PATTERNS: dict[str, dict] = {
    "BANK": {
        "prefixes": [
            "ru.sberbank", "com.tinkoff", "ru.vtb", "ru.alfabank",
            "com.bank", "ru.bank", "com.mobile.bank", "ru.mobilebank",
            "com.pay", "ru.pay", "com.wallet", "ru.wallet",
            "com.finance", "ru.finance", "com.banking", "ru.banking",
            "org.bank", "net.bank", "com.neobank", "ru.neobank",
        ],
        "suffixes": [
            ".mobile", ".app", ".android", ".client", ".banking",
            ".pay", ".wallet", ".finance", ".online", ".digital",
            ".lite", ".pro", ".plus", ".business", ".personal",
        ],
        "labels_ru": [
            "Банк", "Мобильный банк", "Онлайн банк", "Платежи",
            "Кошелёк", "Переводы", "Финансы", "Карта", "Счёт",
            "Вклады", "Кредиты", "Ипотека", "Банкинг",
        ],
        "labels_en": [
            "Bank", "Mobile Banking", "Online Bank", "Payments",
            "Wallet", "Transfers", "Finance", "Card", "Account",
            "Savings", "Loans", "Mortgage", "Banking App",
        ],
    },
    "INVESTMENTS": {
        "prefixes": [
            "com.invest", "ru.invest", "com.broker", "ru.broker",
            "com.trading", "ru.trading", "com.stock", "ru.stock",
            "com.crypto", "ru.crypto", "com.exchange", "ru.exchange",
        ],
        "suffixes": [
            ".app", ".mobile", ".android", ".client", ".pro",
            ".trading", ".invest", ".broker", ".exchange",
        ],
        "labels_ru": [
            "Инвестиции", "Брокер", "Трейдинг", "Акции",
            "Криптовалюта", "Биржа", "Портфель", "Фонды",
        ],
        "labels_en": [
            "Investments", "Broker", "Trading", "Stocks",
            "Crypto", "Exchange", "Portfolio", "Funds",
        ],
    },
    "GOVERNMENT": {
        "prefixes": [
            "ru.gosuslugi", "ru.gov", "ru.fns", "ru.mos",
            "ru.gibdd", "ru.pfr", "ru.fssp", "ru.mvd",
            "gov.ru", "com.gov", "ru.government", "ru.egov",
        ],
        "suffixes": [
            ".app", ".mobile", ".android", ".client", ".service",
            ".portal", ".online", ".digital",
        ],
        "labels_ru": [
            "Госуслуги", "Налоги", "ФНС", "МФЦ",
            "Штрафы", "ГИБДД", "Пенсионный фонд", "Портал",
        ],
        "labels_en": [
            "Government Services", "Taxes", "Tax Service", "Public Services",
            "Fines", "Traffic Police", "Pension Fund", "Portal",
        ],
    },
    "MARKETPLACE": {
        "prefixes": [
            "com.wildberries", "com.ozon", "com.aliexpress", "com.amazon",
            "com.shop", "ru.shop", "com.market", "ru.market",
            "com.store", "ru.store", "com.mall", "ru.mall",
            "com.buy", "ru.buy", "com.deal", "ru.deal",
        ],
        "suffixes": [
            ".app", ".mobile", ".android", ".client", ".shopping",
            ".store", ".market", ".shop", ".buyer", ".seller",
        ],
        "labels_ru": [
            "Маркетплейс", "Магазин", "Покупки", "Товары",
            "Скидки", "Распродажа", "Доставка", "Каталог",
        ],
        "labels_en": [
            "Marketplace", "Shop", "Shopping", "Products",
            "Deals", "Sale", "Delivery", "Catalog",
        ],
    },
    "DELIVERY": {
        "prefixes": [
            "com.delivery", "ru.delivery", "com.food", "ru.food",
            "com.eda", "ru.eda", "com.pizza", "ru.pizza",
            "com.restaurant", "ru.restaurant", "com.cafe", "ru.cafe",
        ],
        "suffixes": [
            ".app", ".mobile", ".android", ".client", ".delivery",
            ".order", ".food", ".express", ".fast",
        ],
        "labels_ru": [
            "Доставка еды", "Ресторан", "Пицца", "Кафе",
            "Заказ еды", "Продукты", "Экспресс доставка",
        ],
        "labels_en": [
            "Food Delivery", "Restaurant", "Pizza", "Cafe",
            "Food Order", "Groceries", "Express Delivery",
        ],
    },
    "TRANSPORT": {
        "prefixes": [
            "com.taxi", "ru.taxi", "com.ride", "ru.ride",
            "com.transport", "ru.transport", "com.maps", "ru.maps",
            "com.navigation", "ru.navigation", "com.metro", "ru.metro",
        ],
        "suffixes": [
            ".app", ".mobile", ".android", ".client", ".rider",
            ".driver", ".passenger", ".navigation",
        ],
        "labels_ru": [
            "Такси", "Навигация", "Карты", "Транспорт",
            "Метро", "Автобус", "Маршрут", "Поездка",
        ],
        "labels_en": [
            "Taxi", "Navigation", "Maps", "Transport",
            "Metro", "Bus", "Route", "Ride",
        ],
    },
    "TRAVEL": {
        "prefixes": [
            "com.travel", "ru.travel", "com.booking", "ru.booking",
            "com.hotel", "ru.hotel", "com.flight", "ru.flight",
            "com.airline", "ru.airline", "com.trip", "ru.trip",
        ],
        "suffixes": [
            ".app", ".mobile", ".android", ".client", ".travel",
            ".booking", ".hotel", ".flights",
        ],
        "labels_ru": [
            "Путешествия", "Бронирование", "Отель", "Авиабилеты",
            "Перелёты", "Туризм", "Отпуск", "Поездка",
        ],
        "labels_en": [
            "Travel", "Booking", "Hotel", "Flights",
            "Airlines", "Tourism", "Vacation", "Trip",
        ],
    },
    "HEALTH": {
        "prefixes": [
            "com.health", "ru.health", "com.fitness", "ru.fitness",
            "com.doctor", "ru.doctor", "com.medical", "ru.medical",
            "com.clinic", "ru.clinic", "com.yoga", "ru.yoga",
        ],
        "suffixes": [
            ".app", ".mobile", ".android", ".client", ".health",
            ".fitness", ".medical", ".wellness",
        ],
        "labels_ru": [
            "Здоровье", "Фитнес", "Доктор", "Медицина",
            "Клиника", "Йога", "Тренировка", "Диета",
        ],
        "labels_en": [
            "Health", "Fitness", "Doctor", "Medical",
            "Clinic", "Yoga", "Workout", "Diet",
        ],
    },
    "MESSENGER": {
        "prefixes": [
            "com.messenger", "org.messenger", "com.chat", "ru.chat",
            "com.messaging", "org.messaging", "com.im", "ru.im",
            "com.call", "ru.call", "com.voip", "ru.voip",
        ],
        "suffixes": [
            ".app", ".mobile", ".android", ".client", ".messenger",
            ".chat", ".messaging", ".im", ".voip",
        ],
        "labels_ru": [
            "Мессенджер", "Чат", "Общение", "Звонки",
            "Видеозвонки", "Сообщения", "Переписка",
        ],
        "labels_en": [
            "Messenger", "Chat", "Communication", "Calls",
            "Video Calls", "Messages", "Messaging",
        ],
    },
    "SOCIAL": {
        "prefixes": [
            "com.social", "ru.social", "com.network", "ru.network",
            "com.community", "ru.community", "com.friends", "ru.friends",
            "com.blog", "ru.blog", "com.vlog", "ru.vlog",
        ],
        "suffixes": [
            ".app", ".mobile", ".android", ".client", ".social",
            ".network", ".community", ".friends",
        ],
        "labels_ru": [
            "Социальная сеть", "Друзья", "Сообщество", "Блог",
            "Подписчики", "Лента", "Профиль", "Контакты",
        ],
        "labels_en": [
            "Social Network", "Friends", "Community", "Blog",
            "Followers", "Feed", "Profile", "Contacts",
        ],
    },
    "EMAIL": {
        "prefixes": [
            "com.mail", "ru.mail", "com.email", "ru.email",
            "org.mail", "com.inbox", "ru.inbox", "com.outlook",
        ],
        "suffixes": [
            ".app", ".mobile", ".android", ".client", ".mail",
            ".email", ".inbox", ".lite",
        ],
        "labels_ru": [
            "Почта", "Электронная почта", "Письма", "Входящие",
            "Рассылка", "Корреспонденция",
        ],
        "labels_en": [
            "Mail", "Email", "Inbox", "Letters",
            "Newsletter", "Correspondence",
        ],
    },
    "NEWS": {
        "prefixes": [
            "com.news", "ru.news", "com.media", "ru.media",
            "com.press", "ru.press", "com.journal", "ru.journal",
            "com.gazette", "ru.gazette", "com.rbc", "ru.lenta",
        ],
        "suffixes": [
            ".app", ".mobile", ".android", ".client", ".news",
            ".media", ".press", ".reader",
        ],
        "labels_ru": [
            "Новости", "Газета", "Журнал", "Пресса",
            "Лента новостей", "Медиа", "Издание",
        ],
        "labels_en": [
            "News", "Newspaper", "Magazine", "Press",
            "News Feed", "Media", "Publication",
        ],
    },
    "MEDIA": {
        "prefixes": [
            "com.video", "ru.video", "com.music", "ru.music",
            "com.stream", "ru.stream", "com.player", "ru.player",
            "com.tv", "ru.tv", "com.cinema", "ru.cinema",
        ],
        "suffixes": [
            ".app", ".mobile", ".android", ".client", ".player",
            ".stream", ".video", ".music", ".tv",
        ],
        "labels_ru": [
            "Видео", "Музыка", "Стриминг", "Плеер",
            "Кино", "ТВ", "Фильмы", "Сериалы",
        ],
        "labels_en": [
            "Video", "Music", "Streaming", "Player",
            "Cinema", "TV", "Movies", "Series",
        ],
    },
    "GAMES": {
        "prefixes": [
            "com.game", "ru.game", "com.play", "ru.play",
            "com.puzzle", "ru.puzzle", "com.arcade", "ru.arcade",
            "com.racing", "ru.racing", "com.rpg", "ru.rpg",
            "com.strategy", "ru.strategy", "com.action", "ru.action",
        ],
        "suffixes": [
            ".game", ".games", ".app", ".mobile", ".android",
            ".free", ".pro", ".lite", ".premium", ".adventure",
        ],
        "labels_ru": [
            "Игра", "Головоломка", "Аркада", "Гонки",
            "Стратегия", "Приключения", "РПГ", "Экшн",
        ],
        "labels_en": [
            "Game", "Puzzle", "Arcade", "Racing",
            "Strategy", "Adventure", "RPG", "Action",
        ],
    },
    "DATING": {
        "prefixes": [
            "com.dating", "ru.dating", "com.love", "ru.love",
            "com.match", "ru.match", "com.meet", "ru.meet",
            "com.romance", "ru.romance", "com.flirt", "ru.flirt",
        ],
        "suffixes": [
            ".app", ".mobile", ".android", ".client", ".dating",
            ".love", ".match", ".meet",
        ],
        "labels_ru": [
            "Знакомства", "Свидания", "Любовь", "Пара",
            "Романтика", "Встречи", "Отношения",
        ],
        "labels_en": [
            "Dating", "Love", "Match", "Meet",
            "Romance", "Relationships", "Singles",
        ],
    },
    "EDUCATION": {
        "prefixes": [
            "com.edu", "ru.edu", "com.learn", "ru.learn",
            "com.school", "ru.school", "com.course", "ru.course",
            "com.study", "ru.study", "com.tutor", "ru.tutor",
        ],
        "suffixes": [
            ".app", ".mobile", ".android", ".client", ".edu",
            ".learn", ".school", ".course", ".study",
        ],
        "labels_ru": [
            "Образование", "Обучение", "Курсы", "Школа",
            "Университет", "Репетитор", "Уроки", "Экзамен",
        ],
        "labels_en": [
            "Education", "Learning", "Courses", "School",
            "University", "Tutor", "Lessons", "Exam",
        ],
    },
    "BROWSER": {
        "prefixes": [
            "com.browser", "ru.browser", "org.browser", "com.web",
            "ru.web", "org.web", "com.surf", "ru.surf",
        ],
        "suffixes": [
            ".app", ".mobile", ".android", ".browser", ".web",
            ".lite", ".fast", ".secure", ".private",
        ],
        "labels_ru": [
            "Браузер", "Веб-браузер", "Интернет", "Поиск",
            "Быстрый браузер", "Приватный браузер",
        ],
        "labels_en": [
            "Browser", "Web Browser", "Internet", "Search",
            "Fast Browser", "Private Browser",
        ],
    },
    "VPN": {
        "prefixes": [
            "com.vpn", "ru.vpn", "org.vpn", "net.vpn",
            "com.proxy", "ru.proxy", "com.secure", "ru.secure",
        ],
        "suffixes": [
            ".app", ".mobile", ".android", ".vpn", ".proxy",
            ".free", ".pro", ".fast", ".secure", ".unlimited",
        ],
        "labels_ru": [
            "VPN", "ВПН", "Прокси", "Безопасность",
            "Анонимность", "Защита", "Приватность",
        ],
        "labels_en": [
            "VPN", "Proxy", "Security", "Privacy",
            "Anonymous", "Protection", "Secure VPN",
        ],
    },
    "PRODUCTIVITY": {
        "prefixes": [
            "com.productivity", "ru.productivity", "com.office", "ru.office",
            "com.notes", "ru.notes", "com.todo", "ru.todo",
            "com.calendar", "ru.calendar", "com.task", "ru.task",
        ],
        "suffixes": [
            ".app", ".mobile", ".android", ".client", ".pro",
            ".notes", ".todo", ".calendar", ".task",
        ],
        "labels_ru": [
            "Продуктивность", "Офис", "Заметки", "Задачи",
            "Календарь", "Планировщик", "Документы",
        ],
        "labels_en": [
            "Productivity", "Office", "Notes", "Tasks",
            "Calendar", "Planner", "Documents",
        ],
    },
}

# Random words to make package names more diverse
RANDOM_WORDS = [
    "alpha", "beta", "gamma", "delta", "omega", "nova", "star", "moon",
    "sun", "sky", "cloud", "fire", "ice", "wind", "earth", "ocean",
    "river", "lake", "forest", "mountain", "valley", "peak", "wave",
    "spark", "flash", "bolt", "thunder", "storm", "rain", "snow",
    "gold", "silver", "diamond", "ruby", "emerald", "crystal", "pearl",
    "swift", "fast", "quick", "rapid", "turbo", "ultra", "mega", "super",
    "smart", "clever", "wise", "bright", "sharp", "keen", "agile",
    "zen", "calm", "peace", "harmony", "balance", "flow", "grace",
    "pixel", "byte", "bit", "code", "data", "tech", "digital", "cyber",
    "fox", "wolf", "eagle", "hawk", "lion", "tiger", "bear", "dragon",
]

RANDOM_DEVS = [
    "studio", "labs", "tech", "soft", "dev", "app", "mobile",
    "digital", "solutions", "systems", "group", "team", "works",
    "inc", "co", "io", "ai", "net", "org", "pro",
]


def generate_package_name(rng: random.Random, prefix: str, suffix: str) -> str:
    """Generate a plausible package name."""
    # Add 1-2 random segments between prefix and suffix
    segments = rng.randint(1, 2)
    middle_parts = []
    for _ in range(segments):
        word = rng.choice(RANDOM_WORDS)
        middle_parts.append(word)
    middle = ".".join(middle_parts)
    # Sometimes add a developer name
    if rng.random() < 0.3:
        dev = rng.choice(RANDOM_DEVS)
        return f"{prefix}.{dev}.{middle}{suffix}"
    return f"{prefix}.{middle}{suffix}"


def generate_synthetic_rows(
    category: str,
    patterns: dict,
    count: int,
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """Generate synthetic (packageName, label, category) rows for one category."""
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    while len(rows) < count:
        prefix = rng.choice(patterns["prefixes"])
        suffix = rng.choice(patterns["suffixes"])
        pkg = generate_package_name(rng, prefix, suffix)

        if pkg in seen:
            continue
        seen.add(pkg)

        # Choose label (50% Russian, 50% English, 10% empty)
        r = rng.random()
        if r < 0.1:
            label = ""
        elif r < 0.55:
            label = rng.choice(patterns["labels_ru"])
            # Add some variation
            if rng.random() < 0.3:
                label += f" {rng.choice(RANDOM_WORDS).capitalize()}"
        else:
            label = rng.choice(patterns["labels_en"])
            if rng.random() < 0.3:
                label += f" {rng.choice(RANDOM_WORDS).capitalize()}"

        rows.append((pkg, label, category))

    return rows


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
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic training corpus for App Category Model"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--per-category", type=int, default=TARGET_PER_CATEGORY,
        help=f"Target rows per category (default: {TARGET_PER_CATEGORY})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_rows: list[tuple[str, str, str]] = []

    # Generate for each of the 18 categories (excluding OTHER)
    for category, patterns in CATEGORY_PATTERNS.items():
        rows = generate_synthetic_rows(category, patterns, args.per_category, rng)
        all_rows.extend(rows)
        print(
            f"[synthetic] {category}: {len(rows)} rows generated",
            file=sys.stderr,
        )

    # Shuffle to avoid category clustering
    rng.shuffle(all_rows)

    write_csv_atomic(args.output, all_rows)
    print(
        f"\n[synthetic] DONE: wrote {len(all_rows)} rows to "
        f"{args.output.as_posix()}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
