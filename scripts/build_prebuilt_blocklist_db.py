"""
Собирает app/src/main/assets/prebuilt_blocklist.db из spam_numbers.csv.

Зачем это нужно:
    Раньше CsvSpamImporter парсил 33-мегабайтный CSV (~2.4M строк) и
    вставлял каждый номер по одному в Room. На реальном устройстве это
    ~290 строк/сек (WAL-fsync на каждый INSERT × уникальный индекс →
    часы импорта на S24). Идея: подготовить готовый sqlite на этапе
    сборки и подложить его как asset. Kotlin-стороне останется только
    скопировать файл в filesDir и читать через SQLiteDatabase в read-only
    режиме. Никакого Room — Room для готового файла требует точного
    identity_hash, который проще не воспроизводить.

Формат входа (`spam_numbers.csv`):
    +74951234567          ← exact number
    prefix:+7800          ← prefix entry
    regex:8800.*          ← raw regex
    # comments

Формат выхода (`prebuilt_blocklist.db`):
    Таблица `prebuilt_blocked(normalizedNumber TEXT PRIMARY KEY,
    originalNumber TEXT, pattern TEXT)`. PK = уникальный индекс по
    normalizedNumber. Записи с pattern != NULL — regex/prefix-маски,
    Kotlin-сторона их прогонит через `containsMatchIn`.

    Таблица `meta(key TEXT PRIMARY KEY, value TEXT)` хранит версию.

Запуск:
    python scripts/build_prebuilt_blocklist_db.py
"""
from __future__ import annotations

import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "app" / "src" / "main" / "assets" / "spam_numbers.csv"
DB_OUT = ROOT / "app" / "src" / "main" / "assets" / "prebuilt_blocklist.db"

# Должен совпадать с PrebuiltBlocklistReader.BUNDLED_VERSION на стороне
# Kotlin. Когда поднимаешь — Kotlin-сторона перекопирует ассет в filesDir
# при следующем старте.
BUNDLED_VERSION = 1


def normalize(raw: str) -> str | None:
    """Урезанный аналог [com.antispam.blocker.util.PhoneNormalizer.normalize].

    Должен давать тот же ключ, что PhoneNormalizer.kt — иначе exact-lookup
    из BlockListRepository не найдёт совпадения. Логика:
      - "+7XXX..." → "+" + только цифры
      - "8XXXXXXXXXX" (11 цифр) → "+7" + последние 10
      - "7XXXXXXXXXX" (11 цифр) → "+7" + последние 10
      - 10 цифр → считаем РФ-номером, "+7" + 10 цифр
      - всё остальное с ≥ 7 цифр → "+" + цифры
    """
    raw = raw.strip()
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None
    if raw.startswith("+"):
        return "+" + digits
    if len(digits) == 11 and digits.startswith("8"):
        return "+7" + digits[1:]
    if len(digits) == 11 and digits.startswith("7"):
        return "+7" + digits[1:]
    if len(digits) == 10:
        return "+7" + digits
    if len(digits) >= 7:
        return "+" + digits
    return None


def main() -> int:
    if not CSV_PATH.exists():
        print(f"FAIL: {CSV_PATH} не найден", file=sys.stderr)
        return 1

    DB_OUT.parent.mkdir(parents=True, exist_ok=True)
    if DB_OUT.exists():
        DB_OUT.unlink()

    started = time.time()
    conn = sqlite3.connect(DB_OUT)
    # PRAGMA на этапе сборки — для скорости. На устройстве БД открывается
    # в режиме по умолчанию и эти настройки не наследуются.
    conn.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA page_size = 4096;

        CREATE TABLE prebuilt_blocked (
            normalizedNumber TEXT PRIMARY KEY NOT NULL,
            originalNumber   TEXT NOT NULL,
            pattern          TEXT
        );
        CREATE INDEX idx_prebuilt_blocked_pattern
            ON prebuilt_blocked(pattern)
            WHERE pattern IS NOT NULL;

        CREATE TABLE meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )

    exact_rows: list[tuple[str, str, None]] = []
    pattern_rows: list[tuple[str, str, str]] = []
    seen_exact: set[str] = set()
    seen_patterns: set[str] = set()
    skipped = 0

    with CSV_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("prefix:"):
                prefix = stripped.removeprefix("prefix:").strip()
                if not prefix:
                    continue
                pat = "^" + re.escape(prefix)
                if pat in seen_patterns:
                    continue
                seen_patterns.add(pat)
                pattern_rows.append((prefix, prefix, pat))
            elif stripped.startswith("regex:"):
                pat = stripped.removeprefix("regex:").strip()
                if not pat or pat in seen_patterns:
                    continue
                try:
                    re.compile(pat)
                except re.error:
                    skipped += 1
                    continue
                seen_patterns.add(pat)
                pattern_rows.append((pat, pat, pat))
            else:
                normalized = normalize(stripped)
                if not normalized or normalized in seen_exact:
                    skipped += 1
                    continue
                seen_exact.add(normalized)
                exact_rows.append((normalized, stripped, None))

    cur = conn.cursor()
    chunk = 50_000
    for i in range(0, len(exact_rows), chunk):
        cur.executemany(
            "INSERT OR IGNORE INTO prebuilt_blocked "
            "(normalizedNumber, originalNumber, pattern) VALUES (?, ?, ?)",
            exact_rows[i : i + chunk],
        )
    cur.executemany(
        "INSERT OR IGNORE INTO prebuilt_blocked "
        "(normalizedNumber, originalNumber, pattern) VALUES (?, ?, ?)",
        pattern_rows,
    )
    cur.executemany(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        [
            ("version", str(BUNDLED_VERSION)),
            ("exact_count", str(len(exact_rows))),
            ("pattern_count", str(len(pattern_rows))),
            ("built_at", str(int(time.time()))),
        ],
    )

    conn.commit()
    conn.execute("VACUUM")
    conn.close()

    size_mb = DB_OUT.stat().st_size / (1024 * 1024)
    print(
        f"Готово: {DB_OUT.relative_to(ROOT)}  "
        f"exact={len(exact_rows):,}  pattern={len(pattern_rows):,}  "
        f"skipped={skipped:,}  size={size_mb:.1f}MB  "
        f"time={time.time() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
