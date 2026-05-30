"""
Скачать и распарсить официальный план нумерации РФ (Россвязь/Минцифры).

Источники:
  - https://opendata.digital.gov.ru/registry/numeric/
  - Старые CSV: https://rossvyaz.ru/data/ABC-3xx.csv, ABC-4xx.csv, ABC-7xx.csv, ABC-8xx.csv, ABC-9xx.csv

Формат выходного CSV:
  def_code,start_number,end_number,operator,region,capacity,number_type
"""

import csv
import os
import urllib.request
import re
from bisect import bisect_right
from typing import List, Dict, Optional, Tuple

# URL-шаблоны Россвязи (старый формат, всё ещё доступен)
ROSSVYAZ_URLS = {
    'ABC-3xx': 'https://opendata.digital.gov.ru/downloads/ABC-3xx.csv',
    'ABC-4xx': 'https://opendata.digital.gov.ru/downloads/ABC-4xx.csv',
    'ABC-8xx': 'https://opendata.digital.gov.ru/downloads/ABC-8xx.csv',
    'DEF-9xx': 'https://opendata.digital.gov.ru/downloads/DEF-9xx.csv',
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'ru', 'raw')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'ru_numbering_plan.csv')

# Россвязь CSV: ; разделитель, кодировка windows-1251
# Колонки: АВС/ DEF; От; До; Ёмкость; Оператор; Регион
# Пример строки: 916;0;999999;1000000;МТС;Москва


def download_rossvyaz_csv(url: str) -> Optional[str]:
    """Скачать CSV с Россвязи."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            # Попробовать windows-1251, затем utf-8
            try:
                return raw.decode('windows-1251')
            except UnicodeDecodeError:
                return raw.decode('utf-8')
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None


def parse_rossvyaz_csv(text: str) -> List[Dict]:
    """Распарсить CSV Россвязи в записи."""
    records = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split(';')
        if len(parts) < 6:
            continue

        try:
            def_code = int(parts[0].strip())
            start_num = int(parts[1].strip())
            end_num = int(parts[2].strip())
            capacity = int(parts[3].strip())
            operator = parts[4].strip().strip('"')
            region = parts[5].strip().strip('"')
        except (ValueError, IndexError):
            continue

        # Определить тип номера
        if 900 <= def_code <= 999:
            number_type = 'mobile'
        elif 300 <= def_code <= 499:
            number_type = 'landline'
        elif def_code == 800:
            number_type = 'tollfree'
        elif def_code == 700 or def_code == 780:
            number_type = 'virtual'
        else:
            number_type = 'other'

        records.append({
            'def_code': def_code,
            'start_number': start_num,
            'end_number': end_num,
            'operator': operator,
            'region': region,
            'capacity': capacity,
            'number_type': number_type,
        })

    return records


def load_existing_csv(path: str) -> List[Dict]:
    """Загрузить ранее скачанный CSV."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['def_code'] = int(row['def_code'])
            row['start_number'] = int(row['start_number'])
            row['end_number'] = int(row['end_number'])
            row['capacity'] = int(row['capacity'])
            records.append(row)
    return records


def save_csv(records: List[Dict], path: str):
    """Сохранить записи в CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'def_code', 'start_number', 'end_number',
            'operator', 'region', 'capacity', 'number_type'
        ])
        writer.writeheader()
        writer.writerows(records)


class NumberingPlan:
    """Быстрый поиск оператора/региона по номеру."""

    def __init__(self, records: List[Dict]):
        # Индекс: def_code → отсортированные диапазоны. Линейный scan по
        # сотням тысяч строк Россвязи стал узким местом на multi-million CSV.
        self._index: Dict[int, List[Tuple[int, int, Dict]]] = {}
        for r in records:
            dc = r['def_code']
            if dc not in self._index:
                self._index[dc] = []
            self._index[dc].append((r['start_number'], r['end_number'], r))
        self._starts: Dict[int, List[int]] = {}
        for dc, ranges in self._index.items():
            ranges.sort(key=lambda item: item[0])
            self._starts[dc] = [start for start, _end, _record in ranges]

    def lookup(self, normalized_number: str) -> Optional[Dict]:
        """
        Найти оператора и регион по нормализованному номеру (+7XXXXXXXXXX).

        Возвращает dict с operator, region, number_type или None.
        """
        if not normalized_number or not normalized_number.startswith('+7'):
            return None

        digits = normalized_number[2:]  # убираем +7
        if len(digits) < 7:
            return None

        try:
            def_code = int(digits[:3])
            subscriber = int(digits[3:])
        except ValueError:
            return None

        ranges = self._index.get(def_code)
        starts = self._starts.get(def_code)
        if not ranges or not starts:
            return None
        idx = bisect_right(starts, subscriber) - 1
        if idx >= 0:
            start_number, end_number, r = ranges[idx]
            if start_number <= subscriber <= end_number:
                return {
                    'operator': r['operator'],
                    'region': r['region'],
                    'number_type': r['number_type'],
                    'def_code': def_code,
                }

        return None

    def is_valid_ru_range(self, normalized_number: str) -> bool:
        """Проверить что номер попадает в зарегистрированный диапазон РФ."""
        return self.lookup(normalized_number) is not None


def main():
    """Скачать и сохранить план нумерации РФ."""
    all_records = []

    for name, url in ROSSVYAZ_URLS.items():
        print(f"Downloading {name} from {url}...")
        text = download_rossvyaz_csv(url)
        if text:
            records = parse_rossvyaz_csv(text)
            print(f"  Parsed {len(records)} ranges")
            all_records.extend(records)
        else:
            print(f"  Failed, trying existing file...")

    if not all_records:
        # Попробовать загрузить ранее скачанный
        print("No data downloaded, checking existing CSV...")
        all_records = load_existing_csv(OUTPUT_FILE)

    if not all_records:
        print("ERROR: No numbering plan data available!")
        print("Please manually download from https://rossvyaz.ru/deyatelnost/resurs-numeracii/vypiska-iz-reestra-sistemy-i-plana-numeracii")
        return

    # Дедупликация
    seen = set()
    unique = []
    for r in all_records:
        key = (r['def_code'], r['start_number'], r['end_number'])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    save_csv(unique, OUTPUT_FILE)
    print(f"\nSaved {len(unique)} ranges to {OUTPUT_FILE}")

    # Статистика
    types = {}
    for r in unique:
        t = r['number_type']
        types[t] = types.get(t, 0) + 1
    print("Breakdown by type:")
    for t, c in sorted(types.items()):
        print(f"  {t}: {c}")


if __name__ == '__main__':
    main()
