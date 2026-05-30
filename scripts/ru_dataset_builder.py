"""
РФ-ориентированный сборщик датасета для TFLite модели.

Читает raw CSV из datasets/ru/raw/, объединяет источники,
формирует labels (ALLOW/WARN/BLOCK) и генерирует признаки
в формате, совместимом с train_ru_metadata_models.py.

Usage:
    python scripts/ru_dataset_builder.py
    python scripts/ru_dataset_builder.py --add-synthetic 2000
    python scripts/ru_dataset_builder.py --output datasets/ru/processed/ru_call_features.csv
"""

import csv
import os
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Импорты из sibling-скриптов
import sys
sys.path.insert(0, os.path.dirname(__file__))
from ru_number_normalizer import (
    normalize_ru_phone, is_russian_number, is_mobile_ru,
    is_landline_ru, is_tollfree_ru, is_short_code, get_def_code
)
from ru_numbering_plan import NumberingPlan, load_existing_csv as load_numbering_csv

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'ru')
RAW_DIR = os.path.join(BASE_DIR, 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')

LABEL_ALLOW = 0
LABEL_WARN = 1
LABEL_BLOCK = 2

# Приоритет источников: выше = надёжнее
SOURCE_PRIORITY = {
    'user_feedback': 10,
    'whitelist_official': 9,
    'blacklist_moshelovka': 8,
    'blacklist_spravportal': 7,
    'reviews_neberitrubku': 5,
    'reviews_zvonili': 4,
    'synthetic': 1,
}

# Категории, которые однозначно → BLOCK
BLOCK_CATEGORIES = {
    'телефонное мошенничество', 'мошенничество', 'мошенник',
    'вымогательство', 'фишинг', 'scam', 'fraud',
}

# Категории, которые → WARN
WARN_CATEGORIES = {
    'спам', 'нежелательный звонок', 'реклама', 'колл-центр',
    'телемаркетинг', 'опрос', 'навязывание услуг', 'робоколл',
    'сборщик задолженностей', 'коллекторы',
}

# Категории → ALLOW
ALLOW_CATEGORIES = {
    'банк', 'служба поддержки', 'доставка', 'официальный',
    'экстренная служба', 'медицина', 'государственный',
}


def load_raw_csv(filename: str) -> List[Dict]:
    """Загрузить CSV из raw/."""
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_whitelist() -> set:
    """Загрузить официальный whitelist."""
    numbers = set()
    for row in load_raw_csv('whitelist_official_ru.csv'):
        norm = normalize_ru_phone(row.get('normalized_number', ''), reject_non_ru=True)
        if norm:
            numbers.add(norm)
    return numbers


def load_blacklists() -> Dict[str, Dict]:
    """
    Загрузить все blacklist-источники.

    Возвращает dict: normalized_number → {source, category, confidence}
    """
    result = {}

    # Мошеловка ОНФ
    for row in load_raw_csv('blacklist_moshelovka.csv'):
        norm = normalize_ru_phone(row.get('normalized_number', ''), reject_non_ru=True)
        if norm:
            result[norm] = {
                'source': 'blacklist_moshelovka',
                'category': row.get('category', ''),
                'confidence': float(row.get('confidence', '1.0')),
            }

    # SpravPortal
    for row in load_raw_csv('blacklist_spravportal.csv'):
        norm = normalize_ru_phone(row.get('normalized_number', ''), reject_non_ru=True)
        if norm:
            existing = result.get(norm)
            new_conf = float(row.get('confidence', '0.8'))
            # Берём источник с более высоким приоритетом
            if not existing or SOURCE_PRIORITY.get('blacklist_spravportal', 0) > SOURCE_PRIORITY.get(existing['source'], 0):
                result[norm] = {
                    'source': 'blacklist_spravportal',
                    'category': row.get('category', ''),
                    'confidence': new_conf,
                }

    return result


def load_reviews() -> Dict[str, List[Dict]]:
    """
    Загрузить отзывы из neberitrubku/zvonili.

    Возвращает dict: normalized_number → [{source, rating, negative_count, positive_count, categories}]
    """
    result = defaultdict(list)

    for filename, source_name in [
        ('reviews_neberitrubku.csv', 'reviews_neberitrubku'),
        ('reviews_zvonili.csv', 'reviews_zvonili'),
    ]:
        for row in load_raw_csv(filename):
            norm = normalize_ru_phone(row.get('normalized_number', ''), reject_non_ru=True)
            if norm:
                result[norm].append({
                    'source': source_name,
                    'rating': row.get('rating', ''),
                    'negative_count': int(row.get('negative_count', '0') or '0'),
                    'positive_count': int(row.get('positive_count', '0') or '0'),
                    'categories': row.get('categories', ''),
                })

    return dict(result)


def determine_label(
    number: str,
    in_whitelist: bool,
    blacklist_info: Optional[Dict],
    reviews: List[Dict],
) -> Tuple[int, float, str]:
    """
    Определить label для номера.

    Возвращает (label, weight, source).
    """
    # 1. Официальный whitelist → ALLOW с высоким весом
    if in_whitelist:
        return LABEL_ALLOW, 2.0, 'whitelist_official'

    # 2. Blacklist → BLOCK
    if blacklist_info:
        cat_lower = blacklist_info['category'].lower()
        if any(bc in cat_lower for bc in BLOCK_CATEGORIES):
            return LABEL_BLOCK, 2.0, blacklist_info['source']
        if any(wc in cat_lower for wc in WARN_CATEGORIES):
            return LABEL_WARN, 1.5, blacklist_info['source']
        # По умолчанию blacklist → BLOCK
        return LABEL_BLOCK, 1.5, blacklist_info['source']

    # 3. Отзывы
    if reviews:
        total_neg = sum(r['negative_count'] for r in reviews)
        total_pos = sum(r['positive_count'] for r in reviews)
        all_cats = ' '.join(r['categories'].lower() for r in reviews)

        # Явное мошенничество в отзывах
        if any(bc in all_cats for bc in BLOCK_CATEGORIES) and total_neg > total_pos * 3:
            return LABEL_BLOCK, 1.0, 'reviews'

        # Спам/реклама
        if any(wc in all_cats for wc in WARN_CATEGORIES) and total_neg > total_pos:
            return LABEL_WARN, 1.0, 'reviews'

        # Больше негативных → WARN
        if total_neg > 5 and total_neg > total_pos * 2:
            return LABEL_WARN, 0.8, 'reviews'

        # Больше позитивных → ALLOW
        if total_pos > total_neg * 2 and total_pos >= 3:
            return LABEL_ALLOW, 0.8, 'reviews'

    # 4. Не найден нигде — неизвестный номер
    return LABEL_WARN, 0.3, 'unknown'


def compute_features(
    number: str,
    label: int,
    numbering_plan: Optional[NumberingPlan],
) -> List[float]:
    """
    Вычислить 20 признаков CallFeatures для данного номера.

    Это — «средний» профиль пользователя, т.к. реальные контекстные
    данные (контакты, время, приложения) доступны только на устройстве.
    Для обучения используем правдоподобные значения.
    """
    is_ru = float(is_russian_number(number))
    is_mobile = float(is_mobile_ru(number))
    is_landline = float(is_landline_ru(number))
    is_tollfree = float(is_tollfree_ru(number))
    is_short = float(is_short_code(number))
    is_foreign = 0.0  # РФ-only датасет

    # Оператор/регион из плана нумерации
    operator_info = numbering_plan.lookup(number) if numbering_plan else None
    is_valid_range = float(operator_info is not None)

    # prefixRisk: эвристика по DEF-коду
    def_code = get_def_code(number)
    prefix_risk = 0.0
    if def_code:
        # Некоторые DEF-коды чаще используются спамерами
        high_risk_defs = {900, 901, 902, 903, 904, 905, 906, 908, 909, 950, 951, 952, 953}
        medium_risk_defs = {910, 911, 912, 913, 914, 915, 916, 917, 918, 919,
                            920, 921, 922, 923, 924, 925, 926, 927, 928, 929,
                            930, 931, 932, 933, 934, 935, 936, 937, 938, 939}
        if def_code in high_risk_defs:
            prefix_risk = 0.6
        elif def_code in medium_risk_defs:
            prefix_risk = 0.3
        elif is_tollfree:
            prefix_risk = 0.1  # 8-800 обычно легитимные
        elif is_landline:
            prefix_risk = 0.2  # городские — нейтрально

    # inBlacklist / inAllowlist из label
    in_blacklist = 1.0 if label == LABEL_BLOCK else 0.0
    in_allowlist = 1.0 if label == LABEL_ALLOW else 0.0

    # Контекстные признаки — для обучения используем правдоподобные средние
    # На устройстве они будут реальными
    is_contact = 0.0  # неизвестные номера
    call_frequency = random.betavariate(2, 5) if label == LABEL_WARN else random.betavariate(1, 8)
    is_night_time = 1.0 if label == LABEL_BLOCK and random.random() < 0.25 else 0.0
    recent_bank_app = 1.0 if label == LABEL_BLOCK and random.random() < 0.3 else 0.0
    recent_gov_app = 0.0
    recent_marketplace_app = 0.0
    recent_messenger_app = 0.0
    previously_rejected = 1.0 if label == LABEL_BLOCK and random.random() < 0.4 else 0.0
    hidden_number = 0.0  # РФ-only датасет — номера видимые
    caller_verify_failed = 1.0 if label == LABEL_BLOCK and random.random() < 0.5 else 0.0

    # UserProfileVector — средние значения
    user_vulnerability = random.betavariate(2, 5)
    user_business_activity = random.betavariate(3, 4)
    contacts_available = 1.0  # предполагаем что контакты доступны
    usage_access_available = random.choice([0.0, 1.0])

    # Для ALLOW номеров — корректировки
    if label == LABEL_ALLOW:
        is_contact = 1.0 if random.random() < 0.4 else 0.0
        call_frequency = random.betavariate(5, 2)  # чаще звонят
        caller_verify_failed = 0.0
        previously_rejected = 0.0

    return [
        is_contact, is_ru, is_foreign, is_short,
        prefix_risk, min(call_frequency, 1.0), is_night_time,
        recent_bank_app, recent_gov_app, recent_marketplace_app,
        recent_messenger_app, previously_rejected, in_blacklist,
        in_allowlist, hidden_number, caller_verify_failed,
        user_vulnerability, user_business_activity,
        contacts_available, usage_access_available
    ]


def generate_synthetic_ru_samples(n: int, numbering_plan: Optional[NumberingPlan]) -> List[Dict]:
    """Генерировать синтетические РФ-образцы для балансировки."""
    samples = []
    # Основные DEF-коды для генерации
    def_codes = list(range(900, 970)) + [495, 498, 499, 800]

    for _ in range(n):
        def_code = random.choice(def_codes)
        subscriber = random.randint(0, 9999999 if def_code < 900 else 9999999)
        number = f'+7{def_code}{subscriber:07d}'

        # Определить тип
        if def_code == 800:
            label = LABEL_ALLOW
            weight = 1.0
        elif def_code in (495, 498, 499):
            label = random.choices([LABEL_ALLOW, LABEL_WARN], weights=[0.6, 0.4])[0]
            weight = 0.5
        else:
            label = random.choices([LABEL_ALLOW, LABEL_WARN, LABEL_BLOCK], weights=[0.3, 0.4, 0.3])[0]
            weight = 0.5

        features = compute_features(number, label, numbering_plan)
        samples.append({
            'number': number,
            'label': label,
            'weight': weight,
            'source': 'synthetic',
            'features': features,
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description='Build RF-only training dataset')
    parser.add_argument('--add-synthetic', type=int, default=0,
                        help='Add N synthetic RF samples for balancing')
    parser.add_argument('--output', type=str,
                        default=os.path.join(PROCESSED_DIR, 'ru_call_features.csv'),
                        help='Output training CSV path')
    parser.add_argument('--labeled-output', type=str,
                        default=os.path.join(PROCESSED_DIR, 'ru_numbers_labeled.csv'),
                        help='Output labeled numbers CSV path')
    args = parser.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Загрузить план нумерации
    print("Loading numbering plan...")
    numbering_records = load_numbering_csv(
        os.path.join(RAW_DIR, 'ru_numbering_plan.csv')
    )
    numbering_plan = NumberingPlan(numbering_records) if numbering_records else None
    if numbering_plan:
        print(f"  Loaded {len(numbering_records)} ranges")
    else:
        print("  WARNING: No numbering plan loaded, running without operator/region enrichment")

    # 2. Загрузить whitelist
    print("Loading whitelist...")
    whitelist = load_whitelist()
    print(f"  {len(whitelist)} official whitelist numbers")

    # 3. Загрузить blacklists
    print("Loading blacklists...")
    blacklists = load_blacklists()
    print(f"  {len(blacklists)} blacklisted numbers")

    # 4. Загрузить отзывы
    print("Loading reviews...")
    reviews = load_reviews()
    print(f"  {len(reviews)} numbers with reviews")

    # 5. Собрать все уникальные номера
    all_numbers = set()
    all_numbers.update(whitelist)
    all_numbers.update(blacklists.keys())
    all_numbers.update(reviews.keys())
    print(f"\nTotal unique RF numbers: {len(all_numbers)}")

    # 6. Определить labels и признаки
    print("Building dataset...")
    labeled_rows = []
    feature_rows = []

    for number in sorted(all_numbers):
        in_wl = number in whitelist
        bl_info = blacklists.get(number)
        rev_list = reviews.get(number, [])

        label, weight, source = determine_label(number, in_wl, bl_info, rev_list)
        features = compute_features(number, label, numbering_plan)

        labeled_rows.append({
            'normalized_number': number,
            'label': ['ALLOW', 'WARN', 'BLOCK'][label],
            'weight': weight,
            'source': source,
        })

        feature_rows.append(features + [label])

    # 7. Добавить синтетику если нужно
    if args.add_synthetic > 0:
        print(f"Adding {args.add_synthetic} synthetic samples...")
        synth = generate_synthetic_ru_samples(args.add_synthetic, numbering_plan)
        for s in synth:
            labeled_rows.append({
                'normalized_number': s['number'],
                'label': ['ALLOW', 'WARN', 'BLOCK'][s['label']],
                'weight': s['weight'],
                'source': s['source'],
            })
            feature_rows.append(s['features'] + [s['label']])

    # 8. Сохранить labeled CSV
    with open(args.labeled_output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['normalized_number', 'label', 'weight', 'source'])
        writer.writeheader()
        writer.writerows(labeled_rows)
    print(f"Saved {len(labeled_rows)} labeled numbers to {args.labeled_output}")

    # 9. Сохранить features CSV
    header = [
        'isContact', 'isRussianNumber', 'isForeignNumber', 'isShortCode',
        'prefixRisk', 'callFrequency', 'isNightTime', 'recentBankApp',
        'recentGovApp', 'recentMarketplaceApp', 'recentMessengerApp',
        'previouslyRejected', 'inBlacklist', 'inAllowlist',
        'hiddenNumber', 'callerVerifyFailed', 'userVulnerability',
        'userBusinessActivity', 'contactsAvailable', 'usageAccessAvailable',
        'label'
    ]
    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in feature_rows:
            writer.writerow(row)
    print(f"Saved {len(feature_rows)} feature rows to {args.output}")

    # 10. Статистика
    label_counts = defaultdict(int)
    for row in labeled_rows:
        label_counts[row['label']] += 1
    print(f"\nLabel distribution:")
    for lbl in ['ALLOW', 'WARN', 'BLOCK']:
        print(f"  {lbl}: {label_counts.get(lbl, 0)}")

    if not blacklists and not reviews:
        print("\n⚠ No blacklist/review data found. Dataset contains only whitelist + synthetic.")
        print("  To get real data, populate raw CSV files from:")
        print("  - SpravPortal API (scripts/ru_collect_sources.py)")
        print("  - Мошеловка ОНФ (moshelovka.onf.ru/blacklist/)")
        print("  - neberitrubku.ru / zvonili.com")
        print("  Or use --add-synthetic to add synthetic samples.")


if __name__ == '__main__':
    main()
