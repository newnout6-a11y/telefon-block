"""
РФ-нормализация телефонных номеров.

Приводит номера к формату +7XXXXXXXXXX (E.164 для РФ).
Отбрасывает не-РФ номера при включённом фильтре.
"""

import re
from typing import Optional


# РФ мобильные DEF-коды (3-значные, начинаются с 9)
RU_MOBILE_DEF_CODES = set(range(900, 1000))  # 900-999

# РФ городские ABC-коды (актуальный реестр Россвязи / план нумерации).
# Раньше тут было `set(range(300, 500))` (300-499 оптом), но половина этого
# диапазона нерабочая — Россия использует только подмножество, остальное
# либо не выделено, либо делегировано Казахстану / Узбекистану внутри +7.
# Открытый «гибкий» диапазон давал ложные ALLOW из любого мусорного источника
# (см. инцидент с yandex_maps fallback regex).  Список ниже — **геокоды,
# реально используемые в РФ** (январь 2025), плюс служебные 800/804.
RU_LANDLINE_ABC_CODES = {
    # 30x — Сибирь / Дальний Восток (Бурятия, Забайкалье)
    301, 302,
    # 34x — Урал/Поволжье (Ижевск, Пермь, Екатеринбург, Тюмень, Уфа и т.п.)
    341, 342, 343, 345, 346, 347, 349,
    # 35x — Челябинск, Курган, Оренбург, Калмыкия и т.п.
    351, 352, 353, 365,
    # 38x — Сибирь (Омск, Томск, Новосибирск, Кемерово, Алтай, Хакасия и т.п.)
    381, 382, 383, 384, 385, 388, 390, 391, 394, 395,
    # 41x — Дальний Восток (Якутия, Магадан, Камчатка, Сахалин, Амур)
    411, 413, 415, 416,
    # 42x — Хабаровск, Приморье, Сахалин
    421, 422, 423, 424, 425, 426, 427,
    # 47x — ЦФО юг (Курск, Белгород, Воронеж, Липецк, Тамбов и т.п.)
    471, 472, 473, 474, 475,
    # 48x — ЦФО север+Москва, Подмосковье, Ярославль, Владимир, Иваново и т.п.
    481, 482, 483, 484, 485, 486, 487, 491, 492, 493, 494, 495, 496, 497, 498, 499,
    # 81x — С-З (СПб, ЛО, Карелия, Мурманск, Новгород, Вологда, Архангельск)
    812, 813, 814, 815, 816, 817, 818,
    # 82x — Север (Коми, Псков и т.п.)
    820, 821, 822,
    # 83x — Поволжье север (Нижний, Йошкар-Ола, Саранск, Чебоксары)
    831, 833, 834, 835,
    # 84x — Поволжье центр (Пенза, Казань, Волгоград, Саратов, Самара, Ульяновск)
    841, 842, 843, 844, 845, 846, 847, 848, 855,
    # 85x — Поволжье юг
    851, 852, 853,
    # 86x — Северный Кавказ (Краснодар, Сочи, Ростов, Ставрополь, КБР, СО)
    861, 862, 863, 865, 866, 867,
    # 87x — Северный Кавказ юг (Чечня, Дагестан, КЧР, Ингушетия, Калмыкия, Адыгея)
    871, 872, 873, 877, 878, 879,
}

# РФ 8-800 / 8-804 (toll-free)
RU_TOLLFREE_CODES = {800, 804}

# Короткие/экстренные номера РФ
RU_EMERGENCY_SHORT = {'101', '102', '103', '104', '112'}

# Общий паттерн: цифры из номера
DIGITS_RE = re.compile(r'\d')


def normalize_ru_phone(raw: str, reject_non_ru: bool = True) -> Optional[str]:
    """
    Нормализовать номер к +7XXXXXXXXXX.

    Возвращает None если:
    - номер не похож на телефонный
    - reject_non_ru=True и номер не российский
    """
    if not raw or not raw.strip():
        return None

    s = raw.strip()

    # Короткие экстренные номера
    digits = DIGITS_RE.findall(s)
    digit_str = ''.join(digits)

    if digit_str in RU_EMERGENCY_SHORT:
        return f'+7{digit_str}'

    # Убираем лидирующую 8 или 7 для 11-значных российских номеров
    if len(digit_str) == 11 and digit_str[0] in ('7', '8'):
        core = digit_str[1:]
        return f'+7{core}'

    # 10-значный номер без кода страны — предполагаем РФ
    if len(digit_str) == 10 and digit_str[0] in ('9', '3', '4', '8'):
        return f'+7{digit_str}'

    # Международный формат с +
    if s.startswith('+') and len(digit_str) >= 11:
        country_code = digit_str[:1] if digit_str[0] == '7' else digit_str[:2]
        if country_code == '7':
            return f'+7{digit_str[1:11]}'
        elif reject_non_ru:
            return None
        else:
            return f'+{digit_str}'

    # 7-9 цифр — может быть городской без кода
    if 7 <= len(digit_str) <= 9:
        return None  # недостаточно данных для нормализации

    # Не распознано
    return None


def is_russian_number(normalized: str) -> bool:
    """Проверить что нормализованный номер — российский."""
    if not normalized or not normalized.startswith('+7'):
        return False
    digits = normalized[2:]  # убираем +7
    if not digits:
        return False
    # Короткие экстренные (101/102/103/104/112) — это валидный РФ ввод.
    if digits in RU_EMERGENCY_SHORT:
        return True
    if len(digits) < 3:
        return False
    def_code = int(digits[:3])
    return def_code in RU_MOBILE_DEF_CODES or def_code in RU_LANDLINE_ABC_CODES or def_code in RU_TOLLFREE_CODES


def is_mobile_ru(normalized: str) -> bool:
    """Мобильный РФ номер (DEF 9xx)."""
    if not normalized or not normalized.startswith('+7'):
        return False
    digits = normalized[2:]
    if len(digits) < 3:
        return False
    return int(digits[:3]) in RU_MOBILE_DEF_CODES


def is_landline_ru(normalized: str) -> bool:
    """Городской РФ номер (ABC 3xx-4xx)."""
    if not normalized or not normalized.startswith('+7'):
        return False
    digits = normalized[2:]
    if len(digits) < 3:
        return False
    return int(digits[:3]) in RU_LANDLINE_ABC_CODES


def is_tollfree_ru(normalized: str) -> bool:
    """8-800 номер."""
    if not normalized or not normalized.startswith('+7'):
        return False
    digits = normalized[2:]
    if len(digits) < 3:
        return False
    return int(digits[:3]) in RU_TOLLFREE_CODES


def is_short_code(normalized: str) -> bool:
    """Короткий/экстренный номер."""
    if not normalized or not normalized.startswith('+7'):
        return False
    digits = normalized[2:]
    return digits in RU_EMERGENCY_SHORT or (len(digits) <= 4 and digits.isdigit())


# Placeholder/junk patterns: 6+ trailing zeros after a real def-code.
# Real toll-free hotlines like 8800-555-3535 don't end in long zero runs.
# Crawlers occasionally pick up `+74950000000`-style entries from listings
# that use round numbers as fillers.
_PLACEHOLDER_TRAILING_ZEROS_RE = re.compile(r'0{6,}$')


def is_placeholder_number(normalized: Optional[str]) -> bool:
    """True если номер выглядит как заглушка (≥6 нулей в конце)."""
    if not normalized or not normalized.startswith('+7'):
        return False
    digits = normalized[2:]
    if len(digits) < 7:
        return False
    return bool(_PLACEHOLDER_TRAILING_ZEROS_RE.search(digits))


def is_valid_ru_phone(normalized: Optional[str]) -> bool:
    """Композитная проверка: РФ-номер И не заглушка.

    Используется в `merge_crawler_shards.py` и `ru_metadata_dataset_builder.py`,
    чтобы отфильтровать казахские (+77XX), невалидные def-коды (+70XX/+71XX/+72XX)
    и round-number заглушки (например +74950000000) до попадания в обучающий датасет.
    """
    if not normalized:
        return False
    if not is_russian_number(normalized):
        return False
    if is_placeholder_number(normalized):
        return False
    return True


def get_def_code(normalized: str) -> Optional[int]:
    """Получить DEF/ABC код из нормализованного номера."""
    if not normalized or not normalized.startswith('+7'):
        return None
    digits = normalized[2:]
    if len(digits) < 3:
        return None
    return int(digits[:3])


if __name__ == '__main__':
    # Quick test
    test_numbers = [
        '+79854430013', '89854430013', '79854430013',
        '+79161234567', '+74957754747', '+78005553535',
        '88005553535', '103', '112', '+12125551234',
        '89161234567', '+79001234567', '89001234567',
    ]
    for n in test_numbers:
        norm = normalize_ru_phone(n)
        if norm:
            print(f'{n:20s} → {norm:15s} RU={is_russian_number(norm)} mobile={is_mobile_ru(norm)} landline={is_landline_ru(norm)} tollfree={is_tollfree_ru(norm)} short={is_short_code(norm)}')
        else:
            print(f'{n:20s} → None (rejected)')
