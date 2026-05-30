"""
Feature engineering для РФ metadata-pipeline.

Содержит единый порядок компактных признаков для TFLite/Android.
"""

import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from ru_number_normalizer import (
    normalize_ru_phone,
    is_russian_number,
    is_mobile_ru,
    is_landline_ru,
    is_tollfree_ru,
    is_short_code,
)

FEATURES_VERSION = 4

# Перевод заголовков CSV: английский → русский
FIELD_TO_RU = {
    'normalized_number': 'номер',
    'source': 'источник',
    'label': 'метка',
    'label_id': 'id_метки',
    'weight': 'вес',
    'label_hint': 'подсказка_метки',
    'evidence_type': 'тип_доказательства',
    'evidence_text': 'текст_доказательства',
    'full_text': 'полный_текст',
    'page_title': 'заголовок_страницы',
    'negative_count': 'негативных',
    'positive_count': 'позитивных',
    'neutral_count': 'нейтральных',
    'review_count': 'отзывов',
    'view_count': 'просмотров',
    'related_count': 'связанных',
    'categories': 'категории',
    'source_confidence': 'уверенность_источника',
    'source_reliability': 'надёжность_источника',
    'detail_date': 'дата_детали',
    'fraud_hits': 'мошеннических_совпадений',
    'warn_hits': 'предупреждений',
    'url': 'ссылка',
    'collected_at': 'собрано',
    'search_volume': 'объём_поиска',
    'search_volume_log': 'лог_объёма_поиска',
    'last_review_at': 'последний_отзыв',
    'first_seen_at': 'первое_появление',
    'negative_ratio': 'доля_негативных',
    'positive_ratio': 'доля_позитивных',
    'review_velocity_48h': 'скорость_отзывов_48ч',
    'review_velocity_7d': 'скорость_отзывов_7д',
    'has_fraud_category': 'есть_мошенничество',
    'has_telemarketing_category': 'есть_телемаркетинг',
    'has_finance_category': 'есть_финансы',
    'number_type': 'тип_номера',
    'def_code': 'деф_код',
    'operator': 'оператор',
    'region': 'регион',
    'timezone_offset': 'смещение_часового_пояса',
    'is_mvno': 'мвно',
    'view_count_log': 'лог_просмотров',
    'isContact': 'в_контактах',
    'isRussianNumber': 'российский_номер',
    'isForeignNumber': 'иностранный_номер',
    'isShortCode': 'короткий_номер',
    'isStandardLen': 'стандартная_длина',
    'isTollFree8800': 'бесплатный_8800',
    'isGeographical': 'географический',
    'isMobileRu': 'мобильный_рф',
    'isValidRuRange': 'валидный_диапазон_рф',
    'spoofingPrefixFlag': 'флаг_подмены_префикса',
    'digitEntropy': 'энтропия_цифр',
    'repeatDigitRatio': 'доля_повторов_цифр',
    'maxSameDigitRun': 'макс_повтор_цифры',
    'beautifulNumberFlag': 'красивый_номер',
    'prefixRisk': 'риск_префикса',
    'callFrequency': 'частота_звонков',
    'isNightTime': 'ночное_время',
    'recentBankApp': 'недавнее_приложение_банка',
    'recentGovApp': 'недавнее_приложение_госуслуг',
    'recentMarketplaceApp': 'недавнее_приложение_маркетплейса',
    'recentMessengerApp': 'недавнее_приложение_мессенджера',
    'previouslyRejected': 'ранее_отклонён',
    'inBlacklist': 'в_чёрном_списке',
    'inAllowlist': 'в_белом_списке',
    'hiddenNumber': 'скрытый_номер',
    'callerVerifyFailed': 'проверка_звонящего_не_пройдена',
    'userVulnerability': 'уязвимость_пользователя',
    'userBusinessActivity': 'деловая_активность',
    'contactsAvailable': 'контакты_доступны',
    'usageAccessAvailable': 'доступ_к_использованию',
    'reputationScore': 'рейтинг_репутации',
    'sourceConfidence': 'уверенность_источника_компакт',
    # Phase 3 (v3) extension: 15 новых фичей для cold-start.
    'operatorMts': 'оператор_мтс',
    'operatorMegafon': 'оператор_мегафон',
    'operatorBeeline': 'оператор_билайн',
    'operatorTele2': 'оператор_теле2',
    'operatorMvno': 'оператор_мвно',
    'defCodeRisk': 'риск_деф_кода',
    'prefixBlockShare': 'доля_блока_в_префиксе',
    'prefixWarnShare': 'доля_предупр_в_префиксе',
    'prefixSeenLog': 'лог_видимости_префикса',
    'reviewsLog': 'лог_отзывов',
    'negativeRatio': 'доля_негативных_компакт',
    'searchVolumeLog': 'лог_объёма_поиска_компакт',
    'hasFraudCategory': 'есть_мошенничество_компакт',
    'hasTelemarketingCategory': 'есть_телемаркетинг_компакт',
    'noMetadata': 'нет_метаданных',
    # Phase 4B (v4): +5 cold-survivable фичи на основе мульти-резолюционных префиксов.
    'prefixBlockShare3': 'доля_блока_префикс3',
    'prefixBlockShare7': 'доля_блока_префикс7',
    'prefixEntropy': 'энтропия_лейблов_префикса',
    'defCodeOperatorRisk': 'риск_деф_код_оператор',
    'prefixSampleSize': 'размер_выборки_префикса',
}
RU_TO_FIELD = {v: k for k, v in FIELD_TO_RU.items()}


def translate_headers(fields: Sequence[str], to_ru: bool = True) -> List[str]:
    """Перевести список заголовков: en→ru (to_ru=True) или ru→en (to_ru=False)."""
    mapping = FIELD_TO_RU if to_ru else RU_TO_FIELD
    return [mapping.get(f, f) for f in fields]


def translate_row(row: Dict, to_ru: bool = True) -> Dict:
    """Перевести ключи словаря строки: en→ru или ru→en."""
    mapping = FIELD_TO_RU if to_ru else RU_TO_FIELD
    return {mapping.get(k, k): v for k, v in row.items()}

# Базовые 32 фичи v2 (оставлены без изменений в порядке для совместимости артефактов).
_BASE_COMPACT_FEATURES_V2 = [
    'isContact',
    'isRussianNumber',
    'isForeignNumber',
    'isShortCode',
    'isStandardLen',
    'isTollFree8800',
    'isGeographical',
    'isMobileRu',
    'isValidRuRange',
    'spoofingPrefixFlag',
    'digitEntropy',
    'repeatDigitRatio',
    'maxSameDigitRun',
    'beautifulNumberFlag',
    'prefixRisk',
    'callFrequency',
    'isNightTime',
    'recentBankApp',
    'recentGovApp',
    'recentMarketplaceApp',
    'recentMessengerApp',
    'previouslyRejected',
    'inBlacklist',
    'inAllowlist',
    'hiddenNumber',
    'callerVerifyFailed',
    'userVulnerability',
    'userBusinessActivity',
    'contactsAvailable',
    'usageAccessAvailable',
    'reputationScore',
    'sourceConfidence',
]

# Phase 3: +15 фичей. Все runtime-вычислимые на Android из шиплемых JSON-лукапов:
#   * operator_bucket.json (def_code → mts/megafon/beeline/tele2/mvno/other)
#   * def_code_risk.json   (def_code → P(BLOCK | def_code))
#   * prefix_histogram.json(6-digit prefix → {block_share, warn_share, seen_count})
# При cold-start аугментации reputation/reviews/negativeRatio/searchVolume/hasFraud/
# hasTelemarketing зануляются для X% BLOCK/WARN строк, чтобы модель училась распознавать
# спам без метаданных, опираясь на operator/def_code/prefix-сигналы.
_COMPACT_FEATURES_V3_EXTENSION = [
    'operatorMts',
    'operatorMegafon',
    'operatorBeeline',
    'operatorTele2',
    'operatorMvno',
    'defCodeRisk',
    'prefixBlockShare',
    'prefixWarnShare',
    'prefixSeenLog',
    'reviewsLog',
    'negativeRatio',
    'searchVolumeLog',
    'hasFraudCategory',
    'hasTelemarketingCategory',
    'noMetadata',
]

# Phase 4B: +5 cold-survivable фичи. Мульти-резолюционные prefix histograms
# (3-digit для робастности, 7-digit для точности), label entropy, operator×def_code
# crossfeature, sample-size confidence. Все вычислимы на устройстве из shipped JSONs:
#   * prefix_histogram_3.json (3-digit phone prefix → block_share)
#   * prefix_histogram_7.json (7-digit phone prefix → block_share)
#   * prefix_histogram.json   (entropy field в каждом entry — добавлено в 4B)
#   * def_code_operator_risk.json (operator_bucket × def_code → block_share)
# noMetadata остаётся индикатором cold-start, новые фичи дают модели offline-сигнал.
_COMPACT_FEATURES_V4_EXTENSION = [
    'prefixBlockShare3',
    'prefixBlockShare7',
    'prefixEntropy',
    'defCodeOperatorRisk',
    'prefixSampleSize',
]

COMPACT_FEATURES = (
    _BASE_COMPACT_FEATURES_V2
    + _COMPACT_FEATURES_V3_EXTENSION
    + _COMPACT_FEATURES_V4_EXTENSION
)

FULL_METADATA_FIELDS = [
    'normalized_number',
    'label',
    'label_id',
    'weight',
    'source',
    'source_confidence',
    'negative_count',
    'positive_count',
    'neutral_count',
    'review_count',
    'negative_ratio',
    'positive_ratio',
    'search_volume',
    'search_volume_log',
    'review_velocity_48h',
    'review_velocity_7d',
    'has_fraud_category',
    'has_telemarketing_category',
    'has_finance_category',
    'number_type',
    'def_code',
    'operator',
    'region',
    'timezone_offset',
    'is_mvno',
    'source_reliability',
    'view_count',
    'view_count_log',
    'related_count',
    'detail_date',
] + COMPACT_FEATURES

LABEL_TO_ID = {
    'ALLOW': 0,
    'WARN': 1,
    'BLOCK': 2,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

FRAUD_KEYWORDS = {
    'мошен', 'фрод', 'scam', 'fraud', 'фишинг', 'вымог', 'служба безопасности',
    'подозр', 'обман', 'карта', 'банк мошен', 'безопасности банка',
}

TELEMARKETING_KEYWORDS = {
    'спам', 'реклама', 'телемаркетинг', 'колл', 'опрос', 'робот', 'робозвон',
    'навязы', 'займ', 'кредит', 'мфо', 'коллектор', 'продажи',
}

FINANCE_KEYWORDS = {
    'банк', 'кредит', 'займ', 'мфо', 'карта', 'финанс', 'страхов',
}

MVNO_HINTS = {
    'tinkoff', 'тинькофф мобайл', 'yota', 'йота', 'сбермобайл', 'ростелеком',
    'газпромбанк мобайл', 'втб мобайл', 'алло инкогнито',
}


def digits_only(value: Optional[str]) -> str:
    return ''.join(ch for ch in (value or '') if ch.isdigit())


def clamp01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None or value == '':
            return default
        return int(float(str(value).replace(',', '.')))
    except Exception:
        return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == '':
            return default
        return float(str(value).replace(',', '.'))
    except Exception:
        return default


def shannon_entropy(digits: str) -> float:
    if not digits:
        return 0.0
    counts = {d: digits.count(d) for d in set(digits)}
    entropy = 0.0
    for count in counts.values():
        p = count / len(digits)
        entropy -= p * math.log2(p)
    return entropy / math.log2(10)  # normalize roughly to 0..1


def repeat_digit_ratio(digits: str) -> float:
    if len(digits) <= 1:
        return 0.0
    repeats = sum(1 for i in range(1, len(digits)) if digits[i] == digits[i - 1])
    return repeats / (len(digits) - 1)


def max_same_digit_run(digits: str) -> float:
    if not digits:
        return 0.0
    best = 1
    current = 1
    for i in range(1, len(digits)):
        if digits[i] == digits[i - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return min(best / max(len(digits), 1), 1.0)


def has_monotonic_run(digits: str, min_run: int = 4) -> bool:
    if len(digits) < min_run:
        return False
    for i in range(0, len(digits) - min_run + 1):
        chunk = digits[i:i + min_run]
        asc = all(int(chunk[j]) == (int(chunk[j - 1]) + 1) % 10 for j in range(1, len(chunk)))
        desc = all(int(chunk[j]) == (int(chunk[j - 1]) - 1) % 10 for j in range(1, len(chunk)))
        if asc or desc:
            return True
    return False


def beautiful_number_flag(digits: str) -> bool:
    if not digits:
        return False
    tail = digits[-7:]
    return (
        repeat_digit_ratio(tail) >= 0.45
        or max_same_digit_run(tail) >= 4 / max(len(tail), 1)
        or bool(re.search(r'(\d{2,3})\1+', tail))
        or has_monotonic_run(tail)
    )


def spoofing_prefix_flag(raw_number: Optional[str], normalized_number: Optional[str]) -> bool:
    raw = (raw_number or normalized_number or '').strip().replace(' ', '').replace('-', '').replace('(', '').replace(')', '')
    digits = digits_only(raw)
    if raw.startswith('+84') and digits.startswith('8495'):
        return True
    if raw.startswith('0084') and digits.startswith('008495'):
        return True
    if normalized_number and normalized_number.startswith('+84') and digits_only(normalized_number).startswith('8495'):
        return True
    return False


def parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    for fmt in ('%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%d.%m.%Y', '%d.%m.%Y %H:%M'):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None


def review_velocity(last_review_at: Optional[str], first_seen_at: Optional[str], review_count: int, window_days: int) -> float:
    last_dt = parse_date(last_review_at)
    first_dt = parse_date(first_seen_at)
    if not last_dt or not first_dt or review_count <= 0:
        return 0.0
    age_days = max((last_dt - first_dt).total_seconds() / 86400.0, 1.0)
    return clamp01((review_count / age_days) / max(window_days, 1))


def category_flags(categories: Optional[str]) -> Dict[str, float]:
    text = (categories or '').lower()
    return {
        'has_fraud_category': 1.0 if any(k in text for k in FRAUD_KEYWORDS) else 0.0,
        'has_telemarketing_category': 1.0 if any(k in text for k in TELEMARKETING_KEYWORDS) else 0.0,
        'has_finance_category': 1.0 if any(k in text for k in FINANCE_KEYWORDS) else 0.0,
    }


def operator_bucket(operator: Optional[str]) -> str:
    text = (operator or '').strip().lower()
    if not text:
        return 'unknown'
    if 'мтс' in text or 'mts' in text:
        return 'mts'
    if 'мегафон' in text or 'megafon' in text:
        return 'megafon'
    if 'билайн' in text or 'beeline' in text or 'вымпел' in text:
        return 'beeline'
    if 'tele2' in text or 'теле2' in text or 't2' in text:
        return 'tele2'
    if any(h in text for h in MVNO_HINTS):
        return 'mvno'
    return 'other'


def stable_bucket(value: Optional[str], buckets: int = 16) -> int:
    text = (value or '').lower().strip()
    if not text:
        return 0
    acc = 0
    for ch in text:
        acc = (acc * 31 + ord(ch)) % 1000003
    return acc % buckets


def number_type(normalized_number: Optional[str]) -> str:
    if not normalized_number:
        return 'unknown'
    if is_short_code(normalized_number):
        return 'short'
    if is_tollfree_ru(normalized_number):
        return 'tollfree'
    if is_mobile_ru(normalized_number):
        return 'mobile'
    if is_landline_ru(normalized_number):
        return 'landline'
    if normalized_number.startswith('+7'):
        return 'ru_other'
    return 'foreign'


_PREFIX_RISK_TABLE_CACHE: Optional[Dict] = None
_PREFIX_RISK_TABLE_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'app', 'src', 'main', 'assets', 'prefix_risk.json',
)


def _load_prefix_risk_table(path: Optional[str] = None) -> Optional[Dict]:
    """Загрузить data-driven таблицу P(BLOCK|prefix). Кэшируется."""
    global _PREFIX_RISK_TABLE_CACHE
    if _PREFIX_RISK_TABLE_CACHE is not None and path is None:
        return _PREFIX_RISK_TABLE_CACHE
    target = path or _PREFIX_RISK_TABLE_PATH_DEFAULT
    if not os.path.isfile(target):
        return None
    try:
        with open(target, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if path is None:
            _PREFIX_RISK_TABLE_CACHE = data
        return data
    except Exception:
        return None


def reset_prefix_risk_cache() -> None:
    global _PREFIX_RISK_TABLE_CACHE
    _PREFIX_RISK_TABLE_CACHE = None


# ---------------------------------------------------------------------------
# Phase 3: data-driven лукапы для operator_bucket / def_code_risk / prefix_histogram.
# Все файлы генерирует build_assets_from_dataset.py и шипятся в app/src/main/assets/.
# Android читает их в FeatureExtractor.kt, чтобы train и runtime видели одни и те же фичи.
# ---------------------------------------------------------------------------

_OPERATOR_BUCKET_TABLE_CACHE: Optional[Dict] = None
_OPERATOR_BUCKET_TABLE_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'app', 'src', 'main', 'assets', 'operator_bucket.json',
)

_DEF_CODE_RISK_TABLE_CACHE: Optional[Dict] = None
_DEF_CODE_RISK_TABLE_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'app', 'src', 'main', 'assets', 'def_code_risk.json',
)

_PREFIX_HISTOGRAM_TABLE_CACHE: Optional[Dict] = None
_PREFIX_HISTOGRAM_TABLE_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'app', 'src', 'main', 'assets', 'prefix_histogram.json',
)

# Phase 4B: multi-resolution prefix histograms + def_code×operator cross-feature.
_PREFIX_HISTOGRAM3_TABLE_CACHE: Optional[Dict] = None
_PREFIX_HISTOGRAM3_TABLE_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'app', 'src', 'main', 'assets', 'prefix_histogram_3.json',
)

_PREFIX_HISTOGRAM7_TABLE_CACHE: Optional[Dict] = None
_PREFIX_HISTOGRAM7_TABLE_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'app', 'src', 'main', 'assets', 'prefix_histogram_7.json',
)

_DEF_CODE_OPERATOR_RISK_TABLE_CACHE: Optional[Dict] = None
_DEF_CODE_OPERATOR_RISK_TABLE_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'app', 'src', 'main', 'assets', 'def_code_operator_risk.json',
)

OPERATOR_BUCKETS = ('mts', 'megafon', 'beeline', 'tele2', 'mvno', 'other', 'unknown')


def _load_json_cached(path: Optional[str], default_path: str, cache_attr_name: str) -> Optional[Dict]:
    """Общий загрузчик JSON-файлов с кэшем в глобальной переменной модуля."""
    global_dict = globals()
    if path is None and global_dict.get(cache_attr_name) is not None:
        return global_dict[cache_attr_name]
    target = path or default_path
    if not os.path.isfile(target):
        return None
    try:
        with open(target, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if path is None:
            global_dict[cache_attr_name] = data
        return data
    except Exception:
        return None


def _load_operator_bucket_table(path: Optional[str] = None) -> Optional[Dict]:
    return _load_json_cached(path, _OPERATOR_BUCKET_TABLE_PATH_DEFAULT, '_OPERATOR_BUCKET_TABLE_CACHE')


def _load_def_code_risk_table(path: Optional[str] = None) -> Optional[Dict]:
    return _load_json_cached(path, _DEF_CODE_RISK_TABLE_PATH_DEFAULT, '_DEF_CODE_RISK_TABLE_CACHE')


def _load_prefix_histogram_table(path: Optional[str] = None) -> Optional[Dict]:
    return _load_json_cached(path, _PREFIX_HISTOGRAM_TABLE_PATH_DEFAULT, '_PREFIX_HISTOGRAM_TABLE_CACHE')


def _load_prefix_histogram3_table(path: Optional[str] = None) -> Optional[Dict]:
    return _load_json_cached(path, _PREFIX_HISTOGRAM3_TABLE_PATH_DEFAULT, '_PREFIX_HISTOGRAM3_TABLE_CACHE')


def _load_prefix_histogram7_table(path: Optional[str] = None) -> Optional[Dict]:
    return _load_json_cached(path, _PREFIX_HISTOGRAM7_TABLE_PATH_DEFAULT, '_PREFIX_HISTOGRAM7_TABLE_CACHE')


def _load_def_code_operator_risk_table(path: Optional[str] = None) -> Optional[Dict]:
    return _load_json_cached(
        path,
        _DEF_CODE_OPERATOR_RISK_TABLE_PATH_DEFAULT,
        '_DEF_CODE_OPERATOR_RISK_TABLE_CACHE',
    )


def reset_phase3_caches() -> None:
    global _OPERATOR_BUCKET_TABLE_CACHE, _DEF_CODE_RISK_TABLE_CACHE, _PREFIX_HISTOGRAM_TABLE_CACHE
    global _PREFIX_HISTOGRAM3_TABLE_CACHE, _PREFIX_HISTOGRAM7_TABLE_CACHE
    global _DEF_CODE_OPERATOR_RISK_TABLE_CACHE
    _OPERATOR_BUCKET_TABLE_CACHE = None
    _DEF_CODE_RISK_TABLE_CACHE = None
    _PREFIX_HISTOGRAM_TABLE_CACHE = None
    _PREFIX_HISTOGRAM3_TABLE_CACHE = None
    _PREFIX_HISTOGRAM7_TABLE_CACHE = None
    _DEF_CODE_OPERATOR_RISK_TABLE_CACHE = None


def _def_code_from_number(normalized_number: Optional[str]) -> str:
    """Первые 3 цифры после +7 / 8. Для +7800/+7495 это соответственно '800', '495' и т.д."""
    if not normalized_number:
        return ''
    digits = digits_only(normalized_number)
    if len(digits) >= 4 and digits[0] in ('7', '8'):
        return digits[1:4]
    if len(digits) >= 3:
        return digits[:3]
    return ''


def resolve_operator_bucket(
    normalized_number: Optional[str],
    metadata: Dict,
) -> str:
    """Определить бакет оператора.

    Приоритет:
      1) metadata['operator_bucket'] (уже посчитано в dataset_builder).
      2) metadata['operator'] или metadata['оператор'] (сырая строка) → operator_bucket().
      3) Шиплемый def_code → bucket лукап (operator_bucket.json) — cold-start путь.
      4) 'unknown'.
    """
    bucket = (metadata.get('operator_bucket') or '').strip().lower()
    if bucket in OPERATOR_BUCKETS:
        return bucket
    operator_str = metadata.get('operator') or metadata.get('оператор') or ''
    if operator_str:
        b = operator_bucket(operator_str)
        if b in OPERATOR_BUCKETS:
            return b
    table = _load_operator_bucket_table()
    if table is not None:
        def_code = _def_code_from_number(normalized_number)
        mapping = table.get('def_codes', {})
        if def_code and def_code in mapping:
            b = str(mapping[def_code]).lower()
            if b in OPERATOR_BUCKETS:
                return b
    return 'unknown'


def operator_bucket_one_hot(bucket: str) -> Dict[str, float]:
    bucket = (bucket or '').lower()
    return {
        'operatorMts': 1.0 if bucket == 'mts' else 0.0,
        'operatorMegafon': 1.0 if bucket == 'megafon' else 0.0,
        'operatorBeeline': 1.0 if bucket == 'beeline' else 0.0,
        'operatorTele2': 1.0 if bucket == 'tele2' else 0.0,
        'operatorMvno': 1.0 if bucket == 'mvno' else 0.0,
    }


def infer_def_code_risk(
    normalized_number: Optional[str],
    metadata: Dict,
) -> float:
    """P(BLOCK | def_code) из data-driven лукапа. Cold-start дружелюбный: опирается только на номер."""
    if not normalized_number:
        return 0.0
    if metadata.get('inAllowlist'):
        return 0.0
    table = _load_def_code_risk_table()
    if table is None:
        return 0.0
    def_code = _def_code_from_number(normalized_number)
    if not def_code:
        return float(table.get('fallback_risk', 0.1))
    risks = table.get('def_codes', {})
    if def_code in risks:
        return clamp01(float(risks[def_code]))
    return float(table.get('fallback_risk', 0.1))


def infer_prefix_histogram(
    normalized_number: Optional[str],
    metadata: Dict,
) -> Dict[str, float]:
    """3-D embedding префикса: doli BLOCK / WARN в трейне + log(seen_count)/log(seen_max).

    Для cold-start это поведение видов BLOCK по одному и тому же 6-значному префиксу — богатый
    сигнал эквивалентный явному prefix-embedding (без in-graph embedding-слоя, mobile-friendly).
    """
    out = {'prefixBlockShare': 0.0, 'prefixWarnShare': 0.0, 'prefixSeenLog': 0.0}
    if not normalized_number:
        return out
    table = _load_prefix_histogram_table()
    if table is None:
        return out
    plen = int(table.get('prefix_length', 6))
    prefix = normalized_number[:plen]
    prefixes = table.get('prefixes', {})
    entry = prefixes.get(prefix)
    if entry is None:
        # Бэкофф на 5/4-значный префикс (сворачивание для свежих номеров).
        for k in (5, 4):
            short = normalized_number[:k]
            if short in prefixes:
                entry = prefixes[short]
                break
    if not entry:
        return out
    out['prefixBlockShare'] = clamp01(float(entry.get('block_share', 0.0)))
    out['prefixWarnShare'] = clamp01(float(entry.get('warn_share', 0.0)))
    seen = float(entry.get('seen_count', 0.0))
    seen_max = float(table.get('seen_log_norm', math.log1p(50.0)))
    if seen_max <= 0:
        seen_max = math.log1p(50.0)
    out['prefixSeenLog'] = clamp01(math.log1p(max(seen, 0.0)) / seen_max)
    return out


# ---------------------------------------------------------------------------
# Phase 4B: multi-resolution prefix histograms + def_code×operator cross-feature.
# ---------------------------------------------------------------------------

def _lookup_prefix_block_share_at(
    normalized_number: Optional[str],
    table: Optional[Dict],
    fallback: float = 0.0,
) -> float:
    """Достаём block_share из histogram-таблицы по нужному prefix_length."""
    if not normalized_number or not table:
        return fallback
    plen = int(table.get('prefix_length', 6))
    prefix = normalized_number[:plen]
    prefixes = table.get('prefixes', {})
    entry = prefixes.get(prefix)
    if entry is None:
        return fallback
    return clamp01(float(entry.get('block_share', fallback)))


def infer_prefix_block_share_3(
    normalized_number: Optional[str],
    metadata: Dict,
) -> float:
    """3-digit (т.е. 5-char `+7XXX` = def_code) prefix block share. Грубее, но
    надёжнее чем 6-char — почти всегда есть значение для любого RU номера.
    """
    if not normalized_number:
        return 0.0
    if metadata.get('inAllowlist'):
        return 0.0
    table = _load_prefix_histogram3_table()
    return _lookup_prefix_block_share_at(normalized_number, table)


def infer_prefix_block_share_7(
    normalized_number: Optional[str],
    metadata: Dict,
) -> float:
    """7-digit (т.е. 9-char `+7XXXXXXX`) prefix block share. Точнее на знакомых
    номерах, но шумнее на новых. При отсутствии записи откатываемся на ноль —
    не подменяем сигналом более крупного префикса (его уже даёт prefixBlockShare/3).
    """
    if not normalized_number:
        return 0.0
    if metadata.get('inAllowlist'):
        return 0.0
    table = _load_prefix_histogram7_table()
    return _lookup_prefix_block_share_at(normalized_number, table)


def _shannon_entropy_3way(p_block: float, p_warn: float) -> float:
    """Shannon-энтропия нормализована к [0..1] на 3 классах (max=log2(3))."""
    p_block = max(0.0, min(1.0, p_block))
    p_warn = max(0.0, min(1.0, p_warn))
    p_allow = max(0.0, 1.0 - p_block - p_warn)
    total = p_block + p_warn + p_allow
    if total <= 0:
        return 0.0
    parts = (p_block / total, p_warn / total, p_allow / total)
    h = 0.0
    for p in parts:
        if p > 0.0:
            h -= p * math.log2(p)
    max_h = math.log2(3)
    return clamp01(h / max_h) if max_h > 0 else 0.0


def infer_prefix_entropy(
    normalized_number: Optional[str],
    metadata: Dict,
) -> float:
    """Энтропия label-распределения на 6-char prefix (тот же лукап, что у `infer_prefix_histogram`).

    Высокая энтропия → префикс смешанный (часть BLOCK, часть ALLOW) — сигнал к
    осторожности (часто WARN). Низкая → префикс «чистый» в одну сторону.

    Сначала пытается прочитать precomputed `entropy` из entry (build_assets
    проставляет его в Phase 4B). Если поля нет — считает через Shannon
    из block_share/warn_share (back-compat для старых assets без поля).
    """
    if not normalized_number:
        return 0.0
    table = _load_prefix_histogram_table()
    if table is None:
        return 0.0
    plen = int(table.get('prefix_length', 6))
    prefix = normalized_number[:plen]
    prefixes = table.get('prefixes', {})
    entry = prefixes.get(prefix)
    if entry is None:
        for k in (5, 4):
            short = normalized_number[:k]
            if short in prefixes:
                entry = prefixes[short]
                break
    if not entry:
        return 0.0
    if 'entropy' in entry:
        return clamp01(float(entry['entropy']))
    return _shannon_entropy_3way(
        float(entry.get('block_share', 0.0)),
        float(entry.get('warn_share', 0.0)),
    )


def infer_def_code_operator_risk(
    normalized_number: Optional[str],
    metadata: Dict,
) -> float:
    """P(BLOCK | operator_bucket × def_code) — cross-фича оператора и DEF-кода.

    Даёт более тонкий сигнал чем defCodeRisk: один и тот же DEF-код может
    иметь разную долю BLOCK в зависимости от того, какой именно оператор
    обслуживает диапазон (e.g. MVNO часто чаще даёт спам чем основной MNO).
    """
    if not normalized_number:
        return 0.0
    if metadata.get('inAllowlist'):
        return 0.0
    table = _load_def_code_operator_risk_table()
    if table is None:
        return 0.0
    bucket = resolve_operator_bucket(normalized_number, metadata)
    def_code = _def_code_from_number(normalized_number)
    if not bucket or bucket == 'unknown' or not def_code:
        return float(table.get('fallback_risk', 0.0))
    risks = table.get('buckets', {}).get(bucket, {})
    if def_code in risks:
        return clamp01(float(risks[def_code]))
    return float(table.get('fallback_risk', 0.0))


def infer_prefix_sample_size(
    normalized_number: Optional[str],
    metadata: Dict,
) -> float:
    """Confidence в block/warn-share по 6-char prefix: насколько много раз он
    встречался в трейне. Линейная нормализация (0 → нет данных, 1 → ≥30 примеров).

    Дополняет `prefixSeenLog`: lognorm даёт нелинейную «насыщенность», sampleSize
    линейную «уверенность» — модель видит оба сигнала и взвешивает сама.
    """
    if not normalized_number:
        return 0.0
    table = _load_prefix_histogram_table()
    if table is None:
        return 0.0
    plen = int(table.get('prefix_length', 6))
    prefix = normalized_number[:plen]
    prefixes = table.get('prefixes', {})
    entry = prefixes.get(prefix)
    if entry is None:
        for k in (5, 4):
            short = normalized_number[:k]
            if short in prefixes:
                entry = prefixes[short]
                break
    if not entry:
        return 0.0
    saturation = float(table.get('sample_size_saturation', 30.0))
    if saturation <= 0:
        saturation = 30.0
    seen = float(entry.get('seen_count', 0.0))
    return clamp01(seen / saturation)


def compute_no_metadata_flag(metadata: Dict) -> float:
    """1 если у номера ни отзывов, ни search-volume, ни списков, ни категории — cold-start индикатор."""
    has_reviews = safe_int(metadata.get('review_count')) > 0
    has_search = safe_int(metadata.get('search_volume')) > 0
    has_views = safe_int(metadata.get('view_count')) > 0
    has_in_lists = bool(metadata.get('inBlacklist') or metadata.get('inAllowlist'))
    flags = category_flags(metadata.get('categories', ''))
    has_category = bool(flags.get('has_fraud_category') or flags.get('has_telemarketing_category') or flags.get('has_finance_category'))
    return 0.0 if (has_reviews or has_search or has_views or has_in_lists or has_category) else 1.0


def infer_prefix_risk(normalized_number: Optional[str], metadata: Dict) -> float:
    """Оценка риска по префиксу номера.

    Сначала пытается использовать data-driven таблицу `prefix_risk.json`
    (генерируется `scripts/build_assets_from_dataset.py` из обучающих данных).
    Тот же файл грузит и Android — единый источник правды train/runtime.

    Если файл недоступен — fallback на старую эвристику по DEF-коду.

    Важно: не использует metadata['label'] — это было бы leakage в обучении.
    """
    if not normalized_number:
        return 0.0
    if metadata.get('inAllowlist'):
        return 0.0

    table = _load_prefix_risk_table()
    if table is not None:
        plen = int(table.get('prefix_length', 6))
        prefix = normalized_number[:plen]
        prefixes = table.get('prefixes', {})
        if prefix in prefixes:
            return float(prefixes[prefix])
        # Fallback: shorter prefixes (best-effort) and known curated overrides.
        for k in (5, 4):
            short = normalized_number[:k]
            if short in prefixes:
                return float(prefixes[short])
        if normalized_number.startswith('+84'):
            return 0.8
        return float(table.get('fallback_risk', 0.1))

    # Legacy fallback (no prefix_risk.json available, e.g. unit tests).
    digits = digits_only(normalized_number)
    if normalized_number.startswith('+7800'):
        return 0.25
    if normalized_number.startswith('+7495') or normalized_number.startswith('+7499'):
        return 0.35
    if len(digits) >= 4 and digits.startswith('79'):
        def_code = safe_int(digits[1:4], 0)
        if def_code in {900, 901, 902, 903, 904, 905, 906, 908, 909, 950, 951, 952, 953, 958, 966, 969}:
            return 0.65
        return 0.3
    if normalized_number.startswith('+84'):
        return 0.8
    return 0.1


def compute_reputation_score(negative_ratio: float, review_count: int, search_volume: int, flags: Dict[str, float]) -> float:
    review_component = clamp01(math.log1p(review_count) / math.log1p(100))
    search_component = clamp01(math.log1p(search_volume) / math.log1p(10000))
    fraud_boost = 0.25 if flags.get('has_fraud_category', 0.0) else 0.0
    telemarketing_boost = 0.12 if flags.get('has_telemarketing_category', 0.0) else 0.0
    return clamp01(negative_ratio * 0.55 + review_component * 0.15 + search_component * 0.15 + fraud_boost + telemarketing_boost)


def compact_feature_vector(
    normalized_number: Optional[str],
    label: str,
    metadata: Dict,
    raw_number: Optional[str] = None,
) -> Dict[str, float]:
    digits = digits_only(normalized_number or raw_number)
    is_ru = bool(normalized_number and is_russian_number(normalized_number))
    is_foreign = bool(normalized_number and normalized_number.startswith('+') and not normalized_number.startswith('+7'))
    n_type = number_type(normalized_number)
    flags = category_flags(metadata.get('categories', ''))

    negative = safe_int(metadata.get('negative_count'))
    positive = safe_int(metadata.get('positive_count'))
    neutral = safe_int(metadata.get('neutral_count'))
    review_count = safe_int(metadata.get('review_count'), negative + positive + neutral)
    if review_count <= 0:
        review_count = negative + positive + neutral
    search_volume = safe_int(metadata.get('search_volume'))
    total_votes = max(negative + positive + neutral, review_count, 1)
    negative_ratio = negative / total_votes

    source_confidence = clamp01(safe_float(metadata.get('source_confidence'), safe_float(metadata.get('confidence'), 0.5)))
    reputation = compute_reputation_score(negative_ratio, review_count, search_volume, flags)

    operator = metadata.get('operator', '')
    is_valid_range = bool(metadata.get('is_valid_ru_range'))
    if metadata.get('numbering_match') is not None:
        is_valid_range = bool(metadata.get('numbering_match'))

    # label больше не используется внутри compact_feature_vector (раньше — для inBlacklist/inAllowlist/prefixRisk,
    # это было leakage). Параметр сохраняем для обратной совместимости вызывающих.
    _ = label

    values = {
        'isContact': 1.0 if metadata.get('isContact') else 0.0,
        'isRussianNumber': 1.0 if is_ru else 0.0,
        'isForeignNumber': 1.0 if is_foreign else 0.0,
        'isShortCode': 1.0 if n_type == 'short' else 0.0,
        'isStandardLen': 1.0 if len(digits) == 11 and digits.startswith(('7', '8')) else 0.0,
        'isTollFree8800': 1.0 if n_type == 'tollfree' else 0.0,
        'isGeographical': 1.0 if n_type == 'landline' else 0.0,
        'isMobileRu': 1.0 if n_type == 'mobile' else 0.0,
        'isValidRuRange': 1.0 if is_valid_range else 0.0,
        'spoofingPrefixFlag': 1.0 if spoofing_prefix_flag(raw_number, normalized_number) else 0.0,
        'digitEntropy': clamp01(shannon_entropy(digits)),
        'repeatDigitRatio': clamp01(repeat_digit_ratio(digits)),
        'maxSameDigitRun': clamp01(max_same_digit_run(digits)),
        'beautifulNumberFlag': 1.0 if beautiful_number_flag(digits) else 0.0,
        # Не передаём label в metadata: infer_prefix_risk не должен видеть целевую метку.
        'prefixRisk': clamp01(infer_prefix_risk(normalized_number, metadata)),
        'callFrequency': clamp01(safe_float(metadata.get('callFrequency'), 0.0)),
        'isNightTime': 1.0 if metadata.get('isNightTime') else 0.0,
        'recentBankApp': 1.0 if metadata.get('recentBankApp') else 0.0,
        'recentGovApp': 1.0 if metadata.get('recentGovApp') else 0.0,
        'recentMarketplaceApp': 1.0 if metadata.get('recentMarketplaceApp') else 0.0,
        'recentMessengerApp': 1.0 if metadata.get('recentMessengerApp') else 0.0,
        'previouslyRejected': 1.0 if metadata.get('previouslyRejected') else 0.0,
        # inBlacklist/inAllowlist отражают реальное состояние в metadata (статические бандлы/пользовательские списки).
        # НЕ связаны с label — иначе это leakage: модель выучивал бы свой же ответ из фичи
        # и проваливалась бы в проде на неизвестных номерах.
        'inBlacklist': 1.0 if metadata.get('inBlacklist') else 0.0,
        'inAllowlist': 1.0 if metadata.get('inAllowlist') else 0.0,
        'hiddenNumber': 1.0 if metadata.get('hiddenNumber') else 0.0,
        'callerVerifyFailed': 1.0 if metadata.get('callerVerifyFailed') else 0.0,
        'userVulnerability': clamp01(safe_float(metadata.get('userVulnerability'), 0.35)),
        'userBusinessActivity': clamp01(safe_float(metadata.get('userBusinessActivity'), 0.45)),
        'contactsAvailable': 1.0 if metadata.get('contactsAvailable', True) else 0.0,
        'usageAccessAvailable': 1.0 if metadata.get('usageAccessAvailable') else 0.0,
        'reputationScore': reputation,
        'sourceConfidence': source_confidence,
    }

    # --- Phase 3 (v3) extension: 15 новых фичей. ---
    bucket = resolve_operator_bucket(normalized_number, metadata)
    values.update(operator_bucket_one_hot(bucket))

    values['defCodeRisk'] = clamp01(infer_def_code_risk(normalized_number, metadata))

    hist = infer_prefix_histogram(normalized_number, metadata)
    values['prefixBlockShare'] = hist['prefixBlockShare']
    values['prefixWarnShare'] = hist['prefixWarnShare']
    values['prefixSeenLog'] = hist['prefixSeenLog']

    # Reputation-explicit фичи: разворачиваем то, что внутри reputationScore было
    # «свёрнуто» в один скаляр. Модель получает явный сигнал, обнуляемый при cold-start aug.
    values['reviewsLog'] = clamp01(math.log1p(review_count) / math.log1p(100))
    values['negativeRatio'] = clamp01(negative_ratio)
    values['searchVolumeLog'] = clamp01(math.log1p(search_volume) / math.log1p(10000))

    values['hasFraudCategory'] = float(flags.get('has_fraud_category', 0.0))
    values['hasTelemarketingCategory'] = float(flags.get('has_telemarketing_category', 0.0))

    values['noMetadata'] = compute_no_metadata_flag(metadata)

    # Phase 4B: multi-resolution prefix signals + def_code×operator cross.
    values['prefixBlockShare3'] = clamp01(infer_prefix_block_share_3(normalized_number, metadata))
    values['prefixBlockShare7'] = clamp01(infer_prefix_block_share_7(normalized_number, metadata))
    values['prefixEntropy'] = clamp01(infer_prefix_entropy(normalized_number, metadata))
    values['defCodeOperatorRisk'] = clamp01(infer_def_code_operator_risk(normalized_number, metadata))
    values['prefixSampleSize'] = clamp01(infer_prefix_sample_size(normalized_number, metadata))

    return values


def compact_row(normalized_number: Optional[str], label: str, metadata: Dict, raw_number: Optional[str] = None) -> Sequence[float]:
    values = compact_feature_vector(normalized_number, label, metadata, raw_number)
    return [values[name] for name in COMPACT_FEATURES]
