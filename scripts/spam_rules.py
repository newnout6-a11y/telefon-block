"""Post-model rule engine для предиктов спам-блокера.

Зачем оно нужно
----------------
TFLite-модель учится на 32 фичах с активной репутацией. На «холодных» номерах
(когда метадаты по номеру нет ни в одном источнике) модель опирается главным
образом на структурные признаки + `prefixRisk`. По наблюдениям, она почти не
отдаёт BLOCK/WARN, если кроме умеренного `prefixRisk` нет других сигналов
(см. session 6df1a48d, тест на 5 номерах).

Это правильное поведение для precision (BLOCK precision = 0.9975 в текущей
модели), но плохо для recall на холодном старте. Чтобы не понижать порог BLOCK
(и не ловить ложных тревог), мы добавляем мягкий слой WARN-эскалации:

    cold-start AND prefixRisk >= COLDSTART_PREFIX_RISK_WARN AND not in allowlist
    AND not is contact AND not is allowlisted-toll-free  ⇒  WARN

WARN на Android проявляется как беззвучный + heads-up уведомление, не блок.
Пользователь сам решит ответить или нет, и его обратная связь (FeedbackHandler)
сместит веса.

Эта же логика реализована в `SmartSpamDetector.kt` через RiskFactor
`prefix_risk_high` — Python-копия нужна чтобы `scripts/spam_predict.py` показывал
тот же вердикт, что увидит пользователь на устройстве.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Порог prefixRisk, выше которого холодный +7-номер считается «подозрительным
# по префиксу» — на основе fallback_risk (0.5712) + ~8 пунктов запаса.
# Изменение этого порога — точка калибровки precision/recall на холодном старте.
COLDSTART_PREFIX_RISK_WARN = 0.65


@dataclass(frozen=True)
class RuleHit:
    """Результат срабатывания одного правила."""

    rule_id: str
    verdict_override: Optional[str]  # ALLOW / WARN / BLOCK / None
    reason: str
    points: int  # для совместимости с rule-engine на Android (display)


def _is_cold_start(feature_source: Optional[str]) -> bool:
    """`spam_predict.py` помечает source='dataset', если фичи взяты из
    processed/ru_metadata_features.csv (т.е. номер уже размечен), и source='cold'
    в остальных случаях."""
    return feature_source == 'cold'


def _ru_unknown_mobile_or_landline(features: Dict[str, float]) -> bool:
    return (
        features.get('isRussianNumber', 0.0) >= 0.5
        and (features.get('isMobileRu', 0.0) >= 0.5 or features.get('isGeographical', 0.0) >= 0.5)
        and features.get('isContact', 0.0) < 0.5
        and features.get('inAllowlist', 0.0) < 0.5
        and features.get('isShortCode', 0.0) < 0.5
        and features.get('isTollFree8800', 0.0) < 0.5
    )


def evaluate_rules(
    features: Dict[str, float],
    feature_source: Optional[str],
) -> List[RuleHit]:
    """Применить все известные правила и вернуть список сработавших.

    Args:
        features: компакт-вектор 32 фичей (см. ru_metadata_features.COMPACT_FEATURES).
        feature_source: 'cold' если cold-start, 'dataset' если фичи из labeled CSV.
    """
    hits: List[RuleHit] = []

    if _is_cold_start(feature_source) and _ru_unknown_mobile_or_landline(features):
        prefix_risk = float(features.get('prefixRisk', 0.0))
        if prefix_risk >= COLDSTART_PREFIX_RISK_WARN:
            hits.append(
                RuleHit(
                    rule_id='prefix_risk_high',
                    verdict_override='WARN',
                    reason=(
                        f'Холодный старт + высокий риск префикса '
                        f'(prefixRisk={prefix_risk:.2f} ≥ {COLDSTART_PREFIX_RISK_WARN:.2f})'
                    ),
                    points=int(round(prefix_risk * 25)),
                )
            )

    return hits


_LEVEL = {'ALLOW': 0, 'WARN': 1, 'BLOCK': 2}


def _max_verdict(*verdicts: Optional[str]) -> str:
    """Возвращает «самый строгий» вердикт из списка (ALLOW < WARN < BLOCK)."""
    best = 'ALLOW'
    for v in verdicts:
        if v is not None and _LEVEL.get(v, 0) > _LEVEL[best]:
            best = v
    return best


def apply_rules(
    model_verdict: str,
    features: Dict[str, float],
    feature_source: Optional[str],
) -> Tuple[str, List[RuleHit]]:
    """Послойно поверх модельного вердикта применить правила.

    Правила могут только ПОВЫСИТЬ строгость (ALLOW→WARN→BLOCK), не понизить.
    Это сознательное решение: даже если правило ошибочно сработало, мы максимум
    уведомим пользователя, а не разблокируем заведомо плохой номер.

    Returns:
        (final_verdict, applied_hits)
    """
    hits = evaluate_rules(features, feature_source)
    if not hits:
        return model_verdict, []
    rule_verdicts = [h.verdict_override for h in hits]
    final = _max_verdict(model_verdict, *rule_verdicts)
    if final == model_verdict:
        # Правила сработали, но не повысили вердикт (редкий кейс).
        return model_verdict, []
    return final, hits


__all__ = [
    'COLDSTART_PREFIX_RISK_WARN',
    'RuleHit',
    'evaluate_rules',
    'apply_rules',
]
