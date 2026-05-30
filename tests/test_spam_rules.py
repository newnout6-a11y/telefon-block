"""Тесты для post-model rule engine (scripts/spam_rules.py).

Главная цель — зафиксировать поведение для cold-start prefix-risk эскалации:

* WARN добавляется поверх модельного ALLOW когда холодный +7-номер с
  prefixRisk >= 0.65 не в allowlist и не contact.
* Правило никогда не понижает вердикт (ALLOW из правила не перебивает
  модельный BLOCK).
* dataset-источник (известный размеченный номер) правило не трогает —
  там модель уже видит inBlacklist/inAllowlist напрямую.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from spam_rules import (  # noqa: E402
    COLDSTART_PREFIX_RISK_WARN,
    apply_rules,
    evaluate_rules,
)


def _base_features(**overrides):
    """Дефолтный «холодный +7-мобильный, не в контактах»."""
    base = {
        'isContact': 0.0,
        'isRussianNumber': 1.0,
        'isForeignNumber': 0.0,
        'isShortCode': 0.0,
        'isStandardLen': 1.0,
        'isTollFree8800': 0.0,
        'isGeographical': 0.0,
        'isMobileRu': 1.0,
        'isValidRuRange': 1.0,
        'spoofingPrefixFlag': 0.0,
        'digitEntropy': 0.8,
        'repeatDigitRatio': 0.1,
        'maxSameDigitRun': 0.18,
        'beautifulNumberFlag': 0.0,
        'prefixRisk': 0.5,
        'callFrequency': 0.0,
        'isNightTime': 0.0,
        'recentBankApp': 0.0,
        'recentGovApp': 0.0,
        'recentMarketplaceApp': 0.0,
        'recentMessengerApp': 0.0,
        'previouslyRejected': 0.0,
        'inBlacklist': 0.0,
        'inAllowlist': 0.0,
        'hiddenNumber': 0.0,
        'callerVerifyFailed': 0.0,
        'userVulnerability': 0.35,
        'userBusinessActivity': 0.45,
        'contactsAvailable': 1.0,
        'usageAccessAvailable': 0.0,
        'reputationScore': 0.0,
        'sourceConfidence': 0.5,
    }
    base.update(overrides)
    return base


class TestColdStartPrefixRisk:
    def test_high_prefix_risk_escalates_allow_to_warn(self):
        features = _base_features(prefixRisk=0.85)
        verdict, hits = apply_rules('ALLOW', features, feature_source='cold')
        assert verdict == 'WARN'
        assert len(hits) == 1
        assert hits[0].rule_id == 'prefix_risk_high'
        assert hits[0].verdict_override == 'WARN'

    def test_threshold_is_inclusive(self):
        features = _base_features(prefixRisk=COLDSTART_PREFIX_RISK_WARN)
        verdict, hits = apply_rules('ALLOW', features, feature_source='cold')
        assert verdict == 'WARN'
        assert len(hits) == 1

    def test_below_threshold_keeps_allow(self):
        features = _base_features(prefixRisk=COLDSTART_PREFIX_RISK_WARN - 0.01)
        verdict, hits = apply_rules('ALLOW', features, feature_source='cold')
        assert verdict == 'ALLOW'
        assert hits == []

    def test_landline_high_risk_also_escalates(self):
        features = _base_features(prefixRisk=0.85, isMobileRu=0.0, isGeographical=1.0)
        verdict, hits = apply_rules('ALLOW', features, feature_source='cold')
        assert verdict == 'WARN'
        assert hits[0].rule_id == 'prefix_risk_high'

    def test_short_code_does_not_escalate(self):
        features = _base_features(prefixRisk=0.99, isShortCode=1.0, isStandardLen=0.0,
                                  isMobileRu=0.0)
        verdict, hits = apply_rules('ALLOW', features, feature_source='cold')
        assert verdict == 'ALLOW'

    def test_tollfree_8800_does_not_escalate(self):
        features = _base_features(prefixRisk=0.85, isTollFree8800=1.0, isMobileRu=0.0)
        verdict, hits = apply_rules('ALLOW', features, feature_source='cold')
        assert verdict == 'ALLOW'

    def test_contact_skips_escalation(self):
        features = _base_features(prefixRisk=0.85, isContact=1.0)
        verdict, hits = apply_rules('ALLOW', features, feature_source='cold')
        assert verdict == 'ALLOW'

    def test_allowlist_skips_escalation(self):
        features = _base_features(prefixRisk=0.85, inAllowlist=1.0)
        verdict, hits = apply_rules('ALLOW', features, feature_source='cold')
        assert verdict == 'ALLOW'


class TestNonColdSource:
    def test_dataset_source_skipped(self):
        # Если фичи пришли из processed/ru_metadata_features.csv (размеченный
        # номер), правило не вмешивается — модель сама видит inBlacklist.
        features = _base_features(prefixRisk=0.85)
        verdict, hits = apply_rules('ALLOW', features, feature_source='dataset')
        assert verdict == 'ALLOW'
        assert hits == []

    def test_unknown_source_skipped(self):
        features = _base_features(prefixRisk=0.85)
        verdict, hits = apply_rules('ALLOW', features, feature_source=None)
        assert verdict == 'ALLOW'


class TestNeverDowngrade:
    def test_block_stays_block_when_rule_says_warn(self):
        features = _base_features(prefixRisk=0.85)
        verdict, hits = apply_rules('BLOCK', features, feature_source='cold')
        assert verdict == 'BLOCK'
        # Hits возвращаются пустыми, потому что правило не подняло вердикт.
        assert hits == []

    def test_warn_stays_warn_with_high_prefix_risk(self):
        features = _base_features(prefixRisk=0.85)
        verdict, hits = apply_rules('WARN', features, feature_source='cold')
        assert verdict == 'WARN'
        assert hits == []  # уже WARN, правило не повысило


class TestForeignNumber:
    def test_foreign_high_prefix_risk_no_escalation(self):
        # Иностранные номера не подпадают под isRussianNumber-проверку правила.
        features = _base_features(prefixRisk=0.85, isRussianNumber=0.0,
                                  isForeignNumber=1.0, isMobileRu=0.0)
        verdict, hits = apply_rules('ALLOW', features, feature_source='cold')
        assert verdict == 'ALLOW'


class TestEvaluateRulesDirect:
    def test_evaluate_returns_hits_without_changing_verdict(self):
        features = _base_features(prefixRisk=0.85)
        hits = evaluate_rules(features, feature_source='cold')
        assert len(hits) == 1
        assert hits[0].rule_id == 'prefix_risk_high'
        assert hits[0].verdict_override == 'WARN'
        assert hits[0].points >= 16  # 0.85 * 25 ≈ 21
