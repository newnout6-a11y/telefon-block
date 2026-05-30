import os
import sys
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from ru_metadata_features import (
    COMPACT_FEATURES, compact_feature_vector, compact_row,
    shannon_entropy, spoofing_prefix_flag, repeat_digit_ratio,
    max_same_digit_run, beautiful_number_flag, compute_reputation_score,
    category_flags, number_type, infer_prefix_risk, clamp01,
    LABEL_TO_ID, ID_TO_LABEL,
)
from ru_number_normalizer import (
    normalize_ru_phone, is_russian_number, is_mobile_ru,
    is_landline_ru, is_tollfree_ru, is_short_code,
    is_placeholder_number, is_valid_ru_phone,
)


class TestFeatureCount:
    def test_feature_count_is_52(self):
        # Phase 3: 32 (v2) + 15 (operator + def_code + prefix histogram + reputation explicit + categories + noMetadata).
        # Phase 4B: +5 cold-survivable (prefixBlockShare3/7, prefixEntropy, defCodeOperatorRisk, prefixSampleSize) → 52.
        assert len(COMPACT_FEATURES) == 52

    def test_no_duplicate_features(self):
        assert len(COMPACT_FEATURES) == len(set(COMPACT_FEATURES))

    def test_v4_features_present(self):
        # Phase 4B: явная проверка 5 новых cold-survivable фич в схеме.
        for name in (
            'prefixBlockShare3', 'prefixBlockShare7', 'prefixEntropy',
            'defCodeOperatorRisk', 'prefixSampleSize',
        ):
            assert name in COMPACT_FEATURES, f'{name} missing from COMPACT_FEATURES'


class TestCompactFeatureVector:
    def test_has_all_features(self):
        # inBlacklist теперь отражает metadata, а не label (был leakage).
        features = compact_feature_vector(
            '+79001234567', 'BLOCK',
            {'negative_count': 10, 'positive_count': 0, 'review_count': 10,
             'categories': 'мошенничество', 'source_confidence': 0.9, 'is_valid_ru_range': True,
             'inBlacklist': True}
        )
        assert list(features.keys()) == COMPACT_FEATURES
        assert features['inBlacklist'] == 1.0
        assert features['reputationScore'] > 0.5

    def test_label_does_not_set_inblacklist(self):
        # Регресс-тест на фикс leakage: BLOCK label без metadata.inBlacklist => inBlacklist=0.
        features = compact_feature_vector(
            '+79001234567', 'BLOCK',
            {'negative_count': 10, 'review_count': 10, 'categories': 'мошенничество'}
        )
        assert features['inBlacklist'] == 0.0
        assert features['inAllowlist'] == 0.0

    def test_allow_label(self):
        features = compact_feature_vector(
            '+78005553535', 'ALLOW',
            {'negative_count': 0, 'positive_count': 5, 'review_count': 5,
             'categories': 'банк', 'source_confidence': 0.9, 'is_valid_ru_range': True,
             'inAllowlist': True}
        )
        assert features['inAllowlist'] == 1.0
        assert features['inBlacklist'] == 0.0

    def test_compact_row_length(self):
        row = compact_row('+79161234567', 'WARN', {'negative_count': 3, 'review_count': 5})
        assert len(row) == 52

    def test_all_values_in_range(self):
        features = compact_feature_vector(
            '+79161234567', 'WARN',
            {'negative_count': 3, 'review_count': 5, 'source_confidence': 0.5}
        )
        for name, val in features.items():
            assert isinstance(val, float), f'{name} is not float'
            assert 0.0 <= val <= 1.0, f'{name}={val} out of [0,1]'


class TestEntropy:
    def test_low_entropy(self):
        assert shannon_entropy('77777777777') < 0.1

    def test_high_entropy(self):
        assert shannon_entropy('79001234567') > 0.3

    def test_empty(self):
        assert shannon_entropy('') == 0.0

    def test_single_digit(self):
        assert shannon_entropy('1') == 0.0


class TestDigitPatterns:
    def test_repeat_digit_ratio(self):
        assert repeat_digit_ratio('1111') >= 0.9
        assert repeat_digit_ratio('1234') == 0.0

    def test_max_same_digit_run(self):
        assert max_same_digit_run('11122333') > 0.3
        assert max_same_digit_run('12345678') < 0.2

    def test_beautiful_number(self):
        assert beautiful_number_flag('1111111') is True
        assert beautiful_number_flag('1234567') is True


class TestSpoofing:
    def test_spoofing_prefix_flag(self):
        assert spoofing_prefix_flag('+84 95 123-45-67', '+84951234567') is True
        assert spoofing_prefix_flag('+7 495 123-45-67', '+74951234567') is False


class TestNormalizer:
    def test_basic(self):
        assert normalize_ru_phone('8 (800) 555-35-35') == '+78005553535'

    def test_plus7(self):
        assert normalize_ru_phone('+7 916 123-45-67') == '+79161234567'

    def test_international(self):
        assert normalize_ru_phone('+7 (495) 123-45-67') == '+74951234567'

    def test_invalid(self):
        assert normalize_ru_phone('not a number') is None

    def test_is_russian(self):
        assert is_russian_number('+79161234567') is True
        assert is_russian_number('+1234567890') is False

    def test_is_mobile(self):
        assert is_mobile_ru('+79161234567') is True
        assert is_mobile_ru('+74951234567') is False

    def test_is_landline(self):
        assert is_landline_ru('+74951234567') is True
        assert is_landline_ru('+79161234567') is False

    def test_is_tollfree(self):
        assert is_tollfree_ru('+78005553535') is True
        assert is_tollfree_ru('+79161234567') is False

    def test_is_short_code(self):
        assert is_short_code('+7112') is True
        assert is_short_code('+79161234567') is False

    def test_is_russian_rejects_invalid_def(self):
        # def-codes 0XX/1XX/2XX не существуют в РФ — должны отбрасываться.
        assert is_russian_number('+70327490161') is False
        assert is_russian_number('+71678334567') is False
        assert is_russian_number('+72251234567') is False
        # def-code 6XX/7XX относятся к Казахстану, не к России (общий +7).
        assert is_russian_number('+77001234149') is False
        assert is_russian_number('+76001234567') is False

    def test_is_placeholder_number(self):
        assert is_placeholder_number('+79000000000') is True
        assert is_placeholder_number('+78000000000') is True
        assert is_placeholder_number('+74950000000') is True
        # Реальный номер с нулями в середине — не placeholder.
        assert is_placeholder_number('+74951234567') is False
        assert is_placeholder_number('+79161234567') is False
        # Защита от None / коротких.
        assert is_placeholder_number(None) is False
        assert is_placeholder_number('+7') is False

    def test_is_valid_ru_phone_composite(self):
        # Валидные РФ номера проходят.
        assert is_valid_ru_phone('+79161234567') is True
        assert is_valid_ru_phone('+74951234567') is True
        assert is_valid_ru_phone('+78005553535') is True
        # Невалидные def-коды отбрасываются.
        assert is_valid_ru_phone('+70327490161') is False
        # Казахстан отбрасывается.
        assert is_valid_ru_phone('+77001234149') is False
        # Placeholder отбрасывается даже при валидном def-коде.
        assert is_valid_ru_phone('+74950000000') is False
        assert is_valid_ru_phone('+78000000000') is False


class TestNumberType:
    def test_mobile(self):
        assert number_type('+79161234567') == 'mobile'

    def test_tollfree(self):
        assert number_type('+78005553535') == 'tollfree'

    def test_landline(self):
        assert number_type('+74951234567') == 'landline'

    def test_foreign(self):
        assert number_type('+1234567890') == 'foreign'

    def test_unknown(self):
        assert number_type(None) == 'unknown'


class TestPrefixRisk:
    def test_tollfree_risk(self):
        assert infer_prefix_risk('+78005553535', {}) > 0.0

    def test_allowlist_zero(self):
        assert infer_prefix_risk('+79161234567', {'inAllowlist': True}) == 0.0

    def test_high_risk_def(self):
        risk = infer_prefix_risk('+79001234567', {})
        assert risk > 0.5


class TestReputationScore:
    def test_high_negative(self):
        flags = {'has_fraud_category': 1.0, 'has_telemarketing_category': 0.0}
        score = compute_reputation_score(0.9, 50, 1000, flags)
        assert score > 0.7

    def test_low_negative(self):
        flags = {'has_fraud_category': 0.0, 'has_telemarketing_category': 0.0}
        score = compute_reputation_score(0.1, 5, 10, flags)
        assert score < 0.3


class TestCategoryFlags:
    def test_fraud(self):
        flags = category_flags('мошенничество телефонное')
        assert flags['has_fraud_category'] == 1.0

    def test_telemarketing(self):
        flags = category_flags('спам реклама')
        assert flags['has_telemarketing_category'] == 1.0

    def test_finance(self):
        flags = category_flags('банк кредит')
        assert flags['has_finance_category'] == 1.0

    def test_clean(self):
        flags = category_flags('доставка еды')
        assert flags['has_fraud_category'] == 0.0


class TestClamp01:
    def test_normal(self):
        assert clamp01(0.5) == 0.5

    def test_over(self):
        assert clamp01(1.5) == 1.0

    def test_under(self):
        assert clamp01(-0.5) == 0.0

    def test_nan(self):
        assert clamp01(float('nan')) == 0.0

    def test_inf(self):
        assert clamp01(float('inf')) == 0.0


class TestSchemaSync:
    def test_feature_names_match_kotlin(self):
        kotlin_path = os.path.join(
            ROOT, 'app', 'src', 'main', 'java', 'com', 'antispam', 'blocker',
            'domain', 'tracking', 'DecisionTracker.kt'
        )
        if not os.path.exists(kotlin_path):
            return
        import re
        with open(kotlin_path, 'r', encoding='utf-8') as f:
            text = f.read()
        marker = 'val FEATURE_NAMES: List<String> = listOf('
        start = text.find(marker)
        if start < 0:
            return
        start += len(marker)
        end = text.find(')', start)
        block = text[start:end]
        kotlin_names = re.findall(r'"([A-Za-z0-9_]+)"', block)
        assert kotlin_names == COMPACT_FEATURES, f'Kotlin={kotlin_names} != Python={COMPACT_FEATURES}'

    def test_model_card_features_match(self):
        card_path = os.path.join(ROOT, 'app', 'src', 'main', 'assets', 'model_card.json')
        if not os.path.exists(card_path):
            return
        with open(card_path, 'r', encoding='utf-8') as f:
            card = json.load(f)
        assert card.get('features') == COMPACT_FEATURES
        assert card.get('feature_count') == len(COMPACT_FEATURES)


class TestColdThresholds:
    """Phase 4A: shipped model_card должен содержать cold_thresholds; Android
    выбирает их при cold-start (noMetadata=1, нет list-подсказок). Если
    `cold_thresholds` отсутствует — считаем это legacy-картой и не падаем.
    """

    def _load_card(self):
        card_path = os.path.join(ROOT, 'app', 'src', 'main', 'assets', 'model_card.json')
        if not os.path.exists(card_path):
            return None
        with open(card_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_cold_thresholds_block_warn_present(self):
        card = self._load_card()
        if card is None or 'cold_thresholds' not in card:
            return  # legacy card — back-compat OK
        ct = card['cold_thresholds']
        assert 'block_threshold' in ct, ct
        assert 'warn_threshold' in ct, ct
        assert 0.0 < ct['block_threshold'] < 1.0
        assert 0.0 < ct['warn_threshold'] < 1.0

    def test_cold_thresholds_more_conservative_than_warm(self):
        # Cold-start: model уверен реже -> threshold должен быть >= warm
        # (иначе ловим больше шума). Защита от случайной заливки cold < warm.
        card = self._load_card()
        if card is None or 'cold_thresholds' not in card:
            return
        warm = card.get('thresholds')
        cold = card.get('cold_thresholds')
        if not warm or not cold:
            return
        assert cold['block_threshold'] >= warm['block_threshold'], (
            f"cold block ({cold['block_threshold']}) < warm block ({warm['block_threshold']}) — "
            f"подозрительно: cold должен быть строже"
        )

    def test_cold_thresholds_tuning_info_lists_mask_features(self):
        card = self._load_card()
        if card is None or 'cold_thresholds' not in card:
            return
        info = card['cold_thresholds'].get('tuning_info', {})
        if not info:
            return
        mask = info.get('mask_features', [])
        # Эти фичи должны быть offline-unavailable: на устройстве без интернета
        # они всегда нули, поэтому zerout-им их и при cold-tuning порогов.
        for f in ('inAllowlist', 'inBlacklist', 'reputationScore'):
            assert f in mask, f'{f} expected in cold mask, got {mask}'


class TestDecisionTracking:
    def test_tracking_stats_fields(self):
        assert ID_TO_LABEL == {0: 'ALLOW', 1: 'WARN', 2: 'BLOCK'}

    def test_label_to_id_roundtrip(self):
        for label, lid in LABEL_TO_ID.items():
            assert ID_TO_LABEL[lid] == label


class TestColdAugBalanced:
    """Phase 4C: balanced cold-aug — selective spam-only masking, oversampling
    WARN до уровня BLOCK, weight bump на masked рядах.
    """

    def _import_train(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'train_kd_distillation',
            os.path.join(ROOT, 'scripts', 'train_kd_distillation.py'),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_balanced_aug_targets_only_spam_classes(self):
        m = self._import_train()
        np = m.np
        X = np.random.RandomState(0).rand(40, len(m.COMPACT_FEATURES)).astype(np.float32)
        y = np.array([0] * 20 + [1] * 5 + [2] * 15)  # 20 ALLOW, 5 WARN, 15 BLOCK
        # Default: warn_oversample_cap=1.0 — без oversampling, каждая spam-строка
        # маскируется один раз. Балансировка через class_weights.
        Xa, ya, w, info = m.concat_masked_aug_balanced(
            X, y, mask_indices=[0, 1, 2],
            target_classes=(m.LABEL_TO_ID['BLOCK'], m.LABEL_TO_ID['WARN']),
            weight_multiplier=1.5, seed=42,
        )
        # Размер masked subset = 5 WARN + 15 BLOCK = 20 (no oversampling).
        assert info['masked_added'] == 20
        # Оригинальные ALLOW не должны попасть в masked subset.
        n_orig = len(X)
        masked_y = ya[n_orig:]
        assert m.LABEL_TO_ID['ALLOW'] not in set(masked_y.tolist())
        # Должны быть BLOCK (15) и WARN (5).
        assert (masked_y == m.LABEL_TO_ID['BLOCK']).sum() == 15
        assert (masked_y == m.LABEL_TO_ID['WARN']).sum() == 5
        # Weights: original=1.0, masked=1.5.
        assert (w[:n_orig] == 1.0).all()
        assert (w[n_orig:] == 1.5).all()

    def test_balanced_aug_with_oversample_cap(self):
        """warn_oversample_cap=4.0 → WARN дублируется до 4×original (capped)."""
        m = self._import_train()
        np = m.np
        X = np.random.RandomState(0).rand(40, len(m.COMPACT_FEATURES)).astype(np.float32)
        y = np.array([0] * 20 + [1] * 5 + [2] * 15)
        Xa, ya, w, info = m.concat_masked_aug_balanced(
            X, y, mask_indices=[0, 1, 2],
            target_classes=(m.LABEL_TO_ID['BLOCK'], m.LABEL_TO_ID['WARN']),
            weight_multiplier=1.5, seed=42,
            warn_oversample_cap=4.0,
        )
        n_orig = len(X)
        masked_y = ya[n_orig:]
        # WARN: min(15, 5*4)=15; BLOCK: 15 (cap не применяется т.к. это max).
        assert (masked_y == m.LABEL_TO_ID['WARN']).sum() == 15
        assert (masked_y == m.LABEL_TO_ID['BLOCK']).sum() == 15

    def test_balanced_aug_zeroes_mask_columns(self):
        m = self._import_train()
        np = m.np
        rng = np.random.RandomState(7)
        X = rng.rand(20, len(m.COMPACT_FEATURES)).astype(np.float32) + 0.5  # все > 0
        y = np.array([0] * 8 + [1] * 4 + [2] * 8)
        mask_idx = [0, 1, 2]
        Xa, ya, w, info = m.concat_masked_aug_balanced(
            X, y, mask_indices=mask_idx,
            target_classes=(m.LABEL_TO_ID['BLOCK'], m.LABEL_TO_ID['WARN']),
            weight_multiplier=1.5, seed=7,
        )
        n_orig = len(X)
        # Masked rows: указанные колонки = 0.0.
        for col in mask_idx:
            assert (Xa[n_orig:, col] == 0.0).all(), f'col {col} not zeroed in masked rows'
        # Оригинальные строки не тронуты.
        assert np.allclose(Xa[:n_orig], X)

    def test_legacy_aug_with_weight_multiplier(self):
        """Phase 4C: legacy concat_masked_aug поддерживает weight_multiplier на masked."""
        m = self._import_train()
        np = m.np
        X = np.random.RandomState(0).rand(20, len(m.COMPACT_FEATURES)).astype(np.float32)
        y = np.array([0] * 12 + [1] * 4 + [2] * 4)
        n_orig = len(X)
        # Без буста: все веса = 1.0.
        Xa, ya, w, info = m.concat_masked_aug(
            X, y, mask_indices=[0, 1], mask_prob=0.5, seed=42,
        )
        assert len(Xa) == len(ya) == len(w)
        assert (w[:n_orig] == 1.0).all()
        assert (w[n_orig:] == 1.0).all()
        # С бустом: оригинальные = 1.0, masked = 1.5.
        Xa2, ya2, w2, info2 = m.concat_masked_aug(
            X, y, mask_indices=[0, 1], mask_prob=0.5, seed=42,
            weight_multiplier=1.5,
        )
        assert (w2[:n_orig] == 1.0).all()
        assert (w2[n_orig:] == 1.5).all()

    def test_legacy_aug_full_doubling(self):
        """Phase 4C: mask_prob=1.0 → masked rows == orig rows."""
        m = self._import_train()
        np = m.np
        X = np.random.RandomState(0).rand(20, len(m.COMPACT_FEATURES)).astype(np.float32)
        y = np.array([0] * 12 + [1] * 4 + [2] * 4)
        Xa, ya, w, info = m.concat_masked_aug(
            X, y, mask_indices=[0, 1], mask_prob=1.0, seed=42,
            weight_multiplier=1.5,
        )
        assert len(Xa) == 2 * len(X)
        # Class proportions сохранены при удвоении.
        for c in (0, 1, 2):
            assert (ya == c).sum() == 2 * (y == c).sum()

    def test_legacy_aug_stratified_preserves_class_ratio(self):
        """Phase 4C: stratified mask_prob=0.5 — выборка пропорциональна классам."""
        m = self._import_train()
        np = m.np
        X = np.random.RandomState(0).rand(40, len(m.COMPACT_FEATURES)).astype(np.float32)
        # 20 ALLOW, 4 WARN, 16 BLOCK.
        y = np.array([0] * 20 + [1] * 4 + [2] * 16)
        Xa, ya, w, info = m.concat_masked_aug(
            X, y, mask_indices=[0, 1], mask_prob=0.5, seed=42,
            stratified=True,
        )
        n_orig = len(X)
        masked_y = ya[n_orig:]
        # Stratified: каждый класс масштабируется пропорционально.
        assert (masked_y == 0).sum() == 10  # 20 * 0.5
        assert (masked_y == 1).sum() == 2   # 4 * 0.5
        assert (masked_y == 2).sum() == 8   # 16 * 0.5


class TestPhase4DBalancer:
    """Phase 4D: dataset-level balancer that breaks the
    ALLOW ⇄ noMetadata=1 shortcut. The balancer must mutate ALLOW rows
    (Strategy A) or drop them (Strategy B) deterministically, must
    leave non-ALLOW rows untouched, and the optional Strategy C must
    only zero the cold-mask half of BLOCK/WARN duplicates while
    preserving every other field.
    """

    def _import_balancer(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'cold_start_balancer',
            os.path.join(ROOT, 'scripts', 'cold_start_balancer.py'),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _make_cold_allow_entries(self, count: int):
        return [
            {
                'number': f'+791234567{i:02d}',
                'label': 'ALLOW',
                'weight': 0.5,
                'source': 'numbering_plan_personal_mobile',
                'reputation': {},
                'numbering': {'operator': 'mts', 'def_code': '912'},
                'in_static_allowlist': False,
                'in_public_blacklist': False,
                'is_contact': False,
            }
            for i in range(count)
        ]

    def test_inject_synthetic_metadata_changes_no_metadata(self):
        import random as _random
        csb = self._import_balancer()
        rows = self._make_cold_allow_entries(100)
        # Sanity: every input row currently looks "noMetadata=1".
        assert sum(csb._has_no_metadata(r) for r in rows) == 100

        out = csb.inject_synthetic_metadata_into_allow(rows, 0.30, _random.Random(42))
        assert len(out) == 100  # injection never drops rows
        warm_after = sum(1 for r in out if not csb._has_no_metadata(r))
        # ~30 rows should have flipped to noMetadata=0 (allow ±1 for rounding).
        assert 28 <= warm_after <= 32

        # Each warm row must carry a real review_count and look "positive":
        # positive_ratio >= 0.7, negative_ratio <= 0.1.
        for r in out:
            if not r.get('phase4d_synthetic'):
                continue
            rep = r['reputation']
            rc = int(rep['review_count'])
            assert 1 <= rc <= 10
            assert int(rep['positive_count']) >= int(round(0.7 * rc - 1))
            assert int(rep['negative_count']) <= int(round(0.1 * rc + 1))
            assert 1 <= int(rep['search_volume']) <= 50
            assert 0.70 <= float(rep['source_confidence']) <= 0.95

    def test_subsample_allow_drops_only_unprotected(self):
        import random as _random
        csb = self._import_balancer()
        rows = self._make_cold_allow_entries(100)
        # First 20 rows are protected (inAllowlist=True).
        for r in rows[:20]:
            r['in_static_allowlist'] = True
        # Next 20 rows are protected (isContact=True).
        for r in rows[20:40]:
            r['is_contact'] = True

        out = csb.subsample_allow_no_metadata(rows, 0.50, _random.Random(7))
        # 50 % of the 60 unprotected rows = 30 rows dropped → 70 left.
        assert len(out) == 70
        # Every protected number must still be in the output.
        protected_numbers = {r['number'] for r in rows[:40]}
        out_numbers = {r['number'] for r in out}
        assert protected_numbers <= out_numbers

    def test_shadow_cold_rows_zero_only_mask_features(self):
        import importlib.util
        import random as _random
        csb = self._import_balancer()
        # Use the actual feature builder so we exercise the full
        # compact_feature_vector path on the shadow row.
        spec = importlib.util.spec_from_file_location(
            'ru_metadata_features',
            os.path.join(ROOT, 'scripts', 'ru_metadata_features.py'),
        )
        rmf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rmf)

        block_entry = {
            'number': '+79611607830',
            'label': 'BLOCK',
            'weight': 1.5,
            'source': 'reviews_neberitrubku',
            'reputation': {
                'negative_count': 12, 'positive_count': 0, 'neutral_count': 1,
                'review_count': 13, 'search_volume': 500,
                'categories': 'мошенничество',
                'source_confidence': 0.9,
                'source': 'reviews_neberitrubku',
            },
            'numbering': {'operator': 'mts', 'def_code': '961'},
            'in_static_allowlist': False,
            'in_public_blacklist': True,
            'is_contact': False,
        }

        out = csb.add_shadow_cold_block_warn([block_entry], 1.0, _random.Random(1))
        assert len(out) == 2
        original, shadow = out

        def vec(entry):
            meta = {**(entry['reputation'] or {}), **entry['numbering']}
            meta['inAllowlist'] = entry['in_static_allowlist']
            meta['inBlacklist'] = entry['in_public_blacklist']
            return rmf.compact_feature_vector(entry['number'], entry['label'], meta)

        v_orig = vec(original)
        v_shadow = vec(shadow)
        # Mask features must be zero in the shadow row.
        for f in csb.COLD_START_MASK_FEATURES:
            assert v_shadow[f] == 0.0, f'shadow {f} not zeroed (={v_shadow[f]})'
        # noMetadata must be 1.
        assert v_shadow['noMetadata'] == 1.0
        # Operator / prefix-side features must be identical to the original
        # (the shadow row only changes the metadata half).
        for f in (
            'isMobileRu', 'isStandardLen',
            'operatorMts', 'operatorMegafon', 'operatorBeeline',
            'operatorTele2', 'operatorMvno',
            'defCodeRisk', 'defCodeOperatorRisk',
            'prefixBlockShare', 'prefixBlockShare3', 'prefixBlockShare7',
            'prefixEntropy', 'prefixSampleSize',
        ):
            assert v_orig[f] == v_shadow[f], f'{f} drifted: orig={v_orig[f]} shadow={v_shadow[f]}'

    def test_balancer_deterministic_with_seed(self):
        import random as _random
        csb = self._import_balancer()

        def run(seed: int):
            rows = self._make_cold_allow_entries(200)
            for r in rows[:50]:
                r['in_static_allowlist'] = True
            rng = _random.Random(seed)
            rows = csb.inject_synthetic_metadata_into_allow(rows, 0.30, rng)
            rows = csb.subsample_allow_no_metadata(rows, 0.25, rng)
            return [(r['number'], int(bool(r.get('phase4d_synthetic'))),
                     int((r['reputation'] or {}).get('review_count', 0) or 0))
                    for r in rows]

        a = run(42)
        b = run(42)
        c = run(43)
        assert a == b, 'same seed should produce identical output'
        assert a != c, 'different seed should produce different output'
