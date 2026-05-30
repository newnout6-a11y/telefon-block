"""Unit tests for the OSM bulk collector's _categorise() function.

The bulk collector (`scripts/ru_osm_bulk_collector.py`) streams the entire
russia-latest.osm.pbf and calls `_categorise()` for every phone-tagged
element. The `ALLOW 10x` PR loosens the fallback path to capture the ~50%
of phone-tagged elements that previously had no name, by adding two extra
fallback tiers and a NEGATIVE_VALUES filter for street furniture.

This test pins down the new behaviour so future edits can't silently
revert the loosening.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from ru_osm_bulk_collector import (
    _categorise, _is_negative, _phone_tags,
    ORG_HINT_TAGS, ADDR_HINT_TAGS, NEGATIVE_VALUES,
    PHONE_TAG_KEYS, TAG_LOOKUP,
)


class TestExactMatches:
    """Exact (key, value) hits in the CATEGORIES registry must take priority."""

    def test_amenity_hospital_returns_medical(self):
        result = _categorise({'amenity': 'hospital', 'name': 'Городская больница'})
        assert result is not None
        cat, feat, src, conf = result
        assert cat == 'medical'
        assert feat == 'medical'
        assert src == 'osm_medical_hospital'
        assert conf >= 0.90

    def test_amenity_school_returns_education(self):
        result = _categorise({'amenity': 'school', 'name': 'Школа №25'})
        assert result is not None
        cat, feat, src, conf = result
        assert cat == 'education'
        assert src == 'osm_education_school'

    def test_office_company_returns_office(self):
        result = _categorise({'office': 'company', 'name': 'ООО Ромашка'})
        assert result is not None
        cat, _, src, _ = result
        assert cat == 'office_other'

    def test_amenity_post_office_returns_gov(self):
        result = _categorise({'amenity': 'post_office', 'name': 'Почта России'})
        assert result is not None
        cat, _, src, conf = result
        assert cat == 'gov'
        assert conf >= 0.90


class TestWildcardMatches:
    """`healthcare=*` should match any value (the wildcard path)."""

    def test_healthcare_wildcard(self):
        result = _categorise({'healthcare': 'doctor', 'name': 'Кабинет ЛОР'})
        assert result is not None
        cat, _, src, _ = result
        # healthcare is mapped to medical via wildcard
        assert cat == 'medical'
        assert src.startswith('osm_medical_healthcare') or 'healthcare' in src

    def test_craft_wildcard(self):
        # craft=* added in this PR — any craft (carpenter, electrician, etc.)
        result = _categorise({'craft': 'electrician', 'name': 'Электрик Иван'})
        assert result is not None
        cat, _, src, _ = result
        assert cat == 'craft' or cat == 'other'


class TestFallbackTier1OrgHintWithName:
    """Tier-1 fallback: phone+org-hint+name → osm_other_business / 0.72."""

    def test_amenity_with_name(self):
        # An obscure amenity not in the registry, but with a name.
        result = _categorise({'amenity': 'rare_amenity', 'name': 'Уникальное место'})
        assert result is not None
        cat, feat, src, conf = result
        assert cat == 'other'
        assert feat == 'business'
        assert src == 'osm_other_business'
        assert 0.70 <= conf <= 0.75

    def test_shop_with_name(self):
        result = _categorise({'shop': 'esoteric_shop', 'name': 'Чай Кофе'})
        assert result is not None
        _, _, src, _ = result
        assert src == 'osm_other_business'


class TestFallbackTier2OrgHintNoName:
    """Tier-2 fallback (NEW in this PR): phone+org-hint without name → osm_org_hint / 0.66."""

    def test_amenity_without_name(self):
        # Previously dropped silently; now accepted as low-confidence org.
        result = _categorise({'amenity': 'cafe'})
        # amenity=cafe is in CATEGORIES registry, so this hits exact match,
        # not fallback. Use an unmapped one:
        result = _categorise({'amenity': 'unusual_thing'})
        assert result is not None
        _, _, src, conf = result
        assert src == 'osm_org_hint'
        assert 0.64 <= conf <= 0.68

    def test_shop_without_name(self):
        result = _categorise({'shop': 'something_unmapped'})
        assert result is not None
        _, _, src, _ = result
        assert src == 'osm_org_hint'

    def test_office_without_name(self):
        result = _categorise({'office': 'unknown_office_type'})
        assert result is not None
        _, _, src, _ = result
        assert src == 'osm_org_hint'

    def test_industrial_without_name(self):
        # 'industrial' is in ORG_HINT_TAGS — phone + industrial = real org
        result = _categorise({'industrial': 'factory'})
        assert result is not None
        _, _, src, _ = result
        assert src == 'osm_org_hint'


class TestFallbackTier3AddrPlusName:
    """Tier-3 fallback (NEW): phone+addr+name (no org-hint) → osm_addressed_business / 0.62."""

    def test_addr_with_name_no_org_hint(self):
        result = _categorise({
            'name': 'ООО Ромашка',
            'addr:street': 'Тверская',
            'addr:housenumber': '7',
        })
        assert result is not None
        _, _, src, conf = result
        assert src == 'osm_addressed_business'
        assert 0.60 <= conf <= 0.64

    def test_addr_without_name_returns_none(self):
        # Just an address with phone and no name and no org-hint isn't enough.
        result = _categorise({
            'addr:street': 'Тверская',
            'addr:housenumber': '7',
        })
        assert result is None


class TestNegativeFilter:
    """Street furniture (benches, waste baskets, etc.) must be rejected
    even if some prankster added a phone tag."""

    def test_bench_rejected(self):
        # Even with a name, amenity=bench is street furniture.
        result = _categorise({'amenity': 'bench', 'name': 'Лавочка'})
        assert result is None

    def test_waste_basket_rejected(self):
        result = _categorise({'amenity': 'waste_basket'})
        assert result is None

    def test_vending_machine_rejected(self):
        result = _categorise({'amenity': 'vending_machine', 'name': 'Автомат'})
        assert result is None

    def test_street_lamp_rejected(self):
        result = _categorise({'highway': 'street_lamp'})
        assert result is None

    def test_is_negative_helper_consistency(self):
        # All NEGATIVE_VALUES entries must trigger _is_negative.
        for (key, val) in NEGATIVE_VALUES:
            tags = {key: val if val != '*' else 'anything'}
            assert _is_negative(tags) is True, f'{key}={val} should be negative'


class TestNoSignalReturnsNone:
    """Tags with phones but neither org-hint nor addr+name should return None."""

    def test_only_phone_returns_none(self):
        # `phone` tag alone (without context) — _categorise just sees it via the
        # caller; here we simulate an element with phone but no org-hint.
        result = _categorise({'unrelated_key': 'value'})
        assert result is None

    def test_random_tags_returns_none(self):
        result = _categorise({'website': 'https://example.com', 'opening_hours': '24/7'})
        assert result is None


class TestExtendedCategories:
    """The PR adds craft, public_transport, shop_extended, amenity_extended.
    These should all hit exact-match path, not fallback."""

    def test_aeroway_terminal_hits_exact(self):
        result = _categorise({'aeroway': 'terminal', 'name': 'Шереметьево T1'})
        assert result is not None
        # Should hit exact match (not fallback)
        _, _, src, conf = result
        assert 'osm_' in src
        assert conf >= 0.85

    def test_public_transport_station_hits_exact(self):
        result = _categorise({'public_transport': 'station', 'name': 'ЖД вокзал'})
        assert result is not None
        _, _, src, conf = result
        assert conf >= 0.80

    def test_shop_clothes_hits_exact(self):
        result = _categorise({'shop': 'clothes', 'name': 'Бутик'})
        assert result is not None
        cat, feat, src, conf = result
        # shop=clothes added in 'shop_extended' category
        assert src == 'osm_shop_extended_clothes'

    def test_amenity_cinema_hits_exact(self):
        result = _categorise({'amenity': 'cinema', 'name': 'Каро'})
        assert result is not None
        _, feat, src, conf = result
        # amenity=cinema added in amenity_extended
        assert feat == 'tourism'


class TestRegressionNoYieldDrop:
    """A representative sample of element shapes — verify each yields a
    result. This is the regression net: if any future refactor returns
    None for these, ALLOW yield will drop."""

    SHAPES = [
        # (description, tags, expect_some_result)
        ('hospital with name',     {'amenity': 'hospital', 'name': 'Х'}, True),
        ('school no name',         {'amenity': 'school'}, True),
        ('office company',         {'office': 'company', 'name': 'Y'}, True),
        ('mystery shop with name', {'shop': 'mystery', 'name': 'Z'}, True),
        ('mystery shop no name',   {'shop': 'mystery'}, True),
        ('industrial only',        {'industrial': 'factory'}, True),
        ('addr+name no orgish',    {'name': 'A', 'addr:street': 'B', 'addr:housenumber': '1'}, True),
        ('craft electrician',      {'craft': 'electrician'}, True),
        ('aeroway terminal',       {'aeroway': 'terminal'}, True),
        ('bench (negative)',       {'amenity': 'bench', 'name': 'B'}, False),
        ('only phone (no context)',{'random_key': 'v'}, False),
    ]

    def test_all_shapes_yield_expected(self):
        for desc, tags, should_match in self.SHAPES:
            result = _categorise(tags)
            if should_match:
                assert result is not None, f'expected match for {desc} (tags={tags})'
            else:
                assert result is None, f'expected None for {desc} (tags={tags})'


class TestPhoneTagExtraction:
    """The expanded PHONE_TAG_KEYS catches more OSM tag conventions —
    `phone:reception`, `contact:landline`, etc. — without duplicating the
    same number when multiple keys carry the same digits."""

    def test_extracts_from_basic_phone(self):
        tags = {'phone': '+7 495 123-45-67'}
        out = _phone_tags(tags)
        assert any('495' in p for p in out)

    def test_extracts_from_contact_phone(self):
        tags = {'contact:phone': '+7-495-987-65-43'}
        out = _phone_tags(tags)
        assert len(out) >= 1

    def test_extracts_from_reception(self):
        tags = {'phone:reception': '+7 800 100-00-00'}
        out = _phone_tags(tags)
        assert len(out) >= 1

    def test_extracts_from_emergency(self):
        tags = {'phone:emergency': '+7 495 999-99-99'}
        out = _phone_tags(tags)
        assert len(out) >= 1

    def test_extracts_from_landline(self):
        tags = {'phone:landline': '+7 495 111-22-33'}
        out = _phone_tags(tags)
        assert len(out) >= 1

    def test_extracts_from_multiple_keys(self):
        tags = {
            'phone': '+7 495 111-11-11',
            'contact:phone': '+7 495 222-22-22',
            'phone:reception': '+7 495 333-33-33',
        }
        out = _phone_tags(tags)
        # All 3 distinct numbers should be extracted.
        assert len(out) >= 3

    def test_no_phone_returns_empty(self):
        tags = {'name': 'Тест', 'amenity': 'cafe'}
        out = _phone_tags(tags)
        assert out == []

    def test_phone_tag_keys_includes_basics(self):
        # Don't accidentally remove the four core keys.
        for key in ('phone', 'contact:phone', 'phone:mobile', 'contact:mobile'):
            assert key in PHONE_TAG_KEYS, f'{key} missing from PHONE_TAG_KEYS'


class TestSourceLabelStability:
    """Source labels are used in downstream dataset stats and shouldn't
    silently change format (e.g. from 'osm_medical_hospital' to 'osm:medical:hospital')."""

    def test_format_is_underscore_prefixed_osm(self):
        result = _categorise({'amenity': 'hospital'})
        assert result is not None
        _, _, src, _ = result
        assert src.startswith('osm_')
        assert ':' not in src
        assert ' ' not in src

    def test_fallback_sources_distinct(self):
        # The three fallback tiers each have distinct source labels, so
        # downstream stats can tell them apart.
        s1 = _categorise({'amenity': 'unmapped', 'name': 'A'})[2]
        s2 = _categorise({'amenity': 'unmapped'})[2]
        s3 = _categorise({'name': 'A', 'addr:street': 'B', 'addr:housenumber': '1'})[2]
        assert s1 == 'osm_other_business'
        assert s2 == 'osm_org_hint'
        assert s3 == 'osm_addressed_business'
        assert len({s1, s2, s3}) == 3
