"""Phase 4D: balance the ALLOW ⇄ noMetadata=1 shortcut at the dataset level.

Background: the Phase 4C dataset has the degenerate correlation
``P(noMetadata=1 | ALLOW) ≈ 0.99`` and ``P(noMetadata=1 | BLOCK ∪ WARN) ≈ 0``.
The TFLite student internalises this as the dominant rule
``noMetadata=1 → ALLOW`` and the cold-start eval slice (where the 9
metadata features are zeroed) collapses to >0.99 ALLOW for prefix-spam
numbers. Phase 4A/4B/4C mitigated the symptom at training time via
per-batch feature masking; Phase 4D breaks the shortcut at the data
source so the model can no longer learn it.

The functions below operate on a list of "entries" — pre-feature-record
dicts with the layout produced by
``ru_metadata_dataset_builder.build_entry()``. Each entry carries the
underlying ``reputation`` metadata dict plus everything needed to
recompute the compact feature vector (``numbering``, ``label``,
``in_allowlist`` / ``in_blacklist`` flags, etc.). Each function takes
``(entries, ..., rng)`` and returns a new list of entries; the caller
is responsible for re-running ``build_feature_record`` on the mutated
entries afterwards so the compact feature vector is consistent with
the new metadata.

All randomness flows through the seeded ``rng`` parameter — the same
seed produces the same balanced output bit-for-bit.
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Sequence


# Cold-start mask features (must match
# ``train_kd_distillation.COLD_START_MASK_FEATURES``). Strategy C zeros
# exactly these in the shadow rows and forces ``noMetadata=1`` so the
# val/test split can contain BLOCK/WARN rows that look exactly like
# what the model sees on a real cold-start phone call.
COLD_START_MASK_FEATURES: Sequence[str] = (
    'inAllowlist', 'inBlacklist',
    'reputationScore', 'sourceConfidence',
    'reviewsLog', 'negativeRatio', 'searchVolumeLog',
    'hasFraudCategory', 'hasTelemarketingCategory',
)


def _has_no_metadata(entry: Dict) -> bool:
    """Mirror of ``ru_metadata_features.compute_no_metadata_flag`` but on
    the pre-feature-record entry dict. Returns True if the entry would
    produce ``noMetadata=1`` after ``build_feature_record``.
    """
    reputation = entry.get('reputation') or {}
    if int(reputation.get('review_count', 0) or 0) > 0:
        return False
    if int(reputation.get('search_volume', 0) or 0) > 0:
        return False
    if int(reputation.get('view_count', 0) or 0) > 0:
        return False
    if entry.get('in_static_allowlist') or entry.get('in_public_blacklist'):
        return False
    categories = str(reputation.get('categories', '') or '')
    if any(token in categories.lower() for token in (
        'мошен', 'фрод', 'scam', 'fraud', 'спам', 'реклам', 'телемаркет',
        'банк', 'кредит', 'займ', 'мфо', 'карта', 'финанс', 'страхов',
    )):
        return False
    return True


def inject_synthetic_metadata_into_allow(
    entries: List[Dict], target_fraction: float, rng: random.Random,
) -> List[Dict]:
    """Strategy A: enrich ``target_fraction`` of ALLOW entries that
    currently look cold (no reviews / no search / no list / no category)
    with plausible *positive* online metadata so ``noMetadata=0``.

    The injected fields are randomised per row to avoid creating a
    fixed-template synthetic cluster the model could memorise:

    - ``review_count``      ~ uniform 1..10
    - ``positive_count``    ~ ``review_count * uniform(0.7, 1.0)`` (>=1)
    - ``negative_count``    ~ ``review_count * uniform(0.0, 0.1)``
    - ``neutral_count``     filled to match ``review_count``
    - ``search_volume``     ~ uniform 1..50
    - ``source_confidence`` ~ uniform 0.7..0.95

    Categories are left empty for the majority of entries; a small
    fraction (~20 %) get the ``верифицированный_бизнес`` tag so the
    model also sees ALLOW rows with a non-spam category present.

    The function returns a *new* list (the input is not mutated); each
    modified entry is a deep copy with a fresh ``reputation`` dict so
    callers re-running ``build_feature_record`` will pick up the change.
    Entries that are not selected (or are not eligible) are passed
    through unchanged.
    """
    if target_fraction <= 0:
        return list(entries)

    eligible_idx: List[int] = []
    for i, entry in enumerate(entries):
        if entry.get('label') != 'ALLOW':
            continue
        if not _has_no_metadata(entry):
            continue
        eligible_idx.append(i)

    n_inject = int(round(len(eligible_idx) * float(target_fraction)))
    n_inject = max(0, min(n_inject, len(eligible_idx)))
    selected = set(rng.sample(eligible_idx, n_inject)) if n_inject else set()

    out: List[Dict] = []
    for i, entry in enumerate(entries):
        if i not in selected:
            out.append(entry)
            continue

        new_entry = copy.deepcopy(entry)
        rep = dict(new_entry.get('reputation') or {})
        review_count = rng.randint(1, 10)
        positive_ratio = rng.uniform(0.70, 1.00)
        negative_ratio = rng.uniform(0.00, 0.10)
        positive = max(1, int(round(review_count * positive_ratio)))
        negative = int(round(review_count * negative_ratio))
        # cap so positive+negative <= review_count
        if positive + negative > review_count:
            negative = max(0, review_count - positive)
        neutral = max(0, review_count - positive - negative)

        rep['review_count'] = review_count
        rep['positive_count'] = positive
        rep['negative_count'] = negative
        rep['neutral_count'] = neutral
        rep['search_volume'] = rng.randint(1, 50)
        rep['source_confidence'] = round(rng.uniform(0.70, 0.95), 4)
        # ~20% of injected rows get a benign verified-business category;
        # the rest stay empty so the synthetic cluster has internal
        # variance rather than a single fixed signature.
        if rng.random() < 0.20:
            rep['categories'] = 'верифицированный_бизнес'
        else:
            rep.setdefault('categories', '')
        # Mark provenance so tests / downstream tooling can identify
        # synthetic rows; the field is ignored by the feature builder.
        rep['phase4d_synthetic'] = 1

        new_entry['reputation'] = rep
        new_entry['phase4d_synthetic'] = True
        out.append(new_entry)

    return out


def subsample_allow_no_metadata(
    entries: List[Dict], drop_fraction: float, rng: random.Random,
) -> List[Dict]:
    """Strategy B: drop ``drop_fraction`` of ALLOW entries where
    ``noMetadata=1`` AND ``inAllowlist=0`` AND ``isContact=0``.

    The protection ensures we never drop ALLOW labels that the device
    can prove independently (static whitelist hit / address-book hit)
    — those rules are the runtime safety net we want the model to keep
    learning. Everything else (numbering-plan personal-mobile rows,
    legitimate-collector rows without a static-whitelist tag, etc.)
    is fair game for subsampling.
    """
    if drop_fraction <= 0:
        return list(entries)

    droppable_idx: List[int] = []
    for i, entry in enumerate(entries):
        if entry.get('label') != 'ALLOW':
            continue
        if not _has_no_metadata(entry):
            continue
        if entry.get('in_static_allowlist'):
            continue
        if entry.get('is_contact'):
            continue
        droppable_idx.append(i)

    n_drop = int(round(len(droppable_idx) * float(drop_fraction)))
    n_drop = max(0, min(n_drop, len(droppable_idx)))
    drop_set = set(rng.sample(droppable_idx, n_drop)) if n_drop else set()

    return [entry for i, entry in enumerate(entries) if i not in drop_set]


def add_shadow_cold_block_warn(
    entries: List[Dict], shadow_fraction: float, rng: random.Random,
) -> List[Dict]:
    """Strategy C (optional): for ``shadow_fraction`` of BLOCK and WARN
    entries, append a duplicate "shadow" entry that emulates a
    cold-start observation of the same number — the 9 cold-mask
    metadata features are zeroed (via blanked reputation) and
    ``noMetadata=1`` is forced.

    The shadow entries reuse the original numbering / prefix /
    operator information (those are runtime-cheap signals that *are*
    available cold), only the online-reputation half is masked. This
    parallels Phase 4C feature-masking augmentation but moves it into
    the saved CSV so val/test reflect cold-start reality during
    early-stopping, threshold tuning and final reporting.
    """
    if shadow_fraction <= 0:
        return list(entries)

    eligible_idx = [
        i for i, e in enumerate(entries)
        if e.get('label') in ('BLOCK', 'WARN')
    ]
    n_shadow = int(round(len(eligible_idx) * float(shadow_fraction)))
    n_shadow = max(0, min(n_shadow, len(eligible_idx)))
    selected = set(rng.sample(eligible_idx, n_shadow)) if n_shadow else set()

    out = list(entries)
    for i in sorted(selected):
        original = entries[i]
        shadow = copy.deepcopy(original)
        # Blank the reputation half: this drives reviewsLog / negativeRatio
        # / searchVolumeLog / hasFraudCategory / hasTelemarketingCategory
        # / reputationScore / sourceConfidence to zero in
        # compact_feature_vector.
        shadow['reputation'] = {
            'source': str((original.get('reputation') or {}).get('source', '')) + '|phase4d_shadow',
            'source_confidence': 0.0,
            'source_reliability': 0.0,
        }
        # And the static-list half is hidden so inAllowlist=inBlacklist=0.
        shadow['in_static_allowlist'] = False
        shadow['in_public_blacklist'] = False
        shadow['phase4d_shadow'] = True
        out.append(shadow)

    return out
