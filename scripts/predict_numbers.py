"""Predict spam probability for arbitrary phone numbers using the trained
spam_model_binary.tflite + spam_model_leak_free.tflite.

Cold-start mode (offline-on-device simulation):
  - 9 metadata features zeroed out
  - noMetadata=1
  - other features derived from the phone number itself (numbering plan,
    operator/region buckets, prefix histograms, def-code risk).

Run:
  C:\\Users\\Redmi\\AppData\\Local\\Programs\\Python\\Python312\\python.exe scripts\\predict_numbers.py "+7 952 480 36 89" "+7 911 616 767 5"
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ru_metadata_features import (  # noqa: E402
    COMPACT_FEATURES, FIELD_TO_RU,
    category_flags, compact_feature_vector, compact_row,
    compute_reputation_score,
    infer_prefix_risk, number_type, operator_bucket, parse_date,
    review_velocity, safe_float, safe_int, stable_bucket,
)
from ru_number_normalizer import get_def_code, is_valid_ru_phone, normalize_ru_phone  # noqa: E402
from ru_numbering_plan import NumberingPlan, load_existing_csv as load_numbering_csv  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "datasets" / "ru" / "raw"
ASSETS_DIR = REPO_ROOT / "app" / "src" / "main" / "assets" / "experimental"

BINARY_TFLITE = ASSETS_DIR / "spam_model_binary.tflite"
LEAKFREE_TFLITE = ASSETS_DIR / "spam_model_leak_free.tflite"
BINARY_CARD = ASSETS_DIR / "model_card_binary.json"
LEAKFREE_CARD = ASSETS_DIR / "model_card_leak_free.json"


def infer_timezone_offset(region: str) -> int:
    text = (region or "").lower()
    if any(k in text for k in ["камчат", "чукот"]): return 12
    if any(k in text for k in ["магадан", "сахалин"]): return 11
    if any(k in text for k in ["якут", "примор", "хабаров"]): return 10
    if any(k in text for k in ["иркут", "бурят"]): return 8
    if any(k in text for k in ["краснояр", "кемеров", "томск", "новосибир"]): return 7
    if "омск" in text: return 6
    if any(k in text for k in ["екатерин", "свердлов", "челябин", "перм", "тюмень"]): return 5
    if any(k in text for k in ["самар", "саратов", "удмурт"]): return 4
    if "калининград" in text: return 2
    return 3


def build_cold_features(number: str, plan: NumberingPlan) -> List[float]:
    """Build a 52-feature compact row in cold-start mode (no metadata)."""
    match = plan.lookup(number) if plan else None
    operator = match.get("operator", "") if match else ""
    region = match.get("region", "") if match else ""
    n_type = match.get("number_type") if match else number_type(number)
    op_bucket = operator_bucket(operator)
    numbering = {
        "numbering_match": match is not None,
        "is_valid_ru_range": match is not None,
        "operator": operator,
        "region": region,
        "number_type": n_type,
        "def_code": get_def_code(number) or "",
        "operator_bucket": op_bucket,
        "region_bucket": stable_bucket(region),
        "timezone_offset": infer_timezone_offset(region),
        "is_mvno": 1 if op_bucket == "mvno" else 0,
    }
    # Cold view: empty reputation, noMetadata=1, no allowlist/blacklist signal.
    meta = {
        **numbering,
        "source_confidence": 0.5,
        "source_reliability": 0.5,
        "inAllowlist": False,
        "inBlacklist": False,
        "contactsAvailable": True,  # we don't know but the trainer wrote True for everyone
        "negative_count": 0,
        "positive_count": 0,
        "neutral_count": 0,
        "review_count": 0,
        "search_volume": 0,
        "view_count": 0,
        "related_count": 0,
        "categories": "",
        "last_review_at": "",
        "first_seen_at": "",
        "detail_date": "",
    }
    # compact_row(number, label, meta) returns the 52-feature vector.
    # We pass label='ALLOW' as placeholder — it isn't used for inference, only
    # for label-derived features which are already zeroed in cold view.
    row = compact_row(number, "ALLOW", meta)
    # row is a list of 52 floats.
    # Ensure 9 leak-free features are zero and noMetadata=1.
    leak_features = [
        "inAllowlist", "inBlacklist", "reputationScore", "sourceConfidence",
        "reviewsLog", "negativeRatio", "searchVolumeLog",
        "hasFraudCategory", "hasTelemarketingCategory",
    ]
    for fname in leak_features:
        if fname in COMPACT_FEATURES:
            row[COMPACT_FEATURES.index(fname)] = 0.0
    if "noMetadata" in COMPACT_FEATURES:
        row[COMPACT_FEATURES.index("noMetadata")] = 1.0
    return row


def predict_one(interp, X: np.ndarray) -> np.ndarray:
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    interp.set_tensor(in_d["index"], X.reshape(1, -1).astype(np.float32))
    interp.invoke()
    return interp.get_tensor(out_d["index"]).reshape(-1)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/predict_numbers.py '+7...' '+7...'")
        return 2

    print("Loading numbering plan...", flush=True)
    plan_csv = RAW_DIR / "ru_numbering_plan.csv"
    plan_records = load_numbering_csv(str(plan_csv))
    plan = NumberingPlan(plan_records) if plan_records else None
    print(f"  {len(plan_records)} numbering ranges", flush=True)

    print("Loading binary model...", flush=True)
    from ai_edge_litert.interpreter import Interpreter
    binary_interp = Interpreter(model_path=str(BINARY_TFLITE))
    binary_interp.allocate_tensors()
    binary_card = json.loads(BINARY_CARD.read_text(encoding="utf-8"))
    binary_thresh = float(binary_card.get("thresholds", {}).get("block_threshold", 0.01))
    print(f"  binary threshold: {binary_thresh:.3f}", flush=True)

    print("Loading leak-free 3-class model...", flush=True)
    leakfree_interp = Interpreter(model_path=str(LEAKFREE_TFLITE))
    leakfree_interp.allocate_tensors()
    leakfree_card = json.loads(LEAKFREE_CARD.read_text(encoding="utf-8"))
    lf_block_thresh = float(leakfree_card.get("thresholds", {}).get("block_threshold", 0.5))
    lf_warn_thresh = float(leakfree_card.get("thresholds", {}).get("warn_threshold", 0.3))
    print(f"  leak-free block_threshold: {lf_block_thresh:.3f}, warn_threshold: {lf_warn_thresh:.3f}", flush=True)
    print()

    for raw in sys.argv[1:]:
        norm = normalize_ru_phone(raw, reject_non_ru=True)
        if not norm or not is_valid_ru_phone(norm):
            print(f"INPUT: {raw!r}  -> NOT A VALID RU NUMBER")
            print()
            continue

        match = plan.lookup(norm) if plan else None
        operator = match.get("operator", "") if match else "?"
        region = match.get("region", "") if match else "?"

        feats = build_cold_features(norm, plan)
        X = np.array(feats, dtype=np.float32)

        # Binary inference.
        bin_p = predict_one(binary_interp, X)
        spam_prob = float(bin_p[0])  # P(spam)
        bin_verdict = "BLOCK" if spam_prob >= binary_thresh else "ALLOW"

        # Leak-free 3-class inference.
        lf_p = predict_one(leakfree_interp, X)
        # 3-class softmax: [ALLOW, WARN, BLOCK]
        if len(lf_p) >= 3:
            p_allow, p_warn, p_block = float(lf_p[0]), float(lf_p[1]), float(lf_p[2])
        else:
            p_allow, p_warn, p_block = 0.0, 0.0, float(lf_p[0])
        if p_block >= lf_block_thresh:
            lf_verdict = "BLOCK"
        elif p_warn >= lf_warn_thresh or (p_allow < 0.5 and p_warn > p_allow):
            lf_verdict = "WARN"
        else:
            lf_verdict = "ALLOW"

        print(f"INPUT: {raw}")
        print(f"  normalized: {norm}")
        print(f"  operator:   {operator or '(unknown)'}")
        print(f"  region:     {region or '(unknown)'}")
        print(f"  -- BINARY model (production candidate) --")
        print(f"     P(spam) = {spam_prob:.4f}   threshold = {binary_thresh:.3f}")
        print(f"     verdict = {bin_verdict}")
        print(f"  -- LEAK-FREE 3-class model (failed FP-rate gate) --")
        print(f"     P(allow)={p_allow:.4f}  P(warn)={p_warn:.4f}  P(block)={p_block:.4f}")
        print(f"     verdict = {lf_verdict}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
