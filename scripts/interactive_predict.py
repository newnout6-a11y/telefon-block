"""Interactive spam predictor with full telemetry input.

Asks for:
  - phone number
  - is_contact (есть в записной книжке)
  - call_frequency (сколько раз звонил)
  - call_time (время суток)
  - average_call_duration_sec (средняя длительность разговора)
  - previously_rejected (отклонял ранее)
  - hidden_number (скрытый номер)
  - recent app usage flags (банк / госуслуги / маркетплейс / мессенджер)
  - whitelist / blacklist hint

Then runs binary model on these features.

Run:
  C:\\Users\\Redmi\\AppData\\Local\\Programs\\Python\\Python312\\python.exe scripts\\interactive_predict.py
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ru_metadata_features import (  # noqa: E402
    COMPACT_FEATURES, FIELD_TO_RU,
    compact_row, number_type, operator_bucket, stable_bucket,
)
from ru_number_normalizer import get_def_code, is_valid_ru_phone, normalize_ru_phone  # noqa: E402
from ru_numbering_plan import NumberingPlan, load_existing_csv as load_numbering_csv  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "datasets" / "ru" / "raw"
ASSETS_DIR = REPO_ROOT / "app" / "src" / "main" / "assets" / "experimental"

BINARY_TFLITE = ASSETS_DIR / "spam_model_binary.tflite"
BINARY_CARD = ASSETS_DIR / "model_card_binary.json"


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


def ask_number() -> str:
    while True:
        raw = input("\n📞 Номер (например +7 952 480 36 89): ").strip()
        if not raw:
            sys.exit(0)
        norm = normalize_ru_phone(raw, reject_non_ru=True)
        if not norm or not is_valid_ru_phone(norm):
            print(f"  ❌ '{raw}' — не валидный российский номер. Попробуй ещё раз.")
            continue
        return norm


def ask_int(prompt: str, default: int, min_v: int = 0, max_v: int = 10000) -> int:
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            v = int(raw)
            if v < min_v or v > max_v:
                print(f"    нужно число в диапазоне {min_v}..{max_v}")
                continue
            return v
        except ValueError:
            print("    нужно целое число")


def ask_yn(prompt: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"  {prompt} [{suffix}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes", "д", "да", "1", "true"}:
            return True
        if raw in {"n", "no", "н", "нет", "0", "false"}:
            return False
        print("    введи y/n")


def ask_float(prompt: str, default: float, min_v: float = 0.0, max_v: float = 1e6) -> float:
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip().replace(",", ".")
        if not raw:
            return default
        try:
            v = float(raw)
            if v < min_v or v > max_v:
                print(f"    нужно число в диапазоне {min_v}..{max_v}")
                continue
            return v
        except ValueError:
            print("    нужно число")


def collect_telemetry(plan: NumberingPlan) -> Dict:
    """Ask the user for all the runtime telemetry the model expects."""
    number = ask_number()

    match = plan.lookup(number) if plan else None
    operator = match.get("operator", "") if match else ""
    region = match.get("region", "") if match else ""
    n_type = match.get("number_type") if match else number_type(number)
    op_bucket = operator_bucket(operator)

    print(f"  Оператор: {operator or '(unknown)'}")
    print(f"  Регион:   {region or '(unknown)'}")
    print(f"  Тип:      {n_type}")
    print()
    print("⚙️  Телеметрия (Enter = значение по умолчанию):")

    is_contact = ask_yn("Есть в записной книжке?", default=False)
    call_count = ask_int("Сколько раз этот номер звонил тебе за последние 30 дней?", default=0, max_v=10000)
    avg_duration = ask_float("Средняя длительность разговоров (сек, 0 = никогда не отвечал)", default=0.0, max_v=10000)
    is_night = ask_yn("Сейчас ночь (22:00–07:00)?", default=False)
    previously_rejected = ask_yn("Раньше ты сбрасывал/блокировал звонки с этого номера?", default=False)
    hidden = ask_yn("Скрытый номер (caller ID не передан)?", default=False)
    in_blacklist = ask_yn("Номер в публичном blacklist (мошеловка / справ-портал)?", default=False)
    in_whitelist = ask_yn("Номер в whitelist (банк/госуслуги/официальная организация)?", default=False)
    recent_bank = ask_yn("Ты недавно (≤1ч) открывал банковское приложение?", default=False)
    recent_gov = ask_yn("Ты недавно открывал госуслуги/налоги?", default=False)
    recent_marketplace = ask_yn("Ты недавно открывал маркетплейс (озон/wb/ya маркет)?", default=False)
    recent_messenger = ask_yn("Ты недавно открывал мессенджер?", default=False)

    # Build the same compact_row that build_feature_record produces.
    # Reputation fields stay zeroed (we don't have a reputation API runtime)
    # except inAllowlist / inBlacklist which are user-provided.
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
    meta = {
        **numbering,
        "source_confidence": 0.5,
        "source_reliability": 0.5,
        "inAllowlist": in_whitelist,
        "inBlacklist": in_blacklist,
        "contactsAvailable": True,
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
    feats = compact_row(number, "ALLOW", meta)

    # Override the runtime-provided features with what the user just said.
    def _set(feat_name: str, value):
        if feat_name in COMPACT_FEATURES:
            feats[COMPACT_FEATURES.index(feat_name)] = float(value)

    _set("isContact", 1.0 if is_contact else 0.0)
    _set("callFrequency", float(call_count))
    _set("isNightTime", 1.0 if is_night else 0.0)
    _set("previouslyRejected", 1.0 if previously_rejected else 0.0)
    _set("hiddenNumber", 1.0 if hidden else 0.0)
    _set("inBlacklist", 1.0 if in_blacklist else 0.0)
    _set("inAllowlist", 1.0 if in_whitelist else 0.0)
    _set("recentBankApp", 1.0 if recent_bank else 0.0)
    _set("recentGovApp", 1.0 if recent_gov else 0.0)
    _set("recentMarketplaceApp", 1.0 if recent_marketplace else 0.0)
    _set("recentMessengerApp", 1.0 if recent_messenger else 0.0)
    # Average duration is a derived signal: very short avg duration (<10s) on a
    # number that called many times is a classic spam pattern.
    # We don't have a dedicated feature for it in the trained model, but we can
    # influence callerVerifyFailed if avg duration is essentially zero AND
    # call_count > 1.
    if call_count > 1 and 0 < avg_duration < 5:
        _set("callerVerifyFailed", 1.0)

    # noMetadata: mark cold only if user didn't give any signal beyond defaults.
    has_signal = any([
        is_contact, call_count > 0, previously_rejected,
        in_blacklist, in_whitelist, hidden,
    ])
    _set("noMetadata", 0.0 if has_signal else 1.0)

    return {
        "number": number,
        "operator": operator,
        "region": region,
        "feats": feats,
        "summary": {
            "is_contact": is_contact,
            "call_count": call_count,
            "avg_duration": avg_duration,
            "is_night": is_night,
            "previously_rejected": previously_rejected,
            "hidden": hidden,
            "in_blacklist": in_blacklist,
            "in_whitelist": in_whitelist,
            "recent_bank": recent_bank,
            "recent_gov": recent_gov,
            "recent_marketplace": recent_marketplace,
            "recent_messenger": recent_messenger,
        },
    }


def predict(interp, X: np.ndarray) -> float:
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    interp.set_tensor(in_d["index"], X.reshape(1, -1).astype(np.float32))
    interp.invoke()
    return float(interp.get_tensor(out_d["index"]).reshape(-1)[0])


def main() -> int:
    print("=" * 65)
    print("  ИНТЕРАКТИВНЫЙ ПРОВЕРЯТОР ДЛЯ БИНАРНОЙ МОДЕЛИ (cold-warm)")
    print("=" * 65)
    print("Введи номер, дай телеметрию — модель скажет ALLOW/BLOCK.")
    print("Пустой номер для выхода.")

    print("\nЗагрузка numbering plan...")
    plan_records = load_numbering_csv(str(RAW_DIR / "ru_numbering_plan.csv"))
    plan = NumberingPlan(plan_records) if plan_records else None
    print(f"  {len(plan_records)} диапазонов")

    print("Загрузка binary модели...")
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=str(BINARY_TFLITE))
    interp.allocate_tensors()
    card = json.loads(BINARY_CARD.read_text(encoding="utf-8"))
    threshold = float(card.get("thresholds", {}).get("block_threshold", 0.01))
    print(f"  threshold = {threshold:.3f}\n")

    while True:
        try:
            data = collect_telemetry(plan)
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return 0

        feats = data["feats"]
        X = np.array(feats, dtype=np.float32)
        spam_prob = predict(interp, X)
        verdict = "BLOCK" if spam_prob >= threshold else "ALLOW"

        print()
        print("─" * 65)
        print(f"  📞 {data['number']}  ({data['operator']}, {data['region']})")
        print(f"  P(spam) = {spam_prob:.4f}    threshold = {threshold:.3f}")
        print(f"  >>> ВЕРДИКТ: {verdict} <<<")
        print("─" * 65)

        # Top-3 most "loaded" features for explainability.
        nonzero = [(COMPACT_FEATURES[i], v) for i, v in enumerate(feats) if abs(v) > 1e-6]
        nonzero.sort(key=lambda kv: -abs(kv[1]))
        print("Активные фичи (топ-10):")
        for name, val in nonzero[:10]:
            print(f"   {name:30s} = {val:.3f}")
        print()


if __name__ == "__main__":
    sys.exit(main())
