"""Predict ALLOW/WARN/BLOCK для одного телефонного номера через TFLite-модель.

Usage:
    python scripts/spam_predict.py +79991234567
    python scripts/spam_predict.py +79991234567 +74957754747 +78005553535
    python scripts/spam_predict.py --features-csv my_features.csv +79991234567

    # Batch mode: read candidates from CSV, write verdicts to CSV.
    # Input CSV must have a ``normalized_number`` column (or ``номер``/``number``).
    # If it also has ``expected_label`` (e.g. cold-start ALLOW candidates from
    # vk_token_collector), output CSV will include a ``disagreement`` flag.
    python scripts/spam_predict.py --from-csv datasets/ru/eval/vk_candidates.csv \
                                   --out-csv  datasets/ru/eval/vk_verdicts.csv

Алгоритм:
    1. Нормализуем входной номер (+7XXXXXXXXXX).
    2. Если в processed/ru_metadata_features.csv (русские заголовки) находим строку с тем же
       номером — берём готовый компакт-вектор оттуда. Это «реалистичный» сценарий: номер уже
       видели в датасете, есть репутация, источник, in/black/whitelist.
    3. Иначе считаем фичи с нуля через compact_feature_vector + пустую metadata. Это
       «холодный старт» — что приложение знает о номере на момент звонка, если его нет
       ни в одном бандл-списке.
    4. Прогоняем через TFLite (FP32, [1, 32] -> [1, 3] softmax).
    5. Применяем thresholds из model_card.json (block_threshold, warn_threshold) если они есть,
       иначе argmax.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from ru_metadata_features import (
    COMPACT_FEATURES,
    FIELD_TO_RU,
    ID_TO_LABEL,
    compact_feature_vector,
)
from ru_number_normalizer import normalize_ru_phone
from spam_rules import apply_rules

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DEFAULT_MODEL = os.path.join(BASE_DIR, 'app', 'src', 'main', 'assets', 'spam_model.tflite')
DEFAULT_CARD = os.path.join(BASE_DIR, 'app', 'src', 'main', 'assets', 'model_card.json')
DEFAULT_FEATURES_CSV = os.path.join(BASE_DIR, 'datasets', 'ru', 'processed', 'ru_metadata_features.csv')


def load_thresholds(card_path: str, cold: bool = False) -> Optional[Tuple[float, float]]:
    """Load (block_threshold, warn_threshold) from model_card.json.

    When `cold=True`, prefer `cold_thresholds` (Phase 4A — calibrated on the
    cold view of val with a min-precision floor). Fall back to warm
    `thresholds` if no cold-specific entry exists. This matters because warm
    thresholds are tuned for full-metadata inputs (block_threshold ≈ 0.24),
    while cold inputs need a much higher floor (typically ≥ 0.58) to avoid
    blocking everything just because the prefix happens to be spammy.
    """
    if not os.path.isfile(card_path):
        return None
    try:
        with open(card_path, 'r', encoding='utf-8') as f:
            card = json.load(f)
    except Exception:
        return None

    if cold:
        cold_thr = card.get('cold_thresholds') or {}
        bt = cold_thr.get('block_threshold')
        wt = cold_thr.get('warn_threshold')
        if bt is not None and wt is not None:
            return float(bt), float(wt)
        # fall through to warm thresholds if cold-specific aren't present.

    thr = card.get('thresholds') or {}
    bt = thr.get('block_threshold')
    wt = thr.get('warn_threshold')
    if bt is None or wt is None:
        return None
    return float(bt), float(wt)


def lookup_features_for(number: str, features_csv: str) -> Optional[Dict[str, float]]:
    """Найти готовый компакт-вектор для номера в processed/ru_metadata_features.csv."""
    if not os.path.isfile(features_csv):
        return None
    ru_to_eng = {v: k for k, v in FIELD_TO_RU.items()}
    with open(features_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            num = row.get('номер') or row.get('normalized_number') or row.get('number')
            if num != number:
                continue
            features: Dict[str, float] = {}
            for eng in COMPACT_FEATURES:
                ru = FIELD_TO_RU.get(eng, eng)
                raw = row.get(ru)
                if raw is None or raw == '':
                    raw = row.get(eng)
                if raw is None or raw == '':
                    return None
                features[eng] = float(raw)
            return features
    return None


COLD_START_MASK_FEATURES: Tuple[str, ...] = (
    'inAllowlist', 'inBlacklist',
    'reputationScore', 'sourceConfidence',
    'reviewsLog', 'negativeRatio', 'searchVolumeLog',
    'hasFraudCategory', 'hasTelemarketingCategory',
)


def features_from_scratch(number: str) -> Dict[str, float]:
    """Холодный старт: метадата = {}, isContact / inBlacklist / inAllowlist = 0.

    После расчёта обнуляем те же 9 cold-mask признаков, что обнуляет тренер
    (`make_cold_view`) во время cold-аугментации, и форсим noMetadata=1.
    Это устраняет mismatch, при котором `compact_feature_vector` отдавал
    `sourceConfidence=0.5` (default), хотя student-MLP обучен ожидать 0
    в cold-view → модель ошибочно считала вход «warm» и предсказывала ALLOW
    с высокой уверенностью.
    """
    features = compact_feature_vector(number, label='UNKNOWN', metadata={})
    for name in COLD_START_MASK_FEATURES:
        if name in features:
            features[name] = 0.0
    features['noMetadata'] = 1.0
    return features


def run_tflite(model_path: str, x: List[float]):
    import numpy as np
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    arr = np.asarray(x, dtype=np.float32).reshape(1, -1)
    interp.set_tensor(in_d['index'], arr)
    interp.invoke()
    return interp.get_tensor(out_d['index']).reshape(-1).tolist()


def verdict_from(probs: List[float], thresholds: Optional[Tuple[float, float]]) -> str:
    allow_p, warn_p, block_p = probs
    if thresholds is not None:
        block_t, warn_t = thresholds
        if block_p >= block_t:
            return 'BLOCK'
        if warn_p >= warn_t:
            return 'WARN'
        return 'ALLOW'
    # argmax fallback
    idx = max(range(3), key=lambda i: probs[i])
    return ID_TO_LABEL.get(idx, 'ALLOW')


def predict_one(number_in: str, model_path: str, thresholds: Optional[Tuple[float, float]],
                features_csv: str, force_cold: bool = False,
                disable_rules: bool = False) -> Dict:
    norm = normalize_ru_phone(number_in)
    if not norm:
        return {'input': number_in, 'error': 'invalid_number'}
    features = None if force_cold else lookup_features_for(norm, features_csv)
    source = 'dataset' if features is not None else 'cold'
    if features is None:
        features = features_from_scratch(norm)
    vec = [features[name] for name in COMPACT_FEATURES]
    probs = run_tflite(model_path, vec)
    model_verdict = verdict_from(probs, thresholds)
    if disable_rules:
        final_verdict, rule_hits = model_verdict, []
    else:
        final_verdict, rule_hits = apply_rules(model_verdict, features, source)
    risk_score = int(round(probs[2] * 100))
    if final_verdict == 'WARN' and model_verdict == 'ALLOW':
        # Чтобы UI/CLI видел, что эскалация произошла (а не «модель сама решила
        # WARN»), бьём risk_score хотя бы до warn-порога * 100.
        warn_floor = int(round((thresholds[1] if thresholds else 0.10) * 100))
        risk_score = max(risk_score, warn_floor)
    return {
        'input': number_in,
        'normalized': norm,
        'feature_source': source,
        'probs': {'ALLOW': probs[0], 'WARN': probs[1], 'BLOCK': probs[2]},
        'model_verdict': model_verdict,
        'verdict': final_verdict,
        'risk_score': risk_score,
        'rule_overrides': [
            {'rule_id': h.rule_id, 'verdict': h.verdict_override, 'reason': h.reason}
            for h in rule_hits
        ],
        'features': features,
    }


BATCH_VERDICT_FIELDS: Tuple[str, ...] = (
    'normalized_number',
    'expected_label',
    'verdict',
    'model_verdict',
    'risk_score',
    'probs_ALLOW',
    'probs_WARN',
    'probs_BLOCK',
    'feature_source',
    'rule_overrides',
    'disagreement',
    'error',
)


def _row_number(row: Dict[str, str]) -> Optional[str]:
    """Find the phone-number column in a candidate CSV row (multiple naming
    conventions supported)."""
    for col in ('normalized_number', 'номер', 'number', 'phone'):
        v = (row.get(col) or '').strip()
        if v:
            return v
    return None


def run_from_csv(in_path: str, out_path: str, model_path: str,
                 thresholds: Optional[Tuple[float, float]], features_csv: str,
                 *, force_cold: bool, disable_rules: bool, limit: Optional[int] = None,
                 progress_every: int = 50) -> Dict[str, int]:
    """Batch verdicts: read candidates CSV → write verdicts CSV.

    If the input CSV has an ``expected_label`` column, the output CSV adds a
    ``disagreement`` flag (True iff the model verdict differs).
    """
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f'input CSV not found: {in_path}')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    counters = {'total': 0, 'errors': 0, 'disagreement': 0}
    label_counts: Dict[str, int] = {}
    with open(in_path, 'r', encoding='utf-8', newline='') as fin, \
            open(out_path, 'w', encoding='utf-8', newline='') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=list(BATCH_VERDICT_FIELDS))
        writer.writeheader()
        for row in reader:
            if limit is not None and counters['total'] >= limit:
                break
            counters['total'] += 1
            num_in = _row_number(row)
            expected = (row.get('expected_label') or '').strip().upper() or ''
            if not num_in:
                counters['errors'] += 1
                writer.writerow({
                    'normalized_number': '', 'expected_label': expected,
                    'verdict': '', 'model_verdict': '', 'risk_score': '',
                    'probs_ALLOW': '', 'probs_WARN': '', 'probs_BLOCK': '',
                    'feature_source': '', 'rule_overrides': '',
                    'disagreement': '', 'error': 'no_number_column',
                })
                continue
            r = predict_one(num_in, model_path, thresholds, features_csv,
                            force_cold=force_cold, disable_rules=disable_rules)
            if 'error' in r:
                counters['errors'] += 1
                writer.writerow({
                    'normalized_number': r.get('normalized', num_in),
                    'expected_label': expected, 'verdict': '', 'model_verdict': '',
                    'risk_score': '', 'probs_ALLOW': '', 'probs_WARN': '', 'probs_BLOCK': '',
                    'feature_source': '', 'rule_overrides': '',
                    'disagreement': '', 'error': r['error'],
                })
                continue
            p = r['probs']
            verdict = r['verdict']
            label_counts[verdict] = label_counts.get(verdict, 0) + 1
            disagree = bool(expected) and (expected != verdict)
            if disagree:
                counters['disagreement'] += 1
            rules_dump = ';'.join(
                f'{h["rule_id"]}->{h["verdict"]}' for h in r.get('rule_overrides', [])
            )
            writer.writerow({
                'normalized_number': r['normalized'],
                'expected_label': expected,
                'verdict': verdict,
                'model_verdict': r['model_verdict'],
                'risk_score': r['risk_score'],
                'probs_ALLOW': f"{p['ALLOW']:.6f}",
                'probs_WARN':  f"{p['WARN']:.6f}",
                'probs_BLOCK': f"{p['BLOCK']:.6f}",
                'feature_source': r['feature_source'],
                'rule_overrides': rules_dump,
                'disagreement': '1' if disagree else '0',
                'error': '',
            })
            if progress_every and counters['total'] % progress_every == 0:
                print(f'  ... processed {counters["total"]} rows '
                      f'(err={counters["errors"]}, disagree={counters["disagreement"]})',
                      file=sys.stderr)
    out = {**counters, **{f'verdict_{k}': v for k, v in label_counts.items()}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('numbers', nargs='*', help='Один или несколько телефонных номеров (опционально, если задан --from-csv).')
    ap.add_argument('--model', default=DEFAULT_MODEL, help=f'Путь к .tflite (default: {DEFAULT_MODEL}).')
    ap.add_argument('--card', default=DEFAULT_CARD, help=f'Путь к model_card.json (default: {DEFAULT_CARD}).')
    ap.add_argument('--features-csv', default=DEFAULT_FEATURES_CSV,
                    help='Lookup CSV с метадатой по номерам (default: ru_metadata_features.csv).')
    ap.add_argument('--cold', action='store_true',
                    help='Игнорировать lookup CSV; считать фичи с нуля (как «неизвестный номер»).')
    ap.add_argument('--no-rules', action='store_true',
                    help='Отключить post-model rule engine (cold-start prefix-risk WARN и т.п.).')
    ap.add_argument('--show-features', action='store_true', help='Распечатать все 32 фичи.')
    ap.add_argument('--json', action='store_true', help='Вывести результат как JSON.')
    ap.add_argument('--from-csv', dest='from_csv', default=None,
                    help='Batch mode: read candidates from this CSV (cols: normalized_number, '
                         'optionally expected_label).')
    ap.add_argument('--out-csv', dest='out_csv', default=None,
                    help='Output CSV for batch mode (cols: normalized_number, verdict, '
                         'probs_*, disagreement, …). Required with --from-csv.')
    ap.add_argument('--limit', type=int, default=None,
                    help='Limit number of rows processed in --from-csv mode.')
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        print(f'ERROR: model not found: {args.model}', file=sys.stderr)
        return 2

    thresholds = load_thresholds(args.card, cold=args.cold)
    if thresholds:
        which = 'cold_thresholds' if args.cold else 'thresholds'
        print(f'  {which} from model_card.json: block={thresholds[0]:.3f} warn={thresholds[1]:.3f}')
    else:
        print('  no thresholds in model_card.json — falling back to argmax')

    if args.from_csv:
        if not args.out_csv:
            print('ERROR: --from-csv requires --out-csv', file=sys.stderr)
            return 2
        summary = run_from_csv(args.from_csv, args.out_csv, args.model,
                               thresholds, args.features_csv,
                               force_cold=args.cold, disable_rules=args.no_rules,
                               limit=args.limit)
        print(f'\nbatch summary: {summary}')
        print(f'  wrote → {args.out_csv}')
        return 0

    if not args.numbers:
        ap.error('provide phone numbers or use --from-csv')

    results = [predict_one(n, args.model, thresholds, args.features_csv,
                           force_cold=args.cold, disable_rules=args.no_rules)
               for n in args.numbers]

    if args.json:
        for r in results:
            r.pop('features', None) if not args.show_features else None
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    for r in results:
        if 'error' in r:
            print(f'\n{r["input"]:20s} → ERROR: {r["error"]}')
            continue
        p = r['probs']
        print(f'\n{r["input"]:20s} → {r["normalized"]:15s} [{r["feature_source"]}]')
        print(f'  probs:    ALLOW={p["ALLOW"]:.4f}  WARN={p["WARN"]:.4f}  BLOCK={p["BLOCK"]:.4f}')
        if r.get('rule_overrides'):
            print(f'  model:    {r["model_verdict"]}')
            for hit in r['rule_overrides']:
                print(f'  rule[+]:  {hit["rule_id"]} → {hit["verdict"]}  ({hit["reason"]})')
        print(f'  verdict:  {r["verdict"]}     risk_score={r["risk_score"]}')
        if args.show_features:
            for k, v in r['features'].items():
                print(f'    {k:24s} = {v:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
