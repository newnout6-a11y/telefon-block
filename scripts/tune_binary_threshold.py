"""Find the best block_threshold for spam_model_binary.tflite on cold_eval_5k.

Sweeps thresholds 0.01..0.50 and prints precision/recall/FP-rate. Picks the
threshold that maximizes BLOCK F1 while keeping FP rate <= 0.20 and precision
>= 0.85.

Run:
  C:\\Users\\Redmi\\AppData\\Local\\Programs\\Python\\Python312\\python.exe scripts\\tune_binary_threshold.py
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ru_metadata_features import COMPACT_FEATURES, FIELD_TO_RU  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TFLITE = REPO_ROOT / "app" / "src" / "main" / "assets" / "experimental" / "spam_model_binary.tflite"
EVAL = REPO_ROOT / "datasets" / "ru" / "eval" / "cold_eval_5k_labeled.csv"
CARD = REPO_ROOT / "app" / "src" / "main" / "assets" / "experimental" / "model_card_binary.json"

LABEL_MAP = {"ALLOW": 0, "WARN": 1, "BLOCK": 2}


def load_eval():
    with open(EVAL, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    headers = list(rows[0].keys())
    label_key = "метка" if "метка" in headers else "label"
    X_rows, y_list = [], []
    no_meta_idx = COMPACT_FEATURES.index("noMetadata") if "noMetadata" in COMPACT_FEATURES else -1
    for row in rows:
        lbl = str(row[label_key]).strip().upper()
        if lbl not in LABEL_MAP:
            continue
        y_list.append(LABEL_MAP[lbl])
        feat_row = []
        for name in COMPACT_FEATURES:
            col = name if name in headers else FIELD_TO_RU.get(name, name)
            feat_row.append(float(row.get(col, 0.0) or 0.0))
        # Cold mode: zero out 9 metadata features.
        leak_features = [
            "inAllowlist", "inBlacklist", "reputationScore", "sourceConfidence",
            "reviewsLog", "negativeRatio", "searchVolumeLog",
            "hasFraudCategory", "hasTelemarketingCategory",
        ]
        for fname in leak_features:
            if fname in COMPACT_FEATURES:
                feat_row[COMPACT_FEATURES.index(fname)] = 0.0
        if no_meta_idx >= 0:
            feat_row[no_meta_idx] = 1.0
        X_rows.append(feat_row)
    return np.array(X_rows, dtype=np.float32), np.array(y_list, dtype=np.int64)


def main() -> int:
    print(f"loading eval from {EVAL}", flush=True)
    X, y = load_eval()
    print(f"eval rows: {len(X)}, classes: ALLOW={int((y==0).sum())} WARN={int((y==1).sum())} BLOCK={int((y==2).sum())}", flush=True)

    # Inference via ai-edge-litert.
    print(f"loading {TFLITE}", flush=True)
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=str(TFLITE))
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    print("running inference...", flush=True)
    probs = np.zeros(len(X), dtype=np.float32)
    for i, row in enumerate(X):
        interp.set_tensor(in_d["index"], row.reshape(1, -1).astype(np.float32))
        interp.invoke()
        probs[i] = interp.get_tensor(out_d["index"]).reshape(-1)[0]

    # In binary merge_block: y=0 (ALLOW) is negative, y=1 or y=2 (WARN/BLOCK) is positive.
    is_spam = (y >= 1)  # spam = WARN or BLOCK

    # Sweep thresholds.
    print(f"\n{'thresh':>7} {'P':>6} {'R':>6} {'F1':>6} {'FP_rate':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}", flush=True)
    print("-" * 70)
    best_f1 = -1.0
    best_thresh = None
    best_metrics = None
    for thresh in np.arange(0.01, 0.51, 0.01):
        pred = (probs >= thresh).astype(np.int64)
        tp = int(((pred == 1) & is_spam).sum())
        fp = int(((pred == 1) & ~is_spam).sum())
        fn = int(((pred == 0) & is_spam).sum())
        tn = int(((pred == 0) & ~is_spam).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        passes = prec >= 0.85 and rec >= 0.55 and fp_rate <= 0.20
        marker = "  <-- BEST" if passes and f1 > best_f1 else ""
        if passes and f1 > best_f1:
            best_f1, best_thresh, best_metrics = f1, thresh, (prec, rec, f1, fp_rate, tp, fp, fn, tn)
        print(f"{thresh:>7.2f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {fp_rate:>8.3f} {tp:>5} {fp:>5} {fn:>5} {tn:>5}{marker}", flush=True)

    print()
    if best_thresh is not None:
        prec, rec, f1, fp_rate, tp, fp, fn, tn = best_metrics
        print(f"BEST: threshold={best_thresh:.2f} F1={f1:.3f} P={prec:.3f} R={rec:.3f} FP_rate={fp_rate:.3f}", flush=True)

        # Update model card in place.
        card = json.loads(CARD.read_text(encoding="utf-8"))
        card.setdefault("thresholds", {})["block_threshold"] = float(best_thresh)
        card["thresholds"]["block_precision"] = float(prec)
        card["thresholds"]["block_recall"] = float(rec)
        card["thresholds"]["block_f1"] = float(f1)
        card["thresholds"]["allow_fp_rate"] = float(fp_rate)
        card["thresholds"]["tuned_on"] = "cold_eval_5k_labeled.csv"
        CARD.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"updated {CARD} with new threshold")
    else:
        print("NO threshold passes all 3 gates. Showing the best by F1 alone:")
        # Find max-F1 regardless of gates.
        best_thresh_any, best_f1_any, best_metrics_any = None, -1.0, None
        for thresh in np.arange(0.01, 0.51, 0.01):
            pred = (probs >= thresh).astype(np.int64)
            tp = int(((pred == 1) & is_spam).sum())
            fp = int(((pred == 1) & ~is_spam).sum())
            fn = int(((pred == 0) & is_spam).sum())
            tn = int(((pred == 0) & ~is_spam).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            if f1 > best_f1_any:
                best_f1_any, best_thresh_any, best_metrics_any = f1, thresh, (prec, rec, f1, fp_rate)
        prec, rec, f1, fp_rate = best_metrics_any
        print(f"max F1: threshold={best_thresh_any:.2f} F1={f1:.3f} P={prec:.3f} R={rec:.3f} FP_rate={fp_rate:.3f}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
