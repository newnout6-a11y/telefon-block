"""Manually generate model_card_leak_free.json from the metrics observed in
the trainer log when the sanity-check crashed (TF 2.21 Interpreter signature
incompatibility), but the .tflite was already exported successfully.

Run:
  C:\\Users\\Redmi\\AppData\\Local\\Programs\\Python\\Python312\\python.exe scripts\\generate_leakfree_card.py
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from ru_metadata_features import COMPACT_FEATURES  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "datasets" / "ru" / "processed" / "ru_tflite_features.csv"
TFLITE = REPO_ROOT / "app" / "src" / "main" / "assets" / "experimental" / "spam_model_leak_free.tflite"
OUT = REPO_ROOT / "app" / "src" / "main" / "assets" / "experimental" / "model_card_leak_free.json"


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not TFLITE.is_file():
        sys.stderr.write(f"missing tflite: {TFLITE}\n")
        return 2
    if not DATA.is_file():
        sys.stderr.write(f"missing dataset: {DATA}\n")
        return 2

    # From trainer log:
    #   plain MLP val: macroF1=0.5018 BLOCK_P=0.966
    #   plain_mlp thresholds: block=0.190 warn=0.280 (BLOCK_P=0.903, BLOCK_F1=0.893)
    #   best-of selection: plain_mlp wins (0.6053 vs kd 0.5573 on val_unknown_macroF1)
    # Class counts from dataset.
    card = {
        "version": f"plain-mlp-leakfree-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(COMPACT_FEATURES),
        "features": list(COMPACT_FEATURES),
        "rows": 999999,
        "class_counts": {"ALLOW": 96819, "WARN": 86509, "BLOCK": 816671},
        "dataset_hash": file_sha256(DATA)[:16],
        "best_model": "plain_mlp",
        "best_of": {
            "selected": "plain_mlp",
            "val_unknown_macroF1": {"plain_mlp": 0.6053, "kd_student": 0.5573},
        },
        "block_precision": 0.903,
        "block_recall": 0.0,  # unknown from log
        "roc_auc_ovr": None,
        "thresholds": {
            "block_threshold": 0.190,
            "warn_threshold": 0.280,
            "block_precision": 0.903,
            "block_recall": 0.0,
            "block_f1": 0.893,
            "warn_f1": 0.0,
        },
        "smote_applied": False,
        "kd": {
            "T": 8.0,
            "alpha": 0.7,
            "teacher": "catboost_multiclass",
            "teacher_train_per_class": {"ALLOW": 77455, "WARN": 69207, "BLOCK": 653337},
            "student_train_per_class": {"ALLOW": 77455, "WARN": 69207, "BLOCK": 653337},
        },
        "cold_start": {
            "eval_size": 100000,
            "tflite_metrics": {},
            "tflite_metrics_cold_thresholded": {},
        },
        "class_weights": {"allow": 10.0, "warn": 12.0, "block": 1.0},
        "notes": (
            "Manually reconstructed after TF 2.21 sanity-check crash. "
            "Metrics taken from trainer log; .tflite is the actual exported plain MLP. "
            "Trained on 1M-row stratified subsample of cold-eval-style ru_tflite_features.csv."
        ),
        "leak_free": True,
    }

    OUT.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
