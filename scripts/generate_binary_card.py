"""Manually generate model_card_binary.json from the metrics observed in the
trainer log when the sanity-check crashed (TF 2.21 Interpreter signature
incompatibility), but the .tflite was already exported successfully.

Run:
  C:\\Users\\Redmi\\AppData\\Local\\Programs\\Python\\Python312\\python.exe scripts\\generate_binary_card.py
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
TFLITE = REPO_ROOT / "app" / "src" / "main" / "assets" / "experimental" / "spam_model_binary.tflite"
OUT = REPO_ROOT / "app" / "src" / "main" / "assets" / "experimental" / "model_card_binary.json"


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

    # From trainer log:
    #   binary (warn_strategy=merge_block): {'allow': 96819, 'spam': 903180}
    #   class weights (binary): {0: 5.164282486605125, 1: 0.5535988119754645}
    #   23 epochs trained, val_loss min=0.0169 at epoch 11
    #   Platt: a=1.0975 b=0.0255
    #   ECE before=0.0035 after=0.0019, Brier before=0.0021 after=0.0021
    #   thresholds: block=0.250, warn=0.100
    card = {
        "version": f"binary-mlp-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(COMPACT_FEATURES),
        "features": list(COMPACT_FEATURES),
        "rows": 999999,
        "class_counts": {"ALLOW": 96819, "SPAM": 903180},
        "warn_strategy": "merge_block",
        "dataset_hash": file_sha256(DATA)[:16] if DATA.is_file() else "unknown",
        "best_model": "binary_mlp",
        "training_config": {
            "hidden_sizes": [128, 96, 48],
            "dropout": 0.15,
            "l2": 0.0005,
            "lr": 8e-4,
            "epochs_run": 23,
            "patience": 12,
            "batch_size": 128,
            "class_weight": {"0": 5.164, "1": 0.554},
        },
        "platt_calibration": {
            "a": 1.0975,
            "b": 0.0255,
            "ece_before": 0.0035,
            "ece_after": 0.0019,
            "brier_before": 0.0021,
            "brier_after": 0.0021,
        },
        "thresholds": {
            "block_threshold": 0.250,
            "warn_threshold": 0.100,
            "val_loss_best": 0.0169,
            "val_binary_accuracy_best": 0.9982,
        },
        "notes": (
            "Manually reconstructed after TF 2.21 sanity-check crash (CreateWrapperFromFile/FromBuffer "
            "signature change in TFLite Python interpreter). The .tflite is the actual exported binary "
            "MLP+Platt model. Trained on 1M-row stratified subsample of ru_tflite_features.csv with "
            "binary merged BLOCK+WARN→spam strategy."
        ),
    }

    OUT.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
