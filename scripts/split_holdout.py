"""Split ru_tflite_features.csv into train (~2.46M) + holdout (500k).

Stratified random split (preserves class balance), seed=42 for reproducibility.

Outputs:
  datasets/ru/processed/ru_tflite_features_train.csv  - used for training
  datasets/ru/eval/cold_eval_holdout_500k.csv         - cold-start hold-out

Also overwrites datasets/ru/processed/ru_tflite_features.csv with the train
split so that pipeline_orchestrator.py preflight + train_full_pipeline.ps1
pick up the train-only file.

Run:
  C:\\Users\\Redmi\\AppData\\Local\\Programs\\Python\\Python312\\python.exe scripts\\split_holdout.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROC = REPO_ROOT / "datasets" / "ru" / "processed"
EVAL = REPO_ROOT / "datasets" / "ru" / "eval"
SRC = PROC / "ru_tflite_features.csv"
TRAIN = PROC / "ru_tflite_features_train.csv"
HOLDOUT = EVAL / "cold_eval_holdout_500k.csv"
HOLDOUT_SIZE = 500_000
SEED = 42


def main() -> int:
    if not SRC.is_file():
        sys.stderr.write(f"source missing: {SRC}\n")
        return 2

    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] reading {SRC}...", flush=True)
    df = pd.read_csv(SRC, dtype=str, low_memory=False)
    print(f"[{time.time()-t0:6.1f}s] loaded {len(df):,} rows, {len(df.columns)} cols", flush=True)

    label_col = df.columns[-1]
    print(f"[{time.time()-t0:6.1f}s] label column: {label_col!r}; distribution:", flush=True)
    print(df[label_col].value_counts().to_string())

    # Stratified sample: take HOLDOUT_SIZE total, proportional per class.
    total = len(df)
    if HOLDOUT_SIZE >= total:
        sys.stderr.write(f"holdout size {HOLDOUT_SIZE} >= total {total}, refusing\n")
        return 2

    holdout_per_class = (
        df[label_col].value_counts()
        .mul(HOLDOUT_SIZE / total)
        .round()
        .astype(int)
    )
    print(f"[{time.time()-t0:6.1f}s] holdout per class:", flush=True)
    print(holdout_per_class.to_string())

    parts = []
    for cls, n in holdout_per_class.items():
        sub = df[df[label_col] == cls].sample(n=int(n), random_state=SEED)
        parts.append(sub)
    holdout_df = pd.concat(parts).sort_index()
    train_df = df.drop(holdout_df.index)
    print(
        f"[{time.time()-t0:6.1f}s] split: train={len(train_df):,}, holdout={len(holdout_df):,}",
        flush=True,
    )

    EVAL.mkdir(parents=True, exist_ok=True)
    print(f"[{time.time()-t0:6.1f}s] writing {HOLDOUT}...", flush=True)
    holdout_df.to_csv(HOLDOUT, index=False, encoding="utf-8")
    print(f"[{time.time()-t0:6.1f}s] writing {TRAIN}...", flush=True)
    train_df.to_csv(TRAIN, index=False, encoding="utf-8")

    # Replace the canonical training file with the train-only split so the
    # pipeline orchestrator's preflight + trainers pick it up automatically.
    print(f"[{time.time()-t0:6.1f}s] overwriting {SRC} with train split...", flush=True)
    train_df.to_csv(SRC, index=False, encoding="utf-8")

    print(
        f"[{time.time()-t0:6.1f}s] done. train rows: {len(train_df):,}; "
        f"holdout rows: {len(holdout_df):,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
