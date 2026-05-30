"""Stratified subsample of ru_tflite_features.csv to a manageable size for training.

Backs up the original 2.46M-row file as ru_tflite_features_full.csv, then
overwrites ru_tflite_features.csv with a 1M-row stratified subsample that
preserves class proportions.

Run:
  C:\\Users\\Redmi\\AppData\\Local\\Programs\\Python\\Python312\\python.exe scripts\\subsample_train.py
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import time

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "datasets" / "ru" / "processed" / "ru_tflite_features.csv"
BACKUP = REPO_ROOT / "datasets" / "ru" / "processed" / "ru_tflite_features_full.csv"
TARGET = 1_000_000
SEED = 42


def main() -> int:
    if not SRC.is_file():
        sys.stderr.write(f"missing: {SRC}\n")
        return 2

    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] reading {SRC.name}...", flush=True)
    df = pd.read_csv(SRC, dtype=str, low_memory=False)
    n = len(df)
    label_col = df.columns[-1]
    print(f"[{time.time()-t0:6.1f}s] loaded {n:,}, distrib: {dict(df[label_col].value_counts())}", flush=True)

    if n <= TARGET:
        print(f"[{time.time()-t0:6.1f}s] already <= {TARGET}, no-op", flush=True)
        return 0

    if not BACKUP.is_file():
        print(f"[{time.time()-t0:6.1f}s] backing up to {BACKUP.name}...", flush=True)
        shutil.copy2(SRC, BACKUP)
        print(f"[{time.time()-t0:6.1f}s] backup done", flush=True)
    else:
        print(f"[{time.time()-t0:6.1f}s] backup already exists, skipping", flush=True)

    target_per_class = (
        df[label_col].value_counts()
        .mul(TARGET / n)
        .round()
        .astype(int)
    )
    print(f"[{time.time()-t0:6.1f}s] target per class: {dict(target_per_class)}", flush=True)

    parts = []
    for cls, k in target_per_class.items():
        sub = df[df[label_col] == cls].sample(n=int(k), random_state=SEED)
        parts.append(sub)
    out = pd.concat(parts).sample(frac=1, random_state=SEED).sort_index()
    print(f"[{time.time()-t0:6.1f}s] subsample: {len(out):,}", flush=True)

    out.to_csv(SRC, index=False, encoding="utf-8")
    print(f"[{time.time()-t0:6.1f}s] wrote {SRC} ({SRC.stat().st_size/1024/1024:.0f}MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
