"""Take a stratified 5k subset from cold_eval_holdout_500k.csv -> cold_eval_5k.csv.

5000 rows are statistically more than enough for cold-start eval gate (binomial
CI is ±1% at p=0.5). Keeps the same class proportions as the full hold-out.

Run:
  C:\\Users\\Redmi\\AppData\\Local\\Programs\\Python\\Python312\\python.exe scripts\\subset_holdout.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "datasets" / "ru" / "eval" / "cold_eval_holdout_500k.csv"
OUT = REPO_ROOT / "datasets" / "ru" / "eval" / "cold_eval_5k.csv"
TARGET = 5000
SEED = 42


def main() -> int:
    if not SRC.is_file():
        sys.stderr.write(f"missing: {SRC}\n")
        return 2

    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] reading {SRC.name}...", flush=True)
    df = pd.read_csv(SRC, dtype=str, low_memory=False)
    label_col = df.columns[-1]

    n = len(df)
    print(f"[{time.time()-t0:6.1f}s] loaded {n:,}, distrib: {dict(df[label_col].value_counts())}", flush=True)

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
    out = pd.concat(parts).sort_index()
    print(f"[{time.time()-t0:6.1f}s] subset: {len(out):,}", flush=True)

    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"[{time.time()-t0:6.1f}s] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
