"""Convert label IDs (0/1/2) in cold_eval_5k.csv -> string labels (ALLOW/WARN/BLOCK)
so that eval_golden_set.py LABEL_MAP picks them up.

Run:
  C:\\Users\\Redmi\\AppData\\Local\\Programs\\Python\\Python312\\python.exe scripts\\fix_eval_labels.py
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "datasets" / "ru" / "eval" / "cold_eval_5k.csv"
DST = REPO_ROOT / "datasets" / "ru" / "eval" / "cold_eval_5k_labeled.csv"

ID_TO_LABEL = {"0": "ALLOW", "1": "WARN", "2": "BLOCK"}


def main() -> int:
    if not SRC.is_file():
        sys.stderr.write(f"missing: {SRC}\n")
        return 2

    print(f"reading {SRC.name}...", flush=True)
    df = pd.read_csv(SRC, dtype=str, low_memory=False)

    # Last column is the label (Russian header 'метка', integer IDs).
    label_col = df.columns[-1]
    print(f"label column: {label_col!r}", flush=True)
    print(f"current values: {df[label_col].value_counts().to_dict()}", flush=True)

    df[label_col] = df[label_col].map(ID_TO_LABEL).fillna(df[label_col])
    print(f"converted values: {df[label_col].value_counts().to_dict()}", flush=True)

    df.to_csv(DST, index=False, encoding="utf-8")
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
