"""Build a fine-tune feedback CSV from a spam_predict verdicts file.

Use case: VK Service Token collector produced
``datasets/ru/eval/vk_candidates.csv`` (ALLOW-leaning cold-start candidates).
``spam_predict.py --from-csv ... --out-csv vk_verdicts.csv --no-rules`` then
scored every number with the pure cold model.

The model flags some of those numbers as BLOCK (78/474 in our smoke run).
The user reviewed them and decided they are *not* spam (real users posting on
public classifieds walls). We want the model to learn that — **without**
adding the numbers to the bundle whitelist (which would just hard-override
the model). The right mechanism is feedback fine-tune:

  * take the rows where ``verdict == BLOCK`` (or another target),
  * compute the same cold-start COMPACT_FEATURES that the model saw,
  * write a ``user_action=not_spam`` feedback row,
  * pipe it into ``scripts/online_fine_tune.py --feedback ...``

``online_fine_tune.py`` then mixes the corrections into a re-training run
(weight 3x for BLOCK→ALLOW corrections, alpha=0.3 by default) and exports a
new TFLite + model card. Re-running ``spam_predict`` on the same numbers
shows how many corrections were absorbed.

Typical usage::

    # 1. build feedback CSV from a verdicts file
    python scripts/fine_tune_from_verdicts.py \
        --verdicts /tmp/vk_verdicts.csv \
        --target-verdict BLOCK \
        --user-action not_spam \
        --out /tmp/vk_feedback.csv

    # 2. fine-tune model from those corrections (TFLite to /tmp for safety)
    python scripts/online_fine_tune.py \
        --feedback /tmp/vk_feedback.csv \
        --export-tflite \
        --tflite-output /tmp/spam_model_ft.tflite \
        --model-card-output /tmp/model_card_ft.json

    # 3. verify the model now predicts ALLOW on the same numbers
    python scripts/spam_predict.py \
        --from-csv /tmp/vk_candidates.csv \
        --out-csv  /tmp/vk_verdicts_ft.csv \
        --model    /tmp/spam_model_ft.tflite \
        --card     /tmp/model_card_ft.json \
        --no-rules

The bridge intentionally does **not** call ``online_fine_tune`` itself —
keeping the steps separate makes it easy to inspect intermediate CSVs and
test in isolation.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from ru_metadata_features import COMPACT_FEATURES  # noqa: E402
from spam_predict import features_from_scratch  # noqa: E402

VALID_TARGET_VERDICTS = ('BLOCK', 'WARN', 'ALLOW', 'ANY')
VALID_USER_ACTIONS = ('not_spam', 'fraud', 'dismiss')


def _row_number(row: Dict[str, str]) -> Optional[str]:
    """Find the phone column in a verdicts row (multiple naming conventions)."""
    for col in ('normalized_number', 'номер', 'number', 'phone'):
        v = (row.get(col) or '').strip()
        if v:
            return v
    return None


def select_rows(rows: Iterable[Dict[str, str]], target_verdict: str) -> List[Dict[str, str]]:
    """Filter verdicts rows by ``verdict`` (or ``model_verdict``) column.

    ``target_verdict='ANY'`` keeps everything. Empty / missing ``verdict``
    rows are skipped — they likely had errors during prediction.
    """
    target = target_verdict.upper()
    out: List[Dict[str, str]] = []
    for row in rows:
        if (row.get('error') or '').strip():
            continue
        v = (row.get('verdict') or row.get('model_verdict') or '').strip().upper()
        if not v:
            continue
        if target == 'ANY' or v == target:
            out.append(row)
    return out


FEEDBACK_FIELDS: Tuple[str, ...] = (
    'normalized_number',
    'verdict',
    'user_action',
    *COMPACT_FEATURES,
)


def build_feedback_records(
    selected: Sequence[Dict[str, str]],
    user_action: str,
    *,
    verdict_override: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Convert verdicts rows → feedback records consumable by ``online_fine_tune``.

    Features are recomputed via ``features_from_scratch`` (cold-start mask)
    so the feedback row matches the input vector the model originally saw.

    ``verdict_override`` lets the caller pin the per-row ``verdict`` column
    independent of what the verdicts CSV said. This is useful when the
    target is ``ANY`` and the caller wants the entire batch treated as one
    correction class.
    """
    out: List[Dict[str, object]] = []
    for row in selected:
        number = _row_number(row)
        if not number:
            continue
        feats = features_from_scratch(number)
        verdict = (
            verdict_override.upper() if verdict_override
            else (row.get('verdict') or row.get('model_verdict') or '').strip().upper()
        )
        record: Dict[str, object] = {
            'normalized_number': number,
            'verdict': verdict,
            'user_action': user_action,
        }
        for feat in COMPACT_FEATURES:
            record[feat] = float(feats.get(feat, 0.0))
        out.append(record)
    return out


def write_feedback_csv(path: str, records: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(FEEDBACK_FIELDS))
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, '') for k in FEEDBACK_FIELDS})


def read_verdicts(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f'verdicts CSV not found: {path}')
    with open(path, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--verdicts', required=True,
                    help='Input verdicts CSV (output of spam_predict.py --from-csv).')
    ap.add_argument('--target-verdict', default='BLOCK', choices=list(VALID_TARGET_VERDICTS),
                    help='Filter verdicts rows by this verdict (default BLOCK).')
    ap.add_argument('--user-action', default='not_spam', choices=list(VALID_USER_ACTIONS),
                    help='What the user said about these numbers (default not_spam).')
    ap.add_argument('--verdict-override', default=None,
                    help='Force a specific verdict on the output rows '
                         '(useful with --target-verdict=ANY).')
    ap.add_argument('--out', required=True,
                    help='Output feedback CSV consumable by online_fine_tune.py.')
    ap.add_argument('--limit', type=int, default=None,
                    help='Cap on number of feedback rows (debug aid).')
    args = ap.parse_args()

    rows = read_verdicts(args.verdicts)
    selected = select_rows(rows, args.target_verdict)
    if args.limit is not None:
        selected = selected[: max(0, int(args.limit))]
    if not selected:
        print(f'no rows matched verdict={args.target_verdict} in {args.verdicts}',
              file=sys.stderr)
        return 0
    records = build_feedback_records(
        selected,
        user_action=args.user_action,
        verdict_override=args.verdict_override,
    )
    write_feedback_csv(args.out, records)
    print(f'wrote {len(records)} feedback rows to {args.out} '
          f'(verdict={args.target_verdict}, user_action={args.user_action})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
