# Contributing

## Branch Model

- **`main`** — production branch. All PRs target `main`.
- Feature branches: `kiro/<feature>`, `devin/<session-id>-<feature>`, `fix/<issue>`.
- Direct pushes to `main` are prohibited. Use PRs with at least 1 review.

## Before Merging

1. **All existing tests pass**: `python3 -m pytest tests/ -q`
2. **Golden-set eval passes** (when model changes):
   ```bash
   python3 scripts/eval_golden_set.py \
       --model app/src/main/assets/spam_model.tflite \
       --card  app/src/main/assets/model_card.json \
       --golden datasets/ru/eval/cold_eval_600.csv \
       --cold
   ```
3. **Python syntax check**: `python3 -m py_compile scripts/<modified_file>.py`
4. **Android build** (when Kotlin changes): `./gradlew assembleDebug`

## Model Changes

- Never commit model retrains directly to `main` without eval gate passing.
- Experimental models go to `app/src/main/assets/experimental/`.
- Promotion to production `spam_model.tflite` requires:
  - Golden-set eval pass (PR-6)
  - Manual review of precision/recall deltas
  - Separate PR with clear before/after metrics in description

## User Feedback

- User "not spam" taps go to personal allowlist (DataStore), NOT to model retrain.
- Batch retrains happen weekly from accumulated `training_data` table export.
- Never retrain the model for fewer than 100 verified feedback examples.
