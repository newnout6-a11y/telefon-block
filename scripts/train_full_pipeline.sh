#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# FULL TRAINING PIPELINE — от сырых данных до обученной leak-free модели.
#
# Запуск:
#   chmod +x scripts/train_full_pipeline.sh
#   ./scripts/train_full_pipeline.sh
#
# Что делает (последовательно):
#   1. Pre-flight проверки (через pipeline_orchestrator.py preflight)
#   2. Собирает датасет (ru_metadata_dataset_builder.py) — если нужно
#   3. Обучает leak-free 3-class KD модель (train_kd_distillation.py --leak-free)
#   4. Обучает бинарную модель с Platt калибровкой (train_binary_model.py)
#   5. Прогоняет обе модели через eval gate (eval_golden_set.py)
#   6. Печатает summary с рекомендациями по промоушену
#
# Зависимости:
#   pip install tensorflow catboost scikit-learn numpy
#
# Данные:
#   datasets/ru/processed/ru_tflite_features.csv должен существовать
#   (сгенерируй через ru_metadata_dataset_builder.py или скачай из Drive)
# ═══════════════════════════════════════════════════════════════════════════

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
ASSETS_DIR="$REPO_ROOT/app/src/main/assets"
EXPERIMENTAL="$ASSETS_DIR/experimental"
DATA="$REPO_ROOT/datasets/ru/processed/ru_tflite_features.csv"
EVAL_CSV="$REPO_ROOT/datasets/ru/eval/cold_eval_600.csv"

# ─── Defaults (match pipeline_orchestrator.py DEFAULT_THRESHOLDS / DEFAULT_SEED) ──
SEED=42
MIN_BLOCK_PRECISION=0.85
MIN_BLOCK_RECALL=0.55
MAX_ALLOW_FP_RATE=0.20
SKIP_BINARY=false
SKIP_EVAL=false
FORCE_REBUILD_DATASET=false

# ─── Parse CLI flags (subtask 5.1) ──────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            SEED="$2"; shift 2 ;;
        --min-block-precision)
            MIN_BLOCK_PRECISION="$2"; shift 2 ;;
        --min-block-recall)
            MIN_BLOCK_RECALL="$2"; shift 2 ;;
        --max-allow-fp-rate)
            MAX_ALLOW_FP_RATE="$2"; shift 2 ;;
        --skip-binary)
            SKIP_BINARY=true; shift ;;
        --skip-eval)
            SKIP_EVAL=true; shift ;;
        --force-rebuild-dataset)
            FORCE_REBUILD_DATASET=true; shift ;;
        *)
            echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

# ─── Manifest path (set after manifest-init, used by trap) ───────────────
MANIFEST=""
FINAL_EXIT=0

# ─── Trap: manifest-finalize on any exit (subtask 5.5) ───────────────────
cleanup() {
    if [ -n "$MANIFEST" ] && [ -f "$MANIFEST" ]; then
        python3 "$SCRIPTS_DIR/pipeline_orchestrator.py" manifest-finalize \
            --manifest "$MANIFEST" \
            --exit-code "$FINAL_EXIT" || true
    fi
}
trap cleanup EXIT

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║       FULL TRAINING PIPELINE (leak-free + binary)            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Data:    $DATA"
echo "Eval:    $EVAL_CSV"
echo "Assets:  $EXPERIMENTAL"
echo "Seed:    $SEED"
echo ""

# ─── Pre-flight via pipeline_orchestrator.py (subtask 5.2) ───────────────
echo "─── Pre-flight checks ───────────────────────────────────────────"

if [ "$FORCE_REBUILD_DATASET" = true ]; then
    echo "Force rebuild requested — running dataset builder first..."
    python3 "$SCRIPTS_DIR/ru_metadata_dataset_builder.py" || {
        echo "ERROR: Dataset builder failed" >&2
        FINAL_EXIT=2
        exit 2
    }
fi

set +e
python3 "$SCRIPTS_DIR/pipeline_orchestrator.py" preflight \
    --dataset-path "$DATA" \
    --eval-csv-path "$EVAL_CSV"
PREFLIGHT_EXIT=$?
set -e

if [ $PREFLIGHT_EXIT -eq 10 ]; then
    echo "Dataset not found — running dataset builder..."
    python3 "$SCRIPTS_DIR/ru_metadata_dataset_builder.py" || {
        echo "ERROR: Dataset builder failed" >&2
        FINAL_EXIT=2
        exit 2
    }
    # Retry preflight after building dataset
    set +e
    python3 "$SCRIPTS_DIR/pipeline_orchestrator.py" preflight \
        --dataset-path "$DATA" \
        --eval-csv-path "$EVAL_CSV"
    PREFLIGHT_EXIT=$?
    set -e
    if [ $PREFLIGHT_EXIT -ne 0 ]; then
        echo "ERROR: Pre-flight failed after dataset rebuild (exit=$PREFLIGHT_EXIT)" >&2
        FINAL_EXIT=$PREFLIGHT_EXIT
        exit $PREFLIGHT_EXIT
    fi
elif [ $PREFLIGHT_EXIT -eq 2 ]; then
    echo "ERROR: Pre-flight checks failed (exit=2)" >&2
    FINAL_EXIT=2
    exit 2
elif [ $PREFLIGHT_EXIT -ne 0 ]; then
    echo "ERROR: Pre-flight checks failed (exit=$PREFLIGHT_EXIT)" >&2
    FINAL_EXIT=$PREFLIGHT_EXIT
    exit $PREFLIGHT_EXIT
fi

echo "✓ Pre-flight passed"
echo ""

# ─── Initialize Run_Manifest ─────────────────────────────────────────────
MANIFEST=$(python3 "$SCRIPTS_DIR/pipeline_orchestrator.py" manifest-init \
    --seed "$SEED" \
    --dataset-path "$DATA" \
    --eval-csv-path "$EVAL_CSV")

if [ -z "$MANIFEST" ] || [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest-init failed to create manifest" >&2
    FINAL_EXIT=2
    exit 2
fi

echo "Manifest: $MANIFEST"
echo ""

# ─── Step 1: Leak-free 3-class KD model (subtask 5.3) ───────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1/4: Training LEAK-FREE 3-class KD model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

STEP_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)

set +e
python3 "$SCRIPTS_DIR/train_kd_distillation.py" \
    --data "$DATA" \
    --leak-free \
    --optuna-trials 0 \
    --student-epochs 120 \
    --student-patience 15 \
    --student-batch 128 \
    --hidden-sizes "128,96,48" \
    --min-block-precision 0.90 \
    --warn-class-weight 5.0 \
    --block-class-weight 1.2 \
    --allow-class-weight 1.0 \
    --weight-decay 5e-4 \
    --label-smoothing 0.08 \
    --seed "$SEED"
KD_EXIT=$?
set -e

STEP_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python3 "$SCRIPTS_DIR/pipeline_orchestrator.py" manifest-step \
    --manifest "$MANIFEST" \
    --name "train-leak-free" \
    --started-at "$STEP_START" \
    --finished-at "$STEP_END" \
    --exit-code "$KD_EXIT" \
    --artifact "$EXPERIMENTAL/spam_model_leak_free.tflite" \
    --artifact "$EXPERIMENTAL/model_card_leak_free.json"

if [ $KD_EXIT -ne 0 ]; then
    echo "ERROR: Leak-free KD training failed (exit=$KD_EXIT)" >&2
    FINAL_EXIT=$KD_EXIT
    exit $KD_EXIT
fi

echo ""
echo "✓ Leak-free KD model done"
echo "  → $EXPERIMENTAL/spam_model_leak_free.tflite"
echo ""

# ─── Step 2: Binary model + Platt calibration (subtask 5.3, 5.4) ────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2/4: Training BINARY model + Platt calibration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$SKIP_BINARY" = true ]; then
    echo "⚠ Skipped (--skip-binary)"
    STEP_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    STEP_END="$STEP_START"
    python3 "$SCRIPTS_DIR/pipeline_orchestrator.py" manifest-step \
        --manifest "$MANIFEST" \
        --name "train-binary" \
        --started-at "$STEP_START" \
        --finished-at "$STEP_END" \
        --exit-code 0 \
        --skipped \
        --skipped-reason "user passed --skip-binary"
    BIN_EXIT=0
    BIN_SKIPPED=true
else
    STEP_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    set +e
    python3 "$SCRIPTS_DIR/train_binary_model.py" \
        --data "$DATA" \
        --binary-warn-strategy merge_block \
        --hidden-sizes "128,96,48" \
        --dropout 0.15 \
        --l2 5e-4 \
        --epochs 120 \
        --batch 128 \
        --patience 12 \
        --lr 8e-4 \
        --min-block-precision 0.90 \
        --seed "$SEED"
    BIN_EXIT=$?
    set -e

    STEP_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    python3 "$SCRIPTS_DIR/pipeline_orchestrator.py" manifest-step \
        --manifest "$MANIFEST" \
        --name "train-binary" \
        --started-at "$STEP_START" \
        --finished-at "$STEP_END" \
        --exit-code "$BIN_EXIT" \
        --artifact "$EXPERIMENTAL/spam_model_binary.tflite" \
        --artifact "$EXPERIMENTAL/model_card_binary.json"

    BIN_SKIPPED=false

    if [ $BIN_EXIT -ne 0 ]; then
        echo "⚠ Binary model training failed (exit=$BIN_EXIT) — continuing with eval for leak-free only"
    else
        echo ""
        echo "✓ Binary model done"
        echo "  → $EXPERIMENTAL/spam_model_binary.tflite"
    fi
fi
echo ""

# ─── Step 3: Eval gate — leak-free 3-class (subtask 5.3, 5.4) ───────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3/4: Eval gate — leak-free 3-class on cold_eval_600"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$SKIP_EVAL" = true ]; then
    echo "⚠ Skipped (--skip-eval)"
    STEP_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    STEP_END="$STEP_START"
    python3 "$SCRIPTS_DIR/pipeline_orchestrator.py" manifest-step \
        --manifest "$MANIFEST" \
        --name "eval-leak-free" \
        --started-at "$STEP_START" \
        --finished-at "$STEP_END" \
        --exit-code 0 \
        --skipped \
        --skipped-reason "user passed --skip-eval"
    EVAL_RESULT_KD=0
else
    STEP_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    set +e
    python3 "$SCRIPTS_DIR/eval_golden_set.py" \
        --model "$EXPERIMENTAL/spam_model_leak_free.tflite" \
        --card "$EXPERIMENTAL/model_card_leak_free.json" \
        --golden "$EVAL_CSV" \
        --cold \
        --min-block-precision "$MIN_BLOCK_PRECISION" \
        --min-block-recall "$MIN_BLOCK_RECALL" \
        --max-allow-fp-rate "$MAX_ALLOW_FP_RATE" \
        --output-json "$EXPERIMENTAL/eval_leak_free.json"
    EVAL_RESULT_KD=$?
    set -e

    STEP_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    if [ $EVAL_RESULT_KD -eq 2 ]; then
        # I/O error — abort pipeline
        python3 "$SCRIPTS_DIR/pipeline_orchestrator.py" manifest-step \
            --manifest "$MANIFEST" \
            --name "eval-leak-free" \
            --started-at "$STEP_START" \
            --finished-at "$STEP_END" \
            --exit-code "$EVAL_RESULT_KD" \
            --artifact "$EXPERIMENTAL/eval_leak_free.json"
        echo "ERROR: Eval gate I/O error for leak-free (exit=2) — aborting" >&2
        FINAL_EXIT=2
        exit 2
    elif [ $EVAL_RESULT_KD -eq 1 ]; then
        # Gate failed — continue
        python3 "$SCRIPTS_DIR/pipeline_orchestrator.py" manifest-step \
            --manifest "$MANIFEST" \
            --name "eval-leak-free" \
            --started-at "$STEP_START" \
            --finished-at "$STEP_END" \
            --exit-code "$EVAL_RESULT_KD" \
            --gate-failed \
            --artifact "$EXPERIMENTAL/eval_leak_free.json"
        echo "✗ Leak-free model FAILED eval gate (exit=1) — continuing"
    else
        python3 "$SCRIPTS_DIR/pipeline_orchestrator.py" manifest-step \
            --manifest "$MANIFEST" \
            --name "eval-leak-free" \
            --started-at "$STEP_START" \
            --finished-at "$STEP_END" \
            --exit-code "$EVAL_RESULT_KD" \
            --artifact "$EXPERIMENTAL/eval_leak_free.json"
        echo "✓ Leak-free model PASSED eval gate"
    fi
fi
echo ""

# ─── Step 4: Eval gate — binary (subtask 5.3, 5.4) ──────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4/4: Eval gate — binary on cold_eval_600"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$SKIP_EVAL" = true ] || [ "$SKIP_BINARY" = true ]; then
    SKIP_REASON=""
    if [ "$SKIP_EVAL" = true ]; then
        SKIP_REASON="user passed --skip-eval"
    elif [ "$SKIP_BINARY" = true ]; then
        SKIP_REASON="user passed --skip-binary"
    fi
    echo "⚠ Skipped ($SKIP_REASON)"
    STEP_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    STEP_END="$STEP_START"
    python3 "$SCRIPTS_DIR/pipeline_orchestrator.py" manifest-step \
        --manifest "$MANIFEST" \
        --name "eval-binary" \
        --started-at "$STEP_START" \
        --finished-at "$STEP_END" \
        --exit-code 0 \
        --skipped \
        --skipped-reason "$SKIP_REASON"
    EVAL_RESULT_BIN=0
elif [ $BIN_EXIT -ne 0 ]; then
    echo "⚠ Skipped (binary training failed)"
    STEP_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    STEP_END="$STEP_START"
    python3 "$SCRIPTS_DIR/pipeline_orchestrator.py" manifest-step \
        --manifest "$MANIFEST" \
        --name "eval-binary" \
        --started-at "$STEP_START" \
        --finished-at "$STEP_END" \
        --exit-code 0 \
        --skipped \
        --skipped-reason "binary training failed with exit $BIN_EXIT"
    EVAL_RESULT_BIN=0
else
    STEP_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    set +e
    python3 "$SCRIPTS_DIR/eval_golden_set.py" \
        --model "$EXPERIMENTAL/spam_model_binary.tflite" \
        --card "$EXPERIMENTAL/model_card_binary.json" \
        --golden "$EVAL_CSV" \
        --cold \
        --min-block-precision "$MIN_BLOCK_PRECISION" \
        --min-block-recall "$MIN_BLOCK_RECALL" \
        --max-allow-fp-rate "$MAX_ALLOW_FP_RATE" \
        --output-json "$EXPERIMENTAL/eval_binary.json"
    EVAL_RESULT_BIN=$?
    set -e

    STEP_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    if [ $EVAL_RESULT_BIN -eq 2 ]; then
        # I/O error — abort pipeline
        python3 "$SCRIPTS_DIR/pipeline_orchestrator.py" manifest-step \
            --manifest "$MANIFEST" \
            --name "eval-binary" \
            --started-at "$STEP_START" \
            --finished-at "$STEP_END" \
            --exit-code "$EVAL_RESULT_BIN" \
            --artifact "$EXPERIMENTAL/eval_binary.json"
        echo "ERROR: Eval gate I/O error for binary (exit=2) — aborting" >&2
        FINAL_EXIT=2
        exit 2
    elif [ $EVAL_RESULT_BIN -eq 1 ]; then
        # Gate failed — continue
        python3 "$SCRIPTS_DIR/pipeline_orchestrator.py" manifest-step \
            --manifest "$MANIFEST" \
            --name "eval-binary" \
            --started-at "$STEP_START" \
            --finished-at "$STEP_END" \
            --exit-code "$EVAL_RESULT_BIN" \
            --gate-failed \
            --artifact "$EXPERIMENTAL/eval_binary.json"
        echo "✗ Binary model FAILED eval gate (exit=1) — continuing"
    else
        python3 "$SCRIPTS_DIR/pipeline_orchestrator.py" manifest-step \
            --manifest "$MANIFEST" \
            --name "eval-binary" \
            --started-at "$STEP_START" \
            --finished-at "$STEP_END" \
            --exit-code "$EVAL_RESULT_BIN" \
            --artifact "$EXPERIMENTAL/eval_binary.json"
        echo "✓ Binary model PASSED eval gate"
    fi
fi
echo ""

# ─── Summary (subtask 5.5) ───────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo ""

python3 "$SCRIPTS_DIR/pipeline_orchestrator.py" summary \
    --manifest "$MANIFEST" \
    --experimental-dir "$EXPERIMENTAL" \
    --prod-model-card "$ASSETS_DIR/model_card.json" || true

echo ""
echo "Done."
