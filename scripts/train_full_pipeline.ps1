<#
.SYNOPSIS
    Full training pipeline for antispam TFLite models (leak-free + binary).

.DESCRIPTION
    PowerShell equivalent of scripts/train_full_pipeline.sh.
    Orchestrates: preflight -> dataset build -> train leak-free -> train binary -> eval x2 -> summary.
    All trained models go to app/src/main/assets/experimental/ — production artifacts are never touched.

.EXAMPLE
    .\scripts\train_full_pipeline.ps1
    .\scripts\train_full_pipeline.ps1 -SkipBinary -Seed 123
    .\scripts\train_full_pipeline.ps1 -SkipEval -ForceRebuildDataset
#>

param(
    [int]$Seed = 42,
    [double]$MinBlockPrecision = 0.85,
    [double]$MinBlockRecall = 0.55,
    [double]$MaxAllowFpRate = 0.20,
    [switch]$SkipBinary,
    [switch]$SkipEval,
    [switch]$ForceRebuildDataset,
    [string]$PythonExe = ''
)

$ErrorActionPreference = 'Stop'

# ─── Resolve paths ────────────────────────────────────────────────────────
$ScriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptsDir
$AssetsDir = Join-Path $RepoRoot 'app\src\main\assets'
$Experimental = Join-Path $AssetsDir 'experimental'
$Data = Join-Path $RepoRoot 'datasets\ru\processed\ru_tflite_features.csv'
$EvalCsv = Join-Path $RepoRoot 'datasets\ru\eval\cold_eval_600.csv'

# ─── Resolve Python executable (subtask 6.1) ─────────────────────────────
# Priority: param value -> hardcoded path -> python from PATH
if (-not $PythonExe) {
    $hardcoded = 'C:\Users\Redmi\AppData\Local\Programs\Python\Python312\python.exe'
    if (Test-Path $hardcoded) {
        $PythonExe = $hardcoded
    } else {
        $PythonExe = 'python'
    }
}

# ─── Helper: get UTC timestamp ────────────────────────────────────────────
function Get-UtcTimestamp {
    (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
}

# ─── Banner ───────────────────────────────────────────────────────────────
Write-Host "================================================================="
Write-Host "       FULL TRAINING PIPELINE (leak-free + binary)               "
Write-Host "================================================================="
Write-Host ""
Write-Host "Python:  $PythonExe"
Write-Host "Data:    $Data"
Write-Host "Eval:    $EvalCsv"
Write-Host "Assets:  $Experimental"
Write-Host ""

# ─── Preflight (subtask 6.2) ─────────────────────────────────────────────
Write-Host "--- Preflight checks ---"
& $PythonExe "$ScriptsDir\pipeline_orchestrator.py" preflight
$preflightExit = $LASTEXITCODE

if ($preflightExit -eq 10) {
    # Dataset missing — run builder then retry preflight
    Write-Host "Dataset not found, running dataset builder..."
    & $PythonExe "$ScriptsDir\ru_metadata_dataset_builder.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Dataset builder failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    & $PythonExe "$ScriptsDir\pipeline_orchestrator.py" preflight
    $preflightExit = $LASTEXITCODE
}

if ($preflightExit -ne 0) {
    Write-Host "ERROR: Preflight failed with exit code $preflightExit"
    exit $preflightExit
}

Write-Host "Preflight passed."
Write-Host ""

# ─── Manifest init ────────────────────────────────────────────────────────
$ManifestPath = ''
$FinalExitCode = 0

try {
    $ManifestPath = (& $PythonExe "$ScriptsDir\pipeline_orchestrator.py" manifest-init --seed $Seed) | Select-Object -Last 1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: manifest-init failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    $ManifestPath = $ManifestPath.Trim()
    Write-Host "Manifest: $ManifestPath"
    Write-Host ""

    # ─── ForceRebuildDataset (subtask 6.4) ────────────────────────────────
    if ($ForceRebuildDataset) {
        $stepStart = Get-UtcTimestamp
        Write-Host "--- Force rebuilding dataset ---"
        & $PythonExe "$ScriptsDir\ru_metadata_dataset_builder.py"
        $buildExit = $LASTEXITCODE
        $stepEnd = Get-UtcTimestamp
        & $PythonExe "$ScriptsDir\pipeline_orchestrator.py" manifest-step `
            --manifest $ManifestPath `
            --name "dataset-build" `
            --started-at $stepStart `
            --finished-at $stepEnd `
            --exit-code $buildExit `
            --artifact "datasets/ru/processed/ru_tflite_features.csv"
        if ($buildExit -ne 0) {
            Write-Host "ERROR: Dataset builder failed with exit code $buildExit"
            $FinalExitCode = $buildExit
            exit $buildExit
        }
        Write-Host ""
    }

    # ─── Step 1: Train leak-free KD model (subtask 6.3) ──────────────────
    Write-Host "================================================================="
    Write-Host "Step 1: Training LEAK-FREE 3-class KD model"
    Write-Host "================================================================="
    Write-Host ""

    $stepStart = Get-UtcTimestamp
    & $PythonExe "$ScriptsDir\train_kd_distillation.py" `
        --data $Data `
        --leak-free `
        --optuna-trials 0 `
        --student-epochs 120 `
        --student-patience 15 `
        --student-batch 128 `
        --hidden-sizes "128,96,48" `
        --min-block-precision 0.90 `
        --warn-class-weight 12.0 `
        --block-class-weight 1.0 `
        --allow-class-weight 10.0 `
        --weight-decay 5e-4 `
        --label-smoothing 0.08 `
        --seed $Seed
    $kdExit = $LASTEXITCODE
    $stepEnd = Get-UtcTimestamp

    & $PythonExe "$ScriptsDir\pipeline_orchestrator.py" manifest-step `
        --manifest $ManifestPath `
        --name "train-leak-free" `
        --started-at $stepStart `
        --finished-at $stepEnd `
        --exit-code $kdExit `
        --artifact "app/src/main/assets/experimental/spam_model_leak_free.tflite" `
        --artifact "app/src/main/assets/experimental/model_card_leak_free.json"

    if ($kdExit -ne 0) {
        Write-Host "ERROR: Leak-free training failed with exit code $kdExit"
        $FinalExitCode = $kdExit
        exit $kdExit
    }
    Write-Host "Leak-free KD model done."
    Write-Host ""

    # ─── Step 2: Train binary model (subtask 6.3, 6.4) ───────────────────
    if ($SkipBinary) {
        Write-Host "--- Skipping binary training (-SkipBinary) ---"
        $stepStart = Get-UtcTimestamp
        & $PythonExe "$ScriptsDir\pipeline_orchestrator.py" manifest-step `
            --manifest $ManifestPath `
            --name "train-binary" `
            --started-at $stepStart `
            --finished-at $stepStart `
            --exit-code 0 `
            --skipped `
            --skipped-reason "SkipBinary flag set"
        Write-Host ""
    } else {
        Write-Host "================================================================="
        Write-Host "Step 2: Training BINARY model + Platt calibration"
        Write-Host "================================================================="
        Write-Host ""

        $stepStart = Get-UtcTimestamp
        & $PythonExe "$ScriptsDir\train_binary_model.py" `
            --data $Data `
            --binary-warn-strategy merge_block `
            --hidden-sizes "128,96,48" `
            --dropout 0.15 `
            --l2 5e-4 `
            --epochs 120 `
            --batch 128 `
            --patience 12 `
            --lr 8e-4 `
            --min-block-precision 0.90 `
            --seed $Seed
        $binExit = $LASTEXITCODE
        $stepEnd = Get-UtcTimestamp

        & $PythonExe "$ScriptsDir\pipeline_orchestrator.py" manifest-step `
            --manifest $ManifestPath `
            --name "train-binary" `
            --started-at $stepStart `
            --finished-at $stepEnd `
            --exit-code $binExit `
            --artifact "app/src/main/assets/experimental/spam_model_binary.tflite" `
            --artifact "app/src/main/assets/experimental/model_card_binary.json"

        if ($binExit -ne 0) {
            Write-Host "WARNING: Binary training failed with exit code $binExit (continuing for leak-free eval)"
            $FinalExitCode = $binExit
        } else {
            Write-Host "Binary model done."
        }
        Write-Host ""
    }

    # ─── Step 3: Eval gate — leak-free (subtask 6.3, 6.4) ────────────────
    if ($SkipEval) {
        Write-Host "--- Skipping eval (-SkipEval) ---"
        $stepStart = Get-UtcTimestamp
        & $PythonExe "$ScriptsDir\pipeline_orchestrator.py" manifest-step `
            --manifest $ManifestPath `
            --name "eval-leak-free" `
            --started-at $stepStart `
            --finished-at $stepStart `
            --exit-code 0 `
            --skipped `
            --skipped-reason "SkipEval flag set"
        & $PythonExe "$ScriptsDir\pipeline_orchestrator.py" manifest-step `
            --manifest $ManifestPath `
            --name "eval-binary" `
            --started-at $stepStart `
            --finished-at $stepStart `
            --exit-code 0 `
            --skipped `
            --skipped-reason "SkipEval flag set"
        Write-Host ""
    } else {
        Write-Host "================================================================="
        Write-Host "Step 3: Eval gate - leak-free 3-class on cold_eval_600"
        Write-Host "================================================================="
        Write-Host ""

        $stepStart = Get-UtcTimestamp
        & $PythonExe "$ScriptsDir\eval_golden_set.py" `
            --model "$Experimental\spam_model_leak_free.tflite" `
            --card "$Experimental\model_card_leak_free.json" `
            --golden $EvalCsv `
            --cold `
            --min-block-precision $MinBlockPrecision `
            --min-block-recall $MinBlockRecall `
            --max-allow-fp-rate $MaxAllowFpRate `
            --output-json "$Experimental\eval_leak_free.json"
        $evalKdExit = $LASTEXITCODE
        $stepEnd = Get-UtcTimestamp

        $gateFailed = $null
        if ($evalKdExit -eq 1) {
            $gateFailed = '--gate-failed'
            Write-Host "Leak-free model FAILED eval gate (exit=1)"
        } elseif ($evalKdExit -eq 2) {
            # I/O or parse error — abort
            & $PythonExe "$ScriptsDir\pipeline_orchestrator.py" manifest-step `
                --manifest $ManifestPath `
                --name "eval-leak-free" `
                --started-at $stepStart `
                --finished-at $stepEnd `
                --exit-code $evalKdExit `
                --artifact "app/src/main/assets/experimental/eval_leak_free.json"
            Write-Host "ERROR: Eval gate (leak-free) I/O error, aborting (exit=$evalKdExit)"
            $FinalExitCode = $evalKdExit
            exit $evalKdExit
        } else {
            Write-Host "Leak-free model PASSED eval gate"
        }

        # Record manifest step
        $manifestStepArgs = @(
            "$ScriptsDir\pipeline_orchestrator.py", 'manifest-step',
            '--manifest', $ManifestPath,
            '--name', 'eval-leak-free',
            '--started-at', $stepStart,
            '--finished-at', $stepEnd,
            '--exit-code', $evalKdExit,
            '--artifact', 'app/src/main/assets/experimental/eval_leak_free.json'
        )
        if ($gateFailed) { $manifestStepArgs += $gateFailed }
        & $PythonExe @manifestStepArgs
        Write-Host ""

        # ─── Step 4: Eval gate — binary (subtask 6.3, 6.4) ───────────────
        if ($SkipBinary -or ($binExit -and $binExit -ne 0)) {
            Write-Host "--- Skipping binary eval (binary training skipped or failed) ---"
            $stepStart = Get-UtcTimestamp
            & $PythonExe "$ScriptsDir\pipeline_orchestrator.py" manifest-step `
                --manifest $ManifestPath `
                --name "eval-binary" `
                --started-at $stepStart `
                --finished-at $stepStart `
                --exit-code 0 `
                --skipped `
                --skipped-reason "Binary training was skipped or failed"
            Write-Host ""
        } else {
            Write-Host "================================================================="
            Write-Host "Step 4: Eval gate - binary on cold_eval_600"
            Write-Host "================================================================="
            Write-Host ""

            $stepStart = Get-UtcTimestamp
            & $PythonExe "$ScriptsDir\eval_golden_set.py" `
                --model "$Experimental\spam_model_binary.tflite" `
                --card "$Experimental\model_card_binary.json" `
                --golden $EvalCsv `
                --cold `
                --min-block-precision $MinBlockPrecision `
                --min-block-recall $MinBlockRecall `
                --max-allow-fp-rate $MaxAllowFpRate `
                --output-json "$Experimental\eval_binary.json"
            $evalBinExit = $LASTEXITCODE
            $stepEnd = Get-UtcTimestamp

            $gateFailed = $null
            if ($evalBinExit -eq 1) {
                $gateFailed = '--gate-failed'
                Write-Host "Binary model FAILED eval gate (exit=1)"
            } elseif ($evalBinExit -eq 2) {
                & $PythonExe "$ScriptsDir\pipeline_orchestrator.py" manifest-step `
                    --manifest $ManifestPath `
                    --name "eval-binary" `
                    --started-at $stepStart `
                    --finished-at $stepEnd `
                    --exit-code $evalBinExit `
                    --artifact "app/src/main/assets/experimental/eval_binary.json"
                Write-Host "ERROR: Eval gate (binary) I/O error, aborting (exit=$evalBinExit)"
                $FinalExitCode = $evalBinExit
                exit $evalBinExit
            } else {
                Write-Host "Binary model PASSED eval gate"
            }

            $manifestStepArgs = @(
                "$ScriptsDir\pipeline_orchestrator.py", 'manifest-step',
                '--manifest', $ManifestPath,
                '--name', 'eval-binary',
                '--started-at', $stepStart,
                '--finished-at', $stepEnd,
                '--exit-code', $evalBinExit,
                '--artifact', 'app/src/main/assets/experimental/eval_binary.json'
            )
            if ($gateFailed) { $manifestStepArgs += $gateFailed }
            & $PythonExe @manifestStepArgs
            Write-Host ""
        }
    }

    # ─── Summary ──────────────────────────────────────────────────────────
    Write-Host "================================================================="
    Write-Host "SUMMARY"
    Write-Host "================================================================="
    Write-Host ""
    & $PythonExe "$ScriptsDir\pipeline_orchestrator.py" summary `
        --manifest $ManifestPath `
        --experimental-dir $Experimental `
        --prod-model-card "$AssetsDir\model_card.json"
    Write-Host ""

} finally {
    # ─── Manifest finalize (subtask 6.5) ──────────────────────────────────
    # Always finalize the manifest regardless of success or failure.
    if ($ManifestPath -and (Test-Path $ManifestPath)) {
        & $PythonExe "$ScriptsDir\pipeline_orchestrator.py" manifest-finalize `
            --manifest $ManifestPath `
            --exit-code $FinalExitCode
    }
}

exit $FinalExitCode
