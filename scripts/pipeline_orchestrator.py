"""Pipeline orchestrator for the antispam model training pipeline.

Тонкое Python-ядро для `scripts/train_full_pipeline.sh` и
`scripts/train_full_pipeline.ps1`. Содержит общие pre-flight проверки,
управление Run_Manifest и финальный summary.

На текущей итерации (task 1) реализован только каркас:
константы из дизайна и argparse с подкомандами-заглушками.
Каждая подкоманда возвращает фиксированный exit code 99.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import importlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
from typing import NoReturn, Optional

# ---------------------------------------------------------------------------
# Конфигурационные дефолты (Component 4 of design.md)
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "datasets" / "ru" / "processed" / "ru_tflite_features.csv"
EVAL_CSV_PATH = REPO_ROOT / "datasets" / "ru" / "eval" / "cold_eval_600.csv"
EXPERIMENTAL_DIR = REPO_ROOT / "app" / "src" / "main" / "assets" / "experimental"
PROD_MODEL_CARD = REPO_ROOT / "app" / "src" / "main" / "assets" / "model_card.json"
REPORTS_DIR = REPO_ROOT / "datasets" / "ru" / "reports" / "training"

DEFAULT_THRESHOLDS = {
    # Источник дефолтов: текущий scripts/train_full_pipeline.sh (cold-eval baseline).
    # Не путать с eval_golden_set.py CLI-дефолтами (0.85 / 0.55 / 0.15) и со «строгим»
    # вариантом из шапки скрипта (0.90 / 0.60 / 0.10). Прод model_card.json показывает
    # cold-метрики ≈ block_precision 0.95 / block_recall 0.69 / allow_fp 0.16, поэтому
    # baseline сделан мягче по recall и FP — иначе даже текущая prod-модель не проходит.
    # Для строгого регрессионного gate используются override-флаги (Req 5.3, 8.5).
    "min_block_precision": 0.85,
    "min_block_recall": 0.55,
    "max_allow_fp_rate": 0.20,
}

DEFAULT_SEED = 42
REQUIRED_PACKAGES = ("tensorflow", "catboost", "sklearn", "numpy")
MIN_PYTHON = (3, 10)
LEAK_FREE_FORBIDDEN_FEATURES = (
    "reputationScore",
    "sourceConfidence",
    "reviewsLog",
    "negativeRatio",
    "searchVolumeLog",
    "hasFraudCategory",
    "hasTelemarketingCategory",
    "inAllowlist",
    "inBlacklist",
)

# Exit code, возвращаемый всеми подкомандами-заглушками до фактической имплементации.
NOT_IMPLEMENTED_EXIT_CODE = 99


# ---------------------------------------------------------------------------
# Subcommand stubs
# ---------------------------------------------------------------------------


def _not_implemented(subcommand: str) -> int:
    """Печатает сообщение и возвращает фиксированный exit code 99."""
    sys.stderr.write(
        f"pipeline_orchestrator: subcommand '{subcommand}' is not implemented yet\n"
    )
    return NOT_IMPLEMENTED_EXIT_CODE


def cmd_preflight(args: argparse.Namespace) -> int:
    """Validate runtime environment for the training pipeline.

    Implemented checks (subtasks 2.1, 2.3, 2.5):
      * Python interpreter version >= MIN_PYTHON.
      * All REQUIRED_PACKAGES importable via importlib.
      * Dataset CSV exists (exit 10 if missing) and has >=1 data row
        (exit 2 if header-only or completely empty).
      * Eval CSV exists and has >=100 data rows (exit 2 otherwise).

    On the first detected failure the function prints a single ASCII-safe
    line to stderr and returns the appropriate exit code. The function
    returns the exit code instead of calling sys.exit() so that main()
    retains control of process termination semantics. When all checks
    pass the function returns 0.
    """
    # --- Python version check (Requirements 1.5, 2.1) ---------------------
    version = sys.version_info
    if (version.major, version.minor) < MIN_PYTHON:
        sys.stderr.write(
            "preflight: Python {req_major}.{req_minor}+ is required, "
            "found {cur_major}.{cur_minor}.{cur_micro}\n".format(
                req_major=MIN_PYTHON[0],
                req_minor=MIN_PYTHON[1],
                cur_major=version.major,
                cur_minor=version.minor,
                cur_micro=version.micro,
            )
        )
        return 2

    # --- Required packages check (Requirements 2.1, 2.2) ------------------
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            sys.stderr.write(
                "preflight: missing package: {pkg}. "
                "Install via: pip install tensorflow catboost scikit-learn numpy\n".format(
                    pkg=package,
                )
            )
            return 2

    # --- Dataset existence + non-emptiness check (Requirements 2.3, 2.4, 2.5) ---
    # Контракт обёртки: exit 10 означает «датасета нет, нужно собрать» — обёртка
    # вызовет ru_metadata_dataset_builder.py и повторит preflight. Exit 2 —
    # фатально (после повторного preflight файл всё ещё пустой/отсутствует).
    dataset_path: pathlib.Path = args.dataset_path
    if not dataset_path.is_file():
        sys.stderr.write(
            "preflight: dataset not found at {path}; run ru_metadata_dataset_builder.py\n".format(
                path=dataset_path,
            )
        )
        return 10

    # Стримово считаем data-строки (без заголовка), чтобы не грузить весь CSV в память.
    data_rows = 0
    with dataset_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        # Header пропускаем; пустой файл (нет даже заголовка) тоже трактуем как 0 строк.
        try:
            next(reader)
        except StopIteration:
            pass
        else:
            for _ in reader:
                data_rows += 1

    if data_rows == 0:
        sys.stderr.write(
            "preflight: dataset {path} is empty; required raw inputs: "
            "ru_call_features.csv, ru_numbers_labeled.csv, ru_reputation_raw.csv\n".format(
                path=dataset_path,
            )
        )
        return 2

    # --- Eval CSV existence + minimum row count check (Requirements 2.6, 2.7) ---
    # Cold_Eval_Set обязан содержать >=100 data-строк, иначе eval gate был бы
    # статистически бессмысленным. Любое нарушение — фатально (exit 2): тихий
    # пропуск шагов 3 и 4 запрещён требованием 2.7.
    eval_csv_path: pathlib.Path = args.eval_csv_path
    eval_data_rows = 0
    if eval_csv_path.is_file():
        with eval_csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            try:
                next(reader)
            except StopIteration:
                pass
            else:
                # Считаем data-строки и выходим на >=100 ради ранней остановки
                # на больших eval-наборах.
                for _ in reader:
                    eval_data_rows += 1
                    if eval_data_rows >= 100:
                        break

    if eval_data_rows < 100:
        sys.stderr.write(
            "preflight: eval CSV at {path} is missing or has fewer than 100 data rows; "
            "expected datasets/ru/eval/cold_eval_600.csv with >=100 rows\n".format(
                path=eval_csv_path,
            )
        )
        return 2

    return 0


def _sha256_file(path: pathlib.Path) -> str:
    """Compute SHA-256 of `path` reading 64 KB chunks (stream-friendly).

    Avoids loading the entire CSV into memory — Run_Manifest нужен и для больших
    датасетов, и читать всё в память только ради хэша было бы расточительно.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_data_rows(path: pathlib.Path) -> int:
    """Count CSV data rows (excluding header), like cmd_preflight does."""
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            next(reader)
        except StopIteration:
            return 0
        for _ in reader:
            rows += 1
    return rows


def _atomic_write_json(path: pathlib.Path, data: dict) -> None:
    """Write `data` as JSON to `path` atomically via temp file + ``os.replace``.

    Делится между ``cmd_manifest_step`` и ``cmd_manifest_finalize``: оба шага
    обязаны не оставлять полу-записанный манифест, если процесс прервут
    в середине ``json.dump``. Алгоритм:

      1. Создать sibling temp file в `path.parent` (та же FS — это ключ к
         атомарности ``os.replace``: cross-device rename атомарным не будет).
      2. Записать туда JSON в utf-8 c indent=2 (консистентно с manifest-init).
      3. ``os.replace(tmp, path)`` — atomic поверх API ОС (POSIX rename(2),
         Windows MoveFileExW с ``MOVEFILE_REPLACE_EXISTING``); работает с
         Python 3.3+.
      4. При любой OSError — подчистить temp file и пробросить наверх:
         caller форматирует диагностическое сообщение с контекстом
         (``manifest-step:``/``manifest-finalize:``) и возвращает свой
         exit code.

    ``tempfile.NamedTemporaryFile(delete=False)`` гарантирует уникальное
    имя с PID/random-суффиксом — защищает от коллизий с гипотетическими
    параллельными писателями (хотя контракт пайплайна — sequential).
    """
    tmp_path: Optional[pathlib.Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix="manifest_",
            suffix=".json.tmp",
            delete=False,
            mode="w",
            encoding="utf-8",
        ) as tmp_fh:
            tmp_path = pathlib.Path(tmp_fh.name)
            json.dump(data, tmp_fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except OSError:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _git_head_sha(repo_root: pathlib.Path) -> str:
    """Return current git HEAD SHA, or "unknown" on any failure.

    Failure modes treated identically (Req 7.4): non-zero exit, git binary
    missing (FileNotFoundError), or timeout. Wrapped as a small helper so
    tests can monkeypatch it independently of `_git_is_dirty`.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    sha = result.stdout.strip()
    return sha if sha else "unknown"


def _git_is_dirty(repo_root: pathlib.Path) -> Optional[bool]:
    """Return True/False for working-tree dirty state, or None if git fails.

    `git status --porcelain` prints one line per modified/untracked entry; an
    empty stdout means the tree is clean. None preserves «не смогли определить»
    semantics from Req 7.4 and serializes to JSON null.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def cmd_manifest_init(args: argparse.Namespace) -> int:
    """Create a fresh Run_Manifest JSON for the current pipeline run.

    Implements subtask 3.1 (Requirements 7.1, 7.2, 7.3, 7.4):
      * Streams SHA-256 of dataset and eval CSV without loading them whole.
      * Captures git HEAD SHA and dirty flag (best effort — никаких прерываний
        при отсутствии git).
      * Captures host OS and Python version.
      * Writes the manifest to
        ``<reports_dir>/training_run_<UTC_TIMESTAMP>.json`` with empty
        ``steps: []`` (subsequent ``manifest-step`` calls will append).
      * Prints the absolute manifest path to stdout so wrappers can capture it.

    Возвращает 0 при успехе, 2 при любой неожиданной I/O-ошибке. Сообщения об
    ошибках идут в stderr, как и в `cmd_preflight`.
    """
    try:
        # Single timestamp instance — UTC `now` используется и для имени файла,
        # и для started_at, чтобы они гарантированно соответствовали друг другу
        # (иначе между двумя вызовами datetime.now() могла бы пройти секунда).
        now = datetime.datetime.now(datetime.timezone.utc)
        timestamp_filename = now.strftime("%Y%m%dT%H%M%SZ")
        started_at_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        dataset_path: pathlib.Path = args.dataset_path
        eval_csv_path: pathlib.Path = args.eval_csv_path
        reports_dir: pathlib.Path = args.reports_dir

        # SHA-256 + row count для датасета. Ошибки I/O пробрасываются в общий
        # try/except внизу — это сценарий "файл удалили между preflight и
        # manifest-init", и пользователь увидит их через stderr/exit 2.
        dataset_sha256 = _sha256_file(dataset_path)
        dataset_row_count = _count_data_rows(dataset_path)

        # SHA-256 для eval CSV. Row count для eval не требуется в схеме
        # Run_Manifest, поэтому считаем только хэш.
        eval_csv_sha256 = _sha256_file(eval_csv_path)

        # Пути в манифесте — относительные от REPO_ROOT, если возможно. Это
        # делает прогон переносимым (можно открыть JSON на другой машине и
        # сразу понять расположение датасета). Если путь снаружи репозитория —
        # сохраняем абсолютный (Path.relative_to бросит ValueError).
        def _rel(path: pathlib.Path) -> str:
            try:
                return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
            except ValueError:
                return str(path.resolve())

        # `platform.platform()` даёт стабильную полную строку вида
        # "Windows-10-10.0.19045-SP0" / "Linux-5.15.0-...-x86_64-with-glibc2.35",
        # эквивалент host_os из дизайна.
        host_os = platform.platform()
        python_version = platform.python_version()

        git_sha = _git_head_sha(REPO_ROOT)
        git_dirty = _git_is_dirty(REPO_ROOT)

        manifest = {
            "schema_version": 1,
            "started_at": started_at_iso,
            "finished_at": None,
            "final_exit_code": None,
            "host_os": host_os,
            "python_version": python_version,
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            "seed": args.seed,
            "dataset_path": _rel(dataset_path),
            "dataset_sha256": dataset_sha256,
            "dataset_row_count": dataset_row_count,
            "eval_csv_path": _rel(eval_csv_path),
            "eval_csv_sha256": eval_csv_sha256,
            "thresholds": dict(DEFAULT_THRESHOLDS),
            "steps": [],
        }

        reports_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = reports_dir / f"training_run_{timestamp_filename}.json"

        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)

        # Печатаем абсолютный путь одной строкой — обёртка перехватит его как
        # `MANIFEST=$(python ... manifest-init ...)`.
        sys.stdout.write(str(manifest_path.resolve()) + "\n")
        return 0
    except OSError as exc:
        sys.stderr.write(f"manifest-init: I/O error: {exc}\n")
        return 2


def cmd_manifest_step(args: argparse.Namespace) -> int:
    """Atomically append a single step record to an existing Run_Manifest.

    Implements subtask 3.2 (Requirements 5.5, 7.3, 8.3, 8.4). The function:
      * Reads the existing manifest JSON from ``args.manifest``.
      * Builds a new step dict with required fields (``name``, ``started_at``,
        ``finished_at``, ``exit_code``, ``artifact_paths``) plus optional
        ``gate_failed`` / ``skipped`` / ``skipped_reason`` when their CLI
        values are not None.
      * Appends the dict to ``manifest["steps"]``.
      * Writes the updated manifest to a sibling temp file and atomically
        renames it onto the original via ``os.replace`` (works on POSIX
        and Windows). This prevents readers from observing a partial file
        if the wrapper is interrupted mid-write — relevant when the
        pipeline aggregates many step appends sequentially.

    Returns 0 on success, 2 on missing manifest, JSON decode error, or any
    OSError. All failure messages go to stderr in line with the rest of the
    orchestrator (see ``cmd_preflight`` / ``cmd_manifest_init``).
    """
    manifest_path: pathlib.Path = args.manifest

    # Чтение существующего манифеста. Отсутствующий файл — фатально (exit 2):
    # пайплайн не может «продолжить как ни в чём не бывало», если manifest-init
    # не отработал ранее.
    if not manifest_path.is_file():
        sys.stderr.write(
            f"manifest-step: manifest not found at {manifest_path}\n"
        )
        return 2

    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(
            f"manifest-step: failed to read manifest at {manifest_path}: {exc}\n"
        )
        return 2

    # Сборка step dict. `artifact_paths` всегда присутствует (пустой список,
    # если артефактов нет) — это упрощает downstream-парсинг summary.
    step: dict = {
        "name": args.name,
        "started_at": args.started_at,
        "finished_at": args.finished_at,
        "exit_code": args.exit_code,
        "artifact_paths": list(args.artifacts or []),
    }

    # Опциональные поля включаются только если CLI-флаг был явно задан.
    # `BooleanOptionalAction` с `default=None` различает "флаг не передан"
    # (None) и "флаг передан как --no-foo" (False), поэтому проверяем именно
    # `is not None`, а не truthiness.
    if args.gate_failed is not None:
        step["gate_failed"] = args.gate_failed
    if args.skipped is not None:
        step["skipped"] = args.skipped
    if args.skipped_reason is not None:
        step["skipped_reason"] = args.skipped_reason

    # `steps` должен существовать после manifest-init; если ключа нет (или он
    # не список) — мягко инициализируем, чтобы битый manifest не валил весь
    # пайплайн.
    steps = manifest.get("steps")
    if not isinstance(steps, list):
        steps = []
        manifest["steps"] = steps
    steps.append(step)

    # Атомарная запись через общий helper (см. _atomic_write_json).
    try:
        _atomic_write_json(manifest_path, manifest)
        return 0
    except OSError as exc:
        sys.stderr.write(
            f"manifest-step: failed to write manifest at {manifest_path}: {exc}\n"
        )
        return 2


def _load_eval_json(experimental_dir: pathlib.Path, filename: str) -> dict:
    """Load an eval JSON file and extract key metrics.

    Returns a dict with keys:
      - ``block_precision``: float or None
      - ``block_recall``: float or None
      - ``allow_fp_rate``: float or None
      - ``gate_passed``: bool or None
      - ``evaluated``: True (file exists and was parsed)

    If the file does not exist, returns ``{"evaluated": False}``.
    If a metric key is missing inside the JSON, the corresponding value is None.
    """
    path = experimental_dir / filename
    if not path.is_file():
        return {"evaluated": False}

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {"evaluated": False}

    metrics = data.get("metrics") if isinstance(data, dict) else None
    if not isinstance(metrics, dict):
        metrics = {}

    # Extract gate_passed: eval_golden_set.py writes "status": "PASS"/"FAIL".
    # Design refers to this as gate_passed (bool).
    status = data.get("status") if isinstance(data, dict) else None
    if isinstance(status, str):
        gate_passed: bool | None = status.upper() == "PASS"
    else:
        # Fallback: check if there's an explicit gate_passed field
        gp = data.get("gate_passed") if isinstance(data, dict) else None
        gate_passed = bool(gp) if gp is not None else None

    return {
        "evaluated": True,
        "block_precision": metrics.get("block_precision"),
        "block_recall": metrics.get("block_recall"),
        "allow_fp_rate": metrics.get("allow_fp_rate"),
        "gate_passed": gate_passed,
    }


def _load_prod_model_card(prod_model_card_path: pathlib.Path) -> dict | str:
    """Load the production model card and extract key metrics.

    Returns a dict with keys ``block_precision``, ``block_recall``,
    ``allow_fp_rate`` (each float or None) when the file exists and is
    parseable. Returns the string ``"unavailable"`` when the file is
    missing or cannot be decoded.

    The prod model card stores cold-eval metrics in two possible locations:
      1. ``metrics.block_precision`` / ``metrics.block_recall`` /
         ``metrics.allow_fp_rate`` — the canonical eval JSON layout.
      2. ``cold_thresholds.block_precision`` / ``cold_thresholds.block_recall``
         — the layout used by ``train_kd_distillation.py``.

    The function checks ``metrics`` first; if absent, falls back to
    ``cold_thresholds``. ``allow_fp_rate`` may not be present in either
    location — in that case it is set to None.
    """
    if not prod_model_card_path.is_file():
        return "unavailable"

    try:
        with prod_model_card_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return "unavailable"

    if not isinstance(data, dict):
        return "unavailable"

    # Try canonical `metrics` sub-object first (eval JSON style).
    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        return {
            "block_precision": metrics.get("block_precision"),
            "block_recall": metrics.get("block_recall"),
            "allow_fp_rate": metrics.get("allow_fp_rate"),
        }

    # Fallback: cold_thresholds (train_kd_distillation.py model card style).
    cold = data.get("cold_thresholds")
    if isinstance(cold, dict):
        return {
            "block_precision": cold.get("block_precision"),
            "block_recall": cold.get("block_recall"),
            "allow_fp_rate": cold.get("allow_fp_rate"),
        }

    # Last resort: top-level keys (some model cards put them at root).
    bp = data.get("block_precision")
    br = data.get("block_recall")
    afp = data.get("allow_fp_rate")
    if bp is not None or br is not None:
        return {
            "block_precision": bp,
            "block_recall": br,
            "allow_fp_rate": afp,
        }

    return "unavailable"


def _fmt_metric(value) -> str:
    """Format a metric value for the ASCII table: float as 4-decimal, else 'N/A'."""
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "N/A"


def _determine_eligibility(
    model_eval: dict,
    prod_metrics: dict | str,
) -> tuple[bool, str]:
    """Apply the promotion eligibility rule (Requirement 9.1, 5.5, 6.5).

    Returns (eligible: bool, reason: str).

    Rules:
      1. If gate_passed is False -> eligible = False regardless of metrics.
      2. If prod_metrics == "unavailable" -> eligible = gate_passed
         (don't block on missing prod baseline).
      3. Otherwise: eligible = (block_precision >= prod.block_precision)
         AND (block_recall >= prod.block_recall - 0.05).
    """
    if not model_eval.get("evaluated"):
        return False, "not evaluated"

    gate_passed = model_eval.get("gate_passed")

    # Rule 1: gate_passed == False -> not eligible
    if gate_passed is False:
        return False, "eval gate failed"

    # Rule 2: prod metrics unavailable -> eligible = gate_passed
    if prod_metrics == "unavailable":
        if gate_passed:
            return True, "gate passed (no prod baseline)"
        # gate_passed is None (unknown) — treat as not eligible
        return False, "gate status unknown (no prod baseline)"

    # Rule 3: compare against prod metrics
    bp = model_eval.get("block_precision")
    br = model_eval.get("block_recall")
    prod_bp = prod_metrics.get("block_precision") if isinstance(prod_metrics, dict) else None
    prod_br = prod_metrics.get("block_recall") if isinstance(prod_metrics, dict) else None

    # If we can't compare (missing metrics), fall back to gate_passed
    if bp is None or br is None or prod_bp is None or prod_br is None:
        if gate_passed:
            return True, "gate passed (metrics incomplete for comparison)"
        return False, "metrics incomplete for comparison"

    precision_ok = bp >= prod_bp
    recall_ok = br >= prod_br - 0.05

    if precision_ok and recall_ok:
        return True, "meets prod baseline"
    elif not precision_ok and not recall_ok:
        return False, f"precision < prod ({bp:.4f} < {prod_bp:.4f}) and recall dropped > 0.05"
    elif not precision_ok:
        return False, f"precision < prod ({bp:.4f} < {prod_bp:.4f})"
    else:
        return False, f"recall dropped > 0.05 ({br:.4f} < {prod_br:.4f} - 0.05)"


def _build_ascii_table(rows: list[dict]) -> str:
    """Build an ASCII table using +, -, | characters only.

    Each row dict has keys: model, block_p, block_r, allow_fp, gate_passed,
    eligible, reason.
    """
    headers = ["Model", "Block_P", "Block_R", "Allow_FP", "Gate", "Eligible", "Reason"]
    # Compute column widths
    col_widths = [len(h) for h in headers]
    str_rows: list[list[str]] = []
    for row in rows:
        cells = [
            row["model"],
            row["block_p"],
            row["block_r"],
            row["allow_fp"],
            row["gate_passed"],
            row["eligible"],
            row["reason"],
        ]
        str_rows.append(cells)
        for i, cell in enumerate(cells):
            col_widths[i] = max(col_widths[i], len(cell))

    # Build separator line
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    # Build header line
    header_line = "|" + "|".join(
        f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)
    ) + "|"

    # Build data lines
    lines = [sep, header_line, sep]
    for cells in str_rows:
        data_line = "|" + "|".join(
            f" {c:<{col_widths[i]}} " for i, c in enumerate(cells)
        ) + "|"
        lines.append(data_line)
    lines.append(sep)

    return "\n".join(lines)


def cmd_summary(args: argparse.Namespace) -> int:
    """Print a human-readable comparison of trained models vs prod.

    Implements subtasks 9.1–9.4:
      9.1 — Load eval JSONs from EXPERIMENTAL_DIR.
      9.2 — Load prod model card for baseline comparison.
      9.3 — Apply promotion eligibility rule and print ASCII table.
      9.4 — Print OS-appropriate copy commands for eligible models.
    """
    experimental_dir: pathlib.Path = args.experimental_dir
    prod_model_card_path: pathlib.Path = args.prod_model_card

    # --- 9.1: Load eval JSONs from EXPERIMENTAL_DIR -----------------------
    eval_leak_free = _load_eval_json(experimental_dir, "eval_leak_free.json")
    eval_binary = _load_eval_json(experimental_dir, "eval_binary.json")

    # --- 9.2: Load prod model card ----------------------------------------
    prod_metrics = _load_prod_model_card(prod_model_card_path)

    # --- 9.3: Apply promotion rule and build ASCII table ------------------
    models = [
        ("leak_free", eval_leak_free),
        ("binary", eval_binary),
    ]

    table_rows: list[dict] = []
    eligibility_results: list[tuple[str, bool, str]] = []

    for model_name, model_eval in models:
        eligible, reason = _determine_eligibility(model_eval, prod_metrics)
        eligibility_results.append((model_name, eligible, reason))

        if model_eval.get("evaluated"):
            gate_str = "PASS" if model_eval.get("gate_passed") else "FAIL"
            if model_eval.get("gate_passed") is None:
                gate_str = "N/A"
        else:
            gate_str = "N/A"

        table_rows.append({
            "model": model_name,
            "block_p": _fmt_metric(model_eval.get("block_precision")),
            "block_r": _fmt_metric(model_eval.get("block_recall")),
            "allow_fp": _fmt_metric(model_eval.get("allow_fp_rate")),
            "gate_passed": gate_str,
            "eligible": "YES" if eligible else "NO",
            "reason": reason,
        })

    # Print prod baseline info
    sys.stdout.write("\n=== Model Training Summary ===\n\n")
    if prod_metrics == "unavailable":
        sys.stdout.write("Prod model card: unavailable (no baseline for comparison)\n\n")
    else:
        sys.stdout.write(
            f"Prod baseline: block_p={_fmt_metric(prod_metrics.get('block_precision'))}"
            f"  block_r={_fmt_metric(prod_metrics.get('block_recall'))}"
            f"  allow_fp={_fmt_metric(prod_metrics.get('allow_fp_rate'))}\n\n"
        )

    # Print ASCII table
    table_str = _build_ascii_table(table_rows)
    sys.stdout.write(table_str + "\n\n")

    # --- 9.4: Print OS-appropriate promotion commands ---------------------
    sys.stdout.write("=== Promotion Commands ===\n\n")

    is_windows = os.name == "nt"

    for model_name, eligible, reason in eligibility_results:
        tflite_src = f"app/src/main/assets/experimental/spam_model_{model_name}.tflite"
        tflite_dst = "app/src/main/assets/spam_model.tflite"
        card_src = f"app/src/main/assets/experimental/model_card_{model_name}.json"
        card_dst = "app/src/main/assets/model_card.json"

        if is_windows:
            cmd_tflite = f"Copy-Item {tflite_src} {tflite_dst} -Force"
            cmd_card = f"Copy-Item {card_src} {card_dst} -Force"
        else:
            cmd_tflite = f"cp {tflite_src} {tflite_dst}"
            cmd_card = f"cp {card_src} {card_dst}"

        if eligible:
            sys.stdout.write(f"# {model_name} (eligible):\n")
            sys.stdout.write(f"  {cmd_tflite}\n")
            sys.stdout.write(f"  {cmd_card}\n\n")
        else:
            sys.stdout.write(f"# {model_name} (not recommended: {reason}):\n")
            sys.stdout.write(f"  {cmd_tflite}\n")
            sys.stdout.write(f"  {cmd_card}\n\n")

    return 0


def cmd_manifest_finalize(args: argparse.Namespace) -> int:
    """Append `finished_at` and `final_exit_code` to the Run_Manifest.

    Implements subtask 3.3 (Requirement 7.5). The function is invoked by the
    wrapper from a Bash ``trap EXIT`` / PowerShell ``finally`` block, so it
    MUST be idempotent: if the manifest is already finalized (both
    ``finished_at`` is a non-null string and ``final_exit_code`` is an int),
    the function returns 0 without overwriting and prints a notice to
    stderr. Otherwise it stamps the current UTC timestamp and the provided
    exit code, then atomically rewrites the manifest.

    Возвращает:
      * 0 при успешном финализировании ИЛИ при идемпотентном повторном вызове;
      * 2 при отсутствии манифеста, JSON decode error или OSError записи.
    """
    manifest_path: pathlib.Path = args.manifest

    # Отсутствующий файл — фатально (exit 2): wrapper не может «корректно
    # завершить» прогон, в котором manifest-init не отработал.
    if not manifest_path.is_file():
        sys.stderr.write(
            f"manifest-finalize: manifest not found at {manifest_path}\n"
        )
        return 2

    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(
            f"manifest-finalize: failed to read manifest at {manifest_path}: {exc}\n"
        )
        return 2

    # Идемпотентность. Признак «уже финализировано» — оба поля одновременно
    # заполнены валидными типами:
    #   * `finished_at`: non-empty str (manifest-init пишет туда null);
    #   * `final_exit_code`: int (manifest-init пишет туда null; bool в Python —
    #     подкласс int, исключаем явно, чтобы не интерпретировать True как 1).
    # Этот сценарий штатный (Bash trap EXIT может вызвать finalize дважды
    # при повторном получении сигнала), поэтому возвращаем 0, а не ошибку.
    existing_finished = manifest.get("finished_at")
    existing_exit_code = manifest.get("final_exit_code")
    already_finalized = (
        isinstance(existing_finished, str)
        and existing_finished
        and isinstance(existing_exit_code, int)
        and not isinstance(existing_exit_code, bool)
    )
    if already_finalized:
        sys.stderr.write(
            "manifest-finalize: manifest already finalized at "
            f"{existing_finished} with exit_code={existing_exit_code}\n"
        )
        return 0

    # Свежий timestamp в том же формате, что и manifest-init (UTC, без
    # микросекунд) — поля started_at/finished_at таким образом форматно
    # консистентны.
    now = datetime.datetime.now(datetime.timezone.utc)
    manifest["finished_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["final_exit_code"] = args.exit_code

    try:
        _atomic_write_json(manifest_path, manifest)
        return 0
    except OSError as exc:
        sys.stderr.write(
            f"manifest-finalize: failed to write manifest at {manifest_path}: {exc}\n"
        )
        return 2


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline_orchestrator",
        description=(
            "Python core for the antispam model training pipeline. "
            "Invoked by scripts/train_full_pipeline.{sh,ps1}."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # preflight ------------------------------------------------------------
    p_pre = subparsers.add_parser(
        "preflight",
        help="Validate Python version, packages, dataset and eval CSV.",
    )
    p_pre.add_argument(
        "--strict",
        dest="strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat warnings as errors (default: enabled).",
    )
    p_pre.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        default=DATASET_PATH,
        help=f"Path to the training dataset CSV (default: {DATASET_PATH}).",
    )
    p_pre.add_argument(
        "--eval-csv-path",
        type=pathlib.Path,
        default=EVAL_CSV_PATH,
        help=f"Path to the cold eval CSV (default: {EVAL_CSV_PATH}).",
    )
    p_pre.set_defaults(func=cmd_preflight)

    # manifest-init --------------------------------------------------------
    p_init = subparsers.add_parser(
        "manifest-init",
        help="Create a new Run_Manifest JSON for the current pipeline run.",
    )
    p_init.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_init.add_argument("--dataset-path", type=pathlib.Path, default=DATASET_PATH)
    p_init.add_argument("--eval-csv-path", type=pathlib.Path, default=EVAL_CSV_PATH)
    p_init.add_argument("--reports-dir", type=pathlib.Path, default=REPORTS_DIR)
    p_init.set_defaults(func=cmd_manifest_init)

    # manifest-step --------------------------------------------------------
    p_step = subparsers.add_parser(
        "manifest-step",
        help="Atomically append a step record to an existing Run_Manifest.",
    )
    p_step.add_argument("--manifest", type=pathlib.Path, required=True)
    p_step.add_argument("--name", required=True)
    p_step.add_argument("--started-at", required=True)
    p_step.add_argument("--finished-at", required=True)
    p_step.add_argument("--exit-code", type=int, required=True)
    p_step.add_argument(
        "--artifact",
        dest="artifacts",
        action="append",
        default=[],
        help="Path to an artifact produced by the step (repeatable).",
    )
    p_step.add_argument(
        "--gate-failed",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p_step.add_argument(
        "--skipped",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p_step.add_argument("--skipped-reason", default=None)
    p_step.set_defaults(func=cmd_manifest_step)

    # summary --------------------------------------------------------------
    p_sum = subparsers.add_parser(
        "summary",
        help="Print a human-readable comparison of trained models vs prod.",
    )
    p_sum.add_argument("--manifest", type=pathlib.Path, required=True)
    p_sum.add_argument(
        "--experimental-dir",
        type=pathlib.Path,
        default=EXPERIMENTAL_DIR,
    )
    p_sum.add_argument(
        "--prod-model-card",
        type=pathlib.Path,
        default=PROD_MODEL_CARD,
    )
    p_sum.set_defaults(func=cmd_summary)

    # manifest-finalize ----------------------------------------------------
    p_fin = subparsers.add_parser(
        "manifest-finalize",
        help="Append finished_at and final_exit_code to the Run_Manifest.",
    )
    p_fin.add_argument("--manifest", type=pathlib.Path, required=True)
    p_fin.add_argument("--exit-code", type=int, required=True)
    p_fin.set_defaults(func=cmd_manifest_finalize)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _entrypoint() -> NoReturn:
    sys.exit(main())


if __name__ == "__main__":
    _entrypoint()
