"""Unit-тесты для подзадачи 2.2 model-training-pipeline.

Покрывают `scripts/pipeline_orchestrator.py::cmd_preflight` в части проверки
окружения (Python version + обязательные пакеты).

Кейсы:
* Python < 3.10 → exit 2, в stderr указаны минимальная и текущая версии.
* `tensorflow` отсутствует → exit 2 + точная подсказка `pip install ...`.
* `catboost` отсутствует (остальные импорты успешны) → exit 2 + та же подсказка.
* Окружение полностью ок → текущая реализация падает в `NOT_IMPLEMENTED_EXIT_CODE`
  (это поведение должно осознанно поменяться при имплементации подзадач 2.3/2.5).

Тесты НЕ зависят от наличия настоящих `tensorflow`, `catboost`, `sklearn`,
`numpy` в окружении: `importlib.import_module` подменяется фейком на уровне
модуля под тестом.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import sys
from collections import namedtuple
from typing import Iterable

import pytest


# ---------------------------------------------------------------------------
# Загрузка модуля без мутации sys.path и без зависимости от scripts/__init__.py
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ORCH_PATH = REPO_ROOT / "scripts" / "pipeline_orchestrator.py"

# Точная подсказка установки из Requirement 2.2 — должна совпадать с тем,
# что печатает cmd_preflight.
INSTALL_HINT = "pip install tensorflow catboost scikit-learn numpy"


def _load_orchestrator():
    """Загрузить scripts/pipeline_orchestrator.py как изолированный модуль.

    Используется `spec_from_file_location` вместо `sys.path.insert`, чтобы:
    * не зависеть от наличия `scripts/__init__.py`;
    * не оставлять глобальных побочных эффектов между тестами разных модулей.
    """
    spec = importlib.util.spec_from_file_location(
        "pipeline_orchestrator_under_test", ORCH_PATH
    )
    assert spec is not None and spec.loader is not None, (
        f"Не удалось получить ModuleSpec для {ORCH_PATH}"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def orchestrator():
    return _load_orchestrator()


# ---------------------------------------------------------------------------
# Хелперы для построения фейковых sys.version_info / importlib
# ---------------------------------------------------------------------------

# Имитируем форму sys.version_info: cmd_preflight использует .major / .minor / .micro.
VersionInfo = namedtuple(
    "VersionInfo", "major minor micro releaselevel serial"
)


def _mk_version(major: int, minor: int, micro: int = 0) -> VersionInfo:
    return VersionInfo(major, minor, micro, "final", 0)


def _mk_args(
    dataset_path: pathlib.Path | None = None,
    eval_csv_path: pathlib.Path | None = None,
) -> argparse.Namespace:
    """Argparse-namespace для cmd_preflight.

    `dataset_path` соответствует `--dataset-path`, `eval_csv_path` —
    `--eval-csv-path` из argparse-обвязки. Тесты, которые не доходят до
    проверок датасета/eval (Python/пакеты упали раньше), могут не
    передавать их.
    """
    return argparse.Namespace(
        dataset_path=dataset_path,
        eval_csv_path=eval_csv_path,
    )


class _FakeImportlib:
    """Минимальный двойник модуля importlib с настраиваемым `import_module`.

    Подменяется как атрибут на загруженном орхестраторе, чтобы не трогать
    глобальный `importlib` (что могло бы сломать pytest-инфраструктуру).
    """

    def __init__(self, missing: Iterable[str]):
        self._missing = set(missing)
        self.calls: list[str] = []

    def import_module(self, name: str):  # сигнатура совместима с importlib
        self.calls.append(name)
        if name in self._missing:
            raise ImportError(f"No module named '{name}' (mocked)")
        return object()


# ---------------------------------------------------------------------------
# Тесты
# ---------------------------------------------------------------------------


def test_python_too_old_returns_exit_2_and_mentions_versions(
    orchestrator, monkeypatch, capsys
):
    """Python < MIN_PYTHON → exit 2, stderr содержит требуемую и текущую версии."""
    monkeypatch.setattr(sys, "version_info", _mk_version(3, 9, 7))

    rc = orchestrator.cmd_preflight(_mk_args())

    assert rc == 2, "При устаревшей версии Python ожидался exit code 2"
    captured = capsys.readouterr()
    err = captured.err
    assert "3.10" in err, f"ожидалась минимальная требуемая версия 3.10 в stderr, got: {err!r}"
    assert "3.9" in err, f"ожидалась текущая версия 3.9 в stderr, got: {err!r}"
    # До шага проверки пакетов дойти не должны.
    assert INSTALL_HINT not in err


def test_missing_tensorflow_returns_exit_2_with_install_hint(
    orchestrator, monkeypatch, capsys
):
    """Отсутствие tensorflow → exit 2 + точная строка установки из Req 2.2."""
    monkeypatch.setattr(sys, "version_info", _mk_version(3, 12, 0))
    fake_importlib = _FakeImportlib(missing={"tensorflow"})
    monkeypatch.setattr(orchestrator, "importlib", fake_importlib)

    rc = orchestrator.cmd_preflight(_mk_args())

    assert rc == 2
    err = capsys.readouterr().err
    assert "tensorflow" in err, f"имя пакета должно фигурировать в stderr, got: {err!r}"
    assert INSTALL_HINT in err, (
        f"stderr должен содержать точную подсказку установки {INSTALL_HINT!r}, "
        f"got: {err!r}"
    )
    # Должны были остановиться на первом провалившемся пакете.
    assert fake_importlib.calls == ["tensorflow"]


def test_missing_catboost_returns_exit_2_with_install_hint(
    orchestrator, monkeypatch, capsys
):
    """tensorflow ок, catboost отсутствует → exit 2 + та же подсказка."""
    monkeypatch.setattr(sys, "version_info", _mk_version(3, 11, 4))
    fake_importlib = _FakeImportlib(missing={"catboost"})
    monkeypatch.setattr(orchestrator, "importlib", fake_importlib)

    rc = orchestrator.cmd_preflight(_mk_args())

    assert rc == 2
    err = capsys.readouterr().err
    assert "catboost" in err, f"имя пакета должно фигурировать в stderr, got: {err!r}"
    assert INSTALL_HINT in err
    # Проверяется в порядке REQUIRED_PACKAGES; tensorflow должен был успешно
    # импортироваться до catboost.
    assert fake_importlib.calls[:2] == ["tensorflow", "catboost"]


def test_environment_ok_returns_exit_0(
    orchestrator, monkeypatch, tmp_path, capsys
):
    """Версия, пакеты, датасет и eval CSV ок → preflight возвращает 0.

    После имплементации подзадачи 2.5 cmd_preflight больше не уходит в
    NOT_IMPLEMENTED_EXIT_CODE, а возвращает 0 при полностью валидном
    окружении.
    """
    monkeypatch.setattr(sys, "version_info", _mk_version(3, 12, 0))
    fake_importlib = _FakeImportlib(missing=set())
    monkeypatch.setattr(orchestrator, "importlib", fake_importlib)

    # Минимальный валидный датасет: header + 1 data-строка.
    dataset_path = tmp_path / "ru_tflite_features.csv"
    dataset_path.write_text("col_a,col_b\n1,2\n", encoding="utf-8")

    # Валидный eval CSV: header + 100 data-строк.
    eval_csv_path = tmp_path / "cold_eval_600.csv"
    eval_lines = ["phone,label"] + [f"+7900000{i:04d},spam" for i in range(100)]
    eval_csv_path.write_text("\n".join(eval_lines) + "\n", encoding="utf-8")

    rc = orchestrator.cmd_preflight(
        _mk_args(dataset_path=dataset_path, eval_csv_path=eval_csv_path)
    )

    assert rc == 0, f"Полностью валидное окружение должно давать exit 0, got {rc}"
    err = capsys.readouterr().err
    # При успешных импортах подсказка установки не должна появляться.
    assert INSTALL_HINT not in err
    # Все обязательные пакеты должны быть проверены.
    assert list(fake_importlib.calls) == list(orchestrator.REQUIRED_PACKAGES)


# ---------------------------------------------------------------------------
# Подзадача 2.4: проверка датасета (Requirements 2.3, 2.4, 2.5)
# ---------------------------------------------------------------------------

# Имена сырых входов, которые cmd_preflight перечисляет в сообщении об
# ошибке для пустого/header-only датасета (см. требование 2.5).
_RAW_INPUTS = (
    "ru_call_features.csv",
    "ru_numbers_labeled.csv",
    "ru_reputation_raw.csv",
)


def _passing_env(orchestrator, monkeypatch) -> _FakeImportlib:
    """Подготовить окружение так, чтобы preflight дошёл до проверки датасета.

    Подменяет `sys.version_info` на 3.12.0 и `importlib` внутри орхестратора
    на фейк без отсутствующих пакетов. Возвращает фейковый importlib на
    случай, если тесту нужно проверить, что все пакеты были импортированы.
    """
    monkeypatch.setattr(sys, "version_info", _mk_version(3, 12, 0))
    fake_importlib = _FakeImportlib(missing=set())
    monkeypatch.setattr(orchestrator, "importlib", fake_importlib)
    return fake_importlib


def test_dataset_missing_returns_exit_10(orchestrator, monkeypatch, tmp_path, capsys):
    """Файл датасета отсутствует → exit 10, stderr указывает путь и builder."""
    _passing_env(orchestrator, monkeypatch)

    missing_path = tmp_path / "missing.csv"
    assert not missing_path.exists(), "sanity: файл не должен существовать"

    rc = orchestrator.cmd_preflight(_mk_args(dataset_path=missing_path))

    assert rc == 10, (
        f"При отсутствующем датасете ожидался exit 10 (сигнал «нужно собрать»), got {rc}"
    )
    err = capsys.readouterr().err
    assert str(missing_path) in err, (
        f"stderr должен упоминать путь к датасету {missing_path}, got: {err!r}"
    )
    assert "ru_metadata_dataset_builder.py" in err, (
        f"stderr должен подсказывать запустить ru_metadata_dataset_builder.py, got: {err!r}"
    )


def test_dataset_completely_empty_returns_exit_2(
    orchestrator, monkeypatch, tmp_path, capsys
):
    """0-байтовый датасет → exit 2, stderr перечисляет три сырых входа."""
    _passing_env(orchestrator, monkeypatch)

    dataset_path = tmp_path / "ru_tflite_features.csv"
    dataset_path.write_bytes(b"")
    assert dataset_path.stat().st_size == 0, "sanity: файл должен быть пустым"

    rc = orchestrator.cmd_preflight(_mk_args(dataset_path=dataset_path))

    assert rc == 2, f"При пустом датасете ожидался exit 2, got {rc}"
    err = capsys.readouterr().err
    for raw_input in _RAW_INPUTS:
        assert raw_input in err, (
            f"stderr должен упоминать сырой вход {raw_input}, got: {err!r}"
        )


def test_dataset_header_only_returns_exit_2(
    orchestrator, monkeypatch, tmp_path, capsys
):
    """Только заголовок без data-строк → exit 2, stderr перечисляет три сырых входа."""
    _passing_env(orchestrator, monkeypatch)

    dataset_path = tmp_path / "ru_tflite_features.csv"
    dataset_path.write_text("col_a,col_b\n", encoding="utf-8")

    rc = orchestrator.cmd_preflight(_mk_args(dataset_path=dataset_path))

    assert rc == 2, f"Header-only датасет должен давать exit 2, got {rc}"
    err = capsys.readouterr().err
    for raw_input in _RAW_INPUTS:
        assert raw_input in err, (
            f"stderr должен упоминать сырой вход {raw_input}, got: {err!r}"
        )


def test_dataset_with_one_data_row_passes_dataset_check(
    orchestrator, monkeypatch, tmp_path, capsys
):
    """Header + ≥1 data-строка + валидный eval CSV → preflight успешен (exit 0).

    После подзадачи 2.5 проверка eval CSV подключена; при полностью
    валидном окружении cmd_preflight возвращает 0.
    """
    _passing_env(orchestrator, monkeypatch)

    dataset_path = tmp_path / "ru_tflite_features.csv"
    dataset_path.write_text("col_a,col_b\n1,2\n", encoding="utf-8")

    # Валидный eval CSV: header + 100 data-строк.
    eval_csv_path = tmp_path / "cold_eval_600.csv"
    eval_lines = ["phone,label"] + [f"+7900000{i:04d},spam" for i in range(100)]
    eval_csv_path.write_text("\n".join(eval_lines) + "\n", encoding="utf-8")

    rc = orchestrator.cmd_preflight(
        _mk_args(dataset_path=dataset_path, eval_csv_path=eval_csv_path)
    )

    assert rc == 0, (
        "Валидный датасет (1+ data-строк) и валидный eval CSV (>=100 строк) "
        f"должны давать exit 0. Получили {rc}."
    )
    err = capsys.readouterr().err
    # Ошибочные сообщения уровня датасета (отсутствует / пустой) появляться не должны.
    assert "is empty" not in err, f"не ожидалось сообщение про пустой датасет, got: {err!r}"
    assert "dataset not found" not in err, (
        f"не ожидалось сообщение про отсутствующий датасет, got: {err!r}"
    )


# ---------------------------------------------------------------------------
# Подзадача 2.6: проверка eval CSV (Requirements 2.6, 2.7)
# ---------------------------------------------------------------------------


def _write_eval_csv(path: pathlib.Path, rows: int) -> None:
    """Записать eval CSV с заголовком и `rows` data-строками.

    Формат намеренно тривиальный (две колонки) — preflight только считает
    строки и не валидирует схему cold_eval_600.csv на этой подзадаче.
    """
    lines = ["col_a,col_b"] + [f"value_{i},{i}" for i in range(rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_valid_dataset(path: pathlib.Path) -> None:
    """Минимальный валидный датасет: header + 1 data-строка.

    Нужен, чтобы preflight прошёл проверку датасета и дошёл до eval CSV.
    """
    path.write_text("col_a,col_b\n1,2\n", encoding="utf-8")


def test_eval_csv_missing_returns_exit_2(
    orchestrator, monkeypatch, tmp_path, capsys
):
    """Eval CSV отсутствует → exit 2; stderr упоминает cold_eval_600.csv."""
    _passing_env(orchestrator, monkeypatch)

    dataset_path = tmp_path / "ru_tflite_features.csv"
    _write_valid_dataset(dataset_path)

    eval_csv_path = tmp_path / "cold_eval_600.csv"
    assert not eval_csv_path.exists(), "sanity: eval CSV не должен существовать"

    rc = orchestrator.cmd_preflight(
        _mk_args(dataset_path=dataset_path, eval_csv_path=eval_csv_path)
    )

    assert rc == 2, f"При отсутствующем eval CSV ожидался exit 2, got {rc}"
    err = capsys.readouterr().err
    assert "cold_eval_600.csv" in err, (
        f"stderr должен упоминать ожидаемое имя cold_eval_600.csv, got: {err!r}"
    )


def test_eval_csv_too_few_rows_returns_exit_2(
    orchestrator, monkeypatch, tmp_path, capsys
):
    """Eval CSV с 50 data-строками → exit 2; stderr упоминает cold_eval_600.csv."""
    _passing_env(orchestrator, monkeypatch)

    dataset_path = tmp_path / "ru_tflite_features.csv"
    _write_valid_dataset(dataset_path)

    eval_csv_path = tmp_path / "cold_eval_600.csv"
    _write_eval_csv(eval_csv_path, rows=50)

    rc = orchestrator.cmd_preflight(
        _mk_args(dataset_path=dataset_path, eval_csv_path=eval_csv_path)
    )

    assert rc == 2, (
        f"Eval CSV с <100 data-строк должен давать exit 2, got {rc}"
    )
    err = capsys.readouterr().err
    assert "cold_eval_600.csv" in err, (
        f"stderr должен упоминать ожидаемое имя cold_eval_600.csv, got: {err!r}"
    )


def test_eval_csv_with_100_rows_passes(
    orchestrator, monkeypatch, tmp_path, capsys
):
    """Eval CSV ровно со 100 data-строк → preflight успешен (exit 0)."""
    _passing_env(orchestrator, monkeypatch)

    dataset_path = tmp_path / "ru_tflite_features.csv"
    _write_valid_dataset(dataset_path)

    eval_csv_path = tmp_path / "cold_eval_600.csv"
    _write_eval_csv(eval_csv_path, rows=100)

    rc = orchestrator.cmd_preflight(
        _mk_args(dataset_path=dataset_path, eval_csv_path=eval_csv_path)
    )

    assert rc == 0, (
        f"Eval CSV с >=100 data-строк должен давать exit 0, got {rc}"
    )
    err = capsys.readouterr().err
    # При успехе сообщение об eval CSV не должно появляться.
    assert "eval CSV" not in err, (
        f"не ожидалось сообщение об ошибке eval CSV, got: {err!r}"
    )


# ---------------------------------------------------------------------------
# Подзадача 3.4: Unit-тесты манифеста (Requirements 7.3, 7.4, 7.5)
# ---------------------------------------------------------------------------


def _mk_manifest_init_args(
    tmp_path: pathlib.Path,
    seed: int = 42,
) -> tuple[argparse.Namespace, pathlib.Path, pathlib.Path]:
    """Подготовить аргументы и файлы для cmd_manifest_init.

    Создаёт минимальный валидный датасет и eval CSV в tmp_path, возвращает
    (args, dataset_path, eval_csv_path).
    """
    dataset_path = tmp_path / "ru_tflite_features.csv"
    dataset_path.write_text("col_a,col_b\n1,2\n", encoding="utf-8")

    eval_csv_path = tmp_path / "cold_eval_600.csv"
    eval_lines = ["phone,label"] + [f"+7900000{i:04d},spam" for i in range(100)]
    eval_csv_path.write_text("\n".join(eval_lines) + "\n", encoding="utf-8")

    reports_dir = tmp_path / "reports" / "training"

    args = argparse.Namespace(
        seed=seed,
        dataset_path=dataset_path,
        eval_csv_path=eval_csv_path,
        reports_dir=reports_dir,
    )
    return args, dataset_path, eval_csv_path


def test_manifest_init_git_success(orchestrator, monkeypatch, tmp_path, capsys):
    """manifest-init с работающим git → git_sha и git_dirty заполнены корректно.

    Мокаем subprocess.run так, чтобы `git rev-parse HEAD` вернул известный SHA,
    а `git status --porcelain` вернул пустой stdout (clean tree).
    Ожидаем: exit 0, JSON содержит git_sha == мок-значение, git_dirty == False.
    """
    import subprocess as real_subprocess

    args, _, _ = _mk_manifest_init_args(tmp_path)

    fake_sha = "abc1234def5678"

    def fake_subprocess_run(cmd, **kwargs):
        if "rev-parse" in cmd:
            result = real_subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=fake_sha + "\n", stderr=""
            )
            return result
        if "status" in cmd and "--porcelain" in cmd:
            result = real_subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )
            return result
        # Fallback — не должен вызываться в этом тесте.
        return real_subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr=""
        )

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_subprocess_run)

    rc = orchestrator.cmd_manifest_init(args)

    assert rc == 0, f"manifest-init должен вернуть 0 при успехе, got {rc}"

    # Проверяем stdout: должен содержать путь к манифесту.
    captured = capsys.readouterr()
    manifest_path = pathlib.Path(captured.out.strip())
    assert manifest_path.exists(), f"Манифест не создан по пути {manifest_path}"

    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["git_sha"] == fake_sha, (
        f"git_sha должен быть {fake_sha!r}, got {manifest['git_sha']!r}"
    )
    assert manifest["git_dirty"] is False, (
        f"git_dirty должен быть False (clean tree), got {manifest['git_dirty']!r}"
    )
    # Проверяем обязательные поля из Req 7.3.
    assert manifest["schema_version"] == 1
    assert manifest["started_at"] is not None
    assert manifest["seed"] == 42
    assert manifest["steps"] == []
    assert "dataset_sha256" in manifest
    assert "eval_csv_sha256" in manifest
    assert manifest["dataset_row_count"] == 1


def test_manifest_init_git_failure(orchestrator, monkeypatch, tmp_path, capsys):
    """manifest-init при недоступном git → git_sha == "unknown", git_dirty == None.

    Мокаем subprocess.run так, чтобы оба git-вызова бросали FileNotFoundError
    (git не установлен). Ожидаем: exit 0 (не прерываем), JSON содержит
    git_sha == "unknown" и git_dirty == null (None в Python → null в JSON).
    """
    args, _, _ = _mk_manifest_init_args(tmp_path)

    def fake_subprocess_run(cmd, **kwargs):
        raise FileNotFoundError("git not found (mocked)")

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_subprocess_run)

    rc = orchestrator.cmd_manifest_init(args)

    assert rc == 0, f"manifest-init не должен падать при отсутствии git, got exit {rc}"

    captured = capsys.readouterr()
    manifest_path = pathlib.Path(captured.out.strip())
    assert manifest_path.exists()

    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["git_sha"] == "unknown", (
        f"При недоступном git git_sha должен быть 'unknown', got {manifest['git_sha']!r}"
    )
    assert manifest["git_dirty"] is None, (
        f"При недоступном git git_dirty должен быть null/None, got {manifest['git_dirty']!r}"
    )


def test_manifest_step_two_sequential_calls_preserve_both(
    orchestrator, tmp_path
):
    """Два последовательных вызова manifest-step → steps содержит оба элемента.

    Создаём минимальный манифест вручную, вызываем cmd_manifest_step дважды
    с разными именами шагов, проверяем, что оба шага сохранились и
    оригинальные поля манифеста не потёрты.
    """
    import json

    manifest_path = tmp_path / "training_run_test.json"
    initial_manifest = {
        "schema_version": 1,
        "started_at": "2025-01-01T00:00:00Z",
        "finished_at": None,
        "final_exit_code": None,
        "seed": 42,
        "steps": [],
    }
    manifest_path.write_text(
        json.dumps(initial_manifest, indent=2), encoding="utf-8"
    )

    # Первый вызов: шаг "train-leak-free"
    args_step1 = argparse.Namespace(
        manifest=manifest_path,
        name="train-leak-free",
        started_at="2025-01-01T00:01:00Z",
        finished_at="2025-01-01T00:10:00Z",
        exit_code=0,
        artifacts=["experimental/spam_model_leak_free.tflite"],
        gate_failed=None,
        skipped=None,
        skipped_reason=None,
    )
    rc1 = orchestrator.cmd_manifest_step(args_step1)
    assert rc1 == 0, f"Первый manifest-step должен вернуть 0, got {rc1}"

    # Второй вызов: шаг "train-binary"
    args_step2 = argparse.Namespace(
        manifest=manifest_path,
        name="train-binary",
        started_at="2025-01-01T00:11:00Z",
        finished_at="2025-01-01T00:20:00Z",
        exit_code=0,
        artifacts=["experimental/spam_model_binary.tflite"],
        gate_failed=None,
        skipped=None,
        skipped_reason=None,
    )
    rc2 = orchestrator.cmd_manifest_step(args_step2)
    assert rc2 == 0, f"Второй manifest-step должен вернуть 0, got {rc2}"

    # Проверяем результат.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Оригинальные поля не потёрты.
    assert manifest["schema_version"] == 1
    assert manifest["started_at"] == "2025-01-01T00:00:00Z"
    assert manifest["seed"] == 42

    # Оба шага присутствуют.
    steps = manifest["steps"]
    assert len(steps) == 2, f"Ожидалось 2 шага, got {len(steps)}"
    assert steps[0]["name"] == "train-leak-free"
    assert steps[0]["exit_code"] == 0
    assert steps[0]["artifact_paths"] == ["experimental/spam_model_leak_free.tflite"]
    assert steps[1]["name"] == "train-binary"
    assert steps[1]["exit_code"] == 0
    assert steps[1]["artifact_paths"] == ["experimental/spam_model_binary.tflite"]


def test_manifest_finalize_idempotent(orchestrator, tmp_path, capsys):
    """Повторный вызов manifest-finalize не дублирует finished_at / final_exit_code.

    Первый вызов записывает поля. Второй вызов обнаруживает, что манифест
    уже финализирован, и возвращает 0 без перезаписи. Проверяем, что
    значения первого вызова сохранились неизменными.
    """
    import json

    manifest_path = tmp_path / "training_run_test.json"
    initial_manifest = {
        "schema_version": 1,
        "started_at": "2025-01-01T00:00:00Z",
        "finished_at": None,
        "final_exit_code": None,
        "seed": 42,
        "steps": [
            {
                "name": "train-leak-free",
                "started_at": "2025-01-01T00:01:00Z",
                "finished_at": "2025-01-01T00:10:00Z",
                "exit_code": 0,
                "artifact_paths": [],
            }
        ],
    }
    manifest_path.write_text(
        json.dumps(initial_manifest, indent=2), encoding="utf-8"
    )

    # Первый вызов finalize.
    args_fin = argparse.Namespace(manifest=manifest_path, exit_code=0)
    rc1 = orchestrator.cmd_manifest_finalize(args_fin)
    assert rc1 == 0, f"Первый finalize должен вернуть 0, got {rc1}"

    manifest_after_first = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_after_first["finished_at"] is not None, (
        "finished_at должен быть заполнен после первого finalize"
    )
    assert manifest_after_first["final_exit_code"] == 0
    first_finished_at = manifest_after_first["finished_at"]

    # Второй вызов finalize (идемпотентность).
    rc2 = orchestrator.cmd_manifest_finalize(args_fin)
    assert rc2 == 0, f"Повторный finalize должен вернуть 0 (идемпотентно), got {rc2}"

    manifest_after_second = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_after_second["finished_at"] == first_finished_at, (
        "finished_at не должен меняться при повторном вызове finalize"
    )
    assert manifest_after_second["final_exit_code"] == 0, (
        "final_exit_code не должен меняться при повторном вызове finalize"
    )

    # Проверяем, что оригинальные поля не потёрты.
    assert manifest_after_second["schema_version"] == 1
    assert manifest_after_second["seed"] == 42
    assert len(manifest_after_second["steps"]) == 1

    # Проверяем stderr: при повторном вызове должно быть уведомление.
    err = capsys.readouterr().err
    assert "already finalized" in err, (
        f"При повторном finalize ожидалось уведомление 'already finalized' в stderr, got: {err!r}"
    )


# ---------------------------------------------------------------------------
# Подзадача 9.5: Unit-тесты summary (Requirements 6.3, 6.4, 6.5, 9.1–9.4)
# ---------------------------------------------------------------------------


def _mk_summary_args(
    experimental_dir: pathlib.Path,
    prod_model_card: pathlib.Path,
    manifest: pathlib.Path | None = None,
) -> argparse.Namespace:
    """Build argparse.Namespace for cmd_summary."""
    if manifest is None:
        manifest = experimental_dir / "dummy_manifest.json"
    return argparse.Namespace(
        experimental_dir=experimental_dir,
        prod_model_card=prod_model_card,
        manifest=manifest,
    )


def _write_eval_json(
    directory: pathlib.Path,
    filename: str,
    *,
    block_precision: float = 0.92,
    block_recall: float = 0.70,
    allow_fp_rate: float = 0.12,
    status: str = "PASS",
) -> None:
    """Write a minimal eval JSON file in the format expected by cmd_summary."""
    import json

    data = {
        "status": status,
        "metrics": {
            "block_precision": block_precision,
            "block_recall": block_recall,
            "allow_fp_rate": allow_fp_rate,
        },
    }
    path = directory / filename
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_prod_model_card(
    path: pathlib.Path,
    *,
    block_precision: float = 0.90,
    block_recall: float = 0.68,
    allow_fp_rate: float = 0.16,
) -> None:
    """Write a minimal prod model card JSON."""
    import json

    data = {
        "metrics": {
            "block_precision": block_precision,
            "block_recall": block_recall,
            "allow_fp_rate": allow_fp_rate,
        },
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class TestSummaryProdMetricsUnavailable:
    """Prod model card отсутствует → prod_metrics == 'unavailable', exit 0."""

    def test_prod_card_missing_prints_unavailable_and_exits_0(
        self, orchestrator, tmp_path, capsys
    ):
        """Если prod model card не существует, summary печатает 'unavailable'."""
        experimental_dir = tmp_path / "experimental"
        experimental_dir.mkdir()

        # Создаём eval JSON для leak_free с gate_passed = True
        _write_eval_json(experimental_dir, "eval_leak_free.json", status="PASS")
        _write_eval_json(experimental_dir, "eval_binary.json", status="PASS")

        prod_card_path = tmp_path / "model_card.json"
        # Не создаём файл — он отсутствует

        args = _mk_summary_args(experimental_dir, prod_card_path)
        rc = orchestrator.cmd_summary(args)

        assert rc == 0, f"summary должен вернуть 0, got {rc}"
        out = capsys.readouterr().out
        assert "unavailable" in out, (
            f"При отсутствующем prod model card stdout должен содержать 'unavailable', "
            f"got: {out!r}"
        )


class TestSummaryGateFailedNotRecommended:
    """gate_passed == false → модель помечена 'not recommended'."""

    def test_gate_failed_model_marked_not_recommended(
        self, orchestrator, tmp_path, capsys
    ):
        """Eval binary с gate_passed=false → в выводе 'not recommended'."""
        experimental_dir = tmp_path / "experimental"
        experimental_dir.mkdir()

        # leak_free проходит gate
        _write_eval_json(experimental_dir, "eval_leak_free.json", status="PASS")
        # binary НЕ проходит gate
        _write_eval_json(experimental_dir, "eval_binary.json", status="FAIL")

        prod_card_path = tmp_path / "model_card.json"
        _write_prod_model_card(prod_card_path)

        args = _mk_summary_args(experimental_dir, prod_card_path)
        rc = orchestrator.cmd_summary(args)

        assert rc == 0, f"summary должен вернуть 0, got {rc}"
        out = capsys.readouterr().out
        assert "not recommended" in out, (
            f"При gate_passed=false модель должна быть помечена 'not recommended', "
            f"got: {out!r}"
        )
        # Проверяем, что binary помечена как NO в таблице
        assert "NO" in out, (
            f"Таблица должна содержать 'NO' для не-eligible модели, got: {out!r}"
        )


class TestSummaryOsWindows:
    """os.name == 'nt' → команды промоушена используют Copy-Item."""

    def test_windows_uses_copy_item(
        self, orchestrator, monkeypatch, tmp_path, capsys
    ):
        """На Windows summary печатает Copy-Item для eligible моделей."""
        monkeypatch.setattr(os, "name", "nt")

        experimental_dir = tmp_path / "experimental"
        experimental_dir.mkdir()

        _write_eval_json(experimental_dir, "eval_leak_free.json", status="PASS")
        _write_eval_json(experimental_dir, "eval_binary.json", status="PASS")

        prod_card_path = tmp_path / "model_card.json"
        _write_prod_model_card(prod_card_path)

        args = _mk_summary_args(experimental_dir, prod_card_path)
        rc = orchestrator.cmd_summary(args)

        assert rc == 0, f"summary должен вернуть 0, got {rc}"
        out = capsys.readouterr().out
        assert "Copy-Item" in out, (
            f"На Windows (os.name=='nt') должна быть команда Copy-Item, got: {out!r}"
        )
        # Не должно быть unix-стиля cp (без Copy-Item контекста)
        # Проверяем, что Copy-Item присутствует для tflite
        assert "Copy-Item" in out and "-Force" in out, (
            f"Команда Copy-Item должна содержать -Force, got: {out!r}"
        )


class TestSummaryOsPosix:
    """os.name == 'posix' → команды промоушена используют cp."""

    def test_posix_uses_cp(
        self, orchestrator, monkeypatch, tmp_path, capsys
    ):
        """На POSIX summary печатает cp для eligible моделей."""
        monkeypatch.setattr(os, "name", "posix")

        experimental_dir = tmp_path / "experimental"
        experimental_dir.mkdir()

        _write_eval_json(experimental_dir, "eval_leak_free.json", status="PASS")
        _write_eval_json(experimental_dir, "eval_binary.json", status="PASS")

        prod_card_path = tmp_path / "model_card.json"
        _write_prod_model_card(prod_card_path)

        args = _mk_summary_args(experimental_dir, prod_card_path)
        rc = orchestrator.cmd_summary(args)

        assert rc == 0, f"summary должен вернуть 0, got {rc}"
        out = capsys.readouterr().out
        assert "cp " in out, (
            f"На POSIX (os.name=='posix') должна быть команда cp, got: {out!r}"
        )
        # Не должно быть Copy-Item
        assert "Copy-Item" not in out, (
            f"На POSIX не должно быть Copy-Item, got: {out!r}"
        )
