"""
Тесты эквивалентности обёрток: train_full_pipeline.sh и train_full_pipeline.ps1.

Парсит содержимое обоих файлов через regex и проверяет, что ключевые CLI-вызовы
присутствуют в обеих обёртках с одинаковыми флагами. Без фактического запуска скриптов.

Requirements: 1.2, 1.4
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BASH_WRAPPER = REPO_ROOT / "scripts" / "train_full_pipeline.sh"
PS_WRAPPER = REPO_ROOT / "scripts" / "train_full_pipeline.ps1"


@pytest.fixture
def bash_content() -> str:
    """Read the Bash wrapper content."""
    assert BASH_WRAPPER.is_file(), f"Bash wrapper not found: {BASH_WRAPPER}"
    return BASH_WRAPPER.read_text(encoding="utf-8")


@pytest.fixture
def ps_content() -> str:
    """Read the PowerShell wrapper content."""
    assert PS_WRAPPER.is_file(), f"PowerShell wrapper not found: {PS_WRAPPER}"
    return PS_WRAPPER.read_text(encoding="utf-8")


# ─── Helper: extract key invocations ─────────────────────────────────────────


def _has_kd_leak_free_with_seed(content: str) -> bool:
    """Check that content invokes train_kd_distillation.py with --leak-free and --seed."""
    # Allow multiline commands (backslash or backtick continuation)
    # We look for the script name, then --leak-free and --seed somewhere in the
    # same logical command block (within ~30 lines).
    pattern = r"train_kd_distillation\.py"
    matches = list(re.finditer(pattern, content))
    if not matches:
        return False
    for m in matches:
        # Grab a window of text after the match to check for flags
        window = content[m.start() : m.start() + 1500]
        has_leak_free = bool(re.search(r"--leak-free", window))
        has_seed = bool(re.search(r"--seed", window))
        if has_leak_free and has_seed:
            return True
    return False


def _has_binary_model_with_strategy_and_seed(content: str) -> bool:
    """Check that content invokes train_binary_model.py with --binary-warn-strategy merge_block and --seed."""
    pattern = r"train_binary_model\.py"
    matches = list(re.finditer(pattern, content))
    if not matches:
        return False
    for m in matches:
        window = content[m.start() : m.start() + 1500]
        has_strategy = bool(re.search(r"--binary-warn-strategy\s+merge_block", window))
        has_seed = bool(re.search(r"--seed", window))
        if has_strategy and has_seed:
            return True
    return False


def _has_eval_golden_set_cold_output_json(content: str) -> bool:
    """Check that content invokes eval_golden_set.py with --cold and --output-json."""
    pattern = r"eval_golden_set\.py"
    matches = list(re.finditer(pattern, content))
    if not matches:
        return False
    for m in matches:
        window = content[m.start() : m.start() + 1500]
        has_cold = bool(re.search(r"--cold", window))
        has_output_json = bool(re.search(r"--output-json", window))
        if has_cold and has_output_json:
            return True
    return False


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestKdDistillationInvocation:
    """Both wrappers must invoke train_kd_distillation.py --leak-free --seed."""

    def test_bash_has_kd_leak_free_seed(self, bash_content: str):
        assert _has_kd_leak_free_with_seed(bash_content), (
            "Bash wrapper missing: train_kd_distillation.py --leak-free --seed"
        )

    def test_ps_has_kd_leak_free_seed(self, ps_content: str):
        assert _has_kd_leak_free_with_seed(ps_content), (
            "PowerShell wrapper missing: train_kd_distillation.py --leak-free --seed"
        )


class TestBinaryModelInvocation:
    """Both wrappers must invoke train_binary_model.py --binary-warn-strategy merge_block --seed."""

    def test_bash_has_binary_strategy_seed(self, bash_content: str):
        assert _has_binary_model_with_strategy_and_seed(bash_content), (
            "Bash wrapper missing: train_binary_model.py --binary-warn-strategy merge_block --seed"
        )

    def test_ps_has_binary_strategy_seed(self, ps_content: str):
        assert _has_binary_model_with_strategy_and_seed(ps_content), (
            "PowerShell wrapper missing: train_binary_model.py --binary-warn-strategy merge_block --seed"
        )


class TestEvalGoldenSetInvocation:
    """Both wrappers must invoke eval_golden_set.py with --cold and --output-json."""

    def test_bash_has_eval_cold_output_json(self, bash_content: str):
        assert _has_eval_golden_set_cold_output_json(bash_content), (
            "Bash wrapper missing: eval_golden_set.py --cold --output-json"
        )

    def test_ps_has_eval_cold_output_json(self, ps_content: str):
        assert _has_eval_golden_set_cold_output_json(ps_content), (
            "PowerShell wrapper missing: eval_golden_set.py --cold --output-json"
        )


class TestEquivalentFlags:
    """Verify that both wrappers pass the same set of key flags to each tool."""

    def test_kd_same_hidden_sizes(self, bash_content: str, ps_content: str):
        """Both wrappers pass --hidden-sizes '128,96,48' to KD trainer."""
        pattern = r'--hidden-sizes\s+["\']?128,96,48["\']?'
        assert re.search(pattern, bash_content), "Bash missing --hidden-sizes 128,96,48"
        assert re.search(pattern, ps_content), "PS missing --hidden-sizes 128,96,48"

    def test_binary_same_hidden_sizes(self, bash_content: str, ps_content: str):
        """Both wrappers pass --hidden-sizes '128,96,48' to binary trainer."""
        # Find all binary model invocation sites and check any has hidden-sizes
        for label, content in [("Bash", bash_content), ("PS", ps_content)]:
            matches = list(re.finditer(r"train_binary_model\.py", content))
            assert matches, f"{label} missing train_binary_model.py invocation"
            found = False
            for m in matches:
                window = content[m.start() : m.start() + 1500]
                if re.search(r'--hidden-sizes\s+["\']?128,96,48["\']?', window):
                    found = True
                    break
            assert found, (
                f"{label} missing --hidden-sizes 128,96,48 in binary trainer call"
            )

    def test_eval_threshold_flags_present(self, bash_content: str, ps_content: str):
        """Both wrappers pass threshold flags to eval_golden_set.py."""
        for label, content in [("Bash", bash_content), ("PS", ps_content)]:
            matches = list(re.finditer(r"eval_golden_set\.py", content))
            assert matches, f"{label} missing eval_golden_set.py invocation"
            # Check at least one invocation has all threshold flags
            found = False
            for m in matches:
                window = content[m.start() : m.start() + 1500]
                has_precision = bool(re.search(r"--min-block-precision", window))
                has_recall = bool(re.search(r"--min-block-recall", window))
                has_fp = bool(re.search(r"--max-allow-fp-rate", window))
                if has_precision and has_recall and has_fp:
                    found = True
                    break
            assert found, (
                f"{label} missing threshold flags (--min-block-precision, "
                f"--min-block-recall, --max-allow-fp-rate) in eval_golden_set.py call"
            )

    def test_seed_default_42(self, bash_content: str, ps_content: str):
        """Both wrappers default seed to 42."""
        # Bash: SEED=42 or --seed 42
        assert re.search(r"SEED=42|--seed\s+42", bash_content), (
            "Bash wrapper does not default seed to 42"
        )
        # PowerShell: $Seed = 42 in param block
        assert re.search(r"\$Seed\s*=\s*42", ps_content), (
            "PowerShell wrapper does not default seed to 42"
        )
