"""Tests for the App Category training pipeline.

Covers:
  - Module importability (task 1.1 sanity check)
  - CharNGramVocab (task 7.1, Requirements 2.2, 2.8)
  - encode_dataset (task 7.1, Requirements 2.2, 2.8)
  - compare_enum_order (task 7.9, Requirement 2.10)
  - write_atomic (task 7.9, Requirement 2.10)
  - evaluate() function (task 7.5, Requirement 2.4)
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


def test_train_app_category_classifier_module_importable() -> None:
    """Skeleton sanity check: the training module loads."""
    module = importlib.import_module("train_app_category_classifier")
    assert hasattr(module, "CATEGORIES"), "CATEGORIES list missing"
    assert hasattr(module, "KOTLIN_APP_CATEGORY_ORDER"), (
        "KOTLIN_APP_CATEGORY_ORDER must be committed in task 1.1 as the "
        "single source of truth for the Python side"
    )


# ── Tests for CharNGramVocab (task 7.1, Requirements 2.2, 2.8) ─────────────

from train_app_category_classifier import (
    CharNGramVocab,
    PAD_ID,
    UNK_ID,
    encode_dataset,
)


class TestCharNGramVocabInit:
    """Unit tests for CharNGramVocab.__init__ validation."""

    def test_valid_minimal_vocab(self) -> None:
        """Minimal valid vocab with just PAD and UNK."""
        vocab = CharNGramVocab(["<PAD>", "<UNK>"])
        assert vocab.tokens == ["<PAD>", "<UNK>"]
        assert vocab.token_to_id == {"<PAD>": 0, "<UNK>": 1}

    def test_valid_vocab_with_ngrams(self) -> None:
        """Vocab with real n-gram tokens."""
        tokens = ["<PAD>", "<UNK>", "com", "org", "app"]
        vocab = CharNGramVocab(tokens)
        assert len(vocab.tokens) == 5
        assert vocab.token_to_id["com"] == 2
        assert vocab.token_to_id["org"] == 3
        assert vocab.token_to_id["app"] == 4

    def test_rejects_non_list(self) -> None:
        """Non-list input raises TypeError."""
        with pytest.raises(TypeError, match="must be a list"):
            CharNGramVocab(("<PAD>", "<UNK>"))  # type: ignore[arg-type]

    def test_rejects_too_short(self) -> None:
        """Less than 2 tokens raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 tokens"):
            CharNGramVocab(["<PAD>"])

    def test_rejects_wrong_pad(self) -> None:
        """tokens[0] != '<PAD>' raises ValueError."""
        with pytest.raises(ValueError, match="tokens\\[0\\] must be '<PAD>'"):
            CharNGramVocab(["<UNK>", "<PAD>"])

    def test_rejects_wrong_unk(self) -> None:
        """tokens[1] != '<UNK>' raises ValueError."""
        with pytest.raises(ValueError, match="tokens\\[1\\] must be '<UNK>'"):
            CharNGramVocab(["<PAD>", "com"])

    def test_rejects_empty_token(self) -> None:
        """Empty string token raises ValueError."""
        with pytest.raises(ValueError, match="is empty"):
            CharNGramVocab(["<PAD>", "<UNK>", ""])

    def test_rejects_duplicate_tokens(self) -> None:
        """Duplicate tokens raise ValueError."""
        with pytest.raises(ValueError, match="duplicate"):
            CharNGramVocab(["<PAD>", "<UNK>", "com", "com"])

    def test_rejects_non_str_token(self) -> None:
        """Non-string token raises TypeError."""
        with pytest.raises(TypeError, match="must be a str"):
            CharNGramVocab(["<PAD>", "<UNK>", 123])  # type: ignore[list-item]


class TestCharNGramVocabBuild:
    """Unit tests for CharNGramVocab.build."""

    def test_build_basic(self) -> None:
        """Build from simple rows produces valid vocab."""
        rows = [("com.example.app", "Example")]
        vocab = CharNGramVocab.build(rows, n_grams=(3,), max_size=10)
        assert vocab.tokens[0] == "<PAD>"
        assert vocab.tokens[1] == "<UNK>"
        assert len(vocab.tokens) <= 10

    def test_build_respects_max_size(self) -> None:
        """Vocab size never exceeds max_size."""
        rows = [
            ("com.example.app", "Example App"),
            ("org.another.package", "Another"),
            ("net.third.thing", "Third Thing"),
        ]
        vocab = CharNGramVocab.build(rows, n_grams=(3, 4, 5), max_size=5)
        assert len(vocab.tokens) <= 5

    def test_build_deterministic(self) -> None:
        """Same input produces identical vocab (Requirement 2.1)."""
        rows = [
            ("com.sberbank.online", "Сбербанк"),
            ("ru.gosuslugi.mobile", "Госуслуги"),
            ("com.google.android.gm", "Gmail"),
        ]
        v1 = CharNGramVocab.build(rows, n_grams=(3, 4, 5), max_size=100)
        v2 = CharNGramVocab.build(rows, n_grams=(3, 4, 5), max_size=100)
        assert v1.tokens == v2.tokens

    def test_build_handles_none_label(self) -> None:
        """Rows with None label don't crash."""
        rows = [("com.example.app", None)]
        vocab = CharNGramVocab.build(rows, n_grams=(3,), max_size=10)
        assert vocab.tokens[0] == "<PAD>"

    def test_build_handles_empty_label(self) -> None:
        """Rows with empty label don't crash."""
        rows = [("com.example.app", "")]
        vocab = CharNGramVocab.build(rows, n_grams=(3,), max_size=10)
        assert vocab.tokens[0] == "<PAD>"

    def test_build_rejects_max_size_below_2(self) -> None:
        """max_size < 2 raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be >= 2"):
            CharNGramVocab.build([("a", "b")], max_size=1)

    def test_build_rejects_empty_n_grams(self) -> None:
        """Empty n_grams tuple raises ValueError."""
        with pytest.raises(ValueError, match="n_grams must be a non-empty"):
            CharNGramVocab.build([("a", "b")], n_grams=())

    def test_build_rejects_non_positive_n(self) -> None:
        """n_grams with zero or negative raises ValueError."""
        with pytest.raises(ValueError, match="positive ints"):
            CharNGramVocab.build([("a", "b")], n_grams=(0,))

    def test_build_frequency_ordering(self) -> None:
        """Most frequent n-grams appear first after sentinels."""
        # "aaa" repeated many times should dominate
        rows = [("aaa", "aaa")] * 10 + [("xyz", "xyz")]
        vocab = CharNGramVocab.build(rows, n_grams=(3,), max_size=10)
        # "aaa" should be the most frequent 3-gram from "aaa aaa"
        # The text is "aaa aaa" → 3-grams: "aaa", "aa ", "a a", " aa", "aaa"
        # "aaa" appears twice per row × 10 rows = 20 times
        assert "aaa" in vocab.tokens
        # It should be right after the sentinels (highest frequency)
        assert vocab.tokens[2] == "aaa"

    def test_build_alphabetic_tiebreak(self) -> None:
        """Ties in frequency are broken alphabetically ascending."""
        # Create rows where "abc" and "abd" appear exactly once each
        rows = [("abcabd", "")]
        vocab = CharNGramVocab.build(rows, n_grams=(3,), max_size=10)
        # Both "abc" and "abd" appear once; "abc" < "abd" alphabetically
        abc_idx = vocab.tokens.index("abc")
        abd_idx = vocab.tokens.index("abd")
        assert abc_idx < abd_idx


class TestCharNGramVocabEncode:
    """Unit tests for CharNGramVocab.encode."""

    @pytest.fixture
    def vocab(self) -> CharNGramVocab:
        """Build a small vocab for encode tests."""
        rows = [
            ("com.example.app", "Example"),
            ("org.test.pkg", "Test App"),
        ]
        return CharNGramVocab.build(rows, n_grams=(3, 4, 5), max_size=100)

    def test_encode_returns_correct_length(self, vocab) -> None:
        """Encode always returns array of length max_len."""
        result = vocab.encode("com.example.app", "Example", max_len=64)
        assert result.shape == (64,)

    def test_encode_custom_max_len(self, vocab) -> None:
        """Custom max_len is respected."""
        result = vocab.encode("com.example.app", "Example", max_len=32)
        assert result.shape == (32,)

    def test_encode_dtype_int32(self, vocab) -> None:
        """Encode returns int32 dtype."""
        result = vocab.encode("com.example.app", "Example")
        assert result.dtype == np.int32

    def test_encode_pad_id_for_short_input(self, vocab) -> None:
        """Short inputs are right-padded with PAD_ID."""
        result = vocab.encode("ab", None, max_len=64)
        # "ab" is too short for any 3-gram, so all ids should be PAD
        assert all(v == PAD_ID for v in result)

    def test_encode_known_ngrams_not_unk(self, vocab) -> None:
        """Known n-grams from training data get their real ids."""
        result = vocab.encode("com.example.app", "Example")
        # At least some tokens should not be UNK or PAD
        non_special = [v for v in result if v != PAD_ID and v != UNK_ID]
        assert len(non_special) > 0

    def test_encode_unknown_ngrams_get_unk_id(self) -> None:
        """N-grams not in vocab get UNK_ID."""
        # Build a tiny vocab that only knows "aaa"
        vocab = CharNGramVocab(["<PAD>", "<UNK>", "aaa"])
        result = vocab.encode("xyz", None, max_len=10)
        # "xyz" has one 3-gram "xyz" which is not in vocab → UNK
        assert result[0] == UNK_ID

    def test_encode_none_label(self, vocab) -> None:
        """None label doesn't crash and uses packageName only."""
        result = vocab.encode("com.example.app", None)
        assert result.shape == (64,)

    def test_encode_empty_label(self, vocab) -> None:
        """Empty string label treated same as None."""
        r1 = vocab.encode("com.example.app", None)
        r2 = vocab.encode("com.example.app", "")
        np.testing.assert_array_equal(r1, r2)

    def test_encode_truncates_long_input(self) -> None:
        """Long inputs are truncated to max_len."""
        rows = [("a" * 200, "b" * 200)]
        vocab = CharNGramVocab.build(rows, n_grams=(3, 4, 5), max_size=500)
        result = vocab.encode("a" * 200, "b" * 200, max_len=10)
        assert result.shape == (10,)
        # No PAD_ID should appear since input is long enough
        assert PAD_ID not in result

    def test_encode_rejects_non_str_packagename(self, vocab) -> None:
        """Non-string packageName raises TypeError."""
        with pytest.raises(TypeError, match="packageName must be a str"):
            vocab.encode(123, None)  # type: ignore[arg-type]

    def test_encode_rejects_non_str_label(self, vocab) -> None:
        """Non-string, non-None label raises TypeError."""
        with pytest.raises(TypeError, match="label must be a str or None"):
            vocab.encode("com.example", 123)  # type: ignore[arg-type]

    def test_encode_rejects_non_positive_max_len(self, vocab) -> None:
        """max_len <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_len must be positive"):
            vocab.encode("com.example", None, max_len=0)

    def test_encode_ngram_order_3_then_4_then_5(self) -> None:
        """N-grams are concatenated in order n=3, n=4, n=5."""
        # Build vocab that knows specific n-grams of different sizes
        tokens = ["<PAD>", "<UNK>", "abc", "abcd", "abcde"]
        vocab = CharNGramVocab(tokens)
        # Input "abcde" has:
        #   3-grams: abc, bcd, cde → ids: 2, UNK, UNK
        #   4-grams: abcd, bcde → ids: 3, UNK
        #   5-grams: abcde → ids: 4
        result = vocab.encode("abcde", None, max_len=10)
        # First come 3-grams, then 4-grams, then 5-grams
        assert result[0] == 2  # "abc" → id 2
        assert result[1] == UNK_ID  # "bcd" → UNK
        assert result[2] == UNK_ID  # "cde" → UNK
        assert result[3] == 3  # "abcd" → id 3
        assert result[4] == UNK_ID  # "bcde" → UNK
        assert result[5] == 4  # "abcde" → id 4
        # Rest is PAD
        assert result[6] == PAD_ID


class TestCharNGramVocabSerialize:
    """Unit tests for CharNGramVocab.serialize."""

    def test_serialize_format(self) -> None:
        """Serialize produces one token per line with trailing newline."""
        vocab = CharNGramVocab(["<PAD>", "<UNK>", "com", "org"])
        result = vocab.serialize()
        assert result == "<PAD>\n<UNK>\ncom\norg\n"

    def test_serialize_pad_on_line_0(self) -> None:
        """<PAD> is always on line 0."""
        vocab = CharNGramVocab(["<PAD>", "<UNK>", "abc"])
        lines = vocab.serialize().split("\n")
        assert lines[0] == "<PAD>"

    def test_serialize_unk_on_line_1(self) -> None:
        """<UNK> is always on line 1."""
        vocab = CharNGramVocab(["<PAD>", "<UNK>", "abc"])
        lines = vocab.serialize().split("\n")
        assert lines[1] == "<UNK>"

    def test_serialize_trailing_newline(self) -> None:
        """Serialized string ends with newline."""
        vocab = CharNGramVocab(["<PAD>", "<UNK>"])
        result = vocab.serialize()
        assert result.endswith("\n")

    def test_serialize_no_empty_lines(self) -> None:
        """No empty lines in serialized output (except trailing split artifact)."""
        vocab = CharNGramVocab(["<PAD>", "<UNK>", "abc", "def", "ghi"])
        result = vocab.serialize()
        # Split by \n gives tokens + one empty string at end from trailing \n
        lines = result.split("\n")
        # All lines except the last (empty from trailing \n) should be non-empty
        for line in lines[:-1]:
            assert line != ""

    def test_serialize_ascending_id_order(self) -> None:
        """Tokens are in ascending id order (line k = token id k)."""
        tokens = ["<PAD>", "<UNK>", "aaa", "bbb", "ccc"]
        vocab = CharNGramVocab(tokens)
        result = vocab.serialize()
        lines = result.rstrip("\n").split("\n")
        assert lines == tokens

    def test_serialize_roundtrip(self) -> None:
        """Serialized vocab can be parsed back to the same token list."""
        tokens = ["<PAD>", "<UNK>", "com", "org", "net"]
        vocab = CharNGramVocab(tokens)
        serialized = vocab.serialize()
        # Parse back
        parsed_tokens = serialized.rstrip("\n").split("\n")
        assert parsed_tokens == tokens
        # Reconstruct vocab
        vocab2 = CharNGramVocab(parsed_tokens)
        assert vocab2.tokens == vocab.tokens
        assert vocab2.token_to_id == vocab.token_to_id


class TestEncodeDataset:
    """Unit tests for encode_dataset (task 7.1, Requirements 2.2, 2.8)."""

    @pytest.fixture
    def vocab(self) -> CharNGramVocab:
        """Build a vocab for encode_dataset tests."""
        rows = [
            ("com.sberbank.online", "Сбербанк"),
            ("ru.gosuslugi.mobile", "Госуслуги"),
            ("com.google.android.gm", "Gmail"),
            ("org.telegram.messenger", "Telegram"),
        ]
        return CharNGramVocab.build(rows, n_grams=(3, 4, 5), max_size=200)

    @pytest.fixture(autouse=True)
    def _require_tensorflow(self):
        pytest.importorskip("tensorflow")

    def test_encode_dataset_returns_tf_dataset(self, vocab) -> None:
        """encode_dataset returns a tf.data.Dataset."""
        import tensorflow as tf

        df = _FakeDataFrame([
            {"packageName": "com.sberbank.online", "label": "Сбербанк", "category": "BANK"},
        ])
        ds = encode_dataset(df, vocab, batch_size=1)
        assert isinstance(ds, tf.data.Dataset)

    def test_encode_dataset_excludes_other(self, vocab) -> None:
        """Rows with category=OTHER are excluded."""

        df = _FakeDataFrame([
            {"packageName": "com.sberbank.online", "label": "Сбербанк", "category": "BANK"},
            {"packageName": "com.unknown.app", "label": "Unknown", "category": "OTHER"},
        ])
        ds = encode_dataset(df, vocab, batch_size=10)
        # Should only have 1 row (BANK), not 2
        count = 0
        for features, labels in ds:
            count += features.shape[0]
        assert count == 1

    def test_encode_dataset_correct_label_ids(self, vocab) -> None:
        """Label ids match KOTLIN_APP_CATEGORY_ORDER index."""
        from train_app_category_classifier import KOTLIN_APP_CATEGORY_ORDER

        df = _FakeDataFrame([
            {"packageName": "com.sberbank.online", "label": "Сбербанк", "category": "BANK"},
            {"packageName": "org.telegram.messenger", "label": "Telegram", "category": "MESSENGER"},
        ])
        ds = encode_dataset(df, vocab, batch_size=10)
        for features, labels in ds:
            label_list = labels.numpy().tolist()
            assert KOTLIN_APP_CATEGORY_ORDER.index("BANK") in label_list
            assert KOTLIN_APP_CATEGORY_ORDER.index("MESSENGER") in label_list

    def test_encode_dataset_feature_shape(self, vocab) -> None:
        """Features have shape [batch, 64] (default max_len)."""
        df = _FakeDataFrame([
            {"packageName": "com.sberbank.online", "label": "Сбербанк", "category": "BANK"},
            {"packageName": "org.telegram.messenger", "label": "Telegram", "category": "MESSENGER"},
        ])
        ds = encode_dataset(df, vocab, batch_size=10)
        for features, labels in ds:
            assert features.shape[1] == 64

    def test_encode_dataset_handles_missing_label_column(self, vocab) -> None:
        """Works when DataFrame has no 'label' column."""
        df = _FakeDataFrame([
            {"packageName": "com.sberbank.online", "category": "BANK"},
        ])
        ds = encode_dataset(df, vocab, batch_size=1)
        count = 0
        for features, labels in ds:
            count += features.shape[0]
        assert count == 1

    def test_encode_dataset_handles_nan_label(self, vocab) -> None:
        """NaN label values are treated as empty string."""
        df = _FakeDataFrame([
            {"packageName": "com.sberbank.online", "label": float("nan"), "category": "BANK"},
        ])
        ds = encode_dataset(df, vocab, batch_size=1)
        count = 0
        for features, labels in ds:
            count += features.shape[0]
        assert count == 1

    def test_encode_dataset_empty_after_filtering(self, vocab) -> None:
        """All-OTHER DataFrame produces empty dataset with correct shape."""
        df = _FakeDataFrame([
            {"packageName": "com.unknown.app", "label": "Unknown", "category": "OTHER"},
        ])
        ds = encode_dataset(df, vocab, batch_size=1)
        count = 0
        for features, labels in ds:
            count += features.shape[0]
            # Even empty, features should have 64 columns
            assert features.shape[1] == 64
        assert count == 0

    def test_encode_dataset_rejects_non_int_batch_size(self, vocab) -> None:
        """Non-int batch_size raises TypeError."""
        df = _FakeDataFrame([
            {"packageName": "com.example", "label": "", "category": "BANK"},
        ])
        with pytest.raises(TypeError, match="batch_size must be an int"):
            encode_dataset(df, vocab, batch_size=1.5)  # type: ignore[arg-type]

    def test_encode_dataset_rejects_non_positive_batch_size(self, vocab) -> None:
        """batch_size <= 0 raises ValueError."""
        df = _FakeDataFrame([
            {"packageName": "com.example", "label": "", "category": "BANK"},
        ])
        with pytest.raises(ValueError, match="batch_size must be positive"):
            encode_dataset(df, vocab, batch_size=0)

    def test_encode_dataset_rejects_wrong_vocab_type(self) -> None:
        """Non-CharNGramVocab raises TypeError."""
        df = _FakeDataFrame([
            {"packageName": "com.example", "label": "", "category": "BANK"},
        ])
        with pytest.raises(TypeError, match="vocab must be a CharNGramVocab"):
            encode_dataset(df, "not_a_vocab", batch_size=1)  # type: ignore[arg-type]


# ── Tests for compare_enum_order (task 7.9, Requirement 2.10) ──────────────

from train_app_category_classifier import (
    ENUM_ORDER_MISSING_SENTINEL,
    compare_enum_order,
    write_atomic,
)


class TestCompareEnumOrder:
    """Unit tests for compare_enum_order."""

    def test_identical_lists_returns_none(self) -> None:
        """Identical lists should return None (no divergence)."""
        a = ["BANK", "INVESTMENTS", "GOVERNMENT"]
        b = ["BANK", "INVESTMENTS", "GOVERNMENT"]
        assert compare_enum_order(a, b) is None

    def test_empty_lists_returns_none(self) -> None:
        """Two empty lists are identical."""
        assert compare_enum_order([], []) is None

    def test_single_element_identical(self) -> None:
        assert compare_enum_order(["BANK"], ["BANK"]) is None

    def test_first_element_differs(self) -> None:
        """Divergence at index 0."""
        result = compare_enum_order(["BANK", "EMAIL"], ["GAMES", "EMAIL"])
        assert result == (0, "GAMES", "BANK")

    def test_middle_element_differs(self) -> None:
        """Divergence at a middle index."""
        result = compare_enum_order(
            ["BANK", "INVESTMENTS", "GOVERNMENT"],
            ["BANK", "INVESTMENTS", "EMAIL"],
        )
        assert result == (2, "EMAIL", "GOVERNMENT")

    def test_python_list_longer(self) -> None:
        """Python list has extra entries beyond kotlin list length."""
        result = compare_enum_order(
            ["BANK", "INVESTMENTS", "GOVERNMENT"],
            ["BANK", "INVESTMENTS"],
        )
        assert result == (2, ENUM_ORDER_MISSING_SENTINEL, "GOVERNMENT")

    def test_kotlin_list_longer(self) -> None:
        """Kotlin list has extra entries beyond python list length."""
        result = compare_enum_order(
            ["BANK", "INVESTMENTS"],
            ["BANK", "INVESTMENTS", "GOVERNMENT"],
        )
        assert result == (2, "GOVERNMENT", ENUM_ORDER_MISSING_SENTINEL)

    def test_case_sensitive(self) -> None:
        """Comparison is case-sensitive."""
        result = compare_enum_order(["bank"], ["BANK"])
        assert result == (0, "BANK", "bank")

    def test_full_kotlin_order_matches_itself(self) -> None:
        """The canonical KOTLIN_APP_CATEGORY_ORDER matches itself."""
        from train_app_category_classifier import KOTLIN_APP_CATEGORY_ORDER

        assert compare_enum_order(
            KOTLIN_APP_CATEGORY_ORDER, list(KOTLIN_APP_CATEGORY_ORDER)
        ) is None

    def test_does_not_mutate_inputs(self) -> None:
        """Function should not mutate the input lists."""
        a = ["BANK", "EMAIL"]
        b = ["BANK", "GAMES"]
        a_copy = list(a)
        b_copy = list(b)
        compare_enum_order(a, b)
        assert a == a_copy
        assert b == b_copy


# ── Tests for write_atomic (task 7.9, Requirement 2.10) ────────────────────


class TestWriteAtomic:
    """Unit tests for write_atomic."""

    def test_write_str_content(self, tmp_path: Path) -> None:
        """String content is written as UTF-8 with LF line endings."""
        target = tmp_path / "output.txt"
        write_atomic(target, "hello\nworld\n")
        raw = target.read_bytes()
        assert raw == b"hello\nworld\n"
        # No BOM
        assert not raw.startswith(b"\xef\xbb\xbf")

    def test_write_bytes_content(self, tmp_path: Path) -> None:
        """Bytes content is written verbatim in binary mode."""
        target = tmp_path / "output.bin"
        data = b"\x00\x01\x02\xff"
        write_atomic(target, data)
        assert target.read_bytes() == data

    def test_atomic_replaces_existing_file(self, tmp_path: Path) -> None:
        """Existing file at the target path is atomically replaced."""
        target = tmp_path / "output.txt"
        target.write_text("old content", encoding="utf-8")
        write_atomic(target, "new content")
        assert target.read_text(encoding="utf-8") == "new content"

    def test_no_tmp_file_left_behind(self, tmp_path: Path) -> None:
        """The .tmp staging file should not remain after write."""
        target = tmp_path / "output.txt"
        write_atomic(target, "data")
        tmp_file = target.with_suffix(target.suffix + ".tmp")
        assert not tmp_file.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Parent directories are created on demand."""
        target = tmp_path / "sub" / "dir" / "output.txt"
        write_atomic(target, "nested")
        assert target.read_text(encoding="utf-8") == "nested"

    def test_str_no_crlf_on_windows(self, tmp_path: Path) -> None:
        """LF newlines are preserved (no CRLF translation)."""
        target = tmp_path / "lf.txt"
        write_atomic(target, "line1\nline2\n")
        raw = target.read_bytes()
        assert b"\r\n" not in raw
        assert raw == b"line1\nline2\n"

    def test_unicode_str_content(self, tmp_path: Path) -> None:
        """Unicode string content is written as UTF-8."""
        target = tmp_path / "unicode.txt"
        content = "Привет мир\n日本語\n"
        write_atomic(target, content)
        assert target.read_text(encoding="utf-8") == content

    def test_bytearray_content(self, tmp_path: Path) -> None:
        """bytearray is accepted as bytes-like content."""
        target = tmp_path / "output.bin"
        data = bytearray(b"\xde\xad\xbe\xef")
        write_atomic(target, data)
        assert target.read_bytes() == bytes(data)

    def test_invalid_content_type_raises(self, tmp_path: Path) -> None:
        """Non-str/bytes content raises TypeError."""
        target = tmp_path / "output.txt"
        with pytest.raises(TypeError, match="content must be bytes or str"):
            write_atomic(target, 12345)  # type: ignore[arg-type]

    def test_path_coercion_from_str(self, tmp_path: Path) -> None:
        """String path argument is coerced to Path."""
        target = str(tmp_path / "coerced.txt")
        write_atomic(target, "works")  # type: ignore[arg-type]
        assert Path(target).read_text(encoding="utf-8") == "works"


# ── Tests for evaluate() (task 7.5, Requirement 2.4) ──────────────────────


class _FakeDataFrame:
    """Minimal DataFrame-like object for testing evaluate()."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.columns = list(rows[0].keys()) if rows else []

    def itertuples(self, index: bool = True):
        from collections import namedtuple

        if not self._rows:
            return iter([])
        Row = namedtuple("Row", self.columns)  # type: ignore[misc]
        return iter(Row(**r) for r in self._rows)


def _make_perfect_model(test_rows: list[dict], vocab_obj):
    """Return a model that always predicts the correct category.

    The model's `predict` method returns a one-hot softmax vector for
    the correct class based on pre-computed encodings.
    """
    from train_app_category_classifier import KOTLIN_APP_CATEGORY_ORDER

    class PerfectModel:
        def __init__(self):
            self._lookup: dict[bytes, int] = {}
            for row in test_rows:
                cat = row["category"]
                if cat == "OTHER":
                    continue
                pkg = row["packageName"]
                label = row.get("label", "")
                if not isinstance(label, str):
                    label = ""
                encoded = vocab_obj.encode(pkg, label)
                key = encoded.tobytes()
                label_id = KOTLIN_APP_CATEGORY_ORDER.index(cat)
                self._lookup[key] = label_id

        def predict(self, X, verbose=0):
            n = X.shape[0]
            probs = np.zeros((n, 18), dtype=np.float32)
            for i in range(n):
                key = X[i].tobytes()
                label_id = self._lookup.get(key, 0)
                probs[i, label_id] = 1.0
            return probs

    return PerfectModel()


def _make_random_model(seed: int = 123):
    """Return a model that returns random softmax predictions."""

    class RandomModel:
        def predict(self, X, verbose=0):
            rng = np.random.default_rng(seed)
            n = X.shape[0]
            logits = rng.standard_normal((n, 18)).astype(np.float32)
            # softmax
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return RandomModel()


@pytest.fixture
def small_vocab():
    """Build a tiny CharNGramVocab from a few sample packages."""
    from train_app_category_classifier import CharNGramVocab

    rows = [
        ("com.sberbank.online", "Сбербанк"),
        ("ru.gosuslugi.mobile", "Госуслуги"),
        ("com.google.android.gm", "Gmail"),
        ("org.telegram.messenger", "Telegram"),
        ("com.example.game", "Fun Game"),
    ]
    return CharNGramVocab.build(rows, n_grams=(3, 4, 5), max_size=200)


@pytest.fixture
def synthetic_test_rows():
    """A tiny synthetic test set with known categories."""
    return [
        {"packageName": "com.sberbank.online", "label": "Сбербанк", "category": "BANK"},
        {"packageName": "ru.vtb24.mobile", "label": "ВТБ", "category": "BANK"},
        {"packageName": "ru.gosuslugi.mobile", "label": "Госуслуги", "category": "GOVERNMENT"},
        {"packageName": "com.google.android.gm", "label": "Gmail", "category": "EMAIL"},
        {"packageName": "org.telegram.messenger", "label": "Telegram", "category": "MESSENGER"},
        {"packageName": "com.example.game", "label": "Fun Game", "category": "GAMES"},
    ]


class TestEvaluate:
    """Tests for evaluate() — task 7.5, Requirement 2.4."""

    def test_returns_required_keys(self, small_vocab, synthetic_test_rows):
        """evaluate() returns dict with top1_accuracy, macro_f1, per_category."""
        from train_app_category_classifier import evaluate

        model = _make_random_model()
        test_df = _FakeDataFrame(synthetic_test_rows)
        result = evaluate(model, test_df, small_vocab)

        assert "top1_accuracy" in result
        assert "macro_f1" in result
        assert "per_category" in result

    def test_per_category_has_18_keys(self, small_vocab, synthetic_test_rows):
        """per_category must have exactly 18 keys (BANK..PRODUCTIVITY, no OTHER)."""
        from train_app_category_classifier import (
            KOTLIN_APP_CATEGORY_ORDER,
            evaluate,
        )

        model = _make_random_model()
        test_df = _FakeDataFrame(synthetic_test_rows)
        result = evaluate(model, test_df, small_vocab)

        per_cat = result["per_category"]
        assert len(per_cat) == 18
        expected_keys = set(KOTLIN_APP_CATEGORY_ORDER[:18])
        assert set(per_cat.keys()) == expected_keys

    def test_per_category_entries_have_precision_recall_f1(
        self, small_vocab, synthetic_test_rows
    ):
        """Each per_category entry must have precision, recall, f1 floats."""
        from train_app_category_classifier import evaluate

        model = _make_random_model()
        test_df = _FakeDataFrame(synthetic_test_rows)
        result = evaluate(model, test_df, small_vocab)

        for cat, metrics in result["per_category"].items():
            assert "precision" in metrics, f"{cat} missing precision"
            assert "recall" in metrics, f"{cat} missing recall"
            assert "f1" in metrics, f"{cat} missing f1"
            assert 0.0 <= metrics["precision"] <= 1.0, f"{cat} precision out of range"
            assert 0.0 <= metrics["recall"] <= 1.0, f"{cat} recall out of range"
            assert 0.0 <= metrics["f1"] <= 1.0, f"{cat} f1 out of range"

    def test_perfect_model_gives_accuracy_1(self, small_vocab, synthetic_test_rows):
        """A model that always predicts correctly should yield accuracy=1.0."""
        from train_app_category_classifier import evaluate

        model = _make_perfect_model(synthetic_test_rows, small_vocab)
        test_df = _FakeDataFrame(synthetic_test_rows)
        result = evaluate(model, test_df, small_vocab)

        assert result["top1_accuracy"] == 1.0

    def test_perfect_model_gives_f1_1_for_present_categories(
        self, small_vocab, synthetic_test_rows
    ):
        """A perfect model should have F1=1.0 for categories with support."""
        from train_app_category_classifier import evaluate

        model = _make_perfect_model(synthetic_test_rows, small_vocab)
        test_df = _FakeDataFrame(synthetic_test_rows)
        result = evaluate(model, test_df, small_vocab)

        # Categories present in the test set should have perfect metrics.
        present_cats = {"BANK", "GOVERNMENT", "EMAIL", "MESSENGER", "GAMES"}
        for cat in present_cats:
            assert result["per_category"][cat]["precision"] == 1.0, (
                f"{cat} precision should be 1.0"
            )
            assert result["per_category"][cat]["recall"] == 1.0, (
                f"{cat} recall should be 1.0"
            )
            assert result["per_category"][cat]["f1"] == 1.0, (
                f"{cat} f1 should be 1.0"
            )

    def test_accuracy_in_valid_range(self, small_vocab, synthetic_test_rows):
        """top1_accuracy must be in [0, 1]."""
        from train_app_category_classifier import evaluate

        model = _make_random_model()
        test_df = _FakeDataFrame(synthetic_test_rows)
        result = evaluate(model, test_df, small_vocab)

        assert 0.0 <= result["top1_accuracy"] <= 1.0

    def test_macro_f1_in_valid_range(self, small_vocab, synthetic_test_rows):
        """macro_f1 must be in [0, 1]."""
        from train_app_category_classifier import evaluate

        model = _make_random_model()
        test_df = _FakeDataFrame(synthetic_test_rows)
        result = evaluate(model, test_df, small_vocab)

        assert 0.0 <= result["macro_f1"] <= 1.0

    def test_other_category_excluded(self, small_vocab):
        """Rows with category=OTHER should be excluded from evaluation."""
        from train_app_category_classifier import evaluate

        rows = [
            {"packageName": "com.sberbank.online", "label": "Сбербанк", "category": "BANK"},
            {"packageName": "com.unknown.app", "label": "Unknown", "category": "OTHER"},
        ]
        model = _make_perfect_model(rows, small_vocab)
        test_df = _FakeDataFrame(rows)
        result = evaluate(model, test_df, small_vocab)

        # Only 1 row should be evaluated (the BANK one), OTHER is excluded.
        assert result["top1_accuracy"] == 1.0

    def test_empty_test_set_returns_zero_metrics(self, small_vocab):
        """An empty test set (all OTHER) should return zero metrics."""
        from train_app_category_classifier import evaluate

        rows = [
            {"packageName": "com.unknown.app", "label": "Unknown", "category": "OTHER"},
        ]
        model = _make_random_model()
        test_df = _FakeDataFrame(rows)
        result = evaluate(model, test_df, small_vocab)

        assert result["top1_accuracy"] == 0.0
        assert result["macro_f1"] == 0.0
        assert len(result["per_category"]) == 18

    def test_missing_label_column_handled(self, small_vocab):
        """evaluate() works when the DataFrame has no 'label' column."""
        from train_app_category_classifier import evaluate

        rows = [
            {"packageName": "com.sberbank.online", "category": "BANK"},
            {"packageName": "org.telegram.messenger", "category": "MESSENGER"},
        ]
        test_df = _FakeDataFrame(rows)
        model = _make_random_model(seed=42)
        result = evaluate(model, test_df, small_vocab)

        # Should not crash; metrics should be valid
        assert 0.0 <= result["top1_accuracy"] <= 1.0
        assert 0.0 <= result["macro_f1"] <= 1.0
        assert len(result["per_category"]) == 18

    def test_macro_f1_is_mean_of_per_category_f1(self, small_vocab, synthetic_test_rows):
        """macro_f1 should equal the unweighted mean of all 18 per-category F1 scores."""
        from train_app_category_classifier import evaluate

        model = _make_random_model(seed=99)
        test_df = _FakeDataFrame(synthetic_test_rows)
        result = evaluate(model, test_df, small_vocab)

        f1_values = [v["f1"] for v in result["per_category"].values()]
        expected_macro_f1 = sum(f1_values) / len(f1_values)
        assert abs(result["macro_f1"] - expected_macro_f1) < 1e-7


# ── Tests for TFLite conversion + size budget (task 7.6, Requirements 2.6, 2.7, 2.11) ──


from train_app_category_classifier import (
    TFLITE_SIZE_BUDGET_BYTES,
    convert_to_tflite_quantized,
    enforce_size_budget,
)


class TestConvertToTfliteQuantized:
    """Unit tests for convert_to_tflite_quantized — task 7.6, Requirement 2.6."""

    def test_returns_bytes(self) -> None:
        """convert_to_tflite_quantized must return bytes."""
        tf = pytest.importorskip("tensorflow")

        # Build a minimal Keras model for conversion.
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(64,), dtype=tf.int32),
            tf.keras.layers.Embedding(100, 16),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(18, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        result = convert_to_tflite_quantized(model)
        assert isinstance(result, bytes)

    def test_result_is_nonempty(self) -> None:
        """The TFLite blob must be non-empty."""
        tf = pytest.importorskip("tensorflow")

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(64,), dtype=tf.int32),
            tf.keras.layers.Embedding(100, 16),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(18, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        result = convert_to_tflite_quantized(model)
        assert len(result) > 0

    def test_uses_dynamic_range_quantization(self) -> None:
        """The converted model should be smaller than the unquantized version."""
        tf = pytest.importorskip("tensorflow")

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(64,), dtype=tf.int32),
            tf.keras.layers.Embedding(500, 32),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(18, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        # Unquantized conversion for comparison.
        converter_unquant = tf.lite.TFLiteConverter.from_keras_model(model)
        unquant_blob = converter_unquant.convert()

        quant_blob = convert_to_tflite_quantized(model)

        # Dynamic-range quantization (int8 weights) should produce a
        # smaller blob than the default fp32 conversion.
        assert len(quant_blob) < len(unquant_blob)


class TestEnforceSizeBudget:
    """Unit tests for enforce_size_budget — task 7.6, Requirements 2.7, 2.11."""

    def test_budget_constant_is_1mb(self) -> None:
        """TFLITE_SIZE_BUDGET_BYTES must be exactly 1 048 576."""
        assert TFLITE_SIZE_BUDGET_BYTES == 1_048_576

    def test_within_budget_returns_none(self, tmp_path: Path) -> None:
        """Blobs at or below 1 MiB should pass without exit."""
        blob = b"\x00" * TFLITE_SIZE_BUDGET_BYTES  # exactly 1 MiB
        tmp_file = tmp_path / "model.tflite.tmp"
        tmp_file.write_bytes(b"staged")

        result = enforce_size_budget(blob, [tmp_file])
        assert result is None
        # .tmp file should still exist (not cleaned up).
        assert tmp_file.exists()

    def test_exactly_at_budget_passes(self, tmp_path: Path) -> None:
        """A blob of exactly 1_048_576 bytes should NOT trigger exit."""
        blob = b"\x42" * 1_048_576
        result = enforce_size_budget(blob, [])
        assert result is None

    def test_one_byte_over_budget_exits_with_code_2(self, tmp_path: Path) -> None:
        """A blob of 1_048_577 bytes should trigger sys.exit(2)."""
        blob = b"\x00" * (TFLITE_SIZE_BUDGET_BYTES + 1)
        with pytest.raises(SystemExit) as exc_info:
            enforce_size_budget(blob, [])
        assert exc_info.value.code == 2

    def test_large_blob_exits_with_code_2(self, tmp_path: Path) -> None:
        """Any blob exceeding 1 MiB should trigger sys.exit(2)."""
        blob = b"\xff" * (TFLITE_SIZE_BUDGET_BYTES + 1000)
        with pytest.raises(SystemExit) as exc_info:
            enforce_size_budget(blob, [])
        assert exc_info.value.code == 2

    def test_cleanup_removes_tmp_files(self, tmp_path: Path) -> None:
        """On budget overflow, all .tmp files in tmp_paths are removed."""
        tmp1 = tmp_path / "app_category_model.tflite.tmp"
        tmp2 = tmp_path / "app_category_vocab.txt.tmp"
        tmp3 = tmp_path / "app_category_card.json.tmp"
        tmp1.write_bytes(b"model data")
        tmp2.write_text("vocab data", encoding="utf-8")
        tmp3.write_text("{}", encoding="utf-8")

        blob = b"\x00" * (TFLITE_SIZE_BUDGET_BYTES + 1)
        with pytest.raises(SystemExit) as exc_info:
            enforce_size_budget(blob, [tmp1, tmp2, tmp3])
        assert exc_info.value.code == 2

        # All .tmp files should be cleaned up.
        assert not tmp1.exists()
        assert not tmp2.exists()
        assert not tmp3.exists()

    def test_cleanup_tolerates_missing_tmp_files(self, tmp_path: Path) -> None:
        """Cleanup should not fail if a .tmp file doesn't exist."""
        nonexistent = tmp_path / "does_not_exist.tmp"
        blob = b"\x00" * (TFLITE_SIZE_BUDGET_BYTES + 1)
        with pytest.raises(SystemExit) as exc_info:
            enforce_size_budget(blob, [nonexistent])
        assert exc_info.value.code == 2

    def test_does_not_remove_non_tmp_files(self, tmp_path: Path) -> None:
        """Only paths in tmp_paths are cleaned; other files are untouched."""
        final_file = tmp_path / "app_category_model.tflite"
        final_file.write_bytes(b"existing artifact")

        tmp_file = tmp_path / "app_category_model.tflite.tmp"
        tmp_file.write_bytes(b"staged")

        blob = b"\x00" * (TFLITE_SIZE_BUDGET_BYTES + 1)
        with pytest.raises(SystemExit) as exc_info:
            enforce_size_budget(blob, [tmp_file])
        assert exc_info.value.code == 2

        # The final artifact must remain untouched.
        assert final_file.exists()
        assert final_file.read_bytes() == b"existing artifact"
        # The .tmp file should be cleaned up.
        assert not tmp_file.exists()

    def test_stderr_message_on_overflow(self, tmp_path: Path, capsys) -> None:
        """On overflow, a descriptive message is printed to stderr."""
        blob = b"\x00" * (TFLITE_SIZE_BUDGET_BYTES + 42)
        with pytest.raises(SystemExit):
            enforce_size_budget(blob, [])
        captured = capsys.readouterr()
        assert "1048576" in captured.err
        assert str(TFLITE_SIZE_BUDGET_BYTES + 42) in captured.err


# ── Tests for build_char_cnn_model (task 7.2, Requirement 2.2) ─────────────

from train_app_category_classifier import build_char_cnn_model

# Check if TensorFlow is available for model-building tests.
try:
    import tensorflow as tf  # type: ignore[import-not-found]

    _HAS_TF = True
except ImportError:
    _HAS_TF = False

_skip_no_tf = pytest.mark.skipif(
    not _HAS_TF, reason="TensorFlow not installed"
)


class TestBuildCharCnnModelValidation:
    """Tests for build_char_cnn_model parameter validation (no TF needed)."""

    def test_vocab_size_must_be_int(self) -> None:
        with pytest.raises(TypeError, match="vocab_size must be an int"):
            build_char_cnn_model(vocab_size="100")  # type: ignore[arg-type]

    def test_vocab_size_bool_rejected(self) -> None:
        with pytest.raises(TypeError, match="vocab_size must be an int"):
            build_char_cnn_model(vocab_size=True)  # type: ignore[arg-type]

    def test_vocab_size_minimum_2(self) -> None:
        with pytest.raises(ValueError, match="vocab_size must be >= 2"):
            build_char_cnn_model(vocab_size=1)

    def test_max_len_must_be_int(self) -> None:
        with pytest.raises(TypeError, match="max_len must be an int"):
            build_char_cnn_model(vocab_size=100, max_len=64.0)  # type: ignore[arg-type]

    def test_max_len_minimum_1(self) -> None:
        with pytest.raises(ValueError, match="max_len must be >= 1"):
            build_char_cnn_model(vocab_size=100, max_len=0)

    def test_embed_dim_must_be_int(self) -> None:
        with pytest.raises(TypeError, match="embed_dim must be an int"):
            build_char_cnn_model(vocab_size=100, embed_dim=32.0)  # type: ignore[arg-type]

    def test_embed_dim_minimum_1(self) -> None:
        with pytest.raises(ValueError, match="embed_dim must be >= 1"):
            build_char_cnn_model(vocab_size=100, embed_dim=0)

    def test_conv_filters_must_be_int(self) -> None:
        with pytest.raises(TypeError, match="conv_filters must be an int"):
            build_char_cnn_model(vocab_size=100, conv_filters="128")  # type: ignore[arg-type]

    def test_conv_filters_bool_rejected(self) -> None:
        with pytest.raises(TypeError, match="conv_filters must be an int"):
            build_char_cnn_model(vocab_size=100, conv_filters=True)  # type: ignore[arg-type]

    def test_conv_filters_minimum_1(self) -> None:
        with pytest.raises(ValueError, match="conv_filters must be >= 1"):
            build_char_cnn_model(vocab_size=100, conv_filters=0)

    def test_num_classes_must_be_int(self) -> None:
        with pytest.raises(TypeError, match="num_classes must be an int"):
            build_char_cnn_model(vocab_size=100, num_classes=18.0)  # type: ignore[arg-type]

    def test_num_classes_bool_rejected(self) -> None:
        with pytest.raises(TypeError, match="num_classes must be an int"):
            build_char_cnn_model(vocab_size=100, num_classes=False)  # type: ignore[arg-type]

    def test_num_classes_minimum_1(self) -> None:
        with pytest.raises(ValueError, match="num_classes must be >= 1"):
            build_char_cnn_model(vocab_size=100, num_classes=0)

    def test_kernel_sizes_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError, match="kernel_sizes must be a non-empty tuple"):
            build_char_cnn_model(vocab_size=100, kernel_sizes=())

    def test_kernel_sizes_entries_must_be_positive_ints(self) -> None:
        with pytest.raises(ValueError, match="kernel_sizes entries must be positive ints"):
            build_char_cnn_model(vocab_size=100, kernel_sizes=(3, 0, 7))

    def test_kernel_sizes_rejects_bool_entries(self) -> None:
        with pytest.raises(ValueError, match="kernel_sizes entries must be positive ints"):
            build_char_cnn_model(vocab_size=100, kernel_sizes=(3, True, 7))


@_skip_no_tf
class TestBuildCharCnnModelArchitecture:
    """Tests for build_char_cnn_model model topology (requires TensorFlow)."""

    def test_default_params_builds_model(self) -> None:
        """Default parameters produce a valid Keras model."""
        model = build_char_cnn_model(vocab_size=1000)
        assert model is not None
        assert model.name == "app_category_cnn"

    def test_input_shape(self) -> None:
        """Model input shape matches (None, max_len) with int32 dtype."""
        model = build_char_cnn_model(vocab_size=1000, max_len=64)
        input_shape = model.input_shape
        assert input_shape == (None, 64)

    def test_output_shape_default(self) -> None:
        """Model output shape is (None, 18) with default num_classes."""
        model = build_char_cnn_model(vocab_size=1000, num_classes=18)
        output_shape = model.output_shape
        assert output_shape == (None, 18)

    def test_output_shape_custom_classes(self) -> None:
        """Model output shape respects custom num_classes."""
        model = build_char_cnn_model(vocab_size=500, num_classes=5)
        assert model.output_shape == (None, 5)

    def test_custom_max_len(self) -> None:
        """Model respects custom max_len for input shape."""
        model = build_char_cnn_model(vocab_size=500, max_len=128)
        assert model.input_shape == (None, 128)

    def test_three_conv_branches_present(self) -> None:
        """Model contains three Conv1D layers (one per kernel size)."""
        model = build_char_cnn_model(vocab_size=1000, kernel_sizes=(3, 5, 7))
        conv_layers = [l for l in model.layers if "conv1d" in l.name.lower()]
        assert len(conv_layers) == 3

    def test_conv_layer_names_reflect_kernel_sizes(self) -> None:
        """Conv1D layers are named with their kernel sizes."""
        model = build_char_cnn_model(vocab_size=1000, kernel_sizes=(3, 5, 7))
        layer_names = [l.name for l in model.layers]
        assert "conv1d_k3" in layer_names
        assert "conv1d_k5" in layer_names
        assert "conv1d_k7" in layer_names

    def test_global_max_pool_layers_present(self) -> None:
        """Model contains GlobalMaxPooling1D layers for each branch."""
        model = build_char_cnn_model(vocab_size=1000, kernel_sizes=(3, 5, 7))
        pool_layers = [l for l in model.layers if "global_max_pool" in l.name]
        assert len(pool_layers) == 3

    def test_embedding_layer_present(self) -> None:
        """Model contains an Embedding layer with correct vocab_size."""
        model = build_char_cnn_model(vocab_size=5000, embed_dim=32)
        embed_layers = [
            l for l in model.layers
            if isinstance(l, tf.keras.layers.Embedding)
        ]
        assert len(embed_layers) == 1
        assert embed_layers[0].input_dim == 5000
        assert embed_layers[0].output_dim == 32

    def test_dense_softmax_output(self) -> None:
        """Final Dense layer uses softmax activation."""
        model = build_char_cnn_model(vocab_size=1000, num_classes=18)
        output_layer = model.layers[-1]
        assert isinstance(output_layer, tf.keras.layers.Dense)
        assert output_layer.units == 18
        # Keras stores activation config; check it's softmax.
        activation_name = output_layer.get_config()["activation"]
        assert activation_name == "softmax"

    def test_concat_layer_present(self) -> None:
        """Model contains a Concatenate layer merging the branches."""
        model = build_char_cnn_model(vocab_size=1000, kernel_sizes=(3, 5, 7))
        concat_layers = [
            l for l in model.layers
            if isinstance(l, tf.keras.layers.Concatenate)
        ]
        assert len(concat_layers) == 1

    def test_model_predict_shape(self) -> None:
        """Model.predict returns correct output shape for a batch."""
        model = build_char_cnn_model(vocab_size=500, max_len=32, num_classes=18)
        # Create a dummy input batch of 4 samples.
        dummy_input = tf.zeros((4, 32), dtype=tf.int32)
        output = model.predict(dummy_input, verbose=0)
        assert output.shape == (4, 18)

    def test_model_output_sums_to_one(self) -> None:
        """Softmax output probabilities sum to ~1.0 for each sample."""
        model = build_char_cnn_model(vocab_size=500, max_len=32, num_classes=18)
        dummy_input = tf.constant([[1, 2, 3] + [0] * 29], dtype=tf.int32)
        output = model.predict(dummy_input, verbose=0)
        assert abs(float(output.sum()) - 1.0) < 1e-5

    def test_single_kernel_size(self) -> None:
        """Model works with a single kernel size (no Concatenate needed)."""
        model = build_char_cnn_model(vocab_size=500, kernel_sizes=(5,))
        assert model.output_shape == (None, 18)
        # No Concatenate layer when there's only one branch.
        concat_layers = [
            l for l in model.layers
            if isinstance(l, tf.keras.layers.Concatenate)
        ]
        assert len(concat_layers) == 0

    def test_custom_conv_filters(self) -> None:
        """Conv1D layers respect the conv_filters parameter."""
        model = build_char_cnn_model(vocab_size=500, conv_filters=64)
        conv_layers = [
            l for l in model.layers
            if "conv1d" in l.name.lower()
        ]
        for conv in conv_layers:
            assert conv.filters == 64

    def test_model_is_not_compiled(self) -> None:
        """Model is returned uncompiled (caller applies optimizer/loss)."""
        model = build_char_cnn_model(vocab_size=500)
        # An uncompiled model has no optimizer set.
        assert model.optimizer is None


# ── Tests for main() orchestration (task 7.10, Requirements 2.1–2.12) ─────


from train_app_category_classifier import (
    load_splits,
    parse_args,
    main as train_main,
    set_random_seed,
)


class TestParseArgs:
    """Tests for parse_args (task 7.10, Requirement 2.1)."""

    def test_defaults(self) -> None:
        """All defaults match the documented paths."""
        ns = parse_args([])
        assert ns.seed == 42
        assert str(ns.train).replace("\\", "/").endswith(
            "datasets/categories/train.csv"
        )
        assert str(ns.val).replace("\\", "/").endswith(
            "datasets/categories/val.csv"
        )
        assert str(ns.test).replace("\\", "/").endswith(
            "datasets/categories/test.csv"
        )
        assert str(ns.output).replace("\\", "/").endswith(
            "app/src/main/assets/app_category_model.tflite"
        )
        assert str(ns.vocab).replace("\\", "/").endswith(
            "app/src/main/assets/app_category_vocab.txt"
        )
        assert str(ns.card).replace("\\", "/").endswith(
            "app/src/main/assets/app_category_card.json"
        )

    def test_custom_seed_and_paths(self, tmp_path: Path) -> None:
        """All CLI flags override defaults."""
        train_path = tmp_path / "train.csv"
        val_path = tmp_path / "val.csv"
        test_path = tmp_path / "test.csv"
        out_path = tmp_path / "model.tflite"
        vocab_path = tmp_path / "vocab.txt"
        card_path = tmp_path / "card.json"

        ns = parse_args([
            "--train", str(train_path),
            "--val", str(val_path),
            "--test", str(test_path),
            "--seed", "123",
            "--output", str(out_path),
            "--vocab", str(vocab_path),
            "--card", str(card_path),
        ])

        assert ns.seed == 123
        assert ns.train == train_path
        assert ns.val == val_path
        assert ns.test == test_path
        assert ns.output == out_path
        assert ns.vocab == vocab_path
        assert ns.card == card_path


class TestLoadSplits:
    """Tests for load_splits (task 7.10)."""

    def _write_csv(self, path: Path, rows: list[tuple[str, str, str]]) -> None:
        """Write a CSV with the standard header and the given rows."""
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "packageName,label,category\n"
        for pkg, label, cat in rows:
            body += f"{pkg},{label},{cat}\n"
        path.write_bytes(body.encode("utf-8"))

    def test_loads_three_splits(self, tmp_path: Path) -> None:
        """All three splits are loaded into _CsvFrame objects."""
        train = tmp_path / "train.csv"
        val = tmp_path / "val.csv"
        test = tmp_path / "test.csv"
        self._write_csv(train, [("com.a", "A", "BANK"), ("com.b", "B", "GAMES")])
        self._write_csv(val, [("com.c", "C", "EMAIL")])
        self._write_csv(test, [("com.d", "D", "MESSENGER")])

        train_df, val_df, test_df = load_splits(train, val, test)
        assert len(train_df) == 2
        assert len(val_df) == 1
        assert len(test_df) == 1
        assert train_df.columns == ["packageName", "label", "category"]

    def test_missing_train_raises(self, tmp_path: Path) -> None:
        """A missing train CSV surfaces a FileNotFoundError."""
        train = tmp_path / "missing.csv"
        val = tmp_path / "val.csv"
        test = tmp_path / "test.csv"
        self._write_csv(val, [("com.a", "A", "BANK")])
        self._write_csv(test, [("com.b", "B", "GAMES")])

        with pytest.raises(FileNotFoundError, match="split CSV not found"):
            load_splits(train, val, test)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        """A truly empty CSV (no header) raises ValueError."""
        train = tmp_path / "train.csv"
        val = tmp_path / "val.csv"
        test = tmp_path / "test.csv"
        train.write_bytes(b"")
        self._write_csv(val, [("com.a", "A", "BANK")])
        self._write_csv(test, [("com.b", "B", "GAMES")])

        with pytest.raises(ValueError, match="empty"):
            load_splits(train, val, test)

    def test_wrong_header_raises(self, tmp_path: Path) -> None:
        """A CSV with the wrong column header raises ValueError."""
        train = tmp_path / "train.csv"
        val = tmp_path / "val.csv"
        test = tmp_path / "test.csv"
        train.write_bytes(b"package,label,category\ncom.a,A,BANK\n")
        self._write_csv(val, [("com.a", "A", "BANK")])
        self._write_csv(test, [("com.b", "B", "GAMES")])

        with pytest.raises(ValueError, match="unexpected header"):
            load_splits(train, val, test)

    def test_csvframe_itertuples_yields_named_rows(self, tmp_path: Path) -> None:
        """itertuples returns namedtuples with the expected attribute names."""
        train = tmp_path / "train.csv"
        val = tmp_path / "val.csv"
        test = tmp_path / "test.csv"
        self._write_csv(train, [("com.a", "Label A", "BANK")])
        self._write_csv(val, [("com.b", "B", "GAMES")])
        self._write_csv(test, [("com.c", "C", "EMAIL")])

        train_df, _, _ = load_splits(train, val, test)
        records = list(train_df.itertuples(index=False))
        assert len(records) == 1
        assert records[0].packageName == "com.a"
        assert records[0].label == "Label A"
        assert records[0].category == "BANK"


class TestSetRandomSeed:
    """Tests for set_random_seed (task 7.10, Requirement 2.1)."""

    def test_seeds_python_random(self) -> None:
        """random.random() returns the same first draw after seeding."""
        import random

        set_random_seed(42)
        a = random.random()
        set_random_seed(42)
        b = random.random()
        assert a == b

    def test_seeds_numpy(self) -> None:
        """numpy.random.rand() returns the same first draw after seeding."""
        set_random_seed(7)
        a = np.random.rand(3)
        set_random_seed(7)
        b = np.random.rand(3)
        np.testing.assert_array_equal(a, b)

    def test_rejects_non_int_seed(self) -> None:
        """Non-int seed raises TypeError."""
        with pytest.raises(TypeError, match="seed must be an int"):
            set_random_seed(3.14)  # type: ignore[arg-type]

    def test_rejects_bool_seed(self) -> None:
        """bool is not accepted as int seed."""
        with pytest.raises(TypeError, match="seed must be an int"):
            set_random_seed(True)  # type: ignore[arg-type]


class TestMainOrchestration:
    """Tests for main() orchestrator (task 7.10, Requirements 2.1–2.12)."""

    def _write_csv(self, path: Path, rows: list[tuple[str, str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "packageName,label,category\n"
        for pkg, label, cat in rows:
            body += f"{pkg},{label},{cat}\n"
        path.write_bytes(body.encode("utf-8"))

    def test_missing_train_returns_1(self, tmp_path: Path, capsys) -> None:
        """Missing input CSV → exit code 1 (Requirement 2.12 fail-closed)."""
        # train.csv is intentionally NOT created.
        val = tmp_path / "val.csv"
        test = tmp_path / "test.csv"
        out = tmp_path / "model.tflite"
        vocab = tmp_path / "vocab.txt"
        card = tmp_path / "card.json"
        self._write_csv(val, [("com.a", "A", "BANK")])
        self._write_csv(test, [("com.b", "B", "GAMES")])

        rc = train_main([
            "--train", str(tmp_path / "missing.csv"),
            "--val", str(val),
            "--test", str(test),
            "--output", str(out),
            "--vocab", str(vocab),
            "--card", str(card),
        ])
        assert rc == 1
        # No artifacts may be produced.
        assert not out.exists()
        assert not vocab.exists()
        assert not card.exists()
        # Stderr must mention the failure (Requirement 2.12).
        captured = capsys.readouterr()
        assert "FAIL" in captured.err

    def test_empty_train_returns_1(self, tmp_path: Path, capsys) -> None:
        """A train CSV with only a header (zero data rows) → exit 1."""
        train = tmp_path / "train.csv"
        val = tmp_path / "val.csv"
        test = tmp_path / "test.csv"
        out = tmp_path / "model.tflite"
        vocab = tmp_path / "vocab.txt"
        card = tmp_path / "card.json"
        # Header-only train CSV is loadable but empty.
        train.write_bytes(b"packageName,label,category\n")
        self._write_csv(val, [("com.a", "A", "BANK")])
        self._write_csv(test, [("com.b", "B", "GAMES")])

        rc = train_main([
            "--train", str(train),
            "--val", str(val),
            "--test", str(test),
            "--output", str(out),
            "--vocab", str(vocab),
            "--card", str(card),
        ])
        assert rc == 1
        assert not out.exists()
        assert not vocab.exists()
        assert not card.exists()
        captured = capsys.readouterr()
        assert "FAIL" in captured.err

    def test_preexisting_artifacts_unchanged_on_load_failure(
        self, tmp_path: Path
    ) -> None:
        """Pre-existing artifacts are not modified on early failure
        (Requirement 2.12).
        """
        out = tmp_path / "model.tflite"
        vocab = tmp_path / "vocab.txt"
        card = tmp_path / "card.json"
        out.write_bytes(b"old-tflite")
        vocab.write_bytes(b"old-vocab\n")
        card.write_bytes(b"{\"old\": true}\n")

        # All splits missing → exit 1.
        rc = train_main([
            "--train", str(tmp_path / "no-train.csv"),
            "--val", str(tmp_path / "no-val.csv"),
            "--test", str(tmp_path / "no-test.csv"),
            "--output", str(out),
            "--vocab", str(vocab),
            "--card", str(card),
        ])
        assert rc == 1
        # Pre-existing artifacts must be untouched (SHA256-equivalent).
        assert out.read_bytes() == b"old-tflite"
        assert vocab.read_bytes() == b"old-vocab\n"
        assert card.read_bytes() == b"{\"old\": true}\n"
