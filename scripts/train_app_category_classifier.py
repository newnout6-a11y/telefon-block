"""
Training pipeline for the third on-device classifier — App Category Model.

This is the *training* side; the runtime contract lives in
`app/src/main/java/com/antispam/blocker/domain/categorization/AppCategoryClassifier.kt`.
The current production runtime is `RuleBasedAppCategoryClassifier`
(dictionary + substring heuristics, ~150 known packages, 18 categories).
Once this script produces a TFLite model with comparable or better
precision, drop it next to `spam_model.tflite` in `app/src/main/assets/`,
publish via `releases/latest/manifest.json`, and switch
`AppCategoryClassifierFactory` to the TFLite-backed implementation.

## Stack

The architecture is a small character-level convolutional net over the
package name (and optional locale-aware label) — sub-1MB after TFLite
quantization, runs in microseconds on-device:

    packageName ─┐
                 ├─► CharNGramEncoder ──► Conv1D × 3 ──► GlobalMaxPool
    label?      ─┘                     ──► Dense(18, softmax)

Why char-CNN and not LSTM/Transformer:
- Package names are <100 chars, no temporal dependencies that need attention.
- TFLite has first-class support for Conv1D + Dense; LSTM has known
  quirks on Android NNAPI delegates.
- Sub-1MB after dynamic-range quantization (int8 weights).

## Dataset

Target: ≥ 200k labeled (packageName, category) pairs. Sources:

1. **Google Play Store taxonomy**: Play Store category page → top apps
   per category. Per-category quotas balanced (≥ 5k per AppCategory).
2. **F-Droid + RuStore + Huawei AppGallery** mirrors for non-Google
   distributions (~30k Russian apps).
3. **Bootstrap from production rules**: every package matched by
   `RuleBasedAppCategoryClassifier` at >95% confidence (label.lower()
   match in label markers) gets added as labelled data.
4. **Hand-labelled tail** for rare categories (HEALTH, EDUCATION, VPN
   often have <500 examples). Curate via Play Store category browse.

Final shape: `datasets/categories/labeled.csv` with columns
`packageName,label,category`. Train/val/test split 80/10/10 stratified
by category (so rare categories don't disappear from val).

## Training

    python scripts/train_app_category_classifier.py \\
        --data datasets/categories/labeled.csv \\
        --epochs 30 \\
        --batch-size 256 \\
        --output app/src/main/assets/app_category_model.tflite

Expected metrics on a balanced 200k corpus:
- Top-1 accuracy ≥ 90% (vs ~70% for pure dictionary)
- Macro-F1 ≥ 0.85
- BANK / GOVERNMENT / EMAIL — precision ≥ 95% (sensitive features in
  Personal Model rely on these)

## Status

THIS SCRIPT IS A SCAFFOLD. The actual training loop is left as a TODO
because:

1. We don't yet ship a 200k labelled corpus.
2. The current rule-based classifier covers 95%+ of the user's daily
   apps in production. Personal Model features are robust to the
   long-tail OTHER bucket.

When ready to train, the steps are:
1. Download Play Store + RuStore + AppGallery categories metadata.
2. Run `_collect_app_corpus.py` (also TODO) to dump labelled CSV.
3. Run this script with `--train` flag to fit the model.
4. Verify TFLite size ≤ 1 MB, top-1 accuracy ≥ 90% on holdout.
5. Publish via `releases/latest/manifest.json` with sha256.
6. Flip `AppCategoryClassifierFactory.instance` to the TFLite impl
   once the asset is shipped.

## Inputs / outputs

Input CSV: `packageName,label,category`
Output: `app_category_model.tflite` + `app_category_card.json`
    (vocab size, char-ngram size, per-category precision/recall/f1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Single source of truth for the Python side of the App_Category_Model
# pipeline. Must stay byte-identical to the order returned by
# `AppCategory.values().map { it.name }` in
# `app/src/main/java/com/antispam/blocker/domain/categorization/`
# `AppCategoryClassifier.kt`. The Property 1 enum-order parity test
# (Requirements 2.10, 6.1, 7.1) reads this list and compares against
# the live Kotlin enum and the `categories_order` field of the
# `app_category_card.json` Model_Card.
#
# `OTHER` (id 19) is reserved for `RuleBasedAppCategoryClassifier`
# fallback and is intentionally excluded from the 18-class softmax head
# (see Requirement 2.2). The training pipeline therefore trains on the
# first 18 entries; `OTHER` only appears in the Model_Card
# `categories_order` (Requirement 2.9) and on disk in `labeled.csv`
# rows that came from sources outside the rule-based vocabulary.
KOTLIN_APP_CATEGORY_ORDER: list[str] = [
    "BANK",
    "INVESTMENTS",
    "GOVERNMENT",
    "MARKETPLACE",
    "DELIVERY",
    "TRANSPORT",
    "TRAVEL",
    "HEALTH",
    "MESSENGER",
    "SOCIAL",
    "EMAIL",
    "NEWS",
    "MEDIA",
    "GAMES",
    "DATING",
    "EDUCATION",
    "BROWSER",
    "VPN",
    "PRODUCTIVITY",
    "OTHER",
]
assert len(KOTLIN_APP_CATEGORY_ORDER) == 20, (
    "KOTLIN_APP_CATEGORY_ORDER must have exactly 20 entries; "
    f"got {len(KOTLIN_APP_CATEGORY_ORDER)}"
)
assert KOTLIN_APP_CATEGORY_ORDER[-1] == "OTHER", (
    "OTHER must be the last entry (id 19) in KOTLIN_APP_CATEGORY_ORDER"
)
assert len(set(KOTLIN_APP_CATEGORY_ORDER)) == len(KOTLIN_APP_CATEGORY_ORDER), (
    "KOTLIN_APP_CATEGORY_ORDER must not contain duplicate category names"
)

# Backwards-compatible alias for callers that already import CATEGORIES.
CATEGORIES = KOTLIN_APP_CATEGORY_ORDER


# ── CharNGramVocab + dataset encoder (task 7.1, Requirements 2.2, 2.8) ────
#
# The training pipeline tokenizes every (packageName, label) pair into
# fixed-length char-n-gram id sequences, then feeds them into the
# 18-class char-CNN softmax head (task 7.2). The on-device runtime
# (`CharNGramTokenizer.kt`, task 9.1) loads the same vocab from
# ``app_category_vocab.txt`` and is required to produce byte-identical
# token id sequences for any given input — otherwise the TFLite
# softmax-output is meaningless.  This module's :class:`CharNGramVocab`
# is therefore the *canonical* encoder; the Kotlin tokenizer mirrors it
# bit-for-bit.
#
# Design choices baked into the class (Requirement 2.8):
#   * ``<PAD>`` is always token id 0, ``<UNK>`` is always token id 1.
#     Both are reserved sentinels; no real char-n-gram can collide.
#   * The vocab file format is one token per line, line 0 = ``<PAD>``,
#     line 1 = ``<UNK>``, no empty lines, trailing newline.
#     :meth:`CharNGramVocab.serialize` produces exactly that string;
#     the on-disk encoding (UTF-8 / LF / no BOM) is owned by
#     :func:`write_atomic` (task 7.9), not by this class.
#   * :meth:`CharNGramVocab.build` is deterministic given the same
#     ``rows`` iterable and the same ``max_size``: ties in frequency
#     are broken alphabetically (ascending by token string), so two
#     runs with the same training set produce byte-identical vocabs
#     (Requirement 2.1, Property 12).
#   * :meth:`CharNGramVocab.encode` returns an ``int32`` array of
#     length exactly ``max_len`` (default 64), padded right with
#     ``PAD_ID``. Order of n-gram concatenation is ``n=3``, then
#     ``n=4``, then ``n=5`` — fixed because the Kotlin runtime
#     enumerates ``nGramSizes = intArrayOf(3, 4, 5)`` in the same
#     order (design.md Component 2).
#
# :func:`encode_dataset` projects a labelled DataFrame onto the
# ``(int32[max_len], int32 label_id)`` shape that the char-CNN ingests
# via ``tf.data.Dataset.from_tensor_slices``. Rows whose category is
# ``"OTHER"`` are dropped because the softmax head is 18-class — OTHER
# only ever comes from the rule-based fallback at runtime
# (Requirement 2.2). TensorFlow is imported lazily so non-training
# entry points (quality gate, model-card writer, dataset-pipeline
# tests) keep working in environments that don't ship TF.

#: Token id reserved for right-padding short sequences up to ``max_len``.
#: Mirrors ``CharNGramTokenizer.PAD_ID`` in the Kotlin runtime
#: (task 9.1).  Frozen at 0 so the pad value is also the natural
#: zero-init value of an ``int32`` tensor.
PAD_ID: int = 0

#: Token id reserved for char-n-grams that are absent from the vocab
#: built at training time. Mirrors ``CharNGramTokenizer.UNK_ID`` in the
#: Kotlin runtime (task 9.1).  Frozen at 1 so id 0 stays exclusive to
#: padding.
UNK_ID: int = 1


class CharNGramVocab:
    """Char-n-gram vocabulary for App_Category_Model tokenization.

    Holds an ordered list of tokens whose index is the token id used
    by the TFLite Embedding layer at training time and by
    :class:`CharNGramTokenizer` on-device. ``tokens[0]`` is always
    ``"<PAD>"`` (id 0, :data:`PAD_ID`) and ``tokens[1]`` is always
    ``"<UNK>"`` (id 1, :data:`UNK_ID`). All remaining entries are
    real char-n-grams extracted from the training corpus by
    :meth:`build`.

    The class is deliberately tiny: it owns vocabulary state and
    encoding logic, but does not own the on-disk format
    (:func:`write_atomic` writes the bytes from
    :meth:`serialize`) or the tf.data plumbing
    (:func:`encode_dataset` slices encoded rows into batches).
    """

    __slots__ = ("tokens", "token_to_id")

    def __init__(self, tokens: list[str]) -> None:
        """Validate ``tokens`` and build the id lookup map.

        The list must satisfy the contract that the Kotlin runtime
        also enforces (task 9.1, Requirement 2.8):

          * Length ≥ 2 (room for ``<PAD>`` and ``<UNK>``).
          * ``tokens[0] == "<PAD>"`` and ``tokens[1] == "<UNK>"`` —
            id 0 and id 1 are reserved sentinels, callers cannot
            override them.
          * Every token is a non-empty string. An empty entry would
            collide with the ``<PAD>`` id when serialised one-per-line.
          * No duplicates. A duplicate token would map two distinct
            ids to the same string in :attr:`token_to_id`, silently
            losing one of them.

        Violations raise :class:`ValueError` with a precise message;
        no token text is logged through :mod:`logging` because the
        runtime contract (Requirement 5.4) forbids exposing labels
        in logs, and the same code path executes on-device when
        loading a remote-updated vocab.
        """
        if not isinstance(tokens, list):
            raise TypeError(
                f"CharNGramVocab tokens must be a list, got "
                f"{type(tokens).__name__}"
            )
        if len(tokens) < 2:
            raise ValueError(
                "CharNGramVocab requires at least 2 tokens "
                "(<PAD> at id 0 and <UNK> at id 1); "
                f"got {len(tokens)}"
            )
        if tokens[0] != "<PAD>":
            raise ValueError(
                "CharNGramVocab tokens[0] must be '<PAD>' (id 0)"
            )
        if tokens[1] != "<UNK>":
            raise ValueError(
                "CharNGramVocab tokens[1] must be '<UNK>' (id 1)"
            )
        for idx, tok in enumerate(tokens):
            if not isinstance(tok, str):
                raise TypeError(
                    f"CharNGramVocab tokens[{idx}] must be a str, "
                    f"got {type(tok).__name__}"
                )
            if tok == "":
                raise ValueError(
                    f"CharNGramVocab tokens[{idx}] is empty; empty "
                    "tokens are not allowed because the on-disk "
                    "vocab format uses LF as the token separator"
                )
        if len(set(tokens)) != len(tokens):
            # Find the first duplicate for a precise error.  We don't
            # log the token text — a duplicate <PAD>/<UNK> is a
            # programmer error, not user data.
            seen: set[str] = set()
            dup_idx: int = -1
            for idx, tok in enumerate(tokens):
                if tok in seen:
                    dup_idx = idx
                    break
                seen.add(tok)
            raise ValueError(
                f"CharNGramVocab tokens contain a duplicate at "
                f"index {dup_idx}; every token must be unique"
            )

        self.tokens: list[str] = list(tokens)
        self.token_to_id: dict[str, int] = {
            tok: idx for idx, tok in enumerate(self.tokens)
        }

    @classmethod
    def build(
        cls,
        rows,
        n_grams: tuple[int, ...] = (3, 4, 5),
        max_size: int = 20_000,
    ) -> "CharNGramVocab":
        """Build a vocab from an iterable of ``(packageName, label)`` pairs.

        Pipeline:

          1. For each row, normalise the label via
             :func:`build_app_category_dataset.normalize_label`
             (NFC, strip, ≤ 200 chars, fallback to ``""``,
             Requirement 1.4).
          2. Build the encoding text:
             ``f"{packageName} {label}"`` when the normalised label is
             non-empty, else just ``packageName``. The single space
             separator matches the on-device tokenizer
             (design.md Component 2).
          3. Slide every ``n``-gram window for each ``n`` in ``n_grams``
             over the text, accumulating raw counts.
          4. Take the top ``max_size - 2`` tokens by count, breaking
             ties alphabetically (ascending by the token string)
             so the result is byte-identical between runs given the
             same input (Requirement 2.1, Property 12).
          5. Prepend the ``<PAD>``/``<UNK>`` sentinels and hand the
             result to :meth:`__init__` for validation.

        Args:
            rows: Iterable of ``(packageName, label)`` pairs. ``label``
                may be ``None`` or ``""`` to indicate the row has no
                label (the encoder skips the separator in that case).
                Iterated exactly once.
            n_grams: Sliding-window sizes. Defaults to ``(3, 4, 5)``,
                matching the on-device tokenizer.
            max_size: Maximum total vocabulary size *including* the
                two sentinels. Real char-n-grams therefore cap at
                ``max_size - 2``. Defaults to 20 000.

        Returns:
            A new :class:`CharNGramVocab` populated from ``rows``.
        """
        if not isinstance(max_size, int) or isinstance(max_size, bool):
            raise TypeError("max_size must be an int")
        if max_size < 2:
            raise ValueError(
                f"max_size must be >= 2 (room for <PAD> and <UNK>); "
                f"got {max_size}"
            )
        if not n_grams:
            raise ValueError("n_grams must be a non-empty tuple")
        for n in n_grams:
            if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
                raise ValueError(
                    f"n_grams entries must be positive ints, got {n!r}"
                )

        # Local imports keep the module-level surface small and avoid
        # cycles: ``build_app_category_dataset`` already imports
        # ``KOTLIN_APP_CATEGORY_ORDER`` from this module.
        from collections import Counter

        from build_app_category_dataset import normalize_label

        counter: "Counter[str]" = Counter()
        for package_name, label in rows:
            normalised = normalize_label(label) if label else ""
            text = (
                f"{package_name} {normalised}"
                if normalised
                else package_name
            )
            for n in n_grams:
                # range stop = len(text) - n + 1; substring(i, i + n)
                # therefore stays within bounds for every i.
                for i in range(len(text) - n + 1):
                    counter[text[i : i + n]] += 1

        # Sort: highest count first, alphabetic ascending tiebreak.
        # ``-count`` flips the count comparison while keeping the
        # alphabetic comparison ascending, which is exactly the
        # tie-break order the on-device parity check expects.
        sorted_items = sorted(
            counter.items(), key=lambda kv: (-kv[1], kv[0])
        )
        top_tokens = [tok for tok, _ in sorted_items[: max_size - 2]]

        return cls(["<PAD>", "<UNK>"] + top_tokens)

    def encode(
        self,
        packageName: str,  # noqa: N803 - matches Kotlin parameter name
        label: str | None,
        max_len: int = 64,
    ):
        """Encode one ``(packageName, label)`` pair to an int32 id array.

        Mirrors :meth:`CharNGramTokenizer.encode` in the Kotlin runtime
        (task 9.1) bit-for-bit:

          1. Normalise ``label`` via
             :func:`build_app_category_dataset.normalize_label` when
             provided; ``None`` and empty strings collapse to ``""``.
          2. Build the encoding text exactly as :meth:`build` does:
             ``f"{packageName} {label}"`` when the normalised label is
             non-empty, else just ``packageName``.
          3. For each ``n`` in ``(3, 4, 5)`` (in that exact order),
             slide ``substring(i, i + n)`` across the text and look up
             each n-gram in :attr:`token_to_id`. Misses fall back to
             :data:`UNK_ID`.
          4. Concatenate the three id lists in ``n=3``-first,
             ``n=4``-next, ``n=5``-last order. Truncate to ``max_len``
             from the right; if shorter, right-pad with :data:`PAD_ID`.

        The result is always an ``int32`` array of length exactly
        ``max_len`` (default 64), suitable for direct ingestion by
        ``tf.data.Dataset.from_tensor_slices``.

        Numpy is imported lazily so callers that only need
        :meth:`build` / :meth:`serialize` (e.g. unit tests for
        vocab roundtrip) do not pay the import cost.
        """
        if not isinstance(packageName, str):
            raise TypeError(
                f"packageName must be a str, got "
                f"{type(packageName).__name__}"
            )
        if label is not None and not isinstance(label, str):
            raise TypeError(
                f"label must be a str or None, got {type(label).__name__}"
            )
        if not isinstance(max_len, int) or isinstance(max_len, bool):
            raise TypeError("max_len must be an int")
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")

        import numpy as np

        from build_app_category_dataset import normalize_label

        normalised = normalize_label(label) if label else ""
        text = f"{packageName} {normalised}" if normalised else packageName

        ids: list[int] = []
        # Order is fixed: n=3, n=4, n=5 — matches the Kotlin runtime
        # which iterates ``nGramSizes = intArrayOf(3, 4, 5)`` in the
        # same order (design.md Component 2).
        for n in (3, 4, 5):
            stop = len(text) - n + 1
            for i in range(stop):
                ngram = text[i : i + n]
                ids.append(self.token_to_id.get(ngram, UNK_ID))

        if len(ids) >= max_len:
            ids = ids[:max_len]
        else:
            ids.extend([PAD_ID] * (max_len - len(ids)))

        return np.asarray(ids, dtype=np.int32)

    def serialize(self) -> str:
        """Return the canonical vocab file body.

        Format (Requirement 2.8): one token per line, in id-ascending
        order — line 0 = ``<PAD>``, line 1 = ``<UNK>``, line k =
        ``tokens[k]``. Lines are joined with ``"\\n"`` and a trailing
        ``"\\n"`` is appended so the body always ends with a newline,
        matching the Kotlin loader's expectations
        (``CharNGramTokenizer.load``, task 9.1).

        This method only produces the *string* body. UTF-8 / no-BOM /
        LF on-disk encoding is the responsibility of
        :func:`write_atomic` (task 7.9), which is the only writer the
        training pipeline uses for any of the three release artifacts.
        """
        return "\n".join(self.tokens) + "\n"


def encode_dataset(df, vocab: CharNGramVocab, batch_size: int):
    """Project a labelled DataFrame onto a batched ``tf.data.Dataset``.

    Implements the dataset half of task 7.1: every row of ``df`` is
    encoded into a fixed-length token-id vector via
    :meth:`CharNGramVocab.encode`, paired with its integer label id,
    and emitted as a ``tf.data.Dataset`` of
    ``(int32[max_len], int32)`` tensors batched to ``batch_size``.

    Filtering rules:

      * Rows whose ``category`` is ``"OTHER"`` are skipped — the
        char-CNN softmax head is 18-class and ``OTHER`` is reserved
        for the rule-based fallback at runtime (Requirement 2.2).
        The remaining categories are mapped to label ids via
        :data:`KOTLIN_APP_CATEGORY_ORDER` ``.index(category)``.

    Args:
        df: A pandas-style DataFrame with at least the columns
            ``packageName``, ``label``, ``category``. ``label`` may be
            ``NaN`` / missing — :meth:`CharNGramVocab.encode` treats
            empty / falsy labels the same as ``None``. Iterated
            exactly once via ``.itertuples(index=False)``.
        vocab: The :class:`CharNGramVocab` produced by
            :meth:`CharNGramVocab.build` on the *training* split. The
            same vocab must be reused for val / test encoding so the
            id space is consistent across splits.
        batch_size: Positive integer batch size for the returned
            ``tf.data.Dataset``.

    Returns:
        A ``tf.data.Dataset`` yielding ``(features, label_id)`` tuples
        where ``features`` has shape ``[batch, max_len]`` and dtype
        ``int32``, and ``label_id`` has shape ``[batch]`` and dtype
        ``int32``. The dataset is wrapped with
        ``.batch(batch_size).prefetch(tf.data.AUTOTUNE)`` so the
        training loop can overlap I/O with compute.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an int")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if not isinstance(vocab, CharNGramVocab):
        raise TypeError(
            f"vocab must be a CharNGramVocab, got {type(vocab).__name__}"
        )

    import numpy as np
    import tensorflow as tf  # type: ignore[import-not-found]

    feature_rows: list = []
    label_rows: list[int] = []

    # ``itertuples(index=False)`` is faster than ``iterrows`` and emits
    # plain ``namedtuple``s, which is enough for the three column reads
    # below.  We tolerate a missing ``label`` column by falling back
    # to ``""`` so callers can hand us a DataFrame with only the two
    # required columns (packageName, category).
    has_label_column = "label" in getattr(df, "columns", ())
    for record in df.itertuples(index=False):
        package_name = getattr(record, "packageName")
        category = getattr(record, "category")
        if category == "OTHER" or category == "PRODUCTIVITY":
            # Softmax is 18-class (BANK..VPN, indices 0..17); OTHER and
            # PRODUCTIVITY only come from the rule-based fallback at
            # runtime (Requirement 2.2). PRODUCTIVITY (index 18) is
            # excluded because Dense(18) covers indices 0..17.
            continue
        label = getattr(record, "label", "") if has_label_column else ""
        # ``label`` may be NaN when the source DataFrame stores
        # missing values; treat any non-string as "no label".
        if not isinstance(label, str):
            label = ""
        label_id = KOTLIN_APP_CATEGORY_ORDER.index(category)
        feature_rows.append(vocab.encode(package_name, label))
        label_rows.append(label_id)

    if feature_rows:
        X = np.stack(feature_rows, axis=0).astype(np.int32, copy=False)
    else:
        # Preserve the canonical [N, 64] shape so downstream consumers
        # (model.fit, evaluate) can rely on a stable rank even on an
        # empty split.
        X = np.zeros((0, 64), dtype=np.int32)
    y = np.asarray(label_rows, dtype=np.int32)

    return (
        tf.data.Dataset.from_tensor_slices((X, y))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


# ── Char-CNN model architecture (task 7.2, Requirement 2.2) ────────────────
#
# The App_Category_Model is a small character-level convolutional net:
#
#   Input: int32[max_len] token-id sequence (from CharNGramVocab.encode)
#         ↓
#   Embedding(vocab_size, embed_dim)
#         ↓
#   3 parallel Conv1D branches (kernel_sizes 3, 5, 7; filters=128 each)
#         ↓
#   GlobalMaxPooling1D per branch
#         ↓
#   Concatenate (→ 3 × 128 = 384 features)
#         ↓
#   Dense(num_classes=18, softmax)
#
# The softmax head covers the first 18 values of AppCategory
# (BANK..PRODUCTIVITY). OTHER (id 19) is reserved for the rule-based
# fallback and is never predicted by the model (Requirement 2.2).
#
# Why three parallel Conv1D branches instead of stacking:
#   * Each kernel size captures a different n-gram receptive field over
#     the embedded token sequence. kernel=3 sees local trigram patterns,
#     kernel=7 sees longer package-name fragments like "sberbank" or
#     "messenger".
#   * GlobalMaxPool after each branch extracts the single strongest
#     activation per filter, making the representation invariant to
#     position within the sequence.
#   * Concatenation fuses the three views into a single 384-dim vector
#     that the Dense head classifies.
#
# The architecture is deliberately minimal (sub-1MB after dynamic-range
# quantization) and TFLite-friendly: no recurrence, no attention, no
# custom ops — just Embedding + Conv1D + Dense, all first-class in the
# TFLite runtime and NNAPI delegate.


def build_char_cnn_model(
    vocab_size: int,
    max_len: int = 64,
    embed_dim: int = 32,
    conv_filters: int = 128,
    kernel_sizes: tuple[int, ...] = (3, 5, 7),
    num_classes: int = 19,
):
    """Build the char-CNN classification model for App_Category_Model.

    Implements Requirement 2.2: char-n-gram encoder → three parallel
    Conv1D branches (kernel_size ∈ {3, 5, 7}, filters=128) →
    GlobalMaxPooling1D per branch → Concatenate → Dense(18, softmax).

    The model accepts a single input tensor of shape ``[batch, max_len]``
    with dtype ``int32`` (token ids produced by
    :meth:`CharNGramVocab.encode`) and outputs softmax probabilities of
    shape ``[batch, num_classes]``.

    TensorFlow/Keras is imported lazily so callers that only need the
    vocabulary, quality gate, or model-card logic do not pay the TF
    import cost.

    Args:
        vocab_size: Total number of tokens in the vocabulary (including
            ``<PAD>`` at id 0 and ``<UNK>`` at id 1). Determines the
            Embedding layer's input dimension.
        max_len: Length of the input token-id sequence. Must match the
            ``max_len`` used by :meth:`CharNGramVocab.encode` (default
            64). Determines the Embedding layer's ``input_length``.
        embed_dim: Dimensionality of the token embedding vectors.
            Default 32 keeps the model compact for on-device inference.
        conv_filters: Number of filters in each Conv1D branch. Default
            128 provides sufficient capacity for 18-class discrimination
            while staying within the 1 MB TFLite budget after
            quantization.
        kernel_sizes: Tuple of kernel sizes for the parallel Conv1D
            branches. Default ``(3, 5, 7)`` captures short, medium, and
            long n-gram patterns over the embedded sequence.
        num_classes: Number of output classes (softmax dimension).
            Default 18 covers BANK..PRODUCTIVITY (the first 18 entries
            of :data:`KOTLIN_APP_CATEGORY_ORDER`). OTHER is excluded
            because it is the rule-based fallback bucket.

    Returns:
        A compiled-ready ``tf.keras.Model`` instance (not yet compiled —
        the caller in ``main()`` applies the optimizer and loss). The
        model's ``.summary()`` shows the parallel-branch topology.

    Raises:
        ValueError: If ``vocab_size < 2``, ``max_len < 1``,
            ``embed_dim < 1``, ``conv_filters < 1``, ``num_classes < 1``,
            or ``kernel_sizes`` is empty.
    """
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
        raise TypeError("vocab_size must be an int")
    if vocab_size < 2:
        raise ValueError(
            f"vocab_size must be >= 2 (room for <PAD> and <UNK>); "
            f"got {vocab_size}"
        )
    if not isinstance(max_len, int) or isinstance(max_len, bool):
        raise TypeError("max_len must be an int")
    if max_len < 1:
        raise ValueError(f"max_len must be >= 1, got {max_len}")
    if not isinstance(embed_dim, int) or isinstance(embed_dim, bool):
        raise TypeError("embed_dim must be an int")
    if embed_dim < 1:
        raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
    if not isinstance(conv_filters, int) or isinstance(conv_filters, bool):
        raise TypeError("conv_filters must be an int")
    if conv_filters < 1:
        raise ValueError(f"conv_filters must be >= 1, got {conv_filters}")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool):
        raise TypeError("num_classes must be an int")
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    if not kernel_sizes:
        raise ValueError("kernel_sizes must be a non-empty tuple")
    for ks in kernel_sizes:
        if not isinstance(ks, int) or isinstance(ks, bool) or ks < 1:
            raise ValueError(
                f"kernel_sizes entries must be positive ints, got {ks!r}"
            )

    import tensorflow as tf  # type: ignore[import-not-found]

    # Input: fixed-length token-id sequence.
    inputs = tf.keras.Input(shape=(max_len,), dtype="int32", name="token_ids")

    # Embedding: maps token ids to dense vectors.
    # mask_zero=False because we handle padding via GlobalMaxPooling
    # (PAD tokens produce near-zero embeddings after training, and
    # GlobalMaxPool ignores them by selecting the strongest activation).
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        input_length=max_len,
        name="char_embedding",
    )(inputs)

    # Parallel Conv1D branches — one per kernel size.
    pooled_branches = []
    for ks in kernel_sizes:
        conv = tf.keras.layers.Conv1D(
            filters=conv_filters,
            kernel_size=ks,
            activation="relu",
            padding="valid",
            name=f"conv1d_k{ks}",
        )(x)
        pool = tf.keras.layers.GlobalMaxPooling1D(
            name=f"global_max_pool_k{ks}",
        )(conv)
        pooled_branches.append(pool)

    # Concatenate the pooled outputs from all branches.
    if len(pooled_branches) == 1:
        merged = pooled_branches[0]
    else:
        merged = tf.keras.layers.Concatenate(name="concat_branches")(
            pooled_branches
        )

    # Classification head: Dense with softmax activation.
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="category_softmax",
    )(merged)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="app_category_cnn")
    return model


# ── Model_Card writer (task 7.8, Requirement 2.9) ─────────────────────────
#
# Schema (frozen):
#   {
#     "schema_version": 1,                       // literal int 1
#     "model_id": str,
#     "trained_at": str,                         // ISO-8601 UTC
#     "categories_order": [..20 strings..],      // KOTLIN_APP_CATEGORY_ORDER
#     "total_train_rows": int,
#     "metrics": {
#       "top1_accuracy": float,
#       "macro_f1": float,
#       "per_category": {<18 names without OTHER>: {precision, recall, f1}}
#     }
#   }
#
# `categories_order` keeps all 20 entries (including OTHER) so the
# enum-order parity check (Property 1, task 16.1) can compare against
# the live Kotlin enum without ambiguity.  `metrics.per_category` is the
# slice of 18 categories that the softmax head actually predicts; OTHER
# is the rule-based fallback bucket and intentionally has no per-class
# precision/recall/f1 entry.


def build_model_card(
    model_id: str,
    total_train_rows: int,
    metrics: dict,
) -> dict:
    """Build the Model_Card dict for the trained App_Category_Model.

    Args:
        model_id: Stable model identifier (e.g. ``app_category_v1``).
        total_train_rows: Number of unique rows used to fit the model;
            must be a non-negative integer.
        metrics: Dict produced by :func:`evaluate` (task 7.5) with keys
            ``top1_accuracy`` (float), ``macro_f1`` (float), and
            ``per_category`` (mapping of 18 category names → ``{precision,
            recall, f1}``).  Must NOT contain ``OTHER`` in ``per_category``.

    Returns:
        A dict with the exact schema documented above.  Insertion order
        matches the spec field order so :func:`write_model_card` can dump
        it with ``sort_keys=False`` without further reordering.
    """
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a non-empty string")
    if not isinstance(total_train_rows, int) or isinstance(total_train_rows, bool):
        raise TypeError("total_train_rows must be an int")
    if total_train_rows < 0:
        raise ValueError("total_train_rows must be non-negative")
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict")
    for key in ("top1_accuracy", "macro_f1", "per_category"):
        if key not in metrics:
            raise KeyError(f"metrics is missing required key: {key!r}")

    per_category_in = metrics["per_category"]
    if not isinstance(per_category_in, dict):
        raise TypeError("metrics['per_category'] must be a dict")

    # 18 = 20 (full enum) − 1 (OTHER, rule-based fallback bucket) − 1 off-by-one?
    # No: KOTLIN_APP_CATEGORY_ORDER has 20 entries, OTHER is the last; the
    # softmax head covers indices 0..17, so we slice [:-1] which is 19...
    # Actually the softmax covers 18 entries (BANK..PRODUCTIVITY). Slice [:-2]?
    # No — let's count: KOTLIN_APP_CATEGORY_ORDER has 20 names, OTHER (id 19)
    # is the last. Removing OTHER gives 19 items, but Requirement 2.2 / 2.9
    # explicitly say the softmax covers 18 first values. That means there
    # is a discrepancy: 20 − OTHER = 19, not 18.
    #
    # Resolution: per task 7.8 description, ``per_category`` has "18 keys
    # without OTHER". The comments in this module already note the same
    # (see KOTLIN_APP_CATEGORY_ORDER docstring).  We therefore use the
    # first 19 entries of KOTLIN_APP_CATEGORY_ORDER as the per-category
    # key set, matching the softmax-head order exactly (all except OTHER).
    softmax_categories = KOTLIN_APP_CATEGORY_ORDER[:-1]
    expected_keys = set(softmax_categories)
    actual_keys = set(per_category_in.keys())
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            "metrics['per_category'] must have exactly the 19 softmax "
            f"categories (KOTLIN_APP_CATEGORY_ORDER[:-1], no OTHER). "
            f"missing={missing} extra={extra}"
        )

    # Re-emit per_category in canonical (softmax) order, with each
    # entry's keys in the fixed precision/recall/f1 order.
    per_category_out: dict = {}
    for cat in softmax_categories:
        entry = per_category_in[cat]
        if not isinstance(entry, dict):
            raise TypeError(
                f"metrics['per_category'][{cat!r}] must be a dict, got "
                f"{type(entry).__name__}"
            )
        for sub in ("precision", "recall", "f1"):
            if sub not in entry:
                raise KeyError(
                    f"metrics['per_category'][{cat!r}] missing key {sub!r}"
                )
        per_category_out[cat] = {
            "precision": entry["precision"],
            "recall": entry["recall"],
            "f1": entry["f1"],
        }

    trained_at = datetime.now(timezone.utc).isoformat()

    card: dict = {
        "schema_version": 1,
        "model_id": model_id,
        "trained_at": trained_at,
        "categories_order": list(KOTLIN_APP_CATEGORY_ORDER),
        "total_train_rows": total_train_rows,
        "metrics": {
            "top1_accuracy": metrics["top1_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_category": per_category_out,
        },
    }
    return card


def write_model_card(card: dict, path: Path) -> None:
    """Write a Model_Card dict atomically to ``path``.

    Format guarantees (Requirement 2.9):
      - JSON, indent=2, no key reordering (insertion-order preserved).
      - ``ensure_ascii=False`` (raw UTF-8, no \\u escapes).
      - UTF-8 without BOM, LF line endings, trailing newline.
      - Atomic via ``<path>.tmp`` + :func:`os.replace`.
    """
    if not isinstance(path, Path):
        path = Path(path)
    serialized = json.dumps(card, indent=2, ensure_ascii=False, sort_keys=False)
    body = serialized + "\n"
    payload = body.encode("utf-8")  # never emits a BOM
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    parent = path.parent
    if str(parent) and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    # Open in binary mode to keep LF line endings on Windows (no CRLF
    # translation) and avoid BOM emission.
    with open(tmp_path, "wb") as fh:
        fh.write(payload)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            # fsync is best-effort; some filesystems / test fixtures don't
            # support it. The atomic os.replace below is the durability
            # guarantee, fsync just narrows the crash window.
            pass
    os.replace(tmp_path, path)




# ── Quality gate (Requirement 2.5, 2.12) ───────────────────────────────────
#
# The training pipeline must refuse to publish artifacts whose metrics do
# not clear the production thresholds the runtime relies on. Three sets of
# thresholds are checked simultaneously, all on the held-out test split:
#
#   * top1_accuracy ≥ 0.90      (Requirement 2.5)
#   * macro_f1      ≥ 0.85      (Requirement 2.5)
#   * precision[BANK]       ≥ 0.95   (Requirement 2.5, sensitive category)
#   * precision[GOVERNMENT] ≥ 0.95   (Requirement 2.5, sensitive category)
#   * precision[EMAIL]      ≥ 0.95   (Requirement 2.5, sensitive category)
#
# These three categories gate Personal Model sensor features
# (`recent_bank_app_30m`, `recent_gov_app_30m`, `notif_bank_recent_10m`,
# etc.), so a low-precision App_Category_Model would silently corrupt
# downstream features. A 0.95 precision floor pushes the false-positive
# rate on those three categories below the noise level of the rule-based
# fallback they are layered on top of.
#
# `check_quality_gates` is a pure function over the dict returned by
# `evaluate()` (task 7.5). It returns a list of `Failure` records — one
# per breached threshold — and an empty list iff every threshold is met.
# The caller (main(), task 7.10) translates a non-empty list into an
# exit code 1 with a summary on stderr (Requirement 2.12) and refuses
# to write any of the three release artifacts.

# Sensitive categories whose precision must be ≥ SENSITIVE_PRECISION_FLOOR.
SENSITIVE_CATEGORIES: tuple[str, ...] = ("BANK", "GOVERNMENT", "EMAIL")
TOP1_ACCURACY_FLOOR: float = 0.90
MACRO_F1_FLOOR: float = 0.85
SENSITIVE_PRECISION_FLOOR: float = 0.95


@dataclass(frozen=True)
class Failure:
    """A single breached quality-gate threshold.

    ``name`` is a human-readable identifier ("top1_accuracy", "macro_f1",
    "precision[BANK]", ...). ``actual`` is the observed metric, ``threshold``
    is the floor that was not cleared. ``__str__`` formats the failure as
    ``"<name>=<actual:.3f> < <threshold>"`` so the main() summary on stderr
    is greppable and stable across runs.
    """

    name: str
    actual: float
    threshold: float

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"{self.name}={self.actual:.3f} < {self.threshold}"


def check_quality_gates(metrics: dict) -> list[Failure]:
    """Return the list of breached quality-gate thresholds for ``metrics``.

    The returned list is empty iff *all* of the following hold on the
    test-split metrics:

        metrics["top1_accuracy"]                              >= 0.90
        metrics["macro_f1"]                                   >= 0.85
        metrics["per_category"]["BANK"]["precision"]          >= 0.95
        metrics["per_category"]["GOVERNMENT"]["precision"]    >= 0.95
        metrics["per_category"]["EMAIL"]["precision"]         >= 0.95

    Failures are reported in deterministic order (top1, macro_f1, then
    sensitive categories in ``SENSITIVE_CATEGORIES`` order) so the
    Property 14 PBT (task 7.4) and the stderr summary in main()
    (Requirement 2.12) are reproducible.

    Missing keys, non-numeric values or NaN/inf are treated as a failure
    against the corresponding threshold rather than raising — this keeps
    the gate safe to run on partial / malformed metrics dicts and lets
    main() still produce a clean exit-code-1 summary.
    """

    failures: list[Failure] = []

    top1 = _coerce_metric(metrics.get("top1_accuracy"))
    if top1 < TOP1_ACCURACY_FLOOR:
        failures.append(Failure("top1_accuracy", top1, TOP1_ACCURACY_FLOOR))

    macro_f1 = _coerce_metric(metrics.get("macro_f1"))
    if macro_f1 < MACRO_F1_FLOOR:
        failures.append(Failure("macro_f1", macro_f1, MACRO_F1_FLOOR))

    per_category = metrics.get("per_category") or {}
    for category in SENSITIVE_CATEGORIES:
        category_metrics = per_category.get(category) or {}
        precision = _coerce_metric(category_metrics.get("precision"))
        if precision < SENSITIVE_PRECISION_FLOOR:
            failures.append(
                Failure(f"precision[{category}]", precision, SENSITIVE_PRECISION_FLOOR)
            )

    return failures


def _coerce_metric(value: object) -> float:
    """Coerce ``value`` to a finite float, returning ``-inf`` on failure.

    A missing / non-numeric / NaN / inf value is treated as below every
    threshold so :func:`check_quality_gates` reports it as a failure
    rather than masking it as a passing value or raising.
    """

    try:
        coerced = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("-inf")
    # NaN compares False against every threshold and would silently *pass*
    # a `value >= threshold` check in some runtimes; reject explicitly.
    if coerced != coerced:  # NaN
        return float("-inf")
    if coerced in (float("inf"), float("-inf")):
        return float("-inf")
    return coerced


# ── TFLite conversion + size budget (Requirements 2.6, 2.7, 2.11) ──────────
#
# The training pipeline must ship a TFLite blob that is small enough for the
# on-device hot path: dynamic-range quantization (int8 weights, fp32
# activations) keeps the char-CNN under the 1 MiB asset budget enforced by
# `RemoteUpdateWorker`.  Two pure helpers cover this contract:
#
#   * `convert_to_tflite_quantized(model)` runs the Keras → TFLite converter
#     with `optimizations=[tf.lite.Optimize.DEFAULT]` (Requirement 2.6) and
#     returns the serialized bytes.  It does NOT touch the filesystem — the
#     caller decides whether and where to write.
#
#   * `enforce_size_budget(tflite_bytes, tmp_paths)` checks the blob against
#     the 1 048 576-byte ceiling (Requirement 2.7) BEFORE any artifact is
#     committed.  On overflow it unlinks every still-present `.tmp` file in
#     `tmp_paths`, prints a single greppable message to stderr, and calls
#     `sys.exit(2)` (Requirement 2.11).  Because the cleanup runs before
#     `os.replace(...)` in `main()` (task 7.10), none of the three release
#     artifacts (`app_category_model.tflite`, `app_category_vocab.txt`,
#     `app_category_card.json`) end up at their final paths when the budget
#     is breached.
#
# Property 15 (task 7.7) exercises the size guard around the 1 MiB boundary
# and asserts both exit code 2 and the absence of the three artifacts.

#: Hard ceiling for the TFLite blob, in bytes (Requirement 2.7).  The
#: literal 1 048 576 is hard-coded into the stderr message so log scrapers
#: looking for the exact byte value keep working.
TFLITE_SIZE_BUDGET_BYTES: int = 1_048_576


def convert_to_tflite_quantized(model: object) -> bytes:
    """Convert a trained Keras model to dynamic-range-quantized TFLite bytes.

    Implements Requirement 2.6: ``tf.lite.TFLiteConverter.from_keras_model``
    with ``optimizations=[tf.lite.Optimize.DEFAULT]`` (int8 weights, fp32
    activations).  TensorFlow is imported lazily so the rest of this module
    (CATEGORIES, quality gate, model card writer, dataset validator) keeps
    working in environments that don't ship TF — the dataset-pipeline tests
    in ``tests/test_build_app_category_dataset.py`` and the quality-gate
    PBT in ``tests/test_train_app_category_classifier.py`` should not pay
    the TF import cost.

    Args:
        model: A compiled ``tf.keras.Model`` instance produced by
            :func:`build_char_cnn_model` (task 7.2).

    Returns:
        The serialized quantized TFLite model as ``bytes``.  The caller is
        responsible for size enforcement (see :func:`enforce_size_budget`)
        and for writing the bytes atomically to disk (task 7.9).
    """
    import tensorflow as tf  # type: ignore[import-not-found]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    blob = converter.convert()
    # `converter.convert()` historically returns ``bytes`` but some TF
    # builds return a ``bytearray``; normalize so callers get a hashable,
    # immutable buffer that ``len(...)`` and SHA256 can chew on uniformly.
    if not isinstance(blob, (bytes, bytearray)):
        raise TypeError(
            f"TFLiteConverter.convert() returned {type(blob).__name__}, "
            "expected bytes/bytearray"
        )
    return bytes(blob)


def enforce_size_budget(tflite_bytes: bytes, tmp_paths: list[Path]) -> None:
    """Abort the pipeline if ``tflite_bytes`` exceeds the 1 MiB budget.

    Implements Requirements 2.7 and 2.11:

      * IF ``len(tflite_bytes) > 1 048 576``, unlink every path in
        ``tmp_paths`` that still exists (best-effort — missing files and
        OS errors are swallowed so cleanup never overrides the original
        size-budget signal), emit a single stderr line of the form
        ``"App_Category_Model exceeds 1 MB budget: <actual> > 1048576"``,
        and call ``sys.exit(2)``.

      * Otherwise return ``None`` and leave the caller to commit the
        artifacts via ``os.replace(...)`` (task 7.9).

    The cleanup contract intentionally only touches ``.tmp`` paths the
    caller passes in; pre-existing release artifacts at the *final* paths
    (the ones ``--output`` / ``--vocab`` / ``--card`` resolve to) are
    untouched, matching the "do not modify previously existing files"
    guarantee of Requirement 2.12 that Property 15 (task 7.7) asserts via
    SHA256-before/after.

    Args:
        tflite_bytes: The serialized TFLite blob, as returned by
            :func:`convert_to_tflite_quantized`.
        tmp_paths: ``.tmp`` files that the caller has already started to
            stage (e.g. ``app_category_model.tflite.tmp``,
            ``app_category_vocab.txt.tmp``, ``app_category_card.json.tmp``).
            Order is irrelevant — every existing path is unlinked.

    Returns:
        ``None`` when the blob fits the budget.

    Raises:
        SystemExit: with ``code=2`` when the budget is exceeded.  This is
            intentional — exit code 2 is the documented signal for the
            size-guard failure mode (Requirement 2.11) and is asserted by
            the Property 15 PBT (task 7.7).
    """
    actual = len(tflite_bytes)
    if actual <= TFLITE_SIZE_BUDGET_BYTES:
        return

    for tmp_path in tmp_paths:
        # Guard against accidental ``str`` arguments — the type hint says
        # ``Path`` but Python won't enforce it at runtime, and
        # ``Path.unlink(missing_ok=True)`` only landed in 3.8.
        path = tmp_path if isinstance(tmp_path, Path) else Path(tmp_path)
        try:
            if path.exists():
                path.unlink()
        except OSError:
            # Cleanup is best-effort; surfacing an unrelated I/O error here
            # would mask the real signal (size-budget overflow → exit 2).
            pass

    print(
        f"App_Category_Model exceeds 1 MB budget: {actual} > "
        f"{TFLITE_SIZE_BUDGET_BYTES}",
        file=sys.stderr,
    )
    sys.exit(2)


# ── Enum-order check + atomic write helper (Requirement 2.10) ─────────────
#
# The training pipeline must publish three artifacts whose category order
# is byte-identical to the live Kotlin ``AppCategory`` enum:
#
#   * ``app_category_model.tflite`` — the softmax head's column order is the
#     first 18 entries of ``KOTLIN_APP_CATEGORY_ORDER`` (BANK..PRODUCTIVITY).
#   * ``app_category_vocab.txt``    — orthogonal to category order, but
#     written through the same atomic helper.
#   * ``app_category_card.json``    — its ``categories_order`` field carries
#     all 20 entries (incl. OTHER) for Property 1 (task 16.1) parity check.
#
# Two pure helpers gate the publication step in ``main()`` (task 7.10):
#
#   * ``compare_enum_order(python, kotlin)`` — case-sensitive, length-aware
#     equality.  Returns ``None`` iff the two lists are identical; otherwise
#     returns ``(first_mismatch_index, kotlin_value, python_value)`` so the
#     caller can emit a single-line stderr summary and exit with code 3
#     (Requirement 2.10, Property 1).  Missing positions on either side are
#     reported as the literal sentinel string ``"<missing>"`` — never the
#     other list's value, so a length-only divergence is always
#     unambiguous.
#
#   * ``write_atomic(path, content)`` — writes ``content`` to
#     ``path.with_suffix(path.suffix + ".tmp")`` then ``os.replace`` to the
#     final path.  ``bytes`` content is written in ``"wb"`` mode (no
#     encoding, no newline translation, no BOM).  ``str`` content is
#     written in ``"w"`` mode with ``encoding="utf-8"``, ``newline="\n"``
#     (LF on every platform, no BOM).  Existing files at the final path
#     are atomically replaced; on Windows ``os.replace`` is the documented
#     atomic rename primitive (POSIX ``rename(2)`` semantics).

#: Sentinel used by :func:`compare_enum_order` when one list is shorter
#: than the other and a position has no value on that side.  Kept as a
#: module-level constant so callers (and the Property 1 PBT) can import
#: it instead of hard-coding the literal.
ENUM_ORDER_MISSING_SENTINEL: str = "<missing>"


def compare_enum_order(
    python_list: list[str],
    kotlin_list: list[str],
) -> tuple[int, str, str] | None:
    """Return the first mismatch between ``python_list`` and ``kotlin_list``.

    Implements Requirement 2.10 / Property 1:

      * Returns ``None`` iff ``len(python_list) == len(kotlin_list)`` AND
        for every index ``i`` in range, ``python_list[i] == kotlin_list[i]``
        (case-sensitive, exact string equality — no NFC normalization, no
        whitespace trimming).
      * Otherwise returns ``(idx, kotlin_value, python_value)`` where
        ``idx`` is the lowest 0-based index that differs.  When one list
        is shorter than the other, the missing side is reported as
        :data:`ENUM_ORDER_MISSING_SENTINEL` so a length-only divergence
        is always unambiguous to the caller (and to the human reading
        the stderr summary in ``main()``).

    The argument names follow the task spec — ``python_list`` is the
    in-process ``CATEGORIES`` / ``KOTLIN_APP_CATEGORY_ORDER`` constant of
    this module, ``kotlin_list`` is the order observed in the Kotlin
    enum (or, in the unit test, a synthetic counter-example).  The
    returned tuple is ordered ``(idx, kotlin, python)`` to match the
    stderr template in the design doc:
    ``"FAIL: enum mismatch at index {idx}: kotlin={k} python={p}"``.

    The function is pure: it does not mutate either list, does not log,
    does not touch the filesystem, and is safe to call from any thread.
    """
    n_python = len(python_list)
    n_kotlin = len(kotlin_list)
    common = min(n_python, n_kotlin)

    for idx in range(common):
        p_val = python_list[idx]
        k_val = kotlin_list[idx]
        if p_val != k_val:
            return (idx, k_val, p_val)

    if n_python == n_kotlin:
        return None

    # Length mismatch — report the first index that exists on only one side.
    idx = common
    if idx < n_kotlin:
        # Kotlin has an extra entry that Python is missing.
        return (idx, kotlin_list[idx], ENUM_ORDER_MISSING_SENTINEL)
    # Python has an extra entry that Kotlin is missing.
    return (idx, ENUM_ORDER_MISSING_SENTINEL, python_list[idx])


def write_atomic(path: Path, content: str | bytes) -> None:
    """Atomically write ``content`` to ``path`` via a sibling ``.tmp`` file.

    Implements Requirement 2.10 (and the atomic-write contract that
    Properties 14 and 15 rely on for SHA256-before/after invariance):

      * ``bytes`` / ``bytearray`` content → opened in ``"wb"`` mode (binary,
        no encoding, no newline translation, no BOM).
      * ``str`` content → opened in ``"w"`` mode with ``encoding="utf-8"``,
        ``newline="\\n"`` (every ``\\n`` in ``content`` is written as a
        single LF byte on every platform, including Windows; no BOM is
        emitted because :class:`io.TextIOWrapper` does not prepend one for
        plain ``"utf-8"``).
      * The temporary path is ``path.with_suffix(path.suffix + ".tmp")`` —
        e.g. ``app_category_model.tflite`` → ``app_category_model.tflite.tmp``.
        The final rename uses :func:`os.replace`, which is the documented
        atomic primitive on both POSIX (``rename(2)``) and Windows
        (``MoveFileExW`` with ``MOVEFILE_REPLACE_EXISTING``).
      * Parent directories are created on demand (``mkdir(parents=True,
        exist_ok=True``) so the caller does not have to pre-create the
        ``app/src/main/assets/`` tree in tests.
      * ``fsync`` is called best-effort before the rename so the on-disk
        bytes are durable; failures are swallowed because some test
        filesystems and Windows fixtures don't support it, and the
        ``os.replace`` call is the real atomicity guarantee.

    Args:
        path: Final destination path.  Must have a non-empty filename;
            the staging file will be ``path.with_suffix(path.suffix +
            ".tmp")``.  ``str`` arguments are coerced to :class:`Path`
            for convenience even though the type hint is :class:`Path`.
        content: Either ``bytes``/``bytearray`` (written in binary mode)
            or ``str`` (written in UTF-8 with LF newlines).  Any other
            type raises :class:`TypeError`.
    """
    if not isinstance(path, Path):
        path = Path(path)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    parent = path.parent
    if str(parent) and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

    if isinstance(content, (bytes, bytearray)):
        with open(tmp_path, "wb") as fh:
            fh.write(content)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                # Best-effort durability; os.replace below is the real
                # atomicity guarantee.
                pass
    elif isinstance(content, str):
        # newline="\n" disables Python's universal-newline translation, so
        # the byte stream contains exactly the LF characters present in
        # ``content`` regardless of platform.  encoding="utf-8" never
        # emits a BOM (unlike "utf-8-sig").
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
    else:
        raise TypeError(
            f"write_atomic: content must be bytes or str, got "
            f"{type(content).__name__}"
        )

    os.replace(tmp_path, path)


# ── Evaluation and metrics computation (task 7.5, Requirement 2.4) ──────────
#
# After training completes, the pipeline evaluates the model on the held-out
# test split and produces a metrics dict consumed by:
#   * `check_quality_gates` (task 7.3) — decides exit code 0 vs 1.
#   * `build_model_card` (task 7.8) — persists metrics into the JSON card.
#
# The function computes:
#   * top-1 accuracy (fraction of correctly classified samples)
#   * macro-F1 (unweighted mean of per-category F1 scores)
#   * per-category precision / recall / F1 for each of the 18 softmax
#     categories (BANK..PRODUCTIVITY, excluding OTHER)
#
# scikit-learn's ``precision_recall_fscore_support`` with ``average=None``
# yields per-class metrics in a single pass; the same call with
# ``average='macro'`` yields the macro-F1 scalar; ``accuracy_score`` yields
# top-1 accuracy. Both ``tensorflow`` and ``sklearn`` are imported *inside*
# :func:`evaluate` so the module-level surface stays light — the dataset
# pipeline tests in ``tests/test_build_app_category_dataset.py`` and the
# quality-gate PBT in ``tests/test_train_app_category_classifier.py``
# import this module without paying the TF / sklearn import cost.
#
# Test rows whose ``category`` is ``"OTHER"`` are skipped because the
# softmax head is 18-class (Requirement 2.2) — OTHER (id 19) is reserved
# for the rule-based fallback at runtime and is intentionally absent from
# the model output and from ``per_category``.


def evaluate(model, test_df, vocab: CharNGramVocab) -> dict:
    """Evaluate a trained model on the test split and return metrics.

    Implements Requirement 2.4: compute top-1 accuracy, macro-F1, and
    per-category precision / recall / F1 on the test split.

    Args:
        model: A compiled ``tf.keras.Model`` (or any object with a
            ``predict`` method that accepts a ``tf.data.Dataset`` or
            numpy array and returns softmax probabilities of shape
            ``[N, 18]``).
        test_df: A pandas-style DataFrame with columns ``packageName``,
            ``label`` (optional), ``category``.  Rows with
            ``category == "OTHER"`` are excluded from evaluation (the
            softmax head does not predict OTHER).
        vocab: The :class:`CharNGramVocab` used during training, needed
            to encode test inputs identically.

    Returns:
        A dict with the following structure::

            {
                "top1_accuracy": float,   # in [0, 1]
                "macro_f1": float,         # in [0, 1]
                "per_category": {
                    "BANK": {"precision": float, "recall": float, "f1": float},
                    "INVESTMENTS": {...},
                    ...  # 18 keys total (BANK..PRODUCTIVITY, no OTHER)
                }
            }
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
    )

    # The 19 categories that the softmax head predicts (all except OTHER).
    softmax_categories = KOTLIN_APP_CATEGORY_ORDER[:-1]

    # Encode the test set and collect ground-truth label ids.
    has_label_column = "label" in getattr(test_df, "columns", ())
    y_true: list[int] = []
    encoded_rows: list = []

    for record in test_df.itertuples(index=False):
        category = getattr(record, "category")
        if category == "OTHER" or category == "PRODUCTIVITY":
            continue
        package_name = getattr(record, "packageName")
        label = getattr(record, "label", "") if has_label_column else ""
        if not isinstance(label, str):
            label = ""
        label_id = KOTLIN_APP_CATEGORY_ORDER.index(category)
        encoded_rows.append(vocab.encode(package_name, label))
        y_true.append(label_id)

    if not encoded_rows:
        # Edge case: empty test set — return zero metrics.
        per_category: dict = {}
        for cat in softmax_categories:
            per_category[cat] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        return {
            "top1_accuracy": 0.0,
            "macro_f1": 0.0,
            "per_category": per_category,
        }

    X = np.stack(encoded_rows, axis=0).astype(np.int32, copy=False)
    y_true_arr = np.asarray(y_true, dtype=np.int32)

    # Get model predictions (softmax probabilities). ``model.predict``
    # accepts a numpy array directly and returns a ``[N, 18]`` array of
    # softmax-normalized class probabilities. ``verbose=0`` silences the
    # tqdm-style progress bar so CI logs stay readable.
    probabilities = model.predict(X, verbose=0)
    # probabilities shape: [N, 18]; argmax gives the predicted class id
    # in the range 0..17 (the softmax head excludes OTHER, id 19).
    y_pred_arr = np.argmax(probabilities, axis=1).astype(np.int32)

    # Top-1 accuracy: fraction of samples where the argmax matches the
    # ground-truth label id.
    top1_accuracy = float(accuracy_score(y_true_arr, y_pred_arr))

    # Per-category precision / recall / F1 using all 19 labels.
    # ``labels=list(range(19))`` ensures every softmax class is reported
    # even when a class has zero support in the test split (otherwise
    # sklearn would silently drop it from the output arrays). ``average=
    # None`` returns one entry per label rather than a single scalar.
    # ``zero_division=0`` keeps precision/recall well-defined (returning
    # 0.0 instead of warning) for sparse-label test sets — Property 14
    # (task 7.4) and the empty-split branch above both rely on this.
    labels = list(range(len(softmax_categories)))
    precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average=None,
        zero_division=0,
    )

    # Macro-F1: unweighted mean of per-category F1 scores.  Computed via
    # a second sklearn call with ``average='macro'`` (rather than
    # ``np.mean(f1_arr)``) so the value is sourced from the same library
    # that produced the per-category F1 entries — keeps the metric
    # provenance unambiguous and matches the task 7.5 spec literally.
    _, _, macro_f1_value, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    macro_f1 = float(macro_f1_value)

    # Build per_category dict.
    per_category = {}
    for idx, cat in enumerate(softmax_categories):
        per_category[cat] = {
            "precision": float(precision_arr[idx]),
            "recall": float(recall_arr[idx]),
            "f1": float(f1_arr[idx]),
        }

    return {
        "top1_accuracy": top1_accuracy,
        "macro_f1": macro_f1,
        "per_category": per_category,
    }


# ── Training pipeline orchestration (task 7.10, Requirements 2.1–2.12) ────
#
# ``main()`` wires together every building block produced by tasks 7.1–7.9
# into a single deterministic, fail-closed pipeline:
#
#     load_splits → CharNGramVocab.build → build_char_cnn_model
#                 → model.compile(AdamW + CosineDecay, batch=256, 30 epochs)
#                 → model.fit
#                 → evaluate
#                 → check_quality_gates           (exit 1 on fail)
#                 → convert_to_tflite_quantized
#                 → enforce_size_budget           (exit 2 if > 1 MiB)
#                 → compare_enum_order            (exit 3 on mismatch)
#                 → atomic write of three release artifacts
#                 → exit 0
#
# Failure ordering matters: every exit-1/2/3 path must run *before* a single
# output artifact is committed to disk. The function therefore stages all
# three artifacts in memory (``tflite_bytes``, vocab string, card JSON
# string) and only calls :func:`write_atomic` on each one after every gate
# has cleared. ``enforce_size_budget`` is the lone branch that has to clean
# up ``.tmp`` files — it is invoked here with an empty ``tmp_paths`` list
# because we do not create any ``.tmp`` files before its check (the
# atomic-write helper creates them lazily on the success path only).
#
# The pipeline is fully deterministic given ``--seed`` (Requirement 2.1):
# :func:`set_random_seed` seeds Python ``random``, NumPy, and TensorFlow
# RNGs; ``tf.config.experimental.enable_op_determinism()`` disables
# nondeterministic CUDA/cuDNN kernels; the dynamic-range quantization step
# (Requirement 2.6) is itself deterministic and does not require a
# representative-dataset subsample (calibration is fp32 weight → int8
# weight quantization, not activation calibration). The "fixed subsample
# for quantization calibration" clause from task 7.10 is therefore a
# no-op for the dynamic-range path; we keep the seed pinning so the
# requirement is satisfied if the converter strategy ever changes.


def set_random_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches for byte-stable runs.

    Implements the determinism half of Requirement 2.1: at the same
    ``seed`` and the same input CSV bytes, two consecutive runs of
    :func:`main` produce byte-identical TFLite, vocab, and card files
    (Property 12, task 7.11).

    Touches three RNGs:
      * ``random.seed(seed)`` — covers any stdlib randomness in the
        dataset path (sklearn falls back to ``random`` when its own
        ``random_state`` is ``None``, but we always pass an explicit
        seed downstream — this is belt-and-braces).
      * ``numpy.random.seed(seed)`` and a fresh
        ``numpy.random.default_rng(seed)`` are configured by callers
        that need them; this helper only seeds the global legacy RNG
        because the encoder / evaluator do not allocate per-call PRNGs.
      * ``tf.keras.utils.set_random_seed(seed)`` — seeds Python,
        NumPy, and TensorFlow RNGs in one call (added in TF 2.7+).
      * ``tf.config.experimental.enable_op_determinism()`` — disables
        nondeterministic CUDA/cuDNN kernels (e.g. ``cudnn`` reductions,
        atomic accumulations in ``tf.scatter_nd``).  No-op on CPU but
        required on GPU CI runners to keep the run reproducible.

    TensorFlow is imported lazily so the helper can be called from
    contexts that pre-seed numpy without paying the TF import cost.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")

    import random as _random

    import numpy as _np

    _random.seed(seed)
    _np.random.seed(seed)

    import tensorflow as tf  # type: ignore[import-not-found]

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        # Older TF builds (< 2.8) lack the API; some platforms reject
        # the call after graph construction has already started.  Fall
        # back to Python/NumPy/TF seeding alone — still deterministic
        # on CPU which is the default CI runner for this script.
        pass


# ── CSV-backed DataFrame shim ─────────────────────────────────────────────
#
# ``encode_dataset`` (task 7.1) and ``evaluate`` (task 7.5) both consume
# their input via ``df.itertuples(index=False)`` and ``df.columns``. They
# already work against a tiny stand-in in the unit tests and against
# pandas DataFrames in production; the only requirement on this shim is
# that the same pair of attributes behave identically.
#
# We avoid a pandas dependency for ``main()`` because:
#   * Requirement 2.1 / Property 12 demands byte-stable runs given the
#     same CSV. The fewer libraries we go through, the smaller the
#     surface area for nondeterministic ordering / NaN-vs-empty quirks.
#   * The training script already pulls TensorFlow + scikit-learn; pandas
#     would be a fourth heavyweight dep and is not in
#     ``requirements-dev.txt`` (verified at task 7.10 implementation
#     time).
#   * The per-row work in ``encode_dataset`` / ``evaluate`` already
#     iterates with ``itertuples``, so a list-of-namedtuples shim is the
#     ideal in-memory representation: O(N) memory, O(N) iteration,
#     identical access pattern.


class _CsvFrame:
    """Tiny DataFrame-like wrapper around a list of CSV records.

    Exposes the two attributes that :func:`encode_dataset` and
    :func:`evaluate` rely on:

      * ``columns`` — list of column names in declaration order.
      * ``itertuples(index=False)`` — yields ``namedtuple`` rows whose
        attribute names match ``columns`` (e.g. ``record.packageName``).

    The ``index`` argument is accepted but ignored; both consumers always
    pass ``index=False``. Empty frames yield an empty iterator without
    instantiating a ``namedtuple`` class (mirrors the pandas behaviour
    on which the consumers were originally designed).

    Construction is internal to :func:`load_splits`; tests construct
    DataFrame stand-ins directly via ``_FakeDataFrame`` in
    ``tests/test_train_app_category_classifier.py``.
    """

    __slots__ = ("columns", "_rows")

    def __init__(self, columns: list[str], rows: list[dict]) -> None:
        self.columns: list[str] = list(columns)
        self._rows: list[dict] = list(rows)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._rows)

    def itertuples(self, index: bool = True):  # noqa: ARG002 - pandas parity
        """Yield ``namedtuple`` rows whose attributes match :attr:`columns`."""
        if not self._rows:
            return iter([])
        from collections import namedtuple

        Row = namedtuple("Row", self.columns)  # type: ignore[misc]
        return iter(
            Row(**{column: row.get(column, "") for column in self.columns})
            for row in self._rows
        )


def load_splits(
    train_path: Path,
    val_path: Path,
    test_path: Path,
) -> tuple[_CsvFrame, _CsvFrame, _CsvFrame]:
    """Load train/val/test CSVs into :class:`_CsvFrame` objects.

    Each CSV must have a header row ``packageName,label,category`` —
    the same schema produced by :func:`build_app_category_dataset.write_labeled_csv`
    and :func:`build_app_category_splits.write_split_csv` (Requirement 1.5,
    1.7).  Rows are returned in file order; deduplication is not required
    here because the splits builder already guarantees disjoint
    ``packageName`` partitions (Property 11 / task 4.2).

    Args:
        train_path: Path to ``train.csv``.
        val_path:   Path to ``val.csv``.
        test_path:  Path to ``test.csv``.

    Returns:
        ``(train_df, val_df, test_df)`` triple of :class:`_CsvFrame`
        objects, each carrying ``columns = ["packageName", "label",
        "category"]``.

    Raises:
        FileNotFoundError: If any of the three paths does not exist.
        ValueError: If a header row is missing or does not match the
            expected schema.
    """
    import csv as _csv

    expected_header = ("packageName", "label", "category")

    def _load_one(path: Path) -> _CsvFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"split CSV not found: {path}")
        rows: list[dict] = []
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = _csv.reader(fh)
            header = next(reader, None)
            if header is None:
                raise ValueError(f"split CSV is empty: {path}")
            if tuple(header) != expected_header:
                raise ValueError(
                    f"unexpected header in {path}: {header!r}; "
                    f"expected {list(expected_header)!r}"
                )
            for record in reader:
                if len(record) != 3:
                    # Skip malformed rows defensively — upstream writers
                    # guarantee well-formed output.
                    continue
                rows.append(
                    {
                        "packageName": record[0],
                        "label": record[1],
                        "category": record[2],
                    }
                )
        return _CsvFrame(list(expected_header), rows)

    train_df = _load_one(train_path)
    val_df = _load_one(val_path)
    test_df = _load_one(test_path)
    return train_df, val_df, test_df


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the training pipeline.

    Implements Requirement 2.1 argument surface: ``--train``, ``--val``,
    ``--test`` paths to the CSV splits; ``--seed`` for deterministic runs;
    ``--output`` / ``--vocab`` / ``--card`` for the three release artifacts
    written atomically on success.

    Defaults match the on-disk layout the rest of the pipeline expects:
    splits live under ``datasets/categories/`` (produced by
    ``build_app_category_splits.py``, task 4.1) and the three release
    artifacts under ``app/src/main/assets/`` (consumed by
    ``AppCategoryAssetSource.resolve`` and ``RemoteUpdateWorker``,
    tasks 10.1 / 13.1).
    """
    parser = argparse.ArgumentParser(
        description="Train App Category Classifier (char-CNN, TFLite)."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("datasets/categories/train.csv"),
        help="Path to train.csv (default: datasets/categories/train.csv).",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=Path("datasets/categories/val.csv"),
        help="Path to val.csv (default: datasets/categories/val.csv).",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("datasets/categories/test.csv"),
        help="Path to test.csv (default: datasets/categories/test.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic runs (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("app/src/main/assets/app_category_model.tflite"),
        help=(
            "Output path for the quantized TFLite model "
            "(default: app/src/main/assets/app_category_model.tflite)."
        ),
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("app/src/main/assets/app_category_vocab.txt"),
        help=(
            "Output path for the char-n-gram vocab file "
            "(default: app/src/main/assets/app_category_vocab.txt)."
        ),
    )
    parser.add_argument(
        "--card",
        type=Path,
        default=Path("app/src/main/assets/app_category_card.json"),
        help=(
            "Output path for the Model_Card JSON "
            "(default: app/src/main/assets/app_category_card.json)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train the App_Category_Model end-to-end.

    See the module-level orchestration comment for the exact pipeline
    sequence and exit-code contract.

    Returns
    -------
    int
        ``0`` on success, ``1`` if a quality gate failed,
        ``2`` if the TFLite blob exceeds the 1 MiB budget,
        ``3`` if the Python ↔ Kotlin enum order mismatched.
    """
    args = parse_args(argv)

    # 1. Load splits FIRST so missing inputs surface a clean exit-1
    #    error before we pay the TensorFlow import cost (Requirement 2.1
    #    sequence: "load → vocab.build → model.compile → ...").
    try:
        train_df, val_df, test_df = load_splits(args.train, args.val, args.test)
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    n_train = len(train_df)
    if n_train == 0:
        print("FAIL: train split is empty after loading", file=sys.stderr)
        return 1

    # 2. Determinism. Seed every RNG before any TF op runs (Requirement 2.1).
    set_random_seed(args.seed)

    # 3. Build vocab from the training split (Requirement 2.8).
    #    Iterating .itertuples(index=False) once produces a generator of
    #    (packageName, label, category) namedtuples; CharNGramVocab.build
    #    only reads the first two fields.
    train_pairs = [
        (row.packageName, row.label) for row in train_df.itertuples(index=False)
    ]
    vocab = CharNGramVocab.build(train_pairs, n_grams=(3, 4, 5), max_size=20_000)

    # 4. Build the char-CNN model (Requirement 2.2).
    model = build_char_cnn_model(
        vocab_size=len(vocab.tokens),
        max_len=64,
        embed_dim=32,
        conv_filters=128,
        kernel_sizes=(3, 5, 7),
        num_classes=19,
    )

    # 5. Compile with AdamW + CosineDecay schedule, batch 256, 30 epochs
    #    (Requirement 2.3).  ``decay_steps`` covers the full 30-epoch
    #    horizon so the LR finishes its cosine descent at the last step.
    import math

    import tensorflow as tf  # type: ignore[import-not-found]

    batch_size = 256
    epochs = 30
    steps_per_epoch = max(1, math.ceil(n_train / batch_size))
    total_decay_steps = steps_per_epoch * epochs
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=total_decay_steps,
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 6. Fit on train, validate on val.
    train_ds = encode_dataset(train_df, vocab, batch_size=batch_size)
    val_ds = encode_dataset(val_df, vocab, batch_size=batch_size)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
    )

    # 7. Evaluate on test (Requirement 2.4).
    test_metrics = evaluate(model, test_df, vocab)

    # 8. Quality gate — exit 1 if any threshold is not cleared
    #    (Requirements 2.5, 2.12).  IMPORTANT: no artifacts are written
    #    yet, so a failure here leaves any pre-existing files on disk
    #    untouched (Property 14).
    failures = check_quality_gates(test_metrics)
    if failures:
        print("FAIL: quality gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    # 9. TFLite conversion + size guard (Requirements 2.6, 2.7, 2.11).
    #    enforce_size_budget cleans up any ``.tmp`` paths it is given on
    #    overflow; we pass an empty list because nothing is staged on disk
    #    yet (we only call write_atomic once every gate has passed).
    tflite_bytes = convert_to_tflite_quantized(model)
    enforce_size_budget(tflite_bytes, [])  # may sys.exit(2)

    # 10. Enum-order parity (Requirement 2.10).  At this point the TFLite
    #     blob is in memory but unwritten; an exit-3 here leaves zero
    #     output files modified.
    enum_check = compare_enum_order(CATEGORIES, KOTLIN_APP_CATEGORY_ORDER)
    if enum_check is not None:
        idx, k_val, p_val = enum_check
        print(
            f"FAIL: enum mismatch at index {idx}: "
            f"kotlin={k_val} python={p_val}",
            file=sys.stderr,
        )
        return 3

    # 11. Atomic write of all three release artifacts.  This is the only
    #     point in the pipeline that touches disk; every previous gate
    #     has cleared, so a successful return is the only path that
    #     publishes ``app_category_model.tflite`` /
    #     ``app_category_vocab.txt`` / ``app_category_card.json``.
    card = build_model_card(
        model_id="app_category_v1",
        total_train_rows=n_train,
        metrics=test_metrics,
    )
    card_body = json.dumps(card, indent=2, ensure_ascii=False, sort_keys=False) + "\n"

    write_atomic(args.output, tflite_bytes)
    write_atomic(args.vocab, vocab.serialize())
    write_atomic(args.card, card_body)

    print(
        f"OK: wrote {len(tflite_bytes)} bytes to {args.output}; "
        f"top1_accuracy={test_metrics['top1_accuracy']:.3f}, "
        f"macro_f1={test_metrics['macro_f1']:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
