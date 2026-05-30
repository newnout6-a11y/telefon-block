package com.antispam.blocker.domain.model

import android.content.Context
import org.json.JSONObject

/**
 * Структурированная карточка модели: версия, дата, метрики, размер вектора.
 *
 * Загружается из `assets/model_card.json` или из `filesDir/model_card.json`,
 * если worker дообучения положит туда новую версию.
 */
data class ModelCard(
    val version: String,
    val createdAt: String,
    val featureCount: Int,
    val rows: Int,
    val classCounts: Map<String, Int>,
    val blockPrecision: Float,
    val blockRecall: Float,
    val rocAuc: Float?,
    val datasetHash: String?,
    val thresholds: Thresholds? = null,
    val coldThresholds: Thresholds? = null,
    val outputFormat: OutputFormat = OutputFormat.THREE_CLASS_SOFTMAX,
    val notes: String? = null
) {
    /**
     * Формат выхода TFLite-модели.
     *
     * Эволюция:
     *   * THREE_CLASS_SOFTMAX (legacy, default) — выход [1, 3] = [allow, warn, block],
     *     softmax-распределение. Тёплые/холодные пороги применяются над `block` и `warn`
     *     индивидуально. WARN-класс при таком формате на практике не работает
     *     (см. model_card cold_thresholded ALLOW precision = 0.42), потому что
     *     эвристика label'ов делает WARN неотличимым от BLOCK по фичам.
     *
     *   * BINARY_SIGMOID (PR-2) — выход [1, 1] = p_spam ∈ [0, 1], уже откалибровано
     *     внутри графа через Platt scaling: p_spam = sigmoid(a * logit + b),
     *     где a/b замораживаются как константы в TFLite. WARN получаем как
     *     зону неопределённости [warn_threshold, block_threshold) над p_spam.
     */
    enum class OutputFormat(val cardValue: String, val tensorSize: Int) {
        THREE_CLASS_SOFTMAX("3class_softmax", 3),
        BINARY_SIGMOID("binary_sigmoid", 1),
    }
    /**
     * Per-class probability thresholds tuned on validation set.
     *
     * Inference logic in SpamModel.kt:
     *   if block >= blockThreshold -> BLOCK
     *   else if warn >= warnThreshold -> WARN
     *   else -> ALLOW
     *
     * Phase 4A: there are now two threshold sets in model_card.json:
     *   - `thresholds`: warm path — tuned on full val with all metadata available.
     *   - `cold_thresholds`: cold path — tuned on val with all online metadata
     *      (lists, reputation, reviews, categories) zeroed and noMetadata=1.
     *
     * SpamModel picks `coldThresholds` at inference when the runtime feature
     * vector has `noMetadata=true` AND no list-based hint (inAllowlist=0,
     * inBlacklist=0). Otherwise it uses `thresholds`. If `coldThresholds` is
     * absent (legacy card), the warm thresholds are used in both modes —
     * matches pre-Phase-4A behavior.
     *
     * If thresholds is null or any field is NaN, inference falls back to argmax.
     */
    data class Thresholds(
        val blockThreshold: Float,
        val warnThreshold: Float
    )

    companion object {
        private const val ASSET_NAME = "model_card.json"

        fun load(context: Context): ModelCard? {
            return loadFromFiles(context) ?: loadFromAssets(context)
        }

        private fun loadFromFiles(context: Context): ModelCard? {
            val file = java.io.File(context.filesDir, ASSET_NAME)
            if (!file.exists()) return null
            return runCatching { parse(file.readText(Charsets.UTF_8)) }.getOrNull()
        }

        private fun loadFromAssets(context: Context): ModelCard? {
            return runCatching {
                context.assets.open(ASSET_NAME).bufferedReader(Charsets.UTF_8).use { it.readText() }
            }.getOrNull()?.let { runCatching { parse(it) }.getOrNull() }
        }

        private fun parseThresholds(thresholdsObj: JSONObject?): Thresholds? {
            if (thresholdsObj == null) return null
            val bt = thresholdsObj.optDouble("block_threshold", Double.NaN)
            val wt = thresholdsObj.optDouble("warn_threshold", Double.NaN)
            if (bt.isNaN() || wt.isNaN()) return null
            // Sanity: block_threshold MUST be ≥ warn_threshold. На текущем
            // обученном `model_card.json` (`scripts/train_kd_distillation.py`)
            // эти ключи местами перепутаны (block=0.24, warn=0.58). Чтобы
            // вердикт BLOCK не выдавался при p_spam < warn — меняем местами
            // и логируем. Корректировка нужна только для битых карточек;
            // на правильно сгенерированных это no-op.
            val (blockFinal, warnFinal) = if (bt < wt) {
                android.util.Log.w(
                    "ModelCard",
                    "Inverted thresholds detected (block=$bt < warn=$wt); swapping at parse-time"
                )
                wt.toFloat() to bt.toFloat()
            } else {
                bt.toFloat() to wt.toFloat()
            }
            return Thresholds(blockThreshold = blockFinal, warnThreshold = warnFinal)
        }

        fun parse(json: String): ModelCard? {
            return runCatching {
                val obj = JSONObject(json)
                val classObj = obj.optJSONObject("class_counts")
                val classCounts = mutableMapOf<String, Int>()
                if (classObj != null) {
                    val keys = classObj.keys()
                    while (keys.hasNext()) {
                        val k = keys.next()
                        classCounts[k] = classObj.optInt(k, 0)
                    }
                }
                val thresholds = parseThresholds(obj.optJSONObject("thresholds"))
                // Phase 4A: cold thresholds (выбираются на устройстве по noMetadata).
                // Если блок `cold_thresholds` отсутствует в карточке (старая версия) —
                // парсинг вернёт null, SpamModel будет применять warm thresholds везде.
                val coldThresholds = parseThresholds(obj.optJSONObject("cold_thresholds"))
                // PR-2: формат выхода. Legacy = 3class_softmax (выход [1,3]),
                // новый = binary_sigmoid (выход [1,1] с уже встроенной Platt-калибровкой).
                val outputFormatRaw = obj.optString("output_format", "").lowercase()
                val outputFormat = when (outputFormatRaw) {
                    "binary_sigmoid", "binary" -> OutputFormat.BINARY_SIGMOID
                    "3class_softmax", "softmax", "" -> OutputFormat.THREE_CLASS_SOFTMAX
                    else -> {
                        android.util.Log.w(
                            "ModelCard",
                            "Unknown output_format='$outputFormatRaw' in model_card.json; defaulting to 3class_softmax"
                        )
                        OutputFormat.THREE_CLASS_SOFTMAX
                    }
                }
                ModelCard(
                    version = obj.optString("version", "unknown"),
                    createdAt = obj.optString("created_at", ""),
                    featureCount = obj.optInt("feature_count", 0),
                    rows = obj.optInt("rows", 0),
                    classCounts = classCounts,
                    blockPrecision = obj.optDouble("block_precision", 0.0).toFloat(),
                    blockRecall = obj.optDouble("block_recall", 0.0).toFloat(),
                    rocAuc = obj.optDouble("roc_auc_ovr", Double.NaN).takeUnless { it.isNaN() }?.toFloat(),
                    datasetHash = obj.optString("dataset_hash", "").ifBlank { null },
                    thresholds = thresholds,
                    coldThresholds = coldThresholds,
                    outputFormat = outputFormat,
                    notes = obj.optString("notes", "").ifBlank { null }
                )
            }.getOrNull()
        }
    }
}
