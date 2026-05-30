package com.antispam.blocker.domain.scoring

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.BufferedReader
import java.io.File
import java.io.FileReader
import java.io.InputStreamReader
import kotlin.math.ln

/**
 * Phase 3: histogram lookup для prefix → 3 числа в 0..1:
 *   - prefixBlockShare — доля BLOCK среди всех номеров с этим префиксом (smoothed)
 *   - prefixWarnShare  — доля WARN  (smoothed)
 *   - prefixSeenLog    — log(1 + seen_count) / seen_log_norm
 *
 * Phase 4B: тот же класс используется для multi-resolution лукапов через
 * фабрики `get(context, assetName)`:
 *   - prefix_histogram.json    (6-char = 4 phone digits, default)
 *   - prefix_histogram_3.json  (5-char = 3 phone digits, def_code only)
 *   - prefix_histogram_7.json  (9-char = 7 phone digits, fine-grained)
 *
 * Дополнительно entry может содержать precomputed `entropy` (Shannon, [0..1])
 * — для prefixEntropy фичи (лейбл-смешанность префикса). Также `seen_count`
 * нормализуется через `sample_size_saturation` (linear, default=30) в фичу
 * prefixSampleSize.
 *
 * Используется как функциональный аналог prefix-embedding (16d) — мобильно
 * через JSON-лукап, без in-graph весов.
 *
 * Если ассета нет — все фичи = 0.
 */
class PrefixHistogramTable private constructor(
    private val prefixes: Map<String, Entry>,
    private val prefixLength: Int,
    private val seenLogNorm: Float,
    private val sampleSizeSaturation: Float,
    private val overallBlockRate: Float,
    private val overallWarnRate: Float,
    val version: String?,
) {
    data class Entry(
        val blockShare: Float,
        val warnShare: Float,
        val seenCount: Int,
        val entropy: Float,
    )

    fun isEmpty(): Boolean = prefixes.isEmpty()

    /** Возвращает (block_share, warn_share, seen_log) — все в 0..1. */
    fun lookup(normalized: String?): Triple<Float, Float, Float> {
        if (normalized == null || normalized.isEmpty() || prefixes.isEmpty()) {
            return Triple(0f, 0f, 0f)
        }
        val entry = entryOrFallback(normalized)
        if (entry != null) {
            val seenLog = (ln(1.0 + entry.seenCount) / seenLogNorm.toDouble())
                .toFloat().coerceIn(0f, 1f)
            return Triple(entry.blockShare, entry.warnShare, seenLog)
        }
        return Triple(overallBlockRate, overallWarnRate, 0f)
    }

    /** Phase 4B: только block_share (для prefixBlockShare3 / prefixBlockShare7). */
    fun blockShareFor(normalized: String?): Float {
        if (normalized == null || normalized.isEmpty() || prefixes.isEmpty()) return 0f
        val key = normalized.take(prefixLength)
        return prefixes[key]?.blockShare ?: 0f
    }

    /** Phase 4B: precomputed Shannon-энтропия лейблов на префиксе (0..1). */
    fun entropyFor(normalized: String?): Float {
        if (normalized == null || normalized.isEmpty() || prefixes.isEmpty()) return 0f
        val entry = entryOrFallback(normalized) ?: return 0f
        return entry.entropy
    }

    /** Phase 4B: linear-scaled confidence по seen_count, насыщается на N образцах. */
    fun sampleSizeFor(normalized: String?): Float {
        if (normalized == null || normalized.isEmpty() || prefixes.isEmpty()) return 0f
        val entry = entryOrFallback(normalized) ?: return 0f
        val saturation = if (sampleSizeSaturation > 0f) sampleSizeSaturation else 30f
        return (entry.seenCount.toFloat() / saturation).coerceIn(0f, 1f)
    }

    private fun entryOrFallback(normalized: String): Entry? {
        val key = normalized.take(prefixLength)
        prefixes[key]?.let { return it }
        // shorter-prefix fallback (5, 4) — даёт хоть какой-то сигнал.
        for (k in intArrayOf(5, 4)) {
            if (k >= prefixLength) continue
            prefixes[normalized.take(k)]?.let { return it }
        }
        return null
    }

    companion object {
        private const val TAG = "PrefixHistogramTable"
        const val ASSET_PATH_DEFAULT = "prefix_histogram.json"
        const val ASSET_PATH_3 = "prefix_histogram_3.json"
        const val ASSET_PATH_7 = "prefix_histogram_7.json"

        @Volatile private var INSTANCES: MutableMap<String, PrefixHistogramTable> = HashMap()

        /** Default (6-char / 4-phone-digit) prefix histogram. Back-compat with Phase 3. */
        fun get(context: Context): PrefixHistogramTable = get(context, ASSET_PATH_DEFAULT)

        /** Phase 4B: загрузка по имени ассета (one instance per asset). */
        fun get(context: Context, assetName: String): PrefixHistogramTable {
            INSTANCES[assetName]?.let { return it }
            synchronized(this) {
                INSTANCES[assetName]?.let { return it }
                val loaded = load(context.applicationContext, assetName)
                INSTANCES[assetName] = loaded
                return loaded
            }
        }

        fun invalidate() {
            synchronized(this) { INSTANCES.clear() }
        }

        private fun load(context: Context, assetName: String): PrefixHistogramTable {
            val fromFile = runCatching {
                val file = File(context.filesDir, assetName)
                if (file.exists() && file.length() > 0) {
                    FileReader(file).use { it.readText() }
                } else null
            }.getOrNull()

            val json = fromFile ?: runCatching {
                context.assets.open(assetName).use { stream ->
                    BufferedReader(InputStreamReader(stream)).readText()
                }
            }.getOrNull()

            if (json.isNullOrBlank()) {
                Log.w(TAG, "asset $assetName unavailable")
                return PrefixHistogramTable(emptyMap(), 6, 1f, 30f, 0f, 0f, null)
            }

            return try {
                val obj = JSONObject(json)
                val version = obj.optString("version", null)
                val prefixLength = obj.optInt("prefix_length", 6)
                val seenLogNorm = obj.optDouble("seen_log_norm", 1.0).toFloat().coerceAtLeast(1e-6f)
                val sampleSizeSaturation = obj.optDouble("sample_size_saturation", 30.0).toFloat()
                val overallBlockRate = obj.optDouble("overall_block_rate", 0.0).toFloat()
                val overallWarnRate = obj.optDouble("overall_warn_rate", 0.0).toFloat()
                val prefObj = obj.optJSONObject("prefixes")
                val map = HashMap<String, Entry>(prefObj?.length() ?: 0)
                if (prefObj != null) {
                    val it = prefObj.keys()
                    while (it.hasNext()) {
                        val k = it.next()
                        val entry = prefObj.optJSONObject(k) ?: continue
                        val b = entry.optDouble("block_share", 0.0).toFloat().coerceIn(0f, 1f)
                        val w = entry.optDouble("warn_share", 0.0).toFloat().coerceIn(0f, 1f)
                        val n = entry.optInt("seen_count", 0)
                        val e = entry.optDouble("entropy", 0.0).toFloat().coerceIn(0f, 1f)
                        map[k] = Entry(b, w, n, e)
                    }
                }
                val sourceTag = if (fromFile != null) "filesDir" else "assets"
                Log.i(TAG, "loaded ${map.size} prefixes from $assetName ($sourceTag, v=$version, len=$prefixLength)")
                PrefixHistogramTable(
                    map, prefixLength, seenLogNorm, sampleSizeSaturation,
                    overallBlockRate, overallWarnRate, version,
                )
            } catch (t: Throwable) {
                Log.w(TAG, "$assetName parse failed", t)
                PrefixHistogramTable(emptyMap(), 6, 1f, 30f, 0f, 0f, null)
            }
        }
    }
}
