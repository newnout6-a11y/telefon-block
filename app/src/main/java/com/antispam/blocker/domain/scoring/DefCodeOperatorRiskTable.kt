package com.antispam.blocker.domain.scoring

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.BufferedReader
import java.io.File
import java.io.FileReader
import java.io.InputStreamReader

/**
 * Phase 4B: cross-feature P(BLOCK | operator_bucket × def_code).
 *
 * Источник: `scripts/build_assets_from_dataset.py` пишет
 * assets/def_code_operator_risk.json со структурой:
 *   {
 *     "fallback_risk": 0.57,
 *     "buckets": {
 *       "mts":     { "495": 0.31, "812": 0.05, ... },
 *       "megafon": { ... },
 *       "beeline": { ... },
 *       "tele2":   { ... },
 *       "mvno":    { ... },
 *       "other":   { ... }
 *     }
 *   }
 *
 * Сглажено с prior=overall_block_rate (α=20). При оффлайн-инференсе позволяет
 * различить риск номера в зависимости от того, какой именно оператор
 * обслуживает диапазон с этим DEF-кодом — например, MVNO в коде «495» обычно
 * чаще даёт спам, чем основной MNO.
 *
 * Если ассета нет или комбинации (bucket, def_code) — возвращает fallback_risk.
 */
class DefCodeOperatorRiskTable private constructor(
    private val byBucket: Map<String, Map<String, Float>>,
    val fallbackRisk: Float,
    val version: String?,
) {

    /** Риск 0..1 по нормализованному номеру (+7XXX…) и operator_bucket. */
    fun riskFor(normalized: String?, bucket: String): Float {
        if (normalized == null) return fallbackRisk
        if (!normalized.startsWith("+7")) return fallbackRisk
        val digits = normalized.filter { it.isDigit() }
        if (digits.length < 4) return fallbackRisk
        val def = digits.substring(1, 4)
        val bucketMap = byBucket[bucket] ?: return fallbackRisk
        return bucketMap[def] ?: fallbackRisk
    }

    companion object {
        private const val TAG = "DefCodeOperatorRiskTable"
        private const val ASSET_PATH = "def_code_operator_risk.json"
        private const val DEFAULT_FALLBACK = 0.5f

        @Volatile private var INSTANCE: DefCodeOperatorRiskTable? = null

        fun get(context: Context): DefCodeOperatorRiskTable {
            INSTANCE?.let { return it }
            synchronized(this) {
                INSTANCE?.let { return it }
                val loaded = load(context.applicationContext)
                INSTANCE = loaded
                return loaded
            }
        }

        fun invalidate() {
            synchronized(this) { INSTANCE = null }
        }

        private fun load(context: Context): DefCodeOperatorRiskTable {
            val fromFile = runCatching {
                val file = File(context.filesDir, ASSET_PATH)
                if (file.exists() && file.length() > 0) {
                    FileReader(file).use { it.readText() }
                } else null
            }.getOrNull()

            val json = fromFile ?: runCatching {
                context.assets.open(ASSET_PATH).use { stream ->
                    BufferedReader(InputStreamReader(stream)).readText()
                }
            }.getOrNull()

            if (json.isNullOrBlank()) {
                Log.w(TAG, "asset $ASSET_PATH unavailable; using fallback=$DEFAULT_FALLBACK")
                return DefCodeOperatorRiskTable(emptyMap(), DEFAULT_FALLBACK, null)
            }

            return try {
                val obj = JSONObject(json)
                val version = obj.optString("version", null)
                val fallback = obj.optDouble("fallback_risk", DEFAULT_FALLBACK.toDouble()).toFloat()
                val bucketsObj = obj.optJSONObject("buckets")
                val byBucket = HashMap<String, Map<String, Float>>(bucketsObj?.length() ?: 0)
                if (bucketsObj != null) {
                    val bucketKeys = bucketsObj.keys()
                    while (bucketKeys.hasNext()) {
                        val bucketName = bucketKeys.next()
                        val bucketObj = bucketsObj.optJSONObject(bucketName) ?: continue
                        val codeMap = HashMap<String, Float>(bucketObj.length())
                        val codeKeys = bucketObj.keys()
                        while (codeKeys.hasNext()) {
                            val code = codeKeys.next()
                            codeMap[code] = bucketObj.getDouble(code).toFloat()
                        }
                        byBucket[bucketName] = codeMap
                    }
                }
                val sourceTag = if (fromFile != null) "filesDir" else "assets"
                val totalPairs = byBucket.values.sumOf { it.size }
                Log.i(TAG, "loaded $totalPairs (bucket, def_code) pairs from $ASSET_PATH " +
                    "($sourceTag, version=$version, buckets=${byBucket.keys}, fb=$fallback)")
                DefCodeOperatorRiskTable(byBucket, fallback, version)
            } catch (t: Throwable) {
                Log.w(TAG, "$ASSET_PATH parse failed", t)
                DefCodeOperatorRiskTable(emptyMap(), DEFAULT_FALLBACK, null)
            }
        }
    }
}
