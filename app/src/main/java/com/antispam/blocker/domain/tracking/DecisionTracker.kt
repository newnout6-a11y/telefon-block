package com.antispam.blocker.domain.tracking

import com.antispam.blocker.data.db.dao.DecisionRecordDao
import com.antispam.blocker.data.db.entity.DecisionRecord
import com.antispam.blocker.domain.scoring.CallFeatures
import com.antispam.blocker.domain.scoring.RiskScore
import kotlinx.coroutines.flow.Flow
import org.json.JSONArray
import org.json.JSONObject

/**
 * Полнофункциональный аудит решений ИИ.
 *
 * Записывает каждое решение детектора со снимком 32 фич, raw model probabilities,
 * вердиктом и активными факторами. Используется DebugScreen-ом и feedback metric.
 */
class DecisionTracker(
    private val dao: DecisionRecordDao,
    private val modelVersionProvider: () -> String? = { null }
) {

    suspend fun record(
        rawNumber: String?,
        normalizedNumber: String?,
        features: CallFeatures,
        risk: RiskScore,
        /**
         * Финальный вердикт после `FusionDecider.decide` — ОН и должен
         * показываться как «вердикт» в журнале и в карточке вкладки «ИИ»,
         * чтобы UI не показывал «BLOCK 82%», а звонок при этом был
         * пропущен. Если `null`, fusion не применялся (старые вызовы /
         * fast-path) — пишем `risk.verdict` без коррекции.
         */
        finalVerdict: com.antispam.blocker.domain.detector.Verdict? = null,
        /**
         * Тег ветки `FusionDecider` (`device_veto_allow`,
         * `single_block_downgrade_warn`, …). Когда [finalVerdict] не равен
         * `risk.verdict`, тег добавляется в reasons как «Решение
         * скорректировано: <тег>», чтобы пользователь видел, ПОЧЕМУ карточка
         * показывает не то, что он ожидал.
         */
        fusionRationaleTag: String? = null,
    ): Long {
        val effectiveVerdict = finalVerdict ?: risk.verdict
        val correctionApplied = finalVerdict != null && finalVerdict != risk.verdict
        val effectiveReasons = if (correctionApplied) {
            buildList {
                add("Финальный вердикт: ${effectiveVerdict.name} (модель: ${risk.verdict.name})")
                if (!fusionRationaleTag.isNullOrBlank()) {
                    add("Корректировка: $fusionRationaleTag")
                }
                addAll(risk.reasons)
            }
        } else {
            risk.reasons
        }

        val featuresJson = featuresToJson(features).toString()
        val reasonsJson = JSONArray(effectiveReasons).toString()
        val factorsJson = JSONArray(risk.activeFactorIds).toString()

        val record = DecisionRecord(
            timestamp = System.currentTimeMillis(),
            rawNumber = rawNumber,
            normalizedNumber = normalizedNumber,
            verdict = effectiveVerdict.name,
            score = risk.score,
            source = risk.source,
            confidence = risk.confidence.name,
            modelAllowProb = risk.allowProb,
            modelWarnProb = risk.warnProb,
            modelBlockProb = risk.blockProb,
            modelInputSize = risk.modelInputSize,
            featuresJson = featuresJson,
            reasonsJson = reasonsJson,
            activeFactorsJson = factorsJson,
            ruleScore = risk.ruleScore,
            warnThreshold = risk.warnThreshold,
            blockThreshold = risk.blockThreshold,
            modelVersion = modelVersionProvider()
        )
        return dao.insert(record)
    }

    fun observeRecent(limit: Int = 100): Flow<List<DecisionRecord>> = dao.observeRecent(limit)

    suspend fun getRecent(limit: Int = 100): List<DecisionRecord> = dao.getRecent(limit)

    suspend fun stats(): TrackingStats {
        val total = dao.count()
        val withFeedback = dao.countWithFeedback()
        val agreeing = dao.countAgreeingFeedback()
        return TrackingStats(
            total = total,
            blockCount = dao.countByVerdict("BLOCK"),
            warnCount = dao.countByVerdict("WARN"),
            allowCount = dao.countByVerdict("ALLOW"),
            // Backwards-compat: исторически Server_Model (TFLite) писалась
            // как `tflite_model`, потом как `cloud_model` — оба значения
            // считаем за «модельные» решения, чтобы старые decision_records
            // правильно учитывались в счётчике.
            modelDecisions = dao.countBySource("server_model") +
                dao.countBySource("cloud_model") +
                dao.countBySource("tflite_model"),
            ruleDecisions = dao.countBySource("rule_engine"),
            feedbackCount = withFeedback,
            agreementRate = if (withFeedback > 0) agreeing.toFloat() / withFeedback else null
        )
    }

    suspend fun setUserAction(id: Long, action: String) {
        dao.setUserAction(id, action, System.currentTimeMillis())
    }

    suspend fun pruneOlderThan(cutoff: Long): Int = dao.deleteOlderThan(cutoff)

    suspend fun clear() = dao.clear()

    private fun featuresToJson(features: CallFeatures): JSONObject {
        val arr = features.toFloatArray()
        val obj = JSONObject()
        FEATURE_NAMES.forEachIndexed { idx, name ->
            obj.put(name, arr.getOrElse(idx) { 0f }.toDouble())
        }
        return obj
    }

    companion object {
        /**
         * Имена признаков должны быть синхронизированы с
         * `scripts/ru_metadata_features.py:COMPACT_FEATURES`.
         */
        val FEATURE_NAMES: List<String> = listOf(
            "isContact",
            "isRussianNumber",
            "isForeignNumber",
            "isShortCode",
            "isStandardLen",
            "isTollFree8800",
            "isGeographical",
            "isMobileRu",
            "isValidRuRange",
            "spoofingPrefixFlag",
            "digitEntropy",
            "repeatDigitRatio",
            "maxSameDigitRun",
            "beautifulNumberFlag",
            "prefixRisk",
            "callFrequency",
            "isNightTime",
            "recentBankApp",
            "recentGovApp",
            "recentMarketplaceApp",
            "recentMessengerApp",
            "previouslyRejected",
            "inBlacklist",
            "inAllowlist",
            "hiddenNumber",
            "callerVerifyFailed",
            "userVulnerability",
            "userBusinessActivity",
            "contactsAvailable",
            "usageAccessAvailable",
            "reputationScore",
            "sourceConfidence",
            // Phase 3 v3 +15 features:
            "operatorMts",
            "operatorMegafon",
            "operatorBeeline",
            "operatorTele2",
            "operatorMvno",
            "defCodeRisk",
            "prefixBlockShare",
            "prefixWarnShare",
            "prefixSeenLog",
            "reviewsLog",
            "negativeRatio",
            "searchVolumeLog",
            "hasFraudCategory",
            "hasTelemarketingCategory",
            "noMetadata",
            // Phase 4B v4 +5 cold-survivable features:
            "prefixBlockShare3",
            "prefixBlockShare7",
            "prefixEntropy",
            "defCodeOperatorRisk",
            "prefixSampleSize"
        )
    }
}

data class TrackingStats(
    val total: Int,
    val blockCount: Int,
    val warnCount: Int,
    val allowCount: Int,
    val modelDecisions: Int,
    val ruleDecisions: Int,
    val feedbackCount: Int,
    val agreementRate: Float?
)
