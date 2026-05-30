package com.antispam.blocker.domain.scoring

import com.antispam.blocker.domain.detector.Verdict

data class RiskScore(
    val score: Int,
    val level: RiskLevel,
    val verdict: Verdict,
    val reasons: List<String>,
    val confidence: Confidence = Confidence.MEDIUM,
    val source: String = "rule_engine",
    val modelProbabilities: FloatArray = floatArrayOf(0f, 0f, 0f),
    val ruleScore: Int = score,
    val activeFactorIds: List<String> = emptyList(),
    val warnThreshold: Int = 35,
    val blockThreshold: Int = 70,
    val modelInputSize: Int = 0
) {
    enum class Confidence { LOW, MEDIUM, HIGH }

    val allowProb: Float get() = modelProbabilities.getOrElse(0) { 0f }
    val warnProb: Float get() = modelProbabilities.getOrElse(1) { 0f }
    val blockProb: Float get() = modelProbabilities.getOrElse(2) { 0f }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is RiskScore) return false
        return score == other.score &&
            level == other.level &&
            verdict == other.verdict &&
            reasons == other.reasons &&
            confidence == other.confidence &&
            source == other.source &&
            modelProbabilities.contentEquals(other.modelProbabilities) &&
            ruleScore == other.ruleScore &&
            activeFactorIds == other.activeFactorIds &&
            warnThreshold == other.warnThreshold &&
            blockThreshold == other.blockThreshold &&
            modelInputSize == other.modelInputSize
    }

    override fun hashCode(): Int {
        var result = score
        result = 31 * result + level.hashCode()
        result = 31 * result + verdict.hashCode()
        result = 31 * result + reasons.hashCode()
        result = 31 * result + confidence.hashCode()
        result = 31 * result + source.hashCode()
        result = 31 * result + modelProbabilities.contentHashCode()
        result = 31 * result + ruleScore
        result = 31 * result + activeFactorIds.hashCode()
        result = 31 * result + warnThreshold
        result = 31 * result + blockThreshold
        result = 31 * result + modelInputSize
        return result
    }
}
