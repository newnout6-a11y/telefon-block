package com.antispam.blocker.domain.scoring

import android.util.Log
import com.antispam.blocker.data.db.entity.BlockedNumber
import com.antispam.blocker.data.db.entity.TrainingData
import com.antispam.blocker.data.prefs.FeedbackLearningStore
import com.antispam.blocker.data.repository.BlockListRepository
import com.antispam.blocker.util.PhoneNormalizer
import kotlinx.coroutines.flow.first

enum class UserAction {
    ANSWER,        // Ответил на WARN → скорее всего не спам
    NOT_SPAM,      // Нажал «Не спам» на WARN/BLOCK
    IS_SCAM,       // Нажал «Мошенник» на WARN
    DISMISS,       // Просто сбросил WARN → нейтрально
    UNBLOCK,       // Разблокировал из BLOCK
    MARK_SPAM      // Отметил как спам из ALLOW
}

class FeedbackHandler(
    private val feedbackStore: FeedbackLearningStore,
    private val blockListRepo: BlockListRepository,
    private val trainingDataDao: com.antispam.blocker.data.db.dao.TrainingDataDao
) {
    suspend fun handleFeedback(
        number: String,
        originalVerdict: String,
        action: UserAction,
        activeFactors: List<RiskFactor>
    ) {
        val alpha = FeedbackLearningStore.ALPHA
        val weights = feedbackStore.getAllWeights()

        // 1. Update weights via EMA for top-3 contributing factors
        val topFactors = activeFactors
            .sortedByDescending { it.points * it.weight }
            .take(3)

        for (factor in topFactors) {
            val currentWeight = weights[factor.id] ?: 1.0f
            val delta = when (action) {
                UserAction.NOT_SPAM, UserAction.UNBLOCK -> -0.1f  // reduce weight
                UserAction.IS_SCAM, UserAction.MARK_SPAM -> 0.1f  // increase weight
                UserAction.ANSWER -> -0.05f
                UserAction.DISMISS -> 0f
            }
            val newWeight = currentWeight + alpha * (currentWeight + delta - currentWeight)
            feedbackStore.setWeight(factor.id, newWeight)
            Log.d("FeedbackHandler", "Updated weight ${factor.id}: $currentWeight → $newWeight")
        }

        // 2. Remember number in appropriate list
        val normalized = PhoneNormalizer.normalize(number)
        if (normalized != null) {
            when (action) {
                UserAction.IS_SCAM, UserAction.MARK_SPAM -> {
                    try { blockListRepo.addToBlockList(normalized, BlockedNumber.Source.FEEDBACK) } catch (_: Exception) {}
                }
                UserAction.NOT_SPAM, UserAction.UNBLOCK -> {
                    try { blockListRepo.addToAllowList(normalized) } catch (_: Exception) {}
                    // Дополнительно копим per-prefix override-сигнал: если по
                    // одному DEF-кодовому префиксу набирается ≥ N «не-спам»
                    // отметок за 30 дней — Home-экран предложит занести
                    // весь префикс в персональный prefix-allowlist одним тапом.
                    feedbackStore.extractPrefixOrNull(normalized)?.let { prefix ->
                        try { feedbackStore.recordPrefixNotSpam(prefix) } catch (_: Exception) {}
                    }
                }
                else -> {}
            }
        }

        // 3. Save training data with user action
        if (normalized != null) {
            val correctedLabel = when (action) {
                UserAction.NOT_SPAM, UserAction.UNBLOCK, UserAction.ANSWER -> "allow"
                UserAction.IS_SCAM, UserAction.MARK_SPAM -> "block"
                UserAction.DISMISS -> originalVerdict
            }
            trainingDataDao.insert(
                TrainingData(
                    normalizedNumber = normalized,
                    featuresJson = "[]",
                    label = correctedLabel,
                    weight = if (action == UserAction.DISMISS) 0.5f else 2.0f,
                    userAction = action.name.lowercase(),
                    timestamp = System.currentTimeMillis()
                )
            )
        }

        // 4. Adapt thresholds after enough feedback
        feedbackStore.incrementFeedbackCount()
        val count = feedbackStore.feedbackCount.first()
        if (count >= FeedbackLearningStore.MIN_FEEDBACK_FOR_THRESHOLD) {
            adaptThresholds(action)
        }
    }

    private suspend fun adaptThresholds(action: UserAction) {
        val currentWarn = feedbackStore.warnThreshold.first()
        val currentBlock = feedbackStore.blockThreshold.first()

        // Shift thresholds based on feedback pattern
        val warnDelta = when (action) {
            UserAction.NOT_SPAM, UserAction.UNBLOCK -> 0.02f   // raise warn threshold → fewer warnings
            UserAction.IS_SCAM, UserAction.MARK_SPAM -> -0.02f // lower warn threshold → more warnings
            else -> 0f
        }
        val blockDelta = when (action) {
            UserAction.UNBLOCK -> 0.03f   // raise block threshold → fewer blocks
            UserAction.MARK_SPAM -> -0.02f // lower block threshold → more blocks
            else -> 0f
        }

        feedbackStore.setWarnThreshold(currentWarn + warnDelta)
        feedbackStore.setBlockThreshold(currentBlock + blockDelta)
    }
}
