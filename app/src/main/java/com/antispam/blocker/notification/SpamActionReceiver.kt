package com.antispam.blocker.notification

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import com.antispam.blocker.SpamBlockerApp
import com.antispam.blocker.data.prefs.FeedbackLearningStore
import com.antispam.blocker.data.repository.BlockListRepository
import com.antispam.blocker.domain.personal.ExplicitLabel
import com.antispam.blocker.domain.scoring.FeedbackHandler
import com.antispam.blocker.domain.scoring.RiskFactor
import com.antispam.blocker.domain.scoring.UserAction
import com.antispam.blocker.util.PhoneNormalizer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/**
 * Принимает тапы по action-кнопкам уведомлений «Подозрение на спам» /
 * «Звонок заблокирован». Кроме обновления списков (BlockedNumber/AllowedNumber)
 * вызывает [FeedbackHandler] — это запускает EMA-обновление весов сработавших
 * факторов и адаптацию порогов в [FeedbackLearningStore], а также пишет
 * человеко-проверенную строку в `training_data` (с `userAction != null`).
 *
 * Маппинг кнопка × вердикт → [UserAction]:
 *   - WARN-нотификация:
 *       «В чёрный список» → IS_SCAM
 *       «Это не спам»     → NOT_SPAM
 *   - BLOCK-нотификация (после уже отклонённого звонка):
 *       «Это не спам»     → UNBLOCK
 *
 * Дополнительно (Req 4.5–4.8) принимает «Был ли это спам? Да / Нет» —
 * [SpamWarningNotifier.ACTION_SPAM_YES] / [SpamWarningNotifier.ACTION_SPAM_NO].
 * Эти действия идут **отдельным путём**: они НЕ дёргают [FeedbackHandler] и
 * cloud-side EMA, а кормят исключительно [com.antispam.blocker.domain.personal.OnlineTrainer.applyExplicitLabel]
 * с `label = ExplicitLabel.BLOCK / ALLOW`. SGD-шаг по сохранённому
 * `FeatureSnapshot` идёт с весом `EXPLICIT_WEIGHT` (1.5f), строго больше
 * IMPLICIT_WEIGHT (Req 4.8). Перепутать пути нельзя — `callEventId` берётся
 * из `EXTRA_CALL_EVENT_ID`, и оба канала намеренно изолированы (см. KDoc в
 * `SpamWarningNotifier`).
 */
class SpamActionReceiver : BroadcastReceiver() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onReceive(context: Context, intent: Intent) {
        // Personal-classifier feedback (Req 4.5–4.8): отдельный путь, который
        // НЕ дёргает FeedbackHandler / cloud-side EMA. Он кормит исключительно
        // OnlineTrainer.applyExplicitLabel(callEventId, label) — SGD-шаг по
        // сохранённому FeatureSnapshot. См. KDoc в SpamWarningNotifier.
        when (intent.action) {
            SpamWarningNotifier.ACTION_SPAM_YES -> {
                handlePersonalFeedback(context, intent, ExplicitLabel.BLOCK)
                return
            }
            SpamWarningNotifier.ACTION_SPAM_NO -> {
                handlePersonalFeedback(context, intent, ExplicitLabel.ALLOW)
                return
            }
        }

        val number = intent.getStringExtra(SpamWarningNotifier.EXTRA_NUMBER) ?: return
        val notifId = intent.getIntExtra(SpamWarningNotifier.EXTRA_NOTIF_ID, -1)
        val originalVerdict = intent.getStringExtra(SpamWarningNotifier.EXTRA_ORIG_VERDICT)
            ?: "warn"
        val activeFactorIds = intent.getStringExtra(SpamWarningNotifier.EXTRA_ACTIVE_FACTORS)
            ?.split(',')
            ?.filter { it.isNotBlank() }
            ?: emptyList()

        val app = SpamBlockerApp.instance
        val db = app.database
        val blockListRepo = BlockListRepository(
            db.blockedNumberDao(),
            db.allowedNumberDao(),
            PhoneNormalizer
        )
        val feedbackStore = FeedbackLearningStore(context.applicationContext)
        val feedbackHandler = FeedbackHandler(
            feedbackStore = feedbackStore,
            blockListRepo = blockListRepo,
            trainingDataDao = db.trainingDataDao()
        )

        val userAction = when (intent.action) {
            SpamWarningNotifier.ACTION_BLOCK -> when (originalVerdict) {
                "block" -> UserAction.MARK_SPAM // wasn't really visible but keep semantics
                else -> UserAction.IS_SCAM       // WARN→Block: пометить как мошенника
            }
            SpamWarningNotifier.ACTION_ALLOW -> when (originalVerdict) {
                "block" -> UserAction.UNBLOCK    // BLOCK→Allow: разблокировать
                "allow" -> UserAction.MARK_SPAM  // редкий путь (ALLOW→spam) — нет UI пока
                else -> UserAction.NOT_SPAM      // WARN→Allow: «не спам»
            }
            else -> {
                Log.w(TAG, "unknown action ${intent.action}")
                return
            }
        }

        scope.launch {
            // Восстанавливаем порядок факторов: order сохранён в EXTRA_ACTIVE_FACTORS
            // (отсортированы по points*weight ещё в SmartSpamDetector). Stub points
            // декремент, чтобы сохранить топ-N после .sortedByDescending в FeedbackHandler.
            val factorStubs: List<RiskFactor> = activeFactorIds.mapIndexed { idx, id ->
                RiskFactor(
                    id = id,
                    displayName = id,
                    points = activeFactorIds.size - idx,
                    reason = "",
                    weight = 1.0f,
                    isActive = true
                )
            }

            try {
                feedbackHandler.handleFeedback(
                    number = number,
                    originalVerdict = originalVerdict,
                    action = userAction,
                    activeFactors = factorStubs
                )
                Log.i(TAG, "feedback applied: number=$number action=$userAction orig=$originalVerdict factors=${activeFactorIds.size}")
            } catch (t: Throwable) {
                Log.w(TAG, "feedback failed", t)
            }

            // Дыра #2.b: тапы по cloud-side кнопкам («В чёрный список» / «Это не спам» /
            // «Разблокировать») теперь дополнительно дают SGD-step Device_Model.
            // До этого Device_Model училась только на отдельных кнопках «Был ли это
            // спам? Да/Нет», которые юзеры в реальности тапают редко. SGD идёт с
            // EXPLICIT_WEIGHT (1.5f) — строго больше, чем неявный label из
            // SpamCallScreeningService (0.5f), так что осознанное действие юзера
            // всегда перевешивает наш собственный authoritative-сигнал.
            val callEventId = intent.getLongExtra(SpamWarningNotifier.EXTRA_CALL_EVENT_ID, -1L)
            if (callEventId != -1L) {
                val explicitLabel = when (userAction) {
                    UserAction.IS_SCAM, UserAction.MARK_SPAM -> ExplicitLabel.BLOCK
                    UserAction.UNBLOCK, UserAction.NOT_SPAM, UserAction.ANSWER -> ExplicitLabel.ALLOW
                    UserAction.DISMISS -> null  // нейтральное действие, SGD не делаем
                }
                if (explicitLabel != null) {
                    try {
                        SpamBlockerApp.instance.onlineTrainer
                            .applyExplicitLabel(callEventId, explicitLabel)
                        Log.i(TAG, "explicit label applied via cloud-side action: id=$callEventId label=$explicitLabel")
                    } catch (t: Throwable) {
                        Log.w(TAG, "applyExplicitLabel from cloud-side action failed", t)
                    }
                }
            }
        }

        if (notifId > 0) {
            val nm = context.getSystemService(Context.NOTIFICATION_SERVICE) as android.app.NotificationManager
            nm.cancel(notifId)
        }
    }

    /**
     * Per-call personal-classifier feedback (Req 4.5–4.8).
     *
     * Кормит исключительно `OnlineTrainer.applyExplicitLabel(callEventId, label)`
     * — это делает SGD-шаг по сохранённому `FeatureSnapshot` с весом
     * `DeviceModel.EXPLICIT_WEIGHT` (1.5f, строго больше IMPLICIT_WEIGHT,
     * Req 4.8). Сознательно **не** дёргает [FeedbackHandler.handleFeedback],
     * чтобы не путать cloud-side EMA весов factor-id'ов с on-device SGD по
     * нормализованному фичевому вектору (см. KDoc в [SpamWarningNotifier]).
     *
     * Если `callEventId` отсутствует или равен -1 — это означает, что
     * уведомление было выпущено старой версией кода или поломанным intent'ом;
     * молча выходим без побочных эффектов вместо того чтобы искажать веса.
     */
    private fun handlePersonalFeedback(
        context: Context,
        intent: Intent,
        label: ExplicitLabel
    ) {
        val callEventId = intent.getLongExtra(SpamWarningNotifier.EXTRA_CALL_EVENT_ID, -1L)
        if (callEventId == -1L) {
            Log.w(TAG, "personal feedback ${intent.action}: missing EXTRA_CALL_EVENT_ID")
            return
        }
        val notifId = intent.getIntExtra(SpamWarningNotifier.EXTRA_NOTIF_ID, -1)

        scope.launch {
            try {
                SpamBlockerApp.instance.onlineTrainer
                    .applyExplicitLabel(callEventId, label)
                Log.i(TAG, "explicit label applied: callEventId=$callEventId label=$label")
            } catch (t: Throwable) {
                Log.w(TAG, "applyExplicitLabel failed for callEventId=$callEventId", t)
            }
        }

        if (notifId > 0) {
            val nm = context.getSystemService(Context.NOTIFICATION_SERVICE)
                as android.app.NotificationManager
            nm.cancel(notifId)
        }
    }

    private companion object {
        const val TAG = "SpamActionReceiver"
    }
}
