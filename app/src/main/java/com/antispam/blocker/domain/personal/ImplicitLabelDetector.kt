package com.antispam.blocker.domain.personal

import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.workDataOf
import com.antispam.blocker.data.db.entity.CallEvent
import java.util.concurrent.TimeUnit

/**
 * Implicit on-device label produced by [ImplicitLabelDetector].
 *
 * The numeric `y` is consumed by [OnlineTrainer]'s SGD step:
 * `BLOCK = 1f`, `ALLOW = 0f` (positive class is BLOCK).
 *
 * Mirrors the design enum (see design.md "Online_Trainer" section); declared
 * here because the detector is the upstream producer of these labels and
 * because [OnlineTrainer] (task 10.1) is downstream of this module.
 */
enum class ImplicitLabel(val y: Float) {
    ALLOW(0f),
    BLOCK(1f),
}

/**
 * Output of [ImplicitLabelDetector.detect].
 *
 * @property label the implicit label to apply to the call (`ALLOW` or `BLOCK`).
 * @property isDeferred when `true`, the caller must NOT apply the label
 *   immediately. Instead it should schedule a re-check after 24 hours via
 *   [MissedNoCallbackRecheckWorker]; the worker then decides whether the
 *   label is still valid (no callback to that number occurred in the window)
 *   and applies it through `OnlineTrainer.applyImplicitLabel`. Currently only
 *   the `INCOMING + MISSED` rule (Req 4.3) is deferred — every other rule
 *   that fires emits a synchronous label.
 */
data class ImplicitLabelResult(
    val label: ImplicitLabel,
    val isDeferred: Boolean = false,
)

/**
 * Pure, side-effect-free implementation of the implicit-label decision table
 * from design §"ImplicitLabelDetector" / Property 10.
 *
 * Maps each `(direction, state, durationMs)` triplet (plus per-number history)
 * to an [ImplicitLabelResult] or `null` when no rule fires:
 *
 * | Condition                                                                                                                                          | Result                  |
 * |----------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
 * | `INCOMING ∧ ANSWERED ∧ durationMs ≥ 15s`                                                                                                            | `ALLOW` synchronously   |
 * | `INCOMING ∧ (REJECTED ∨ (ANSWERED ∧ durationMs < 15s))`                                                                                             | `BLOCK` synchronously   |
 * | `INCOMING ∧ MISSED`                                                                                                                                  | `BLOCK` deferred 24h    |
 * | `OUTGOING ∧ ∃ h ∈ history : h.normalizedNumber = event.normalizedNumber ∧ h.direction = INCOMING ∧ h.state ∈ {MISSED, REJECTED}`                   | `ALLOW` synchronously   |
 * | otherwise                                                                                                                                            | `null`                  |
 *
 * The MISSED rule emits a *deferred* candidate: per Req 4.3 the actual BLOCK
 * label only applies if no outgoing call to the same number occurs within the
 * 24-hour callback window. That second-stage check lives in
 * [MissedNoCallbackRecheckWorker]; this detector merely flags the candidate
 * via [ImplicitLabelResult.isDeferred] so the orchestrator can schedule the
 * re-check via [enqueueDeferredMissedRecheck].
 *
 * No I/O, no DAO calls, no clock reads — the function is total in its inputs
 * which makes it trivially property-testable (Property 10, task 9.2).
 *
 * Requirements: 4.1, 4.2, 4.3, 4.4.
 */
class ImplicitLabelDetector {

    /**
     * Time window for the OUTGOING rule (Req 4.4): only prior MISSED/REJECTED
     * events within this window count as a signal that the number is "ours".
     * 30 days prevents spurious ALLOW labels from years-old missed calls.
     */
    val outgoingRuleWindowMs: Long = 30L * 24 * 60 * 60 * 1000

    /**
     * Returns the implicit label produced by [event] given prior [history], or
     * `null` if no rule from the table fires (e.g. `state = UNKNOWN`, or an
     * OUTGOING event with no prior MISSED/REJECTED history for that number).
     *
     * @param event the just-finished call event being classified.
     * @param history previously recorded `CallEvent`s for the same user. Only
     *   entries whose `normalizedNumber` matches `event.normalizedNumber` and
     *   whose `direction = INCOMING` with `state ∈ {MISSED, REJECTED}` matter
     *   for the OUTGOING rule; the rest are ignored. Time ordering of
     *   [history] is irrelevant — the rule is purely set-existential per
     *   Property 10 row 4.
     */
    fun detect(event: CallEvent, history: List<CallEvent>): ImplicitLabelResult? {
        return when (event.direction) {
            CallEvent.Direction.OUTGOING -> outgoingRule(event, history)
            CallEvent.Direction.INCOMING -> incomingRule(event)
        }
    }

    /**
     * Req 4.4: an outgoing call to a number that previously MISSED or
     * REJECTED us emits a synchronous `ALLOW` candidate. We require the
     * historical event to be INCOMING to match Property 10 row 4 verbatim
     * (`incoming(...)`); same `normalizedNumber` is mandatory.
     */
    private fun outgoingRule(event: CallEvent, history: List<CallEvent>): ImplicitLabelResult? {
        val number = event.normalizedNumber ?: return null
        val windowStart = event.startedAt - outgoingRuleWindowMs
        val priorIncomingMissedOrRejected = history.any { h ->
            h.normalizedNumber == number &&
                h.direction == CallEvent.Direction.INCOMING &&
                (h.state == CallEvent.CallState.MISSED || h.state == CallEvent.CallState.REJECTED) &&
                h.startedAt >= windowStart
        }
        return if (priorIncomingMissedOrRejected) {
            ImplicitLabelResult(ImplicitLabel.ALLOW)
        } else {
            null
        }
    }

    /**
     * Incoming rules from Req 4.1–4.3. The ANSWERED branch splits on
     * [ANSWERED_ALLOW_MIN_MS] (15 s, Req 4.1 / 4.2). MISSED returns deferred
     * BLOCK because the no-callback condition can only be evaluated 24 h
     * later by [MissedNoCallbackRecheckWorker].
     */
    private fun incomingRule(event: CallEvent): ImplicitLabelResult? = when (event.state) {
        CallEvent.CallState.ANSWERED ->
            if (event.durationMs >= ANSWERED_ALLOW_MIN_MS) {
                ImplicitLabelResult(ImplicitLabel.ALLOW)
            } else {
                ImplicitLabelResult(ImplicitLabel.BLOCK)
            }
        CallEvent.CallState.REJECTED -> ImplicitLabelResult(ImplicitLabel.BLOCK)
        CallEvent.CallState.MISSED -> ImplicitLabelResult(ImplicitLabel.BLOCK, isDeferred = true)
        CallEvent.CallState.UNKNOWN -> null
    }

    companion object {
        /**
         * Threshold (15 s) above which an ANSWERED incoming call is treated
         * as ALLOW per Req 4.1; calls answered for less than this become a
         * BLOCK candidate per Req 4.2.
         */
        const val ANSWERED_ALLOW_MIN_MS: Long = 15_000L

        /**
         * 24-hour callback window used by the deferred MISSED rule (Req 4.3).
         * The detector itself does not look at the clock — this constant is
         * exposed so [enqueueDeferredMissedRecheck] and the worker share a
         * single source of truth.
         */
        const val MISSED_CALLBACK_WINDOW_MS: Long = 24L * 60 * 60 * 1000

        /** Input-data key used by [MissedNoCallbackRecheckWorker]. */
        internal const val KEY_CALL_EVENT_ID = "callEventId"

        /** Input-data key used by [MissedNoCallbackRecheckWorker]. */
        internal const val KEY_NORMALIZED_NUMBER = "normalizedNumber"

        /** Input-data key used by [MissedNoCallbackRecheckWorker]. */
        internal const val KEY_ORIGINAL_STARTED_AT = "originalStartedAt"

        /** Unique-work prefix; full name is `"missed-recheck-<callEventId>"`. */
        private const val UNIQUE_WORK_PREFIX = "missed-recheck-"

        /** Returns the unique-work name used for the per-event recheck. */
        internal fun uniqueWorkName(callEventId: Long): String = UNIQUE_WORK_PREFIX + callEventId
    }
}

/**
 * Schedules the 24-hour MISSED-without-callback re-check via WorkManager.
 *
 * Call this exactly when [ImplicitLabelDetector.detect] returns a result with
 * [ImplicitLabelResult.isDeferred] = `true`. The worker fires after
 * [ImplicitLabelDetector.MISSED_CALLBACK_WINDOW_MS] and either applies a
 * `BLOCK` implicit label (if no outgoing call to the same number happened in
 * the window) or quietly drops it.
 *
 * Uses [ExistingWorkPolicy.KEEP] to make the call idempotent — re-enqueueing
 * the same call event id is a no-op.
 */
fun ImplicitLabelDetector.enqueueDeferredMissedRecheck(
    workManager: WorkManager,
    callEventId: Long,
    normalizedNumber: String,
    originalStartedAt: Long,
) {
    val data = workDataOf(
        ImplicitLabelDetector.KEY_CALL_EVENT_ID to callEventId,
        ImplicitLabelDetector.KEY_NORMALIZED_NUMBER to normalizedNumber,
        ImplicitLabelDetector.KEY_ORIGINAL_STARTED_AT to originalStartedAt,
    )
    val request = OneTimeWorkRequestBuilder<MissedNoCallbackRecheckWorker>()
        .setInitialDelay(ImplicitLabelDetector.MISSED_CALLBACK_WINDOW_MS, TimeUnit.MILLISECONDS)
        .setInputData(data)
        .build()
    workManager.enqueueUniqueWork(
        ImplicitLabelDetector.uniqueWorkName(callEventId),
        ExistingWorkPolicy.KEEP,
        request,
    )
}
