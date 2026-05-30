package com.antispam.blocker.domain.personal

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.antispam.blocker.SpamBlockerApp
import com.antispam.blocker.data.db.entity.CallEvent
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Second-stage classifier for the MISSED-without-callback rule (Req 4.3).
 *
 * [ImplicitLabelDetector] flags a MISSED incoming event as a *deferred* BLOCK
 * candidate; this worker is enqueued via
 * [enqueueDeferredMissedRecheck] with a 24-hour `setInitialDelay`. When it
 * eventually fires:
 *
 * 1. It re-reads the per-number history from [com.antispam.blocker.data.db.dao.CallEventDao].
 * 2. If **no** outgoing call to the same `normalizedNumber` happened in the
 *    `[originalStartedAt, originalStartedAt + 24 h]` window, the BLOCK label
 *    becomes effective and is dispatched through [OnlineTrainerHandle.applyImplicitLabel].
 * 3. Otherwise the candidate is silently dropped (the user clearly recognised
 *    the caller, so we must not penalise the model).
 *
 * The worker has no opinion of its own about ALLOW vs BLOCK semantics — it
 * simply re-evaluates the deferral condition and forwards the result to
 * `OnlineTrainer`. That keeps [ImplicitLabelDetector.detect] pure and lets us
 * unit-test the deferral logic directly via DAO fakes.
 *
 * Failures are logged and converted to [Result.retry] so transient I/O errors
 * don't permanently lose the label.
 *
 * Requirements: 4.3.
 */
class MissedNoCallbackRecheckWorker(
    appContext: Context,
    params: WorkerParameters,
) : CoroutineWorker(appContext, params) {

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val callEventId = inputData.getLong(ImplicitLabelDetector.KEY_CALL_EVENT_ID, -1L)
        val normalizedNumber = inputData.getString(ImplicitLabelDetector.KEY_NORMALIZED_NUMBER)
        val originalStartedAt = inputData.getLong(ImplicitLabelDetector.KEY_ORIGINAL_STARTED_AT, -1L)

        if (callEventId < 0 || normalizedNumber.isNullOrBlank() || originalStartedAt < 0) {
            Log.w(TAG, "missing input data: id=$callEventId number=$normalizedNumber t=$originalStartedAt")
            // Bad input is not retryable — drop quietly.
            return@withContext Result.success()
        }

        return@withContext try {
            val app = SpamBlockerApp.instance
            val callEventDao = app.database.callEventDao()
            val history = callEventDao.getByNumber(normalizedNumber)

            if (hadCallbackInWindow(history, originalStartedAt)) {
                Log.i(
                    TAG,
                    "callback found within 24h for callEventId=$callEventId; dropping deferred BLOCK",
                )
                return@withContext Result.success()
            }

            val trainer = OnlineTrainerLocator.resolve(app)
            if (trainer == null) {
                // OnlineTrainer (task 10.1) has not been wired into the app yet.
                // Log and succeed — the deferred candidate is dropped rather than
                // crashing the call-screening hot path.
                Log.w(TAG, "OnlineTrainer not available; dropping deferred BLOCK for $callEventId")
                return@withContext Result.success()
            }
            trainer.applyImplicitLabel(callEventId, ImplicitLabel.BLOCK)
            Log.i(TAG, "applied deferred implicit BLOCK for callEventId=$callEventId")
            Result.success()
        } catch (t: Throwable) {
            Log.w(TAG, "missed-no-callback recheck failed for callEventId=$callEventId", t)
            Result.retry()
        }
    }

    /**
     * Returns `true` iff [history] contains an OUTGOING call to the same
     * number within the 24-hour window `[originalStartedAt,
     * originalStartedAt + 24 h]`. The window is inclusive on both ends to
     * match Property 10 row 3 (`t ∈ [e.startedAt, e.startedAt + 24h]`).
     */
    private fun hadCallbackInWindow(history: List<CallEvent>, originalStartedAt: Long): Boolean {
        val windowEnd = originalStartedAt + ImplicitLabelDetector.MISSED_CALLBACK_WINDOW_MS
        return history.any { h ->
            h.direction == CallEvent.Direction.OUTGOING &&
                h.startedAt in originalStartedAt..windowEnd
        }
    }

    companion object {
        private const val TAG = "MissedRecheckWrk"
    }
}

/**
 * Minimal abstraction over the not-yet-implemented [OnlineTrainer] (task 10.1)
 * so [MissedNoCallbackRecheckWorker] doesn't need a forward reference at
 * compile time.
 *
 * Task 10.1 will provide a concrete `OnlineTrainer` and either implement this
 * interface directly or expose an instance through [SpamBlockerApp]; the
 * worker resolves it via [OnlineTrainerLocator] at run time.
 */
interface OnlineTrainerHandle {
    suspend fun applyImplicitLabel(callEventId: Long, label: ImplicitLabel)
}

/**
 * Late-bound lookup for the runtime [OnlineTrainerHandle].
 *
 * [OnlineTrainer] (task 10.1) is exposed via [SpamBlockerApp.onlineTrainer]
 * and implements [OnlineTrainerHandle] directly, so this resolver simply
 * forwards the application-scoped instance. Returning the lazy property is
 * safe — initialization is idempotent and runs on the worker's IO
 * dispatcher.
 */
internal object OnlineTrainerLocator {
    fun resolve(app: SpamBlockerApp): OnlineTrainerHandle? = app.onlineTrainer
}
