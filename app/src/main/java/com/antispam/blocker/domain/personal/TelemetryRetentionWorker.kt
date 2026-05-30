package com.antispam.blocker.domain.personal

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.antispam.blocker.SpamBlockerApp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Daily retention sweep for the four on-device telemetry tables introduced
 * by the personal classifier feature: `call_event`, `notification_event`,
 * `app_usage_event` and `feature_snapshot`.
 *
 * Removes any row whose timestamp (`startedAt` / `timestamp` / `foregroundAt`
 * depending on the table) is strictly older than `now − 90 days`. This keeps
 * the on-device dataset bounded and aligned with the privacy contract from
 * Requirement 2.4.
 *
 * Scheduling lives in task 14.2 (`SpamBlockerApp.onCreate`); this class only
 * implements the work itself.
 *
 * Failure handling mirrors [com.antispam.blocker.data.worker.RemoteUpdateWorker]:
 * any [Throwable] is logged and converted to [Result.retry] so WorkManager
 * tries again later instead of marking the unique work permanently failed.
 */
class TelemetryRetentionWorker(
    appContext: Context,
    params: WorkerParameters,
) : CoroutineWorker(appContext, params) {

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        return@withContext try {
            val now = System.currentTimeMillis()
            val cutoff = now - RETENTION_WINDOW_MS
            val db = SpamBlockerApp.instance.database

            val deletedCallEvents = db.callEventDao().deleteOlderThan(cutoff)
            // Notification events нужны только для фич recent_10m —
            // хранить их 90 дней бессмысленно. 24h с запасом.
            val deletedNotifications = db.notificationEventDao().deleteOlderThan(now - NOTIFICATION_RETENTION_MS)
            val deletedAppUsage = db.appUsageEventDao().deleteOlderThan(cutoff)
            val deletedSnapshots = db.featureSnapshotDao().deleteOlderThan(cutoff)

            Log.i(
                TAG,
                "retention sweep cutoff=$cutoff notifCutoff=${now - NOTIFICATION_RETENTION_MS} " +
                    "callEvent=$deletedCallEvents " +
                    "notificationEvent=$deletedNotifications " +
                    "appUsageEvent=$deletedAppUsage " +
                    "featureSnapshot=$deletedSnapshots"
            )
            Result.success()
        } catch (t: Throwable) {
            Log.w(TAG, "telemetry retention sweep failed", t)
            Result.retry()
        }
    }

    companion object {
        private const val TAG = "TelemetryRetentionWrk"

        /** Unique work name used by [androidx.work.WorkManager]. */
        const val UNIQUE_NAME = "TelemetryRetention"

        /** 90 calendar days expressed in milliseconds. */
        private const val RETENTION_WINDOW_MS: Long = 90L * 24 * 60 * 60 * 1000
        /** 1 hour — notification events are only used for recent_10m features. */
        private const val NOTIFICATION_RETENTION_MS: Long = 60 * 60 * 1000
    }
}
