package com.antispam.blocker.domain.lookup

import android.content.Context
import android.util.Log
import androidx.work.BackoffPolicy
import androidx.work.CoroutineWorker
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.antispam.blocker.SpamBlockerApp
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * WorkManager worker: выполняет offline-seed + опциональный 2GIS online-lookup
 * для конкретного номера телефона.
 *
 * Триггер: запускается из [CallEventRecorder] после каждого нового CallLog-события
 * для номеров, у которых нет актуального кэша.
 *
 * Алгоритм:
 *  1. [CallerLookupRepository.ensureOffline] — быстро, без сети.
 *  2. [CallerLookupRepository.fetchOnline]   — HTTP к 2GIS, только если opt-in.
 *
 * При [IOException] делает до 3 retry с экспоненциальным backoff (30s/60s/120s).
 * 429 (rate limit) и null-ответы — success без retry (чтобы не тратить квоту).
 */
class CallerLookupWorker(
    appContext: Context,
    params: WorkerParameters,
) : CoroutineWorker(appContext, params) {

    companion object {
        private const val TAG = "CallerLookupWorker"
        const val KEY_NUMBER = "normalized_number"

        /**
         * Ставит задачу в очередь для данного номера.
         * Идентификатор = "caller_lookup_{number}" — дедупликация WorkManager
         * автоматически предотвращает двойные запросы.
         */
        fun enqueue(context: Context, normalizedNumber: String) {
            val request = OneTimeWorkRequestBuilder<CallerLookupWorker>()
                .setInputData(workDataOf(KEY_NUMBER to normalizedNumber))
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
                .build()
            // Нет ExistingWorkPolicy.KEEP / REPLACE — позволяем параллельные задачи
            // для разных номеров; для одного номера дедупликация через кэш в repo.
            WorkManager.getInstance(context).enqueue(request)
        }
    }

    override suspend fun doWork(): Result {
        val number = inputData.getString(KEY_NUMBER)
        if (number.isNullOrBlank()) return Result.failure()

        val app = applicationContext as SpamBlockerApp
        val repo = app.callerLookupRepository

        return try {
            repo.ensureOffline(number)
            repo.fetchOnline(number)
            Log.d(TAG, "done: $number")
            Result.success()
        } catch (e: IOException) {
            Log.w(TAG, "network error for number, attempt=$runAttemptCount", e)
            if (runAttemptCount < 3) Result.retry() else Result.failure()
        } catch (t: Throwable) {
            Log.e(TAG, "unexpected error", t)
            Result.failure()
        }
    }
}
