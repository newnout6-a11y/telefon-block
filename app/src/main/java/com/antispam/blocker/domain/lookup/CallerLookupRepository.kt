package com.antispam.blocker.domain.lookup

import android.util.Log
import com.antispam.blocker.data.db.dao.CallerLookupDao
import com.antispam.blocker.data.db.entity.CallerLookup
import com.antispam.blocker.data.prefs.CallerLookupSettingsStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext

/**
 * Оркестратор определения звонящего.
 *
 * Стратегия (для каждого номера N):
 *  1. Кэш актуален → возвращаем.
 *  2. Offline-lookup (libphonenumber) → сохраняем.
 *  3. Если 2GIS включён — WorkManager worker делает online-уточнение async.
 *
 * UI подписывается на [observe] — поток обновляется автоматически.
 */
class CallerLookupRepository(
    private val dao: CallerLookupDao,
    private val offlineLookup: OfflineCallerLookup,
    private val twoGisLookup: TwoGisCallerLookup,
    private val settingsStore: CallerLookupSettingsStore,
) {
    companion object {
        private const val TAG = "CallerLookupRepo"
    }

    /** Flow для UI — обновляется после любого upsert. */
    fun observe(normalizedNumber: String): Flow<CallerInfo?> =
        dao.observe(normalizedNumber).map { it?.toCallerInfo() }

    /**
     * Гарантирует наличие актуального оффлайн-кэша для номера.
     * Не делает сетевых запросов. Вызывается из Worker и UI при необходимости.
     */
    suspend fun ensureOffline(normalizedNumber: String): CallerInfo? {
        val cached = dao.getByNumber(normalizedNumber)
        if (cached != null && !cached.isStale()) return cached.toCallerInfo()

        val info = withContext(Dispatchers.Default) {
            offlineLookup.lookup(normalizedNumber)
        } ?: return cached?.toCallerInfo()

        dao.upsert(info.toEntity())
        Log.d(TAG, "offline $normalizedNumber → ${info.subtitle}")
        return info
    }

    /**
     * 2GIS online-lookup. Вызывается только из [CallerLookupWorker].
     * @throws java.io.IOException при сетевой ошибке — worker делает retry.
     */
    suspend fun fetchOnline(normalizedNumber: String) {
        val enabled = settingsStore.twoGisEnabledFlow.first()
        if (!enabled) return
        val apiKey = settingsStore.apiKeyFlow.first()
        if (apiKey.isNullOrBlank()) return

        // Не трогаем свежую 2GIS-запись
        val cached = dao.getByNumber(normalizedNumber)
        if (cached != null && cached.source == "2gis" && !cached.isStale()) return

        val info = withContext(Dispatchers.IO) {
            twoGisLookup.lookup(normalizedNumber, apiKey)
        } ?: return  // null = ошибка связи → не кэшируем

        dao.upsert(info.toEntity())
        Log.d(TAG, "2gis  $normalizedNumber → ${info.subtitle ?: "negative"}")
    }

    /** Удаляет старые записи (вызывается из TelemetryRetentionWorker). */
    suspend fun purgeStale() {
        dao.purgeStale(System.currentTimeMillis() - CallerLookup.TTL_ONLINE_MS * 2)
    }
}
