package com.antispam.blocker.data.db.dao

import androidx.room.Dao
import androidx.room.Query
import androidx.room.Upsert
import com.antispam.blocker.data.db.entity.CallerLookup
import kotlinx.coroutines.flow.Flow

@Dao
interface CallerLookupDao {

    /** Реактивный поток для UI — обновляется автоматически при upsert. */
    @Query("SELECT * FROM caller_lookup WHERE normalizedNumber = :number LIMIT 1")
    fun observe(number: String): Flow<CallerLookup?>

    /** Однократное чтение для бизнес-логики (не suspend для совместимости с sync-кодом). */
    @Query("SELECT * FROM caller_lookup WHERE normalizedNumber = :number LIMIT 1")
    suspend fun getByNumber(number: String): CallerLookup?

    /** INSERT OR REPLACE — Room @Upsert (Room 2.5+). */
    @Upsert
    suspend fun upsert(lookup: CallerLookup)

    /** Удаляет устаревшие записи с lookedUpAt < threshold. */
    @Query("DELETE FROM caller_lookup WHERE lookedUpAt < :threshold")
    suspend fun purgeStale(threshold: Long)

    /** Удаляет все оффлайн-записи (используется при обновлении базы РКН). */
    @Query("DELETE FROM caller_lookup WHERE source = 'offline'")
    suspend fun purgeOffline()
}
