package com.antispam.blocker.data.db.dao

import androidx.room.*
import com.antispam.blocker.data.db.entity.CallRecord
import kotlinx.coroutines.flow.Flow

@Dao
interface CallRecordDao {

    @Query("SELECT * FROM call_records ORDER BY timestamp DESC")
    fun getAll(): Flow<List<CallRecord>>

    @Query("SELECT * FROM call_records WHERE timestamp >= :from ORDER BY timestamp DESC")
    fun getSince(from: Long): Flow<List<CallRecord>>

    @Query("SELECT COUNT(*) FROM call_records WHERE verdict = 'BLOCK' AND timestamp >= :from")
    fun countBlockedSince(from: Long): Flow<Int>

    @Query("SELECT COUNT(*) FROM call_records WHERE verdict = 'WARN' AND timestamp >= :from")
    fun countWarnedSince(from: Long): Flow<Int>

    @Query("SELECT COUNT(*) FROM call_records WHERE verdict = 'ALLOW' AND timestamp >= :from")
    fun countAllowedSince(from: Long): Flow<Int>

    @Query("SELECT COUNT(*) FROM call_records WHERE normalizedNumber = :number AND timestamp >= :since")
    suspend fun countByNumberSince(number: String, since: Long): Int

    /**
     * Кол-во записей по номеру за окно с заданным `verdict` (BLOCK/WARN/ALLOW).
     * Используется фичей `previouslyRejected` в [com.antispam.blocker.domain.scoring.FeatureExtractor]
     * — там нужен именно «реально отклонённый», а не «звонил недавно». Без
     * фильтра по вердикту фича триггерилась на любом повторном звонке от
     * знакомого и заставляла модель кричать BLOCK по легитимным номерам.
     */
    @Query("SELECT COUNT(*) FROM call_records WHERE normalizedNumber = :number AND verdict = :verdict AND timestamp >= :since")
    suspend fun countByNumberAndVerdictSince(number: String, verdict: String, since: Long): Int

    @Query("SELECT COUNT(*) FROM call_records WHERE normalizedNumber LIKE :prefixPattern AND timestamp >= :since")
    suspend fun countByPrefixSince(prefixPattern: String, since: Long): Int

    @Insert
    suspend fun insert(record: CallRecord): Long

    /**
     * Перезаписать вердикт для всех существующих записей по этому номеру.
     * Используется кнопками «+» / «✓» в журнале: пользователь жмёт «в чёрный
     * список» — мы добавляем номер в `blocked_numbers`, а здесь подменяем
     * `verdict` на BLOCK у уже залогированных звонков, чтобы UI-фильтр
     * `record.verdict != Verdict.BLOCK` корректно прятал кнопку и журнал
     * перестал противоречить «Чёрному списку».
     */
    @Query("UPDATE call_records SET verdict = :verdict WHERE normalizedNumber = :number")
    suspend fun updateVerdictByNumber(number: String, verdict: String): Int

    /** Все уникальные normalizedNumber для бэкфилла CallerID. */
    @Query("SELECT DISTINCT normalizedNumber FROM call_records WHERE normalizedNumber IS NOT NULL")
    suspend fun getDistinctNumbers(): List<String>

    @Query("DELETE FROM call_records WHERE timestamp < :before")
    suspend fun deleteOlderThan(before: Long)
}
