package com.antispam.blocker.data.db.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.antispam.blocker.data.db.entity.DecisionRecord
import kotlinx.coroutines.flow.Flow

@Dao
interface DecisionRecordDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(record: DecisionRecord): Long

    @Query("SELECT * FROM decision_records ORDER BY timestamp DESC LIMIT :limit")
    fun observeRecent(limit: Int): Flow<List<DecisionRecord>>

    @Query("SELECT * FROM decision_records ORDER BY timestamp DESC LIMIT :limit")
    suspend fun getRecent(limit: Int): List<DecisionRecord>

    @Query("SELECT * FROM decision_records WHERE id = :id LIMIT 1")
    suspend fun getById(id: Long): DecisionRecord?

    @Query("SELECT COUNT(*) FROM decision_records")
    suspend fun count(): Int

    @Query("SELECT COUNT(*) FROM decision_records WHERE verdict = :verdict")
    suspend fun countByVerdict(verdict: String): Int

    @Query("SELECT COUNT(*) FROM decision_records WHERE source = :source")
    suspend fun countBySource(source: String): Int

    @Query("SELECT COUNT(*) FROM decision_records WHERE userAction IS NOT NULL")
    suspend fun countWithFeedback(): Int

    @Query("""
        SELECT COUNT(*) FROM decision_records
        WHERE userAction IS NOT NULL
          AND (
                (verdict IN ('BLOCK','WARN') AND userAction IN ('IS_SCAM','MARK_SPAM','DISMISS'))
             OR (verdict = 'ALLOW' AND userAction IN ('NOT_SPAM','UNBLOCK','ANSWER'))
          )
    """)
    suspend fun countAgreeingFeedback(): Int

    @Query("UPDATE decision_records SET userAction = :action, userActionTimestamp = :ts WHERE id = :id")
    suspend fun setUserAction(id: Long, action: String, ts: Long)

    @Query("DELETE FROM decision_records WHERE timestamp < :cutoff")
    suspend fun deleteOlderThan(cutoff: Long): Int

    @Query("DELETE FROM decision_records")
    suspend fun clear()
}
