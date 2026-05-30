package com.antispam.blocker.data.db.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import com.antispam.blocker.data.db.entity.CallEvent

/**
 * DAO for [CallEvent] rows used by Device_Model (personal classifier).
 *
 * Exposes per-number lookups, prefix-bucket counts (e.g. last 7 days) and
 * a retention sweep keyed on `startedAt`.
 */
@Dao
interface CallEventDao {

    @Insert
    suspend fun insert(event: CallEvent): Long

    @Query("SELECT * FROM call_event WHERE normalizedNumber = :normalizedNumber ORDER BY startedAt DESC")
    suspend fun getByNumber(normalizedNumber: String): List<CallEvent>

    /** Total CallEvent rows ever recorded — UI Privacy «Прозрачность данных». */
    @Query("SELECT COUNT(*) FROM call_event")
    suspend fun countAll(): Int

    /**
     * Recent N records from the on-device call event log. Used by the
     * Privacy → Transparency block in Settings to prove to the user that
     * the app actually reads CallLog: we display masked numbers + relative
     * timestamps, so they can visually confirm their own recent calls.
     * Returned in DESC order by startedAt.
     */
    @Query("SELECT * FROM call_event ORDER BY startedAt DESC LIMIT :limit")
    suspend fun recent(limit: Int): List<CallEvent>

    /**
     * Counts events whose `normalizedNumber` starts with [prefix] and whose
     * `startedAt` is at or after [sinceMs]. The caller computes the cutoff
     * (e.g. 7d window) so this method stays trivial and pure.
     */
    @Query(
        "SELECT COUNT(*) FROM call_event " +
            "WHERE normalizedNumber LIKE :prefix || '%' AND startedAt >= :sinceMs"
    )
    suspend fun countByPrefixSince(prefix: String, sinceMs: Long): Int

    @Query("DELETE FROM call_event WHERE startedAt < :cutoffMs")
    suspend fun deleteOlderThan(cutoffMs: Long): Int

    // ── Bulk export / import (PersonalDataPortabilityService) ──

    /**
     * Returns all rows in stable insertion order — used by export to produce
     * a deterministic JSON payload (Req 2.5).
     */
    @Query("SELECT * FROM call_event ORDER BY id")
    suspend fun getAllForExport(): List<CallEvent>

    /** Wipes the table — used by import to atomically replace contents (Req 2.6). */
    @Query("DELETE FROM call_event")
    suspend fun deleteAll(): Int

    @Insert
    suspend fun insertAll(events: List<CallEvent>)
}
