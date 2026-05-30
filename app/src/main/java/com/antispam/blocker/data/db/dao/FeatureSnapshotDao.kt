package com.antispam.blocker.data.db.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import com.antispam.blocker.data.db.entity.FeatureSnapshot

/**
 * DAO for [FeatureSnapshot] rows — the exact feature vector Device_Model
 * saw at decision time.
 *
 * A snapshot may be persisted before its matching `call_event` row exists,
 * so [updateCallEventId] is exposed for post-hoc linking (used by task 12.2
 * when an inserted CallEvent id becomes known after the verdict).
 */
@Dao
interface FeatureSnapshotDao {

    @Insert
    suspend fun insert(snapshot: FeatureSnapshot): Long

    @Query("SELECT * FROM feature_snapshot WHERE callEventId = :callEventId LIMIT 1")
    suspend fun getByCallEventId(callEventId: Long): FeatureSnapshot?

    /**
     * Direct primary-key lookup. Used by [com.antispam.blocker.domain.personal.OnlineTrainer]
     * as a fallback when the caller passes a snapshot id directly instead of a
     * `call_event.id` — see task 12.2: in v1 the screening service writes the
     * snapshot with `callEventId = null` (no CallEvent row is created at decision
     * time yet) and propagates `FeatureSnapshot.id` to the notifier as the
     * feedback handle, so the trainer's `getByCallEventId(id) ?: getById(id)`
     * funnel resolves both kinds of identifiers transparently. Also used by
     * `ExplainabilityDetailScreen` to load a snapshot by its primary key for
     * the long-press detail view (Req 6.2).
     */
    @Query("SELECT * FROM feature_snapshot WHERE id = :snapshotId LIMIT 1")
    suspend fun getById(snapshotId: Long): FeatureSnapshot?

    /**
     * Latest snapshot for a given normalized number, used by the call-log
     * long-press menu (task 18.2 / Req 4.7, 6.2): the journal stores
     * `CallRecord` rows that don't carry a snapshot id, so we resolve the
     * matching `feature_snapshot` row by `normalizedNumber` and pick the
     * most recent one (decision time = `timestamp`).
     */
    @Query(
        "SELECT * FROM feature_snapshot WHERE normalizedNumber = :number " +
            "ORDER BY timestamp DESC LIMIT 1"
    )
    suspend fun getLatestForNumber(number: String): FeatureSnapshot?

    @Query("UPDATE feature_snapshot SET callEventId = :callEventId WHERE id = :snapshotId")
    suspend fun updateCallEventId(snapshotId: Long, callEventId: Long): Int

    @Query("DELETE FROM feature_snapshot WHERE timestamp < :cutoffMs")
    suspend fun deleteOlderThan(cutoffMs: Long): Int

    // ── Bulk export / import (PersonalDataPortabilityService) ──

    @Query("SELECT * FROM feature_snapshot ORDER BY id")
    suspend fun getAllForExport(): List<FeatureSnapshot>

    @Query("DELETE FROM feature_snapshot")
    suspend fun deleteAll(): Int

    @Insert
    suspend fun insertAll(snapshots: List<FeatureSnapshot>)
}
