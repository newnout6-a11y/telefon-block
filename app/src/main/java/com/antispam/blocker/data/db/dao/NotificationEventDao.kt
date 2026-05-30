package com.antispam.blocker.data.db.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import com.antispam.blocker.data.db.entity.NotificationEvent

/**
 * DAO for [NotificationEvent] rows captured by the personal notification
 * listener. Persists ONLY package name, derived [NotificationEvent.CategoryBucket]
 * and timestamp — never the notification body or extras (Requirements 1.5, 7.10).
 */
@Dao
interface NotificationEventDao {

    @Insert
    suspend fun insert(event: NotificationEvent): Long

    /** Total notification events ever recorded — UI Privacy «Прозрачность данных». */
    @Query("SELECT COUNT(*) FROM notification_event")
    suspend fun countAll(): Int

    /** N most recent notification events for transparency UI (DESC by timestamp). */
    @Query("SELECT * FROM notification_event ORDER BY timestamp DESC LIMIT :limit")
    suspend fun recent(limit: Int): List<NotificationEvent>

    /**
     * Returns notification events for the given [bucket] whose `timestamp`
     * falls inside the half-open window `[sinceMs, untilMs)`.
     */
    @Query(
        "SELECT * FROM notification_event " +
            "WHERE categoryBucket = :bucket AND timestamp >= :sinceMs AND timestamp < :untilMs " +
            "ORDER BY timestamp DESC"
    )
    suspend fun queryByCategoryWithin(
        bucket: NotificationEvent.CategoryBucket,
        sinceMs: Long,
        untilMs: Long
    ): List<NotificationEvent>

    /**
     * Convenience count for the same window — Device_Model typically only
     * needs "did any happen" / "how many" and avoids materialising rows.
     */
    @Query(
        "SELECT COUNT(*) FROM notification_event " +
            "WHERE categoryBucket = :bucket AND timestamp >= :sinceMs AND timestamp < :untilMs"
    )
    suspend fun countByCategoryWithin(
        bucket: NotificationEvent.CategoryBucket,
        sinceMs: Long,
        untilMs: Long
    ): Int

    @Query("DELETE FROM notification_event WHERE timestamp < :cutoffMs")
    suspend fun deleteOlderThan(cutoffMs: Long): Int

    // ── Bulk export / import (PersonalDataPortabilityService) ──

    @Query("SELECT * FROM notification_event ORDER BY id")
    suspend fun getAllForExport(): List<NotificationEvent>

    @Query("DELETE FROM notification_event")
    suspend fun deleteAll(): Int

    @Insert
    suspend fun insertAll(events: List<NotificationEvent>)
}
