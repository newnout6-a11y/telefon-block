package com.antispam.blocker.data.db.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import com.antispam.blocker.data.db.entity.AppUsageEvent
import com.antispam.blocker.data.db.entity.NotificationEvent

/**
 * DAO for [AppUsageEvent] rows derived from `UsageStatsManager`.
 *
 * Reuses [NotificationEvent.CategoryBucket] for a single, consistent
 * category enum across notification and usage signals.
 */
@Dao
interface AppUsageEventDao {

    @Insert
    suspend fun insert(event: AppUsageEvent): Long

    /** Total app-usage events ever recorded — UI Privacy «Прозрачность данных». */
    @Query("SELECT COUNT(*) FROM app_usage_event")
    suspend fun countAll(): Int

    /** N most recent app-usage events for transparency UI (DESC by foregroundAt). */
    @Query("SELECT * FROM app_usage_event ORDER BY foregroundAt DESC LIMIT :limit")
    suspend fun recent(limit: Int): List<AppUsageEvent>

    /**
     * Returns foreground-app events for the given [bucket] whose
     * `foregroundAt` falls inside the half-open window `[sinceMs, untilMs)`.
     */
    @Query(
        "SELECT * FROM app_usage_event " +
            "WHERE categoryBucket = :bucket AND foregroundAt >= :sinceMs AND foregroundAt < :untilMs " +
            "ORDER BY foregroundAt DESC"
    )
    suspend fun queryByCategoryWithin(
        bucket: NotificationEvent.CategoryBucket,
        sinceMs: Long,
        untilMs: Long
    ): List<AppUsageEvent>

    @Query("DELETE FROM app_usage_event WHERE foregroundAt < :cutoffMs")
    suspend fun deleteOlderThan(cutoffMs: Long): Int

    // ── Bulk export / import (PersonalDataPortabilityService) ──

    @Query("SELECT * FROM app_usage_event ORDER BY id")
    suspend fun getAllForExport(): List<AppUsageEvent>

    @Query("DELETE FROM app_usage_event")
    suspend fun deleteAll(): Int

    @Insert
    suspend fun insertAll(events: List<AppUsageEvent>)
}
