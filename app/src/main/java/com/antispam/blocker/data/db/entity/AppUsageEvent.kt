package com.antispam.blocker.data.db.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Foreground app usage event sourced from `UsageStatsManager`.
 *
 * Used by Device_Model to derive "recently opened bank/gov/marketplace/
 * messenger app" features. Reuses [NotificationEvent.CategoryBucket] for
 * a single, consistent category enum across notification and usage signals.
 *
 * Indexed by `foregroundAt` for time-window queries and retention.
 */
@Entity(
    tableName = "app_usage_event",
    indices = [Index("foregroundAt")]
)
data class AppUsageEvent(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val packageName: String,
    val categoryBucket: NotificationEvent.CategoryBucket,
    val foregroundAt: Long
)
