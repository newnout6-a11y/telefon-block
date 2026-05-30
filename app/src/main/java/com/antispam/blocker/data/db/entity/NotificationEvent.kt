package com.antispam.blocker.data.db.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Notification event captured by `PersonalNotificationListenerService`.
 *
 * Stores ONLY package name, derived [CategoryBucket] and timestamp — the
 * notification body, title and any extras MUST NEVER be persisted (see
 * Requirements 1.5, 7.10).
 *
 * Indexed by `timestamp` for time-window queries (last 10 minutes, retention).
 */
@Entity(
    tableName = "notification_event",
    indices = [Index("timestamp")]
)
data class NotificationEvent(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val packageName: String,
    val categoryBucket: CategoryBucket,
    val timestamp: Long
) {
    enum class CategoryBucket { BANK, MARKETPLACE, MESSENGER, EMAIL, OTHER }
}
