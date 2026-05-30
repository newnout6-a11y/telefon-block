package com.antispam.blocker.data.db.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * On-device call telemetry event used by Device_Model (personal classifier).
 *
 * Captures direction, state, duration and timestamps for both incoming and
 * outgoing calls. Stored locally only; subject to 90-day retention by
 * `TelemetryRetentionWorker` and per-source toggles in Settings → Privacy.
 *
 * Indexed by `startedAt` for retention/aggregation scans, and by
 * `normalizedNumber` for per-number lookups (answer rate, prefix counts).
 */
@Entity(
    tableName = "call_event",
    indices = [Index("startedAt"), Index("normalizedNumber")]
)
data class CallEvent(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val normalizedNumber: String?,
    val isHidden: Boolean,
    val direction: Direction,
    val state: CallState,
    val durationMs: Long,
    val startedAt: Long,
    val endedAt: Long? = null,
    val isContact: Boolean = false
) {
    enum class Direction { INCOMING, OUTGOING }
    enum class CallState { ANSWERED, REJECTED, MISSED, UNKNOWN }
}
