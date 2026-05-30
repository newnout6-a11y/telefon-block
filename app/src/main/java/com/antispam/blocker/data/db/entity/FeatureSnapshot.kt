package com.antispam.blocker.data.db.entity

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Snapshot of the Device_Model feature vector taken at decision time.
 *
 * Captures the exact numbers the model saw when it produced a verdict, so
 * that a later SGD step (triggered by an Implicit/Explicit label) is applied
 * to the same vector — see Requirement 2.3.
 *
 * `callEventId` is a nullable foreign key into [CallEvent] with
 * `ON DELETE SET NULL`: a snapshot may be persisted before the matching
 * call_event row exists, and we never want call-event deletion to cascade
 * away the audit trail of decisions.
 *
 * `featuresJson` is a JSON object `{feature_name: float}` covering all
 * features in `DeviceFeatures.NAMES`. `featureSchemaVersion` lets the
 * trainer skip stale snapshots after a feature schema bump.
 */
@Entity(
    tableName = "feature_snapshot",
    foreignKeys = [
        ForeignKey(
            entity = CallEvent::class,
            parentColumns = ["id"],
            childColumns = ["callEventId"],
            onDelete = ForeignKey.SET_NULL
        )
    ],
    indices = [Index("callEventId"), Index("timestamp")]
)
data class FeatureSnapshot(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val callEventId: Long?,
    val normalizedNumber: String?,
    val timestamp: Long,
    val featuresJson: String,
    val featureSchemaVersion: Int,
    val weightsHash: String?,
    val deviceProbBlock: Float
)
