package com.antispam.blocker.data.db.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import com.antispam.blocker.domain.detector.Verdict

@Entity(
    tableName = "call_records",
    indices = [Index(value = ["featureSnapshotId"])],
)
data class CallRecord(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val normalizedNumber: String?,
    val originalNumber: String?,
    val verdict: Verdict,
    val timestamp: Long = System.currentTimeMillis(),
    val ruleName: String? = null,
    /**
     * P1 fix: id строки `feature_snapshot`, относящейся ИМЕННО к этому звонку
     * (а не последняя по номеру). Без этой ссылки long-press в журнале по
     * старой записи открывал snapshot последнего звонка с того же номера,
     * и юзер видел "чужой" explainability-вектор. `null` для записей,
     * созданных до фикса, либо для fast-path вердиктов (allow-/block-list /
     * emergency / disabled), где snapshot вообще не сохранялся.
     */
    val featureSnapshotId: Long? = null,
)
