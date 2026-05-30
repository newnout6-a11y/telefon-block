package com.antispam.blocker.data.db.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "blocked_numbers",
    indices = [Index(value = ["normalizedNumber"], unique = true)]
)
data class BlockedNumber(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val normalizedNumber: String,
    val originalNumber: String,
    val addedAt: Long = System.currentTimeMillis(),
    val source: Source = Source.MANUAL,
    val pattern: String? = null
) {
    enum class Source { MANUAL, PREBUILT, REPORT, FEEDBACK }
}
