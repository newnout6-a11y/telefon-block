package com.antispam.blocker.data.db.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "answer_bot_messages",
    indices = [Index(value = ["timestamp"])]
)
data class AnswerBotMessageEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0L,

    val normalizedNumber: String,

    val transcription: String? = null,

    val audioPath: String,

    val durationMs: Long = 0L,

    val played: Boolean = false,

    val spam: Boolean? = null,

    val timestamp: Long = System.currentTimeMillis(),
)
