package com.antispam.blocker.data.db.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "allowed_numbers",
    indices = [Index(value = ["normalizedNumber"], unique = true)]
)
data class AllowedNumber(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val normalizedNumber: String,
    val originalNumber: String,
    val addedAt: Long = System.currentTimeMillis()
)
