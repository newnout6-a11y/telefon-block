package com.antispam.blocker.data.db.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "training_data",
    indices = [Index(value = ["normalizedNumber", "timestamp"], unique = true)]
)
data class TrainingData(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val normalizedNumber: String,
    val featuresJson: String,
    val label: String,
    val weight: Float = 1.0f,
    val userAction: String? = null,
    val timestamp: Long = System.currentTimeMillis()
)
