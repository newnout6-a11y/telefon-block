package com.antispam.blocker.data.db.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Полный снимок решения детектора по конкретному звонку.
 *
 * Хранит вердикт, источник (model/rule/blacklist/...), сырой выход модели
 * и компактный JSON со всеми 32 признаками + active reasons. Используется
 * Debug/Insight-экранами для аудита того, как ИИ принимает решения.
 */
@Entity(
    tableName = "decision_records",
    indices = [Index(value = ["timestamp"]), Index(value = ["normalizedNumber"])]
)
data class DecisionRecord(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val timestamp: Long,
    val rawNumber: String?,
    val normalizedNumber: String?,
    val verdict: String,
    val score: Int,
    val source: String,
    val confidence: String,
    val modelAllowProb: Float,
    val modelWarnProb: Float,
    val modelBlockProb: Float,
    val modelInputSize: Int,
    val featuresJson: String,
    val reasonsJson: String,
    val activeFactorsJson: String,
    val ruleScore: Int,
    val warnThreshold: Int,
    val blockThreshold: Int,
    val userAction: String? = null,
    val userActionTimestamp: Long? = null,
    val modelVersion: String? = null
)
