package com.antispam.blocker.data.db.dao

import androidx.room.*
import com.antispam.blocker.data.db.entity.TrainingData
import kotlinx.coroutines.flow.Flow

@Dao
interface TrainingDataDao {

    @Query("SELECT * FROM training_data ORDER BY timestamp DESC")
    fun getAll(): Flow<List<TrainingData>>

    @Query("SELECT * FROM training_data ORDER BY timestamp DESC LIMIT :limit")
    suspend fun getLatest(limit: Int): List<TrainingData>

    @Query("SELECT COUNT(*) FROM training_data")
    fun countAll(): Flow<Int>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(data: TrainingData): Long

    @Query("DELETE FROM training_data WHERE timestamp < :before")
    suspend fun deleteOlderThan(before: Long)

    @Query("DELETE FROM training_data")
    suspend fun deleteAll()
}
