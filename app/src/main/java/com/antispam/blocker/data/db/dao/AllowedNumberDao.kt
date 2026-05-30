package com.antispam.blocker.data.db.dao

import androidx.room.*
import com.antispam.blocker.data.db.entity.AllowedNumber
import kotlinx.coroutines.flow.Flow

@Dao
interface AllowedNumberDao {

    @Query("SELECT * FROM allowed_numbers ORDER BY addedAt DESC")
    fun getAll(): Flow<List<AllowedNumber>>

    @Query("SELECT EXISTS(SELECT 1 FROM allowed_numbers WHERE normalizedNumber = :normalizedNumber LIMIT 1)")
    suspend fun contains(normalizedNumber: String): Boolean

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insert(number: AllowedNumber): Long

    @Query("DELETE FROM allowed_numbers WHERE normalizedNumber = :normalizedNumber")
    suspend fun deleteByNumber(normalizedNumber: String)

    @Query("DELETE FROM allowed_numbers")
    suspend fun deleteAll()
}
