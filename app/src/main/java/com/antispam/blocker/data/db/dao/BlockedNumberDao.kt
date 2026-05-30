package com.antispam.blocker.data.db.dao

import androidx.room.*
import com.antispam.blocker.data.db.entity.BlockedNumber
import kotlinx.coroutines.flow.Flow

@Dao
interface BlockedNumberDao {

    @Query("SELECT * FROM blocked_numbers ORDER BY addedAt DESC")
    fun getAll(): Flow<List<BlockedNumber>>

    @Query("SELECT EXISTS(SELECT 1 FROM blocked_numbers WHERE normalizedNumber = :normalizedNumber LIMIT 1)")
    suspend fun contains(normalizedNumber: String): Boolean

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insert(number: BlockedNumber): Long

    /**
     * Bulk insert. Используется [com.antispam.blocker.data.repository.BlockListRepository.importPrebuilt]
     * для загрузки больших словарей (`spam_numbers.csv` ~ 2.4 млн строк).
     * Прогон через [insert] по одной строке упирается в WAL-fsync и тратит
     * часы — `@Insert` со списком кладёт всё одной prepared-statement-серией
     * внутри одной транзакции. На S24 это десятки секунд вместо часов.
     */
    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertAll(numbers: List<BlockedNumber>): List<Long>

    @Query("DELETE FROM blocked_numbers WHERE normalizedNumber = :normalizedNumber")
    suspend fun deleteByNumber(normalizedNumber: String)

    @Query("DELETE FROM blocked_numbers")
    suspend fun deleteAll()

    @Query("DELETE FROM blocked_numbers WHERE source = 'PREBUILT'")
    suspend fun deletePrebuilt()

    @Query("SELECT * FROM blocked_numbers WHERE pattern IS NOT NULL")
    suspend fun getAllPatterns(): List<BlockedNumber>

    @Query("SELECT COUNT(*) FROM blocked_numbers")
    fun countAll(): kotlinx.coroutines.flow.Flow<Int>

    @Query("SELECT COUNT(*) FROM blocked_numbers WHERE source = 'PREBUILT'")
    fun countPrebuilt(): kotlinx.coroutines.flow.Flow<Int>
}
