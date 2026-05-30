package com.antispam.blocker.data.db.dao

import androidx.room.Dao
import androidx.room.Query
import androidx.room.Upsert
import com.antispam.blocker.data.db.entity.AnswerBotMessageEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface AnswerBotMessageDao {

    @Query("SELECT * FROM answer_bot_messages ORDER BY timestamp DESC")
    fun observeAll(): Flow<List<AnswerBotMessageEntity>>

    @Query("SELECT COUNT(*) FROM answer_bot_messages WHERE played = 0")
    fun unreadCount(): Flow<Int>

    @Query("SELECT * FROM answer_bot_messages WHERE id = :id LIMIT 1")
    suspend fun getById(id: Long): AnswerBotMessageEntity?

    @Upsert
    suspend fun upsert(message: AnswerBotMessageEntity)

    @Query("UPDATE answer_bot_messages SET played = 1 WHERE id = :id")
    suspend fun markPlayed(id: Long)

    @Query("UPDATE answer_bot_messages SET spam = :spam WHERE id = :id")
    suspend fun markSpam(id: Long, spam: Boolean)

    @Query("DELETE FROM answer_bot_messages WHERE timestamp < :threshold")
    suspend fun purgeOlderThan(threshold: Long)
}
