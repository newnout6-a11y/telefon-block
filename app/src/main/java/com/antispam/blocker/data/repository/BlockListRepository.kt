package com.antispam.blocker.data.repository

import com.antispam.blocker.data.assets.PrebuiltBlocklistReader
import com.antispam.blocker.data.db.dao.AllowedNumberDao
import com.antispam.blocker.data.db.dao.BlockedNumberDao
import com.antispam.blocker.data.db.entity.AllowedNumber
import com.antispam.blocker.data.db.entity.BlockedNumber
import com.antispam.blocker.util.PhoneNormalizer
import kotlinx.coroutines.flow.Flow

class BlockListRepository(
    private val blockedDao: BlockedNumberDao,
    private val allowedDao: AllowedNumberDao,
    private val phoneNormalizer: PhoneNormalizer,
    /**
     * Read-only словарь из 2.4M+ известных спам-номеров, упакованный как
     * sqlite-asset (`assets/prebuilt_blocklist.db`) и копируемый в filesDir
     * при первом запуске. Опциональный — для unit-тестов и старых
     * вызовов BlockListRepository(...) без префикса (например,
     * `BlockListRepository(...).addToBlockList`) можно передавать null,
     * тогда prebuilt-проверка просто пропускается.
     */
    private val prebuiltReader: PrebuiltBlocklistReader? = null
) {

    val allBlocked: Flow<List<BlockedNumber>> = blockedDao.getAll()
    val allAllowed: Flow<List<AllowedNumber>> = allowedDao.getAll()
    val totalCount: Flow<Int> = blockedDao.countAll()
    val prebuiltCount: Flow<Int> = blockedDao.countPrebuilt()

    private var cachedPatterns: List<Regex>? = null

    suspend fun isBlocked(rawNumber: String): Boolean {
        val normalized = phoneNormalizer.normalize(rawNumber) ?: return false

        // 1. Prebuilt-словарь (sqlite-asset). На горячем пути — один EXISTS
        // с уникальным индексом, микросекунды.
        if (prebuiltReader?.contains(normalized) == true) return true

        // 2. Ручные/feedback-блокировки + regex-маски в Room.
        if (blockedDao.contains(normalized)) return true

        val patterns = cachedPatterns ?: blockedDao.getAllPatterns()
            .mapNotNull { entry ->
                try { Regex(entry.pattern!!) } catch (_: Exception) { null }
            }.also { cachedPatterns = it }

        if (patterns.any { it.containsMatchIn(normalized) }) return true

        // 3. Prebuilt regex/префикс-маски (≤ 30 штук, кэшируются).
        if (prebuiltReader?.findPatternMatch(normalized) != null) return true

        return false
    }

    suspend fun isAllowed(rawNumber: String): Boolean {
        val normalized = phoneNormalizer.normalize(rawNumber) ?: return false
        return allowedDao.contains(normalized)
    }

    suspend fun addToBlockList(rawNumber: String, source: BlockedNumber.Source = BlockedNumber.Source.MANUAL, pattern: String? = null) {
        val rawTrimmed = rawNumber.trim()
        if (rawTrimmed.isBlank()) return

        if (pattern != null) {
            // Маска: не требуем валидного телефонного формата.
            // Проверяем что regex сам компилируется, иначе запись бесполезна.
            try {
                Regex(pattern)
            } catch (_: Exception) {
                return
            }
            blockedDao.insert(
                BlockedNumber(
                    normalizedNumber = rawTrimmed, // для маски храним как есть (юзер увидит в списке)
                    originalNumber = rawTrimmed,
                    source = source,
                    pattern = pattern
                )
            )
            cachedPatterns = null
            return
        }

        // Обычный номер — строгая нормализация
        val normalized = phoneNormalizer.normalize(rawTrimmed) ?: return
        blockedDao.insert(
            BlockedNumber(
                normalizedNumber = normalized,
                originalNumber = rawTrimmed,
                source = source,
                pattern = null
            )
        )
    }

    suspend fun addToAllowList(rawNumber: String) {
        val normalized = phoneNormalizer.normalize(rawNumber) ?: return
        allowedDao.insert(
            AllowedNumber(
                normalizedNumber = normalized,
                originalNumber = rawNumber
            )
        )
    }

    suspend fun removeFromBlockList(normalizedNumber: String) {
        blockedDao.deleteByNumber(normalizedNumber)
        cachedPatterns = null
    }

    suspend fun removeFromAllowList(normalizedNumber: String) {
        allowedDao.deleteByNumber(normalizedNumber)
    }

    /** Полная очистка всех заблокированных номеров (включая встроенную базу). */
    suspend fun clearAllBlocked() {
        blockedDao.deleteAll()
        cachedPatterns = null
    }

    /** Удалить только PREBUILT-записи (оставить ручные и по сообщениям). */
    suspend fun clearPrebuilt() {
        blockedDao.deletePrebuilt()
        cachedPatterns = null
    }

    suspend fun importPrebuilt(numbers: List<String>) {
        if (numbers.isEmpty()) return
        // Перегоняем нормализацию в один проход без обращений к БД, а затем
        // льём всё батчами по 1000 строк в одной prepared-statement-серии.
        // Старый цикл вида `for (num) { dao.insert(BlockedNumber(...)) }`
        // делал по WAL-fsync на КАЖДЫЙ insert — 2.4M строк = часы. Сейчас
        // получается единая транзакция → секунды.
        val entities = ArrayList<BlockedNumber>(numbers.size)
        for (num in numbers) {
            val normalized = phoneNormalizer.normalize(num) ?: continue
            entities += BlockedNumber(
                normalizedNumber = normalized,
                originalNumber = num,
                source = BlockedNumber.Source.PREBUILT
            )
        }
        if (entities.isEmpty()) return
        val chunkSize = 1000
        var offset = 0
        while (offset < entities.size) {
            val end = (offset + chunkSize).coerceAtMost(entities.size)
            blockedDao.insertAll(entities.subList(offset, end))
            offset = end
        }
    }
}
