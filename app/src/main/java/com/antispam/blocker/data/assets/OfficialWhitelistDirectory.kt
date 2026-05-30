package com.antispam.blocker.data.assets

import android.content.Context
import com.antispam.blocker.util.PhoneNormalizer
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * In-memory справочник официального РФ whitelist'а из
 * `assets/official_ru_whitelist.csv`. Используется UI-слоем, чтобы
 * показывать «банки / операторы / экстренные службы» отдельной секцией
 * и не давать пользователю их случайно удалить — они импортируются
 * автоматически и должны держаться вечно.
 *
 * Хранение источника: парсим CSV один раз при первом обращении,
 * результат кэшируется на уровне application'а. Файл маленький (≈ 2 КБ),
 * так что блокирующего IO здесь нет.
 */
object OfficialWhitelistDirectory {

    private data class Entry(val normalized: String, val name: String, val category: String)

    @Volatile private var cache: Map<String, Entry>? = null

    private fun loadIfNeeded(context: Context): Map<String, Entry> {
        cache?.let { return it }
        val map = HashMap<String, Entry>()
        try {
            context.assets.open("official_ru_whitelist.csv").use { stream ->
                BufferedReader(InputStreamReader(stream)).useLines { lines ->
                    for (line in lines) {
                        val trimmed = line.trim()
                        if (trimmed.isBlank() || trimmed.startsWith("#")) continue
                        val parts = trimmed.split(",")
                        if (parts.isEmpty()) continue
                        val number = parts.getOrNull(0)?.trim().orEmpty()
                        val name = parts.getOrNull(1)?.trim().orEmpty()
                        val category = parts.getOrNull(2)?.trim().orEmpty()
                        val normalized = PhoneNormalizer.normalize(number) ?: continue
                        map[normalized] = Entry(normalized, name, category)
                    }
                }
            }
        } catch (_: Throwable) {
            // Если ассет битый/отсутствует — UI просто покажет всё как «Ваши».
        }
        cache = map
        return map
    }

    /** Возвращает имя организации (например, «Сбербанк») или null если номер не из whitelist'а. */
    fun nameFor(context: Context, normalizedNumber: String): String? =
        loadIfNeeded(context)[normalizedNumber]?.name?.takeIf { it.isNotBlank() }

    /** Возвращает категорию ("bank" / "support" / "emergency" / "government") или null. */
    fun categoryFor(context: Context, normalizedNumber: String): String? =
        loadIfNeeded(context)[normalizedNumber]?.category?.takeIf { it.isNotBlank() }

    /** Set всех номеров из официального whitelist'а — для O(1)-проверки в UI. */
    fun officialSet(context: Context): Set<String> = loadIfNeeded(context).keys
}
