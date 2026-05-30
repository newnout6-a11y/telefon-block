package com.antispam.blocker.data.assets

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.util.Log
import java.io.File

/**
 * Read-only доступ к prebuilt-БД заблокированных номеров, лежащей в
 * `assets/prebuilt_blocklist.db`. На первом запуске (или при смене
 * [BUNDLED_VERSION]) копируем ассет в `filesDir/prebuilt_blocklist.db`,
 * затем открываем как обычную SQLite. Никакого Room — Room для подхвата
 * готового файла требует точного `identity_hash` в `room_master_table`,
 * который при ручной сборке файла повторить трудно. SQLiteDatabase
 * работает с любым валидным sqlite-файлом и стоит на горячем пути всего
 * один `EXISTS`-запрос с уникальным индексом — это микросекунды.
 *
 * Раньше этот же словарь грузился через [CsvSpamImporter] из 33-МБ CSV
 * (~2.4M строк) — в Room insert-цикле это 290 строк/сек на S24, то есть
 * часы. Здесь — копия файла (≈ 128 МБ) занимает секунды и сразу даёт
 * проиндексированный словарь.
 *
 * Источник истины — `scripts/build_prebuilt_blocklist_db.py`.
 *
 * Threading: [SQLiteDatabase] потокобезопасна для конкурентных read'ов
 * после открытия с [SQLiteDatabase.OPEN_READONLY]; [contains]
 * и [findPattern]-методы можно дёргать с любых корутиновых dispatcher'ов.
 */
class PrebuiltBlocklistReader(private val context: Context) {

    @Volatile private var db: SQLiteDatabase? = null
    @Volatile private var compiledPatterns: List<Pair<String, Regex>>? = null

    /**
     * Открыть БД, копируя ассет в `filesDir` при необходимости. Безопасно
     * вызывать многократно — повторные вызовы возвращаются мгновенно.
     */
    @Synchronized
    fun ensureOpen(): SQLiteDatabase? {
        db?.let { return it }
        val target = File(context.filesDir, FILE_NAME)
        try {
            if (shouldReinstall(target)) {
                copyFromAsset(target)
            }
            val opened = SQLiteDatabase.openDatabase(
                target.absolutePath,
                /* factory = */ null,
                SQLiteDatabase.OPEN_READONLY,
            )
            // Проверка целостности: сверяем реальный row_count с meta.row_count.
            // Если partial-copy на low-storage — число строк не совпадёт.
            try {
                opened.rawQuery("SELECT value FROM meta WHERE key = 'row_count' LIMIT 1", null)
                    .use { cursor ->
                        if (cursor.moveToFirst()) {
                            val expected = cursor.getString(0)?.toIntOrNull() ?: 0
                            val actual = opened.rawQuery(
                                "SELECT COUNT(*) FROM prebuilt_blocked", null
                            ).use { c -> if (c.moveToFirst()) c.getInt(0) else 0 }
                            if (actual != expected) {
                                Log.w(TAG, "row_count mismatch: expected=$expected actual=$actual — possible corrupt copy")
                            }
                        } else {
                            Log.w(TAG, "meta.row_count not found in prebuilt DB")
                        }
                    }
            } catch (t: Throwable) {
                Log.w(TAG, "row_count check failed", t)
            }
            db = opened
            return opened
        } catch (t: Throwable) {
            Log.w(TAG, "Не смогли открыть prebuilt blocklist", t)
            return null
        }
    }

    /**
     * `true`, если [normalizedNumber] лежит точным совпадением в prebuilt
     * списке. Не матчит регексы/префиксы — для них есть [findPatternMatch].
     */
    fun contains(normalizedNumber: String): Boolean {
        if (normalizedNumber.isBlank()) return false
        val database = ensureOpen() ?: return false
        return try {
            database.rawQuery(
                "SELECT 1 FROM prebuilt_blocked WHERE normalizedNumber = ? LIMIT 1",
                arrayOf(normalizedNumber),
            ).use { cursor -> cursor.moveToFirst() }
        } catch (t: Throwable) {
            Log.w(TAG, "contains() failed for $normalizedNumber", t)
            false
        }
    }

    /**
     * Возвращает первый regex/префиксный паттерн, совпавший с [normalizedNumber],
     * или `null`. Patterns кэшируются после первого чтения (их ≤ 30 на
     * текущий ассет, так что это бюджетно).
     */
    fun findPatternMatch(normalizedNumber: String): String? {
        if (normalizedNumber.isBlank()) return null
        val patterns = patternsOrLoad() ?: return null
        for ((raw, regex) in patterns) {
            if (regex.containsMatchIn(normalizedNumber)) return raw
        }
        return null
    }

    /** Пара (originalPattern, compiledRegex) — отлаженный список или null при пустой/сломанной БД. */
    private fun patternsOrLoad(): List<Pair<String, Regex>>? {
        compiledPatterns?.let { return it }
        val database = ensureOpen() ?: return null
        return try {
            val list = mutableListOf<Pair<String, Regex>>()
            database.rawQuery(
                "SELECT pattern FROM prebuilt_blocked WHERE pattern IS NOT NULL",
                null,
            ).use { cursor ->
                while (cursor.moveToNext()) {
                    val pat = cursor.getString(0) ?: continue
                    try {
                        list += pat to Regex(pat)
                    } catch (_: Exception) {
                        // битый regex в ассете игнорируем — лучше пропустить
                        // одну запись, чем уронить весь матчинг.
                    }
                }
            }
            compiledPatterns = list
            list
        } catch (t: Throwable) {
            Log.w(TAG, "Не смогли загрузить patterns", t)
            null
        }
    }

    /**
     * Возвращает true, если в `filesDir` ещё нет файла, или если у
     * существующего файла другая версия по сравнению с [BUNDLED_VERSION].
     */
    private fun shouldReinstall(target: File): Boolean {
        if (!target.exists() || target.length() == 0L) return true
        return try {
            SQLiteDatabase.openDatabase(
                target.absolutePath,
                null,
                SQLiteDatabase.OPEN_READONLY,
            ).use { existing ->
                existing.rawQuery("SELECT value FROM meta WHERE key = 'version' LIMIT 1", null)
                    .use { cursor ->
                        if (!cursor.moveToFirst()) return@use true
                        val installed = cursor.getString(0)?.toIntOrNull() ?: -1
                        installed != BUNDLED_VERSION
                    }
            }
        } catch (_: Throwable) {
            // Если файл битый — переустанавливаем.
            true
        }
    }

    private fun copyFromAsset(target: File) {
        val tmp = File(target.parentFile, "${target.name}.tmp")
        context.assets.open(FILE_NAME).use { input ->
            tmp.outputStream().use { output ->
                input.copyTo(output, bufferSize = 64 * 1024)
            }
        }
        if (target.exists()) target.delete()
        if (!tmp.renameTo(target)) {
            // На некоторых Android-сборках renameTo через границу
            // возвращает false — fallback на покопирно-удаление.
            tmp.copyTo(target, overwrite = true)
            tmp.delete()
        }
        compiledPatterns = null
        // Если БД уже была открыта старой версии — закроем, чтобы при
        // следующем `ensureOpen` подхватить новый файл.
        db?.close()
        db = null
        Log.i(TAG, "Установили prebuilt blocklist v$BUNDLED_VERSION (${target.length()} bytes)")
    }

    /** Закрыть открытую БД. Обычно не нужно вызывать — процесс держит её на всё время. */
    @Synchronized
    fun close() {
        db?.close()
        db = null
        compiledPatterns = null
    }

    companion object {
        private const val TAG = "PrebuiltBlocklist"
        private const val FILE_NAME = "prebuilt_blocklist.db"

        /**
         * Должен совпадать с `BUNDLED_VERSION` в
         * `scripts/build_prebuilt_blocklist_db.py`. Когда поднимаешь
         * там — поднимай и здесь, иначе клиенты не пересоздадут локальную копию.
         */
        const val BUNDLED_VERSION = 1
    }
}

/**
 * Маленький helper-extension, чтобы [SQLiteDatabase.use] компилировался
 * (стандартный `use` ожидает [java.io.Closeable], а SQLiteDatabase реализует
 * `Closeable` только с API 16+, что у нас выполнено — но Kotlin не всегда
 * это видит без явной подсказки).
 */
private inline fun <R> SQLiteDatabase.use(block: (SQLiteDatabase) -> R): R {
    try {
        return block(this)
    } finally {
        try { close() } catch (_: Throwable) {}
    }
}
