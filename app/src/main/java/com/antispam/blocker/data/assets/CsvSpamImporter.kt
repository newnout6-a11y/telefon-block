package com.antispam.blocker.data.assets

import android.content.Context
import com.antispam.blocker.data.repository.BlockListRepository
import com.antispam.blocker.data.db.entity.BlockedNumber
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import java.io.InputStreamReader

class CsvSpamImporter(
    private val context: Context,
    private val repository: BlockListRepository
) {

    companion object {
        // Поднимай версию, когда меняешь spam_numbers.csv — база
        // автоматически переимпортируется при следующем запуске приложения.
        // v5: переход на auto-generated bundle из crawler-обогащённого датасета
        //     (~8004 BLOCK номеров + 26 префиксов вместо 61 строки вручную).
        // v6: префиксы из CSV теперь пишутся как regex-блок в БД (а не в
        //     settings.prefixList, который удалён вместе с мёртвым rule-engine).
        const val BUNDLED_DB_VERSION = 6
        private const val PREFS = "spam_import"
        private const val KEY_VERSION = "bundled_version"
        private const val KEY_REMOTE_SHA = "remote_sha256"
        const val ASSET_NAME = "spam_numbers.csv"
    }

    suspend fun importIfFirstRun() = importInternal(force = false, useRemoteIfAvailable = true)

    suspend fun reimport() = importInternal(force = true, useRemoteIfAvailable = true)

    /**
     * Перенакатить базу из конкретного скачанного файла (вызывается из
     * RemoteUpdateWorker после успешной загрузки и SHA256-верификации).
     */
    suspend fun importFromRemote(file: File, sha256: String) {
        withContext(Dispatchers.IO) {
            if (!file.exists() || file.length() == 0L) return@withContext
            doImport(force = true, source = file)
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .edit()
                .putInt(KEY_VERSION, BUNDLED_DB_VERSION)
                .putString(KEY_REMOTE_SHA, sha256)
                .apply()
        }
    }

    /** Сбросить метку импорта (например, после очистки всей базы). */
    fun resetImportFlag() {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit()
            .remove(KEY_VERSION)
            .remove(KEY_REMOTE_SHA)
            .apply()
    }

    private suspend fun importInternal(force: Boolean, useRemoteIfAvailable: Boolean) {
        withContext(Dispatchers.IO) {
            val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            val installedVersion = prefs.getInt(KEY_VERSION, 0)
            if (!force && installedVersion >= BUNDLED_DB_VERSION) return@withContext

            val remoteFile = if (useRemoteIfAvailable) File(context.filesDir, ASSET_NAME) else null
            val source: Any = if (remoteFile != null && remoteFile.exists() && remoteFile.length() > 0) {
                remoteFile
            } else {
                "asset"
            }
            doImport(force = force || installedVersion > 0, source = source)
            prefs.edit().putInt(KEY_VERSION, BUNDLED_DB_VERSION).apply()
        }
    }

    private suspend fun doImport(force: Boolean, source: Any) {
        if (force) {
            repository.clearPrebuilt()
        }

        val exactNumbers = mutableListOf<String>()
        val regexEntries = mutableListOf<Pair<String, String>>()
        val prefixes = mutableListOf<String>()

        try {
            val inputStream: InputStream = when (source) {
                is File -> FileInputStream(source)
                else -> context.assets.open(ASSET_NAME)
            }
            inputStream.use { stream ->
                BufferedReader(InputStreamReader(stream)).use { reader ->
                    var line = reader.readLine()
                    while (line != null) {
                        val trimmed = line.trim()
                        if (trimmed.isNotBlank() && !trimmed.startsWith("#")) {
                            when {
                                trimmed.startsWith("prefix:") -> prefixes.add(trimmed.removePrefix("prefix:"))
                                trimmed.startsWith("regex:") -> {
                                    val pattern = trimmed.removePrefix("regex:")
                                    regexEntries.add(pattern to pattern)
                                }
                                else -> exactNumbers.add(trimmed)
                            }
                        }
                        line = reader.readLine()
                    }
                }
            }
        } catch (_: Exception) {
            return
        }

        if (exactNumbers.isNotEmpty()) {
            repository.importPrebuilt(exactNumbers)
        }

        for ((raw, pattern) in regexEntries) {
            repository.addToBlockList(raw, source = BlockedNumber.Source.PREBUILT, pattern = pattern)
        }

        // Префиксы сохраняем как regex-записи в blocklist —
        // BlockListRepository.isBlocked умеет матчить их по containsMatchIn.
        for (raw in prefixes) {
            val normalized = raw.trim()
            if (normalized.isBlank()) continue
            val pattern = "^" + Regex.escape(normalized)
            repository.addToBlockList(
                rawNumber = normalized,
                source = BlockedNumber.Source.PREBUILT,
                pattern = pattern
            )
        }
    }
}
