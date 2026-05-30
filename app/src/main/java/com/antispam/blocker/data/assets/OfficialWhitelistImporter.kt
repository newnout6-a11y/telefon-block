package com.antispam.blocker.data.assets

import android.content.Context
import com.antispam.blocker.data.repository.BlockListRepository
import com.antispam.blocker.util.PhoneNormalizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Импортер официальных РФ whitelist-номеров (банки, операторы, экстренные службы).
 * Добавляет номера в allowlist, чтобы модель никогда не блокировала их.
 */
class OfficialWhitelistImporter(
    private val context: Context,
    private val repository: BlockListRepository
) {

    companion object {
        private const val ASSET_FILE = "official_ru_whitelist.csv"
        private const val PREFS = "official_whitelist"
        private const val KEY_IMPORTED = "imported_version"
        const val CURRENT_VERSION = 1
    }

    suspend fun importIfFirstRun() = importInternal(force = false)

    suspend fun reimport() = importInternal(force = true)

    private suspend fun importInternal(force: Boolean) {
        withContext(Dispatchers.IO) {
            val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            val installedVersion = prefs.getInt(KEY_IMPORTED, 0)
            if (!force && installedVersion >= CURRENT_VERSION) return@withContext

            try {
                val inputStream = context.assets.open(ASSET_FILE)
                BufferedReader(InputStreamReader(inputStream)).use { reader ->
                    var line = reader.readLine()
                    var count = 0
                    while (line != null) {
                        val trimmed = line.trim()
                        if (trimmed.isNotBlank() && !trimmed.startsWith("#")) {
                            // Format: normalized_number,name,category
                            val parts = trimmed.split(",")
                            val number = parts[0].trim()
                            val normalized = PhoneNormalizer.normalize(number)
                            if (normalized != null) {
                                try {
                                    repository.addToAllowList(normalized)
                                    count++
                                } catch (_: Exception) {
                                    // Already in allowlist — ok
                                }
                            }
                        }
                        line = reader.readLine()
                    }
                    prefs.edit().putInt(KEY_IMPORTED, CURRENT_VERSION).apply()
                    android.util.Log.d("WhitelistImporter", "Imported $count official whitelist numbers (v$CURRENT_VERSION)")
                }
            } catch (e: Exception) {
                android.util.Log.w("WhitelistImporter", "Failed to import whitelist", e)
            }
        }
    }
}
