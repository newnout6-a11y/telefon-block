package com.antispam.blocker.data.prefs

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/**
 * Настройки определения звонящего (CallerID).
 *
 * Содержит:
 *  - [twoGisEnabledFlow] — включён ли 2GIS online-lookup (по умолчанию false, opt-in).
 *  - [apiKeyFlow]        — API-ключ 2GIS (хранится в DataStore, не в коде).
 *
 * DataStore name = "caller_lookup_settings", отдельный от "device_model".
 */
private val Context.callerLookupDataStore: DataStore<Preferences> by preferencesDataStore(
    name = "caller_lookup_settings"
)

class CallerLookupSettingsStore(context: Context) {

    private val store: DataStore<Preferences> = context.applicationContext.callerLookupDataStore

    private val KEY_TWO_GIS_ENABLED = booleanPreferencesKey("two_gis_enabled")
    private val KEY_TWO_GIS_API_KEY = stringPreferencesKey("two_gis_api_key")

    /** true = разрешить online-lookup через 2GIS (по умолчанию false — opt-in). */
    val twoGisEnabledFlow: Flow<Boolean> = store.data.map { it[KEY_TWO_GIS_ENABLED] ?: false }

    /**
     * API-ключ, введённый пользователем в Settings.
     * null или blank = online-lookup отключён даже если [twoGisEnabledFlow] = true.
     */
    val apiKeyFlow: Flow<String?> = store.data.map {
        it[KEY_TWO_GIS_API_KEY]?.takeIf { k -> k.isNotBlank() }
    }

    suspend fun setTwoGisEnabled(enabled: Boolean) {
        store.edit { it[KEY_TWO_GIS_ENABLED] = enabled }
    }

    suspend fun setApiKey(key: String) {
        store.edit { it[KEY_TWO_GIS_API_KEY] = key.trim() }
    }
}
