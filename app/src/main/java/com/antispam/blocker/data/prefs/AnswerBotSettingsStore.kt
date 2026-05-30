package com.antispam.blocker.data.prefs

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.answerBotStore by preferencesDataStore("answer_bot_settings")

/**
 * DataStore preferences for the AnswerBot feature.
 *
 * Keys:
 *   - enabled: whether auto-answer is active (default true — always on per spec)
 *   - maxDurationSec: maximum recording duration in seconds (default 45)
 */
class AnswerBotSettingsStore(private val context: Context) {

    private val store: DataStore<Preferences> get() = context.answerBotStore

    val enabledFlow: Flow<Boolean> = store.data.map { prefs ->
        prefs[KEY_ENABLED] ?: true
    }

    val maxDurationSecFlow: Flow<Int> = store.data.map { prefs ->
        prefs[KEY_MAX_DURATION_SEC] ?: 45
    }

    suspend fun setEnabled(enabled: Boolean) {
        store.updateData { it.toMutablePreferences().apply { set(KEY_ENABLED, enabled) } }
    }

    suspend fun setMaxDurationSec(seconds: Int) {
        store.updateData { it.toMutablePreferences().apply { set(KEY_MAX_DURATION_SEC, seconds) } }
    }

    companion object {
        private val KEY_ENABLED = booleanPreferencesKey("answerbot_enabled")
        private val KEY_MAX_DURATION_SEC = intPreferencesKey("answerbot_max_duration_sec")
    }
}
