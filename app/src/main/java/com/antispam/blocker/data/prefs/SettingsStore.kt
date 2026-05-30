package com.antispam.blocker.data.prefs

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.*
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.runBlocking

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "settings")

class SettingsStore(private val context: Context) {

    private val store get() = context.dataStore

    val protectionEnabled: Flow<Boolean> = boolPref("protection_enabled", true)

    val skipCallLogForBlocked: Flow<Boolean> = boolPref("skip_call_log_for_blocked", false)

    // Master-toggle on-device персонального классификатора (Device_Model).
    // По умолчанию включено — Device_Model участвует в fusion'е после warm-up.
    // Когда выключено, SmartSpamDetector пропускает вызов Device_Model и финальный
    // вердикт принимается только по Cloud_Model + rule engine.
    val personalClassifierEnabled: Flow<Boolean> = boolPref("personal_classifier_enabled", true)

    // Kill-switch для on-device char-CNN классификатора App_Category_Model
    // (см. spec app-category-ml-classifier, Req 3.6b).
    // По умолчанию включено — TFLite-путь активен поверх RuleBasedAppCategoryClassifier
    // с confidence-gated fallback. Когда выключено, AppCategoryClassifierFactory
    // возвращает singleton RuleBasedAppCategoryClassifier на всё время жизни процесса.
    val tfliteAppCategoryEnabled: Flow<Boolean> = boolPref("tflite_app_category_enabled", true)

    // По умолчанию включено — раз в 6ч app тянет publicный manifest с GitHub raw.
    // Юзер может выключить в Settings, тогда worker отменяется.
    val dbUpdateEnabled: Flow<Boolean> = boolPref("db_update_enabled", true)
    val dbUpdateUrl: Flow<String> = stringPref("db_update_url", "")

    val lastUpdateAt: Flow<Long> = longPref("last_update_at", 0L)
    val lastUpdateVersion: Flow<String> = stringPref("last_update_version", "")

    suspend fun set(key: String, value: Boolean) {
        store.edit { it[booleanPreferencesKey(key)] = value }
    }

    suspend fun set(key: String, value: String) {
        store.edit { it[stringPreferencesKey(key)] = value }
    }

    /** Включить/выключить on-device персональный классификатор (Device_Model). */
    suspend fun setPersonalClassifierEnabled(enabled: Boolean) {
        store.edit { it[booleanPreferencesKey("personal_classifier_enabled")] = enabled }
    }

    /** Включить/выключить TFLite App_Category_Model (Req 3.6b). */
    suspend fun setTfliteAppCategoryEnabled(enabled: Boolean) {
        store.edit { it[booleanPreferencesKey("tflite_app_category_enabled")] = enabled }
    }

    suspend fun setLastUpdateAt(value: Long) {
        store.edit { it[longPreferencesKey("last_update_at")] = value }
    }

    suspend fun setLastUpdateVersion(value: String) {
        store.edit { it[stringPreferencesKey("last_update_version")] = value }
    }

    /** Синхронный снимок — для use из WorkManager-воркера. */
    fun dbUpdateEnabledSnapshot(): Boolean = runBlocking { dbUpdateEnabled.first() }
    fun dbUpdateUrlSnapshot(): String = runBlocking { dbUpdateUrl.first() }

    /**
     * Синхронный снимок kill-switch'а App_Category_Model TFLite-пути.
     *
     * Используется `AppCategoryClassifierFactory.getOrCreate` (вызывается с горячего
     * пути `PersonalNotificationListenerService.onNotificationPosted`) — блокирующее
     * чтение приемлемо по образу `dbUpdateEnabledSnapshot` (см. Req 3.6b).
     */
    fun tfliteAppCategoryEnabledSnapshot(): Boolean =
        runBlocking { tfliteAppCategoryEnabled.first() }

    // --- helpers ---

    private fun boolPref(key: String, default: Boolean): Flow<Boolean> =
        store.data.map { it[booleanPreferencesKey(key)] ?: default }

    private fun stringPref(key: String, default: String): Flow<String> =
        store.data.map { it[stringPreferencesKey(key)] ?: default }

    private fun longPref(key: String, default: Long): Flow<Long> =
        store.data.map { it[longPreferencesKey(key)] ?: default }
}
