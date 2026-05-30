package com.antispam.blocker.data.prefs

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/**
 * Хранилище состояния персонального on-device-классификатора (Device_Model):
 *
 * - **Веса и bias** логистической регрессии — JSON-строкой и Float'ом соответственно.
 *   Веса keyed by `device_weights_v{schema}` — при bump'е [SCHEMA_VERSION] (см. дизайн
 *   "Backward compatibility веса") ключ меняется, старые веса остаются «висеть» как
 *   орфаны и игнорируются (фактический reset к defaults делает `DeviceModel.resetToDefaults()`).
 *
 * - **Telemetry sources toggles** (Req 7.4) — четыре независимых per-source флага.
 *   Defaults: всё `true`, кроме `source_notifications_enabled = false` — Notification
 *   Listener Access требует отдельного user-action в системных настройках, поэтому
 *   стартовать «выключенным» безопаснее, чем притворяться, что у нас уже есть доступ.
 *
 * - **Warm_Up_Window state** — `device_installed_at` и `device_label_count` (Req 5.9).
 *   На свежей установке либо после [reset] оба обнуляются, окно стартует заново.
 *
 * - **Feature schema version** — мониторится снаружи (`DeviceModel`/`OnlineTrainer`),
 *   чтобы при mismatch'е со снэпшотом не применять SGD-шаг к чужому вектору
 *   (см. дизайн §"OnlineTrainer.applyExplicitLabel").
 *
 * Concurrency: read-modify-write идёт через `DataStore.edit`, который сам сериализует
 * операции по файлу. Доп. mutex'ы здесь не нужны — параллельный feedback из UI и
 * SGD-шаг из фонового worker'а корректно сериализуются на уровне DataStore.
 */
private val Context.deviceModelDataStore: DataStore<Preferences> by preferencesDataStore(
    name = DEVICE_MODEL_DATASTORE_NAME
)

private const val DEVICE_MODEL_DATASTORE_NAME = "device_model"

class DeviceModelStore(
    private val context: Context,
    private val nowMillis: () -> Long = System::currentTimeMillis,
) {

    private val store: DataStore<Preferences> get() = context.deviceModelDataStore

    // ── Weights & bias ────────────────────────────────────────────────────

    /**
     * Веса как JSON-объект `{feature_name: weight}`. `null`, если ещё ни разу не записывали
     * (свежая установка / после [reset]) — caller должен инициализировать через
     * `DefaultWeightsLoader` + [setWeightsJson].
     */
    val weightsJsonFlow: Flow<String?> = store.data.map { it[weightsKey(SCHEMA_VERSION)] }

    suspend fun setWeightsJson(json: String) {
        store.edit { it[weightsKey(SCHEMA_VERSION)] = json }
    }

    val biasFlow: Flow<Float> = store.data.map { it[BIAS_KEY] ?: DEFAULT_BIAS }

    suspend fun setBias(b: Float) {
        store.edit { it[BIAS_KEY] = b }
    }

    // ── Warm_Up_Window state ──────────────────────────────────────────────

    /** Timestamp первой инициализации модели (для `now - installedAt >= 14d` ветки Warm_Up_Window). */
    val installedAtFlow: Flow<Long> = store.data.map { it[INSTALLED_AT_KEY] ?: 0L }

    suspend fun setInstalledAt(t: Long) {
        store.edit { it[INSTALLED_AT_KEY] = t }
    }

    /** Total Implicit + Explicit labels — для `labelCount >= 30` ветки Warm_Up_Window. */
    val labelCountFlow: Flow<Int> = store.data.map { it[LABEL_COUNT_KEY] ?: 0 }

    suspend fun incrementLabelCount() {
        store.edit { prefs ->
            prefs[LABEL_COUNT_KEY] = (prefs[LABEL_COUNT_KEY] ?: 0) + 1
        }
    }

    // ── Feature schema version ────────────────────────────────────────────

    /**
     * Версия схемы фич, под которую записаны текущие веса. `0`, если ещё не инициализировано.
     * При расхождении с `DeviceFeatures.SCHEMA_VERSION` `DeviceModel` обязан сделать
     * `resetToDefaults()` (см. design §"Backward compatibility веса").
     */
    val featureSchemaFlow: Flow<Int> = store.data.map { it[FEATURE_SCHEMA_KEY] ?: 0 }

    suspend fun setFeatureSchema(v: Int) {
        store.edit { it[FEATURE_SCHEMA_KEY] = v }
    }

    // ── Per-source toggles (Req 7.4) ──────────────────────────────────────

    val sourceCallLogEnabledFlow: Flow<Boolean> = store.data.map {
        it[SOURCE_CALL_LOG_KEY] ?: DEFAULT_SOURCE_CALL_LOG_ENABLED
    }

    suspend fun setSourceCallLogEnabled(enabled: Boolean) {
        store.edit { it[SOURCE_CALL_LOG_KEY] = enabled }
    }

    val sourceContactsEnabledFlow: Flow<Boolean> = store.data.map {
        it[SOURCE_CONTACTS_KEY] ?: DEFAULT_SOURCE_CONTACTS_ENABLED
    }

    suspend fun setSourceContactsEnabled(enabled: Boolean) {
        store.edit { it[SOURCE_CONTACTS_KEY] = enabled }
    }

    val sourceAppUsageEnabledFlow: Flow<Boolean> = store.data.map {
        it[SOURCE_APP_USAGE_KEY] ?: DEFAULT_SOURCE_APP_USAGE_ENABLED
    }

    suspend fun setSourceAppUsageEnabled(enabled: Boolean) {
        store.edit { it[SOURCE_APP_USAGE_KEY] = enabled }
    }

    val sourceNotificationsEnabledFlow: Flow<Boolean> = store.data.map {
        it[SOURCE_NOTIFICATIONS_KEY] ?: DEFAULT_SOURCE_NOTIFICATIONS_ENABLED
    }

    suspend fun setSourceNotificationsEnabled(enabled: Boolean) {
        store.edit { it[SOURCE_NOTIFICATIONS_KEY] = enabled }
    }

    // ── End-of-call recorder cursor ───────────────────────────────────────

    /**
     * High-water mark of `CallLog.Calls.DATE` already processed by
     * `CallEventRecorder`. Default `0L` — on a fresh install (or after
     * [reset]) the recorder treats the entire CallLog as already-seen
     * implicitly via its initial catch-up read, then advances the cursor as
     * new rows arrive.
     *
     * Note: [reset] (Wipe) clears this key alongside everything else, which
     * is the desired behaviour — after Wipe we want a clean slate, not a
     * replay of years of CallLog history.
     */
    val lastSeenCallLogTimestampFlow: Flow<Long> = store.data.map {
        it[LAST_SEEN_CALL_LOG_KEY] ?: 0L
    }

    suspend fun setLastSeenCallLogTimestamp(t: Long) {
        store.edit { it[LAST_SEEN_CALL_LOG_KEY] = t }
    }

    // ── Reset (Req 2.7, 7.6) ──────────────────────────────────────────────

    /**
     * Полная очистка хранилища Device_Model: веса, bias, label count, feature schema,
     * все per-source toggles. После очистки сразу пишется новый `installed_at = now()`,
     * чтобы Warm_Up_Window рестартовал отсчёт корректно (Req 5.9: «с момента установки
     * приложения или последнего Wipe прошло ≥ 14 календарных дней»).
     *
     * Toggle'ы возвращаются к дефолтам неявно — их `Flow`-getters читают [SOURCE_*_KEY] и
     * фолбэчатся на `DEFAULT_SOURCE_*_ENABLED`, когда ключи отсутствуют.
     */
    suspend fun reset() {
        val now = nowMillis()
        store.edit { prefs ->
            prefs.clear()
            prefs[INSTALLED_AT_KEY] = now
        }
    }

    companion object {
        /**
         * Версия схемы фич, встраиваемая в ключ весов. Должна совпадать с
         * `DeviceFeatures.SCHEMA_VERSION` (см. `domain/personal/DeviceFeatures.kt`).
         * При изменении набора/порядка фич — бампим оба константа синхронно, и старые
         * записи под ключом `device_weights_v{old}` автоматически становятся орфанами.
         */
        const val SCHEMA_VERSION: Int = 1

        const val DEFAULT_BIAS: Float = 0f

        const val DEFAULT_SOURCE_CALL_LOG_ENABLED: Boolean = true
        const val DEFAULT_SOURCE_CONTACTS_ENABLED: Boolean = true
        const val DEFAULT_SOURCE_APP_USAGE_ENABLED: Boolean = true

        /**
         * Notification Listener Access требует отдельного user-action в системных
         * настройках; стартуем выключенным, чтобы не делать вид, что доступ уже есть.
         */
        const val DEFAULT_SOURCE_NOTIFICATIONS_ENABLED: Boolean = false

        // ── Preference keys ──
        private fun weightsKey(schema: Int) = stringPreferencesKey("device_weights_v$schema")

        private val BIAS_KEY = floatPreferencesKey("device_bias")
        private val INSTALLED_AT_KEY = longPreferencesKey("device_installed_at")
        private val LABEL_COUNT_KEY = intPreferencesKey("device_label_count")
        private val FEATURE_SCHEMA_KEY = intPreferencesKey("device_feature_schema")

        private val SOURCE_CALL_LOG_KEY = booleanPreferencesKey("source_call_log_enabled")
        private val SOURCE_CONTACTS_KEY = booleanPreferencesKey("source_contacts_enabled")
        private val SOURCE_APP_USAGE_KEY = booleanPreferencesKey("source_app_usage_enabled")
        private val SOURCE_NOTIFICATIONS_KEY = booleanPreferencesKey("source_notifications_enabled")

        private val LAST_SEEN_CALL_LOG_KEY = longPreferencesKey("last_seen_call_log_ts")
    }
}
