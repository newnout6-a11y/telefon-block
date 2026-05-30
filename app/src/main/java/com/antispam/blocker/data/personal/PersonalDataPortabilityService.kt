package com.antispam.blocker.data.personal

import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.room.withTransaction
import com.antispam.blocker.data.db.AppDatabase
import com.antispam.blocker.data.db.entity.AppUsageEvent
import com.antispam.blocker.data.db.entity.CallEvent
import com.antispam.blocker.data.db.entity.FeatureSnapshot
import com.antispam.blocker.data.db.entity.NotificationEvent
import com.antispam.blocker.data.prefs.DeviceModelStore
import com.antispam.blocker.domain.personal.DeviceFeatures
import com.antispam.blocker.domain.personal.DeviceModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException

/**
 * Экспорт / импорт всей персональной телеметрии Device_Model в один JSON-файл
 * (Req 2.5, 2.6).
 *
 * Поток данных:
 *  - **Экспорт**: дампит все строки четырёх таблиц (`call_event`, `notification_event`,
 *    `app_usage_event`, `feature_snapshot`) плюс снимок состояния модели из
 *    [DeviceModelStore] (`weights_json`, `bias`, `installed_at`, `label_count`)
 *    в один JSON-объект и пишет в `Uri`, выбранный пользователем через SAF.
 *  - **Импорт**: читает тот же JSON, валидирует `schema_version` против
 *    [DeviceFeatures.SCHEMA_VERSION] (Req 2.6 — стабильная схема), внутри
 *    Room-транзакции полностью заменяет содержимое четырёх таблиц на загруженные
 *    строки и затем перезаписывает веса/bias/installedAt/labelCount через
 *    [DeviceModelStore]. При несовпадении схемы импорт отвергается с лог-записью —
 *    SGD-шаги и predict никогда не должны применяться к снапшотам чужой схемы.
 *
 * Безопасность приватности (Req 7.1, 7.3): сервис работает только с локальными
 * `Uri` (SAF document picker); никаких сетевых вызовов, никаких аналитических
 * SDK. Файл выбирает сам пользователь.
 *
 * Wipe (Req 2.7) реализуется отдельно — см. задачу 16.1.
 */
class PersonalDataPortabilityService(
    private val context: Context,
    private val database: AppDatabase,
    private val store: DeviceModelStore,
    private val deviceModel: DeviceModel,
) {

    /**
     * Сериализует все четыре таблицы и состояние модели в JSON и пишет в [uri]
     * через SAF (`openOutputStream(uri, "wt")` — truncate-then-write, чтобы при
     * перезаписи существующего файла не оставалось хвоста от старого экспорта).
     *
     * Бросает [IOException], если `Uri` нельзя открыть на запись (отозванное
     * разрешение SAF, удалённый документ и т.п.) — caller'у нужно показать
     * ошибку пользователю, а не молча проглотить.
     */
    suspend fun exportToJson(uri: Uri) = withContext(Dispatchers.IO) {
        val callEvents = database.callEventDao().getAllForExport()
        val notificationEvents = database.notificationEventDao().getAllForExport()
        val appUsageEvents = database.appUsageEventDao().getAllForExport()
        val featureSnapshots = database.featureSnapshotDao().getAllForExport()

        val payload = JSONObject().apply {
            put(KEY_SCHEMA_VERSION, DeviceFeatures.SCHEMA_VERSION)
            put(KEY_EXPORT_TIMESTAMP, System.currentTimeMillis())
            put(KEY_CALL_EVENTS, JSONArray().apply {
                callEvents.forEach { put(callEventToJson(it)) }
            })
            put(KEY_NOTIFICATION_EVENTS, JSONArray().apply {
                notificationEvents.forEach { put(notificationEventToJson(it)) }
            })
            put(KEY_APP_USAGE_EVENTS, JSONArray().apply {
                appUsageEvents.forEach { put(appUsageEventToJson(it)) }
            })
            put(KEY_FEATURE_SNAPSHOTS, JSONArray().apply {
                featureSnapshots.forEach { put(featureSnapshotToJson(it)) }
            })
            // Snapshot текущего состояния модели; `weights_json` может быть null
            // если ещё ни разу не записывали (свежая установка / после reset).
            put(KEY_WEIGHTS_JSON, store.weightsJsonFlow.first() ?: JSONObject.NULL)
            put(KEY_BIAS, store.biasFlow.first().toDouble())
            put(KEY_INSTALLED_AT, store.installedAtFlow.first())
            put(KEY_LABEL_COUNT, store.labelCountFlow.first())
        }

        val resolver = context.contentResolver
        val outputStream = resolver.openOutputStream(uri, "wt")
            ?: throw IOException("Cannot open output stream for $uri")
        outputStream.use { os ->
            os.bufferedWriter(Charsets.UTF_8).use { it.write(payload.toString()) }
        }
    }

    /**
     * Читает JSON из [uri] и заменяет содержимое четырёх таблиц + состояние модели.
     *
     * Контракт:
     * 1. Если `schema_version` в файле не совпадает с [DeviceFeatures.SCHEMA_VERSION] —
     *    отвергаем импорт (Req 2.6, "stable schema"); логируем и возвращаем
     *    `Result.failure`. Никаких изменений в БД и DataStore не делаем.
     * 2. Замена содержимого четырёх таблиц идёт **внутри одной Room-транзакции**
     *    (`AppDatabase.withTransaction { ... }`) — либо все таблицы заменены, либо
     *    ничего, никаких полу-импортированных состояний.
     * 3. Веса/bias/installedAt/labelCount пишутся через [DeviceModelStore] **после**
     *    успешного коммита транзакции. DataStore-операции сами сериализуются на
     *    уровне файла, поэтому отдельный мьютекс не нужен.
     *
     * Возвращает [Result.success] при успехе, [Result.failure] с пояснением — при
     * отказе (mismatch, IO, parse error). Любая ошибка логируется через [Log.w].
     */
    suspend fun importFromJson(uri: Uri): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            val resolver = context.contentResolver
            val raw = resolver.openInputStream(uri)?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }
                ?: throw IOException("Cannot open input stream for $uri")

            val obj = JSONObject(raw)
            val fileSchema = obj.getInt(KEY_SCHEMA_VERSION)
            if (fileSchema != DeviceFeatures.SCHEMA_VERSION) {
                Log.w(
                    TAG,
                    "import rejected: schema_version mismatch (file=$fileSchema, " +
                        "expected=${DeviceFeatures.SCHEMA_VERSION})"
                )
                throw SchemaVersionMismatchException(
                    fileVersion = fileSchema,
                    expectedVersion = DeviceFeatures.SCHEMA_VERSION,
                )
            }

            val callEvents = obj.optJSONArray(KEY_CALL_EVENTS).toCallEvents()
            val notificationEvents = obj.optJSONArray(KEY_NOTIFICATION_EVENTS).toNotificationEvents()
            val appUsageEvents = obj.optJSONArray(KEY_APP_USAGE_EVENTS).toAppUsageEvents()
            val featureSnapshots = obj.optJSONArray(KEY_FEATURE_SNAPSHOTS).toFeatureSnapshots()

            // Атомарная замена содержимого четырёх таблиц.
            //
            // Порядок важен из-за foreign key `feature_snapshot.callEventId →
            // call_event.id` (`ON DELETE SET NULL`): сначала чистим snapshot'ы
            // (FK ни на что не указывает после), затем call_event; вставляем в
            // обратном порядке — call_event перед feature_snapshot, чтобы FK
            // мог разрешиться, если запись приходит с непустым callEventId.
            database.withTransaction {
                database.featureSnapshotDao().deleteAll()
                database.callEventDao().deleteAll()
                database.notificationEventDao().deleteAll()
                database.appUsageEventDao().deleteAll()

                if (callEvents.isNotEmpty()) database.callEventDao().insertAll(callEvents)
                if (notificationEvents.isNotEmpty()) {
                    database.notificationEventDao().insertAll(notificationEvents)
                }
                if (appUsageEvents.isNotEmpty()) {
                    database.appUsageEventDao().insertAll(appUsageEvents)
                }
                if (featureSnapshots.isNotEmpty()) {
                    database.featureSnapshotDao().insertAll(featureSnapshots)
                }
            }

            // Восстанавливаем состояние модели после успешной транзакции.
            val weightsJson = if (obj.isNull(KEY_WEIGHTS_JSON)) null else obj.getString(KEY_WEIGHTS_JSON)
            if (weightsJson != null) {
                store.setWeightsJson(weightsJson)
            }
            store.setBias(obj.getDouble(KEY_BIAS).toFloat())
            store.setInstalledAt(obj.getLong(KEY_INSTALLED_AT))
            // Прямой setter для labelCount отсутствует — атомарно «догоняем» счётчик
            // до требуемого значения через incrementLabelCount(). На импорте это
            // выполняется один раз, поэтому стоимость допустима.
            //
            // Если в файле labelCount меньше текущего — это импорт «старого»
            // снапшота поверх более свежей истории; мы намеренно не уменьшаем
            // счётчик, чтобы не сбрасывать Warm_Up_Window назад.
            val targetLabelCount = obj.getInt(KEY_LABEL_COUNT)
            val currentLabelCount = store.labelCountFlow.first()
            val delta = targetLabelCount - currentLabelCount
            if (delta > 0) {
                repeat(delta) { store.incrementLabelCount() }
            }
        }.onFailure { Log.w(TAG, "importFromJson failed: ${it.message}", it) }
    }

    // ── Wipe (Req 2.7, 7.6) ───────────────────────────────────────────────

    /**
     * Полный сброс on-device состояния Device_Model.
     *
     * Атомарно очищает все четыре таблицы телеметрии в рамках одной Room-транзакции
     * (`call_event`, `notification_event`, `app_usage_event`, `feature_snapshot`),
     * затем восстанавливает веса/bias к defaults через [DeviceModel.resetToDefaults]
     * и очищает DataStore через [DeviceModelStore.reset]. Последний шаг записывает
     * `installedAt = now` и `labelCount = 0`, благодаря чему Warm_Up_Window
     * рестартует отсчёт «с нуля» (Req 5.9).
     *
     * Порядок удаления внутри транзакции тот же, что и в [importFromJson]:
     * сначала `feature_snapshot` (FK на `call_event(id)` `ON DELETE SET NULL`),
     * затем сами `call_event`. Для двух оставшихся таблиц порядок безразличен.
     *
     * Контракт «либо всё, либо ничего» соблюдается на уровне БД (`withTransaction`)
     * и на уровне DataStore (`reset` использует `prefs.clear()` под file-lock'ом).
     * Сбой между ними оставит БД пустой, но веса — старыми; это безопасный
     * fail-state — `predict` продолжит работать, а Warm_Up_Window просто не
     * рестартует. Caller обязан логировать исключение и предложить пользователю
     * повторить операцию.
     */
    suspend fun wipeAll() = withContext(Dispatchers.IO) {
        database.withTransaction {
            database.featureSnapshotDao().deleteAll()
            database.callEventDao().deleteAll()
            database.notificationEventDao().deleteAll()
            database.appUsageEventDao().deleteAll()
        }
        deviceModel.resetToDefaults()
        store.reset()
    }

    // ── JSON encoders ─────────────────────────────────────────────────────

    private fun callEventToJson(e: CallEvent): JSONObject = JSONObject().apply {
        put("id", e.id)
        put("normalizedNumber", e.normalizedNumber ?: JSONObject.NULL)
        put("isHidden", e.isHidden)
        put("direction", e.direction.name)
        put("state", e.state.name)
        put("durationMs", e.durationMs)
        put("startedAt", e.startedAt)
        put("endedAt", e.endedAt ?: JSONObject.NULL)
        put("isContact", e.isContact)
    }

    private fun notificationEventToJson(e: NotificationEvent): JSONObject = JSONObject().apply {
        put("id", e.id)
        put("packageName", e.packageName)
        put("categoryBucket", e.categoryBucket.name)
        put("timestamp", e.timestamp)
    }

    private fun appUsageEventToJson(e: AppUsageEvent): JSONObject = JSONObject().apply {
        put("id", e.id)
        put("packageName", e.packageName)
        put("categoryBucket", e.categoryBucket.name)
        put("foregroundAt", e.foregroundAt)
    }

    private fun featureSnapshotToJson(s: FeatureSnapshot): JSONObject = JSONObject().apply {
        put("id", s.id)
        put("callEventId", s.callEventId ?: JSONObject.NULL)
        put("normalizedNumber", s.normalizedNumber ?: JSONObject.NULL)
        put("timestamp", s.timestamp)
        put("featuresJson", s.featuresJson)
        put("featureSchemaVersion", s.featureSchemaVersion)
        put("weightsHash", s.weightsHash ?: JSONObject.NULL)
        put("deviceProbBlock", s.deviceProbBlock.toDouble())
    }

    // ── JSON decoders ─────────────────────────────────────────────────────

    private fun JSONArray?.toCallEvents(): List<CallEvent> {
        if (this == null) return emptyList()
        return List(length()) { i ->
            val o = getJSONObject(i)
            CallEvent(
                id = o.optLong("id", 0L),
                normalizedNumber = o.optStringOrNull("normalizedNumber"),
                isHidden = o.getBoolean("isHidden"),
                direction = CallEvent.Direction.valueOf(o.getString("direction")),
                state = CallEvent.CallState.valueOf(o.getString("state")),
                durationMs = o.getLong("durationMs"),
                startedAt = o.getLong("startedAt"),
                endedAt = if (o.isNull("endedAt")) null else o.getLong("endedAt"),
                isContact = o.optBoolean("isContact", false),
            )
        }
    }

    private fun JSONArray?.toNotificationEvents(): List<NotificationEvent> {
        if (this == null) return emptyList()
        return List(length()) { i ->
            val o = getJSONObject(i)
            NotificationEvent(
                id = o.optLong("id", 0L),
                packageName = o.getString("packageName"),
                categoryBucket = NotificationEvent.CategoryBucket.valueOf(o.getString("categoryBucket")),
                timestamp = o.getLong("timestamp"),
            )
        }
    }

    private fun JSONArray?.toAppUsageEvents(): List<AppUsageEvent> {
        if (this == null) return emptyList()
        return List(length()) { i ->
            val o = getJSONObject(i)
            AppUsageEvent(
                id = o.optLong("id", 0L),
                packageName = o.getString("packageName"),
                categoryBucket = NotificationEvent.CategoryBucket.valueOf(o.getString("categoryBucket")),
                foregroundAt = o.getLong("foregroundAt"),
            )
        }
    }

    private fun JSONArray?.toFeatureSnapshots(): List<FeatureSnapshot> {
        if (this == null) return emptyList()
        return List(length()) { i ->
            val o = getJSONObject(i)
            FeatureSnapshot(
                id = o.optLong("id", 0L),
                callEventId = if (o.isNull("callEventId")) null else o.getLong("callEventId"),
                normalizedNumber = o.optStringOrNull("normalizedNumber"),
                timestamp = o.getLong("timestamp"),
                featuresJson = o.getString("featuresJson"),
                featureSchemaVersion = o.getInt("featureSchemaVersion"),
                weightsHash = o.optStringOrNull("weightsHash"),
                deviceProbBlock = o.getDouble("deviceProbBlock").toFloat(),
            )
        }
    }

    private fun JSONObject.optStringOrNull(key: String): String? =
        if (isNull(key)) null else optString(key, "").takeIf { has(key) }

    /**
     * Бросается изнутри [importFromJson] при mismatch'е `schema_version`. Caller
     * наблюдает её через `Result.exceptionOrNull()` и может показать пользователю
     * понятную ошибку «файл от другой версии модели».
     */
    class SchemaVersionMismatchException(
        val fileVersion: Int,
        val expectedVersion: Int,
    ) : IllegalStateException(
        "feature schema mismatch: file=$fileVersion, expected=$expectedVersion"
    )

    private companion object {
        const val TAG = "PersonalPortability"

        const val KEY_SCHEMA_VERSION = "schema_version"
        const val KEY_EXPORT_TIMESTAMP = "export_timestamp"
        const val KEY_CALL_EVENTS = "call_events"
        const val KEY_NOTIFICATION_EVENTS = "notification_events"
        const val KEY_APP_USAGE_EVENTS = "app_usage_events"
        const val KEY_FEATURE_SNAPSHOTS = "feature_snapshots"
        const val KEY_WEIGHTS_JSON = "weights_json"
        const val KEY_BIAS = "bias"
        const val KEY_INSTALLED_AT = "installed_at"
        const val KEY_LABEL_COUNT = "label_count"
    }
}
