package com.antispam.blocker

import android.app.Application
import androidx.work.Constraints
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import com.antispam.blocker.data.assets.PrebuiltBlocklistReader
import com.antispam.blocker.data.db.AppDatabase
import com.antispam.blocker.data.personal.PersonalDataPortabilityService
import com.antispam.blocker.data.prefs.DeviceModelStore
import com.antispam.blocker.data.prefs.ProfileVectorStore
import com.antispam.blocker.data.prefs.SettingsStore
import com.antispam.blocker.data.repository.BlockListRepository
import com.antispam.blocker.data.repository.CallEventRecorder
import com.antispam.blocker.data.prefs.CallerLookupSettingsStore
import com.antispam.blocker.data.prefs.AnswerBotSettingsStore
import com.antispam.blocker.data.worker.RemoteUpdateWorker
import com.antispam.blocker.domain.lookup.CallerLookupRepository
import com.antispam.blocker.domain.lookup.CallerLookupWorker
import com.antispam.blocker.domain.lookup.OfflineCallerLookup
import com.antispam.blocker.domain.lookup.TwoGisCallerLookup
import com.antispam.blocker.domain.lookup.RussianPhoneLookup
import com.antispam.blocker.domain.answerbot.VoskRecognizer
import com.antispam.blocker.domain.model.ModelCard
import com.antispam.blocker.domain.personal.DefaultWeightsLoader
import com.antispam.blocker.domain.personal.DeviceModel
import com.antispam.blocker.domain.personal.OnlineTrainer
import com.antispam.blocker.domain.personal.TelemetryRetentionWorker
import com.antispam.blocker.domain.scoring.InstalledAppScanner
import com.antispam.blocker.domain.scoring.UserProfileVector
import com.antispam.blocker.util.PhoneNormalizer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.util.concurrent.TimeUnit

class SpamBlockerApp : Application() {

    private val appScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    val database: AppDatabase by lazy { AppDatabase.getInstance(this) }
    val settingsStore: SettingsStore by lazy { SettingsStore(this) }
    val profileVectorStore: ProfileVectorStore by lazy { ProfileVectorStore(this) }
    val modelCard: ModelCard? get() = ModelCard.load(this)
    val modelVersion: String get() = modelCard?.version ?: "local-rule-fallback"

    /**
     * Read-only sqlite-словарь известного спам-наполнения (~2.4 млн номеров +
     * regex/префиксы). Лежит как asset, копируется в filesDir один раз при
     * первом запуске. На горячем пути — `EXISTS` с уникальным индексом,
     * микросекунды. Заменяет старый CsvSpamImporter, который заливал тот
     * же словарь по одному номеру через Room (часы вместо секунд).
     */
    val prebuiltBlocklistReader: PrebuiltBlocklistReader by lazy {
        PrebuiltBlocklistReader(this)
    }

    /**
     * Persistent state for the on-device personal classifier (weights, bias,
     * label count, per-source toggles, Warm_Up_Window state).
     */
    val deviceModelStore: DeviceModelStore by lazy { DeviceModelStore(this) }

    /**
     * Loader for the shipped default weights asset
     * (`assets/device_model_default_weights.json`). Falls back to
     * [DefaultWeightsLoader.FALLBACK] if the asset is missing or malformed.
     */
    val defaultWeightsLoader: DefaultWeightsLoader by lazy {
        DefaultWeightsLoader(assets)
    }

    /**
     * Pure-Kotlin logistic-regression Device_Model. All read-modify-write is
     * serialized through the model's internal `Mutex`; concurrent feedback
     * from `SpamActionReceiver`, the explainability screen, and the
     * deferred MISSED-recheck worker is safe.
     */
    val deviceModel: DeviceModel by lazy {
        DeviceModel(deviceModelStore, defaultWeightsLoader)
    }

    /**
     * Funnel for implicit/explicit labels into [DeviceModel.sgdStep].
     * Resolved at run time by [com.antispam.blocker.domain.personal.OnlineTrainerLocator]
     * so [com.antispam.blocker.domain.personal.MissedNoCallbackRecheckWorker]
     * can reach it without a forward reference at compile time.
     */
    val onlineTrainer: OnlineTrainer by lazy {
        OnlineTrainer(
            deviceModel = deviceModel,
            featureSnapshotDao = database.featureSnapshotDao(),
            store = deviceModelStore,
        )
    }

    /**
     * Экспорт/импорт всей on-device телеметрии Device_Model в JSON и полный
     * Wipe (Req 2.5, 2.6, 2.7). UI Settings-экрана достаёт сервис отсюда; у нас
     * нет DI-фреймворка, поэтому держим единый lazy-инстанс на уровне Application.
     */
    val personalDataPortabilityService: PersonalDataPortabilityService by lazy {
        PersonalDataPortabilityService(
            context = this,
            database = database,
            store = deviceModelStore,
            deviceModel = deviceModel,
        )
    }

    /** Настройки определения звонящего (2GIS toggle + API key). */
    val callerLookupSettingsStore: CallerLookupSettingsStore by lazy {
        CallerLookupSettingsStore(this)
    }

    /** Настройки автоответчика (enabled + max duration). */
    val answerBotSettingsStore: AnswerBotSettingsStore by lazy {
        AnswerBotSettingsStore(this)
    }

    /**
     * Vosk ASR recognizer для AnswerBot. Модель загружается из filesDir/answerbot/,
     * куда копируется из assets/answerbot/ при первом запуске.
     *
     * НЕ lazy — проверяет наличие модели при КАЖДОМ обращении, чтобы
     * избежать race condition: если первый доступ случился до окончания
     * копирования из assets, null не кэшируется навсегда.
     */
    val voskRecognizer: VoskRecognizer?
        get() {
            val modelDir = java.io.File(filesDir, "answerbot/vosk-model-small-ru-0.22")
            val sentinel = java.io.File(modelDir, ".copy_done")
            return if (sentinel.exists()) {
                VoskRecognizer(modelDir)
            } else {
                null
            }
        }

    /**
     * Репозиторий определения звонящего: кэш (offline libphonenumber) +
     * опциональный 2GIS online-lookup.
     */
    val callerLookupRepository: CallerLookupRepository by lazy {
        CallerLookupRepository(
            dao           = database.callerLookupDao(),
            offlineLookup = OfflineCallerLookup(),
            twoGisLookup  = TwoGisCallerLookup(),
            settingsStore = callerLookupSettingsStore,
        )
    }

    /**
     * End-of-call recorder: watches `CallLog.Calls`, writes [com.antispam.blocker.data.db.entity.CallEvent]
     * rows for every new entry, links the matching pre-existing
     * [com.antispam.blocker.data.db.entity.FeatureSnapshot] from the
     * screening service, and dispatches the implicit label through
     * [onlineTrainer]. Closes the implicit branch of Req 4.1–4.4 — without
     * it `previously_rejected`, `prev_missed_no_callback_24h`,
     * `same_prefix_call_count_7d_norm` and friends would be stuck at 0.
     * Initialised in [onCreate] after [RemoteUpdateWorker.schedule].
     */
    val callEventRecorder: CallEventRecorder by lazy {
        CallEventRecorder(
            context = this,
            callEventDao = database.callEventDao(),
            featureSnapshotDao = database.featureSnapshotDao(),
            callRecordDao = database.callRecordDao(),
            deviceModelStore = deviceModelStore,
            onlineTrainer = onlineTrainer,
        )
    }

    private var _profileVector: UserProfileVector = UserProfileVector()
    val profileVector: UserProfileVector get() = _profileVector

    override fun onCreate() {
        super.onCreate()
        instance = this

        // ContactsCache: до этой правки `init(...)` дёргался только из
        // SpamCallScreeningService.onCreate(), а тот запускается лишь на
        // первый входящий звонок. До звонка кэш = null, и UI Privacy
        // («Прозрачность данных») честно показывал «кэш не прогрет», даже
        // если разрешение READ_CONTACTS уже выдано. Прогреваем
        // в Application.onCreate, чтобы сразу после старта приложения
        // кэш был готов к скринингу и наблюдаемости.
        com.antispam.blocker.data.cache.ContactsCache.init(this)

        appScope.launch {
            // Прогреваем prebuilt-БД заранее: распаковка 128-МБ ассета в
            // filesDir идёт пару секунд на первом старте. Делаем это в IO
            // ещё до того, как пользователь нажмёт «Журнал звонков», чтобы
            // первый incoming-call не блокировал screening-сервис.
            prebuiltBlocklistReader.ensureOpen()

            // Распаковываем Vosk модель из assets/answerbot/ в filesDir/answerbot/
            // при первом запуске (или если модель по какой-то причине отсутствует).
            // Sentinel-файл .copy_done защищает от частичной копии при kill процесса
            // и служит сигналом для voskRecognizer, что модель готова.
            val modelAssetsPath = "answerbot/vosk-model-small-ru-0.22"
            val modelFilesDir = java.io.File(filesDir, modelAssetsPath)
            val sentinel = java.io.File(modelFilesDir, ".copy_done")
            if (!sentinel.exists()) {
                try {
                    modelFilesDir.deleteRecursively()
                    sentinel.delete()
                    copyAssetDir(modelAssetsPath, modelFilesDir)
                    sentinel.createNewFile()
                    android.util.Log.i("SpamBlockerApp", "Vosk model copied to $modelFilesDir")
                } catch (t: Throwable) {
                    android.util.Log.w("SpamBlockerApp", "Vosk model copy failed", t)
                }
            }

            // Загружаем реестр РКН (336 KB) в память — точные регионы
            // для мобильных +7 9XX вместо просто "Россия".
            RussianPhoneLookup.load(applicationContext)

            // Однократная миграция: пользователи, у которых уже накопились
            // PREBUILT-записи в Room (старый CsvSpamImporter заливал их
            // часами по одной), получат пустую таблицу `blocked_numbers`
            // от наших prebuilt-записей — теперь словарь живёт в sqlite-asset.
            // Ручные/feedback-записи (source != PREBUILT) сохраняются.
            val prefs = getSharedPreferences("prebuilt_migration", MODE_PRIVATE)
            if (!prefs.getBoolean("migrated_to_sqlite_asset", false)) {
                try {
                    val repo = BlockListRepository(
                        database.blockedNumberDao(),
                        database.allowedNumberDao(),
                        PhoneNormalizer,
                        prebuiltReader = prebuiltBlocklistReader,
                    )
                    repo.clearPrebuilt()
                    prefs.edit().putBoolean("migrated_to_sqlite_asset", true).apply()
                } catch (t: Throwable) {
                    android.util.Log.w("SpamBlockerApp", "Не смогли вычистить старые PREBUILT", t)
                }
            }

            // Load profile vector from DataStore, enrich with installed apps
            val stored = profileVectorStore.getVector()
            val enriched = InstalledAppScanner(this@SpamBlockerApp).enrichProfile(stored)
            _profileVector = enriched

            // WarmUpGate fix: installedAtFlow по умолчанию = 0L, а с этим
            // дефолтом (now - 0) >= 14d всегда true → warm-up считается
            // завершённым уже на первом запуске, Device-модель сразу
            // голосует на дефолтных весах (σ(bias=-0.5) ≈ 0.378), и
            // FusionDecider попадает в single_block_downgrade_warn —
            // любой Cloud BLOCK понижается до WARN на свежей установке.
            // Сидим installedAt = now() только если он ещё не был выставлен,
            // чтобы не сбрасывать счётчик 14 дней у юзеров после Wipe.
            if (deviceModelStore.installedAtFlow.first() == 0L) {
                deviceModelStore.setInstalledAt(System.currentTimeMillis())
                android.util.Log.i(
                    "SpamBlockerApp",
                    "WarmUp: seeded installedAt for fresh install",
                )
            }
        }

        // Раз в 6ч тянем обновлённый manifest.json + spam_numbers.csv + prefix_risk.json
        // + spam_model.tflite с публичной статики (по умолчанию — GitHub raw нашего репо).
        // Фактический сетевой запрос worker делает только если settings.dbUpdateEnabled = true.
        RemoteUpdateWorker.schedule(this)

        // End-of-call recorder: подключается к CallLog ContentObserver и
        // конвертирует системные записи звонков в CallEvent + диспатчит
        // implicit-label в OnlineTrainer (Req 4.1–4.4).
        callEventRecorder.init(this)

        // Раз в 24ч чистим on-device телеметрию старше 90 дней (требование 2.4).
        // Запускается только при достаточном заряде батареи.
        val retentionConstraints = Constraints.Builder()
            .setRequiresBatteryNotLow(true)
            .build()
        val retentionRequest = PeriodicWorkRequestBuilder<TelemetryRetentionWorker>(
            24, TimeUnit.HOURS,
        ).setConstraints(retentionConstraints).build()
        WorkManager.getInstance(this).enqueueUniquePeriodicWork(
            TelemetryRetentionWorker.UNIQUE_NAME,
            ExistingPeriodicWorkPolicy.UPDATE, // был KEEP — не давал обновить retention-window
            retentionRequest,
        )

        // Раз в час чистит notification_event и app_usage_event старше 1 часа.
        // Фичи recent_10m / recent_30m используют только 10-30 минут — старше мусор.
        val notifCleanupRequest = PeriodicWorkRequestBuilder<com.antispam.blocker.domain.personal.TelemetryRetentionWorker>(
            1, TimeUnit.HOURS,
        ).setConstraints(Constraints.Builder().setRequiresBatteryNotLow(true).build()).build()
        WorkManager.getInstance(this).enqueueUniquePeriodicWork(
            "NotifCleanupHourly",
            ExistingPeriodicWorkPolicy.UPDATE,
            notifCleanupRequest,
        )

        // Разовый бэкфилл CallerID: запускаем определение для всех номеров
        // которые уже лежат в журнале. Без этого subtitle появляется только
        // для новых звонков. Флаг гарантирует один проход за жизнь установки.
        val callerIdPrefs = getSharedPreferences("caller_id_backfill", MODE_PRIVATE)
        if (!callerIdPrefs.getBoolean("v1_done", false)) {
            appScope.launch {
                try {
                    val numbers = database.callRecordDao().getDistinctNumbers()
                    android.util.Log.i("SpamBlockerApp", "CallerID backfill: ${numbers.size} numbers")
                    numbers.forEach { CallerLookupWorker.enqueue(this@SpamBlockerApp, it) }
                    callerIdPrefs.edit().putBoolean("v1_done", true).apply()
                } catch (t: Throwable) {
                    android.util.Log.w("SpamBlockerApp", "CallerID backfill failed", t)
                }
            }
        }
        // v2 backfill: номера, которые не получили caller ID при v1 из-за
        // NetworkType.CONNECTED (баг в CallerLookupWorker — оффлайн-лукапу
        // сеть не нужна, но worker не запускался без соединения). После
        // удаления network constraint перезапускаем для всех номеров —
        // worker идемпотентен (проверяет кэш перед запросом к libphonenumber).
        if (!callerIdPrefs.getBoolean("v2_done", false)) {
            appScope.launch {
                try {
                    val numbers = database.callRecordDao().getDistinctNumbers()
                    android.util.Log.i("SpamBlockerApp", "CallerID v2 backfill: ${numbers.size} numbers")
                    numbers.forEach { CallerLookupWorker.enqueue(this@SpamBlockerApp, it) }
                    callerIdPrefs.edit().putBoolean("v2_done", true).apply()
                } catch (t: Throwable) {
                    android.util.Log.w("SpamBlockerApp", "CallerID v2 backfill failed", t)
                }
            }
        }

        // Однократный сброс оффлайн-кэша после внедрения реестра РКН:
        // старые записи содержат region="Россия", нужно перезаписать точными данными.
        val rknPrefs = getSharedPreferences("rkn_lookup", MODE_PRIVATE)
        if (!rknPrefs.getBoolean("purge_v1_done", false)) {
            appScope.launch {
                try {
                    database.callerLookupDao().purgeOffline()
                    val numbers = database.callRecordDao().getDistinctNumbers()
                    android.util.Log.i("SpamBlockerApp", "RKN purge+requeue: ${numbers.size} numbers")
                    numbers.forEach { CallerLookupWorker.enqueue(this@SpamBlockerApp, it) }
                    rknPrefs.edit().putBoolean("purge_v1_done", true).apply()
                } catch (t: Throwable) {
                    android.util.Log.w("SpamBlockerApp", "RKN purge failed", t)
                }
            }
        }

        // Раз в сутки чистим старые аудиофайлы автоответчика (>30 дней)
        val answerBotCleanupPrefs = getSharedPreferences("answerbot_cleanup", MODE_PRIVATE)
        val lastCleanup = answerBotCleanupPrefs.getLong("last_cleanup", 0L)
        if (System.currentTimeMillis() - lastCleanup > 24 * 3_600_000L) {
            appScope.launch {
                try {
                    val dir = java.io.File(filesDir, "answerbot")
                    val cutoff = System.currentTimeMillis() - 30L * 24 * 3_600_000L
                    dir.listFiles()?.filter { it.isFile && it.lastModified() < cutoff }?.forEach { it.delete() }
                    answerBotCleanupPrefs.edit().putLong("last_cleanup", System.currentTimeMillis()).apply()
                } catch (t: Throwable) {
                    android.util.Log.w("SpamBlockerApp", "AnswerBot cleanup failed", t)
                }
            }
        }

        // Одноразовая чистка: notification_event и app_usage_event старше 24h.
        // Раньше TelemetryRetentionWorker держал их 90 дней, но фичи
        // recent_10m / recent_30m используют только 10–30 минут.
        val telemetryTrimPrefs = getSharedPreferences("telemetry_trim", MODE_PRIVATE)
        if (!telemetryTrimPrefs.getBoolean("trim_v1_done", false)) {
            appScope.launch {
                try {
                    val cutoff = System.currentTimeMillis() - 24L * 3_600_000L
                    val delNotif = database.notificationEventDao().deleteOlderThan(cutoff)
                    val delApp = database.appUsageEventDao().deleteOlderThan(cutoff)
                    android.util.Log.i("SpamBlockerApp", "Telemetry trim v1: notif=$delNotif appUsage=$delApp")
                    telemetryTrimPrefs.edit().putBoolean("trim_v1_done", true).apply()
                } catch (t: Throwable) {
                    android.util.Log.w("SpamBlockerApp", "Telemetry trim failed", t)
                }
            }
        }
        // v2: агрессивнее — 1 час вместо 24 часов. Фичам хватит 10 минут.
        if (!telemetryTrimPrefs.getBoolean("trim_v2_done", false)) {
            appScope.launch {
                try {
                    val cutoff = System.currentTimeMillis() - 3_600_000L
                    val delNotif = database.notificationEventDao().deleteOlderThan(cutoff)
                    val delApp = database.appUsageEventDao().deleteOlderThan(cutoff)
                    android.util.Log.i("SpamBlockerApp", "Telemetry trim v2: notif=$delNotif appUsage=$delApp")
                    telemetryTrimPrefs.edit().putBoolean("trim_v2_done", true).apply()
                } catch (t: Throwable) {
                    android.util.Log.w("SpamBlockerApp", "Telemetry trim v2 failed", t)
                }
            }
        }
    }

    suspend fun updateProfileVector(vector: UserProfileVector) {
        _profileVector = vector
        profileVectorStore.saveVector(vector)
    }

    /**
     * Рекурсивно копирует директорию из assets в filesDir.
     * Используется для распаковки Vosk модели (много файлов ~45 MB).
     */
    private fun copyAssetDir(assetPath: String, targetDir: java.io.File) {
        targetDir.mkdirs()
        val list = assets.list(assetPath) ?: return
        for (name in list) {
            val childPath = "$assetPath/$name"
            val childFile = java.io.File(targetDir, name)
            // Если assets.list(childPath) непустой — это поддиректория
            val subList = runCatching { assets.list(childPath) }.getOrNull()
            if (subList != null && subList.isNotEmpty()) {
                copyAssetDir(childPath, childFile)
            } else {
                assets.open(childPath).use { input ->
                    java.io.FileOutputStream(childFile).use { output ->
                        input.copyTo(output)
                    }
                }
            }
        }
    }

    companion object {
        lateinit var instance: SpamBlockerApp
            private set
    }
}
