package com.antispam.blocker.service

import android.content.Intent
import android.os.Build
import android.telecom.Call
import android.telecom.CallScreeningService
import com.antispam.blocker.SpamBlockerApp
import com.antispam.blocker.data.assets.OfficialWhitelistImporter
import com.antispam.blocker.data.cache.ContactsCache
import com.antispam.blocker.data.db.dao.FeatureSnapshotDao
import com.antispam.blocker.data.db.entity.FeatureSnapshot
import com.antispam.blocker.data.prefs.FeedbackLearningStore
import com.antispam.blocker.data.prefs.SettingsStore
import com.antispam.blocker.data.repository.BlockListRepository
import com.antispam.blocker.data.repository.CallLogRepository
import com.antispam.blocker.domain.detector.Verdict
import com.antispam.blocker.domain.lookup.CallerLookupWorker
import com.antispam.blocker.domain.personal.DeviceFeatureExtractor
import com.antispam.blocker.domain.personal.DeviceFeatures
import com.antispam.blocker.domain.personal.FusionDecider
import com.antispam.blocker.domain.personal.ImplicitLabel
import com.antispam.blocker.domain.personal.WarmUpGate
import com.antispam.blocker.domain.scoring.FeatureExtractor
import com.antispam.blocker.domain.scoring.FeedbackHandler
import com.antispam.blocker.domain.scoring.RiskScore
import com.antispam.blocker.domain.scoring.SmartSpamDetector
import com.antispam.blocker.domain.scoring.UserProfileVector
import com.antispam.blocker.domain.tracking.DecisionTracker
import com.antispam.blocker.notification.SpamWarningNotifier
import com.antispam.blocker.overlay.SpamAlertOverlayService
import com.antispam.blocker.util.PhoneNormalizer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

class SpamCallScreeningService : CallScreeningService() {

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private lateinit var smartDetector: SmartSpamDetector
    private lateinit var callLogRepo: CallLogRepository
    private lateinit var blockListRepo: BlockListRepository
    private lateinit var settings: SettingsStore
    private lateinit var notifier: SpamWarningNotifier
    private lateinit var featureExtractor: FeatureExtractor
    private lateinit var profileVector: UserProfileVector
    private lateinit var feedbackHandler: FeedbackHandler
    private lateinit var feedbackStore: FeedbackLearningStore
    private lateinit var decisionTracker: DecisionTracker

    // Device_Model wiring (tasks 12.2 / 12.3): persists the exact feature
    // vector seen at decision time and fuses Cloud_Model + Device_Model
    // votes (with whitelist/blacklist overrides and Warm_Up gating) into
    // the final verdict before responding / notifying.
    private lateinit var featureSnapshotDao: FeatureSnapshotDao
    private lateinit var fusionDecider: FusionDecider
    private lateinit var warmUpGate: WarmUpGate

    override fun onCreate() {
        super.onCreate()
        val app = SpamBlockerApp.instance
        val db = app.database
        settings = app.settingsStore
        feedbackStore = FeedbackLearningStore(this)

        // Прогреваем in-memory кэш контактов до того, как на сервис придёт
        // первый incoming звонок: ContactsCache.contains() будет O(1) вместо
        // ContentResolver.query() в FeatureExtractor.checkIsContact() на каждом
        // звонке. Observer внутри кэша подхватывает правки справочника.
        ContactsCache.init(this)

        blockListRepo = BlockListRepository(
            db.blockedNumberDao(),
            db.allowedNumberDao(),
            PhoneNormalizer,
            prebuiltReader = SpamBlockerApp.instance.prebuiltBlocklistReader
        )
        callLogRepo = CallLogRepository(db.callRecordDao())
        notifier = SpamWarningNotifier(this)
        featureExtractor = FeatureExtractor(
            context = this,
            callRecordDao = db.callRecordDao(),
            blockListRepo = blockListRepo,
            // Включает per-prefix override (Home-чип «занести в персональный
            // allowlist»): isAllowlist в фичах будет true и для номеров, чей
            // префикс юзер сам пометил как доверенный.
            feedbackStore = feedbackStore
        )

        smartDetector = SmartSpamDetector(
            context = this,
            blockListRepo = blockListRepo,
            callRecordDao = db.callRecordDao(),
            settings = settings,
            featureExtractor = featureExtractor,
            feedbackStore = feedbackStore,
            // Device_Model контур (задачи 12.x): экстрактор фич + LR-модель.
            // Сама модель и DataStore-стор живут в SpamBlockerApp по-singletonу,
            // чтобы веса не пересоздавались между сервисами и UI.
            deviceFeatureExtractor = DeviceFeatureExtractor(
                context = this,
                callEventDao = db.callEventDao(),
                notificationEventDao = db.notificationEventDao(),
                store = app.deviceModelStore,
            ),
            deviceModel = app.deviceModel,
        )

        feedbackHandler = FeedbackHandler(
            feedbackStore = feedbackStore,
            blockListRepo = blockListRepo,
            trainingDataDao = db.trainingDataDao()
        )

        decisionTracker = DecisionTracker(
            dao = db.decisionRecordDao(),
            modelVersionProvider = { app.modelVersion }
        )

        // Device_Model wiring: snapshot DAO is read here so the screening
        // service can persist `feature_snapshot` rows BEFORE notifying /
        // responding (Req 2.3). FusionDecider is stateless. WarmUpGate uses
        // the same singleton DeviceModelStore that the model itself reads,
        // so installedAt / labelCount stay consistent across SGD steps and
        // the gate check.
        featureSnapshotDao = db.featureSnapshotDao()
        fusionDecider = FusionDecider()
        warmUpGate = WarmUpGate(app.deviceModelStore)

        profileVector = app.profileVector

        // Импорт официального РФ whitelist (банки, операторы, экстренные службы)
        val whitelistImporter = OfficialWhitelistImporter(this, blockListRepo)
        serviceScope.launch { whitelistImporter.importIfFirstRun() }
    }

    override fun onScreenCall(callDetails: Call.Details) {
        val handle = callDetails.handle
        val number = handle?.schemeSpecificPart
        val isHidden = number.isNullOrBlank()

        android.util.Log.d("SpamBlocker", "Incoming call: $number (hidden: $isHidden)")

        serviceScope.launch {
            try {
                val normalized = PhoneNormalizer.normalize(number)

                // Триггерим оффлайн-определение звонящего (libphonenumber) сразу
                // на этапе скрининга, не дожидаясь конца звонка и записи в CallLog.
                // Это закрывает дыру для BLOCK-звонков (BLOCKED_TYPE=6), которые
                // CallEventRecorder пропускает — без этого вызова caller ID для
                // заблокированных номеров никогда бы не заполнялся.
                if (normalized != null && !isHidden) {
                    try {
                        CallerLookupWorker.enqueue(this@SpamCallScreeningService, normalized)
                    } catch (t: Throwable) {
                        android.util.Log.w("SpamBlocker", "CallerLookupWorker enqueue failed", t)
                    }
                }

                val scoringResult = smartDetector.scoreWithFeatures(number, isHidden, callDetails, profileVector)
                val riskScore = scoringResult.risk

                android.util.Log.d("SpamBlocker", "Risk score: ${riskScore.score} level=${riskScore.level} verdict=${riskScore.verdict} reasons=${riskScore.reasons}")

                // Task 12.2: persist FeatureSnapshot BEFORE notifying / responding,
                // when scoreWithFeatures actually produced device-model fields. On
                // fast-path verdicts (allow-/block-list / emergency / disabled) and
                // when personalClassifierEnabled = false, scoringResult device-side
                // fields are null and we skip the snapshot write — there is nothing
                // for OnlineTrainer to learn from in those cases.
                //
                // V1 simplification: callEventId is set to null. End-of-call CallEvent
                // creation lands in a future task (see tasks.md "12.2 Notes"); until
                // then we propagate the snapshot's primary key directly to the
                // notifier as the feedback handle. OnlineTrainer.applyLabel resolves
                // both shapes via getByCallEventId(id) ?: getById(id), so explicit
                // feedback works end-to-end.
                var snapshotId: Long = -1L
                val snapshotJson = scoringResult.deviceFeaturesSnapshotJson
                val deviceProb = scoringResult.deviceProbBlock
                if (snapshotJson != null && deviceProb != null) {
                    try {
                        snapshotId = featureSnapshotDao.insert(
                            FeatureSnapshot(
                                callEventId = null,
                                normalizedNumber = normalized,
                                timestamp = System.currentTimeMillis(),
                                featuresJson = snapshotJson,
                                featureSchemaVersion = DeviceFeatures.SCHEMA_VERSION,
                                weightsHash = null,
                                deviceProbBlock = deviceProb,
                            )
                        )
                    } catch (t: Throwable) {
                        // Never crash the screening pipeline if snapshot insert fails.
                        // The call still gets a verdict — explicit feedback for this
                        // particular call will silently no-op (OnlineTrainer logs and
                        // returns), which is the lesser evil.
                        android.util.Log.w("SpamBlocker", "FeatureSnapshot insert failed", t)
                    }
                }

                // Task 12.3: fuse Cloud_Model + Device_Model + user lists into the
                // final verdict (Req 5.1–5.10). Whitelist/blacklist override both
                // models; while Warm_Up is incomplete or device-vote is unavailable,
                // FusionDecider falls back to cloud-only (warmup_cloud_only branch).
                val isInWhitelist = if (normalized != null) blockListRepo.isAllowed(normalized) else false
                val isInBlacklist = if (normalized != null) blockListRepo.isBlocked(normalized) else false
                val isWarmUpComplete = warmUpGate.isComplete()

                // RiskScore.confidence is a 3-level enum; FusionDecider needs a
                // numeric cutoff to compare against HIGH_CONFIDENCE = 0.70f. Map
                // HIGH→1.0, MEDIUM→0.7, LOW→0.3 so HIGH always votes-for-block
                // and LOW never does — matches the design intent that "confident
                // BLOCK" means the rule engine produced a HIGH-confidence vote.
                val cloudConfidence = when (riskScore.confidence) {
                    RiskScore.Confidence.HIGH -> 1.0f
                    RiskScore.Confidence.MEDIUM -> 0.69f
                    RiskScore.Confidence.LOW -> 0.3f
                }
                val fusion = fusionDecider.decide(
                    FusionDecider.FusionInput(
                        cloudVerdict = riskScore.verdict,
                        cloudConfidence = cloudConfidence,
                        deviceVerdict = scoringResult.deviceVerdict,
                        deviceProbBlock = scoringResult.deviceProbBlock,
                        isInWhitelist = isInWhitelist,
                        isInBlacklist = isInBlacklist,
                        isWarmUpComplete = isWarmUpComplete,
                    )
                )
                val finalVerdict = fusion.finalVerdict
                android.util.Log.d(
                    "SpamBlocker",
                    "Fusion: cloud=${riskScore.verdict}/${riskScore.confidence} " +
                        "device=${scoringResult.deviceVerdict} warmup=$isWarmUpComplete " +
                        "wl=$isInWhitelist bl=$isInBlacklist -> $finalVerdict (${fusion.rationaleTag})"
                )

                // P1 fix: прокидываем snapshotId напрямую — иначе долгое нажатие в
                // журнале берёт самый свежий snapshot по номеру, что для повторных
                // звонков с того же номера показывает "чужой" вектор.
                callLogRepo.record(
                    normalizedNumber = normalized,
                    originalNumber = number,
                    verdict = finalVerdict,
                    ruleName = riskScore.source,
                    featureSnapshotId = if (snapshotId > 0L) snapshotId else null,
                )

                // Снимок фичей для аудит-лога DecisionTracker. Переиспользуем
                // тот, что уже посчитал SmartSpamDetector — для модельного и
                // rule-based путей. На fast-path вердиктах (allow-/block-list /
                // emergency / disabled) фичи не нужны DecisionTracker'у при
                // той же логике, что и до рефакторинга — он спокойно считает
                // их сам.
                val features = scoringResult.features
                    ?: featureExtractor.extract(number, isHidden, callDetails, profileVector)

                try {
                    // DecisionTracker теперь видит ФИНАЛЬНЫЙ (fused) вердикт —
                    // тот, что реально применён к звонку. Раньше писался cloud-
                    // vote, и карточка в «ИИ» показывала «BLOCK 82%», хотя
                    // звонок прошёл (UX-баг: «модель назвала спамом, но не
                    // заблокировала»). Теперь карточка показывает финальный
                    // вердикт, а в reasons добавляется строка про коррекцию,
                    // чтобы было видно ПОЧЕМУ модель и финал не совпали
                    // (например, `device_veto_allow` или
                    // `single_block_downgrade_warn`).
                    decisionTracker.record(
                        rawNumber = number,
                        normalizedNumber = normalized,
                        features = features,
                        risk = riskScore,
                        finalVerdict = finalVerdict,
                        fusionRationaleTag = fusion.rationaleTag,
                    )
                } catch (e: Exception) {
                    android.util.Log.w("SpamBlocker", "Failed to record decision", e)
                }

                // Раньше здесь писалось в `training_data` с label = вердикт самой
                // модели — это создавало рекурсивный сигнал (модель «учится» на
                // собственных предсказаниях, замораживая ошибки). Теперь
                // `decision_record` (через DecisionTracker выше) хранит весь
                // аудит решения, а `training_data` пополняется ТОЛЬКО через
                // FeedbackHandler — когда юзер реально подтвердил вердикт
                // («Спам» / «Не спам» / «Разблокировать»). Без человеческой
                // разметки мы тут ничего не пишем.

                val displayNumber = number ?: "Скрытый номер"
                // V1: snapshot.id is propagated as the feedback handle (see 12.2
                // notes above). When end-of-call CallEvent linking lands later,
                // FeatureSnapshotDao.updateCallEventId will fix this up and the
                // notifier will receive the real CallEvent id instead.
                val feedbackHandle: Long? = if (snapshotId > 0L) snapshotId else null

                when (finalVerdict) {
                    Verdict.BLOCK -> {
                        val skipLog = settings.skipCallLogForBlocked.first()
                        respondToCall(callDetails, buildBlockResponse(skipLog))
                        notifier.showBlocked(
                            displayNumber,
                            reasons = riskScore.reasons,
                            originalVerdict = finalVerdict.name.lowercase(),
                            activeFactorIds = riskScore.activeFactorIds,
                            callEventId = feedbackHandle
                        )
                        // Закрываем дыру в обучении Device-модели: после
                        // нашего собственного BLOCK звонок попадает в
                        // системный CallLog как BLOCKED_TYPE=6, который
                        // CallEventRecorder.mapCallType намеренно пропускает
                        // (чтобы не дублировать). Без этого вызова
                        // FeatureSnapshot остался бы orphan (callEventId=null)
                        // и SGD-step никогда бы не выполнился — модель
                        // забыла бы, что мы только что заблокировали этот
                        // номер. IMPLICIT_WEIGHT (0.5f) меньше, чем у
                        // явного «Был ли это спам? Да» (1.5f), так что
                        // юзер всегда может перебить наш authoritative-
                        // сигнал тапом по нотификации.
                        if (snapshotId > 0L && snapshotJson != null && deviceProb != null) {
                            try {
                                SpamBlockerApp.instance.onlineTrainer
                                    .applyImplicitLabel(snapshotId, ImplicitLabel.BLOCK)
                            } catch (t: Throwable) {
                                android.util.Log.w(
                                    "SpamBlocker",
                                    "implicit BLOCK label failed for snapshotId=$snapshotId",
                                    t,
                                )
                            }
                        }
                    }
                    Verdict.WARN -> {
                        respondToCall(callDetails, buildWarnResponse())
                        notifier.showWarning(
                            displayNumber,
                            reasons = riskScore.reasons,
                            originalVerdict = finalVerdict.name.lowercase(),
                            activeFactorIds = riskScore.activeFactorIds,
                            callEventId = feedbackHandle
                        )
                        SpamAlertOverlayService.show(
                            this@SpamCallScreeningService,
                            displayNumber,
                            riskScore.reasons.firstOrNull()
                        )
                    }
                    Verdict.ALLOW -> respondToCall(callDetails, buildAllowResponse())
                }
            } catch (e: Exception) {
                android.util.Log.e("SpamBlocker", "Error in onScreenCall", e)
                respondToCall(callDetails, buildAllowResponse())
            }
        }
    }

    private fun buildBlockResponse(skipCallLog: Boolean): CallResponse {
        return CallResponse.Builder()
            .setDisallowCall(true)
            .setRejectCall(true)
            .setSkipCallLog(skipCallLog)
            .setSkipNotification(true)
            .build()
    }

    private fun buildWarnResponse(): CallResponse {
        return CallResponse.Builder()
            .setSilenceCall(true)
            .setSkipNotification(true)
            .build()
    }

    private fun buildAllowResponse(): CallResponse {
        return CallResponse.Builder().build()
    }
}
