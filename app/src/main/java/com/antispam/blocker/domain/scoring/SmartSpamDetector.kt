package com.antispam.blocker.domain.scoring

import android.content.Context
import android.telecom.Call
import android.util.Log
import com.antispam.blocker.data.cache.ContactsCache
import com.antispam.blocker.data.db.dao.CallRecordDao
import com.antispam.blocker.data.prefs.SettingsStore
import com.antispam.blocker.data.repository.BlockListRepository
import com.antispam.blocker.domain.detector.Verdict
import com.antispam.blocker.domain.model.ModelCard
import com.antispam.blocker.domain.model.SpamModel
import com.antispam.blocker.domain.personal.DeviceFeatureExtractor
import com.antispam.blocker.domain.personal.DeviceFeatures
import com.antispam.blocker.domain.personal.DeviceModel
import com.antispam.blocker.util.PhoneNormalizer
import kotlinx.coroutines.flow.first
import org.json.JSONObject

class SmartSpamDetector(
    private val context: Context,
    private val blockListRepo: BlockListRepository,
    private val callRecordDao: CallRecordDao,
    private val settings: SettingsStore,
    private val featureExtractor: FeatureExtractor,
    private val feedbackStore: com.antispam.blocker.data.prefs.FeedbackLearningStore,
    /**
     * Экстрактор фич для on-device персонального классификатора (Device_Model).
     * Всегда передаётся `SpamCallScreeningService` — `null` оставлен только для
     * unit-тестов, которые не должны лезть в этот контур.
     */
    private val deviceFeatureExtractor: DeviceFeatureExtractor? = null,
    /**
     * Сама Device_Model (логистическая регрессия). Если не передана —
     * Device-вердикт не считается, поля `deviceVerdict / deviceProbBlock /
     * deviceFeaturesSnapshotJson / topContributions` в [ScoringResult] остаются
     * `null` и downstream `FusionDecider` падает в branch «warmup_cloud_only».
     */
    private val deviceModel: DeviceModel? = null,
) {
    private val spamModel: SpamModel by lazy {
        SpamModel(context).also {
            it.loadModel()
        }
    }

    /**
     * Сбрасывает дрейф feedback-порогов, если версия model_card изменилась
     * с момента последнего вызова. Дрейф валиден только для конкретно той
     * модели, на которой был накоплен — иначе старый сдвиг бьёт по новой
     * модели, которая может уже учитывать ту же ситуацию через свои
     * пробабилитные пороги.
     */
    private suspend fun reconcileFeedbackWithModelVersion() {
        val card = ModelCard.load(context) ?: return
        val pinned = feedbackStore.pinnedModelVersion.first()
        if (pinned == card.version) return
        Log.i(
            "SmartSpamDetector",
            "model_card version changed: pinned=$pinned new=${card.version}; " +
                "resetting feedback threshold drift"
        )
        feedbackStore.resetThresholdsForModel(card.version)
    }

    /**
     * Тонкая обёртка над [scoreWithFeatures] для вызовов, которым не нужен
     * снимок [CallFeatures] (UI / тесты / spam-rules скрипты).
     */
    suspend fun score(
        number: String?,
        isHidden: Boolean,
        callDetails: Call.Details?,
        profileVector: UserProfileVector
    ): RiskScore = scoreWithFeatures(number, isHidden, callDetails, profileVector).risk

    /**
     * Возвращает [ScoringResult] вместо «голого» [RiskScore], чтобы вызывающий
     * мог переиспользовать уже посчитанный [CallFeatures] (см. [DecisionTracker]).
     * На fast-path'ах (защита выключена / экстренный / абсолютные списки)
     * фичи не считаются — [ScoringResult.features] = null. На модельном пути
     * фичи возвращаются «как есть» — это экономит повторный вызов
     * [FeatureExtractor.extract] на каждом входящем звонке.
     */
    suspend fun scoreWithFeatures(
        number: String?,
        isHidden: Boolean,
        callDetails: Call.Details?,
        profileVector: UserProfileVector
    ): ScoringResult {
        reconcileFeedbackWithModelVersion()

        val warnThreshold = (feedbackStore.warnThreshold.first() * 100).toInt()
        val blockThreshold = (feedbackStore.blockThreshold.first() * 100).toInt()

        if (!settings.protectionEnabled.first()) {
            return ScoringResult(
                risk = RiskScore(
                    score = 0, level = RiskLevel.SAFE, verdict = Verdict.ALLOW,
                    reasons = emptyList(), confidence = RiskScore.Confidence.LOW, source = "disabled",
                    warnThreshold = warnThreshold, blockThreshold = blockThreshold
                ),
                features = null
            )
        }

        // Emergency whitelist: never block emergency numbers
        if (number != null && isEmergencyNumber(number)) {
            return ScoringResult(
                risk = RiskScore(
                    score = 0, level = RiskLevel.SAFE, verdict = Verdict.ALLOW,
                    reasons = listOf("Экстренная служба"),
                    confidence = RiskScore.Confidence.HIGH, source = "emergency_whitelist",
                    activeFactorIds = listOf("emergency"),
                    warnThreshold = warnThreshold, blockThreshold = blockThreshold
                ),
                features = null
            )
        }

        // Absolute lists checked BEFORE model
        if (number != null) {
            if (blockListRepo.isAllowed(number)) {
                return ScoringResult(
                    risk = RiskScore(
                        score = 0, level = RiskLevel.SAFE, verdict = Verdict.ALLOW,
                        reasons = listOf("В белом списке"),
                        confidence = RiskScore.Confidence.HIGH, source = "allowlist",
                        activeFactorIds = listOf("allowlist"),
                        warnThreshold = warnThreshold, blockThreshold = blockThreshold
                    ),
                    features = null
                )
            }
            if (blockListRepo.isBlocked(number)) {
                return ScoringResult(
                    risk = RiskScore(
                        score = 100, level = RiskLevel.DANGEROUS, verdict = Verdict.BLOCK,
                        reasons = listOf("В чёрном списке"),
                        confidence = RiskScore.Confidence.HIGH, source = "blacklist",
                        activeFactorIds = listOf("blacklist"),
                        warnThreshold = warnThreshold, blockThreshold = blockThreshold
                    ),
                    features = null
                )
            }

            // Contact fast-path: номер из записной книжки получает максимальный
            // траст и не прогоняется через ML/rule-engine. Иначе модель видит
            // факторы вроде `previously_rejected` или нейтральный prefixRisk и
            // выдаёт «82% BLOCK», после чего FusionDecider всё равно ALLOW'ит
            // через device_veto_allow — UX выглядит так, будто «ИИ называет
            // контакт спамом». Лечим в источнике: контакт — это явный whitelist
            // на уровне источника решения (Req 4.5: контакты = высший траст).
            //
            // Blacklist выше уже проверен и перебивает контакт — если юзер сам
            // внёс кого-то из контактов в чёрный список, это его решение.
            val normalizedForContact = PhoneNormalizer.normalize(number)
            if (normalizedForContact != null && ContactsCache.contains(normalizedForContact) == true) {
                return ScoringResult(
                    risk = RiskScore(
                        score = 0, level = RiskLevel.SAFE, verdict = Verdict.ALLOW,
                        reasons = listOf("Номер из контактов"),
                        confidence = RiskScore.Confidence.HIGH, source = "contact",
                        activeFactorIds = listOf("contact"),
                        warnThreshold = warnThreshold, blockThreshold = blockThreshold
                    ),
                    features = null
                )
            }
        }

        // Extract features
        val features = featureExtractor.extract(number, isHidden, callDetails, profileVector)
        val factors = evaluateFactors(features, profileVector)
        val activeFactors = factors.filter { it.isActive }
        val activeReasons = activeFactors.map { it.reason }
        // IDs in descending priority (points × weight). Notification action receiver
        // reconstructs RiskFactor stubs in this order so FeedbackHandler picks the
        // actually-influential top-3 to EMA-adjust.
        val activeIds = activeFactors
            .sortedByDescending { it.points * it.weight }
            .map { it.id }

        val learnedWeights = feedbackStore.getAllWeights()
        val ruleTotalScore = activeFactors.sumOf { factor ->
            val learnedWeight = learnedWeights[factor.id] ?: factor.weight
            (factor.points * learnedWeight).toInt()
        }.coerceIn(0, 100)

        // Try TFLite model first.
        val modelResult = spamModel.predict(features)
        if (modelResult != null) {
            // Rule-based safety net: rules могут поднять вердикт модели, но не опустить.
            // Главный кейс — `prefix_call_wave` (5+ звонков с одного DEF-кода за неделю):
            // эта фича не входит в ML-вход (модель 32-фичная), и без override модель
            // не увидит спам-волну. Аналогично «caller_verify_failed» / «spoofing_prefix»
            // — сильные локальные сигналы, которые должны бить ALLOW от модели.
            val waveBoost = activeFactors.firstOrNull { it.id == "prefix_call_wave" }
                ?.let { (it.points * it.weight).toInt() } ?: 0
            val hardOverride = activeFactors.any {
                it.id == "spoofing_prefix" || it.id == "caller_verify_failed"
            }

            val (overrideVerdict, overrideScore) = when (modelResult.verdict) {
                Verdict.BLOCK -> modelResult.verdict to modelResult.score
                Verdict.WARN -> {
                    if (waveBoost >= 30 || hardOverride) {
                        Verdict.BLOCK to maxOf(modelResult.score, blockThreshold)
                    } else modelResult.verdict to modelResult.score
                }
                Verdict.ALLOW -> when {
                    waveBoost >= 40 || hardOverride ->
                        Verdict.BLOCK to maxOf(modelResult.score, blockThreshold)
                    waveBoost >= 20 || ruleTotalScore >= warnThreshold ->
                        Verdict.WARN to maxOf(modelResult.score, warnThreshold)
                    else -> modelResult.verdict to modelResult.score
                }
            }
            val overrideLevel = when {
                overrideScore >= 70 -> RiskLevel.DANGEROUS
                overrideScore >= 35 -> RiskLevel.SUSPICIOUS
                else -> RiskLevel.SAFE
            }

            val deviceExtras = maybeRunDeviceModel(number, isHidden, callDetails)

            return ScoringResult(
                risk = modelResult.copy(
                    verdict = overrideVerdict,
                    score = overrideScore,
                    level = overrideLevel,
                    reasons = activeReasons.ifEmpty { modelResult.reasons },
                    activeFactorIds = activeIds,
                    ruleScore = ruleTotalScore,
                    warnThreshold = warnThreshold,
                    blockThreshold = blockThreshold
                ),
                features = features,
                deviceVerdict = deviceExtras?.verdict,
                deviceProbBlock = deviceExtras?.probBlock,
                deviceFeaturesSnapshotJson = deviceExtras?.snapshotJson,
                topContributions = deviceExtras?.topContributions,
            )
        }

        // Fallback: rule-based scoring
        val level = when {
            ruleTotalScore >= 70 -> RiskLevel.DANGEROUS
            ruleTotalScore >= 35 -> RiskLevel.SUSPICIOUS
            else -> RiskLevel.SAFE
        }

        val verdict = when {
            ruleTotalScore >= blockThreshold -> Verdict.BLOCK
            ruleTotalScore >= warnThreshold -> Verdict.WARN
            else -> Verdict.ALLOW
        }

        val confidence = when {
            activeFactors.any { it.id == "blacklist" || it.id == "allowlist" } -> RiskScore.Confidence.HIGH
            activeFactors.size >= 3 -> RiskScore.Confidence.MEDIUM
            else -> RiskScore.Confidence.LOW
        }

        val deviceExtras = maybeRunDeviceModel(number, isHidden, callDetails)

        return ScoringResult(
            risk = RiskScore(
                score = ruleTotalScore,
                level = level,
                verdict = verdict,
                reasons = activeReasons,
                confidence = confidence,
                source = "rule_engine",
                ruleScore = ruleTotalScore,
                activeFactorIds = activeIds,
                warnThreshold = warnThreshold,
                blockThreshold = blockThreshold
            ),
            features = features,
            deviceVerdict = deviceExtras?.verdict,
            deviceProbBlock = deviceExtras?.probBlock,
            deviceFeaturesSnapshotJson = deviceExtras?.snapshotJson,
            topContributions = deviceExtras?.topContributions,
        )
    }

    /**
     * Запускает Device_Model на хот-пас (после Cloud_Model или rule-based fallback'а),
     * собирает фичи через [DeviceFeatureExtractor], кладёт результат предсказания
     * в [DeviceScoringExtras]. Возвращает `null` если:
     *
     *   - вся фича on-device-классификатора выключена через
     *     `SettingsStore.personalClassifierEnabled = false` (Req 7.4, 7.5);
     *   - extractor / model не были переданы в конструктор (unit-тесты);
     *   - сборка фич или predict упали с исключением — лог пишется, но мы не
     *     роняем call-screening pipeline.
     *
     * Не вызывается на fast-path вердиктах (защита выключена / экстренный /
     * абсолютные allow-/block-списки) — там вызов даже не нужен, finальное
     * решение уже принято и device-вердикт всё равно перекрывается whitelist'ом /
     * blacklist'ом в [com.antispam.blocker.domain.personal.FusionDecider].
     */
    private suspend fun maybeRunDeviceModel(
        number: String?,
        isHidden: Boolean,
        callDetails: Call.Details?,
    ): DeviceScoringExtras? {
        val extractor = deviceFeatureExtractor ?: return null
        val model = deviceModel ?: return null

        val enabled = try {
            settings.personalClassifierEnabled.first()
        } catch (t: Throwable) {
            Log.w(TAG, "personalClassifierEnabled read failed; treating as enabled", t)
            true
        }
        if (!enabled) return null

        return try {
            val normalized = PhoneNormalizer.normalize(number)
            val features = extractor.extract(
                normalizedNumber = normalized,
                isHidden = isHidden,
                callDetails = callDetails,
            )
            val prediction = model.predict(features)
            DeviceScoringExtras(
                verdict = prediction.verdict,
                probBlock = prediction.probBlock,
                snapshotJson = serializeFeaturesJson(features),
                topContributions = prediction.topContributions,
            )
        } catch (t: Throwable) {
            // Никогда не роняем основной pipeline из-за on-device-контура.
            Log.w(TAG, "Device_Model invocation failed; falling back to cloud-only", t)
            null
        }
    }

    /**
     * Снапшот, отдаваемый downstream'у через [ScoringResult]. Поля
     * соответствуют новым полям [ScoringResult] (см. задачу 12.1) и далее
     * переиспользуются `SpamCallScreeningService` для записи в `feature_snapshot`
     * (12.2) и `FusionDecider.decide` (12.3).
     */
    private data class DeviceScoringExtras(
        val verdict: com.antispam.blocker.domain.personal.DeviceVerdict,
        val probBlock: Float,
        val snapshotJson: String,
        val topContributions: List<com.antispam.blocker.domain.personal.FeatureContribution>,
    )

    private fun serializeFeaturesJson(features: DeviceFeatures): String {
        // Каноничная форма: {"feature_name": value} в порядке DeviceFeatures.NAMES.
        // Используется FeatureSnapshot.featuresJson (Req 2.3) — round-trip
        // гарантирован, потому что и parser и producer ходят по одному и тому же
        // списку имён.
        val obj = JSONObject()
        val values = features.toFloatArray()
        for (i in 0 until DeviceFeatures.SIZE) {
            obj.put(DeviceFeatures.NAMES[i], values[i].toDouble())
        }
        return obj.toString()
    }

    private fun evaluateFactors(features: CallFeatures, profile: UserProfileVector): List<RiskFactor> {
        val vulnerabilityMultiplier = 1f + (profile.vulnerabilityScore / 200f)
        val businessMultiplier = 1f - (profile.businessActivity / 300f)

        return listOf(
            RiskFactor(
                id = "hidden_number",
                displayName = "Скрытый номер",
                points = 50,
                reason = "Скрытый номер",
                weight = vulnerabilityMultiplier
            ).takeIf { features.hiddenNumber },
            RiskFactor(
                id = "not_contact",
                displayName = "Не в контактах",
                points = 20,
                reason = "Не в контактах",
                weight = if (features.contactsAvailable) 1f else 0.5f
            ).takeIf { !features.isContact && !features.hiddenNumber },
            RiskFactor(
                id = "russian_unknown",
                displayName = "Неизвестный +7",
                points = 15,
                reason = "Неизвестный российский номер",
                weight = businessMultiplier
            ).takeIf { features.isRussianNumber && !features.isContact },
            RiskFactor(
                id = "foreign_number",
                displayName = "Иностранный номер",
                points = 10,
                reason = "Иностранный номер",
                weight = if (profile.hasForeignContacts) 0.3f else 1f
            ).takeIf { features.isForeignNumber },
            RiskFactor(
                id = "short_code",
                displayName = "Короткий номер",
                points = 5,
                reason = "Короткий номер",
                weight = if (profile.hasHomePhone) 0.3f else 0.7f
            ).takeIf { features.isShortCode },
            RiskFactor(
                id = "spoofing_prefix",
                displayName = "Имитация российского префикса",
                points = 55,
                reason = "Номер похож на подмену российского кода",
                weight = vulnerabilityMultiplier
            ).takeIf { features.spoofingPrefixFlag },
            RiskFactor(
                id = "invalid_ru_range",
                displayName = "Неизвестный диапазон РФ",
                points = 15,
                reason = "Номер не похож на валидный диапазон РФ",
                weight = businessMultiplier
            ).takeIf { features.isRussianNumber && !features.isValidRuRange && !features.isShortCode },
            RiskFactor(
                id = "tollfree_8800",
                displayName = "Федеральный 8-800",
                points = 8,
                reason = "Федеральный номер 8-800",
                weight = if (features.inAllowlist) 0f else 0.7f
            ).takeIf { features.isTollFree8800 },
            RiskFactor(
                id = "beautiful_number",
                displayName = "Шаблонный номер",
                points = 10,
                reason = "Номер содержит повторяющийся цифровой паттерн",
                weight = businessMultiplier
            ).takeIf { features.beautifulNumberFlag && !features.inAllowlist },
            RiskFactor(
                id = "reputation_score",
                displayName = "Репутационный риск",
                points = (features.reputationScore * 45).toInt(),
                reason = "Репутационные признаки номера повышают риск",
                weight = 1f
            ).takeIf { features.reputationScore > 0.45f && !features.inAllowlist },
            RiskFactor(
                id = "prefix_risk",
                displayName = "Подозрительный префикс",
                points = (features.prefixRisk * 25).toInt(),
                reason = "Подозрительный префикс номера",
                weight = businessMultiplier
            ).takeIf { features.prefixRisk > 0.3f && features.prefixRisk < PREFIX_RISK_HIGH_THRESHOLD },
            // Cold-start доминирующий префикс-риск: prefixRisk >= 0.65 → агрессивнее
            // эскалируем WARN. На холодных номерах (которых нет в чёрном/белом списках)
            // это часто единственный значимый сигнал. Зеркалит правило `prefix_risk_high`
            // в scripts/spam_rules.py.
            RiskFactor(
                id = "prefix_risk_high",
                displayName = "Высокий риск по префиксу",
                points = (features.prefixRisk * 45).toInt(),
                reason = "Префикс этого номера часто фигурирует в жалобах",
                weight = if (features.inAllowlist || features.isContact) 0f else businessMultiplier
            ).takeIf {
                features.prefixRisk >= PREFIX_RISK_HIGH_THRESHOLD
                        && !features.inAllowlist
                        && !features.isContact
            },
            RiskFactor(
                id = "night_time",
                displayName = "Ночное время",
                points = 20,
                reason = "Звонок в ночное время",
                weight = vulnerabilityMultiplier
            ).takeIf { features.isNightTime },
            RiskFactor(
                id = "recent_bank_app",
                displayName = "Недавно в банковском приложении",
                points = 25,
                reason = "Вы недавно были в банковском приложении",
                weight = (profile.digitalActivity / 100f).coerceIn(0.3f, 1.5f)
            ).takeIf { features.recentBankApp },
            RiskFactor(
                id = "recent_gov_app",
                displayName = "Недавно в Госуслугах",
                points = 15,
                reason = "Вы недавно были в Госуслугах",
                weight = (profile.digitalActivity / 100f).coerceIn(0.3f, 1.2f)
            ).takeIf { features.recentGovApp },
            RiskFactor(
                id = "recent_marketplace",
                displayName = "Недавно в маркетплейсе",
                points = 10,
                reason = "Вы недавно были в маркетплейсе",
                weight = (profile.adsActivity / 100f).coerceIn(0.2f, 1f)
            ).takeIf { features.recentMarketplaceApp },
            RiskFactor(
                id = "call_frequency",
                displayName = "Частые звонки",
                points = (features.callFrequency * 30).toInt(),
                reason = "Частые повторные звонки",
                weight = 1f
            ).takeIf { features.callFrequency > 0.5f },
            RiskFactor(
                id = "previously_rejected",
                displayName = "Ранее отклонён",
                points = 15,
                reason = "Вы ранее отклоняли этот номер",
                weight = 1f
            ).takeIf { features.previouslyRejected },
            RiskFactor(
                id = "caller_verify_failed",
                displayName = "Номер не прошёл проверку",
                points = 30,
                reason = "Номер не прошёл проверку оператора",
                weight = vulnerabilityMultiplier
            ).takeIf { features.callerVerifyFailed },
            // Спам-волна по DEF-коду: 5+ разных номеров с одного префикса за 7 дней.
            // Чем больше — тем выше риск. Не входит в ML-вход (модель 32 фичи),
            // работает как rule-based override после инференса модели.
            RiskFactor(
                id = "prefix_call_wave",
                displayName = "Спам-волна по префиксу",
                points = (15 + minOf(features.prefixCallFrequency7d, 30) * 2).coerceAtMost(60),
                reason = "С этого префикса звонят последние 7 дней (${features.prefixCallFrequency7d} звонков)",
                weight = if (features.inAllowlist || features.isContact) 0f else 1f
            ).takeIf { features.prefixCallFrequency7d >= 5 && !features.inAllowlist }
        ).filterNotNull()
    }

    private fun isEmergencyNumber(number: String): Boolean {
        val cleaned = number.replace(Regex("[^\\d]"), "")
        return cleaned in EMERGENCY_NUMBERS
    }

    companion object {
        private const val TAG = "SmartSpamDetector"
        private val EMERGENCY_NUMBERS = setOf("112", "101", "102", "103", "104")
        // Зеркалит COLDSTART_PREFIX_RISK_WARN из scripts/spam_rules.py.
        // Выше этого порога prefixRisk считается доминирующим сигналом и
        // отдельный фактор `prefix_risk_high` поднимает вердикт до WARN
        // даже если модель сама дала ALLOW.
        const val PREFIX_RISK_HIGH_THRESHOLD = 0.65f
    }
}
