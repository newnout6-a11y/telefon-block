package com.antispam.blocker.domain.scoring

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.telecom.Call
import android.telecom.Connection
import com.antispam.blocker.data.cache.ContactsCache
import com.antispam.blocker.data.db.dao.CallRecordDao
import com.antispam.blocker.data.prefs.FeedbackLearningStore
import com.antispam.blocker.data.repository.BlockListRepository
import com.antispam.blocker.util.PhoneNormalizer
import kotlin.math.ln
import java.util.Calendar

class FeatureExtractor(
    private val context: Context,
    private val callRecordDao: CallRecordDao,
    private val blockListRepo: BlockListRepository? = null,
    /**
     * Опционально — даёт доступ к персональному prefix-allowlist (см.
     * [FeedbackLearningStore.prefixAllowlist]). Когда задан, расширяет
     * `inAllowlist` для номеров, чей префикс юзер пометил как «доверенный»
     * через Home-чип.
     */
    private val feedbackStore: FeedbackLearningStore? = null
) {
    private val recentContextProvider = RecentUserContextProvider(context)

    suspend fun extract(
        number: String?,
        isHidden: Boolean,
        callDetails: Call.Details?,
        profileVector: UserProfileVector
    ): CallFeatures {
        val normalized = number?.let { PhoneNormalizer.normalize(it) }
        val hasContactsPermission = context.checkSelfPermission(Manifest.permission.READ_CONTACTS) == PackageManager.PERMISSION_GRANTED
        val hasCallLogPermission = context.checkSelfPermission(Manifest.permission.READ_CALL_LOG) == PackageManager.PERMISSION_GRANTED

        val isContact = if (hasContactsPermission && normalized != null) {
            checkIsContact(normalized)
        } else false

        val isRussian = normalized != null && normalized.startsWith("+7")
        val isForeign = normalized != null && !normalized.startsWith("+7") && !normalized.startsWith("8")
        val rawDigits = number?.filter { it.isDigit() }.orEmpty()
        val normalizedDigits = normalized?.filter { it.isDigit() }.orEmpty()
        val isShort = rawDigits.length in 2..6 || normalizedDigits.length in 2..6
        val isStandardLen = normalizedDigits.length == 11 && normalizedDigits.startsWith("7")
        val isTollFree8800 = normalized?.startsWith("+7800") == true || rawDigits.startsWith("8800")
        val defCode = getRuDefCode(normalized)
        val isMobileRu = defCode in 900..999
        val isGeographical = defCode in 300..499
        val isValidRuRange = isShort || isTollFree8800 || isMobileRu || isGeographical
        val spoofingPrefixFlag = isSpoofingPrefix(number, normalized)
        val digitEntropy = calculateDigitEntropy(normalizedDigits.ifBlank { rawDigits })
        val repeatDigitRatio = calculateRepeatDigitRatio(normalizedDigits.ifBlank { rawDigits })
        val maxSameDigitRun = calculateMaxSameDigitRun(normalizedDigits.ifBlank { rawDigits })
        val beautifulNumberFlag = isBeautifulNumber(normalizedDigits.ifBlank { rawDigits })

        val prefixRisk = if (normalized != null) calculatePrefixRisk(normalized) else 0f

        val callFrequency = if (hasCallLogPermission && normalized != null) {
            calculateCallFrequency(normalized)
        } else 0.5f

        val prefixCallFrequency7d = if (hasCallLogPermission && normalized != null) {
            calculatePrefixCallFrequency7d(normalized)
        } else 0

        val isNight = isNightTime()

        val previouslyRejected = if (hasCallLogPermission && normalized != null) {
            // Фильтруем по `verdict = BLOCK` — без этого фича триггерилась на
            // любом повторном звонке от знакомого (включая ALLOW-записи) и
            // ML-модель ошибочно поднимала score до 80%+ для контактов и
            // обычных номеров. Окно — 30 дней (раньше было 24h, что слишком
            // коротко для «реально отклонённого» сигнала, но сейчас фильтр
            // по вердикту делает это безопасным).
            val cutoff = System.currentTimeMillis() - 30L * 24 * 60 * 60_000L
            callRecordDao.countByNumberAndVerdictSince(
                normalized,
                com.antispam.blocker.domain.detector.Verdict.BLOCK.name,
                cutoff
            ) > 0
        } else false

        val callerVerifyFailed = callDetails?.callerNumberVerificationStatus == Connection.VERIFICATION_STATUS_FAILED

        // Recent user context from UsageStats (requires Usage Access permission)
        val recentContext = recentContextProvider.getRecentContext()

        val (vulnerability, business) = profileVector.toFeatureValues()

        val inBlacklist = if (blockListRepo != null && normalized != null) {
            blockListRepo.isBlocked(normalized)
        } else false

        val inAllowlist = if (normalized != null) {
            val byList = blockListRepo?.isAllowed(normalized) == true
            // Юзерский per-prefix override: если префикс в персональном
            // allowlist — считаем номер «своим», даже если конкретный
            // нормализованный номер в allowed_numbers ещё не лежит.
            byList || (feedbackStore?.isPrefixAllowed(normalized) == true)
        } else false
        val reputationScore = calculateLocalReputation(prefixRisk, inBlacklist, inAllowlist, spoofingPrefixFlag, beautifulNumberFlag)
        val sourceConfidence = when {
            inBlacklist || inAllowlist -> 1f
            spoofingPrefixFlag -> 0.85f
            prefixRisk > 0.5f -> 0.65f
            else -> 0.35f
        }

        // --- Phase 3: 15 новых фичей через shipped JSON-лукапы. ---
        val operatorBucket = operatorBucketTable.bucketFor(normalized)
        val defCodeRiskValue = if (normalized != null) defCodeRiskTable.riskFor(normalized) else 0f
        val (prefixBlockShare, prefixWarnShare, prefixSeenLog) =
            if (normalized != null) prefixHistogramTable.lookup(normalized)
            else Triple(0f, 0f, 0f)
        // --- Phase 4B: multi-resolution prefix + def_code×operator cross. ---
        val prefixBlockShare3 =
            if (normalized != null && !inAllowlist) prefixHistogram3Table.blockShareFor(normalized) else 0f
        val prefixBlockShare7 =
            if (normalized != null && !inAllowlist) prefixHistogram7Table.blockShareFor(normalized) else 0f
        val prefixEntropy =
            if (normalized != null) prefixHistogramTable.entropyFor(normalized) else 0f
        val defCodeOperatorRiskValue =
            if (normalized != null && !inAllowlist) defCodeOperatorRiskTable.riskFor(normalized, operatorBucket) else 0f
        val prefixSampleSize =
            if (normalized != null) prefixHistogramTable.sampleSizeFor(normalized) else 0f
        // Reputation/категории доступны только при онлайн-метаданных. На устройстве
        // оффлайн всегда 0 → noMetadata=1, что и есть «true cold-start» сигнал,
        // на котором обучен Phase 3 student.
        val reviewsLog = 0f
        val negativeRatio = 0f
        val searchVolumeLog = 0f
        val hasFraudCategory = false
        val hasTelemarketingCategory = false
        // `noMetadata` отражает отсутствие *онлайн*-репутационных сигналов и
        // категорий, и НЕ должен зависеть от блок-/вайт-листов или rule-based
        // `reputationScore`. На устройстве это всегда true (нет онлайн-скрейпа
        // отзывов/категорий), но условие оставлено явным — если когда-то
        // появится on-device кэш онлайн-репутации, флаг автоматически
        // переключится в false и student-MLP перейдёт в warm-режим.
        // SpamModel.predict() дополнительно проверяет `!inAllowlist && !inBlacklist`
        // прежде чем использовать cold_thresholds — это держит cold/warm
        // решение независимым от наличия списка.
        val noMetadata = reviewsLog == 0f && negativeRatio == 0f &&
            searchVolumeLog == 0f && !hasFraudCategory && !hasTelemarketingCategory

        return CallFeatures(
            isContact = isContact,
            isRussianNumber = isRussian,
            isForeignNumber = isForeign,
            isShortCode = isShort,
            isStandardLen = isStandardLen,
            isTollFree8800 = isTollFree8800,
            isGeographical = isGeographical,
            isMobileRu = isMobileRu,
            isValidRuRange = isValidRuRange,
            spoofingPrefixFlag = spoofingPrefixFlag,
            digitEntropy = digitEntropy,
            repeatDigitRatio = repeatDigitRatio,
            maxSameDigitRun = maxSameDigitRun,
            beautifulNumberFlag = beautifulNumberFlag,
            prefixRisk = prefixRisk,
            callFrequency = callFrequency,
            isNightTime = isNight,
            recentBankApp = recentContext.recentBankApp,
            recentGovApp = recentContext.recentGovApp,
            recentMarketplaceApp = recentContext.recentMarketplaceApp,
            recentMessengerApp = recentContext.recentMessengerApp,
            previouslyRejected = previouslyRejected,
            inBlacklist = inBlacklist,
            inAllowlist = inAllowlist,
            hiddenNumber = isHidden,
            callerVerifyFailed = callerVerifyFailed,
            userVulnerability = vulnerability,
            userBusinessActivity = business,
            contactsAvailable = hasContactsPermission,
            usageAccessAvailable = recentContextProvider.isUsageAccessGranted(),
            reputationScore = reputationScore,
            sourceConfidence = sourceConfidence,
            // Phase 3 features (15):
            operatorMts = operatorBucket == OperatorBucketTable.BUCKET_MTS,
            operatorMegafon = operatorBucket == OperatorBucketTable.BUCKET_MEGAFON,
            operatorBeeline = operatorBucket == OperatorBucketTable.BUCKET_BEELINE,
            operatorTele2 = operatorBucket == OperatorBucketTable.BUCKET_TELE2,
            operatorMvno = operatorBucket == OperatorBucketTable.BUCKET_MVNO,
            defCodeRisk = defCodeRiskValue,
            prefixBlockShare = prefixBlockShare,
            prefixWarnShare = prefixWarnShare,
            prefixSeenLog = prefixSeenLog,
            reviewsLog = reviewsLog,
            negativeRatio = negativeRatio,
            searchVolumeLog = searchVolumeLog,
            hasFraudCategory = hasFraudCategory,
            hasTelemarketingCategory = hasTelemarketingCategory,
            noMetadata = noMetadata,
            // Phase 4B features (5):
            prefixBlockShare3 = prefixBlockShare3,
            prefixBlockShare7 = prefixBlockShare7,
            prefixEntropy = prefixEntropy,
            defCodeOperatorRisk = defCodeOperatorRiskValue,
            prefixSampleSize = prefixSampleSize,
            prefixCallFrequency7d = prefixCallFrequency7d
        )
    }

    private fun checkIsContact(normalizedNumber: String): Boolean {
        // Сначала пробуем in-memory кэш (`ContactsCache`) — он прогревается в
        // SpamCallScreeningService.onCreate() и обновляется по ContentObserver
        // на ContactsContract.Contacts. Это убирает ContentResolver.query на
        // hot-path входящих звонков. Если кэш ещё не готов (первичный bind не
        // завершился) или разрешение `READ_CONTACTS` отозвано — `contains()`
        // возвращает null и мы падаем на прежний путь через ContentResolver.
        ContactsCache.contains(normalizedNumber)?.let { return it }

        val uri = android.provider.ContactsContract.CommonDataKinds.Phone.CONTENT_URI
        val projection = arrayOf(android.provider.ContactsContract.CommonDataKinds.Phone.NORMALIZED_NUMBER)
        val selection = "${android.provider.ContactsContract.CommonDataKinds.Phone.NORMALIZED_NUMBER} = ?"
        context.contentResolver.query(uri, projection, selection, arrayOf(normalizedNumber), null)?.use { cursor ->
            return cursor.moveToFirst()
        }
        return false
    }

    private val prefixRiskTable = PrefixRiskTable.get(context)
    // Phase 3 lookup tables.
    private val operatorBucketTable = OperatorBucketTable.get(context)
    private val defCodeRiskTable = DefCodeRiskTable.get(context)
    private val prefixHistogramTable = PrefixHistogramTable.get(context)
    // Phase 4B lookup tables.
    private val prefixHistogram3Table =
        PrefixHistogramTable.get(context, PrefixHistogramTable.ASSET_PATH_3)
    private val prefixHistogram7Table =
        PrefixHistogramTable.get(context, PrefixHistogramTable.ASSET_PATH_7)
    private val defCodeOperatorRiskTable = DefCodeOperatorRiskTable.get(context)

    private fun calculatePrefixRisk(normalized: String): Float =
        prefixRiskTable.riskFor(normalized)

    private suspend fun calculateCallFrequency(normalized: String): Float {
        // Окно 7 дней: ловит «уже звонил недавно» паттерн как для друзей/знакомых,
        // так и для назойливых спамеров. Делитель 10 — нормировка к [0..1]:
        // 10+ звонков за неделю с одного номера → 1.0.
        val since = System.currentTimeMillis() - 7L * 24 * 60 * 60_000L
        val count = callRecordDao.countByNumberSince(normalized, since)
        return (count / 10f).coerceIn(0f, 1f)
    }

    /**
     * Кол-во звонков с того же DEF-кода (первые 4 цифры после +7) за 7 дней.
     * +7 961 1xx-xxxx → префикс «+7961», ловит спам-волны вида
     * «несколько разных номеров с одного DEF-кода за короткий период».
     * Возвращает абсолютный счёт, не нормализован — даёт рулу `SmartSpamDetector`
     * принять решение по порогу.
     */
    private suspend fun calculatePrefixCallFrequency7d(normalized: String): Int {
        if (!normalized.startsWith("+7") || normalized.length < 5) return 0
        val prefix = normalized.take(5)
        val since = System.currentTimeMillis() - 7L * 24 * 60 * 60_000L
        return callRecordDao.countByPrefixSince("$prefix%", since)
    }

    private fun isNightTime(): Boolean {
        val hour = Calendar.getInstance().get(Calendar.HOUR_OF_DAY)
        return hour >= 22 || hour < 8
    }

    private fun getRuDefCode(normalized: String?): Int {
        if (normalized == null || !normalized.startsWith("+7")) return -1
        val digits = normalized.filter { it.isDigit() }
        if (digits.length < 4) return -1
        return digits.substring(1, 4).toIntOrNull() ?: -1
    }

    private fun isSpoofingPrefix(raw: String?, normalized: String?): Boolean {
        val cleanedRaw = raw.orEmpty().replace(Regex("[\\s\\-().]"), "")
        val rawDigits = cleanedRaw.filter { it.isDigit() }
        val normalizedDigits = normalized.orEmpty().filter { it.isDigit() }
        return (cleanedRaw.startsWith("+84") && rawDigits.startsWith("8495")) ||
                (rawDigits.startsWith("008495")) ||
                (normalized.orEmpty().startsWith("+84") && normalizedDigits.startsWith("8495"))
    }

    private fun calculateDigitEntropy(digits: String): Float {
        if (digits.isBlank()) return 0f
        val counts = digits.groupingBy { it }.eachCount()
        val entropy = counts.values.sumOf { count ->
            val p = count.toDouble() / digits.length
            -p * (ln(p) / ln(2.0))
        }
        return (entropy / (ln(10.0) / ln(2.0))).toFloat().coerceIn(0f, 1f)
    }

    private fun calculateRepeatDigitRatio(digits: String): Float {
        if (digits.length <= 1) return 0f
        val repeats = (1 until digits.length).count { digits[it] == digits[it - 1] }
        return (repeats.toFloat() / (digits.length - 1)).coerceIn(0f, 1f)
    }

    private fun calculateMaxSameDigitRun(digits: String): Float {
        if (digits.isBlank()) return 0f
        var best = 1
        var current = 1
        for (i in 1 until digits.length) {
            if (digits[i] == digits[i - 1]) {
                current++
                if (current > best) best = current
            } else {
                current = 1
            }
        }
        return (best.toFloat() / digits.length).coerceIn(0f, 1f)
    }

    private fun isBeautifulNumber(digits: String): Boolean {
        if (digits.isBlank()) return false
        val tail = digits.takeLast(7)
        val hasPairPattern = Regex("(\\d{2,3})\\1+").containsMatchIn(tail)
        val hasLongRun = calculateMaxSameDigitRun(tail) >= (4f / tail.length.coerceAtLeast(1))
        val hasManyRepeats = calculateRepeatDigitRatio(tail) >= 0.45f
        return hasPairPattern || hasLongRun || hasManyRepeats
    }

    private fun calculateLocalReputation(
        prefixRisk: Float,
        inBlacklist: Boolean,
        inAllowlist: Boolean,
        spoofingPrefixFlag: Boolean,
        beautifulNumberFlag: Boolean
    ): Float {
        if (inAllowlist) return 0f
        if (inBlacklist) return 1f
        var score = prefixRisk * 0.55f
        if (spoofingPrefixFlag) score += 0.35f
        if (beautifulNumberFlag) score += 0.1f
        return score.coerceIn(0f, 1f)
    }
}
