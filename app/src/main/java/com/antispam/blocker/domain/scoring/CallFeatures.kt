package com.antispam.blocker.domain.scoring

data class CallFeatures(
    val isContact: Boolean = false,
    val isRussianNumber: Boolean = false,
    val isForeignNumber: Boolean = false,
    val isShortCode: Boolean = false,
    val isStandardLen: Boolean = false,
    val isTollFree8800: Boolean = false,
    val isGeographical: Boolean = false,
    val isMobileRu: Boolean = false,
    val isValidRuRange: Boolean = false,
    val spoofingPrefixFlag: Boolean = false,
    val digitEntropy: Float = 0f,
    val repeatDigitRatio: Float = 0f,
    val maxSameDigitRun: Float = 0f,
    val beautifulNumberFlag: Boolean = false,
    val prefixRisk: Float = 0f,
    val callFrequency: Float = 0f,
    val isNightTime: Boolean = false,
    val recentBankApp: Boolean = false,
    val recentGovApp: Boolean = false,
    val recentMarketplaceApp: Boolean = false,
    val recentMessengerApp: Boolean = false,
    val previouslyRejected: Boolean = false,
    val inBlacklist: Boolean = false,
    val inAllowlist: Boolean = false,
    val hiddenNumber: Boolean = false,
    val callerVerifyFailed: Boolean = false,
    val userVulnerability: Float = 0f,
    val userBusinessActivity: Float = 0f,
    val contactsAvailable: Boolean = false,
    val usageAccessAvailable: Boolean = false,
    val reputationScore: Float = 0f,
    val sourceConfidence: Float = 0f,
    // --- Phase 3 (v3): +15 фичей, total = 47.
    // Operator bucket one-hot (5 фичей): mts/megafon/beeline/tele2/mvno.
    val operatorMts: Boolean = false,
    val operatorMegafon: Boolean = false,
    val operatorBeeline: Boolean = false,
    val operatorTele2: Boolean = false,
    val operatorMvno: Boolean = false,
    // P(BLOCK | def_code), shipped как assets/def_code_risk.json (0..1).
    val defCodeRisk: Float = 0f,
    // 6-digit prefix histogram (0..1) — shipped как assets/prefix_histogram.json.
    val prefixBlockShare: Float = 0f,
    val prefixWarnShare: Float = 0f,
    val prefixSeenLog: Float = 0f,
    // Reputation explicit (0..1, log-normalized).
    val reviewsLog: Float = 0f,
    val negativeRatio: Float = 0f,
    val searchVolumeLog: Float = 0f,
    // Категории.
    val hasFraudCategory: Boolean = false,
    val hasTelemarketingCategory: Boolean = false,
    // Cold-start indicator: 1 если все metadata-фичи нулевые.
    val noMetadata: Boolean = false,
    // --- Phase 4B (v4): +5 cold-survivable фичей, total = 52.
    // Multi-resolution prefix histograms (3-digit/7-digit phone digits после +7).
    val prefixBlockShare3: Float = 0f,
    val prefixBlockShare7: Float = 0f,
    // Shannon-энтропия лейблов на 6-char prefix (precomputed, [0..1]).
    val prefixEntropy: Float = 0f,
    // P(BLOCK | operator_bucket × def_code) — cross feature.
    val defCodeOperatorRisk: Float = 0f,
    // Linear-scaled confidence по seen_count (saturates at ~30).
    val prefixSampleSize: Float = 0f,
    /**
     * Runtime-only field, не входит в ML-вход (модель обучена на 52 фичах).
     * Считает звонки с того же 4-знакового префикса (DEF-код + первая цифра)
     * за последние 7 дней. Используется в SmartSpamDetector как rule-based
     * boost: множество звонков с одного префикса → спам-волна.
     */
    val prefixCallFrequency7d: Int = 0
) {
    /**
     * Сборка 52-мерного входного вектора для student-MLP.
     *
     * Phase 4D — опция `maskColdStart`: зеркалирует `make_cold_view` из
     * `scripts/train_kd_distillation.py` — обнуляет 9 cold-mask фичей
     * (`inAllowlist`, `inBlacklist`, `reputationScore`, `sourceConfidence`,
     * `reviewsLog`, `negativeRatio`, `searchVolumeLog`, `hasFraudCategory`,
     * `hasTelemarketingCategory`) и форсит `noMetadata=1`. Это нужно только
     * при действительном cold-start (ни в листах, ни в онлайн-метадате), чтобы
     * on-device вход совпадал с распределением, на котором калиброваны
     * `cold_thresholds` (Phase 4A) и cold-aug Phase 4C/4D. Без этого
     * rule-based `reputationScore`/`sourceConfidence` с устройства смещают
     * вход в «warm-подобную» зону, и студент ошибочно отвечает
     * `ALLOW≈0.999` даже для явно спамных префиксов.
     */
    fun toFloatArray(maskColdStart: Boolean = false): FloatArray = floatArrayOf(
        if (isContact) 1f else 0f,
        if (isRussianNumber) 1f else 0f,
        if (isForeignNumber) 1f else 0f,
        if (isShortCode) 1f else 0f,
        if (isStandardLen) 1f else 0f,
        if (isTollFree8800) 1f else 0f,
        if (isGeographical) 1f else 0f,
        if (isMobileRu) 1f else 0f,
        if (isValidRuRange) 1f else 0f,
        if (spoofingPrefixFlag) 1f else 0f,
        digitEntropy,
        repeatDigitRatio,
        maxSameDigitRun,
        if (beautifulNumberFlag) 1f else 0f,
        prefixRisk,
        callFrequency,
        if (isNightTime) 1f else 0f,
        if (recentBankApp) 1f else 0f,
        if (recentGovApp) 1f else 0f,
        if (recentMarketplaceApp) 1f else 0f,
        if (recentMessengerApp) 1f else 0f,
        if (previouslyRejected) 1f else 0f,
        if (maskColdStart) 0f else (if (inBlacklist) 1f else 0f),
        if (maskColdStart) 0f else (if (inAllowlist) 1f else 0f),
        if (hiddenNumber) 1f else 0f,
        if (callerVerifyFailed) 1f else 0f,
        userVulnerability,
        userBusinessActivity,
        if (contactsAvailable) 1f else 0f,
        if (usageAccessAvailable) 1f else 0f,
        if (maskColdStart) 0f else reputationScore,
        if (maskColdStart) 0f else sourceConfidence,
        // --- Phase 3 extension (15 features) ---
        if (operatorMts) 1f else 0f,
        if (operatorMegafon) 1f else 0f,
        if (operatorBeeline) 1f else 0f,
        if (operatorTele2) 1f else 0f,
        if (operatorMvno) 1f else 0f,
        defCodeRisk,
        prefixBlockShare,
        prefixWarnShare,
        prefixSeenLog,
        if (maskColdStart) 0f else reviewsLog,
        if (maskColdStart) 0f else negativeRatio,
        if (maskColdStart) 0f else searchVolumeLog,
        if (maskColdStart) 0f else (if (hasFraudCategory) 1f else 0f),
        if (maskColdStart) 0f else (if (hasTelemarketingCategory) 1f else 0f),
        if (maskColdStart) 1f else (if (noMetadata) 1f else 0f),
        // --- Phase 4B extension (5 features) ---
        prefixBlockShare3,
        prefixBlockShare7,
        prefixEntropy,
        defCodeOperatorRisk,
        prefixSampleSize
    )

    companion object {
        const val FEATURE_COUNT = 52
    }
}
