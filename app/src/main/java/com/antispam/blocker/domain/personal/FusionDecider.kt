package com.antispam.blocker.domain.personal

import com.antispam.blocker.domain.detector.Verdict

/**
 * Verdict produced by the on-device personal model.
 *
 * - [ALLOW]: device model is confident the call is legitimate.
 * - [WARN]: device model is uncertain.
 * - [BLOCK_HIGH]: device model is confidently flagging the call as spam
 *   (only this level of confidence is allowed to vote for BLOCK in fusion).
 *
 * Lives in the same file as [FusionDecider] for cohesion: the fusion
 * decision is the only consumer of this enum besides the Device_Model
 * itself.
 */
enum class DeviceVerdict {
    ALLOW,
    WARN,
    BLOCK_HIGH,
}

/**
 * Pure, stateless combiner of cloud + device verdicts and user
 * whitelist/blacklist overrides into the final [Verdict] used by
 * `SpamCallScreeningService`.
 *
 * Algorithm: **consensus-required BLOCK + Device_Veto on ALLOW**
 * (см. requirements.md Req 5.1–5.12 и design.md §"FusionDecider").
 *
 * Ключевая идея: false BLOCK строго дороже false ALLOW (Req 4.9), а
 * Cloud_Model работает на 52 фичах без доступа к персональной телеметрии
 * (контакты, история ответов, OUTGOING after MISSED). Поэтому одиночный
 * Cloud BLOCK без подтверждения от Device_Model **не имеет права**
 * заблокировать звонок — он эскалируется максимум до WARN. И наоборот:
 * Device_Model имеет veto на Cloud BLOCK, если у неё на руках сильный
 * персональный сигнал «свой» (`is_contact = -3.0`, высокий `answer_rate`,
 * `prev_outgoing_after_missed = -1.5` → итоговый `probBlock < WARN_THRESHOLD`).
 *
 * Шаги:
 *
 * 1. whitelist override → ALLOW
 * 2. blacklist override → BLOCK
 * 3. Warm_Up not complete OR no device vote → fall back to Cloud_Model only.
 *    В этом fallback одиночный Cloud BLOCK может стать финальным BLOCK,
 *    иначе устройство первые 14 дней не сможет блокировать вообще ничего.
 * 4. consensus BLOCK: cloud BLOCK ≥ HIGH_CONFIDENCE И device BLOCK_HIGH → BLOCK
 * 5. Device_Veto: cloud BLOCK ≥ HIGH_CONFIDENCE И device ALLOW → ALLOW
 *    (Device_Model видит контакт / answer rate / OUTGOING after MISSED,
 *    у Cloud этой телеметрии нет — доверяем Device)
 * 6. одиночный BLOCK-голос (cloud-only или device-only) → WARN, не BLOCK
 * 7. оба ALLOW → ALLOW
 * 8. иначе (любая другая комбинация WARN / ALLOW) → WARN
 *
 * No I/O, no side effects: тотальная функция, тривиально property-testable.
 */
class FusionDecider {

    data class FusionInput(
        val cloudVerdict: Verdict,
        val cloudConfidence: Float,
        val deviceVerdict: DeviceVerdict?,
        /**
         * Сырая `p(BLOCK)` от Device_Model — нужна, чтобы отличить
         * «реально знает что свой» от «нейтральный вектор на дефолтных
         * весах». На свежей установке `sigmoid(bias = −0.5) ≈ 0.378`,
         * что ниже [DeviceModel.WARN_THRESHOLD] (0.45) → DeviceVerdict.ALLOW,
         * хотя у модели НЕТ персонального сигнала «свой». Без этого поля
         * device-veto перебивает Cloud_Model BLOCK на любых номерах при
         * первом запуске. Veto активируется только когда `probBlock` ниже
         * [STRONG_ALLOW_PROB_THRESHOLD] — это требует, чтобы хотя бы одна
         * из сильных отрицательных фич (is_contact = −3.0,
         * prev_outgoing_after_missed = −1.5, answer_rate × −1.5) реально
         * сработала.
         *
         * `null` → device-вердикт недоступен (не тот же случай, что в
         * `deviceVerdict == null`, но обе ветки fallback'ятся на cloud-only).
         */
        val deviceProbBlock: Float?,
        val isInWhitelist: Boolean,
        val isInBlacklist: Boolean,
        val isWarmUpComplete: Boolean,
    )

    data class FusionOutput(
        val finalVerdict: Verdict,
        val rationaleTag: String,
    )

    fun decide(input: FusionInput): FusionOutput {
        // 1. User whitelist always wins.
        if (input.isInWhitelist) {
            return FusionOutput(Verdict.ALLOW, "whitelist_override")
        }

        // 2. User blacklist wins over both models.
        if (input.isInBlacklist) {
            return FusionOutput(Verdict.BLOCK, "blacklist_override")
        }

        // 3. Warm-up incomplete or no device vote → cloud only.
        //    В этом fallback одиночный Cloud BLOCK эскалируется к финальному
        //    BLOCK как раньше — иначе первые 14 дней / до 30 меток устройство
        //    не сможет блокировать вообще ничего, что хуже false-positive риска.
        if (!input.isWarmUpComplete || input.deviceVerdict == null) {
            return FusionOutput(input.cloudVerdict, "warmup_cloud_only")
        }

        val cloudIsBlock =
            input.cloudVerdict == Verdict.BLOCK && input.cloudConfidence >= HIGH_CONFIDENCE
        val deviceIsBlock = input.deviceVerdict == DeviceVerdict.BLOCK_HIGH
        val deviceIsAllow = input.deviceVerdict == DeviceVerdict.ALLOW

        // 4. Consensus BLOCK — обе модели независимо проголосовали за BLOCK
        //    на высоком уровне уверенности (Req 4.10 «несколько независимых
        //    сигналов»). Только эта ветка может выдать финальный BLOCK по
        //    решению моделей; всё остальное максимум WARN.
        if (cloudIsBlock && deviceIsBlock) {
            return FusionOutput(Verdict.BLOCK, "consensus_block")
        }

        // 5. Device_Veto: Cloud видит спам, Device знает «свой» номер
        //    (`is_contact`, `answer_rate`, `prev_outgoing_after_missed`).
        //    У Cloud_Model нет доступа к этой телеметрии — доверяем Device.
        //    Это и есть основной фикс UX: cloud-only false BLOCK на
        //    врача / курьера / банк из контактов теперь невозможен.
        //
        //    КРИТИЧНО: требуем `probBlock < STRONG_ALLOW_PROB_THRESHOLD`,
        //    а не просто `DeviceVerdict.ALLOW`. На дефолтных весах
        //    `sigmoid(bias = −0.5) ≈ 0.378`, что само по себе уже
        //    DeviceVerdict.ALLOW для нейтрального (нулевого) вектора — без
        //    этого порога veto срабатывал бы на первом же установленном
        //    номере, у которого Device_Model нет ни одного персонального
        //    сигнала. Сильно отрицательные фичи (is_contact = −3.0 →
        //    σ ≈ 0.030, prev_outgoing_after_missed = −1.5 → σ ≈ 0.118)
        //    легко перешагивают порог 0.20 и активируют veto, как и
        //    задумано.
        val deviceStrongAllow = deviceIsAllow &&
            input.deviceProbBlock != null &&
            input.deviceProbBlock < STRONG_ALLOW_PROB_THRESHOLD
        if (cloudIsBlock && deviceStrongAllow) {
            return FusionOutput(Verdict.ALLOW, "device_veto_allow")
        }

        // 6. Одиночный BLOCK-голос (cloud-only без device-подтверждения,
        //    или device-only без cloud-подтверждения, или cloud BLOCK с
        //    confidence < HIGH_CONFIDENCE при device WARN/BLOCK_HIGH) →
        //    понижаем до WARN. Req 4.9: false BLOCK дороже false ALLOW.
        if (cloudIsBlock || deviceIsBlock) {
            return FusionOutput(Verdict.WARN, "single_block_downgrade_warn")
        }

        // 7. ALLOW требует согласия обеих моделей.
        if (input.cloudVerdict == Verdict.ALLOW && input.deviceVerdict == DeviceVerdict.ALLOW) {
            return FusionOutput(Verdict.ALLOW, "agree_allow")
        }

        // 8. Любая другая комбинация (WARN от любой стороны, несогласие
        //    ALLOW vs WARN, и т.п.) → WARN (Req 4.11, 5.6).
        return FusionOutput(Verdict.WARN, "either_warn")
    }

    companion object {
        /**
         * Cloud confidence cutoff above which a Cloud_Model BLOCK verdict
         * is treated as a confident BLOCK vote in fusion. Mirrors
         * `RiskScore.confidence == HIGH` in the existing rule engine.
         */
        const val HIGH_CONFIDENCE = 0.70f

        /**
         * Порог `p(BLOCK)`, ниже которого Device_Model получает право veto'ить
         * Cloud BLOCK. Должен быть **ниже** дефолтного предсказания на
         * нулевом векторе (`σ(bias) ≈ 0.378` при `bias = −0.5`), иначе veto
         * срабатывает на любом номере при первом запуске. Сильные «свой»-
         * сигналы (is_contact, prev_outgoing_after_missed) сдвигают `σ`
         * ниже 0.15, так что 0.20 уверенно пропускает реальные позитивные
         * случаи и режет ложные.
         */
        const val STRONG_ALLOW_PROB_THRESHOLD: Float = 0.20f
    }
}
