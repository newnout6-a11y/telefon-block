package com.antispam.blocker.domain.lookup

import android.util.Log
import com.google.i18n.phonenumbers.NumberParseException
import com.google.i18n.phonenumbers.PhoneNumberUtil
import com.google.i18n.phonenumbers.geocoding.PhoneNumberOfflineGeocoder
import com.google.i18n.phonenumbers.PhoneNumberToCarrierMapper
import java.util.Locale

/**
 * Оффлайн-определение номера через реестр РКН + libphonenumber.
 *
 * Стратегия для российских мобильных (+7 9XX):
 *   1. [RussianPhoneLookup] — точный регион из реестра нумерации РКН
 *      (например "Краснодарский край" вместо "Россия"). Должен быть
 *      загружен заранее через [RussianPhoneLookup.load].
 *   2. libphonenumber geocoder — fallback и для всех остальных номеров.
 *   3. libphonenumber carrier mapper — оператор для любых номеров,
 *      не перекрытых реестром (зарубежные и т.д.).
 *
 * Работает полностью без сети.
 */
class OfflineCallerLookup {

    private val phoneUtil     = PhoneNumberUtil.getInstance()
    private val geocoder      = PhoneNumberOfflineGeocoder.getInstance()
    private val carrierMapper = PhoneNumberToCarrierMapper.getInstance()

    companion object {
        private const val TAG = "OfflineCallerLookup"
        private val RU_LOCALE = Locale("ru", "RU")
        // libphonenumber возвращает это для мобильных +7 — слишком обобщённо
        private const val VAGUE_RU = "Россия"
    }

    /**
     * Синхронный lookup. Никогда не бросает.
     *
     * @param normalizedNumber E.164-нормализованный номер (+79991234567).
     * @return [CallerInfo] или null, если парсинг не удался / ничего не определено.
     */
    fun lookup(normalizedNumber: String): CallerInfo? {
        return try {
            val parsed = phoneUtil.parse(normalizedNumber, "ZZ")

            // ── 1. Реестр РКН: точный регион + нормализованный оператор ──────
            val rknResult = RussianPhoneLookup.lookup(normalizedNumber)

            // ── 2. libphonenumber geocoder ──────────────────────────────────
            val geocoderRegion = geocoder
                .getDescriptionForNumber(parsed, RU_LOCALE)
                .trim()
                .ifEmpty { null }

            // Предпочитаем РКН, если он дал что-то конкретнее "Россия"
            val region: String? = when {
                rknResult != null                          -> rknResult.region
                geocoderRegion != VAGUE_RU                -> geocoderRegion
                else                                       -> geocoderRegion
            }

            // ── 3. Оператор: РКН → libphonenumber carrier ──────────────────
            val carrierLib = carrierMapper
                .getNameForNumber(parsed, RU_LOCALE)
                .trim()
                .ifEmpty { null }

            val carrier: String? = rknResult?.operator ?: carrierLib

            val numberType = when (phoneUtil.getNumberType(parsed)) {
                PhoneNumberUtil.PhoneNumberType.MOBILE,
                PhoneNumberUtil.PhoneNumberType.FIXED_LINE_OR_MOBILE -> CallerInfo.NumberType.MOBILE
                PhoneNumberUtil.PhoneNumberType.FIXED_LINE            -> CallerInfo.NumberType.FIXED_LINE
                PhoneNumberUtil.PhoneNumberType.TOLL_FREE             -> CallerInfo.NumberType.TOLL_FREE
                PhoneNumberUtil.PhoneNumberType.PREMIUM_RATE          -> CallerInfo.NumberType.PREMIUM_RATE
                PhoneNumberUtil.PhoneNumberType.SHARED_COST           -> CallerInfo.NumberType.SHARED_COST
                PhoneNumberUtil.PhoneNumberType.VOIP                  -> CallerInfo.NumberType.VOIP
                else                                                   -> CallerInfo.NumberType.UNKNOWN
            }

            if (region == null && carrier == null && numberType == CallerInfo.NumberType.UNKNOWN) {
                return null
            }

            CallerInfo(
                normalizedNumber = normalizedNumber,
                region = region,
                carrier = carrier,
                numberType = numberType,
                source = CallerInfo.Source.OFFLINE,
            )
        } catch (e: NumberParseException) {
            Log.d(TAG, "Cannot parse number: ${e.message}")
            null
        } catch (t: Throwable) {
            Log.w(TAG, "Unexpected error", t)
            null
        }
    }
}
