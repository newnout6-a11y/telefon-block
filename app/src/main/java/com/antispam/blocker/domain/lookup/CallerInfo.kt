package com.antispam.blocker.domain.lookup

import com.antispam.blocker.data.db.entity.CallerLookup

/**
 * Результат определения звонящего — view-модель поверх [CallerLookup].
 *
 * Используется в UI и Repository; не зависит от Room напрямую.
 */
data class CallerInfo(
    val normalizedNumber: String,
    /** Название организации (из 2GIS). Null для частных/мобильных номеров. */
    val orgName: String? = null,
    /** Адрес организации (из 2GIS). */
    val address: String? = null,
    /** Регион из geocoder: "Москва", "Санкт-Петербург", "Россия", ... */
    val region: String? = null,
    /** Оператор: "МТС", "Билайн", "МегаФон" и т.д. */
    val carrier: String? = null,
    val numberType: NumberType = NumberType.UNKNOWN,
    val source: Source = Source.OFFLINE,
) {
    enum class NumberType(val displayRu: String) {
        MOBILE("мобильный"),
        FIXED_LINE("городской"),
        TOLL_FREE("бесплатный"),
        PREMIUM_RATE("платный"),
        SHARED_COST("общий"),
        VOIP("VoIP"),
        UNKNOWN("неизвестный"),
    }

    enum class Source { OFFLINE, TWO_GIS, NEGATIVE }

    /**
     * Строка-подпись для второй строки в журнале звонков.
     *
     * Примеры:
     *   "Пятёрочка, ул. Ленина 5"  (2GIS с адресом)
     *   "МТС, Москва"               (оффлайн с оператором и регионом)
     *   "мобильный, Москва"         (оффлайн без оператора)
     *   null                        (ничего не определено)
     */
    val subtitle: String?
        get() {
            if (orgName != null) {
                return listOfNotNull(orgName, address).joinToString(", ").ifEmpty { null }
            }
            val typeLabel = if (numberType != NumberType.UNKNOWN) numberType.displayRu else null
            val parts = listOfNotNull(carrier ?: typeLabel, region)
            return parts.joinToString(", ").ifEmpty { null }
        }
}

// ── Маппинг entity <-> domain ────────────────────────────────────────────────

fun CallerLookup.toCallerInfo() = CallerInfo(
    normalizedNumber = normalizedNumber,
    orgName = orgName,
    address = address,
    region = region,
    carrier = carrier,
    numberType = when (numberType) {
        "mobile"       -> CallerInfo.NumberType.MOBILE
        "fixed_line"   -> CallerInfo.NumberType.FIXED_LINE
        "toll_free"    -> CallerInfo.NumberType.TOLL_FREE
        "premium_rate" -> CallerInfo.NumberType.PREMIUM_RATE
        "shared_cost"  -> CallerInfo.NumberType.SHARED_COST
        "voip"         -> CallerInfo.NumberType.VOIP
        else           -> CallerInfo.NumberType.UNKNOWN
    },
    source = when (source) {
        "2gis"     -> CallerInfo.Source.TWO_GIS
        "negative" -> CallerInfo.Source.NEGATIVE
        else       -> CallerInfo.Source.OFFLINE
    },
)

fun CallerInfo.toEntity(nowMs: Long = System.currentTimeMillis()) = CallerLookup(
    normalizedNumber = normalizedNumber,
    orgName = orgName,
    address = address,
    region = region,
    carrier = carrier,
    numberType = when (numberType) {
        CallerInfo.NumberType.MOBILE       -> "mobile"
        CallerInfo.NumberType.FIXED_LINE   -> "fixed_line"
        CallerInfo.NumberType.TOLL_FREE    -> "toll_free"
        CallerInfo.NumberType.PREMIUM_RATE -> "premium_rate"
        CallerInfo.NumberType.SHARED_COST  -> "shared_cost"
        CallerInfo.NumberType.VOIP         -> "voip"
        else                               -> "unknown"
    },
    source = when (source) {
        CallerInfo.Source.TWO_GIS  -> "2gis"
        CallerInfo.Source.NEGATIVE -> "negative"
        else                       -> "offline"
    },
    lookedUpAt = nowMs,
)
