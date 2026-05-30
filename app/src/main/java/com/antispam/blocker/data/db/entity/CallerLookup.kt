package com.antispam.blocker.data.db.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Кэш-запись определения звонящего.
 *
 * Источников два:
 *   - "offline" — libphonenumber (регион + оператор + тип, работает без сети)
 *   - "2gis"    — 2GIS Catalog API (название организации + адрес, opt-in)
 *
 * Стратегия обновления:
 *   offline-запись живёт [TTL_OFFLINE_MS] (7 дней).
 *   2gis-запись живёт [TTL_ONLINE_MS] (30 дней).
 *   Отрицательный кэш (found=false) — [TTL_NEGATIVE_MS] (2 дня), чтобы
 *   при появлении номера в базе 2GIS он подтянулся без долгого ожидания.
 */
@Entity(
    tableName = "caller_lookup",
    indices = [Index(value = ["normalizedNumber"], unique = true)]
)
data class CallerLookup(
    @PrimaryKey
    val normalizedNumber: String,

    /** Название организации из 2GIS (null для частных/неизвестных номеров). */
    val orgName: String? = null,

    /** Адрес организации из 2GIS. */
    val address: String? = null,

    /** Регион из geocoder: "Москва", "Санкт-Петербург", "Россия", … */
    val region: String? = null,

    /** Оператор из carrier mapper: "МТС", "Билайн", "МегаФон", … */
    val carrier: String? = null,

    /**
     * Тип номера: "mobile", "fixed_line", "toll_free", "premium_rate",
     * "shared_cost", "voip", "unknown".
     */
    val numberType: String = "unknown",

    /**
     * Источник последнего успешного lookup:
     * "offline" | "2gis" | "negative" (номер не найден в 2GIS).
     */
    val source: String = "offline",

    /** Timestamp последнего lookup в миллисекундах. */
    val lookedUpAt: Long = System.currentTimeMillis(),
) {
    companion object {
        const val TTL_ONLINE_MS  = 30L * 24 * 3_600_000  // 30 дней
        const val TTL_OFFLINE_MS =  7L * 24 * 3_600_000  // 7 дней
        const val TTL_NEGATIVE_MS = 2L * 24 * 3_600_000  // 2 дня (негативный кэш)
    }

    fun isStale(nowMs: Long = System.currentTimeMillis()): Boolean {
        val ttl = when (source) {
            "2gis"     -> TTL_ONLINE_MS
            "negative" -> TTL_NEGATIVE_MS
            else       -> TTL_OFFLINE_MS
        }
        return nowMs - lookedUpAt > ttl
    }
}
