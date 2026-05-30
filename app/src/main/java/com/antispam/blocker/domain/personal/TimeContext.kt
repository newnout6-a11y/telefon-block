package com.antispam.blocker.domain.personal

import java.util.Calendar

/**
 * Time-of-call context derived from system clock.
 *
 * Used downstream by the Device_Model feature extractor to compute the
 * `is_night_time` and `is_weekend` features (Requirement 1.6).
 *
 * @property hour Hour-of-day in `[0..23]` (24-hour clock, system default time zone).
 * @property dayOfWeek Day-of-week in `[1..7]` matching [Calendar.SUNDAY]..[Calendar.SATURDAY].
 * @property isWeekend `true` when [dayOfWeek] is [Calendar.SATURDAY] or [Calendar.SUNDAY].
 */
data class TimeContext(
    val hour: Int,
    val dayOfWeek: Int,
    val isWeekend: Boolean,
) {
    companion object {
        /**
         * Derives a [TimeContext] from the given epoch-millisecond timestamp using the
         * system default time zone via [Calendar].
         */
        fun derive(t: Long): TimeContext {
            val calendar = Calendar.getInstance().apply { timeInMillis = t }
            val hour = calendar.get(Calendar.HOUR_OF_DAY)
            val dayOfWeek = calendar.get(Calendar.DAY_OF_WEEK)
            val isWeekend = dayOfWeek == Calendar.SATURDAY || dayOfWeek == Calendar.SUNDAY
            return TimeContext(
                hour = hour,
                dayOfWeek = dayOfWeek,
                isWeekend = isWeekend,
            )
        }
    }
}
