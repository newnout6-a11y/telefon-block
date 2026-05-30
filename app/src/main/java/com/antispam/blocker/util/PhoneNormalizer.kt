package com.antispam.blocker.util

import com.google.i18n.phonenumbers.PhoneNumberUtil
import java.util.Locale

object PhoneNormalizer {

    private val phoneUtil = PhoneNumberUtil.getInstance()

    private val defaultRegion: String by lazy {
        Locale.getDefault().country.ifEmpty { "RU" }
    }

    fun normalize(rawNumber: String?): String? {
        if (rawNumber.isNullOrBlank()) return null

        // Ранний отсев явного мусора: строка должна состоять в основном
        // из цифр, +, пробелов, дефисов и скобок. Если есть буквы или
        // HTML/JS-символы (< > / " etc.) — это не номер.
        if (!looksLikePhone(rawNumber)) return null

        // Короткие номера (экстренные службы, городские) — нормализуем как есть
        val digitsOnly = rawNumber.filter { it.isDigit() }
        if (digitsOnly.length in 2..6) {
            return digitsOnly
        }

        return try {
            val parsed = phoneUtil.parse(rawNumber, defaultRegion)
            if (!phoneUtil.isValidNumber(parsed) && !phoneUtil.isPossibleNumber(parsed)) {
                return null
            }
            phoneUtil.format(parsed, PhoneNumberUtil.PhoneNumberFormat.E164)
        } catch (_: Exception) {
            val digits = rawNumber.filter { it.isDigit() || it == '+' }
            if (digitsOnly.length < 7 || digitsOnly.length > 15) return null
            digits.ifBlank { null }
        }
    }

    fun isEmergencyNumber(number: String): Boolean {
        val cleaned = number.replace(Regex("[^\\d]"), "")
        return cleaned in EMERGENCY_NUMBERS
    }

    /** Быстрая проверка: строка должна выглядеть как номер телефона. */
    private fun looksLikePhone(s: String): Boolean {
        val trimmed = s.trim()
        if (trimmed.length > 20) return false
        // разрешённые символы
        val allowed = trimmed.all { c ->
            c.isDigit() || c == '+' || c == ' ' || c == '-' ||
                    c == '(' || c == ')' || c == '.'
        }
        if (!allowed) return false
        val digits = trimmed.count { it.isDigit() }
        return digits in 2..15
    }

    private val EMERGENCY_NUMBERS = setOf("112", "101", "102", "103", "104")
}
