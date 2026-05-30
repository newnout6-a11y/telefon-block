package com.antispam.blocker.domain.lookup

import android.util.Log
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

/**
 * Определение звонящего через 2GIS Catalog API (бесплатный tier).
 *
 * Для бизнес-номеров возвращает название и адрес организации.
 * Для мобильных/частных — пустой список (кэшируется как "negative").
 *
 * Требует API-ключ (бесплатная регистрация: dev.2gis.ru).
 * Лимит: 5 000 запросов/день.
 *
 * Privacy: номер передаётся на серверы 2GIS — только opt-in (toggle в Settings).
 */
class TwoGisCallerLookup {

    companion object {
        private const val TAG = "TwoGisCallerLookup"
        private const val BASE_URL = "https://catalog.api.2gis.com/3.0/items"
        private const val CONNECT_MS = 5_000
        private const val READ_MS    = 8_000
    }

    /**
     * @return CallerInfo(source=TWO_GIS) если найдено,
     *         CallerInfo(source=NEGATIVE) если 2GIS вернул пустой список,
     *         null при сетевой / серверной ошибке (не кэшировать).
     * @throws IOException при сетевой ошибке — WorkManager retry.
     */
    @Throws(IOException::class)
    fun lookup(normalizedNumber: String, apiKey: String): CallerInfo? {
        val phone = normalizedNumber.trimStart('+')
        val url = "$BASE_URL?phone=${enc(phone)}" +
            "&fields=items.name,items.address_name" +
            "&key=${enc(apiKey)}"

        val conn = URL(url).openConnection() as HttpURLConnection
        conn.connectTimeout = CONNECT_MS
        conn.readTimeout    = READ_MS
        conn.requestMethod  = "GET"
        conn.setRequestProperty("Accept", "application/json")
        conn.setRequestProperty("User-Agent", "SpamBlockerApp/1.0")

        return try {
            val code = conn.responseCode
            when {
                code == 429 -> { Log.w(TAG, "rate limit"); null }  // не retry сразу
                code != 200 -> throw IOException("HTTP $code")
                else -> {
                    val body = conn.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
                    parseResponse(normalizedNumber, body)
                }
            }
        } finally {
            conn.disconnect()
        }
    }

    private fun parseResponse(normalizedNumber: String, json: String): CallerInfo? {
        return try {
        val result = JSONObject(json).optJSONObject("result")
            ?: return negativeInfo(normalizedNumber)
        val items = result.optJSONArray("items")
        if (items == null || items.length() == 0) return negativeInfo(normalizedNumber)

        val item    = items.getJSONObject(0)
        val name    = item.optString("name").trim().ifEmpty { null }
        val address = item.optString("address_name").trim().ifEmpty { null }
        if (name == null) return negativeInfo(normalizedNumber)

        CallerInfo(
            normalizedNumber = normalizedNumber,
            orgName = name,
            address = address,
            source = CallerInfo.Source.TWO_GIS,
        )
    } catch (t: Throwable) {
        Log.w(TAG, "JSON parse failed", t)
        null
        }
    }

    private fun negativeInfo(n: String) = CallerInfo(n, source = CallerInfo.Source.NEGATIVE)

    private fun enc(s: String): String = URLEncoder.encode(s, "UTF-8")
}
