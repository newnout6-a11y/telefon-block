package com.antispam.blocker.domain.lookup

import android.content.Context
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Оффлайн-определение оператора и региона РФ по реестру нумерации РКН.
 *
 * Бинарный ассет phone_lookup.bin (336 KB) собирается скриптом
 * scripts/build_phone_lookup.py из DEF-9xx.csv (opendata.digital.gov.ru).
 *
 * Формат (big-endian):
 *   magic[4]="PLKU" + N[4] + Nop[2] + Nreg[2]
 *   строковая таблица: (Nop + Nreg) строк, каждая = 1-byte-len + UTF-8 байты
 *   N × запись: from_key[8] + to_key[8] + op_idx[2] + reg_idx[2]
 *   ключ = DEF * 10_000_000 + 7-значный-номер-абонента
 *
 * Вызов [load] обязателен до [lookup]. Безопасно вызывать несколько раз.
 */
object RussianPhoneLookup {

    data class Result(val operator: String, val region: String)

    private const val TAG = "RussianPhoneLookup"
    private const val ASSET = "phone_lookup.bin"

    private var operators:  Array<String> = emptyArray()
    private var regions:    Array<String> = emptyArray()
    private var fromKeys:   LongArray  = LongArray(0)
    private var toKeys:     LongArray  = LongArray(0)
    private var opIndices:  ShortArray = ShortArray(0)
    private var regIndices: ShortArray = ShortArray(0)

    @Volatile private var loaded = false

    /**
     * Загружает ассет в память (~344 KB).
     * Идемпотентен: повторный вызов игнорируется.
     * Рекомендуется вызывать из фонового потока / [SpamBlockerApp.onCreate]
     * через coroutine Dispatchers.IO.
     */
    @Synchronized
    fun load(context: Context) {
        if (loaded) return
        try {
            val bytes = context.assets.open(ASSET).use { it.readBytes() }
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)

            val magic = ByteArray(4).also { buf.get(it) }
            check(magic.contentEquals(byteArrayOf('P'.code.toByte(), 'L'.code.toByte(),
                                                   'K'.code.toByte(), 'U'.code.toByte()))) {
                "Bad magic in $ASSET"
            }

            val n    = buf.int
            val nOp  = buf.short.toInt() and 0xFFFF
            val nReg = buf.short.toInt() and 0xFFFF

            fun readStrings(count: Int): Array<String> = Array(count) {
                val len = buf.get().toInt() and 0xFF
                val b = ByteArray(len).also { buf.get(it) }
                String(b, Charsets.UTF_8)
            }

            operators  = readStrings(nOp)
            regions    = readStrings(nReg)
            // Записи в файле interleaved: from(8)+to(8)+op(2)+reg(2) на строку.
            // Читаем построчно, иначе from/to/op/reg перемешиваются.
            fromKeys   = LongArray(n)
            toKeys     = LongArray(n)
            opIndices  = ShortArray(n)
            regIndices = ShortArray(n)
            for (i in 0 until n) {
                fromKeys[i]   = buf.long
                toKeys[i]     = buf.long
                opIndices[i]  = buf.short
                regIndices[i] = buf.short
            }

            loaded = true
            Log.d(TAG, "loaded $n entries, $nOp operators, $nReg regions")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load $ASSET — will use libphonenumber fallback", e)
        }
    }

    /**
     * Ищет оператора и регион по номеру в E.164-формате.
     *
     * @param e164 Нормализованный номер, например "+79161234567".
     * @return [Result] или null, если номер не для RU, формат неверный,
     *         ассет не загружен или диапазон не найден.
     */
    fun lookup(e164: String): Result? {
        if (!loaded) return null

        val digits = if (e164.startsWith("+")) e164.drop(1) else e164
        // Только российские мобильные: 79XXXXXXXXX (11 цифр, DEF начинается с 9)
        if (digits.length < 11 || !digits.startsWith("7")) return null
        val defCode = digits.substring(1, 4).toLongOrNull() ?: return null
        if (defCode < 900 || defCode > 999) return null   // только мобильные DEF
        val subNum  = digits.substring(4, 11).toLongOrNull() ?: return null
        val key     = defCode * 10_000_000L + subNum

        // Бинарный поиск по диапазонам
        var lo = 0; var hi = fromKeys.size - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            when {
                key < fromKeys[mid] -> hi = mid - 1
                key > toKeys[mid]   -> lo = mid + 1
                else -> {
                    val op  = operators[opIndices[mid].toInt() and 0xFFFF]
                    val reg = regions[regIndices[mid].toInt() and 0xFFFF]
                    return Result(op, reg)
                }
            }
        }
        return null
    }
}
