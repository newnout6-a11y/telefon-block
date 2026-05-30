package com.antispam.blocker.data.cache

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.ContactsContract
import android.util.Log
import java.util.concurrent.ConcurrentHashMap

/**
 * Lightweight contact-name resolver for UI strings.
 *
 * Журнал звонков и карточка во вкладке «ИИ» должны показывать имя из
 * адресной книги (например «Мама») вместо сырого `+7912…`. Сама
 * [ContactsCache] хранит только нормализованные номера — расширять её до
 * `Map<String, String>` неудобно: она прогревается на hot-path
 * `SpamCallScreeningService.onCreate` и любое изменение её shape'а влияет
 * на скрининг. Здесь — отдельный простой LRU-кэш с lookup'ом через
 * `ContactsContract.PhoneLookup`, который читает справочник по индексу и
 * сам нормализует номер. Один промах заполняет кэш, дальше всё O(1).
 *
 * Пример:
 * ```
 * val displayName = ContactNameLookup.resolveOrNull(context, "+79991234567")
 *     ?: rawNumber  // fallback на номер если контакт не найден
 * ```
 *
 * Кэш сбрасывается через [invalidate] при изменении адресной книги
 * (в [ContactsCache] есть свой ContentObserver — повесим инвалидацию
 * туда же без жёсткой связки между классами).
 */
object ContactNameLookup {

    private const val TAG = "ContactNameLookup"

    /** Карта `normalized number → display name`. Sentinel `""` означает «не найден». */
    private val cache = ConcurrentHashMap<String, String>()

    /**
     * Возвращает имя контакта по номеру или `null`, если контакта нет /
     * нет разрешения / номер пустой. Никогда не бросает.
     *
     * UI вызывает это синхронно из Composable; чтение `PhoneLookup` —
     * один лёгкий cursor query по индексу, так что для журнала из
     * нескольких десятков строк задержки не видно. Если потребуется
     * полностью убрать главный поток — вынеси в `LaunchedEffect` с
     * `rememberSaveable` кэшем имён.
     */
    fun resolveOrNull(context: Context, number: String?): String? {
        if (number.isNullOrBlank()) return null
        cache[number]?.let { cached ->
            return cached.ifEmpty { null }
        }
        val granted = context.checkSelfPermission(Manifest.permission.READ_CONTACTS) ==
            PackageManager.PERMISSION_GRANTED
        if (!granted) return null

        val name = lookupInternal(context, number)
        cache[number] = name.orEmpty() // sentinel "" = «известно что не найден»
        return name
    }

    /** Сбрасывает кэш — вызвать при изменении контактов. */
    fun invalidate() {
        cache.clear()
    }

    private fun lookupInternal(context: Context, number: String): String? {
        return try {
            val uri = Uri.withAppendedPath(
                ContactsContract.PhoneLookup.CONTENT_FILTER_URI,
                Uri.encode(number)
            )
            val projection = arrayOf(ContactsContract.PhoneLookup.DISPLAY_NAME)
            context.contentResolver.query(uri, projection, null, null, null)?.use { c ->
                if (c.moveToFirst()) {
                    val idx = c.getColumnIndex(ContactsContract.PhoneLookup.DISPLAY_NAME)
                    if (idx >= 0) c.getString(idx) else null
                } else null
            }
        } catch (t: Throwable) {
            Log.w(TAG, "PhoneLookup failed for masked number", t)
            null
        }
    }
}
