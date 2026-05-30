package com.antispam.blocker.data.cache

import android.Manifest
import android.content.ContentResolver
import android.content.Context
import android.content.pm.PackageManager
import android.database.ContentObserver
import android.net.Uri
import android.os.Handler
import android.os.HandlerThread
import android.provider.ContactsContract
import android.util.Log
import com.antispam.blocker.util.PhoneNormalizer
import java.util.concurrent.atomic.AtomicReference

/**
 * In-memory snapshot of contact phone numbers, keyed by E.164-normalized form.
 *
 * Why a singleton with eager pre-load:
 *   `FeatureExtractor.checkIsContact()` ранее на каждый incoming звонок
 *   делал `ContentResolver.query(ContactsContract.CommonDataKinds.Phone…)`,
 *   что добавляло видимую задержку к scoring-pipeline (особенно на
 *   устройствах с большим телефонным справочником и плохим IO). Этот кэш
 *   разово прогружается при старте `SpamCallScreeningService` и затем
 *   обновляется реактивно по `ContentObserver` на
 *   `ContactsContract.Contacts.CONTENT_URI` — изменения справочника (новый
 *   контакт, переименование, удаление) попадают в кэш в фоне.
 *
 * Permission-aware:
 *   Если у приложения нет `READ_CONTACTS`, [init] всё равно регистрирует
 *   observer (на случай если разрешение появится позже), но первичная
 *   загрузка пропускается, и [contains] возвращает `null` —
 *   FeatureExtractor падает на безопасный default (`isContact = false`).
 *
 * Thread safety:
 *   Состояние держится в `AtomicReference<Set<String>>`. Reads — lock-free.
 *   Refresh идёт в выделенный `HandlerThread`, запись атомарна.
 */
object ContactsCache {

    private const val TAG = "ContactsCache"

    /** `null` — кэш ещё не прогружен (первичная загрузка не успела завершиться). */
    private val snapshot = AtomicReference<Set<String>?>(null)

    @Volatile private var initialized = false
    @Volatile private var observer: ContentObserver? = null
    @Volatile private var workerThread: HandlerThread? = null

    /**
     * Регистрирует observer и запускает первичную загрузку. Идемпотентен:
     * повторные вызовы — no-op. Безопасно вызывать из `Service.onCreate()`.
     */
    @Synchronized
    fun init(context: Context) {
        if (initialized) return
        initialized = true

        val appContext = context.applicationContext
        val thread = HandlerThread("ContactsCache").also { it.start() }
        workerThread = thread
        val handler = Handler(thread.looper)

        val resolver = appContext.contentResolver
        val obs = object : ContentObserver(handler) {
            override fun onChange(selfChange: Boolean, uri: Uri?) {
                handler.post {
                    // При любом изменении справочника сбрасываем кэш имён
                    // (UI журнала) — иначе переименованный контакт продолжит
                    // отображаться по старому имени до перезапуска приложения.
                    ContactNameLookup.invalidate()
                    reloadInternal(appContext)
                }
            }
        }
        runCatching {
            resolver.registerContentObserver(
                ContactsContract.Contacts.CONTENT_URI,
                /* notifyForDescendants = */ true,
                obs
            )
        }.onFailure { Log.w(TAG, "register observer failed", it) }
        observer = obs

        handler.post { reloadInternal(appContext) }
    }

    /**
     * Быстрый O(1) контактный lookup.
     *
     * @return `true` — номер найден среди контактов; `false` — точно нет;
     *         `null` — кэш ещё не готов или разрешение не выдано, нужно
     *         падать на ContentResolver-fallback.
     */
    fun contains(normalizedNumber: String?): Boolean? {
        if (normalizedNumber.isNullOrBlank()) return false
        val set = snapshot.get() ?: return null
        return set.contains(normalizedNumber)
    }

    /**
     * Размер прогретого in-memory кэша.
     *
     * @return -1 если кэш ещё не загружен (первичный bind не завершился
     *         или `READ_CONTACTS` не выдан), иначе — количество
     *         нормализованных номеров в адресной книге.
     *
     * Используется UI Settings → Privacy для строки «Контактов прогрето: N»,
     * чтобы пользователь видел, что приложение реально читает справочник
     * (а не просто проставило галку разрешения).
     */
    fun size(): Int = snapshot.get()?.size ?: -1

    /**
     * Маскированный sample из кэша — возвращает [n] нормализованных номеров,
     * по которым пользователь сможет визуально подтвердить, что приложение
     * реально прочитало его адресную книгу. Для приватности номера маскируются:
     * `+7XXX•••XX12` (видны только страновой код, кусок DEF и последние 2
     * цифры). Кэш — `Set<String>`, порядок не гарантирован, но на одних и
     * тех же данных стабилен.
     *
     * Возвращает пустой список если кэш ещё не прогрет.
     */
    fun sampleMaskedNumbers(limit: Int = 3): List<String> {
        val set = snapshot.get() ?: return emptyList()
        return set.asSequence()
            .take(limit)
            .map(::maskNumber)
            .toList()
    }

    private fun maskNumber(normalized: String): String {
        if (normalized.length <= 6) return normalized
        // Оставляем «+7» (или другой префикс) + 3 цифры DEF-кода + последние 2.
        val keepHead = normalized.take(5)
        val keepTail = normalized.takeLast(2)
        return "$keepHead•••$keepTail"
    }

    /** Принудительная синхронная перезагрузка — для тестов. */
    fun forceReloadBlocking(context: Context) {
        reloadInternal(context.applicationContext)
    }

    /** Сброс состояния — для unit-тестов. Не используется в продакшене. */
    @Synchronized
    fun resetForTest() {
        val obs = observer
        if (obs != null) {
            runCatching { /* nothing — resolver is global, observer dies with thread */ }
        }
        observer = null
        workerThread?.quitSafely()
        workerThread = null
        snapshot.set(null)
        initialized = false
    }

    private fun reloadInternal(context: Context) {
        val granted = context.checkSelfPermission(Manifest.permission.READ_CONTACTS) ==
            PackageManager.PERMISSION_GRANTED
        if (!granted) {
            // Без разрешения держим snapshot=null → FeatureExtractor пойдёт
            // в свой fallback и не будет считать «нет в кэше = не контакт».
            snapshot.set(null)
            return
        }
        val resolver: ContentResolver = context.contentResolver
        val numbers = HashSet<String>(256)
        val projection = arrayOf(
            ContactsContract.CommonDataKinds.Phone.NORMALIZED_NUMBER,
            ContactsContract.CommonDataKinds.Phone.NUMBER
        )
        try {
            resolver.query(
                ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                projection, null, null, null
            )?.use { cursor ->
                val normalizedIdx = cursor.getColumnIndex(
                    ContactsContract.CommonDataKinds.Phone.NORMALIZED_NUMBER
                )
                val rawIdx = cursor.getColumnIndex(
                    ContactsContract.CommonDataKinds.Phone.NUMBER
                )
                while (cursor.moveToNext()) {
                    val normalized = if (normalizedIdx >= 0) cursor.getString(normalizedIdx) else null
                    if (!normalized.isNullOrBlank()) {
                        numbers += normalized
                    } else if (rawIdx >= 0) {
                        // Контакт без NORMALIZED_NUMBER (бывает на старых импортах) —
                        // нормализуем сами через libphonenumber.
                        val raw = cursor.getString(rawIdx)
                        PhoneNormalizer.normalize(raw)?.let { numbers += it }
                    }
                }
            }
            snapshot.set(numbers)
            Log.d(TAG, "reloaded ${numbers.size} contacts")
        } catch (t: Throwable) {
            Log.w(TAG, "reload failed", t)
        }
    }
}
