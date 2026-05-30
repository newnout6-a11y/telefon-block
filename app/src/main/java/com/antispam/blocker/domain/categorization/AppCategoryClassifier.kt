package com.antispam.blocker.domain.categorization

import android.content.Context
import android.util.Log
import com.antispam.blocker.SpamBlockerApp
import org.tensorflow.lite.Interpreter

/**
 * Третья «ИИ» в архитектуре наряду со Server Model (TFLite-классификатор
 * спам-номеров) и Personal Model (on-device логистическая регрессия по
 * 17 фичам). Задача — категоризировать приложение по `packageName` (и,
 * опционально, локализованному label из PackageManager) на семантические
 * бакеты, которые потом потребляют:
 *
 *  - `RecentUserContextProvider.getRecentContext` → фичи Personal Model
 *    `recent_*_30m` и `notif_*_recent_10m`.
 *  - `PersonalNotificationListenerService.bucketFor` → запись в
 *    `notification_event.categoryBucket`.
 *  - UI «Прозрачность данных» в Settings → Privacy.
 *
 * Сейчас реализация [RuleBasedAppCategoryClassifier] — словарь из ~150
 * известных российских пакетов плюс substring-эвристики (см. ниже).
 * В будущем сюда подключится [TFLiteAppCategoryClassifier]: char-n-gram
 * encoder + 18-class softmax, обученный на (packageName, Play Store
 * category) корпусе. Модель — sub-1MB TFLite, лежит в `assets/` и
 * вытягивается через `RemoteUpdateWorker` так же, как Server Model.
 */
interface AppCategoryClassifier {

    /**
     * @param packageName Android package id, например `ru.sberbankmobile`.
     * @param label Опциональное локализованное имя приложения из
     *   `PackageManager.getApplicationInfo(packageName).loadLabel(...)`.
     *   Помогает rule-based реализации различать пакеты с
     *   обфусцированными именами (китайские маркетплейсы и т.п.).
     */
    fun classify(packageName: String, label: String? = null): AppCategory
}

/**
 * 18 категорий, упорядоченных по убыванию приоритета: если пакет
 * матчится в нескольких словарях (например, банк+инвестиции), берём
 * первую совпавшую сверху.
 *
 * Отдельный enum, а не `NotificationEvent.CategoryBucket`, потому что
 * Room-схема `notification_event` уже содержит более узкий enum
 * (BANK / MARKETPLACE / MESSENGER / EMAIL / OTHER) — мы не хотим
 * ломать миграцию. Маппинг на узкий enum происходит в
 * [AppCategory.toNotificationBucket].
 */
enum class AppCategory {
    BANK,
    INVESTMENTS,
    GOVERNMENT,
    MARKETPLACE,
    DELIVERY,
    TRANSPORT,
    TRAVEL,
    HEALTH,
    MESSENGER,
    SOCIAL,
    EMAIL,
    NEWS,
    MEDIA,
    GAMES,
    DATING,
    EDUCATION,
    BROWSER,
    VPN,
    PRODUCTIVITY,
    OTHER;

    /**
     * Маппинг 18-категорийного пространства на 5-категорийный enum,
     * используемый в Room (`NotificationEvent.CategoryBucket`).
     * Бакеты, которые Personal Model ещё не использует как фичу,
     * сворачиваются в OTHER.
     */
    fun toNotificationBucket(): String = when (this) {
        BANK, INVESTMENTS -> "BANK"
        MARKETPLACE, DELIVERY -> "MARKETPLACE"
        MESSENGER -> "MESSENGER"
        EMAIL -> "EMAIL"
        else -> "OTHER"
    }
}

/**
 * Production-ready реализация на правилах + словаре. Покрывает
 * большинство популярных российских и международных приложений
 * без ML-модели.
 *
 * Стратегия:
 * 1. Точное совпадение в словаре [KNOWN_PACKAGES] — самый сильный сигнал.
 * 2. Substring heuristics над `packageName.lowercase()` — словесные
 *    маркеры типа `bank`/`pay`/`shop`/`messenger`. Достаточно одного
 *    совпадения, в порядке приоритета категорий.
 * 3. То же самое над `label.lowercase()` если `label != null`.
 * 4. Иначе [AppCategory.OTHER].
 *
 * Производительность: hot-path метод, вызывается из NotificationListener
 * на каждое уведомление. Все коллекции — `Set<String>` или
 * `List<Pair<String, AppCategory>>`, lookup O(1) / линейный по словарю
 * substring-маркеров (~30 элементов). На устройстве это микросекунды.
 */
class RuleBasedAppCategoryClassifier : AppCategoryClassifier {

    override fun classify(packageName: String, label: String?): AppCategory {
        if (packageName.isBlank()) return AppCategory.OTHER
        KNOWN_PACKAGES[packageName]?.let { return it }

        val lower = packageName.lowercase()
        for ((marker, category) in PACKAGE_MARKERS) {
            if (marker in lower) return category
        }

        if (!label.isNullOrBlank()) {
            val labelLower = label.lowercase()
            for ((marker, category) in LABEL_MARKERS) {
                if (marker in labelLower) return category
            }
        }

        return AppCategory.OTHER
    }

    private companion object {

        // ── Точно известные пакеты ─────────────────────────────────────────
        // Только публичные приложения с предсказуемой категорией. Если у
        // пакета функционал «банк + инвестиции», маппим в BANK (приоритет
        // выше) — для целей фич Personal Model это эквивалентно.
        val KNOWN_PACKAGES: Map<String, AppCategory> = mapOf(
            // ── Banks (RU) ────────────────────────────────────────────────
            "ru.sberbankmobile" to AppCategory.BANK,
            "com.idamob.tinkoff.android" to AppCategory.BANK,
            "ru.tinkoff.app" to AppCategory.BANK,
            "ru.vtb.mobile" to AppCategory.BANK,
            "ru.vtb24.mobilebanking.android" to AppCategory.BANK,
            "ru.alfabank.mobile.android" to AppCategory.BANK,
            "ru.alfabank.oavdo.amc" to AppCategory.BANK,
            "ru.gazprombank.mobile" to AppCategory.BANK,
            "ru.rshb.v1" to AppCategory.BANK,
            "ru.mkb.mobile" to AppCategory.BANK,
            "ru.psbc.mbank" to AppCategory.BANK,
            "ru.rosbank.mobile" to AppCategory.BANK,
            "ru.otp.mobile" to AppCategory.BANK,
            "ru.raiffeisen" to AppCategory.BANK,
            "ru.raiffeisennews" to AppCategory.BANK,
            "ru.rsb.mobile" to AppCategory.BANK,
            "com.openbank.mobile" to AppCategory.BANK,
            "ru.sovcombank.mobile" to AppCategory.BANK,
            "ru.sovcombank.halva" to AppCategory.BANK,
            "ru.pochta.bank.mobile" to AppCategory.BANK,
            "com.homecredit.mobilebanking" to AppCategory.BANK,
            "ru.bss.mobile" to AppCategory.BANK,
            "ru.ubrir.mobile" to AppCategory.BANK,
            "ru.crpt.client" to AppCategory.BANK,
            "ru.bcs.mobile" to AppCategory.BANK,
            "ru.qiwi.client" to AppCategory.BANK,
            "ru.yandex.money" to AppCategory.BANK,
            "ru.yoo.money" to AppCategory.BANK,
            "ru.mirpay" to AppCategory.BANK,

            // ── Investments (RU) ─────────────────────────────────────────
            "com.tinkoff.investing" to AppCategory.INVESTMENTS,
            "ru.sberbank.android.investor" to AppCategory.INVESTMENTS,
            "ru.alfabank.investments" to AppCategory.INVESTMENTS,
            "ru.bcs.broker" to AppCategory.INVESTMENTS,
            "ru.finam.tradetrust" to AppCategory.INVESTMENTS,
            "ru.openbroker.app" to AppCategory.INVESTMENTS,
            "ru.vtb24.brokerage" to AppCategory.INVESTMENTS,

            // ── Government (RU) ──────────────────────────────────────────
            "ru.rostelecom.gosuslugi" to AppCategory.GOVERNMENT,
            "ru.gosuslugi" to AppCategory.GOVERNMENT,
            "ru.gosuslugi.api" to AppCategory.GOVERNMENT,
            "ru.minsvyaz.gosuslugi" to AppCategory.GOVERNMENT,
            "ru.fns.lkfl.app" to AppCategory.GOVERNMENT,
            "ru.fns.docs" to AppCategory.GOVERNMENT,
            "ru.mos.app" to AppCategory.GOVERNMENT,
            "ru.mos.metro" to AppCategory.GOVERNMENT,
            "ru.tinkoff.acquiring.fns" to AppCategory.GOVERNMENT,
            "ru.gibdd.gibddru" to AppCategory.GOVERNMENT,
            "ru.cdek.client" to AppCategory.GOVERNMENT,

            // ── Marketplaces (RU) ────────────────────────────────────────
            "com.wildberries.ru" to AppCategory.MARKETPLACE,
            "com.ozon.android" to AppCategory.MARKETPLACE,
            "ru.beru.android" to AppCategory.MARKETPLACE,
            "com.sbermarket" to AppCategory.MARKETPLACE,
            "ru.megamarket" to AppCategory.MARKETPLACE,
            "com.yandex.market" to AppCategory.MARKETPLACE,
            "ru.lamoda.lamoda" to AppCategory.MARKETPLACE,
            "com.aliexpress.buyer" to AppCategory.MARKETPLACE,
            "com.aliexpress.aer" to AppCategory.MARKETPLACE,
            "com.joom" to AppCategory.MARKETPLACE,
            "ru.avito" to AppCategory.MARKETPLACE,
            "com.yandex.metrocollector" to AppCategory.MARKETPLACE,
            "ru.auto.ara" to AppCategory.MARKETPLACE,
            "ru.youla" to AppCategory.MARKETPLACE,

            // ── Delivery / Food (RU) ─────────────────────────────────────
            "ru.samokat.android" to AppCategory.DELIVERY,
            "com.icemobile.deliveryclub" to AppCategory.DELIVERY,
            "ru.foodfox.client" to AppCategory.DELIVERY,
            "ru.yandex.eda" to AppCategory.DELIVERY,
            "com.deliveryclub" to AppCategory.DELIVERY,
            "ru.tinkoff.eda" to AppCategory.DELIVERY,
            "com.kuhnyanarayone" to AppCategory.DELIVERY,
            "ru.lavka" to AppCategory.DELIVERY,
            "ru.ozon.express" to AppCategory.DELIVERY,
            "ru.burgerking.android" to AppCategory.DELIVERY,
            "com.mcdonalds.app" to AppCategory.DELIVERY,
            "ru.dodopizza" to AppCategory.DELIVERY,
            "ru.tasty" to AppCategory.DELIVERY,

            // ── Transport / Taxi ─────────────────────────────────────────
            "ru.yandex.taxi" to AppCategory.TRANSPORT,
            "com.ubercab" to AppCategory.TRANSPORT,
            "com.citymobil.passenger" to AppCategory.TRANSPORT,
            "sinet.startup.inDriver" to AppCategory.TRANSPORT,
            "com.bolt.deliveryclient" to AppCategory.TRANSPORT,
            "ru.yandex.metro" to AppCategory.TRANSPORT,
            "ru.yandex.maps" to AppCategory.TRANSPORT,
            "ru.dublgis.dgismobile" to AppCategory.TRANSPORT,

            // ── Travel ───────────────────────────────────────────────────
            "ru.aviasales" to AppCategory.TRAVEL,
            "ru.onetwotrip.client" to AppCategory.TRAVEL,
            "ru.tutu.tutu" to AppCategory.TRAVEL,
            "com.booking" to AppCategory.TRAVEL,
            "com.airbnb.android" to AppCategory.TRAVEL,
            "ru.rzd.pass" to AppCategory.TRAVEL,
            "ru.aeroflot" to AppCategory.TRAVEL,
            "com.utair.utair" to AppCategory.TRAVEL,
            "ru.s7airlines" to AppCategory.TRAVEL,

            // ── Health ───────────────────────────────────────────────────
            "ru.sberbank.health" to AppCategory.HEALTH,
            "ru.emias.app" to AppCategory.HEALTH,
            "ru.docdoc.docdoc" to AppCategory.HEALTH,
            "ru.zdorov.app" to AppCategory.HEALTH,
            "ru.medsi.medsi" to AppCategory.HEALTH,

            // ── Messengers ───────────────────────────────────────────────
            "com.whatsapp" to AppCategory.MESSENGER,
            "org.telegram.messenger" to AppCategory.MESSENGER,
            "org.telegram.messenger.web" to AppCategory.MESSENGER,
            "com.viber.voip" to AppCategory.MESSENGER,
            "org.thoughtcrime.securesms" to AppCategory.MESSENGER,
            "ch.threema.app" to AppCategory.MESSENGER,
            "com.discord" to AppCategory.MESSENGER,
            "com.skype.raider" to AppCategory.MESSENGER,
            "com.snapchat.android" to AppCategory.MESSENGER,
            "vk.messenger.android" to AppCategory.MESSENGER,
            "ru.mail.icq" to AppCategory.MESSENGER,
            "ru.tamtam.tamtam" to AppCategory.MESSENGER,
            "ru.maxmessenger.app" to AppCategory.MESSENGER,

            // ── Social ───────────────────────────────────────────────────
            "com.vkontakte.android" to AppCategory.SOCIAL,
            "ru.ok.android" to AppCategory.SOCIAL,
            "com.twitter.android" to AppCategory.SOCIAL,
            "com.instagram.android" to AppCategory.SOCIAL,
            "com.zhiliaoapp.musically" to AppCategory.SOCIAL, // TikTok
            "com.ss.android.ugc.trill" to AppCategory.SOCIAL, // TikTok lite
            "com.reddit.frontpage" to AppCategory.SOCIAL,
            "com.facebook.katana" to AppCategory.SOCIAL,
            "com.linkedin.android" to AppCategory.SOCIAL,

            // ── Email ────────────────────────────────────────────────────
            "com.google.android.gm" to AppCategory.EMAIL,
            "ru.mail.mailapp" to AppCategory.EMAIL,
            "ru.yandex.mail" to AppCategory.EMAIL,
            "com.microsoft.office.outlook" to AppCategory.EMAIL,
            "com.yahoo.mobile.client.android.mail" to AppCategory.EMAIL,
            "ch.protonmail.android" to AppCategory.EMAIL,

            // ── News (RU) ────────────────────────────────────────────────
            "ru.rbc.rbcnews" to AppCategory.NEWS,
            "ru.lentaru" to AppCategory.NEWS,
            "ru.gazeta.app" to AppCategory.NEWS,
            "ru.tass.android" to AppCategory.NEWS,
            "ru.interfax.android" to AppCategory.NEWS,
            "ru.rt.android" to AppCategory.NEWS,
            "ru.kommersant.kommersant" to AppCategory.NEWS,

            // ── Media / Video ────────────────────────────────────────────
            "com.google.android.youtube" to AppCategory.MEDIA,
            "tv.twitch.android.app" to AppCategory.MEDIA,
            "ru.rutube.app" to AppCategory.MEDIA,
            "ru.kinopoisk" to AppCategory.MEDIA,
            "ru.ivi.client" to AppCategory.MEDIA,
            "ru.more.tv" to AppCategory.MEDIA,
            "ru.okko.android" to AppCategory.MEDIA,
            "com.spbtv.android.premier" to AppCategory.MEDIA,
            "com.spotify.music" to AppCategory.MEDIA,
            "ru.yandex.music" to AppCategory.MEDIA,
            "ru.zvuk.app" to AppCategory.MEDIA,

            // ── Browsers ─────────────────────────────────────────────────
            "com.android.chrome" to AppCategory.BROWSER,
            "org.mozilla.firefox" to AppCategory.BROWSER,
            "com.yandex.browser" to AppCategory.BROWSER,
            "com.microsoft.emmx" to AppCategory.BROWSER,
            "com.opera.browser" to AppCategory.BROWSER,
            "com.sec.android.app.sbrowser" to AppCategory.BROWSER,
            "com.duckduckgo.mobile.android" to AppCategory.BROWSER,

            // ── VPN ──────────────────────────────────────────────────────
            "com.nordvpn.android" to AppCategory.VPN,
            "com.expressvpn.vpn" to AppCategory.VPN,
            "ch.protonvpn.android" to AppCategory.VPN,
            "free.vpn.unblock.proxy.vpnpro" to AppCategory.VPN,
            "com.cloudflare.onedotonedotonedotone" to AppCategory.VPN,

            // ── Dating ───────────────────────────────────────────────────
            "com.tinder" to AppCategory.DATING,
            "ru.mamba.client.v3" to AppCategory.DATING,
            "com.bumble.app" to AppCategory.DATING,
            "com.badoo.mobile" to AppCategory.DATING,
            "com.pure.android" to AppCategory.DATING,

            // ── Education ────────────────────────────────────────────────
            "com.skyeng.skyengvocab" to AppCategory.EDUCATION,
            "com.yandex.practicum" to AppCategory.EDUCATION,
            "ru.foxford.android" to AppCategory.EDUCATION,
            "com.duolingo" to AppCategory.EDUCATION,

            // ── Productivity ─────────────────────────────────────────────
            "com.google.android.calendar" to AppCategory.PRODUCTIVITY,
            "com.microsoft.office.outlook" to AppCategory.PRODUCTIVITY,
            "com.todoist" to AppCategory.PRODUCTIVITY,
            "com.notesnook.android" to AppCategory.PRODUCTIVITY,
            "com.evernote" to AppCategory.PRODUCTIVITY,
            "com.notion.android" to AppCategory.PRODUCTIVITY,
        )

        // ── Substring markers по `packageName` ─────────────────────────────
        // Порядок имеет значение: первое совпадение выигрывает. Сильные
        // маркеры (точные слова «bank», «pay») идут раньше слабых
        // («mobile», «android»).
        val PACKAGE_MARKERS: List<Pair<String, AppCategory>> = listOf(
            ".bank" to AppCategory.BANK,
            "bank." to AppCategory.BANK,
            "banking" to AppCategory.BANK,
            ".pay" to AppCategory.BANK,
            "wallet" to AppCategory.BANK,
            "broker" to AppCategory.INVESTMENTS,
            "invest" to AppCategory.INVESTMENTS,
            "trading" to AppCategory.INVESTMENTS,
            "crypto" to AppCategory.INVESTMENTS,
            "exchange" to AppCategory.INVESTMENTS,
            "binance" to AppCategory.INVESTMENTS,

            "gosuslug" to AppCategory.GOVERNMENT,
            ".gov" to AppCategory.GOVERNMENT,
            "government" to AppCategory.GOVERNMENT,
            ".fns" to AppCategory.GOVERNMENT,
            ".fssp" to AppCategory.GOVERNMENT,
            ".mvd" to AppCategory.GOVERNMENT,

            "shop" to AppCategory.MARKETPLACE,
            "market" to AppCategory.MARKETPLACE,
            "wildberries" to AppCategory.MARKETPLACE,
            "ozon" to AppCategory.MARKETPLACE,

            "delivery" to AppCategory.DELIVERY,
            "doordash" to AppCategory.DELIVERY,
            "ubereats" to AppCategory.DELIVERY,
            "samokat" to AppCategory.DELIVERY,
            "lavka" to AppCategory.DELIVERY,
            ".eda" to AppCategory.DELIVERY,
            ".pizza" to AppCategory.DELIVERY,

            "taxi" to AppCategory.TRANSPORT,
            ".uber" to AppCategory.TRANSPORT,
            "yandex.taxi" to AppCategory.TRANSPORT,
            "rideshare" to AppCategory.TRANSPORT,
            "scooter" to AppCategory.TRANSPORT,

            "booking" to AppCategory.TRAVEL,
            "airline" to AppCategory.TRAVEL,
            "aviasales" to AppCategory.TRAVEL,
            "trip" to AppCategory.TRAVEL,
            "hotel" to AppCategory.TRAVEL,
            "flight" to AppCategory.TRAVEL,

            "messenger" to AppCategory.MESSENGER,
            "messaging" to AppCategory.MESSENGER,
            "whatsapp" to AppCategory.MESSENGER,
            "telegram" to AppCategory.MESSENGER,
            "viber" to AppCategory.MESSENGER,
            "signal" to AppCategory.MESSENGER,
            ".chat" to AppCategory.MESSENGER,

            "vkontakte" to AppCategory.SOCIAL,
            "instagram" to AppCategory.SOCIAL,
            "twitter" to AppCategory.SOCIAL,
            "tiktok" to AppCategory.SOCIAL,
            "facebook" to AppCategory.SOCIAL,
            "snapchat" to AppCategory.SOCIAL,

            "mail" to AppCategory.EMAIL,
            "outlook" to AppCategory.EMAIL,

            "news" to AppCategory.NEWS,
            ".rbc" to AppCategory.NEWS,
            "lenta" to AppCategory.NEWS,

            "youtube" to AppCategory.MEDIA,
            "kinopoisk" to AppCategory.MEDIA,
            ".music" to AppCategory.MEDIA,
            ".video" to AppCategory.MEDIA,
            "spotify" to AppCategory.MEDIA,
            "twitch" to AppCategory.MEDIA,

            "browser" to AppCategory.BROWSER,
            "chrome" to AppCategory.BROWSER,
            "firefox" to AppCategory.BROWSER,
            "opera" to AppCategory.BROWSER,
            "edge" to AppCategory.BROWSER,

            "vpn" to AppCategory.VPN,

            "doctor" to AppCategory.HEALTH,
            "health" to AppCategory.HEALTH,
            "medsi" to AppCategory.HEALTH,
            "fitness" to AppCategory.HEALTH,

            "dating" to AppCategory.DATING,
            "tinder" to AppCategory.DATING,
            "mamba" to AppCategory.DATING,

            "edu." to AppCategory.EDUCATION,
            "english" to AppCategory.EDUCATION,
            "duolingo" to AppCategory.EDUCATION,
            "skyeng" to AppCategory.EDUCATION,
            "lessons" to AppCategory.EDUCATION,

            "calendar" to AppCategory.PRODUCTIVITY,
            "todoist" to AppCategory.PRODUCTIVITY,
            "notion" to AppCategory.PRODUCTIVITY,
            "evernote" to AppCategory.PRODUCTIVITY,

            ".game" to AppCategory.GAMES,
            "games" to AppCategory.GAMES,
            "playrix" to AppCategory.GAMES,
            "nexters" to AppCategory.GAMES,
            "supercell" to AppCategory.GAMES,
            "tencent" to AppCategory.GAMES,
        )

        // ── Substring markers по локализованному label ────────────────────
        // Используем когда packageName бесполезен (обфускация).
        val LABEL_MARKERS: List<Pair<String, AppCategory>> = listOf(
            "банк" to AppCategory.BANK,
            "карта" to AppCategory.BANK,
            "оплата" to AppCategory.BANK,
            "инвест" to AppCategory.INVESTMENTS,
            "брокер" to AppCategory.INVESTMENTS,
            "госуслуг" to AppCategory.GOVERNMENT,
            "налог" to AppCategory.GOVERNMENT,
            "магазин" to AppCategory.MARKETPLACE,
            "доставк" to AppCategory.DELIVERY,
            "такси" to AppCategory.TRANSPORT,
            "месенджер" to AppCategory.MESSENGER,
            "мессенджер" to AppCategory.MESSENGER,
            "почта" to AppCategory.EMAIL,
            "новост" to AppCategory.NEWS,
            "клиник" to AppCategory.HEALTH,
            "доктор" to AppCategory.HEALTH,
            "знакомств" to AppCategory.DATING,
            "обучен" to AppCategory.EDUCATION,
            "образован" to AppCategory.EDUCATION,
            "браузер" to AppCategory.BROWSER,
            "видео" to AppCategory.MEDIA,
            "музык" to AppCategory.MEDIA,
            "игр" to AppCategory.GAMES,
            "vpn" to AppCategory.VPN,
            "путешеств" to AppCategory.TRAVEL,
            "перелёт" to AppCategory.TRAVEL,
            "отель" to AppCategory.TRAVEL,
        )
    }
}

/**
 * Singleton-обёртка, доступная без DI. На горячем пути
 * (PersonalNotificationListenerService.onNotificationPosted) используется
 * через [classify].
 *
 * ## Selection table (Req 3.6, 3.7, 3.9)
 *
 * Возвращает [TFLiteAppCategoryClassifier] **только** когда выполнены все три
 * условия:
 *
 * 1. `AppCategoryAssetSource.resolve(...)` вернул не-null
 *    (model+vocab пара доступна в `filesDir` или APK assets);
 * 2. `SettingsStore.tfliteAppCategoryEnabledSnapshot()` == `true`
 *    (kill-switch не выключен);
 * 3. `tryCreateTFLite(...)` не выбросил исключение
 *    (TFLite Interpreter инициализировался, output shape валидный, vocab распарсился).
 *
 * При любой "F" из триплета — стабильный singleton [RuleBasedAppCategoryClassifier].
 *
 * ## Lazy reinit на обновлениях ассетов
 *
 * После успешного скачивания обновлённого `app_category_model.tflite`,
 * `app_category_vocab.txt` или `app_category_card.json` через
 * `RemoteUpdateWorker.doWork()` вызывается
 * [TFLiteAppCategoryClassifier.invalidate], которая бампит
 * `assetEpoch`. Следующий вызов [classify] увидит расхождение между
 * cached-epoch и текущим, закроет старый Interpreter и создаст новый —
 * без перезапуска процесса.
 *
 * ## Privacy contract на init-failure-логе
 *
 * При неудачной TFLite-инициализации логируется только факт
 * (`"TFLiteAppCategoryClassifier init failed; falling back"`) и сам
 * `Throwable`; **никаких** входов `packageName`/`label` (Req 5.4).
 */
object AppCategoryClassifierFactory {
    private const val TAG = "AppCategoryFactory"

    private val ruleBased: RuleBasedAppCategoryClassifier by lazy {
        RuleBasedAppCategoryClassifier()
    }

    /**
     * Currently active classifier — либо TFLite (если все условия
     * selection table выполнены), либо singleton rule-based.
     *
     * `@Volatile` гарантирует видимость записи между потоками без
     * дополнительной синхронизации на read-path (горячий путь
     * NL-сервиса).
     */
    @Volatile
    private var cached: AppCategoryClassifier? = null

    /**
     * Asset-epoch на момент создания текущего [cached] инстанса.
     * Если `TFLiteAppCategoryClassifier.currentAssetEpoch()` отличается
     * от этого значения, [cached] пересоздаётся.
     */
    @Volatile
    private var cachedEpoch: Long = -1L

    /** Lock для пересоздания. Read-path lock-free через [cached]. */
    private val lock = Any()

    /**
     * Возвращает текущий активный классификатор. Создаёт новый инстанс
     * при первом вызове или при расхождении `cachedEpoch` и текущего
     * `assetEpoch` (после [TFLiteAppCategoryClassifier.invalidate]).
     */
    fun getOrCreate(context: Context): AppCategoryClassifier {
        val current = cached
        val currentEpoch = TFLiteAppCategoryClassifier.currentAssetEpoch()
        if (current != null && cachedEpoch == currentEpoch) {
            return current
        }
        return synchronized(lock) {
            val again = cached
            if (again != null && cachedEpoch == currentEpoch) {
                again
            } else {
                val fresh = createClassifier(context.applicationContext)
                cached = fresh
                cachedEpoch = currentEpoch
                fresh
            }
        }
    }

    /**
     * Сбрасывает кэш — следующий [getOrCreate] создаст новый инстанс.
     * Вызывается из тестов; в production бамп делает
     * [TFLiteAppCategoryClassifier.invalidate].
     */
    @Synchronized
    fun invalidate() {
        cached = null
        cachedEpoch = -1L
    }

    /**
     * Selection table:
     *
     *  | assetSource | killSwitch | initSucceeds | result |
     *  |:-----------:|:----------:|:------------:|:------:|
     *  |     ✓       |     on     |      ✓       | TFLite |
     *  |     ✗       |     —      |      —       |  Rule  |
     *  |     ✓       |    off     |      —       |  Rule  |
     *  |     ✓       |     on     |      ✗       |  Rule  |
     */
    private fun createClassifier(appContext: Context): AppCategoryClassifier {
        val assetSource = AppCategoryAssetSource.resolve(appContext)
            ?: return ruleBased

        val killSwitchOn = try {
            SpamBlockerApp.instance.settingsStore.tfliteAppCategoryEnabledSnapshot()
        } catch (_: Throwable) {
            // Если SpamBlockerApp ещё не инициализирован (тесты, ранний bootstrap) —
            // безопасный default = rule-based.
            return ruleBased
        }
        if (!killSwitchOn) return ruleBased

        return tryCreateTFLite(appContext, ruleBased, assetSource) ?: ruleBased
    }

    /**
     * Пытается создать [TFLiteAppCategoryClassifier]. Возвращает `null`
     * при любом исключении (Interpreter init, vocab parse, output shape
     * валидация). Лог не содержит значений входов (Req 5.4).
     */
    private fun tryCreateTFLite(
        appContext: Context,
        rules: RuleBasedAppCategoryClassifier,
        assetSource: AppCategoryAssetSource,
    ): TFLiteAppCategoryClassifier? = try {
        val tokenizer = CharNGramTokenizer.load(assetSource.vocabSource)
        val interpreter = Interpreter(assetSource.modelByteBuffer)
        TFLiteAppCategoryClassifier(
            context = appContext,
            ruleBased = rules,
            tokenizer = tokenizer,
            interpreter = interpreter,
        )
    } catch (t: Throwable) {
        Log.w(TAG, "TFLiteAppCategoryClassifier init failed; falling back", t)
        null
    }

    /**
     * Backward-compatible static API — горячий путь
     * `PersonalNotificationListenerService.onNotificationPosted` и
     * `RecentUserContextProvider`. Резолвит инстанс через
     * `SpamBlockerApp.instance` (Application-context).
     *
     * Если `SpamBlockerApp` ещё не инициализирован (вызов в раннем
     * bootstrap или unit-тесте без mocked Application), возвращает
     * rule-based singleton — это безопасный default.
     */
    fun classify(packageName: String, label: String? = null): AppCategory {
        val classifier = try {
            getOrCreate(SpamBlockerApp.instance)
        } catch (_: Throwable) {
            ruleBased
        }
        return classifier.classify(packageName, label)
    }
}
