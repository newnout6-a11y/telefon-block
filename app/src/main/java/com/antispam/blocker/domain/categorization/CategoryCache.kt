package com.antispam.blocker.domain.categorization

/**
 * LRU-кэш фиксированной ёмкости, мапящий `packageName → AppCategory`.
 * Используется горячим путём `TFLiteAppCategoryClassifier.classify(...)`,
 * чтобы повторные запросы из `PersonalNotificationListenerService.onNotificationPosted`
 * отдавались из памяти за единицы микросекунд (cache-hit ≤ 100 µs p99,
 * см. Requirement 3.10).
 *
 * Реализация — `LinkedHashMap` в режиме `accessOrder = true` плюс
 * переопределённый [LinkedHashMap.removeEldestEntry], который вытесняет
 * least-recently-used запись, как только `size > capacity`. Все мутаторы
 * (`get` / `put` / `size` / `clear`) — `@Synchronized` поверх единого
 * монитора инстанса, потому что в кэш одновременно ходят как минимум
 * три потока:
 *  - NotificationListener-поток (`onNotificationPosted` → classify),
 *  - UI-поток экрана «Прозрачность данных» (`AppCategoryClassifierFactory.classify`),
 *  - WorkManager / RemoteUpdateWorker callback `invalidate` (`clear`).
 *
 * Это **wrapper**, а не подкласс `LinkedHashMap` — наружу выставлены
 * ровно четыре метода. Прямой доступ к `keys` / `values` / `entries`
 * закрыт намеренно, чтобы случайная итерация (например, через
 * `Gson.toJson(cache)` или сериализацию в `Bundle`) не могла утечь
 * содержимое кэша за пределы process-memory privacy-границы.
 *
 * ## Privacy contract — process memory only
 *
 * Содержимое кэша живёт **исключительно в куче JVM работающего процесса**.
 * Оно НИКОГДА не персистится и не покидает память процесса:
 *  - never persisted to **SharedPreferences**,
 *  - never persisted to **Room** / `AppDatabase` / любой DAO,
 *  - never persisted to **files** в `filesDir`, `cacheDir`, `externalFilesDir`,
 *    MediaStore, dropbox или иной точке хранения,
 *  - never crosses **IPC** boundaries (`ContentProvider`, AIDL, Binder, intents),
 *  - never logged через `android.util.Log` и не отправляется по сети.
 *
 * При завершении процесса (system kill, low-memory eviction, user
 * clear-cache, device reboot) весь кэш полностью теряется — это
 * is-by-design (см. Requirement 5.9): предсказанные категории
 * приложений пользователя НЕ должны переживать рантайм, который их
 * вычислил.
 *
 * @param capacity Максимальное число записей; при `size > capacity`
 *   вытесняется LRU-entry. Default для рантайма — 500
 *   (см. `TFLiteAppCategoryClassifier.DEFAULT_CACHE_CAPACITY`).
 *
 * @see <a href="../../../../../../../../.kiro/specs/app-category-ml-classifier/design.md">design.md → Component 3: Category_Cache</a>
 */
class CategoryCache(private val capacity: Int) {

    init {
        require(capacity >= 1) { "CategoryCache capacity must be ≥ 1, got $capacity" }
    }

    /**
     * `accessOrder = true` ⇒ `get`/`put` перемещают элемент в "most recently
     * used" хвост; eldest = head = least recently used. Initial bucket
     * capacity взят равным заявленной ёмкости — в типовом use-case (cap=500)
     * это исключает ре-хеширование на горячем пути.
     */
    private val map: LinkedHashMap<String, AppCategory> =
        object : LinkedHashMap<String, AppCategory>(capacity, LOAD_FACTOR, /* accessOrder = */ true) {
            override fun removeEldestEntry(
                eldest: MutableMap.MutableEntry<String, AppCategory>?,
            ): Boolean = size > capacity
        }

    @Synchronized
    fun get(packageName: String): AppCategory? = map[packageName]

    @Synchronized
    fun put(packageName: String, category: AppCategory) {
        map[packageName] = category
    }

    @Synchronized
    fun size(): Int = map.size

    @Synchronized
    fun clear() {
        map.clear()
    }

    private companion object {
        private const val LOAD_FACTOR = 0.75f
    }
}
