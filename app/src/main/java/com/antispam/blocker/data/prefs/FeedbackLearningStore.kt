package com.antispam.blocker.data.prefs

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.core.stringSetPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import org.json.JSONArray

// File-level delegate ⇒ AndroidX DataStore singleton on the
// `feedback_learning.preferences_pb` file. Если объявить делегат внутри
// класса, каждая новая инстанция [FeedbackLearningStore] открывает свой
// DataStore поверх одного и того же файла, и androidx.datastore валит
// процесс с IllegalStateException "There are multiple DataStores active
// for the same file ..." — это и был краш «вернулся на главную → вылет».
private val Context.feedbackLearningDataStore: DataStore<Preferences> by preferencesDataStore(
    name = "feedback_learning"
)

/**
 * Хранит адаптивные параметры скоринга — веса факторов, дрейф порогов
 * (warn/block), количество обработанных feedback'ов.
 *
 * **Привязка к версии модели.** [warnThreshold] и [blockThreshold] меняются
 * через [setWarnThreshold] / [setBlockThreshold], которые вызывает
 * [com.antispam.blocker.domain.scoring.FeedbackHandler.adaptThresholds] на
 * основе пользовательского feedback'а. Этот дрейф валиден только для текущей
 * обученной модели — приехавшая ретрейном новая модель приходит со своими
 * порогами в [com.antispam.blocker.domain.model.ModelCard], и накопленный
 * под старую модель сдвиг для новой может быть вреден (например, юзер часто
 * жал «не спам» по поликлиникам → блок-порог поднялся → теперь новая модель,
 * которая уже сама поликлиники различает, получает лишний +0.1 к порогу и
 * пропускает реальный спам).
 *
 * Поэтому при смене версии модели (отслеживается через [pinnedModelVersion])
 * порогами и счётчик feedback'ов сбрасываются — см. [resetThresholdsForModel].
 * Веса факторов (`w_*`) при этом сохраняются: они индексируются по id фактора
 * и переживают смену модели (id факторов глобально стабилен).
 */
class FeedbackLearningStore(private val context: Context) {

    // Factor weights (key = factor_id, default = 1.0)
    fun weight(factorId: String): Flow<Float> =
        context.feedbackLearningDataStore.data.map { it[floatPreferencesKey("w_$factorId")] ?: 1.0f }

    suspend fun setWeight(factorId: String, weight: Float) {
        context.feedbackLearningDataStore.edit { it[floatPreferencesKey("w_$factorId")] = weight.coerceIn(0.1f, 3.0f) }
    }

    suspend fun getAllWeights(): Map<String, Float> {
        val prefs = context.feedbackLearningDataStore.data.first().asMap()
        return prefs.filterKeys { it.name.startsWith("w_") }
            .mapKeys { it.key.name.removePrefix("w_") }
            .mapValues { (it.value as? Number)?.toFloat() ?: 1.0f }
    }

    // Adaptive thresholds
    val warnThreshold: Flow<Float> = context.feedbackLearningDataStore.data.map {
        it[floatPreferencesKey("warn_threshold")] ?: DEFAULT_WARN_THRESHOLD
    }
    val blockThreshold: Flow<Float> = context.feedbackLearningDataStore.data.map {
        it[floatPreferencesKey("block_threshold")] ?: DEFAULT_BLOCK_THRESHOLD
    }

    suspend fun setWarnThreshold(value: Float) {
        context.feedbackLearningDataStore.edit { it[floatPreferencesKey("warn_threshold")] = value.coerceIn(0.15f, 0.50f) }
    }

    suspend fun setBlockThreshold(value: Float) {
        context.feedbackLearningDataStore.edit { it[floatPreferencesKey("block_threshold")] = value.coerceIn(0.50f, 0.85f) }
    }

    // Feedback count for threshold adaptation
    val feedbackCount: Flow<Int> = context.feedbackLearningDataStore.data.map {
        (it[floatPreferencesKey("feedback_count")]?.toInt()) ?: 0
    }

    suspend fun incrementFeedbackCount() {
        context.feedbackLearningDataStore.edit { prefs ->
            val current = (prefs[floatPreferencesKey("feedback_count")]?.toInt()) ?: 0
            prefs[floatPreferencesKey("feedback_count")] = (current + 1).toFloat()
        }
    }

    /** Версия модели, для которой накоплен текущий дрейф порогов и счётчик feedback'ов. */
    val pinnedModelVersion: Flow<String?> = context.feedbackLearningDataStore.data.map {
        it[stringPreferencesKey("pinned_model_version")]
    }

    /**
     * Сбрасывает дрейф порогов и счётчик feedback'ов на дефолтные значения
     * и пины версии к [newVersion]. Вызывается при первом обнаружении новой
     * версии model_card — обычно из [com.antispam.blocker.domain.scoring.SmartSpamDetector]
     * перед первым `score()`-проходом.
     */
    suspend fun resetThresholdsForModel(newVersion: String) {
        context.feedbackLearningDataStore.edit { prefs ->
            prefs[floatPreferencesKey("warn_threshold")] = DEFAULT_WARN_THRESHOLD
            prefs[floatPreferencesKey("block_threshold")] = DEFAULT_BLOCK_THRESHOLD
            prefs[floatPreferencesKey("feedback_count")] = 0f
            prefs[stringPreferencesKey("pinned_model_version")] = newVersion
        }
    }

    suspend fun resetAll() {
        context.feedbackLearningDataStore.edit { it.clear() }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Per-prefix override: пользователь несколько раз отметил «не спам» по
    // номерам с одного префикса (например, +74953 — Москва-Юг). Считаем
    // такие отметки в окне 30 дней. Когда счётчик переваливает за
    // [PREFIX_OVERRIDE_THRESHOLD], в Home-экране показывается чип с
    // предложением «занести префикс в персональный allowlist».
    //
    // Хранение: на префикс — JSON-массив timestamps в `prefix_notspam_<P>`.
    // Рост ограничен `PREFIX_OVERRIDE_MAX_TIMESTAMPS` (мусор за пределами
    // окна выбрасывается на каждом insert, плюс жёсткий ceiling — на случай
    // если юзер за день нажмёт «не спам» сотни раз).
    //
    // Итоговый персональный allowlist префиксов хранится как
    // [PREFIX_ALLOWLIST_KEY] (StringSet). FeatureExtractor подхватывает его
    // и OR-ит к [com.antispam.blocker.data.repository.BlockListRepository.isAllowed].
    // ──────────────────────────────────────────────────────────────────────

    /** Снэпшот персонального prefix-allowlist'а. */
    val prefixAllowlist: Flow<Set<String>> = context.feedbackLearningDataStore.data.map {
        it[stringSetPreferencesKey(PREFIX_ALLOWLIST_KEY)].orEmpty()
    }

    /**
     * `true`, если хотя бы один из накопленных в [prefixAllowlist] префиксов
     * совпадает с началом [normalized]. Выполняется на горячем пути scoring'а,
     * поэтому минимизирует аллокации.
     */
    suspend fun isPrefixAllowed(normalized: String): Boolean {
        if (normalized.isEmpty()) return false
        val set = prefixAllowlist.first()
        if (set.isEmpty()) return false
        for (p in set) if (normalized.startsWith(p)) return true
        return false
    }

    suspend fun addPrefixToAllowlist(prefix: String) {
        if (prefix.isBlank()) return
        context.feedbackLearningDataStore.edit { prefs ->
            val current = prefs[stringSetPreferencesKey(PREFIX_ALLOWLIST_KEY)].orEmpty()
            prefs[stringSetPreferencesKey(PREFIX_ALLOWLIST_KEY)] = current + prefix
            // После добавления в allowlist чип больше не нужен — чистим
            // накопленные timestamps и возможный dismiss-flag.
            prefs.remove(stringPreferencesKey("prefix_notspam_$prefix"))
            prefs.remove(booleanPreferencesKey("prefix_dismissed_$prefix"))
        }
    }

    suspend fun removePrefixFromAllowlist(prefix: String) {
        context.feedbackLearningDataStore.edit { prefs ->
            val current = prefs[stringSetPreferencesKey(PREFIX_ALLOWLIST_KEY)].orEmpty()
            prefs[stringSetPreferencesKey(PREFIX_ALLOWLIST_KEY)] = current - prefix
        }
    }

    /**
     * Записывает один «не-спам»-feedback по префиксу [prefix] (формат
     * "+7XXXX" — см. [extractPrefixOrNull]). Старые записи (>30d) и переполнение
     * `PREFIX_OVERRIDE_MAX_TIMESTAMPS` усекаются здесь же — отдельной чистки
     * не требуется.
     */
    suspend fun recordPrefixNotSpam(prefix: String, nowMillis: Long = System.currentTimeMillis()) {
        if (prefix.isBlank()) return
        val key = stringPreferencesKey("prefix_notspam_$prefix")
        context.feedbackLearningDataStore.edit { prefs ->
            // Если префикс уже в персональном allowlist — пропускаем, чтобы
            // не плодить пустые записи.
            val allowlist = prefs[stringSetPreferencesKey(PREFIX_ALLOWLIST_KEY)].orEmpty()
            if (prefix in allowlist) return@edit
            val previous = prefs[key]
            val list = mutableListOf<Long>()
            if (!previous.isNullOrBlank()) {
                runCatching {
                    val arr = JSONArray(previous)
                    for (i in 0 until arr.length()) list += arr.getLong(i)
                }
            }
            val cutoff = nowMillis - PREFIX_OVERRIDE_WINDOW_MS
            list.removeAll { it < cutoff }
            list += nowMillis
            // Жёсткий ceiling: храним только последние N — экономия места и
            // защита от raid'а «не-спам»-нажатий за один день.
            val trimmed = if (list.size > PREFIX_OVERRIDE_MAX_TIMESTAMPS) {
                list.subList(list.size - PREFIX_OVERRIDE_MAX_TIMESTAMPS, list.size)
            } else list
            val arr = JSONArray()
            for (ts in trimmed) arr.put(ts)
            prefs[key] = arr.toString()
        }
    }

    /** Скрывает чип-предложение для конкретного префикса. */
    suspend fun dismissPrefixOverride(prefix: String) {
        if (prefix.isBlank()) return
        context.feedbackLearningDataStore.edit { prefs ->
            prefs[booleanPreferencesKey("prefix_dismissed_$prefix")] = true
        }
    }

    /**
     * Поток кандидатов на per-prefix override: префиксы, по которым набралось
     * ≥ [PREFIX_OVERRIDE_THRESHOLD] «не-спам»-отметок за последние 30 дней,
     * и которые юзер ещё не дисмиссил и не занёс в персональный allowlist.
     */
    fun prefixOverrideCandidates(
        nowMillis: () -> Long = System::currentTimeMillis
    ): Flow<List<PrefixOverrideCandidate>> {
        return context.feedbackLearningDataStore.data.map { prefs ->
            val cutoff = nowMillis() - PREFIX_OVERRIDE_WINDOW_MS
            val allowlist = prefs[stringSetPreferencesKey(PREFIX_ALLOWLIST_KEY)].orEmpty()
            val out = mutableListOf<PrefixOverrideCandidate>()
            for ((k, v) in prefs.asMap()) {
                val name = k.name
                if (!name.startsWith(PREFIX_NOTSPAM_PREFIX)) continue
                val prefix = name.removePrefix(PREFIX_NOTSPAM_PREFIX)
                if (prefix in allowlist) continue
                if (prefs[booleanPreferencesKey("prefix_dismissed_$prefix")] == true) continue
                val raw = v as? String ?: continue
                val timestamps = runCatching {
                    val arr = JSONArray(raw)
                    val list = ArrayList<Long>(arr.length())
                    for (i in 0 until arr.length()) list += arr.getLong(i)
                    list
                }.getOrNull() ?: continue
                val recent = timestamps.filter { it >= cutoff }
                if (recent.size >= PREFIX_OVERRIDE_THRESHOLD) {
                    out += PrefixOverrideCandidate(
                        prefix = prefix,
                        count = recent.size,
                        lastSeenMillis = recent.maxOrNull() ?: 0L
                    )
                }
            }
            out.sortedByDescending { it.count }
        }
    }

    /**
     * Достаёт префикс для override-учёта из E.164-нормализованного номера.
     * Возвращает `null` для коротких номеров, не-РФ номеров и всего, что
     * нельзя осмысленно объединить в DEF-кодовую группу.
     */
    fun extractPrefixOrNull(normalized: String?): String? {
        if (normalized.isNullOrBlank()) return null
        if (!normalized.startsWith("+7")) return null
        if (normalized.length < PREFIX_OVERRIDE_LENGTH) return null
        return normalized.take(PREFIX_OVERRIDE_LENGTH)
    }

    data class PrefixOverrideCandidate(
        val prefix: String,
        val count: Int,
        val lastSeenMillis: Long
    )

    // ──────────────────────────────────────────────────────────────────────
    // PR-5: Per-number personal allowlist.
    //
    // Когда юзер нажимает «Это не спам» → конкретный номер добавляется сюда.
    // FeatureExtractor проверяет этот список ДО модели. Если номер в нём —
    // verdict = ALLOW, model не вызывается.
    //
    // Ротация: номера старше 90 дней без повторного подтверждения удаляются
    // автоматически (см. pruneExpiredNumbers). Это предотвращает безграничный
    // рост DataStore и ситуацию, когда номер мошенника, случайно попавший
    // в allowlist 3 месяца назад, навсегда остаётся безопасным.
    //
    // Формат хранения: StringSet, каждый элемент = "number|timestampMs".
    // ──────────────────────────────────────────────────────────────────────

    /** Поток нормализованных номеров в personal allowlist (без timestamp-суффиксов). */
    val personalAllowlist: Flow<Set<String>> = context.feedbackLearningDataStore.data.map { prefs ->
        val raw = prefs[stringSetPreferencesKey(PERSONAL_ALLOWLIST_KEY)].orEmpty()
        raw.mapNotNull { entry -> entry.substringBefore('|').takeIf { it.isNotBlank() } }.toSet()
    }

    /** Быстрая проверка: номер в personal allowlist? */
    suspend fun isNumberPersonallyAllowed(normalized: String): Boolean {
        if (normalized.isBlank()) return false
        val set = personalAllowlist.first()
        return normalized in set
    }

    /** Добавляет номер в personal allowlist с текущим timestamp. */
    suspend fun addNumberToPersonalAllowlist(normalized: String, nowMillis: Long = System.currentTimeMillis()) {
        if (normalized.isBlank()) return
        val entry = "$normalized|$nowMillis"
        context.feedbackLearningDataStore.edit { prefs ->
            val current = prefs[stringSetPreferencesKey(PERSONAL_ALLOWLIST_KEY)].orEmpty()
            // Удаляем старую запись этого номера (если есть), чтобы обновить timestamp.
            val cleaned = current.filter { !it.startsWith("$normalized|") }.toSet()
            prefs[stringSetPreferencesKey(PERSONAL_ALLOWLIST_KEY)] = cleaned + entry
        }
    }

    /** Удаляет номер из personal allowlist. */
    suspend fun removeNumberFromPersonalAllowlist(normalized: String) {
        if (normalized.isBlank()) return
        context.feedbackLearningDataStore.edit { prefs ->
            val current = prefs[stringSetPreferencesKey(PERSONAL_ALLOWLIST_KEY)].orEmpty()
            prefs[stringSetPreferencesKey(PERSONAL_ALLOWLIST_KEY)] =
                current.filter { !it.startsWith("$normalized|") }.toSet()
        }
    }

    /**
     * Удаляет записи старше [PERSONAL_ALLOWLIST_TTL_MS] (90 дней).
     * Вызывается при старте сервиса (SpamCallScreeningService.onCreate).
     */
    suspend fun pruneExpiredNumbers(nowMillis: Long = System.currentTimeMillis()) {
        val cutoff = nowMillis - PERSONAL_ALLOWLIST_TTL_MS
        context.feedbackLearningDataStore.edit { prefs ->
            val current = prefs[stringSetPreferencesKey(PERSONAL_ALLOWLIST_KEY)].orEmpty()
            val kept = current.filter { entry ->
                val ts = entry.substringAfter('|', "0").toLongOrNull() ?: 0L
                ts >= cutoff
            }.toSet()
            if (kept.size != current.size) {
                prefs[stringSetPreferencesKey(PERSONAL_ALLOWLIST_KEY)] = kept
            }
        }
    }

    companion object {
        const val ALPHA = 0.05f // EMA smoothing factor
        const val MIN_FEEDBACK_FOR_THRESHOLD = 5
        const val DEFAULT_WARN_THRESHOLD = 0.35f
        const val DEFAULT_BLOCK_THRESHOLD = 0.70f

        /** Хранит «+7» + 4 цифры DEF-кода → группа вида "+74953" (Москва-Юг). */
        const val PREFIX_OVERRIDE_LENGTH = 6
        const val PREFIX_OVERRIDE_THRESHOLD = 5
        const val PREFIX_OVERRIDE_WINDOW_MS = 30L * 24 * 60 * 60_000L
        const val PREFIX_OVERRIDE_MAX_TIMESTAMPS = 50

        /** Personal allowlist: номера старше 90 дней без повторного подтверждения удаляются. */
        const val PERSONAL_ALLOWLIST_TTL_MS = 90L * 24 * 60 * 60_000L

        private const val PREFIX_ALLOWLIST_KEY = "prefix_allowlist"
        private const val PREFIX_NOTSPAM_PREFIX = "prefix_notspam_"
        private const val PERSONAL_ALLOWLIST_KEY = "personal_number_allowlist"
    }
}
