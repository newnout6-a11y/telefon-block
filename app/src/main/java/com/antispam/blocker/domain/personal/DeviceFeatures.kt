package com.antispam.blocker.domain.personal

/**
 * Фиксированная схема фич Device_Model — персональной on-device логистической регрессии.
 *
 * Порядок имён в [NAMES] **важен**: индекс каждой фичи зашит в default weights
 * (`assets/device_model_default_weights.json`), в [Feature_Snapshot.featuresJson] и в
 * формулу SGD-шага. Любое изменение порядка или удаление/добавление фичи требует
 * увеличения [SCHEMA_VERSION].
 *
 * Размер вектора держится в коридоре `[15, 20]` согласно требованию 3.1; текущая
 * версия — 17 фич, типы по табличке из design "Components and Interfaces →
 * DeviceFeatures": булевые источники нормализованы как `0f / 1f`, агрегаты по
 * call_event приведены в `[0f, 1f]` нормализующими функциями экстрактора.
 *
 * Validates: Requirements 3.1 (фиксированный набор фич), 2.3 (стабильный snapshot).
 */
data class DeviceFeatures(
    val values: FloatArray
) {
    init {
        require(values.size == SIZE) {
            "DeviceFeatures.values.size must be $SIZE (got ${values.size}); " +
                "schema_version=$SCHEMA_VERSION"
        }
    }

    /**
     * Возвращает «сырой» вектор для скоринга/SGD без копирования.
     *
     * Контракт: вызывающий код должен относиться к результату как к read-only —
     * мутация массива нарушит гарантии Feature_Snapshot (Req 2.3).
     */
    fun toFloatArray(): FloatArray = values

    /**
     * Equality на содержимом массива, а не на ссылке.
     *
     * `data class` по умолчанию делает reference equality для `FloatArray`,
     * что бесполезно: два одинаковых снимка не сравнятся. Property-tests и
     * round-trip через [Feature_Snapshot.featuresJson] требуют content-based
     * сравнения.
     */
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is DeviceFeatures) return false
        return values.contentEquals(other.values)
    }

    override fun hashCode(): Int = values.contentHashCode()

    override fun toString(): String =
        "DeviceFeatures(schema=$SCHEMA_VERSION, values=${values.contentToString()})"

    companion object {
        /**
         * Версия схемы фич. Bump при любом изменении набора, порядка или
         * семантики фич — `OnlineTrainer` должен отбрасывать снимки с
         * несовпадающей версией, чтобы SGD-шаг не применился к чужим числам.
         */
        const val SCHEMA_VERSION: Int = 1

        /**
         * Имена фич в каноническом порядке.
         *
         * Соответствует таблице из design "DeviceFeatures":
         *  0..3   — контекст звонка (контакт, прошлое поведение, время суток/неделя)
         *  4..5   — производные от call_event (missed/no-callback, outgoing-after-missed)
         *  6..9   — недавно открытые приложения по категориям
         * 10..11  — недавние уведомления (только bucket'ы)
         * 12..13  — оператор/short-code
         * 14..15  — нормализованные агрегаты по номеру/префиксу
         * 16     — скрытый номер
         */
        val NAMES: List<String> = listOf(
            "is_contact",
            "previously_rejected",
            "is_night_time",
            "is_weekend",
            "prev_missed_no_callback_24h",
            "prev_outgoing_after_missed",
            "recent_bank_app_30m",
            "recent_gov_app_30m",
            "recent_marketplace_app_30m",
            "recent_messenger_app_30m",
            "notif_bank_recent_10m",
            "notif_marketplace_recent_10m",
            "same_carrier_as_user",
            "is_short_code",
            "same_prefix_call_count_7d_norm",
            "answer_rate_for_number_norm",
            "hidden_number"
        )

        /** Размер фича-вектора. Гарантировано совпадает с [NAMES].size. */
        val SIZE: Int = NAMES.size
    }
}
