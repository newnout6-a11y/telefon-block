package com.antispam.blocker.domain.categorization

import android.content.Context
import android.util.Log
import androidx.annotation.VisibleForTesting
import org.tensorflow.lite.Interpreter
import java.util.concurrent.atomic.AtomicLong

/**
 * ML-реализация [AppCategoryClassifier] на базе char-CNN TFLite-модели.
 * Классифицирует Android-приложение по `packageName` (и опциональному
 * `label`) в одну из 18 категорий `AppCategory` (BANK..PRODUCTIVITY).
 * `OTHER` (id 19) не предсказывается Dense-слоем — он зарезервирован
 * для rule-based fallback.
 *
 * ## Confidence-gated fallback
 *
 * Если softmax-вероятность top-1-категории строго меньше
 * [confidenceThreshold] (по умолчанию 0.6), вызов делегируется в
 * [ruleBased]. Это исключает «загрязнение» сенсорных фич Personal Model
 * (BANK/GOVERNMENT/EMAIL) низкокачественными ML-предсказаниями для
 * long-tail пакетов.
 *
 * ## Privacy contract
 *
 * - Читает **только** аргументы `packageName` и `label`, переданные в [classify].
 * - Никогда не обращается к `Notification.extras`, `PackageManager`, `LocationManager` и т.п.
 * - Никогда не логирует значения `packageName`/`label` через `android.util.Log`.
 * - Никаких новых runtime permissions, никаких сетевых вызовов.
 *
 * ## Threading
 *
 * `classify` вызывается с горячего пути `PersonalNotificationListenerService.onNotificationPosted`
 * (NL-сервисный поток). Доступ к [cache] сериализуется внутренним `synchronized`
 * в [CategoryCache]. TFLite [Interpreter] — single-thread safe.
 *
 * @see <a href="../../../../../../../../.kiro/specs/app-category-ml-classifier/design.md">design.md → Component 1: TFLiteAppCategoryClassifier</a>
 */
class TFLiteAppCategoryClassifier(
    private val context: Context,
    private val ruleBased: RuleBasedAppCategoryClassifier,
    private val confidenceThreshold: Float = DEFAULT_CONFIDENCE_THRESHOLD,
    private val cacheCapacity: Int = DEFAULT_CACHE_CAPACITY,
    private val tokenizer: CharNGramTokenizer,
    private val interpreter: Interpreter,
) : AppCategoryClassifier {

    private val cache = CategoryCache(cacheCapacity)

    init {
        val outputShape = interpreter.getOutputTensor(0).shape()
        val valid = outputShape.contentEquals(intArrayOf(1, EXPECTED_OUTPUT_DIM)) ||
            outputShape.contentEquals(intArrayOf(EXPECTED_OUTPUT_DIM))
        if (!valid) {
            throw IllegalStateException(
                "expected output shape [1,$EXPECTED_OUTPUT_DIM] or [$EXPECTED_OUTPUT_DIM], " +
                    "got ${outputShape.contentToString()}"
            )
        }
    }

    override fun classify(packageName: String, label: String?): AppCategory {
        // Step 1: Cache lookup — return immediately on hit (Req 3.4).
        cache.get(packageName)?.let { return it }

        return try {
            // Steps 2–5: tokenize → inference → confidence gate.
            val softmax = runInference(packageName, label)
            val top1Idx = argmax(softmax)
            val top1Conf = softmax[top1Idx]

            // Step 4: Defensive confidence gate (Req 3.5, 5.1).
            // Reject non-finite, out-of-range, or below-threshold predictions.
            val result = if (!top1Conf.isFinite() || top1Conf < 0f || top1Conf > 1f || top1Conf < confidenceThreshold) {
                // Low confidence → delegate to rule-based (Req 3.5).
                ruleBased.classify(packageName, label)
            } else {
                // Step 5: High confidence → use ML prediction.
                AppCategory.values()[top1Idx]
            }

            // Cache the final result (including rule-based fallback) (Req 3.4).
            cache.put(packageName, result)
            result
        } catch (t: Throwable) {
            // Step 6: Any exception → rule-based fallback (Req 3.11).
            val fallback = ruleBased.classify(packageName, label)
            cache.put(packageName, fallback)
            // Log the exception fact without packageName/label values (Req 5.4).
            Log.w(TAG, "tflite inference threw", t)
            fallback
        }
    }

    /**
     * Возвращает softmax-вероятность top-1-категории для данного входа.
     * Выполняет шаги 1–3 (tokenize → run → argmax) без записи в кэш.
     *
     * Используется property-тестом Requirement 7.4 — тесту нужно явно
     * сравнить уверенность с порогом и сравнить два результата `classify`
     * (TFLite vs rule-based).
     */
    @VisibleForTesting
    fun softmaxTop1Confidence(packageName: String, label: String?): Float {
        val softmax = runInference(packageName, label)
        val top1Idx = argmax(softmax)
        return softmax[top1Idx]
    }

    /**
     * Tokenizes input and runs TFLite inference, returning the raw
     * softmax output array of size [EXPECTED_OUTPUT_DIM].
     */
    private fun runInference(packageName: String, label: String?): FloatArray {
        val tokenIds = tokenizer.encode(packageName, label)
        // TFLite expects input shape [1, maxLen] for batched models.
        val input = arrayOf(tokenIds)
        val output = Array(1) { FloatArray(EXPECTED_OUTPUT_DIM) }
        interpreter.run(input, output)
        return output[0]
    }

    /**
     * Returns the index of the maximum value in [arr].
     */
    private fun argmax(arr: FloatArray): Int {
        var maxIdx = 0
        var maxVal = arr[0]
        for (i in 1 until arr.size) {
            if (arr[i] > maxVal) {
                maxVal = arr[i]
                maxIdx = i
            }
        }
        return maxIdx
    }

    companion object {
        const val DEFAULT_CONFIDENCE_THRESHOLD = 0.6f
        const val DEFAULT_CACHE_CAPACITY = 500
        const val EXPECTED_OUTPUT_DIM = 19

        private const val TAG = "TFLiteAppCatClassifier"

        /**
         * Epoch-счётчик ассетов. Бампится при каждом вызове [invalidate]
         * (после успешного скачивания обновлённых ассетов через
         * RemoteUpdateWorker). `AppCategoryClassifierFactory` сравнивает
         * текущий epoch с сохранённым — при расхождении пересоздаёт
         * инстанс классификатора (lazy reinit без перезапуска процесса).
         */
        private val assetEpoch = AtomicLong(0L)

        /**
         * Бамп asset-epoch — следующий `Factory.classify` пересоздаст инстанс.
         * Вызывается из `RemoteUpdateWorker` после успешного обновления
         * `app_category_model.tflite`, `app_category_vocab.txt` или
         * `app_category_card.json`.
         */
        fun invalidate() {
            assetEpoch.incrementAndGet()
        }

        /**
         * Текущее значение asset-epoch. Используется
         * `AppCategoryClassifierFactory.getOrCreate()` для определения,
         * нужно ли пересоздать инстанс классификатора.
         */
        fun currentAssetEpoch(): Long = assetEpoch.get()
    }
}
