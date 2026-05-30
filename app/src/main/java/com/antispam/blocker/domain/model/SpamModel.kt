package com.antispam.blocker.domain.model

import android.content.Context
import com.antispam.blocker.domain.detector.Verdict
import com.antispam.blocker.domain.scoring.CallFeatures
import com.antispam.blocker.domain.scoring.RiskLevel
import com.antispam.blocker.domain.scoring.RiskScore
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.atomic.AtomicLong

class SpamModel(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var isModelLoaded = false
    private var warmThresholds: ModelCard.Thresholds? = null
    private var coldThresholds: ModelCard.Thresholds? = null
    /** Формат выхода tflite, прочитанный из model_card.json. По умолчанию
     *  THREE_CLASS_SOFTMAX (legacy). PR-2 добавил BINARY_SIGMOID. */
    private var outputFormat: ModelCard.OutputFormat = ModelCard.OutputFormat.THREE_CLASS_SOFTMAX
    /** Эпоха ассета на момент последней удачной загрузки. Сравнивается с
     *  глобальным [assetEpoch] на каждом predict — если глобальный счётчик
     *  ушёл вперёд (RemoteUpdateWorker дёрнул [invalidate]), interpreter
     *  закрывается и tflite/model_card перечитываются. */
    private var loadedEpoch: Long = -1L

    fun loadModel(): Boolean {
        return try {
            val modelBuffer = loadModelFile()
            val options = Interpreter.Options().apply {
                setNumThreads(2)
            }
            interpreter?.close()
            interpreter = Interpreter(modelBuffer, options)
            // Load thresholds from model_card.json (if present); falls back to argmax otherwise.
            // Phase 4A: load both warm and cold sets. Cold thresholds are picked at
            // inference time when the feature vector signals cold-start (no metadata).
            val card = ModelCard.load(context)
            warmThresholds = card?.thresholds
            coldThresholds = card?.coldThresholds
            outputFormat = card?.outputFormat ?: ModelCard.OutputFormat.THREE_CLASS_SOFTMAX
            // Sanity: размер выходного тензора должен совпасть с ожиданием формата.
            val interpOutputSize = interpreter?.getOutputTensor(0)?.shape()?.lastOrNull()
            if (interpOutputSize != null && interpOutputSize != outputFormat.tensorSize) {
                android.util.Log.w(
                    "SpamModel",
                    "model_card output_format=${outputFormat.cardValue} expects " +
                        "tensor size=${outputFormat.tensorSize} but tflite has $interpOutputSize. " +
                        "Falling back to argmax-style output for safety."
                )
                outputFormat = if (interpOutputSize == 1)
                    ModelCard.OutputFormat.BINARY_SIGMOID
                else
                    ModelCard.OutputFormat.THREE_CLASS_SOFTMAX
            }
            isModelLoaded = true
            loadedEpoch = assetEpoch.get()
            true
        } catch (e: Exception) {
            android.util.Log.e("SpamModel", "Failed to load TFLite model", e)
            isModelLoaded = false
            false
        }
    }

    /** Эпоха ассета актуальная для всех инстансов SpamModel в процессе. */
    @Synchronized
    private fun reloadIfStale() {
        if (loadedEpoch != assetEpoch.get() && isModelLoaded) {
            android.util.Log.i(
                "SpamModel",
                "Asset epoch advanced (was=$loadedEpoch now=${assetEpoch.get()}); reloading tflite + model_card"
            )
            loadModel()
        }
    }

    fun predict(features: CallFeatures): RiskScore? {
        reloadIfStale()
        val interp = interpreter ?: return null
        if (!isModelLoaded) return null

        return try {
            // Phase 4D: определяем cold-start ДО сборки входного вектора,
            // чтобы применить make_cold_view-маску (обнуление 9 фичей +
            // noMetadata=1) в toFloatArray. Без этого rule-based
            // `reputationScore`/`sourceConfidence` смещают вход в «warm-подобную»
            // зону, и студент ошибочно отвечает ALLOW≈0.999.
            val isColdStart = features.noMetadata
                && !features.inAllowlist
                && !features.inBlacklist
            val input = features.toFloatArray(maskColdStart = isColdStart)
            val expectedInputSize = interp.getInputTensor(0).shape().lastOrNull() ?: input.size
            if (expectedInputSize != input.size) {
                android.util.Log.w("SpamModel", "Model input size mismatch: model=$expectedInputSize features=${input.size}")
                return null
            }
            val inputBuffer = ByteBuffer.allocateDirect(input.size * 4).order(ByteOrder.nativeOrder())
            input.forEach { inputBuffer.putFloat(it) }
            inputBuffer.rewind()

            val outputSize = outputFormat.tensorSize
            val outputBuffer = ByteBuffer.allocateDirect(outputSize * 4).order(ByteOrder.nativeOrder())

            interp.run(inputBuffer, outputBuffer)
            outputBuffer.rewind()

            // PR-2: единая логика — выводим вердикт + распределение
            // вероятностей по 3 классам ([allow, warn, block]) в обоих форматах.
            // В случае BINARY_SIGMOID выход сетки — одна калиброванная p_spam,
            // вокруг которой и определяются warn/block. WARN-зона —
            // [warn_threshold, block_threshold) над p_spam.
            val probs = parseModelOutput(outputBuffer, outputFormat)
            val allow = probs[0]
            val warn = probs[1]
            val block = probs[2]
            val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0

            // Apply per-class thresholds from model_card.json (when available);
            // otherwise fall back to plain argmax.
            //
            // Phase 4A: pick cold thresholds when the feature vector indicates a true
            // cold-start — noMetadata=1 AND no list hint. Otherwise use warm thresholds.
            // If coldThresholds is null (legacy model_card.json), behavior is identical
            // to the warm-only path.
            val thr = if (isColdStart && coldThresholds != null) coldThresholds else warmThresholds
            val verdict = if (thr != null) {
                when {
                    block >= thr.blockThreshold -> Verdict.BLOCK
                    warn >= thr.warnThreshold -> Verdict.WARN
                    else -> Verdict.ALLOW
                }
            } else {
                when (maxIdx) {
                    0 -> Verdict.ALLOW
                    1 -> Verdict.WARN
                    2 -> Verdict.BLOCK
                    else -> Verdict.ALLOW
                }
            }
            // Confidence still tracks argmax probability so UI gauges keep meaning.
            val confidence = probs[maxIdx]

            val score = (block * 100).toInt().coerceIn(0, 100)
            val level = when {
                score >= 70 -> RiskLevel.DANGEROUS
                score >= 35 -> RiskLevel.SUSPICIOUS
                else -> RiskLevel.SAFE
            }

            RiskScore(
                score = score,
                level = level,
                verdict = verdict,
                reasons = emptyList(),
                confidence = when {
                    confidence > 0.8f -> RiskScore.Confidence.HIGH
                    confidence > 0.5f -> RiskScore.Confidence.MEDIUM
                    else -> RiskScore.Confidence.LOW
                },
                source = "server_model",
                modelProbabilities = floatArrayOf(allow, warn, block),
                ruleScore = score,
                activeFactorIds = emptyList(),
                modelInputSize = expectedInputSize
            )
        } catch (e: Exception) {
            android.util.Log.e("SpamModel", "Prediction failed", e)
            null
        }
    }

    /**
     * Преобразует сырой выход tflite в тройку (allow, warn, block).
     *
     * PR-2:
     *   * THREE_CLASS_SOFTMAX: тензор [allow, warn, block] напрямую.
     *   * BINARY_SIGMOID: тензор [p_spam], откалиброван внутри графа.
     *     В этом формате модель **не делит** spam на warn vs block —
     *     это делается за её пределами через два порога над p_spam.
     *     Чтобы downstream UI и threshold-логика остались единообразными,
     *     раскладываем p_spam так:
     *       allow = 1 - p_spam
     *       warn  = p_spam   (триггерится по warn_threshold)
     *       block = p_spam   (триггерится по block_threshold)
     *     Тогда сравнения `block >= block_threshold` и
     *     `warn >= warn_threshold` работают идентично 3-class коду.
     *     Выводимое argmax будет ALLOW при p<0.5, иначе BLOCK — нормально,
     *     WARN-вердикт идёт через threshold-ветку, а не argmax.
     */
    private fun parseModelOutput(
        outputBuffer: ByteBuffer,
        format: ModelCard.OutputFormat
    ): FloatArray {
        return when (format) {
            ModelCard.OutputFormat.THREE_CLASS_SOFTMAX -> {
                val allow = outputBuffer.getFloat()
                val warn = outputBuffer.getFloat()
                val block = outputBuffer.getFloat()
                floatArrayOf(allow, warn, block)
            }
            ModelCard.OutputFormat.BINARY_SIGMOID -> {
                val pSpam = outputBuffer.getFloat().coerceIn(0f, 1f)
                floatArrayOf(1f - pSpam, pSpam, pSpam)
            }
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
        isModelLoaded = false
    }

    private fun loadModelFile(): ByteBuffer {
        val file = java.io.File(context.filesDir, MODEL_FILENAME)
        if (!file.exists()) {
            // Try loading from assets as fallback
            val assetFd = context.assets.openFd(MODEL_FILENAME)
            val inputStream = FileInputStream(assetFd.fileDescriptor)
            val buffer = inputStream.channel.map(
                FileChannel.MapMode.READ_ONLY,
                assetFd.startOffset,
                assetFd.declaredLength
            )
            return buffer
        }
        val inputStream = FileInputStream(file)
        val buffer = inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY,
            0,
            file.length()
        )
        return buffer
    }

    companion object {
        const val MODEL_FILENAME = "spam_model.tflite"
        const val INPUT_SIZE = CallFeatures.FEATURE_COUNT

        /**
         * Глобальный счётчик «ассеты обновлены, нужно перезагрузить tflite +
         * model_card». Бамп этого значения форсит первый же [predict] на любом
         * живом инстансе SpamModel перечитать файлы.
         *
         * Используется [com.antispam.blocker.data.worker.RemoteUpdateWorker]:
         * после удачного скачивания spam_model.tflite или model_card.json
         * вызывается [invalidate]. Это позволяет получить свежую модель в
         * том же процессе без ожидания пересоздания CallScreeningService.
         */
        private val assetEpoch = AtomicLong(0L)

        fun invalidate() {
            assetEpoch.incrementAndGet()
        }
    }
}
