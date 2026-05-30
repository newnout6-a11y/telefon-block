package com.antispam.blocker.domain.categorization

import android.content.Context
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

/**
 * Выбирает, откуда читать `app_category_model.tflite` и
 * `app_category_vocab.txt` — из `filesDir` (обновлённые через
 * RemoteUpdateWorker) или из APK assets (initial fallback, поставляемый
 * внутри APK при первой установке).
 *
 * ## Atomic source rule
 *
 * Если в `filesDir` лежит только один из двух файлов (например,
 * `app_category_model.tflite` скачался, а `app_category_vocab.txt` ещё
 * нет — RemoteUpdateWorker процессит ассеты последовательно и может
 * упасть посередине), мы **не миксуем** новый model с старым vocab.
 * Берём атомарную пару целиком: либо оба из `filesDir`, либо оба из
 * APK assets. Без этого можно получить катастрофическое расхождение
 * токенизации между обучением и рантаймом.
 *
 * ## Memory-mapping
 *
 * TFLite-модель загружается через `FileChannel.map(READ_ONLY, ...)`
 * для zero-copy доступа — `Interpreter` принимает `ByteBuffer` напрямую,
 * без копирования в heap. Для APK assets используется
 * `AssetFileDescriptor.startOffset` / `declaredLength`, потому что
 * ассеты упакованы внутри APK-архива со смещением.
 *
 * @see <a href="../../../../../../../../.kiro/specs/app-category-ml-classifier/design.md">design.md → Component 4: AppCategoryAssetSource</a>
 */
data class AppCategoryAssetSource(
    /** Memory-mapped ByteBuffer TFLite-модели из выбранного источника. */
    val modelByteBuffer: ByteBuffer,
    /** Источник vocab-файла для загрузки в [CharNGramTokenizer]. */
    val vocabSource: VocabSource,
    /** Откуда была взята пара — для логов и тестов. */
    val origin: Origin,
) {

    enum class Origin { FILES_DIR, APK_ASSETS }

    companion object {
        const val MODEL_FILENAME = "app_category_model.tflite"
        const val VOCAB_FILENAME = "app_category_vocab.txt"

        /**
         * Разрешает источник ассетов для App Category Model.
         *
         * Приоритет:
         * 1. Оба файла в `filesDir` → [Origin.FILES_DIR].
         * 2. Иначе оба файла из APK `assets/` → [Origin.APK_ASSETS].
         * 3. Если ни в одном месте полной пары нет → `null`
         *    (Factory переключится на rule-based).
         *
         * **Atomic source rule**: если в `filesDir` есть только один из
         * двух файлов, берём APK pair целиком — не миксуем.
         *
         * @return готовый [AppCategoryAssetSource] или `null` если ассеты
         *   недоступны ни в одном из источников.
         */
        fun resolve(context: Context): AppCategoryAssetSource? {
            // ── 1. Попытка filesDir (оба файла должны присутствовать) ────
            val modelFile = File(context.filesDir, MODEL_FILENAME)
            val vocabFile = File(context.filesDir, VOCAB_FILENAME)

            if (modelFile.exists() && vocabFile.exists()) {
                return fromFilesDir(modelFile, vocabFile)
            }

            // ── 2. Fallback на APK assets ───────────────────────────────
            return fromApkAssets(context)
        }

        /**
         * Загружает пару из `filesDir`. Memory-map модели через FileChannel.
         */
        private fun fromFilesDir(modelFile: File, vocabFile: File): AppCategoryAssetSource? {
            return try {
                val modelBuffer = FileInputStream(modelFile).use { fis ->
                    fis.channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        0,
                        modelFile.length(),
                    )
                }
                AppCategoryAssetSource(
                    modelByteBuffer = modelBuffer,
                    vocabSource = VocabSource.FromFile(vocabFile),
                    origin = Origin.FILES_DIR,
                )
            } catch (_: Exception) {
                // IO-ошибка при чтении filesDir — пробуем APK assets ниже
                null
            }
        }

        /**
         * Загружает пару из APK assets. Использует `AssetFileDescriptor`
         * для memory-map TFLite с учётом смещения внутри APK-архива.
         *
         * Возвращает `null` если ассеты отсутствуют в APK (теоретически
         * невозможно для release-сборки, но защищаемся).
         */
        private fun fromApkAssets(context: Context): AppCategoryAssetSource? {
            return try {
                val assetFd = context.assets.openFd(MODEL_FILENAME)
                val modelBuffer = FileInputStream(assetFd.fileDescriptor).use { fis ->
                    fis.channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        assetFd.startOffset,
                        assetFd.declaredLength,
                    )
                }
                AppCategoryAssetSource(
                    modelByteBuffer = modelBuffer,
                    vocabSource = VocabSource.FromAsset(VOCAB_FILENAME) {
                        context.assets.open(VOCAB_FILENAME)
                    },
                    origin = Origin.APK_ASSETS,
                )
            } catch (_: Exception) {
                // Ассеты отсутствуют в APK — модель недоступна
                null
            }
        }
    }
}
