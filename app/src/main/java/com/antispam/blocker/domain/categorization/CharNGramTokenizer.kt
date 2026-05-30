package com.antispam.blocker.domain.categorization

import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.text.Normalizer

/**
 * Воспроизводит в Kotlin ту же char-n-gram-токенизацию, что используется
 * в `train_app_category_classifier.py`. Без побайтового совпадения
 * токенизации softmax-выход TFLite будет случайным мусором.
 *
 * ## Формат файла `app_category_vocab.txt`
 *
 * - Кодировка: UTF-8 без BOM.
 * - Line endings: LF (`\n`), не CRLF.
 * - По одной строке на токен, в порядке возрастания id
 *   (строка 0 → id 0 = `<PAD>`, строка 1 → id 1 = `<UNK>`, строка K → id K).
 * - Без пустых строк.
 * - Trailing newline после последней строки.
 *
 * ## Encode-алгоритм (побайтово совпадает с Python)
 *
 * 1. Нормализация input: `text = packageName + (label?.let { " $it_nfc" } ?: "")`.
 *    `normalizeLabel` = NFC-нормализация, trim, take(200); пустая → опускается.
 * 2. Для каждого `n ∈ {3, 4, 5}`: для каждой позиции `i ∈ [0, text.length - n]`
 *    — извлечь n-gram `text.substring(i, i+n)`, lookup `tokenToId[ngram] ?: UNK_ID`.
 * 3. Конкатенация всех id-списков в порядке `n=3 first, потом n=4, потом n=5`.
 * 4. Truncate до `maxLen`, либо pad с `PAD_ID` справа до `maxLen`.
 *
 * @see <a href="../../../../../../../../.kiro/specs/app-category-ml-classifier/design.md">design.md → Component 2: CharNGramTokenizer</a>
 */
class CharNGramTokenizer(
    private val tokenToId: Map<String, Int>,
    private val maxLen: Int = DEFAULT_MAX_LEN,
    private val nGramSizes: IntArray = intArrayOf(3, 4, 5),
) {

    /**
     * Кодирует `packageName` (и опциональный `label`) в массив token-id
     * фиксированной длины [maxLen], right-padded с [PAD_ID].
     *
     * @param packageName Android package id, например `ru.sberbankmobile`.
     * @param label Опциональное локализованное имя приложения.
     * @return IntArray длины [maxLen].
     */
    fun encode(packageName: String, label: String?): IntArray {
        val normalizedLabel = label?.let { normalizeLabel(it) }
        val text = if (!normalizedLabel.isNullOrEmpty()) {
            "$packageName $normalizedLabel"
        } else {
            packageName
        }

        val ids = mutableListOf<Int>()
        for (n in nGramSizes) {
            for (i in 0..text.length - n) {
                val ngram = text.substring(i, i + n)
                ids.add(tokenToId[ngram] ?: UNK_ID)
            }
        }

        val result = IntArray(maxLen) { PAD_ID }
        val copyLen = minOf(ids.size, maxLen)
        for (i in 0 until copyLen) {
            result[i] = ids[i]
        }
        return result
    }

    companion object {
        const val PAD_ID = 0
        const val UNK_ID = 1
        const val DEFAULT_MAX_LEN = 64

        private const val PAD_TOKEN = "<PAD>"
        private const val UNK_TOKEN = "<UNK>"

        /**
         * Загружает vocab из указанного [source] и возвращает готовый
         * [CharNGramTokenizer]. Валидирует структуру файла:
         * - нет пустых строк,
         * - нет дубликатов,
         * - `tokens[0] == "<PAD>"`,
         * - `tokens[1] == "<UNK>"`.
         *
         * При любом нарушении бросает [IllegalStateException] без
         * раскрытия значений токенов (privacy contract).
         */
        fun load(source: VocabSource): CharNGramTokenizer {
            val lines = readLines(source)
            validate(lines)
            val tokenToId = LinkedHashMap<String, Int>(lines.size)
            for ((index, token) in lines.withIndex()) {
                tokenToId[token] = index
            }
            return CharNGramTokenizer(tokenToId)
        }

        /**
         * Сериализует список токенов в формат vocab-файла: UTF-8 без BOM,
         * LF line endings, по одному токену на строку, без пустых строк,
         * trailing newline.
         *
         * Используется для round-trip property test (Requirement 7.5).
         */
        fun writeVocab(tokens: List<String>, sink: OutputStream) {
            sink.bufferedWriter(Charsets.UTF_8).use { writer ->
                for (token in tokens) {
                    writer.write(token)
                    writer.write("\n")
                }
            }
        }

        private fun readLines(source: VocabSource): List<String> = when (source) {
            is VocabSource.FromFile -> source.file.bufferedReader(Charsets.UTF_8)
                .use { it.readLines() }
            is VocabSource.FromAsset -> source.inputStream().bufferedReader(Charsets.UTF_8)
                .use { it.readLines() }
            is VocabSource.FromBytes -> source.data.inputStream().bufferedReader(Charsets.UTF_8)
                .use { it.readLines() }
        }

        private fun validate(lines: List<String>) {
            check(lines.isNotEmpty()) {
                "Vocab file is empty"
            }
            check(lines.size >= 2) {
                "Vocab file must contain at least 2 tokens (<PAD> and <UNK>)"
            }
            check(lines[0] == PAD_TOKEN) {
                "Vocab file: token at index 0 must be <PAD>"
            }
            check(lines[1] == UNK_TOKEN) {
                "Vocab file: token at index 1 must be <UNK>"
            }
            val seen = HashSet<String>(lines.size)
            for ((index, token) in lines.withIndex()) {
                check(token.isNotEmpty()) {
                    "Vocab file: empty token at line $index"
                }
                check(seen.add(token)) {
                    "Vocab file: duplicate token at line $index"
                }
            }
        }

        /**
         * NFC-нормализация label: NFC, trim, max 200 code points.
         * Пустая строка после обработки → возвращает пустую строку.
         */
        private fun normalizeLabel(label: String): String {
            val nfc = Normalizer.normalize(label, Normalizer.Form.NFC)
            val trimmed = nfc.trim()
            if (trimmed.isEmpty()) return ""
            // Ограничение по code points (Unicode символам), не по char count
            val codePointCount = trimmed.codePointCount(0, trimmed.length)
            if (codePointCount > 200) return ""
            return trimmed
        }
    }
}

/**
 * Источник данных для загрузки vocab-файла. Sealed class позволяет
 * единообразно обрабатывать три сценария:
 * - [FromFile]: файл из `filesDir` (обновлённый через RemoteUpdateWorker).
 * - [FromAsset]: имя ассета из APK `assets/` (initial fallback).
 * - [FromBytes]: байтовый массив (для тестов и round-trip property).
 */
sealed class VocabSource {
    data class FromFile(val file: File) : VocabSource()
    data class FromAsset(val name: String, private val opener: () -> InputStream) : VocabSource() {
        fun inputStream(): InputStream = opener()
    }
    data class FromBytes(val data: ByteArray) : VocabSource() {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is FromBytes) return false
            return data.contentEquals(other.data)
        }

        override fun hashCode(): Int = data.contentHashCode()
    }
}
