# Implementation Plan: App Category ML Classifier

## Overview

Эта спецификация добавляет on-device char-CNN TFLite-классификатор приложений (`App_Category_Model`) поверх существующего `RuleBasedAppCategoryClassifier`, с confidence-gated fallback. Реализация делится на пять крупных направлений:

1. **Python offline-пайплайн** — crawlers + Dataset_Builder + stratified splits + Training_Pipeline (char-CNN, AdamW, dynamic-range-quantization, quality gates).
2. **Kotlin runtime** — `TFLiteAppCategoryClassifier`, `CharNGramTokenizer`, `CategoryCache`, `AppCategoryAssetSource`, расширение `AppCategoryClassifierFactory`.
3. **Дистрибуция** — расширение `RemoteUpdateWorker` тремя новыми ассетами (`app_category_model.tflite`, `app_category_vocab.txt`, `app_category_card.json`) и обновление `build_release_manifest.py`.
4. **UI-интеграция** — отображение категории в `PrivacyTransparencyScreen` для каждого live-сэмпла foreground-приложения.
5. **Property-based и structural тесты** — 16 свойств из дизайн-документа реализуются как PBT (Hypothesis в Python, kotest-property в Kotlin); отдельный набор structural / smoke / integration тестов покрывает контракт-инварианты Requirements 3, 4, 5, 6.

Конвенция:

- Сначала пишутся чистые компоненты с минимальными зависимостями (нормализация, dedup, токенизатор, кэш).
- Затем компоненты, которые их используют (Dataset_Builder, Training_Pipeline, TFLite-классификатор).
- В конце — интеграция в Factory, Settings, RemoteUpdateWorker и UI; финальная пачка cross-cutting тестов (enum-order, toNotificationBucket, regression, sensitive integration, structural smokes, distribution E2E).

Каждое property test sub-task ссылается на конкретное Property из `design.md` и Requirements clause из `requirements.md`. Тестовые sub-tasks помечены `*` (опциональные); core implementation tasks — нет.

## Tasks

- [x] 1. Set up project structure for new Python pipeline
  - [x] 1.1 Create directory layout and module skeletons
    - Создать каталоги `scripts/crawlers/`, `datasets/categories/raw/`, `tests/`.
    - Создать пустые модули: `scripts/build_app_category_dataset.py`, `scripts/build_app_category_splits.py`, `scripts/crawlers/play_store_crawler.py`, `scripts/crawlers/rustore_crawler.py`, `scripts/crawlers/huawei_appgallery_crawler.py`, `tests/test_build_app_category_dataset.py`, `tests/test_build_app_category_splits.py`, `tests/test_train_app_category_classifier.py`.
    - В каждом модуле положить минимальный `if __name__ == "__main__": ...` или pytest-стуб.
    - Расширить `requirements-dev.txt` (или эквивалент): добавить `tensorflow`, `scikit-learn`, `hypothesis`, `pytest` фиксированных версий.
    - Закоммитить статический `KOTLIN_APP_CATEGORY_ORDER` (20 строк) в `scripts/train_app_category_classifier.py` как single-source-of-truth для Python-стороны.
    - _Requirements: 1.1, 2.10_

- [x] 2. Implement Dataset_Builder pure functions
  - [x] 2.1 Implement label normalization (`normalize_label`)
    - Реализовать функцию `normalize_label(s: str) -> str` в `scripts/build_app_category_dataset.py`: NFC-нормализация, strip whitespace, max 200 code points, возврат `""` если результат пуст или > 200 символов.
    - _Requirements: 1.4_

  - [ ]* 2.2 Write property test for label normalization
    - **Property 9: Label normalization is well-formed**
    - **Validates: Requirements 1.4**
    - Hypothesis-стратегия: `st.text(alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x10FFFF), min_size=0, max_size=1000)`.
    - Файл: `tests/test_build_app_category_dataset.py::test_normalize_label_property`.
    - Проверить все 4 инварианта Property 9 (NFC, no leading/trailing whitespace, ≤200, fallback to `""`).

  - [x] 2.3 Implement source merge with priority (`merge_sources`)
    - Реализовать `merge_sources(rows: list[Row]) -> list[Row]` с приоритетом `BOOTSTRAP > PLAY > RUSTORE > APPGALLERY`, дедуп по `packageName` case-sensitive, label берётся из выбранной приоритетной записи.
    - _Requirements: 1.2, 1.3_

  - [ ]* 2.4 Write property test for source merge dedup
    - **Property 10: Source merge dedup with priority**
    - **Validates: Requirements 1.2, 1.3**
    - Hypothesis: генерировать `dict[str, list[(source_enum, label, category)]]`, проверять что в выходе ровно один rec на packageName с правильным priority.

  - [x] 2.5 Implement category mapping and row counters
    - Реализовать функции `map_category(raw: str) -> tuple[str, bool]` (нормализация в `AppCategory`-имя или `OTHER` с флагом unknown) и dataclass `Counters` (`total_input_rows`, `dropped_rows`, `unknown_category_rows`, `corpus_rows`, `per_category_counts`).
    - Реализовать `is_blank_package(row) -> bool` (пакет пустой после strip).
    - _Requirements: 1.9, 1.10_

  - [ ]* 2.6 Write property test for counters
    - **Property 13: Counters reflect actual decisions**
    - **Validates: Requirements 1.9, 1.10, 1.11**
    - Hypothesis: генерировать произвольные mix'ы валидных / blank-package / unknown-category строк, проверять все 5 инвариантов после прогона `process_rows`.

- [ ] 3. Implement Dataset_Builder orchestrator
  - [x] 3.1 Implement `labeled.csv` writer
    - UTF-8 без BOM, LF line endings (`csv.writer(..., lineterminator='\n')`), header `packageName,label,category`, trailing newline.
    - Atomic-write через `.tmp` + `os.replace`.
    - _Requirements: 1.5_

  - [x] 3.2 Implement `build_report.json` writer
    - Поля: `total_input_rows`, `dropped_rows`, `unknown_category_rows`, `corpus_rows`, `per_category_counts` (20 ключей включая OTHER), `seed`, `built_at` (ISO-8601 UTC).
    - _Requirements: 1.11_

  - [x] 3.3 Implement corpus size validation
    - После dedup: assert `len(unique_packages) >= 200_000` и `per_category_counts[cat] >= 5_000` для всех `cat ∈ AppCategory \ {OTHER}`. На несоответствии — exit code 1 с явным сообщением.
    - _Requirements: 1.6_

  - [x] 3.4 Wire `build_app_category_dataset.py` main()
    - Принять CLI args: `--seed=42`, пути к raw CSV-источникам, путь к Bootstrap_Seed.
    - Последовательность: load raw → bootstrap from `RuleBasedAppCategoryClassifier` known packages → normalize labels → map categories → drop blanks → merge sources → write `labeled.csv` → write `build_report.json` → validate corpus size.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.9, 1.10, 1.11_

  - [ ]* 3.5 Write CSV format byte-inspection test
    - Toy corpus → write `labeled.csv` → читать сырые байты файла; assert: нет BOM (`\xEF\xBB\xBF`), нет `\r` (только LF), header байт-равен `packageName,label,category\n`, trailing newline присутствует.
    - Файл: `tests/test_build_app_category_dataset.py::test_csv_format_byte_inspection`.
    - _Requirements: 1.5_

  - [ ]* 3.6 Write build_report.json schema test
    - Toy corpus → `build_report.json` → `json.load`; assert все 7 полей присутствуют с корректными типами (`total_input_rows: int`, `built_at: str` matches ISO-8601, `per_category_counts` имеет ровно 20 ключей включая `OTHER`).
    - Файл: `tests/test_build_app_category_dataset.py::test_build_report_schema`.
    - _Requirements: 1.11_

- [x] 4. Implement Splits builder
  - [x] 4.1 Implement `build_app_category_splits.py`
    - sklearn `train_test_split(stratify=y, random_state=seed)` дважды (80/20, потом 50/50 на 20%).
    - Disjoint-инвариант обеспечивается split'ом по индексам уникальных packages после dedup.
    - Записать `train.csv`, `val.csv`, `test.csv` с тем же CSV-форматом (UTF-8, LF, trailing newline) atomic-write.
    - _Requirements: 1.7, 1.8_

  - [ ]* 4.2 Write property test for stratified split
    - **Property 11: Stratified split is disjoint and total**
    - **Validates: Requirements 1.7**
    - Hypothesis: генерировать корпуса 100..2000 строк с минимум 10 на категорию; проверять disjoint, total, stratified-tolerance.

- [x] 5. Implement crawler scripts
  - [x] 5.1 Implement Google Play Store crawler
    - `scripts/crawlers/play_store_crawler.py`: итерация по category pages, извлечение `(packageName, displayName, googleCategory)`, запись в `datasets/categories/raw/play_store.csv` (UTF-8, LF, header `packageName,label,category`).
    - _Requirements: 1.1_

  - [x] 5.2 Implement RuStore crawler
    - `scripts/crawlers/rustore_crawler.py`, формат как у Play Store, выход в `raw/rustore.csv`.
    - _Requirements: 1.1_

  - [x] 5.3 Implement Huawei AppGallery crawler
    - `scripts/crawlers/huawei_appgallery_crawler.py`, выход в `raw/appgallery.csv`.
    - _Requirements: 1.1_

  - [ ]* 5.4 Write per-crawler unit tests with recorded HTTP fixtures
    - Файлы: `tests/test_play_store_crawler.py`, `tests/test_rustore_crawler.py`, `tests/test_huawei_appgallery_crawler.py`.
    - Mock HTTP responses (recorded fixtures в `tests/fixtures/`); запуск crawler-функции; assert структура output `(packageName, displayName, googleCategory)` тройки, корректная UTF-8 LF CSV запись.
    - _Requirements: 1.1_

- [x] 6. Checkpoint - Dataset pipeline tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement Training_Pipeline core
  - [x] 7.1 Implement CharNGramVocab and dataset encoder (Python side)
    - Класс `CharNGramVocab`: `build(rows, n_grams=(3,4,5), max_size)`, метод `encode(packageName, label, max_len=64) -> np.ndarray[int]`, `serialize() -> str` (UTF-8 без BOM, LF, по одной строке на токен в порядке возрастания id, `<PAD>` на строке 0, `<UNK>` на строке 1, trailing newline).
    - Реализовать `encode_dataset(df, vocab, batch_size)` через `tf.data.Dataset`.
    - _Requirements: 2.2, 2.8_

  - [x] 7.2 Implement char-CNN model architecture
    - `build_char_cnn_model(vocab_size, max_len=64, embed_dim=32, conv_filters=128, kernel_sizes=(3,5,7), num_classes=18)`: Embedding → 3 параллельных Conv1D → GlobalMaxPool → Concat → Dense(18, softmax).
    - _Requirements: 2.2_

  - [ ]* 7.2a Write architecture smoke test
    - Файл: `tests/test_train_app_category_classifier.py::test_architecture`.
    - Build model → inspect `model.layers`; assert: 3 Conv1D с `filters=128` и `kernel_size ∈ {3,5,7}`, 3 GlobalMaxPool, Concatenate, Dense(18) с softmax-активацией.
    - _Requirements: 2.2_

  - [x] 7.3 Implement quality gate function
    - `check_quality_gates(metrics: dict) -> list[Failure]`: возвращает `[]` IFF `top1_accuracy ≥ 0.90 ∧ macro_f1 ≥ 0.85 ∧ precision[BANK,GOVERNMENT,EMAIL] ≥ 0.95`.
    - _Requirements: 2.5_

  - [ ]* 7.4 Write property test for quality gate function
    - **Property 14: Quality gate function is correct**
    - **Validates: Requirements 2.5, 2.12**
    - Hypothesis: генерировать словари `metrics` со значениями ∈ [0,1]; проверить логическую эквивалентность `failures == []` ⇔ все 5 условий выполнены; проверить также что при `failures != []` файлы по `args.output`/`args.vocab`/`args.card` остаются неизменными (через SHA256 до/после в фикстуре).

  - [x] 7.5 Implement evaluation and metrics computation
    - `evaluate(model, test_df, vocab) -> dict` с полями `top1_accuracy`, `macro_f1`, `per_category` (18 ключей с `precision`/`recall`/`f1`).
    - _Requirements: 2.4_

  - [ ]* 7.5a Write metrics population test
    - Toy training run на mini-датасете; assert `metrics` dict содержит `top1_accuracy`, `macro_f1`, `per_category` (18 ключей без `OTHER`), все значения ∈ [0,1].
    - Файл: `tests/test_train_app_category_classifier.py::test_metrics_populated`.
    - _Requirements: 2.4_

  - [x] 7.6 Implement TFLite conversion with dynamic-range quantization
    - `convert_to_tflite_quantized(model) -> bytes`: `tf.lite.TFLiteConverter` с `optimizations=[tf.lite.Optimize.DEFAULT]` (int8 weights, fp32 activations).
    - Size check: `len(b) > 1_048_576` → exit 2, cleanup `.tmp`-файлов, не записывать ни один из трёх артефактов.
    - _Requirements: 2.6, 2.7, 2.11_

  - [ ]* 7.7 Write property test for TFLite size guard with cleanup
    - **Property 15: TFLite size guard with cleanup**
    - **Validates: Requirements 2.7, 2.11**
    - Hypothesis: blob ∈ {boundary cases: 1_048_576 ± 1 байт, случайные большие/маленькие байтовые последовательности}. Проверить exit code 2, отсутствие output-файла, неизменность ранее существующих файлов через SHA256.

  - [x] 7.8 Implement Model_Card writer
    - JSON-формат: `schema_version=1`, `model_id`, `trained_at`, `categories_order` (20 строк, включая OTHER), `total_train_rows`, `metrics.top1_accuracy`, `metrics.macro_f1`, `metrics.per_category` (18 ключей без OTHER, каждый с `precision`/`recall`/`f1`).
    - _Requirements: 2.9_

  - [ ]* 7.8a Write Model_Card schema test
    - Synthetic metrics → render card → parse `app_category_card.json` → assert: `schema_version == 1`, `categories_order` длиной ровно 20 (включая `OTHER`), `metrics.per_category` ровно 18 ключей (без `OTHER`), все метрики ∈ [0,1], `total_train_rows >= 0`.
    - Файл: `tests/test_train_app_category_classifier.py::test_card_schema`.
    - _Requirements: 2.9_

  - [x] 7.9 Implement enum-order check and atomic write helper
    - `compare_enum_order(python_list, kotlin_list)` возвращает `None` если списки идентичны или `(idx, k_val, p_val)` первой расхождения.
    - `write_atomic(path, content)`: `.tmp` → `os.replace`.
    - _Requirements: 2.10_

  - [ ]* 7.9a Write atomic write test
    - Mock `os.replace` → run `write_atomic(target, content)` → assert: создан `<target>.tmp`, потом `os.replace(.tmp, target)` вызван ровно один раз. Сторонний crash между write и replace оставляет `.tmp` cleaned via `try/finally`.
    - Файл: `tests/test_train_app_category_classifier.py::test_atomic_write`.
    - _Requirements: 2.6_

  - [x] 7.10 Wire `train_app_category_classifier.py` main()
    - Аргументы: `--train`, `--val`, `--test`, `--seed=42`, `--output`, `--vocab`, `--card`.
    - Детерминизм: `set_random_seed`, `tf.config.experimental.enable_op_determinism()`, фиксированный subsample для quantization-калибровки.
    - Последовательность: load → vocab.build → model.compile (AdamW, CosineDecay 1e-3, batch 256, 30 эпох) → fit → evaluate → quality gate (exit 1 при fail) → tflite convert (exit 2 при > 1 MB) → enum check (exit 3 при mismatch) → atomic write 3 артефактов → exit 0.
    - _Requirements: 2.1, 2.3, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12_

  - [ ]* 7.10a Write optimizer config test
    - Inspect `model.optimizer` после compile; assert: класс `AdamW`, начальный LR == 1e-3, schedule == `CosineDecay`, batch_size == 256, epochs == 30 (читается через config-arg или `fit_config`).
    - Файл: `tests/test_train_app_category_classifier.py::test_optimizer_config`.
    - _Requirements: 2.3_

  - [ ]* 7.11 Write property test for pipeline determinism
    - **Property 12: Pipeline determinism (Dataset_Builder + Training_Pipeline)**
    - **Validates: Requirements 1.8, 2.1**
    - Запустить `build_app_category_dataset.py` + `build_app_category_splits.py` дважды на одних и тех же raw CSV с `--seed=42`, проверить SHA256 равенство всех output-файлов.
    - Запустить `train_app_category_classifier.py` дважды на тех же splits с `--seed=42` (на маленьком подкорпусе для скорости), проверить SHA256 равенство `app_category_model.tflite`, `app_category_vocab.txt`, `app_category_card.json`.

- [x] 8. Checkpoint - Training pipeline tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement Kotlin runtime — pure components
  - [x] 9.1 Implement `CharNGramTokenizer`
    - Файл: `app/src/main/java/com/antispam/blocker/domain/categorization/CharNGramTokenizer.kt`.
    - Поля: `tokenToId: Map<String, Int>`, `maxLen=64`, `nGramSizes=intArrayOf(3,4,5)`, константы `PAD_ID=0`, `UNK_ID=1`.
    - Метод `encode(packageName, label?) -> IntArray[maxLen]`: NFC-нормализация label, конкатенация `"${packageName} ${label}"`, char-n-gram токенизация для каждого `n ∈ {3,4,5}`, lookup в `tokenToId` (миссы → `UNK_ID`), truncate/pad до `maxLen` справа `PAD_ID`.
    - Метод `companion fun load(source: VocabSource)`: парсинг файла построчно, валидация (нет пустых, нет дублей, `tokens[0]=="<PAD>"`, `tokens[1]=="<UNK>"`); на нарушении бросает `IllegalStateException` без значений токенов.
    - Метод `companion fun writeVocab(tokens, sink: BufferedSink)`: запись UTF-8 без BOM, LF, без пустых строк, trailing newline.
    - Sealed class `VocabSource { FromFile(File); FromAsset(name); FromBytes(ByteArray) }`.
    - _Requirements: 2.8_

  - [ ]* 9.2 Write property test for tokenizer round-trip
    - **Property 5: Tokenizer_Vocab round-trip**
    - **Validates: Requirements 2.8, 7.5**
    - Файл: `app/src/test/java/com/antispam/blocker/categorization/CharNGramTokenizerRoundTripTest.kt`.
    - kotest-property: генерировать списки уникальных строк `T` (с `t_0="<PAD>"`, `t_1="<UNK>"`, остальные ≠ "" и попарно различны), проверять `writeVocab→load` round-trip, отсутствие BOM, отсутствие CR (`0x0D`), `tokenToIdMap` равенство.

  - [x] 9.3 Implement `CategoryCache`
    - Файл: `app/src/main/java/com/antispam/blocker/domain/categorization/CategoryCache.kt`.
    - `LinkedHashMap<String, AppCategory>(capacity, 0.75f, accessOrder=true)` с `removeEldestEntry`, обёрнут в `@Synchronized` методы `get/put/size/clear`.
    - **Privacy contract**: only-process-memory, не персиститься.
    - _Requirements: 3.4, 5.9_

  - [ ]* 9.4 Write property test for cache idempotence + LRU eviction
    - **Property 6: Category_Cache idempotence, capacity, and LRU eviction**
    - **Validates: Requirements 3.4, 7.6**
    - Файл: `app/src/test/java/com/antispam/blocker/categorization/CategoryCacheIdempotencyTest.kt`.
    - Тестировать через `TFLiteAppCategoryClassifier` с моком `Interpreter` и счётчиком `inferCount`; явно assert: 5 вызовов = 1 inference, capacity=500, eviction `p_1` после `p_501`.

- [x] 10. Implement Kotlin runtime — asset source
  - [x] 10.1 Implement `AppCategoryAssetSource`
    - Файл: `app/src/main/java/com/antispam/blocker/domain/categorization/AppCategoryAssetSource.kt`.
    - `data class` с полями `modelByteBuffer: ByteBuffer`, `vocabSource: VocabSource`, `origin: Origin {FILES_DIR, APK_ASSETS}`.
    - `companion fun resolve(context): AppCategoryAssetSource?`: priority filesDir-pair → APK-pair; **atomic source rule** — если в filesDir только один из двух, берём APK pair полностью; если обоих нет нигде, возвращает `null`.
    - Memory-map TFLite через `FileChannel.map(MapMode.READ_ONLY, ...)`.
    - _Requirements: 3.2, 4.5, 4.7_

- [ ] 11. Implement TFLiteAppCategoryClassifier
  - [x] 11.1 Implement constructor and shape validation
    - Файл: `app/src/main/java/com/antispam/blocker/domain/categorization/TFLiteAppCategoryClassifier.kt`.
    - Constructor принимает `context`, `ruleBased`, `confidenceThreshold=0.6f`, `cacheCapacity=500`, `tokenizer`, `interpreter`.
    - В init: `interpreter.getOutputTensor(0).shape() ∈ {[1,18], [18]}`; иначе `IllegalStateException("expected output shape [1,18] or [18], got <actual>")` (без значений `packageName`/`label`).
    - Constants: `DEFAULT_CONFIDENCE_THRESHOLD=0.6f`, `DEFAULT_CACHE_CAPACITY=500`, `EXPECTED_OUTPUT_DIM=18`.
    - `companion object { fun invalidate(); fun currentAssetEpoch() }` через `AtomicLong`.
    - _Requirements: 3.1, 3.8, 7.3_

  - [ ]* 11.2 Write property test for TFLite output dimension
    - **Property 3: TFLite output dimension is 18**
    - **Validates: Requirements 3.8, 3.9, 7.3**
    - Файл: `app/src/androidTest/java/com/antispam/blocker/categorization/TFLiteAppCategoryClassifierShapeTest.kt`.
    - kotest-property: генерировать TFLite-моки с output shape ∈ `{[1,18], [18], [17], [19], [1,17], [1,19], [2,18], [], [1,1,18]}`; проверить, что инициализация success IFF shape ∈ `{[1,18],[18]}`; иначе `IllegalStateException` и Factory переключается на singleton rule-based.

  - [x] 11.3 Implement classify() with cache, inference, confidence gate, and exception fallback
    - Алгоритм:
      1. `cache.get(packageName)` → hit: return.
      2. tokenize → `interpreter.run(tokenIds)` → `softmax: FloatArray(18)`.
      3. `top1Idx = argmax(softmax)`, `top1Conf = softmax[top1Idx]`.
      4. Defensive: `if (!top1Conf.isFinite() || top1Conf < 0f || top1Conf > 1f || top1Conf < confidenceThreshold)` → `ruleBased.classify(...)`, cache rule-based result, return.
      5. Иначе: `result = AppCategory.values()[top1Idx]`, cache, return.
      6. Любое исключение из 2–5 → catch → `ruleBased.classify(...)`, cache rule-based, `Log.w(TAG, "tflite inference threw", t)` **без** значений.
    - Реализовать `@VisibleForTesting fun softmaxTop1Confidence(packageName, label): Float` (выполняет шаги 1–3 без cache.put).
    - _Requirements: 3.3, 3.4, 3.5, 3.11, 3.12, 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 11.4 Write property test for confidence-gated fallback (≥ 200 iterations)
    - **Property 4: TFLite-unavailability ⇒ rule-based equivalence**
    - **Validates: Requirements 3.5, 3.11, 7.4**
    - Файл: `app/src/test/java/com/antispam/blocker/categorization/ConfidenceGatedFallbackPropertyTest.kt`.
    - kotest-property с min iterations 200: `packageName` ∈ Unicode `[U+0020..U+007E ∪ U+0400..U+04FF]` длина 0..100; `label` той же грамматики длина 0..200.
    - Для каждой пары: если `softmaxTop1Confidence < 0.6f` → assert `tfliteClassifier.classify(...) == ruleBased.classify(...)`.
    - Параллельно: подменять Interpreter моком, бросающим `RuntimeException`; assert тот же равенство-инвариант.
    - При нарушении вывести `(packageName, label, softmaxTop1Confidence, tflite_result, rule_based_result)`.

  - [ ]* 11.5 Write property test for no-PII-in-logs
    - **Property 8: Privacy — no input values in logs**
    - **Validates: Requirements 3.9, 3.11, 5.4, 5.9**
    - Файл: `app/src/test/java/com/antispam/blocker/categorization/PrivacyNoLogsPropertyTest.kt`.
    - Spy на `android.util.Log.{v,d,i,w,e}` через Robolectric / shadow; на каждой path of execution (cache hit, успешный inference, low-confidence fallback, exception fallback, init shape failure) проверить: ни tag, ни msg, ни throwable не содержат подстроки `packageName` или `label`. Также assert: filesDir / cacheDir / SharedPreferences / Room не получили ни одной записи относительно классификации.

  - [ ]* 11.6 Write strict-context test for classify() (no extra Android API access)
    - **Validates: Requirements 3.3, 5.1, 5.2, 5.3**
    - Файл: `app/src/test/java/com/antispam/blocker/categorization/TFLiteClassifierContextStrictTest.kt`.
    - Mock `Context` со strict mode, throwing на каждый запрещённый API (`PackageManager.getInstalledPackages/queryIntentActivities`, `LocationManager`, `ClipboardManager`, `MediaRecorder`, `AudioRecord`, `BiometricPrompt`, `content://sms`, `Notification.extras` и др.); запустить серию `classify(...)` вызовов; assert ни один из throws не сработал.

  - [ ]* 11.7 Write no-network-on-classify smoke test
    - **Validates: Requirements 5.5**
    - Файл: `app/src/test/java/com/antispam/blocker/categorization/NoNetworkOnClassifyTest.kt`.
    - MockWebServer; полный цикл `Factory.classify(...)` × 100 на разных входах; assert `server.requestCount == 0` (ни одного network call вне manifest pull).

- [ ] 12. Wire AppCategoryClassifierFactory and SettingsStore
  - [x] 12.1 Add `tfliteAppCategoryEnabled` flag to SettingsStore
    - Файл: `app/src/main/java/com/antispam/blocker/data/prefs/SettingsStore.kt`.
    - Добавить: `val tfliteAppCategoryEnabled: Flow<Boolean> = boolPref("tflite_app_category_enabled", true)`, `suspend fun setTfliteAppCategoryEnabled(enabled: Boolean)`, `fun tfliteAppCategoryEnabledSnapshot(): Boolean = runBlocking { tfliteAppCategoryEnabled.first() }`.
    - _Requirements: 3.6_

  - [ ] 12.2 Extend `AppCategoryClassifierFactory` with TFLite path
    - Файл: `app/src/main/java/com/antispam/blocker/domain/categorization/AppCategoryClassifier.kt` (тот же файл, не редактировать существующий enum / interface / RuleBased).
    - Добавить `@Volatile cached`, `assetEpoch`, `lock`, `getOrCreate()`, `invalidate()`, `tryCreateTFLite(context, rules)`.
    - Правило выбора: TFLite активен IFF `(assetSource != null) ∧ tfliteAppCategoryEnabledSnapshot() ∧ tryCreateTFLite не выбросил`.
    - При неудачной инициализации `tryCreateTFLite` ловит throwable, лог `Log.w(TAG, "TFLiteAppCategoryClassifier init failed; falling back", t)` **без** значений, возврат `null`.
    - _Requirements: 3.6, 3.7, 3.9_

  - [ ]* 12.3 Write property test for factory selection table
    - **Property 7: Factory selection table**
    - **Validates: Requirements 3.6, 3.7**
    - Файл: `app/src/test/java/com/antispam/blocker/categorization/FactorySelectionPropertyTest.kt`.
    - Перебрать все 8 комбинаций `(assetsAvailable, killSwitch, initSucceeds) ∈ (T,F)^3`; assert: возвращается `TFLiteAppCategoryClassifier` IFF все три true; иначе **тот же самый** `RuleBasedAppCategoryClassifier` инстанс на 100 последовательных вызовах в рамках процесса.

- [ ] 13. Extend RemoteUpdateWorker for new assets
  - [x] 13.1 Add three new entries to `ALLOWED_FILES` and invalidate dispatch
    - Файл: `app/src/main/java/com/antispam/blocker/data/worker/RemoteUpdateWorker.kt`.
    - В `ALLOWED_FILES`: добавить `"app_category_model.tflite"`, `"app_category_vocab.txt"`, `"app_category_card.json"`.
    - В `when (entry.localName) { ... }` добавить ветку для всех трёх → `TFLiteAppCategoryClassifier.invalidate()`.
    - SHA256/size валидация, `.tmp` rename, retry — без изменений (унаследовано).
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 13.2 Update release manifest builder
    - Файл: `scripts/build_release_manifest.py`.
    - Добавить три новых имени в whitelist filenames; для каждого посчитать `sha256` (lowercase hex, 64 символа), `size` (int bytes), `url` (без префикса `/`).
    - _Requirements: 4.1_

  - [ ]* 13.3 Write manifest entries parser smoke test
    - **Validates: Requirements 4.1**
    - Файл: `app/src/test/java/com/antispam/blocker/categorization/AppCategoryManifestEntriesTest.kt`.
    - Загрузить fixture `manifest.json` со всеми 3 новыми entries; парсинг через существующий `RemoteUpdateWorker.parseManifest`; assert каждая entry имеет `sha256` длиной ровно 64 hex-символа в lowercase, `size >= 0`, `url` не начинается с `/`.

  - [ ]* 13.4 Write ALLOWED_FILES reflection test
    - **Validates: Requirements 4.2**
    - Файл: `app/src/test/java/com/antispam/blocker/categorization/RemoteUpdateWorkerAllowedFilesTest.kt`.
    - Reflection на `RemoteUpdateWorker.ALLOWED_FILES` (или companion-object); assert содержит ровно `"app_category_model.tflite"`, `"app_category_vocab.txt"`, `"app_category_card.json"` в дополнение к существующим entries.

  - [ ]* 13.5 Write WorkManager schedule smoke test
    - **Validates: Requirements 4.6, 6.6**
    - Файл: `app/src/androidTest/java/com/antispam/blocker/categorization/RemoteUpdateWorkerScheduleTest.kt`.
    - `WorkManager.getInstance(context).getWorkInfosForUniqueWork(RemoteUpdateWorker.UNIQUE_NAME)`; assert ровно один periodic worker с интервалом 6h ± 30 min; reflective scan production-кода: нет нового `App_Category_Model`-specific worker class.

- [ ] 14. Bundle initial APK assets
  - [ ] 14.1 Add initial bundled assets
    - Положить в `app/src/main/assets/`: `app_category_model.tflite`, `app_category_vocab.txt`, `app_category_card.json` (initial fallback версии, сгенерированные `train_app_category_classifier.py` на референсном корпусе).
    - Убедиться, что `app/build.gradle.kts` `aaptOptions { noCompress("tflite") }` (если ещё нет) применяется — для memory-map.
    - _Requirements: 4.7_

- [ ] 15. Wire Privacy Transparency screen
  - [ ] 15.1 Bind category to live foreground events
    - Файл: `app/src/main/java/com/antispam/blocker/ui/screens/PrivacyTransparencyScreen.kt`.
    - Для каждого сэмпла из `RecentUserContextProvider.recentForegroundEvents` вызвать `AppCategoryClassifierFactory.classify(packageName, label)` и отобразить рядом с сэмплом.
    - _Requirements: 3.13_

  - [ ]* 15.2 Write Compose UI test for category display
    - **Validates: Requirements 3.13**
    - Файл: `app/src/androidTest/java/com/antispam/blocker/categorization/PrivacyTransparencyCategoryDisplayTest.kt`.
    - Compose `PrivacyTransparencyScreen` с тестовым `RecentUserContextProvider`-моком из 5 foreground-сэмплов; assert: каждая карточка содержит видимый текст с именем `AppCategory` (одного из 20 значений) рядом с `packageName`.

- [ ] 16. Cross-cutting backward-compatibility tests
  - [ ]* 16.1 Write smoke test for AppCategory enum order parity
    - **Property 1: Enum-order parity Kotlin ↔ Python ↔ Card**
    - **Validates: Requirements 2.10, 6.1, 6.7, 7.1, 7.2**
    - Файл: `app/src/test/java/com/antispam/blocker/categorization/AppCategoryEnumOrderSmokeTest.kt`.
    - Прочитать `categories_order` из `app/src/main/assets/app_category_card.json`, сравнить с `AppCategory.values().map { it.name }` и со статически закоммиченным `KOTLIN_APP_CATEGORY_ORDER` (импорт из единого места). На несоответствии: вывести первую расходящуюся позицию, оба значения; assertion fail → `:app:testDebugUnitTest` не пройдёт.

  - [ ]* 16.2 Write test for `toNotificationBucket()` total mapping
    - **Property 2: `toNotificationBucket()` total mapping**
    - **Validates: Requirements 6.2**
    - Файл: `app/src/test/java/com/antispam/blocker/categorization/ToNotificationBucketTest.kt`.
    - Проитерировать все 20 значений `AppCategory`, проверить точную таблицу маппинга; assert множество `{c.toNotificationBucket() | c ∈ values()} == {"BANK","MARKETPLACE","MESSENGER","EMAIL","OTHER"}`.

  - [ ]* 16.3 Write golden-file regression test for Personal Model features
    - **Property 16: Personal Model features regression**
    - **Validates: Requirements 6.5**
    - Файл: `app/src/test/java/com/antispam/blocker/categorization/DeviceFeatureExtractorRegressionTest.kt`.
    - Зафиксировать в репозитории golden snapshot (JSON) `(callEvents, notificationEvents, appUsageEvents, now)` → `expectedFeatures: float[17]`.
    - Прогнать `DeviceFeatureExtractor.extract(...)` после внедрения этой спеки; assert бит-равенство всех 17 фич, особое внимание на 6 категория-зависимых.

  - [ ]* 16.4 Write integration test for sensitive categories (BANK / GOVERNMENT / EMAIL)
    - **Validates: Requirements 7.7**
    - Файл: `app/src/test/java/com/antispam/blocker/categorization/SensitiveCategoryIntegrationTest.kt`.
    - 9 пар `(packageName, expected_category)`: 3 BANK (`ru.sberbank.online`, `ru.vtb24.mobilebanking.android`, `com.idamob.tinkoff.android`), 3 GOVERNMENT (`ru.gosuslugi.mobile`, `ru.gibdd.mobile`, `ru.fns.taxes`), 3 EMAIL (`com.google.android.gm`, `ru.mail.mailapp`, `com.yandex.mail`).
    - Для каждой пары явно проверить: `(a) tflite.classify == expected` ИЛИ `(b) softmaxTop1Confidence < 0.6f ∧ ruleBased.classify == expected`. Pass IFF (a) ∨ (b).
    - При fail вывести пару, `softmaxTop1Confidence`, фактическую категорию.

  - [ ]* 16.5 Write AppCategoryClassifier interface shape reflection test
    - **Validates: Requirements 3.1, 6.4, 6.7**
    - Файл: `app/src/test/java/com/antispam/blocker/categorization/AppCategoryClassifierInterfaceShapeTest.kt`.
    - Reflection: `KClass<AppCategoryClassifier>.declaredMembers`; assert ровно один метод `classify` с сигнатурой `(packageName: String, label: String? = null) -> AppCategory`. Любое добавление метода в интерфейс → assertion fail.

  - [ ]* 16.6 Write banned dependencies test
    - **Validates: Requirements 5.8**
    - Файл: `app/src/test/java/com/antispam/blocker/categorization/BannedDependenciesTest.kt`.
    - Парсить `app/build.gradle.kts` (release configuration); assert: ни одна dependency не соответствует pattern `com.google.firebase:firebase-analytics`, `com.google.firebase:firebase-crashlytics`, `com.appmetrica:*`, `io.sentry:*`, `com.bugsnag:*`, `com.amplitude:*`, `com.mixpanel:*`.

  - [ ]* 16.7 Write notification_event schema smoke test
    - **Validates: Requirements 6.3**
    - Файл: `app/src/test/java/com/antispam/blocker/categorization/NotificationEventSchemaSmokeTest.kt`.
    - Сравнить `app/schemas/com.antispam.blocker.data.db.AppDatabase/5.json` (snapshot before) с актуальной schema generation; assert table `notification_event` byte-equal до и после внедрения этой спецификации (если миграция не нужна по Req 6.3).

  - [ ]* 16.8 Write no-new-worker reflection test
    - **Validates: Requirements 6.6**
    - Файл: `app/src/test/java/com/antispam/blocker/categorization/NoNewWorkerTest.kt`.
    - Reflective scan production-package на subclasses `androidx.work.CoroutineWorker`/`ListenableWorker`; assert: только existing `RemoteUpdateWorker` и `TelemetryRetentionWorker` (или их аналоги в проекте) — без нового App_Category_Model worker class.

- [ ] 17. Distribution end-to-end tests
  - [ ]* 17.1 Write E2E asset update test
    - **Validates: Requirements 4.5**
    - Файл: `app/src/androidTest/java/com/antispam/blocker/categorization/AppCategoryAssetUpdateE2ETest.kt`.
    - Pre-create fake `filesDir/app_category_model.tflite` + `app_category_vocab.txt` (старая версия) → init Factory → call `classify(...)` → record category → MockWebServer serves manifest + bodies для новой версии → run `RemoteUpdateWorker` → assert files в `filesDir` обновились с корректным sha256/size → call `TFLiteAppCategoryClassifier.invalidate()` → next `Factory.classify(...)` использует новый файл (без перезапуска процесса).

  - [ ]* 17.2 Write fresh-install fallback test
    - **Validates: Requirements 4.7**
    - Файл: `app/src/androidTest/java/com/antispam/blocker/categorization/AppCategoryFreshInstallTest.kt`.
    - Empty `filesDir`; APK содержит bundled `app_category_model.tflite` + `app_category_vocab.txt`; assert `Factory.classify(...)` работает без сетевого вызова, использует APK assets, возвращает корректные категории.

  - [ ]* 17.3 Write SHA256 mismatch retry test
    - **Validates: Requirements 4.4**
    - Файл: `app/src/androidTest/java/com/antispam/blocker/categorization/RemoteUpdateShaFailTest.kt`.
    - MockWebServer возвращает body с неправильным sha256 для одного из 3 новых файлов; запуск `RemoteUpdateWorker.doWork()`; assert: `Result.retry()`, `.tmp` файл удалён, существующий локальный файл не заменён.

  - [ ]* 17.4 Write size mismatch retry test
    - **Validates: Requirements 4.4**
    - Файл: `app/src/androidTest/java/com/antispam/blocker/categorization/RemoteUpdateSizeFailTest.kt`.
    - MockWebServer возвращает body с size != manifest.size для одного из 3 новых файлов; assert `Result.retry()`, `.tmp` удалён.

- [ ] 18. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional property/structural/integration tests and can be skipped for faster MVP.
- Each property test sub-task explicitly cites a Property from `design.md` and the validating Requirements clause from `requirements.md`.
- The 16 properties from the design map to optional test sub-tasks: 2.2 (P9), 2.4 (P10), 2.6 (P13), 4.2 (P11), 7.4 (P14), 7.7 (P15), 7.11 (P12), 9.2 (P5), 9.4 (P6), 11.2 (P3), 11.4 (P4), 11.5 (P8), 12.3 (P7), 16.1 (P1), 16.2 (P2), 16.3 (P16). Plus integration test 16.4 for Req 7.7 (which is example-based, not a universal property).
- Structural / smoke tests cover non-PBT contract invariants from Requirements 3, 4, 5, 6: 3.5, 3.6, 7.2a, 7.5a, 7.8a, 7.9a, 7.10a (Python pipeline structure); 11.6, 11.7, 13.3, 13.4, 13.5, 16.5, 16.6, 16.7, 16.8 (Kotlin runtime + distribution + privacy + backward-compat); 15.2 (UI). Distribution E2E tests in section 17 cover Req 4.4, 4.5, 4.7.
- Performance-инвариант Req 3.10 (cache-hit ≤ 100 µs p99, cache-miss ≤ 5 ms p99) реализуется отдельным microbenchmark в `androidTest` через `androidx.benchmark` — не PBT и не блокер для PR (purposefully out of scope per design's Testing Strategy "Performance benchmarks").
- Качество метрик на real corpus (`top1 ≥ 0.90`, `macro_f1 ≥ 0.85`, `BANK/GOV/EMAIL precision ≥ 0.95`) — гейт самого Training_Pipeline (Req 2.5 / Property 14), а не unit-теста. Скрипт возвращает exit code 1 при failure.
- Checkpoints (tasks 6, 8, 18) are placed at major phase transitions: end of dataset pipeline, end of training pipeline, and end of all integration.
- `AppCategory` enum, `toNotificationBucket()`, и `AppCategoryClassifier` interface MUST NOT be edited — только `AppCategoryClassifierFactory` внутри того же файла расширяется (Req 6.1, 6.2, 6.4).
- Initial bundled assets (task 14) require running the full training pipeline once on the seed corpus before the first APK release.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "2.3", "2.5", "3.1", "3.3", "4.1", "5.1", "5.2", "5.3", "7.1", "7.2", "7.3", "7.5", "7.6", "7.8", "7.9", "9.1", "9.3", "10.1", "12.1", "13.2", "16.2", "16.5", "16.6", "16.7", "16.8"] },
    { "id": 2, "tasks": ["2.2", "2.4", "3.2", "5.4", "7.2a", "7.5a", "7.8a", "7.9a", "9.2", "11.1", "13.3", "13.4"] },
    { "id": 3, "tasks": ["3.4", "3.5", "3.6", "4.2", "7.10", "7.10a", "11.3", "13.1"] },
    { "id": 4, "tasks": ["2.6", "7.4", "7.7", "7.11", "9.4", "11.2", "11.4", "11.5", "11.6", "11.7", "12.2", "13.5", "14.1"] },
    { "id": 5, "tasks": ["12.3", "15.1", "16.1", "16.3", "16.4", "17.2"] },
    { "id": 6, "tasks": ["15.2", "17.1", "17.3", "17.4"] }
  ]
}
```
