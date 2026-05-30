# Requirements Document

## Introduction

Эта спецификация описывает третью «ИИ» в архитектуре приложения — **App Category Model**, on-device TFLite-классификатор приложений по `packageName` (с опциональным локализованным `label`) на 18 семантических категорий. Модель — сиблинг двух уже существующих:

- **Server Model** — облачная TFLite-модель спам-номеров (`app/src/main/assets/spam_model.tflite`), описана в спецификации [`model-training-pipeline`](../model-training-pipeline/requirements.md).
- **Personal Model** — on-device логистическая регрессия по 17 фичам, описана в спецификации [`on-device-personal-spam-classifier`](../on-device-personal-spam-classifier/requirements.md).

Контекст: текущий рантайм — `RuleBasedAppCategoryClassifier` (~150 известных пакетов + substring-эвристики), production-ready, но ограничен. Для long-tail приложений он fallback-ит в `OTHER`, что ослабляет фичи Personal Model `recent_bank_app_30m`, `recent_gov_app_30m`, `recent_marketplace_app_30m`, `recent_messenger_app_30m`, `notif_bank_recent_10m`, `notif_marketplace_recent_10m`. Эта спецификация добавляет обучаемую модель поверх — но НЕ замещает rule-based: rule-based остаётся обязательным fallback-путём.

Ключевые свойства фичи:

- Обучение полностью offline (Python + TensorFlow) на корпусе ≥ 200k (packageName, label, category) пар, собранных из Google Play Store, RuStore, Huawei AppGallery и bootstrap-выборки из `RuleBasedAppCategoryClassifier`.
- Архитектура — char-CNN энкодер (n-gram 3..5) + Conv1D × 3 + GlobalMaxPool + Dense(18, softmax). Размер после dynamic-range-квантизации ≤ 1 MB.
- Дистрибуция через тот же канал, что Server Model: `releases/latest/manifest.json` + `RemoteUpdateWorker` с проверкой SHA256, в существующем 6-часовом цикле обновления (см. `on-device-personal-spam-classifier`, общая инфраструктура ассетов).
- Confidence-gated fallback: если уверенность TFLite ниже порога — возвращается результат `RuleBasedAppCategoryClassifier`. Это исключает «загрязнение» сенсорных фич Personal Model (BANK/GOVERNMENT/EMAIL) низкокачественными предсказаниями.
- Жёсткая приватность: классификатор читает только `packageName` и (опционально) `applicationInfo.loadLabel`. Никогда не читает `Notification.extras`, никогда не логирует labels off-device, никогда не делает сетевых вызовов вне существующего manifest pull.
- Property-based корректность: enum order Kotlin ↔ Python CATEGORIES, dimension TFLite-выхода = 18, и инвариант «низкая уверенность ⇒ rule-based ответ» — все три проверяются автоматически.

## Non-Goals

В рамки v1 фичи **явно не входят** следующие возможности:

- Fine-tuning на user-specific app usage — это утечёт информацию о том, какие приложения у пользователя установлены.
- Federated learning, gradient sharing, любая отправка фич, labels или весов на сервер.
- On-device training App Category Model (in contrast to Personal Model, который учится online; App Category Model — read-only TFLite).
- Image/icon-based фичи (sub-1MB target несовместим с обработкой иконок).
- Замена `RuleBasedAppCategoryClassifier` — он остаётся как fallback и source of truth для bootstrap-данных.
- Изменение `AppCategory.toNotificationBucket()` маппинга на узкий 5-категорийный enum, который пишется в Room-таблицу `notification_event.categoryBucket`. Этот контракт фиксируется этой спецификацией: миграция БД не требуется.
- Чтение `Notification.extras`, `Notification.tickerText`, заголовка или текста уведомления любым кодом, ассоциированным с App Category Model.
- Любые сетевые вызовы из App Category Model или TFLite-инференс-пути, кроме существующего `RemoteUpdateWorker` manifest pull.
- Сторонние аналитические SDK, Firebase Analytics, Crashlytics или любые другие каналы, через которые могла бы утечь телеметрия о категориях, предсказанных для приложений пользователя.
- Кроссдевайсная синхронизация LRU-кэша или предсказаний.

## Glossary

- **App_Category_Model**: TFLite-модель char-CNN, классифицирующая Android-приложение по `packageName` (+ опциональный `label`) в одну из 18 категорий `AppCategory`. Размер ≤ 1 MB после dynamic-range-квантизации, лежит в `app/src/main/assets/app_category_model.tflite`, обновляется через `RemoteUpdateWorker`.
- **AppCategory**: Kotlin-enum из 20 значений: BANK, INVESTMENTS, GOVERNMENT, MARKETPLACE, DELIVERY, TRANSPORT, TRAVEL, HEALTH, MESSENGER, SOCIAL, EMAIL, NEWS, MEDIA, GAMES, DATING, EDUCATION, BROWSER, VPN, PRODUCTIVITY, OTHER. Порядок значений MUST совпадать с порядком списка `CATEGORIES` в `scripts/train_app_category_classifier.py` и порядком softmax-выхода TFLite (где `OTHER` зарезервирован для rule-based fallback и не предсказывается напрямую TFLite-моделью; softmax-выход покрывает 18 первых значений).
- **AppCategoryClassifier**: интерфейс с одним методом `classify(packageName: String, label: String? = null): AppCategory`, существует в `app/src/main/java/com/antispam/blocker/domain/categorization/AppCategoryClassifier.kt`. Этот контракт сохраняется обеими реализациями.
- **RuleBasedAppCategoryClassifier**: production-ready реализация на правилах + словаре (~150 пакетов + substring-маркеры). Остаётся как fallback и не модифицируется этой спецификацией.
- **TFLiteAppCategoryClassifier**: новая реализация `AppCategoryClassifier`, добавляемая этой спецификацией. Загружает `app_category_model.tflite` и `app_category_vocab.txt`, выполняет инференс char-CNN, делегирует в `RuleBasedAppCategoryClassifier` при низкой уверенности или ошибках загрузки.
- **AppCategoryClassifierFactory**: singleton-фабрика в том же файле, что и интерфейс. Выбирает между `TFLiteAppCategoryClassifier` и `RuleBasedAppCategoryClassifier` на основании доступности модели и значения kill-switch в `SettingsStore`.
- **Confidence_Threshold**: численный порог softmax-вероятности (значение по умолчанию 0.6), ниже которого предсказание `App_Category_Model` отбрасывается, и вызов делегируется в `RuleBasedAppCategoryClassifier`.
- **Category_Cache**: LRU-кэш фиксированной ёмкостью 500 записей `packageName → AppCategory`, поддерживаемый `TFLiteAppCategoryClassifier` для удержания горячего пути (`PersonalNotificationListenerService.onNotificationPosted`) на микросекундах.
- **Bootstrap_Seed**: подмножество обучающего корпуса, полученное прогоном `RuleBasedAppCategoryClassifier` с `confidence ≥ HIGH_BOOTSTRAP_CONFIDENCE` (точное совпадение пакета в `KNOWN_PACKAGES` или совпадение по `LABEL_MARKERS`). Гарантирует, что новая модель не теряет ни одного пакета, который текущая rule-based реализация уже классифицирует уверенно.
- **Labelled_Corpus**: финальный CSV-файл `datasets/categories/labeled.csv` со схемой `packageName,label,category`, минимум 200k уникальных пакетов, ≥ 5k на категорию (исключая `OTHER`), train/val/test split 80/10/10 stratified by category.
- **Model_Card**: JSON-файл `app_category_card.json`, содержащий per-category precision / recall / F1, общее число строк обучения, и явный список `categories_order` (20 элементов, включая `OTHER`) для верификации совпадения с Kotlin-enum.
- **Tokenizer_Vocab**: текстовый файл `app_category_vocab.txt`, описывающий словарь char-n-gram, который использовался при обучении. Кодировка UTF-8 без BOM, LF line endings, по одному char-n-gram токену на строку, в порядке убывания id, без пустых строк, с trailing newline. Загружается в рантайме `TFLiteAppCategoryClassifier` для воспроизведения той же токенизации, что в Python.
- **Notification_Bucket**: узкий 5-категорийный enum (BANK / MARKETPLACE / MESSENGER / EMAIL / OTHER), хранимый в `notification_event.categoryBucket` Room-таблицы. Маппится из `AppCategory` через метод `AppCategory.toNotificationBucket()`. Этой спецификацией не модифицируется.
- **Sensitive_Categories**: подмножество `AppCategory` {BANK, GOVERNMENT, EMAIL}, которое гейтит сенсорные фичи Personal Model и поэтому требует precision ≥ 95% (более жёсткий порог, чем общая макроточность модели).
- **Dataset_Builder**: коллекция Python-скриптов в `scripts/`, реализующих сбор корпуса из источников и его трансформацию в `Labelled_Corpus` + train/val/test split.
- **Training_Pipeline**: единый воспроизводимый Python-скрипт `scripts/train_app_category_classifier.py`, потребляющий `train.csv`/`val.csv`/`test.csv` и производящий три артефакта: TFLite-модель, Tokenizer_Vocab, Model_Card.
- **Build_System**: Gradle-сборка приложения, включая unit-test и lint фазы CI.
- **Property_Test_Suite**: набор автоматических тестов на устройстве и в JVM, реализующих property-based и integration проверки инвариантов `App_Category_Model`.

## Requirements

### Requirement 1: Dataset construction

**User Story:** Как data scientist, я хочу детерминированно собирать сбалансированный корпус (packageName, label, category) из публичных источников и production-кода, чтобы каждая из 18 категорий была представлена достаточным числом примеров для обучения, и чтобы любой коллега мог воспроизвести тот же датасет с тем же seed.

#### Acceptance Criteria

1. THE Dataset_Builder SHALL собирать (packageName, label, category) тройки из Google Play Store, RuStore и Huawei AppGallery через скрипты-краулеры, расположенные в каталоге `scripts/`.
2. THE Dataset_Builder SHALL прогнать `RuleBasedAppCategoryClassifier` по списку известных пакетов; для каждого пакета, для которого классификатор активно нашёл совпадение с высокой уверенностью (точное совпадение в `KNOWN_PACKAGES` или совпадение по `LABEL_MARKERS`), SHALL добавить тройку (packageName, label, category) в Bootstrap_Seed; пакеты, для которых классификатор не нашёл уверенных совпадений, SHALL NOT попадать в Bootstrap_Seed.
3. THE Dataset_Builder SHALL дедуплицировать строки по `packageName` (case-sensitive), сохраняя для каждого пакета первое встретившееся значение `label` и категорию из источника с наивысшим приоритетом (Bootstrap_Seed > Google Play Store > RuStore > Huawei AppGallery).
4. THE Dataset_Builder SHALL нормализовать `label`: приводить к Unicode NFC, обрезать ведущие/завершающие пробелы; IF получившаяся строка пустая ИЛИ длина в Unicode-символах превышает 200, THEN записывать в колонку `label` пустую строку.
5. THE Dataset_Builder SHALL производить итоговый файл `datasets/categories/labeled.csv` с заголовком первой строкой `packageName,label,category`, последующими строками данных в том же порядке колонок, кодировкой UTF-8 без BOM, LF line endings и trailing newline.
6. THE Labelled_Corpus SHALL содержать не менее 200 000 уникальных пакетов и не менее 5 000 пакетов в каждой из 18 категорий `AppCategory`, исключая `OTHER`.
7. THE Dataset_Builder SHALL производить три файла `train.csv`, `val.csv`, `test.csv` в каталоге `datasets/categories/` с тем же форматом, что у `labeled.csv` (см. п. 5), разбивая Labelled_Corpus в пропорции 80/10/10 stratified by category, при этом ни один `packageName` SHALL NOT появляться более чем в одном split.
8. THE Dataset_Builder SHALL принимать аргумент `--seed` со значением по умолчанию 42; для одинаковых значений `--seed` и одинакового входного корпуса THE Dataset_Builder SHALL производить байт-идентичные файлы `train.csv`, `val.csv`, `test.csv` (включая порядок строк).
9. IF в строке источника отсутствует поле `packageName` ИЛИ оно пустое после strip, THEN THE Dataset_Builder SHALL атомарно увеличить счётчик `dropped_rows` в финальном отчёте независимо от того, был ли последующий пропуск строки успешным; затем SHALL пропустить строку.
10. IF значение `category` в источнике после `.toUpperCase().strip()` не совпадает ни с одним из 20 значений `AppCategory`, THEN THE Dataset_Builder SHALL отображать его в категорию `OTHER` И вести отдельный счётчик `unknown_category_rows` в финальном отчёте.
11. THE Dataset_Builder SHALL записывать финальный отчёт в `datasets/categories/build_report.json` со следующими полями: `total_input_rows`, `dropped_rows`, `unknown_category_rows`, `corpus_rows`, `per_category_counts` (объект из 20 ключей), `seed`, `built_at` (ISO-8601 UTC).

### Requirement 2: Model training and packaging

**User Story:** Как data scientist, я хочу один воспроизводимый скрипт `scripts/train_app_category_classifier.py`, который из готового CSV-сплита тренирует char-CNN, сохраняет квантизованный TFLite ≤ 1 MB и Model_Card с метриками — чтобы релиз-инженер мог собрать идентичный артефакт без моего участия.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL принимать аргументы `--train`, `--val`, `--test` (пути к CSV-сплитам, по умолчанию `datasets/categories/train.csv`, `datasets/categories/val.csv`, `datasets/categories/test.csv`) и `--seed` (целое, по умолчанию 42); при одинаковом seed и одинаковом содержимом CSV-сплитов SHALL производить байт-идентичные `app_category_model.tflite`, `app_category_vocab.txt` и `app_category_card.json`.
2. THE Training_Pipeline SHALL реализовывать архитектуру: char-n-gram энкодер (n ∈ {3,4,5}) над `packageName` и опциональным `label`, три параллельные ветви Conv1D с `filters=128` и `kernel_size ∈ {3,5,7}`, GlobalMaxPool по каждой ветви, конкатенация, Dense(18) с softmax-активацией; Dense-слой покрывает 18 первых значений `AppCategory` (без `OTHER`, который зарезервирован для rule-based fallback и не предсказывается TFLite-моделью).
3. THE Training_Pipeline SHALL использовать оптимизатор AdamW с начальной скоростью обучения `1e-3`, cosine-decay-расписанием, batch size 256, 30 эпох.
4. WHEN тренировка завершается, THE Training_Pipeline SHALL вычислить на test-сплите top-1 accuracy, macro-F1, а также precision / recall / F1 по каждой из 18 категорий.
5. WHEN top-1 accuracy ≥ 0.90 И macro-F1 ≥ 0.85 И precision по каждой из BANK, GOVERNMENT, EMAIL ≥ 0.95 на test-сплите вычислены и проверены численно (не просто факт завершения вычислений), THE Training_Pipeline SHALL записать три артефакта: TFLite-файл (см. п. 6), Tokenizer_Vocab (см. п. 8) и Model_Card (см. п. 9), и SHALL завершиться с exit code 0.
6. THE Training_Pipeline SHALL экспортировать обученную модель в TFLite c dynamic-range-квантизацией (int8 weights), результат записывать в путь, указанный аргументом `--output` (по умолчанию `app/src/main/assets/app_category_model.tflite`), атомарно через временный файл `.tmp` с последующим rename.
7. THE App_Category_Model TFLite-файл SHALL занимать на диске не более 1 048 576 байт (1 MB).
8. THE Training_Pipeline SHALL экспортировать Tokenizer_Vocab в файл, путь которого задаётся аргументом `--vocab` (по умолчанию `app/src/main/assets/app_category_vocab.txt`), кодировкой UTF-8 без BOM, LF line endings, по одному char-n-gram токену на строку в порядке убывания id, без пустых строк, с trailing newline.
9. THE Training_Pipeline SHALL записывать Model_Card в путь `--card` (по умолчанию `app/src/main/assets/app_category_card.json`) со следующими полями: `categories_order` (список из 20 строк, идентичный имени значений `AppCategory.values()` в Kotlin), `total_train_rows` (целое неотрицательное), `metrics.top1_accuracy` (число в [0, 1]), `metrics.macro_f1` (число в [0, 1]), `metrics.per_category` (объект из 18 ключей — имена категорий, исключая `OTHER`, со значениями `{precision: число в [0, 1], recall: число в [0, 1], f1: число в [0, 1]}`).
10. THE Training_Pipeline SHALL сравнивать `categories_order` из Model_Card с порядком `AppCategory` enum в Kotlin (через статически закоммиченный список `KOTLIN_APP_CATEGORY_ORDER` в скрипте); IF длина списков разная ИЛИ хотя бы в одном индексе значения case-sensitive не равны, THEN THE Training_Pipeline SHALL завершиться с exit code 3, вывести в stderr сообщение, описывающее первую позицию расхождения и оба значения, и не записать ни TFLite, ни vocab, ни card.
11. IF размер записанного TFLite превышает 1 048 576 байт, THEN THE Training_Pipeline SHALL удалить TFLite-файл, vocab-файл и card-файл, завершиться с exit code 2 и вывести в stderr сообщение о превышении.
12. IF любая из метрик из п. 5 ниже своего порога, THEN THE Training_Pipeline SHALL завершиться с exit code 1 (явное отклонение релиза с ненулевым кодом), не записать ни одного из трёх артефактов, не модифицировать ранее существующие файлы по тем же путям, и вывести в stderr перечень метрик, не достигших порога с указанием фактического и ожидаемого значения для каждой.

### Requirement 3: Android integration

**User Story:** Как пользователь Personal Model, я хочу, чтобы фичи `recent_bank_app_30m`, `recent_gov_app_30m`, `recent_marketplace_app_30m`, `recent_messenger_app_30m`, `notif_bank_recent_10m`, `notif_marketplace_recent_10m` корректно срабатывали на длинном хвосте моих установленных приложений (а не только на ~150 известных пакетах), без замедления горячего пути уведомлений и без новых рантайм-разрешений.

#### Acceptance Criteria

1. THE TFLiteAppCategoryClassifier SHALL реализовывать существующий интерфейс `AppCategoryClassifier` без изменения его сигнатуры `fun classify(packageName: String, label: String? = null): AppCategory`.
2. WHEN TFLiteAppCategoryClassifier создаётся, THE TFLiteAppCategoryClassifier SHALL загружать TFLite-модель и Tokenizer_Vocab по следующему правилу приоритета: если файлы `app_category_model.tflite` и `app_category_vocab.txt` присутствуют в `filesDir`, использовать их; иначе использовать одноимённые файлы из `app/src/main/assets/`.
3. WHEN TFLiteAppCategoryClassifier выполняет инференс, THE TFLiteAppCategoryClassifier SHALL читать только переданные параметры `packageName: String` и `label: String?`, и SHALL NOT обращаться к каким-либо иным полям, объектам или системным API для получения дополнительного входа.
4. THE TFLiteAppCategoryClassifier SHALL поддерживать LRU-кэш Category_Cache с ключом `packageName` и значением `AppCategory`, фиксированной ёмкостью 500 записей; при cache-hit SHALL возвращать сохранённое значение без вызова TFLite-инференса и без делегирования в `RuleBasedAppCategoryClassifier`; при cache-miss после получения итогового результата вызова `classify` (включая результат, возвращённый из rule-based fallback по п. 5 и п. 11) SHALL записать пару (`packageName`, итоговая `AppCategory`) в кэш; при достижении ёмкости 500 записей SHALL вытеснять least-recently-used запись.
5. IF softmax-вероятность top-1-категории, вычисленная TFLite-инференсом, строго меньше Confidence_Threshold (значение по умолчанию 0.6), THEN THE TFLiteAppCategoryClassifier SHALL делегировать вызов в `RuleBasedAppCategoryClassifier.classify(packageName, label)` и вернуть его результат как итог `classify`.
6. THE AppCategoryClassifierFactory SHALL выбирать `TFLiteAppCategoryClassifier` тогда и только тогда, когда одновременно выполняются три условия: (a) хотя бы одна из двух пар ассетов доступна — `app_category_model.tflite` + `app_category_vocab.txt` в `filesDir` ИЛИ те же файлы в APK assets; (b) kill-switch `tfliteAppCategoryEnabled` в `SettingsStore` находится в значении `true` (значение по умолчанию `true`); (c) при первом обращении к фабрике загрузка модели и проверка shape output-tensor по п. 8 завершились без выброса исключения.
7. IF любое из трёх условий из п. 6 не выполняется, THEN THE AppCategoryClassifierFactory SHALL возвращать singleton `RuleBasedAppCategoryClassifier` для всех последующих вызовов в течение жизни процесса.
8. WHEN TFLiteAppCategoryClassifier инициализируется, THE TFLiteAppCategoryClassifier SHALL проверить, что output-tensor загруженной TFLite-модели имеет shape `[1, 18]` либо `[18]` после squeeze, и SHALL отвергать любую другую форму выходного тензора.
9. IF проверка shape по п. 8 не проходит, THEN THE TFLiteAppCategoryClassifier SHALL пометить себя как unavailable, AppCategoryClassifierFactory SHALL переключиться на singleton `RuleBasedAppCategoryClassifier`, и факт ошибки SHALL быть залогирован через `android.util.Log` без включения значений `packageName` или `label` в лог-сообщение.
10. THE TFLiteAppCategoryClassifier SHALL возвращать результат метода `classify` при cache-hit за время не более 100 микросекунд по p99 и при cache-miss за время не более 5 миллисекунд по p99, измеренное на устройстве референс-уровня (Snapdragon 6xx серии и выше) в одиночном потоке вызова с batch=1, при выборке не менее 1000 последовательных вызовов.
11. IF TFLite-инференс выбрасывает любое исключение во время вызова `classify`, THEN THE TFLiteAppCategoryClassifier SHALL поймать исключение, делегировать вызов в `RuleBasedAppCategoryClassifier.classify(packageName, label)`, вернуть его результат как итог `classify`, и SHALL залогировать факт исключения через `android.util.Log` без включения значений `packageName` или `label` в лог-сообщение.
12. THE TFLiteAppCategoryClassifier SHALL NOT добавлять новых рантайм-разрешений Android, SHALL NOT инициировать сетевые запросы, SHALL NOT обращаться к файловой системе вне `app/src/main/assets/` и `filesDir`, и SHALL NOT обращаться к `PackageManager` за пределами тех вызовов, которые уже выполняет `RuleBasedAppCategoryClassifier`.
13. WHILE Settings → Privacy → «Прозрачность данных» screen видим пользователю, THE screen SHALL отображать рядом с каждым live-сэмплом foreground-приложения из `RecentUserContextProvider.recentForegroundEvents` категорию `AppCategory`, возвращённую `AppCategoryClassifierFactory.classify(packageName, label)` для соответствующего сэмпла.

### Requirement 4: Distribution and updates

**User Story:** Как релиз-инженер, я хочу публиковать новые версии App_Category_Model тем же способом, что и Server Model — через `releases/latest/manifest.json` с SHA256 — и быть уверен, что при сетевой ошибке или повреждении файла приложение откатится на rule-based реализацию, а не упадёт.

#### Acceptance Criteria

1. THE manifest.json SHALL объявлять записи `app_category_model.tflite`, `app_category_vocab.txt` и `app_category_card.json` со следующими полями каждая: `sha256` (lowercase hex длиной ровно 64 символа), `size` (целое неотрицательное в байтах), `url` (относительный путь от base URL без префикса `/`).
2. THE RemoteUpdateWorker SHALL добавлять `app_category_model.tflite`, `app_category_vocab.txt` и `app_category_card.json` в существующий список разрешённых файлов (`ALLOWED_FILES`).
3. WHEN RemoteUpdateWorker обнаруживает новую версию manifest, THE RemoteUpdateWorker SHALL для каждого из трёх ассетов выполнить последовательность: скачать в `filesDir/<filename>.tmp`, посчитать SHA256, сравнить с заявленным в manifest, проверить размер байт против `size` из manifest, при совпадении атомарно переименовать `.tmp` → финальное имя; при любой ошибке IO/сети, mismatch SHA256 или mismatch размера — удалить `.tmp`-файл.
4. IF SHA256 ИЛИ size любого из трёх скачанных файлов (`app_category_model.tflite`, `app_category_vocab.txt`, `app_category_card.json`) не совпадают с заявленными в manifest, THEN THE RemoteUpdateWorker SHALL не заменять существующий локальный файл, удалить соответствующий `.tmp`-файл, и SHALL возвращать `Result.retry()`.
5. WHEN App_Category_Model успешно обновился через RemoteUpdateWorker (все три файла прошли валидацию и переименованы), THE TFLiteAppCategoryClassifier SHALL переинициализироваться при следующем создании экземпляра (lazy reinitialization при первом вызове `AppCategoryClassifierFactory.classify` после обновления), используя обновлённые файлы из `filesDir` приоритетно над bundled APK-ассетом, без перезапуска процесса.
6. THE RemoteUpdateWorker SHALL опрашивать manifest в окне 6 часов ± 30 минут от предыдущего опроса для App_Category_Model и существующих ассетов в одном цикле, и SHALL NOT добавлять отдельного канала обновлений или отдельного периодического worker'а для App_Category_Model.
7. THE App_Category_Model SHALL поставляться внутри APK как initial fallback в `app/src/main/assets/app_category_model.tflite`, чтобы первый запуск приложения после установки уже мог использовать TFLite-путь без сетевого вызова.

### Requirement 5: Privacy

**User Story:** Как пользователь, я хочу быть уверен, что новый ML-классификатор приложений не получает никаких новых полей, не пишет никаких labels в логи и не делает никаких сетевых вызовов помимо уже существующего обновления ассетов.

#### Acceptance Criteria

1. THE TFLiteAppCategoryClassifier SHALL читать только параметры `packageName: String` и `label: String?`, переданные в метод `classify`, и SHALL NOT обращаться к Android-API для получения дополнительных полей о приложении (`PackageManager.getInstalledApplications`, `getInstalledPackages`, `queryIntentActivities` и т.п.).
2. THE TFLiteAppCategoryClassifier SHALL NOT читать `Notification.extras`, `Notification.tickerText`, `Notification.contentText`, `Notification.contentTitle`, `Notification.bigContentView`, `Notification.publicVersion` или любое другое поле объекта `Notification`.
3. THE TFLiteAppCategoryClassifier SHALL NOT обращаться к `LocationManager`, `MediaRecorder`, `AudioRecord`, `BiometricPrompt`, `ClipboardManager`, или к URI `content://sms`, `content://mms`, `content://carriers`.
4. THE TFLiteAppCategoryClassifier SHALL NOT логировать значения `packageName`, `label` или предсказанной `AppCategory` через `android.util.Log`, не записывать их в `SharedPreferences`, Room/AppDatabase, `filesDir`, `cacheDir`, `externalFilesDir`, MediaStore, dropbox или logcat-каналы.
5. THE TFLiteAppCategoryClassifier SHALL инициировать только те исходящие сетевые соединения (HTTP, HTTPS, WebSocket, raw socket), которые уже выполняет `RemoteUpdateWorker` для пула manifest и ассетов; ни `packageName`, ни `label`, ни softmax-выход, ни предсказанная `AppCategory` SHALL быть включены в полезную нагрузку этих или иных запросов.
6. THE App_Category_Model SHALL обучаться исключительно на offline-корпусе из `datasets/categories/`; THE App_Category_Model SHALL NOT использовать данные, накопленные на устройстве пользователя (включая Category_Cache, `notification_event` Room-таблицу, `app_usage_event` или любую другую on-device телеметрию) для дообучения.
7. THE App_Category_Model SHALL NOT использовать federated learning, gradient sharing, model averaging или любую другую форму отправки сигнала о пользовательских приложениях на сервер.
8. THE app release-вариант build SHALL NOT включать в Gradle-зависимости пакеты com.google.firebase:firebase-analytics, com.google.firebase:firebase-crashlytics, com.appmetrica:*, io.sentry:*, com.bugsnag:*, com.amplitude:*, com.mixpanel:* или любые иные SDK, которые могли бы получить доступ к `packageName`, `label` или предсказанной `AppCategory`.
9. THE Category_Cache SHALL храниться только в process memory, не персиститься через `SharedPreferences`, Room/AppDatabase, `filesDir`, `cacheDir`, IPC (`ContentProvider`, AIDL); THE Category_Cache SHALL быть сброшен при завершении процесса приложения.

### Requirement 6: Backwards compatibility

**User Story:** Как разработчик Personal Model, я хочу быть уверен, что переключение `AppCategoryClassifierFactory` на TFLite-реализацию не ломает существующие фичи и не требует Room-миграции.

#### Acceptance Criteria

1. THE `AppCategory` enum в `app/src/main/java/com/antispam/blocker/domain/categorization/AppCategoryClassifier.kt` SHALL содержать ровно 20 значений в указанном порядке: BANK (id 0), INVESTMENTS (1), GOVERNMENT (2), MARKETPLACE (3), DELIVERY (4), TRANSPORT (5), TRAVEL (6), HEALTH (7), MESSENGER (8), SOCIAL (9), EMAIL (10), NEWS (11), MEDIA (12), GAMES (13), DATING (14), EDUCATION (15), BROWSER (16), VPN (17), PRODUCTIVITY (18), OTHER (19); порядок имён, возвращаемый `AppCategory.values().map { it.name }`, SHALL быть байт-эквивалентен этому списку.
2. THE method `AppCategory.toNotificationBucket(): String` SHALL возвращать одно из ровно 5 значений {"BANK", "MARKETPLACE", "MESSENGER", "EMAIL", "OTHER"} согласно следующему маппингу: BANK → "BANK", INVESTMENTS → "BANK", MARKETPLACE → "MARKETPLACE", DELIVERY → "MARKETPLACE", MESSENGER → "MESSENGER", EMAIL → "EMAIL", и каждое из остальных 14 значений (GOVERNMENT, TRANSPORT, TRAVEL, HEALTH, SOCIAL, NEWS, MEDIA, GAMES, DATING, EDUCATION, BROWSER, VPN, PRODUCTIVITY, OTHER) → "OTHER".
3. THE Room schema `notification_event` SHALL остаться без миграции в результате внедрения этой спецификации в случае, когда TFLite-реализация не требует изменения схемы; IF TFLite-реализация требует хранения дополнительных метаданных или изменения способа персистенции notification buckets, THEN миграция Room SHALL допускаться, при этом `AppDatabase` version SHALL корректно повышаться через стандартный `Migration` объект, и существующие данные `notification_event` SHALL мигрироваться без потерь.
4. THE interface `AppCategoryClassifier` в `AppCategoryClassifier.kt` SHALL содержать ровно один метод `classify` со следующей сигнатурой: имя `classify`, два параметра — первый `packageName: String`, второй `label: String? = null` со значением по умолчанию `null`, тип возврата `AppCategory`; новых методов в интерфейс SHALL не добавляться.
5. THE Personal Model features `recent_bank_app_30m`, `recent_gov_app_30m`, `recent_marketplace_app_30m`, `recent_messenger_app_30m`, `notif_bank_recent_10m`, `notif_marketplace_recent_10m` в `DeviceFeatureExtractor` SHALL: (a) сохранить исходный код методов их вычисления без модификации; (b) для одинаковых input-событий и одинакового `now` возвращать бит-равные значения фич; (c) проходить существующие unit-тесты без модификации тестов.
6. THE App_Category_Model SHALL предпочтительно переиспользовать `RemoteUpdateWorker` и `manifest.json` инфраструктуру Server Model и Personal Model: SHALL NOT добавляться отдельный класс `WorkManager`-worker для дистрибуции App_Category_Model в штатном режиме, SHALL NOT добавляться второй файл manifest, SHALL NOT использоваться периодичность опроса отличная от существующих 6 часов ± 30 минут (см. Requirement 4 п. 6); IF существующая shared-инфраструктура временно недоступна (RemoteUpdateWorker возвращает persistent failure более 24 часов) ИЛИ требует изменений, несовместимых с App_Category_Model, THEN допускается fallback на отдельную инфраструктуру (отдельный worker или отдельный manifest-источник) с явной отметкой в release notes и kill-switch для отключения.
7. THE Build_System SHALL завершать сборку с ошибкой, IF любое из условий п. 1, п. 2, п. 4 нарушено в исходном коде — через unit-тест в `unitTest`-фазе, проверяющий enum order, `toNotificationBucket()` маппинг и сигнатуру интерфейса.

### Requirement 7: Property-based correctness invariants

**User Story:** Как разработчик, я хочу автоматическими проверками на каждой сборке гарантировать, что Kotlin-enum и Python-список категорий не разъехались, что output TFLite имеет ровно 18 классов, и что confidence-gated fallback действительно равен rule-based ответу при низкой уверенности.

#### Acceptance Criteria

1. WHEN Build_System выполняет фазу `unitTest` или CI-пайплайн (Gradle `:app:testDebugUnitTest`), THE Build_System SHALL запускать smoke-тест, который читает `categories_order` из `app/src/main/assets/app_category_card.json` и сравнивает его с `AppCategory.values().map { it.name }`, включая `OTHER` в обоих списках, в одинаковом порядке и одинаковой длине.
2. IF smoke-тест из п. 1 находит расхождение в длине списков ИЛИ в порядке элементов, THEN THE Build_System SHALL завершить сборку с ненулевым exit code, вывести в stderr сообщение об ошибке, указывающее первую позицию расхождения и оба значения, и не публиковать APK-артефакт.
3. WHEN TFLiteAppCategoryClassifier инициализируется, THE TFLiteAppCategoryClassifier SHALL проверить, что dimension softmax-выхода TFLite-модели равно 18; IF проверка не проходит, THEN THE TFLiteAppCategoryClassifier SHALL выбросить исключение инициализации с фактическим и ожидаемым размером, и SHALL не оставаться в ready-to-infer состоянии (см. также Requirement 3 п. 8).
4. THE Property_Test_Suite SHALL содержать property-тест с не менее 200 итерациями, который для случайно сгенерированных строк `packageName` (длина 0..100, символы из Unicode-диапазонов U+0020..U+007E ASCII и U+0400..U+04FF Cyrillic) и опциональных `label` (длина 0..200, те же диапазоны) проверяет инвариант: если `TFLiteAppCategoryClassifier.softmaxTop1Confidence(packageName, label) < Confidence_Threshold`, то `TFLiteAppCategoryClassifier.classify(packageName, label) == RuleBasedAppCategoryClassifier.classify(packageName, label)`; при нарушении SHALL выводить пару (`packageName`, `label`), значение `softmaxTop1Confidence` и обе предсказанные категории.
5. THE Property_Test_Suite SHALL содержать round-trip-тест на парсер Tokenizer_Vocab: для каждого char-n-gram токена `t`, прочитанного из `app_category_vocab.txt`, повторная сериализация через тот же writer SHALL давать байт-идентичный поток без исключений; при нарушении SHALL выводить индекс токена и расходящийся байтовый вывод.
6. THE Property_Test_Suite SHALL содержать тест на идемпотентность Category_Cache: для произвольно выбранного `packageName`, ровно 5 последовательных вызовов `classify(packageName)` SHALL возвращать байт-идентичные значения `AppCategory` (явная проверка попарного равенства результатов), при этом mock TFLite-runtime SHALL зафиксировать ровно один вызов инференса (явный assert: счётчик == 1); тест SHALL проверять оба условия независимо, не полагаясь на общую оценку фреймворка; при нарушении SHALL выводить итоговые значения и счётчик инференсов.
7. THE Property_Test_Suite SHALL содержать integration-тест с ровно 9 пар (3 для BANK, 3 для GOVERNMENT, 3 для EMAIL) `(packageName, expected_category)`; для каждой пары тест SHALL явно проверять одно из двух логических условий: (a) `TFLiteAppCategoryClassifier.classify(packageName, null) == expected_category`, либо (b) `softmaxTop1Confidence(packageName, null) < Confidence_Threshold` И `RuleBasedAppCategoryClassifier.classify(packageName, null) == expected_category`; пара считается прошедшей если выполнено (a) ИЛИ (b); при нарушении (ни (a), ни (b) не выполнено) SHALL выводить пару, `softmaxTop1Confidence` и фактически предсказанную категорию.
