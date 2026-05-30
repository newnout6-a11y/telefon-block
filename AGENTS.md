# AGENTS.md — заметки для будущих ассистентов и разработчика

Краткий справочник по правкам, инвариантам и нерешённым задачам, чтобы не
переоткрывать одно и то же. Дополняй при крупных изменениях.

## Build / verification

- Сборка debug APK: `./gradlew.bat :app:assembleDebug --no-daemon`
- Только компиляция Kotlin: `./gradlew.bat :app:compileDebugKotlin --no-daemon`
- Принудительная перекомпиляция после правок: `--rerun-tasks`
- APK после сборки: `app/build/outputs/apk/debug/app-debug.apk` (~116 MB
  из-за prebuilt blocklist sqlite asset 128 MB)
- Python тесты: `py -m pytest tests/ -v` (на Windows используется `py`,
  не `python` / `python3`)

## Тестовая инфраструктура

В `app/src/test/java/` сейчас почти ничего нет — единственный unit-тест
`CategoryCacheTest`. Критичные модули **`FusionDecider`**, **`OnlineTrainer`**,
**`WarmUpGate`**, **`ImplicitLabelDetector`**, **`SmartSpamDetector`** без
покрытия. Любой рефакторинг этих файлов рискованный.

## Известные edge case'ы (исправленные)

### Дыра #1 — implicit BLOCK SGD теряется на нашем собственном блоке (FIXED)

`SpamCallScreeningService.onScreenCall` после `Verdict.BLOCK` дёргает
`OnlineTrainer.applyImplicitLabel(snapshotId, BLOCK)` напрямую, потому что
`CallEventRecorder.mapCallType` пропускает `CallLog.Calls.BLOCKED_TYPE = 6`
строки и без этого SGD-step не происходил. См.
`app/src/main/java/com/antispam/blocker/service/SpamCallScreeningService.kt`
ветку `Verdict.BLOCK`.

### Дыра #2 — кнопки нотификации не учили Device-модель (FIXED)

`SpamWarningNotifier.buildActionIntent` теперь кладёт `EXTRA_CALL_EVENT_ID`
для `ACTION_BLOCK`/`ACTION_ALLOW`, и `SpamActionReceiver.onReceive` после
`feedbackHandler.handleFeedback(...)` параллельно дёргает
`OnlineTrainer.applyExplicitLabel(callEventId, label)` с весом
`EXPLICIT_WEIGHT = 1.5f`. До этого Device-модель училась только на отдельных
`ACTION_SPAM_YES`/`ACTION_SPAM_NO` кнопках, которые юзер тапает редко.

### WarmUpGate `installedAt = 0L` (FIXED)

`SpamBlockerApp.onCreate` теперь сидит `installedAt = now()` при первом
запуске, если `installedAtFlow.first() == 0L`. До этого `(now - 0) >= 14d`
всегда true → warm-up считался завершённым на первом старте, Device-модель
голосовала на дефолтных весах (probBlock ≈ 0.378), и FusionDecider попадал
в `single_block_downgrade_warn` → Cloud BLOCK всегда становился WARN.

### `fallbackToDestructiveMigration` в release (FIXED)

`AppDatabase.getInstance` теперь применяет `fallbackToDestructiveMigration()`
только при `BuildConfig.DEBUG = true`. На release Room упадёт, если новой
миграции нет — это _полезный_ фейл (разработчик должен добавить миграцию),
вместо тихой потери истории звонков, training_data, decision_record и т.д.

### CallEventRecorder читал весь CallLog при первом запуске (FIXED)

`CallEventRecorder.init` теперь при `lastSeen == 0L` (свежая установка)
сидит cursor на `now() - 14d` (синхронно с `WarmUpGate.WARMUP_DAYS_MS`).
До этого подтягивались все строки из системного журнала, что блокировало
startup на телефонах с 5+ годами истории и не давало SGD (snapshot для
исторических нет всё равно).

### `min_app_db_version` enforcement (FIXED)

`AppDatabase.SCHEMA_VERSION` теперь публичная константа. `RemoteUpdateWorker`
сверяет `manifest.minAppDbVersion` с ней: если манифест требует более новую
схему, чем у APK — лог + `Result.success()` без применения. Юзер получит
обновление после апдейта APK. До этого поле парсилось, но игнорировалось,
и несовместимые ассеты применялись, ломая модель.

### CallRecord ↔ FeatureSnapshot association (FIXED — MIGRATION_5_6)

Схема БД bump'нута 5 → 6: добавлена nullable колонка
`call_records.featureSnapshotId`. `SpamCallScreeningService.callLogRepo.record`
теперь передаёт snapshotId напрямую, а `CallLogScreen.resolveSnapshotId`
сначала смотрит `record.featureSnapshotId` и только потом fallback на
`getLatestForNumber`. Это лечит UX-баг long-press по старой записи в журнале
(до фикса показывался snapshot последнего звонка с того же номера, а не
снимок именно ЭТОЙ записи).

## SGD diagnostic logging

`DeviceModel.sgdStep` и `OnlineTrainer.applyLabel` логируют в `SpamBlocker_SGD`
тег. Просмотр на устройстве:

```
adb logcat -s SpamBlocker_SGD:I SpamBlocker:I
```

Что должно появляться:
- `applyLabel id=42 y=1.0 w=1.5 snapshot=42 schema=1`
- `sgd y=1.0 w=1.5 p=0.3870->0.4012 d=+0.0142 bias=-0.4862`
- `labelCount -> 7 (warmup at 30)`

## App Category Model — initial bundle (P1, отложено)

Артефакты `app_category_model.tflite` / `app_category_vocab.txt` /
`app_category_card.json` **не** лежат в `app/src/main/assets/`. Сейчас
`AppCategoryClassifierFactory.createClassifier` корректно возвращает
`RuleBasedAppCategoryClassifier` (selection table в KDoc) — поэтому это
**не data loss / crash**, а только производительность long-tail apps.

Чтобы обучить и подложить:

```bash
# 1. Датасет уже собран:
ls datasets/categories/{train,val,test}.csv  # 182k/22k/22k

# 2. Обучить (требует TF, ~20-60 мин CPU):
py scripts/train_app_category_classifier.py \
    --train datasets/categories/train.csv \
    --val   datasets/categories/val.csv \
    --test  datasets/categories/test.csv \
    --output app/src/main/assets/app_category_model.tflite \
    --vocab  app/src/main/assets/app_category_vocab.txt \
    --card   app/src/main/assets/app_category_card.json

# 3. Проверить enum-order parity (Property 1 в spec):
py -m pytest tests/test_train_app_category_classifier.py -k enum_order
```

После размещения артефактов в `assets/` `AppCategoryClassifierFactory`
автоматически переключится на TFLite-ветку при следующем старте процесса.

## Манифест-обновления и подпись (P0 #3 — graceful verify)

`RemoteUpdateWorker` тянет `manifest.json` + ассеты с GitHub raw, проверяет
SHA256 каждого файла. SHA256 защищает от mid-flight corruption, но **не
от подменённого manifest** (supply-chain). Поддерживается opt-in проверка
ECDSA-подписи:

- При наличии `app/src/main/assets/manifest_pubkey.pem` (DER-encoded
  ECDSA P-256 public key) И `manifest.json.sig` рядом с `manifest.json`
  на сервере, worker верифицирует подпись через
  `Signature.getInstance("SHA256withECDSA")`. При несовпадении —
  `Result.retry()` без применения.
- Если pubkey **или** .sig отсутствуют — лог `verify: skipped (no pubkey
  / no signature)` и worker продолжает по старому пути (обратная
  совместимость с существующими manifest).

### Как сгенерировать пару и подписать

```powershell
# 1. Сгенерировать ECDSA P-256 keypair (одноразово; private key хранить
#    в безопасном месте, в репо НЕ коммитить):
py scripts/sign_manifest.py keygen --out releases/keys/

# Появятся:
#   releases/keys/manifest_priv.pem  — приватный ключ (СЕКРЕТ)
#   releases/keys/manifest_pub.pem   — публичный ключ (DER → base64)

# 2. Положить публичный ключ в APK:
copy releases/keys/manifest_pub.pem app/src/main/assets/manifest_pubkey.pem

# 3. На каждый релиз — подписать manifest.json:
py scripts/sign_manifest.py sign \
    --key releases/keys/manifest_priv.pem \
    --manifest releases/latest/manifest.json
# Создаст releases/latest/manifest.json.sig (binary ECDSA signature)
```

После этого все APK с обновлённым `manifest_pubkey.pem` будут отказываться
применять manifest без валидной подписи.

## CallerID — определение звонящего (SCHEMA_VERSION 7)

Реализовано в ветке `feature/on-device-personal-classifier` (коммит рядом с этим).

### Компоненты
- `data/db/entity/CallerLookup.kt` — Room-entity, таблица `caller_lookup`
- `data/db/dao/CallerLookupDao.kt` — `observe(number): Flow`, `upsert`, `purgeStale`
- `domain/lookup/CallerInfo.kt` — view-model + extension fun `toCallerInfo` / `toEntity`
- `domain/lookup/OfflineCallerLookup.kt` — libphonenumber geocoder+carrier (без сети)
- `domain/lookup/TwoGisCallerLookup.kt` — 2GIS Catalog API opt-in HTTP client
- `domain/lookup/CallerLookupRepository.kt` — оркестратор (cache → offline → 2GIS)
- `data/prefs/CallerLookupSettingsStore.kt` — DataStore: `twoGisEnabledFlow` + `apiKeyFlow`
- `domain/lookup/CallerLookupWorker.kt` — WorkManager, дёргается после каждого CallLog-события

### Зависимости
```
com.googlecode.libphonenumber:geocoder:2.249  // регион ("Москва")
com.googlecode.libphonenumber:carrier:1.239   // оператор ("МТС")
```

### Поток данных
1. `CallEventRecorder.handleRow` → `CallerLookupWorker.enqueue(ctx, normalizedNumber)`
2. Worker → `repo.ensureOffline(number)` → libphonenumber → `caller_lookup` в Room
3. Если 2GIS включён → `repo.fetchOnline(number)` → HTTP к 2GIS → upsert в Room
4. `CallLogScreen` подписывается на `repo.observe(number): Flow<CallerInfo?>` → subtitle

### Настройка 2GIS
- Settings → "Определение звонящего" → включить toggle + вставить API key
- Бесплатный ключ: [dev.2gis.ru](https://dev.2gis.ru) (регистрация, 5k запросов/день)
- Без ключа работает только оффлайн (регион + оператор из libphonenumber)

### TTL кэша
- offline-запись: 7 дней
- 2GIS-запись: 30 дней
- negative (не найдено в 2GIS): 2 дня

### MIGRATION_6_7
Чисто аддитивная: создаёт таблицу `caller_lookup`. Существующие данные не трогает.


## Прочие нерешённые крупные пункты (P1)

- **Manifest URL хардкоден** на личный GitHub-репо. На момент правок 19.05.26
  сервер юзера мёртв, манифест ниоткуда не тянется. RemoteUpdateWorker
  graceful'но retry'ится, батарею не ест. Когда сервер вернётся — перейти
  на доменное имя.
- ~~`min_app_db_version` не enforced~~ — FIXED.
- ~~CallRecord ↔ FeatureSnapshot~~ — FIXED via MIGRATION_5_6.
- ~~Threshold drift в FeedbackLearningStore~~ — **ложная тревога**: уже
  clamp'нут (`warnThreshold ∈ [0.15, 0.50]`, `blockThreshold ∈ [0.50, 0.85]`,
  weights ∈ [0.1, 3.0]).
- **PrebuiltBlocklistReader** — нет CRC/row-count проверки, partial-copy
  на low-storage не детектится → половина словаря молча теряется. Требует
  пересборки `assets/prebuilt_blocklist.db` через
  `scripts/build_prebuilt_blocklist_db.py` с добавлением row_count в meta.
- **App Category Model artifacts** — обучить через
  `scripts/train_app_category_classifier.py` (датасет уже готов в
  `datasets/categories/`), положить TFLite + vocab + card в
  `app/src/main/assets/`. Fallback к rule-based работает корректно, но
  long-tail apps классифицируются по substring-эвристикам.
- **Тестовая инфраструктура** — нет unit-тестов на FusionDecider,
  OnlineTrainer, WarmUpGate, ImplicitLabelDetector, SmartSpamDetector.
  Любой рефакторинг этих модулей сейчас рискованный.
- **SettingsScreen 1343 LOC** — God-screen, разбить на под-секции.
- **UsageStats permission banner на HomeScreen** — без активного prompt'а
  юзер не находит «Settings → Special access → Usage access», и фичи
  `recent_*_30m` Device-модели всегда нулевые.

## Conventions

- **CRLF** в Kotlin-исходниках. `edit` tool иногда промахивается на
  многострочных заменах — для пакетных правок использовать одноразовые
  Python-скрипты в `scripts/_apply_*.py`, удалять после применения.
- **Логи на русском в комментариях** допустимы, но имена тегов logcat и
  identifier'ы — английские. Тег `SpamBlocker_SGD` для диагностики
  обучения.
- Скрипт `scripts/_apply_*.py` — disposable, не коммитить.
- `.kiro/specs/<feature>/` — требования/дизайн/таски в Kiro-формате.
  `PROGRESS_JOURNAL.md` рядом — для статуса задач, когда `task_update`
  падает с EPERM на Windows из-за антивируса.
