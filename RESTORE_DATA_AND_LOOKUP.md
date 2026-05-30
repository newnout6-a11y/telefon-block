# RESTORE — восстановление данных и логики поиска номеров

Этот репозиторий содержит **только исходный код**. Базы данных, датасеты и
крупные бинарные ассеты намеренно НЕ загружены на GitHub (лимит 100 МБ на файл
и просто чтобы не тащить гигабайты данных в git).

Документ для следующего ИИ/разработчика: как восстановить всё, что отсутствует,
и как доделать незавершённую логику определения звонящего (caller lookup) и
голосового автоответчика (answer bot).

---

## 1. Чего НЕТ в репозитории (и почему)

### 1.1. Крупные ассеты приложения (`app/src/main/assets/`)

| Файл / папка | Размер | Назначение | Чем собрать |
|---|---|---|---|
| `prebuilt_blocklist.db` | ~128 МБ | Prebuilt SQLite-словарь спам-номеров | `scripts/build_prebuilt_blocklist_db.py` |
| `spam_numbers.csv` | ~32 МБ | Плоский список спам-номеров | `scripts/build_assets_from_dataset.py` |
| `prefix_histogram_7.json` | ~23 МБ | Гистограмма префиксов (7 знаков), исходник | `scripts/build_prefix_binary.py` |
| `phone_lookup.bin` | ~336 КБ | Бинарный реестр операторов/регионов РФ | `scripts/build_phone_lookup.py` (см. §3) |
| `answerbot/vosk-model-small-ru-0.22/` | ~87 МБ | Offline STT-модель Vosk (RU) | `scripts/download_vosk_model.ps1` (см. §4) |
| `answerbot/greeting.ogg` | — | Аудио-приветствие автоответчика | записать вручную, см. `GREETING_README.txt` |

> `prefix_histogram_7.phbin` (~2.5 МБ, компактный бинарный формат) **оставлен**
> в репозитории — он маленький и нужен рантайму. Крупный JSON-исходник убран.

### 1.2. Папка `datasets/` (~4.8 ГБ) — целиком отсутствует

Сырые и обработанные CSV для обучения моделей. Полностью регенерируемы через
скрипты пайплайна (`scripts/ru_*`, `scripts/build_*`, `scripts/train_*`).
В `.gitignore` уже прописаны самые крупные. Структура, которую ожидают скрипты:

```
datasets/
  categories/{train,val,test,labeled}.csv     # для app-category классификатора
  categories/raw/{appgallery,rustore,synthetic}.csv
  ru/raw/...                                   # сырые краулы репутации номеров
  ru/processed/...                             # фичи для обучения
  ru/eval/...                                  # golden-set / cold-eval
```

Точные имена и форматы см. в `.kiro/specs/` и в самих скриптах сборки.

### 1.3. Прочее, что не коммитится

- `releases/latest/*` — артефакты релиза (раздаются через GitHub Releases, не git).
- `build/`, `.gradle/`, `.kotlin/`, `catboost_info/` — сборочный мусор.
- `local.properties` — путь к Android SDK на конкретной машине. Создать локально:
  ```
  sdk.dir=C\:\\Users\\<user>\\AppData\\Local\\Android\\Sdk
  ```
- `batch_*.tar.gz` — локальные дампы датасетов.

---

## 2. Быстрый старт сборки APK

```powershell
# 1. Создать local.properties с путём к Android SDK (см. §1.3).
# 2. (опционально) Подложить крупные ассеты — без них приложение
#    собирается, но блок-словарь/STT/greeting не работают.
# 3. Сборка:
./gradlew.bat :app:assembleDebug --no-daemon
```

APK: `app/build/outputs/apk/debug/app-debug.apk`.

Зависимости поиска/STT уже прописаны в `app/build.gradle.kts`:
```
com.googlecode.libphonenumber:libphonenumber:8.13.26
com.googlecode.libphonenumber:geocoder:2.249
com.googlecode.libphonenumber:carrier:1.239
com.alphacephei:vosk-android:0.3.47@aar
net.java.dev.jna:jna:5.13.0@aar
```

---

## 3. Логика определения звонящего (caller lookup) — НЕ доделана

Цель: показывать в журнале звонков оператора + регион (offline) и название
организации (online через 2ГИС).

### 3.1. Что уже есть в коде (готово)

```
domain/lookup/
  CallerInfo.kt              # view-model + toEntity / toCallerInfo
  OfflineCallerLookup.kt     # libphonenumber geocoder+carrier
  TwoGisCallerLookup.kt      # 2ГИС Catalog API (HTTP, opt-in)
  CallerLookupRepository.kt  # оркестратор: cache -> offline -> 2gis
  CallerLookupWorker.kt      # WorkManager, дёргается после CallLog-события
  RussianPhoneLookup.kt      # парсер бинарного phone_lookup.bin (RKN DEF-9xx)
data/db/entity/CallerLookup.kt           # Room entity (таблица caller_lookup)
data/db/dao/CallerLookupDao.kt           # observe / upsert / purgeStale
data/prefs/CallerLookupSettingsStore.kt  # DataStore: twoGisEnabled + apiKey
```

Поток данных:
1. `CallEventRecorder.handleRow` -> `CallerLookupWorker.enqueue(ctx, number)`
2. Worker -> `repo.ensureOffline(number)` -> libphonenumber -> Room
3. Если 2ГИС включён -> `repo.fetchOnline(number)` -> HTTP -> Room
4. `CallLogScreen` подписан на `repo.observe(number): Flow<CallerInfo?>`

### 3.2. Чего НЕ хватает (что доделать)

1. **Ассет `phone_lookup.bin` отсутствует** — без него `RussianPhoneLookup.lookup()`
   всегда возвращает `null` и работает только libphonenumber-fallback (грубее).
   Как собрать — см. §3.3.

2. **Источник реестра РКН.** `build_phone_lookup.py` сейчас читает захардкоженный
   путь `C:/tmp/def9.csv`. Нужно:
   - скачать актуальный DEF-9xx реестр с opendata.digital.gov.ru
     (Минцифры, «Выписка из реестра российской системы и плана нумерации»,
     коды DEF 9xx — мобильные);
   - сохранить как `C:/tmp/def9.csv` (разделитель `;`, UTF-8/CP1251);
   - параметризовать путь через `argparse` вместо хардкода (TODO).

3. **Вызов `RussianPhoneLookup.load(context)` нигде не выполняется.** Парсер готов,
   но его никто не инициализирует. Доделать:
   - вызвать `RussianPhoneLookup.load(applicationContext)` в `SpamBlockerApp.onCreate`
     на `Dispatchers.IO` (идемпотентно, ~344 КБ в память);
   - в `OfflineCallerLookup` сначала пробовать `RussianPhoneLookup.lookup(e164)`
     (точные операторы/регионы РКН), при `null` — fallback на libphonenumber.

4. **2ГИС по умолчанию выключен.** Проверить, что в `SettingsScreen` есть toggle
   «Определение звонящего» + поле API-ключа, связанные с
   `CallerLookupSettingsStore`. Бесплатный ключ: dev.2gis.ru (5000 запросов/день).
   Без ключа online-ветка просто пропускается.

5. **Тесты отсутствуют.** Покрыть юнит-тестами:
   - `RussianPhoneLookup` — корректность бинарного парсинга и бинарного поиска
     по диапазонам (формат `PLKU`, big-endian, key = DEF*1e7 + subscriber).
   - `TwoGisCallerLookup.parseResponse` — happy/empty/negative/ошибка.

### 3.3. Сборка `phone_lookup.bin`

Формат файла (big-endian), уже реализован в обоих концах
(`build_phone_lookup.py` пишет, `RussianPhoneLookup.kt` читает):

```
magic[4]="PLKU" + N[4] + Nop[2] + Nreg[2]
string table: (Nop+Nreg) строк, каждая = 1-byte-len + UTF-8 байты
N x запись (interleaved): from_key[8] + to_key[8] + op_idx[2] + reg_idx[2]
key = DEF * 10_000_000 + 7-значный-номер-абонента
```

```powershell
# 1. Положить реестр РКН в C:/tmp/def9.csv (CSV ; -разделитель).
# 2. Собрать ассет:
py scripts/build_phone_lookup.py
# 3. Появится app/src/main/assets/phone_lookup.bin (~336 КБ)
#    + лог C:/tmp/lookup_build.txt со спот-чеком номеров.
```

Скрипт нормализует операторов (`_OP_MAP`) и регионы (`_REGION_REPLACE`) —
при изменении формата реестра проверить эти таблицы.

---

## 4. Голосовой автоответчик (answer bot) — НЕ доделан

```
domain/answerbot/
  VoskRecognizer.kt              # обёртка Vosk STT
  SilenceDetector.kt             # детекция тишины
  AnswerBotTranscriptionWorker.kt
service/SpamAnswerBotService.kt
ui/screens/AnswerBotMessagesScreen.kt
data/db/entity/AnswerBotMessageEntity.kt + dao/AnswerBotMessageDao.kt
data/prefs/AnswerBotSettingsStore.kt
```

Чего не хватает:
1. **Vosk-модель** (`assets/answerbot/vosk-model-small-ru-0.22/`, ~87 МБ):
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/download_vosk_model.ps1
   # тянет vosk-model-small-ru-0.22.zip с alphacephei.com и распаковывает.
   ```
2. **Приветствие** `assets/answerbot/greeting.ogg` — записать вручную
   (см. `GREETING_README.txt`). В репозитории README остаётся, само .ogg — нет.
3. Прогнать end-to-end: входящий звонок -> проигрывание greeting ->
   запись/распознавание -> сохранение транскрипта в Room -> экран сообщений.

---

## 5. Где смотреть детали

- `AGENTS.md` — заметки по инвариантам, исправленным edge-case'ам, известным
  нерешённым P0/P1 пунктам (manifest-подпись, min_app_db_version, миграции БД).
- `.kiro/specs/<feature>/` — требования/дизайн/таски в Kiro-формате.
- `scripts/` — весь пайплайн сбора датасетов, обучения и сборки ассетов.
- `docs/research/` — исследования по определению мошеннических звонков.

> ВАЖНО: при возврате рабочего сервера обновлений — manifest URL сейчас
> захардкожен на мёртвый GitHub-репо (см. `RemoteUpdateWorker` и AGENTS.md).
