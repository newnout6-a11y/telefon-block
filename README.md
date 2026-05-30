# Блокировщик спама (SpamBlocker)

Android-приложение (Kotlin + Jetpack Compose) для блокировки мошеннических звонков. Работает **полностью без интернета**.

## Возможности

- Блокировка звонков через системный `CallScreeningService` (Android 10+)
- Гибкая реакция: **BLOCK** (сброс) или **WARN** (беззвучный + heads-up уведомление)
- 7 правил с приоритетом: белый список → контакты → чёрный список → предзагруженная база → скрытые номера → подозрительные префиксы → нестандартная длина
- **ML-модели**: LogisticRegression, RandomForest, CatBoost, TFLite MLP — 32 компактных признака
- **SMOTE** балансировка классов + oversampling редких паттернов
- **Optuna** гиперпараметрический поиск + 5-fold кросс-валидация
- **Per-class threshold tuning** с PR-компромиссом (argmax F1)
- **Platt/Isotonic калибровка** для LR и RF
- **Drift detection** (KS-test) между training и production данными
- **Feature importance**: RF Gini + Mutual Information + Permutation Importance
- **Decision tracking**: Room БД с 32 признаками, вероятностями, вердиктом, причинами
- **Model debug screen**: последние 50 решений, статистика, feedback agreement rate
- **ML dashboard HTML**: precision/recall/F1/ROC + confusion matrix + importance plots
- В один тап «Это мошенник» / «Это не спам» прямо из журнала или уведомления
- UI на русском (Jetpack Compose + Material 3)

## Требования

- **Android Studio** Hedgehog или новее (2023.1+)
- **JDK 17+** (Android Studio JBR)
- Устройство или эмулятор с **Android 10 (API 29)** или новее
- **Python 3.10+** (для ML-пайплайна)

## QUICKSTART

### 1. Проверка окружения

```powershell
.\run.ps1 doctor
# или
python tools/spam_cli.py doctor
```

### 2. Сборка датасета

```powershell
.\run.ps1 build-dataset
# с синтетикой для теста:
.\run.ps1 build-dataset -SmokeSynthetic 80
```

### 3. Обучение моделей

```powershell
# базовое обучение
.\run.ps1 train

# обучение + экспорт TFLite + графики
.\run.ps1 train -ExportTflite -Plots

# с Optuna гиперпараметрическим поиском (30 trials)
.\run.ps1 train -ExportTflite -OptunaTrials 30 -Plots

# без SMOTE
.\run.ps1 train -NoSmote
```

### 3.1. Leak-free training (PR-1)

Опция `--leak-free` физически зануляет 9 metadata-фич (`reputationScore`,
`sourceConfidence`, `reviewsLog`, `negativeRatio`, `searchVolumeLog`,
`hasFraudCategory`, `hasTelemarketingCategory`, `inAllowlist`, `inBlacklist`)
во **всех** сплитах перед обучением. Эти фичи на устройстве недоступны без
интернета, поэтому обучение на них даёт train-test mismatch (warm BLOCK
recall ≈ 0.9999, но на cold-thresholded eval ALLOW precision проседает до 0.42).

```bash
python3 scripts/train_kd_distillation.py --leak-free
```

Артефакты пишутся в `app/src/main/assets/experimental/spam_model_leak_free.tflite`
и `experimental/model_card_leak_free.json` — прод (`spam_model.tflite`) не
подменяется. После проверки на golden-set отдельный PR обновит Android, чтобы
читать этот файл.

### 4. Экспорт модели

```powershell
.\run.ps1 export -Plots
```

### 4.1. Бинарная модель + Platt калибровка (PR-2)

Альтернатива 3-классовому KD-пайплайну. WARN-класс убран как самостоятельный
выход (он не работал — F1 = 0 в warm режиме / 0.30 в cold), вместо этого
WARN получаем как «зону неопределённости» на калиброванной шкале p_spam:
`p ≥ block_threshold` → BLOCK, `p ≥ warn_threshold` → WARN, иначе ALLOW.

```bash
# Обучение бинарной модели (WARN сливается в spam=1 по умолчанию)
python3 scripts/train_binary_model.py

# Альтернативные стратегии WARN
python3 scripts/train_binary_model.py --binary-warn-strategy drop
python3 scripts/train_binary_model.py --binary-warn-strategy merge_allow
```

Что делается:

1. Сэмплинг: WARN→spam=1 (по умолчанию), BLOCK→spam=1, ALLOW→spam=0.
2. Обучение MLP с binary cross-entropy + class-balanced weights.
3. **Platt scaling** на отдельном hold-out из val: `p_spam = sigmoid(a*z + b)`.
   Параметры a, b встраиваются как константы прямо в TFLite-граф, поэтому
   интерпретатор сразу возвращает калиброванную вероятность.
4. ECE и Brier до/после калибровки записываются в model_card для аудита.
5. `output_format: "binary_sigmoid"` в карточке. Android-сторона (PR-2)
   читает это поле и парсит выход [1,1] вместо [1,3].

Артефакты пишутся в `app/src/main/assets/experimental/spam_model_binary.tflite` —
прод (`spam_model.tflite`) не подменяется. Промоутить можно после прогона
на golden-set (PR-6).

### 5. Drift detection

```powershell
.\run.ps1 drift -DriftReference datasets/ru/processed/ru_tflite_features.csv
```

### 6. Проверка качества данных

```powershell
.\run.ps1 quality
```

### 7. Сборка Android APK

```powershell
.\run.ps1 android-build
# или
.\gradlew.bat assembleDebug
```

### 8. Тесты

```powershell
# Python тесты (46 тестов)
python -m pytest tests/ -v

# Android тесты
.\gradlew.bat test
```

### 9. Сбор cold-start ALLOW-кандидатов из VK (опционально)

Для оценки cold-start поведения модели можно собрать публичные посты пользователей с RU-классифайд-стен ВКонтакте. Это **не** идёт в тренировочную выборку — пишется в отдельный `datasets/ru/eval/vk_candidates.csv` и прогоняется через модель ниже.

**1. Создай VK Standalone-приложение** на https://vk.com/apps?act=manage:
- «Создать приложение» → тип **«Standalone-приложение»** → название любое (например `spam-eval-collector`).
- Откроется страница приложения → **«Настройки»** → найди **«Сервисный ключ доступа»** (Service Access Token) и скопируй строку.
- Положи токен в env: `export VK_SERVICE_TOKEN="<строка>"`.

Этот токен — не user-OAuth: он даёт доступ только к **публичным** методам API. Не требует логина, SMS, 2FA.

**2. Запусти сборщик**:
```bash
python scripts/vk_token_collector.py \
    --output datasets/ru/eval/vk_candidates.csv \
    --max-calls 200 --max-groups 10
```

Источники (только wall-based, что работает с service-token):
- `wall.search query="+7"` по seed-списку публичных «барахолок» — самый плотный источник
- `wall.get` по тем же группам с фильтром `from_id > 0` (только посты юзеров, не админов)
- `wall.getComments` (опционально, флаг `--with-comments`)

Что фильтруется:
- посты от лица сообщества (`from_id <= 0`)
- номера, уже лежащие в `datasets/ru/processed/ru_reputation_raw.csv` (known фрод)
- не-российские номера (через `normalize_ru_phone(reject_non_ru=True)`)

Выход: CSV со схемой `normalized_number, vk_source, vk_object_id, vk_object_url, sample_text, expected_label=ALLOW, collected_at`.

**3. Прогон через модель** (находим disagreement: где cold-start модель не согласна с дефолтным ALLOW):
```bash
python scripts/spam_predict.py --cold \
    --from-csv datasets/ru/eval/vk_candidates.csv \
    --out-csv  datasets/ru/eval/vk_verdicts.csv
```

В выходном CSV колонка `disagreement=1` отмечает номера, где модель/правила говорят что-то отличное от ALLOW — это материал для fine-tune (см. `scripts/online_fine_tune.py`) или для калибровки порогов в `model_card.json`.

## Makefile (Linux/macOS/Git Bash)

```bash
make doctor          # проверка окружения
make train-export    # обучение + экспорт + графики
make train-optuna    # обучение с Optuna
make export          # экспорт TFLite
make drift           # drift detection
make quality         # проверка качества данных
make validate        # валидация схемы
make test            # Python тесты
make android-build   # сборка APK
make clean           # очистка отчётов
```

## Как собрать (Android Studio)

1. Откройте Android Studio.
2. **File → Open** → выберите папку проекта.
3. Android Studio скачает Gradle, SDK, зависимости (первый раз ~5–10 минут).
4. После синхронизации нажмите **Run ▶** (Shift+F10).

## Первый запуск

1. При первом запуске откроется **экран онбординга**.
2. Выдайте разрешения: **Контакты → Журнал звонков → Уведомления**.
3. Назначьте приложение **«Приложением для определения номера и фильтрации вызовов»**.
4. Готово! Входящие звонки проходят через фильтр.

## ML Pipeline

```text
datasets/ru/raw/           → исходные CSV (whitelist, blacklists, reputation, evidence)
datasets/ru/processed/     → ru_tflite_features.csv (32 признака + label)
datasets/ru/reports/       → JSON/HTML/MD отчёты + PNG графики (gitignore)

scripts/
├── ru_reputation_crawler.py       — ночной краулер 6 источников (spravportal, callfilter, zvonili, moshelovka, bloha, getscam)
├── ru_collect_sources.py          — безопасный оффлайн-сборщик данных
├── ru_dataset_builder.py          — сборка датасета из raw CSV
├── ru_metadata_dataset_builder.py — raw reputation → labels → features CSV
├── ru_metadata_features.py        — 32 компактных признака (shared Python-Kotlin)
├── train_ru_metadata_models.py    — обучение LR/RF/CatBoost/TFLite + Optuna/SMOTE/CV
├── online_fine_tune.py            — онлайн-подстройка из фидбека
├── ru_number_normalizer.py        — РФ-нормализация номеров
├── ru_numbering_plan.py           — план нумерации РФ (Россвязь)
├── validate_feature_schema.py     — проверка Kotlin↔Python схемы (32 признака)
├── validate_ru_data.py            — валидация данных
└── check_data_quality.py          — качество raw/processed CSV

tools/
└── spam_cli.py                    — CLI: doctor/train/export/drift/quality/status/collect
```

### 32 Compact Features

| Категория | Признаки |
|-----------|----------|
| Контакт | `isContact`, `isRuNumber`, `isForeignNumber`, `isShortCode` |
| Структура номера | `isStandardLength`, `is8800`, `isGeographical`, `isMobile`, `isValidRuRange`, `spoofingPrefixFlag` |
| Паттерны цифр | `digitEntropy`, `repeatDigitRatio`, `maxSameDigitRun`, `beautifulNumberFlag` |
| Префикс/риск | `prefixRisk` |
| Локальный контекст | `callFrequency7d`, `isNightCall`, `recentBankGov`, `recentMarketplace`, `recentMessenger`, `wasRejected` |
| Источники | `inBlacklist`, `inAllowlist`, `isHiddenNumber`, `callerVerification` |
| Пользователь | `userVulnerability`, `businessActivity` |
| Разрешения | `permissionsAvailable` |
| Репутация | `reputationScore`, `sourceConfidence` |

## Структура проекта

```text
app/src/main/java/com/antispam/blocker/
├── SpamBlockerApp.kt        — Application, импорт CSV
├── MainActivity.kt          — точка входа UI
├── service/                 — CallScreeningService
├── domain/
│   ├── detector/            — движок правил (SpamDetector, 7+ правил)
│   ├── scoring/             — Smart Risk Engine (CallFeatures, FeatureExtractor, SpamModel, FeedbackHandler, UserProfileVector)
│   ├── model/               — SpamModel (TFLite), ModelCard
│   └── tracking/            — DecisionTracker (Room audit)
├── data/
│   ├── db/                  — Room БД (entities, DAOs, AppDatabase)
│   ├── prefs/               — DataStore (Settings, FeedbackLearning, ProfileVector)
│   ├── repository/          — BlockListRepository, CallLogRepository
│   ├── assets/              — CsvSpamImporter, OfficialWhitelistImporter
│   └── worker/              — ModelFineTuneWorker, SpamDbUpdateWorker
├── notification/            — SpamWarningNotifier, SpamActionReceiver
├── ui/screens/              — Home, CallLog, Blacklist, Rules, Settings, Onboarding, Questionnaire, ModelDebug
└── util/                    — PhoneNormalizer, RoleManagerHelper

docs/
└── research/               — Исследовательские документы по проекту
```

## Python-зависимости (опционально)

```
pip install scikit-learn numpy imbalanced-learn optuna catboost tensorflow matplotlib scipy
```

| Пакет | Назначение |
|-------|-----------|
| scikit-learn | LR, RF, CV, metrics, calibration |
| numpy | Массивы |
| imbalanced-learn | SMOTE |
| optuna | Гиперпараметрический поиск |
| catboost | CatBoost модель |
| tensorflow | TFLite MLP |
| matplotlib | Графики (CM, importance, MI) |
| scipy | KS-test (drift detection) |

## Известные ограничения

- **Android 9 и старше — не поддерживается** (`minSdk = 29`).
- Только **одно** приложение может быть фильтром звонков.
- На Xiaomi/Huawei/Oppo — отключите оптимизацию батареи.

## Лицензия

MIT
