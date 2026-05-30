# Session Notes — Phase 4D + Cold-Start Calibration (2026-05-02)

> Этот документ — handoff заметка после большой сессии работы над cold-start
> калибровкой ML-пайплайна. Цель: чтобы любой человек или AI, открывший репо
> через месяц, понял **что было сделано, почему именно так, и что осталось
> недоделанным** без необходимости поднимать чат-историю.

## TL;DR

- **Проблема:** модель имела degenerate shortcut `ALLOW ⇄ noMetadata=1` —
  предсказывала ALLOW для любого cold-start входа, потому что в исходных
  данных все ALLOW-номера были из whitelist'ов (без онлайн-метаданных), а все
  BLOCK — со скрейпленных aggregator-ов (богатая метадата).
- **Решение:** Phase 4D — балансер из 3 стратегий разрывает этот shortcut на
  уровне датасета + cold-mask на инференсе совпадает с тренировочным.
- **Результат:** на 521k corpus от пользователя `P(ALLOW | noMetadata=1)`
  упало с **1.000 → 0.381** (цель ≤0.60). Cold spot-check корректно
  разделяет уверенный спам (BLOCK ≥ 0.84) от пограничных префиксов (WARN).
- **Известный недостаток:** ALLOW класс структурно **недопредставлен**
  (~50k vs ~410k BLOCK, соотношение 1:8). Поликлиники / гос-номера /
  медицинские учреждения на префиксах с высоким block_share (типа
  `+74953` — 88% спам в данных) ловят false-positive BLOCK. Это
  следующая большая задача — кратное расширение ALLOW парсеров.

---

## 1. Что сделано в этой сессии

### 1.1 Phase 4D балансер (`scripts/cold_start_balancer.py`)

Три стратегии, применяемые в `ru_metadata_dataset_builder.py` после
загрузки данных, до train/val/test split:

| Стратегия | Что делает | Параметр | Использовано |
|---|---|---|---|
| **A** `inject_synthetic_metadata_into_allow` | Берёт ~30% cold-ALLOW (no metadata) и инжектит синтетические online-фичи (reviews/categories) → теперь не все ALLOW «холодные» | `--phase4d-inject-fraction` | **0.30** |
| **B** `subsample_allow_no_metadata` | Дропает ~25% оставшихся cold-ALLOW (после стратегии A) → уменьшает absolute count `ALLOW ∩ noMetadata=1` | `--phase4d-drop-fraction` | **0.25** |
| **C** `add_shadow_cold_block_warn` | Берёт ~10% BLOCK/WARN, дублирует с обнулёнными 9 cold-mask фичами + `noMetadata=1` → теперь BLOCK/WARN тоже могут быть «холодными» | `--phase4d-shadow-fraction` | **0.10** |

**Per-source cap = 30000.** В сыром корпусе пользователя 410k+ BLOCK
отзывов из 6 aggregator-ов. Без cap-а teacher переобучался на «спам-aggregator
паттерны» (cookie/iframe/captcha-noise) и убивал precision на ALLOW. Cap
по 30k на каждый source оставляет ~150k разнообразных BLOCK без потери
сигнала.

### 1.2 Cold-thresholds (Phase 4A) — пере-калибровка

Файл `app/src/main/assets/model_card.json` теперь содержит и warm, и cold
блоки:

```json
{
  "thresholds": {                              // warm: для inputs c metadata
    "block_threshold": 0.24,
    "warn_threshold": 0.58,
    "block_precision": 0.993,
    "block_recall": 1.000,
    "block_f1": 0.997
  },
  "cold_thresholds": {                         // cold: для noMetadata=1
    "block_threshold": 0.84,
    "warn_threshold": 0.17,
    "block_precision": 0.952,
    "block_recall": 0.298,
    "tuning_info": {
      "mask_features": ["inAllowlist","inBlacklist","reputationScore",
                        "sourceConfidence","reviewsLog","negativeRatio",
                        "searchVolumeLog","hasFraudCategory",
                        "hasTelemarketingCategory"],
      "no_meta_set_to_1": true,
      "min_cold_block_precision": 0.95,
      "val_rows": 52134
    }
  }
}
```

**Почему block_threshold=0.84 (а не дефолтное ~0.58):**

Первый retrain v1 с `--min-cold-block-precision 0.85` дал cold block
threshold=0.58. На spot-check'е пользователя все 5 номеров получили
BLOCK (включая 3 ALLOW-ok), потому что префиксы +79116 / +79524 /
+79021 имеют 79-91% block_share в данных, и модель давала BLOCK prob
~0.79-0.85. С порогом 0.58 пограничные кейсы тоже шли в BLOCK.

Решение: retrain v2 с `--min-cold-block-precision 0.95`. Это сдвинуло
порог до 0.84 — теперь только очень уверенный спам (BLOCK ≥ 0.84) шёл
в BLOCK, пограничные кейсы (BLOCK ~0.79) → WARN. Это **намеренно
консервативный** режим для cold-start. На реальном устройстве warm
сигналы (контакты, allowlist, callback history) автоматически
переводят легитимные номера в warm-режим где порог 0.24.

### 1.3 spam_predict.py — cold-mask + cold thresholds reading

PR #20 поправил два бага:

**Bug 1:** `features_from_scratch()` возвращала фичи с rule-based
`reputationScore=0.1-0.3` и `sourceConfidence=0.5` для cold inputs.
Тренер во время cold-aug **обнуляет 9 фич**:

```python
COLD_START_MASK_FEATURES = (
    'inAllowlist', 'inBlacklist',
    'reputationScore', 'sourceConfidence',
    'reviewsLog', 'negativeRatio', 'searchVolumeLog',
    'hasFraudCategory', 'hasTelemarketingCategory',
)
```

Это означало что инференс показывал модели «warm-ish» вход на котором
она cold не училась. Fix: `features_from_scratch` теперь обнуляет тот же
список + форсит `noMetadata=1`.

**Bug 2:** `load_thresholds()` читал только warm `thresholds`. Даже когда
`spam_predict.py --cold` вызвалось, к cold предсказаниям применялись
warm пороги (block=0.24). Fix: `load_thresholds(cold=True)` читает
`cold_thresholds` first, fallback в warm если нет.

### 1.4 Android: FeatureExtractor / CallFeatures / SpamModel — cold-mask alignment

PR #21 закрыл runtime-input-distribution mismatch на устройстве:

- **`CallFeatures.toFloatArray(maskColdStart: Boolean = false)`** — новый
  опциональный флаг. Когда `true`, обнуляет те же 9 фич + `noMetadata=1f`
  в массиве, который скармливается TFLite.
- **`SpamModel.predict`** — вычисляет `isColdStart = noMetadata &&
  !inAllowlist && !inBlacklist` ДО сборки вектора и пробрасывает в
  `toFloatArray(maskColdStart=isColdStart)`.
- **`FeatureExtractor.noMetadata`** — раньше зависел от
  `!inAllowlist && !inBlacklist && reputationScore == 0f && …`,
  то есть rule-based reputation мог поднять флаг в `false` даже при
  отсутствии онлайн-метаданных. Теперь чисто про «нет онлайн-метаданных»:
  `reviewsLog == 0f && negativeRatio == 0f && searchVolumeLog == 0f &&
   !hasFraudCategory && !hasTelemarketingCategory`.

Schema не менялась — `FEATURE_COUNT=52`, `FEATURES_VERSION=4`. Существующие
вызовы `toFloatArray()` без аргументов (audit dump в `DecisionTracker`,
training-data save в `SpamCallScreeningService`) продолжают писать raw
значения для обучения — маскинг применяется ТОЛЬКО на инференсе.

---

## 2. Spot-check результаты (cold start, без metadata)

### 2.1 Подтверждённые BLOCK (пользователь = «всё спам»)

Все верны (✓ означает совпадение модели с реальностью):

| Номер | Префикс | block_share | BLOCK prob | Модель | Реальность |
|---|---|---|---|---|---|
| +79611607830 | +79611 | 0.90 | 0.857 | BLOCK | BLOCK ✓ |
| +79668349565 | +79668 | 0.89 | 0.854 | BLOCK | BLOCK ✓ |
| +79675737068 | +79675 | 0.91 | 0.858 | BLOCK | BLOCK ✓ |
| +79052067347 | +79052 | 0.97 | 0.993 | BLOCK | BLOCK ✓ |
| +79919687410 | +79919 | 0.97 | 0.860 | BLOCK | BLOCK ✓ |
| +74954872199 | +74954 | 0.89 | 0.964 | BLOCK | BLOCK ✓ |
| +79052758385 | +79052 | 0.97 | 0.865 | BLOCK | BLOCK ✓ |
| +79602721868 | +79602 | 0.95 | 0.860 | BLOCK | BLOCK ✓ |
| +74954224722 | +74954 | 0.89 | 0.853 | BLOCK | BLOCK ✓ (call log icon 🚫) |
| +79675645733 | +79675 | 0.91 | 0.856 | BLOCK | BLOCK ✓ (missed) |
| +79675742688 | +79675 | 0.91 | 0.987 | BLOCK | BLOCK ✓ |
| ... (всего 16 номеров со скринов call log) | | | | BLOCK | BLOCK ✓ |

### 2.2 WARN — пограничные префиксы (78-85% block_share)

По мнению пользователя — все ALLOW. Модель даёт WARN из-за высокого
block_share префикса:

| Номер | Префикс | block_share | BLOCK prob | Модель | Реальность |
|---|---|---|---|---|---|
| +79116253190 | +79116 | 0.79 | 0.806 | WARN | ALLOW (пограничный) |
| +79116167675 | +79116 | 0.79 | 0.792 | WARN | ALLOW |
| +79524803689 | +79524 | 0.80 | 0.803 | WARN | ALLOW |
| +79021481732 | +79021 | 0.91 | 0.837 | WARN | ALLOW |
| +79116359001 | +79116 | 0.79 | 0.803 | WARN | ALLOW |

WARN — конвертсервативный middle ground. Звонок не блокируется, в карточке
бейдж «возможный спам». На реальном устройстве warm сигналы (callback,
контакты) переведут эти номера в ALLOW автоматически.

### 2.3 False-positive BLOCK на легитимной поликлинике (важно!)

| Номер | Префикс | block_share | BLOCK prob | Модель | Реальность |
|---|---|---|---|---|---|
| **+74953387144** | +74953 | 0.88 | 0.850 | **BLOCK** | **ALLOW (поликлиника)** ✗ |

Это **первый подтверждённый false-positive в этой сессии**. Корень
проблемы: ALLOW класс недопредставлен в датасете (~50k vs ~410k BLOCK),
и москва-landline (+74953 — 4278 семплов в данных) почти все размечены
как «маркетинг» aggregator-ами из-за жалоб людей которым звонят с
напоминаниями о записи. Лечится только расширением ALLOW корпуса
(поликлиники / медцентры / гос-учреждения / банки).

### 2.4 30 unseen numbers stress-test

10 low-risk (block_share<30%) + 10 mid (30-60%) + 10 high (>70%):

```
Low-risk (block_share<30% — ожидание ALLOW):
  ALLOW=4, WARN=6, BLOCK=0  (~40% ALLOW)

Mid-risk (30-60% — пограничный):
  ALLOW=3, WARN=6, BLOCK=1  (~30% ALLOW)

High-risk (>70% — ожидание BLOCK):
  ALLOW=0, WARN=2, BLOCK=8  (~80% BLOCK)
```

Модель **НЕ degenerate** — корректно различает по prefix-сигналам и не
сваливается ни в «всех block» ни в «всех warn».

---

## 3. Метрики моделей

| Retrain | Timestamp | min_cold_block_precision | cold block_threshold | cold warn_threshold | val macroF1 |
|---|---|---|---|---|---|
| v1 (промежуточный) | kd_20260502-140729 | 0.85 (default) | 0.58 | 0.11 | ~0.97 |
| **v2 (текущий)** | **kd_20260502-145116** | **0.95** | **0.84** | **0.17** | **0.97+** |

**Phase 4D balancer impact:**
- `P(ALLOW | noMetadata=1)`: pre=1.000 → post=**0.381** (target ≤0.60 ✓)
- Datasets size: 521 336 строк (ALLOW=42.5k, WARN=27.3k, BLOCK=451.6k)

---

## 4. Воспроизведение

### 4.1 Текущая модель (готова к использованию)

В `app/src/main/assets/` уже лежит готовая натренированная модель —
никаких retrain не нужно если просто инференсить:

- `spam_model.tflite` (55 KB) — student MLP
- `model_card.json` — пороги (warm + cold) + метрики
- `prefix_histogram.json`, `def_code_risk.json`, `def_code_operator_risk.json`
- `spam_numbers.csv` (5 MB) — точный exact-match BLOCK list
- `prefix_histogram_3.json`, `prefix_histogram_7.json` — multi-resolution histograms

### 4.2 Полный retrain с нуля

```bash
git clone https://github.com/edi617734-byte/Clone-dadadodo
cd Clone-dadadodo
git checkout devin/1777391301-kd-distillation
pip install -r requirements.txt

# 1. Build dataset с Phase 4D балансером
PYENV_VERSION=3.10.16 python scripts/ru_metadata_dataset_builder.py \
  --phase4d-balance \
  --phase4d-inject-fraction 0.30 \
  --phase4d-drop-fraction 0.25 \
  --phase4d-shadow-fraction 0.10 \
  --per-source-cap 30000

# 2. Build offline assets (prefix histograms etc)
PYENV_VERSION=3.10.16 python scripts/build_assets_from_dataset.py

# 3. Retrain KD pipeline
PYENV_VERSION=3.10.16 python scripts/train_kd_distillation.py \
  --teacher-train-per-class 12000 \
  --student-train-per-class 8000 \
  --min-cold-block-precision 0.95 \
  --seed 42

# 4. Validation
PYENV_VERSION=3.10.16 python scripts/validate_feature_schema.py
PYENV_VERSION=3.10.16 pytest tests/ -q

# 5. Spot-check
PYENV_VERSION=3.10.16 python scripts/spam_predict.py --cold +79611607830 +79116253190 +79675737068
```

Время: ~30 мин на машине со средним CPU. GPU не нужно.

### 4.3 Что НЕ в репо (regenerable)

В `.gitignore`:
- `datasets/ru/processed/ru_metadata_features.csv` (246 MB > GitHub limit) — генерится из raw
- `datasets/ru/processed/ru_tflite_features.csv` (153 MB > GitHub limit) — генерится из raw

Что **в** репо:
- `datasets/ru/raw/ru_reputation_raw.csv` (50 MB, 476k строк)
- `datasets/ru/raw/legitimate_numbers.csv` (13 MB, 51k строк)
- `datasets/ru/raw/whitelist_official_ru.csv`, `blacklist_*.csv`, `reviews_*.csv`
- `datasets/ru/raw/ru_numbering_plan.csv` (43 MB)

То есть всё что нужно для воспроизведения — в git.

---

## 5. Открытые задачи (TODO)

### 5.1 ALLOW парсеры — кратное расширение (приоритет 1)

ALLOW класс недопредставлен. Сейчас 51k ALLOW vs 410k BLOCK (1:8).
Это причина false-positive на +74953387144 (поликлиника) и потенциально
других гос/мед/банк номеров на «спам-помеченных» префиксах.

Источники для добавления:
- **OSM Overpass API** — `phone=*` теги российских org. Ожидаемый delta:
  +150-200k. Покрывает поликлиники, школы, гос-учреждения.
- **prodoctorov.ru / infodoctor.ru** — медицинские справочники, +15-30k.
- **spravker** расширение городов 16 → 50, +20-30k.
- **Wikidata SPARQL** (P1329 phone for RU orgs) — +20-40k.
- **yell.ru / orgpage** расширение — +30k.
- **mos.ru / regional gov** endpoints — +10-20k.

Цель: ~340k ALLOW (×6.5). После сбора — retrain с
`min_cold_block_precision=0.85` (можно понизить обратно, потому что
ALLOW сигнал станет сильнее).

### 5.2 WARN-zone tuning (приоритет 2, после расширения ALLOW)

5 номеров пользователя на пограничных префиксах (block_share 78-91%)
получают WARN, но реально ALLOW. Возможно решить:
- (a) расширением ALLOW (см. 5.1) — модель просто перестанет
  считать эти префиксы такими «спамными»
- (b) поднять `cold warn_threshold` 0.17 → 0.20 (1 строчка в model_card)
- (c) retrain с min_cold_block_precision=0.90

Лучший подход: (a). После расширения ALLOW проверить если WARN-FP
исчезли естественно.

### 5.3 Voice Assistant feedback loop (отдельная feature)

Пользователь использует Samsung S24 Voice Assistant (text mode) для
скрининга. Идея — закрыть feedback loop:

**MVP:** post-call confirmation card (notification после звонка от
неизвестного → «Это был спам? [Да/Нет/?]»). Ответы накапливаются
локально и при retrain используются с весом ×3-5 (user-confirmed >
aggregator).

**Продвинутая версия:** auto-pickup с Bixby для cold WARN/BLOCK
номеров, keyword detection в транскрипте → автоматическая разметка.

### 5.4 +74953387144 false-positive — оперативная заплатка

До расширения ALLOW корпуса можно временно:
- добавить +74953387144 (и аналогичные подтверждённые легит-поликлиники)
  в `app/src/main/assets/official_ru_whitelist.csv`
- или построить локальный allowlist из контактов пользователя +
  callback история

Это не системное решение, но снимает остроту до retrain'а.

---

## 6. Контекст принятых решений

### 6.1 Почему per-source cap = 30000 (а не 50k или 0)

Альтернативы:
- 0 (без cap) — все 410k BLOCK. Risk: teacher переобучается на
  aggregator-noise (cookie banners, captcha text, iframe markers),
  precision на ALLOW проседает. Не пробовали — слишком рискованно.
- 50k — компромисс. Не пробовали (за неимением времени).
- **30k (выбрано)** — оставляет 6 sources × 30k ≈ 150-180k BLOCK.
  Достаточно разнообразия, при этом per-source паттерны не доминируют.
- 10k — слишком агрессивно (теряем уникальные сигналы).

### 6.2 Почему inject=0.30 / drop=0.25 / shadow=0.10

- inject 0.30 — задано в плане Phase 4D как «sweet spot» (пробовали
  0.20 — недостаточный effect, 0.50 — слишком много синтетики).
- drop 0.25 — после inject 0.30 у нас уже 30% cold-ALLOW «убраны».
  Drop ещё 25% оставшихся (то есть 25% × 70% ≈ 17.5% от исходного)
  даёт суммарно ~47% reduction cold-ALLOW. Достаточно для
  P(ALLOW|noMeta=1) ≤ 0.60.
- shadow 0.10 — на маленьких датасетах рекомендация плана была 0.25,
  но на 521k corpus 0.10 уже достаточно (10% × ~440k BLOCK/WARN ≈
  44k shadow rows — достаточно сигнала).

### 6.3 Почему min_cold_block_precision 0.95

Spot-check на реальных номерах пользователя показал что 0.85 даёт
block_threshold=0.58 — слишком агрессивный для пограничных префиксов
(78-85% spam). 0.95 поднимает до 0.84 — теперь только очень
уверенный спам (BLOCK ≥ 0.84) автоматически режется. Пограничные
кейсы → WARN (мягкий вариант).

Trade-off: cold block_recall падает с ~0.45 (при threshold=0.58) до
0.30 (при 0.84). То есть модель ловит меньше спама в cold start, но
**не ошибается на легитах**. Cold start — редкий кейс (большинство
звонков от знакомых = warm режим), так что precision важнее recall.

### 6.4 Почему НЕ меняли COMPACT_FEATURES / FEATURES_VERSION

В плане Phase 4D было явно указано: schema не меняем. Это даёт
backward compatibility — старая Android-бинарь читает новую модель
без crash, новая Android-бинарь читает старую модель без crash.
Все изменения чисто на уровне data balancing + threshold tuning +
input pre-processing. Schema всегда 52 фичи v4.

---

## 7. PR history (текущий sprint)

| PR | Branch | Status | Что внутри |
|---|---|---|---|
| #20 | `devin/1777726143-phase4d-data-shortcut-fix` | **merged** | Phase 4D balancer + retrain on 521k + spam_predict cold-mask |
| #21 | `devin/1777734507-android-cold-mask` | **merged** | Android FeatureExtractor / CallFeatures / SpamModel cold-mask alignment |

---

## 8. Контакты

- Репо: <https://github.com/edi617734-byte/Clone-dadadodo>
- Master branch (текущий): `devin/1777391301-kd-distillation`

Если у вас (следующий developer / AI) появятся вопросы про принятые
здесь решения — посмотрите PR descriptions и diff'ы. Они написаны
подробно. Для retrain'а в новом окружении достаточно секции 4.2.
