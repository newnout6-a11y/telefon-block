# Requirements Document

## Introduction

Эта спецификация описывает воспроизводимый и кросс-платформенный пайплайн обучения двух TFLite-моделей антиспама — `leak_free` (3-class KD) и `binary` (бинарная + Platt-калибровка) — поверх уже существующих Python-скриптов и shell-обёрток репозитория (`scripts/train_full_pipeline.sh`, `scripts/train_kd_distillation.py`, `scripts/train_binary_model.py`, `scripts/eval_golden_set.py`, `scripts/ru_metadata_dataset_builder.py`).

Пайплайн обязан:
- Корректно работать на Windows (PowerShell), Linux и macOS без обращения к Git Bash как к скрытой зависимости.
- Гарантировать, что `leak_free` остаётся обязательным режимом обучения, потому что 9 metadata-фич недоступны на устройстве при отсутствии интернета (риск train-test mismatch).
- Останавливаться с понятной ошибкой при отсутствии входных датасетов или eval-набора, не делая «тихих» пропусков.
- Не трогать продовые артефакты (`app/src/main/assets/spam_model.tflite`, `model_card.json`) автоматически: промоушен — отдельное явное действие.

Не-цели: переписывать ML-логику, менять архитектуру нейросетей или CatBoost-учителя, добавлять новые источники данных, изменять формат датасета.

## Glossary

- **Pipeline_Runner**: оркестратор, который последовательно вызывает шаги пайплайна (pre-flight → dataset build → train leak-free → train binary → eval gate ×2 → summary). Существует в двух эквивалентных реализациях: `scripts/train_full_pipeline.sh` (Bash) и `scripts/train_full_pipeline.ps1` (PowerShell).
- **Pre_Flight_Checker**: подсистема Pipeline_Runner, выполняющая проверки окружения (Python, pip-пакеты, наличие сырых данных, наличие eval CSV) до старта тяжёлых шагов.
- **Dataset_Builder**: существующий скрипт `scripts/ru_metadata_dataset_builder.py`, генерирующий `datasets/ru/processed/ru_tflite_features.csv` из сырых данных.
- **Leak_Free_Trainer**: вызов `scripts/train_kd_distillation.py --leak-free`, обучающий 3-class KD MLP-студента без metadata-фич, недоступных оффлайн.
- **Binary_Trainer**: вызов `scripts/train_binary_model.py`, обучающий бинарную MLP с Platt-калибровкой.
- **Eval_Gate**: вызов `scripts/eval_golden_set.py` против `datasets/ru/eval/cold_eval_600.csv` с порогами precision/recall/FP-rate. Возвращает exit code 0 (passed), 1 (failed), 2 (I/O error).
- **Experimental_Artifacts**: каталог `app/src/main/assets/experimental/`, куда пишутся `spam_model_leak_free.tflite`, `model_card_leak_free.json`, `spam_model_binary.tflite`, `model_card_binary.json`, `eval_leak_free.json`, `eval_binary.json`.
- **Production_Artifacts**: файлы `app/src/main/assets/spam_model.tflite` и `app/src/main/assets/model_card.json`, попадающие в APK.
- **Train_Test_Mismatch**: расхождение между распределением фич, виденных при обучении, и тем, что доступно на устройстве (9 metadata-фич: `reputationScore`, `sourceConfidence`, `reviewsLog`, `negativeRatio`, `searchVolumeLog`, `hasFraudCategory`, `hasTelemarketingCategory`, `inAllowlist`, `inBlacklist`).
- **Cold_Eval_Set**: hold-out CSV `datasets/ru/eval/cold_eval_600.csv`, используемый Eval_Gate с флагом `--cold`.
- **Run_Manifest**: JSON-файл с метаданными прогона (seed, версии скриптов, git SHA, контрольная сумма датасета, времена шагов, exit codes), пишется в `datasets/ru/reports/training/training_run_<timestamp>.json`. Каталог `training/` отделён от общих отчётов специально, чтобы `make clean` (который сносит `datasets/ru/reports/*.json`) не удалял историю прогонов.
- **Promotion_Step**: ручная процедура копирования одной из моделей из Experimental_Artifacts в Production_Artifacts по объективным критериям сравнения.

## Requirements

### Requirement 1: Кросс-платформенный запуск пайплайна

**User Story:** Как разработчик на Windows, я хочу запускать полный пайплайн обучения одной командой в нативной оболочке, чтобы не зависеть от Git Bash или WSL.

#### Acceptance Criteria

1. THE Pipeline_Runner SHALL предоставлять PowerShell-обёртку `scripts/train_full_pipeline.ps1`, эквивалентную по последовательности шагов и по передаваемым аргументам Bash-версии `scripts/train_full_pipeline.sh`.
2. WHEN пользователь запускает `scripts/train_full_pipeline.ps1` на Windows с установленным Python 3.10+ и необходимыми пакетами, THE Pipeline_Runner SHALL завершаться с тем же набором артефактов в Experimental_Artifacts, что и `scripts/train_full_pipeline.sh` на Linux/macOS при идентичном входном датасете и одинаковом seed.
3. WHEN PowerShell-обёртка вызывает Python, THE Pipeline_Runner SHALL использовать команду `python` (а не `python3`), потому что в Windows-дистрибутивах Python `python3` обычно отсутствует.
4. THE Pipeline_Runner SHALL передавать одинаковые значения CLI-флагов в `train_kd_distillation.py`, `train_binary_model.py` и `eval_golden_set.py` независимо от ОС, чтобы результаты обучения были воспроизводимы между платформами.
5. IF в системе отсутствует интерпретатор Python ≥ 3.10, THEN THE Pre_Flight_Checker SHALL прерывать выполнение с exit code ≥ 2 и выводить сообщение, указывающее минимальную требуемую версию.

### Requirement 2: Pre-flight проверки окружения и данных

**User Story:** Как разработчик, я хочу, чтобы пайплайн проверял окружение и данные до начала тяжёлых шагов, чтобы не тратить время на ошибку через 30 минут после старта.

#### Acceptance Criteria

1. WHEN Pipeline_Runner стартует, THE Pre_Flight_Checker SHALL проверять наличие следующих Python-пакетов: `tensorflow`, `catboost`, `scikit-learn`, `numpy`.
2. IF любой из обязательных пакетов отсутствует, THEN THE Pre_Flight_Checker SHALL прерывать выполнение с exit code ≥ 2, указывать имя пакета и предлагать команду установки `pip install tensorflow catboost scikit-learn numpy`.
3. WHEN Pre_Flight_Checker проверяет данные, THE Pre_Flight_Checker SHALL сначала искать `datasets/ru/processed/ru_tflite_features.csv`.
4. IF `datasets/ru/processed/ru_tflite_features.csv` отсутствует, THEN THE Pre_Flight_Checker SHALL запускать Dataset_Builder (`python scripts/ru_metadata_dataset_builder.py`) до перехода к шагу обучения.
5. IF после запуска Dataset_Builder файл `datasets/ru/processed/ru_tflite_features.csv` всё ещё отсутствует или имеет ноль строк данных, THEN THE Pre_Flight_Checker SHALL прерывать выполнение с exit code ≥ 2 и сообщением, какие сырые входы потребуются (`ru_call_features.csv`, `ru_numbers_labeled.csv`, `ru_reputation_raw.csv`).
6. WHEN Pre_Flight_Checker проверяет eval CSV, THE Pre_Flight_Checker SHALL убеждаться, что `datasets/ru/eval/cold_eval_600.csv` существует и содержит минимум 100 строк.
7. IF Cold_Eval_Set отсутствует или содержит меньше 100 строк, THEN THE Pre_Flight_Checker SHALL прерывать выполнение с exit code ≥ 2 и явным сообщением о недостающем eval-наборе, вместо «тихого» пропуска шагов 3 и 4.
8. WHILE Pre_Flight_Checker выполняется, THE Pipeline_Runner SHALL не запускать ни Leak_Free_Trainer, ни Binary_Trainer, ни Eval_Gate.

### Requirement 3: Обучение leak-free 3-class KD модели

**User Story:** Как ML-инженер, я хочу гарантированно получать leak-free KD-модель, чтобы избежать train-test mismatch при оффлайн-инференсе на устройстве.

#### Acceptance Criteria

1. THE Leak_Free_Trainer SHALL вызывать `scripts/train_kd_distillation.py` с флагом `--leak-free`.
2. THE Leak_Free_Trainer SHALL передавать фиксированный seed (по умолчанию 42), параметры архитектуры `--hidden-sizes "128,96,48"`, `--student-epochs 120`, `--student-patience 15`, `--student-batch 128` и весовые коэффициенты классов согласно текущей версии `scripts/train_full_pipeline.sh`.
3. WHEN обучение завершается успешно, THE Leak_Free_Trainer SHALL записывать `app/src/main/assets/experimental/spam_model_leak_free.tflite` и `app/src/main/assets/experimental/model_card_leak_free.json`.
4. IF любая из 9 metadata-фич Train_Test_Mismatch попадает во входной набор фич студента, THEN THE Leak_Free_Trainer SHALL прерывать обучение с ненулевым exit code и сообщением, какие фичи нарушают leak-free контракт.
5. WHERE требуется отладка, THE Leak_Free_Trainer SHALL поддерживать ту же CLI, что и `scripts/train_kd_distillation.py`, без обёртывания флагов в Pipeline_Runner-специфичные имена.

### Requirement 4: Обучение бинарной модели с Platt-калибровкой

**User Story:** Как ML-инженер, я хочу также получать бинарную модель с калиброванными вероятностями, чтобы сравнивать её с 3-class KD по объективным метрикам.

#### Acceptance Criteria

1. THE Binary_Trainer SHALL вызывать `scripts/train_binary_model.py` с флагом `--binary-warn-strategy merge_block`.
2. THE Binary_Trainer SHALL передавать тот же seed, что и Leak_Free_Trainer, чтобы прогон был воспроизводим.
3. WHEN обучение завершается успешно, THE Binary_Trainer SHALL записывать `app/src/main/assets/experimental/spam_model_binary.tflite` и `app/src/main/assets/experimental/model_card_binary.json`, включая параметры Platt-калибровки в model card.
4. IF Binary_Trainer завершается с ненулевым exit code, THEN THE Pipeline_Runner SHALL продолжать выполнение к шагу 3 Eval_Gate для leak-free, но пропускать шаг 4 Eval_Gate для binary и фиксировать факт пропуска в Run_Manifest.

### Requirement 5: Eval gate на cold_eval_600

**User Story:** Как релиз-инженер, я хочу, чтобы каждая модель проходила через golden-set перед промоушеном, чтобы не выкатить регрессию.

#### Acceptance Criteria

1. THE Eval_Gate SHALL вызывать `scripts/eval_golden_set.py` с флагом `--cold` и Cold_Eval_Set в качестве `--golden`.
2. THE Eval_Gate SHALL применять пороги по умолчанию: `--min-block-precision 0.85`, `--min-block-recall 0.55`, `--max-allow-fp-rate 0.20`.
3. THE Pipeline_Runner SHALL принимать опциональные параметры `MinBlockPrecision`, `MinBlockRecall`, `MaxAllowFpRate` и пробрасывать их в Eval_Gate без модификации, чтобы пороги оставались параметризуемыми, но имели задокументированные дефолты.
4. WHEN Eval_Gate завершается, THE Pipeline_Runner SHALL записывать JSON-результат в `app/src/main/assets/experimental/eval_leak_free.json` (для leak-free) или `app/src/main/assets/experimental/eval_binary.json` (для binary).
5. IF Eval_Gate возвращает exit code 1 (gate failed), THEN THE Pipeline_Runner SHALL продолжать остальные шаги и финальный summary, но помечать соответствующую модель как `gate_failed: true` в Run_Manifest.
6. IF Eval_Gate возвращает exit code 2 (I/O или parse error), THEN THE Pipeline_Runner SHALL прерывать выполнение с тем же exit code и пробрасывать stderr Eval_Gate в свой вывод.

### Requirement 6: Защита продовых артефактов

**User Story:** Как поддерживающий продакшен, я хочу, чтобы пайплайн никогда не перезаписывал боевую модель автоматически, чтобы случайный запуск не выкатил неоттестированную версию.

#### Acceptance Criteria

1. THE Pipeline_Runner SHALL писать все обученные модели и eval-результаты только в `app/src/main/assets/experimental/`.
2. THE Pipeline_Runner SHALL NOT модифицировать `app/src/main/assets/spam_model.tflite` и `app/src/main/assets/model_card.json` ни на одном из шагов обучения или eval.
3. WHEN Pipeline_Runner завершает все шаги, THE Pipeline_Runner SHALL выводить в stdout явные команды копирования (PowerShell `Copy-Item` для Windows, `cp` для Linux/macOS) для ручного промоушена каждой из двух моделей.
4. THE Pipeline_Runner SHALL включать в финальный summary сравнение ключевых метрик (block precision, block recall, allow FP rate) обеих моделей из `eval_leak_free.json` и `eval_binary.json`, чтобы инженер мог принять решение о промоушене.
5. WHERE одна из моделей не прошла Eval_Gate, THE Pipeline_Runner SHALL явно помечать её как «не рекомендуется к промоушену» в финальном summary.

### Requirement 7: Воспроизводимость и логирование

**User Story:** Как ML-инженер, я хочу полный след прогона (seed, git SHA, времена, exit codes), чтобы отладить регрессию через неделю.

#### Acceptance Criteria

1. WHEN Pipeline_Runner стартует, THE Pipeline_Runner SHALL фиксировать timestamp начала прогона в формате ISO-8601 (UTC).
2. THE Pipeline_Runner SHALL писать Run_Manifest в `datasets/ru/reports/training/training_run_<timestamp>.json`, где `<timestamp>` — UTC-время старта в формате `YYYYMMDDTHHMMSSZ`. Каталог `training/` намеренно отделён от общего `datasets/ru/reports/`, чтобы `make clean` не сносил историю прогонов.
3. THE Run_Manifest SHALL содержать поля: `started_at`, `finished_at`, `host_os`, `python_version`, `git_sha`, `git_dirty`, `seed`, `dataset_path`, `dataset_sha256`, `dataset_row_count`, `eval_csv_path`, `eval_csv_sha256`, `steps[]` (по одному элементу на шаг с `name`, `started_at`, `finished_at`, `exit_code`, `artifact_paths[]`).
4. IF получение git SHA невозможно (репозиторий не git, git недоступен), THEN THE Pipeline_Runner SHALL записывать `git_sha: "unknown"` и `git_dirty: null` без прерывания.
5. WHEN Pipeline_Runner завершает прогон (успешно или с ошибкой gate), THE Pipeline_Runner SHALL дописывать `finished_at` и финальный exit code в Run_Manifest до выхода.
6. THE Pipeline_Runner SHALL передавать значение `--seed` в Leak_Free_Trainer и Binary_Trainer одинаковым, чтобы два прогона с одинаковым seed на одинаковом датасете давали идентичные TFLite-артефакты по контрольной сумме (в пределах детерминизма TensorFlow на данной платформе).

### Requirement 8: Параметризация порогов и шагов

**User Story:** Как разработчик, я хочу гибко переопределять пороги Eval_Gate и пропускать тяжёлые шаги при отладке, чтобы быстрее итерироваться.

#### Acceptance Criteria

1. THE Pipeline_Runner SHALL поддерживать опциональный параметр `SkipBinary` (PowerShell switch / `--skip-binary` Bash), который пропускает шаги Binary_Trainer и Eval_Gate для binary.
2. THE Pipeline_Runner SHALL поддерживать опциональный параметр `SkipEval` (PowerShell switch / `--skip-eval` Bash), который пропускает оба Eval_Gate-шага.
3. WHERE передан `SkipBinary`, THE Pipeline_Runner SHALL фиксировать пропуск в Run_Manifest и не выводить рекомендации по промоушену binary в summary.
4. WHERE передан `SkipEval`, THE Pipeline_Runner SHALL помечать обе модели как `not_evaluated` в Run_Manifest и явно требовать ручного запуска `eval_golden_set.py` перед любым промоушеном в summary.
5. THE Pipeline_Runner SHALL передавать переопределения порогов (`MinBlockPrecision`, `MinBlockRecall`, `MaxAllowFpRate`) в оба вызова Eval_Gate (для leak-free и для binary) одинаково, чтобы сравнение моделей оставалось честным.

### Requirement 9: Документация процедуры промоушена

**User Story:** Как новый член команды, я хочу понимать критерии промоушена модели в продакшен, чтобы не делать копирование наугад.

#### Acceptance Criteria

1. THE Pipeline_Runner SHALL включать в финальный summary критерии решения о промоушене в человекочитаемом виде: «Промоутить модель X, если её block precision ≥ block precision текущей prod-модели И её block recall не упал более чем на 0.05 относительно prod».
2. THE Pipeline_Runner SHALL читать `app/src/main/assets/model_card.json` (текущий prod), если он существует, и выводить его метрики рядом с метриками новых моделей для прямого сравнения.
3. IF `app/src/main/assets/model_card.json` отсутствует, THEN THE Pipeline_Runner SHALL помечать prod-метрики как `unavailable` и не блокировать промоушен по этому основанию.
4. THE Pipeline_Runner SHALL включать в summary одну строку с командой копирования для каждой модели, готовую к копипасту в текущей оболочке (PowerShell `Copy-Item` под Windows, `cp` под POSIX).
