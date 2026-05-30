# Progress Journal — app-category-ml-classifier

Мини-журнал выполненных тасков. Обновляю после каждой подзадачи. Используется как страховка от рассинхрона `tasks.md` и системного meta.json (`task_update completed` периодически валится с EPERM на Windows из-за антивирусного скана `.tmp` → нужно отметить статусы вручную в Kiro UI позже).

## Статус-коды
- ✅ implementation done + tests/build verified by sub-agent
- 🟡 implementation done, status в `tasks.md` ещё не помечен из-за EPERM
- ⏳ pending
- `*` — optional (PBT/integration test)

---

## Wave 0–2 (завершены до текущей сессии)

- ✅ 1.1  — project structure + module skeletons
- ✅ 2.1  — `normalize_label` (Req 1.4)
- ✅ 2.3  — `merge_sources` priority dedup (Req 1.2, 1.3)
- ✅ 2.5  — `map_category`, `Counters`, `is_blank_package` (Req 1.9, 1.10)
- ✅ 3.1  — `write_labeled_csv` (Req 1.5)
- ✅ 3.2  — `write_build_report` (Req 1.11)
- ✅ 3.3  — `validate_corpus_size` (Req 1.6)
- ✅ 4.1  — `build_app_category_splits.py` (Req 1.7, 1.8)
- ✅ 5.1  — Play Store crawler
- ✅ 5.2  — RuStore crawler
- ✅ 5.3  — Huawei AppGallery crawler
- ✅ 7.1  — `CharNGramVocab` + `encode_dataset` (Req 2.2, 2.8)
- ✅ 7.2  — `build_char_cnn_model` (Req 2.2)
- ✅ 7.3  — `check_quality_gates` (Req 2.5)
- ✅ 7.5  — `evaluate` (Req 2.4)
- ✅ 7.6  — `convert_to_tflite_quantized` + size guard (Req 2.6, 2.7, 2.11)
- ✅ 7.8  — Model_Card writer (Req 2.9)
- ✅ 7.9  — `compare_enum_order` + `write_atomic` (Req 2.10)
- ✅ 9.1  — `CharNGramTokenizer` (Req 2.8)
- ✅ 9.3  — `CategoryCache` (Req 3.4, 5.9)
- ✅ 10.1 — `AppCategoryAssetSource` (Req 3.2, 4.5, 4.7)
- ✅ 11.1 — TFLite ctor + shape validation + invalidate epoch (Req 3.1, 3.8, 7.3)
- ✅ 12.1 — `tfliteAppCategoryEnabled` flag в SettingsStore (Req 3.6)
- ✅ 13.2 — release manifest builder (Req 4.1)

## Wave 3 (текущая сессия — 17.05.2026)

- 🟡 3.4  — `build_app_category_dataset.py` main() wired
   - Дописан `import argparse`, прогнаны юнит-тесты (10 passed), smoke-run на 4 синтетических CSV: bootstrap-priority, blank-drop, unknown-category, oversize-label, exit 1 на 200k/5k floor.
   - Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.9, 1.10, 1.11
- 🟡 7.10 — `train_app_category_classifier.py` main() wired
   - Реализованы `set_random_seed`, `_CsvFrame`, `load_splits`, `parse_args(--train/--val/--test/--seed/--output/--vocab/--card)`, `main()` с последовательностью load → vocab.build → AdamW+CosineDecay → fit → evaluate → quality gate (exit 1) → tflite convert + size guard (exit 2) → enum check (exit 3) → atomic write × 3 → exit 0.
   - На fail-путях артефакты не пишутся (atomicity guarantee).
   - 96 passed, 29 skipped (TF-skips); 28 passed в sibling-модулях.
   - Requirements: 2.1, 2.3, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12
- 🟡 11.3 — `TFLiteAppCategoryClassifier.classify()` + `softmaxTop1Confidence`
   - Уже было реализовано в 11.1 (cache hit return → infer → argmax → defensive confidence gate → exception fallback с `Log.w(TAG, "tflite inference threw", t)` без значений).
   - `:app:compileDebugKotlin` exit 0; getDiagnostics clean.
   - Requirements: 3.3, 3.4, 3.5, 3.11, 3.12, 5.1–5.5
- 🟡 13.1 — RemoteUpdateWorker `ALLOWED_FILES` + invalidate dispatch для трёх app-category ассетов
   - Уже было реализовано; verified `:app:compileDebugKotlin` exit 0.
   - Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6

> **EPERM-блокер**: `task_update status=completed` для всех 4 задач Wave 3 валится на rename `.tmp → meta.json` (Windows Defender / антивирус). Пользователь подтвердил, что отметит вручную в Kiro UI позже.

## Wave 4 (следующая — после ручной отметки Wave 3)

- ⏳ 12.2 — расширить `AppCategoryClassifierFactory` TFLite-веткой (Req 3.6, 3.7, 3.9)
- ⏳ 14.1 — bundle initial APK assets (Req 4.7)
- ⏳ 2.6*  — PBT counters (Property 13)
- ⏳ 7.4*  — PBT quality gate (Property 14)
- ⏳ 7.7*  — PBT TFLite size guard cleanup (Property 15)
- ⏳ 7.11* — PBT pipeline determinism (Property 12)
- ⏳ 9.4*  — PBT cache idempotence + LRU (Property 6)
- ⏳ 11.2* — PBT TFLite output dimension (Property 3)
- ⏳ 11.4* — PBT confidence-gated fallback (Property 4)
- ⏳ 11.5* — PBT no-PII-in-logs (Property 8)

## Wave 5 (финальная)

- ⏳ 12.3* — PBT factory selection table (Property 7)
- ⏳ 15.1  — Privacy Transparency screen (Req 3.13)
- ⏳ 16.1* — enum-order parity smoke test (Property 1)
- ⏳ 16.3* — DeviceFeatureExtractor regression (Property 16)
- ⏳ 16.4* — sensitive categories integration (Req 7.7)

## Чекпойнты

- ⏳ 6.   — checkpoint after dataset pipeline
- ⏳ 8.   — checkpoint after training pipeline
- ⏳ 17.  — final checkpoint

## Уже завершённые опциональные тесты

- ✅ 16.2* — пометка статуса в meta показывает `not_started`, но фактически unclear; перепроверить при wave 5 запуске.

---

_Last update: после Wave 3 (3.4 / 7.10 / 11.3 / 13.1)._
