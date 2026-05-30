package com.antispam.blocker.domain.scoring

import com.antispam.blocker.domain.personal.DeviceVerdict
import com.antispam.blocker.domain.personal.FeatureContribution

/**
 * Возвращается из [SmartSpamDetector.scoreWithFeatures]. Прокидывает наверх
 * уже посчитанный [CallFeatures]-снимок, чтобы [com.antispam.blocker.domain.tracking.DecisionTracker]
 * не вызывал [FeatureExtractor.extract] второй раз на каждый звонок.
 *
 * [features] = `null` для fast-path вердиктов (защита выключена / экстренный
 * номер / абсолютные allow-/block-списки) — там скоринг не доходит до сборки
 * вектора. Если потребителю всё равно нужен снимок (например, для DecisionTracker),
 * пусть вызовет extract сам.
 *
 * Поля Device_Model (`deviceVerdict`, `deviceProbBlock`, `deviceFeaturesSnapshotJson`,
 * `topContributions`) заполняются только когда [SmartSpamDetector] прошёл хот-пас
 * (т.е. для fast-path вердиктов они тоже `null`) и когда настройка
 * `SettingsStore.personalClassifierEnabled = true`. Эти поля переиспользуются
 * downstream'ом (`FusionDecider`, `SpamWarningNotifier`, `ExplainabilityDetailScreen`,
 * `FeatureSnapshotDao`) — задача 12.1, см. design.md → "Pipeline Integration".
 *
 * Дефолты `null` сохраняют бинарную совместимость с существующими вызывающими.
 */
data class ScoringResult(
    val risk: RiskScore,
    val features: CallFeatures?,
    val deviceVerdict: DeviceVerdict? = null,
    val deviceProbBlock: Float? = null,
    val deviceFeaturesSnapshotJson: String? = null,
    val topContributions: List<FeatureContribution>? = null,
)

