package com.antispam.blocker.ui.screens

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.antispam.blocker.SpamBlockerApp
import com.antispam.blocker.data.db.entity.FeatureSnapshot
import com.antispam.blocker.data.prefs.FeedbackLearningStore
import com.antispam.blocker.data.repository.BlockListRepository
import com.antispam.blocker.domain.personal.DeviceFeatures
import com.antispam.blocker.domain.personal.DevicePrediction
import com.antispam.blocker.domain.personal.ExplicitLabel
import com.antispam.blocker.domain.personal.FeatureContribution
import com.antispam.blocker.ui.components.GlassCard
import com.antispam.blocker.ui.components.MonoLabelText
import com.antispam.blocker.ui.components.SectionHeader
import com.antispam.blocker.ui.theme.AllowGreen
import com.antispam.blocker.ui.theme.Amber
import com.antispam.blocker.ui.theme.BlockRed
import com.antispam.blocker.ui.theme.Ink
import com.antispam.blocker.ui.theme.InkBorder
import com.antispam.blocker.ui.theme.InkSurface
import com.antispam.blocker.ui.theme.TextPrimary
import com.antispam.blocker.ui.theme.TextSecondary
import com.antispam.blocker.ui.theme.TextTertiary
import com.antispam.blocker.ui.theme.WarnAmber
import com.antispam.blocker.util.PhoneNormalizer
import kotlinx.coroutines.launch
import org.json.JSONException
import org.json.JSONObject
import kotlin.math.max

/**
 * Detail screen for Device_Model explainability (Req 6.2, 6.3, 6.4).
 *
 * Loads a [FeatureSnapshot] by id, runs the current Device_Model on the
 * stored feature vector to obtain [DevicePrediction.topContributions], and
 * renders the top 5 features as a horizontal bar chart on a Compose Canvas.
 * Bars are colored by sign: positive (pushes toward BLOCK) → [BlockRed],
 * negative (pushes toward ALLOW) → [AllowGreen].
 *
 * Three CTA buttons in fixed order at the bottom:
 *   1. «Не спам» → ExplicitLabel.ALLOW + add to global allow list +
 *      personal allowlist; closes the screen.
 *   2. «Подтвердить спам» → ExplicitLabel.BLOCK; closes the screen.
 *   3. «Игнорировать» → just closes (no label produced — Req 4.6).
 *
 * The route arg is the FeatureSnapshot primary key. The trainer however is
 * keyed by callEventId — so for SGD we pass `snapshot.callEventId` when it
 * is linked. If the snapshot was never linked (callEventId == null), we
 * still record the allowlist additions but skip the SGD step rather than
 * fabricating a callEventId.
 */
@Composable
fun ExplainabilityDetailScreen(
    snapshotId: Long,
    onClose: () -> Unit,
) {
    val app = SpamBlockerApp.instance
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var snapshot by remember(snapshotId) { mutableStateOf<FeatureSnapshot?>(null) }
    var prediction by remember(snapshotId) { mutableStateOf<DevicePrediction?>(null) }
    var loadError by remember(snapshotId) { mutableStateOf<String?>(null) }
    var actionInFlight by remember { mutableStateOf(false) }

    LaunchedEffect(snapshotId) {
        val loaded = app.database.featureSnapshotDao().getById(snapshotId)
        snapshot = loaded
        if (loaded == null) {
            loadError = "Запись не найдена"
            return@LaunchedEffect
        }
        if (loaded.featureSchemaVersion != DeviceFeatures.SCHEMA_VERSION) {
            loadError = "Снимок устарел: schema=${loaded.featureSchemaVersion}, " +
                "ожидалась ${DeviceFeatures.SCHEMA_VERSION}"
            return@LaunchedEffect
        }
        val features = parseFeatures(loaded.featuresJson)
        if (features == null) {
            loadError = "Не удалось разобрать features"
            return@LaunchedEffect
        }
        prediction = app.deviceModel.predict(features)
    }

    Scaffold(
        containerColor = Ink,
        contentWindowInsets = WindowInsets(0),
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp, vertical = 24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            SectionHeader(
                eyebrow = "// EXPLAIN",
                title = "Почему это решение",
            )

            HeaderCard(snapshot = snapshot, prediction = prediction)

            when {
                loadError != null -> {
                    GlassCard(modifier = Modifier.fillMaxWidth()) {
                        Column(
                            modifier = Modifier.padding(16.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            MonoLabelText(text = "ERROR", color = BlockRed)
                            Text(
                                text = loadError!!,
                                style = MaterialTheme.typography.bodyMedium,
                                color = TextPrimary,
                            )
                        }
                    }
                }
                prediction == null -> {
                    GlassCard(modifier = Modifier.fillMaxWidth()) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(24.dp),
                            contentAlignment = Alignment.Center,
                        ) {
                            CircularProgressIndicator(color = Amber)
                        }
                    }
                }
                else -> {
                    ContributionsCard(prediction!!.topContributions)
                }
            }

            ActionButtons(
                enabled = !actionInFlight && snapshot != null,
                onNotSpam = {
                    actionInFlight = true
                    val s = snapshot
                    scope.launch {
                        try {
                            s?.callEventId?.let { ce ->
                                app.onlineTrainer.applyExplicitLabel(ce, ExplicitLabel.ALLOW)
                            }
                            val number = s?.normalizedNumber
                            if (!number.isNullOrBlank()) {
                                val blockListRepo = BlockListRepository(
                                    app.database.blockedNumberDao(),
                                    app.database.allowedNumberDao(),
                                    PhoneNormalizer,
                                )
                                blockListRepo.addToAllowList(number)
                                FeedbackLearningStore(context).addNumberToPersonalAllowlist(number)
                            }
                        } finally {
                            onClose()
                        }
                    }
                },
                onConfirmSpam = {
                    actionInFlight = true
                    val s = snapshot
                    scope.launch {
                        try {
                            s?.callEventId?.let { ce ->
                                app.onlineTrainer.applyExplicitLabel(ce, ExplicitLabel.BLOCK)
                            }
                        } finally {
                            onClose()
                        }
                    }
                },
                onIgnore = onClose,
            )
        }
    }
}

@Composable
private fun HeaderCard(
    snapshot: FeatureSnapshot?,
    prediction: DevicePrediction?,
) {
    GlassCard(modifier = Modifier.fillMaxWidth(), accentBorder = true) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            MonoLabelText(text = "INCOMING NUMBER", color = TextTertiary)
            Text(
                text = snapshot?.normalizedNumber?.takeIf { it.isNotBlank() } ?: "Скрытый номер",
                style = MaterialTheme.typography.titleLarge,
                color = TextPrimary,
            )
            val storedProb = snapshot?.deviceProbBlock
            val livePct = prediction?.probBlock?.let { (it * 100f).toInt() }
            val storedPct = storedProb?.let { (it * 100f).toInt() }
            val pct = livePct ?: storedPct
            if (pct != null) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    MonoLabelText(text = "P(BLOCK)", color = TextTertiary)
                    Text(
                        text = "$pct%",
                        style = MaterialTheme.typography.headlineSmall,
                        color = colorForProb(pct),
                    )
                }
            }
        }
    }
}

@Composable
private fun ContributionsCard(top: List<FeatureContribution>) {
    GlassCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            MonoLabelText(text = "TOP-5 FEATURES", color = TextTertiary)
            Text(
                text = "Что повлияло на решение",
                style = MaterialTheme.typography.titleMedium,
                color = TextPrimary,
            )
            Text(
                text = "Вклад = вес × значение. Положительный (красный) тянет к BLOCK, отрицательный (зелёный) — к ALLOW.",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
            )
            Spacer(Modifier.height(4.dp))
            ContributionsChart(top)
        }
    }
}

@Composable
private fun ContributionsChart(top: List<FeatureContribution>) {
    if (top.isEmpty()) {
        Text(
            text = "Нет данных для отображения.",
            style = MaterialTheme.typography.bodySmall,
            color = TextSecondary,
        )
        return
    }
    val maxAbs = max(top.maxOf { kotlin.math.abs(it.signed) }, 1e-6f)

    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        for (c in top) {
            ContributionRow(contribution = c, maxAbs = maxAbs)
        }
    }
}

@Composable
private fun ContributionRow(contribution: FeatureContribution, maxAbs: Float) {
    val color = if (contribution.signed >= 0f) BlockRed else AllowGreen
    val ratio = (kotlin.math.abs(contribution.signed) / maxAbs).coerceIn(0f, 1f)

    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = contribution.name,
                style = MaterialTheme.typography.bodyMedium,
                color = TextPrimary,
            )
            Text(
                text = formatSigned(contribution.signed),
                style = MaterialTheme.typography.labelLarge,
                color = color,
            )
        }
        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .height(10.dp),
        ) {
            val trackColor = InkSurface
            drawRect(
                color = trackColor,
                topLeft = Offset.Zero,
                size = size,
            )
            val barWidth = size.width * ratio
            if (barWidth > 0f) {
                drawRect(
                    color = color,
                    topLeft = Offset.Zero,
                    size = Size(barWidth, size.height),
                )
            }
        }
        Text(
            text = "вес ${formatFloat(contribution.weight)}  •  значение ${formatFloat(contribution.value)}",
            style = MaterialTheme.typography.bodySmall,
            color = TextTertiary,
        )
    }
}

@Composable
private fun ActionButtons(
    enabled: Boolean,
    onNotSpam: () -> Unit,
    onConfirmSpam: () -> Unit,
    onIgnore: () -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Button(
            onClick = onNotSpam,
            enabled = enabled,
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = AllowGreen,
                contentColor = Ink,
                disabledContainerColor = AllowGreen.copy(alpha = 0.4f),
                disabledContentColor = Ink,
            ),
        ) {
            Text("Не спам")
        }
        Button(
            onClick = onConfirmSpam,
            enabled = enabled,
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = BlockRed,
                contentColor = Ink,
                disabledContainerColor = BlockRed.copy(alpha = 0.4f),
                disabledContentColor = Ink,
            ),
        ) {
            Text("Подтвердить спам")
        }
        OutlinedButton(
            onClick = onIgnore,
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            border = BorderStroke(1.dp, InkBorder),
            colors = ButtonDefaults.outlinedButtonColors(contentColor = TextPrimary),
        ) {
            Text("Игнорировать")
        }
    }
}

private fun parseFeatures(json: String): DeviceFeatures? = try {
    val obj = JSONObject(json)
    val out = FloatArray(DeviceFeatures.SIZE)
    var ok = true
    for (i in 0 until DeviceFeatures.SIZE) {
        val name = DeviceFeatures.NAMES[i]
        if (!obj.has(name)) { ok = false; break }
        out[i] = obj.getDouble(name).toFloat()
    }
    if (ok) DeviceFeatures(out) else null
} catch (_: JSONException) {
    null
}

private fun colorForProb(pct: Int): Color = when {
    pct >= 80 -> BlockRed
    pct >= 45 -> WarnAmber
    else -> AllowGreen
}

private fun formatSigned(value: Float): String {
    val sign = if (value >= 0f) "+" else ""
    return "$sign${formatFloat(value)}"
}

private fun formatFloat(value: Float): String =
    String.format(java.util.Locale.US, "%.3f", value)
