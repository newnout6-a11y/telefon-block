package com.antispam.blocker.ui.screens

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Analytics
import androidx.compose.material.icons.rounded.AutoAwesome
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.antispam.blocker.SpamBlockerApp
import com.antispam.blocker.data.cache.ContactNameLookup
import com.antispam.blocker.data.db.entity.DecisionRecord
import com.antispam.blocker.domain.tracking.DecisionTracker
import com.antispam.blocker.domain.tracking.TrackingStats
import com.antispam.blocker.ui.components.GlassCard
import com.antispam.blocker.ui.components.MonoLabelText
import com.antispam.blocker.ui.components.SectionHeader
import com.antispam.blocker.ui.theme.*
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import org.json.JSONArray
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelDebugScreen() {
    val app = SpamBlockerApp.instance
    val tracker = remember {
        DecisionTracker(app.database.decisionRecordDao()) { app.modelVersion }
    }
    val scope = rememberCoroutineScope()
    val modelCard = app.modelCard
    var records by remember { mutableStateOf<List<DecisionRecord>>(emptyList()) }
    var stats by remember { mutableStateOf<TrackingStats?>(null) }

    LaunchedEffect(Unit) {
        tracker.observeRecent(50).collectLatest {
            records = it
            stats = tracker.stats()
        }
    }

    Scaffold(
        containerColor = Ink,
        contentWindowInsets = WindowInsets(0)
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            contentPadding = PaddingValues(horizontal = 20.dp, vertical = 24.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            item {
                SectionHeader(eyebrow = "// AI OBSERVABILITY", title = "ИИ и отслеживание")
                Spacer(Modifier.height(4.dp))
            }

            // Server Model card
            item {
                GlassCard(modifier = Modifier.fillMaxWidth(), accentBorder = true) {
                    Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Icon(Icons.Rounded.AutoAwesome, null, tint = Amber)
                            Text("Server Model", style = MaterialTheme.typography.titleMedium, color = TextPrimary)
                        }
                        if (modelCard == null) {
                            Text("Card не найден — работает Rules-fallback.", style = MaterialTheme.typography.bodySmall, color = WarnAmber)
                        } else {
                            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                MetricChip("VERSION", modelCard.version, Modifier.weight(1f))
                                MetricChip("FEATURES", modelCard.featureCount.toString(), Modifier.weight(1f))
                            }
                            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                MetricChip("ROWS", modelCard.rows.toString(), Modifier.weight(1f))
                                MetricChip("BLOCK PREC", "${(modelCard.blockPrecision * 100).toInt()}%", Modifier.weight(1f))
                            }
                            if (modelCard.rocAuc != null) {
                                MetricChip("ROC-AUC", "%.3f".format(modelCard.rocAuc), Modifier.fillMaxWidth())
                            }
                        }
                    }
                }
            }

            // Accuracy dashboard
            item {
                val s = stats
                GlassCard(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Icon(Icons.Rounded.Analytics, null, tint = Amber)
                            Text("Дашборд", style = MaterialTheme.typography.titleMedium, color = TextPrimary)
                        }

                        // Verdict distribution
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            DashboardCard("BLOCK", s?.blockCount ?: 0, BlockRed, Modifier.weight(1f))
                            DashboardCard("WARN", s?.warnCount ?: 0, WarnAmber, Modifier.weight(1f))
                            DashboardCard("ALLOW", s?.allowCount ?: 0, AllowGreen, Modifier.weight(1f))
                        }

                        Divider(color = InkBorder)

                        // Accuracy row
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text("Совпадение с фидбеком", style = MaterialTheme.typography.bodyMedium, color = TextSecondary)
                            val agreeRate = s?.agreementRate
                            Text(
                                if (agreeRate != null) "${(agreeRate * 100).toInt()}%" else "—",
                                style = MaterialTheme.typography.titleLarge,
                                color = when {
                                    agreeRate == null -> TextTertiary
                                    agreeRate >= 0.8f -> AllowGreen
                                    agreeRate >= 0.5f -> WarnAmber
                                    else -> BlockRed
                                }
                            )
                        }
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text("Всего решений", style = MaterialTheme.typography.bodyMedium, color = TextSecondary)
                            Text((s?.total ?: 0).toString(), style = MaterialTheme.typography.bodyLarge, color = TextPrimary)
                        }
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text("С фидбеком", style = MaterialTheme.typography.bodyMedium, color = TextSecondary)
                            Text((s?.feedbackCount ?: 0).toString(), style = MaterialTheme.typography.bodyLarge, color = TextPrimary)
                        }
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text("Model / Rules", style = MaterialTheme.typography.bodyMedium, color = TextSecondary)
                            Text(
                                "${s?.modelDecisions ?: 0} / ${s?.ruleDecisions ?: 0}",
                                style = MaterialTheme.typography.bodyLarge,
                                color = TextPrimary
                            )
                        }

                        if (records.isNotEmpty()) {
                            Divider(color = InkBorder)
                            MonoLabelText(text = "ТРЕНД ВЕРОЯТНОСТЕЙ", color = TextTertiary)
                            VerdictSparkline(records = records, modifier = Modifier.fillMaxWidth().height(48.dp))
                            Row(horizontalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
                                SparklineLegend("BLOCK", BlockRed)
                                SparklineLegend("WARN", WarnAmber)
                                SparklineLegend("ALLOW", AllowGreen)
                            }
                        }
                    }
                }
            }

            // Last decisions
            item { MonoLabelText(text = "ПОСЛЕДНИЕ 20 РЕШЕНИЙ", color = TextTertiary) }

            if (records.isEmpty()) {
                item {
                    GlassCard(modifier = Modifier.fillMaxWidth()) {
                        Text("Пока нет записанных решений.", modifier = Modifier.padding(16.dp), style = MaterialTheme.typography.bodyMedium, color = TextSecondary)
                    }
                }
            } else {
                items(records.take(20), key = { it.id }) { record ->
                    DecisionCard(record)
                }
            }
        }
    }
}

@Composable
private fun DashboardCard(label: String, value: Int, color: Color, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .background(InkSurface, RoundedCornerShape(12.dp))
            .padding(horizontal = 12.dp, vertical = 10.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(value.toString(), style = MaterialTheme.typography.headlineSmall, color = color, fontWeight = FontWeight.Bold)
        Text(label, style = MaterialTheme.typography.labelSmall, color = TextTertiary)
    }
}

@Composable
private fun MetricChip(label: String, value: String, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .background(InkSurface, RoundedCornerShape(12.dp))
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        MonoLabelText(text = label, color = TextTertiary)
        Text(value, style = MaterialTheme.typography.titleMedium, color = TextPrimary, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun DecisionCard(record: DecisionRecord) {
    val verdictColor = when (record.verdict) {
        "BLOCK" -> BlockRed
        "WARN" -> WarnAmber
        else -> AllowGreen
    }
    GlassCard(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(record.verdict, color = verdictColor, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                    val sourceLabel = mapSource(record.source)
                    Text(sourceLabel, style = MaterialTheme.typography.labelSmall, color = TextTertiary)
                    val fusionTag = extractFusionTag(record.reasonsJson)
                    if (fusionTag != null) {
                        Text(fusionTag, style = MaterialTheme.typography.labelSmall, color = Amber)
                    }
                }
                Text(formatTime(record.timestamp), style = MaterialTheme.typography.labelSmall, color = TextTertiary)
            }

            val context = LocalContext.current
            val displayName = remember(record.normalizedNumber, record.rawNumber) {
                record.normalizedNumber?.let { ContactNameLookup.resolveOrNull(context, it) }
                    ?: record.rawNumber ?: record.normalizedNumber ?: "Скрытый номер"
            }
            Text(displayName, style = MaterialTheme.typography.bodyMedium, color = TextPrimary, maxLines = 1)

            ProbabilityBars(record)

            val reasons = parseJsonArray(record.reasonsJson)
            if (reasons.isNotEmpty()) {
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    reasons.take(4).forEach { reason ->
                        Text(
                            "• $reason",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSecondary,
                            maxLines = 2,
                            softWrap = true,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ProbabilityBars(record: DecisionRecord) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        ProbBar("ALLOW", record.modelAllowProb, AllowGreen)
        ProbBar("WARN", record.modelWarnProb, WarnAmber)
        ProbBar("BLOCK", record.modelBlockProb, BlockRed)
    }
}

@Composable
private fun ProbBar(label: String, value: Float, color: Color) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        MonoLabelText(text = label, color = TextTertiary, modifier = Modifier.width(48.dp))
        LinearProgressIndicator(
            progress = value.coerceIn(0f, 1f),
            modifier = Modifier
                .weight(1f)
                .height(8.dp),
            color = color,
            trackColor = InkBorder
        )
        Text("${(value * 100).toInt()}%", color = TextSecondary, style = MaterialTheme.typography.bodySmall)
    }
}

private fun parseJsonArray(json: String): List<String> {
    return runCatching {
        val arr = JSONArray(json)
        List(arr.length()) { idx -> arr.optString(idx) }
    }.getOrDefault(emptyList())
}

private fun formatTime(ts: Long): String {
    return SimpleDateFormat("HH:mm:ss dd.MM", Locale.getDefault()).format(Date(ts))
}

@Composable
private fun VerdictSparkline(records: List<DecisionRecord>, modifier: Modifier = Modifier) {
    val blockProbs = records.map { it.modelBlockProb }
    val warnProbs = records.map { it.modelWarnProb }
    val allowProbs = records.map { it.modelAllowProb }

    Canvas(modifier = modifier) {
        val w = size.width
        val h = size.height
        if (blockProbs.size < 2) return@Canvas

        val drawSparkline: (List<Float>, Color) -> Unit = { data, color ->
            val step = w / (data.size - 1).coerceAtLeast(1)
            for (i in 0 until data.size - 1) {
                val x1 = i * step
                val x2 = (i + 1) * step
                val y1 = h - data[i].coerceIn(0f, 1f) * h
                val y2 = h - data[i + 1].coerceIn(0f, 1f) * h
                drawLine(
                    color = color,
                    start = Offset(x1, y1),
                    end = Offset(x2, y2),
                    strokeWidth = 2.dp.toPx(),
                    cap = StrokeCap.Round
                )
            }
        }

        drawSparkline(allowProbs, AllowGreen.copy(alpha = 0.5f))
        drawSparkline(warnProbs, WarnAmber.copy(alpha = 0.7f))
        drawSparkline(blockProbs, BlockRed.copy(alpha = 0.9f))
    }
}

@Composable
private fun SparklineLegend(label: String, color: Color) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        Canvas(modifier = Modifier.size(8.dp)) {
            drawCircle(color = color)
        }
        MonoLabelText(text = label, color = TextSecondary)
    }
}

private fun mapSource(source: String): String = when (source) {
    "server_model" -> "модель"
    "rule_engine" -> "правила"
    "blacklist" -> "ч.список"
    "allowlist" -> "б.список"
    "contact" -> "контакт"
    "disabled" -> "выкл"
    "emergency_whitelist" -> "экстрен"
    "cloud_model", "tflite_model" -> "модель"
    else -> source
}

private fun extractFusionTag(reasonsJson: String): String? {
    val reasons = parseJsonArray(reasonsJson)
    val fusionLine = reasons.firstOrNull { it.startsWith("Корректировка:") }
    return fusionLine?.removePrefix("Корректировка:")
}
