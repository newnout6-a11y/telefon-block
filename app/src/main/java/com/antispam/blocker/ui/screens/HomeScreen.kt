package com.antispam.blocker.ui.screens

import android.app.AppOpsManager
import android.content.Intent
import android.os.Build
import android.os.Process
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Shield
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.antispam.blocker.SpamBlockerApp
import com.antispam.blocker.data.prefs.FeedbackLearningStore
import com.antispam.blocker.ui.components.GlassCard
import com.antispam.blocker.ui.components.HeroBackground
import com.antispam.blocker.ui.components.MetricCounter
import com.antispam.blocker.ui.components.MonoLabelText
import com.antispam.blocker.ui.components.StatusPill
import com.antispam.blocker.ui.theme.*
import com.antispam.blocker.util.RoleManagerHelper
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.util.Calendar

@Preview(showBackground = true, backgroundColor = 0xFF0A0A0B)
@Composable
fun HomeScreenPreview() {
        SpamBlockerTheme {
        HomeScreen(
            protectionEnabled = true,
            isRoleHeld = true,
            canDrawOverlay = true,
            blockedToday = 12,
            warnedToday = 5,
            allowedToday = 45,
            blockedWeek = 31,
            warnedWeek = 14,
            allowedWeek = 203,
            usageStatsGranted = true,
            notifListenerGranted = true,
            onProtectionChanged = {},
            onRequestOverlay = {},
            onRequestRole = {}
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    protectionEnabled: Boolean,
    isRoleHeld: Boolean,
    canDrawOverlay: Boolean,
    blockedToday: Int,
    warnedToday: Int,
    allowedToday: Int = 0,
    blockedWeek: Int = 0,
    warnedWeek: Int = 0,
    allowedWeek: Int = 0,
    usageStatsGranted: Boolean = true,
    notifListenerGranted: Boolean = true,
    onProtectionChanged: (Boolean) -> Unit,
    onRequestOverlay: () -> Unit,
    onRequestRole: () -> Unit,
    prefixOverrides: List<FeedbackLearningStore.PrefixOverrideCandidate> = emptyList(),
    onAcceptPrefixOverride: (String) -> Unit = {},
    onDismissPrefixOverride: (String) -> Unit = {}
) {
    val isActive = protectionEnabled && isRoleHeld

    val statusColor by animateColorAsState(
        targetValue = when {
            isActive -> Amber
            !isRoleHeld -> BlockRed
            else -> TextTertiary
        },
        animationSpec = tween(500),
        label = "status_color"
    )

    val infinite = rememberInfiniteTransition(label = "hero_pulse")
    val pulse by infinite.animateFloat(
        initialValue = 1f,
        targetValue = if (isActive) 1.08f else 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(2200, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse"
    )

    Scaffold(
        containerColor = Ink,
        contentWindowInsets = WindowInsets(0)
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp, vertical = 24.dp),
            verticalArrangement = Arrangement.spacedBy(20.dp)
        ) {
            // --- Top brand row ---
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    MonoLabelText(text = "SENTINEL • v3.0", color = TextTertiary)
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = "Antispam",
                        style = MaterialTheme.typography.headlineLarge,
                        color = TextPrimary
                    )
                }
                StatusPill(
                    text = when {
                        isActive -> "active"
                        !isRoleHeld -> "unassigned"
                        else -> "paused"
                    },
                    color = statusColor
                )
            }

            // --- Hero status card ---
            HeroBackground(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 320.dp)
            ) {
                if (isActive) {
                    Box(
                        modifier = Modifier
                            .align(Alignment.Center)
                            .size(260.dp)
                            .scale(pulse)
                            .background(
                                Brush.radialGradient(
                                    0f to AmberGlow,
                                    1f to Color.Transparent
                                ),
                                shape = CircleShape
                            )
                    )
                }

                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(24.dp),
                    verticalArrangement = Arrangement.SpaceBetween
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        MonoLabelText(text = "// CALL SCREENING", color = TextTertiary)
                        Switch(
                            checked = protectionEnabled,
                            onCheckedChange = onProtectionChanged,
                            colors = SwitchDefaults.colors(
                                checkedThumbColor = TextOnAccent,
                                checkedTrackColor = Amber,
                                uncheckedThumbColor = TextSecondary,
                                uncheckedTrackColor = InkSurface,
                                uncheckedBorderColor = InkBorder
                            )
                        )
                    }

                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 12.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Box(
                            modifier = Modifier
                                .size(104.dp)
                                .clip(CircleShape)
                                .background(statusColor.copy(alpha = 0.12f))
                                .border(1.dp, statusColor.copy(alpha = 0.4f), CircleShape),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Default.Shield,
                                contentDescription = null,
                                tint = statusColor,
                                modifier = Modifier.size(48.dp)
                            )
                        }
                    }

                    Column {
                        Text(
                            text = when {
                                isActive -> "Защита активна"
                                !isRoleHeld -> "Не назначен фильтром"
                                else -> "Защита приостановлена"
                            },
                            style = MaterialTheme.typography.headlineMedium,
                            color = TextPrimary
                        )
                        Spacer(Modifier.height(6.dp))
                        Text(
                            text = when {
                                isActive -> "Входящие звонки анализируются нейросетью и правилами в реальном времени."
                                !isRoleHeld -> "Назначьте приложение фильтром в системных настройках Android."
                                else -> "Включите защиту, чтобы начать блокировать спам-звонки."
                            },
                            style = MaterialTheme.typography.bodyMedium,
                            color = TextSecondary
                        )
                    }
                }
            }

            // --- Stats grid (24h) ---
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                // All three cards built identically — only color and data differ.
                for ((value, label, accent) in listOf(
                    Triple(blockedToday, "БЛОК / 24Ч", BlockRed),
                    Triple(warnedToday, "WARN / 24Ч", WarnAmber),
                    Triple(allowedToday, "ОК / 24Ч", AllowGreen),
                )) {
                    Box(Modifier.weight(1f)) {
                        Surface(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(20.dp),
                            color = InkElevated,
                            border = BorderStroke(1.dp, InkBorder),
                        ) {
                            Column(
                                modifier = Modifier.fillMaxWidth().padding(16.dp),
                                verticalArrangement = Arrangement.spacedBy(8.dp),
                            ) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Box(Modifier.size(6.dp).clip(CircleShape).background(accent))
                                    Spacer(Modifier.width(6.dp))
                                    Text(label.uppercase(), style = MaterialTheme.typography.labelSmall, color = TextTertiary, modifier = Modifier.weight(1f))
                                }
                                Text(
                                    text = value.toString().padStart(2, '0'),
                                    style = MaterialTheme.typography.displaySmall,
                                    color = accent,
                                    modifier = Modifier.fillMaxWidth(),
                                )
                            }
                        }
                    }
                }
            }

            // --- Weekly stats ---
            GlassCard(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    MonoLabelText(text = "// ЗА НЕДЕЛЮ", color = TextTertiary)
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        StatLine("Заблокировано", blockedWeek, BlockRed)
                        StatLine("Подозрения", warnedWeek, WarnAmber)
                        StatLine("Пропущено", allowedWeek, AllowGreen)
                    }
                }
            }

            // --- Permission banners ---
            if (!usageStatsGranted) {
                val ctx = LocalContext.current
                GlassCard(modifier = Modifier.fillMaxWidth(), accentBorder = true) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Warning, null, tint = WarnAmber, modifier = Modifier.size(20.dp))
                            Text("Включите доступ к статистике", style = MaterialTheme.typography.titleSmall, color = TextPrimary)
                        }
                        Text(
                            "Без UsageStats модель не видит недавние приложения → фичи recent_*_30m всегда нулевые → точность ниже.",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSecondary
                        )
                        OutlinedButton(
                            onClick = {
                                try {
                                    ctx.startActivity(Intent(android.provider.Settings.ACTION_USAGE_ACCESS_SETTINGS).apply {
                                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                                    })
                                } catch (_: Throwable) {}
                            },
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(12.dp),
                            border = BorderStroke(1.dp, InkBorder),
                            colors = ButtonDefaults.outlinedButtonColors(contentColor = TextPrimary)
                        ) {
                            Text("Открыть настройки Usage Access")
                        }
                    }
                }
            }

            if (!notifListenerGranted) {
                val ctx = LocalContext.current
                GlassCard(modifier = Modifier.fillMaxWidth(), accentBorder = true) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Warning, null, tint = WarnAmber, modifier = Modifier.size(20.dp))
                            Text("Включите слушатель уведомлений", style = MaterialTheme.typography.titleSmall, color = TextPrimary)
                        }
                        Text(
                            "Без Notification Listener фичи notif_*_recent_10m всегда нулевые → модель не видит банки/маркетплейсы.",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSecondary
                        )
                        OutlinedButton(
                            onClick = {
                                try {
                                    ctx.startActivity(Intent(android.provider.Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS).apply {
                                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                                    })
                                } catch (_: Throwable) {}
                            },
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(12.dp),
                            border = BorderStroke(1.dp, InkBorder),
                            colors = ButtonDefaults.outlinedButtonColors(contentColor = TextPrimary)
                        ) {
                            Text("Открыть настройки слушателя")
                        }
                    }
                }
            }

            // --- Per-prefix override suggestions ---
            for (candidate in prefixOverrides) {
                PrefixOverrideCard(
                    candidate = candidate,
                    onAccept = { onAcceptPrefixOverride(candidate.prefix) },
                    onDismiss = { onDismissPrefixOverride(candidate.prefix) }
                )
            }

            // --- Warning card if role is not held ---
            if (!isRoleHeld) {
                GlassCard(
                    modifier = Modifier.fillMaxWidth(),
                    accentBorder = true
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                            Icon(
                                imageVector = Icons.Default.Warning,
                                contentDescription = null,
                                tint = Amber,
                                modifier = Modifier.size(22.dp)
                            )
                            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                                Text(
                                    text = "Требуется действие",
                                    style = MaterialTheme.typography.titleSmall,
                                    color = TextPrimary
                                )
                                Text(
                                    text = "Назначьте Antispam Sentinel приложением для фильтра и идентификации звонков. Без этой роли Android не передаёт нам входящие — защита не работает.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = TextSecondary
                                )
                            }
                        }
                        Button(
                            onClick = onRequestRole,
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Amber,
                                contentColor = TextOnAccent
                            ),
                            shape = androidx.compose.foundation.shape.RoundedCornerShape(12.dp)
                        ) {
                            Text("Назначить приложением по умолчанию")
                        }
                    }
                }
            }

            // --- Warning card if overlay permission is missing ---
            if (!canDrawOverlay) {
                GlassCard(
                    modifier = Modifier.fillMaxWidth(),
                    accentBorder = true
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                            Icon(
                                imageVector = Icons.Default.Warning,
                                contentDescription = null,
                                tint = WarnAmber,
                                modifier = Modifier.size(22.dp)
                            )
                            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                                Text(
                                    text = "Включите плашку поверх звонка",
                                    style = MaterialTheme.typography.titleSmall,
                                    color = TextPrimary
                                )
                                Text(
                                    text = "Без этого разрешения предупреждение о подозрительном звонке не появится поверх экрана входящего вызова.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = TextSecondary
                                )
                            }
                        }
                        Button(
                            onClick = onRequestOverlay,
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Amber,
                                contentColor = TextOnAccent
                            ),
                            shape = androidx.compose.foundation.shape.RoundedCornerShape(12.dp)
                        ) {
                            Text("Открыть настройки")
                        }
                    }
                }
            }

            // --- Footer tagline ---
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.Center
            ) {
                MonoLabelText(
                    text = "OFFLINE FIRST  •  RU",
                    color = TextTertiary
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen() {
    val context = LocalContext.current
    val app = SpamBlockerApp.instance
    val settings = app.settingsStore
    val callLogRepo = remember {
        com.antispam.blocker.data.repository.CallLogRepository(app.database.callRecordDao())
    }

    val protectionEnabled by settings.protectionEnabled.collectAsState(initial = true)

    // Состояние двух системных разрешений (call screening role + overlay) обновляется
    // при возврате из системных настроек — оба слушают ON_RESUME активити.
    var isRoleHeld by remember {
        mutableStateOf(RoleManagerHelper.isCallScreeningRoleHeld(context))
    }
    var canDrawOverlay by remember {
        mutableStateOf(android.provider.Settings.canDrawOverlays(context))
    }
    androidx.compose.runtime.DisposableEffect(Unit) {
        val activity = context as? android.app.Activity
        val lifecycle = (activity as? androidx.lifecycle.LifecycleOwner)?.lifecycle
        val observer = androidx.lifecycle.LifecycleEventObserver { _, event ->
            if (event == androidx.lifecycle.Lifecycle.Event.ON_RESUME) {
                canDrawOverlay = android.provider.Settings.canDrawOverlays(context)
                isRoleHeld = RoleManagerHelper.isCallScreeningRoleHeld(context)
            }
        }
        lifecycle?.addObserver(observer)
        onDispose { lifecycle?.removeObserver(observer) }
    }

    val startOfDay = remember {
        val cal = Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
        cal.timeInMillis
    }

    val blockedToday by callLogRepo.blockedCountSince(startOfDay).collectAsState(initial = 0)
    val warnedToday by callLogRepo.warnedCountSince(startOfDay).collectAsState(initial = 0)
    val allowedToday by callLogRepo.allowedCountSince(startOfDay).collectAsState(initial = 0)

    val startOfWeek = remember {
        val cal = Calendar.getInstance().apply {
            set(Calendar.DAY_OF_WEEK, Calendar.MONDAY)
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
        cal.timeInMillis
    }
    val blockedWeek by callLogRepo.blockedCountSince(startOfWeek).collectAsState(initial = 0)
    val warnedWeek by callLogRepo.warnedCountSince(startOfWeek).collectAsState(initial = 0)
    val allowedWeek by callLogRepo.allowedCountSince(startOfWeek).collectAsState(initial = 0)

    // UsageStats permission check
    val usageStatsGranted = remember {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val appOps = context.getSystemService(android.content.Context.APP_OPS_SERVICE) as AppOpsManager
            try {
                val mode = appOps.checkOpNoThrow(
                    AppOpsManager.OPSTR_GET_USAGE_STATS,
                    Process.myUid(),
                    context.packageName,
                )
                mode == AppOpsManager.MODE_ALLOWED
            } catch (_: Throwable) { false }
        } else true
    }

    // Notification Listener check
    val notifListenerGranted = remember {
        try {
            val flat = Settings.Secure.getString(context.contentResolver, "enabled_notification_listeners") ?: ""
            flat.contains(context.packageName)
        } catch (_: Throwable) { false }
    }

    // Per-prefix override-кандидаты (≥5 «не-спам»-отметок по одному префиксу
    // за 30 дней) — отрисовываются чипами над warning-картами и предлагают
    // занести префикс в персональный allowlist одним тапом.
    val feedbackStore = remember { FeedbackLearningStore(context) }
    val prefixOverrides by feedbackStore.prefixOverrideCandidates()
        .collectAsState(initial = emptyList())

    val scope = rememberCoroutineScope()

    val roleLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartActivityForResult(),
    ) { _ ->
        // По возвращении из системного диалога обновим состояние через
        // ON_RESUME-обработчик уже выше. Колбэк нужен только чтобы
        // система знала, что мы дождались результата.
        isRoleHeld = RoleManagerHelper.isCallScreeningRoleHeld(context)
    }

    HomeScreen(
        protectionEnabled = protectionEnabled,
        isRoleHeld = isRoleHeld,
        canDrawOverlay = canDrawOverlay,
        blockedToday = blockedToday,
        warnedToday = warnedToday,
        allowedToday = allowedToday,
        blockedWeek = blockedWeek,
        warnedWeek = warnedWeek,
        allowedWeek = allowedWeek,
        usageStatsGranted = usageStatsGranted,
        notifListenerGranted = notifListenerGranted,
        onProtectionChanged = { enabled ->
            scope.launch { settings.set("protection_enabled", enabled) }
        },
        onRequestOverlay = {
            val intent = android.content.Intent(
                android.provider.Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                android.net.Uri.parse("package:${context.packageName}")
            ).apply {
                addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            try {
                context.startActivity(intent)
            } catch (_: Exception) {
                // fallback — общие настройки оверлея
                context.startActivity(
                    android.content.Intent(android.provider.Settings.ACTION_MANAGE_OVERLAY_PERMISSION)
                        .addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
                )
            }
        },
        onRequestRole = {
            // RoleManagerHelper.requestCallScreeningRole закрывает SDK-гэп:
            // на API < 29 RoleManager отсутствует, и тогда вместо системного
            // диалога просто открываем общие настройки приложений по умолчанию.
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                RoleManagerHelper.requestCallScreeningRole(roleLauncher, context)
            } else {
                try {
                    context.startActivity(
                        android.content.Intent(android.provider.Settings.ACTION_MANAGE_DEFAULT_APPS_SETTINGS)
                            .addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
                    )
                } catch (_: Throwable) { /* no-op */ }
            }
        },
        prefixOverrides = prefixOverrides,
        onAcceptPrefixOverride = { prefix ->
            scope.launch { feedbackStore.addPrefixToAllowlist(prefix) }
        },
        onDismissPrefixOverride = { prefix ->
            scope.launch { feedbackStore.dismissPrefixOverride(prefix) }
        }
    )
}

/**
 * Чип-предложение «занести префикс +7XXXX в персональный allowlist».
 * Отрисовывается на Home, когда юзер ≥5 раз за последние 30 дней нажал
 * «не спам» по номерам с одного DEF-кодового префикса. Кнопки —
 * принять (добавить в [FeedbackLearningStore.prefixAllowlist]) или
 * отказаться (dismiss-flag, чип больше не появится).
 */
@Composable
private fun PrefixOverrideCard(
    candidate: FeedbackLearningStore.PrefixOverrideCandidate,
    onAccept: () -> Unit,
    onDismiss: () -> Unit
) {
    GlassCard(
        modifier = Modifier.fillMaxWidth(),
        accentBorder = true
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            MonoLabelText(text = "// PREFIX OVERRIDE", color = Amber)
            Text(
                text = "Префикс ${candidate.prefix} — ${candidate.count} раз вы отмечали «не спам»",
                style = MaterialTheme.typography.titleSmall,
                color = TextPrimary
            )
            Text(
                text = "Занести в персональный allowlist? Будущие звонки с номеров этой группы перестанут попадать под подозрение.",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                OutlinedButton(
                    onClick = onDismiss,
                    modifier = Modifier.weight(1f),
                    shape = androidx.compose.foundation.shape.RoundedCornerShape(12.dp),
                    border = androidx.compose.foundation.BorderStroke(1.dp, InkBorder),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = TextSecondary)
                ) {
                    Text("Не сейчас")
                }
                Button(
                    onClick = onAccept,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Amber,
                        contentColor = TextOnAccent
                    ),
                    shape = androidx.compose.foundation.shape.RoundedCornerShape(12.dp)
                ) {
                    Text("Занести")
                }
            }
        }
    }
}

@Composable
private fun StatLine(label: String, value: Int, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value.toString(), style = MaterialTheme.typography.titleLarge, color = color)
        Text(label, style = MaterialTheme.typography.labelSmall, color = TextTertiary)
    }
}

