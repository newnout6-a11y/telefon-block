package com.antispam.blocker.ui.screens

import android.Manifest
import android.app.AppOpsManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Process
import android.provider.Settings
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Shield
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import com.antispam.blocker.SpamBlockerApp
import com.antispam.blocker.data.assets.CsvSpamImporter
import com.antispam.blocker.data.repository.BlockListRepository
import com.antispam.blocker.data.worker.RemoteUpdateWorker
import com.antispam.blocker.domain.model.ModelCard
import com.antispam.blocker.ui.components.GlassCard
import com.antispam.blocker.ui.components.MonoLabelText
import com.antispam.blocker.ui.components.SectionHeader
import com.antispam.blocker.ui.theme.*
import com.antispam.blocker.util.PhoneNormalizer
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.text.input.PasswordVisualTransformation
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private const val DEFAULT_DB_URL_PLACEHOLDER = RemoteUpdateWorker.DEFAULT_MANIFEST_URL

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen() {
    val app = SpamBlockerApp.instance
    val settings = app.settingsStore
    val scope = rememberCoroutineScope()
    val context = LocalContext.current

    val callerLookupStore = app.callerLookupSettingsStore
    val twoGisEnabled by callerLookupStore.twoGisEnabledFlow.collectAsState(initial = false)
    val savedApiKey by callerLookupStore.apiKeyFlow.collectAsState(initial = null)
    var apiKeyInput by rememberSaveable { mutableStateOf("") }
    LaunchedEffect(savedApiKey) { apiKeyInput = savedApiKey ?: "" }

    val dbUpdateEnabled by settings.dbUpdateEnabled.collectAsState(initial = true)
    val dbUpdateUrl by settings.dbUpdateUrl.collectAsState(initial = "")
    val lastUpdateAt by settings.lastUpdateAt.collectAsState(initial = 0L)
    val lastUpdateVersion by settings.lastUpdateVersion.collectAsState(initial = "")
    val skipCallLogForBlocked by settings.skipCallLogForBlocked.collectAsState(initial = false)

    // ── Privacy / Device_Model state (Req 7.4, 7.6, 7.7) ─────────────────
    val deviceModelStore = app.deviceModelStore
    val portabilityService = app.personalDataPortabilityService
    val sourceCallLogEnabled by deviceModelStore.sourceCallLogEnabledFlow
        .collectAsState(initial = true)
    val sourceContactsEnabled by deviceModelStore.sourceContactsEnabledFlow
        .collectAsState(initial = true)
    val sourceAppUsageEnabled by deviceModelStore.sourceAppUsageEnabledFlow
        .collectAsState(initial = true)
    val sourceNotificationsEnabled by deviceModelStore.sourceNotificationsEnabledFlow
        .collectAsState(initial = false)
    val labelCount by deviceModelStore.labelCountFlow.collectAsState(initial = 0)
    var showWipeDialog by remember { mutableStateOf(false) }

    // ── Just-in-time permission flow (Req 7.8, 7.9) ──────────────────────
    // Включение per-source toggle'а — это и есть "первая активация" источника.
    // До системного prompt'а показываем rationale, объясняющий, какие именно
    // фичи Device_Model получают данные из этого источника. Если разрешение
    // уже выдано (например, выдали в onboarding), persist'им сразу без диалога.
    var pendingRationale by remember { mutableStateOf<PrivacyRationale?>(null) }

    val callLogPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        scope.launch { deviceModelStore.setSourceCallLogEnabled(granted) }
        if (!granted) {
            Toast.makeText(
                context,
                "Без разрешения сигналы из журнала звонков не учитываются",
                Toast.LENGTH_LONG
            ).show()
        }
    }
    val contactsPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        scope.launch { deviceModelStore.setSourceContactsEnabled(granted) }
        if (!granted) {
            Toast.makeText(
                context,
                "Без разрешения is_contact не работает",
                Toast.LENGTH_LONG
            ).show()
        }
    }
    // Special Access (UsageStats, NotificationListener) запрашиваются через
    // системные настройки — у них нет runtime-prompt'а. Возвращаемся
    // через StartActivityForResult и пере-проверяем actual state.
    val usageAccessLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartActivityForResult()
    ) { _ ->
        val granted = isUsageAccessGranted(context)
        scope.launch { deviceModelStore.setSourceAppUsageEnabled(granted) }
        if (!granted) {
            Toast.makeText(
                context,
                "Включите Antispam Sentinel в списке «Доступ к данным об использовании»",
                Toast.LENGTH_LONG
            ).show()
        }
    }
    val notificationListenerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartActivityForResult()
    ) { _ ->
        val granted = isNotificationListenerEnabled(context)
        scope.launch { deviceModelStore.setSourceNotificationsEnabled(granted) }
        if (!granted) {
            Toast.makeText(
                context,
                "Включите Antispam Sentinel в списке доступа к уведомлениям",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    pendingRationale?.let { rationale ->
        AlertDialog(
            onDismissRequest = { pendingRationale = null },
            containerColor = InkElevated,
            title = { Text(rationale.title, color = TextPrimary) },
            text = {
                Text(
                    rationale.featureExplanation,
                    color = TextSecondary
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        val action = rationale.onConfirmed
                        pendingRationale = null
                        action()
                    }
                ) { Text("Продолжить", color = Amber) }
            },
            dismissButton = {
                TextButton(onClick = { pendingRationale = null }) {
                    Text("Отмена", color = TextSecondary)
                }
            }
        )
    }

    // SAF launchers for export / import (Req 2.5, 2.6).
    // CreateDocument возвращает Uri выбранного нового файла, OpenDocument — Uri
    // существующего; оба варианта пробрасываем в IO-корутины, чтобы основной
    // поток не блокировался файловыми операциями.
    val exportLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("application/json")
    ) { uri ->
        if (uri != null) {
            scope.launch {
                runCatching {
                    withContext(Dispatchers.IO) { portabilityService.exportToJson(uri) }
                }.onSuccess {
                    Toast.makeText(context, "Экспорт выполнен", Toast.LENGTH_SHORT).show()
                }.onFailure {
                    Toast.makeText(context, "Ошибка экспорта: ${it.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }
    val importLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            scope.launch {
                val result = withContext(Dispatchers.IO) { portabilityService.importFromJson(uri) }
                if (result.isSuccess) {
                    Toast.makeText(context, "Импорт выполнен", Toast.LENGTH_SHORT).show()
                } else {
                    val msg = result.exceptionOrNull()?.message ?: "неизвестная ошибка"
                    Toast.makeText(context, "Ошибка импорта: $msg", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    val blockListRepo = remember {
        BlockListRepository(app.database.blockedNumberDao(), app.database.allowedNumberDao(), PhoneNormalizer)
    }
    val totalBlocked by blockListRepo.totalCount.collectAsState(initial = 0)
    val prebuiltBlocked by blockListRepo.prebuiltCount.collectAsState(initial = 0)
    var reimporting by remember { mutableStateOf(false) }
    var showClearDialog by remember { mutableStateOf(false) }

    if (showClearDialog) {
        AlertDialog(
            onDismissRequest = { showClearDialog = false },
            containerColor = InkElevated,
            title = { Text("Очистить всю базу?", color = TextPrimary) },
            text = {
                Text(
                    "Будут удалены все заблокированные номера: встроенная база, скачанные из интернета и добавленные вручную. Белый список не будет затронут.",
                    color = TextSecondary
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showClearDialog = false
                        scope.launch {
                            blockListRepo.clearAllBlocked()
                            val importer = CsvSpamImporter(context, blockListRepo)
                            importer.resetImportFlag()
                            // Сразу подтягиваем встроенную базу, чтобы защита не осталась пустой
                            importer.importIfFirstRun()
                        }
                    }
                ) { Text("Очистить", color = BlockRed) }
            },
            dismissButton = {
                TextButton(onClick = { showClearDialog = false }) {
                    Text("Отмена", color = TextSecondary)
                }
            }
        )
    }

    if (showWipeDialog) {
        AlertDialog(
            onDismissRequest = { showWipeDialog = false },
            containerColor = InkElevated,
            title = { Text("Стереть всю историю обучения?", color = TextPrimary) },
            text = {
                Text(
                    "Будут удалены все события телеметрии (звонки, уведомления, активность приложений) и сохранённые feature-снапшоты. Веса персональной модели вернутся к defaults, окно прогрева начнётся заново.",
                    color = TextSecondary
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showWipeDialog = false
                        scope.launch {
                            withContext(Dispatchers.IO) { portabilityService.wipeAll() }
                            Toast.makeText(
                                context,
                                "История обучения очищена",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    }
                ) { Text("Стереть", color = BlockRed) }
            },
            dismissButton = {
                TextButton(onClick = { showWipeDialog = false }) {
                    Text("Отмена", color = TextSecondary)
                }
            }
        )
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
                SectionHeader(
                    eyebrow = "// SYSTEM",
                    title = "Настройки"
                )
                Spacer(Modifier.height(4.dp))
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                item {
                    GlassCard(
                        modifier = Modifier.fillMaxWidth(),
                        accentBorder = true
                    ) {
                        Column(
                            modifier = Modifier.padding(16.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                Icon(
                                    Icons.Rounded.Shield,
                                    contentDescription = null,
                                    tint = Amber,
                                    modifier = Modifier.size(18.dp)
                                )
                                MonoLabelText(text = "RESTRICTED SETTINGS", color = Amber)
                            }
                            Text(
                                text = "Если приложение установлено не из Google Play",
                                style = MaterialTheme.typography.titleMedium,
                                color = TextPrimary
                            )
                            Text(
                                text = "Перейдите в Настройки → Приложения → Блокировщик спама → ⋮ → Разрешить ограниченные настройки",
                                style = MaterialTheme.typography.bodySmall,
                                color = TextSecondary
                            )
                        }
                    }
                }
            }

            item {
                SettingsCard(
                    eyebrow = "CALL LOG",
                    title = "Журнал звонков",
                    description = "Не записывать заблокированные звонки в системный журнал Android"
                ) {
                    ToggleRow(
                        label = "Скрыть из журнала",
                        checked = skipCallLogForBlocked,
                        onCheckedChange = { scope.launch { settings.set("skip_call_log_for_blocked", it) } }
                    )
                }
            }

            item {
                SettingsCard(
                    eyebrow = "BUILT-IN DB",
                    title = "Встроенная база спам-номеров",
                    description = "Предзагруженный список известных спам-номеров, префиксов и масок. Работает без интернета."
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            MonoLabelText(text = "ENTRIES", color = TextTertiary)
                            Text(
                                text = totalBlocked.toString(),
                                style = MaterialTheme.typography.titleLarge,
                                color = TextPrimary
                            )
                        }
                        Column(modifier = Modifier.weight(1f)) {
                            MonoLabelText(text = "PREBUILT", color = TextTertiary)
                            Text(
                                text = prebuiltBlocked.toString(),
                                style = MaterialTheme.typography.titleLarge,
                                color = Amber
                            )
                        }
                    }
                    Spacer(Modifier.height(12.dp))
                    OutlinedButton(
                        onClick = {
                            if (!reimporting) {
                                reimporting = true
                                scope.launch {
                                    try {
                                        val repo = BlockListRepository(
                                            app.database.blockedNumberDao(),
                                            app.database.allowedNumberDao(),
                                            PhoneNormalizer
                                        )
                                        CsvSpamImporter(context, repo).reimport()
                                    } finally {
                                        reimporting = false
                                    }
                                }
                            }
                        },
                        enabled = !reimporting,
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(12.dp),
                        border = androidx.compose.foundation.BorderStroke(1.dp, InkBorder),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = TextPrimary)
                    ) {
                        Text(if (reimporting) "Импорт…" else "Переимпортировать встроенную базу")
                    }
                    Spacer(Modifier.height(8.dp))
                    OutlinedButton(
                        onClick = { showClearDialog = true },
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(12.dp),
                        border = androidx.compose.foundation.BorderStroke(1.dp, BlockRed.copy(alpha = 0.5f)),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = BlockRed)
                    ) {
                        Text("Очистить всю базу")
                    }
                }
            }

            item {
                SettingsCard(
                    eyebrow = "DATABASE",
                    title = "Облачное обновление базы",
                    description = "Раз в 6 часов приложение тянет публичный manifest со списком спам-номеров и data-driven prefix-risk таблицы. Каждый файл проверяется по SHA256, никакие персональные данные наружу не отправляются."
                ) {
                    ToggleRow(
                        label = "Автоматические обновления",
                        checked = dbUpdateEnabled,
                        onCheckedChange = { enabled ->
                            scope.launch {
                                settings.set("db_update_enabled", enabled)
                                if (enabled) {
                                    RemoteUpdateWorker.schedule(context)
                                } else {
                                    RemoteUpdateWorker.cancel(context)
                                }
                            }
                        }
                    )
                    if (dbUpdateEnabled) {
                        Spacer(Modifier.height(12.dp))
                        val versionLabel = if (lastUpdateVersion.isNotBlank()) lastUpdateVersion else "—"
                        val whenLabel = if (lastUpdateAt > 0L) {
                            android.text.format.DateUtils.getRelativeTimeSpanString(
                                lastUpdateAt,
                                System.currentTimeMillis(),
                                android.text.format.DateUtils.MINUTE_IN_MILLIS
                            ).toString()
                        } else "никогда"
                        Text(
                            text = "Последнее обновление: $whenLabel\nВерсия базы: $versionLabel",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextTertiary
                        )
                        Spacer(Modifier.height(12.dp))
                        OutlinedTextField(
                            value = dbUpdateUrl,
                            onValueChange = { scope.launch { settings.set("db_update_url", it) } },
                            label = { Text("URL manifest.json (опционально)", color = TextTertiary) },
                            placeholder = {
                                Text(DEFAULT_DB_URL_PLACEHOLDER, color = TextTertiary, maxLines = 1)
                            },
                            modifier = Modifier.fillMaxWidth(),
                            singleLine = true,
                            shape = RoundedCornerShape(12.dp),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Amber,
                                unfocusedBorderColor = InkBorder,
                                focusedTextColor = TextPrimary,
                                unfocusedTextColor = TextPrimary,
                                cursorColor = Amber
                            )
                        )
                        Spacer(Modifier.height(8.dp))
                        Text(
                            text = "Если поле пустое, используется ${RemoteUpdateWorker.DEFAULT_MANIFEST_URL}",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextTertiary
                        )
                        Spacer(Modifier.height(8.dp))
                        OutlinedButton(
                            onClick = {
                                val request = OneTimeWorkRequestBuilder<RemoteUpdateWorker>().build()
                                WorkManager.getInstance(context).enqueue(request)
                            },
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(12.dp),
                            border = androidx.compose.foundation.BorderStroke(1.dp, InkBorder),
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = TextPrimary
                            )
                        ) {
                            Text("Обновить сейчас")
                        }
                    }
                }
            }

            item {
                ModelFreshnessCard(
                    lastUpdateAt = lastUpdateAt,
                    lastUpdateVersion = lastUpdateVersion
                )
            }

            item {
                val activeSourcesCount = listOf(
                    sourceCallLogEnabled,
                    sourceContactsEnabled,
                    sourceAppUsageEnabled,
                    sourceNotificationsEnabled,
                ).count { it }
                SettingsCard(
                    eyebrow = "PRIVACY & ON-DEVICE LEARNING",
                    title = "Персональный классификатор",
                    description = "Personal Spam Classifier учится только на этом устройстве. Веса, телеметрия и feature-снапшоты никуда не уходят — управляйте источниками точечно."
                ) {
                    MonoLabelText(text = "ИСТОЧНИКИ ТЕЛЕМЕТРИИ", color = TextTertiary)
                    ToggleRow(
                        label = "Журнал звонков",
                        checked = sourceCallLogEnabled,
                        onCheckedChange = { enabled ->
                            if (!enabled) {
                                scope.launch { deviceModelStore.setSourceCallLogEnabled(false) }
                            } else if (hasRuntimePermission(context, Manifest.permission.READ_CALL_LOG)) {
                                scope.launch { deviceModelStore.setSourceCallLogEnabled(true) }
                            } else {
                                pendingRationale = PrivacyRationale(
                                    title = "Доступ к журналу звонков",
                                    featureExplanation = "Personal Spam Classifier использует журнал звонков, чтобы посчитать «отклонял ли я этот номер раньше», «как часто звонят с этого префикса за 7 дней» и «отвечал ли я на этот номер». Затронутые фичи: previously_rejected, prev_missed_no_callback_24h, prev_outgoing_after_missed, same_prefix_call_count_7d_norm, answer_rate_for_number_norm. Без этого доступа сигналы будут нулями.",
                                    onConfirmed = {
                                        callLogPermissionLauncher.launch(Manifest.permission.READ_CALL_LOG)
                                    }
                                )
                            }
                        }
                    )
                    ToggleRow(
                        label = "Контакты",
                        checked = sourceContactsEnabled,
                        onCheckedChange = { enabled ->
                            if (!enabled) {
                                scope.launch { deviceModelStore.setSourceContactsEnabled(false) }
                            } else if (hasRuntimePermission(context, Manifest.permission.READ_CONTACTS)) {
                                scope.launch { deviceModelStore.setSourceContactsEnabled(true) }
                            } else {
                                pendingRationale = PrivacyRationale(
                                    title = "Доступ к контактам",
                                    featureExplanation = "Контакты используются только для одной фичи: is_contact (вес ≈ −3.0 — самый сильный сигнал в сторону ALLOW). Имена и номера никуда не уходят и не сохраняются — на горячем пути проверяется только membership «есть ли номер в адресной книге».",
                                    onConfirmed = {
                                        contactsPermissionLauncher.launch(Manifest.permission.READ_CONTACTS)
                                    }
                                )
                            }
                        }
                    )
                    ToggleRow(
                        label = "Использование приложений",
                        checked = sourceAppUsageEnabled,
                        onCheckedChange = { enabled ->
                            if (!enabled) {
                                scope.launch { deviceModelStore.setSourceAppUsageEnabled(false) }
                            } else if (isUsageAccessGranted(context)) {
                                scope.launch { deviceModelStore.setSourceAppUsageEnabled(true) }
                            } else {
                                pendingRationale = PrivacyRationale(
                                    title = "Доступ к данным об использовании",
                                    featureExplanation = "Чтобы понимать «вы только что заходили в банк / Госуслуги / маркетплейс / мессенджер за последние 30 минут», Personal Spam Classifier читает foreground-события из UsageStats. Затронутые фичи: recent_bank_app_30m, recent_gov_app_30m, recent_marketplace_app_30m, recent_messenger_app_30m. Это special access — нужно включить Antispam Sentinel в системных настройках.",
                                    onConfirmed = {
                                        usageAccessLauncher.launch(
                                            Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS)
                                                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                                        )
                                    }
                                )
                            }
                        }
                    )
                    ToggleRow(
                        label = "Уведомления",
                        checked = sourceNotificationsEnabled,
                        onCheckedChange = { enabled ->
                            if (!enabled) {
                                scope.launch { deviceModelStore.setSourceNotificationsEnabled(false) }
                            } else if (isNotificationListenerEnabled(context)) {
                                scope.launch { deviceModelStore.setSourceNotificationsEnabled(true) }
                            } else {
                                pendingRationale = PrivacyRationale(
                                    title = "Доступ к уведомлениям",
                                    featureExplanation = "Используется только пакет приложения, отправившего уведомление, для фич notif_bank_recent_10m и notif_marketplace_recent_10m. Тело уведомлений никогда не читается и нигде не сохраняется. Доступ включается в системных настройках в списке Notification Listener.",
                                    onConfirmed = {
                                        notificationListenerLauncher.launch(
                                            Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS)
                                                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                                        )
                                    }
                                )
                            }
                        }
                    )

                    Spacer(Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            MonoLabelText(text = "CALLS PROCESSED", color = TextTertiary)
                            Text(
                                text = labelCount.toString(),
                                style = MaterialTheme.typography.titleLarge,
                                color = TextPrimary
                            )
                        }
                        Column(modifier = Modifier.weight(1f)) {
                            MonoLabelText(text = "ACTIVE SOURCES", color = TextTertiary)
                            Text(
                                text = "$activeSourcesCount / 4",
                                style = MaterialTheme.typography.titleLarge,
                                color = Amber
                            )
                        }
                    }

                    Spacer(Modifier.height(8.dp))
                    // Диагностика что приложение реально читает контакты:
                    // ContactsCache прогревается в SpamCallScreeningService.onCreate,
                    // здесь показываем количество и подсвечиваем status.
                    val contactsCacheSize = remember { com.antispam.blocker.data.cache.ContactsCache.size() }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .background(
                                    color = when {
                                        contactsCacheSize < 0 -> TextTertiary
                                        contactsCacheSize == 0 -> WarnAmber
                                        else -> AllowGreen
                                    },
                                    shape = androidx.compose.foundation.shape.CircleShape
                                )
                        )
                        Spacer(Modifier.width(8.dp))
                        Text(
                            text = when {
                                contactsCacheSize < 0 -> "Контакты: кэш не прогрет (ждём первичную загрузку)"
                                contactsCacheSize == 0 -> "Контакты: 0 — справочник пуст или разрешение отозвано"
                                else -> "Контакты: $contactsCacheSize прогрето в кэше"
                            },
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSecondary,
                        )
                    }

                    Spacer(Modifier.height(12.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        OutlinedButton(
                            onClick = {
                                // CreateDocument сам подставит расширение из MIME-типа,
                                // но имя по умолчанию задаём явно для удобства пользователя.
                                exportLauncher.launch("antispam_personal_export.json")
                            },
                            modifier = Modifier.weight(1f),
                            shape = RoundedCornerShape(12.dp),
                            border = androidx.compose.foundation.BorderStroke(1.dp, InkBorder),
                            colors = ButtonDefaults.outlinedButtonColors(contentColor = TextPrimary)
                        ) {
                            Text("Экспорт")
                        }
                        OutlinedButton(
                            onClick = {
                                // OpenDocument фильтрует system picker по MIME-типу;
                                // поддерживаем только application/json — формат экспорта.
                                importLauncher.launch(arrayOf("application/json"))
                            },
                            modifier = Modifier.weight(1f),
                            shape = RoundedCornerShape(12.dp),
                            border = androidx.compose.foundation.BorderStroke(1.dp, InkBorder),
                            colors = ButtonDefaults.outlinedButtonColors(contentColor = TextPrimary)
                        ) {
                            Text("Импорт")
                        }
                    }

                    Spacer(Modifier.height(8.dp))
                    OutlinedButton(
                        onClick = { showWipeDialog = true },
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(12.dp),
                        border = androidx.compose.foundation.BorderStroke(
                            1.dp,
                            BlockRed.copy(alpha = 0.5f)
                        ),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = BlockRed)
                    ) {
                        Text("Стереть всю историю обучения")
                    }
                }
            }

            item {
                // Прозрачность данных: пользователь видит «доказательства»,
                // что приложение реально читает контакты и журнал звонков —
                // счётчики и маскированные превью. Не выгружает наружу,
                // только подтверждает локальное чтение для UX-доверия.
                DataTransparencyCard(
                    callEventDao = remember { app.database.callEventDao() },
                    refreshKey = labelCount,
                )
            }

            item {
                SettingsCard(
                    eyebrow = "// CALLER ID",
                    title = "Определение звонящего",
                    description = "Показывает кто звонил в журнале: оператор, регион и (опционально) название организации из 2GIS."
                ) {
                    // Оффлайн всегда включён — работает без интернета
                    MonoLabelText(text = "OFFLINE: LIBPHONENUMBER  ●  ВСЕГДА АКТИВНО", color = AllowGreen)
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = "Регион и оператор определяются без интернета на основе DEF-кодов номера.",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary,
                    )
                    Spacer(Modifier.height(12.dp))
                    ToggleRow(
                        label = "2GIS online (opt-in)",
                        checked = twoGisEnabled,
                        onCheckedChange = { enabled ->
                            scope.launch { callerLookupStore.setTwoGisEnabled(enabled) }
                        }
                    )
                    Text(
                        text = "Название организации для бизнес-номеров (Пятёрочка, банк, кафе). Номер уходит на серверы 2GIS — только с вашего согласия.",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextTertiary,
                    )
                    if (twoGisEnabled) {
                        Spacer(Modifier.height(8.dp))
                        OutlinedTextField(
                            value = apiKeyInput,
                            onValueChange = { apiKeyInput = it },
                            label = { Text("API-ключ 2GIS (dev.2gis.ru)") },
                            placeholder = { Text("Вставьте ключ сюда") },
                            visualTransformation = PasswordVisualTransformation(),
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth(),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Amber,
                                focusedLabelColor = Amber,
                                cursorColor = Amber,
                                unfocusedBorderColor = InkBorder,
                                unfocusedLabelColor = TextTertiary,
                            ),
                        )
                        Spacer(Modifier.height(4.dp))
                        OutlinedButton(
                            onClick = {
                                scope.launch { callerLookupStore.setApiKey(apiKeyInput) }
                            },
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(10.dp),
                            border = androidx.compose.foundation.BorderStroke(1.dp, Amber),
                            colors = ButtonDefaults.outlinedButtonColors(contentColor = Amber),
                        ) {
                            Text("Сохранить ключ")
                        }
                        if (!savedApiKey.isNullOrBlank()) {
                            Spacer(Modifier.height(4.dp))
                            MonoLabelText(
                                text = "KEY SAVED  ●  ONLINE LOOKUP ACTIVE",
                                color = AllowGreen,
                            )
                        }
                    }
                }
            }

            item {
                SettingsCard(
                    eyebrow = "ABOUT",
                    title = "Antispam Sentinel v3.0",
                    description = "Приложение анализирует входящие звонки с помощью нейросети и адаптивных правил. Защита обучается на ваших реакциях. Работает полностью без интернета."
                )
            }

            item {
                Spacer(Modifier.height(12.dp))
                MonoLabelText(
                    text = "BUILT WITH KOTLIN  •  JETPACK COMPOSE  •  ROOM",
                    color = TextTertiary,
                    modifier = Modifier.fillMaxWidth()
                )
            }
        }
    }
}

@Composable
private fun SettingsCard(
    eyebrow: String,
    title: String,
    description: String,
    content: (@Composable ColumnScope.() -> Unit)? = null
) {
    GlassCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            MonoLabelText(text = eyebrow, color = TextTertiary)
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium,
                color = TextPrimary
            )
            Text(
                text = description,
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary
            )
            if (content != null) {
                Spacer(Modifier.height(4.dp))
                content()
            }
        }
    }
}

@Composable
private fun ToggleRow(
    label: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = TextPrimary
        )
        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange,
            colors = SwitchDefaults.colors(
                checkedThumbColor = TextOnAccent,
                checkedTrackColor = Amber,
                uncheckedThumbColor = TextSecondary,
                uncheckedTrackColor = InkSurface,
                uncheckedBorderColor = InkBorder
            )
        )
    }
}

/**
 * Состояние модели: дата последнего успешного manifest-pull, версия
 * [ModelCard.version] (id обученной модели в файле `model_card.json`)
 * и индикатор pending-загрузки — есть ли ENQUEUED/RUNNING WorkInfo в
 * `RemoteUpdateWorker.UNIQUE_NAME`. Чисто диагностическая карточка —
 * не меняет поведения, но даёт понять, насколько свежая база и модель
 * сейчас работают и идёт ли скачивание прямо сейчас.
 */
@Composable
private fun ModelFreshnessCard(
    lastUpdateAt: Long,
    lastUpdateVersion: String
) {
    val context = LocalContext.current

    // ModelCard.load() читает с диска (filesDir или assets), поэтому
    // прогружаем в IO-thread через LaunchedEffect. Re-load случается, когда
    // прилетает новая версия base'ы (lastUpdateVersion обновляется).
    var modelCard by remember { mutableStateOf<ModelCard?>(null) }
    LaunchedEffect(lastUpdateVersion) {
        modelCard = withContext(Dispatchers.IO) { ModelCard.load(context) }
    }

    val workLive = remember(context) {
        WorkManager.getInstance(context)
            .getWorkInfosForUniqueWorkLiveData(RemoteUpdateWorker.UNIQUE_NAME)
    }
    val workInfos by workLive.observeAsState(initial = emptyList())

    val pending = workInfos.any { it.state == WorkInfo.State.RUNNING }
    val scheduled = workInfos.any { it.state == WorkInfo.State.ENQUEUED }

    val whenLabel = if (lastUpdateAt > 0L) {
        android.text.format.DateUtils.getRelativeTimeSpanString(
            lastUpdateAt,
            System.currentTimeMillis(),
            android.text.format.DateUtils.MINUTE_IN_MILLIS
        ).toString()
    } else "никогда"

    val manifestVersion = lastUpdateVersion.ifBlank { "—" }
    val cardVersion = modelCard?.version ?: "—"

    GlassCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            MonoLabelText(text = "MODEL FRESHNESS · SERVER MODEL", color = TextTertiary)
            Text(
                text = "Server Model (TFLite)",
                style = MaterialTheme.typography.titleMedium,
                color = TextPrimary
            )
            Text(
                text = "Серверная TFLite-модель, общая для всех пользователей. Загружается раз в 6 ч с публичного manifest. Personal Model (персональная) живёт отдельно — ниже в разделе PRIVACY.",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary
            )

            Spacer(Modifier.height(4.dp))

            FreshnessRow(label = "Последний pull manifest", value = whenLabel)
            FreshnessRow(label = "Версия manifest", value = manifestVersion)
            FreshnessRow(label = "Версия model_card", value = cardVersion)
            modelCard?.thresholds?.let { t ->
                FreshnessRow(
                    label = "Пороги (warm)",
                    value = "warn=${"%.2f".format(t.warnThreshold)} • block=${"%.2f".format(t.blockThreshold)}"
                )
            }

            Spacer(Modifier.height(4.dp))
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .background(
                            color = when {
                                pending -> Amber
                                scheduled -> AllowGreen
                                else -> TextTertiary
                            },
                            shape = androidx.compose.foundation.shape.CircleShape
                        )
                )
                Text(
                    text = when {
                        pending -> "Идёт скачивание обновления…"
                        scheduled -> "Запланировано (раз в 6ч)"
                        else -> "Загрузка не запланирована"
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = if (pending) Amber else TextTertiary
                )
            }
        }
    }
}

@Composable
private fun FreshnessRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = TextTertiary
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodySmall,
            color = TextPrimary
        )
    }
}




/**
 * Описание rationale-диалога, показываемого в момент первой активации
 * Telemetry_Source (Req 7.8, 7.9). [onConfirmed] стартует ровно одно
 * системное действие: либо запрос runtime-permission, либо переход в
 * системные настройки Special Access — выбор зависит от источника.
 */
private data class PrivacyRationale(
    val title: String,
    val featureExplanation: String,
    val onConfirmed: () -> Unit,
)

private fun hasRuntimePermission(context: Context, permission: String): Boolean =
    ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED

private fun isUsageAccessGranted(context: Context): Boolean = try {
    val appOps = context.getSystemService(Context.APP_OPS_SERVICE) as AppOpsManager
    // minSdk=29: unsafeCheckOpNoThrow доступен напрямую, fallback не нужен.
    val mode = appOps.unsafeCheckOpNoThrow(
        AppOpsManager.OPSTR_GET_USAGE_STATS,
        Process.myUid(),
        context.packageName
    )
    mode == AppOpsManager.MODE_ALLOWED
} catch (_: Throwable) {
    false
}

private fun isNotificationListenerEnabled(context: Context): Boolean = try {
    NotificationManagerCompat
        .getEnabledListenerPackages(context)
        .contains(context.packageName)
} catch (_: Throwable) {
    false
}


/**
 * Карточка «Прозрачность данных» в разделе PRIVACY.
 *
 * Что показывает:
 * - Контакты: количество прогретых записей в [com.antispam.blocker.data.cache.ContactsCache]
 *   плюс 3 маскированных номера (`+7XXX•••XX`) — пользователь сразу видит,
 *   что приложение реально читает его адресную книгу.
 * - Журнал звонков: суммарное число записей в `call_event` (Room) плюс 3
 *   последние записи с маскированным номером и относительным временем —
 *   то же подтверждение для CallLog.
 *
 * Зачем: до этой карточки у пользователя не было визуального
 * подтверждения, что Personal Model реально персонализируется на его
 * данных. Теперь он сам видит свои свежие звонки и контакты в UI.
 *
 * Ничего из этого не пишется в Room заново, не отправляется в сеть, не
 * логируется. Все номера маскируются на стороне UI [maskPhoneForDisplay].
 *
 * @param refreshKey любое значение, изменение которого триггерит перечит
 *   из Room (используем `labelCount`, чтобы карточка обновлялась при
 *   каждом новом обработанном звонке).
 */
@Composable
private fun DataTransparencyCard(
    callEventDao: com.antispam.blocker.data.db.dao.CallEventDao,
    refreshKey: Int,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val app = SpamBlockerApp.instance
    val notificationEventDao = remember { app.database.notificationEventDao() }
    val appUsageEventDao = remember { app.database.appUsageEventDao() }

    var contactsTick by remember { mutableIntStateOf(0) }
    val contactsCount = remember(refreshKey, contactsTick) {
        com.antispam.blocker.data.cache.ContactsCache.size()
    }
    val contactsSamples = remember(refreshKey, contactsTick) {
        com.antispam.blocker.data.cache.ContactsCache.sampleMaskedNumbers(3)
    }
    val hasContactsPermission = remember(contactsTick) {
        hasRuntimePermission(context, Manifest.permission.READ_CONTACTS)
    }

    val contactsPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            // Сразу принудительно перезагружаем кэш, чтобы счётчик
            // в этой карточке стал зелёным без ожидания.
            scope.launch {
                withContext(Dispatchers.IO) {
                    com.antispam.blocker.data.cache.ContactsCache.forceReloadBlocking(context)
                }
                contactsTick++
            }
        } else {
            Toast.makeText(
                context,
                "Без READ_CONTACTS Personal Model не сможет считать фичу is_contact",
                Toast.LENGTH_LONG,
            ).show()
        }
    }

    val callLogPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (!granted) {
            Toast.makeText(
                context,
                "Без READ_CALL_LOG не получится посчитать «previously rejected», «answer rate» и аггрегаты по 7д.",
                Toast.LENGTH_LONG,
            ).show()
        }
        // Refresh tick — контакты или журнал, оба триггерят перечит счётчиков.
        contactsTick++
    }

    // Special-access launchers (notification listener + usage stats):
    // нет runtime-permission диалога — только системные настройки.
    val notificationListenerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartActivityForResult(),
    ) { _ -> contactsTick++ }
    val usageAccessLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartActivityForResult(),
    ) { _ -> contactsTick++ }

    var callEventCount by remember(refreshKey) { mutableIntStateOf(-1) }
    var callEventSamples by remember(refreshKey) {
        mutableStateOf<List<com.antispam.blocker.data.db.entity.CallEvent>>(emptyList())
    }
    var notificationEventCount by remember(refreshKey, contactsTick) { mutableIntStateOf(-1) }
    var notificationEventSamples by remember(refreshKey, contactsTick) {
        mutableStateOf<List<com.antispam.blocker.data.db.entity.NotificationEvent>>(emptyList())
    }
    var appUsageEventCount by remember(refreshKey, contactsTick) { mutableIntStateOf(-1) }
    var appUsageEventSamples by remember(refreshKey, contactsTick) {
        mutableStateOf<List<com.antispam.blocker.domain.scoring.RecentUserContextProvider.ForegroundEventSample>>(emptyList())
    }
    LaunchedEffect(refreshKey, contactsTick) {
        callEventCount = withContext(Dispatchers.IO) {
            runCatching { callEventDao.countAll() }.getOrDefault(0)
        }
        callEventSamples = withContext(Dispatchers.IO) {
            runCatching { callEventDao.recent(3) }.getOrDefault(emptyList())
        }
        notificationEventCount = withContext(Dispatchers.IO) {
            runCatching { notificationEventDao.countAll() }.getOrDefault(0)
        }
        notificationEventSamples = withContext(Dispatchers.IO) {
            runCatching { notificationEventDao.recent(3) }.getOrDefault(emptyList())
        }
        // App usage таблица в Room специально не наполняется (Personal Model
        // читает UsageStats live через RecentUserContextProvider). Поэтому
        // здесь делаем то же — берём live-foreground-выборку из системы.
        val provider = com.antispam.blocker.domain.scoring.RecentUserContextProvider(context)
        appUsageEventSamples = withContext(Dispatchers.IO) {
            runCatching { provider.recentForegroundEvents(limit = 3) }.getOrDefault(emptyList())
        }
        appUsageEventCount = appUsageEventSamples.size
    }

    GlassCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            MonoLabelText(text = "DATA TRANSPARENCY", color = TextTertiary)
            Text(
                text = "Прозрачность данных",
                style = MaterialTheme.typography.titleMedium,
                color = TextPrimary,
            )
            Text(
                text = "Доказательства, что Personal Model реально учится на вашем телефоне. Все номера маскируются. Ничего из этого не покидает устройство.",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
            )

            // ── Контакты ──────────────────────────────────────────────────
            TransparencyBlock(
                title = "Контакты",
                statusEmoji = when {
                    !hasContactsPermission -> "🔴"
                    contactsCount < 0 -> "⚪"
                    contactsCount == 0 -> "🟡"
                    else -> "🟢"
                },
                statusText = when {
                    !hasContactsPermission -> "Разрешение не выдано"
                    contactsCount < 0 -> "Кэш не прогрет"
                    contactsCount == 0 -> "Справочник пуст"
                    else -> "$contactsCount записей прогрето"
                },
                samples = contactsSamples,
                emptyHint = if (!hasContactsPermission) {
                    "Без разрешения READ_CONTACTS Personal Model не может определить, входящий ли это контакт. Это самый сильный сигнал в сторону ALLOW (вес ≈ −3.0)."
                } else if (contactsCount < 0) {
                    "Прогрев идёт в фоне. Если кнопка ниже не помогает — попробуйте перезапустить приложение."
                } else {
                    "Справочник прочитан, но в нём нет ни одной записи."
                },
                action = if (!hasContactsPermission) {
                    "Дать разрешение" to {
                        contactsPermissionLauncher.launch(Manifest.permission.READ_CONTACTS)
                    }
                } else if (contactsCount <= 0) {
                    "Перечитать сейчас" to {
                        scope.launch {
                            withContext(Dispatchers.IO) {
                                com.antispam.blocker.data.cache.ContactsCache.forceReloadBlocking(context)
                            }
                            contactsTick++
                        }
                    }
                } else null,
            )

            // ── Журнал звонков ─────────────────────────────────────────────
            val hasCallLogPermission = remember(refreshKey) {
                hasRuntimePermission(context, Manifest.permission.READ_CALL_LOG)
            }
            TransparencyBlock(
                title = "Журнал звонков",
                statusEmoji = when {
                    !hasCallLogPermission -> "🔴"
                    callEventCount < 0 -> "⚪"
                    callEventCount == 0 -> "🟡"
                    else -> "🟢"
                },
                statusText = when {
                    !hasCallLogPermission -> "Разрешение не выдано"
                    callEventCount < 0 -> "Загружается"
                    callEventCount == 0 -> "Пока нет записей"
                    else -> "$callEventCount записей"
                },
                samples = callEventSamples.map { event ->
                    val masked = maskPhoneForDisplay(event.normalizedNumber.orEmpty())
                    val ago = android.text.format.DateUtils.getRelativeTimeSpanString(
                        event.startedAt,
                        System.currentTimeMillis(),
                        android.text.format.DateUtils.MINUTE_IN_MILLIS,
                    )
                    "$masked · $ago"
                },
                emptyHint = if (!hasCallLogPermission) {
                    "Без READ_CALL_LOG Personal Model не сможет посчитать «отклонял ли я этот номер раньше», «отвечал ли я на этот номер», «как часто звонят с этого префикса за 7 дней»."
                } else {
                    "Каждый новый звонок попадает сюда автоматически. Записи появятся после первого входящего."
                },
                action = if (!hasCallLogPermission) {
                    "Дать разрешение" to {
                        callLogPermissionLauncher.launch(Manifest.permission.READ_CALL_LOG)
                    }
                } else null,
            )

            // ── Уведомления (NotificationListenerService) ──────────────────
            // Special-access — runtime-permission диалога нет, отправляем юзера
            // прямиком в системные настройки. Записываем только packageName +
            // categoryBucket + timestamp, никакого тела уведомлений.
            val hasNotifAccess = remember(contactsTick) {
                isNotificationListenerEnabled(context)
            }
            TransparencyBlock(
                title = "Уведомления",
                statusEmoji = when {
                    !hasNotifAccess -> "🔴"
                    notificationEventCount < 0 -> "⚪"
                    notificationEventCount == 0 -> "🟡"
                    else -> "🟢"
                },
                statusText = when {
                    !hasNotifAccess -> "Доступ Notification Listener выключен"
                    notificationEventCount < 0 -> "Загружается"
                    notificationEventCount == 0 -> "Пока нет событий"
                    else -> "$notificationEventCount событий"
                },
                samples = notificationEventSamples.map { event ->
                    val app = maskPackageForDisplay(event.packageName)
                    val ago = android.text.format.DateUtils.getRelativeTimeSpanString(
                        event.timestamp,
                        System.currentTimeMillis(),
                        android.text.format.DateUtils.MINUTE_IN_MILLIS,
                    )
                    "$app · ${event.categoryBucket} · $ago"
                },
                emptyHint = if (!hasNotifAccess) {
                    "Без Notification Listener Personal Model не сможет считать notif_bank_recent_10m / notif_marketplace_recent_10m. Только packageName и категория, тело уведомлений никогда не читается."
                } else {
                    "Любое новое уведомление от banking/marketplace/messenger приложений попадёт сюда."
                },
                action = if (!hasNotifAccess) {
                    "Открыть настройки" to {
                        notificationListenerLauncher.launch(
                            android.content.Intent(android.provider.Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS)
                                .addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK),
                        )
                    }
                } else null,
            )

            // ── App Usage (UsageStats) ────────────────────────────────────
            val hasUsageAccess = remember(contactsTick) {
                isUsageAccessGranted(context)
            }
            TransparencyBlock(
                title = "Использование приложений",
                statusEmoji = when {
                    !hasUsageAccess -> "🔴"
                    appUsageEventCount < 0 -> "⚪"
                    appUsageEventCount == 0 -> "🟡"
                    else -> "🟢"
                },
                statusText = when {
                    !hasUsageAccess -> "UsageStats доступ выключен"
                    appUsageEventCount < 0 -> "Загружается"
                    appUsageEventCount == 0 -> "Пока нет событий"
                    else -> "$appUsageEventCount событий"
                },
                samples = appUsageEventSamples.map { event ->
                    val app = maskPackageForDisplay(event.packageName)
                    val ago = android.text.format.DateUtils.getRelativeTimeSpanString(
                        event.timestamp,
                        System.currentTimeMillis(),
                        android.text.format.DateUtils.MINUTE_IN_MILLIS,
                    )
                    "$app · ${event.categoryBucket} · $ago"
                },
                emptyHint = if (!hasUsageAccess) {
                    "Без UsageStats Personal Model не считает recent_bank_app_30m / recent_gov_app_30m / recent_marketplace_app_30m / recent_messenger_app_30m."
                } else {
                    "Откройте банк / Госуслуги / маркетплейс / мессенджер — последний foreground появится здесь сразу."
                },
                action = if (!hasUsageAccess) {
                    "Открыть настройки" to {
                        usageAccessLauncher.launch(
                            android.content.Intent(android.provider.Settings.ACTION_USAGE_ACCESS_SETTINGS)
                                .addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK),
                        )
                    }
                } else null,
            )
        }
    }
}

@Composable
private fun TransparencyBlock(
    title: String,
    statusEmoji: String,
    statusText: String,
    samples: List<String>,
    emptyHint: String,
    action: Pair<String, () -> Unit>? = null,
) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(text = statusEmoji, style = MaterialTheme.typography.bodyLarge)
            Spacer(Modifier.width(8.dp))
            Text(
                text = title,
                style = MaterialTheme.typography.titleSmall,
                color = TextPrimary,
            )
            Spacer(Modifier.weight(1f))
            Text(
                text = statusText,
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
            )
        }
        if (samples.isEmpty()) {
            Text(
                text = emptyHint,
                style = MaterialTheme.typography.bodySmall,
                color = TextTertiary,
                modifier = Modifier.padding(start = 28.dp),
            )
        } else {
            for (s in samples) {
                Text(
                    text = "·  $s",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                    modifier = Modifier.padding(start = 24.dp),
                )
            }
        }
        if (action != null) {
            Spacer(Modifier.height(4.dp))
            OutlinedButton(
                onClick = action.second,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(start = 28.dp),
                shape = RoundedCornerShape(10.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, Amber),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = Amber),
            ) {
                Text(action.first)
            }
        }
    }
}

/**
 * Маска телефонного номера для UI: «+7XXX•••XX12». Оставляем код страны,
 * 3 цифры DEF-кода и последние 2 цифры; всё остальное прячем за «•••».
 * Безопасно для коротких/нестандартных номеров (USSD, экстренные).
 */
private fun maskPhoneForDisplay(normalized: String): String {
    if (normalized.length <= 6) return normalized
    val keepHead = normalized.take(5)
    val keepTail = normalized.takeLast(2)
    return "$keepHead•••$keepTail"
}

/**
 * Маска package-имени для UI: «com.tinkoff» → «com.tinkoff» (короткие
 * показываем целиком), «com.android.chrome.mobile» → «com.android…mobile».
 * Pro-приложения часто содержат нечувствительные данные (брэнд), полная
 * маскировка не требуется — но для длинных пакетов делаем читаемый вариант.
 */
private fun maskPackageForDisplay(pkg: String): String {
    if (pkg.length <= 28) return pkg
    return pkg.take(14) + "…" + pkg.takeLast(10)
}
