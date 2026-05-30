package com.antispam.blocker.ui.screens

import android.widget.Toast
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.AddCircle
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.antispam.blocker.SpamBlockerApp
import androidx.compose.runtime.collectAsState
import com.antispam.blocker.data.cache.ContactNameLookup
import com.antispam.blocker.data.db.entity.CallRecord
import com.antispam.blocker.data.repository.BlockListRepository
import com.antispam.blocker.domain.detector.Verdict
import com.antispam.blocker.domain.personal.ExplicitLabel
import com.antispam.blocker.ui.components.GlassCard
import com.antispam.blocker.ui.components.MonoLabelText
import com.antispam.blocker.ui.components.SectionHeader
import com.antispam.blocker.ui.components.StatusPill
import com.antispam.blocker.ui.theme.*
import com.antispam.blocker.util.PhoneNormalizer
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*

/**
 * Журнал звонков. Long-press по строке открывает контекстное меню
 * (Req 4.7, 6.2):
 *
 *   1. «Открыть детали»          → переход на ExplainabilityDetailScreen
 *      по `feature_snapshot.id`, найденному через
 *      `FeatureSnapshotDao.getLatestForNumber(normalizedNumber)`.
 *   2. «Отметить как спам»       → `OnlineTrainer.applyExplicitLabel(snapshotId, BLOCK)`.
 *   3. «Отметить как легитимный» → `OnlineTrainer.applyExplicitLabel(snapshotId, ALLOW)`.
 *
 * Если для номера нет ни одного `feature_snapshot` (например, вердикт
 * пришёл с fast-path до того, как Device_Model вычислил вектор), то
 * пользователю показывается toast «Нет данных Device_Model для этого
 * номера», и меню не открывается. Внутренние IconButton'ы (add to
 * blocklist / allowlist) обрабатывают свои нажатия независимо от
 * combinedClickable родителя.
 *
 * `navController` опциональный (`null` по умолчанию), чтобы Composable
 * можно было превьюить без NavHost; в реальном приложении его передаёт
 * `AppNavigation`.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CallLogScreen(navController: NavController? = null) {
    val app = SpamBlockerApp.instance
    val callLogRepo = remember { com.antispam.blocker.data.repository.CallLogRepository(app.database.callRecordDao()) }
    val blockListRepo = remember {
        BlockListRepository(app.database.blockedNumberDao(), app.database.allowedNumberDao(), PhoneNormalizer)
    }
    val featureSnapshotDao = remember { app.database.featureSnapshotDao() }
    val onlineTrainer = remember { app.onlineTrainer }
    val callerLookupRepo = remember { app.callerLookupRepository }
    val scope = rememberCoroutineScope()

    val records by callLogRepo.allRecords.collectAsState(initial = emptyList())

    Scaffold(
        containerColor = Ink,
        contentWindowInsets = WindowInsets(0)
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            contentPadding = PaddingValues(horizontal = 20.dp, vertical = 24.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            item {
                SectionHeader(
                    eyebrow = "// EVENT STREAM",
                    title = "Журнал"
                )
                Spacer(Modifier.height(4.dp))
                MonoLabelText(
                    text = "${records.size} events logged",
                    color = TextTertiary
                )
                Spacer(Modifier.height(16.dp))
            }

            if (records.isEmpty()) {
                item {
                    GlassCard(modifier = Modifier.fillMaxWidth()) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(32.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text(
                                    text = "Пока нет записей",
                                    style = MaterialTheme.typography.titleMedium,
                                    color = TextPrimary
                                )
                                Spacer(Modifier.height(4.dp))
                                Text(
                                    text = "Как только поступит звонок, он появится здесь",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = TextSecondary
                                )
                            }
                        }
                    }
                }
            } else {
                items(records, key = { it.id }) { record ->
                    CallRecordItem(
                        record = record,
                        callerLookupRepo = callerLookupRepo,
                        onBlock = {
                            val num = record.originalNumber ?: return@CallRecordItem
                            scope.launch {
                                blockListRepo.addToBlockList(num)
                                // Без этого UPDATE запись в журнале остаётся
                                // со старым вердиктом WARN/ALLOW, и иконка
                                // «+» не пропадает (см. CallLogRepository.updateVerdict).
                                val normalized = record.normalizedNumber
                                if (!normalized.isNullOrBlank()) {
                                    callLogRepo.updateVerdict(normalized, Verdict.BLOCK)
                                }
                            }
                        },
                        onAllow = {
                            val num = record.originalNumber ?: return@CallRecordItem
                            scope.launch {
                                blockListRepo.addToAllowList(num)
                                val normalized = record.normalizedNumber
                                if (!normalized.isNullOrBlank()) {
                                    callLogRepo.updateVerdict(normalized, Verdict.ALLOW)
                                }
                            }
                        },
                        resolveSnapshotId = { number ->
                            // P1 fix: предпочитаем точный snapshot этой записи. Для
                            // старых строк (до миграции 5→6) поле = null, fallback
                            // на getLatestForNumber.
                            record.featureSnapshotId
                                ?: featureSnapshotDao.getLatestForNumber(number)?.id
                        },
                        onOpenDetail = { snapshotId ->
                            navController?.navigate("explain/$snapshotId")
                        },
                        onMarkSpam = { snapshotId ->
                            scope.launch {
                                onlineTrainer.applyExplicitLabel(snapshotId, ExplicitLabel.BLOCK)
                            }
                        },
                        onMarkLegit = { snapshotId ->
                            scope.launch {
                                onlineTrainer.applyExplicitLabel(snapshotId, ExplicitLabel.ALLOW)
                            }
                        },
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun CallRecordItem(
    record: CallRecord,
    callerLookupRepo: com.antispam.blocker.domain.lookup.CallerLookupRepository,
    onBlock: () -> Unit,
    onAllow: () -> Unit,
    resolveSnapshotId: suspend (String) -> Long?,
    onOpenDetail: (Long) -> Unit,
    onMarkSpam: (Long) -> Unit,
    onMarkLegit: (Long) -> Unit,
) {
    val (color, label) = when (record.verdict) {
        Verdict.BLOCK -> BlockRed to "заблокирован"
        Verdict.WARN -> WarnAmber to "подозрительный"
        Verdict.ALLOW -> AllowGreen to "разрешён"
    }

    val dateFormat = remember { SimpleDateFormat("dd.MM HH:mm", Locale.getDefault()) }
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var menuExpanded by remember { mutableStateOf(false) }
    // Снэпшот резолвится лениво — только когда пользователь действительно
    // долго нажал на строку, чтобы не дёргать DAO для каждой строки
    // журнала на старте.
    var menuSnapshotId by remember { mutableStateOf<Long?>(null) }

    // Подписываемся на callerInfo из кэша — обновится при записи Worker'а.
    val callerInfo by remember(record.normalizedNumber) {
        record.normalizedNumber
            ?.let { callerLookupRepo.observe(it) }
            ?: kotlinx.coroutines.flow.flowOf(null)
    }.collectAsState(initial = null)

    GlassCard(
        modifier = Modifier
            .fillMaxWidth()
            .combinedClickable(
                onClick = { /* row-tap зарезервирован — внутренние IconButton'ы перехватывают свои клики */ },
                onLongClick = {
                    val number = record.normalizedNumber
                    if (number.isNullOrBlank()) {
                        Toast.makeText(
                            context,
                            "Нет данных Device_Model для этого номера",
                            Toast.LENGTH_SHORT,
                        ).show()
                        return@combinedClickable
                    }
                    menuSnapshotId = null
                    menuExpanded = true
                    scope.launch {
                        val resolved = resolveSnapshotId(number)
                        if (resolved == null) {
                            menuExpanded = false
                            Toast.makeText(
                                context,
                                "Нет данных Device_Model для этого номера",
                                Toast.LENGTH_SHORT,
                            ).show()
                        } else {
                            menuSnapshotId = resolved
                        }
                    }
                },
            ),
    ) {
        Box {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    // Если номер из контактов — показываем имя («Мама»)
                    // вместо сырого +7…. PhoneLookup кэшируется, на пары
                    // десятков строк журнала задержка не заметна.
                    val displayName = remember(record.normalizedNumber) {
                        record.normalizedNumber
                            ?.let { ContactNameLookup.resolveOrNull(context, it) }
                            ?: record.originalNumber
                            ?: "Скрытый номер"
                    }
                    Text(
                        text = displayName,
                        style = MaterialTheme.typography.titleMedium,
                        color = TextPrimary
                    )
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        StatusPill(text = label, color = color)
                        if (record.ruleName != null) {
                            MonoLabelText(
                                text = "· ${record.ruleName}",
                                color = TextTertiary
                            )
                        }
                    }
                    // Subtitle: МТС/Москва или название организации из 2GIS
                    val infoSubtitle = callerInfo?.subtitle
                    if (infoSubtitle != null) {
                        MonoLabelText(
                            text = infoSubtitle,
                            color = TextTertiary
                        )
                    }
                    MonoLabelText(
                        text = dateFormat.format(Date(record.timestamp)),
                        color = TextTertiary
                    )
                }

                if (record.verdict != Verdict.BLOCK) {
                    IconButton(onClick = onBlock) {
                        Icon(
                            Icons.Rounded.AddCircle,
                            contentDescription = "В чёрный список",
                            tint = BlockRed
                        )
                    }
                }
                if (record.verdict != Verdict.ALLOW) {
                    IconButton(onClick = onAllow) {
                        Icon(
                            Icons.Rounded.CheckCircle,
                            contentDescription = "В белый список",
                            tint = AllowGreen
                        )
                    }
                }
            }

            DropdownMenu(
                expanded = menuExpanded,
                onDismissRequest = { menuExpanded = false },
            ) {
                val snapshotId = menuSnapshotId
                DropdownMenuItem(
                    text = { Text("Открыть детали") },
                    enabled = snapshotId != null,
                    onClick = {
                        menuExpanded = false
                        snapshotId?.let(onOpenDetail)
                    },
                )
                DropdownMenuItem(
                    text = { Text("Отметить как спам") },
                    enabled = snapshotId != null,
                    onClick = {
                        menuExpanded = false
                        snapshotId?.let(onMarkSpam)
                    },
                )
                DropdownMenuItem(
                    text = { Text("Отметить как легитимный") },
                    enabled = snapshotId != null,
                    onClick = {
                        menuExpanded = false
                        snapshotId?.let(onMarkLegit)
                    },
                )
            }
        }
    }
}
