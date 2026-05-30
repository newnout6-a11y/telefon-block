package com.antispam.blocker.ui.screens

import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Add
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Lock
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.antispam.blocker.SpamBlockerApp
import com.antispam.blocker.data.assets.OfficialWhitelistDirectory
import com.antispam.blocker.data.repository.BlockListRepository
import com.antispam.blocker.ui.components.GlassCard
import com.antispam.blocker.ui.components.MonoLabelText
import com.antispam.blocker.ui.components.SectionHeader
import com.antispam.blocker.ui.components.StatusPill
import com.antispam.blocker.ui.theme.*
import com.antispam.blocker.util.PhoneNormalizer
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BlacklistScreen() {
    val app = SpamBlockerApp.instance
    val context = LocalContext.current
    val repo = remember {
        BlockListRepository(
            app.database.blockedNumberDao(),
            app.database.allowedNumberDao(),
            PhoneNormalizer,
            prebuiltReader = app.prebuiltBlocklistReader,
        )
    }
    val scope = rememberCoroutineScope()

    var showWhitelist by remember { mutableStateOf(false) }
    var newNumber by remember { mutableStateOf("") }
    var newWhite by remember { mutableStateOf("") }
    var isRegex by remember { mutableStateOf(false) }
    var officialExpanded by remember { mutableStateOf(false) }

    val blocked by repo.allBlocked.collectAsState(initial = emptyList())
    val allowed by repo.allAllowed.collectAsState(initial = emptyList())

    // Делим allowlist на «системный» (загруженный OfficialWhitelistImporter
    // из официального CSV — банки / операторы / экстренные службы — их
    // нельзя удалить, они импортируются автоматически) и «пользовательский»
    // (то, что юзер сам добавил или отметил «не спам»).
    val officialSet = remember { OfficialWhitelistDirectory.officialSet(context) }
    val userAllowed = allowed.filter { it.normalizedNumber !in officialSet }
    val officialAllowed = allowed.filter { it.normalizedNumber in officialSet }

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
                    eyebrow = "// LISTS",
                    title = if (showWhitelist) "Белый список" else "Чёрный список"
                )
                Spacer(Modifier.height(16.dp))

                // Segmented toggle
                SegmentedToggle(
                    isWhitelist = showWhitelist,
                    onChange = { showWhitelist = it }
                )
                Spacer(Modifier.height(16.dp))
            }

            if (!showWhitelist) {
                item {
                    // Информер про prebuilt-словарь: 2.4M номеров, упакованных
                    // как sqlite-asset, не лежат в Room — поэтому их не видно
                    // в списке ниже, но защита по ним работает на горячем пути.
                    PrebuiltInfoCard()
                    Spacer(Modifier.height(16.dp))

                    AddNumberCard(
                        value = newNumber,
                        onValueChange = { newNumber = it },
                        isRegex = isRegex,
                        onRegexChange = { isRegex = it },
                        onAdd = {
                            if (newNumber.isNotBlank()) {
                                scope.launch {
                                    val pattern = if (isRegex) {
                                        // Маска: оставляем только цифры.
                                        // Regex.containsMatchIn сам ищет подстроку,
                                        // так что "8800" совпадёт с "+78001234567".
                                        val digits = newNumber.filter { it.isDigit() }
                                        if (digits.isBlank()) null else digits
                                    } else null
                                    repo.addToBlockList(newNumber, pattern = pattern)
                                    newNumber = ""
                                }
                            }
                        }
                    )
                    Spacer(Modifier.height(16.dp))
                }
            }

            if (showWhitelist) {
                // Поле ввода для добавления в белый список
                item {
                    AddNumberCard(
                        value = newWhite,
                        onValueChange = { newWhite = it },
                        isRegex = false,
                        onRegexChange = {},
                        label = "НОМЕР ТЕЛЕФОНА",
                        placeholder = "+7 495 123-45-67",
                        addLabel = "Добавить номер",
                        onAdd = {
                            if (newWhite.isNotBlank()) {
                                scope.launch {
                                    repo.addToAllowList(newWhite)
                                    newWhite = ""
                                }
                            }
                        },
                        showMaskToggle = false,
                    )
                    Spacer(Modifier.height(16.dp))
                }

                // Ваши номера (можно удалять)
                item {
                    MonoLabelText(text = "ВАШИ НОМЕРА · ${userAllowed.size}", color = TextTertiary)
                    Spacer(Modifier.height(8.dp))
                }
                if (userAllowed.isEmpty()) {
                    item {
                        EmptyHintCard(
                            title = "Пусто",
                            subtitle = "Добавьте сюда номера, которые точно не должны блокироваться, или отметьте звонок в журнале как «не спам»."
                        )
                        Spacer(Modifier.height(16.dp))
                    }
                } else {
                    items(userAllowed, key = { "user-${it.normalizedNumber}" }) { item ->
                        ListRow(
                            original = item.originalNumber,
                            pattern = null,
                            badge = null,
                            accentColor = AllowGreen,
                            onRemove = {
                                scope.launch { repo.removeFromAllowList(item.normalizedNumber) }
                            },
                        )
                    }
                    item { Spacer(Modifier.height(16.dp)) }
                }

                // Официальный whitelist — свёрнут по умолчанию, удалять нельзя
                item {
                    OfficialWhitelistHeader(
                        count = officialAllowed.size,
                        expanded = officialExpanded,
                        onToggle = { officialExpanded = !officialExpanded },
                    )
                    Spacer(Modifier.height(8.dp))
                }
                if (officialExpanded) {
                    items(officialAllowed, key = { "official-${it.normalizedNumber}" }) { item ->
                        val name = OfficialWhitelistDirectory.nameFor(context, item.normalizedNumber)
                        val category = OfficialWhitelistDirectory.categoryFor(context, item.normalizedNumber)
                        ListRow(
                            original = if (name != null) "${item.originalNumber} · $name" else item.originalNumber,
                            pattern = null,
                            badge = category,
                            accentColor = AllowGreen,
                            onRemove = null, // системные удалять нельзя
                        )
                    }
                }
            } else {
                // Чёрный список: показываем только пользовательские/feedback/маски
                item {
                    MonoLabelText(text = "${blocked.size} entries", color = TextTertiary)
                    Spacer(Modifier.height(8.dp))
                }

                if (blocked.isEmpty()) {
                    item {
                        EmptyHintCard(
                            title = "Личных блокировок нет",
                            subtitle = "Встроенный словарь из 2.4 млн номеров уже работает — он защищает все звонки автоматически. Добавьте сюда свой номер, чтобы заблокировать его персонально."
                        )
                    }
                } else {
                    items(blocked, key = { it.normalizedNumber }) { item ->
                        ListRow(
                            original = item.originalNumber,
                            pattern = item.pattern,
                            badge = null,
                            accentColor = BlockRed,
                            onRemove = {
                                scope.launch { repo.removeFromBlockList(item.normalizedNumber) }
                            },
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun PrebuiltInfoCard() {
    GlassCard(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                Icons.Rounded.Lock,
                contentDescription = null,
                tint = Amber,
                modifier = Modifier.size(20.dp),
            )
            Spacer(Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Встроенный словарь",
                    style = MaterialTheme.typography.titleSmall,
                    color = TextPrimary,
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    text = "≈ 2 416 000 известных спам-номеров и 26 префиксов проверяются автоматически. Этот список нельзя редактировать; он живёт отдельно от ваших персональных блокировок.",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                )
            }
        }
    }
}

@Composable
private fun OfficialWhitelistHeader(
    count: Int,
    expanded: Boolean,
    onToggle: () -> Unit,
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        color = androidx.compose.ui.graphics.Color.Transparent,
        onClick = onToggle,
    ) {
        Row(
            modifier = Modifier.padding(vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                Icons.Rounded.Lock,
                contentDescription = null,
                tint = TextTertiary,
                modifier = Modifier.size(14.dp),
            )
            Spacer(Modifier.width(8.dp))
            MonoLabelText(
                text = "ОФИЦИАЛЬНЫЙ WHITELIST · $count",
                color = TextTertiary,
            )
            Spacer(Modifier.weight(1f))
            Text(
                text = if (expanded) "СКРЫТЬ" else "ПОКАЗАТЬ",
                style = MaterialTheme.typography.labelSmall,
                color = Amber,
            )
        }
    }
}

@Composable
private fun EmptyHintCard(title: String, subtitle: String) {
    GlassCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(text = title, style = MaterialTheme.typography.titleMedium, color = TextPrimary)
            Spacer(Modifier.height(4.dp))
            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
            )
        }
    }
}

@Composable
private fun SegmentedToggle(
    isWhitelist: Boolean,
    onChange: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(100.dp))
            .border(1.dp, InkBorder, RoundedCornerShape(100.dp))
            .padding(4.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        SegmentChip(
            label = "Чёрный",
            selected = !isWhitelist,
            onClick = { onChange(false) },
            modifier = Modifier.weight(1f)
        )
        SegmentChip(
            label = "Белый",
            selected = isWhitelist,
            onClick = { onChange(true) },
            modifier = Modifier.weight(1f)
        )
    }
}

@Composable
private fun SegmentChip(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(100.dp),
        color = if (selected) Amber else androidx.compose.ui.graphics.Color.Transparent,
        onClick = onClick
    ) {
        Box(
            modifier = Modifier.padding(vertical = 10.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.titleSmall,
                color = if (selected) TextOnAccent else TextSecondary
            )
        }
    }
}

@Composable
private fun AddNumberCard(
    value: String,
    onValueChange: (String) -> Unit,
    isRegex: Boolean,
    onRegexChange: (Boolean) -> Unit,
    onAdd: () -> Unit,
    label: String = if (isRegex) "МАСКА НОМЕРА" else "НОМЕР ТЕЛЕФОНА",
    placeholder: String = if (isRegex) "8800" else "+7 495 123-45-67",
    addLabel: String = if (isRegex) "Добавить маску" else "Добавить номер",
    showMaskToggle: Boolean = true,
) {
    GlassCard(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            MonoLabelText(text = label, color = TextTertiary)
            OutlinedTextField(
                value = value,
                onValueChange = { input ->
                    val filtered = if (isRegex) {
                        input.filter { it.isDigit() || it == '+' || it == '*' }
                    } else {
                        input.filter { it.isDigit() || it == '+' || it == ' ' || it == '-' || it == '(' || it == ')' }
                    }
                    onValueChange(filtered)
                },
                placeholder = { Text(text = placeholder, color = TextTertiary) },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Phone),
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Amber,
                    unfocusedBorderColor = InkBorder,
                    focusedTextColor = TextPrimary,
                    unfocusedTextColor = TextPrimary,
                    cursorColor = Amber
                ),
                shape = RoundedCornerShape(12.dp)
            )

            if (showMaskToggle) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Switch(
                        checked = isRegex,
                        onCheckedChange = onRegexChange,
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = TextOnAccent,
                            checkedTrackColor = Amber,
                            uncheckedThumbColor = TextSecondary,
                            uncheckedTrackColor = InkSurface,
                            uncheckedBorderColor = InkBorder
                        )
                    )
                    Spacer(Modifier.width(12.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Режим маски",
                            style = MaterialTheme.typography.titleSmall,
                            color = if (isRegex) Amber else TextPrimary
                        )
                        Text(
                            text = if (isRegex)
                                "Введите часть номера — заблокирует все номера, содержащие эти цифры. Например: 8800 заблокирует +78001234567, 78008889999 и т.п."
                            else
                                "Заблокировать один конкретный номер.",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSecondary
                        )
                    }
                }
            }

            Button(
                onClick = onAdd,
                enabled = value.isNotBlank(),
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Amber,
                    contentColor = TextOnAccent,
                    disabledContainerColor = InkSurface,
                    disabledContentColor = TextTertiary
                ),
                shape = RoundedCornerShape(12.dp)
            ) {
                Icon(Icons.Rounded.Add, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text(addLabel)
            }
        }
    }
}

@Composable
private fun ListRow(
    original: String,
    pattern: String?,
    badge: String?,
    accentColor: androidx.compose.ui.graphics.Color,
    onRemove: (() -> Unit)?,
) {
    GlassCard(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(
                    text = original,
                    style = MaterialTheme.typography.titleMedium,
                    color = TextPrimary
                )
                if (pattern != null) {
                    StatusPill(text = "маска", color = accentColor)
                } else if (badge != null) {
                    StatusPill(text = badge, color = accentColor)
                }
            }
            if (onRemove != null) {
                IconButton(onClick = onRemove) {
                    Icon(
                        Icons.Rounded.Close,
                        contentDescription = "Удалить",
                        tint = TextTertiary
                    )
                }
            } else {
                Icon(
                    Icons.Rounded.Lock,
                    contentDescription = "Системная запись",
                    tint = TextTertiary,
                    modifier = Modifier.size(18.dp),
                )
            }
        }
    }
}
