package com.antispam.blocker.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ArrowForward
import androidx.compose.material.icons.rounded.Check
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.antispam.blocker.SpamBlockerApp
import com.antispam.blocker.domain.scoring.*
import com.antispam.blocker.ui.components.GlassCard
import com.antispam.blocker.ui.components.MonoLabelText
import com.antispam.blocker.ui.theme.*
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun QuestionnaireScreen(onComplete: () -> Unit) {
    val app = SpamBlockerApp.instance
    val scope = rememberCoroutineScope()
    var currentQuestion by remember { mutableIntStateOf(0) }
    val answers = remember { mutableStateOf(QuestionnaireAnswers()) }

    val questions = remember { buildQuestions() }
    val totalQuestions = questions.size

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
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Header
            MonoLabelText(
                text = "PROFILE · QUESTION ${(currentQuestion + 1).coerceAtMost(totalQuestions)} / $totalQuestions",
                color = Amber
            )
            Text(
                text = "Расскажите о себе",
                style = MaterialTheme.typography.headlineMedium,
                color = TextPrimary
            )
            Text(
                text = "Ответы помогут настроить защиту под вас. Все данные остаются на устройстве.",
                style = MaterialTheme.typography.bodyMedium,
                color = TextSecondary
            )

            // Progress bar
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                repeat(totalQuestions) { index ->
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(3.dp)
                            .clip(RoundedCornerShape(100.dp))
                            .background(
                                when {
                                    index < currentQuestion -> Amber
                                    index == currentQuestion -> Amber.copy(alpha = 0.45f)
                                    else -> InkBorder
                                }
                            )
                    )
                }
            }

            Spacer(Modifier.height(8.dp))

            // Current question
            val q = questions.getOrNull(currentQuestion)
            if (q != null) {
                GlassCard(modifier = Modifier.fillMaxWidth(), accentBorder = true) {
                    Column(
                        modifier = Modifier.padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        MonoLabelText(text = q.category, color = TextTertiary)
                        Text(
                            text = q.text,
                            style = MaterialTheme.typography.titleLarge,
                            color = TextPrimary
                        )
                        if (q.subtitle != null) {
                            Text(
                                text = q.subtitle,
                                style = MaterialTheme.typography.bodySmall,
                                color = TextSecondary
                            )
                        }

                        // Options
                        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            q.options.forEach { option ->
                                val selected = q.isSelected(answers.value)
                                val isThisSelected = selected == option.value
                                Surface(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .border(
                                            width = 1.dp,
                                            color = if (isThisSelected) Amber else InkBorder,
                                            shape = RoundedCornerShape(12.dp)
                                        ),
                                    shape = RoundedCornerShape(12.dp),
                                    color = if (isThisSelected) Amber.copy(alpha = 0.08f) else InkSurface,
                                    onClick = {
                                        q.onSelect(answers, option.value)
                                    }
                                ) {
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(horizontal = 16.dp, vertical = 12.dp),
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.SpaceBetween
                                    ) {
                                        Text(
                                            text = option.label,
                                            style = MaterialTheme.typography.bodyLarge,
                                            color = if (isThisSelected) Amber else TextPrimary
                                        )
                                        if (isThisSelected) {
                                            Icon(
                                                Icons.Rounded.Check,
                                                contentDescription = null,
                                                tint = Amber,
                                                modifier = Modifier.size(18.dp)
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Spacer(Modifier.height(8.dp))

            // Navigation buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                if (currentQuestion > 0) {
                    OutlinedButton(
                        onClick = { currentQuestion-- },
                        modifier = Modifier.weight(1f).height(52.dp),
                        shape = RoundedCornerShape(14.dp),
                        border = androidx.compose.foundation.BorderStroke(1.dp, InkBorder),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = TextPrimary)
                    ) {
                        Text("Назад")
                    }
                }

                Button(
                    onClick = {
                        if (currentQuestion < totalQuestions - 1) {
                            currentQuestion++
                        } else {
                            scope.launch {
                                val vector = UserProfileVector.fromQuestionnaire(answers.value)
                                val enriched = InstalledAppScanner(app).enrichProfile(vector)
                                app.updateProfileVector(enriched)
                                app.profileVectorStore.saveFromQuestionnaire(answers.value)
                                onComplete()
                            }
                        }
                    },
                    modifier = Modifier
                        .weight(1f)
                        .height(52.dp),
                    shape = RoundedCornerShape(14.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Amber,
                        contentColor = TextOnAccent
                    )
                ) {
                    Text(
                        if (currentQuestion < totalQuestions - 1) "Далее" else "Завершить",
                        style = MaterialTheme.typography.titleMedium
                    )
                    Spacer(Modifier.width(6.dp))
                    Icon(Icons.Rounded.ArrowForward, contentDescription = null, modifier = Modifier.size(18.dp))
                }
            }

            if (currentQuestion > 0) {
                TextButton(
                    onClick = { onComplete() },
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.textButtonColors(contentColor = TextTertiary)
                ) {
                    Text("Пропустить опрос")
                }
            }
        }
    }
}

private data class QuestionOption(val label: String, val value: String)

private data class QuestionDef(
    val category: String,
    val text: String,
    val subtitle: String? = null,
    val options: List<QuestionOption>,
    val isSelected: (QuestionnaireAnswers) -> String?,
    val onSelect: (MutableState<QuestionnaireAnswers>, String) -> Unit
)

private fun buildQuestions(): List<QuestionDef> = listOf(
    // 1. Возраст
    QuestionDef(
        category = "КТО ВЫ",
        text = "Ваш возраст?",
        options = listOf(
            QuestionOption("До 25", "under_25"),
            QuestionOption("25–40", "25_40"),
            QuestionOption("40–60", "40_60"),
            QuestionOption("60+", "60_plus")
        ),
        isSelected = { a -> a.age.name },
        onSelect = { a, v -> a.value = a.value.copy(age = AgeRange.valueOf(v.uppercase().replace("__", "_").replace("+", "_PLUS"))) }
    ),
    // 2. Род деятельности
    QuestionDef(
        category = "КТО ВЫ",
        text = "Основной род деятельности?",
        options = listOf(
            QuestionOption("Работник офиса", "OFFICE"),
            QuestionOption("Фрилансер", "FREELANCE"),
            QuestionOption("Предприниматель", "ENTREPRENEUR"),
            QuestionOption("Пенсионер", "PENSIONER"),
            QuestionOption("Студент", "STUDENT"),
            QuestionOption("Другое", "OTHER")
        ),
        isSelected = { a -> a.occupation.name },
        onSelect = { a, v -> a.value = a.value.copy(occupation = Occupation.valueOf(v)) }
    ),
    // 3. Звонки по работе
    QuestionDef(
        category = "КТО ВЫ",
        text = "Получаете ли много звонков от незнакомых номеров по работе?",
        options = listOf(
            QuestionOption("Да", "YES"),
            QuestionOption("Иногда", "SOMETIMES"),
            QuestionOption("Нет", "NO")
        ),
        isSelected = { a -> a.workCalls.name },
        onSelect = { a, v -> a.value = a.value.copy(workCalls = CallFrequency.valueOf(v)) }
    ),
    // 4. Мобильный банкинг
    QuestionDef(
        category = "ДЕНЬГИ И БАНКИ",
        text = "Пользуетесь мобильным банкингом?",
        options = listOf(
            QuestionOption("Да", "yes"),
            QuestionOption("Нет", "no")
        ),
        isSelected = { a -> if (a.mobileBanking) "yes" else "no" },
        onSelect = { a, v -> a.value = a.value.copy(mobileBanking = v == "yes") }
    ),
    // 5. Количество банковских приложений
    QuestionDef(
        category = "ДЕНЬГИ И БАНКИ",
        text = "Сколько банковских приложений установлено?",
        options = listOf(
            QuestionOption("Ни одного", "ZERO"),
            QuestionOption("1–2", "ONE_TWO"),
            QuestionOption("3 и больше", "THREE_PLUS")
        ),
        isSelected = { a -> a.bankAppCount.name },
        onSelect = { a, v -> a.value = a.value.copy(bankAppCount = BankAppCount.valueOf(v)) }
    ),
    // 6. Покупки онлайн
    QuestionDef(
        category = "ДЕНЬГИ И БАНКИ",
        text = "Как часто покупаете онлайн?",
        options = listOf(
            QuestionOption("Часто", "OFTEN"),
            QuestionOption("Иногда", "SOMETIMES"),
            QuestionOption("Редко", "RARELY"),
            QuestionOption("Нет", "NO")
        ),
        isSelected = { a -> a.onlinePurchases.name },
        onSelect = { a, v -> a.value = a.value.copy(onlinePurchases = PurchaseFrequency.valueOf(v)) }
    ),
    // 7. Маркетплейсы
    QuestionDef(
        category = "ПОКУПКИ И ОБЪЯВЛЕНИЯ",
        text = "Пользуетесь маркетплейсами (Wildberries, Ozon)?",
        options = listOf(
            QuestionOption("Да", "yes"),
            QuestionOption("Нет", "no")
        ),
        isSelected = { a -> if (a.usesMarketplaces) "yes" else "no" },
        onSelect = { a, v -> a.value = a.value.copy(usesMarketplaces = v == "yes") }
    ),
    // 8. Авито/объявления
    QuestionDef(
        category = "ПОКУПКИ И ОБЪЯВЛЕНИЯ",
        text = "Размещаете объявления на Авито или Юле?",
        options = listOf(
            QuestionOption("Часто", "OFTEN"),
            QuestionOption("Иногда", "SOMETIMES"),
            QuestionOption("Нет", "NO")
        ),
        isSelected = { a -> a.adsPosting.name },
        onSelect = { a, v -> a.value = a.value.copy(adsPosting = PostFrequency.valueOf(v)) }
    ),
    // 9. Доставка еды
    QuestionDef(
        category = "ПОКУПКИ И ОБЪЯВЛЕНИЯ",
        text = "Заказываете доставку еды?",
        options = listOf(
            QuestionOption("Часто", "OFTEN"),
            QuestionOption("Иногда", "SOMETIMES"),
            QuestionOption("Нет", "NO")
        ),
        isSelected = { a -> a.deliveryUsage.name },
        onSelect = { a, v -> a.value = a.value.copy(deliveryUsage = PostFrequency.valueOf(v)) }
    ),
    // 10. WhatsApp
    QuestionDef(
        category = "МЕССЕНДЖЕРЫ",
        text = "Пользуетесь WhatsApp?",
        options = listOf(
            QuestionOption("Да", "yes"),
            QuestionOption("Нет", "no")
        ),
        isSelected = { a -> if (a.usesWhatsApp) "yes" else "no" },
        onSelect = { a, v -> a.value = a.value.copy(usesWhatsApp = v == "yes") }
    ),
    // 11. Telegram
    QuestionDef(
        category = "МЕССЕНДЖЕРЫ",
        text = "Пользуетесь Telegram?",
        options = listOf(
            QuestionOption("Да", "yes"),
            QuestionOption("Нет", "no")
        ),
        isSelected = { a -> if (a.usesTelegram) "yes" else "no" },
        onSelect = { a, v -> a.value = a.value.copy(usesTelegram = v == "yes") }
    ),
    // 12. Звонки от незнакомцев в мессенджерах
    QuestionDef(
        category = "МЕССЕНДЖЕРЫ",
        text = "Звонят ли незнакомые люди в мессенджерах?",
        options = listOf(
            QuestionOption("Да", "YES"),
            QuestionOption("Иногда", "SOMETIMES"),
            QuestionOption("Нет", "NO")
        ),
        isSelected = { a -> a.messengerStrangerCalls.name },
        onSelect = { a, v -> a.value = a.value.copy(messengerStrangerCalls = CallFrequency.valueOf(v)) }
    ),
    // 13. Заграничные контакты
    QuestionDef(
        category = "СВЯЗИ",
        text = "Звонят ли вам из-за границы?",
        options = listOf(
            QuestionOption("Часто", "OFTEN"),
            QuestionOption("Редко", "RARELY"),
            QuestionOption("Нет", "NO")
        ),
        isSelected = { a -> a.foreignContacts.name },
        onSelect = { a, v -> a.value = a.value.copy(foreignContacts = ForeignContacts.valueOf(v)) }
    ),
    // 14. Домашний телефон
    QuestionDef(
        category = "СВЯЗИ",
        text = "Есть ли у вас домашний (городской) телефон?",
        options = listOf(
            QuestionOption("Да", "yes"),
            QuestionOption("Нет", "no")
        ),
        isSelected = { a -> if (a.hasHomePhone) "yes" else "no" },
        onSelect = { a, v -> a.value = a.value.copy(hasHomePhone = v == "yes") }
    ),
    // 15. Частота спама
    QuestionDef(
        category = "ОПЫТ СО СПАМОМ",
        text = "Как часто звонят спамеры?",
        options = listOf(
            QuestionOption("Каждый день", "EVERY_DAY"),
            QuestionOption("Несколько в неделю", "SEVERAL_WEEK"),
            QuestionOption("Редко", "RARELY"),
            QuestionOption("Почти никогда", "ALMOST_NONE")
        ),
        isSelected = { a -> a.spamFrequency.name },
        onSelect = { a, v -> a.value = a.value.copy(spamFrequency = SpamFrequency.valueOf(v)) }
    ),
    // 16. Опыт мошенничества
    QuestionDef(
        category = "ОПЫТ СО СПАМОМ",
        text = "Сталкивались ли с телефонным мошенничеством?",
        subtitle = "Ваш ответ поможет настроить чувствительность защиты",
        options = listOf(
            QuestionOption("Да, потерял деньги", "LOST_MONEY"),
            QuestionOption("Чуть не попался, но вовремя понял", "CAUGHT_IN_TIME"),
            QuestionOption("Нет", "NO")
        ),
        isSelected = { a -> a.scamExperience.name },
        onSelect = { a, v -> a.value = a.value.copy(scamExperience = ScamExperience.valueOf(v)) }
    ),
    // 17. Осведомлённость
    QuestionDef(
        category = "ОПЫТ СО СПАМОМ",
        text = "Насколько вы разбираетесь в телефонных мошенничествах?",
        options = listOf(
            QuestionOption("Хорошо знаю схемы", "GOOD"),
            QuestionOption("Поверхностно", "SUPERFICIAL"),
            QuestionOption("Почти ничего не знаю", "NONE")
        ),
        isSelected = { a -> a.scamAwareness.name },
        onSelect = { a, v -> a.value = a.value.copy(scamAwareness = AwarenessLevel.valueOf(v)) }
    ),
    // 18. Приоритет защиты
    QuestionDef(
        category = "НАСТРОЙКА ЗАЩИТЫ",
        text = "Что важнее?",
        options = listOf(
            QuestionOption("Не пропустить важный звонок", "DONT_MISS"),
            QuestionOption("Баланс", "BALANCE"),
            QuestionOption("Не слышать спам", "NO_SPAM")
        ),
        isSelected = { a -> a.protectionPriority.name },
        onSelect = { a, v -> a.value = a.value.copy(protectionPriority = ProtectionPriority.valueOf(v)) }
    ),
    // 19. Ответ на незнакомые номера
    QuestionDef(
        category = "НАСТРОЙКА ЗАЩИТЫ",
        text = "Обычно отвечаете на звонки с незнакомых номеров?",
        options = listOf(
            QuestionOption("Всегда", "ALWAYS"),
            QuestionOption("Иногда", "SOMETIMES"),
            QuestionOption("Никогда", "NEVER")
        ),
        isSelected = { a -> a.answerStrangers.name },
        onSelect = { a, v -> a.value = a.value.copy(answerStrangers = AnswerStrangers.valueOf(v)) }
    ),
    // 20. Автоблокировка
    QuestionDef(
        category = "НАСТРОЙКА ЗАЩИТЫ",
        text = "Как поступать с подозрительными звонками?",
        options = listOf(
            QuestionOption("Блокировать автоматически", "BLOCK"),
            QuestionOption("Показать предупреждение", "WARN"),
            QuestionOption("Решать каждый раз сам", "DECIDE_EACH")
        ),
        isSelected = { a -> a.autoBlockPreference.name },
        onSelect = { a, v -> a.value = a.value.copy(autoBlockPreference = AutoBlockPreference.valueOf(v)) }
    )
)
