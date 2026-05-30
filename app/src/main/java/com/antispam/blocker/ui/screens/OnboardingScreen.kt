package com.antispam.blocker.ui.screens

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ArrowForward
import androidx.compose.material.icons.rounded.Check
import androidx.compose.material.icons.rounded.Contacts
import androidx.compose.material.icons.rounded.Notifications
import androidx.compose.material.icons.rounded.PhoneInTalk
import androidx.compose.material.icons.rounded.Security
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.antispam.blocker.ui.components.GlassCard
import com.antispam.blocker.ui.components.MonoLabelText
import com.antispam.blocker.ui.theme.*
import com.antispam.blocker.util.RoleManagerHelper

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun OnboardingScreen(onComplete: () -> Unit) {
    val context = LocalContext.current
    var currentStep by remember { mutableIntStateOf(0) }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { _ ->
        currentStep++
    }

    val roleLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { _ ->
        onComplete()
    }

    val steps = listOf(
        OnboardingStep(Icons.Rounded.Contacts, "Доступ к контактам", "Чтобы разрешять звонки от ваших контактов", Manifest.permission.READ_CONTACTS),
        OnboardingStep(Icons.Rounded.PhoneInTalk, "Доступ к журналу звонков", "Для отображения заблокированных звонков", Manifest.permission.READ_CALL_LOG),
        OnboardingStep(Icons.Rounded.Notifications, "Уведомления", "Для предупреждений о подозрительных звонках", Manifest.permission.POST_NOTIFICATIONS),
        OnboardingStep(Icons.Rounded.Security, "Фильтр звонков", "Назначить приложение для фильтрации входящих звонков", null),
        OnboardingStep(Icons.Rounded.Settings, "Плашка поверх звонка", "Чтобы предупреждение появлялось поверх экрана входящего вызова. Включите «Поверх других приложений» в системных настройках.", "OVERLAY_PERMISSION"),
        OnboardingStep(Icons.Rounded.Settings, "Ограниченные настройки", "Если переключатели серые: Настройки → Приложения → Блокировщик спама → ⋮ → Разрешить ограниченные настройки", "RESTRICTED_SETTINGS_INFO")
    )

    val infinite = rememberInfiniteTransition(label = "onboard_pulse")
    val glowScale by infinite.animateFloat(
        initialValue = 0.9f,
        targetValue = 1.1f,
        animationSpec = infiniteRepeatable(
            animation = tween(2800, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "glow"
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
                .padding(horizontal = 24.dp, vertical = 32.dp)
        ) {
            // --- Hero brand ---
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(220.dp),
                contentAlignment = Alignment.Center
            ) {
                Box(
                    modifier = Modifier
                        .size(220.dp)
                        .scale(glowScale)
                        .background(
                            Brush.radialGradient(
                                0f to AmberGlow,
                                1f to Color.Transparent
                            ),
                            shape = CircleShape
                        )
                )
                Box(
                    modifier = Modifier
                        .size(96.dp)
                        .clip(CircleShape)
                        .background(Amber.copy(alpha = 0.15f))
                        .border(1.dp, Amber.copy(alpha = 0.4f), CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Rounded.Security,
                        contentDescription = null,
                        tint = Amber,
                        modifier = Modifier.size(44.dp)
                    )
                }
            }

            Spacer(Modifier.height(12.dp))

            MonoLabelText(
                text = "SETUP · STEP ${(currentStep + 1).coerceAtMost(steps.size)} / ${steps.size}",
                color = Amber
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = "Добро пожаловать",
                style = MaterialTheme.typography.displaySmall,
                color = TextPrimary
            )
            Spacer(Modifier.height(6.dp))
            Text(
                text = "Настроим блокировщик спама за минуту. Все данные остаются на устройстве.",
                style = MaterialTheme.typography.bodyMedium,
                color = TextSecondary
            )

            Spacer(Modifier.height(28.dp))

            // Progress track
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                steps.forEachIndexed { index, _ ->
                    val done = index < currentStep
                    val current = index == currentStep
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(3.dp)
                            .clip(RoundedCornerShape(100.dp))
                            .background(
                                when {
                                    done -> Amber
                                    current -> Amber.copy(alpha = 0.45f)
                                    else -> InkBorder
                                }
                            )
                    )
                }
            }

            Spacer(Modifier.height(20.dp))

            // Steps list
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                steps.forEachIndexed { index, step ->
                    val isDone = index < currentStep
                    val isCurrent = index == currentStep
                    StepRow(
                        step = step,
                        isDone = isDone,
                        isCurrent = isCurrent,
                        number = index + 1
                    )
                }
            }

            Spacer(Modifier.height(24.dp))

            // CTA
            val step = steps.getOrNull(currentStep)
            Button(
                onClick = {
                    val s = steps.getOrNull(currentStep) ?: run { onComplete(); return@Button }
                    when {
                        s.permission == "RESTRICTED_SETTINGS_INFO" -> {
                            // Open the app's system details page so the user can find the ⋮
                            // menu in one tap instead of navigating through Settings →
                            // Приложения → Блокировщик спама by hand. After returning from
                            // this activity onboarding completes.
                            val intent = Intent(
                                Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                                android.net.Uri.parse("package:${context.packageName}")
                            ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                            try { context.startActivity(intent) } catch (_: Exception) {}
                            currentStep++
                            if (currentStep >= steps.size) onComplete()
                        }
                        s.permission == "OVERLAY_PERMISSION" -> {
                            if (Settings.canDrawOverlays(context)) {
                                currentStep++
                            } else {
                                val intent = Intent(
                                    Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                                    android.net.Uri.parse("package:${context.packageName}")
                                ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                                try { context.startActivity(intent) } catch (_: Exception) {}
                                currentStep++
                            }
                            if (currentStep >= steps.size) onComplete()
                        }
                        s.permission != null -> {
                            val needed = mutableListOf<String>()
                            if (ContextCompat.checkSelfPermission(context, s.permission)
                                != PackageManager.PERMISSION_GRANTED
                            ) {
                                needed.add(s.permission)
                            }
                            if (needed.isEmpty()) {
                                currentStep++
                            } else {
                                permissionLauncher.launch(needed.toTypedArray())
                            }
                        }
                        else -> {
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                                RoleManagerHelper.requestCallScreeningRole(roleLauncher, context)
                            } else {
                                onComplete()
                            }
                        }
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                shape = RoundedCornerShape(14.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Amber,
                    contentColor = TextOnAccent
                )
            ) {
                Text(
                    when {
                        currentStep >= steps.size -> "Готово"
                        step?.permission == "RESTRICTED_SETTINGS_INFO" -> "Открыть настройки приложения"
                        step?.permission == "OVERLAY_PERMISSION" -> "Открыть настройки"
                        step?.permission != null -> "Разрешить"
                        else -> "Назначить фильтром"
                    },
                    style = MaterialTheme.typography.titleMedium
                )
                Spacer(Modifier.width(8.dp))
                Icon(Icons.Rounded.ArrowForward, contentDescription = null, modifier = Modifier.size(18.dp))
            }

            AnimatedVisibility(
                visible = currentStep > 0 && currentStep < steps.size,
                enter = fadeIn(),
                exit = fadeOut()
            ) {
                TextButton(
                    onClick = { onComplete() },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 8.dp),
                    colors = ButtonDefaults.textButtonColors(contentColor = TextTertiary)
                ) {
                    Text("Пропустить настройку")
                }
            }
        }
    }
}

@Composable
private fun StepRow(
    step: OnboardingStep,
    isDone: Boolean,
    isCurrent: Boolean,
    number: Int
) {
    GlassCard(
        modifier = Modifier.fillMaxWidth(),
        accentBorder = isCurrent
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(
                        when {
                            isDone -> Amber.copy(alpha = 0.15f)
                            isCurrent -> Amber.copy(alpha = 0.08f)
                            else -> InkSurface
                        }
                    )
                    .border(
                        width = 1.dp,
                        color = when {
                            isDone -> Amber.copy(alpha = 0.5f)
                            isCurrent -> Amber.copy(alpha = 0.35f)
                            else -> InkBorder
                        },
                        shape = CircleShape
                    ),
                contentAlignment = Alignment.Center
            ) {
                if (isDone) {
                    Icon(
                        imageVector = Icons.Rounded.Check,
                        contentDescription = null,
                        tint = Amber,
                        modifier = Modifier.size(20.dp)
                    )
                } else {
                    Icon(
                        imageVector = step.icon,
                        contentDescription = null,
                        tint = if (isCurrent) Amber else TextTertiary,
                        modifier = Modifier.size(20.dp)
                    )
                }
            }

            Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                MonoLabelText(
                    text = "${number.toString().padStart(2, '0')}",
                    color = when {
                        isDone -> Amber
                        isCurrent -> AmberSoft
                        else -> TextTertiary
                    }
                )
                Text(
                    text = step.title,
                    style = MaterialTheme.typography.titleMedium,
                    color = if (isCurrent || isDone) TextPrimary else TextSecondary
                )
                Text(
                    text = step.description,
                    style = MaterialTheme.typography.bodySmall,
                    color = TextTertiary
                )
            }
        }
    }
}

private data class OnboardingStep(
    val icon: ImageVector,
    val title: String,
    val description: String,
    val permission: String?
)
