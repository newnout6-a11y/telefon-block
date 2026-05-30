package com.antispam.blocker.ui.theme

import android.app.Activity
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val AppDarkColorScheme = darkColorScheme(
    primary = Amber,
    onPrimary = TextOnAccent,
    primaryContainer = AmberContainer,
    onPrimaryContainer = AmberSoft,

    secondary = AmberSoft,
    onSecondary = TextOnAccent,
    secondaryContainer = InkSurface,
    onSecondaryContainer = TextPrimary,

    tertiary = AllowGreen,
    onTertiary = TextOnAccent,
    tertiaryContainer = AllowGreenSoft,
    onTertiaryContainer = AllowGreen,

    background = Ink,
    onBackground = TextPrimary,

    surface = InkElevated,
    onSurface = TextPrimary,
    surfaceVariant = InkSurface,
    onSurfaceVariant = TextSecondary,

    error = BlockRed,
    onError = TextOnAccent,
    errorContainer = BlockRedSoft,
    onErrorContainer = BlockRed,

    outline = InkBorder,
    outlineVariant = InkBorderBright,
)

@Composable
fun SpamBlockerTheme(
    content: @Composable () -> Unit
) {
    val colorScheme = AppDarkColorScheme

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            // Edge-to-edge: transparent system bars, dark icons off
            window.statusBarColor = Ink.toArgb()
            window.navigationBarColor = Ink.toArgb()
            val insetsController = WindowCompat.getInsetsController(window, view)
            insetsController.isAppearanceLightStatusBars = false
            insetsController.isAppearanceLightNavigationBars = false
            WindowCompat.setDecorFitsSystemWindows(window, false)
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = AppTypography,
        content = content
    )
}
