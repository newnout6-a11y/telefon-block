package com.antispam.blocker.ui.theme

import androidx.compose.ui.graphics.Color

// === Factory.ai-inspired palette ===
// Deep, near-black background with warm amber accents and muted neutrals.

// --- Surfaces ---
val Ink = Color(0xFF0A0A0B)           // app background
val InkElevated = Color(0xFF111113)    // elevated surface (cards)
val InkSurface = Color(0xFF161619)     // card surface
val InkBorder = Color(0xFF23232A)      // subtle borders
val InkBorderBright = Color(0xFF2E2E38) // hovered/active border

// --- Accent: warm amber (Factory signature) ---
val Amber = Color(0xFFFF6B1A)          // primary accent
val AmberSoft = Color(0xFFFFB070)      // soft accent text
val AmberGlow = Color(0x33FF6B1A)      // 20% glow for shadows/overlays
val AmberContainer = Color(0xFF2A1509) // tonal container

// --- Text ---
val TextPrimary = Color(0xFFF5F5F7)    // high-emphasis
val TextSecondary = Color(0xFFA0A0AB)  // medium-emphasis
val TextTertiary = Color(0xFF6B6B78)   // low-emphasis
val TextOnAccent = Color(0xFF0A0A0B)

// --- Verdict colors (muted, modern) ---
val BlockRed = Color(0xFFFF4D6D)       // rose-red for blocks
val WarnAmber = Color(0xFFFFB84D)      // warm amber for warnings
val AllowGreen = Color(0xFF4ADE80)     // emerald for allows

val BlockRedSoft = Color(0x1AFF4D6D)
val WarnAmberSoft = Color(0x1AFFB84D)
val AllowGreenSoft = Color(0x1A4ADE80)

// --- Gradient stops for hero sections ---
val GradientStart = Color(0xFF1A0F08)
val GradientMid = Color(0xFF0F0A08)
val GradientEnd = Color(0xFF0A0A0B)
