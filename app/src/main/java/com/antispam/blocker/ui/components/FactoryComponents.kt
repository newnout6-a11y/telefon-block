package com.antispam.blocker.ui.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.antispam.blocker.ui.theme.*

/**
 * Glassmorphism-style card with subtle border.
 * Mirrors factory.ai's card treatment: faint border, elevated ink surface.
 */
@Composable
fun GlassCard(
    modifier: Modifier = Modifier,
    accentBorder: Boolean = false,
    content: @Composable () -> Unit
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(20.dp),
        color = InkElevated,
        border = BorderStroke(
            width = 1.dp,
            color = if (accentBorder) Amber.copy(alpha = 0.35f) else InkBorder
        )
    ) {
        content()
    }
}

/**
 * Hero background: radial amber glow over ink gradient.
 * Use as top-of-screen accent panel.
 */
@Composable
fun HeroBackground(
    modifier: Modifier = Modifier,
    content: @Composable BoxScope.() -> Unit
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(24.dp))
            .background(
                Brush.verticalGradient(
                    0f to GradientStart,
                    0.6f to GradientMid,
                    1f to GradientEnd
                )
            ),
        content = content
    )
}

/**
 * Compact mono-label: uppercase tracked monospace text used for technical labels.
 * Factory.ai uses this for system-level descriptors.
 */
@Composable
fun MonoLabelText(
    text: String,
    modifier: Modifier = Modifier,
    color: Color = TextTertiary
) {
    Text(
        text = text.uppercase(),
        style = MonoLabel,
        color = color,
        modifier = modifier
    )
}

/**
 * Pill-shaped status badge. Used across screens to indicate verdict / state.
 */
@Composable
fun StatusPill(
    text: String,
    color: Color,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(100.dp),
        color = color.copy(alpha = 0.12f),
        border = BorderStroke(1.dp, color.copy(alpha = 0.35f))
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(6.dp)
                    .clip(androidx.compose.foundation.shape.CircleShape)
                    .background(color)
            )
            Text(
                text = text.uppercase(),
                color = color,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Medium,
                fontSize = 10.sp,
                letterSpacing = 1.sp
            )
        }
    }
}

/**
 * Large metric counter, monospaced for tech feel.
 */
@Composable
fun MetricCounter(
    value: Int,
    label: String,
    color: Color,
    modifier: Modifier = Modifier
) {
    GlassCard(modifier = modifier) {
        Column(
            modifier = Modifier.fillMaxWidth().padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(6.dp)
                        .clip(androidx.compose.foundation.shape.CircleShape)
                        .background(color)
                )
                MonoLabelText(text = label, modifier = Modifier.weight(1f))
            }
            Text(
                text = value.toString().padStart(2, '0'),
                style = MonoDisplay,
                color = color,
                modifier = Modifier.fillMaxWidth()
            )
        }
    }
}

/**
 * Section header with tracked mono title.
 */
@Composable
fun SectionHeader(
    eyebrow: String,
    title: String,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        MonoLabelText(text = eyebrow, color = Amber)
        Text(
            text = title,
            style = MaterialTheme.typography.headlineMedium,
            color = TextPrimary
        )
    }
}
