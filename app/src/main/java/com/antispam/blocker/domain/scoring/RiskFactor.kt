package com.antispam.blocker.domain.scoring

data class RiskFactor(
    val id: String,
    val displayName: String,
    val points: Int,
    val reason: String,
    val weight: Float = 1.0f,
    val isActive: Boolean = true
)
