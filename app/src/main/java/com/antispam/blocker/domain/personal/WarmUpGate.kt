package com.antispam.blocker.domain.personal

import com.antispam.blocker.data.prefs.DeviceModelStore
import kotlinx.coroutines.flow.first

/**
 * Monotonic clock abstraction injected into [WarmUpGate] for testability.
 *
 * Production callers use [SystemClock]; tests substitute a deterministic fake so
 * the `now − installedAt ≥ 14 days` branch can be exercised without sleeping.
 */
fun interface Clock {
    fun now(): Long
}

/** Default [Clock] backed by [System.currentTimeMillis]. */
val SystemClock: Clock = Clock { System.currentTimeMillis() }

/**
 * Warm_Up_Window completion gate (Requirements 5.7–5.10).
 *
 * The Device_Model is silent while the warm-up window is active: `FusionDecider`
 * ignores its vote and falls back to Cloud_Model + rule-based heuristics
 * (Req 5.7, 5.8). The window completes as soon as ONE of the following holds
 * (whichever comes first, Req 5.9):
 *
 * - `now − installedAt ≥ 14 days` — calendar branch, anchored to install/wipe time
 *   stored as `installedAtFlow` in [DeviceModelStore].
 * - `labelCount ≥ 30` — label-count branch, total of Implicit_Label + Explicit_Label
 *   tracked via `labelCountFlow` in [DeviceModelStore].
 *
 * Once [isComplete] flips to `true`, `FusionDecider` enables the full fusion rule
 * with Device_Model participating (Req 5.10). After a Wipe, [DeviceModelStore.reset]
 * resets `installedAt := now()` and `labelCount := 0`, so the gate flips back to
 * `false` and the warm-up restarts.
 *
 * Note on the `installedAt = 0L` boundary: an unset value (`0L`) means warm-up
 * was never started. In that case `(now − 0) ≥ 14 days` always evaluates to
 * `true` on real devices (system clock is well past 1970+14d), so the formula
 * as written is spec-faithful — fresh installs must call [DeviceModelStore.reset]
 * (or seed `installedAt`) at first launch to anchor the window correctly.
 */
class WarmUpGate(
    private val store: DeviceModelStore,
    private val clock: Clock = SystemClock,
) {
    /**
     * Returns `true` iff the Warm_Up_Window is complete per Req 5.9 — either the
     * 14-day calendar threshold elapsed since install/wipe, or the user has
     * accumulated ≥ 30 labels. Reads each backing flow once via [first].
     */
    suspend fun isComplete(): Boolean {
        val installedAt = store.installedAtFlow.first()
        val labelCount = store.labelCountFlow.first()
        return (clock.now() - installedAt) >= WARMUP_DAYS_MS || labelCount >= WARMUP_LABELS
    }

    companion object {
        /** 14 calendar days in milliseconds — calendar branch threshold (Req 5.9). */
        const val WARMUP_DAYS_MS: Long = 14L * 24 * 60 * 60 * 1000

        /** Label-count branch threshold (Req 5.9): Implicit + Explicit labels combined. */
        const val WARMUP_LABELS: Int = 30
    }
}
