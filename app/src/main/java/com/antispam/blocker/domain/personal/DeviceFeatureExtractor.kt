package com.antispam.blocker.domain.personal

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.telecom.Call
import android.telephony.TelephonyManager
import android.util.Log
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import com.antispam.blocker.data.cache.ContactsCache
import com.antispam.blocker.data.db.dao.CallEventDao
import com.antispam.blocker.data.db.dao.NotificationEventDao
import com.antispam.blocker.data.db.entity.CallEvent
import com.antispam.blocker.data.db.entity.NotificationEvent
import com.antispam.blocker.data.prefs.DeviceModelStore
import com.antispam.blocker.domain.scoring.RecentUserContextProvider
import kotlinx.coroutines.flow.first

/**
 * Builds a [DeviceFeatures] vector for the on-device personal classifier
 * (Device_Model). Runs on the call-screening hot path, so:
 *
 * - **Never throws**. Every per-source read is wrapped in a try/catch
 *   that downgrades any [SecurityException] (or unexpected [Throwable])
 *   to the neutral default value `0f`. The final `DeviceFeatures.values`
 *   array is always sized exactly [DeviceFeatures.SIZE].
 * - **Per-source gating** (Req 1.9, 1.10, 7.5). For every Telemetry_Source
 *   the extractor checks BOTH the corresponding `DeviceModelStore.source*Enabled`
 *   toggle AND the matching Android runtime permission / special access.
 *   If either is off/denied, the features driven by that source are left
 *   at `0f` (degraded mode). The model continues to predict on whatever
 *   sources remain available.
 *
 * Source ↔ feature index mapping (canonical order from
 * [DeviceFeatures.NAMES]):
 *
 * | # | Feature                            | Source                                     |
 * |---|------------------------------------|--------------------------------------------|
 * | 0 | is_contact                         | ContactsCache, READ_CONTACTS               |
 * | 1 | previously_rejected                | call_event 30d, READ_CALL_LOG              |
 * | 2 | is_night_time                      | TimeContext (no permission)                |
 * | 3 | is_weekend                         | TimeContext (no permission)                |
 * | 4 | prev_missed_no_callback_24h        | call_event 24h, READ_CALL_LOG              |
 * | 5 | prev_outgoing_after_missed         | call_event 7d, READ_CALL_LOG               |
 * | 6 | recent_bank_app_30m                | UsageStats, GET_USAGE_STATS                |
 * | 7 | recent_gov_app_30m                 | UsageStats, GET_USAGE_STATS                |
 * | 8 | recent_marketplace_app_30m         | UsageStats, GET_USAGE_STATS                |
 * | 9 | recent_messenger_app_30m           | UsageStats, GET_USAGE_STATS                |
 * |10 | notif_bank_recent_10m              | NotificationEventDao 10m, listener access  |
 * |11 | notif_marketplace_recent_10m       | NotificationEventDao 10m, listener access  |
 * |12 | same_carrier_as_user               | TelephonyManager, READ_PHONE_STATE         |
 * |13 | is_short_code                      | PhoneNormalizer (no permission)            |
 * |14 | same_prefix_call_count_7d_norm     | call_event 7d, READ_CALL_LOG               |
 * |15 | answer_rate_for_number_norm        | call_event 30d, READ_CALL_LOG              |
 * |16 | hidden_number                      | Call.Details / isHidden (no permission)    |
 *
 * Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 1.9, 1.10, 7.5.
 */
class DeviceFeatureExtractor(
    private val context: Context,
    private val callEventDao: CallEventDao,
    private val notificationEventDao: NotificationEventDao,
    private val store: DeviceModelStore,
    private val recentContextProvider: RecentUserContextProvider =
        RecentUserContextProvider(context),
) {

    /**
     * Builds a feature vector. Returns a non-null [DeviceFeatures] whose
     * `values.size == DeviceFeatures.SIZE` even when every Telemetry_Source
     * is disabled or denied.
     *
     * @param normalizedNumber E.164-normalized inbound number, or `null`
     *   for hidden / un-parsable numbers. Drives features 0, 1, 4, 5,
     *   13, 14, 15.
     * @param isHidden `true` when the inbound caller-ID was withheld.
     *   Maps directly onto feature #16 (`hidden_number`) regardless of
     *   any permission state — the value is observable from
     *   [Call.Details] without telephony permissions.
     * @param callDetails Optional Telecom call descriptor. Currently
     *   unused on the hot path beyond the [isHidden] derivation done by
     *   the caller, but kept in the signature for forward-compat with
     *   future call-detail-derived features.
     * @param now Wall-clock time in epoch millis, defaults to
     *   `System.currentTimeMillis()`. Injectable for tests.
     */
    suspend fun extract(
        normalizedNumber: String?,
        isHidden: Boolean,
        @Suppress("UNUSED_PARAMETER") callDetails: Call.Details?,
        now: Long = System.currentTimeMillis(),
    ): DeviceFeatures {
        val values = FloatArray(DeviceFeatures.SIZE) // zero-initialized — degraded default

        // ── Source toggles (DataStore) ────────────────────────────────────
        // Read each per-source toggle once. On any failure we conservatively
        // treat the source as disabled — degraded mode is always safe.
        val callLogEnabled = readToggle(store.sourceCallLogEnabledFlow.let { flow ->
            { flow.first() }
        }, default = DeviceModelStore.DEFAULT_SOURCE_CALL_LOG_ENABLED)
        val contactsEnabled = readToggle(store.sourceContactsEnabledFlow.let { flow ->
            { flow.first() }
        }, default = DeviceModelStore.DEFAULT_SOURCE_CONTACTS_ENABLED)
        val appUsageEnabled = readToggle(store.sourceAppUsageEnabledFlow.let { flow ->
            { flow.first() }
        }, default = DeviceModelStore.DEFAULT_SOURCE_APP_USAGE_ENABLED)
        val notificationsEnabled = readToggle(store.sourceNotificationsEnabledFlow.let { flow ->
            { flow.first() }
        }, default = DeviceModelStore.DEFAULT_SOURCE_NOTIFICATIONS_ENABLED)

        // ── Runtime permissions / special access ──────────────────────────
        val callLogGranted = hasPermission(Manifest.permission.READ_CALL_LOG)
        val contactsGranted = hasPermission(Manifest.permission.READ_CONTACTS)
        val phoneStateGranted = hasPermission(Manifest.permission.READ_PHONE_STATE)
        val usageAccessGranted = safeBool { recentContextProvider.isUsageAccessGranted() }
        val notificationListenerGranted = isNotificationListenerEnabled()

        // ── #2, #3: TimeContext (always available, no permission) ─────────
        val timeContext = TimeContext.derive(now)
        values[2] = if (timeContext.hour >= 22 || timeContext.hour < 8) 1f else 0f
        values[3] = if (timeContext.isWeekend) 1f else 0f

        // ── #16: hidden_number (always available — derived from Telecom
        //        callDetails by the caller, no permission required). ──────
        values[16] = if (isHidden) 1f else 0f

        // ── #13: is_short_code (always available — purely string logic). ──
        values[13] = if (isShortCode(normalizedNumber)) 1f else 0f

        // ── #0: is_contact ────────────────────────────────────────────────
        if (contactsEnabled && contactsGranted && !normalizedNumber.isNullOrBlank()) {
            try {
                if (ContactsCache.contains(normalizedNumber) == true) {
                    values[0] = 1f
                }
            } catch (e: SecurityException) {
                Log.w(TAG, "is_contact: SecurityException, defaulting to 0f", e)
            } catch (t: Throwable) {
                Log.w(TAG, "is_contact: failed, defaulting to 0f", t)
            }
        }

        // ── Call-log-driven features (#1, #4, #5, #14, #15) ───────────────
        // Single getByNumber() call shared across #1, #4, #5, #15 to avoid
        // hitting the DB four times on the hot path.
        if (callLogEnabled && callLogGranted && !normalizedNumber.isNullOrBlank()) {
            try {
                val history = callEventDao.getByNumber(normalizedNumber)
                values[1] = featurePreviouslyRejected(history, now)
                values[4] = featurePrevMissedNoCallback24h(history, now)
                values[5] = featurePrevOutgoingAfterMissed(history, now)
                values[15] = featureAnswerRateNorm(history, now)
            } catch (e: SecurityException) {
                Log.w(TAG, "call_log per-number aggregates: SecurityException, defaulting to 0f", e)
            } catch (t: Throwable) {
                Log.w(TAG, "call_log per-number aggregates: failed, defaulting to 0f", t)
            }

            // #14: same_prefix_call_count_7d_norm — separate prefix query.
            try {
                values[14] = featureSamePrefixCount7dNorm(normalizedNumber, now)
            } catch (e: SecurityException) {
                Log.w(TAG, "same_prefix_call_count_7d_norm: SecurityException, defaulting to 0f", e)
            } catch (t: Throwable) {
                Log.w(TAG, "same_prefix_call_count_7d_norm: failed, defaulting to 0f", t)
            }
        }

        // ── App-usage-driven features (#6–#9) ─────────────────────────────
        if (appUsageEnabled && usageAccessGranted) {
            try {
                val recent = recentContextProvider.getRecentContext()
                values[6] = if (recent.recentBankApp) 1f else 0f
                values[7] = if (recent.recentGovApp) 1f else 0f
                values[8] = if (recent.recentMarketplaceApp) 1f else 0f
                values[9] = if (recent.recentMessengerApp) 1f else 0f
            } catch (e: SecurityException) {
                Log.w(TAG, "recent_app: SecurityException, defaulting to 0f", e)
            } catch (t: Throwable) {
                Log.w(TAG, "recent_app: failed, defaulting to 0f", t)
            }
        }

        // ── Notification-driven features (#10, #11) ───────────────────────
        if (notificationsEnabled && notificationListenerGranted) {
            try {
                val sinceMs = now - NOTIF_WINDOW_MS
                val bankCount = notificationEventDao.countByCategoryWithin(
                    NotificationEvent.CategoryBucket.BANK, sinceMs, now
                )
                values[10] = if (bankCount > 0) 1f else 0f
                val marketCount = notificationEventDao.countByCategoryWithin(
                    NotificationEvent.CategoryBucket.MARKETPLACE, sinceMs, now
                )
                values[11] = if (marketCount > 0) 1f else 0f
            } catch (e: SecurityException) {
                Log.w(TAG, "notif_recent_10m: SecurityException, defaulting to 0f", e)
            } catch (t: Throwable) {
                Log.w(TAG, "notif_recent_10m: failed, defaulting to 0f", t)
            }
        }

        // ── #12: same_carrier_as_user ─────────────────────────────────────
        // Simplified heuristic per task spec: compare the user's SIM country
        // (from TelephonyManager) with the inbound number's country prefix.
        // For a +7… inbound number we treat `simCountryIso == "ru"` as "same
        // carrier country" — good enough for v1, since Cloud_Model already
        // owns finer-grained operator-bucket reasoning. Falls back to 0f on
        // SecurityException, missing TelephonyManager, or denied permission.
        if (phoneStateGranted) {
            try {
                values[12] = featureSameCarrierAsUser(normalizedNumber)
            } catch (e: SecurityException) {
                Log.w(TAG, "same_carrier_as_user: SecurityException, defaulting to 0f", e)
            } catch (t: Throwable) {
                Log.w(TAG, "same_carrier_as_user: failed, defaulting to 0f", t)
            }
        }

        return DeviceFeatures(values)
    }

    // ── Per-feature pure helpers ──────────────────────────────────────────

    private fun featurePreviouslyRejected(history: List<CallEvent>, now: Long): Float {
        val cutoff = now - WINDOW_30D_MS
        val any = history.any {
            it.state == CallEvent.CallState.REJECTED && it.startedAt >= cutoff
        }
        return if (any) 1f else 0f
    }

    private fun featurePrevMissedNoCallback24h(history: List<CallEvent>, now: Long): Float {
        val cutoff = now - WINDOW_24H_MS
        // Most-recent MISSED event inside the 24h window…
        val recentMissed = history.firstOrNull {
            it.state == CallEvent.CallState.MISSED && it.startedAt >= cutoff
        } ?: return 0f
        // …and no OUTGOING to the same number after that MISSED.
        val outgoingSince = history.any {
            it.direction == CallEvent.Direction.OUTGOING &&
                it.startedAt > recentMissed.startedAt
        }
        return if (!outgoingSince) 1f else 0f
    }

    private fun featurePrevOutgoingAfterMissed(history: List<CallEvent>, now: Long): Float {
        val cutoff = now - WINDOW_7D_MS
        // Any OUTGOING within last 7d that had an earlier MISSED/REJECTED
        // for the same number (any time, not just inside the 7d window —
        // the OUTGOING itself just has to be recent).
        val outgoings = history.filter {
            it.direction == CallEvent.Direction.OUTGOING && it.startedAt >= cutoff
        }
        if (outgoings.isEmpty()) return 0f
        val any = outgoings.any { out ->
            history.any { earlier ->
                earlier.startedAt < out.startedAt &&
                    (earlier.state == CallEvent.CallState.MISSED ||
                        earlier.state == CallEvent.CallState.REJECTED)
            }
        }
        return if (any) 1f else 0f
    }

    private fun featureAnswerRateNorm(history: List<CallEvent>, now: Long): Float {
        val cutoff = now - WINDOW_30D_MS
        val window = history.filter { it.startedAt >= cutoff }
        if (window.isEmpty()) return 0f
        val answered = window.count { it.state == CallEvent.CallState.ANSWERED }
        return (answered.toFloat() / window.size.toFloat()).coerceIn(0f, 1f)
    }

    private suspend fun featureSamePrefixCount7dNorm(
        normalized: String,
        now: Long,
    ): Float {
        // Prefix-5 includes the leading "+", e.g. "+7961" — matches the
        // bucketing used by FeatureExtractor.calculatePrefixCallFrequency7d.
        if (!normalized.startsWith("+") || normalized.length < 5) return 0f
        val prefix5 = normalized.take(5)
        val sinceMs = now - WINDOW_7D_MS
        val count = callEventDao.countByPrefixSince(prefix5, sinceMs)
        return (count.coerceAtMost(PREFIX_COUNT_NORM_CAP).toFloat() /
            PREFIX_COUNT_NORM_CAP.toFloat()).coerceIn(0f, 1f)
    }

    private fun featureSameCarrierAsUser(normalized: String?): Float {
        if (normalized.isNullOrBlank()) return 0f
        val tm = context.getSystemService(Context.TELEPHONY_SERVICE) as? TelephonyManager
            ?: return 0f
        // simCountryIso is the safest carrier-derived signal that does not
        // require READ_PHONE_NUMBERS or PRIVILEGED_PHONE_STATE on modern API
        // levels. Still gated on READ_PHONE_STATE upstream for consistency
        // with Req 1.9 and the Privacy section of the design doc.
        val simIso = (tm.simCountryIso ?: "").lowercase()
        // Simplified heuristic (documented): RU number ↔ RU SIM. Falls back
        // to 0f for any other country combination.
        return if (normalized.startsWith("+7") && simIso == "ru") 1f else 0f
    }

    // ── Permission + listener helpers ─────────────────────────────────────

    private fun hasPermission(permission: String): Boolean = try {
        ContextCompat.checkSelfPermission(context, permission) ==
            PackageManager.PERMISSION_GRANTED
    } catch (t: Throwable) {
        // Defensive — checkSelfPermission isn't documented to throw, but
        // we treat any failure as "not granted" to keep the hot path safe.
        Log.w(TAG, "checkSelfPermission($permission) failed; treating as denied", t)
        false
    }

    /**
     * Whether the user has granted Notification Listener Access to this
     * package (a special access, not a runtime permission). Returns
     * `false` on any failure — degraded mode is safe.
     */
    private fun isNotificationListenerEnabled(): Boolean = try {
        NotificationManagerCompat
            .getEnabledListenerPackages(context)
            .contains(context.packageName)
    } catch (t: Throwable) {
        Log.w(TAG, "isNotificationListenerEnabled check failed", t)
        false
    }

    private suspend inline fun readToggle(
        crossinline read: suspend () -> Boolean,
        default: Boolean,
    ): Boolean = try {
        read()
    } catch (t: Throwable) {
        Log.w(TAG, "DeviceModelStore toggle read failed; falling back to default=$default", t)
        default
    }

    private inline fun safeBool(read: () -> Boolean): Boolean = try {
        read()
    } catch (t: Throwable) {
        Log.w(TAG, "safeBool read failed", t)
        false
    }

    private fun isShortCode(normalized: String?): Boolean {
        if (normalized == null) return false
        if (normalized.length !in 2..6) return false
        return normalized.all { it.isDigit() }
    }

    companion object {
        private const val TAG = "DeviceFeatureExtractor"

        private const val WINDOW_24H_MS: Long = 24L * 60 * 60 * 1_000
        private const val WINDOW_7D_MS: Long = 7L * 24 * 60 * 60 * 1_000
        private const val WINDOW_30D_MS: Long = 30L * 24 * 60 * 60 * 1_000
        private const val NOTIF_WINDOW_MS: Long = 10L * 60 * 1_000

        /**
         * Cap used to normalize `same_prefix_call_count_7d_norm` into
         * `[0, 1]`. Any count above this value saturates at 1.0 — beyond
         * ~10 calls/week from the same DEF prefix, the SGD signal is
         * effectively unbounded which causes weight blow-up.
         */
        private const val PREFIX_COUNT_NORM_CAP: Int = 10
    }
}
