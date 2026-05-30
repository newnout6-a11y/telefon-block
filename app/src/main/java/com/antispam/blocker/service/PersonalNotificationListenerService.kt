package com.antispam.blocker.service

import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.util.Log
import com.antispam.blocker.SpamBlockerApp
import com.antispam.blocker.data.db.dao.NotificationEventDao
import com.antispam.blocker.data.db.entity.NotificationEvent
import com.antispam.blocker.data.prefs.DeviceModelStore
import com.antispam.blocker.domain.categorization.AppCategoryClassifierFactory
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

/**
 * `NotificationListenerService` for Device_Model features #10/#11
 * (`notif_bank_recent_10m`, `notif_marketplace_recent_10m`).
 *
 * **Privacy contract (Requirements 1.4, 1.5, 7.10).** This listener captures
 * ONLY:
 *  - [StatusBarNotification.getPackageName] — the source app package
 *  - a derived [NotificationEvent.CategoryBucket] (BANK / MARKETPLACE /
 *    MESSENGER / EMAIL / OTHER) computed locally from the package name
 *  - [StatusBarNotification.getPostTime] — the system post timestamp
 *
 * It MUST NEVER read `sbn.notification.extras`, in particular none of:
 *  - `Notification.EXTRA_TITLE`
 *  - `Notification.EXTRA_TEXT`
 *  - `Notification.EXTRA_BIG_TEXT`
 *  - `Notification.EXTRA_SUB_TEXT`
 *  - `Notification.EXTRA_SUMMARY_TEXT`
 *  - `Notification.EXTRA_INFO_TEXT`
 *
 * The notification body, title, ticker and any other content fields are
 * outside the scope of this feature and are explicitly forbidden by
 * Non-Goals (no SMS body / notification body reading) — see [requirements
 * 1.5](../../../../../../../.kiro/specs/on-device-personal-spam-classifier/requirements.md).
 *
 * **Gating (Requirement 7.4 / 7.5).** Each [onNotificationPosted] call
 * checks [DeviceModelStore.sourceNotificationsEnabledFlow]; if the user
 * has the `source_notifications_enabled` toggle off, the event is
 * dropped without insertion. Defaults to `false` (see
 * [DeviceModelStore.DEFAULT_SOURCE_NOTIFICATIONS_ENABLED]) — Notification
 * Listener Access requires explicit user-action in system settings.
 *
 * **Manifest registration is intentionally NOT done here**; that is
 * task 8.2 (separate change touching `AndroidManifest.xml`).
 */
class PersonalNotificationListenerService : NotificationListenerService() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private lateinit var store: DeviceModelStore
    private lateinit var dao: NotificationEventDao

    override fun onCreate() {
        super.onCreate()
        val app = applicationContext as SpamBlockerApp
        store = DeviceModelStore(applicationContext)
        dao = app.database.notificationEventDao()
    }

    override fun onNotificationPosted(sbn: StatusBarNotification?) {
        // Read ONLY packageName + postTime. Never `sbn.notification.extras`.
        val safe = sbn ?: return
        val pkg = safe.packageName ?: return
        val ts = safe.postTime

        scope.launch {
            // Per-source toggle (Req 7.4/7.5). If the user disabled the
            // notifications source, drop the event silently.
            val enabled = runCatching { store.sourceNotificationsEnabledFlow.first() }
                .getOrDefault(DeviceModelStore.DEFAULT_SOURCE_NOTIFICATIONS_ENABLED)
            if (!enabled) return@launch

            val bucket = bucketFor(pkg)
            try {
                dao.insert(
                    NotificationEvent(
                        packageName = pkg,
                        categoryBucket = bucket,
                        timestamp = ts
                    )
                )
            } catch (t: Throwable) {
                // Never crash the listener thread — Device_Model is best-effort.
                Log.w(TAG, "Failed to persist notification event for $pkg", t)
            }
        }
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "PersonalNotifListener"

        /**
         * Maps an Android package name to a coarse [NotificationEvent.CategoryBucket].
         *
         * Раньше тут жил локальный набор `BANK_PACKAGES / MARKETPLACE_PACKAGES /
         * MESSENGER_PACKAGES / EMAIL_PACKAGES` (~25 пакетов). Сейчас делегируем
         * единому [com.antispam.blocker.domain.categorization.AppCategoryClassifier],
         * который покрывает 18 категорий и ~150 точно известных российских
         * пакетов плюс substring-эвристики для остальных. Маппим 18-категорийное
         * пространство обратно на узкий 5-bucket enum (BANK / MARKETPLACE /
         * MESSENGER / EMAIL / OTHER), чтобы не ломать миграцию Room.
         */
        fun bucketFor(pkg: String): NotificationEvent.CategoryBucket {
            val category = AppCategoryClassifierFactory.classify(pkg)
            return when (category.toNotificationBucket()) {
                "BANK" -> NotificationEvent.CategoryBucket.BANK
                "MARKETPLACE" -> NotificationEvent.CategoryBucket.MARKETPLACE
                "MESSENGER" -> NotificationEvent.CategoryBucket.MESSENGER
                "EMAIL" -> NotificationEvent.CategoryBucket.EMAIL
                else -> NotificationEvent.CategoryBucket.OTHER
            }
        }
    }
}
