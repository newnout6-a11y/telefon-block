package com.antispam.blocker.domain.scoring

import android.app.AppOpsManager
import android.app.usage.UsageEvents
import android.app.usage.UsageStatsManager
import android.content.Context
import android.os.Process
import android.util.Log
import com.antispam.blocker.domain.categorization.AppCategory
import com.antispam.blocker.domain.categorization.AppCategoryClassifierFactory

class RecentUserContextProvider(private val context: Context) {

    data class RecentContext(
        val recentBankApp: Boolean = false,
        val recentGovApp: Boolean = false,
        val recentMarketplaceApp: Boolean = false,
        val recentMessengerApp: Boolean = false
    )

    fun isUsageAccessGranted(): Boolean {
        val appOps = context.getSystemService(Context.APP_OPS_SERVICE) as AppOpsManager
        val mode = appOps.checkOpNoThrow(
            AppOpsManager.OPSTR_GET_USAGE_STATS,
            Process.myUid(),
            context.packageName
        )
        return mode == AppOpsManager.MODE_ALLOWED
    }

    fun getRecentContext(lookbackMs: Long = 30 * 60_000L): RecentContext {
        if (!isUsageAccessGranted()) {
            return RecentContext()
        }

        val usageStatsManager = context.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager
        val endTime = System.currentTimeMillis()
        val startTime = endTime - lookbackMs

        val events = usageStatsManager.queryEvents(startTime, endTime)

        val recentCategories = HashSet<AppCategory>(8)
        while (events.hasNextEvent()) {
            val event = UsageEvents.Event()
            events.getNextEvent(event)
            if (event.eventType == UsageEvents.Event.MOVE_TO_FOREGROUND) {
                recentCategories.add(AppCategoryClassifierFactory.classify(event.packageName))
            }
        }

        return RecentContext(
            recentBankApp = AppCategory.BANK in recentCategories ||
                AppCategory.INVESTMENTS in recentCategories,
            recentGovApp = AppCategory.GOVERNMENT in recentCategories,
            recentMarketplaceApp = AppCategory.MARKETPLACE in recentCategories ||
                AppCategory.DELIVERY in recentCategories,
            recentMessengerApp = AppCategory.MESSENGER in recentCategories
        )
    }

    companion object {
        private const val TAG = "RecentUserContext"
    }

    /**
     * Live-выборка для UI прозрачности данных. Берёт последние [limit]
     * `MOVE_TO_FOREGROUND` событий из UsageStats за окно [lookbackMs] и
     * возвращает их в DESC-порядке по времени. Никуда не пишется,
     * вычисляется на лету при открытии Settings → Privacy. Используется
     * вместо несуществующей `app_usage_event` таблицы (которую в
     * текущей реализации никто не наполняет — Personal Model читает
     * UsageStats тоже live через [getRecentContext]).
     */
    data class ForegroundEventSample(
        val packageName: String,
        val timestamp: Long,
        val categoryBucket: String,
    )

    fun recentForegroundEvents(
        limit: Int = 3,
        lookbackMs: Long = 24L * 60 * 60_000L,
    ): List<ForegroundEventSample> {
        if (!isUsageAccessGranted()) return emptyList()
        return try {
            val usageStatsManager = context.getSystemService(Context.USAGE_STATS_SERVICE)
                as? UsageStatsManager ?: return emptyList()
            val endTime = System.currentTimeMillis()
            val startTime = endTime - lookbackMs
            val events = usageStatsManager.queryEvents(startTime, endTime)
            val out = ArrayDeque<ForegroundEventSample>(limit)
            val seenPackages = mutableSetOf<String>()
            // queryEvents возвращает в ASC-порядке; идём с конца, держим
            // только первые `limit` уникальных пакетов с последним
            // foreground'ом.
            val all = ArrayList<ForegroundEventSample>()
            while (events.hasNextEvent()) {
                val ev = UsageEvents.Event()
                events.getNextEvent(ev)
                if (ev.eventType != UsageEvents.Event.MOVE_TO_FOREGROUND) continue
                all += ForegroundEventSample(
                    packageName = ev.packageName,
                    timestamp = ev.timeStamp,
                    categoryBucket = bucketFor(ev.packageName),
                )
            }
            // DESC по времени, дедуп по packageName, top N.
            for (sample in all.sortedByDescending { it.timestamp }) {
                if (sample.packageName in seenPackages) continue
                seenPackages += sample.packageName
                out.addLast(sample)
                if (out.size >= limit) break
            }
            out.toList()
        } catch (t: Throwable) {
            Log.w(TAG, "recentForegroundEvents failed", t)
            emptyList()
        }
    }

    private fun bucketFor(pkg: String): String =
        AppCategoryClassifierFactory.classify(pkg).name
}
