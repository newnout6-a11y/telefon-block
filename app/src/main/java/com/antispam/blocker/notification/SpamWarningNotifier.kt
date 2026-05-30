package com.antispam.blocker.notification

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Build
import androidx.core.app.NotificationCompat
import com.antispam.blocker.MainActivity
import com.antispam.blocker.R
import java.util.concurrent.atomic.AtomicInteger

/**
 * Показывает уведомления о входящих подозрительных/заблокированных звонках.
 *
 * Для того чтобы уведомление реально появилось ПОВЕРХ системного incoming call UI,
 * используется:
 *  - отдельный канал с IMPORTANCE_HIGH
 *  - CATEGORY_CALL + PRIORITY_MAX
 *  - setFullScreenIntent(...) — единственный способ показать нотификацию поверх
 *    активного звонка на современных Android (требует USE_FULL_SCREEN_INTENT).
 *  - уникальный notificationId для каждого звонка — чтобы подряд идущие
 *    уведомления не перетирали друг друга.
 */
class SpamWarningNotifier(private val context: Context) {

    companion object {
        const val CHANNEL_WARN = "spam_warning_channel"
        const val CHANNEL_BLOCK = "spam_blocked_channel"
        const val CHANNEL_ANSWERBOT = "answerbot_transcription"

        const val ACTION_BLOCK = "com.antispam.blocker.ACTION_BLOCK"
        const val ACTION_ALLOW = "com.antispam.blocker.ACTION_ALLOW"

        /** Personal-classifier feedback (Req 4.5): пользователь нажал «Был ли это спам? Да».
         *  Маршрутизируется отдельно от [ACTION_BLOCK]/[ACTION_ALLOW] — не идёт в
         *  cloud-side EMA через FeedbackHandler, а кормит OnlineTrainer.applyExplicitLabel
         *  с label = BLOCK для on-device SGD-шага по сохранённому FeatureSnapshot. */
        const val ACTION_SPAM_YES = "com.antispam.blocker.PERSONAL_SPAM_YES"
        /** Personal-classifier feedback (Req 4.5): пользователь нажал «Был ли это спам? Нет».
         *  Кормит OnlineTrainer.applyExplicitLabel с label = ALLOW. */
        const val ACTION_SPAM_NO = "com.antispam.blocker.PERSONAL_SPAM_NO"

        const val EXTRA_NUMBER = "extra_number"
        const val EXTRA_NOTIF_ID = "extra_notif_id"
        /** Изначальный вердикт модели (warn/block/allow) — нужен FeedbackHandler'у,
         *  чтобы понять «коррекция в какую сторону» (NOT_SPAM из BLOCK = UNBLOCK,
         *  IS_SCAM из WARN = MARK_SPAM, и т.д.). */
        const val EXTRA_ORIG_VERDICT = "extra_orig_verdict"
        /** CSV списка id факторов которые сработали при принятии решения.
         *  FeedbackHandler делает EMA-обновление весов по топ-3 из них. */
        const val EXTRA_ACTIVE_FACTORS = "extra_active_factors"
        /** id строки в `call_event` (Long), к которой привязан FeatureSnapshot.
         *  Используется только для personal-classifier feedback ([ACTION_SPAM_YES]/
         *  [ACTION_SPAM_NO]) — `OnlineTrainer.applyExplicitLabel(callEventId, label)`
         *  по нему достаёт snapshot и делает SGD-шаг. */
        const val EXTRA_CALL_EVENT_ID = "extra_call_event_id"
    }

    private val nm = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

    init {
        createChannels()
    }

    private fun createChannels() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return

        val warnChannel = NotificationChannel(
            CHANNEL_WARN,
            "Подозрительные звонки",
            NotificationManager.IMPORTANCE_HIGH
        ).apply {
            description = "Предупреждения о подозрительных входящих звонках"
            enableVibration(true)
            enableLights(true)
            lightColor = Color.rgb(0xFF, 0x6B, 0x1A)
            setBypassDnd(true)
            lockscreenVisibility = NotificationCompat.VISIBILITY_PUBLIC
        }

        val blockChannel = NotificationChannel(
            CHANNEL_BLOCK,
            "Заблокированные звонки",
            NotificationManager.IMPORTANCE_DEFAULT
        ).apply {
            description = "Информация о заблокированных звонках"
            enableVibration(false)
            lockscreenVisibility = NotificationCompat.VISIBILITY_PUBLIC
        }

        nm.createNotificationChannel(warnChannel)
        nm.createNotificationChannel(blockChannel)

        val answerBotChannel = NotificationChannel(
            CHANNEL_ANSWERBOT,
            "Расшифровка автоответчика",
            NotificationManager.IMPORTANCE_DEFAULT
        ).apply {
            description = "Расшифрованные голосовые сообщения от заблокированных звонков"
            enableVibration(true)
        }
        nm.createNotificationChannel(answerBotChannel)
    }

    /**
     * Уведомление о подозрительном звонке. Показывается как heads-up поверх
     * incoming call UI через FullScreenIntent.
     *
     * @param reasons причины (RiskFactor.reason / правило). В collapsed-виде
     *   уведомления показывается первая. В expanded-виде (BigTextStyle) — топ-3
     *   списком. Раньше передавался один `ruleName: String?` и остальные причины
     *   терялись — пользователь видел только первый сигнал из ~5 возможных.
     * @param originalVerdict вердикт модели на момент показа («warn»/«block»).
     * @param activeFactorIds список id сработавших факторов (RiskFactor.id).
     *   FeedbackHandler EMA-обновит веса топ-3 из них по фактическому действию.
     * @param callEventId id связанной записи `call_event` для Personal_Classifier
     *   feedback (Req 4.5). Если != null, к уведомлению добавляется ещё одна пара
     *   кнопок «Был ли это спам? Да / Нет», broadcasting [ACTION_SPAM_YES] /
     *   [ACTION_SPAM_NO] с [EXTRA_CALL_EVENT_ID] — это кормит
     *   `OnlineTrainer.applyExplicitLabel`, отдельно от cloud-side
     *   [ACTION_BLOCK]/[ACTION_ALLOW] кнопок.
     */
    fun showWarning(
        number: String,
        reasons: List<String> = emptyList(),
        originalVerdict: String = "warn",
        activeFactorIds: List<String> = emptyList(),
        callEventId: Long? = null
    ) {
        val notifId = generateId(number)

        val blockPending = buildActionIntent(
            ACTION_BLOCK, number, notifId, requestCode = notifId * 10,
            originalVerdict = originalVerdict, activeFactorIds = activeFactorIds,
            callEventId = callEventId,
        )
        val allowPending = buildActionIntent(
            ACTION_ALLOW, number, notifId, requestCode = notifId * 10 + 1,
            originalVerdict = originalVerdict, activeFactorIds = activeFactorIds,
            callEventId = callEventId,
        )
        val contentPending = buildContentIntent(notifId * 10 + 2)

        val collapsedReason = reasons.firstOrNull()
        val collapsedSubtitle = collapsedReason?.let { "• $it" } ?: "Проверено по правилам"
        val expandedReasons = formatReasonsForBigText(reasons)

        val builder = NotificationCompat.Builder(context, CHANNEL_WARN)
            .setSmallIcon(R.drawable.ic_warning)
            .setContentTitle("Подозрение на спам")
            .setContentText("Звонок от $number  $collapsedSubtitle")
            .setStyle(
                NotificationCompat.BigTextStyle()
                    .bigText("Звонок от $number\n$expandedReasons\n\nЕсли это важный звонок — нажмите «Это не спам».")
            )
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setCategory(NotificationCompat.CATEGORY_CALL)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .setColor(Color.rgb(0xFF, 0x6B, 0x1A))
            .setColorized(true)
            .setAutoCancel(true)
            .setOngoing(false)
            .setFullScreenIntent(contentPending, true)
            .setContentIntent(contentPending)
            .addAction(0, "В чёрный список", blockPending)
            .addAction(0, "Это не спам", allowPending)
            .setVibrate(longArrayOf(0, 300, 200, 300))

        // Req 4.5 / 6.1: per-call personal-classifier feedback. Кнопки появляются
        // отдельно от существующих «В чёрный список» / «Это не спам» (которые
        // адресованы cloud-side EMA через FeedbackHandler) и при нажатии
        // отправляют broadcast в SpamActionReceiver, где OnlineTrainer
        // .applyExplicitLabel(callEventId, label) делает SGD-шаг по сохранённому
        // FeatureSnapshot.
        if (callEventId != null) {
            val spamYesPending = buildPersonalFeedbackIntent(
                ACTION_SPAM_YES, number, notifId, callEventId,
                requestCode = notifId * 10 + 5
            )
            val spamNoPending = buildPersonalFeedbackIntent(
                ACTION_SPAM_NO, number, notifId, callEventId,
                requestCode = notifId * 10 + 6
            )
            builder
                .addAction(0, "Был ли это спам? Да", spamYesPending)
                .addAction(0, "Нет", spamNoPending)
        }

        nm.notify(notifId, builder.build())
    }

    /**
     * Уведомление о заблокированном звонке (после того как звонок уже отклонён).
     * Показывается в обычном режиме + даёт возможность вернуть номер в белый список.
     *
     * @param reasons причины блокировки. Collapsed-вид показывает первую,
     *   expanded — топ-3 списком. См. doc у [showWarning].
     * @param originalVerdict вердикт модели на момент блокировки («block»).
     * @param activeFactorIds список сработавших факторов — для FeedbackHandler.
     * @param callEventId id связанной записи `call_event` (Req 4.5). См. [showWarning].
     */
    fun showBlocked(
        number: String,
        reasons: List<String> = emptyList(),
        originalVerdict: String = "block",
        activeFactorIds: List<String> = emptyList(),
        callEventId: Long? = null
    ) {
        val notifId = generateId(number) + 1_000_000 // отдельный диапазон для blocked

        val allowPending = buildActionIntent(
            ACTION_ALLOW, number, notifId, requestCode = notifId * 10 + 3,
            originalVerdict = originalVerdict, activeFactorIds = activeFactorIds,
            callEventId = callEventId,
        )
        val contentPending = buildContentIntent(notifId * 10 + 4)

        val collapsedReason = reasons.firstOrNull()
        val collapsedSubtitle = collapsedReason?.let { "• $it" } ?: ""
        val expandedReasons = formatReasonsForBigText(reasons)

        val builder = NotificationCompat.Builder(context, CHANNEL_BLOCK)
            .setSmallIcon(R.drawable.ic_warning)
            .setContentTitle("Звонок заблокирован")
            .setContentText("$number  $collapsedSubtitle")
            .setStyle(
                NotificationCompat.BigTextStyle()
                    .bigText("Заблокирован звонок от $number\n$expandedReasons\n\nЕсли это ошибка — добавьте номер в белый список.")
            )
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setCategory(NotificationCompat.CATEGORY_CALL)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .setColor(Color.rgb(0xFF, 0x4D, 0x6D))
            .setAutoCancel(true)
            .setContentIntent(contentPending)
            .addAction(0, "Это не спам", allowPending)

        // Req 4.5 / 6.1: см. комментарий в [showWarning].
        if (callEventId != null) {
            val spamYesPending = buildPersonalFeedbackIntent(
                ACTION_SPAM_YES, number, notifId, callEventId,
                requestCode = notifId * 10 + 7
            )
            val spamNoPending = buildPersonalFeedbackIntent(
                ACTION_SPAM_NO, number, notifId, callEventId,
                requestCode = notifId * 10 + 8
            )
            builder
                .addAction(0, "Был ли это спам? Да", spamYesPending)
                .addAction(0, "Нет", spamNoPending)
        }

        nm.notify(notifId, builder.build())
    }

    /**
     * Готовит блок текста с причинами для BigTextStyle: первая идёт без буллета
     * (она же видна в collapsed), остальные с «• ». Берётся максимум 3 — больше
     * визуально уже не влезает в notification и сливается в кашу.
     */
    private fun formatReasonsForBigText(reasons: List<String>): String {
        val top = reasons.distinct().take(3)
        if (top.isEmpty()) return "Проверено по правилам"
        if (top.size == 1) return "• ${top[0]}"
        return top.joinToString(separator = "\n") { "• $it" }
    }

    private fun buildActionIntent(
        action: String,
        number: String,
        notifId: Int,
        requestCode: Int,
        originalVerdict: String,
        activeFactorIds: List<String>,
        // Дыра #2.a: прокидываем callEventId и в cloud-side кнопки
        // («В чёрный список» / «Это не спам»), чтобы SpamActionReceiver
        // мог дёрнуть OnlineTrainer.applyExplicitLabel параллельно с
        // FeedbackHandler. До этого на-tap кормилась только cloud-side EMA,
        // а Device_Model не получала ни одного explicit-сигнала из этой
        // нотификации — отдельные кнопки «Был ли это спам? Да/Нет» юзер
        // нажимает редко, и persistent learning'а не происходило.
        callEventId: Long? = null,
    ): PendingIntent {
        val intent = Intent(context, SpamActionReceiver::class.java).apply {
            this.action = action
            // Уникальный data URI — иначе PendingIntent переиспользует старый intent с чужим номером
            data = Uri.parse("antispam://call/$notifId/$action")
            putExtra(EXTRA_NUMBER, number)
            putExtra(EXTRA_NOTIF_ID, notifId)
            putExtra(EXTRA_ORIG_VERDICT, originalVerdict)
            putExtra(EXTRA_ACTIVE_FACTORS, activeFactorIds.joinToString(","))
            if (callEventId != null) {
                putExtra(EXTRA_CALL_EVENT_ID, callEventId)
            }
        }
        return PendingIntent.getBroadcast(
            context, requestCode, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
    }

    /**
     * Per-call personal-classifier feedback intent (Req 4.5).
     *
     * Маршрутизируется отдельно от [buildActionIntent]: action — [ACTION_SPAM_YES]
     * или [ACTION_SPAM_NO], чтобы [SpamActionReceiver] не путал его с
     * cloud-side EMA-feedback. В extras кладём `EXTRA_CALL_EVENT_ID`, по которому
     * `OnlineTrainer.applyExplicitLabel(callEventId, label)` достаёт сохранённый
     * `FeatureSnapshot` и делает SGD-шаг (Device_Model).
     */
    private fun buildPersonalFeedbackIntent(
        action: String,
        number: String,
        notifId: Int,
        callEventId: Long,
        requestCode: Int
    ): PendingIntent {
        val intent = Intent(context, SpamActionReceiver::class.java).apply {
            this.action = action
            // Уникальный data URI — отдельное пространство от ACTION_BLOCK/ACTION_ALLOW
            data = Uri.parse("antispam://call/$notifId/$action/$callEventId")
            putExtra(EXTRA_NUMBER, number)
            putExtra(EXTRA_NOTIF_ID, notifId)
            putExtra(EXTRA_CALL_EVENT_ID, callEventId)
        }
        return PendingIntent.getBroadcast(
            context, requestCode, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
    }

    private fun buildContentIntent(requestCode: Int): PendingIntent {
        val intent = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        return PendingIntent.getActivity(
            context, requestCode, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
    }

    /** Стабильный id на основе номера, времени и монотонного счётчика.
     *  Счётчик гарантирует уникальность даже для двух звонков с одного
     *  номера в одну секунду (race с notifId collision). */
    private val seq = AtomicInteger(0)

    private fun generateId(number: String): Int {
        val base = (System.currentTimeMillis() / 1000).toInt() and 0x0000FFFF
        val hash = number.hashCode() and 0x0000FFFF
        val counter = seq.incrementAndGet() and 0x0000FFFF
        return (base xor hash xor counter) and 0x0FFFFFFF
    }

    /**
     * Уведомление с расшифровкой голосового сообщения автоответчика.
     *
     * Показывается ПОСЛЕ того как звонящий оставил сообщение и Vosk
     * транскрибировал аудио. Тап по уведомлению открывает экран сообщений.
     */
    fun showAnswerBotTranscription(number: String, transcription: String?, messageId: Long) {
        val notifId = generateId(number) + 2_000_000

        val contentIntent = PendingIntent.getActivity(
            context, notifId * 10,
            Intent(context, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
                putExtra("navigate_to", "answerbot")
                putExtra("answerbot_message_id", messageId)
                data = android.net.Uri.parse("antispam://answerbot/${number}/$messageId")
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )

        val text = if (transcription.isNullOrBlank()) {
            "Голосовое сообщение записано (расшифровка недоступна)"
        } else {
            val preview = if (transcription.length <= 120) transcription
            else transcription.take(117) + "..."
            "\"$preview\""
        }

        val builder = NotificationCompat.Builder(context, CHANNEL_ANSWERBOT)
            .setSmallIcon(R.drawable.ic_shield)
            .setContentTitle("Сообщение от $number")
            .setContentText(text)
            .setStyle(
                if (transcription.isNullOrBlank()) null
                else NotificationCompat.BigTextStyle().bigText(
                    "Сообщение от $number:\n$transcription"
                )
            )
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setAutoCancel(true)
            .setContentIntent(contentIntent)

        nm.notify(notifId, builder.build())
    }
}
