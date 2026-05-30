package com.antispam.blocker.overlay

import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.provider.Settings
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView

/**
 * Показывает плашку-предупреждение поверх системного incoming call UI.
 *
 * Работает через WindowManager + TYPE_APPLICATION_OVERLAY — это единственный
 * способ вывести UI поверх звонка на современных Android. Требует
 * SYSTEM_ALERT_WINDOW permission (юзер включает в системных настройках).
 */
class SpamAlertOverlayService : Service() {

    private var windowManager: WindowManager? = null
    private var overlayView: View? = null
    private val autoHideHandler = Handler(Looper.getMainLooper())
    private val autoHideRunnable = Runnable { removeOverlay() }

    companion object {
        const val EXTRA_NUMBER = "extra_number"
        const val EXTRA_REASON = "extra_reason"
        const val AUTO_HIDE_MS = 15_000L

        /** Запуск оверлея. Безопасно вызывать из любого Service/BroadcastReceiver. */
        fun show(context: Context, number: String, reason: String?) {
            if (!canDrawOverlay(context)) return
            val intent = Intent(context, SpamAlertOverlayService::class.java).apply {
                putExtra(EXTRA_NUMBER, number)
                putExtra(EXTRA_REASON, reason)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startService(intent)
        }

        fun canDrawOverlay(context: Context): Boolean {
            return Settings.canDrawOverlays(context)
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val number = intent?.getStringExtra(EXTRA_NUMBER) ?: "Скрытый номер"
        val reason = intent?.getStringExtra(EXTRA_REASON)

        // Проверка разрешения (может быть отозвано между вызовами)
        if (!canDrawOverlay(this)) {
            stopSelf()
            return START_NOT_STICKY
        }

        removeOverlay() // убираем предыдущий, если был
        showOverlay(number, reason)

        // Автоскрытие через N секунд
        autoHideHandler.removeCallbacks(autoHideRunnable)
        autoHideHandler.postDelayed(autoHideRunnable, AUTO_HIDE_MS)

        return START_NOT_STICKY
    }

    private fun showOverlay(number: String, reason: String?) {
        val wm = getSystemService(WINDOW_SERVICE) as WindowManager
        windowManager = wm

        val view = buildView(number, reason)
        overlayView = view

        val layoutType = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        } else {
            @Suppress("DEPRECATION")
            WindowManager.LayoutParams.TYPE_PHONE
        }

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            layoutType,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                    WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS or
                    WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
            y = dp(40)
        }

        try {
            wm.addView(view, params)
        } catch (_: Exception) {
            stopSelf()
        }
    }

    private fun buildView(number: String, reason: String?): View {
        val ctx = this
        val root = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(18), dp(20), dp(18))
            background = GradientDrawable().apply {
                cornerRadius = dp(20).toFloat()
                setColor(Color.parseColor("#F0111113"))  // ink_elevated, 94% opacity
                setStroke(dp(1), Color.parseColor("#FF6B1A"))  // amber border
            }
            elevation = dp(12).toFloat()
        }

        // Заголовок
        val header = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val badge = TextView(ctx).apply {
            text = "⚠  ПОДОЗРИТЕЛЬНЫЙ ЗВОНОК"
            setTextColor(Color.parseColor("#FF6B1A"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            typeface = android.graphics.Typeface.MONOSPACE
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            letterSpacing = 0.12f
        }
        header.addView(badge)
        root.addView(header)

        // Номер
        val numberView = TextView(ctx).apply {
            text = number
            setTextColor(Color.parseColor("#F5F5F7"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 22f)
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            setPadding(0, dp(8), 0, 0)
        }
        root.addView(numberView)

        // Причина
        if (!reason.isNullOrBlank()) {
            val reasonView = TextView(ctx).apply {
                text = "Правило: $reason"
                setTextColor(Color.parseColor("#A0A0AB"))
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
                setPadding(0, dp(4), 0, 0)
            }
            root.addView(reasonView)
        }

        // Подсказка
        val hint = TextView(ctx).apply {
            text = "Этот звонок помечен как возможный спам"
            setTextColor(Color.parseColor("#A0A0AB"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
            setPadding(0, dp(10), 0, dp(14))
        }
        root.addView(hint)

        // Кнопка закрытия
        val closeBtn = Button(ctx).apply {
            text = "Понятно"
            setTextColor(Color.parseColor("#0A0A0B"))
            setAllCaps(false)
            background = GradientDrawable().apply {
                cornerRadius = dp(12).toFloat()
                setColor(Color.parseColor("#FF6B1A"))
            }
            setOnClickListener { removeOverlay() }
        }
        root.addView(closeBtn, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            dp(48)
        ))

        // Клик по всей плашке тоже закрывает
        root.setOnClickListener { removeOverlay() }

        return root
    }

    private fun removeOverlay() {
        autoHideHandler.removeCallbacks(autoHideRunnable)
        val wm = windowManager
        val view = overlayView
        if (wm != null && view != null) {
            try {
                wm.removeView(view)
            } catch (_: Exception) {
                // view уже удалён
            }
        }
        overlayView = null
        stopSelf()
    }

    override fun onDestroy() {
        removeOverlay()
        super.onDestroy()
    }

    private fun dp(value: Int): Int {
        val density = resources.displayMetrics.density
        return (value * density).toInt()
    }
}
