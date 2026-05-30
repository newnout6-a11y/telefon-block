package com.antispam.blocker.service

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import android.speech.tts.TextToSpeech
import android.telecom.TelecomManager
import android.util.Log
import androidx.core.app.NotificationCompat
import com.antispam.blocker.MainActivity
import com.antispam.blocker.R
import com.antispam.blocker.SpamBlockerApp
import com.antispam.blocker.domain.answerbot.AnswerBotTranscriptionWorker
import com.antispam.blocker.domain.answerbot.SilenceDetector
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream

class SpamAnswerBotService : Service() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    @Volatile private var busy = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val number = intent?.getStringExtra(EXTRA_NUMBER) ?: run {
            stopSelf()
            return START_NOT_STICKY
        }
        val alreadyAccepted = intent.getBooleanExtra(EXTRA_ALREADY_ACCEPTED, false)

        if (busy) {
            Log.w(TAG, "AnswerBot already busy, skipping second call")
            stopSelf(startId)
            return START_NOT_STICKY
        }
        busy = true

        startForeground(NOTIFICATION_RECORDING_ID, buildRecordingNotification())

        scope.launch {
            try {
                runAnswerBot(number, alreadyAccepted)
            } catch (t: Throwable) {
                Log.e(TAG, "AnswerBot failed for $number", t)
            } finally {
                busy = false
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
        }

        return START_REDELIVER_INTENT
    }

    override fun onDestroy() {
        scope.cancel()
        stopRecording()
        super.onDestroy()
    }

    private suspend fun runAnswerBot(number: String, alreadyAccepted: Boolean = false) {
        val tm = getSystemService(Context.TELECOM_SERVICE) as TelecomManager

        if (!alreadyAccepted) {
            if (!tryAcceptCall(tm)) {
                Log.w(TAG, "acceptRingingCall failed or not allowed — falling back to hangup")
                tryHangup(tm)
                return
            }
            minimizeCallUi()
        }

        val outputDir = File(filesDir, "answerbot")
        if (!outputDir.exists()) outputDir.mkdirs()
        val wavFile = File(outputDir, "msg_${System.currentTimeMillis()}.wav")

        // Быстрый оффлайн-лукап: если номер уже есть в кэше caller_lookup,
        // используем orgName/region для персонализированного приветствия.
        val callerInfo = try {
            SpamBlockerApp.instance.callerLookupRepository.ensureOffline(number)
        } catch (t: Throwable) {
            Log.w(TAG, "caller lookup failed, using default greeting", t)
            null
        }

        playGreeting(callerInfo)
        recordCaller(wavFile)

        tryHangup(tm)

        if (wavFile.exists() && wavFile.length() > 0L) {
            val durationMs = estimateWavDuration(wavFile)
            AnswerBotTranscriptionWorker.enqueue(this, wavFile.absolutePath, number, durationMs)
            Log.i(TAG, "recorded ${wavFile.length()} bytes, enqueued transcription for $number")
        }
    }

    private fun tryAcceptCall(tm: TelecomManager): Boolean {
        return try {
            tm.acceptRingingCall()
            Log.i(TAG, "call accepted")
            true
        } catch (e: SecurityException) {
            Log.w(TAG, "ANSWER_PHONE_CALLS permission missing", e)
            false
        } catch (e: RuntimeException) {
            Log.w(TAG, "acceptRingingCall failed", e)
            false
        }
    }

    private fun tryHangup(tm: TelecomManager) {
        try {
            if (tm.isInCall) {
                tm.endCall()
                Log.i(TAG, "call ended")
            }
        } catch (e: SecurityException) {
            Log.w(TAG, "endCall permission missing", e)
        }
    }

    private fun minimizeCallUi() {
        try {
            val homeIntent = Intent(Intent.ACTION_MAIN).apply {
                addCategory(Intent.CATEGORY_HOME)
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            startActivity(homeIntent)
        } catch (t: Throwable) {
            Log.w(TAG, "minimizeCallUi failed", t)
        }
    }

    private fun playGreeting(callerInfo: com.antispam.blocker.domain.lookup.CallerInfo? = null) {
        val greetingFile = buildGreetingText(callerInfo)?.let { text ->
            synthesizeToFile(text)
        }

        var mp: MediaPlayer? = null
        try {
            mp = MediaPlayer()
            if (greetingFile != null && greetingFile.exists() && greetingFile.length() > 0L) {
                mp.setDataSource(greetingFile.absolutePath)
                mp.prepare()
                mp.start()
                while (mp.isPlaying && scope.isActive) { Thread.sleep(100) }
            } else {
                // Fallback to pre-recorded greeting.ogg
                val afd = assets.openFd("answerbot/greeting.ogg")
                mp.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
                afd.close()
                mp.prepare()
                mp.start()
                while (mp.isPlaying && scope.isActive) { Thread.sleep(100) }
            }
        } catch (t: Throwable) {
            Log.w(TAG, "greeting playback failed", t)
        } finally {
            try { mp?.stop() } catch (_: Throwable) {}
            try { mp?.release() } catch (_: Throwable) {}
            if (greetingFile != null) try { greetingFile.delete() } catch (_: Throwable) {}
        }
    }

    /**
     * Строит текст персонализированного приветствия на основе данных о звонящем.
     * Возвращает null если данных недостаточно — тогда играем стандартный greeting.ogg.
     */
    private fun buildGreetingText(info: com.antispam.blocker.domain.lookup.CallerInfo?): String? {
        val org = info?.orgName ?: return null
        return "Здравствуйте. Уважаемый представитель организации «$org», кто вы и по какому вопросу звоните? Сообщение будет записано."
    }

    /** Синтезирует текст через Android TTS в temp WAV-файл. */
    private fun synthesizeToFile(text: String): File? {
        val file = File(cacheDir, "greeting_tts_${System.currentTimeMillis()}.wav")
        try {
            var tts: TextToSpeech? = null
            var done = false
            tts = TextToSpeech(this) { status ->
                if (status == TextToSpeech.SUCCESS) {
                    tts?.setLanguage(java.util.Locale("ru"))
                    tts?.synthesizeToFile(text, null, file, "greeting")
                }
            }
            // Ждём завершения синтеза (synthesizeToFile синхронный в TTS engine)
            val start = System.currentTimeMillis()
            while (!done && System.currentTimeMillis() - start < 5000L) {
                Thread.sleep(200)
                done = file.exists() && file.length() > 0L
            }
            tts?.shutdown()
            return if (file.exists() && file.length() > 0L) file else null
        } catch (t: Throwable) {
            Log.w(TAG, "TTS synthesis failed", t)
            try { file.delete() } catch (_: Throwable) {}
            return null
        }
    }

    private fun recordCaller(wavFile: File) {
        val sampleRate = 16000
        val channelConfig = AudioFormat.CHANNEL_IN_MONO
        val audioFormat = AudioFormat.ENCODING_PCM_16BIT

        val minBuf = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
        val bufferSize = maxOf(minBuf, sampleRate / 5)

        val source = trySource(MediaRecorder.AudioSource.VOICE_COMMUNICATION)
            ?: trySource(MediaRecorder.AudioSource.MIC)
            ?: run {
                Log.e(TAG, "no audio source available")
                return
            }

        val recorder = try {
            AudioRecord(source, sampleRate, channelConfig, audioFormat, bufferSize)
        } catch (t: Throwable) {
            Log.e(TAG, "AudioRecord init failed", t)
            return
        }

        audioRecord = recorder
        isRecording = true

        try {
            recorder.startRecording()
            Log.i(TAG, "recording started with source=$source")

            val allSamples = mutableListOf<ShortArray>()
            val buffer = ShortArray(bufferSize)
            val silence = SilenceDetector()

            val maxDurationSec = 45
            val maxSamples = maxDurationSec * sampleRate
            var totalSamples = 0

            while (isRecording && scope.isActive && totalSamples < maxSamples) {
                val read = recorder.read(buffer, 0, buffer.size)
                if (read <= 0) break

                val chunk = buffer.copyOf(read)
                allSamples.add(chunk)
                totalSamples += read

                if (silence.feedSamples(chunk)) {
                    Log.i(TAG, "silence detected (${silence.silenceDurationSec}s), stopping")
                    break
                }
            }

            recorder.stop()
            Log.i(TAG, "recording stopped, totalSamples=$totalSamples")

            writeWav(wavFile, allSamples, sampleRate)
        } catch (t: Throwable) {
            Log.e(TAG, "recording failed", t)
            try { wavFile.delete() } catch (_: Throwable) {}
        } finally {
            stopRecording()
        }
    }

    private fun trySource(source: Int): Int? {
        return try {
            val testBuf = AudioRecord.getMinBufferSize(16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            val r = AudioRecord(source, 16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, testBuf)
            r.release()
            source
        } catch (t: Throwable) {
            null
        }
    }

    private fun stopRecording() {
        isRecording = false
        audioRecord?.let {
            try { it.stop() } catch (_: Throwable) {}
            try { it.release() } catch (_: Throwable) {}
        }
        audioRecord = null
    }

    private fun writeWav(file: File, chunks: List<ShortArray>, sampleRate: Int) {
        try {
            val totalShorts = chunks.sumOf { it.size }
            val dataSize = totalShorts * 2
            val fos = FileOutputStream(file)
            val dos = DataOutputStream(fos)

            dos.writeBytes("RIFF")
            dos.writeInt(Integer.reverseBytes(36 + dataSize))
            dos.writeBytes("WAVE")
            dos.writeBytes("fmt ")
            dos.writeInt(Integer.reverseBytes(16))
            dos.writeShort(Integer.reverseBytes(1))
            dos.writeShort(Integer.reverseBytes(1))
            dos.writeInt(Integer.reverseBytes(sampleRate))
            dos.writeInt(Integer.reverseBytes(sampleRate * 2))
            dos.writeShort(Integer.reverseBytes(2))
            dos.writeShort(Integer.reverseBytes(16))
            dos.writeBytes("data")
            dos.writeInt(Integer.reverseBytes(dataSize))

            for (chunk in chunks) {
                for (s in chunk) {
                    dos.writeShort(Integer.reverseBytes(s.toInt()))
                }
            }

            dos.close()
        } catch (t: Throwable) {
            Log.e(TAG, "writeWav failed", t)
        }
    }

    private fun estimateWavDuration(file: File): Long {
        val dataSize = file.length() - 44
        if (dataSize <= 0) return 0L
        return dataSize / (16000 * 2) * 1000L
    }

    private fun buildRecordingNotification(): android.app.Notification {
        createRecordingChannel()
        val contentIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_RECORDING)
            .setSmallIcon(R.drawable.ic_shield)
            .setContentTitle(getString(R.string.app_name))
            .setContentText("Идёт запись...")
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_MIN)
            .setContentIntent(contentIntent)
            .build()
    }

    private fun createRecordingChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (nm.getNotificationChannel(CHANNEL_RECORDING) != null) return
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_RECORDING, "Запись звонка", NotificationManager.IMPORTANCE_MIN)
        )
    }

    companion object {
        private const val TAG = "SpamAnswerBot"
        private const val CHANNEL_RECORDING = "answerbot_recording"
        private const val NOTIFICATION_RECORDING_ID = 9001
        const val EXTRA_NUMBER = "normalized_number"
        const val EXTRA_ALREADY_ACCEPTED = "already_accepted"
    }
}
