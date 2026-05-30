package com.antispam.blocker.data.repository

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.database.ContentObserver
import android.net.Uri
import android.os.Handler
import android.os.HandlerThread
import android.provider.CallLog
import android.util.Log
import androidx.work.WorkManager
import com.antispam.blocker.data.cache.ContactsCache
import com.antispam.blocker.data.db.dao.CallEventDao
import com.antispam.blocker.data.db.dao.CallRecordDao
import com.antispam.blocker.data.db.dao.FeatureSnapshotDao
import com.antispam.blocker.data.db.entity.CallEvent
import com.antispam.blocker.data.prefs.DeviceModelStore
import com.antispam.blocker.domain.personal.ImplicitLabelDetector
import com.antispam.blocker.domain.personal.OnlineTrainer
import com.antispam.blocker.domain.lookup.CallerLookupWorker
import com.antispam.blocker.domain.personal.enqueueDeferredMissedRecheck
import com.antispam.blocker.util.PhoneNormalizer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs

/**
 * End-of-call telemetry recorder for Device_Model implicit-label training.
 *
 * Closes the gap between the call-screening hot path
 * ([com.antispam.blocker.service.SpamCallScreeningService]) and the
 * [OnlineTrainer]: the screening service emits a [com.antispam.blocker.data.db.entity.FeatureSnapshot]
 * at decision time but does **not** know the call's eventual duration or
 * outcome (ANSWERED/REJECTED/MISSED). Those facts only land in the system
 * `CallLog` after the call ends. This recorder watches `CallLog.Calls`
 * for new rows, projects them onto our [CallEvent] schema, links them to
 * the matching pre-existing snapshot, and dispatches the implicit label to
 * [OnlineTrainer] (synchronously for ALLOW/BLOCK rules, deferred 24 h via
 * [com.antispam.blocker.domain.personal.MissedNoCallbackRecheckWorker]
 * for the MISSED rule per Req 4.3).
 *
 * Architectural mirror of [ContactsCache]: a dedicated [HandlerThread] +
 * [ContentObserver] keeps every CallLog touch off the call-screening hot
 * path so a slow disk read here cannot delay the screening verdict.
 *
 * Permission-aware: every `ContentResolver.query`/`callEventDao.insert`
 * is wrapped so a runtime [SecurityException] (READ_CALL_LOG revoked
 * after install) degrades to a logged no-op rather than a crash.
 *
 * Privacy-aware: gated on
 * [DeviceModelStore.sourceCallLogEnabledFlow] — when the user has the
 * call-log toggle off, the recorder still listens (so it can resume
 * cheaply when re-enabled) but no-ops on insert/dispatch. Only metadata
 * is captured (number, direction, state, duration, timestamps) — no SMS
 * body, no call audio.
 *
 * Idempotent: the persisted [DeviceModelStore.lastSeenCallLogTimestampFlow]
 * cursor uses a strict `>` comparison, so an observer that fires twice
 * for the same row (or a process restart in the middle of a batch) does
 * not re-insert events.
 *
 * Requirements: 4.1, 4.2, 4.3, 4.4 (closes the implicit branch).
 */
class CallEventRecorder(
    context: Context,
    private val callEventDao: CallEventDao,
    private val featureSnapshotDao: FeatureSnapshotDao,
    private val callRecordDao: CallRecordDao,
    private val deviceModelStore: DeviceModelStore,
    private val onlineTrainer: OnlineTrainer,
    /**
     * Non-default for tests; production omits the parameter and gets a
     * fresh [SupervisorJob]-rooted IO scope so a crash in one batch does
     * not tear down the recorder.
     */
    private val scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO),
) {

    private val appContext: Context = context.applicationContext

    @Volatile private var initialized: Boolean = false
    @Volatile private var observer: ContentObserver? = null
    @Volatile private var workerThread: HandlerThread? = null
    @Volatile private var handler: Handler? = null

    /**
     * Local in-memory mirror of [DeviceModelStore.lastSeenCallLogTimestampFlow]
     * to avoid awaiting a Flow on every observer notification. Seeded from
     * DataStore at [init] time; advanced after each successful batch.
     */
    private val lastSeen: AtomicLong = AtomicLong(0L)

    /**
     * Registers the [ContentObserver] on a dedicated [HandlerThread] and
     * does an initial catch-up read for events newer than the persisted
     * cursor. Idempotent: repeated calls are a no-op, safe to wire from
     * `Application.onCreate`.
     */
    @Synchronized
    fun init(context: Context) {
        if (initialized) return
        initialized = true

        val thread = HandlerThread("CallEventRecorder").also { it.start() }
        workerThread = thread
        val h = Handler(thread.looper)
        handler = h

        // Seed cursor synchronously so the very first observer firing or
        // catch-up read filters correctly. `runBlocking` here is bounded
        // and runs on the worker thread, never on the main thread.
        //
        // P0 #5: на свежей установке persisted cursor = 0L → readNewCallLogRows
        // подтянул бы ВСЕ строки из CallLog (на телефонах с 5+ годами журнала
        // это десятки тысяч строк), что блокирует startup и для исторических
        // строк всё равно не даст SGD-step (FeatureSnapshot не существует).
        // Ограничиваем initial catch-up последними INITIAL_CATCHUP_WINDOW_MS —
        // 14 дней синхронно с WarmUpGate.WARMUP_DAYS_MS, чтобы фича
        // `same_prefix_call_count_7d_norm` (7-дневное окно) сразу имела
        // достаточно данных, а warm-up не упёрся в пустой call_event.
        h.post {
            val seeded = runCatching {
                runBlocking {
                    deviceModelStore.lastSeenCallLogTimestampFlow.first()
                }
            }.getOrDefault(0L)
            val initial = if (seeded == 0L) {
                val now = System.currentTimeMillis()
                val effective = (now - INITIAL_CATCHUP_WINDOW_MS).coerceAtLeast(0L)
                runCatching {
                    runBlocking { deviceModelStore.setLastSeenCallLogTimestamp(effective) }
                }.onFailure { Log.w(TAG, "persist initial cursor failed", it) }
                Log.i(TAG, "fresh-install catch-up cursor set to $effective (now=$now)")
                effective
            } else {
                seeded
            }
            lastSeen.set(initial)
        }

        val obs = object : ContentObserver(h) {
            override fun onChange(selfChange: Boolean, uri: Uri?) {
                h.post { processBatch() }
            }
        }
        runCatching {
            appContext.contentResolver.registerContentObserver(
                CallLog.Calls.CONTENT_URI,
                /* notifyForDescendants = */ true,
                obs,
            )
        }.onFailure { Log.w(TAG, "register observer failed", it) }
        observer = obs

        // Initial catch-up: pick up rows that landed while the process was
        // dead. Filtered by `lastSeen` so a fresh install does not replay
        // years of history (cursor stays at 0 → no rows pass `> 0` until
        // the next call writes a row, which is the desired behaviour).
        h.post { processBatch() }
    }

    /**
     * Unregisters the observer and tears down the worker thread. Idempotent.
     * Production app does not call this — included for symmetry and tests.
     */
    @Synchronized
    fun release() {
        if (!initialized) return
        observer?.let {
            runCatching { appContext.contentResolver.unregisterContentObserver(it) }
        }
        observer = null
        workerThread?.quitSafely()
        workerThread = null
        handler = null
        initialized = false
    }

    /**
     * Reads every CallLog row with `DATE > lastSeen`, projects each onto a
     * [CallEvent], inserts it, links the matching [com.antispam.blocker.data.db.entity.FeatureSnapshot],
     * and dispatches the implicit label.
     *
     * Runs on the dedicated [HandlerThread]; the actual DAO work happens
     * inside [scope] on the IO dispatcher. The high-water mark is
     * advanced only after a batch completes successfully, so a partial
     * failure replays the remaining rows on the next observer firing
     * rather than dropping them.
     */
    private fun processBatch() {
        scope.launch {
            try {
                processBatchInternal()
            } catch (t: Throwable) {
                Log.w(TAG, "processBatch failed", t)
            }
        }
    }

    private suspend fun processBatchInternal() {
        // Gate: when the user has the call-log toggle off, we keep the
        // observer alive (cheap) but do nothing. We deliberately don't
        // advance `lastSeen` here — when the toggle flips back on, the
        // user gets future events only, never replayed history (those
        // rows are skipped here, not persisted).
        val enabled = runCatching { deviceModelStore.sourceCallLogEnabledFlow.first() }
            .getOrDefault(true)
        if (!enabled) return

        if (!hasReadCallLogPermission()) {
            // No permission — nothing we can do until it's granted. Don't
            // advance the cursor so a permission grant later picks up the
            // backlog.
            return
        }

        val cursorStart = lastSeen.get()
        val rows = readNewCallLogRows(cursorStart)
        if (rows.isEmpty()) return

        var maxStartedAt = cursorStart
        for (row in rows) {
            val startedAt = row.dateMs
            var success = false
            try {
                handleRow(row)
                success = true
            } catch (se: SecurityException) {
                Log.w(TAG, "READ_CALL_LOG revoked mid-batch; aborting", se)
                return
            } catch (t: Throwable) {
                Log.w(TAG, "row insert failed at startedAt=$startedAt", t)
            }
            if (success && startedAt > maxStartedAt) maxStartedAt = startedAt
        }

        if (maxStartedAt > cursorStart) {
            lastSeen.set(maxStartedAt)
            runCatching { deviceModelStore.setLastSeenCallLogTimestamp(maxStartedAt) }
                .onFailure { Log.w(TAG, "persist lastSeen failed", it) }
        }
    }

    /**
     * Reads `CallLog.Calls` for rows strictly newer than [sinceMs] and
     * projects each into a lightweight [CallLogRow]. Sorted by `DATE ASC`
     * so the in-place `lastSeen` advance during the loop is monotonic and
     * a partial failure replays only the unprocessed tail.
     */
    private fun readNewCallLogRows(sinceMs: Long): List<CallLogRow> {
        val projection = arrayOf(
            CallLog.Calls.NUMBER,
            CallLog.Calls.TYPE,
            CallLog.Calls.DURATION,
            CallLog.Calls.DATE,
        )
        // Defensive: the CallLog provider doesn't accept "?" in some OEM
        // ROMs reliably, so embed the literal — sinceMs is a long under
        // our control, no injection surface.
        val selection = "${CallLog.Calls.DATE} > $sinceMs"
        return try {
            appContext.contentResolver.query(
                CallLog.Calls.CONTENT_URI,
                projection,
                selection,
                /* selectionArgs = */ null,
                /* sortOrder = */ "${CallLog.Calls.DATE} ASC",
            )?.use { c ->
                val numberIdx = c.getColumnIndex(CallLog.Calls.NUMBER)
                val typeIdx = c.getColumnIndex(CallLog.Calls.TYPE)
                val durationIdx = c.getColumnIndex(CallLog.Calls.DURATION)
                val dateIdx = c.getColumnIndex(CallLog.Calls.DATE)
                if (numberIdx < 0 || typeIdx < 0 || durationIdx < 0 || dateIdx < 0) {
                    return emptyList()
                }
                val out = ArrayList<CallLogRow>(c.count.coerceAtMost(256))
                while (c.moveToNext()) {
                    val rawNumber = c.getString(numberIdx) ?: ""
                    val type = c.getInt(typeIdx)
                    val durationSeconds = c.getLong(durationIdx).coerceAtLeast(0L)
                    val dateMs = c.getLong(dateIdx)
                    out += CallLogRow(rawNumber, type, durationSeconds, dateMs)
                }
                out
            } ?: emptyList()
        } catch (se: SecurityException) {
            Log.w(TAG, "READ_CALL_LOG revoked at runtime", se)
            emptyList()
        } catch (t: Throwable) {
            Log.w(TAG, "CallLog query failed", t)
            emptyList()
        }
    }

    private suspend fun handleRow(row: CallLogRow) {
        val (direction, state) = mapCallType(row.type, row.durationSeconds) ?: return

        // Normalize the number. A row without a number (private/withheld)
        // still has the raw column blank — we set isHidden in that case.
        val isHidden = row.rawNumber.isBlank()
        val normalized = if (isHidden) null else PhoneNormalizer.normalize(row.rawNumber)
        if (!isHidden && normalized == null) {
            // Garbage/unparseable number — skip rather than insert a row
            // we can't ever look up by `getByNumber`.
            return
        }

        val durationMs = row.durationSeconds * 1000L
        val startedAt = row.dateMs
        val endedAt = startedAt + durationMs

        val isContact = if (normalized != null) {
            ContactsCache.contains(normalized) ?: false
        } else {
            false
        }

        val event = CallEvent(
            normalizedNumber = normalized,
            isHidden = isHidden,
            direction = direction,
            state = state,
            durationMs = durationMs,
            startedAt = startedAt,
            endedAt = endedAt,
            isContact = isContact,
        )

        val insertedId = withContext(Dispatchers.IO) {
            try {
                callEventDao.insert(event)
            } catch (se: SecurityException) {
                Log.w(TAG, "callEventDao.insert SecurityException", se)
                -1L
            }
        }
        if (insertedId <= 0L) return

        val savedEvent = event.copy(id = insertedId)

        // Link the most recent matching pre-existing FeatureSnapshot (the
        // one written by SpamCallScreeningService at decision time, which
        // had `callEventId = null`). The link window guards against
        // matching a snapshot from a different, much earlier call to the
        // same number.
        if (normalized != null) {
            var linkedSnapshot: com.antispam.blocker.data.db.entity.FeatureSnapshot? = null
            try {
                val snapshot = featureSnapshotDao.getLatestForNumber(normalized)
                if (snapshot != null &&
                    snapshot.callEventId == null &&
                    abs(snapshot.timestamp - startedAt) <= LINK_WINDOW_MS
                ) {
                    featureSnapshotDao.updateCallEventId(snapshot.id, insertedId)
                    linkedSnapshot = snapshot
                }
            } catch (t: Throwable) {
                Log.w(TAG, "snapshot link failed for id=$insertedId", t)
            }

            // AnswerBot guard: если звонок был обработан автоответчиком (принят,
            // а не отклонён), screening service уже применил BLOCK через Path 2.
            // ImplicitLabelDetector на ANSWERED с duration > 15s выдал бы ALLOW,
            // что противоречит нашему BLOCK-вердикту и разрушает обучение.
            val blockedByUs = linkedSnapshot != null &&
                callRecordDao.countByNumberAndVerdictSince(normalized, "BLOCK", startedAt - LINK_WINDOW_MS) > 0
            if (!blockedByUs) {
                dispatchImplicitLabel(savedEvent, normalized)
            }
        }

        // Enqueue оффлайн+online lookup для показа "кто звонил" в журнале.
        // Запускается только для ненулевых, не-скрытых номеров.
        if (normalized != null && !isHidden) {
            try {
                CallerLookupWorker.enqueue(appContext, normalized)
            } catch (t: Throwable) {
                Log.w(TAG, "CallerLookupWorker enqueue failed", t)
            }
        }
    }

    private suspend fun dispatchImplicitLabel(event: CallEvent, normalizedNumber: String) {
        val history = try {
            callEventDao.getByNumber(normalizedNumber).filter { it.id != event.id }
        } catch (t: Throwable) {
            Log.w(TAG, "history lookup failed for number", t)
            return
        }
        val result = ImplicitLabelDetector().detect(event, history) ?: return
        if (result.isDeferred) {
            try {
                ImplicitLabelDetector().enqueueDeferredMissedRecheck(
                    workManager = WorkManager.getInstance(appContext),
                    callEventId = event.id,
                    normalizedNumber = normalizedNumber,
                    originalStartedAt = event.startedAt,
                )
            } catch (t: Throwable) {
                Log.w(TAG, "enqueue deferred recheck failed", t)
            }
        } else {
            try {
                onlineTrainer.applyImplicitLabel(event.id, result.label)
            } catch (t: Throwable) {
                Log.w(TAG, "applyImplicitLabel failed", t)
            }
        }
    }

    private fun hasReadCallLogPermission(): Boolean {
        return appContext.checkSelfPermission(Manifest.permission.READ_CALL_LOG) ==
            PackageManager.PERMISSION_GRANTED
    }

    /**
     * Maps `CallLog.Calls.TYPE` + `DURATION` into our `(direction, state)`.
     *
     * - INCOMING_TYPE (1) with DURATION > 0 → `(INCOMING, ANSWERED)`
     * - INCOMING_TYPE (1) with DURATION == 0 → `(INCOMING, REJECTED)`
     * - OUTGOING_TYPE (2) → `(OUTGOING, ANSWERED)` regardless of duration
     *   (CallLog itself records reject-by-callee under MISSED/REJECTED on
     *   the receiver side; on our side the call was placed and that's all
     *   the OUTGOING rule of [ImplicitLabelDetector] cares about).
     * - MISSED_TYPE (3) → `(INCOMING, MISSED)`
     * - REJECTED_TYPE (5) → `(INCOMING, REJECTED)`
     * - VOICEMAIL_TYPE (4) → skip (not a real call from the user's POV).
     * - BLOCKED_TYPE (6) → skip (we already have our own block path that
     *   wrote the BLOCK label at decision time; the system also writes
     *   BLOCKED rows for non-spam calls blocked by other apps and we
     *   don't want to double-count those).
     * - anything else (REJECTED == 5 covered above; future OEM-specific
     *   types) → skip rather than guess.
     */
    private fun mapCallType(type: Int, durationSeconds: Long):
        Pair<CallEvent.Direction, CallEvent.CallState>? {
        return when (type) {
            CallLog.Calls.INCOMING_TYPE -> {
                if (durationSeconds > 0L) {
                    CallEvent.Direction.INCOMING to CallEvent.CallState.ANSWERED
                } else {
                    CallEvent.Direction.INCOMING to CallEvent.CallState.REJECTED
                }
            }
            CallLog.Calls.OUTGOING_TYPE ->
                CallEvent.Direction.OUTGOING to CallEvent.CallState.ANSWERED
            CallLog.Calls.MISSED_TYPE ->
                CallEvent.Direction.INCOMING to CallEvent.CallState.MISSED
            CallLog.Calls.REJECTED_TYPE ->
                CallEvent.Direction.INCOMING to CallEvent.CallState.REJECTED
            // VOICEMAIL_TYPE = 4, BLOCKED_TYPE = 6, plus any future
            // OEM-specific type code: explicitly skip.
            else -> null
        }
    }

    private data class CallLogRow(
        val rawNumber: String,
        val type: Int,
        val durationSeconds: Long,
        val dateMs: Long,
    )

    companion object {
        private const val TAG = "CallEventRecorder"

        /**
         * Tolerance window for linking a [com.antispam.blocker.data.db.entity.FeatureSnapshot]
         * (written when the screening service saw the incoming call) to
         * the [CallEvent] (written when the call ended and CallLog
         * surfaced it). 10 minutes comfortably covers a long ringtone
         * plus a long conversation while keeping the link unambiguous —
         * a second call from the same number more than 10 minutes later
         * will produce its own snapshot with a fresh timestamp.
         */
        const val LINK_WINDOW_MS: Long = 10L * 60L * 1000L

        /**
         * Окно catch-up чтения CallLog при первом запуске (P0 #5). Совпадает с
         * [com.antispam.blocker.domain.personal.WarmUpGate.WARMUP_DAYS_MS]:
         * 14 дней — этого хватает фиче `same_prefix_call_count_7d_norm`
         * (7-дневное окно) и оставляет запас на задержку CallLog-провайдера.
         */
        const val INITIAL_CATCHUP_WINDOW_MS: Long = 14L * 24 * 60 * 60 * 1000
    }
}
