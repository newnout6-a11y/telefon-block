package com.antispam.blocker.domain.personal

import android.util.Log
import com.antispam.blocker.data.db.dao.FeatureSnapshotDao
import com.antispam.blocker.data.prefs.DeviceModelStore
import kotlinx.coroutines.flow.first
import org.json.JSONException
import org.json.JSONObject
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

/**
 * Explicit on-device label produced by user feedback in the warning
 * notification or the explainability detail screen.
 *
 * Mirrors [ImplicitLabel]'s `(name, y)` pairing so both label channels can
 * feed [DeviceModel.sgdStep] without conversion: `BLOCK = 1f` (positive
 * class), `ALLOW = 0f`.
 *
 * Explicit feedback weighs strictly more than implicit (Req 4.8) — the
 * sample weight applied here is [DeviceModel.EXPLICIT_WEIGHT] (1.5f) versus
 * [DeviceModel.IMPLICIT_WEIGHT] (0.5f) for [applyImplicitLabel]. A single
 * explicit «Да/Нет» therefore outweighs three implicit signals on the same
 * call.
 */
enum class ExplicitLabel(val y: Float) {
    ALLOW(0f),
    BLOCK(1f),
}

/**
 * Wires implicit and explicit labels through to [DeviceModel.sgdStep].
 *
 * Both `applyImplicitLabel` and `applyExplicitLabel` funnel into the private
 * [applyLabel] which:
 *
 * 1. Looks up the matching [com.antispam.blocker.data.db.entity.FeatureSnapshot]
 *    via [FeatureSnapshotDao.getByCallEventId], falling back to
 *    [FeatureSnapshotDao.getById] when the caller passed a snapshot id
 *    directly (v1 screening pipeline — see task 12.2). **Missing snapshot**
 *    (e.g. retention worker pruned it, or the snapshot was never linked) →
 *    log and return without touching weights. A lost label is the lesser
 *    evil compared to applying SGD to a fabricated vector.
 * 2. Verifies `snapshot.featureSchemaVersion == DeviceFeatures.SCHEMA_VERSION`.
 *    **Schema mismatch** (the snapshot predates a feature-schema bump and is
 *    therefore in a different coordinate system) → log and return without
 *    touching weights, per design §"Database corruption / missing rows".
 * 3. Reconstructs [DeviceFeatures] from `snapshot.featuresJson`, looking up
 *    each feature by name in [DeviceFeatures.NAMES]. **Malformed JSON or a
 *    missing feature key** → log and return without touching weights. The
 *    weights stay deterministic in the face of corrupted state rather than
 *    reading garbage values.
 * 4. Calls [DeviceModel.sgdStep] with `y ∈ {0f, 1f}` and the appropriate
 *    `sampleWeight` from [DeviceModel.Companion].
 * 5. Calls [DeviceModelStore.incrementLabelCount] so [WarmUpGate] can detect
 *    when the warm-up window has been satisfied (Req 5.9 — `labelCount ≥ 30`).
 *
 * Implements [OnlineTrainerHandle] so [MissedNoCallbackRecheckWorker] can
 * resolve the trainer through [OnlineTrainerLocator] without a forward
 * reference at compile time.
 *
 * Thread-safety: this class is stateless beyond its injected dependencies
 * (`DeviceModel` serializes its own writes through a `Mutex`,
 * `FeatureSnapshotDao` is a Room DAO, `DeviceModelStore` serializes through
 * DataStore). Concurrent calls from `SpamActionReceiver`, the explainability
 * screen, and the deferred MISSED-recheck worker are safe.
 *
 * Validates: Requirements 2.3 (snapshot-based retraining), 4.5 (Yes/No
 * explicit feedback), 4.6 (no-feedback ⇒ no explicit weight change), 4.7
 * (long-press confirmation), 4.8 (explicit weight strictly exceeds implicit).
 */
class OnlineTrainer(
    private val deviceModel: DeviceModel,
    private val featureSnapshotDao: FeatureSnapshotDao,
    private val store: DeviceModelStore,
) : OnlineTrainerHandle {

    /** Guard against duplicate SGD steps for the same callEventId/snapshotId. */
    private val processedIds = ConcurrentHashMap.newKeySet<Long>()
    private val lastCleanup = AtomicLong(System.currentTimeMillis())

    /**
     * Apply an implicit label produced by [ImplicitLabelDetector] (Req 4.1–4.4).
     *
     * Sample weight is [DeviceModel.IMPLICIT_WEIGHT] (0.5f). Implicit labels
     * intentionally move weights less than explicit ones so that a small
     * amount of explicit feedback can correct a large amount of implicit
     * signal (Req 4.8).
     */
    override suspend fun applyImplicitLabel(callEventId: Long, label: ImplicitLabel) {
        applyLabel(callEventId, label.y, DeviceModel.IMPLICIT_WEIGHT)
    }

    /**
     * Apply an explicit label produced by user feedback (Req 4.5, 4.7).
     *
     * Sample weight is [DeviceModel.EXPLICIT_WEIGHT] (1.5f) — strictly larger
     * than [DeviceModel.IMPLICIT_WEIGHT] per Req 4.8 so a single explicit
     * «Да/Нет» tap dominates ambient implicit signal on the same call.
     */
    suspend fun applyExplicitLabel(callEventId: Long, label: ExplicitLabel) {
        applyLabel(callEventId, label.y, DeviceModel.EXPLICIT_WEIGHT)
    }

    /**
     * Common SGD path for implicit and explicit labels.
     *
     * The `callEventId` parameter is intentionally polymorphic in v1: callers
     * may pass either a real `call_event.id` (set later when the end-of-call
     * recorder lands) or a [com.antispam.blocker.data.db.entity.FeatureSnapshot.id]
     * (used right now by `SpamCallScreeningService` / `SpamWarningNotifier`,
     * because end-of-call CallEvent linking is a future task and snapshots
     * are persisted with `callEventId = null`). The lookup tries
     * `getByCallEventId` first and falls back to `getById` so the SGD step
     * works for both shapes without churning the public API.
     *
     * Returns silently (and without modifying weights) when the snapshot is
     * missing, schema-mismatched, or malformed — see class KDoc for the
     * rationale. Log lines are intentionally informative so a user wiping
     * weights and replaying a call from the journal can confirm in adb logs
     * whether the SGD step actually happened.
     */
    private suspend fun applyLabel(callEventId: Long, y: Float, sampleWeight: Float) {
        if (!processedIds.add(callEventId)) {
            Log.w(TAG, "duplicate label for id=$callEventId; skipping SGD")
            return
        }
        maybeCleanupDedupSet()

        val snapshot = featureSnapshotDao.getByCallEventId(callEventId)
            ?: featureSnapshotDao.getById(callEventId)
        if (snapshot == null) {
            Log.w(TAG, "no FeatureSnapshot for id=$callEventId; skipping SGD")
            return
        }
        if (snapshot.featureSchemaVersion != DeviceFeatures.SCHEMA_VERSION) {
            Log.w(
                TAG,
                "stale FeatureSnapshot schema=${snapshot.featureSchemaVersion} " +
                    "(expected ${DeviceFeatures.SCHEMA_VERSION}); skipping SGD",
            )
            return
        }
        val features = parseFeatures(snapshot.featuresJson)
        if (features == null) {
            Log.w(TAG, "malformed featuresJson for snapshot id=${snapshot.id}; skipping SGD")
            return
        }
        // Контекстный лог: показывает откуда пришёл label (id snapshot/CallEvent),
        // y и sampleWeight. Парный с подробным логом в DeviceModel.sgdStep —
        // вместе они дают полную картину в `adb logcat -s SpamBlocker_SGD`.
        android.util.Log.i(
            "SpamBlocker_SGD",
            "applyLabel id=$callEventId y=$y w=$sampleWeight " +
                "snapshot=${snapshot.id} schema=${snapshot.featureSchemaVersion}",
        )
        deviceModel.sgdStep(features, y, sampleWeight)
        store.incrementLabelCount()
        val labelCount = store.labelCountFlow.first()
        android.util.Log.i("SpamBlocker_SGD", "labelCount -> $labelCount (warmup at 30)")
    }

    private fun maybeCleanupDedupSet() {
        val now = System.currentTimeMillis()
        if (now - lastCleanup.get() > 3_600_000L) {
            lastCleanup.set(now)
            processedIds.clear()
        }
    }

    /**
     * Parses a `{feature_name: float}` JSON object into a [DeviceFeatures] in
     * canonical [DeviceFeatures.NAMES] order. Returns `null` if the JSON is
     * malformed or any feature name is missing — callers treat this as
     * "skip the SGD step" rather than crashing the screening pipeline.
     */
    private fun parseFeatures(json: String): DeviceFeatures? {
        return try {
            val obj = JSONObject(json)
            val out = FloatArray(DeviceFeatures.SIZE)
            for (i in 0 until DeviceFeatures.SIZE) {
                val name = DeviceFeatures.NAMES[i]
                if (!obj.has(name)) return null
                out[i] = obj.getDouble(name).toFloat()
            }
            DeviceFeatures(out)
        } catch (_: JSONException) {
            null
        }
    }

    private companion object {
        const val TAG = "OnlineTrainer"
    }
}
