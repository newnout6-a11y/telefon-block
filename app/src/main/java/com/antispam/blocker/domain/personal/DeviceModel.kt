package com.antispam.blocker.domain.personal

import com.antispam.blocker.data.prefs.DeviceModelStore
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.json.JSONException
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.exp

/**
 * Single feature contribution to a Device_Model prediction.
 *
 * `signed = weight × value` is precomputed so the UI explainability layer
 * can render bars without re-multiplying. A positive [signed] pushes
 * `p(BLOCK)` up; a negative one pushes it down.
 */
data class FeatureContribution(
    val name: String,
    val weight: Float,
    val value: Float,
    val signed: Float,
)

/**
 * Result of [DeviceModel.predict]: the calibrated probability of BLOCK,
 * the [DeviceVerdict] derived from it via the model thresholds, and the
 * top contributing features for explainability (Req 6.3).
 */
data class DevicePrediction(
    val probBlock: Float,
    val verdict: DeviceVerdict,
    val topContributions: List<FeatureContribution>,
)

/**
 * Personal on-device binary logistic-regression classifier.
 *
 * - **Pure Kotlin**: zero TensorFlow Lite, zero decision-tree / boosting
 *   libraries, zero federated-learning hooks (Req 3.2, 3.6).
 * - **Hot path is lock-free read**: [predict] only reads an
 *   [AtomicReference]-cached snapshot of weights+bias; it does not take
 *   the mutex, so a concurrent SGD step from `OnlineTrainer` cannot stall
 *   the call-screening pipeline.
 * - **Read-modify-write is serialized**: [sgdStep] and [resetToDefaults]
 *   take the mutex, reload current weights, apply the change, persist to
 *   [DeviceModelStore], then atomically publish the new snapshot. Two
 *   concurrent SGD steps cannot lose updates.
 * - **Cold start is automatic**: on the first [predict]/[sgdStep] after a
 *   fresh install, the model loads weights from [DeviceModelStore]; if
 *   the persisted bytes are missing or the schema version mismatches
 *   [DeviceFeatures.SCHEMA_VERSION], the model resets to defaults from
 *   [DefaultWeightsLoader] (Req 3.4).
 *
 * SGD update rule (logistic loss with L2 on weights only):
 *
 * ```
 *   z = bias + Σ w[i] * x[i]
 *   p = σ(z) = 1 / (1 + exp(-z))
 *   for each i:
 *     w[i] -= LEARNING_RATE * sampleWeight * ((p - y) * x[i] + L2 * w[i])
 *   bias  -= LEARNING_RATE * sampleWeight * (p - y)   // no L2 on bias
 * ```
 *
 * Bias deliberately escapes L2 — penalizing the bias would bias the
 * prediction toward `σ(0) = 0.5` regardless of base rate, which is the
 * wrong default for a spam classifier where most calls are legitimate.
 *
 * Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 4.8, 4.9,
 * 6.3.
 */
class DeviceModel(
    private val store: DeviceModelStore,
    private val defaultWeightsLoader: DefaultWeightsLoader,
) {

    /**
     * Immutable snapshot of the model parameters. Held inside an
     * [AtomicReference] so [predict] never sees a torn state during an
     * SGD step.
     */
    private data class Snapshot(val weights: FloatArray, val bias: Float) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is Snapshot) return false
            return bias == other.bias && weights.contentEquals(other.weights)
        }

        override fun hashCode(): Int = 31 * weights.contentHashCode() + bias.hashCode()
    }

    /** Latest committed parameters; `null` until first load. */
    private val cache = AtomicReference<Snapshot?>(null)

    /** Serializes [sgdStep] and [resetToDefaults] (and the cache-prime step). */
    private val mutex = Mutex()

    // ── Public API ────────────────────────────────────────────────────────

    /**
     * Score [features] under the current weights and return a
     * [DevicePrediction] containing `p(BLOCK)`, the verdict, and the
     * top-5 most influential features.
     *
     * Lock-free: reads only the [AtomicReference] cache. The first call
     * after process start may take the mutex briefly to load/initialize
     * weights from [DeviceModelStore].
     */
    suspend fun predict(features: DeviceFeatures): DevicePrediction {
        val snapshot = ensureLoaded()
        val x = features.toFloatArray()
        val w = snapshot.weights

        var z = snapshot.bias
        for (i in 0 until DeviceFeatures.SIZE) {
            z += w[i] * x[i]
        }
        val p = sigmoid(z)
        val verdict = verdictFor(p)
        val top = topContributions(w, x, TOP_K)
        return DevicePrediction(probBlock = p, verdict = verdict, topContributions = top)
    }

    /**
     * Apply one logistic-regression SGD step to the current weights.
     *
     * @param y `1f` for BLOCK, `0f` for ALLOW.
     * @param sampleWeight per-label weight; `IMPLICIT_WEIGHT` for implicit
     *   labels and `EXPLICIT_WEIGHT` for explicit ones (Req 4.8).
     */
    suspend fun sgdStep(features: DeviceFeatures, y: Float, sampleWeight: Float) {
        val x = features.toFloatArray()
        mutex.withLock {
            val current = ensureLoadedLocked()
            val w = current.weights.copyOf()
            var b = current.bias

            // Compute p under the freshly-read weights (not the stale cache).
            var z = b
            for (i in 0 until DeviceFeatures.SIZE) {
                z += w[i] * x[i]
            }
            val p = sigmoid(z)
            val err = p - y // ∂L/∂z

            for (i in 0 until DeviceFeatures.SIZE) {
                val grad = err * x[i] + L2 * w[i]
                w[i] -= LEARNING_RATE * sampleWeight * grad
            }
            // Bias updates without L2 penalty.
            b -= LEARNING_RATE * sampleWeight * err

            val next = Snapshot(weights = w, bias = b)
            persist(next)
            cache.set(next)

            // Диагностический лог в adb logcat: показывает движение модели
            // на каждом SGD-шаге. Без него у юзера/разработчика нет способа
            // подтвердить, что implicit/explicit-сигналы реально двигают веса.
            // Тег `SpamBlocker_SGD` чтобы было удобно фильтровать:
            //   adb logcat -s SpamBlocker_SGD:I
            var zAfter = b
            for (i in 0 until DeviceFeatures.SIZE) {
                zAfter += w[i] * x[i]
            }
            val pAfter = sigmoid(zAfter)
            android.util.Log.i(
                "SpamBlocker_SGD",
                "sgd y=$y w=$sampleWeight " +
                    "p=${"%.4f".format(p)}->${"%.4f".format(pAfter)} " +
                    "d=${"%+.4f".format(pAfter - p)} " +
                    "bias=${"%+.4f".format(b)}",
            )
        }
    }

    /**
     * Reset weights and bias to the shipped defaults (Req 3.4, 7.6).
     * Invoked by Wipe and by the schema-mismatch recovery path.
     */
    suspend fun resetToDefaults() {
        mutex.withLock {
            resetToDefaultsLocked()
        }
    }

    // ── Internals ─────────────────────────────────────────────────────────

    /**
     * Lock-free fast path: returns the cached snapshot if present;
     * otherwise serializes through [mutex] to load it once.
     */
    private suspend fun ensureLoaded(): Snapshot {
        cache.get()?.let { return it }
        return mutex.withLock { ensureLoadedLocked() }
    }

    /**
     * Must be called while holding [mutex]. Returns the current snapshot,
     * loading from [store] (or resetting to defaults if missing /
     * schema-mismatched) on first use.
     */
    private suspend fun ensureLoadedLocked(): Snapshot {
        cache.get()?.let { return it }

        val schema = store.featureSchemaFlow.first()
        val json = store.weightsJsonFlow.first()

        if (json == null || schema != DeviceFeatures.SCHEMA_VERSION) {
            return resetToDefaultsLocked()
        }

        val weights = parseWeightsJson(json)
        if (weights == null) {
            // Persisted JSON is malformed — recover by resetting to defaults
            // rather than throwing on the call-screening hot path.
            return resetToDefaultsLocked()
        }

        val bias = store.biasFlow.first()
        val snapshot = Snapshot(weights = weights, bias = bias)
        cache.set(snapshot)
        return snapshot
    }

    /** Must be called while holding [mutex]. */
    private suspend fun resetToDefaultsLocked(): Snapshot {
        val defaults = defaultWeightsLoader.load()
        val w = defaults.weights.copyOf()
        val snapshot = Snapshot(weights = w, bias = defaults.bias)
        persist(snapshot)
        store.setFeatureSchema(DeviceFeatures.SCHEMA_VERSION)
        cache.set(snapshot)
        return snapshot
    }

    /** Persist weights JSON + bias to [store]. Must be called while holding [mutex]. */
    private suspend fun persist(snapshot: Snapshot) {
        store.setWeightsJson(serializeWeights(snapshot.weights))
        store.setBias(snapshot.bias)
    }

    /**
     * Serialize weights as a `{ "feature_name": weight }` map keyed by
     * [DeviceFeatures.NAMES]. Stable enough for round-trip via
     * [parseWeightsJson] regardless of key iteration order.
     */
    private fun serializeWeights(weights: FloatArray): String {
        val obj = JSONObject()
        for (i in 0 until DeviceFeatures.SIZE) {
            obj.put(DeviceFeatures.NAMES[i], weights[i].toDouble())
        }
        return obj.toString()
    }

    /**
     * Parse a weights JSON object back into a [FloatArray] indexed by
     * [DeviceFeatures.NAMES]. Returns `null` if the JSON is malformed or
     * any feature name is missing — callers should treat this as "reset
     * to defaults" rather than crashing the screening pipeline.
     */
    private fun parseWeightsJson(json: String): FloatArray? {
        return try {
            val obj = JSONObject(json)
            val out = FloatArray(DeviceFeatures.SIZE)
            for (i in 0 until DeviceFeatures.SIZE) {
                val name = DeviceFeatures.NAMES[i]
                if (!obj.has(name)) return null
                out[i] = obj.getDouble(name).toFloat()
            }
            out
        } catch (_: JSONException) {
            null
        }
    }

    private fun verdictFor(p: Float): DeviceVerdict = when {
        p >= BLOCK_HIGH_THRESHOLD -> DeviceVerdict.BLOCK_HIGH
        p >= WARN_THRESHOLD -> DeviceVerdict.WARN
        else -> DeviceVerdict.ALLOW
    }

    /**
     * Top-K feature contributions ranked by `|w[i] * x[i]|` descending,
     * with stable tie-break by feature index ascending so the result is
     * fully deterministic for property tests.
     */
    private fun topContributions(
        weights: FloatArray,
        values: FloatArray,
        k: Int,
    ): List<FeatureContribution> {
        val n = DeviceFeatures.SIZE
        val indices = IntArray(n) { it }
        val signed = FloatArray(n) { i -> weights[i] * values[i] }

        // Sort indices by |signed| desc, then by index asc on ties.
        // Use boxed indices because IntArray.sortedWith is not available
        // directly; n is at most 20 so allocation cost is negligible.
        val ordered = indices.toTypedArray().apply {
            sortWith(
                Comparator<Int> { a, b ->
                    val cmp = abs(signed[b]).compareTo(abs(signed[a]))
                    if (cmp != 0) cmp else a.compareTo(b)
                },
            )
        }

        val take = if (k < n) k else n
        val result = ArrayList<FeatureContribution>(take)
        for (j in 0 until take) {
            val i = ordered[j]
            result.add(
                FeatureContribution(
                    name = DeviceFeatures.NAMES[i],
                    weight = weights[i],
                    value = values[i],
                    signed = signed[i],
                ),
            )
        }
        return result
    }

    private fun sigmoid(z: Float): Float {
        // Numerically-stable sigmoid: avoid overflow in exp(-z) for large
        // negative z by branching on sign.
        return if (z >= 0f) {
            val e = exp(-z.toDouble())
            (1.0 / (1.0 + e)).toFloat()
        } else {
            val e = exp(z.toDouble())
            (e / (1.0 + e)).toFloat()
        }
    }

    companion object {
        const val LEARNING_RATE: Float = 0.05f
        const val L2: Float = 1e-4f

        /**
         * Sample weights for SGD. Explicit feedback must move weights
         * strictly more than implicit (Req 4.8); 0.5 vs 1.5 yields a 3×
         * factor so a single explicit label outweighs three implicit ones.
         */
        const val IMPLICIT_WEIGHT: Float = 0.5f
        const val EXPLICIT_WEIGHT: Float = 1.5f

        /**
         * Verdict thresholds. `BLOCK_HIGH_THRESHOLD = 0.80` is the high
         * confidence cutoff required by Req 4.9 (false BLOCK is strictly
         * costlier than false ALLOW).
         */
        const val BLOCK_HIGH_THRESHOLD: Float = 0.80f
        const val WARN_THRESHOLD: Float = 0.45f

        /** Number of features returned by the explainability detail view (Req 6.3). */
        const val TOP_K: Int = 5
    }
}
