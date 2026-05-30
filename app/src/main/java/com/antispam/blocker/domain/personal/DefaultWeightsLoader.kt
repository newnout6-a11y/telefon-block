package com.antispam.blocker.domain.personal

import android.content.res.AssetManager
import android.util.Log
import org.json.JSONException
import org.json.JSONObject
import java.io.IOException

/**
 * Reads shipped default weights for [Device_Model] from
 * `app/src/main/assets/device_model_default_weights.json` and validates that
 * its schema matches the canonical [DeviceFeatures.NAMES] order.
 *
 * Cold-start contract (Req 3.4): on a fresh install (or after Wipe) the loader
 * MUST return a usable [DefaultWeights] so that the very first incoming call
 * can be scored without any network or training step.
 *
 * Hand-tuned heuristic priors (Req 3.5): the asset and the [FALLBACK] table
 * encode the same numbers — `is_contact = -3.0`, `previously_rejected = +2.0`,
 * `is_night_time = +0.5`, `bias = -0.5`, etc. Should the asset ever be
 * missing, malformed, or schema-mismatched, the loader logs and silently
 * returns [FALLBACK] so on-call hot paths never throw.
 *
 * Validates: Requirements 3.4 (cold-start defaults), 3.5 (specific prior
 * values).
 */
class DefaultWeightsLoader(
    private val assets: AssetManager,
) {

    /**
     * Immutable view of the default-weights asset.
     *
     * `weights.size` is guaranteed to equal [DeviceFeatures.SIZE]; this
     * invariant is enforced by [load] (asset path) and by construction
     * (FALLBACK constant).
     */
    data class DefaultWeights(
        val schemaVersion: Int,
        val featureNames: List<String>,
        val weights: FloatArray,
        val bias: Float,
    ) {
        // FloatArray needs content-based equality — the data-class default uses
        // reference equality, which makes two identical loads compare unequal
        // and breaks property tests.
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is DefaultWeights) return false
            return schemaVersion == other.schemaVersion &&
                featureNames == other.featureNames &&
                weights.contentEquals(other.weights) &&
                bias == other.bias
        }

        override fun hashCode(): Int {
            var result = schemaVersion
            result = 31 * result + featureNames.hashCode()
            result = 31 * result + weights.contentHashCode()
            result = 31 * result + bias.hashCode()
            return result
        }
    }

    /**
     * Loads default weights from [ASSET_PATH], validating schema and feature
     * order. On any [IOException], [JSONException], or validation failure the
     * loader logs a warning and returns [FALLBACK] so the caller keeps a
     * usable model.
     */
    fun load(): DefaultWeights {
        val raw = try {
            assets.open(ASSET_PATH).bufferedReader().use { it.readText() }
        } catch (e: IOException) {
            Log.w(TAG, "asset $ASSET_PATH unavailable, falling back to constants", e)
            return FALLBACK
        }

        return try {
            parseAndValidate(raw)
        } catch (e: JSONException) {
            Log.w(TAG, "$ASSET_PATH parse failed, falling back to constants", e)
            FALLBACK
        } catch (e: IllegalStateException) {
            // Validation errors (schema/feature/weights mismatch) surface as
            // IllegalStateException from check(...) inside parseAndValidate.
            Log.w(TAG, "$ASSET_PATH validation failed, falling back to constants", e)
            FALLBACK
        }
    }

    private fun parseAndValidate(raw: String): DefaultWeights {
        val obj = JSONObject(raw)

        val schemaVersion = obj.getInt("schema_version")
        check(schemaVersion == DeviceFeatures.SCHEMA_VERSION) {
            "schema_version mismatch: asset=$schemaVersion, " +
                "expected=${DeviceFeatures.SCHEMA_VERSION}"
        }

        val featureCount = obj.getInt("feature_count")
        check(featureCount == DeviceFeatures.SIZE) {
            "feature_count mismatch: asset=$featureCount, expected=${DeviceFeatures.SIZE}"
        }

        val featuresArr = obj.getJSONArray("features")
        check(featuresArr.length() == DeviceFeatures.SIZE) {
            "features array size mismatch: asset=${featuresArr.length()}, " +
                "expected=${DeviceFeatures.SIZE}"
        }
        val featureNames = ArrayList<String>(featuresArr.length())
        for (i in 0 until featuresArr.length()) {
            val name = featuresArr.getString(i)
            check(name == DeviceFeatures.NAMES[i]) {
                "features[$i] mismatch: asset=\"$name\", " +
                    "expected=\"${DeviceFeatures.NAMES[i]}\""
            }
            featureNames.add(name)
        }

        val weightsArr = obj.getJSONArray("weights")
        check(weightsArr.length() == featureCount) {
            "weights array size (${weightsArr.length()}) does not match " +
                "feature_count ($featureCount)"
        }
        val weights = FloatArray(weightsArr.length()) { i ->
            weightsArr.getDouble(i).toFloat()
        }

        val bias = obj.getDouble("bias").toFloat()

        return DefaultWeights(
            schemaVersion = schemaVersion,
            featureNames = featureNames,
            weights = weights,
            bias = bias,
        )
    }

    companion object {
        private const val TAG = "DefaultWeightsLoader"

        /** Path inside `assets/` for the shipped default-weights JSON. */
        const val ASSET_PATH: String = "device_model_default_weights.json"

        /**
         * Hard-coded fallback table mirroring
         * `assets/device_model_default_weights.json` byte-for-byte (Req 3.5).
         *
         * Used when the asset is absent or malformed. Keeping these numbers
         * here as a constant means the model can still cold-start even if the
         * APK ships without the asset (e.g. after a corrupt update).
         *
         * Order MUST match [DeviceFeatures.NAMES]. Any change to the asset
         * REQUIRES the same change here, otherwise property test
         * "Default weights load matches asset" (Property 8) will fail.
         */
        val FALLBACK: DefaultWeights = DefaultWeights(
            schemaVersion = DeviceFeatures.SCHEMA_VERSION,
            featureNames = DeviceFeatures.NAMES,
            weights = floatArrayOf(
                -3.0f, // is_contact
                2.0f,  // previously_rejected
                0.5f,  // is_night_time
                0.0f,  // is_weekend
                1.0f,  // prev_missed_no_callback_24h
                -1.5f, // prev_outgoing_after_missed
                0.8f,  // recent_bank_app_30m
                0.5f,  // recent_gov_app_30m
                0.3f,  // recent_marketplace_app_30m
                -0.2f, // recent_messenger_app_30m
                0.7f,  // notif_bank_recent_10m
                0.3f,  // notif_marketplace_recent_10m
                -0.4f, // same_carrier_as_user
                0.0f,  // is_short_code
                1.5f,  // same_prefix_call_count_7d_norm
                -1.5f, // answer_rate_for_number_norm
                1.5f,  // hidden_number
            ),
            bias = -0.5f,
        )
    }
}
