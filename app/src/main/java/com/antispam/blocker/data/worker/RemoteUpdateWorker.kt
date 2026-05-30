package com.antispam.blocker.data.worker

import android.content.Context
import android.util.Log
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import com.antispam.blocker.SpamBlockerApp
import com.antispam.blocker.data.assets.CsvSpamImporter
import com.antispam.blocker.data.db.AppDatabase
import com.antispam.blocker.data.repository.BlockListRepository
import com.antispam.blocker.domain.model.SpamModel
import com.antispam.blocker.domain.categorization.TFLiteAppCategoryClassifier
import com.antispam.blocker.domain.scoring.DefCodeOperatorRiskTable
import com.antispam.blocker.domain.scoring.DefCodeRiskTable
import com.antispam.blocker.domain.scoring.OperatorBucketTable
import com.antispam.blocker.domain.scoring.PrefixHistogramTable
import com.antispam.blocker.domain.scoring.PrefixRiskTable
import com.antispam.blocker.util.PhoneNormalizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.KeyFactory
import java.security.MessageDigest
import java.security.Signature
import java.security.spec.X509EncodedKeySpec
import java.util.Base64
import java.util.concurrent.TimeUnit

/**
 * Раз в N часов тянет manifest.json с публичного URL (по умолчанию —
 * `releases/latest/` нашего GitHub-репозитория) и обновляет локальные копии
 * критичных ассетов в `filesDir`. Каждый файл проверяется по SHA256.
 *
 * Поддерживаемые ассеты (см. [ALLOWED_FILES]):
 *   - spam_numbers.csv               — known-spam список
 *   - prefix_risk.json               — legacy single-number prefix risk
 *   - prefix_histogram.json          — Phase 3 histogram (block/warn/seen)
 *   - prefix_histogram_3.json        — Phase 4B 3-digit prefix histogram
 *   - prefix_histogram_7.json        — Phase 4B 7-digit prefix histogram
 *   - def_code_risk.json             — Phase 3 def-code risk
 *   - def_code_operator_risk.json    — Phase 4B def-code × operator risk
 *   - operator_bucket.json           — Phase 3 operator bucketing
 *   - spam_model.tflite              — TFLite student
 *   - model_card.json                — thresholds (warm + cold) + meta
 *   - app_category_model.tflite      — TFLite app-category char-CNN classifier
 *   - app_category_vocab.txt         — char-n-gram vocabulary for app-category model
 *   - app_category_card.json         — app-category model card (metrics + categories_order)
 *
 * Никаких персональных данных наружу не отправляется — только GET-запросы
 * за публичной статикой. Если manifest недоступен / SHA не сошёлся / 304 —
 * тихо ретраится.
 */
class RemoteUpdateWorker(
    appContext: Context,
    params: WorkerParameters
) : CoroutineWorker(appContext, params) {

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val context = applicationContext
        val app = SpamBlockerApp.instance
        val settings = app.settingsStore
        if (!settings.dbUpdateEnabledSnapshot()) return@withContext Result.success()

        val manifestUrl = inputData.getString(KEY_URL)?.takeIf { it.isNotBlank() }
            ?: settings.dbUpdateUrlSnapshot().takeIf { it.isNotBlank() }
            ?: null  // will use fallback list below

        return@withContext try {
            // PR-4: fallback URL logic. Tries user-configured URL first,
            // then iterates MANIFEST_FALLBACK_URLS until one succeeds.
            val urlsToTry: List<String> = if (manifestUrl != null) {
                listOf(manifestUrl) + MANIFEST_FALLBACK_URLS
            } else {
                MANIFEST_FALLBACK_URLS
            }
            var manifestJson: String? = null
            var usedUrl: String? = null
            for (url in urlsToTry) {
                val result = httpGetText(url)
                if (result != null) {
                    manifestJson = result
                    usedUrl = url
                    break
                }
                Log.w(TAG, "manifest fetch failed for $url, trying next fallback...")
            }
            if (manifestJson == null || usedUrl == null) {
                Log.w(TAG, "all manifest URLs failed (${urlsToTry.size} tried)")
                return@withContext Result.retry()
            }
            Log.i(TAG, "manifest fetched from: $usedUrl")
            val manifestUrl = usedUrl  // shadow for baseUrl computation below

            // P0 #3: graceful ECDSA verify. Если в APK assets лежит
            // manifest_pubkey.pem и на сервере рядом с manifest.json есть
            // manifest.json.sig — подпись валидируется. Иначе шаг
            // skip'ается (обратная совместимость с не-подписанными
            // manifest'ами). При несовпадении подписи → retry, ассеты не
            // применяются (защита от supply-chain подмены модели).
            val verifyOk = verifyManifestSignature(context, manifestJson, manifestUrl)
            if (!verifyOk) {
                Log.w(TAG, "manifest signature verification FAILED — refusing to apply")
                return@withContext Result.retry()
            }

            val manifest = parseManifest(manifestJson) ?: return@withContext Result.retry()

            // P1 enforce: manifest может требовать схему БД новее, чем у
            // приложения (например, новый spam_model.tflite полагается на
            // поля feature_snapshot, которые ещё не существуют в данной
            // версии APK). Применить такой манифест = модель выдаёт чушь
            // или падает. Скипаем — юзер получит обновление после апдейта APK.
            if (manifest.minAppDbVersion > AppDatabase.SCHEMA_VERSION) {
                Log.w(
                    TAG,
                    "manifest requires db v${manifest.minAppDbVersion} " +
                        "but app is on v${AppDatabase.SCHEMA_VERSION} — skipping update",
                )
                return@withContext Result.success()
            }

            val localManifestFile = File(context.filesDir, LOCAL_MANIFEST_NAME)
            val previousManifest = if (localManifestFile.exists()) {
                runCatching { parseManifest(localManifestFile.readText()) }.getOrNull()
            } else null

            if (previousManifest != null && previousManifest.version == manifest.version) {
                Log.i(TAG, "manifest version=${manifest.version} unchanged, skipping")
                return@withContext Result.success()
            }

            val baseUrl = manifestUrl.substringBeforeLast('/') + "/"
            var anyUpdated = false

            for (entry in manifest.files) {
                val target = File(context.filesDir, entry.localName)
                val existingSha = if (target.exists()) sha256(target) else null
                if (existingSha == entry.sha256) {
                    Log.d(TAG, "${entry.localName} already up-to-date (sha=${entry.sha256.take(8)})")
                    continue
                }

                val tmp = File(context.filesDir, entry.localName + ".tmp")
                val ok = httpDownload(baseUrl + entry.url, tmp)
                if (!ok) {
                    Log.w(TAG, "download failed: ${entry.localName}")
                    tmp.delete()
                    return@withContext Result.retry()
                }

                val downloadedSha = sha256(tmp)
                if (downloadedSha != entry.sha256) {
                    Log.w(TAG, "sha256 mismatch for ${entry.localName} expected=${entry.sha256.take(8)} got=${downloadedSha.take(8)}")
                    tmp.delete()
                    return@withContext Result.retry()
                }

                if (entry.size > 0 && tmp.length() != entry.size) {
                    Log.w(TAG, "size mismatch for ${entry.localName} expected=${entry.size} got=${tmp.length()}")
                    tmp.delete()
                    return@withContext Result.retry()
                }

                if (!tmp.renameTo(target)) {
                    target.delete()
                    if (!tmp.renameTo(target)) {
                        Log.w(TAG, "rename failed for ${entry.localName}")
                        tmp.delete()
                        return@withContext Result.retry()
                    }
                }
                Log.i(TAG, "updated ${entry.localName} sha=${entry.sha256.take(8)} size=${target.length()}")
                anyUpdated = true

                when (entry.localName) {
                    "spam_numbers.csv" -> {
                        val repo = BlockListRepository(
                            app.database.blockedNumberDao(),
                            app.database.allowedNumberDao(),
                            PhoneNormalizer
                        )
                        CsvSpamImporter(context, repo).importFromRemote(target, entry.sha256)
                    }
                    "prefix_risk.json" -> PrefixRiskTable.invalidate()
                    "prefix_histogram.json",
                    "prefix_histogram_3.json",
                    "prefix_histogram_7.json" -> PrefixHistogramTable.invalidate()
                    "def_code_risk.json" -> DefCodeRiskTable.invalidate()
                    "def_code_operator_risk.json" -> DefCodeOperatorRiskTable.invalidate()
                    "operator_bucket.json" -> OperatorBucketTable.invalidate()
                    "spam_model.tflite",
                    "model_card.json" -> SpamModel.invalidate()
                    "app_category_model.tflite",
                    "app_category_vocab.txt",
                    "app_category_card.json" -> TFLiteAppCategoryClassifier.invalidate()
                }
            }

            // Сохраняем актуальный манифест локально.
            localManifestFile.writeText(manifestJson)
            if (anyUpdated) {
                settings.setLastUpdateAt(System.currentTimeMillis())
                settings.setLastUpdateVersion(manifest.version)
            }
            Result.success()
        } catch (t: Throwable) {
            Log.w(TAG, "remote update failed", t)
            Result.retry()
        }
    }

    /**
     * Graceful manifest signature verification (P0 #3).
     *
     * Returns:
     *   - `true` if (a) public key is missing from assets OR (b) `.sig` file is
     *     missing on the server (→ feature is opt-in, manifest applied as-is);
     *   - `true` if signature is present AND verifies against the SHA256 hash
     *     of the manifest body using ECDSA P-256;
     *   - `false` if signature is present but verification failed — caller
     *     MUST refuse to apply the manifest.
     */
    private fun verifyManifestSignature(
        context: Context,
        manifestJson: String,
        manifestUrl: String,
    ): Boolean {
        val pubKey = loadPublicKey(context)
        if (pubKey == null) {
            Log.i(TAG, "verify: skipped (no manifest_pubkey.pem in assets)")
            return true
        }
        val sigBytes = httpDownloadBytes(manifestUrl + ".sig")
        if (sigBytes == null) {
            Log.i(TAG, "verify: skipped (no .sig at server side)")
            return true
        }
        return try {
            val sig = Signature.getInstance("SHA256withECDSA")
            sig.initVerify(pubKey)
            sig.update(manifestJson.toByteArray(Charsets.UTF_8))
            val ok = sig.verify(sigBytes)
            if (ok) Log.i(TAG, "verify: OK") else Log.w(TAG, "verify: bad signature")
            ok
        } catch (t: Throwable) {
            Log.w(TAG, "verify: threw — refusing", t)
            false
        }
    }

    /**
     * Loads the bundled ECDSA P-256 public key from `assets/manifest_pubkey.pem`.
     * Accepts either PEM-armoured (`-----BEGIN PUBLIC KEY-----`) or raw base64
     * X.509 SubjectPublicKeyInfo. Returns `null` if file is missing or unparseable.
     */
    private fun loadPublicKey(context: Context): java.security.PublicKey? {
        return try {
            val raw = context.assets.open(PUBKEY_ASSET).bufferedReader().use { it.readText() }
            val base64 = raw
                .replace("-----BEGIN PUBLIC KEY-----", "")
                .replace("-----END PUBLIC KEY-----", "")
                .replace(Regex("\\s+"), "")
            if (base64.isBlank()) return null
            val der = Base64.getDecoder().decode(base64)
            val keyFactory = KeyFactory.getInstance("EC")
            keyFactory.generatePublic(X509EncodedKeySpec(der))
        } catch (t: Throwable) {
            // Asset missing is the normal opt-in case; only log at debug level.
            null
        }
    }

    /** GET binary body. Returns null on any HTTP/network failure. */
    private fun httpDownloadBytes(url: String): ByteArray? {
        return try {
            val conn = (URL(url).openConnection() as HttpURLConnection).apply {
                connectTimeout = 15_000
                readTimeout = 30_000
                requestMethod = "GET"
            }
            try {
                if (conn.responseCode !in 200..299) return null
                conn.inputStream.use { it.readBytes() }
            } finally {
                conn.disconnect()
            }
        } catch (_: Throwable) {
            null
        }
    }

    companion object {
        private const val TAG = "RemoteUpdateWorker"
        /** Bundled ECDSA P-256 public key (PEM/DER) for graceful manifest verify. */
        private const val PUBKEY_ASSET = "manifest_pubkey.pem"
        const val UNIQUE_NAME = "remote_update_periodic"
        private const val KEY_URL = "manifest_url"
        private const val LOCAL_MANIFEST_NAME = "remote_manifest.json"

        const val DEFAULT_MANIFEST_URL =
            // HEAD ref resolves to the repo's default branch — survives branch renames.
            "https://raw.githubusercontent.com/edi617734-byte/Clone-dadadodo/HEAD/releases/latest/manifest.json"

        /**
         * PR-4: Fallback URLs. Worker пробует каждый по очереди; при
         * 4xx/5xx/timeout переходит к следующему. Первый — GitHub, второй —
         * резервный (настраивается владельцем; пока дублирует GitHub через
         * другой CDN-путь для иллюстрации fallback-логики).
         */
        val MANIFEST_FALLBACK_URLS: List<String> = listOf(
            "https://raw.githubusercontent.com/edi617734-byte/Clone-dadadodo/HEAD/releases/latest/manifest.json",
            // TODO: добавить резервный CDN/домен когда появится:
            // "https://spam-blocker-cdn.example.com/releases/latest/manifest.json",
        )

        /** Ставит периодический pull раз в 6 часов, требует Wi-Fi/data. */
        fun schedule(context: Context) {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .build()
            val request = PeriodicWorkRequestBuilder<RemoteUpdateWorker>(
                6, TimeUnit.HOURS,
                30, TimeUnit.MINUTES
            )
                .setConstraints(constraints)
                .setInitialDelay(15, TimeUnit.MINUTES)
                .build()
            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                UNIQUE_NAME,
                ExistingPeriodicWorkPolicy.KEEP,
                request
            )
        }

        /** Снимает периодическую задачу (например, когда юзер выключил обновления). */
        fun cancel(context: Context) {
            WorkManager.getInstance(context).cancelUniqueWork(UNIQUE_NAME)
        }
    }

    // ──────────────────────────────────── helpers ────────────────────────────

    private data class FileEntry(val localName: String, val url: String, val sha256: String, val size: Long)
    private data class Manifest(val version: String, val minAppDbVersion: Int, val files: List<FileEntry>)

    private fun parseManifest(json: String): Manifest? = try {
        val obj = JSONObject(json)
        val version = obj.optString("version", "").ifBlank { return null }
        val minDb = obj.optInt("min_app_db_version", 0)
        val filesObj = obj.optJSONObject("files") ?: return null
        val entries = mutableListOf<FileEntry>()
        val keys = filesObj.keys()
        while (keys.hasNext()) {
            val name = keys.next()
            if (name !in ALLOWED_FILES) continue
            val node = filesObj.optJSONObject(name) ?: continue
            val sha = node.optString("sha256", "").lowercase()
            if (sha.length != 64) continue
            val url = node.optString("url", name)
            val size = node.optLong("size", 0L)
            entries += FileEntry(localName = name, url = url, sha256 = sha, size = size)
        }
        if (entries.isEmpty()) null else Manifest(version, minDb, entries)
    } catch (_: Throwable) {
        null
    }

    private fun httpGetText(url: String): String? = try {
        val conn = (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 15_000
            readTimeout = 30_000
            requestMethod = "GET"
            setRequestProperty("Accept", "application/json, text/plain")
        }
        try {
            if (conn.responseCode in 200..299) conn.inputStream.bufferedReader().use { it.readText() } else null
        } finally {
            conn.disconnect()
        }
    } catch (_: Throwable) {
        null
    }

    private fun httpDownload(url: String, target: File): Boolean {
        return try {
            val conn = (URL(url).openConnection() as HttpURLConnection).apply {
                connectTimeout = 15_000
                readTimeout = 60_000
                requestMethod = "GET"
            }
            try {
                if (conn.responseCode !in 200..299) return false
                FileOutputStream(target).use { out ->
                    conn.inputStream.use { it.copyTo(out) }
                }
                true
            } finally {
                conn.disconnect()
            }
        } catch (_: Throwable) {
            false
        }
    }

    private fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(8 * 1024)
            while (true) {
                val n = input.read(buf)
                if (n <= 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}

private val ALLOWED_FILES = setOf(
    "spam_numbers.csv",
    "prefix_risk.json",
    "prefix_histogram.json",
    "prefix_histogram_3.json",
    "prefix_histogram_7.json",
    "def_code_risk.json",
    "def_code_operator_risk.json",
    "operator_bucket.json",
    "spam_model.tflite",
    "model_card.json",
    "app_category_model.tflite",
    "app_category_vocab.txt",
    "app_category_card.json",
)
