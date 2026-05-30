package com.antispam.blocker.data.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import com.antispam.blocker.BuildConfig
import androidx.room.TypeConverters
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import com.antispam.blocker.data.db.dao.AllowedNumberDao
import com.antispam.blocker.data.db.dao.AnswerBotMessageDao
import com.antispam.blocker.data.db.dao.AppUsageEventDao
import com.antispam.blocker.data.db.dao.BlockedNumberDao
import com.antispam.blocker.data.db.dao.CallEventDao
import com.antispam.blocker.data.db.dao.CallRecordDao
import com.antispam.blocker.data.db.dao.DecisionRecordDao
import com.antispam.blocker.data.db.dao.CallerLookupDao
import com.antispam.blocker.data.db.dao.FeatureSnapshotDao
import com.antispam.blocker.data.db.dao.NotificationEventDao
import com.antispam.blocker.data.db.dao.TrainingDataDao
import com.antispam.blocker.data.db.entity.AllowedNumber
import com.antispam.blocker.data.db.entity.AnswerBotMessageEntity
import com.antispam.blocker.data.db.entity.AppUsageEvent
import com.antispam.blocker.data.db.entity.CallerLookup
import com.antispam.blocker.data.db.entity.BlockedNumber
import com.antispam.blocker.data.db.entity.CallEvent
import com.antispam.blocker.data.db.entity.CallRecord
import com.antispam.blocker.data.db.entity.DecisionRecord
import com.antispam.blocker.data.db.entity.FeatureSnapshot
import com.antispam.blocker.data.db.entity.NotificationEvent
import com.antispam.blocker.data.db.entity.TrainingData
import com.antispam.blocker.data.db.util.CallDirectionConverter
import com.antispam.blocker.data.db.util.CallStateConverter
import com.antispam.blocker.data.db.util.CategoryBucketConverter
import com.antispam.blocker.data.db.util.VerdictConverter

@Database(
    entities = [
        BlockedNumber::class,
        AllowedNumber::class,
        CallRecord::class,
        TrainingData::class,
        DecisionRecord::class,
        CallEvent::class,
        NotificationEvent::class,
        AppUsageEvent::class,
        FeatureSnapshot::class,
        CallerLookup::class,
        AnswerBotMessageEntity::class
    ],
    version = AppDatabase.SCHEMA_VERSION,
    exportSchema = true
)
@TypeConverters(
    VerdictConverter::class,
    CallDirectionConverter::class,
    CallStateConverter::class,
    CategoryBucketConverter::class
)
abstract class AppDatabase : RoomDatabase() {

    abstract fun blockedNumberDao(): BlockedNumberDao
    abstract fun allowedNumberDao(): AllowedNumberDao
    abstract fun callRecordDao(): CallRecordDao
    abstract fun trainingDataDao(): TrainingDataDao
    abstract fun decisionRecordDao(): DecisionRecordDao
    abstract fun callEventDao(): CallEventDao
    abstract fun notificationEventDao(): NotificationEventDao
    abstract fun appUsageEventDao(): AppUsageEventDao
    abstract fun featureSnapshotDao(): FeatureSnapshotDao
    abstract fun callerLookupDao(): CallerLookupDao
    abstract fun answerBotMessageDao(): AnswerBotMessageDao

    companion object {
        /**
         * Текущая версия Room-схемы. Совпадает с `@Database(version = ...)` выше
         * (Kotlin требует константное выражение в аннотации, поэтому ссылаемся
         * как `AppDatabase.SCHEMA_VERSION`). RemoteUpdateWorker сравнивает с
         * `manifest.min_app_db_version` и скипает несовместимые манифесты.
         */
        const val SCHEMA_VERSION: Int = 8

        @Volatile
        private var INSTANCE: AppDatabase? = null

        private val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE blocked_numbers ADD COLUMN pattern TEXT DEFAULT NULL")
            }
        }

        private val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS training_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        normalizedNumber TEXT NOT NULL,
                        featuresJson TEXT NOT NULL,
                        label TEXT NOT NULL,
                        weight REAL NOT NULL DEFAULT 1.0,
                        userAction TEXT,
                        timestamp INTEGER NOT NULL
                    )
                """.trimIndent())
                db.execSQL("CREATE UNIQUE INDEX IF NOT EXISTS index_training_data_normalizedNumber_timestamp ON training_data(normalizedNumber, timestamp)")
            }
        }

        private val MIGRATION_3_4 = object : Migration(3, 4) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS decision_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        rawNumber TEXT,
                        normalizedNumber TEXT,
                        verdict TEXT NOT NULL,
                        score INTEGER NOT NULL,
                        source TEXT NOT NULL,
                        confidence TEXT NOT NULL,
                        modelAllowProb REAL NOT NULL,
                        modelWarnProb REAL NOT NULL,
                        modelBlockProb REAL NOT NULL,
                        modelInputSize INTEGER NOT NULL,
                        featuresJson TEXT NOT NULL,
                        reasonsJson TEXT NOT NULL,
                        activeFactorsJson TEXT NOT NULL,
                        ruleScore INTEGER NOT NULL,
                        warnThreshold INTEGER NOT NULL,
                        blockThreshold INTEGER NOT NULL,
                        userAction TEXT,
                        userActionTimestamp INTEGER,
                        modelVersion TEXT
                    )
                """.trimIndent())
                db.execSQL("CREATE INDEX IF NOT EXISTS index_decision_records_timestamp ON decision_records(timestamp)")
                db.execSQL("CREATE INDEX IF NOT EXISTS index_decision_records_normalizedNumber ON decision_records(normalizedNumber)")
            }
        }

        /**
         * v4 → v5: adds Device_Model on-device telemetry tables.
         *
         * Creates `call_event`, `notification_event`, `app_usage_event` and
         * `feature_snapshot` (with FK on `call_event(id)` ON DELETE SET NULL)
         * plus their secondary indices. Existing tables (`blocked_numbers`,
         * `allowed_numbers`, `call_records`, `training_data`, `decision_records`)
         * are not touched — migration is purely additive.
         *
         * Index names follow Room's auto-generated convention
         * `index_<table>_<column>` so that runtime schema validation against
         * the entities defined in [AppDatabase] passes after migration.
         */
        internal val MIGRATION_4_5 = object : Migration(4, 5) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS call_event (
                        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        normalizedNumber TEXT,
                        isHidden INTEGER NOT NULL,
                        direction TEXT NOT NULL,
                        state TEXT NOT NULL,
                        durationMs INTEGER NOT NULL,
                        startedAt INTEGER NOT NULL,
                        endedAt INTEGER,
                        isContact INTEGER NOT NULL DEFAULT 0
                    )
                """.trimIndent())
                db.execSQL("CREATE INDEX IF NOT EXISTS index_call_event_startedAt ON call_event(startedAt)")
                db.execSQL("CREATE INDEX IF NOT EXISTS index_call_event_normalizedNumber ON call_event(normalizedNumber)")

                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS notification_event (
                        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        packageName TEXT NOT NULL,
                        categoryBucket TEXT NOT NULL,
                        timestamp INTEGER NOT NULL
                    )
                """.trimIndent())
                db.execSQL("CREATE INDEX IF NOT EXISTS index_notification_event_timestamp ON notification_event(timestamp)")

                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS app_usage_event (
                        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        packageName TEXT NOT NULL,
                        categoryBucket TEXT NOT NULL,
                        foregroundAt INTEGER NOT NULL
                    )
                """.trimIndent())
                db.execSQL("CREATE INDEX IF NOT EXISTS index_app_usage_event_foregroundAt ON app_usage_event(foregroundAt)")

                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS feature_snapshot (
                        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        callEventId INTEGER,
                        normalizedNumber TEXT,
                        timestamp INTEGER NOT NULL,
                        featuresJson TEXT NOT NULL,
                        featureSchemaVersion INTEGER NOT NULL,
                        weightsHash TEXT,
                        deviceProbBlock REAL NOT NULL,
                        FOREIGN KEY(callEventId) REFERENCES call_event(id) ON UPDATE NO ACTION ON DELETE SET NULL
                    )
                """.trimIndent())
                db.execSQL("CREATE INDEX IF NOT EXISTS index_feature_snapshot_callEventId ON feature_snapshot(callEventId)")
                db.execSQL("CREATE INDEX IF NOT EXISTS index_feature_snapshot_timestamp ON feature_snapshot(timestamp)")
            }
        }

        /**
         * v5 → v6: добавляет nullable `featureSnapshotId` в `call_records`.
         * Связывает каждую запись журнала с конкретным `feature_snapshot.id`,
         * посчитанным на момент того звонка. Без этого long-press в журнале
         * использовал `featureSnapshotDao.getLatestForNumber(...)` —
         * для номеров с несколькими звонками это возвращало snapshot
         * НЕ ТОГО звонка, по которому юзер long-press'нул.
         *
         * Покрываем только новые записи: старые остаются с `featureSnapshotId = NULL`,
         * UI делает fallback на `getLatestForNumber` для них.
         */
        internal val MIGRATION_5_6 = object : Migration(5, 6) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE call_records ADD COLUMN featureSnapshotId INTEGER DEFAULT NULL")
                db.execSQL("CREATE INDEX IF NOT EXISTS index_call_records_featureSnapshotId ON call_records(featureSnapshotId)")
            }
        }

        /**
         * v6 → v7: добавляет таблицу `caller_lookup` для кэша определения звонящего.
         *
         * Хранит результаты libphonenumber-геокодирования (регион, оператор,
         * тип номера) и опциональные 2GIS-результаты (название организации,
         * адрес). Поле `source` = "offline" | "2gis" | "negative".
         * Поле `lookedUpAt` используется для TTL-инвалидации.
         * Миграция чисто аддитивная — существующие таблицы не затрагиваются.
         */
        internal val MIGRATION_6_7 = object : Migration(6, 7) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS caller_lookup (
                        normalizedNumber TEXT NOT NULL PRIMARY KEY,
                        orgName TEXT,
                        address TEXT,
                        region TEXT,
                        carrier TEXT,
                        numberType TEXT NOT NULL DEFAULT 'unknown',
                        source TEXT NOT NULL DEFAULT 'offline',
                        lookedUpAt INTEGER NOT NULL
                    )
                """.trimIndent())
                db.execSQL(
                    "CREATE UNIQUE INDEX IF NOT EXISTS " +
                    "index_caller_lookup_normalizedNumber ON caller_lookup(normalizedNumber)"
                )
            }
        }

        /**
         * v7 → v8: добавляет таблицу `answer_bot_messages` для автоответчика.
         *
         * Хранит расшифрованные голосовые сообщения спам-звонков: номер,
         * текст расшифровки, путь к аудиофайлу, длительность, флаги
         * прослушано/спам. Миграция чисто аддитивная.
         */
        internal val MIGRATION_7_8 = object : Migration(7, 8) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS answer_bot_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        normalizedNumber TEXT NOT NULL,
                        transcription TEXT,
                        audioPath TEXT NOT NULL,
                        durationMs INTEGER NOT NULL DEFAULT 0,
                        played INTEGER NOT NULL DEFAULT 0,
                        spam INTEGER DEFAULT NULL,
                        timestamp INTEGER NOT NULL
                    )
                """.trimIndent())
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS " +
                    "index_answer_bot_messages_timestamp ON answer_bot_messages(timestamp)"
                )
            }
        }

        fun getInstance(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val builder = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "spam_blocker_db",
                ).addMigrations(MIGRATION_1_2, MIGRATION_2_3, MIGRATION_3_4, MIGRATION_4_5, MIGRATION_5_6, MIGRATION_6_7, MIGRATION_7_8)
                // P0 #2: destructive-fallback только в debug — на release, если
                // окажется, что новой миграции нет, лучше упасть в Logcat и
                // дать разработчику добавить её, чем молча стереть пользовательские
                // данные (call_record, training_data, decision_record,
                // feature_snapshot, веса DataStore-key'ев).
                if (BuildConfig.DEBUG) {
                    builder.fallbackToDestructiveMigration()
                }
                val instance = builder.build()
                INSTANCE = instance
                instance
            }
        }
    }
}
