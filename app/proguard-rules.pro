# =====================================================================
# Spam Blocker — ProGuard rules for release (isMinifyEnabled = true)
# =====================================================================
#
# Release build runs R8 with shrink + obfuscate. Without explicit -keep
# rules, reflection-based loaders inside TFLite, Room, DataStore, WorkManager
# and JSON-org get stripped or renamed → ClassNotFoundException at runtime.
# Each block below pins the things we know reflectively load.

# ---------------------------------------------------------------------
# Room
# ---------------------------------------------------------------------
-keep class * extends androidx.room.RoomDatabase
-keep @androidx.room.Entity class *
-dontwarn androidx.room.paging.**

# ---------------------------------------------------------------------
# TensorFlow Lite — interpreter, NNAPI delegates, support library.
# Without these, Interpreter.run() can crash with NoSuchMethodError /
# UnsatisfiedLinkError on release builds because the native bridges are
# loaded reflectively through JNI shims.
# ---------------------------------------------------------------------
-keep class org.tensorflow.lite.** { *; }
-keepclassmembers class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**

# tensorflow-lite-support reflectively resolves Image / TensorBuffer impls.
-keep class org.tensorflow.lite.support.** { *; }
-dontwarn org.tensorflow.lite.support.**

# Native methods on TFLite must keep original names so JNI binding works.
-keepclasseswithmembernames class * {
    native <methods>;
}

# ---------------------------------------------------------------------
# androidx.datastore (Preferences DataStore reflectively constructs
# `Preferences` proto; minify removes ctors otherwise).
# ---------------------------------------------------------------------
-keep class androidx.datastore.** { *; }
-keep class androidx.datastore.preferences.protobuf.** { *; }
-dontwarn androidx.datastore.**

# ---------------------------------------------------------------------
# WorkManager — workers are instantiated reflectively by class name.
# RemoteUpdateWorker & co. live under the app package.
# ---------------------------------------------------------------------
-keep class androidx.work.impl.** { *; }
-keep class * extends androidx.work.Worker { *; }
-keep class * extends androidx.work.CoroutineWorker { *; }
-keep class * extends androidx.work.ListenableWorker { *; }
-dontwarn androidx.work.**

# Our own Workers (path-explicit so even if the parent rule misses,
# RemoteUpdateWorker still survives obfuscation).
-keep class com.antispam.blocker.data.worker.** { *; }

# ---------------------------------------------------------------------
# Kotlin coroutines — keep service loader entries.
# ---------------------------------------------------------------------
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}
-keepclassmembers class kotlinx.coroutines.** {
    volatile <fields>;
}
-dontwarn kotlinx.coroutines.**

# ---------------------------------------------------------------------
# JSON-org (used by ModelCard.load, RemoteUpdateWorker manifest parser).
# Stable class — but keep just in case future R8 versions get aggressive.
# ---------------------------------------------------------------------
-keep class org.json.** { *; }
-dontwarn org.json.**

# ---------------------------------------------------------------------
# BroadcastReceiver — SpamActionReceiver registered in AndroidManifest
# is instantiated reflectively when notification action fires.
# ---------------------------------------------------------------------
-keep class com.antispam.blocker.notification.SpamActionReceiver { *; }

# ---------------------------------------------------------------------
# Compose / Material3 — already covered by the AGP-bundled rules under
# `proguard-android-optimize.txt`, no extra entries needed.
# ---------------------------------------------------------------------
