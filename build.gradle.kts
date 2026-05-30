plugins {
    id("com.android.application") version "8.7.3" apply false
    id("org.jetbrains.kotlin.android") version "2.0.21" apply false
    id("com.google.devtools.ksp") version "2.0.21-1.0.27" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.21" apply false
}

// ── Workaround: Room (через KSP) использует нативную sqlite-jdbc для верификации @Query.
// На Windows JVM иногда видит java.io.tmpdir == C:\WINDOWS\TEMP, куда обычный пользователь
// не может писать/исполнять — KSP падает с
// "The temp dir [...] must be readable, writable and allow executables".
// Принудительно указываем sqlite-jdbc писабельный каталог под user.home, и
// прокидываем системное свойство в kotlin-daemon (где реально живёт KSP).
run {
    val sqliteTmp = java.io.File(System.getProperty("user.home"), ".gradle/sqlite-tmp")
        .also { it.mkdirs() }
    System.setProperty("org.sqlite.tmpdir", sqliteTmp.absolutePath)
    allprojects {
        tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
            kotlinDaemonJvmArguments.add("-Dorg.sqlite.tmpdir=${sqliteTmp.absolutePath}")
        }
    }
}
