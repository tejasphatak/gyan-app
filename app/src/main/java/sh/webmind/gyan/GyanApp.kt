package sh.webmind.gyan

import android.app.Application
import android.util.Log
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter

/**
 * Application class — catches ALL uncaught exceptions.
 * Saves crash log to file + sends to ntfy.sh.
 */
class GyanApp : Application() {

    override fun onCreate() {
        super.onCreate()

        // Global crash handler
        val defaultHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            try {
                val sw = StringWriter()
                throwable.printStackTrace(PrintWriter(sw))
                val report = """
                    |=== GYAN FATAL CRASH ===
                    |Thread: ${thread.name}
                    |Device: ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}
                    |Android: ${android.os.Build.VERSION.RELEASE} (SDK ${android.os.Build.VERSION.SDK_INT})
                    |
                    |${sw}
                """.trimMargin()

                // Save to file
                File(filesDir, "fatal_crash.txt").writeText(report)

                // Send to ntfy (best-effort, app is dying)
                try {
                    val url = java.net.URL("https://ntfy.sh/gyan-crashes")
                    val conn = url.openConnection() as java.net.HttpURLConnection
                    conn.requestMethod = "POST"
                    conn.setRequestProperty("Title", "FATAL: ${throwable.javaClass.simpleName}")
                    conn.setRequestProperty("Tags", "skull")
                    conn.connectTimeout = 3000
                    conn.readTimeout = 3000
                    conn.doOutput = true
                    conn.outputStream.use { it.write(report.take(4000).toByteArray()) }
                    conn.responseCode  // force send
                    conn.disconnect()
                } catch (_: Exception) {}

                Log.e("GyanCrash", report)
            } catch (_: Exception) {}

            // Call default handler (shows system crash dialog)
            defaultHandler?.uncaughtException(thread, throwable)
        }
    }
}
