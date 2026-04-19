package sh.webmind.gyan.data

import android.content.Context
import android.os.Build
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.PrintWriter
import java.io.StringWriter
import java.util.concurrent.TimeUnit

/**
 * Reports crashes/errors to a simple endpoint.
 * Sends: exception stacktrace, device info, app state.
 */
object CrashReporter {

    // POST errors to this endpoint (a simple GitHub Gist or pastebin)
    // For now, save to local file and show to user
    private val http = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.SECONDS)
        .build()

    private var lastError: String = ""

    fun getLastError(): String = lastError

    /** Capture exception with context info. */
    fun capture(context: Context, error: Throwable, extra: String = "") {
        val sw = StringWriter()
        error.printStackTrace(PrintWriter(sw))

        val report = buildString {
            appendLine("=== Gyan Crash Report ===")
            appendLine("Time: ${java.util.Date()}")
            appendLine("Device: ${Build.MANUFACTURER} ${Build.MODEL}")
            appendLine("Android: ${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})")
            appendLine("ABI: ${Build.SUPPORTED_ABIS.joinToString()}")
            appendLine("RAM: ${Runtime.getRuntime().maxMemory() / 1024 / 1024}MB max, ${Runtime.getRuntime().freeMemory() / 1024 / 1024}MB free")
            if (extra.isNotEmpty()) {
                appendLine("Context: $extra")
            }
            appendLine()
            appendLine("=== Stack Trace ===")
            appendLine(sw.toString())
        }

        lastError = report

        // Save to file
        try {
            val file = java.io.File(context.filesDir, "crash_report.txt")
            file.writeText(report)
        } catch (_: Exception) {}

        // Send to ntfy.sh for remote monitoring
        try {
            val request = Request.Builder()
                .url("https://ntfy.sh/gyan-crashes")
                .post(report.take(4000).toRequestBody("text/plain".toMediaType()))
                .addHeader("Title", "Gyan crash: ${error.javaClass.simpleName}")
                .addHeader("Tags", "warning")
                .build()
            http.newCall(request).execute().close()
        } catch (_: Exception) {}
    }

    /** Get the crash report text for sharing. */
    fun getReport(context: Context): String {
        return try {
            val file = java.io.File(context.filesDir, "crash_report.txt")
            if (file.exists()) file.readText() else lastError
        } catch (_: Exception) {
            lastError
        }
    }
}
