package sh.webmind.gyan.data

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit

/**
 * Downloads model files from GitHub release on first launch.
 * Shows progress via callback.
 */
class ModelDownloader(private val context: Context) {

    companion object {
        // Full 1.24M model — uploaded in chunks to GPU
        private const val MODEL_BASE =
            "https://huggingface.co/TejaDaBheja/gyan-model/resolve/main"
        private const val EMBEDDINGS_FILE = "embeddings.npy"
        private const val METADATA_FILE = "metadata.json"
    }

    private val http = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.MINUTES)
        .build()

    private val modelDir: File
        get() = File(context.filesDir, "model").also { it.mkdirs() }

    val embeddingsFile: File get() = File(modelDir, EMBEDDINGS_FILE)
    val metadataFile: File get() = File(modelDir, METADATA_FILE)

    /** Check if model is already downloaded (and is the right size). */
    fun isModelReady(): Boolean {
        if (!embeddingsFile.exists() || !metadataFile.exists()) return false
        val embSize = embeddingsFile.length()
        return embSize > 1_000_000 && metadataFile.length() > 1_000_000
    }

    /** Delete old model to force re-download. */
    fun clearModel() {
        embeddingsFile.delete()
        metadataFile.delete()
        // Also clear any SQLite DB from old LocalEngine
        java.io.File(embeddingsFile.parentFile, "metadata.db").delete()
    }

    /** Download model files with progress callback. */
    suspend fun download(
        onProgress: (fileName: String, bytesRead: Long, totalBytes: Long) -> Unit,
    ) = withContext(Dispatchers.IO) {
        if (isModelReady()) return@withContext

        downloadFile(
            url = "$MODEL_BASE/$EMBEDDINGS_FILE",
            dest = embeddingsFile,
            name = "Knowledge embeddings",
            onProgress = onProgress,
        )

        downloadFile(
            url = "$MODEL_BASE/$METADATA_FILE",
            dest = metadataFile,
            name = "Knowledge base",
            onProgress = onProgress,
        )
    }

    private fun downloadFile(
        url: String,
        dest: File,
        name: String,
        onProgress: (String, Long, Long) -> Unit,
    ) {
        // Skip if already downloaded
        if (dest.exists() && dest.length() > 1_000_000) {
            onProgress(name, dest.length(), dest.length())
            return
        }

        val request = Request.Builder().url(url).build()
        val response = http.newCall(request).execute()

        if (!response.isSuccessful) {
            throw Exception("Download failed: HTTP ${response.code}")
        }

        val body = response.body ?: throw Exception("Empty response")
        val totalBytes = body.contentLength()
        val tempFile = File(dest.parent, "${dest.name}.tmp")

        body.byteStream().use { input ->
            FileOutputStream(tempFile).use { output ->
                val buffer = ByteArray(8192)
                var bytesRead = 0L

                while (true) {
                    val read = input.read(buffer)
                    if (read == -1) break
                    output.write(buffer, 0, read)
                    bytesRead += read

                    if (bytesRead % (1024 * 1024) < 8192) { // Update every ~1MB
                        onProgress(name, bytesRead, totalBytes)
                    }
                }
            }
        }

        // Atomic move
        tempFile.renameTo(dest)
        onProgress(name, totalBytes, totalBytes)
    }

    /** Delete downloaded model (for cleanup). */
    fun deleteModel() {
        embeddingsFile.delete()
        metadataFile.delete()
    }
}
