package sh.webmind.gyan.ui

import android.app.Application
import android.util.Log
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import sh.webmind.gyan.data.*

class ChatViewModel(private val app: Application) : AndroidViewModel(app) {

    private val downloader = ModelDownloader(app)
    private val gpu = GpuTransformer(app)

    val messages = mutableStateListOf<ChatMessage>()
    val isLoading = mutableStateOf(false)
    val engineStatus = mutableStateOf<HealthResult?>(null)
    val loadingProgress = mutableStateOf("Initializing...")
    val downloadProgress = mutableStateOf(0f)
    val isDownloading = mutableStateOf(false)
    val isReady = mutableStateOf(false)
    val errorReport = mutableStateOf("")

    init {
        viewModelScope.launch {
            try {
                Log.i("Gyan", "Init starting...")
                initEngine()
            } catch (e: Exception) {
                Log.e("Gyan", "Init failed", e)
                CrashReporter.capture(app, e, "initEngine")
                loadingProgress.value = "Error: ${e.message}"
                errorReport.value = CrashReporter.getLastError()
            }
        }
    }

    private suspend fun initEngine() {
        if (downloader.isModelReady()) {
            loadModel()
        } else {
            downloader.clearModel()  // clean up any old/wrong model
            downloadModel()
        }
    }

    private suspend fun loadModel() = withContext(Dispatchers.IO) {
        loadingProgress.value = "Loading AI engine on GPU..."
        val ok = gpu.init(downloader.embeddingsFile, downloader.metadataFile)
        if (ok) {
            loadingProgress.value = "${gpu.pairCount} knowledge pairs on GPU"
            engineStatus.value = HealthResult(ok = true, points = gpu.pairCount)
            isReady.value = true
            Log.i("Gyan", "GPU engine ready: ${gpu.pairCount} pairs")
        } else {
            loadingProgress.value = "GPU init failed. Tap Retry."
            CrashReporter.capture(app, RuntimeException("GPU init returned false"), "loadModel")
            errorReport.value = CrashReporter.getLastError()
        }
    }

    private suspend fun downloadModel() {
        isDownloading.value = true
        loadingProgress.value = "Downloading AI model..."
        try {
            downloader.download { name, bytesRead, totalBytes ->
                if (totalBytes > 0) {
                    val pct = (bytesRead * 100 / totalBytes).toInt()
                    val mbRead = bytesRead / 1_000_000
                    val mbTotal = totalBytes / 1_000_000
                    loadingProgress.value = "$name: ${mbRead}MB / ${mbTotal}MB ($pct%)"
                    downloadProgress.value = bytesRead.toFloat() / totalBytes
                }
            }
            isDownloading.value = false
            loadModel()
        } catch (e: Exception) {
            isDownloading.value = false
            CrashReporter.capture(app, e, "downloadModel")
            errorReport.value = CrashReporter.getLastError()
            loadingProgress.value = "Download failed: ${e.message}\n\nTap Retry"
        }
    }

    fun retryDownload() {
        viewModelScope.launch {
            if (downloader.isModelReady()) loadModel()
            else downloadModel()
        }
    }

    fun sendMessage(text: String) {
        if (text.isBlank() || isLoading.value) return

        messages.add(ChatMessage(role = Role.USER, content = text))
        isLoading.value = true

        val assistantId = java.util.UUID.randomUUID().toString()
        val searchCall = ToolCall(type = ToolType.KB_SEARCH, query = text, status = ToolStatus.RUNNING)
        messages.add(ChatMessage(id = assistantId, role = Role.ASSISTANT, content = "", toolCalls = listOf(searchCall)))

        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) { queryGpu(text) }

            val toolCalls = listOf(searchCall.copy(
                status = if (result.error != null) ToolStatus.FAILED else ToolStatus.DONE,
                result = result.error ?: if (result.answer.isNotEmpty()) "Found" else "No match",
            ))

            val content = when {
                result.error != null -> "Error: ${result.error}"
                result.answer.isEmpty() -> "Searched ${gpu.pairCount} pairs, no match (score=${result.confidence})"
                else -> result.answer
            }

            val idx = messages.indexOfFirst { it.id == assistantId }
            if (idx >= 0) {
                messages[idx] = ChatMessage(
                    id = assistantId, role = Role.ASSISTANT, content = content,
                    toolCalls = toolCalls,
                    metadata = if (result.error == null && result.answer.isNotEmpty()) MessageMetadata(
                        confidence = result.confidence, timeMs = result.timeMs, source = result.source,
                    ) else null,
                )
            }
            isLoading.value = false
        }
    }

    private fun queryGpu(question: String): QueryResult {
        val start = System.currentTimeMillis()
        return try {
            Log.i("Gyan", "Query: '$question'")
            val results = gpu.query(question, topK = 5)
            val elapsed = (System.currentTimeMillis() - start).toInt()

            if (results.isNotEmpty() && results[0].second > 0.3f) {
                val (idx, score) = results[0]
                val answer = gpu.getAnswer(idx)
                Log.i("Gyan", "Match: score=$score, answer='${answer.take(50)}'")
                QueryResult(answer = answer, confidence = score, hops = 0, timeMs = elapsed, source = "gpu")
            } else {
                val topScore = results.firstOrNull()?.second ?: 0f
                Log.w("Gyan", "No match: topScore=$topScore")
                QueryResult(answer = "", confidence = topScore, hops = 0, timeMs = elapsed, source = "no_match")
            }
        } catch (e: Exception) {
            Log.e("Gyan", "Query failed", e)
            CrashReporter.capture(app, e, "queryGpu: $question")
            QueryResult(answer = "", confidence = 0f, hops = 0,
                timeMs = (System.currentTimeMillis() - start).toInt(), source = "", error = e.message)
        }
    }

    fun clearChat() { messages.clear() }
}
