package sh.webmind.gyan.ui

import android.app.Application
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import sh.webmind.gyan.data.*
import sh.webmind.gyan.data.CrashReporter

class ChatViewModel(private val app: Application) : AndroidViewModel(app) {

    private val client = GyanClient()
    private val localEngine = LocalEngine(app)
    private val downloader = ModelDownloader(app)
    private val onnxEncoder = OnnxEncoder(app)

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
                android.util.Log.i("Gyan", "Init starting...")
                initEngine()
                android.util.Log.i("Gyan", "Init complete. Engine=${localEngine.isLoaded}, ONNX=${onnxEncoder.isLoaded}")
            } catch (e: Exception) {
                android.util.Log.e("Gyan", "Init failed", e)
                CrashReporter.capture(app, e, "initEngine")
                loadingProgress.value = "Error: ${e.message}"
                errorReport.value = CrashReporter.getLastError()
            }
        }
    }

    private suspend fun initEngine() {
        if (downloader.isModelReady()) {
            loadingProgress.value = "Loading knowledge base..."
            try {
                localEngine.load(downloader.embeddingsFile, downloader.metadataFile)
                loadingProgress.value = "Loading encoder..."
                onnxEncoder.load()
                loadingProgress.value = "${localEngine.pairCount} knowledge pairs ready"
                engineStatus.value = HealthResult(ok = true, points = localEngine.pairCount)
                isReady.value = true
            } catch (e: Exception) {
                CrashReporter.capture(app, e, "loadModel")
                loadingProgress.value = "Load failed: ${e.message}"
                errorReport.value = CrashReporter.getLastError()
                fallbackToServer()
            }
        } else {
            downloadModel()
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
            loadingProgress.value = "Loading knowledge base..."
            localEngine.load(downloader.embeddingsFile, downloader.metadataFile)
            loadingProgress.value = "Loading encoder..."
            onnxEncoder.load()
            loadingProgress.value = "${localEngine.pairCount} knowledge pairs ready"
            engineStatus.value = HealthResult(ok = true, points = localEngine.pairCount)
            isReady.value = true
        } catch (e: Exception) {
            isDownloading.value = false
            CrashReporter.capture(app, e, "downloadModel")
            errorReport.value = CrashReporter.getLastError()
            loadingProgress.value = "Failed: ${e.message}\n\nTap Retry to try again"
            // Don't fallback to server — show the actual error
        }
    }

    private suspend fun fallbackToServer() {
        loadingProgress.value = "Trying server..."
        val health = client.health()
        engineStatus.value = health
        if (health.ok) {
            loadingProgress.value = "Connected to server (${health.points} pairs)"
            isReady.value = true
        } else {
            loadingProgress.value = "Tap Retry to download the model"
        }
    }

    fun retryDownload() {
        viewModelScope.launch { downloadModel() }
    }

    fun sendMessage(text: String) {
        if (text.isBlank() || isLoading.value) return

        messages.add(ChatMessage(role = Role.USER, content = text))
        isLoading.value = true

        val assistantId = java.util.UUID.randomUUID().toString()
        val searchCall = ToolCall(
            type = ToolType.KB_SEARCH,
            query = text,
            status = ToolStatus.RUNNING,
        )
        messages.add(
            ChatMessage(
                id = assistantId,
                role = Role.ASSISTANT,
                content = "",
                toolCalls = listOf(searchCall),
            )
        )

        viewModelScope.launch {
            val result = if (localEngine.isLoaded) {
                queryLocal(text)
            } else {
                withContext(Dispatchers.IO) { client.query(text) }
            }

            val toolCalls = mutableListOf<ToolCall>()
            if (result.error != null) {
                toolCalls.add(searchCall.copy(status = ToolStatus.FAILED, result = result.error))
            } else {
                toolCalls.add(searchCall.copy(
                    status = ToolStatus.DONE,
                    result = if (result.answer.isNotEmpty()) "Found" else "No match",
                ))
            }

            val content = when {
                result.error != null -> "Error: ${result.error}"
                result.answer.isEmpty() && result.source == "no_match" -> "I searched ${localEngine.pairCount} knowledge pairs but couldn't find a match."
                result.answer.isEmpty() -> "No answer found. (source=${result.source})"
                else -> result.answer
            }

            val idx = messages.indexOfFirst { it.id == assistantId }
            if (idx >= 0) {
                messages[idx] = ChatMessage(
                    id = assistantId,
                    role = Role.ASSISTANT,
                    content = content,
                    toolCalls = toolCalls,
                    metadata = if (result.error == null) MessageMetadata(
                        confidence = result.confidence,
                        hops = result.hops,
                        timeMs = result.timeMs,
                        source = result.source,
                    ) else null,
                )
            }
            isLoading.value = false
        }
    }

    private suspend fun queryLocal(question: String): QueryResult = withContext(Dispatchers.IO) {
        val start = System.currentTimeMillis()
        try {
            // Encode query via on-device ONNX
            val queryEmb = if (onnxEncoder.isLoaded) {
                try {
                    onnxEncoder.encode(question)
                } catch (e: Exception) {
                    CrashReporter.capture(app, e, "onnxEncode failed: $question")
                    return@withContext QueryResult(
                        answer = "", confidence = 0f, hops = 0,
                        timeMs = (System.currentTimeMillis() - start).toInt(),
                        source = "", error = "Encoder error: ${e.message}",
                    )
                }
            } else {
                return@withContext QueryResult(
                    answer = "", confidence = 0f, hops = 0,
                    timeMs = (System.currentTimeMillis() - start).toInt(),
                    source = "", error = "Encoder not loaded",
                )
            }
            val results = localEngine.search(queryEmb, topK = 5)
            if (results.isNotEmpty() && results[0].score > 0.3f) {
                val best = results[0]
                QueryResult(
                    answer = best.answer,
                    confidence = best.score,
                    hops = 0,
                    timeMs = (System.currentTimeMillis() - start).toInt(),
                    source = best.source,
                )
            } else {
                QueryResult(
                    answer = "", confidence = 0f, hops = 0,
                    timeMs = (System.currentTimeMillis() - start).toInt(),
                    source = "no_match",
                )
            }
        } catch (e: Exception) {
            CrashReporter.capture(app, e, "queryLocal: $question")
            QueryResult(
                answer = "", confidence = 0f, hops = 0,
                timeMs = (System.currentTimeMillis() - start).toInt(),
                source = "", error = e.message,
            )
        }
    }

    private fun simpleEncode(text: String): FloatArray {
        val embedding = FloatArray(384)
        val words = text.lowercase().split(Regex("\\W+")).filter { it.length > 1 }
        for (word in words) {
            val hash = word.hashCode()
            for (i in 0 until 384) {
                embedding[i] += ((hash * (i + 1)) % 1000).toFloat() / 1000f
            }
        }
        var norm = 0f
        for (v in embedding) norm += v * v
        norm = kotlin.math.sqrt(norm)
        if (norm > 1e-9f) for (i in embedding.indices) embedding[i] /= norm
        return embedding
    }

    fun clearChat() {
        messages.clear()
    }
}
