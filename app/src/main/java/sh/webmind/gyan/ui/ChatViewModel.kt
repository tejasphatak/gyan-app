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

class ChatViewModel(app: Application) : AndroidViewModel(app) {

    private val client = GyanClient()
    private val localEngine = LocalEngine(app)
    private val downloader = ModelDownloader(app)

    val messages = mutableStateListOf<ChatMessage>()
    val isLoading = mutableStateOf(false)
    val engineStatus = mutableStateOf<HealthResult?>(null)
    val loadingProgress = mutableStateOf("Initializing...")
    val downloadProgress = mutableStateOf(0f) // 0.0 to 1.0
    val isDownloading = mutableStateOf(false)
    val isReady = mutableStateOf(false)

    init {
        viewModelScope.launch {
            initEngine()
        }
    }

    private suspend fun initEngine() {
        if (downloader.isModelReady()) {
            // Model already downloaded — load it
            loadingProgress.value = "Loading knowledge base..."
            try {
                localEngine.load(downloader.embeddingsFile, downloader.metadataFile)
                loadingProgress.value = "${localEngine.pairCount} knowledge pairs ready"
                engineStatus.value = HealthResult(ok = true, points = localEngine.pairCount)
                isReady.value = true
            } catch (e: Exception) {
                loadingProgress.value = "Load failed: ${e.message}"
                fallbackToServer()
            }
        } else {
            // Need to download model
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

            // Load the downloaded model
            loadingProgress.value = "Loading knowledge base..."
            localEngine.load(downloader.embeddingsFile, downloader.metadataFile)
            loadingProgress.value = "${localEngine.pairCount} knowledge pairs ready"
            engineStatus.value = HealthResult(ok = true, points = localEngine.pairCount)
            isReady.value = true
        } catch (e: Exception) {
            isDownloading.value = false
            loadingProgress.value = "Download failed: ${e.message}"
            fallbackToServer()
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
            loadingProgress.value = "Offline — download model to get started"
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
                result.answer.isEmpty() -> "I don't have an answer for that yet."
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
            // Simple text-hash based embedding for now (until ONNX model is bundled)
            // Use word-level cosine: encode question as bag-of-words hash → search
            val queryEmb = simpleEncode(question)
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
            QueryResult(
                answer = "", confidence = 0f, hops = 0,
                timeMs = (System.currentTimeMillis() - start).toInt(),
                source = "", error = e.message,
            )
        }
    }

    /** Simple word-hash encoding — placeholder until ONNX model is bundled. */
    private fun simpleEncode(text: String): FloatArray {
        val embedding = FloatArray(384)
        val words = text.lowercase().split(Regex("\\W+")).filter { it.length > 1 }
        for (word in words) {
            val hash = word.hashCode()
            for (i in 0 until 384) {
                embedding[i] += ((hash * (i + 1)) % 1000).toFloat() / 1000f
            }
        }
        // Normalize
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
