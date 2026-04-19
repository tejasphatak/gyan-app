package sh.webmind.gyan.ui

import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import sh.webmind.gyan.data.*

class ChatViewModel : ViewModel() {

    private val client = GyanClient()

    val messages = mutableStateListOf<ChatMessage>()
    val isLoading = mutableStateOf(false)
    val engineStatus = mutableStateOf<HealthResult?>(null)

    init {
        checkHealth()
    }

    fun checkHealth() {
        viewModelScope.launch {
            engineStatus.value = client.health()
        }
    }

    fun sendMessage(text: String) {
        if (text.isBlank() || isLoading.value) return

        // Add user message
        messages.add(ChatMessage(role = Role.USER, content = text))

        // Build context from last few messages
        val context = messages
            .filter { it.role == Role.ASSISTANT }
            .takeLast(3)
            .joinToString(" ") { it.content.take(200) }

        isLoading.value = true

        // Add placeholder assistant message with search tool call
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
            val result = withContext(Dispatchers.IO) {
                client.query(text, context)
            }

            // Update on main thread — safe for mutableStateListOf
            val toolCalls = mutableListOf<ToolCall>()

            if (result.error != null) {
                toolCalls.add(
                    searchCall.copy(
                        status = ToolStatus.FAILED,
                        result = result.error,
                    )
                )
            } else {
                toolCalls.add(
                    searchCall.copy(
                        status = ToolStatus.DONE,
                        result = if (result.answer.isNotEmpty()) "Found" else "No match",
                    )
                )

                if (result.source == "web-search") {
                    toolCalls.add(
                        ToolCall(
                            type = ToolType.WEB_SEARCH,
                            query = text,
                            status = ToolStatus.DONE,
                            result = if (result.learned) "Learned new answer" else "Found",
                        )
                    )
                }
            }

            val content = when {
                result.error != null -> "Connection error: ${result.error}"
                result.answer.isEmpty() -> "I don't have an answer for that yet."
                else -> result.answer
            }

            // Find and update the placeholder — by ID, safe even if list changed
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
                        learned = result.learned,
                    ) else null,
                )
            }

            isLoading.value = false
        }
    }

    fun clearChat() {
        messages.clear()
    }
}
