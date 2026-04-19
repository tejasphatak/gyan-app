package sh.webmind.gyan.data

import java.util.UUID

/** A single message in the conversation. */
data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val role: Role,
    val content: String,
    val timestamp: Long = System.currentTimeMillis(),
    val toolCalls: List<ToolCall> = emptyList(),
    val metadata: MessageMetadata? = null,
)

enum class Role { USER, ASSISTANT }

/** A tool call (search) shown inline in the response. */
data class ToolCall(
    val id: String = UUID.randomUUID().toString(),
    val type: ToolType,
    val query: String,
    val status: ToolStatus = ToolStatus.RUNNING,
    val result: String? = null,
)

enum class ToolType { KB_SEARCH, WEB_SEARCH }

enum class ToolStatus { RUNNING, DONE, FAILED }

/** Response metadata from the engine. */
data class MessageMetadata(
    val confidence: Float = 0f,
    val hops: Int = 0,
    val timeMs: Int = 0,
    val source: String = "",
    val learned: Boolean = false,
)
