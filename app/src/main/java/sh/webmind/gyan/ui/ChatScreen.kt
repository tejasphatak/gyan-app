package sh.webmind.gyan.ui

import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.launch
import sh.webmind.gyan.data.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(viewModel: ChatViewModel = viewModel()) {
    val listState = rememberLazyListState()
    val scope = rememberCoroutineScope()

    // Auto-scroll on new messages
    LaunchedEffect(viewModel.messages.size) {
        if (viewModel.messages.isNotEmpty()) {
            listState.animateScrollToItem(viewModel.messages.size - 1)
        }
    }

    Scaffold(
        topBar = { ChatTopBar(viewModel) },
        bottomBar = { ChatInput(viewModel) },
    ) { padding ->
        if (viewModel.messages.isEmpty()) {
            EmptyState(modifier = Modifier.padding(padding))
        } else {
            LazyColumn(
                state = listState,
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding),
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                items(viewModel.messages, key = { it.id }) { message ->
                    MessageItem(message)
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ChatTopBar(viewModel: ChatViewModel) {
    val status = viewModel.engineStatus.value

    TopAppBar(
        title = {
            Column {
                Text("Gyan", style = MaterialTheme.typography.titleLarge)
                if (status != null) {
                    Text(
                        text = if (status.ok) "${status.points} knowledge pairs"
                               else "Offline",
                        style = MaterialTheme.typography.bodySmall,
                        color = if (status.ok) MaterialTheme.colorScheme.secondary
                                else MaterialTheme.colorScheme.error,
                    )
                }
            }
        },
        actions = {
            if (viewModel.messages.isNotEmpty()) {
                TextButton(onClick = { viewModel.clearChat() }) {
                    Text("Clear")
                }
            }
        },
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = MaterialTheme.colorScheme.surface,
        ),
    )
}

@Composable
private fun EmptyState(modifier: Modifier = Modifier) {
    Box(
        modifier = modifier.fillMaxSize(),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                "Gyan",
                style = MaterialTheme.typography.titleLarge,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f),
            )
            Text(
                "Ask anything",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.2f),
            )
        }
    }
}

@Composable
private fun ChatInput(viewModel: ChatViewModel) {
    var text by remember { mutableStateOf("") }
    val focusManager = LocalFocusManager.current

    Surface(
        tonalElevation = 2.dp,
        shadowElevation = 8.dp,
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .navigationBarsPadding()
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.Bottom,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            OutlinedTextField(
                value = text,
                onValueChange = { text = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Message") },
                shape = RoundedCornerShape(24.dp),
                maxLines = 5,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(
                    onSend = {
                        if (text.isNotBlank()) {
                            viewModel.sendMessage(text.trim())
                            text = ""
                        }
                    }
                ),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = MaterialTheme.colorScheme.outline,
                    unfocusedBorderColor = MaterialTheme.colorScheme.outlineVariant,
                ),
            )

            // Send button
            FilledIconButton(
                onClick = {
                    if (text.isNotBlank()) {
                        viewModel.sendMessage(text.trim())
                        text = ""
                        focusManager.clearFocus()
                    }
                },
                enabled = text.isNotBlank() && !viewModel.isLoading.value,
                modifier = Modifier.size(48.dp),
                shape = CircleShape,
            ) {
                Text(
                    if (viewModel.isLoading.value) "..." else "\u2191",
                    style = MaterialTheme.typography.titleLarge,
                )
            }
        }
    }
}

@Composable
private fun MessageItem(message: ChatMessage) {
    when (message.role) {
        Role.USER -> UserMessage(message)
        Role.ASSISTANT -> AssistantMessage(message)
    }
}

@Composable
private fun UserMessage(message: ChatMessage) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.End,
    ) {
        Surface(
            shape = RoundedCornerShape(20.dp, 20.dp, 4.dp, 20.dp),
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.widthIn(max = 300.dp),
        ) {
            Text(
                text = message.content,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp),
                color = MaterialTheme.colorScheme.onPrimary,
                style = MaterialTheme.typography.bodyLarge,
            )
        }
    }
}

@Composable
private fun AssistantMessage(message: ChatMessage) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
    ) {
        // Tool calls (search indicators)
        message.toolCalls.forEach { toolCall ->
            ToolCallCard(toolCall)
            Spacer(modifier = Modifier.height(4.dp))
        }

        // Response content
        if (message.content.isNotEmpty()) {
            Surface(
                shape = RoundedCornerShape(4.dp, 20.dp, 20.dp, 20.dp),
                color = MaterialTheme.colorScheme.surfaceVariant,
                modifier = Modifier.widthIn(max = 340.dp),
            ) {
                Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp)) {
                    MarkdownText(
                        markdown = message.content,
                    )

                    // Metadata footer
                    message.metadata?.let { meta ->
                        Spacer(modifier = Modifier.height(8.dp))
                        MetadataFooter(meta)
                    }
                }
            }
        } else if (message.toolCalls.any { it.status == ToolStatus.RUNNING }) {
            // Loading indicator
            Row(
                modifier = Modifier.padding(start = 4.dp, top = 4.dp),
                horizontalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                CircularProgressIndicator(
                    modifier = Modifier.size(14.dp),
                    strokeWidth = 2.dp,
                    color = MaterialTheme.colorScheme.secondary,
                )
                Text(
                    "Thinking...",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.secondary,
                )
            }
        }
    }
}

@Composable
private fun ToolCallCard(toolCall: ToolCall) {
    val icon = when (toolCall.type) {
        ToolType.KB_SEARCH -> "\uD83D\uDD0D" // magnifying glass
        ToolType.WEB_SEARCH -> "\uD83C\uDF10" // globe
    }
    val label = when (toolCall.type) {
        ToolType.KB_SEARCH -> "Knowledge Search"
        ToolType.WEB_SEARCH -> "Web Search"
    }

    Surface(
        shape = RoundedCornerShape(12.dp),
        color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f),
        modifier = Modifier.widthIn(max = 280.dp),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(icon, style = MaterialTheme.typography.bodyMedium)

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    label,
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                Text(
                    toolCall.query,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }

            when (toolCall.status) {
                ToolStatus.RUNNING -> CircularProgressIndicator(
                    modifier = Modifier.size(14.dp),
                    strokeWidth = 2.dp,
                    color = MaterialTheme.colorScheme.secondary,
                )
                ToolStatus.DONE -> Text(
                    "\u2713",
                    color = Color(0xFF4CAF50),
                    style = MaterialTheme.typography.bodyMedium,
                )
                ToolStatus.FAILED -> Text(
                    "\u2717",
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }
    }
}

@Composable
private fun MetadataFooter(meta: MessageMetadata) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        MetaChip("${(meta.confidence * 100).toInt()}%")
        if (meta.hops > 0) MetaChip("${meta.hops} hops")
        MetaChip("${meta.timeMs}ms")
        if (meta.learned) MetaChip("Learned")
    }
}

@Composable
private fun MetaChip(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
    )
}
