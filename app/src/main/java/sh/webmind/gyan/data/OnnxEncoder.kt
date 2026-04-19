package sh.webmind.gyan.data

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * ONNX-based sentence encoder — runs MiniLM on-device.
 * Uses GPU/NNAPI when available, falls back to CPU.
 */
class OnnxEncoder(private val context: Context) {

    private var session: OrtSession? = null
    private var env: OrtEnvironment? = null
    private val tokenizer = SimpleTokenizer()
    private var loaded = false

    /** Load ONNX model. Downloads from assets or bundled. */
    suspend fun load() = withContext(Dispatchers.IO) {
        if (loaded) return@withContext

        env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            // Try NNAPI (dispatches to GPU/DSP/NPU on Android)
            try {
                addNnapi()
            } catch (_: Exception) {
                // NNAPI not available, CPU fallback
            }
        }

        // Load model from assets
        val modelBytes = context.assets.open("model.onnx").use { it.readBytes() }
        session = env!!.createSession(modelBytes, opts)
        loaded = true
    }

    val isLoaded get() = loaded

    /** Encode text to 384-dim normalized embedding. */
    suspend fun encode(text: String): FloatArray = withContext(Dispatchers.IO) {
        val sess = session ?: throw IllegalStateException("Encoder not loaded")
        val ortEnv = env!!

        // Tokenize
        val tokens = tokenizer.tokenize(text)
        val inputIds = LongArray(tokens.size) { tokens[it].toLong() }
        val attentionMask = LongArray(tokens.size) { 1L }
        val tokenTypeIds = LongArray(tokens.size) { 0L }

        val shape = longArrayOf(1, tokens.size.toLong())

        val inputIdsTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(inputIds), shape)
        val attMaskTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(attentionMask), shape)
        val ttIdsTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(tokenTypeIds), shape)

        val inputs = mapOf(
            "input_ids" to inputIdsTensor,
            "attention_mask" to attMaskTensor,
            "token_type_ids" to ttIdsTensor,
        )

        val output = sess.run(inputs)

        // Get token embeddings: (1, seq_len, 384)
        @Suppress("UNCHECKED_CAST")
        val tokenEmbeddings = (output[0].value as Array<Array<FloatArray>>)[0]

        // Mean pooling
        val embedding = FloatArray(384)
        for (dim in 0 until 384) {
            var sum = 0f
            for (tok in tokenEmbeddings.indices) {
                sum += tokenEmbeddings[tok][dim]
            }
            embedding[dim] = sum / tokenEmbeddings.size
        }

        // L2 normalize
        var norm = 0f
        for (v in embedding) norm += v * v
        norm = sqrt(norm)
        if (norm > 1e-9f) {
            for (i in embedding.indices) embedding[i] /= norm
        }

        // Cleanup
        inputIdsTensor.close()
        attMaskTensor.close()
        ttIdsTensor.close()
        output.close()

        embedding
    }
}

/**
 * Minimal WordPiece tokenizer for MiniLM.
 * In production, use HuggingFace's tokenizer or a proper implementation.
 * This is a fallback that splits on whitespace + basic subwords.
 */
class SimpleTokenizer {
    // CLS=101, SEP=102, UNK=100
    fun tokenize(text: String, maxLen: Int = 128): IntArray {
        val words = text.lowercase().split(Regex("\\s+")).filter { it.isNotEmpty() }
        val tokens = mutableListOf(101) // [CLS]

        for (word in words) {
            if (tokens.size >= maxLen - 1) break
            // Simple: hash each word to a token ID in vocab range
            // This is a PLACEHOLDER — real deployment needs proper WordPiece vocab
            val hash = (word.hashCode() and 0x7FFFFFFF) % 30000 + 1000
            tokens.add(hash)
        }

        tokens.add(102) // [SEP]
        return tokens.toIntArray()
    }
}
