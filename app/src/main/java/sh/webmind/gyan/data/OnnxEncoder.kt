package sh.webmind.gyan.data

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * On-device ONNX sentence encoder — runs MiniLM locally.
 * Uses bundled model.onnx + vocab.txt from assets.
 * No internet needed for encoding.
 */
class OnnxEncoder(private val context: Context) {

    private var session: OrtSession? = null
    private var env: OrtEnvironment? = null
    private var vocab: Map<String, Int> = emptyMap()
    private var loaded = false

    val isLoaded get() = loaded

    /** Load ONNX model and tokenizer from assets. */
    suspend fun load() = withContext(Dispatchers.IO) {
        if (loaded) return@withContext

        env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            // Skip NNAPI — many phones have buggy drivers that reject transformer ops
            // CPU is fast enough for single-query encoding (~50ms)
            setIntraOpNumThreads(4)
        }

        // Read model from assets
        val modelBytes = context.assets.open("model.onnx").use { it.readBytes() }
        session = env!!.createSession(modelBytes, opts)

        // Load WordPiece vocabulary
        vocab = loadVocab()
        loaded = true
    }

    private fun loadVocab(): Map<String, Int> {
        val vocabMap = mutableMapOf<String, Int>()
        context.assets.open("vocab.txt").bufferedReader().useLines { lines ->
            lines.forEachIndexed { index, line ->
                vocabMap[line.trim()] = index
            }
        }
        return vocabMap
    }

    /** Encode text to 384-dim normalized embedding. */
    suspend fun encode(text: String): FloatArray = withContext(Dispatchers.IO) {
        val sess = session ?: throw IllegalStateException("Encoder not loaded")
        val ortEnv = env!!

        // WordPiece tokenize
        val tokens = wordPieceTokenize(text)
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
        val outputTensor = output[0]

        // Output shape: [1, seq_len, 384]
        // ONNX Runtime Android may return different Java types
        val seqLen = tokens.size
        val embedding = FloatArray(384)

        val value = outputTensor.value
        when (value) {
            is Array<*> -> {
                // Array<Array<FloatArray>> — [batch][seq][dim]
                @Suppress("UNCHECKED_CAST")
                val batch = value as Array<Array<FloatArray>>
                val tokenEmbs = batch[0]
                for (dim in 0 until 384) {
                    var sum = 0f
                    for (tok in tokenEmbs.indices) sum += tokenEmbs[tok][dim]
                    embedding[dim] = sum / tokenEmbs.size
                }
            }
            is FloatArray -> {
                // Flat float array — [1 * seq_len * 384]
                for (dim in 0 until 384) {
                    var sum = 0f
                    for (tok in 0 until seqLen) sum += value[tok * 384 + dim]
                    embedding[dim] = sum / seqLen
                }
            }
            else -> {
                // Try getting as OnnxTensor float buffer
                val tensor = outputTensor as OnnxTensor
                val floatBuf = tensor.floatBuffer
                for (dim in 0 until 384) {
                    var sum = 0f
                    for (tok in 0 until seqLen) sum += floatBuf.get(tok * 384 + dim)
                    embedding[dim] = sum / seqLen
                }
            }
        }

        // L2 normalize
        var norm = 0f
        for (v in embedding) norm += v * v
        norm = sqrt(norm)
        if (norm > 1e-9f) {
            for (i in embedding.indices) embedding[i] = embedding[i] / norm
        }

        inputIdsTensor.close()
        attMaskTensor.close()
        ttIdsTensor.close()
        output.close()

        embedding
    }

    /** Basic WordPiece tokenization compatible with MiniLM/BERT. */
    private fun wordPieceTokenize(text: String, maxLen: Int = 128): IntArray {
        val CLS = vocab["[CLS]"] ?: 101
        val SEP = vocab["[SEP]"] ?: 102
        val UNK = vocab["[UNK]"] ?: 100

        val tokens = mutableListOf(CLS)

        // Basic pre-tokenization: lowercase, split on whitespace and punctuation
        val words = text.lowercase()
            .replace(Regex("[^\\w\\s]"), " \\$0 ")
            .split(Regex("\\s+"))
            .filter { it.isNotEmpty() }

        for (word in words) {
            if (tokens.size >= maxLen - 1) break

            // Try to find whole word in vocab
            if (word in vocab) {
                tokens.add(vocab[word]!!)
                continue
            }

            // WordPiece: greedily split into subwords
            var start = 0
            var foundSubwords = false
            while (start < word.length) {
                if (tokens.size >= maxLen - 1) break

                var end = word.length
                var matched = false
                while (end > start) {
                    val sub = if (start == 0) word.substring(start, end)
                             else "##" + word.substring(start, end)
                    if (sub in vocab) {
                        tokens.add(vocab[sub]!!)
                        start = end
                        matched = true
                        foundSubwords = true
                        break
                    }
                    end--
                }
                if (!matched) {
                    // Character not in vocab — use UNK for entire word
                    if (!foundSubwords) tokens.add(UNK)
                    break
                }
            }
        }

        tokens.add(SEP)
        return tokens.toIntArray()
    }
}
