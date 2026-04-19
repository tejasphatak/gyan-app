package sh.webmind.gyan.data

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Local on-device engine — loads bundled embeddings + metadata from assets.
 * No server needed. Runs entirely on the phone.
 */
class LocalEngine(private val context: Context) {

    private var embeddings: FloatArray? = null
    private var metadata: List<KBEntry>? = null
    private var dims: Int = 384
    private var count: Int = 0
    private var loaded = false

    data class KBEntry(val question: String, val answer: String, val source: String)

    /** Load model from downloaded files. Call once on startup. */
    suspend fun load(embeddingsFile: java.io.File, metadataFile: java.io.File) = withContext(Dispatchers.IO) {
        if (loaded) return@withContext

        // Load embeddings from numpy .npy file
        val embBytes = embeddingsFile.readBytes()
        val emb = parseNpy(embBytes)
        embeddings = emb.first
        count = emb.second
        dims = emb.third

        // Load metadata
        val metaText = metadataFile.bufferedReader().use { it.readText() }
        val arr = JSONArray(metaText)
        val entries = mutableListOf<KBEntry>()
        for (i in 0 until arr.length()) {
            val obj = arr.getJSONObject(i)
            entries.add(KBEntry(
                question = obj.optString("q", obj.optString("question", "")),
                answer = obj.optString("a", obj.optString("answer", "")),
                source = obj.optString("s", obj.optString("source", "")),
            ))
        }
        metadata = entries
        loaded = true
    }

    val isLoaded get() = loaded
    val pairCount get() = count

    /** Search by cosine similarity. Returns top-K results. */
    fun search(queryEmbedding: FloatArray, topK: Int = 5): List<SearchResult> {
        val emb = embeddings ?: return emptyList()
        val meta = metadata ?: return emptyList()

        // Compute cosine similarity with all embeddings
        // queryEmbedding is already normalized, embeddings stored as fp16→fp32
        val scores = FloatArray(count)
        for (i in 0 until count) {
            var dot = 0f
            val offset = i * dims
            for (j in 0 until dims) {
                dot += queryEmbedding[j] * emb[offset + j]
            }
            scores[i] = dot
        }

        // Find top-K
        val indices = scores.indices.sortedByDescending { scores[it] }.take(topK)

        return indices.map { idx ->
            SearchResult(
                index = idx,
                score = scores[idx],
                question = meta.getOrNull(idx)?.question ?: "",
                answer = meta.getOrNull(idx)?.answer ?: "",
                source = meta.getOrNull(idx)?.source ?: "",
            )
        }
    }

    data class SearchResult(
        val index: Int,
        val score: Float,
        val question: String,
        val answer: String,
        val source: String,
    )

    /** Parse numpy .npy file (fp16 or fp32). Returns (flat array as fp32, rows, cols). */
    private fun parseNpy(bytes: ByteArray): Triple<FloatArray, Int, Int> {
        // Numpy .npy format:
        // 6 bytes magic: \x93NUMPY
        // 1 byte major version
        // 1 byte minor version
        // 2 bytes header length (little-endian)
        // header dict string (contains shape, dtype, order)
        // data

        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        // Skip magic (6) + version (2)
        buf.position(8)
        val headerLen = buf.short.toInt() and 0xFFFF
        val headerBytes = ByteArray(headerLen)
        buf.get(headerBytes)
        val header = String(headerBytes).trim()

        // Parse shape from header: 'shape': (rows, cols)
        val shapeMatch = Regex("""'shape':\s*\((\d+),\s*(\d+)\)""").find(header)
            ?: throw IllegalArgumentException("Cannot parse shape from: $header")
        val rows = shapeMatch.groupValues[1].toInt()
        val cols = shapeMatch.groupValues[2].toInt()

        // Parse dtype
        val isFp16 = header.contains("float16") || header.contains("<f2")
        val isFp32 = header.contains("float32") || header.contains("<f4")

        val data = FloatArray(rows * cols)
        val dataStart = 8 + 2 + headerLen

        if (isFp16) {
            // Read fp16 and convert to fp32
            for (i in 0 until rows * cols) {
                val halfBits = ((bytes[dataStart + i * 2 + 1].toInt() and 0xFF) shl 8) or
                               (bytes[dataStart + i * 2].toInt() and 0xFF)
                data[i] = halfToFloat(halfBits)
            }
        } else if (isFp32) {
            val dataBuf = ByteBuffer.wrap(bytes, dataStart, rows * cols * 4).order(ByteOrder.LITTLE_ENDIAN)
            dataBuf.asFloatBuffer().get(data)
        } else {
            throw IllegalArgumentException("Unsupported dtype in: $header")
        }

        // Normalize each row
        for (i in 0 until rows) {
            var norm = 0f
            val offset = i * cols
            for (j in 0 until cols) {
                norm += data[offset + j] * data[offset + j]
            }
            norm = sqrt(norm)
            if (norm > 1e-9f) {
                for (j in 0 until cols) {
                    data[offset + j] /= norm
                }
            }
        }

        return Triple(data, rows, cols)
    }

    /** Convert IEEE 754 half-precision to single-precision. */
    private fun halfToFloat(hbits: Int): Float {
        val sign = (hbits ushr 15) and 1
        val exp = (hbits ushr 10) and 0x1F
        val mantissa = hbits and 0x3FF

        return when {
            exp == 0 -> {
                // Subnormal or zero
                val f = mantissa.toFloat() / 1024f * (1f / 16384f)
                if (sign == 1) -f else f
            }
            exp == 31 -> {
                // Inf or NaN
                if (mantissa == 0) {
                    if (sign == 1) Float.NEGATIVE_INFINITY else Float.POSITIVE_INFINITY
                } else Float.NaN
            }
            else -> {
                val f = (mantissa.toFloat() / 1024f + 1f) * Math.pow(2.0, (exp - 15).toDouble()).toFloat()
                if (sign == 1) -f else f
            }
        }
    }
}
