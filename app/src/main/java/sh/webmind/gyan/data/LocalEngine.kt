package sh.webmind.gyan.data

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/**
 * Local on-device engine — memory-maps embeddings for zero-copy search.
 * Handles 1.24M × 384 embeddings without loading into heap.
 */
class LocalEngine(private val context: Context) {

    private var embBuffer: MappedByteBuffer? = null
    private var metadata: List<KBEntry>? = null
    private var dims: Int = 384
    private var count: Int = 0
    private var dataOffset: Int = 0
    private var isFp16: Boolean = false
    private var loaded = false

    data class KBEntry(val question: String, val answer: String, val source: String)

    /** Load model using memory-mapped file (no heap allocation for embeddings). */
    suspend fun load(embeddingsFile: File, metadataFile: File) = withContext(Dispatchers.IO) {
        if (loaded) return@withContext

        // Memory-map the embeddings file
        val raf = RandomAccessFile(embeddingsFile, "r")
        val channel = raf.channel
        val mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, raf.length())
        mapped.order(ByteOrder.LITTLE_ENDIAN)

        // Parse numpy header
        mapped.position(8) // skip magic + version
        val headerLen = mapped.short.toInt() and 0xFFFF
        val headerBytes = ByteArray(headerLen)
        mapped.get(headerBytes)
        val header = String(headerBytes).trim()

        val shapeMatch = Regex("""'shape':\s*\((\d+),\s*(\d+)\)""").find(header)
            ?: throw IllegalArgumentException("Cannot parse shape from: $header")
        count = shapeMatch.groupValues[1].toInt()
        dims = shapeMatch.groupValues[2].toInt()
        isFp16 = header.contains("float16") || header.contains("<f2")
        dataOffset = 8 + 2 + headerLen

        embBuffer = mapped
        channel.close()
        raf.close()

        // Stream-parse metadata JSON (don't load entire 543MB string)
        loadMetadataStreaming(metadataFile)

        loaded = true
    }

    /** Parse metadata line-by-line to avoid OOM. */
    private fun loadMetadataStreaming(file: File) {
        val entries = ArrayList<KBEntry>(count)
        val reader = file.bufferedReader(bufferSize = 65536)

        // The JSON is an array of objects. Parse manually to avoid loading entire string.
        val sb = StringBuilder()
        var depth = 0
        var inString = false
        var escape = false
        var objectCount = 0

        reader.use { br ->
            val buf = CharArray(65536)
            while (true) {
                val read = br.read(buf)
                if (read == -1) break

                for (i in 0 until read) {
                    val c = buf[i]

                    if (escape) {
                        sb.append(c)
                        escape = false
                        continue
                    }

                    if (c == '\\' && inString) {
                        sb.append(c)
                        escape = true
                        continue
                    }

                    if (c == '"') {
                        inString = !inString
                        sb.append(c)
                        continue
                    }

                    if (inString) {
                        sb.append(c)
                        continue
                    }

                    when (c) {
                        '{' -> {
                            depth++
                            sb.append(c)
                        }
                        '}' -> {
                            depth--
                            sb.append(c)
                            if (depth == 0) {
                                // Complete object — parse it
                                val objStr = sb.toString()
                                sb.clear()
                                try {
                                    val obj = org.json.JSONObject(objStr)
                                    entries.add(KBEntry(
                                        question = obj.optString("q", obj.optString("question", "")),
                                        answer = obj.optString("a", obj.optString("answer", "")),
                                        source = obj.optString("s", obj.optString("source", "")),
                                    ))
                                    objectCount++
                                } catch (_: Exception) {}
                            }
                        }
                        '[', ']', ',' -> {
                            if (depth == 0) { /* skip array delimiters */ }
                            else sb.append(c)
                        }
                        else -> {
                            if (depth > 0) sb.append(c)
                        }
                    }
                }
            }
        }
        metadata = entries
    }

    val isLoaded get() = loaded
    val pairCount get() = count

    /** Search by cosine similarity using memory-mapped embeddings. */
    fun search(queryEmbedding: FloatArray, topK: Int = 5): List<SearchResult> {
        val buf = embBuffer ?: return emptyList()
        val meta = metadata ?: return emptyList()

        val scores = FloatArray(count)
        val bytesPerElement = if (isFp16) 2 else 4
        val rowBytes = dims * bytesPerElement

        for (i in 0 until count) {
            var dot = 0f
            val rowStart = dataOffset + i.toLong() * rowBytes

            if (isFp16) {
                for (j in 0 until dims) {
                    val pos = (rowStart + j * 2).toInt()
                    val halfBits = (buf.get(pos + 1).toInt() and 0xFF shl 8) or
                                   (buf.get(pos).toInt() and 0xFF)
                    dot += queryEmbedding[j] * halfToFloat(halfBits)
                }
            } else {
                for (j in 0 until dims) {
                    val pos = (rowStart + j * 4).toInt()
                    dot += queryEmbedding[j] * buf.getFloat(pos)
                }
            }
            scores[i] = dot
        }

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

    private fun halfToFloat(hbits: Int): Float {
        val sign = (hbits ushr 15) and 1
        val exp = (hbits ushr 10) and 0x1F
        val mantissa = hbits and 0x3FF

        return when {
            exp == 0 -> {
                val f = mantissa.toFloat() / 1024f * (1f / 16384f)
                if (sign == 1) -f else f
            }
            exp == 31 -> {
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
