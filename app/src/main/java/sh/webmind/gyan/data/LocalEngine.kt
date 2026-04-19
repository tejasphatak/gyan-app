package sh.webmind.gyan.data

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Local on-device engine.
 * - Embeddings: memory-mapped numpy file (zero heap)
 * - Metadata: SQLite database (zero heap, queried by index)
 * - Search: brute-force cosine over mmap
 */
class LocalEngine(private val context: Context) {

    private var embBuffer: MappedByteBuffer? = null
    private var db: SQLiteDatabase? = null
    private var dims: Int = 384
    private var count: Int = 0
    private var dataOffset: Int = 0
    private var isFp16: Boolean = false
    private var loaded = false

    val isLoaded get() = loaded
    val pairCount get() = count

    /** Load model. Embeddings via mmap, metadata via SQLite. */
    suspend fun load(
        embeddingsFile: File,
        metadataFile: File,
        onProgress: ((String) -> Unit)? = null,
    ) = withContext(Dispatchers.IO) {
        if (loaded) return@withContext

        // 1. Memory-map embeddings
        val raf = RandomAccessFile(embeddingsFile, "r")
        val channel = raf.channel
        val mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, raf.length())
        mapped.order(ByteOrder.LITTLE_ENDIAN)

        // Parse numpy header
        mapped.position(8)
        val headerLen = mapped.short.toInt() and 0xFFFF
        val headerBytes = ByteArray(headerLen)
        mapped.get(headerBytes)
        val header = String(headerBytes).trim()

        val shapeMatch = Regex("""'shape':\s*\((\d+),\s*(\d+)\)""").find(header)
            ?: throw IllegalArgumentException("Cannot parse npy header")
        count = shapeMatch.groupValues[1].toInt()
        dims = shapeMatch.groupValues[2].toInt()
        isFp16 = header.contains("float16") || header.contains("<f2")
        dataOffset = 8 + 2 + headerLen

        embBuffer = mapped
        channel.close()
        raf.close()

        onProgress?.invoke("Embeddings loaded: $count × $dims")

        // 2. Convert metadata JSON → SQLite (once, then reuse)
        val dbFile = File(context.filesDir, "model/metadata.db")
        if (!dbFile.exists()) {
            onProgress?.invoke("Converting knowledge base (first time)...")
            convertJsonToSqlite(metadataFile, dbFile, onProgress)
        }
        db = SQLiteDatabase.openDatabase(dbFile.path, null, SQLiteDatabase.OPEN_READONLY)

        loaded = true
    }

    /** Convert metadata.json to SQLite — streaming, no full load. */
    private fun convertJsonToSqlite(jsonFile: File, dbFile: File, onProgress: ((String) -> Unit)? = null) {
        dbFile.parentFile?.mkdirs()
        val database = SQLiteDatabase.openOrCreateDatabase(dbFile, null)
        database.execSQL("CREATE TABLE IF NOT EXISTS meta (idx INTEGER PRIMARY KEY, question TEXT, answer TEXT, source TEXT)")
        database.execSQL("PRAGMA journal_mode=OFF")
        database.execSQL("PRAGMA synchronous=OFF")

        val insertStmt = database.compileStatement(
            "INSERT INTO meta (idx, question, answer, source) VALUES (?, ?, ?, ?)"
        )

        database.beginTransaction()
        try {
            var idx = 0
            val reader = jsonFile.bufferedReader(bufferSize = 65536)
            val sb = StringBuilder()
            var depth = 0
            var inString = false
            var escape = false
            val buf = CharArray(65536)

            reader.use { br ->
                while (true) {
                    val read = br.read(buf)
                    if (read == -1) break

                    for (i in 0 until read) {
                        val c = buf[i]

                        if (escape) { sb.append(c); escape = false; continue }
                        if (c == '\\' && inString) { sb.append(c); escape = true; continue }
                        if (c == '"') { inString = !inString; sb.append(c); continue }
                        if (inString) { sb.append(c); continue }

                        when (c) {
                            '{' -> { depth++; sb.append(c) }
                            '}' -> {
                                depth--
                                sb.append(c)
                                if (depth == 0) {
                                    try {
                                        val obj = org.json.JSONObject(sb.toString())
                                        insertStmt.bindLong(1, idx.toLong())
                                        insertStmt.bindString(2, obj.optString("q", obj.optString("question", "")))
                                        insertStmt.bindString(3, obj.optString("a", obj.optString("answer", "")))
                                        insertStmt.bindString(4, obj.optString("s", obj.optString("source", "")))
                                        insertStmt.executeInsert()
                                        insertStmt.clearBindings()
                                        idx++

                                        if (idx % 50000 == 0) {
                                            database.setTransactionSuccessful()
                                            database.endTransaction()
                                            database.beginTransaction()
                                        }
                                    } catch (_: Exception) {}
                                    sb.clear()
                                }
                            }
                            '[', ']', ',' -> { if (depth > 0) sb.append(c) }
                            else -> { if (depth > 0) sb.append(c) }
                        }
                    }
                }
            }
            database.setTransactionSuccessful()
        } finally {
            database.endTransaction()
            database.close()
        }
    }

    /** Look up answer by embedding index. */
    private fun getEntry(index: Int): Triple<String, String, String> {
        val cursor = db?.rawQuery(
            "SELECT question, answer, source FROM meta WHERE idx = ?",
            arrayOf(index.toString())
        ) ?: return Triple("", "", "")

        return if (cursor.moveToFirst()) {
            val result = Triple(
                cursor.getString(0) ?: "",
                cursor.getString(1) ?: "",
                cursor.getString(2) ?: "",
            )
            cursor.close()
            result
        } else {
            cursor.close()
            Triple("", "", "")
        }
    }

    /** Search by cosine similarity over memory-mapped embeddings. */
    fun search(queryEmbedding: FloatArray, topK: Int = 5): List<SearchResult> {
        val buf = embBuffer ?: return emptyList()

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

        // Top-K
        val indices = scores.indices.sortedByDescending { scores[it] }.take(topK)

        return indices.map { idx ->
            val (question, answer, source) = getEntry(idx)
            SearchResult(
                index = idx,
                score = scores[idx],
                question = question,
                answer = answer,
                source = source,
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
            exp == 31 -> if (mantissa == 0) {
                if (sign == 1) Float.NEGATIVE_INFINITY else Float.POSITIVE_INFINITY
            } else Float.NaN
            else -> {
                val f = (mantissa.toFloat() / 1024f + 1f) * Math.pow(2.0, (exp - 15).toDouble()).toFloat()
                if (sign == 1) -f else f
            }
        }
    }
}
