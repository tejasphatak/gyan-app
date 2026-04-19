package sh.webmind.gyan.data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * HTTP client for the Gyan engine API.
 * Talks to serve.py running on the backend.
 */
class GyanClient(
    private val baseUrl: String = DEFAULT_URL,
) {
    companion object {
        // Emulator uses 10.0.2.2 to reach host. Physical devices need actual IP.
        const val DEFAULT_URL = "http://10.0.2.2:3003"
    }

    private val http = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    private val jsonType = "application/json".toMediaType()

    /** Query the engine. Returns answer + metadata. */
    suspend fun query(question: String, context: String = ""): QueryResult =
        withContext(Dispatchers.IO) {
            try {
                val body = JSONObject().apply {
                    put("question", question)
                    if (context.isNotEmpty()) put("context", context)
                }
                val request = Request.Builder()
                    .url("$baseUrl/query")
                    .post(body.toString().toRequestBody(jsonType))
                    .build()

                val response = http.newCall(request).execute()
                response.use { resp ->
                    val text = resp.body?.string() ?: "{}"
                    val obj = try {
                        JSONObject(text)
                    } catch (e: Exception) {
                        return@withContext QueryResult(
                            answer = text.take(500),
                            confidence = 0f, hops = 0, timeMs = 0,
                            source = "raw", error = "Non-JSON response",
                        )
                    }

                    QueryResult(
                        answer = obj.optString("answer", ""),
                        confidence = obj.optDouble("confidence", 0.0).toFloat(),
                        hops = obj.optInt("hops", 0),
                        timeMs = obj.optInt("timeMs", 0),
                        source = obj.optString("source", ""),
                        learned = obj.optBoolean("learned", false),
                    )
                }
            } catch (e: Exception) {
                QueryResult(
                    answer = "", confidence = 0f, hops = 0, timeMs = 0,
                    source = "", error = e.message ?: "Connection failed",
                )
            }
        }

    /** Teach the engine a new fact. */
    suspend fun learn(question: String, answer: String, source: String = "user") =
        withContext(Dispatchers.IO) {
            try {
                val body = JSONObject().apply {
                    put("question", question)
                    put("answer", answer)
                    put("source", source)
                }
                val request = Request.Builder()
                    .url("$baseUrl/learn")
                    .post(body.toString().toRequestBody(jsonType))
                    .build()
                http.newCall(request).execute().close()
            } catch (_: Exception) {
                // Fire-and-forget
            }
        }

    /** Check engine health. */
    suspend fun health(): HealthResult = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder().url("$baseUrl/health").build()
            http.newCall(request).execute().use { resp ->
                val obj = JSONObject(resp.body?.string() ?: "{}")
                HealthResult(
                    ok = obj.optString("status") == "ok",
                    points = obj.optInt("points", 0),
                )
            }
        } catch (e: Exception) {
            HealthResult(ok = false, points = 0, error = e.message)
        }
    }
}

data class QueryResult(
    val answer: String,
    val confidence: Float,
    val hops: Int,
    val timeMs: Int,
    val source: String,
    val learned: Boolean = false,
    val error: String? = null,
)

data class HealthResult(
    val ok: Boolean,
    val points: Int,
    val error: String? = null,
)
