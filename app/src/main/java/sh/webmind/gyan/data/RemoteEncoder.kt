package sh.webmind.gyan.data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import kotlin.math.sqrt

/**
 * Encode text using HuggingFace's free Inference API.
 * Sends the query text, gets back 384-dim embedding.
 * Only the short query goes over the network — all search is local.
 */
class RemoteEncoder {

    companion object {
        private const val API_URL =
            "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    }

    private val http = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()

    private val jsonType = "application/json".toMediaType()

    /** Encode text to 384-dim normalized embedding via HuggingFace API. */
    suspend fun encode(text: String): FloatArray = withContext(Dispatchers.IO) {
        val body = JSONObject().apply {
            put("inputs", text)
            put("options", JSONObject().apply {
                put("wait_for_model", true)
            })
        }

        val request = Request.Builder()
            .url(API_URL)
            .post(body.toString().toRequestBody(jsonType))
            .build()

        val response = http.newCall(request).execute()
        val responseText = response.body?.string() ?: "[]"

        // Response is [[float, float, ...]] — token embeddings
        // We need to mean-pool them
        val outer = JSONArray(responseText)

        if (outer.length() == 0) throw Exception("Empty response from encoder")

        // Check if it's already pooled (1D array) or token-level (2D)
        val first = outer.get(0)
        val embedding: FloatArray

        if (first is JSONArray) {
            // 2D: token embeddings — mean pool
            val tokenCount = outer.length()
            val dims = (first as JSONArray).length()
            embedding = FloatArray(dims)

            for (t in 0 until tokenCount) {
                val tokenEmb = outer.getJSONArray(t)
                for (d in 0 until dims) {
                    embedding[d] += tokenEmb.getDouble(d).toFloat()
                }
            }
            for (d in embedding.indices) {
                embedding[d] = embedding[d] / tokenCount
            }
        } else {
            // 1D: already pooled
            embedding = FloatArray(outer.length())
            for (i in 0 until outer.length()) {
                embedding[i] = outer.getDouble(i).toFloat()
            }
        }

        // L2 normalize
        var norm = 0f
        for (v in embedding) norm += v * v
        norm = sqrt(norm)
        if (norm > 1e-9f) {
            for (i in embedding.indices) embedding[i] /= norm
        }

        embedding
    }
}
