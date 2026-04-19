package sh.webmind.gyan.data

import android.content.Context
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES31
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * GPU-accelerated cosine similarity search using OpenGL ES 3.1 compute shaders.
 * Runs 1.24M dot products in parallel on the phone's GPU.
 */
class GpuSearch(private val context: Context) {

    companion object {
        private const val TAG = "GyanGPU"
        private const val WORK_GROUP_SIZE = 256
    }

    private var eglDisplay: EGLDisplay? = null
    private var eglContext: EGLContext? = null
    private var eglSurface: EGLSurface? = null
    private var computeProgram: Int = 0
    private var embeddingsSSBO: Int = 0
    private var querySSBO: Int = 0
    private var scoresSSBO: Int = 0
    private var dims: Int = 384
    private var count: Int = 0
    private var ready = false

    val isReady get() = ready
    val pairCount get() = count

    /** Initialize EGL context + compile compute shader. */
    fun init(): Boolean {
        try {
            // Create offscreen EGL context for compute
            eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
            val version = IntArray(2)
            EGL14.eglInitialize(eglDisplay, version, 0, version, 1)

            val configAttribs = intArrayOf(
                EGL14.EGL_RENDERABLE_TYPE, 0x0040,
                EGL14.EGL_NONE
            )
            val configs = arrayOfNulls<EGLConfig>(1)
            val numConfigs = IntArray(1)
            EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0)

            val contextAttribs = intArrayOf(
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 3,
                EGL14.EGL_NONE
            )
            eglContext = EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, contextAttribs, 0)

            val surfaceAttribs = intArrayOf(
                EGL14.EGL_WIDTH, 1,
                EGL14.EGL_HEIGHT, 1,
                EGL14.EGL_NONE
            )
            eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay, configs[0], surfaceAttribs, 0)
            EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)

            Log.i(TAG, "EGL context created. GL version: ${GLES31.glGetString(GLES31.GL_VERSION)}")

            // Compile compute shader
            computeProgram = createComputeProgram()
            if (computeProgram == 0) {
                Log.e(TAG, "Failed to compile compute shader")
                return false
            }

            Log.i(TAG, "GPU search initialized")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "GPU init failed", e)
            return false
        }
    }

    /** Load embeddings into GPU buffer (SSBO). */
    fun loadEmbeddings(embeddingsFile: File): Boolean {
        try {
            // Parse numpy header
            val raf = java.io.RandomAccessFile(embeddingsFile, "r")
            raf.seek(8)
            val headerLen = java.lang.Short.reverseBytes(raf.readShort()).toInt() and 0xFFFF
            val headerBytes = ByteArray(headerLen)
            raf.readFully(headerBytes)
            val header = String(headerBytes).trim()

            val shapeMatch = Regex("""'shape':\s*\((\d+),\s*(\d+)\)""").find(header)
                ?: throw Exception("Bad npy header")
            count = shapeMatch.groupValues[1].toInt()
            dims = shapeMatch.groupValues[2].toInt()
            val isFp16 = header.contains("float16") || header.contains("<f2")
            val dataOffset = 8 + 2 + headerLen

            Log.i(TAG, "Loading $count × $dims embeddings (fp16=$isFp16)")

            // Read all embeddings and convert to fp32
            val totalFloats = count * dims
            val floatBuf = FloatBuffer.allocate(totalFloats)

            raf.seek(dataOffset.toLong())
            if (isFp16) {
                // Read in chunks to avoid OOM
                val chunkRows = 10000
                val chunkBytes = ByteArray(chunkRows * dims * 2)
                var remaining = count
                while (remaining > 0) {
                    val rows = minOf(chunkRows, remaining)
                    val bytes = rows * dims * 2
                    raf.readFully(chunkBytes, 0, bytes)
                    for (i in 0 until rows * dims) {
                        val lo = chunkBytes[i * 2].toInt() and 0xFF
                        val hi = chunkBytes[i * 2 + 1].toInt() and 0xFF
                        floatBuf.put(halfToFloat((hi shl 8) or lo))
                    }
                    remaining -= rows
                }
            } else {
                val chunkBytes = ByteArray(10000 * dims * 4)
                var remaining = count
                while (remaining > 0) {
                    val rows = minOf(10000, remaining)
                    val bytes = rows * dims * 4
                    raf.readFully(chunkBytes, 0, bytes)
                    val bb = ByteBuffer.wrap(chunkBytes, 0, bytes).order(ByteOrder.LITTLE_ENDIAN)
                    val fb = bb.asFloatBuffer()
                    val tmp = FloatArray(rows * dims)
                    fb.get(tmp)
                    floatBuf.put(tmp)
                    remaining -= rows
                }
            }
            raf.close()
            floatBuf.flip()

            Log.i(TAG, "Embeddings loaded to CPU: ${floatBuf.remaining()} floats")

            // Upload to GPU SSBO
            val buffers = IntArray(3)
            GLES31.glGenBuffers(3, buffers, 0)
            embeddingsSSBO = buffers[0]
            querySSBO = buffers[1]
            scoresSSBO = buffers[2]

            // Embeddings buffer
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, embeddingsSSBO)
            GLES31.glBufferData(GLES31.GL_SHADER_STORAGE_BUFFER, totalFloats * 4, floatBuf, GLES31.GL_STATIC_DRAW)

            // Query buffer (384 floats)
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, querySSBO)
            GLES31.glBufferData(GLES31.GL_SHADER_STORAGE_BUFFER, dims * 4, null, GLES31.GL_DYNAMIC_DRAW)

            // Scores buffer (count floats)
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, scoresSSBO)
            GLES31.glBufferData(GLES31.GL_SHADER_STORAGE_BUFFER, count * 4, null, GLES31.GL_DYNAMIC_READ)

            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)

            val err = GLES31.glGetError()
            if (err != GLES31.GL_NO_ERROR) {
                Log.e(TAG, "GL error after buffer upload: $err")
                return false
            }

            Log.i(TAG, "Embeddings uploaded to GPU: ${totalFloats * 4 / 1024 / 1024}MB VRAM")
            ready = true
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load embeddings to GPU", e)
            return false
        }
    }

    /** Run cosine similarity on GPU. Returns top-K indices + scores. */
    fun search(queryEmbedding: FloatArray, topK: Int = 5): List<Pair<Int, Float>> {
        if (!ready) return emptyList()

        // Upload query
        val queryBuf = ByteBuffer.allocateDirect(dims * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        queryBuf.put(queryEmbedding)
        queryBuf.flip()

        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, querySSBO)
        GLES31.glBufferSubData(GLES31.GL_SHADER_STORAGE_BUFFER, 0, dims * 4, queryBuf)

        // Bind buffers
        GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 0, embeddingsSSBO)
        GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 1, querySSBO)
        GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 2, scoresSSBO)

        // Set uniforms
        GLES31.glUseProgram(computeProgram)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(computeProgram, "uCount"), count)
        GLES31.glUniform1i(GLES31.glGetUniformLocation(computeProgram, "uDims"), dims)

        // Dispatch compute
        val numGroups = (count + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE
        GLES31.glDispatchCompute(numGroups, 1, 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)

        // Read back scores
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, scoresSSBO)
        val scoresBuf = ByteBuffer.allocateDirect(count * 4).order(ByteOrder.nativeOrder())
        
        val mapped = GLES31.glMapBufferRange(GLES31.GL_SHADER_STORAGE_BUFFER, 0, count * 4, GLES31.GL_MAP_READ_BIT) as ByteBuffer
        mapped.order(ByteOrder.nativeOrder())
        scoresBuf.put(mapped)
        GLES31.glUnmapBuffer(GLES31.GL_SHADER_STORAGE_BUFFER)
        scoresBuf.flip()
        val scoresFloat = scoresBuf.asFloatBuffer()
        val scores = FloatArray(count)
        scoresFloat.get(scores)

        // Find top-K on CPU (fast for small K)
        return scores.indices
            .sortedByDescending { scores[it] }
            .take(topK)
            .map { it to scores[it] }
    }

    /** Create the compute shader program. */
    private fun createComputeProgram(): Int {
        val shaderSource = """
            #version 310 es
            layout(local_size_x = $WORK_GROUP_SIZE) in;

            layout(std430, binding = 0) readonly buffer Embeddings {
                float embeddings[];
            };

            layout(std430, binding = 1) readonly buffer Query {
                float query[];
            };

            layout(std430, binding = 2) writeonly buffer Scores {
                float scores[];
            };

            uniform int uCount;
            uniform int uDims;

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                if (idx >= uint(uCount)) return;

                float dot = 0.0;
                int offset = int(idx) * uDims;
                for (int j = 0; j < uDims; j++) {
                    dot += embeddings[offset + j] * query[j];
                }
                scores[idx] = dot;
            }
        """.trimIndent()

        val shader = GLES31.glCreateShader(GLES31.GL_COMPUTE_SHADER)
        GLES31.glShaderSource(shader, shaderSource)
        GLES31.glCompileShader(shader)

        val compiled = IntArray(1)
        GLES31.glGetShaderiv(shader, GLES31.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            val log = GLES31.glGetShaderInfoLog(shader)
            Log.e(TAG, "Shader compile failed: $log")
            GLES31.glDeleteShader(shader)
            return 0
        }

        val program = GLES31.glCreateProgram()
        GLES31.glAttachShader(program, shader)
        GLES31.glLinkProgram(program)

        val linked = IntArray(1)
        GLES31.glGetProgramiv(program, GLES31.GL_LINK_STATUS, linked, 0)
        if (linked[0] == 0) {
            val log = GLES31.glGetProgramInfoLog(program)
            Log.e(TAG, "Program link failed: $log")
            GLES31.glDeleteProgram(program)
            return 0
        }

        GLES31.glDeleteShader(shader)
        return program
    }

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

    fun destroy() {
        if (computeProgram != 0) GLES31.glDeleteProgram(computeProgram)
        val bufs = intArrayOf(embeddingsSSBO, querySSBO, scoresSSBO)
        GLES31.glDeleteBuffers(3, bufs, 0)
        EGL14.eglDestroySurface(eglDisplay, eglSurface)
        EGL14.eglDestroyContext(eglDisplay, eglContext)
        EGL14.eglTerminate(eglDisplay)
    }
}
