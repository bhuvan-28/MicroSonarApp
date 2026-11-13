package com.example.microsonarapp

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.*

data class RoomMetrics(
    val width: Float,
    val length: Float,
    val height: Float,
    val volume: Float,
    val material: String,
    val confidence: Float,
    val maxMicLevel: Float,
    val rt60: Float,
    val roomType: String,
    val impulseResponse: FloatArray // Added for visualization
)

class SonarEngine(private val context: Context) {

    private val sampleRate = 48000
    private val chirpDuration = 0.1f // 100 ms
    private val fStart = 18000f
    private val fEnd = 22000f

    private val nFft = 2048
    private val hopLength = 512
    private val nMels = 128
    private val fMin = 17000f
    private val fMax = 23000f

    private val ortEnv = OrtEnvironment.getEnvironment()
    private val ortSession: OrtSession
    private val materials = arrayOf("Concrete", "Drywall", "Wood", "Carpet", "Glass")

    init {
        val modelBytes = context.assets.open("sonar_model.onnx").readBytes()
        ortSession = ortEnv.createSession(modelBytes)
    }

    private fun generateLogChirp(): FloatArray {
        val nSamples = (chirpDuration * sampleRate).toInt()
        val chirp = FloatArray(nSamples)
        val logRatio = ln(fEnd / fStart)
        for (i in 0 until nSamples) {
            val t = i.toFloat() / sampleRate
            val phase = 2.0 * Math.PI * fStart * chirpDuration / logRatio * (exp(t / chirpDuration * logRatio) - 1.0)
            val window = 0.5 * (1.0 - cos(2.0 * Math.PI * i / nSamples))
            chirp[i] = (0.85 * sin(phase) * window).toFloat()
        }
        return chirp
    }

    suspend fun runSweep(isDiagnostic: Boolean = false): RoomMetrics = withContext(Dispatchers.Default) {
        val durationSec = if (isDiagnostic) 1.5f else 0.6f
        val chirp = generateLogChirp()
        val totalRecordSamples = (durationSec * sampleRate).toInt()
        val recordedBuffer = FloatArray(totalRecordSamples)

        val track = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder()
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                .build())
            .setAudioFormat(AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setSampleRate(sampleRate)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build())
            .setBufferSizeInBytes(chirp.size * 4)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()

        track.write(chirp, 0, chirp.size, AudioTrack.WRITE_BLOCKING)

        val minBuf = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        if (minBuf <= 0) {
            throw IllegalStateException("AudioRecord configuration not supported on this device")
        }

        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            max(minBuf, totalRecordSamples * 2)
        )

        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            throw IllegalStateException("Microphone hardware failed to initialize")
        }

        val sessionId = recorder.audioSessionId
        val aecAvailable = AcousticEchoCanceler.isAvailable()
        val agcAvailable = AutomaticGainControl.isAvailable()
        val nsAvailable = NoiseSuppressor.isAvailable()
        var aecEnabled = false
        var agcEnabled = false
        var nsEnabled = false
        try { if (aecAvailable) AcousticEchoCanceler.create(sessionId)?.let { aecEnabled = it.enabled; it.release() } } catch (e: Exception) {}
        try { if (agcAvailable) AutomaticGainControl.create(sessionId)?.let { agcEnabled = it.enabled; it.release() } } catch (e: Exception) {}
        try { if (nsAvailable) NoiseSuppressor.create(sessionId)?.let { nsEnabled = it.enabled; it.release() } } catch (e: Exception) {}

        val shortBuffer = ShortArray(totalRecordSamples)

        recorder.startRecording()
        track.play()

        var readTotal = 0
        while (readTotal < totalRecordSamples) {
            val read = recorder.read(shortBuffer, readTotal, totalRecordSamples - readTotal, AudioRecord.READ_BLOCKING)
            if (read > 0) {
                readTotal += read
            } else {
                throw IllegalStateException("AudioRecord read failed with code: $read")
            }
        }

        var sumSquares = 0.0
        var clipCount = 0
        var maxAmp = 0f
        for (i in 0 until totalRecordSamples) {
            val sample = shortBuffer[i] / 32768.0f
            recordedBuffer[i] = sample
            
            val absVal = abs(sample)
            if (absVal > maxAmp) maxAmp = absVal
            if (absVal >= 0.999f) clipCount++
            sumSquares += sample * sample
        }
        val rms = sqrt(sumSquares / totalRecordSamples).toFloat()
        val clipPercentage = (clipCount.toFloat() / totalRecordSamples) * 100f

        try { recorder.stop() } catch (_: Exception) {}
        recorder.release()
        track.release()

        if (!isDiagnostic) {
            // Save raw recording to cache directory so we can inspect it later
            saveWav("raw_recording.wav", recordedBuffer, sampleRate)
        }

        // 1. Matched filter cross-correlation
        val rir = matchedFilter(recordedBuffer, chirp)

        if (isDiagnostic) {
            var corrMaxVal = 0f
            var corrMaxIdx = 0
            for (i in rir.indices) {
                val absVal = abs(rir[i])
                if (absVal > corrMaxVal) {
                    corrMaxVal = absVal
                    corrMaxIdx = i
                }
            }
            val corrPeakTimeMs = (corrMaxIdx.toFloat() / sampleRate) * 1000f
            val postPeakSamples = rir.size - corrMaxIdx
            val postPeakTimeMs = (postPeakSamples.toFloat() / sampleRate) * 1000f

            val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            saveWav("diag_${ts}_mic.wav", recordedBuffer, sampleRate)
            saveWav("diag_${ts}_chirp.wav", chirp, sampleRate)
            
            File(context.cacheDir, "diag_${ts}_correlation.csv").printWriter().use { out ->
                out.println("time_sec,amplitude")
                for (i in rir.indices) {
                    out.println("${(i.toFloat()/sampleRate)},${rir[i]}")
                }
            }
            
            val json = JSONObject()
            json.put("scan_id", ts)
            json.put("sample_rate", sampleRate)
            json.put("recording_duration_sec", durationSec)
            json.put("chirp_duration_sec", chirpDuration)
            
            val micObj = JSONObject()
            micObj.put("peak_amplitude", maxAmp)
            micObj.put("rms_amplitude", rms)
            micObj.put("clipping_percentage", clipPercentage)
            json.put("mic_metrics", micObj)
            
            val corrObj = JSONObject()
            corrObj.put("peak_index", corrMaxIdx)
            corrObj.put("peak_time_ms", corrPeakTimeMs)
            corrObj.put("peak_amplitude", corrMaxVal)
            corrObj.put("available_tail_samples", postPeakSamples)
            corrObj.put("available_tail_ms", postPeakTimeMs)
            json.put("correlation_metrics", corrObj)
            
            val fxObj = JSONObject()
            fxObj.put("aec_available", aecAvailable)
            fxObj.put("aec_enabled", aecEnabled)
            fxObj.put("agc_available", agcAvailable)
            fxObj.put("agc_enabled", agcEnabled)
            fxObj.put("ns_available", nsAvailable)
            fxObj.put("ns_enabled", nsEnabled)
            json.put("audio_effects", fxObj)
            
            File(context.cacheDir, "diag_${ts}_meta.json").writeText(json.toString(2))
        }

        // Calculate Physics-based Reverberation Time (RT60)
        val rt60 = computeRT60(rir)
        val roomType = when {
            rt60 < 0.35f -> "Small Damped Room (Office / Closet)"
            rt60 in 0.35f..0.75f -> "Medium Standard Room (Bedroom)"
            rt60 in 0.75f..1.2f -> "Large Echoey Space (Hallway / Tile)"
            else -> "Vast Acoustic Environment (Hall / Cave)"
        }

        // 2. Compute proper Mel-Spectrogram (128x128) matching Python training
        val melSpec = computeMelSpectrogram(rir)

        // 3. Run ONNX Inference
        val tensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(melSpec), longArrayOf(1, 1, 128, 128))
        val output = ortSession.run(mapOf("mel_spectrogram" to tensor))

        val dims = (output[0].value as Array<FloatArray>)[0]
        val logits = (output[1].value as Array<FloatArray>)[0]

        // Softmax & argmax
        var maxLogit = logits[0]
        var bestIdx = 0
        for (i in 1 until logits.size) {
            if (logits[i] > maxLogit) {
                maxLogit = logits[i]
                bestIdx = i
            }
        }

        var sumExp = 0.0
        for (v in logits) sumExp += exp(v.toDouble())
        val confidence = (exp(maxLogit.toDouble()) / sumExp).toFloat()

        val maxMicLevel = recordedBuffer.maxOf { abs(it) }

        RoomMetrics(
            width = dims[0],
            length = dims[1],
            height = dims[2],
            volume = dims[0] * dims[1] * dims[2],
            material = materials[bestIdx],
            confidence = confidence,
            maxMicLevel = maxMicLevel,
            rt60 = rt60,
            roomType = roomType,
            impulseResponse = rir // Export the IR for graphing
        )
    }

    private fun matchedFilter(echo: FloatArray, ref: FloatArray): FloatArray {
        val out = FloatArray(echo.size)
        val step = 1
        val maxRefLen = ref.size // Fix: Use the full chirp length for correlation instead of truncating to 2400
        var maxVal = 1e-9f

        for (i in 0 until echo.size - maxRefLen step step) {
            var sum = 0f
            for (j in 0 until maxRefLen) {
                sum += echo[i + j] * ref[j]
            }
            out[i] = sum
            if (abs(sum) > maxVal) maxVal = abs(sum)
        }

        for (i in out.indices) out[i] /= maxVal
        return out
    }

    private fun computeMelSpectrogram(signal: FloatArray): FloatArray {
        val nFrames = 128
        val hop = max(1, (signal.size - nFft) / nFrames)
        val spec2D = Array(nMels) { FloatArray(nFrames) }

        // Window
        val window = FloatArray(nFft) { 0.5f * (1.0f - cos(2.0f * Math.PI.toFloat() * it / nFft)) }

        // Simple Mel filter centers
        val melMin = 2595.0f * log10(1.0f + fMin / 700.0f)
        val melMax = 2595.0f * log10(1.0f + fMax / 700.0f)
        val melBins = FloatArray(nMels + 2) {
            val m = melMin + (melMax - melMin) * it / (nMels + 1)
            700.0f * (10.0f.pow(m / 2595.0f) - 1.0f)
        }
        val binIdx = IntArray(nMels + 2) {
            ((nFft + 1) * melBins[it] / sampleRate).toInt().coerceIn(0, nFft / 2)
        }

        var sMin = Float.MAX_VALUE
        var sMax = -Float.MAX_VALUE

        for (f in 0 until nFrames) {
            val start = (f * hop).coerceIn(0, signal.size - nFft)
            // Power spectrum via DFT on band
            val power = FloatArray(nFft / 2 + 1)
            for (k in binIdx.first() until min(binIdx.last() + 1, nFft / 2)) {
                var re = 0f
                var im = 0f
                val omega = 2.0 * Math.PI * k / nFft
                for (n in 0 until nFft) {
                    val sample = signal[start + n] * window[n]
                    re += (sample * cos(omega * n)).toFloat()
                    im -= (sample * sin(omega * n)).toFloat()
                }
                power[k] = re * re + im * im
            }

            // Mel filterbank integration
            for (m in 0 until nMels) {
                val fLeft = binIdx[m]
                val fCenter = binIdx[m + 1]
                val fRight = binIdx[m + 2]
                var melEnergy = 0f

                for (k in fLeft until fCenter) {
                    if (fCenter > fLeft) melEnergy += power[k] * (k - fLeft).toFloat() / (fCenter - fLeft)
                }
                for (k in fCenter until fRight) {
                    if (fRight > fCenter) melEnergy += power[k] * (fRight - k).toFloat() / (fRight - fCenter)
                }

                val logMel = (10.0f * log10(melEnergy + 1e-9f))
                spec2D[m][f] = logMel
                if (logMel < sMin) sMin = logMel
                if (logMel > sMax) sMax = logMel
            }
        }

        // Normalize to [-1.0, 1.0]
        val range = if (sMax > sMin) sMax - sMin else 1f
        val outTensor = FloatArray(nMels * nFrames)
        var p = 0
        for (m in 0 until nMels) {
            for (f in 0 until nFrames) {
                val norm = 2.0f * (spec2D[m][f] - sMin) / range - 1.0f
                outTensor[p++] = norm
            }
        }
        return outTensor
    }

    private fun saveWav(filename: String, data: FloatArray, sampleRate: Int) {
        val file = File(context.cacheDir, filename)
        val out = FileOutputStream(file)
        
        val pcm = ByteArray(data.size * 2)
        var idx = 0
        for (f in data) {
            val s = (f.coerceIn(-1f, 1f) * 32767).toInt().toShort()
            pcm[idx++] = (s.toInt() and 0xFF).toByte()
            pcm[idx++] = ((s.toInt() shr 8) and 0xFF).toByte()
        }
        
        val totalDataLen = pcm.size + 36
        val byteRate = sampleRate * 2
        
        val header = ByteArray(44)
        header[0] = 'R'.code.toByte(); header[1] = 'I'.code.toByte(); header[2] = 'F'.code.toByte(); header[3] = 'F'.code.toByte()
        header[4] = (totalDataLen and 0xff).toByte(); header[5] = ((totalDataLen shr 8) and 0xff).toByte(); header[6] = ((totalDataLen shr 16) and 0xff).toByte(); header[7] = ((totalDataLen shr 24) and 0xff).toByte()
        header[8] = 'W'.code.toByte(); header[9] = 'A'.code.toByte(); header[10] = 'V'.code.toByte(); header[11] = 'E'.code.toByte()
        header[12] = 'f'.code.toByte(); header[13] = 'm'.code.toByte(); header[14] = 't'.code.toByte(); header[15] = ' '.code.toByte()
        header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0
        header[20] = 1; header[21] = 0
        header[22] = 1; header[23] = 0 // 1 channel
        header[24] = (sampleRate and 0xff).toByte(); header[25] = ((sampleRate shr 8) and 0xff).toByte(); header[26] = ((sampleRate shr 16) and 0xff).toByte(); header[27] = ((sampleRate shr 24) and 0xff).toByte()
        header[28] = (byteRate and 0xff).toByte(); header[29] = ((byteRate shr 8) and 0xff).toByte(); header[30] = ((byteRate shr 16) and 0xff).toByte(); header[31] = ((byteRate shr 24) and 0xff).toByte()
        header[32] = 2; header[33] = 0 // block align
        header[34] = 16; header[35] = 0 // bits per sample
        header[36] = 'd'.code.toByte(); header[37] = 'a'.code.toByte(); header[38] = 't'.code.toByte(); header[39] = 'a'.code.toByte()
        header[40] = (pcm.size and 0xff).toByte(); header[41] = ((pcm.size shr 8) and 0xff).toByte(); header[42] = ((pcm.size shr 16) and 0xff).toByte(); header[43] = ((pcm.size shr 24) and 0xff).toByte()
        
        out.write(header)
        out.write(pcm)
        out.close()
    }

    private fun computeRT60(rir: FloatArray): Float {
        var maxIdx = 0
        var maxVal = 0f
        for (i in rir.indices) {
            val absVal = abs(rir[i])
            if (absVal > maxVal) {
                maxVal = absVal
                maxIdx = i
            }
        }
        if (maxVal < 1e-6f) return 0.0f

        val tailLen = rir.size - maxIdx
        if (tailLen < 100) return 0.0f

        val energy = DoubleArray(tailLen)
        var cumulative = 0.0
        for (i in tailLen - 1 downTo 0) {
            val sample = rir[maxIdx + i].toDouble()
            cumulative += sample * sample
            energy[i] = cumulative
        }

        val maxEnergy = energy[0]
        if (maxEnergy <= 1e-12) return 0.0f

        val dbEnergy = FloatArray(tailLen)
        for (i in 0 until tailLen) {
            val norm = energy[i] / maxEnergy
            dbEnergy[i] = (10.0 * log10(max(norm, 1e-10))).toFloat()
        }

        var idx5 = -1
        var idx25 = -1
        for (i in 0 until tailLen) {
            if (idx5 == -1 && dbEnergy[i] <= -5.0f) {
                idx5 = i
            }
            if (idx25 == -1 && dbEnergy[i] <= -25.0f) {
                idx25 = i
            }
        }

        if (idx5 != -1 && idx25 != -1 && idx25 > idx5) {
            val samplesFor20dB = (idx25 - idx5).toFloat()
            val timeFor20dB = samplesFor20dB / sampleRate
            return (timeFor20dB * 3.0f).coerceIn(0.05f, 3.5f)
        }

        var idxNoise = tailLen - 1
        for (i in 0 until tailLen) {
            if (dbEnergy[i] <= -15.0f) {
                idxNoise = i
                break
            }
        }
        val fallbackSec = (idxNoise.toFloat() / sampleRate) * (60.0f / 15.0f)
        return fallbackSec.coerceIn(0.1f, 3.0f)
    }
}

