package com.example.microsonarapp.api

import com.example.microsonarapp.RoomMetrics

/**
 * GameAcousticProfile represents the processed acoustic data tailored specifically
 * for consumption by a game engine (Unity, Unreal, or Custom Android Game Loop).
 */
data class GameAcousticProfile(
    val environmentTier: EnvironmentTier,
    val reverbTimeSec: Float,
    val suggestedMapScale: Float,
    val suggestedAudioReverbLevel: Float,
    val isReliable: Boolean
)

enum class EnvironmentTier {
    TIGHT_CORRIDOR,  // RT60 < 0.2s
    STANDARD_ROOM,   // RT60 0.2s - 0.6s
    LARGE_HALL,      // RT60 0.6s - 1.2s
    VAST_CAVERN      // RT60 > 1.2s
}

object GameAcousticApi {

    /**
     * Translates raw physical RoomMetrics into actionable game mechanic parameters.
     */
    fun translateMetricsToGameProfile(metrics: RoomMetrics): GameAcousticProfile {
        val rt60 = metrics.rt60
        
        // 1. Determine the level architecture type based on reverb
        val tier = when {
            rt60 < 0.2f -> EnvironmentTier.TIGHT_CORRIDOR
            rt60 in 0.2f..0.6f -> EnvironmentTier.STANDARD_ROOM
            rt60 in 0.6f..1.2f -> EnvironmentTier.LARGE_HALL
            else -> EnvironmentTier.VAST_CAVERN
        }
        
        // 2. Map Scale Multiplier (1.0x is standard). 
        // Tight spaces shrink the map, vast spaces expand it.
        val mapScale = when (tier) {
            EnvironmentTier.TIGHT_CORRIDOR -> 0.75f
            EnvironmentTier.STANDARD_ROOM -> 1.0f
            EnvironmentTier.LARGE_HALL -> 1.5f
            EnvironmentTier.VAST_CAVERN -> 2.5f
        }
        
        // 3. Audio Engine Reverb (0.0 to 1.0 multiplier for game audio system)
        val audioReverb = (rt60 / 2.0f).coerceIn(0.0f, 1.0f)
        
        // 4. Data Reliability (If mic signal is near zero, reject the scan)
        val isReliable = metrics.maxMicLevel > 0.005f
        
        return GameAcousticProfile(
            environmentTier = tier,
            reverbTimeSec = rt60,
            suggestedMapScale = mapScale,
            suggestedAudioReverbLevel = audioReverb,
            isReliable = isReliable
        )
    }
}
