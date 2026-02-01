//! Core API functions for ALICE-Voice
//!
//! This module provides the main entry points for voice encoding and decoding.
//!
//! # Functions
//!
//! | Function | Description | Layer |
//! |----------|-------------|-------|
//! | `voice_to_params` | Voice → Parametric | L2 |
//! | `params_to_voice` | Parametric → Voice | L2 |
//!
//! Note: L3 Semantic Layer is available under Commercial License.

use crate::types::{VoiceResult, VoiceQuality};
use crate::layers::{
    SpectralLayer, SpectralParams,
    ParametricLayer, ParametricParams,
};
use serde::{Deserialize, Serialize};

// Re-export convenience functions from layers
pub use crate::layers::parametric::{voice_to_params, params_to_voice};

/// Voice codec configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCodecConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Frame size in samples
    pub frame_size: usize,
    /// Hop size (frame overlap)
    pub hop_size: usize,
    /// LPC order for parametric layer
    pub lpc_order: usize,
    /// Quality level
    pub quality: VoiceQuality,
}

impl Default for VoiceCodecConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            frame_size: 1024, // 64ms frames (pitch detector needs 2 * max_period)
            hop_size: 512,
            lpc_order: 10,
            quality: VoiceQuality::Medium,
        }
    }
}

impl VoiceCodecConfig {
    /// Create config for specific quality level
    pub fn for_quality(quality: VoiceQuality) -> Self {
        let sample_rate = quality.sample_rate();
        let lpc_order = quality.lpc_order();
        // 64ms frames to satisfy pitch detector (needs 2 * sample_rate/min_f0)
        let frame_size = (sample_rate as f32 * 0.064) as usize;
        let hop_size = frame_size / 2;

        Self {
            sample_rate,
            frame_size,
            hop_size,
            lpc_order,
            quality,
        }
    }

    /// Create config for narrowband (8kHz)
    pub fn narrowband() -> Self {
        Self::for_quality(VoiceQuality::Low)
    }

    /// Create config for wideband (16kHz)
    pub fn wideband() -> Self {
        Self::for_quality(VoiceQuality::Medium)
    }

    /// Create config for super-wideband (32kHz)
    pub fn super_wideband() -> Self {
        Self::for_quality(VoiceQuality::High)
    }

    /// Create config for fullband (48kHz)
    pub fn fullband() -> Self {
        Self::for_quality(VoiceQuality::Ultra)
    }
}

/// Unified voice codec supporting L1-L2 layers
///
/// Note: L3 Semantic Layer is available under Commercial License.
/// See: https://github.com/ext-sakamoro/ALICE-Voice-Commercial
#[derive(Debug)]
pub struct VoiceCodec {
    /// Configuration
    config: VoiceCodecConfig,
    /// L1: Spectral layer
    spectral: SpectralLayer,
    /// L2: Parametric layer
    parametric: ParametricLayer,
}

impl VoiceCodec {
    /// Create new voice codec with configuration
    pub fn new(config: VoiceCodecConfig) -> Self {
        Self {
            spectral: SpectralLayer::new(config.frame_size, config.hop_size)
                .with_quality(config.quality),
            parametric: ParametricLayer::new(
                config.lpc_order,
                config.frame_size,
                config.sample_rate,
            ).with_quality(config.quality),
            config,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(VoiceCodecConfig::default())
    }

    /// Get configuration
    pub fn config(&self) -> &VoiceCodecConfig {
        &self.config
    }

    // ============================================
    // L1: Spectral Layer
    // ============================================

    /// Encode to L1 (Spectral)
    pub fn encode_spectral(&mut self, samples: &[f32]) -> VoiceResult<Vec<SpectralParams>> {
        self.spectral.analyze_stream(samples)
    }

    /// Decode from L1 (Spectral)
    pub fn decode_spectral(&mut self, params: &[SpectralParams]) -> Vec<f32> {
        self.spectral.synthesize_stream(params)
    }

    // ============================================
    // L2: Parametric Layer
    // ============================================

    /// Encode to L2 (Parametric)
    pub fn encode_parametric(&mut self, samples: &[f32]) -> VoiceResult<Vec<ParametricParams>> {
        self.parametric.analyze_stream(samples, self.config.hop_size)
    }

    /// Decode from L2 (Parametric)
    pub fn decode_parametric(&self, params: &[ParametricParams]) -> Vec<f32> {
        self.parametric.synthesize_stream(params, self.config.hop_size)
    }

    // ============================================
    // Utility Functions
    // ============================================

    /// Calculate compression ratio for L1
    pub fn compression_ratio_spectral(&self, samples: &[f32], params: &[SpectralParams]) -> f32 {
        let original_size = samples.len() * 4;
        let compressed_size: usize = params.iter().map(|p| p.encoded_size()).sum();
        original_size as f32 / compressed_size as f32
    }

    /// Calculate compression ratio for L2
    pub fn compression_ratio_parametric(&self, samples: &[f32], params: &[ParametricParams]) -> f32 {
        let original_size = samples.len() * 4;
        let compressed_size: usize = params.iter().map(|p| p.encoded_size()).sum();
        original_size as f32 / compressed_size as f32
    }
}

/// Encoding statistics
#[derive(Debug, Clone, Default)]
pub struct EncodingStats {
    /// Total frames processed
    pub frames_processed: usize,
    /// Total samples processed
    pub samples_processed: usize,
    /// Voiced frames count
    pub voiced_frames: usize,
    /// Unvoiced frames count
    pub unvoiced_frames: usize,
    /// Silent frames count
    pub silent_frames: usize,
    /// Average pitch (Hz)
    pub avg_pitch: f32,
    /// Average energy (dB)
    pub avg_energy: f32,
    /// Compression ratio achieved
    pub compression_ratio: f32,
}

impl EncodingStats {
    /// Compute statistics from parametric params
    pub fn from_parametric(params: &[ParametricParams], original_samples: usize) -> Self {
        if params.is_empty() {
            return Self::default();
        }

        let mut stats = Self::default();
        stats.frames_processed = params.len();
        stats.samples_processed = original_samples;

        let mut pitch_sum = 0.0f32;
        let mut pitch_count = 0;
        let mut energy_sum = 0.0f32;

        for p in params {
            if p.activity.is_voiced {
                stats.voiced_frames += 1;
                if p.pitch.is_voiced && p.pitch.f0 > 0.0 {
                    pitch_sum += p.pitch.f0;
                    pitch_count += 1;
                }
            } else if p.activity.energy_db < -50.0 {
                stats.silent_frames += 1;
            } else {
                stats.unvoiced_frames += 1;
            }
            energy_sum += p.activity.energy_db;
        }

        stats.avg_pitch = if pitch_count > 0 {
            pitch_sum / pitch_count as f32
        } else {
            0.0
        };

        stats.avg_energy = energy_sum / params.len() as f32;

        let original_size = original_samples * 4;
        let compressed_size: usize = params.iter().map(|p| p.encoded_size()).sum();
        stats.compression_ratio = original_size as f32 / compressed_size as f32;

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_creation() {
        let codec = VoiceCodec::default_config();
        assert_eq!(codec.config().sample_rate, 16000);
    }

    #[test]
    fn test_all_layers() {
        let mut codec = VoiceCodec::new(VoiceCodecConfig::wideband());

        // Generate test audio
        let samples: Vec<f32> = (0..8000)
            .map(|i| {
                let t = i as f32 / 16000.0;
                (2.0 * std::f32::consts::PI * 200.0 * t).sin() * 0.5
            })
            .collect();

        // L1: Spectral
        let spectral_params = codec.encode_spectral(&samples).unwrap();
        let spectral_decoded = codec.decode_spectral(&spectral_params);
        assert!(spectral_decoded.len() > 0);

        let ratio1 = codec.compression_ratio_spectral(&samples, &spectral_params);
        println!("L1 compression: {:.1}x", ratio1);

        // L2: Parametric
        let parametric_params = codec.encode_parametric(&samples).unwrap();
        let parametric_decoded = codec.decode_parametric(&parametric_params);
        assert!(parametric_decoded.len() > 0);

        let ratio2 = codec.compression_ratio_parametric(&samples, &parametric_params);
        println!("L2 compression: {:.1}x", ratio2);

        // All layers should achieve some compression
        assert!(ratio1 > 1.0, "L1 should compress");
        assert!(ratio2 > 1.0, "L2 should compress");

        // Note: L3 Semantic Layer is available under Commercial License
    }

    #[test]
    fn test_encoding_stats() {
        let mut codec = VoiceCodec::default_config();

        let samples: Vec<f32> = (0..16000)
            .map(|i| {
                let t = i as f32 / 16000.0;
                (2.0 * std::f32::consts::PI * 150.0 * t).sin() * 0.3
            })
            .collect();

        let params = codec.encode_parametric(&samples).unwrap();
        let stats = EncodingStats::from_parametric(&params, samples.len());

        println!("Stats: {:?}", stats);
        assert!(stats.frames_processed > 0);
        assert!(stats.compression_ratio > 1.0);
    }

    #[test]
    fn test_quality_configs() {
        let narrowband = VoiceCodecConfig::narrowband();
        assert_eq!(narrowband.sample_rate, 8000);
        assert_eq!(narrowband.lpc_order, 8);

        let wideband = VoiceCodecConfig::wideband();
        assert_eq!(wideband.sample_rate, 16000);
        assert_eq!(wideband.lpc_order, 10);

        let fullband = VoiceCodecConfig::fullband();
        assert_eq!(fullband.sample_rate, 48000);
        assert_eq!(fullband.lpc_order, 16);
    }
}
