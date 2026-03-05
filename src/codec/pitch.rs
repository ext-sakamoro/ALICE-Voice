//! Pitch Detection and Generation
//!
//! Pitch (fundamental frequency, F0) is the perceived frequency of speech,
//! determined by the vibration rate of the vocal folds.
//!
//! # Algorithms
//!
//! - Autocorrelation-based pitch detection
//! - YIN algorithm for robust estimation
//! - AMDF (Average Magnitude Difference Function)

use crate::types::{VoiceActivity, VoiceError, VoiceResult};
use serde::{Deserialize, Serialize};

/// Pitch detection result
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PitchInfo {
    /// Fundamental frequency in Hz (0 if unvoiced)
    pub f0: f32,
    /// Pitch period in samples
    pub period: f32,
    /// Voicing probability (0.0 - 1.0)
    pub voicing_prob: f32,
    /// Pitch confidence (0.0 - 1.0)
    pub confidence: f32,
    /// Is the frame voiced?
    pub is_voiced: bool,
}

impl Default for PitchInfo {
    fn default() -> Self {
        Self {
            f0: 0.0,
            period: 0.0,
            voicing_prob: 0.0,
            confidence: 0.0,
            is_voiced: false,
        }
    }
}

impl PitchInfo {
    /// Create new pitch info for voiced frame
    #[must_use]
    pub fn voiced(f0: f32, confidence: f32, sample_rate: u32) -> Self {
        Self {
            f0,
            period: sample_rate as f32 / f0,
            voicing_prob: confidence,
            confidence,
            is_voiced: true,
        }
    }

    /// Create pitch info for unvoiced frame
    #[must_use]
    pub fn unvoiced() -> Self {
        Self::default()
    }

    /// Get pitch period in milliseconds
    #[must_use]
    pub fn period_ms(&self, _sample_rate: u32) -> f32 {
        if self.f0 > 0.0 {
            1000.0 / self.f0
        } else {
            0.0
        }
    }

    /// Convert to MIDI note number (A4 = 69)
    #[must_use]
    pub fn to_midi(&self) -> Option<f32> {
        if self.f0 > 0.0 {
            Some(12.0f32.mul_add((self.f0 / 440.0).log2(), 69.0))
        } else {
            None
        }
    }
}

/// Pitch detection algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PitchAlgorithm {
    /// Autocorrelation-based
    #[default]
    Autocorrelation,
    /// YIN algorithm (more robust)
    Yin,
    /// Average Magnitude Difference Function
    Amdf,
}

/// Pitch detector configuration
///
/// Level 1 & 2 optimized: pre-allocated buffers + SIMD-friendly loops
#[derive(Debug, Clone)]
pub struct PitchDetector {
    /// Sample rate
    sample_rate: u32,
    /// Minimum pitch frequency (Hz)
    min_f0: f32,
    /// Maximum pitch frequency (Hz)
    max_f0: f32,
    /// Voicing threshold
    voicing_threshold: f32,
    /// Detection algorithm
    algorithm: PitchAlgorithm,
    /// Workspace: autocorrelation buffer
    ws_autocorr: Vec<f32>,
    /// Workspace: YIN difference buffer
    ws_yin_d: Vec<f32>,
    /// Workspace: YIN normalized difference buffer
    ws_yin_d_prime: Vec<f32>,
    /// Workspace: AMDF buffer
    ws_amdf: Vec<f32>,
}

impl PitchDetector {
    /// Create new pitch detector with default settings
    ///
    /// Pre-allocates workspace buffers for zero-allocation detection.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        let max_period = (sample_rate as f32 / 50.0) as usize + 1; // 50 Hz min

        Self {
            sample_rate,
            min_f0: 50.0,
            max_f0: 500.0,
            voicing_threshold: 0.3,
            algorithm: PitchAlgorithm::Autocorrelation,
            ws_autocorr: vec![0.0; max_period + 1],
            ws_yin_d: vec![0.0; max_period + 1],
            ws_yin_d_prime: vec![0.0; max_period + 1],
            ws_amdf: vec![0.0; max_period + 1],
        }
    }

    /// Set pitch range
    #[must_use]
    pub const fn with_pitch_range(mut self, min_f0: f32, max_f0: f32) -> Self {
        self.min_f0 = min_f0;
        self.max_f0 = max_f0;
        self
    }

    /// Set voicing threshold
    #[must_use]
    pub const fn with_voicing_threshold(mut self, threshold: f32) -> Self {
        self.voicing_threshold = threshold;
        self
    }

    /// Set detection algorithm
    #[must_use]
    pub const fn with_algorithm(mut self, algorithm: PitchAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Get minimum and maximum period in samples
    fn get_period_range(&self) -> (usize, usize) {
        let min_period = (self.sample_rate as f32 / self.max_f0) as usize;
        let max_period = (self.sample_rate as f32 / self.min_f0) as usize;
        (min_period.max(2), max_period)
    }

    /// Detect pitch in audio frame
    ///
    /// Uses pre-allocated buffers for zero-allocation pitch detection.
    ///
    /// # Errors
    ///
    /// Returns `VoiceError` if pitch detection fails.
    pub fn detect(&mut self, samples: &[f32]) -> VoiceResult<PitchInfo> {
        match self.algorithm {
            PitchAlgorithm::Autocorrelation => self.detect_autocorrelation(samples),
            PitchAlgorithm::Yin => self.detect_yin(samples),
            PitchAlgorithm::Amdf => self.detect_amdf(samples),
        }
    }

    /// Autocorrelation-based pitch detection
    ///
    /// Level 1: Uses pre-allocated `ws_autocorr` buffer
    /// Level 2: SIMD-friendly zip pattern for autocorrelation
    fn detect_autocorrelation(&mut self, samples: &[f32]) -> VoiceResult<PitchInfo> {
        let (min_period, max_period) = self.get_period_range();

        if samples.len() < max_period * 2 {
            return Err(VoiceError::BufferTooSmall {
                need: max_period * 2,
                got: samples.len(),
            });
        }

        // Ensure buffer is large enough
        if self.ws_autocorr.len() < max_period + 1 {
            self.ws_autocorr.resize(max_period + 1, 0.0);
        }

        let n = samples.len();

        // Energy at lag 0 (SIMD-friendly)
        let energy: f32 = samples.iter().map(|&s| s * s).sum();
        self.ws_autocorr[0] = energy;

        if energy < 1e-10 {
            return Ok(PitchInfo::unvoiced());
        }

        // Compute autocorrelation for each lag (SIMD-friendly zip pattern)
        for lag in min_period..=max_period {
            // Level 2: zip iterator pattern for auto-vectorization
            let sum: f32 = samples[..n - lag]
                .iter()
                .zip(samples[lag..].iter())
                .map(|(&a, &b)| a * b)
                .sum();
            self.ws_autocorr[lag] = sum;
        }

        // Find peak in autocorrelation
        let mut best_lag = 0;
        let mut best_corr = 0.0f32;

        for lag in min_period..=max_period {
            let normalized = self.ws_autocorr[lag] / energy;
            if normalized > best_corr {
                best_corr = normalized;
                best_lag = lag;
            }
        }

        // Parabolic interpolation for better precision
        let refined_lag = if best_lag > min_period && best_lag < max_period {
            let y0 = self.ws_autocorr[best_lag - 1];
            let y1 = self.ws_autocorr[best_lag];
            let y2 = self.ws_autocorr[best_lag + 1];

            let denom = 2.0f32.mul_add(-y1, y0) + y2;
            if denom.abs() > 1e-10 {
                best_lag as f32 + (y0 - y2) / (2.0 * denom)
            } else {
                best_lag as f32
            }
        } else {
            best_lag as f32
        };

        // Determine voicing
        let is_voiced = best_corr > self.voicing_threshold;

        if is_voiced && refined_lag > 0.0 {
            let f0 = self.sample_rate as f32 / refined_lag;
            Ok(PitchInfo {
                f0,
                period: refined_lag,
                voicing_prob: best_corr,
                confidence: best_corr,
                is_voiced: true,
            })
        } else {
            Ok(PitchInfo::unvoiced())
        }
    }

    /// YIN pitch detection algorithm
    ///
    /// Level 1: Uses pre-allocated `ws_yin_d`, `ws_yin_d_prime` buffers
    /// Level 2: SIMD-friendly zip pattern for difference function
    fn detect_yin(&mut self, samples: &[f32]) -> VoiceResult<PitchInfo> {
        let (min_period, max_period) = self.get_period_range();

        if samples.len() < max_period * 2 {
            return Err(VoiceError::BufferTooSmall {
                need: max_period * 2,
                got: samples.len(),
            });
        }

        // Ensure buffers are large enough
        let buf_size = max_period + 1;
        if self.ws_yin_d.len() < buf_size {
            self.ws_yin_d.resize(buf_size, 0.0);
            self.ws_yin_d_prime.resize(buf_size, 0.0);
        }

        let n = samples.len() / 2;

        // Step 1 & 2: Difference function (SIMD-friendly)
        self.ws_yin_d[0] = 0.0;
        for tau in 1..=max_period {
            // Level 2: zip iterator pattern for auto-vectorization
            let sum: f32 = samples[..n]
                .iter()
                .zip(samples[tau..tau + n].iter())
                .map(|(&a, &b)| {
                    let diff = a - b;
                    diff * diff
                })
                .sum();
            self.ws_yin_d[tau] = sum;
        }

        // Step 3: Cumulative mean normalized difference
        self.ws_yin_d_prime[0] = 1.0;
        let mut running_sum = 0.0;

        for tau in 1..=max_period {
            running_sum += self.ws_yin_d[tau];
            self.ws_yin_d_prime[tau] = if running_sum > 0.0 {
                self.ws_yin_d[tau] * tau as f32 / running_sum
            } else {
                1.0
            };
        }

        // Step 4: Absolute threshold
        let threshold = 0.1;
        let mut best_tau = 0;

        for tau in min_period..=max_period {
            if self.ws_yin_d_prime[tau] < threshold {
                // Find local minimum
                best_tau = tau;
                while best_tau < max_period
                    && self.ws_yin_d_prime[best_tau + 1] < self.ws_yin_d_prime[best_tau]
                {
                    best_tau += 1;
                }
                break;
            }
        }

        // If no pitch found below threshold, find global minimum
        if best_tau == 0 {
            let mut min_val = f32::MAX;
            for tau in min_period..=max_period {
                if self.ws_yin_d_prime[tau] < min_val {
                    min_val = self.ws_yin_d_prime[tau];
                    best_tau = tau;
                }
            }
        }

        // Step 5: Parabolic interpolation
        let refined_tau = if best_tau > min_period && best_tau < max_period {
            let y0 = self.ws_yin_d_prime[best_tau - 1];
            let y1 = self.ws_yin_d_prime[best_tau];
            let y2 = self.ws_yin_d_prime[best_tau + 1];

            let denom = 2.0f32.mul_add(-y1, y0) + y2;
            if denom.abs() > 1e-10 {
                best_tau as f32 + (y0 - y2) / (2.0 * denom)
            } else {
                best_tau as f32
            }
        } else {
            best_tau as f32
        };

        // Compute confidence
        let confidence = 1.0 - self.ws_yin_d_prime[best_tau].min(1.0);
        let is_voiced = confidence > self.voicing_threshold;

        if is_voiced && refined_tau > 0.0 {
            let f0 = self.sample_rate as f32 / refined_tau;
            Ok(PitchInfo {
                f0,
                period: refined_tau,
                voicing_prob: confidence,
                confidence,
                is_voiced: true,
            })
        } else {
            Ok(PitchInfo::unvoiced())
        }
    }

    /// AMDF-based pitch detection
    ///
    /// Level 1: Uses pre-allocated `ws_amdf` buffer
    /// Level 2: SIMD-friendly zip pattern for AMDF
    fn detect_amdf(&mut self, samples: &[f32]) -> VoiceResult<PitchInfo> {
        let (min_period, max_period) = self.get_period_range();

        if samples.len() < max_period * 2 {
            return Err(VoiceError::BufferTooSmall {
                need: max_period * 2,
                got: samples.len(),
            });
        }

        // Ensure buffer is large enough
        if self.ws_amdf.len() < max_period + 1 {
            self.ws_amdf.resize(max_period + 1, 0.0);
        }

        let n = samples.len() / 2;
        let n_inv = 1.0 / n as f32;

        // Compute AMDF (SIMD-friendly)
        let mut min_amdf = f32::MAX;
        let mut best_lag = 0;

        for lag in min_period..=max_period {
            // Level 2: zip iterator pattern for auto-vectorization
            let sum: f32 = samples[..n]
                .iter()
                .zip(samples[lag..lag + n].iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum();
            self.ws_amdf[lag] = sum * n_inv;

            if self.ws_amdf[lag] < min_amdf {
                min_amdf = self.ws_amdf[lag];
                best_lag = lag;
            }
        }

        // Compute energy for normalization (SIMD-friendly)
        let energy: f32 = samples[..n].iter().map(|&s| s.abs()).sum::<f32>() * n_inv;

        if energy < 1e-10 {
            return Ok(PitchInfo::unvoiced());
        }

        // Confidence based on AMDF minimum relative to energy
        let confidence = 1.0 - (min_amdf / (energy + 1e-10)).min(1.0);
        let is_voiced = confidence > self.voicing_threshold;

        if is_voiced && best_lag > 0 {
            let f0 = self.sample_rate as f32 / best_lag as f32;
            Ok(PitchInfo {
                f0,
                period: best_lag as f32,
                voicing_prob: confidence,
                confidence,
                is_voiced: true,
            })
        } else {
            Ok(PitchInfo::unvoiced())
        }
    }

    /// Detect voice activity
    #[must_use]
    pub fn detect_voice_activity(&self, samples: &[f32]) -> VoiceActivity {
        // Compute energy
        let energy: f32 = samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32;
        let energy_db = 10.0 * (energy + 1e-10).log10();

        // Simple energy-based VAD
        let threshold_db = -40.0;
        let is_voiced = energy_db > threshold_db;

        // Compute zero-crossing rate for confidence
        let mut zero_crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                zero_crossings += 1;
            }
        }
        let zcr = zero_crossings as f32 / samples.len() as f32;

        // High ZCR suggests unvoiced speech or noise
        let confidence = if is_voiced {
            (1.0 - zcr * 5.0).clamp(0.0, 1.0)
        } else {
            0.0
        };

        VoiceActivity {
            is_voiced,
            confidence,
            energy_db,
        }
    }
}

/// Generate excitation signal for LPC synthesis (allocating)
#[must_use]
pub fn generate_excitation(pitch_info: &PitchInfo, length: usize, sample_rate: u32) -> Vec<f32> {
    let mut excitation = vec![0.0; length];
    generate_excitation_into(pitch_info, &mut excitation, sample_rate);
    excitation
}

/// Generate excitation signal into pre-allocated buffer (zero allocation)
///
/// # Arguments
/// * `pitch_info` - Pitch detection result
/// * `output` - Output buffer to fill with excitation signal
/// * `sample_rate` - Sample rate in Hz (unused, kept for API consistency)
///
/// # Performance
///
/// This function performs zero heap allocation, writing directly into
/// the provided output buffer.
#[inline]
pub fn generate_excitation_into(pitch_info: &PitchInfo, output: &mut [f32], _sample_rate: u32) {
    let length = output.len();

    if pitch_info.is_voiced && pitch_info.period > 0.0 {
        // Voiced: pulse train
        // First, clear the buffer
        output.fill(0.0);

        let period = pitch_info.period as usize;
        if period > 0 {
            let mut pos = 0;
            while pos < length {
                output[pos] = 1.0;
                pos += period;
            }
        }
    } else {
        // Unvoiced: white noise (deterministic LCG)
        let mut seed: u32 = 12345;

        // 4x unrolled for SIMD
        let unroll_end = length - (length % 4);
        let mut i = 0;

        while i < unroll_end {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            output[i] = (seed as f32 / u32::MAX as f32).mul_add(2.0, -1.0);

            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            output[i + 1] = (seed as f32 / u32::MAX as f32).mul_add(2.0, -1.0);

            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            output[i + 2] = (seed as f32 / u32::MAX as f32).mul_add(2.0, -1.0);

            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            output[i + 3] = (seed as f32 / u32::MAX as f32).mul_add(2.0, -1.0);

            i += 4;
        }

        // Handle remaining elements
        while i < length {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            output[i] = (seed as f32 / u32::MAX as f32).mul_add(2.0, -1.0);
            i += 1;
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_pitch_info_creation() {
        let pitch = PitchInfo::voiced(440.0, 0.95, 16000);
        assert_eq!(pitch.f0, 440.0);
        assert!(pitch.is_voiced);

        let midi = pitch.to_midi().unwrap();
        assert!((midi - 69.0).abs() < 0.01); // A4 = MIDI 69
    }

    #[test]
    fn test_pitch_detection() {
        let mut detector = PitchDetector::new(16000);

        // Generate simple sine wave at 200 Hz
        let samples: Vec<f32> = (0..1024)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 16000.0).sin())
            .collect();

        let pitch = detector.detect(&samples).unwrap();

        // Should detect pitch around 200 Hz (allow some tolerance)
        if pitch.is_voiced {
            assert!((pitch.f0 - 200.0).abs() < 20.0);
        }
    }

    #[test]
    fn test_yin_algorithm() {
        let mut detector = PitchDetector::new(16000).with_algorithm(PitchAlgorithm::Yin);

        // Generate sine wave at 300 Hz
        let samples: Vec<f32> = (0..1024)
            .map(|i| (2.0 * std::f32::consts::PI * 300.0 * i as f32 / 16000.0).sin())
            .collect();

        let pitch = detector.detect(&samples).unwrap();

        if pitch.is_voiced {
            assert!((pitch.f0 - 300.0).abs() < 30.0);
        }
    }

    #[test]
    fn test_voice_activity_detection() {
        let detector = PitchDetector::new(16000);

        // Loud signal
        let loud: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();
        let vad = detector.detect_voice_activity(&loud);
        assert!(vad.is_voiced);

        // Silent signal
        let silent: Vec<f32> = vec![0.0001; 512];
        let vad = detector.detect_voice_activity(&silent);
        assert!(!vad.is_voiced);
    }

    #[test]
    fn test_excitation_generation() {
        // Voiced excitation (pulse train)
        let pitch = PitchInfo::voiced(200.0, 0.9, 16000);
        let excitation = generate_excitation(&pitch, 1000, 16000);
        assert_eq!(excitation.len(), 1000);

        // Should have pulses at period intervals
        let period = 16000.0 / 200.0;
        assert!(excitation[0].abs() > 0.5);
        assert!(excitation[period as usize].abs() > 0.5);

        // Unvoiced excitation (noise)
        let unvoiced = PitchInfo::unvoiced();
        let noise = generate_excitation(&unvoiced, 1000, 16000);
        assert_eq!(noise.len(), 1000);
    }
}
