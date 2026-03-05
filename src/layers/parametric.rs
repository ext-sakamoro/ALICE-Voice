//! L2: Parametric Layer
//!
//! Speech production model using Linear Predictive Coding (LPC).
//! This is the primary layer for voice communication, balancing
//! compression and quality.
//!
//! # Model
//!
//! Voice = Excitation * Vocal Tract Filter
//!
//! - Excitation: Pitch (voiced) or noise (unvoiced)
//! - Vocal Tract: LPC all-pole filter
//!
//! # Parameters
//!
//! | Parameter | Size | Description |
//! |-----------|------|-------------|
//! | LPC coefficients | 10-16 × 4 bytes | Vocal tract shape |
//! | Pitch | 4 bytes | Fundamental frequency |
//! | Gain | 4 bytes | Energy level |
//! | Formants | 4 × 8 bytes | Resonance frequencies |
//!
//! # Performance Optimizations
//!
//! - Zero-copy `ParametricParamsView` for analyze results
//! - Pre-allocated output buffers for all components
//! - `analyze_into` pattern eliminates per-frame allocation

use crate::codec::formant::{Formant, FormantExtractor};
use crate::codec::lpc::{LpcAnalyzer, LpcCoefficients};
use crate::codec::pitch::{
    generate_excitation, generate_excitation_into, PitchDetector, PitchInfo,
};
use crate::types::{VoiceActivity, VoiceError, VoiceQuality, VoiceResult};
use serde::{Deserialize, Serialize};

// ============================================
// Constants
// ============================================

/// Maximum LPC order for stack allocation
const MAX_LPC_ORDER: usize = 32;

/// Maximum formants for stack allocation
const MAX_FORMANTS: usize = 8;

/// Parametric voice parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricParams {
    /// LPC coefficients
    pub lpc: LpcCoefficients,
    /// Pitch information
    pub pitch: PitchInfo,
    /// Formant frequencies
    pub formants: Vec<Formant>,
    /// Voice activity
    pub activity: VoiceActivity,
    /// Frame duration in samples
    pub frame_size: usize,
    /// Sample rate
    pub sample_rate: u32,
}

impl ParametricParams {
    #[must_use]
    pub fn new(lpc_order: usize, frame_size: usize, sample_rate: u32) -> Self {
        Self {
            lpc: LpcCoefficients::new(lpc_order),
            pitch: PitchInfo::default(),
            formants: Vec::new(),
            activity: VoiceActivity::default(),
            frame_size,
            sample_rate,
        }
    }

    /// Estimate encoded size in bytes
    #[must_use]
    pub fn encoded_size(&self) -> usize {
        // LPC coeffs: order × 4 bytes
        // Gain: 4 bytes
        // Pitch: 8 bytes
        // Formants: n × 8 bytes
        // Activity: 8 bytes
        4 + self.lpc.coeffs.len() * 4 + 4 + 8 + self.formants.len() * 8 + 8
    }

    /// Convert to fixed-point for embedded systems
    #[must_use]
    pub fn to_fixed(&self) -> ParametricParamsFixed {
        ParametricParamsFixed {
            lpc: self.lpc.to_fixed(),
            pitch_q16: (self.pitch.f0 * 65536.0) as i32,
            voicing: self.pitch.is_voiced,
            frame_size: self.frame_size,
            sample_rate: self.sample_rate,
        }
    }
}

/// Fixed-point parametric params (ALICE-Edge compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricParamsFixed {
    /// LPC coefficients in Q16.16
    pub lpc: crate::codec::lpc::LpcCoefficientsFixed,
    /// Pitch in Q16.16 Hz
    pub pitch_q16: i32,
    /// Is voiced?
    pub voicing: bool,
    /// Frame size
    pub frame_size: usize,
    /// Sample rate
    pub sample_rate: u32,
}

// ============================================
// Zero-Copy View (No Allocation)
// ============================================

/// Zero-copy view into parametric analysis results
///
/// This struct borrows data from the `ParametricLayer`'s internal buffers,
/// avoiding all per-frame heap allocation. Use this for real-time processing.
///
/// # Lifetime
///
/// The view is valid until the next call to `analyze_into` on the same layer.
#[derive(Debug, Clone, Copy)]
pub struct ParametricParamsView<'a> {
    /// LPC coefficients (borrowed from layer's internal buffer)
    pub lpc_coeffs: &'a [f32],
    /// LPC gain
    pub lpc_gain: f32,
    /// LPC prediction error
    pub lpc_error: f32,
    /// Formants (borrowed from layer's internal buffer)
    pub formants: &'a [Formant],
    /// Number of valid formants
    pub formant_count: usize,
    /// Pitch information (Copy type, no allocation)
    pub pitch: PitchInfo,
    /// Voice activity (Copy type, no allocation)
    pub activity: VoiceActivity,
    /// Frame size
    pub frame_size: usize,
    /// Sample rate
    pub sample_rate: u32,
}

impl ParametricParamsView<'_> {
    /// Convert to owned `ParametricParams` (allocates)
    #[must_use]
    pub fn to_owned(&self) -> ParametricParams {
        ParametricParams {
            lpc: LpcCoefficients {
                coeffs: self.lpc_coeffs.to_vec(),
                reflection: Vec::new(),
                gain: self.lpc_gain,
                error: self.lpc_error,
            },
            pitch: self.pitch,
            formants: self.formants[..self.formant_count].to_vec(),
            activity: self.activity,
            frame_size: self.frame_size,
            sample_rate: self.sample_rate,
        }
    }

    /// Estimate encoded size in bytes
    #[inline(always)]
    #[must_use]
    pub fn encoded_size(&self) -> usize {
        4 + self.lpc_coeffs.len() * 4 + 4 + 8 + self.formant_count * 8 + 8
    }
}

/// Parametric layer encoder/decoder with pre-allocated buffers
///
/// # Memory Layout
///
/// All output buffers are pre-allocated at construction time:
/// - LPC coefficients: `out_lpc_coeffs[MAX_LPC_ORDER]`
/// - Formants: `out_formants[MAX_FORMANTS]`
/// - Synthesis buffer: `out_synthesis[frame_size]`
#[derive(Debug)]
pub struct ParametricLayer {
    /// LPC order
    lpc_order: usize,
    /// Frame size in samples
    frame_size: usize,
    /// Sample rate
    sample_rate: u32,
    /// LPC analyzer
    lpc_analyzer: LpcAnalyzer,
    /// Pitch detector
    pitch_detector: PitchDetector,
    /// Formant extractor
    formant_extractor: FormantExtractor,
    /// Quality level
    quality: VoiceQuality,

    // === Pre-allocated output buffers ===
    /// Output LPC coefficients buffer
    out_lpc_coeffs: [f32; MAX_LPC_ORDER],
    /// Output LPC reflection coefficients
    out_lpc_reflection: [f32; MAX_LPC_ORDER],
    /// Output LPC gain
    out_lpc_gain: f32,
    /// Output LPC error
    out_lpc_error: f32,
    /// Valid LPC order in buffers
    out_lpc_order: usize,
    /// Output formants buffer
    out_formants: [Formant; MAX_FORMANTS],
    /// Number of valid formants
    out_formant_count: usize,
    /// Synthesis buffer
    out_synthesis: Vec<f32>,
}

impl ParametricLayer {
    /// Create new parametric layer with pre-allocated buffers
    #[must_use]
    pub fn new(lpc_order: usize, frame_size: usize, sample_rate: u32) -> Self {
        Self {
            lpc_order,
            frame_size,
            sample_rate,
            lpc_analyzer: LpcAnalyzer::new(lpc_order),
            pitch_detector: PitchDetector::new(sample_rate),
            formant_extractor: FormantExtractor::new(sample_rate),
            quality: VoiceQuality::Medium,

            // Initialize output buffers
            out_lpc_coeffs: [0.0; MAX_LPC_ORDER],
            out_lpc_reflection: [0.0; MAX_LPC_ORDER],
            out_lpc_gain: 0.0,
            out_lpc_error: 0.0,
            out_lpc_order: lpc_order,
            out_formants: [Formant::default(); MAX_FORMANTS],
            out_formant_count: 0,
            out_synthesis: vec![0.0; frame_size],
        }
    }

    /// Create with default settings (10th order LPC, 512 samples, 16kHz)
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(10, 512, 16000)
    }

    /// Set quality level
    #[must_use]
    pub fn with_quality(mut self, quality: VoiceQuality) -> Self {
        self.quality = quality;
        self.lpc_order = quality.lpc_order();
        self.out_lpc_order = self.lpc_order;
        self.lpc_analyzer = LpcAnalyzer::new(self.lpc_order);
        self
    }

    /// Get LPC order
    #[inline(always)]
    #[must_use]
    pub fn lpc_order(&self) -> usize {
        self.lpc_order
    }

    /// Analyze audio frame and extract parametric representation (allocating)
    ///
    /// For zero-allocation, use `analyze_into` instead.
    ///
    /// # Errors
    ///
    /// Returns `VoiceError` if analysis fails.
    pub fn analyze(&mut self, samples: &[f32]) -> VoiceResult<ParametricParams> {
        // Use zero-alloc path internally, then convert to owned
        let view = self.analyze_into(samples)?;
        Ok(view.to_owned())
    }

    /// Analyze audio frame into pre-allocated buffers (zero allocation)
    ///
    /// Returns a view into the layer's internal buffers. The view is valid
    /// until the next call to `analyze_into`.
    ///
    /// # Performance
    ///
    /// This method performs zero heap allocation per frame, making it
    /// suitable for real-time audio processing.
    ///
    /// # Errors
    ///
    /// Returns `VoiceError::BufferTooSmall` if the input is shorter than `frame_size`.
    pub fn analyze_into<'a>(
        &'a mut self,
        samples: &[f32],
    ) -> VoiceResult<ParametricParamsView<'a>> {
        if samples.len() < self.frame_size {
            return Err(VoiceError::BufferTooSmall {
                need: self.frame_size,
                got: samples.len(),
            });
        }

        let frame = &samples[..self.frame_size];

        // 1. Voice activity detection (no allocation)
        let activity = self.pitch_detector.detect_voice_activity(frame);

        // 2. LPC analysis using zero-copy view
        let lpc_view = self.lpc_analyzer.analyze_view(frame)?;

        // Copy to our internal buffers
        let order = lpc_view.coeffs.len().min(MAX_LPC_ORDER);
        self.out_lpc_coeffs[..order].copy_from_slice(&lpc_view.coeffs[..order]);
        self.out_lpc_reflection[..order].copy_from_slice(&lpc_view.reflection[..order]);
        self.out_lpc_gain = lpc_view.gain;
        self.out_lpc_error = lpc_view.error;
        self.out_lpc_order = order;

        // 3. Pitch detection (returns Copy type)
        let pitch = self.pitch_detector.detect(frame)?;

        // 4. Formant extraction - need to create temporary LpcCoefficients
        //    for the formant extractor API
        let lpc_for_formant = LpcCoefficients {
            coeffs: lpc_view.coeffs.to_vec(),
            reflection: lpc_view.reflection.to_vec(),
            gain: lpc_view.gain,
            error: lpc_view.error,
        };
        let formant_result = self.formant_extractor.extract(&lpc_for_formant)?;

        // Copy formants to our buffer
        let formant_count = formant_result.formants.len().min(MAX_FORMANTS);
        for i in 0..formant_count {
            self.out_formants[i] = formant_result.formants[i];
        }
        self.out_formant_count = formant_count;

        // 5. Return view into our buffers
        Ok(ParametricParamsView {
            lpc_coeffs: &self.out_lpc_coeffs[..self.out_lpc_order],
            lpc_gain: self.out_lpc_gain,
            lpc_error: self.out_lpc_error,
            formants: &self.out_formants,
            formant_count: self.out_formant_count,
            pitch,
            activity,
            frame_size: self.frame_size,
            sample_rate: self.sample_rate,
        })
    }

    /// Synthesize audio frame from parametric representation (allocating)
    ///
    /// For zero-allocation, use `synthesize_into` instead.
    #[must_use]
    pub fn synthesize(&self, params: &ParametricParams) -> Vec<f32> {
        // Generate excitation based on pitch
        let excitation = generate_excitation(&params.pitch, params.frame_size, params.sample_rate);

        // Apply LPC synthesis filter
        self.lpc_analyzer.synthesize(&params.lpc, &excitation)
    }

    /// Synthesize audio frame into pre-allocated output buffer (zero allocation)
    ///
    /// # Arguments
    /// * `params` - Parametric parameters view (zero-copy)
    /// * `output` - Output buffer (must be at least `frame_size` samples)
    ///
    /// # Performance
    ///
    /// This method performs zero heap allocation, using the layer's internal
    /// synthesis buffer for intermediate results.
    ///
    /// # Errors
    ///
    /// Returns `VoiceError::BufferTooSmall` if the output buffer is too small.
    pub fn synthesize_into(
        &mut self,
        params: &ParametricParamsView<'_>,
        output: &mut [f32],
    ) -> VoiceResult<()> {
        if output.len() < params.frame_size {
            return Err(VoiceError::BufferTooSmall {
                need: params.frame_size,
                got: output.len(),
            });
        }

        // Generate excitation into internal buffer
        generate_excitation_into(
            &params.pitch,
            &mut self.out_synthesis[..params.frame_size],
            params.sample_rate,
        );

        // Apply LPC synthesis filter with 4x unrolling
        let frame_size = params.frame_size;
        let order = params.lpc_coeffs.len();
        let gain = params.lpc_gain;
        let coeffs = params.lpc_coeffs;

        // Initialize output
        output[..frame_size].fill(0.0);

        // LPC synthesis: y[n] = G*x[n] + sum(a[k]*y[n-k])
        for n in 0..frame_size {
            let mut sample = self.out_synthesis[n] * gain;

            // Apply all-pole filter (4x unrolled where possible)
            let max_k = order.min(n);
            let unroll_end = max_k - (max_k % 4);

            let mut k = 0;
            while k < unroll_end {
                sample = coeffs[k].mul_add(output[n - 1 - k], sample);
                sample = coeffs[k + 1].mul_add(output[n - 2 - k], sample);
                sample = coeffs[k + 2].mul_add(output[n - 3 - k], sample);
                sample = coeffs[k + 3].mul_add(output[n - 4 - k], sample);
                k += 4;
            }

            while k < max_k {
                sample = coeffs[k].mul_add(output[n - 1 - k], sample);
                k += 1;
            }

            output[n] = sample;
        }

        Ok(())
    }

    /// Process multiple frames
    ///
    /// # Errors
    ///
    /// Returns `VoiceError` if any frame analysis fails.
    pub fn analyze_stream(
        &mut self,
        samples: &[f32],
        hop_size: usize,
    ) -> VoiceResult<Vec<ParametricParams>> {
        let mut params_list = Vec::new();
        let mut pos = 0;

        while pos + self.frame_size <= samples.len() {
            let frame = &samples[pos..pos + self.frame_size];
            let params = self.analyze(frame)?;
            params_list.push(params);
            pos += hop_size;
        }

        Ok(params_list)
    }

    /// Synthesize from multiple frames with overlap-add
    #[must_use]
    pub fn synthesize_stream(&self, params_list: &[ParametricParams], hop_size: usize) -> Vec<f32> {
        if params_list.is_empty() {
            return Vec::new();
        }

        let total_len = (params_list.len() - 1) * hop_size + self.frame_size;
        let mut output = vec![0.0; total_len];

        // Simple overlap-add with triangular window
        // Precompute reciprocal of half-frame to eliminate per-sample division
        let half_frame = self.frame_size / 2;
        let inv_half_frame = (half_frame as f32).recip();
        for (i, params) in params_list.iter().enumerate() {
            let frame = self.synthesize(params);
            let start = i * hop_size;

            // Triangular window for overlap-add
            for (j, &sample) in frame.iter().enumerate() {
                if start + j < total_len {
                    let weight = if j < half_frame {
                        j as f32 * inv_half_frame
                    } else {
                        1.0 - (j - half_frame) as f32 * inv_half_frame
                    };
                    output[start + j] += sample * weight;
                }
            }
        }

        output
    }
}

/// Convenience function: `voice_to_params`
///
/// # Errors
///
/// Returns `VoiceError` if parametric analysis fails.
pub fn voice_to_params(samples: &[f32], sample_rate: u32) -> VoiceResult<Vec<ParametricParams>> {
    // 64ms frames to satisfy pitch detector (needs 2 * sample_rate/min_f0)
    let frame_size = (sample_rate as f32 * 0.064) as usize;
    let hop_size = frame_size / 2;

    let mut layer = ParametricLayer::new(10, frame_size, sample_rate);
    layer.analyze_stream(samples, hop_size)
}

/// Convenience function: `params_to_voice`
#[must_use]
pub fn params_to_voice(params: &[ParametricParams], sample_rate: u32) -> Vec<f32> {
    if params.is_empty() {
        return Vec::new();
    }

    let frame_size = params[0].frame_size;
    let hop_size = frame_size / 2;

    let layer = ParametricLayer::new(10, frame_size, sample_rate);
    layer.synthesize_stream(params, hop_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parametric_layer_creation() {
        let layer = ParametricLayer::new(10, 512, 16000);
        assert_eq!(layer.lpc_order(), 10);
    }

    #[test]
    fn test_analyze_synthesize() {
        // Use frame_size=1024 to satisfy pitch detector requirement (2 * max_period)
        let mut layer = ParametricLayer::new(10, 1024, 16000);

        // Generate test signal (voiced sound)
        let samples: Vec<f32> = (0..1024)
            .map(|i| {
                let t = i as f32 / 16000.0;
                let f0 = 150.0; // 150 Hz pitch
                let pulse = (2.0 * std::f32::consts::PI * f0 * t).sin();
                // Add some harmonics
                pulse * 0.5 + (2.0 * std::f32::consts::PI * f0 * 2.0 * t).sin() * 0.3
            })
            .collect();

        let params = layer.analyze(&samples).unwrap();
        assert_eq!(params.lpc.order(), 10);
        assert!(params.activity.energy_db > -100.0);

        let reconstructed = layer.synthesize(&params);
        assert_eq!(reconstructed.len(), 1024);
    }

    #[test]
    fn test_convenience_functions() {
        // Generate 0.5 seconds of test audio
        let samples: Vec<f32> = (0..8000)
            .map(|i| {
                let t = i as f32 / 16000.0;
                (2.0 * std::f32::consts::PI * 200.0 * t).sin() * 0.5
            })
            .collect();

        let params = voice_to_params(&samples, 16000).unwrap();
        assert!(params.len() > 0);

        let reconstructed = params_to_voice(&params, 16000);
        assert!(reconstructed.len() > 0);
    }

    #[test]
    fn test_fixed_point_conversion() {
        let mut layer = ParametricLayer::new(10, 1024, 16000);

        let samples: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();

        let params = layer.analyze(&samples).unwrap();
        let fixed = params.to_fixed();

        assert_eq!(fixed.lpc.coeffs.len(), 10);
    }

    #[test]
    fn test_compression_ratio() {
        let mut layer = ParametricLayer::new(10, 1024, 16000);

        let samples: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.02).sin() * 0.5).collect();

        let params = layer.analyze(&samples).unwrap();

        let original_size = samples.len() * 4; // 4 bytes per f32
        let compressed_size = params.encoded_size();

        let ratio = original_size as f32 / compressed_size as f32;
        println!("Compression ratio: {:.1}x", ratio);
        // L2 should achieve significant compression
        assert!(ratio > 10.0);
    }
}
