//! L1: Spectral Layer
//!
//! Direct frequency-domain representation using FFT/DCT coefficients.
//! Compatible with ALICE-Streaming-Protocol for unified multimedia handling.
//!
//! # Compression
//!
//! Achieves 10-50x compression by:
//! - DCT transform of audio frames
//! - Quantization of coefficients
//! - Sparse encoding (only non-zero coefficients)
//!
//! # Quality
//!
//! Highest fidelity among the three layers. Suitable for:
//! - Music transmission
//! - Studio-quality voice
//! - Forensic audio
//!
//! # Performance Optimizations
//!
//! - Pre-computed DCT/IDCT matrices (init-time cos calculation)
//! - Pre-allocated workspace buffers (zero per-frame allocation)
//! - 4x loop unrolling for SIMD auto-vectorization
//! - Inplace transforms with buffer reuse

use crate::types::{VoiceResult, VoiceQuality, DEFAULT_FRAME_SIZE, DEFAULT_HOP_SIZE};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

// ============================================
// Constants
// ============================================

/// Maximum supported frame size for stack allocation
const MAX_FRAME_SIZE: usize = 2048;

// ============================================
// SpectralParams (unchanged API)
// ============================================

/// Spectral parameters container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralParams {
    /// DCT/FFT coefficients (sparse representation)
    pub coefficients: Vec<(u16, f32)>, // (index, value)
    /// Frame energy
    pub energy: f32,
    /// Original frame size
    pub frame_size: usize,
    /// Quality level used for encoding
    pub quality: VoiceQuality,
}

impl SpectralParams {
    pub fn new(frame_size: usize) -> Self {
        Self {
            coefficients: Vec::new(),
            energy: 0.0,
            frame_size,
            quality: VoiceQuality::Medium,
        }
    }

    /// Get number of non-zero coefficients
    #[inline(always)]
    pub fn sparsity(&self) -> usize {
        self.coefficients.len()
    }

    /// Estimate encoded size in bytes
    #[inline(always)]
    pub fn encoded_size(&self) -> usize {
        // 2 bytes index + 4 bytes value per coefficient
        4 + self.coefficients.len() * 6
    }
}

// ============================================
// SpectralLayer (Optimized)
// ============================================

/// Spectral layer encoder/decoder with pre-computed tables
///
/// # Memory Layout
///
/// All heavy computation is done at initialization:
/// - DCT matrix: N×N pre-computed cosines
/// - IDCT matrix: N×N pre-computed cosines
/// - Window function: N pre-computed Hann values
///
/// Per-frame operations use pre-allocated workspace buffers.
#[derive(Debug, Clone)]
pub struct SpectralLayer {
    /// Frame size in samples
    frame_size: usize,
    /// Hop size (overlap)
    hop_size: usize,
    /// Quality level
    quality: VoiceQuality,

    // === Pre-computed tables (init-time allocation) ===
    /// DCT-II matrix (row-major, N×N)
    dct_matrix: Vec<f32>,
    /// IDCT matrix (row-major, N×N)
    idct_matrix: Vec<f32>,
    /// Window function (Hann)
    window: Vec<f32>,
    /// Quantization matrix
    quant_matrix: Vec<f32>,

    // === Workspace buffers (reused per frame) ===
    /// Input workspace (windowed samples)
    ws_input: Vec<f32>,
    /// Output workspace (DCT/IDCT result)
    ws_output: Vec<f32>,
    /// Quantized workspace
    ws_quantized: Vec<i32>,
}

impl SpectralLayer {
    /// Create new spectral layer with specified frame and hop size
    ///
    /// Performs all heavy initialization (DCT matrix computation) here.
    pub fn new(frame_size: usize, hop_size: usize) -> Self {
        assert!(frame_size <= MAX_FRAME_SIZE, "Frame size exceeds maximum");

        let mut layer = Self {
            frame_size,
            hop_size,
            quality: VoiceQuality::Medium,

            // Pre-allocate with capacity
            dct_matrix: vec![0.0; frame_size * frame_size],
            idct_matrix: vec![0.0; frame_size * frame_size],
            window: vec![0.0; frame_size],
            quant_matrix: vec![0.0; frame_size],

            // Workspace buffers
            ws_input: vec![0.0; frame_size],
            ws_output: vec![0.0; frame_size],
            ws_quantized: vec![0; frame_size],
        };

        // Initialize all tables
        layer.init_dct_matrix();
        layer.init_idct_matrix();
        layer.init_window();
        layer.init_quant_matrix(50);

        layer
    }

    /// Create with default settings
    pub fn default_config() -> Self {
        Self::new(DEFAULT_FRAME_SIZE, DEFAULT_HOP_SIZE)
    }

    /// Set quality level
    pub fn with_quality(mut self, quality: VoiceQuality) -> Self {
        self.quality = quality;
        let q = match quality {
            VoiceQuality::Low => 25,
            VoiceQuality::Medium => 50,
            VoiceQuality::High => 75,
            VoiceQuality::Ultra => 95,
        };
        self.init_quant_matrix(q);
        self
    }

    // ============================================
    // Table Initialization (Heavy, done once)
    // ============================================

    /// Initialize DCT-II matrix
    ///
    /// Matrix[k,i] = scale(k) * cos(π * k * (2i + 1) / (2N))
    fn init_dct_matrix(&mut self) {
        let n = self.frame_size;
        let scale = (2.0 / n as f32).sqrt();
        let scale0 = 1.0 / (n as f32).sqrt(); // DC component scaling

        for k in 0..n {
            let s = if k == 0 { scale0 } else { scale };
            let row_offset = k * n;

            for i in 0..n {
                let angle = (PI * k as f32 * (2 * i + 1) as f32) / (2 * n) as f32;
                self.dct_matrix[row_offset + i] = s * angle.cos();
            }
        }
    }

    /// Initialize IDCT matrix
    ///
    /// Matrix[i,k] = scale(k) * cos(π * k * (2i + 1) / (2N))
    fn init_idct_matrix(&mut self) {
        let n = self.frame_size;
        let scale = (2.0 / n as f32).sqrt();
        let scale0 = 1.0 / (n as f32).sqrt();

        for i in 0..n {
            let row_offset = i * n;

            for k in 0..n {
                let s = if k == 0 { scale0 } else { scale };
                let angle = (PI * k as f32 * (2 * i + 1) as f32) / (2 * n) as f32;
                self.idct_matrix[row_offset + k] = s * angle.cos();
            }
        }
    }

    /// Initialize Hann window
    fn init_window(&mut self) {
        let n = self.frame_size;
        let scale = 2.0 * PI / (n - 1) as f32;

        for i in 0..n {
            self.window[i] = 0.5 * (1.0 - (scale * i as f32).cos());
        }
    }

    /// Initialize quantization matrix based on quality (1-100)
    fn init_quant_matrix(&mut self, quality: u8) {
        let quality = quality.clamp(1, 100) as f32;
        let scale = if quality < 50.0 {
            5000.0 / quality
        } else {
            200.0 - quality * 2.0
        };

        let n = self.frame_size;
        for i in 0..n {
            // Psychoacoustic: lower frequencies need more precision
            let base = 1.0 + (i as f32 / n as f32) * 15.0;
            self.quant_matrix[i] = (base * scale / 100.0).max(0.1);
        }
    }

    // ============================================
    // Inplace Transforms (Zero Allocation)
    // ============================================

    /// Fast DCT using pre-computed matrix (SGEMV)
    ///
    /// Computes: ws_output = dct_matrix × ws_input
    /// Uses 4x loop unrolling for SIMD auto-vectorization.
    #[inline(always)]
    fn dct_inplace(&mut self) {
        let n = self.frame_size;
        let matrix = &self.dct_matrix;
        let input = &self.ws_input;
        let output = &mut self.ws_output;

        for k in 0..n {
            let row_offset = k * n;
            let mut sum = 0.0f32;

            // 4x unrolled loop for SIMD
            let unroll_end = n - (n % 4);
            let mut i = 0;

            while i < unroll_end {
                sum = input[i].mul_add(matrix[row_offset + i], sum);
                sum = input[i + 1].mul_add(matrix[row_offset + i + 1], sum);
                sum = input[i + 2].mul_add(matrix[row_offset + i + 2], sum);
                sum = input[i + 3].mul_add(matrix[row_offset + i + 3], sum);
                i += 4;
            }

            // Handle remainder
            while i < n {
                sum = input[i].mul_add(matrix[row_offset + i], sum);
                i += 1;
            }

            output[k] = sum;
        }
    }

    /// Fast IDCT using pre-computed matrix (SGEMV)
    ///
    /// Computes: ws_output = idct_matrix × ws_input
    #[inline(always)]
    fn idct_inplace(&mut self) {
        let n = self.frame_size;
        let matrix = &self.idct_matrix;
        let input = &self.ws_input;
        let output = &mut self.ws_output;

        for i in 0..n {
            let row_offset = i * n;
            let mut sum = 0.0f32;

            // 4x unrolled loop
            let unroll_end = n - (n % 4);
            let mut k = 0;

            while k < unroll_end {
                sum = input[k].mul_add(matrix[row_offset + k], sum);
                sum = input[k + 1].mul_add(matrix[row_offset + k + 1], sum);
                sum = input[k + 2].mul_add(matrix[row_offset + k + 2], sum);
                sum = input[k + 3].mul_add(matrix[row_offset + k + 3], sum);
                k += 4;
            }

            while k < n {
                sum = input[k].mul_add(matrix[row_offset + k], sum);
                k += 1;
            }

            output[i] = sum;
        }
    }

    /// Quantize coefficients inplace
    ///
    /// ws_quantized[i] = round(ws_output[i] / quant_matrix[i])
    /// Uses reciprocal multiplication to eliminate repeated float division.
    #[inline(always)]
    fn quantize_inplace(&mut self) {
        let n = self.frame_size;

        for i in 0..n {
            let inv_q = self.quant_matrix[i].recip();
            self.ws_quantized[i] = (self.ws_output[i] * inv_q).round() as i32;
        }
    }

    /// Dequantize coefficients inplace
    ///
    /// ws_input[i] = ws_quantized[i] * quant_matrix[i]
    #[inline(always)]
    fn dequantize_inplace(&mut self) {
        let n = self.frame_size;

        for i in 0..n {
            self.ws_input[i] = self.ws_quantized[i] as f32 * self.quant_matrix[i];
        }
    }

    // ============================================
    // Public API (Backwards Compatible)
    // ============================================

    /// Analyze audio frame and extract spectral parameters
    ///
    /// Uses pre-allocated workspace buffers internally.
    pub fn analyze(&mut self, samples: &[f32]) -> VoiceResult<SpectralParams> {
        if samples.len() < self.frame_size {
            return Err(crate::types::VoiceError::BufferTooSmall {
                need: self.frame_size,
                got: samples.len(),
            });
        }

        let n = self.frame_size;

        // 1. Apply window and copy to ws_input
        let mut energy = 0.0f32;
        for i in 0..n {
            let windowed = samples[i] * self.window[i];
            self.ws_input[i] = windowed;
            energy = windowed.mul_add(windowed, energy);
        }
        let inv_n = (n as f32).recip();
        energy *= inv_n;

        // 2. DCT (ws_input → ws_output)
        self.dct_inplace();

        // 3. Quantize (ws_output → ws_quantized)
        self.quantize_inplace();

        // 4. Sparse encoding (only non-zero coefficients)
        let coefficients: Vec<(u16, f32)> = self.ws_quantized
            .iter()
            .take(n)
            .enumerate()
            .filter(|(_, &c)| c != 0)
            .map(|(i, &c)| (i as u16, c as f32))
            .collect();

        Ok(SpectralParams {
            coefficients,
            energy,
            frame_size: n,
            quality: self.quality,
        })
    }

    /// Synthesize audio frame from spectral parameters
    ///
    /// Uses pre-allocated workspace buffers internally.
    pub fn synthesize(&mut self, params: &SpectralParams) -> Vec<f32> {
        let n = params.frame_size.min(self.frame_size);

        // 1. Clear and reconstruct quantized array
        for i in 0..self.frame_size {
            self.ws_quantized[i] = 0;
        }
        for &(idx, val) in &params.coefficients {
            if (idx as usize) < n {
                self.ws_quantized[idx as usize] = val as i32;
            }
        }

        // 2. Dequantize (ws_quantized → ws_input)
        self.dequantize_inplace();

        // 3. IDCT (ws_input → ws_output)
        self.idct_inplace();

        // 4. Apply synthesis window and return
        //    (output Vec is unavoidable for API compatibility)
        let mut output = vec![0.0f32; n];
        for i in 0..n {
            output[i] = self.ws_output[i] * self.window[i];
        }

        output
    }

    /// Synthesize into pre-allocated buffer (zero-allocation)
    ///
    /// # Returns
    /// Number of samples written
    #[inline(always)]
    pub fn synthesize_into(&mut self, params: &SpectralParams, output: &mut [f32]) -> usize {
        let n = params.frame_size.min(self.frame_size).min(output.len());

        // 1. Clear and reconstruct quantized array
        for i in 0..self.frame_size {
            self.ws_quantized[i] = 0;
        }
        for &(idx, val) in &params.coefficients {
            if (idx as usize) < n {
                self.ws_quantized[idx as usize] = val as i32;
            }
        }

        // 2. Dequantize (ws_quantized → ws_input)
        self.dequantize_inplace();

        // 3. IDCT (ws_input → ws_output)
        self.idct_inplace();

        // 4. Apply synthesis window to output buffer
        for i in 0..n {
            output[i] = self.ws_output[i] * self.window[i];
        }

        n
    }

    /// Process multiple frames with overlap-add
    pub fn analyze_stream(&mut self, samples: &[f32]) -> VoiceResult<Vec<SpectralParams>> {
        let mut params_list = Vec::new();
        let mut pos = 0;

        while pos + self.frame_size <= samples.len() {
            let frame = &samples[pos..pos + self.frame_size];
            let params = self.analyze(frame)?;
            params_list.push(params);
            pos += self.hop_size;
        }

        Ok(params_list)
    }

    /// Synthesize from multiple frames with overlap-add
    pub fn synthesize_stream(&mut self, params_list: &[SpectralParams]) -> Vec<f32> {
        if params_list.is_empty() {
            return Vec::new();
        }

        let total_len = (params_list.len() - 1) * self.hop_size + self.frame_size;
        let mut output = vec![0.0; total_len];
        let mut normalization = vec![0.0; total_len];

        // Pre-allocate frame buffer for zero-alloc inner loop
        let mut frame_buffer = vec![0.0f32; self.frame_size];

        for (i, params) in params_list.iter().enumerate() {
            let written = self.synthesize_into(params, &mut frame_buffer);
            let start = i * self.hop_size;

            for j in 0..written {
                if start + j < total_len {
                    output[start + j] += frame_buffer[j];
                    normalization[start + j] += self.window[j];
                }
            }
        }

        // Normalize by window overlap using reciprocal multiplication
        for i in 0..total_len {
            if normalization[i] > 1e-10 {
                output[i] *= normalization[i].recip();
            }
        }

        output
    }

    // ============================================
    // Legacy API (for backwards compatibility)
    // ============================================

    /// 1D DCT-II transform (allocating version for testing)
    #[allow(dead_code)]
    fn dct(&self, input: &[f32]) -> Vec<f32> {
        let n = input.len().min(self.frame_size);
        let mut output = vec![0.0; n];

        for k in 0..n {
            let row_offset = k * self.frame_size;
            let mut sum = 0.0f32;
            for i in 0..n {
                sum += input[i] * self.dct_matrix[row_offset + i];
            }
            output[k] = sum;
        }

        output
    }

    /// 1D Inverse DCT-II transform (allocating version for testing)
    #[allow(dead_code)]
    fn idct(&self, input: &[f32]) -> Vec<f32> {
        let n = input.len().min(self.frame_size);
        let mut output = vec![0.0; n];

        for i in 0..n {
            let row_offset = i * self.frame_size;
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += input[k] * self.idct_matrix[row_offset + k];
            }
            output[i] = sum;
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_layer_creation() {
        let layer = SpectralLayer::new(512, 256);
        assert_eq!(layer.frame_size, 512);
        assert_eq!(layer.hop_size, 256);
        // Verify matrices are allocated
        assert_eq!(layer.dct_matrix.len(), 512 * 512);
        assert_eq!(layer.idct_matrix.len(), 512 * 512);
    }

    #[test]
    fn test_dct_idct_roundtrip() {
        let layer = SpectralLayer::new(64, 32);

        let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let dct = layer.dct(&input);
        let output = layer.idct(&dct);

        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 0.01, "DCT roundtrip failed: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_analyze_synthesize() {
        let mut layer = SpectralLayer::new(256, 128).with_quality(VoiceQuality::High);

        // Generate test signal
        let samples: Vec<f32> = (0..256)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();

        let params = layer.analyze(&samples).unwrap();
        assert!(params.coefficients.len() > 0);
        assert!(params.energy > 0.0);

        let reconstructed = layer.synthesize(&params);
        assert_eq!(reconstructed.len(), 256);
    }

    #[test]
    fn test_synthesize_into() {
        let mut layer = SpectralLayer::new(256, 128);

        let samples: Vec<f32> = (0..256)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();

        let params = layer.analyze(&samples).unwrap();

        // Use pre-allocated buffer
        let mut output = vec![0.0f32; 256];
        let written = layer.synthesize_into(&params, &mut output);

        assert_eq!(written, 256);
        assert!(output.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_stream_processing() {
        let mut layer = SpectralLayer::new(512, 256);

        // Generate 2 seconds of audio at 16kHz
        let samples: Vec<f32> = (0..32000)
            .map(|i| (2.0 * PI * 300.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();

        let params_list = layer.analyze_stream(&samples).unwrap();
        assert!(params_list.len() > 0);

        let reconstructed = layer.synthesize_stream(&params_list);
        assert!(reconstructed.len() > 0);
    }

    #[test]
    fn test_compression_ratio() {
        let mut layer = SpectralLayer::new(512, 256);

        let samples: Vec<f32> = (0..512)
            .map(|i| (2.0 * PI * 200.0 * i as f32 / 16000.0).sin())
            .collect();

        let params = layer.analyze(&samples).unwrap();

        let original_size = samples.len() * 4; // 4 bytes per f32
        let compressed_size = params.encoded_size();

        let ratio = original_size as f32 / compressed_size as f32;
        // Should achieve some compression
        assert!(ratio > 1.0);
    }
}
