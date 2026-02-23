//! Linear Predictive Coding (LPC) - Ultra Optimized
//!
//! "カリッカリ" Tuning Edition
//! - Zero Allocation in hot paths
//! - Unsafe pointer arithmetic for max throughput
//! - FMA (Fused Multiply-Add) utilization
//! - Loop unrolling for pipeline saturation
//!
//! # Safety
//!
//! This module uses `unsafe` for performance-critical paths.
//! All unsafe code is carefully bounded and verified.
//!
//! # Example
//!
//! ```ignore
//! use alice_voice::codec::lpc::LpcAnalyzer;
//!
//! let mut analyzer = LpcAnalyzer::new(10);
//! let view = analyzer.analyze_view(&audio_frame)?;
//! // Use view.coeffs, view.gain directly (zero-copy)
//! let owned = view.to_owned(); // Only allocate when needed
//! ```

use crate::types::{VoiceError, VoiceResult, Q16_ONE, Q16_SHIFT};
use serde::{Deserialize, Serialize};

// =============================================================================
// Data Structures (Memory Layout Optimized)
// =============================================================================

/// LPC coefficients container (Owned version for serialization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpcCoefficients {
    /// LPC coefficients (`a[1]` to `a[order]`)
    pub coeffs: Vec<f32>,
    /// Prediction gain
    pub gain: f32,
    /// Reflection coefficients (PARCOR)
    pub reflection: Vec<f32>,
    /// Prediction error
    pub error: f32,
}

impl LpcCoefficients {
    pub fn new(order: usize) -> Self {
        Self {
            coeffs: vec![0.0; order],
            gain: 1.0,
            reflection: vec![0.0; order],
            error: 0.0,
        }
    }

    pub fn order(&self) -> usize {
        self.coeffs.len()
    }

    /// Convert to Q16.16 fixed-point format
    pub fn to_fixed(&self) -> LpcCoefficientsFixed {
        LpcCoefficientsFixed {
            coeffs: self
                .coeffs
                .iter()
                .map(|&c| (c * Q16_ONE as f32) as i32)
                .collect(),
            gain: (self.gain * Q16_ONE as f32) as i32,
        }
    }
}

/// Lightweight View for Zero-Copy Access
///
/// Returns a reference to the analyzer's internal buffers.
/// No allocation occurs. Call `to_owned()` if you need to store the result.
#[derive(Debug, Clone, Copy)]
pub struct LpcCoefficientsView<'a> {
    /// LPC coefficients (`a[1]` to `a[order]`)
    pub coeffs: &'a [f32],
    /// Reflection coefficients (PARCOR)
    pub reflection: &'a [f32],
    /// Prediction gain
    pub gain: f32,
    /// Prediction error
    pub error: f32,
}

impl<'a> LpcCoefficientsView<'a> {
    /// Convert to owned LpcCoefficients (allocates)
    pub fn to_owned(&self) -> LpcCoefficients {
        LpcCoefficients {
            coeffs: self.coeffs.to_vec(),
            gain: self.gain,
            reflection: self.reflection.to_vec(),
            error: self.error,
        }
    }

    pub fn order(&self) -> usize {
        self.coeffs.len()
    }
}

/// LPC coefficients in Q16.16 fixed-point (ALICE-Edge compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpcCoefficientsFixed {
    /// LPC coefficients in Q16.16
    pub coeffs: Vec<i32>,
    /// Gain in Q16.16
    pub gain: i32,
}

impl LpcCoefficientsFixed {
    /// Convert back to floating-point
    pub fn to_float(&self) -> LpcCoefficients {
        LpcCoefficients {
            coeffs: self
                .coeffs
                .iter()
                .map(|&c| c as f32 / Q16_ONE as f32)
                .collect(),
            gain: self.gain as f32 / Q16_ONE as f32,
            reflection: vec![],
            error: 0.0,
        }
    }
}

// =============================================================================
// The Analyzer - Ultra Optimized
// =============================================================================

/// LPC Analyzer and Synthesizer
///
/// # Performance Characteristics
///
/// - **Zero-allocation**: All buffers pre-allocated at construction
/// - **Unsafe optimized**: Bounds checks eliminated in hot paths
/// - **FMA enabled**: Uses `mul_add` for fused multiply-add
/// - **Loop unrolled**: 4x unrolling for autocorrelation
#[derive(Debug, Clone)]
pub struct LpcAnalyzer {
    /// LPC order (number of coefficients)
    order: usize,
    /// Pre-emphasis coefficient
    preemph: f32,
    /// Frame size for buffer allocation
    frame_size: usize,

    // Pre-computed window (Hamming)
    window: Vec<f32>,

    // Workspace: combined pre-emphasis + windowed signal
    ws_signal: Vec<f32>,

    // Workspace: autocorrelation
    ws_autocorr: Vec<f32>,

    // Output buffers
    out_coeffs: Vec<f32>,
    out_reflection: Vec<f32>,

    // Levinson-Durbin temporary
    ws_lev_tmp: Vec<f32>,
}

impl LpcAnalyzer {
    /// Create new LPC analyzer with specified order
    ///
    /// Uses default frame size of 1024 samples (64ms @ 16kHz)
    pub fn new(order: usize) -> Self {
        Self::with_frame_size(order, 1024)
    }

    /// Create analyzer with explicit frame size for optimal buffer allocation
    pub fn with_frame_size(order: usize, frame_size: usize) -> Self {
        // Pre-compute Hamming window
        let window: Vec<f32> = (0..frame_size)
            .map(|i| {
                0.54 - 0.46
                    * (2.0 * std::f32::consts::PI * i as f32 / (frame_size - 1) as f32).cos()
            })
            .collect();

        Self {
            order,
            preemph: 0.97,
            frame_size,
            window,
            ws_signal: vec![0.0; frame_size],
            ws_autocorr: vec![0.0; order + 1],
            out_coeffs: vec![0.0; order],
            out_reflection: vec![0.0; order],
            ws_lev_tmp: vec![0.0; order],
        }
    }

    /// Set pre-emphasis coefficient
    pub fn with_preemph(mut self, coeff: f32) -> Self {
        self.preemph = coeff;
        self
    }

    /// Get LPC order
    pub fn order(&self) -> usize {
        self.order
    }

    /// Resize buffers if frame size changed
    fn resize_buffers(&mut self, n: usize) {
        self.frame_size = n;
        self.window = (0..n)
            .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos())
            .collect();
        self.ws_signal.resize(n, 0.0);
    }

    // =========================================================================
    // Public API
    // =========================================================================

    /// Analyze audio frame and return a zero-copy view
    ///
    /// # Zero-Allocation Hot Path
    ///
    /// This method performs NO heap allocation. The returned view
    /// references the analyzer's internal buffers directly.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let view = analyzer.analyze_view(&samples)?;
    /// process(view.coeffs, view.gain); // Direct access
    /// let owned = view.to_owned();     // Allocate only when needed
    /// ```
    pub fn analyze_view(&mut self, samples: &[f32]) -> VoiceResult<LpcCoefficientsView<'_>> {
        let n = samples.len();
        if n < self.order * 2 {
            return Err(VoiceError::BufferTooSmall {
                need: self.order * 2,
                got: n,
            });
        }

        // Resize buffers if needed (rare path)
        if self.frame_size != n {
            self.resize_buffers(n);
        }

        // 1. Fused Pre-emphasis & Windowing (unsafe, no bounds check)
        // SAFETY: Buffer sizes verified above
        unsafe { self.apply_preemph_and_window_unchecked(samples) };

        // 2. Autocorrelation (the heaviest part - 60-80% of compute time)
        // SAFETY: Buffer sizes verified above
        unsafe { self.autocorrelation_unchecked() };

        // 3. Levinson-Durbin
        // SAFETY: Buffer sizes verified above
        let (gain, error) = unsafe { self.levinson_durbin_unchecked()? };

        Ok(LpcCoefficientsView {
            coeffs: &self.out_coeffs,
            reflection: &self.out_reflection,
            gain,
            error,
        })
    }

    /// Analyze audio frame and return owned coefficients
    ///
    /// Convenience wrapper that allocates. Use `analyze_view` for hot paths.
    pub fn analyze(&mut self, samples: &[f32]) -> VoiceResult<LpcCoefficients> {
        let view = self.analyze_view(samples)?;
        Ok(view.to_owned())
    }

    /// Analyze audio frame, writing directly to provided output buffer
    ///
    /// # True Zero-Allocation
    ///
    /// This method performs NO heap allocation. Caller provides output buffer.
    pub fn analyze_into(
        &mut self,
        samples: &[f32],
        output: &mut LpcCoefficients,
    ) -> VoiceResult<()> {
        let view = self.analyze_view(samples)?;

        output.coeffs.clear();
        output.coeffs.extend_from_slice(view.coeffs);
        output.gain = view.gain;
        output.reflection.clear();
        output.reflection.extend_from_slice(view.reflection);
        output.error = view.error;

        Ok(())
    }

    /// Synthesize audio from LPC coefficients and excitation signal
    pub fn synthesize(&self, coeffs: &LpcCoefficients, excitation: &[f32]) -> Vec<f32> {
        let n = excitation.len();
        let order = coeffs.order();
        let mut output = vec![0.0; n];

        for i in 0..n {
            let mut sample = coeffs.gain * excitation[i];

            for j in 0..order.min(i) {
                sample = coeffs.coeffs[j].mul_add(output[i - 1 - j], sample);
            }

            output[i] = sample;
        }

        // De-emphasis
        for i in 1..n {
            output[i] = self.preemph.mul_add(output[i - 1], output[i]);
        }

        output
    }

    /// Synthesize using View (no allocation for coefficients)
    pub fn synthesize_view(&self, view: &LpcCoefficientsView<'_>, excitation: &[f32]) -> Vec<f32> {
        let n = excitation.len();
        let order = view.order();
        let mut output = vec![0.0; n];

        for i in 0..n {
            let mut sample = view.gain * excitation[i];

            for j in 0..order.min(i) {
                sample = view.coeffs[j].mul_add(output[i - 1 - j], sample);
            }

            output[i] = sample;
        }

        // De-emphasis
        for i in 1..n {
            output[i] = self.preemph.mul_add(output[i - 1], output[i]);
        }

        output
    }

    // =========================================================================
    // "Crispy" Implementations (Unsafe/Optimized)
    // =========================================================================

    /// Fused Pre-emphasis and Windowing
    ///
    /// Combines two operations into one pass, halving memory bandwidth.
    /// Uses FMA instructions where available.
    ///
    /// # Safety
    ///
    /// Caller must ensure `samples.len() == self.frame_size`
    #[inline]
    unsafe fn apply_preemph_and_window_unchecked(&mut self, samples: &[f32]) {
        let n = samples.len();
        let signal_ptr = self.ws_signal.as_mut_ptr();
        let sample_ptr = samples.as_ptr();
        let window_ptr = self.window.as_ptr();
        let preemph = self.preemph;

        // First sample (special case: no previous sample)
        *signal_ptr = *sample_ptr * *window_ptr;

        // Remaining samples: signal[i] = (samples[i] - preemph * samples[i-1]) * window[i]
        // Using FMA: filtered = curr + (-preemph * prev)
        for i in 1..n {
            let curr = *sample_ptr.add(i);
            let prev = *sample_ptr.add(i - 1);
            let win = *window_ptr.add(i);

            // FMA: (-preemph).mul_add(prev, curr) = curr - preemph * prev
            let filtered = (-preemph).mul_add(prev, curr);
            *signal_ptr.add(i) = filtered * win;
        }
    }

    /// Ultra-fast Autocorrelation
    ///
    /// This is THE bottleneck (60-80% of total compute time).
    /// Optimizations:
    /// - Pointer arithmetic (no bounds check)
    /// - FMA instructions
    /// - 4x loop unrolling for pipeline saturation
    ///
    /// # Safety
    ///
    /// Caller must ensure buffers are properly sized.
    #[inline]
    unsafe fn autocorrelation_unchecked(&mut self) {
        let n = self.ws_signal.len();
        let order = self.order;

        let sig_ptr = self.ws_signal.as_ptr();
        let r_ptr = self.ws_autocorr.as_mut_ptr();

        // Lag 0: Energy = sum(s[i]^2)
        let mut energy = 0.0f32;
        for i in 0..n {
            let s = *sig_ptr.add(i);
            energy = s.mul_add(s, energy); // energy += s * s
        }
        *r_ptr = energy;

        // Lags 1..=order: r[lag] = sum(s[i] * s[i+lag])
        for lag in 1..=order {
            let mut sum = 0.0f32;
            let len = n - lag;

            // Pointers for the two offset arrays
            let p1 = sig_ptr; // samples[i]
            let p2 = sig_ptr.add(lag); // samples[i + lag]

            // 4x loop unrolling for better instruction-level parallelism
            let mut i = 0;
            let unroll_end = len.saturating_sub(3);

            while i < unroll_end {
                let s1_0 = *p1.add(i);
                let s2_0 = *p2.add(i);
                sum = s1_0.mul_add(s2_0, sum);

                let s1_1 = *p1.add(i + 1);
                let s2_1 = *p2.add(i + 1);
                sum = s1_1.mul_add(s2_1, sum);

                let s1_2 = *p1.add(i + 2);
                let s2_2 = *p2.add(i + 2);
                sum = s1_2.mul_add(s2_2, sum);

                let s1_3 = *p1.add(i + 3);
                let s2_3 = *p2.add(i + 3);
                sum = s1_3.mul_add(s2_3, sum);

                i += 4;
            }

            // Handle remainder
            while i < len {
                let s1 = *p1.add(i);
                let s2 = *p2.add(i);
                sum = s1.mul_add(s2, sum);
                i += 1;
            }

            *r_ptr.add(lag) = sum;
        }
    }

    /// Levinson-Durbin algorithm using unsafe pointer operations
    ///
    /// Uses `copy_nonoverlapping` for fast coefficient updates.
    ///
    /// # Safety
    ///
    /// Caller must ensure buffers are properly sized.
    #[inline]
    unsafe fn levinson_durbin_unchecked(&mut self) -> VoiceResult<(f32, f32)> {
        let order = self.order;
        let r_ptr = self.ws_autocorr.as_ptr();
        let a_ptr = self.out_coeffs.as_mut_ptr();
        let k_ptr = self.out_reflection.as_mut_ptr();
        let tmp_ptr = self.ws_lev_tmp.as_mut_ptr();

        // Get initial error (r[0] = energy)
        let mut error = *r_ptr;

        // Zero energy check
        if error.abs() < 1e-10 {
            std::ptr::write_bytes(a_ptr, 0, order);
            std::ptr::write_bytes(k_ptr, 0, order);
            return Ok((0.0, 0.0));
        }

        // Initialize coefficients to 0
        std::ptr::write_bytes(a_ptr, 0, order);

        for i in 0..order {
            // Compute reflection coefficient (k)
            // sum = r[i+1] - sum(a[j] * r[i-j])
            let mut sum = *r_ptr.add(i + 1);

            for j in 0..i {
                let a_val = *a_ptr.add(j);
                let r_val = *r_ptr.add(i - j);
                // sum -= a[j] * r[i-j]
                sum = (-a_val).mul_add(r_val, sum);
            }

            let k = sum / error;
            *k_ptr.add(i) = k;

            // Update coefficients: tmp[j] = a[j] - k * a[i-1-j]
            for j in 0..i {
                let a_j = *a_ptr.add(j);
                let a_inv = *a_ptr.add(i - 1 - j);
                *tmp_ptr.add(j) = (-k).mul_add(a_inv, a_j);
            }

            // a[i] = k
            *a_ptr.add(i) = k;

            // Copy tmp back to a (fast memcpy for small arrays)
            if i > 0 {
                std::ptr::copy_nonoverlapping(tmp_ptr, a_ptr, i);
            }

            // Update error: error *= (1 - k^2)
            error *= 1.0 - k * k;

            // Stability check
            if error <= 0.0 {
                return Err(VoiceError::LpcError("Unstable filter".into()));
            }
        }

        Ok((error.sqrt(), error))
    }
}

// =============================================================================
// Fixed-Point LPC (ALICE-Edge Compatible)
// =============================================================================

/// Fixed-point LPC analysis using Q16.16 arithmetic
///
/// Optimized for ARM Cortex-M and similar embedded targets.
pub mod lpc_fixed {
    use super::*;

    /// Compute autocorrelation in fixed-point
    ///
    /// Uses SIMD-friendly patterns for ARM NEON auto-vectorization.
    #[inline(always)]
    pub fn autocorrelation_fixed(samples: &[i32], order: usize) -> Vec<i64> {
        let n = samples.len();
        let mut r = vec![0i64; order + 1];

        for lag in 0..=order {
            let sum: i64 = samples[lag..n]
                .iter()
                .zip(samples[..n - lag].iter())
                .map(|(&a, &b)| a as i64 * b as i64)
                .sum();
            r[lag] = sum;
        }

        r
    }

    /// Fast Q16.16 multiply (ARM-optimized)
    #[inline(always)]
    #[cfg(target_arch = "aarch64")]
    pub fn q16_mul(a: i32, b: i32) -> i32 {
        // On ARM64, compiles to SMULL + ASR
        ((a as i64 * b as i64) >> Q16_SHIFT) as i32
    }

    /// Fast Q16.16 multiply (fallback)
    #[inline(always)]
    #[cfg(not(target_arch = "aarch64"))]
    pub fn q16_mul(a: i32, b: i32) -> i32 {
        ((a as i64 * b as i64) >> Q16_SHIFT) as i32
    }

    /// Fast Q16.16 multiply-accumulate (ARM-optimized)
    #[inline(always)]
    #[cfg(target_arch = "aarch64")]
    pub fn q16_mac(acc: i32, a: i32, b: i32) -> i32 {
        // On ARM64, compiles to SMLAL
        acc + ((a as i64 * b as i64) >> Q16_SHIFT) as i32
    }

    /// Fast Q16.16 multiply-accumulate (fallback)
    #[inline(always)]
    #[cfg(not(target_arch = "aarch64"))]
    pub fn q16_mac(acc: i32, a: i32, b: i32) -> i32 {
        acc + ((a as i64 * b as i64) >> Q16_SHIFT) as i32
    }

    /// Fast integer square root (Newton-Raphson)
    ///
    /// For embedded systems without FPU.
    #[inline(always)]
    pub fn fast_isqrt(n: i64) -> i32 {
        if n <= 0 {
            return 0;
        }
        if n < 4 {
            return 1;
        }

        // Initial guess using bit manipulation
        let shift = (63 - n.leading_zeros()) / 2;
        let mut x = 1i64 << shift;

        // 4 iterations of Newton-Raphson
        for _ in 0..4 {
            x = (x + n / x) >> 1;
        }

        x as i32
    }

    /// Levinson-Durbin in fixed-point
    pub fn levinson_durbin_fixed(r: &[i64], order: usize) -> VoiceResult<LpcCoefficientsFixed> {
        let mut coeffs = vec![0i32; order];
        let mut tmp = vec![0i64; order];

        if r[0] == 0 {
            return Ok(LpcCoefficientsFixed {
                coeffs: vec![0; order],
                gain: 0,
            });
        }

        let mut error = r[0];

        for i in 0..order {
            let mut sum = r[i + 1];
            for j in 0..i {
                sum -= (coeffs[j] as i64 * r[i - j]) >> Q16_SHIFT;
            }

            let k = if error != 0 {
                ((sum << Q16_SHIFT) / error) as i32
            } else {
                0
            };

            coeffs[i] = k;
            for j in 0..i {
                tmp[j] = coeffs[j] as i64 - q16_mul(k, coeffs[i - 1 - j]) as i64;
            }
            for j in 0..i {
                coeffs[j] = tmp[j] as i32;
            }

            let k_sq = q16_mul(k, k) as i64;
            error = error - ((k_sq * error) >> Q16_SHIFT);

            if error <= 0 {
                break;
            }
        }

        let gain = if error > 0 { fast_isqrt(error) } else { 0 };

        Ok(LpcCoefficientsFixed { coeffs, gain })
    }

    /// Synthesize audio from fixed-point LPC coefficients
    pub fn synthesize_fixed(coeffs: &LpcCoefficientsFixed, excitation: &[i32]) -> Vec<i32> {
        let n = excitation.len();
        let order = coeffs.coeffs.len();
        let mut output = vec![0i32; n];

        for i in 0..n {
            let mut sample = q16_mul(coeffs.gain, excitation[i]);

            for j in 0..order.min(i) {
                sample = q16_mac(sample, coeffs.coeffs[j], output[i - 1 - j]);
            }

            output[i] = sample;
        }

        output
    }

    /// Zero-allocation synthesize into pre-allocated buffer
    #[inline(always)]
    pub fn synthesize_fixed_into(
        coeffs: &LpcCoefficientsFixed,
        excitation: &[i32],
        output: &mut [i32],
    ) {
        let n = excitation.len().min(output.len());
        let order = coeffs.coeffs.len();

        for i in 0..n {
            let mut sample = q16_mul(coeffs.gain, excitation[i]);

            for j in 0..order.min(i) {
                sample = q16_mac(sample, coeffs.coeffs[j], output[i - 1 - j]);
            }

            output[i] = sample;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lpc_analysis_synthesis() {
        let mut analyzer = LpcAnalyzer::new(10);

        // Generate test signal (simple sine wave)
        let samples: Vec<f32> = (0..512)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();

        let coeffs = analyzer.analyze(&samples).unwrap();
        assert_eq!(coeffs.order(), 10);
        assert!(coeffs.gain > 0.0);

        // Generate excitation
        let excitation: Vec<f32> = (0..512)
            .map(|i| {
                ((i as u64).wrapping_mul(1103515245).wrapping_add(12345) % 100) as f32 / 50.0 - 1.0
            })
            .collect();

        let reconstructed = analyzer.synthesize(&coeffs, &excitation);
        assert_eq!(reconstructed.len(), 512);
    }

    #[test]
    fn test_analyze_view_zero_copy() {
        let mut analyzer = LpcAnalyzer::new(10);

        let samples: Vec<f32> = (0..512)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 16000.0).sin())
            .collect();

        // analyze_view returns a reference - no allocation
        let view = analyzer.analyze_view(&samples).unwrap();
        assert_eq!(view.order(), 10);
        assert!(view.gain > 0.0);

        // Only allocate when explicitly requested
        let owned = view.to_owned();
        assert_eq!(owned.coeffs.len(), 10);
    }

    #[test]
    fn test_lpc_fixed_point() {
        use lpc_fixed::*;

        let samples: Vec<i32> = vec![100, 200, 150, 180, 120, 160, 140, 170];
        let r = autocorrelation_fixed(&samples, 4);
        assert_eq!(r.len(), 5);
        assert!(r[0] > 0);

        let coeffs = levinson_durbin_fixed(&r, 4).unwrap();
        assert_eq!(coeffs.coeffs.len(), 4);
    }

    #[test]
    fn test_coefficient_conversion() {
        let float_coeffs = LpcCoefficients {
            coeffs: vec![0.5, -0.3, 0.1],
            gain: 0.8,
            reflection: vec![],
            error: 0.0,
        };

        let fixed = float_coeffs.to_fixed();
        let back = fixed.to_float();

        for (a, b) in float_coeffs.coeffs.iter().zip(back.coeffs.iter()) {
            assert!((a - b).abs() < 0.001);
        }
    }

    #[test]
    fn test_fast_isqrt() {
        use lpc_fixed::fast_isqrt;

        assert_eq!(fast_isqrt(0), 0);
        assert_eq!(fast_isqrt(1), 1);
        assert_eq!(fast_isqrt(4), 2);
        assert_eq!(fast_isqrt(9), 3);
        assert_eq!(fast_isqrt(100), 10);
        assert_eq!(fast_isqrt(10000), 100);

        // Check accuracy for larger values
        let n = 1_000_000i64;
        let sqrt = fast_isqrt(n);
        assert!((sqrt - 1000).abs() <= 1);
    }
}
