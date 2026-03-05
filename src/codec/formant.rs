// DSP loops access multiple arrays at the same index — standard signal processing pattern.
#![allow(clippy::needless_range_loop)]
//! Formant Extraction - Ultra Optimized ("Crispy" Edition)
//!
//! Tunings:
//! - Zero Heap Allocation (Uses stack buffers for roots)
//! - Fast Approx Math (atan2, sqrt)
//! - `SoA` (Structure of Arrays) layout for SIMD auto-vectorization
//! - Insertion Sort for small arrays
//!
//! # Performance
//!
//! The Durand-Kerner root finder runs entirely on the stack with fixed-size
//! arrays. This eliminates all heap allocation in the hot path and enables
//! aggressive compiler optimization.

use crate::codec::lpc::LpcCoefficients;
use crate::types::VoiceResult;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

// =============================================================================
// Constants
// =============================================================================

/// Maximum supported LPC order for stack allocation.
/// Standard voice LPC is 10-16, so 32 is plenty safe.
const MAX_LPC_ORDER: usize = 32;

/// Maximum Durand-Kerner iterations (reduced for speed)
const MAX_DK_ITERATIONS: usize = 20;

/// Convergence threshold (relaxed for speed)
const DK_EPSILON: f32 = 1e-4;

// =============================================================================
// Data Types
// =============================================================================

/// Single formant with frequency and bandwidth
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct Formant {
    /// Formant frequency in Hz
    pub frequency: f32,
    /// Bandwidth in Hz
    pub bandwidth: f32,
    /// Amplitude (relative)
    pub amplitude: f32,
}

impl Formant {
    #[must_use]
    pub const fn new(frequency: f32, bandwidth: f32) -> Self {
        Self {
            frequency,
            bandwidth,
            amplitude: 1.0,
        }
    }

    #[must_use]
    pub const fn with_amplitude(mut self, amplitude: f32) -> Self {
        self.amplitude = amplitude;
        self
    }
}

/// Formant extraction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormantResult {
    /// Extracted formants (F1, F2, F3, F4, ...)
    pub formants: Vec<Formant>,
    /// Sample rate used for extraction
    pub sample_rate: u32,
}

impl FormantResult {
    #[must_use]
    pub const fn new(sample_rate: u32) -> Self {
        Self {
            formants: Vec::new(),
            sample_rate,
        }
    }

    /// Get formant by index (0 = F1, 1 = F2, etc.)
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Formant> {
        self.formants.get(index)
    }

    /// Get F1 (first formant)
    #[must_use]
    pub fn f1(&self) -> Option<&Formant> {
        self.get(0)
    }

    /// Get F2 (second formant)
    #[must_use]
    pub fn f2(&self) -> Option<&Formant> {
        self.get(1)
    }

    /// Get F3 (third formant)
    #[must_use]
    pub fn f3(&self) -> Option<&Formant> {
        self.get(2)
    }
}

// =============================================================================
// Fast Math Approximations
// =============================================================================

/// Fast approximate atan2
///
/// Error < 0.005 rad, sufficient for formant frequency mapping.
/// Uses polynomial approximation with quadrant mapping.
#[inline(always)]
fn fast_atan2(y: f32, x: f32) -> f32 {
    if x == 0.0 && y == 0.0 {
        return 0.0;
    }

    let abs_y = y.abs();
    let abs_x = x.abs();

    // min / max ratio
    let a = abs_x.min(abs_y) / abs_x.max(abs_y);

    // Polynomial approximation for atan(a) where 0 <= a <= 1
    // atan(a) ≈ a - a^3/3 + a^5/5 - ... (optimized polynomial)
    let s = a * a;
    let r = ((-0.046_496_473f32)
        .mul_add(s, 0.159_314_22)
        .mul_add(s, -0.327_622_77)
        * s)
        .mul_add(a, a);

    // Map back to full circle
    let r = if abs_y > abs_x { 1.570_796_4 - r } else { r };

    // Quadrant correction
    match (x < 0.0, y < 0.0) {
        (true, true) => -core::f32::consts::PI + r,
        (true, false) => core::f32::consts::PI - r,
        (false, true) => -r,
        (false, false) => r,
    }
}

/// Fast magnitude (hypot) approximation
///
/// Uses Alpha max plus beta min algorithm.
/// Max error ~4%, sufficient for formant analysis.
#[inline(always)]
fn fast_magnitude(re: f32, im: f32) -> f32 {
    let abs_re = re.abs();
    let abs_im = im.abs();
    // Coefficients for minimum average error
    0.960_433_87f32.mul_add(abs_re.max(abs_im), 0.397_824_73 * abs_re.min(abs_im))
}

/// Fast approximate natural log for values near 1.0
///
/// Uses Padé approximation. For r in [0.7, 1.0], error < 0.01
#[inline(always)]
fn fast_ln_near_one(x: f32) -> f32 {
    // For x near 1, ln(x) ≈ 2 * (x-1)/(x+1) * (1 + (x-1)^2/(3*(x+1)^2) + ...)
    // Simplified: ln(x) ≈ (x - 1) - (x - 1)^2 / 2 for x close to 1
    let d = x - 1.0;
    (0.5 * d).mul_add(-d, d)
}

// =============================================================================
// The Extractor
// =============================================================================

/// Formant extractor using LPC roots
///
/// # Performance Characteristics
///
/// - **Zero-allocation hot path**: All root finding uses stack arrays
/// - **Fast math**: Approximate atan2 and magnitude
/// - **`SoA` layout**: Separate re[] and im[] arrays for SIMD
/// - **Insertion sort**: Inline-friendly for small arrays
#[derive(Debug, Clone)]
pub struct FormantExtractor {
    /// Sample rate
    sample_rate: u32,
    /// Minimum formant frequency
    min_freq: f32,
    /// Maximum formant frequency
    max_freq: f32,
    /// Maximum bandwidth
    max_bandwidth: f32,
}

impl FormantExtractor {
    /// Create new formant extractor
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            min_freq: 90.0,
            max_freq: (sample_rate / 2) as f32 - 50.0,
            max_bandwidth: 400.0,
        }
    }

    /// Set minimum formant frequency
    #[must_use]
    pub const fn with_min_freq(mut self, freq: f32) -> Self {
        self.min_freq = freq;
        self
    }

    /// Set maximum formant frequency
    #[must_use]
    pub const fn with_max_freq(mut self, freq: f32) -> Self {
        self.max_freq = freq;
        self
    }

    /// Set maximum bandwidth
    #[must_use]
    pub const fn with_max_bandwidth(mut self, bw: f32) -> Self {
        self.max_bandwidth = bw;
        self
    }

    /// Extract formants from LPC coefficients
    ///
    /// # Zero-Allocation Hot Path
    ///
    /// The root finding algorithm runs entirely on the stack.
    /// Only the final output Vec allocation occurs.
    ///
    /// # Errors
    ///
    /// Returns `VoiceError` if LPC order is zero or extraction fails.
    pub fn extract(&self, lpc: &LpcCoefficients) -> VoiceResult<FormantResult> {
        let order = lpc.order();

        // Fallback for oversized LPC (rare)
        if order == 0 {
            return Ok(FormantResult::new(self.sample_rate));
        }

        if order > MAX_LPC_ORDER {
            // Fall back to slower but unlimited version
            return self.extract_fallback(lpc);
        }

        // 1. Find roots using stack-allocated Durand-Kerner
        let (roots_re, roots_im, count) = self.find_roots_fast(&lpc.coeffs);

        // 2. Convert roots to formant candidates (stack array)
        let mut candidates = [Formant::default(); MAX_LPC_ORDER];
        let mut cand_count = 0;

        let sr_factor = self.sample_rate as f32 / (2.0 * PI);
        let bw_factor = -(self.sample_rate as f32) / PI;

        for i in 0..count {
            let re = roots_re[i];
            let im = roots_im[i];

            // Only process positive imaginary (conjugate pairs)
            if im <= 0.0 {
                continue;
            }

            // Stability check using fast magnitude
            let mag = fast_magnitude(re, im);
            if !(0.7..1.0).contains(&mag) {
                continue;
            }

            // Calculate frequency using fast atan2
            let theta = fast_atan2(im, re);
            let frequency = theta * sr_factor;

            // Calculate bandwidth using fast ln approximation
            let bandwidth = bw_factor * fast_ln_near_one(mag);

            // Filter by constraints
            if frequency >= self.min_freq
                && frequency <= self.max_freq
                && bandwidth > 0.0
                && bandwidth < self.max_bandwidth
            {
                let amplitude = 1.0 / (1.0 - mag + 0.01);

                candidates[cand_count] = Formant {
                    frequency,
                    bandwidth,
                    amplitude,
                };
                cand_count += 1;
            }
        }

        // 3. Sort by frequency using insertion sort (fast for small arrays)
        insertion_sort_formants(&mut candidates[..cand_count]);

        // 4. Build output (single allocation)
        let formants = candidates[..cand_count].to_vec();

        Ok(FormantResult {
            formants,
            sample_rate: self.sample_rate,
        })
    }

    /// Durand-Kerner root finder using `SoA` (Structure of Arrays) on stack
    ///
    /// Returns (`re_array`, `im_array`, count)
    ///
    /// # Algorithm
    ///
    /// Solves the LPC polynomial: z^n + a[0]z^(n-1) + ... + a[n-1] = 0
    /// using Durand-Kerner iteration with initial roots on unit circle.
    #[inline]
    #[allow(clippy::unused_self)] // method logically belongs to the extractor
    fn find_roots_fast(
        &self,
        coeffs: &[f32],
    ) -> ([f32; MAX_LPC_ORDER], [f32; MAX_LPC_ORDER], usize) {
        let n = coeffs.len();

        // Stack buffers - SoA layout for SIMD-friendly access
        let mut re = [0.0f32; MAX_LPC_ORDER];
        let mut im = [0.0f32; MAX_LPC_ORDER];

        // Initialize roots uniformly on circle inside unit circle
        let angle_step = 2.0 * PI / n as f32;
        for i in 0..n {
            let theta = angle_step * (i as f32 + 0.5);
            let (sin_t, cos_t) = theta.sin_cos();
            re[i] = 0.9 * cos_t;
            im[i] = 0.9 * sin_t;
        }

        // Durand-Kerner iteration
        for _ in 0..MAX_DK_ITERATIONS {
            let mut max_diff = 0.0f32;

            for i in 0..n {
                let z_re = re[i];
                let z_im = im[i];

                // Evaluate P(z) using Horner's method
                // P(z) = z^n + a[0]z^(n-1) + a[1]z^(n-2) + ... + a[n-1]
                let mut p_re = 1.0f32; // Leading coefficient is 1
                let mut p_im = 0.0f32;

                for k in 0..n {
                    // p = p * z + a[k]
                    let next_re = p_re.mul_add(z_re, -(p_im * z_im)) + coeffs[k];
                    let next_im = p_re.mul_add(z_im, p_im * z_re);
                    p_re = next_re;
                    p_im = next_im;
                }

                // Compute denominator: product of (z - root[j]) for j != i
                let mut den_re = 1.0f32;
                let mut den_im = 0.0f32;

                for j in 0..n {
                    if i != j {
                        let diff_re = z_re - re[j];
                        let diff_im = z_im - im[j];

                        let next_den_re = den_re.mul_add(diff_re, -(den_im * diff_im));
                        let next_den_im = den_re.mul_add(diff_im, den_im * diff_re);
                        den_re = next_den_re;
                        den_im = next_den_im;
                    }
                }

                // Update: z -= P(z) / denominator
                let norm = den_re.mul_add(den_re, den_im * den_im);
                if norm > 1e-15 {
                    let inv_norm = 1.0 / norm;
                    let delta_re = p_re.mul_add(den_re, p_im * den_im) * inv_norm;
                    let delta_im = p_im.mul_add(den_re, -(p_re * den_im)) * inv_norm;

                    re[i] -= delta_re;
                    im[i] -= delta_im;

                    // Track convergence (fast max-based check)
                    let diff_mag = delta_re.abs().max(delta_im.abs());
                    if diff_mag > max_diff {
                        max_diff = diff_mag;
                    }
                }
            }

            // Early exit on convergence
            if max_diff < DK_EPSILON {
                break;
            }
        }

        (re, im, n)
    }

    /// Fallback extraction for LPC orders > `MAX_LPC_ORDER` (rare)
    ///
    /// # Errors
    ///
    /// Returns `VoiceError` if extraction fails.
    #[allow(clippy::unnecessary_wraps)] // keeps consistent API with extract()
    fn extract_fallback(&self, lpc: &LpcCoefficients) -> VoiceResult<FormantResult> {
        let order = lpc.order();
        if order == 0 {
            return Ok(FormantResult::new(self.sample_rate));
        }

        // Use heap-allocated Vec for large orders
        let mut re: Vec<f32> = Vec::with_capacity(order);
        let mut im: Vec<f32> = Vec::with_capacity(order);

        let angle_step = 2.0 * PI / order as f32;
        for i in 0..order {
            let theta = angle_step * (i as f32 + 0.5);
            let (sin_t, cos_t) = theta.sin_cos();
            re.push(0.9 * cos_t);
            im.push(0.9 * sin_t);
        }

        // Durand-Kerner iteration (same algorithm, heap allocated)
        for _ in 0..MAX_DK_ITERATIONS {
            let mut max_diff = 0.0f32;

            for i in 0..order {
                let z_re = re[i];
                let z_im = im[i];

                let mut p_re = 1.0f32;
                let mut p_im = 0.0f32;

                for k in 0..order {
                    let next_re = p_re.mul_add(z_re, -(p_im * z_im)) + lpc.coeffs[k];
                    let next_im = p_re.mul_add(z_im, p_im * z_re);
                    p_re = next_re;
                    p_im = next_im;
                }

                let mut den_re = 1.0f32;
                let mut den_im = 0.0f32;

                for j in 0..order {
                    if i != j {
                        let diff_re = z_re - re[j];
                        let diff_im = z_im - im[j];
                        let next_den_re = den_re.mul_add(diff_re, -(den_im * diff_im));
                        let next_den_im = den_re.mul_add(diff_im, den_im * diff_re);
                        den_re = next_den_re;
                        den_im = next_den_im;
                    }
                }

                let norm = den_re.mul_add(den_re, den_im * den_im);
                if norm > 1e-15 {
                    let inv_norm = 1.0 / norm;
                    let delta_re = p_re.mul_add(den_re, p_im * den_im) * inv_norm;
                    let delta_im = p_im.mul_add(den_re, -(p_re * den_im)) * inv_norm;

                    re[i] -= delta_re;
                    im[i] -= delta_im;

                    let diff_mag = delta_re.abs().max(delta_im.abs());
                    if diff_mag > max_diff {
                        max_diff = diff_mag;
                    }
                }
            }

            if max_diff < DK_EPSILON {
                break;
            }
        }

        // Convert to formants
        let mut formants = Vec::new();
        let sr_factor = self.sample_rate as f32 / (2.0 * PI);
        let bw_factor = -(self.sample_rate as f32) / PI;

        for i in 0..order {
            if im[i] <= 0.0 {
                continue;
            }

            let mag = fast_magnitude(re[i], im[i]);
            if !(0.7..1.0).contains(&mag) {
                continue;
            }

            let theta = fast_atan2(im[i], re[i]);
            let frequency = theta * sr_factor;
            let bandwidth = bw_factor * fast_ln_near_one(mag);

            if frequency >= self.min_freq
                && frequency <= self.max_freq
                && bandwidth > 0.0
                && bandwidth < self.max_bandwidth
            {
                let amplitude = 1.0 / (1.0 - mag + 0.01);
                formants.push(Formant {
                    frequency,
                    bandwidth,
                    amplitude,
                });
            }
        }

        formants.sort_by(|a, b| {
            a.frequency
                .partial_cmp(&b.frequency)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(FormantResult {
            formants,
            sample_rate: self.sample_rate,
        })
    }

    /// Synthesize LPC coefficients from formants (inverse operation)
    #[must_use]
    pub fn synthesize_lpc(&self, formants: &[Formant], order: usize) -> LpcCoefficients {
        let mut poles_re = [0.0f32; MAX_LPC_ORDER];
        let mut poles_im = [0.0f32; MAX_LPC_ORDER];
        let mut pole_count = 0;

        // Convert formants to poles (conjugate pairs)
        for formant in formants {
            if pole_count + 2 > order {
                break;
            }

            let theta = 2.0 * PI * formant.frequency / self.sample_rate as f32;
            let r = (-PI * formant.bandwidth / self.sample_rate as f32).exp();

            let (sin_t, cos_t) = theta.sin_cos();

            // Add conjugate pair
            poles_re[pole_count] = r * cos_t;
            poles_im[pole_count] = r * sin_t;
            pole_count += 1;

            poles_re[pole_count] = r * cos_t;
            poles_im[pole_count] = -r * sin_t;
            pole_count += 1;
        }

        // Compute polynomial coefficients from poles
        let mut coeffs = vec![0.0f32; order];
        let mut poly = vec![0.0f32; order + 1];
        poly[0] = 1.0;

        for i in 0..pole_count.min(order) {
            let pr = poles_re[i];

            for k in (1..=i + 1).rev() {
                if k <= order {
                    poly[k] -= pr * poly[k - 1];
                }
            }
        }

        let n = order.min(poly.len() - 1);
        coeffs[..n].copy_from_slice(&poly[1..=n]);

        LpcCoefficients {
            coeffs,
            gain: 1.0,
            reflection: vec![],
            error: 0.0,
        }
    }
}

/// Insertion sort for formants by frequency (inline-friendly for small arrays)
#[inline]
fn insertion_sort_formants(arr: &mut [Formant]) {
    for i in 1..arr.len() {
        let key = arr[i];
        let mut j = i;
        while j > 0 && arr[j - 1].frequency > key.frequency {
            arr[j] = arr[j - 1];
            j -= 1;
        }
        arr[j] = key;
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_formant_creation() {
        let f = Formant::new(500.0, 80.0).with_amplitude(0.8);
        assert_eq!(f.frequency, 500.0);
        assert_eq!(f.bandwidth, 80.0);
        assert_eq!(f.amplitude, 0.8);
    }

    #[test]
    fn test_formant_extraction() {
        let extractor = FormantExtractor::new(16000);

        let lpc = LpcCoefficients {
            coeffs: vec![
                1.5, -0.8, 0.3, -0.1, 0.05, -0.02, 0.01, -0.005, 0.002, -0.001,
            ],
            gain: 0.1,
            reflection: vec![],
            error: 0.0,
        };

        let result = extractor.extract(&lpc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_formant_result_accessors() {
        let mut result = FormantResult::new(16000);
        result.formants.push(Formant::new(500.0, 80.0));
        result.formants.push(Formant::new(1500.0, 100.0));
        result.formants.push(Formant::new(2500.0, 120.0));

        assert_eq!(result.f1().unwrap().frequency, 500.0);
        assert_eq!(result.f2().unwrap().frequency, 1500.0);
        assert_eq!(result.f3().unwrap().frequency, 2500.0);
    }

    #[test]
    fn test_fast_atan2() {
        // Test quadrants
        let pi = std::f32::consts::PI;

        // First quadrant
        let result = fast_atan2(1.0, 1.0);
        assert!((result - pi / 4.0).abs() < 0.01);

        // Second quadrant
        let result = fast_atan2(1.0, -1.0);
        assert!((result - 3.0 * pi / 4.0).abs() < 0.01);

        // Third quadrant
        let result = fast_atan2(-1.0, -1.0);
        assert!((result + 3.0 * pi / 4.0).abs() < 0.01);

        // Fourth quadrant
        let result = fast_atan2(-1.0, 1.0);
        assert!((result + pi / 4.0).abs() < 0.01);
    }

    #[test]
    fn test_fast_magnitude() {
        // Test against exact sqrt
        let re = 3.0;
        let im = 4.0;
        let fast = fast_magnitude(re, im);
        let exact = re.hypot(im);

        // Error should be < 5%
        assert!((fast - exact).abs() / exact < 0.05);
    }

    #[test]
    fn test_insertion_sort() {
        let mut arr = [
            Formant::new(2000.0, 100.0),
            Formant::new(500.0, 80.0),
            Formant::new(1500.0, 90.0),
            Formant::new(3000.0, 110.0),
        ];

        insertion_sort_formants(&mut arr);

        assert_eq!(arr[0].frequency, 500.0);
        assert_eq!(arr[1].frequency, 1500.0);
        assert_eq!(arr[2].frequency, 2000.0);
        assert_eq!(arr[3].frequency, 3000.0);
    }

    #[test]
    fn test_empty_lpc() {
        let extractor = FormantExtractor::new(16000);

        let lpc = LpcCoefficients {
            coeffs: vec![],
            gain: 0.0,
            reflection: vec![],
            error: 0.0,
        };

        let result = extractor.extract(&lpc).unwrap();
        assert!(result.formants.is_empty());
    }
}
