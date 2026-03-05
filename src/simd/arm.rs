//! ARM NEON SIMD implementations - "Crispy" Tuned Edition
//!
//! Optimizations:
//! - 4-way Loop Unrolling to hide instruction latency
//! - Multiple accumulators to break dependency chains
//! - Pre-reversed coefficients for LPC to eliminate shuffling
//! - Raw pointer arithmetic

use crate::types::{EMBEDDING_DIM, Q16_SHIFT};

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    vaddq_s64, vcombine_s32, vdupq_n_s64, vget_high_s32, vget_low_s32, vgetq_lane_s64, vld1q_s32,
    vmlal_s32, vmull_s32, vshrn_n_s64, vst1q_s32,
};

// ============================================
// Q16.16 Fixed-Point Operations (Scalar)
// ============================================

/// Q16.16 multiply using 64-bit intermediate (scalar)
#[inline(always)]
#[must_use]
pub fn q16_mul(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> Q16_SHIFT) as i32
}

/// Q16.16 multiply-accumulate (scalar)
#[inline(always)]
#[must_use]
pub fn q16_mac(acc: i32, a: i32, b: i32) -> i32 {
    acc + q16_mul(a, b)
}

/// Q16.16 multiply 4 values in parallel (NEON)
///
/// # Safety
/// Caller must ensure `target_feature = "neon"` is available on the current CPU.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[must_use]
pub unsafe fn q16_mul_4x_neon(a: &[i32; 4], b: &[i32; 4]) -> [i32; 4] {
    let va = vld1q_s32(a.as_ptr());
    let vb = vld1q_s32(b.as_ptr());

    let a_lo = vget_low_s32(va);
    let a_hi = vget_high_s32(va);
    let b_lo = vget_low_s32(vb);
    let b_hi = vget_high_s32(vb);

    let prod_lo = vmull_s32(a_lo, b_lo);
    let prod_hi = vmull_s32(a_hi, b_hi);

    let result_lo = vshrn_n_s64(prod_lo, Q16_SHIFT);
    let result_hi = vshrn_n_s64(prod_hi, Q16_SHIFT);

    let result = vcombine_s32(result_lo, result_hi);

    let mut out = [0i32; 4];
    vst1q_s32(out.as_mut_ptr(), result);
    out
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub fn q16_mul_4x_neon(a: &[i32; 4], b: &[i32; 4]) -> [i32; 4] {
    [
        q16_mul(a[0], b[0]),
        q16_mul(a[1], b[1]),
        q16_mul(a[2], b[2]),
        q16_mul(a[3], b[3]),
    ]
}

// ============================================
// Q16.16 Dot Product (Ultra Unrolled)
// ============================================

/// Q16.16 dot product with 4-way unrolling
///
/// Processing 16 elements per iteration allows the CPU to overlap
/// load and multiply-add operations, maximizing throughput.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn q16_dot_product_neon(a: &[i32], b: &[i32]) -> i64 {
    let len = a.len().min(b.len());
    let mut ptr_a = a.as_ptr();
    let mut ptr_b = b.as_ptr();

    unsafe {
        // 4 independent accumulators to break dependency chains
        let mut acc0_lo = vdupq_n_s64(0);
        let mut acc0_hi = vdupq_n_s64(0);
        let mut acc1_lo = vdupq_n_s64(0);
        let mut acc1_hi = vdupq_n_s64(0);

        let mut i = 0;

        // Main loop: Process 16 elements (4 vectors) per iteration
        while i + 16 <= len {
            // Pipeline 0: Elements 0-3
            let va0 = vld1q_s32(ptr_a);
            let vb0 = vld1q_s32(ptr_b);
            acc0_lo = vmlal_s32(acc0_lo, vget_low_s32(va0), vget_low_s32(vb0));
            acc0_hi = vmlal_s32(acc0_hi, vget_high_s32(va0), vget_high_s32(vb0));

            // Pipeline 1: Elements 4-7
            let va1 = vld1q_s32(ptr_a.add(4));
            let vb1 = vld1q_s32(ptr_b.add(4));
            acc1_lo = vmlal_s32(acc1_lo, vget_low_s32(va1), vget_low_s32(vb1));
            acc1_hi = vmlal_s32(acc1_hi, vget_high_s32(va1), vget_high_s32(vb1));

            // Pipeline 2: Elements 8-11 (Reuse acc0)
            let va2 = vld1q_s32(ptr_a.add(8));
            let vb2 = vld1q_s32(ptr_b.add(8));
            acc0_lo = vmlal_s32(acc0_lo, vget_low_s32(va2), vget_low_s32(vb2));
            acc0_hi = vmlal_s32(acc0_hi, vget_high_s32(va2), vget_high_s32(vb2));

            // Pipeline 3: Elements 12-15 (Reuse acc1)
            let va3 = vld1q_s32(ptr_a.add(12));
            let vb3 = vld1q_s32(ptr_b.add(12));
            acc1_lo = vmlal_s32(acc1_lo, vget_low_s32(va3), vget_low_s32(vb3));
            acc1_hi = vmlal_s32(acc1_hi, vget_high_s32(va3), vget_high_s32(vb3));

            ptr_a = ptr_a.add(16);
            ptr_b = ptr_b.add(16);
            i += 16;
        }

        // Combine accumulators
        let sum_lo = vaddq_s64(acc0_lo, acc1_lo);
        let sum_hi = vaddq_s64(acc0_hi, acc1_hi);
        let final_acc = vaddq_s64(sum_lo, sum_hi);

        let mut sum = vgetq_lane_s64(final_acc, 0) + vgetq_lane_s64(final_acc, 1);

        // Handle remainder (scalar)
        while i < len {
            sum += *ptr_a as i64 * *ptr_b as i64;
            ptr_a = ptr_a.add(1);
            ptr_b = ptr_b.add(1);
            i += 1;
        }

        sum >> Q16_SHIFT
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn q16_dot_product_neon(a: &[i32], b: &[i32]) -> i64 {
    // Fallback
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as i64 * y as i64)
        .sum::<i64>()
        >> Q16_SHIFT
}

// ============================================
// Q16.16 Cosine Similarity (Fused & Unrolled)
// ============================================

#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn q16_cosine_similarity_neon(a: &[i32; EMBEDDING_DIM], b: &[i32; EMBEDDING_DIM]) -> i32 {
    unsafe {
        // 3 sets of accumulators, 8-way unrolled (32 iterations for 256 dim)
        let mut dot_lo = vdupq_n_s64(0);
        let mut dot_hi = vdupq_n_s64(0);
        let mut na_lo = vdupq_n_s64(0); // Norm A
        let mut na_hi = vdupq_n_s64(0);
        let mut nb_lo = vdupq_n_s64(0); // Norm B
        let mut nb_hi = vdupq_n_s64(0);

        let mut ptr_a = a.as_ptr();
        let mut ptr_b = b.as_ptr();

        // 256 is divisible by 8, so we unroll 8x per loop -> 32 iterations
        for _ in 0..32 {
            // Block 1 (4 elements)
            let va0 = vld1q_s32(ptr_a);
            let vb0 = vld1q_s32(ptr_b);
            let a0_lo = vget_low_s32(va0);
            let a0_hi = vget_high_s32(va0);
            let b0_lo = vget_low_s32(vb0);
            let b0_hi = vget_high_s32(vb0);

            // Block 2 (4 elements)
            let va1 = vld1q_s32(ptr_a.add(4));
            let vb1 = vld1q_s32(ptr_b.add(4));
            let a1_lo = vget_low_s32(va1);
            let a1_hi = vget_high_s32(va1);
            let b1_lo = vget_low_s32(vb1);
            let b1_hi = vget_high_s32(vb1);

            // Compute Dot
            dot_lo = vmlal_s32(dot_lo, a0_lo, b0_lo);
            dot_hi = vmlal_s32(dot_hi, a0_hi, b0_hi);
            dot_lo = vmlal_s32(dot_lo, a1_lo, b1_lo);
            dot_hi = vmlal_s32(dot_hi, a1_hi, b1_hi);

            // Compute Norm A
            na_lo = vmlal_s32(na_lo, a0_lo, a0_lo);
            na_hi = vmlal_s32(na_hi, a0_hi, a0_hi);
            na_lo = vmlal_s32(na_lo, a1_lo, a1_lo);
            na_hi = vmlal_s32(na_hi, a1_hi, a1_hi);

            // Compute Norm B
            nb_lo = vmlal_s32(nb_lo, b0_lo, b0_lo);
            nb_hi = vmlal_s32(nb_hi, b0_hi, b0_hi);
            nb_lo = vmlal_s32(nb_lo, b1_lo, b1_lo);
            nb_hi = vmlal_s32(nb_hi, b1_hi, b1_hi);

            ptr_a = ptr_a.add(8);
            ptr_b = ptr_b.add(8);
        }

        // Reduction
        let dot_sum = vaddq_s64(dot_lo, dot_hi);
        let dot = vgetq_lane_s64(dot_sum, 0) + vgetq_lane_s64(dot_sum, 1);

        let na_sum = vaddq_s64(na_lo, na_hi);
        let na = vgetq_lane_s64(na_sum, 0) + vgetq_lane_s64(na_sum, 1);

        let nb_sum = vaddq_s64(nb_lo, nb_hi);
        let nb = vgetq_lane_s64(nb_sum, 0) + vgetq_lane_s64(nb_sum, 1);

        // Q16 scaling
        let dot_q16 = dot >> Q16_SHIFT;
        let na_q16 = na >> Q16_SHIFT;
        let nb_q16 = nb >> Q16_SHIFT;

        // Use f64 sqrt for precision (fast_isqrt available but this is more accurate)
        let denom = ((na_q16 as f64).sqrt() * (nb_q16 as f64).sqrt()) as i64;

        if denom > 0 {
            ((dot_q16 << 16) / denom) as i32
        } else {
            0
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn q16_cosine_similarity_neon(a: &[i32; EMBEDDING_DIM], b: &[i32; EMBEDDING_DIM]) -> i32 {
    let mut dot = 0i64;
    let mut norm_a = 0i64;
    let mut norm_b = 0i64;

    for i in 0..EMBEDDING_DIM {
        dot += a[i] as i64 * b[i] as i64;
        norm_a += a[i] as i64 * a[i] as i64;
        norm_b += b[i] as i64 * b[i] as i64;
    }

    let dot_q16 = dot >> Q16_SHIFT;
    let norm_a_q16 = norm_a >> Q16_SHIFT;
    let norm_b_q16 = norm_b >> Q16_SHIFT;

    let denom = ((norm_a_q16 as f64).sqrt() * (norm_b_q16 as f64).sqrt()) as i64;

    if denom > 0 {
        ((dot_q16 << 16) / denom) as i32
    } else {
        0
    }
}

// ============================================
// LPC Filter (Shuffle-Free)
// ============================================

/// Q16.16 LPC filter optimized for linear memory access.
///
/// **Critical Optimization:**
/// To avoid expensive `vrev` (reverse) and `vext` (extract) instructions inside the loop,
/// we pre-reverse the coefficients into a temporary stack buffer.
///
/// `coeffs`: [a1, a2, a3, a4]
/// `rev_coeffs`: [a4, a3, a2, a1]
///
/// Output memory `y` is contiguous: [..., y[n-4], y[n-3], y[n-2], y[n-1]]
///
/// Now `rev_coeffs` and the tail of `y` are aligned for a direct SIMD dot product!
#[cfg(target_arch = "aarch64")]
pub fn q16_lpc_filter_neon(coeffs: &[i32], gain: i32, excitation: &[i32], output: &mut [i32]) {
    let order = coeffs.len();
    let n = excitation.len().min(output.len());

    unsafe {
        // Stack buffer for reversed coefficients (Max order 32 is safe)
        let mut rev_coeffs = [0i32; 32];
        let ptr_rev = rev_coeffs.as_mut_ptr();

        // Pre-reverse coefficients once
        // This moves the shuffle cost out of the O(N) loop to O(order) setup
        for i in 0..order {
            *ptr_rev.add(i) = coeffs[order - 1 - i];
        }

        // Initialize first 'order' samples (scalar fallback)
        for i in 0..order.min(n) {
            let mut sample = q16_mul(gain, excitation[i]);
            for j in 0..i {
                sample = q16_mac(sample, coeffs[j], output[i - 1 - j]);
            }
            output[i] = sample;
        }

        // Main vectorized loop
        if order >= 4 {
            let chunks = order / 4;
            let remainder = order % 4;

            for i in order..n {
                let mut sample = q16_mul(gain, excitation[i]);
                let mut acc_lo = vdupq_n_s64(0);
                let mut acc_hi = vdupq_n_s64(0);

                // Pointer to the start of relevant history in output
                // We need y[i-order] ... y[i-1]
                // Since coeffs are reversed: a[order-1] * y[i-order] + ... + a[0] * y[i-1]
                let ptr_hist = output.as_ptr().add(i - order);

                // Process 4 taps at a time
                for k in 0..chunks {
                    let vc = vld1q_s32(ptr_rev.add(k * 4));
                    let vy = vld1q_s32(ptr_hist.add(k * 4));

                    acc_lo = vmlal_s32(acc_lo, vget_low_s32(vc), vget_low_s32(vy));
                    acc_hi = vmlal_s32(acc_hi, vget_high_s32(vc), vget_high_s32(vy));
                }

                let sum_lo = vaddq_s64(acc_lo, acc_hi);
                let sum = vgetq_lane_s64(sum_lo, 0) + vgetq_lane_s64(sum_lo, 1);

                sample += (sum >> Q16_SHIFT) as i32;

                // Handle remaining taps (use original coeffs for simplicity)
                for j in 0..remainder {
                    sample = q16_mac(sample, coeffs[j], output[i - 1 - j]);
                }

                output[i] = sample;
            }
        } else {
            // Scalar fallback for small order
            for i in order..n {
                let mut sample = q16_mul(gain, excitation[i]);
                for j in 0..order {
                    sample = q16_mac(sample, coeffs[j], output[i - 1 - j]);
                }
                output[i] = sample;
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn q16_lpc_filter_neon(coeffs: &[i32], gain: i32, excitation: &[i32], output: &mut [i32]) {
    // Fallback
    let order = coeffs.len();
    let n = excitation.len().min(output.len());
    for i in 0..n {
        let mut sample = q16_mul(gain, excitation[i]);
        for j in 0..order.min(i) {
            sample = q16_mac(sample, coeffs[j], output[i - 1 - j]);
        }
        output[i] = sample;
    }
}

// ============================================
// Fast Integer Math
// ============================================

/// Fast integer square root (64-bit input, 32-bit output)
///
/// Uses Newton-Raphson with a good initial guess.
/// Returns floor(sqrt(n)).
#[inline]
#[must_use]
pub fn fast_isqrt_64(n: u64) -> u32 {
    if n == 0 {
        return 0;
    }
    if n < 4 {
        return 1;
    }

    // Initial guess: 2^((log2(n) / 2) + 1) to start above the answer
    let log2 = 63 - n.leading_zeros();
    let mut x = 1u64 << ((log2 / 2) + 1);

    // Newton-Raphson iterations
    loop {
        let x1 = (x + n / x) >> 1;
        if x1 >= x {
            break;
        }
        x = x1;
    }

    x as u32
}

/// Fast integer square root (32-bit)
#[inline]
#[must_use]
pub fn fast_isqrt_32(n: u32) -> u16 {
    if n == 0 {
        return 0;
    }

    let shift = (32 - n.leading_zeros()) / 2;
    let mut x = 1u32 << shift;

    loop {
        let x1 = (x + n / x) >> 1;
        if x1 >= x {
            return x as u16;
        }
        x = x1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q16_mul() {
        // 2.5 * 4.0 = 10.0
        let a = (2.5 * 65536.0) as i32;
        let b = (4.0 * 65536.0) as i32;
        let result = q16_mul(a, b);
        let expected = (10.0 * 65536.0) as i32;
        assert!(
            (result - expected).abs() < 2,
            "q16_mul failed: {} vs {}",
            result,
            expected
        );
    }

    #[test]
    fn test_q16_mul_4x() {
        let a = [
            (1.0 * 65536.0) as i32,
            (2.0 * 65536.0) as i32,
            (3.0 * 65536.0) as i32,
            (4.0 * 65536.0) as i32,
        ];
        let b = [
            (2.0 * 65536.0) as i32,
            (3.0 * 65536.0) as i32,
            (4.0 * 65536.0) as i32,
            (5.0 * 65536.0) as i32,
        ];

        #[cfg(target_arch = "aarch64")]
        let result = unsafe { q16_mul_4x_neon(&a, &b) };
        #[cfg(not(target_arch = "aarch64"))]
        let result = q16_mul_4x_neon(&a, &b);

        let expected = [
            (2.0 * 65536.0) as i32,
            (6.0 * 65536.0) as i32,
            (12.0 * 65536.0) as i32,
            (20.0 * 65536.0) as i32,
        ];

        for i in 0..4 {
            assert!(
                (result[i] - expected[i]).abs() < 2,
                "q16_mul_4x[{}] failed: {} vs {}",
                i,
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_fast_isqrt() {
        assert_eq!(fast_isqrt_64(0), 0);
        assert_eq!(fast_isqrt_64(1), 1);
        assert_eq!(fast_isqrt_64(4), 2);
        assert_eq!(fast_isqrt_64(9), 3);
        assert_eq!(fast_isqrt_64(100), 10);
        assert_eq!(fast_isqrt_64(65536), 256);

        // Test larger values
        let n = 1_000_000u64;
        let sqrt = fast_isqrt_64(n);
        assert_eq!(sqrt, 1000);
    }

    #[test]
    fn test_q16_dot_product() {
        let a = [65536i32, 131072, 196608]; // 1.0, 2.0, 3.0 in Q16
        let b = [65536i32, 65536, 65536]; // 1.0, 1.0, 1.0 in Q16

        let result = q16_dot_product_neon(&a, &b);
        // Expected: 1*1 + 2*1 + 3*1 = 6.0 in Q16 = 393216
        let expected = 6 * 65536i64;
        assert!(
            (result - expected).abs() < 10,
            "dot product: {} vs {}",
            result,
            expected
        );
    }

    #[test]
    fn test_q16_cosine_similarity() {
        // Create identical vectors with multiple non-zero elements
        let mut a = [0i32; EMBEDDING_DIM];
        let mut b = [0i32; EMBEDDING_DIM];

        // Set multiple elements to make the test more robust
        for i in 0..10 {
            a[i] = 65536; // 1.0 in Q16
            b[i] = 65536; // 1.0 in Q16
        }

        let sim = q16_cosine_similarity_neon(&a, &b);
        // Should be close to 1.0 (65536 in Q16)
        assert!(sim > 50000, "similarity should be ~1.0: {}", sim);

        // Test orthogonal vectors
        let mut c = [0i32; EMBEDDING_DIM];
        for i in 0..10 {
            c[i + 10] = 65536; // Different indices
        }
        let sim_ortho = q16_cosine_similarity_neon(&a, &c);
        // Should be close to 0.0
        assert!(
            sim_ortho.abs() < 1000,
            "orthogonal similarity should be ~0: {}",
            sim_ortho
        );
    }

    #[test]
    fn test_q16_lpc_filter() {
        // Simple test: gain=1.0, no coefficients
        let coeffs: [i32; 0] = [];
        let gain = 65536; // 1.0
        let excitation = [65536i32, 131072, 196608]; // 1.0, 2.0, 3.0
        let mut output = [0i32; 3];

        q16_lpc_filter_neon(&coeffs, gain, &excitation, &mut output);

        // With no coefficients, output should equal gain * excitation
        assert_eq!(output[0], 65536);
        assert_eq!(output[1], 131072);
        assert_eq!(output[2], 196608);
    }
}
