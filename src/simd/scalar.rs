//! Scalar fallback implementations for non-ARM targets
//!
//! These provide the same API as the ARM NEON implementations
//! but use standard scalar operations.

use crate::types::{EMBEDDING_DIM, Q16_SHIFT};

// Re-export from arm module for consistency
pub use super::arm::{fast_isqrt_32, fast_isqrt_64, q16_mac, q16_mul};

/// Scalar 4x Q16 multiply (no SIMD)
#[inline]
pub fn q16_mul_4x_neon(a: &[i32; 4], b: &[i32; 4]) -> [i32; 4] {
    [
        q16_mul(a[0], b[0]),
        q16_mul(a[1], b[1]),
        q16_mul(a[2], b[2]),
        q16_mul(a[3], b[3]),
    ]
}

/// Scalar dot product
#[inline]
pub fn q16_dot_product_neon(a: &[i32], b: &[i32]) -> i64 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0i64;
    for i in 0..a.len() {
        sum += a[i] as i64 * b[i] as i64;
    }
    sum >> Q16_SHIFT
}

/// Scalar cosine similarity
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

    let norm_a_sqrt = fast_isqrt_64(norm_a_q16 as u64);
    let norm_b_sqrt = fast_isqrt_64(norm_b_q16 as u64);

    let denom = (norm_a_sqrt as i64 * norm_b_sqrt as i64) >> 16;

    if denom > 0 {
        ((dot_q16 << 16) / denom) as i32
    } else {
        0
    }
}

/// Scalar LPC filter
pub fn q16_lpc_filter_neon(coeffs: &[i32], gain: i32, excitation: &[i32], output: &mut [i32]) {
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
