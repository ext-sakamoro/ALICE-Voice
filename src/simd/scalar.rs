//! Scalar fallback implementations for non-ARM targets
//!
//! Same API as the ARM NEON implementations. Each function calls the portable
//! definition in `common`, which gives the same bits as the aarch64 NEON kernel.

use super::common;
use crate::types::EMBEDDING_DIM;

// On ARM this module is compiled only for the comparison test, which does not use these
#[cfg_attr(
    any(target_arch = "aarch64", target_arch = "arm"),
    allow(unused_imports)
)]
pub use super::common::{fast_isqrt_32, fast_isqrt_64, q16_mac, q16_mul};

/// Scalar 4x Q16 multiply (no SIMD)
#[inline]
#[must_use]
pub fn q16_mul_4x_neon(a: &[i32; 4], b: &[i32; 4]) -> [i32; 4] {
    common::q16_mul_4x(a, b)
}

/// Scalar dot product over the common length of `a` and `b`
#[inline]
#[must_use]
pub fn q16_dot_product_neon(a: &[i32], b: &[i32]) -> i64 {
    common::q16_dot_product(a, b)
}

/// Scalar cosine similarity
#[must_use]
pub fn q16_cosine_similarity_neon(a: &[i32; EMBEDDING_DIM], b: &[i32; EMBEDDING_DIM]) -> i32 {
    common::q16_cosine_similarity(a, b)
}

/// Scalar LPC filter
pub fn q16_lpc_filter_neon(coeffs: &[i32], gain: i32, excitation: &[i32], output: &mut [i32]) {
    common::q16_lpc_filter(coeffs, gain, excitation, output);
}
