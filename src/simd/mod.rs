//! SIMD acceleration module
//!
//! Provides target-specific SIMD implementations for hot paths.
//!
//! # Architecture Support
//!
//! | Target | Feature | Implementation |
//! |--------|---------|----------------|
//! | aarch64 | NEON | `arm::neon` |
//! | arm | NEON | `arm::neon` |
//! | `x86_64` | AVX2 | Future |
//! | `x86_64` | SSE4.1 | Future |
//!
//! # Usage
//!
//! The module automatically selects the best implementation at compile time.
//! Functions fall back to scalar implementations on unsupported targets.

// Integer kernels with no intrinsics, shared by both backends
mod common;

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
pub mod arm;

// Re-export commonly used functions
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
pub use arm::*;

// Fallback scalar implementations for non-ARM targets
// Also compiled for tests on ARM so the two backends can be compared bit for bit
#[cfg(any(test, not(any(target_arch = "aarch64", target_arch = "arm"))))]
mod scalar;

#[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
pub use scalar::*;

/// Check if NEON is available at runtime (aarch64)
#[cfg(target_arch = "aarch64")]
#[inline]
#[must_use]
pub const fn has_neon() -> bool {
    // NEON is always available on aarch64
    true
}

/// Check if NEON is available at runtime (arm 32-bit)
#[cfg(target_arch = "arm")]
#[inline]
pub fn has_neon() -> bool {
    #[cfg(target_feature = "neon")]
    {
        true
    }
    #[cfg(not(target_feature = "neon"))]
    {
        false
    }
}

/// Fallback for non-ARM targets
#[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
#[inline]
pub fn has_neon() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_detection() {
        let has = has_neon();
        #[cfg(target_arch = "aarch64")]
        assert!(has, "NEON should be available on aarch64");

        println!("NEON available: {has}");
    }

    /// Deterministic LCG input generator (no external RNG dependency)
    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    fn lcg_fill(seed: &mut u64, out: &mut [i32], span: i32) {
        for v in out.iter_mut() {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *v = ((*seed >> 33) as i32).rem_euclid(2 * span + 1) - span;
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    fn arm_mul_4x(a: &[i32; 4], b: &[i32; 4]) -> [i32; 4] {
        // SAFETY: NEON is mandatory on aarch64
        #[cfg(target_arch = "aarch64")]
        let r = unsafe { arm::q16_mul_4x_neon(a, b) };
        #[cfg(not(target_arch = "aarch64"))]
        let r = arm::q16_mul_4x_neon(a, b);
        r
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    fn check_lpc(coeffs: &[i32], gain: i32, exc: &[i32], what: &str) {
        let mut out_s = vec![0i32; exc.len()];
        let mut out_a = vec![0i32; exc.len()];
        scalar::q16_lpc_filter_neon(coeffs, gain, exc, &mut out_s);
        arm::q16_lpc_filter_neon(coeffs, gain, exc, &mut out_a);
        assert_eq!(out_s, out_a, "lpc {what} order {}", coeffs.len());
    }

    /// The scalar backend (x86_64 and other non-ARM targets) must give the same
    /// bits as the ARM backend for every kernel
    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    #[test]
    fn scalar_matches_arm_bit_for_bit() {
        use crate::types::{EMBEDDING_DIM, Q16_ONE};

        let mut seed = 0x5eed_u64;
        for round in 0..400 {
            let span = match round % 3 {
                0 => 4 * Q16_ONE,
                1 => Q16_ONE / 3,
                _ => i32::MAX / 2,
            };

            let mut a4 = [0i32; 4];
            let mut b4 = [0i32; 4];
            lcg_fill(&mut seed, &mut a4, span);
            lcg_fill(&mut seed, &mut b4, span);
            assert_eq!(
                scalar::q16_mul_4x_neon(&a4, &b4),
                arm_mul_4x(&a4, &b4),
                "mul_4x round {round}"
            );

            let len = round % 41;
            let mut a = vec![0i32; len];
            let mut b = vec![0i32; len + round % 3];
            lcg_fill(&mut seed, &mut a, span);
            lcg_fill(&mut seed, &mut b, span);
            assert_eq!(
                scalar::q16_dot_product_neon(&a, &b),
                arm::q16_dot_product_neon(&a, &b),
                "dot round {round} len {len}"
            );

            let mut ea = [0i32; EMBEDDING_DIM];
            let mut eb = [0i32; EMBEDDING_DIM];
            lcg_fill(&mut seed, &mut ea, span);
            lcg_fill(&mut seed, &mut eb, span);
            assert_eq!(
                scalar::q16_cosine_similarity_neon(&ea, &eb),
                arm::q16_cosine_similarity_neon(&ea, &eb),
                "cosine round {round}"
            );

            // orders 0..=16 cover the < 4 path, every remainder of the 4-lane path,
            // and samples before and after `order`
            let order = round % 17;
            let mut coeffs = vec![0i32; order];
            lcg_fill(&mut seed, &mut coeffs, Q16_ONE / 8);
            let mut exc = vec![0i32; 48];
            lcg_fill(&mut seed, &mut exc, Q16_ONE);
            check_lpc(&coeffs, Q16_ONE / 2 - round as i32, &exc, "random");
        }

        // Edge inputs: zero vectors, extreme Q16 values, sums that go negative
        let zero = [0i32; EMBEDDING_DIM];
        let max = [i32::MAX; EMBEDDING_DIM];
        let min = [i32::MIN; EMBEDDING_DIM];
        let mut alt = [0i32; EMBEDDING_DIM];
        for (i, v) in alt.iter_mut().enumerate() {
            *v = if i % 2 == 0 { i32::MAX } else { i32::MIN };
        }
        let mut neg = [0i32; EMBEDDING_DIM];
        for (i, v) in neg.iter_mut().enumerate() {
            *v = -Q16_ONE * (1 + (i as i32 % 7));
        }
        let mut pos = neg;
        for v in pos.iter_mut() {
            *v = -*v;
        }
        let vecs = [&zero, &max, &min, &alt, &neg, &pos];
        for (i, x) in vecs.iter().enumerate() {
            for (j, y) in vecs.iter().enumerate() {
                assert_eq!(
                    scalar::q16_cosine_similarity_neon(x, y),
                    arm::q16_cosine_similarity_neon(x, y),
                    "cosine edge {i} {j}"
                );
                assert_eq!(
                    scalar::q16_dot_product_neon(&x[..], &y[..]),
                    arm::q16_dot_product_neon(&x[..], &y[..]),
                    "dot edge {i} {j}"
                );
                let x4 = [x[0], x[1], x[2], x[3]];
                let y4 = [y[0], y[1], y[2], y[3]];
                assert_eq!(
                    scalar::q16_mul_4x_neon(&x4, &y4),
                    arm_mul_4x(&x4, &y4),
                    "mul_4x edge {i} {j}"
                );
            }
        }
        for order in 0..=16 {
            // all-negative coefficients with an alternating excitation: the tap sums go negative
            let coeffs = vec![-Q16_ONE / 9; order];
            let exc: Vec<i32> = (0..40)
                .map(|k| if k % 2 == 0 { Q16_ONE } else { -Q16_ONE })
                .collect();
            check_lpc(&coeffs, Q16_ONE, &exc, "negative");
            check_lpc(&coeffs, Q16_ONE, &[0; 40], "zero excitation");
            check_lpc(&coeffs, Q16_ONE, &[], "empty");
        }
    }

    /// Orders above the NEON kernel's 32-entry coefficient buffer must not write past it
    /// and must give the canonical result
    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    #[test]
    fn lpc_high_order_matches_canonical() {
        use crate::types::Q16_ONE;

        let mut seed = 0x0dd_u64;
        for order in 29..=40 {
            let mut coeffs = vec![0i32; order];
            lcg_fill(&mut seed, &mut coeffs, Q16_ONE / 64);
            let mut exc = vec![0i32; 96];
            lcg_fill(&mut seed, &mut exc, Q16_ONE);
            check_lpc(&coeffs, Q16_ONE / 2, &exc, "high order");
        }
    }
}
