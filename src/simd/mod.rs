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

    /// FNV-1a over the little-endian bytes of each value
    fn fnv1a(hash: &mut u64, bytes: &[u8]) {
        for &x in bytes {
            *hash ^= u64::from(x);
            *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    /// Digests of the kernels this target selects (`simd::*`), one per kernel:
    /// [mul_4x, dot, cosine, lpc]
    fn kernel_digests() -> [u64; 4] {
        use crate::types::{EMBEDDING_DIM, Q16_ONE};

        let mut h = [0xcbf2_9ce4_8422_2325_u64; 4];

        fn mul_4x(a: &[i32; 4], b: &[i32; 4]) -> [i32; 4] {
            // SAFETY: NEON is mandatory on aarch64
            #[cfg(target_arch = "aarch64")]
            let r = unsafe { q16_mul_4x_neon(a, b) };
            #[cfg(not(target_arch = "aarch64"))]
            let r = q16_mul_4x_neon(a, b);
            r
        }
        let lpc = |h: &mut u64, coeffs: &[i32], gain: i32, exc: &[i32]| {
            let mut out = vec![0i32; exc.len()];
            q16_lpc_filter_neon(coeffs, gain, exc, &mut out);
            for v in out {
                fnv1a(h, &v.to_le_bytes());
            }
        };

        let mut seed = 0x5eed_u64;
        for round in 0..400usize {
            let span = match round % 3 {
                0 => 4 * Q16_ONE,
                1 => Q16_ONE / 3,
                _ => i32::MAX / 2,
            };
            let mut a4 = [0i32; 4];
            let mut b4 = [0i32; 4];
            lcg_fill(&mut seed, &mut a4, span);
            lcg_fill(&mut seed, &mut b4, span);
            for v in mul_4x(&a4, &b4) {
                fnv1a(&mut h[0], &v.to_le_bytes());
            }

            let len = round % 41;
            let mut a = vec![0i32; len];
            let mut b = vec![0i32; len + round % 3];
            lcg_fill(&mut seed, &mut a, span);
            lcg_fill(&mut seed, &mut b, span);
            fnv1a(&mut h[1], &q16_dot_product_neon(&a, &b).to_le_bytes());

            let mut ea = [0i32; EMBEDDING_DIM];
            let mut eb = [0i32; EMBEDDING_DIM];
            lcg_fill(&mut seed, &mut ea, span);
            lcg_fill(&mut seed, &mut eb, span);
            fnv1a(
                &mut h[2],
                &q16_cosine_similarity_neon(&ea, &eb).to_le_bytes(),
            );

            // orders 0..=40: the < 4 path, every remainder, and above the NEON buffer (32)
            let order = round % 41;
            let mut coeffs = vec![0i32; order];
            lcg_fill(&mut seed, &mut coeffs, Q16_ONE / 64);
            let mut exc = vec![0i32; 64];
            lcg_fill(&mut seed, &mut exc, Q16_ONE);
            lpc(&mut h[3], &coeffs, Q16_ONE / 2 - round as i32, &exc);
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
        for x in vecs {
            for y in vecs {
                let x4 = [x[0], x[1], x[2], x[3]];
                let y4 = [y[0], y[1], y[2], y[3]];
                for v in mul_4x(&x4, &y4) {
                    fnv1a(&mut h[0], &v.to_le_bytes());
                }
                fnv1a(
                    &mut h[1],
                    &q16_dot_product_neon(&x[..], &y[..]).to_le_bytes(),
                );
                fnv1a(&mut h[2], &q16_cosine_similarity_neon(x, y).to_le_bytes());
            }
        }
        for order in 0..=40 {
            let coeffs = vec![-Q16_ONE / 41; order];
            let exc: Vec<i32> = (0..48)
                .map(|k| if k % 2 == 0 { Q16_ONE } else { -Q16_ONE })
                .collect();
            lpc(&mut h[3], &coeffs, Q16_ONE, &exc);
            lpc(&mut h[3], &coeffs, Q16_ONE, &[0; 48]);
            lpc(&mut h[3], &coeffs, Q16_ONE, &[]);
        }

        // Rounding edge inputs (see `rounding_edges`)
        let edges = rounding_edges();
        for &(a, b) in &edges.mul {
            for v in mul_4x(&[a, a, -a, 0], &[b, -b, b, 0]) {
                fnv1a(&mut h[0], &v.to_le_bytes());
            }
            fnv1a(&mut h[1], &q16_dot_product_neon(&[a], &[b]).to_le_bytes());
            fnv1a(
                &mut h[1],
                &q16_dot_product_neon(&[a, 1, -1], &[b, 7, 7]).to_le_bytes(),
            );
        }
        for (x, y) in &edges.cosine {
            fnv1a(&mut h[2], &q16_cosine_similarity_neon(x, y).to_le_bytes());
        }
        for (coeffs, exc) in &edges.lpc {
            lpc(&mut h[3], coeffs, Q16_ONE, exc);
        }
        h
    }

    /// Inputs where every Q16 shift site sees a sum whose low 16 bits are 0xFFFF, so a
    /// kernel that rounds (`(sum + 1) >> 16`) instead of flooring (`sum >> 16`) gives a
    /// different result. Each case asserts the property of its own input. Random inputs
    /// hit this only with probability 2^-16 per sum.
    struct RoundingEdges {
        mul: Vec<(i32, i32)>,
        cosine: Vec<(
            [i32; crate::types::EMBEDDING_DIM],
            [i32; crate::types::EMBEDDING_DIM],
        )>,
        lpc: Vec<(Vec<i32>, Vec<i32>)>,
    }

    fn rounding_edges() -> RoundingEdges {
        use crate::types::EMBEDDING_DIM;

        let low16 = |v: i64| v & 0xFFFF == 0xFFFF;
        let mul = vec![
            (1, 0xFFFF),
            (-1, 1),
            (0x7FFE_FFFF, 1),
            (0xFFFF, 0x1_0001),
            (-0x1_0001, 1),
        ];
        for &(a, b) in &mul {
            assert!(low16(i64::from(a) * i64::from(b)), "mul edge {a} {b}");
        }

        let norm = |v: &[i32; EMBEDDING_DIM]| -> i64 {
            v.iter().map(|&x| i64::from(x) * i64::from(x)).sum()
        };
        // squared norm 2^16 + 0xFFFF (256^2 + 255^2 + 22^2 + 5^2 + 1)
        let mut ca = [0i32; EMBEDDING_DIM];
        ca[..5].copy_from_slice(&[256, 255, 22, 5, 1]);
        assert!(
            low16(norm(&ca)) && norm(&ca) >> 16 == 1,
            "cosine norm edge 1"
        );
        // squared norm 3 * 2^16 + 0xFFFF: rounding it up to 4 moves sqrt across an
        // integer (1.73 -> 2.0), so the denominator changes (511^2 + 31^2 + 7^2 + 3^2 + 3)
        let mut cc = [0i32; EMBEDDING_DIM];
        cc[..7].copy_from_slice(&[511, 31, 7, 3, 1, 1, 1]);
        assert!(
            low16(norm(&cc)) && norm(&cc) >> 16 == 3,
            "cosine norm edge 3"
        );
        let mut neg = ca;
        for v in neg.iter_mut() {
            *v = -*v;
        }
        // (ca, ca): the dot sum and both squared norms are norm(ca)
        let cosine = vec![(ca, ca), (cc, ca), (ca, cc), (cc, cc), (ca, neg)];

        // LPC: the taps j >= order % 4 are summed in 64 bits and shifted once (the
        // others are rounded one by one), so a single coefficient 1 at j = order % 4 makes
        // that sum at sample order + 1 + j equal y[order] = excitation[order] (gain 1.0)
        let mut lpc = Vec::new();
        for order in [4, 5, 6, 7, 8, 31, 32, 33, 36, 40] {
            let j = order % 4;
            for edge in [0xFFFF, -1, 0x1_FFFF] {
                assert!(low16(i64::from(edge)), "lpc edge {edge}");
                let mut coeffs = vec![0i32; order];
                coeffs[j] = 1;
                let mut exc = vec![0i32; order + j + 4];
                exc[order] = edge;
                lpc.push((coeffs, exc));
            }
        }
        RoundingEdges { mul, cosine, lpc }
    }

    /// The rounding edges through both backends (catches a rounding change in either)
    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    #[test]
    fn scalar_matches_arm_on_rounding_edges() {
        use crate::types::Q16_ONE;

        let edges = rounding_edges();
        for &(a, b) in &edges.mul {
            let (x4, y4) = ([a, a, -a, 0], [b, -b, b, 0]);
            assert_eq!(
                scalar::q16_mul_4x_neon(&x4, &y4),
                arm_mul_4x(&x4, &y4),
                "mul_4x {a} {b}"
            );
            assert_eq!(
                scalar::q16_dot_product_neon(&[a], &[b]),
                arm::q16_dot_product_neon(&[a], &[b]),
                "dot {a} {b}"
            );
        }
        for (i, (x, y)) in edges.cosine.iter().enumerate() {
            assert_eq!(
                scalar::q16_cosine_similarity_neon(x, y),
                arm::q16_cosine_similarity_neon(x, y),
                "cosine edge {i}"
            );
        }
        for (coeffs, exc) in &edges.lpc {
            check_lpc(coeffs, Q16_ONE, exc, "rounding edge");
        }
    }

    /// Golden digests of the canonical kernels, checked on every target (no cfg),
    /// so the x86_64 scalar backend is pinned even where the NEON kernel cannot run.
    ///
    /// Recorded from aarch64 NEON (the canonical behavior). To regenerate on an
    /// aarch64 machine:
    /// `cargo test --lib --no-default-features --features std simd::tests::kernels_match_golden -- --nocapture`
    /// and copy the printed `digests` line.
    #[test]
    fn kernels_match_golden() {
        const GOLDEN: [u64; 4] = [
            0xd330_da71_8fe2_124c,
            0x79f3_2db1_0602_9211,
            0xbde2_0e0f_1770_05e3,
            0xdb3f_44a0_898f_5b37,
        ];
        let got = kernel_digests();
        println!("digests = {got:#018x?}");
        let names = ["mul_4x", "dot", "cosine", "lpc"];
        for k in 0..4 {
            assert_eq!(
                got[k], GOLDEN[k],
                "{} digest differs from the aarch64 NEON golden",
                names[k]
            );
        }
    }
}
