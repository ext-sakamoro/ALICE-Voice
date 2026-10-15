//! Architecture-independent kernels shared by `arm` and `scalar`
//!
//! The aarch64 NEON kernels in `arm` define the results. Every other backend
//! (the `scalar` module on x86_64 and the 32-bit ARM fallbacks) calls the
//! portable definitions below, which reproduce the NEON results bit for bit:
//! integer sums wrap like the NEON lanes do, and the operations are grouped the
//! same way. `simd::tests::scalar_matches_arm_bit_for_bit` checks this.

#[cfg(any(test, not(target_arch = "aarch64")))]
use crate::types::EMBEDDING_DIM;
use crate::types::Q16_SHIFT;

/// Q16.16 multiply using 64-bit intermediate (scalar)
#[inline(always)]
#[must_use]
pub const fn q16_mul(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> Q16_SHIFT) as i32
}

/// Q16.16 multiply-accumulate (scalar)
#[inline(always)]
#[must_use]
pub const fn q16_mac(acc: i32, a: i32, b: i32) -> i32 {
    acc + q16_mul(a, b)
}

/// Fast integer square root (64-bit input, 32-bit output)
///
/// Uses Newton-Raphson with a good initial guess.
/// Returns floor(sqrt(n)).
#[inline]
#[must_use]
pub const fn fast_isqrt_64(n: u64) -> u32 {
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
pub const fn fast_isqrt_32(n: u32) -> u16 {
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

// Used by the non-aarch64 backends and by the comparison test
#[cfg(any(test, not(target_arch = "aarch64")))]
/// Q16.16 multiply of 4 lanes: each lane is `q16_mul` (64-bit product, shift, truncate)
#[inline]
#[must_use]
pub const fn q16_mul_4x(a: &[i32; 4], b: &[i32; 4]) -> [i32; 4] {
    [
        q16_mul(a[0], b[0]),
        q16_mul(a[1], b[1]),
        q16_mul(a[2], b[2]),
        q16_mul(a[3], b[3]),
    ]
}

// Used by the non-aarch64 backends and by the comparison test
#[cfg(any(test, not(target_arch = "aarch64")))]
/// Q16.16 dot product over the common length of `a` and `b`
///
/// The 64-bit products are summed with wrapping addition and shifted once.
#[inline]
#[must_use]
pub fn q16_dot_product(a: &[i32], b: &[i32]) -> i64 {
    a.iter()
        .zip(b.iter())
        .fold(0i64, |acc, (&x, &y)| acc.wrapping_add(x as i64 * y as i64))
        >> Q16_SHIFT
}

// Used by the non-aarch64 backends and by the comparison test
#[cfg(any(test, not(target_arch = "aarch64")))]
/// Q16.16 cosine similarity
///
/// The dot product and both squared norms are summed in 64 bits with wrapping
/// addition and shifted to Q16 once. The denominator is
/// `(sqrt(|a|^2) * sqrt(|b|^2)) as i64` with IEEE-754 f64 `sqrt` (correctly
/// rounded, so the same on every target), and the result is
/// `(dot << 16) / denom` truncated to `i32`, or 0 when `denom <= 0`.
#[must_use]
pub fn q16_cosine_similarity(a: &[i32; EMBEDDING_DIM], b: &[i32; EMBEDDING_DIM]) -> i32 {
    let mut dot = 0i64;
    let mut norm_a = 0i64;
    let mut norm_b = 0i64;

    for i in 0..EMBEDDING_DIM {
        dot = dot.wrapping_add(a[i] as i64 * b[i] as i64);
        norm_a = norm_a.wrapping_add(a[i] as i64 * a[i] as i64);
        norm_b = norm_b.wrapping_add(b[i] as i64 * b[i] as i64);
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

// Used by the non-aarch64 backends, and by the aarch64 kernel for orders above its buffer
/// Q16.16 all-pole LPC filter: `y[i] = gain * x[i] + sum_j coeffs[j] * y[i - 1 - j]`
///
/// The rounding follows the NEON kernel:
/// - samples `i < order`, and every sample when `order < 4`: one `q16_mac` per tap
///   in tap order `j = 0, 1, ...` (each product shifted separately)
/// - samples `i >= order` when `order >= 4`: the taps `j >= order % 4` are summed
///   as 64-bit products with wrapping addition and shifted once, then the
///   remaining taps `j < order % 4` are added with one `q16_mac` each
pub fn q16_lpc_filter(coeffs: &[i32], gain: i32, excitation: &[i32], output: &mut [i32]) {
    let order = coeffs.len();
    let n = excitation.len().min(output.len());

    for i in 0..order.min(n) {
        let mut sample = q16_mul(gain, excitation[i]);
        for j in 0..i {
            sample = q16_mac(sample, coeffs[j], output[i - 1 - j]);
        }
        output[i] = sample;
    }

    if order >= 4 {
        let remainder = order % 4;
        for i in order..n {
            let mut sample = q16_mul(gain, excitation[i]);
            let mut sum = 0i64;
            for j in remainder..order {
                sum = sum.wrapping_add(coeffs[j] as i64 * output[i - 1 - j] as i64);
            }
            sample += (sum >> Q16_SHIFT) as i32;
            for j in 0..remainder {
                sample = q16_mac(sample, coeffs[j], output[i - 1 - j]);
            }
            output[i] = sample;
        }
    } else {
        for i in order..n {
            let mut sample = q16_mul(gain, excitation[i]);
            for j in 0..order {
                sample = q16_mac(sample, coeffs[j], output[i - 1 - j]);
            }
            output[i] = sample;
        }
    }
}
