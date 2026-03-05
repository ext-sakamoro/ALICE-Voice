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

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
pub mod arm;

// Re-export commonly used functions
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
pub use arm::*;

// Fallback scalar implementations for non-ARM targets
#[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
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
}
