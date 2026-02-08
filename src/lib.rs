//! ALICE-Voice: Voice-Specialized Procedural Codec
//!
//! "Don't send waveforms. Send the law of speech."
//!
//! This crate provides ultra-efficient voice transmission through parametric encoding.
//! Instead of transmitting raw audio waveforms, we extract and transmit the mathematical
//! laws governing speech production.
//!
//! # Layer Architecture
//!
//! | Layer | Name | Content | Compression |
//! |-------|------|---------|-------------|
//! | L2 | Parametric | LPC + Formants + Pitch | 100-600x |
//! | L1 | Spectral | FFT/DCT coefficients | 10-50x |
//!
//! Note: L3 Semantic Layer (1000x+ compression) is available under Commercial License.
//! See: https://github.com/ext-sakamoro/ALICE-Voice-Commercial
//!
//! # Example
//!
//! ```ignore
//! use alice_voice::{voice_to_params, params_to_voice};
//!
//! // Analyze voice into parameters
//! let params = voice_to_params(&audio_samples, 16000);
//!
//! // Transmit only the parameters (~50 bytes per frame)
//! send_over_network(&params);
//!
//! // Reconstruct voice from parameters
//! let reconstructed = params_to_voice(&params, 16000);
//! ```
//!
//! # Related Projects
//!
//! - [ALICE-Edge](https://github.com/ext-sakamoro/ALICE-Edge) - Embedded LPC (Q16.16)
//! - [ALICE-Streaming-Protocol](https://github.com/ext-sakamoro/ALICE-Streaming-Protocol) - Video streaming
//! - [ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) - Procedural compression

pub mod types;
pub mod layers;
pub mod codec;
pub mod api;
pub mod simd;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "ml")]
pub mod ml_bridge;

// Re-export main types
pub use types::*;
pub use api::*;

// Re-export layer modules
pub use layers::{
    SpectralLayer, SpectralParams,
    ParametricLayer, ParametricParams,
};

// Re-export codec modules
pub use codec::{
    lpc::{LpcAnalyzer, LpcCoefficients},
    formant::{FormantExtractor, Formant},
    pitch::{PitchDetector, PitchInfo},
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Magic bytes for ALICE-Voice packets
pub const VOICE_MAGIC: [u8; 4] = [0x41, 0x56, 0x4F, 0x31]; // "AVO1"

/// Protocol version
pub const VOICE_VERSION: u8 = 1;
