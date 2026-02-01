//! Voice codec modules
//!
//! This module provides the core voice processing algorithms:
//! - LPC (Linear Predictive Coding) analysis and synthesis
//! - Formant extraction and reconstruction
//! - Pitch detection and generation

pub mod lpc;
pub mod formant;
pub mod pitch;

pub use lpc::{LpcAnalyzer, LpcCoefficients};
pub use formant::{FormantExtractor, Formant};
pub use pitch::{PitchDetector, PitchInfo};
