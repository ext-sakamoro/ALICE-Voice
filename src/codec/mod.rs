//! Voice codec modules
//!
//! This module provides the core voice processing algorithms:
//! - LPC (Linear Predictive Coding) analysis and synthesis
//! - Formant extraction and reconstruction
//! - Pitch detection and generation

pub mod formant;
pub mod lpc;
pub mod pitch;

pub use formant::{Formant, FormantExtractor};
pub use lpc::{LpcAnalyzer, LpcCoefficients};
pub use pitch::{PitchDetector, PitchInfo};
