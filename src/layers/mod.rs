//! Voice encoding layers
//!
//! This module provides two encoding layers with different trade-offs:
//!
//! | Layer | Compression | Quality | Use Case |
//! |-------|-------------|---------|----------|
//! | L1 Spectral | 10-50x | Highest | Studio quality |
//! | L2 Parametric | 100-600x | Good | Real-time communication |
//!
//! Note: L3 Semantic Layer is available under Commercial License.
//! See: <https://github.com/ext-sakamoro/ALICE-Voice-Commercial>

pub mod parametric;
pub mod spectral;

pub use parametric::{ParametricLayer, ParametricParams};
pub use spectral::{SpectralLayer, SpectralParams};
