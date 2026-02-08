//! ALICE-Voice × ALICE-ML Bridge
//!
//! Ternary neural inference for voice parameter estimation.
//! Uses 1.58-bit weights for ultra-fast LPC/formant/pitch prediction.

use alice_ml::{TernaryWeight, ternary_matvec};

/// ML-accelerated voice parameter predictor.
///
/// Uses ternary weights (1.58-bit) for zero-multiplication inference
/// of voice codec parameters from spectral features.
pub struct VoicePredictor {
    /// Spectral → LPC coefficient predictor.
    weights_lpc: TernaryWeight,
    /// Spectral → pitch predictor.
    weights_pitch: TernaryWeight,
    /// Input dimension (spectral bins).
    input_dim: usize,
    /// LPC order.
    lpc_order: usize,
}

impl VoicePredictor {
    /// Create a voice predictor from pre-trained ternary weights.
    ///
    /// - `weights_lpc`: ternary values for LPC prediction (lpc_order × input_dim).
    /// - `weights_pitch`: ternary values for pitch prediction (1 × input_dim).
    /// - `input_dim`: number of spectral input features.
    /// - `lpc_order`: LPC coefficient count (typically 10-16).
    pub fn new(
        weights_lpc: &[i8],
        weights_pitch: &[i8],
        input_dim: usize,
        lpc_order: usize,
    ) -> Self {
        Self {
            weights_lpc: TernaryWeight::from_ternary(weights_lpc, lpc_order, input_dim),
            weights_pitch: TernaryWeight::from_ternary(weights_pitch, 1, input_dim),
            input_dim,
            lpc_order,
        }
    }

    /// Predict LPC coefficients from spectral features.
    ///
    /// Zero-allocation: writes directly to `output`.
    pub fn predict_lpc(&self, spectral_input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(spectral_input.len(), self.input_dim);
        debug_assert!(output.len() >= self.lpc_order);
        ternary_matvec(spectral_input, &self.weights_lpc, &mut output[..self.lpc_order]);
    }

    /// Predict pitch from spectral features.
    pub fn predict_pitch(&self, spectral_input: &[f32]) -> f32 {
        debug_assert_eq!(spectral_input.len(), self.input_dim);
        let mut out = [0.0f32; 1];
        ternary_matvec(spectral_input, &self.weights_pitch, &mut out);
        out[0]
    }

    /// Input dimension.
    pub fn input_dim(&self) -> usize { self.input_dim }
    /// LPC order.
    pub fn lpc_order(&self) -> usize { self.lpc_order }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_predictor() {
        // 4 spectral bins → 2 LPC coefficients
        let lpc_weights = [1i8, -1, 0, 1, 0, 1, -1, 0]; // 2×4
        let pitch_weights = [1i8, 0, 1, -1]; // 1×4

        let predictor = VoicePredictor::new(&lpc_weights, &pitch_weights, 4, 2);

        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut lpc_out = [0.0f32; 2];
        predictor.predict_lpc(&input, &mut lpc_out);

        // Row 0: [1,-1,0,1] · [1,2,3,4] = 1-2+0+4 = 3
        // Row 1: [0,1,-1,0] · [1,2,3,4] = 0+2-3+0 = -1
        assert!((lpc_out[0] - 3.0).abs() < 1e-6);
        assert!((lpc_out[1] - (-1.0)).abs() < 1e-6);

        let pitch = predictor.predict_pitch(&input);
        // [1,0,1,-1] · [1,2,3,4] = 1+0+3-4 = 0
        assert!((pitch - 0.0).abs() < 1e-6);
    }
}
