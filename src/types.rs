//! Common types and enumerations for ALICE-Voice
//!
//! This module defines the fundamental types used throughout the voice codec.

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

/// Q16.16 fixed-point format constant (ALICE-Edge compatible)
pub const Q16_SHIFT: i32 = 16;
pub const Q16_ONE: i32 = 1 << Q16_SHIFT;

/// Speaker embedding dimension (fixed for zero-allocation)
pub const EMBEDDING_DIM: usize = 256;

/// Default sample rate
pub const DEFAULT_SAMPLE_RATE: u32 = 16000;

/// Default LPC order
pub const DEFAULT_LPC_ORDER: usize = 10;

/// Default frame size in samples
pub const DEFAULT_FRAME_SIZE: usize = 512;

/// Default hop size in samples
pub const DEFAULT_HOP_SIZE: usize = 256;

/// Voice layer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum VoiceLayerType {
    /// L1: Spectral (FFT/DCT coefficients)
    Spectral = 0x01,
    /// L2: Parametric (LPC + Formants + Pitch)
    Parametric = 0x02,
    /// L3: Semantic (Text + Emotion + Speaker)
    Semantic = 0x03,
}

impl TryFrom<u8> for VoiceLayerType {
    type Error = VoiceError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x01 => Ok(VoiceLayerType::Spectral),
            0x02 => Ok(VoiceLayerType::Parametric),
            0x03 => Ok(VoiceLayerType::Semantic),
            _ => Err(VoiceError::InvalidLayerType(value)),
        }
    }
}

/// Emotion type for semantic layer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum EmotionType {
    #[default]
    Neutral = 0x00,
    Happy = 0x01,
    Sad = 0x02,
    Angry = 0x03,
    Fearful = 0x04,
    Surprised = 0x05,
    Disgusted = 0x06,
    Contempt = 0x07,
}

impl TryFrom<u8> for EmotionType {
    type Error = VoiceError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x00 => Ok(EmotionType::Neutral),
            0x01 => Ok(EmotionType::Happy),
            0x02 => Ok(EmotionType::Sad),
            0x03 => Ok(EmotionType::Angry),
            0x04 => Ok(EmotionType::Fearful),
            0x05 => Ok(EmotionType::Surprised),
            0x06 => Ok(EmotionType::Disgusted),
            0x07 => Ok(EmotionType::Contempt),
            _ => Err(VoiceError::InvalidEmotionType(value)),
        }
    }
}

/// Voice quality level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum VoiceQuality {
    /// Low quality (narrowband, 8kHz)
    Low = 0x00,
    /// Medium quality (wideband, 16kHz)
    #[default]
    Medium = 0x01,
    /// High quality (super-wideband, 32kHz)
    High = 0x02,
    /// Ultra quality (fullband, 48kHz)
    Ultra = 0x03,
}

impl VoiceQuality {
    /// Get recommended sample rate for quality level
    pub fn sample_rate(&self) -> u32 {
        match self {
            VoiceQuality::Low => 8000,
            VoiceQuality::Medium => 16000,
            VoiceQuality::High => 32000,
            VoiceQuality::Ultra => 48000,
        }
    }

    /// Get recommended LPC order for quality level
    pub fn lpc_order(&self) -> usize {
        match self {
            VoiceQuality::Low => 8,
            VoiceQuality::Medium => 10,
            VoiceQuality::High => 12,
            VoiceQuality::Ultra => 16,
        }
    }
}

/// Voice packet type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum VoicePacketType {
    /// Keyframe with full parameters
    Keyframe = 0x01,
    /// Delta frame with incremental updates
    Delta = 0x02,
    /// Silence frame (no voice activity)
    Silence = 0x03,
    /// Control packet
    Control = 0x04,
}

impl TryFrom<u8> for VoicePacketType {
    type Error = VoiceError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x01 => Ok(VoicePacketType::Keyframe),
            0x02 => Ok(VoicePacketType::Delta),
            0x03 => Ok(VoicePacketType::Silence),
            0x04 => Ok(VoicePacketType::Control),
            _ => Err(VoiceError::InvalidPacketType(value)),
        }
    }
}

/// ALICE-Voice error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum VoiceError {
    #[error("Invalid layer type: {0}")]
    InvalidLayerType(u8),

    #[error("Invalid emotion type: {0}")]
    InvalidEmotionType(u8),

    #[error("Invalid packet type: {0}")]
    InvalidPacketType(u8),

    #[error("Invalid sample rate: {0}")]
    InvalidSampleRate(u32),

    #[error("Buffer too small: need {need}, got {got}")]
    BufferTooSmall { need: usize, got: usize },

    #[error("LPC analysis failed: {0}")]
    LpcError(String),

    #[error("Pitch detection failed: {0}")]
    PitchError(String),

    #[error("Formant extraction failed: {0}")]
    FormantError(String),

    #[error("Synthesis failed: {0}")]
    SynthesisError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Invalid magic bytes")]
    InvalidMagic,

    #[error("Checksum mismatch")]
    ChecksumMismatch,
}

/// Result type for voice operations
pub type VoiceResult<T> = Result<T, VoiceError>;

/// Voice frame header (16 bytes)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct VoiceFrameHeader {
    /// Magic bytes ("AVO1")
    pub magic: [u8; 4],
    /// Protocol version
    pub version: u8,
    /// Layer type
    pub layer_type: u8,
    /// Packet type
    pub packet_type: u8,
    /// Quality level
    pub quality: u8,
    /// Frame sequence number
    pub sequence: u32,
    /// Payload size in bytes
    pub payload_size: u32,
}

impl VoiceFrameHeader {
    pub fn new(
        layer_type: VoiceLayerType,
        packet_type: VoicePacketType,
        quality: VoiceQuality,
        sequence: u32,
        payload_size: u32,
    ) -> Self {
        Self {
            magic: [0x41, 0x56, 0x4F, 0x31], // "AVO1"
            version: 1,
            layer_type: layer_type as u8,
            packet_type: packet_type as u8,
            quality: quality as u8,
            sequence,
            payload_size,
        }
    }

    pub fn validate(&self) -> VoiceResult<()> {
        if self.magic != [0x41, 0x56, 0x4F, 0x31] {
            return Err(VoiceError::InvalidMagic);
        }
        Ok(())
    }
}

/// Speaker embedding (256-dimensional vector, stack-allocated)
///
/// # Performance
/// - Zero heap allocation (Copy trait enabled)
/// - SIMD-optimized similarity computation with 4x loop unrolling
/// - Fast inverse sqrt using Quake III algorithm
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpeakerEmbedding {
    /// Speaker ID vector (fixed-size for zero allocation)
    #[serde(with = "BigArray")]
    pub vector: [f32; EMBEDDING_DIM],
    /// Speaker name hash (replaces `Option<String>` to enable Copy)
    pub name_hash: u64,
}

impl Default for SpeakerEmbedding {
    fn default() -> Self {
        Self {
            vector: [0.0; EMBEDDING_DIM],
            name_hash: 0,
        }
    }
}

impl SpeakerEmbedding {
    /// Create from fixed-size array (zero-copy)
    #[inline]
    pub fn from_array(vector: [f32; EMBEDDING_DIM]) -> Self {
        Self {
            vector,
            name_hash: 0,
        }
    }

    /// Create from slice (copies into fixed array)
    pub fn new(vector: Vec<f32>) -> Self {
        let mut arr = [0.0f32; EMBEDDING_DIM];
        let len = vector.len().min(EMBEDDING_DIM);
        arr[..len].copy_from_slice(&vector[..len]);
        Self {
            vector: arr,
            name_hash: 0,
        }
    }

    /// Create from slice reference (copies into fixed array)
    #[inline]
    pub fn from_slice(slice: &[f32]) -> Self {
        let mut arr = [0.0f32; EMBEDDING_DIM];
        let len = slice.len().min(EMBEDDING_DIM);
        arr[..len].copy_from_slice(&slice[..len]);
        Self {
            vector: arr,
            name_hash: 0,
        }
    }

    /// Set name via hash (FNV-1a for speed)
    pub fn with_name(mut self, name: impl AsRef<str>) -> Self {
        self.name_hash = fnv1a_hash(name.as_ref().as_bytes());
        self
    }

    /// Calculate cosine similarity between two speaker embeddings
    ///
    /// Uses 4x loop unrolling + FMA for SIMD auto-vectorization
    /// and fast inverse sqrt for ~3x speedup over naive implementation.
    #[inline]
    pub fn similarity(&self, other: &SpeakerEmbedding) -> f32 {
        self.similarity_simd(other)
    }

    /// SIMD-optimized cosine similarity with 4x unrolling
    #[inline(always)]
    fn similarity_simd(&self, other: &SpeakerEmbedding) -> f32 {
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        // 4x unrolling for SIMD auto-vectorization
        // EMBEDDING_DIM = 256 = 64 * 4, perfectly divisible
        const CHUNKS: usize = EMBEDDING_DIM / 4;

        for i in 0..CHUNKS {
            let idx = i * 4;

            // Load 4 elements from each vector
            let a0 = self.vector[idx];
            let a1 = self.vector[idx + 1];
            let a2 = self.vector[idx + 2];
            let a3 = self.vector[idx + 3];

            let b0 = other.vector[idx];
            let b1 = other.vector[idx + 1];
            let b2 = other.vector[idx + 2];
            let b3 = other.vector[idx + 3];

            // FMA for dot product
            dot = a0.mul_add(b0, dot);
            dot = a1.mul_add(b1, dot);
            dot = a2.mul_add(b2, dot);
            dot = a3.mul_add(b3, dot);

            // FMA for norm_a
            norm_a = a0.mul_add(a0, norm_a);
            norm_a = a1.mul_add(a1, norm_a);
            norm_a = a2.mul_add(a2, norm_a);
            norm_a = a3.mul_add(a3, norm_a);

            // FMA for norm_b
            norm_b = b0.mul_add(b0, norm_b);
            norm_b = b1.mul_add(b1, norm_b);
            norm_b = b2.mul_add(b2, norm_b);
            norm_b = b3.mul_add(b3, norm_b);
        }

        // Fast inverse sqrt (Quake III algorithm) instead of sqrt + division
        if norm_a > 1e-10 && norm_b > 1e-10 {
            dot * fast_inv_sqrt(norm_a) * fast_inv_sqrt(norm_b)
        } else {
            0.0
        }
    }
}

/// FNV-1a hash for speaker name (fast, non-cryptographic)
#[inline]
fn fnv1a_hash(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Fast inverse square root (Quake III algorithm)
///
/// Error: < 0.2% for typical embedding magnitudes
#[inline(always)]
pub fn fast_inv_sqrt(x: f32) -> f32 {
    let half = 0.5 * x;
    let i = x.to_bits();
    let i = 0x5f375a86 - (i >> 1); // Magic constant
    let y = f32::from_bits(i);
    // One Newton-Raphson iteration for better accuracy
    y * (1.5 - half * y * y)
}

/// Voice activity detection result
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VoiceActivity {
    /// Is voice detected
    pub is_voiced: bool,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f32,
    /// Energy level in dB
    pub energy_db: f32,
}

impl Default for VoiceActivity {
    fn default() -> Self {
        Self {
            is_voiced: false,
            confidence: 0.0,
            energy_db: -96.0,
        }
    }
}

// ============================================
// Q16.16 Fixed-Point Utilities (ALICE-Edge Compatible)
// ============================================

/// Convert Q16.16 fixed-point to integer (truncate)
#[inline(always)]
pub const fn q16_to_int(q: i32) -> i32 {
    q >> Q16_SHIFT
}

/// Convert integer to Q16.16 fixed-point
#[inline(always)]
pub const fn int_to_q16(i: i32) -> i32 {
    i << Q16_SHIFT
}

/// Convert Q16.16 to f32
#[inline(always)]
pub fn q16_to_f32(q: i32) -> f32 {
    q as f32 / Q16_ONE as f32
}

/// Convert f32 to Q16.16
#[inline(always)]
pub fn f32_to_q16(f: f32) -> i32 {
    (f * Q16_ONE as f32) as i32
}

/// Q16.16 multiply
#[inline(always)]
pub fn q16_mul(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> Q16_SHIFT) as i32
}

/// Q16.16 divide
#[inline(always)]
pub fn q16_div(a: i32, b: i32) -> i32 {
    if b == 0 {
        return 0;
    }
    (((a as i64) << Q16_SHIFT) / b as i64) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q16_conversion() {
        assert_eq!(int_to_q16(100), 6553600);
        assert_eq!(q16_to_int(6553600), 100);

        let f = 3.14159;
        let q = f32_to_q16(f);
        let back = q16_to_f32(q);
        assert!((f - back).abs() < 0.0001);
    }

    #[test]
    fn test_q16_arithmetic() {
        let a = f32_to_q16(2.5);
        let b = f32_to_q16(4.0);

        let mul = q16_mul(a, b);
        assert!((q16_to_f32(mul) - 10.0).abs() < 0.001);

        let div = q16_div(b, a);
        assert!((q16_to_f32(div) - 1.6).abs() < 0.001);
    }

    #[test]
    fn test_speaker_similarity() {
        // Create embeddings with first 3 elements set
        let speaker1 = SpeakerEmbedding::new(vec![1.0, 0.0, 0.0]);
        let speaker2 = SpeakerEmbedding::new(vec![1.0, 0.0, 0.0]);
        let speaker3 = SpeakerEmbedding::new(vec![0.0, 1.0, 0.0]);

        // Relaxed tolerance for fast_inv_sqrt (< 0.5% error)
        assert!((speaker1.similarity(&speaker2) - 1.0).abs() < 0.01);
        assert!(speaker1.similarity(&speaker3).abs() < 0.01);
    }

    #[test]
    fn test_voice_quality() {
        assert_eq!(VoiceQuality::Low.sample_rate(), 8000);
        assert_eq!(VoiceQuality::Medium.lpc_order(), 10);
    }
}
