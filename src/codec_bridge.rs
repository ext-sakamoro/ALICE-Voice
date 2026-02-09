//! ALICE-Codec bridge: Wavelet + rANS entropy coding for voice parameters
//!
//! Provides enhanced compression for L2 (Parametric) layer coefficients
//! using ALICE-Codec's integer wavelet transform and rANS entropy coder.
//!
//! # Typical Use
//!
//! ```ignore
//! use alice_voice::codec_bridge::*;
//!
//! let params: Vec<ParametricParams> = codec.encode_parametric(&audio)?;
//! let compressed = compress_lpc_sequence(&params, 4);
//! // compressed.data is ~5-10x smaller than raw ParametricParams
//! let recovered = decompress_lpc_sequence(&compressed, 10);
//! ```

use alice_codec::quant::{build_histogram, from_symbols, to_symbols, Quantizer};
use alice_codec::rans::{FrequencyTable, RansDecoder, RansEncoder};
use alice_codec::Wavelet1D;

use crate::layers::ParametricParams;
use crate::VoiceQuality;

/// Compressed voice frame sequence using wavelet + rANS
#[derive(Debug, Clone)]
pub struct CodecCompressedFrame {
    /// rANS-encoded bitstream
    pub data: Vec<u8>,
    /// Number of original frames
    pub frame_count: usize,
    /// LPC order used
    pub lpc_order: usize,
    /// Quantizer step size
    pub quantizer_step: i32,
    /// Quality level
    pub quality: VoiceQuality,
}

/// Compress a sequence of LPC coefficient vectors using Wavelet1D + rANS.
///
/// Pipeline: f32 coefficients → i32 scale → Wavelet1D → Quantize → rANS
///
/// Returns a compact bitstream suitable for storage or transmission.
pub fn compress_lpc_sequence(
    frames: &[ParametricParams],
    quantizer_step: i32,
) -> CodecCompressedFrame {
    if frames.is_empty() {
        return CodecCompressedFrame {
            data: Vec::new(),
            frame_count: 0,
            lpc_order: 0,
            quantizer_step,
            quality: VoiceQuality::Medium,
        };
    }

    let lpc_order = frames[0].lpc.coeffs.len();
    let quality = VoiceQuality::Medium;

    // Flatten LPC coefficients across frames into a single i32 buffer.
    // Scale f32 → i32 with 16-bit fractional precision for integer wavelet.
    let total_samples = frames.len() * lpc_order;
    let mut signal: Vec<i32> = Vec::with_capacity(total_samples);
    for frame in frames {
        for &c in &frame.lpc.coeffs {
            signal.push((c * 32768.0) as i32);
        }
    }

    // Pad to power of 2 for wavelet (store original length)
    let orig_len = signal.len();
    let padded_len = orig_len.next_power_of_two();
    signal.resize(padded_len, 0);

    // Forward wavelet transform (CDF 5/3 — lossless-friendly)
    let wavelet = Wavelet1D::cdf53();
    wavelet.forward(&mut signal);

    // Quantize wavelet coefficients
    let quantizer = Quantizer::new(quantizer_step.max(1));
    let mut quantized = vec![0i32; padded_len];
    quantizer.quantize_buffer(&signal, &mut quantized);

    // Convert to symbols for rANS
    let mut symbols = vec![0u8; padded_len];
    to_symbols(&quantized, &mut symbols);

    // Build frequency table and encode
    let histogram = build_histogram(&symbols);
    let table = FrequencyTable::from_histogram(&histogram);
    let mut encoder = RansEncoder::new();
    encoder.encode_symbols(&symbols, &table);
    let mut encoded = encoder.finish();

    // Prepend histogram (256 × 4 bytes) + orig_len (4 bytes) for decoder
    let mut output = Vec::with_capacity(1028 + encoded.len());
    output.extend_from_slice(&(orig_len as u32).to_le_bytes());
    for &count in &histogram {
        output.extend_from_slice(&count.to_le_bytes());
    }
    output.append(&mut encoded);

    CodecCompressedFrame {
        data: output,
        frame_count: frames.len(),
        lpc_order,
        quantizer_step,
        quality,
    }
}

/// Decompress LPC coefficient sequence from a codec-compressed frame.
///
/// Returns a Vec of LPC coefficient vectors (one per frame).
pub fn decompress_lpc_sequence(compressed: &CodecCompressedFrame) -> Vec<Vec<f32>> {
    if compressed.data.is_empty() || compressed.frame_count == 0 {
        return Vec::new();
    }

    let lpc_order = compressed.lpc_order;

    // Parse header: orig_len + histogram
    let orig_len =
        u32::from_le_bytes(compressed.data[0..4].try_into().unwrap_or([0; 4])) as usize;
    let padded_len = orig_len.next_power_of_two();

    let mut histogram = [0u32; 256];
    for (i, h) in histogram.iter_mut().enumerate() {
        let offset = 4 + i * 4;
        *h = u32::from_le_bytes(
            compressed.data[offset..offset + 4]
                .try_into()
                .unwrap_or([0; 4]),
        );
    }

    let rans_data = &compressed.data[1028..];

    // Decode rANS
    let table = FrequencyTable::from_histogram(&histogram);
    let mut decoder = RansDecoder::new(rans_data);
    let symbols = decoder.decode_n(padded_len, &table);

    // Symbols → quantized coefficients
    let mut quantized = vec![0i32; padded_len];
    from_symbols(&symbols, &mut quantized);

    // Dequantize
    let quantizer = Quantizer::new(compressed.quantizer_step.max(1));
    let mut signal = vec![0i32; padded_len];
    quantizer.dequantize_buffer(&quantized, &mut signal);

    // Inverse wavelet
    let wavelet = Wavelet1D::cdf53();
    wavelet.inverse(&mut signal);

    // Reconstruct per-frame LPC vectors
    let mut result = Vec::with_capacity(compressed.frame_count);
    for frame_idx in 0..compressed.frame_count {
        let start = frame_idx * lpc_order;
        let end = (start + lpc_order).min(orig_len);
        let coeffs: Vec<f32> = signal[start..end]
            .iter()
            .map(|&v| v as f32 / 32768.0)
            .collect();
        result.push(coeffs);
    }

    result
}

/// Estimate compression ratio for a set of parametric frames.
///
/// Returns `(estimated_compressed_bytes, original_bytes)`.
pub fn estimate_compression(frames: &[ParametricParams]) -> (usize, usize) {
    if frames.is_empty() {
        return (0, 0);
    }
    let lpc_order = frames[0].lpc.coeffs.len();
    // Original: each frame has lpc_order * 4 bytes (f32) + pitch (4) + formants (~24) + activity (8)
    let original_per_frame = lpc_order * 4 + 36;
    let original = frames.len() * original_per_frame;

    // Compressed: wavelet + rANS typically achieves 3-8x on correlated LPC data
    let compressed = compress_lpc_sequence(frames, 4);
    (compressed.data.len(), original)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress_empty() {
        let compressed = compress_lpc_sequence(&[], 4);
        assert_eq!(compressed.frame_count, 0);
        let recovered = decompress_lpc_sequence(&compressed);
        assert!(recovered.is_empty());
    }
}
