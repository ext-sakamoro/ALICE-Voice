//! ALICE-Voice × ALICE-Text bridge
//!
//! Voice feature metadata extraction and compressed text indexing, and speech hints from text: the
//! statistics of compressing a transcript with ALICE-Text suggest the codec layer and quality.
//!
//! ```text
//! transcript --ALICE-Text compress--> exception rate, compression ratio --> SpeechHints
//!     exception rate < 0.3        -> VoiceMode::Parametric (LPC), else VoiceMode::Spectral
//!     ratio < 0.3 / < 0.6 / else  -> VoiceQuality::High / Medium / Low
//! ```
//!
//! Author: Moroya Sakamoto

use crate::api::VoiceCodecConfig;
use crate::types::VoiceQuality;
use alice_text::{ALICEText, EncodingMode};

/// Voice metadata for speech-to-text indexing
#[derive(Debug, Clone)]
pub struct VoiceTextMetadata {
    pub transcript_hint: String,
    pub pitch_mean: f32,
    pub energy_db: f32,
    pub duration_ms: u32,
}

/// Extract voice metadata from a PCM audio frame.
/// Computes basic spectral features for indexing purposes.
pub fn extract_voice_metadata(frame: &[f32], sample_rate: u32) -> VoiceTextMetadata {
    let duration_ms = (frame.len() as u32 * 1000) / sample_rate.max(1);

    // Energy in dB
    let energy = frame.iter().map(|s| s * s).sum::<f32>() / frame.len().max(1) as f32;
    let energy_db = if energy > 1e-10 {
        10.0 * energy.log10()
    } else {
        -100.0
    };

    // Simple zero-crossing rate as pitch proxy
    let mut zero_crossings = 0u32;
    for i in 1..frame.len() {
        if (frame[i] >= 0.0) != (frame[i - 1] >= 0.0) {
            zero_crossings += 1;
        }
    }
    let pitch_mean =
        (zero_crossings as f32 * sample_rate as f32) / (2.0 * frame.len().max(1) as f32);

    VoiceTextMetadata {
        transcript_hint: String::new(), // Filled by downstream STT
        pitch_mean,
        energy_db,
        duration_ms,
    }
}

/// Compress voice transcript metadata with ALICE-Text
pub fn compress_voice_transcript(metadata: &VoiceTextMetadata) -> Vec<u8> {
    let serialized = format!(
        "{}|{:.1}|{:.1}|{}",
        metadata.transcript_hint, metadata.pitch_mean, metadata.energy_db, metadata.duration_ms
    );
    let mut compressor = ALICEText::new(EncodingMode::Pattern);
    compressor
        .compress(&serialized)
        .unwrap_or_else(|_| serialized.into_bytes())
}

/// Decompress voice transcript metadata
pub fn decompress_voice_transcript(data: &[u8]) -> Result<VoiceTextMetadata, String> {
    let decompressor = ALICEText::new(EncodingMode::Pattern);
    let text = decompressor
        .decompress(data)
        .or_else(|_| String::from_utf8(data.to_vec()).map_err(|e| e.to_string()))?;

    let parts: Vec<&str> = text.splitn(4, '|').collect();
    if parts.len() < 4 {
        return Err("Invalid metadata format".into());
    }

    Ok(VoiceTextMetadata {
        transcript_hint: parts[0].to_string(),
        pitch_mean: parts[1].parse().unwrap_or(0.0),
        energy_db: parts[2].parse().unwrap_or(-100.0),
        duration_ms: parts[3].parse().unwrap_or(0),
    })
}

/// Speech encoding hints derived from compressing the text with ALICE-Text
#[derive(Clone, Debug)]
pub struct SpeechHints {
    /// Estimated phoneme count (one per three bytes, rounded up)
    pub estimated_phonemes: usize,
    /// The text encoder's exception rate (higher = more irregular)
    pub exception_rate: f64,
    /// Suggested codec layer
    pub suggested_mode: VoiceMode,
    /// The text encoder's compression ratio
    pub text_compression_ratio: f64,
    /// Suggested voice quality
    pub suggested_quality: VoiceQuality,
}

/// Suggested voice codec layer
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoiceMode {
    /// L2: parametric (LPC), for regular speech
    Parametric,
    /// L1: spectral (FFT/DCT), for irregular or musical content
    Spectral,
}

/// The hints for a text of `len` bytes that compressed with `exception_rate` and `ratio`
fn hints(len: usize, exception_rate: f64, ratio: f64) -> SpeechHints {
    let suggested_mode = if exception_rate < 0.3 {
        VoiceMode::Parametric
    } else {
        VoiceMode::Spectral
    };
    let suggested_quality = if ratio < 0.3 {
        VoiceQuality::High
    } else if ratio < 0.6 {
        VoiceQuality::Medium
    } else {
        VoiceQuality::Low
    };
    SpeechHints {
        estimated_phonemes: len.div_ceil(3),
        exception_rate,
        suggested_mode,
        text_compression_ratio: ratio,
        suggested_quality,
    }
}

/// Compress `text` and return the bytes with the hints its statistics give (no statistics: rate and
/// ratio 1.0)
pub fn compress_transcript(text: &str) -> (Vec<u8>, SpeechHints) {
    let mut encoder = ALICEText::new(EncodingMode::Pattern);
    let compressed = encoder.compress(text).unwrap_or_default();
    let (exception_rate, ratio) = encoder
        .last_stats()
        .map_or((1.0, 1.0), |s| (s.exception_rate(), s.compression_ratio()));
    (compressed, hints(text.len(), exception_rate, ratio))
}

/// The speech hints of `text` (as [`compress_transcript`], without keeping the bytes)
pub fn analyze_for_speech(text: &str) -> SpeechHints {
    compress_transcript(text).1
}

/// The codec configuration for the hints' suggested quality
pub fn codec_config_from_hints(hints: &SpeechHints) -> VoiceCodecConfig {
    VoiceCodecConfig::for_quality(hints.suggested_quality)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_metadata() {
        let frame: Vec<f32> = (0..320).map(|i| (i as f32 * 0.01).sin()).collect();
        let meta = extract_voice_metadata(&frame, 16000);
        assert!(meta.pitch_mean > 0.0);
        assert!(meta.energy_db > -100.0);
        assert_eq!(meta.duration_ms, 20); // 320 samples / 16000 Hz = 20ms
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        let meta = VoiceTextMetadata {
            transcript_hint: "hello world".into(),
            pitch_mean: 150.0,
            energy_db: -20.5,
            duration_ms: 500,
        };
        let compressed = compress_voice_transcript(&meta);
        let restored = decompress_voice_transcript(&compressed).unwrap();
        assert_eq!(restored.transcript_hint, "hello world");
        assert_eq!(restored.duration_ms, 500);
    }

    #[test]
    fn the_mode_is_parametric_exactly_below_an_exception_rate_of_0_3() {
        // oracle: the rule the hints state (regular text, few exceptions -> parametric LPC)
        for (rate, want) in [
            (0.0, VoiceMode::Parametric),
            (0.299, VoiceMode::Parametric),
            (0.3, VoiceMode::Spectral),
            (1.0, VoiceMode::Spectral),
        ] {
            assert_eq!(
                hints(30, rate, 0.5).suggested_mode,
                want,
                "exception rate {rate}"
            );
        }
    }

    #[test]
    fn the_quality_follows_the_compression_ratio_bands() {
        for (ratio, want) in [
            (0.0, VoiceQuality::High),
            (0.299, VoiceQuality::High),
            (0.3, VoiceQuality::Medium),
            (0.599, VoiceQuality::Medium),
            (0.6, VoiceQuality::Low),
            (1.0, VoiceQuality::Low),
        ] {
            assert_eq!(
                hints(30, 0.0, ratio).suggested_quality,
                want,
                "ratio {ratio}"
            );
        }
    }

    #[test]
    fn phonemes_are_one_per_three_bytes_rounded_up() {
        for (len, want) in [(0, 0), (1, 1), (3, 1), (4, 2), (30, 10)] {
            assert_eq!(
                hints(len, 0.0, 1.0).estimated_phonemes,
                want,
                "length {len}"
            );
        }
    }

    #[test]
    fn analysis_and_compression_give_the_same_hints_and_the_transcript_round_trips() {
        for text in [
            "Hello hello hello hello hello hello hello hello",
            "2024-01-15 INFO Server started on port 8080",
            "",
        ] {
            let a = analyze_for_speech(text);
            let (compressed, b) = compress_transcript(text);
            assert_eq!(
                (a.estimated_phonemes, a.suggested_mode, a.suggested_quality),
                (b.estimated_phonemes, b.suggested_mode, b.suggested_quality),
                "{text:?}"
            );
            assert_eq!(a.exception_rate.to_bits(), b.exception_rate.to_bits());
            assert_eq!(
                a.text_compression_ratio.to_bits(),
                b.text_compression_ratio.to_bits()
            );
            let back = ALICEText::new(EncodingMode::Pattern)
                .decompress(&compressed)
                .unwrap();
            assert_eq!(
                back, text,
                "the compressed transcript decompresses to the text"
            );
        }
    }

    #[test]
    fn the_codec_config_is_the_one_for_the_suggested_quality() {
        for q in [VoiceQuality::Low, VoiceQuality::Medium, VoiceQuality::High] {
            let mut h = hints(10, 0.1, 0.5);
            h.suggested_quality = q;
            let want = VoiceCodecConfig::for_quality(q);
            let got = codec_config_from_hints(&h);
            assert_eq!(
                (got.sample_rate, got.lpc_order),
                (want.sample_rate, want.lpc_order)
            );
        }
    }

    #[test]
    fn test_silent_frame() {
        let frame = vec![0.0f32; 320];
        let meta = extract_voice_metadata(&frame, 16000);
        assert_eq!(meta.energy_db, -100.0);
        assert_eq!(meta.pitch_mean, 0.0);
    }
}
