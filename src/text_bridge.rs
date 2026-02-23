//! ALICE-Voice × ALICE-Text bridge
//!
//! Voice feature metadata extraction and compressed text indexing.
//!
//! Author: Moroya Sakamoto

use alice_text::ALICEText;

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
    let compressor = ALICEText::new();
    compressor
        .compress(&serialized)
        .unwrap_or_else(|_| serialized.into_bytes())
}

/// Decompress voice transcript metadata
pub fn decompress_voice_transcript(data: &[u8]) -> Result<VoiceTextMetadata, String> {
    let decompressor = ALICEText::new();
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
    fn test_silent_frame() {
        let frame = vec![0.0f32; 320];
        let meta = extract_voice_metadata(&frame, 16000);
        assert_eq!(meta.energy_db, -100.0);
        assert_eq!(meta.pitch_mean, 0.0);
    }
}
