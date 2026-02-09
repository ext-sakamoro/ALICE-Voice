//! ALICE-DB bridge: Voice metrics persistence
//!
//! Stores voice analysis metrics (pitch, gain, voicing probability)
//! into ALICE-DB as time-series for quality monitoring and analytics.
//!
//! # Pipeline
//!
//! ```text
//! ParametricParams → extract metrics → ALICE-DB time-series
//! ALICE-DB → query range → historical voice quality data
//! ```

use alice_db::AliceDB;

use crate::layers::ParametricParams;

/// Persistent store for voice analysis metrics.
pub struct VoiceMetricsSink {
    /// Pitch (f0) values over time
    pitch_db: AliceDB,
    /// LPC gain values over time
    gain_db: AliceDB,
    /// Voicing probability over time
    voicing_db: AliceDB,
}

impl VoiceMetricsSink {
    /// Open or create voice metrics databases at the given directory.
    pub fn open(dir: &str) -> Result<Self, String> {
        let pitch_db = AliceDB::open(format!("{}/pitch", dir))
            .map_err(|e| format!("pitch db: {}", e))?;
        let gain_db = AliceDB::open(format!("{}/gain", dir))
            .map_err(|e| format!("gain db: {}", e))?;
        let voicing_db = AliceDB::open(format!("{}/voicing", dir))
            .map_err(|e| format!("voicing db: {}", e))?;
        Ok(Self {
            pitch_db,
            gain_db,
            voicing_db,
        })
    }

    /// Record voice metrics from a parametric analysis frame.
    pub fn record_frame(&self, timestamp_ms: u64, params: &ParametricParams) {
        let _ = self.pitch_db.put(timestamp_ms as i64, params.pitch.f0);
        let _ = self.gain_db.put(timestamp_ms as i64, params.lpc.gain);
        let _ = self
            .voicing_db
            .put(timestamp_ms as i64, params.pitch.voicing_prob);
    }

    /// Record raw metric values directly.
    pub fn record(&self, timestamp_ms: u64, pitch_f0: f32, gain: f32, voicing_prob: f32) {
        let _ = self.pitch_db.put(timestamp_ms as i64, pitch_f0);
        let _ = self.gain_db.put(timestamp_ms as i64, gain);
        let _ = self.voicing_db.put(timestamp_ms as i64, voicing_prob);
    }

    /// Query pitch history in a time range.
    pub fn query_pitch(&self, from_ms: u64, to_ms: u64) -> Vec<(u64, f32)> {
        self.pitch_db
            .scan(from_ms as i64, to_ms as i64)
            .unwrap_or_default()
            .into_iter()
            .map(|(ts, v)| (ts as u64, v))
            .collect()
    }

    /// Query gain history in a time range.
    pub fn query_gain(&self, from_ms: u64, to_ms: u64) -> Vec<(u64, f32)> {
        self.gain_db
            .scan(from_ms as i64, to_ms as i64)
            .unwrap_or_default()
            .into_iter()
            .map(|(ts, v)| (ts as u64, v))
            .collect()
    }

    /// Query voicing probability history in a time range.
    pub fn query_voicing(&self, from_ms: u64, to_ms: u64) -> Vec<(u64, f32)> {
        self.voicing_db
            .scan(from_ms as i64, to_ms as i64)
            .unwrap_or_default()
            .into_iter()
            .map(|(ts, v)| (ts as u64, v))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_metrics_open() {
        let dir = "/tmp/alice_voice_db_test";
        let result = VoiceMetricsSink::open(dir);
        if result.is_ok() {
            let sink = result.unwrap();
            sink.record(1000, 440.0, 0.8, 0.95);
            let pitches = sink.query_pitch(0, 2000);
            assert!(!pitches.is_empty());
        }
    }
}
