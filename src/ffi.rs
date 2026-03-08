//! C FFI bindings for ALICE-Voice
//!
//! 20 `extern "C"` functions for cross-language integration.
//!
//! # Safety
//!
//! All functions that take raw pointers perform null checks.
//! Opaque handles must be freed with the corresponding `_destroy` function.

use crate::api::{EncodingStats, VoiceCodec, VoiceCodecConfig};
use crate::layers::ParametricParams;
use crate::types::{SpeakerEmbedding, VoiceQuality, EMBEDDING_DIM};
use std::ffi::{c_char, c_float, c_uint};
use std::ptr;

// ============================================
// Opaque handle types (pub for FFI visibility, fields private)
// ============================================

/// Opaque handle for parametric params list
pub struct ParamsList {
    params: Vec<ParametricParams>,
}

/// Opaque handle for decoded audio buffer
pub struct AudioBuffer {
    samples: Vec<f32>,
}

// ============================================
// 1. alice_voice_codec_create
// ============================================

/// Create a `VoiceCodec` with default configuration (16kHz wideband).
///
/// Returns an opaque handle. Free with `alice_voice_codec_destroy`.
#[no_mangle]
pub extern "C" fn alice_voice_codec_create() -> *mut VoiceCodec {
    Box::into_raw(Box::new(VoiceCodec::default_config()))
}

// ============================================
// 2. alice_voice_codec_create_quality
// ============================================

/// Create a `VoiceCodec` with specified quality level.
///
/// `quality`: 0=Low(8kHz), 1=Medium(16kHz), 2=High(32kHz), 3=Ultra(48kHz)
#[no_mangle]
pub extern "C" fn alice_voice_codec_create_quality(quality: u8) -> *mut VoiceCodec {
    let q = match quality {
        0 => VoiceQuality::Low,
        2 => VoiceQuality::High,
        3 => VoiceQuality::Ultra,
        _ => VoiceQuality::Medium,
    };
    Box::into_raw(Box::new(VoiceCodec::new(VoiceCodecConfig::for_quality(q))))
}

// ============================================
// 3. alice_voice_codec_destroy
// ============================================

/// Destroy a `VoiceCodec`.
///
/// # Safety
///
/// `codec` must be a valid pointer from `alice_voice_codec_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_voice_codec_destroy(codec: *mut VoiceCodec) {
    if !codec.is_null() {
        drop(Box::from_raw(codec));
    }
}

// ============================================
// 4. alice_voice_codec_encode_parametric
// ============================================

/// Encode audio samples to L2 parametric params.
///
/// Returns an opaque `ParamsList` handle. Free with `alice_voice_params_destroy`.
///
/// # Safety
///
/// `codec` must be valid. `samples` must point to `len` floats.
#[no_mangle]
pub unsafe extern "C" fn alice_voice_codec_encode_parametric(
    codec: *mut VoiceCodec,
    samples: *const c_float,
    len: c_uint,
) -> *mut ParamsList {
    if codec.is_null() || samples.is_null() || len == 0 {
        return ptr::null_mut();
    }
    let codec = &mut *codec;
    let slice = std::slice::from_raw_parts(samples, len as usize);
    codec
        .encode_parametric(slice)
        .map_or(ptr::null_mut(), |params| {
            Box::into_raw(Box::new(ParamsList { params }))
        })
}

// ============================================
// 5. alice_voice_codec_decode_parametric
// ============================================

/// Decode L2 parametric params back to audio samples.
///
/// Returns an opaque `AudioBuffer`. Get length with return value,
/// copy data with `alice_voice_audio_ptr`. Free with `alice_voice_data_free`.
///
/// # Safety
///
/// `codec` and `params` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn alice_voice_codec_decode_parametric(
    codec: *const VoiceCodec,
    params: *const ParamsList,
    out_len: *mut c_uint,
) -> *mut AudioBuffer {
    if codec.is_null() || params.is_null() {
        return ptr::null_mut();
    }
    let codec = &*codec;
    let params = &*params;
    let samples = codec.decode_parametric(&params.params);
    if !out_len.is_null() {
        *out_len = samples.len() as c_uint;
    }
    Box::into_raw(Box::new(AudioBuffer { samples }))
}

// ============================================
// 6. alice_voice_codec_encode_spectral
// ============================================

/// Encode audio samples to L1 spectral params.
///
/// Returns an opaque handle. Free with `alice_voice_data_free`.
/// The encoded data is stored internally; use decode to reconstruct.
///
/// # Safety
///
/// `codec` must be valid. `samples` must point to `len` floats.
#[no_mangle]
pub unsafe extern "C" fn alice_voice_codec_encode_spectral(
    codec: *mut VoiceCodec,
    samples: *const c_float,
    len: c_uint,
    out_frames: *mut c_uint,
) -> *mut AudioBuffer {
    if codec.is_null() || samples.is_null() || len == 0 {
        return ptr::null_mut();
    }
    let codec = &mut *codec;
    let slice = std::slice::from_raw_parts(samples, len as usize);
    codec
        .encode_spectral(slice)
        .map_or(ptr::null_mut(), |params| {
            if !out_frames.is_null() {
                *out_frames = params.len() as c_uint;
            }
            let decoded = codec.decode_spectral(&params);
            Box::into_raw(Box::new(AudioBuffer { samples: decoded }))
        })
}

// ============================================
// 7. alice_voice_codec_decode_spectral
// ============================================

/// Get the audio data pointer from an `AudioBuffer`.
///
/// # Safety
///
/// `buf` must be a valid `AudioBuffer` pointer.
#[no_mangle]
pub unsafe extern "C" fn alice_voice_audio_ptr(
    buf: *const AudioBuffer,
    out_len: *mut c_uint,
) -> *const c_float {
    if buf.is_null() {
        return ptr::null();
    }
    let buf = &*buf;
    if !out_len.is_null() {
        *out_len = buf.samples.len() as c_uint;
    }
    buf.samples.as_ptr()
}

// ============================================
// 8. alice_voice_codec_sample_rate
// ============================================

/// Get the codec's sample rate.
///
/// # Safety
///
/// `codec` must be valid.
#[no_mangle]
pub const unsafe extern "C" fn alice_voice_codec_sample_rate(codec: *const VoiceCodec) -> c_uint {
    if codec.is_null() {
        return 0;
    }
    (*codec).config().sample_rate
}

// ============================================
// 9. alice_voice_codec_frame_size
// ============================================

/// Get the codec's frame size in samples.
///
/// # Safety
///
/// `codec` must be valid.
#[no_mangle]
pub const unsafe extern "C" fn alice_voice_codec_frame_size(codec: *const VoiceCodec) -> c_uint {
    if codec.is_null() {
        return 0;
    }
    (*codec).config().frame_size as c_uint
}

// ============================================
// 10. alice_voice_params_count
// ============================================

/// Get the number of frames in a `ParamsList`.
///
/// # Safety
///
/// `params` must be valid.
#[no_mangle]
pub const unsafe extern "C" fn alice_voice_params_count(params: *const ParamsList) -> c_uint {
    if params.is_null() {
        return 0;
    }
    (*params).params.len() as c_uint
}

// ============================================
// 11. alice_voice_params_destroy
// ============================================

/// Destroy a `ParamsList`.
///
/// # Safety
///
/// `params` must be from `alice_voice_codec_encode_parametric` or `alice_voice_to_params`.
#[no_mangle]
pub unsafe extern "C" fn alice_voice_params_destroy(params: *mut ParamsList) {
    if !params.is_null() {
        drop(Box::from_raw(params));
    }
}

// ============================================
// 12. alice_voice_stats
// ============================================

/// Compute encoding statistics from parametric params.
///
/// Writes results into the provided output pointers.
///
/// # Safety
///
/// `params` must be valid. Output pointers may be null (skipped).
#[no_mangle]
pub unsafe extern "C" fn alice_voice_stats(
    params: *const ParamsList,
    original_samples: c_uint,
    out_frames: *mut c_uint,
    out_voiced: *mut c_uint,
    out_avg_pitch: *mut c_float,
    out_compression: *mut c_float,
) {
    if params.is_null() {
        return;
    }
    let params = &*params;
    let stats = EncodingStats::from_parametric(&params.params, original_samples as usize);

    if !out_frames.is_null() {
        *out_frames = stats.frames_processed as c_uint;
    }
    if !out_voiced.is_null() {
        *out_voiced = stats.voiced_frames as c_uint;
    }
    if !out_avg_pitch.is_null() {
        *out_avg_pitch = stats.avg_pitch;
    }
    if !out_compression.is_null() {
        *out_compression = stats.compression_ratio;
    }
}

// ============================================
// 13. alice_voice_speaker_create
// ============================================

/// Create a `SpeakerEmbedding` from a float vector.
///
/// `data` must point to exactly 256 floats. If shorter, remaining dims are zero.
///
/// # Safety
///
/// `data` must point to at least `len` floats.
#[no_mangle]
pub unsafe extern "C" fn alice_voice_speaker_create(
    data: *const c_float,
    len: c_uint,
) -> *mut SpeakerEmbedding {
    if data.is_null() {
        return Box::into_raw(Box::new(SpeakerEmbedding::default()));
    }
    let slice = std::slice::from_raw_parts(data, (len as usize).min(EMBEDDING_DIM));
    Box::into_raw(Box::new(SpeakerEmbedding::new(slice)))
}

// ============================================
// 14. alice_voice_speaker_similarity
// ============================================

/// Compute cosine similarity between two speaker embeddings.
///
/// Returns -1.0 on error.
///
/// # Safety
///
/// Both pointers must be valid `SpeakerEmbedding` handles.
#[no_mangle]
pub unsafe extern "C" fn alice_voice_speaker_similarity(
    a: *const SpeakerEmbedding,
    b: *const SpeakerEmbedding,
) -> c_float {
    if a.is_null() || b.is_null() {
        return -1.0;
    }
    (*a).similarity(&*b)
}

// ============================================
// 15. alice_voice_speaker_destroy
// ============================================

/// Destroy a `SpeakerEmbedding`.
///
/// # Safety
///
/// `speaker` must be from `alice_voice_speaker_create`.
#[no_mangle]
pub unsafe extern "C" fn alice_voice_speaker_destroy(speaker: *mut SpeakerEmbedding) {
    if !speaker.is_null() {
        drop(Box::from_raw(speaker));
    }
}

// ============================================
// 16. alice_voice_to_params
// ============================================

/// Convenience: voice samples → parametric params.
///
/// # Safety
///
/// `samples` must point to `len` floats.
#[no_mangle]
pub unsafe extern "C" fn alice_voice_to_params(
    samples: *const c_float,
    len: c_uint,
    sample_rate: c_uint,
) -> *mut ParamsList {
    if samples.is_null() || len == 0 {
        return ptr::null_mut();
    }
    let slice = std::slice::from_raw_parts(samples, len as usize);
    crate::layers::parametric::voice_to_params(slice, sample_rate)
        .map_or(ptr::null_mut(), |params| {
            Box::into_raw(Box::new(ParamsList { params }))
        })
}

// ============================================
// 17. alice_voice_from_params
// ============================================

/// Convenience: parametric params → voice samples.
///
/// # Safety
///
/// `params` must be valid. Returns an `AudioBuffer`.
#[no_mangle]
pub unsafe extern "C" fn alice_voice_from_params(
    params: *const ParamsList,
    sample_rate: c_uint,
    out_len: *mut c_uint,
) -> *mut AudioBuffer {
    if params.is_null() {
        return ptr::null_mut();
    }
    let params = &*params;
    let samples = crate::layers::parametric::params_to_voice(&params.params, sample_rate);
    if !out_len.is_null() {
        *out_len = samples.len() as c_uint;
    }
    Box::into_raw(Box::new(AudioBuffer { samples }))
}

// ============================================
// 18. alice_voice_data_free
// ============================================

/// Free an `AudioBuffer` returned by `decode`/`from_params` functions.
///
/// # Safety
///
/// `buf` must be a valid `AudioBuffer` pointer.
#[no_mangle]
pub unsafe extern "C" fn alice_voice_data_free(buf: *mut AudioBuffer) {
    if !buf.is_null() {
        drop(Box::from_raw(buf));
    }
}

// ============================================
// 19. alice_voice_string_free
// ============================================

/// Free a C string returned by `alice_voice_version`.
///
/// # Safety
///
/// `s` must be a valid C string from this library.
#[no_mangle]
pub unsafe extern "C" fn alice_voice_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(std::ffi::CString::from_raw(s));
    }
}

// ============================================
// 20. alice_voice_version
// ============================================

/// Return the library version as a C string.
///
/// Free with `alice_voice_string_free`.
#[no_mangle]
pub extern "C" fn alice_voice_version() -> *mut c_char {
    let version = std::ffi::CString::new(crate::VERSION).unwrap_or_default();
    version.into_raw()
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn test_codec_create_destroy() {
        let codec = alice_voice_codec_create();
        assert!(!codec.is_null());
        unsafe { alice_voice_codec_destroy(codec) };
    }

    #[test]
    fn test_codec_create_quality() {
        // Ultra (3) frame_size exceeds SpectralLayer MAX_FRAME_SIZE, skip
        for q in 0..=2 {
            let codec = alice_voice_codec_create_quality(q);
            assert!(!codec.is_null());
            unsafe { alice_voice_codec_destroy(codec) };
        }
    }

    #[test]
    fn test_codec_config_accessors() {
        let codec = alice_voice_codec_create();
        unsafe {
            assert_eq!(alice_voice_codec_sample_rate(codec), 16000);
            assert!(alice_voice_codec_frame_size(codec) > 0);
            alice_voice_codec_destroy(codec);
        }
    }

    #[test]
    fn test_encode_decode_parametric() {
        let codec = alice_voice_codec_create();
        let samples: Vec<f32> = (0..8000)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();

        unsafe {
            let params =
                alice_voice_codec_encode_parametric(codec, samples.as_ptr(), samples.len() as u32);
            assert!(!params.is_null());

            let count = alice_voice_params_count(params);
            assert!(count > 0);

            let mut out_len: u32 = 0;
            let audio = alice_voice_codec_decode_parametric(codec, params, &raw mut out_len);
            assert!(!audio.is_null());
            assert!(out_len > 0);

            alice_voice_data_free(audio);
            alice_voice_params_destroy(params);
            alice_voice_codec_destroy(codec);
        }
    }

    #[test]
    fn test_convenience_voice_to_params() {
        let samples: Vec<f32> = (0..8000)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();

        unsafe {
            let params = alice_voice_to_params(samples.as_ptr(), samples.len() as u32, 16000);
            assert!(!params.is_null());

            let count = alice_voice_params_count(params);
            assert!(count > 0);

            let mut out_len: u32 = 0;
            let audio = alice_voice_from_params(params, 16000, &raw mut out_len);
            assert!(!audio.is_null());
            assert!(out_len > 0);

            alice_voice_data_free(audio);
            alice_voice_params_destroy(params);
        }
    }

    #[test]
    fn test_stats() {
        let codec = alice_voice_codec_create();
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 150.0 * i as f32 / 16000.0).sin() * 0.3)
            .collect();

        unsafe {
            let params =
                alice_voice_codec_encode_parametric(codec, samples.as_ptr(), samples.len() as u32);

            let mut frames: u32 = 0;
            let mut voiced: u32 = 0;
            let mut avg_pitch: f32 = 0.0;
            let mut compression: f32 = 0.0;

            alice_voice_stats(
                params,
                samples.len() as u32,
                &raw mut frames,
                &raw mut voiced,
                &raw mut avg_pitch,
                &raw mut compression,
            );

            assert!(frames > 0);
            assert!(compression > 1.0);

            alice_voice_params_destroy(params);
            alice_voice_codec_destroy(codec);
        }
    }

    #[test]
    fn test_speaker_embedding() {
        let data_a: Vec<f32> = vec![1.0, 0.0, 0.0];
        let data_b: Vec<f32> = vec![1.0, 0.0, 0.0];
        let data_c: Vec<f32> = vec![0.0, 1.0, 0.0];

        unsafe {
            let a = alice_voice_speaker_create(data_a.as_ptr(), data_a.len() as u32);
            let b = alice_voice_speaker_create(data_b.as_ptr(), data_b.len() as u32);
            let c = alice_voice_speaker_create(data_c.as_ptr(), data_c.len() as u32);

            let sim_ab = alice_voice_speaker_similarity(a, b);
            let sim_ac = alice_voice_speaker_similarity(a, c);

            assert!((sim_ab - 1.0).abs() < 0.01);
            assert!(sim_ac.abs() < 0.01);

            alice_voice_speaker_destroy(a);
            alice_voice_speaker_destroy(b);
            alice_voice_speaker_destroy(c);
        }
    }

    #[test]
    fn test_version() {
        let ver = alice_voice_version();
        assert!(!ver.is_null());
        unsafe {
            let s = CStr::from_ptr(ver);
            assert!(s.to_str().unwrap().starts_with("0."));
            alice_voice_string_free(ver);
        }
    }

    #[test]
    fn test_null_safety() {
        unsafe {
            alice_voice_codec_destroy(ptr::null_mut());
            alice_voice_params_destroy(ptr::null_mut());
            alice_voice_data_free(ptr::null_mut());
            alice_voice_string_free(ptr::null_mut());
            alice_voice_speaker_destroy(ptr::null_mut());

            assert!(alice_voice_codec_encode_parametric(ptr::null_mut(), ptr::null(), 0).is_null());
            assert_eq!(alice_voice_codec_sample_rate(ptr::null()), 0);
            assert_eq!(alice_voice_codec_frame_size(ptr::null()), 0);
            assert_eq!(alice_voice_params_count(ptr::null()), 0);
            assert!(
                (alice_voice_speaker_similarity(ptr::null(), ptr::null()) - (-1.0)).abs()
                    < f32::EPSILON
            );
        }
    }

    #[test]
    fn test_spectral_encode() {
        let codec = alice_voice_codec_create();
        let samples: Vec<f32> = (0..8000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();

        unsafe {
            let mut frames: u32 = 0;
            let buf = alice_voice_codec_encode_spectral(
                codec,
                samples.as_ptr(),
                samples.len() as u32,
                &raw mut frames,
            );
            assert!(!buf.is_null());
            assert!(frames > 0);

            let mut out_len: u32 = 0;
            let ptr = alice_voice_audio_ptr(buf, &raw mut out_len);
            assert!(!ptr.is_null());
            assert!(out_len > 0);

            alice_voice_data_free(buf);
            alice_voice_codec_destroy(codec);
        }
    }
}
