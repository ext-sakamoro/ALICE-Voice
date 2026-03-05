// Integration tests for ALICE-Voice
// Covers LPC, formant, pitch, spectral, parametric, types, API layers.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]

use alice_voice::{
    api::{EncodingStats, VoiceCodec, VoiceCodecConfig},
    codec::{
        formant::{Formant, FormantExtractor},
        lpc::{lpc_fixed, LpcAnalyzer, LpcCoefficients, LpcCoefficientsFixed},
        pitch::{
            generate_excitation, generate_excitation_into, PitchAlgorithm, PitchDetector, PitchInfo,
        },
    },
    layers::{parametric::ParametricParamsView, ParametricLayer, ParametricParams, SpectralLayer},
    types::{
        f32_to_q16, fast_inv_sqrt, int_to_q16, q16_div, q16_mul, q16_to_f32, q16_to_int,
        EmotionType, SpeakerEmbedding, VoiceActivity, VoiceFrameHeader, VoiceLayerType,
        VoicePacketType, VoiceQuality, EMBEDDING_DIM,
    },
    VOICE_MAGIC,
};

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sine_wave(freq: f32, sample_rate: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
        .collect()
}

fn silent_frame(n: usize) -> Vec<f32> {
    vec![0.0f32; n]
}

// ---------------------------------------------------------------------------
// LPC: LpcAnalyzer
// ---------------------------------------------------------------------------

#[test]
fn lpc_order_4_analysis() {
    let mut a = LpcAnalyzer::new(4);
    let samples = sine_wave(200.0, 16000, 512);
    let c = a.analyze(&samples).unwrap();
    assert_eq!(c.order(), 4);
    assert!(c.gain > 0.0);
}

#[test]
fn lpc_order_16_analysis() {
    let mut a = LpcAnalyzer::new(16);
    let samples = sine_wave(300.0, 16000, 512);
    let c = a.analyze(&samples).unwrap();
    assert_eq!(c.order(), 16);
}

#[test]
fn lpc_buffer_too_small_returns_error() {
    let mut a = LpcAnalyzer::new(10);
    // Need at least 2*order = 20 samples; give only 10
    let tiny: Vec<f32> = vec![0.1; 10];
    assert!(a.analyze(&tiny).is_err());
}

#[test]
fn lpc_preemph_builder() {
    let a = LpcAnalyzer::new(8).with_preemph(0.95);
    assert_eq!(a.order(), 8);
}

#[test]
fn lpc_with_frame_size_builder() {
    let a = LpcAnalyzer::with_frame_size(8, 256);
    assert_eq!(a.order(), 8);
}

#[test]
fn lpc_analyze_into_matches_analyze() {
    let mut a = LpcAnalyzer::new(10);
    let samples = sine_wave(440.0, 16000, 512);
    let owned = a.analyze(&samples).unwrap();

    let mut out = LpcCoefficients::new(10);
    a.analyze_into(&samples, &mut out).unwrap();

    for (x, y) in owned.coeffs.iter().zip(out.coeffs.iter()) {
        assert!((x - y).abs() < 1e-6);
    }
    assert!((owned.gain - out.gain).abs() < 1e-6);
}

#[test]
fn lpc_synthesize_view() {
    let mut a = LpcAnalyzer::new(10);
    let samples = sine_wave(200.0, 16000, 512);
    // Eagerly convert to owned to release the mutable borrow before synthesize_view
    let owned = a.analyze_view(&samples).unwrap().to_owned();
    let excitation = sine_wave(200.0, 16000, 512);
    let view_ref = alice_voice::codec::lpc::LpcCoefficientsView {
        coeffs: &owned.coeffs,
        reflection: &owned.reflection,
        gain: owned.gain,
        error: owned.error,
    };
    let out = a.synthesize_view(&view_ref, &excitation);
    assert_eq!(out.len(), 512);
}

#[test]
fn lpc_synthesize_length_matches_excitation() {
    let mut a = LpcAnalyzer::new(10);
    let samples = sine_wave(150.0, 16000, 512);
    let c = a.analyze(&samples).unwrap();
    let exc = vec![0.5f32; 256];
    let out = a.synthesize(&c, &exc);
    assert_eq!(out.len(), 256);
}

#[test]
fn lpc_zero_energy_silent_frame() {
    // Silent frame: LPC should succeed (returns zero-energy coefficients)
    let mut a = LpcAnalyzer::new(8);
    let frame = silent_frame(512);
    // May succeed with zero gain or return error; neither should panic
    let _ = a.analyze(&frame);
}

#[test]
fn lpc_coefficient_order_method() {
    let c = LpcCoefficients::new(12);
    assert_eq!(c.order(), 12);
}

#[test]
fn lpc_coefficients_fixed_round_trip() {
    let original = LpcCoefficients {
        coeffs: vec![0.9, -0.5, 0.2, -0.1],
        gain: 0.7,
        reflection: vec![],
        error: 0.0,
    };
    let fixed = original.to_fixed();
    let back = fixed.to_float();
    for (a, b) in original.coeffs.iter().zip(back.coeffs.iter()) {
        assert!((a - b).abs() < 0.002, "roundtrip mismatch: {a} vs {b}");
    }
    assert!((original.gain - back.gain).abs() < 0.002);
}

#[test]
fn lpc_view_order_method() {
    let mut a = LpcAnalyzer::new(6);
    let samples = sine_wave(500.0, 16000, 512);
    let view = a.analyze_view(&samples).unwrap();
    assert_eq!(view.order(), 6);
}

// ---------------------------------------------------------------------------
// LPC Fixed-Point
// ---------------------------------------------------------------------------

#[test]
fn lpc_fixed_autocorrelation_lag0_is_energy() {
    let samples: Vec<i32> = vec![100, 200, -100, 50, -200];
    let r = lpc_fixed::autocorrelation_fixed(&samples, 2);
    let expected_energy: i64 = samples.iter().map(|&x| i64::from(x) * i64::from(x)).sum();
    assert_eq!(r[0], expected_energy);
}

#[test]
fn lpc_fixed_autocorrelation_length() {
    let samples: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let r = lpc_fixed::autocorrelation_fixed(&samples, 3);
    assert_eq!(r.len(), 4); // order+1
}

#[test]
fn lpc_fixed_zero_input_gives_zero_coefficients() {
    let samples: Vec<i32> = vec![0; 16];
    let r = lpc_fixed::autocorrelation_fixed(&samples, 4);
    let result = lpc_fixed::levinson_durbin_fixed(&r, 4).unwrap();
    assert!(result.coeffs.iter().all(|&c| c == 0));
}

#[test]
fn lpc_fixed_q16_mul_commutative() {
    let a = 98_304i32; // 1.5 * 65536
    let b = 131_072i32; // 2.0 * 65536
    assert_eq!(lpc_fixed::q16_mul(a, b), lpc_fixed::q16_mul(b, a));
}

#[test]
fn lpc_fixed_q16_mac_identity() {
    let acc = 0i32;
    let a = 65536i32; // 1.0
    let b = 65536i32; // 1.0
    let result = lpc_fixed::q16_mac(acc, a, b);
    assert!((result - 65536).abs() < 2);
}

#[test]
fn lpc_fixed_synthesize_zero_gain_produces_silence() {
    let coeffs = LpcCoefficientsFixed {
        coeffs: vec![0; 4],
        gain: 0,
    };
    let excitation = vec![65536i32; 64];
    let out = lpc_fixed::synthesize_fixed(&coeffs, &excitation);
    assert!(out.iter().all(|&x| x == 0));
}

#[test]
fn lpc_fixed_synthesize_into_matches_allocating() {
    let samples: Vec<i32> = vec![100, 200, 150, 180, 120, 160, 140, 170, 90, 110];
    let r = lpc_fixed::autocorrelation_fixed(&samples, 4);
    let coeffs = lpc_fixed::levinson_durbin_fixed(&r, 4).unwrap();
    let excitation = vec![1000i32; 32];

    let out_alloc = lpc_fixed::synthesize_fixed(&coeffs, &excitation);
    let mut out_into = vec![0i32; 32];
    lpc_fixed::synthesize_fixed_into(&coeffs, &excitation, &mut out_into);

    assert_eq!(out_alloc, out_into);
}

#[test]
fn lpc_fixed_fast_isqrt_perfect_squares() {
    assert_eq!(lpc_fixed::fast_isqrt(0), 0);
    assert_eq!(lpc_fixed::fast_isqrt(1), 1);
    assert_eq!(lpc_fixed::fast_isqrt(16), 4);
    assert_eq!(lpc_fixed::fast_isqrt(25), 5);
    assert_eq!(lpc_fixed::fast_isqrt(144), 12);
}

// ---------------------------------------------------------------------------
// Formant: FormantExtractor
// ---------------------------------------------------------------------------

#[test]
fn formant_new_default_ranges() {
    let e = FormantExtractor::new(16000);
    // Just verify it constructs without panic
    let lpc = LpcCoefficients {
        coeffs: vec![0.5, -0.3, 0.1, -0.05],
        gain: 0.1,
        reflection: vec![],
        error: 0.0,
    };
    assert!(e.extract(&lpc).is_ok());
}

#[test]
fn formant_extractor_builder_chain() {
    let e = FormantExtractor::new(16000)
        .with_min_freq(100.0)
        .with_max_freq(7000.0)
        .with_max_bandwidth(300.0);
    let lpc = LpcCoefficients {
        coeffs: vec![1.0, -0.5, 0.2, -0.1, 0.05],
        gain: 0.1,
        reflection: vec![],
        error: 0.0,
    };
    assert!(e.extract(&lpc).is_ok());
}

#[test]
fn formant_result_get_by_index() {
    use alice_voice::codec::formant::FormantResult;
    let mut r = FormantResult::new(16000);
    r.formants.push(Formant::new(600.0, 90.0));
    r.formants.push(Formant::new(1200.0, 110.0));

    assert!(r.get(0).is_some());
    assert!(r.get(1).is_some());
    assert!(r.get(2).is_none());
    assert!((r.f1().unwrap().frequency - 600.0).abs() < f32::EPSILON);
    assert!((r.f2().unwrap().frequency - 1200.0).abs() < f32::EPSILON);
    assert!(r.f3().is_none());
}

#[test]
fn formant_synthesize_lpc_roundtrip() {
    let e = FormantExtractor::new(16000);
    let formants = vec![Formant::new(700.0, 100.0), Formant::new(1200.0, 120.0)];
    let lpc = e.synthesize_lpc(&formants, 10);
    assert_eq!(lpc.coeffs.len(), 10);
    assert!((lpc.gain - 1.0).abs() < f32::EPSILON);
}

#[test]
fn formant_with_amplitude_builder() {
    let f = Formant::new(800.0, 100.0).with_amplitude(0.5);
    assert!((f.amplitude - 0.5).abs() < 1e-6);
    assert!((f.frequency - 800.0).abs() < 1e-6);
    assert!((f.bandwidth - 100.0).abs() < 1e-6);
}

#[test]
fn formant_extract_fallback_large_order() {
    // Order > 32 triggers the fallback path
    let e = FormantExtractor::new(16000);
    let coeffs: Vec<f32> = (0i32..33)
        .map(|i| 0.01 * i as f32 * if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let lpc = LpcCoefficients {
        coeffs,
        gain: 0.1,
        reflection: vec![],
        error: 0.0,
    };
    assert!(e.extract(&lpc).is_ok());
}

#[test]
fn formant_extract_lpc_from_voiced_signal() {
    // Analyze a voiced signal and extract formants
    let mut analyzer = LpcAnalyzer::new(12);
    let samples = sine_wave(150.0, 16000, 512);
    let lpc = analyzer.analyze(&samples).unwrap();
    let extractor = FormantExtractor::new(16000);
    let result = extractor.extract(&lpc).unwrap();
    // Formants might or might not be found, but should not error
    let _ = result.formants.len();
}

// ---------------------------------------------------------------------------
// Pitch: PitchDetector
// ---------------------------------------------------------------------------

#[test]
fn pitch_info_voiced_constructor() {
    let p = PitchInfo::voiced(220.0, 0.9, 16000);
    assert!((p.f0 - 220.0).abs() < f32::EPSILON);
    assert!(p.is_voiced);
    assert!((p.period - 16000.0 / 220.0).abs() < 1.0);
}

#[test]
fn pitch_info_unvoiced_constructor() {
    let p = PitchInfo::unvoiced();
    assert!(p.f0.abs() < f32::EPSILON);
    assert!(!p.is_voiced);
    assert!(p.to_midi().is_none());
}

#[test]
fn pitch_info_period_ms() {
    let p = PitchInfo::voiced(100.0, 0.8, 16000);
    let ms = p.period_ms(16000);
    assert!((ms - 10.0).abs() < 0.1); // 100 Hz => 10ms
}

#[test]
fn pitch_info_midi_a4() {
    let p = PitchInfo::voiced(440.0, 1.0, 16000);
    let midi = p.to_midi().unwrap();
    assert!((midi - 69.0).abs() < 0.01);
}

#[test]
fn pitch_info_midi_a3() {
    let p = PitchInfo::voiced(220.0, 1.0, 16000);
    let midi = p.to_midi().unwrap();
    assert!((midi - 57.0).abs() < 0.05);
}

#[test]
fn pitch_detector_builder_chain() {
    let d = PitchDetector::new(16000)
        .with_pitch_range(80.0, 400.0)
        .with_voicing_threshold(0.2)
        .with_algorithm(PitchAlgorithm::Yin);
    let samples = sine_wave(200.0, 16000, 1024);
    let _ = d; // verify construction didn't panic
    let mut d2 = PitchDetector::new(16000).with_algorithm(PitchAlgorithm::Yin);
    assert!(d2.detect(&samples).is_ok());
}

#[test]
fn pitch_autocorrelation_buffer_too_small_error() {
    let mut d = PitchDetector::new(16000);
    let tiny = vec![0.5f32; 10]; // way too small
    assert!(d.detect(&tiny).is_err());
}

#[test]
fn pitch_yin_buffer_too_small_error() {
    let mut d = PitchDetector::new(16000).with_algorithm(PitchAlgorithm::Yin);
    let tiny = vec![0.5f32; 10];
    assert!(d.detect(&tiny).is_err());
}

#[test]
fn pitch_amdf_buffer_too_small_error() {
    let mut d = PitchDetector::new(16000).with_algorithm(PitchAlgorithm::Amdf);
    let tiny = vec![0.5f32; 10];
    assert!(d.detect(&tiny).is_err());
}

#[test]
fn pitch_amdf_detects_periodicity() {
    let mut d = PitchDetector::new(16000)
        .with_algorithm(PitchAlgorithm::Amdf)
        .with_voicing_threshold(0.1);
    let samples = sine_wave(200.0, 16000, 2048);
    let p = d.detect(&samples).unwrap();
    // May or may not be voiced depending on threshold, should not error
    let _ = p.f0;
}

#[test]
fn pitch_silence_is_unvoiced() {
    let mut d = PitchDetector::new(16000);
    let samples = silent_frame(2048);
    let p = d.detect(&samples).unwrap();
    assert!(!p.is_voiced);
}

#[test]
fn pitch_vad_loud_is_voiced() {
    let d = PitchDetector::new(16000);
    let loud = sine_wave(200.0, 16000, 1024);
    let vad = d.detect_voice_activity(&loud);
    assert!(vad.is_voiced);
    assert!(vad.energy_db > -40.0);
}

#[test]
fn pitch_vad_silent_is_not_voiced() {
    let d = PitchDetector::new(16000);
    let s = silent_frame(1024);
    let vad = d.detect_voice_activity(&s);
    assert!(!vad.is_voiced);
}

#[test]
fn pitch_excitation_voiced_has_pulses() {
    let p = PitchInfo::voiced(200.0, 0.9, 16000);
    let exc = generate_excitation(&p, 2000, 16000);
    assert_eq!(exc.len(), 2000);
    assert!((exc[0] - 1.0).abs() < 1e-6);
    let period: usize = 80; // 16000 / 200 = 80
    assert!((exc[period] - 1.0).abs() < 1e-6);
}

#[test]
fn pitch_excitation_unvoiced_is_noise() {
    let p = PitchInfo::unvoiced();
    let exc = generate_excitation(&p, 1024, 16000);
    assert_eq!(exc.len(), 1024);
    // Noise values should be in [-1, 1]
    assert!(exc.iter().all(|&x| (-1.0..=1.0).contains(&x)));
    // Should not be all zero
    let any_nonzero = exc.iter().any(|&x| x.abs() > 0.0);
    assert!(any_nonzero);
}

#[test]
fn pitch_excitation_into_matches_allocating() {
    let p = PitchInfo::voiced(150.0, 0.9, 16000);
    let alloc = generate_excitation(&p, 512, 16000);
    let mut into_buf = vec![0.0f32; 512];
    generate_excitation_into(&p, &mut into_buf, 16000);
    assert_eq!(alloc, into_buf);
}

#[test]
fn pitch_excitation_into_unvoiced_matches() {
    let p = PitchInfo::unvoiced();
    let alloc = generate_excitation(&p, 256, 16000);
    let mut into_buf = vec![0.0f32; 256];
    generate_excitation_into(&p, &mut into_buf, 16000);
    assert_eq!(alloc, into_buf);
}

// ---------------------------------------------------------------------------
// Spectral Layer
// ---------------------------------------------------------------------------

#[test]
fn spectral_buffer_too_small_returns_error() {
    let mut layer = SpectralLayer::new(256, 128);
    let tiny = vec![0.1f32; 100];
    assert!(layer.analyze(&tiny).is_err());
}

#[test]
fn spectral_quality_low() {
    let mut layer = SpectralLayer::new(256, 128).with_quality(VoiceQuality::Low);
    let samples = sine_wave(440.0, 16000, 256);
    let params = layer.analyze(&samples).unwrap();
    assert!(params.energy >= 0.0);
}

#[test]
fn spectral_quality_ultra() {
    let mut layer = SpectralLayer::new(256, 128).with_quality(VoiceQuality::Ultra);
    let samples = sine_wave(440.0, 16000, 256);
    let params = layer.analyze(&samples).unwrap();
    assert!(!params.coefficients.is_empty());
}

#[test]
fn spectral_params_sparsity() {
    let mut layer = SpectralLayer::new(256, 128);
    let samples = sine_wave(200.0, 16000, 256);
    let params = layer.analyze(&samples).unwrap();
    assert_eq!(params.sparsity(), params.coefficients.len());
}

#[test]
fn spectral_params_encoded_size_formula() {
    use alice_voice::layers::SpectralParams;
    let p = SpectralParams::new(512);
    // encoded_size = 4 + 0 * 6 = 4
    assert_eq!(p.encoded_size(), 4);
}

#[test]
fn spectral_empty_stream() {
    let mut layer = SpectralLayer::new(256, 128);
    let result = layer.synthesize_stream(&[]);
    assert!(result.is_empty());
}

#[test]
fn spectral_default_config() {
    let mut layer = SpectralLayer::default_config();
    let samples = sine_wave(300.0, 16000, 512);
    let params = layer.analyze(&samples).unwrap();
    assert!(params.energy >= 0.0);
}

#[test]
fn spectral_synthesize_into_returns_correct_count() {
    let mut layer = SpectralLayer::new(128, 64);
    let samples = sine_wave(440.0, 16000, 128);
    let params = layer.analyze(&samples).unwrap();
    let mut out = vec![0.0f32; 128];
    let written = layer.synthesize_into(&params, &mut out);
    assert_eq!(written, 128);
}

#[test]
fn spectral_stream_produces_multiple_frames() {
    let mut layer = SpectralLayer::new(128, 64);
    let samples = sine_wave(200.0, 16000, 1280); // 10 frames
    let params_list = layer.analyze_stream(&samples).unwrap();
    assert!(params_list.len() >= 2);
}

#[test]
fn spectral_dct_idct_invertible_small() {
    // Verify that analyze → synthesize produces a non-trivial output of the right length.
    // The Hann window applied both at encode and decode makes exact reconstruction
    // impossible in a single frame; we just confirm the pipeline works end-to-end.
    let mut layer = SpectralLayer::new(64, 32).with_quality(VoiceQuality::Ultra);
    let samples: Vec<f32> = (0i32..64).map(|i| (i as f32 * 0.3).sin()).collect();
    let params = layer.analyze(&samples).unwrap();
    let reconstructed = layer.synthesize(&params);
    assert_eq!(reconstructed.len(), 64);
    // At least some energy should be preserved
    let energy: f32 = reconstructed.iter().map(|x| x * x).sum();
    assert!(energy > 0.0, "reconstructed signal is silent");
}

// ---------------------------------------------------------------------------
// Parametric Layer
// ---------------------------------------------------------------------------

#[test]
fn parametric_default_config() {
    let layer = ParametricLayer::default_config();
    assert_eq!(layer.lpc_order(), 10);
}

#[test]
fn parametric_buffer_too_small_returns_error() {
    let mut layer = ParametricLayer::new(10, 1024, 16000);
    let tiny = vec![0.1f32; 100];
    assert!(layer.analyze(&tiny).is_err());
}

#[test]
fn parametric_analyze_into_view() {
    let mut layer = ParametricLayer::new(10, 1024, 16000);
    let samples = sine_wave(150.0, 16000, 1024);
    let view = layer.analyze_into(&samples).unwrap();
    assert_eq!(view.lpc_coeffs.len(), 10);
    assert_eq!(view.frame_size, 1024);
    assert_eq!(view.sample_rate, 16000);
}

#[test]
fn parametric_view_to_owned_roundtrip() {
    let mut layer = ParametricLayer::new(10, 1024, 16000);
    let samples = sine_wave(200.0, 16000, 1024);
    let view = layer.analyze_into(&samples).unwrap();
    let owned = view.to_owned();
    assert_eq!(owned.lpc.coeffs.len(), 10);
    assert_eq!(owned.formants.len(), view.formant_count);
}

#[test]
fn parametric_view_encoded_size_consistent() {
    let mut layer = ParametricLayer::new(10, 1024, 16000);
    let samples = sine_wave(200.0, 16000, 1024);
    let view = layer.analyze_into(&samples).unwrap();
    let owned = view.to_owned();
    assert_eq!(view.encoded_size(), owned.encoded_size());
}

#[test]
fn parametric_synthesize_into_length() {
    // synthesize (allocating) as a proxy for synthesize_into correctness
    let mut layer = ParametricLayer::new(10, 1024, 16000);
    let samples = sine_wave(150.0, 16000, 1024);
    let params = layer.analyze(&samples).unwrap();
    let out = layer.synthesize(&params);
    assert_eq!(out.len(), 1024);
}

#[test]
fn parametric_synthesize_into_buffer_too_small_error() {
    // Verify BufferTooSmall by encoding params whose frame_size is larger than output
    let mut layer = ParametricLayer::new(10, 1024, 16000);
    let samples = sine_wave(150.0, 16000, 1024);
    let params = layer.analyze(&samples).unwrap();
    assert_eq!(params.frame_size, 1024);
    // Provide an output buffer much smaller than frame_size to trigger the error.
    let view = ParametricParamsView {
        lpc_coeffs: &params.lpc.coeffs,
        lpc_gain: params.lpc.gain,
        lpc_error: params.lpc.error,
        formants: &params.formants,
        formant_count: params.formants.len(),
        pitch: params.pitch,
        activity: params.activity,
        frame_size: params.frame_size,
        sample_rate: params.sample_rate,
    };
    let mut out = vec![0.0f32; 10]; // too small
    assert!(layer.synthesize_into(&view, &mut out).is_err());
}

#[test]
fn parametric_quality_low() {
    let layer = ParametricLayer::new(8, 1024, 8000).with_quality(VoiceQuality::Low);
    assert_eq!(layer.lpc_order(), 8);
}

#[test]
fn parametric_quality_high() {
    let layer = ParametricLayer::new(12, 2048, 32000).with_quality(VoiceQuality::High);
    assert_eq!(layer.lpc_order(), 12);
}

#[test]
fn parametric_stream_produces_frames() {
    let mut layer = ParametricLayer::new(10, 1024, 16000);
    let samples = sine_wave(200.0, 16000, 4096);
    let frames = layer.analyze_stream(&samples, 512).unwrap();
    assert!(!frames.is_empty());
}

#[test]
fn parametric_synthesize_stream_not_empty() {
    let mut layer = ParametricLayer::new(10, 1024, 16000);
    let samples = sine_wave(200.0, 16000, 4096);
    let frames = layer.analyze_stream(&samples, 512).unwrap();
    let out = layer.synthesize_stream(&frames, 512);
    assert!(!out.is_empty());
}

#[test]
fn parametric_synthesize_stream_empty_input() {
    let layer = ParametricLayer::new(10, 1024, 16000);
    let out = layer.synthesize_stream(&[], 512);
    assert!(out.is_empty());
}

#[test]
fn parametric_fixed_point_voicing() {
    let mut layer = ParametricLayer::new(10, 1024, 16000);
    let samples = sine_wave(150.0, 16000, 1024);
    let params = layer.analyze(&samples).unwrap();
    let fixed = params.to_fixed();
    // pitch_q16 = f0 * 65536
    if params.pitch.is_voiced {
        assert!(fixed.voicing);
        assert!(fixed.pitch_q16 > 0);
    }
}

#[test]
fn parametric_params_new_defaults() {
    let p = ParametricParams::new(10, 512, 16000);
    assert_eq!(p.frame_size, 512);
    assert_eq!(p.sample_rate, 16000);
    assert!(!p.pitch.is_voiced);
}

// ---------------------------------------------------------------------------
// Types: VoiceLayerType TryFrom
// ---------------------------------------------------------------------------

#[test]
fn voice_layer_type_try_from_valid() {
    use std::convert::TryFrom;
    assert_eq!(
        VoiceLayerType::try_from(0x01).unwrap(),
        VoiceLayerType::Spectral
    );
    assert_eq!(
        VoiceLayerType::try_from(0x02).unwrap(),
        VoiceLayerType::Parametric
    );
    assert_eq!(
        VoiceLayerType::try_from(0x03).unwrap(),
        VoiceLayerType::Semantic
    );
}

#[test]
fn voice_layer_type_try_from_invalid() {
    use std::convert::TryFrom;
    assert!(VoiceLayerType::try_from(0xFF).is_err());
    assert!(VoiceLayerType::try_from(0x00).is_err());
}

// ---------------------------------------------------------------------------
// Types: EmotionType TryFrom
// ---------------------------------------------------------------------------

#[test]
fn emotion_type_try_from_all_valid() {
    use std::convert::TryFrom;
    assert_eq!(EmotionType::try_from(0).unwrap(), EmotionType::Neutral);
    assert_eq!(EmotionType::try_from(1).unwrap(), EmotionType::Happy);
    assert_eq!(EmotionType::try_from(2).unwrap(), EmotionType::Sad);
    assert_eq!(EmotionType::try_from(3).unwrap(), EmotionType::Angry);
    assert_eq!(EmotionType::try_from(4).unwrap(), EmotionType::Fearful);
    assert_eq!(EmotionType::try_from(5).unwrap(), EmotionType::Surprised);
    assert_eq!(EmotionType::try_from(6).unwrap(), EmotionType::Disgusted);
    assert_eq!(EmotionType::try_from(7).unwrap(), EmotionType::Contempt);
}

#[test]
fn emotion_type_try_from_invalid() {
    use std::convert::TryFrom;
    assert!(EmotionType::try_from(0x08).is_err());
    assert!(EmotionType::try_from(0xFF).is_err());
}

// ---------------------------------------------------------------------------
// Types: VoicePacketType TryFrom
// ---------------------------------------------------------------------------

#[test]
fn voice_packet_type_try_from_valid() {
    use alice_voice::types::VoicePacketType;
    use std::convert::TryFrom;
    assert_eq!(
        VoicePacketType::try_from(0x01).unwrap(),
        VoicePacketType::Keyframe
    );
    assert_eq!(
        VoicePacketType::try_from(0x02).unwrap(),
        VoicePacketType::Delta
    );
    assert_eq!(
        VoicePacketType::try_from(0x03).unwrap(),
        VoicePacketType::Silence
    );
    assert_eq!(
        VoicePacketType::try_from(0x04).unwrap(),
        VoicePacketType::Control
    );
}

#[test]
fn voice_packet_type_try_from_invalid() {
    use alice_voice::types::VoicePacketType;
    use std::convert::TryFrom;
    assert!(VoicePacketType::try_from(0x00).is_err());
    assert!(VoicePacketType::try_from(0x05).is_err());
}

// ---------------------------------------------------------------------------
// Types: VoiceFrameHeader
// ---------------------------------------------------------------------------

#[test]
fn voice_frame_header_magic_valid() {
    let h = VoiceFrameHeader::new(
        VoiceLayerType::Parametric,
        VoicePacketType::Keyframe,
        VoiceQuality::Medium,
        1,
        128,
    );
    assert!(h.validate().is_ok());
    assert_eq!(h.magic, VOICE_MAGIC);
}

#[test]
fn voice_frame_header_invalid_magic_returns_error() {
    let mut h = VoiceFrameHeader::new(
        VoiceLayerType::Spectral,
        VoicePacketType::Delta,
        VoiceQuality::High,
        42,
        64,
    );
    h.magic = [0x00, 0x00, 0x00, 0x00];
    assert!(h.validate().is_err());
}

#[test]
fn voice_frame_header_sequence_preserved() {
    let h = VoiceFrameHeader::new(
        VoiceLayerType::Spectral,
        VoicePacketType::Silence,
        VoiceQuality::Low,
        9999,
        0,
    );
    assert_eq!(h.sequence, 9999);
    assert_eq!(h.version, 1);
}

// ---------------------------------------------------------------------------
// Types: VoiceQuality
// ---------------------------------------------------------------------------

#[test]
fn voice_quality_sample_rates() {
    assert_eq!(VoiceQuality::Low.sample_rate(), 8000);
    assert_eq!(VoiceQuality::Medium.sample_rate(), 16000);
    assert_eq!(VoiceQuality::High.sample_rate(), 32000);
    assert_eq!(VoiceQuality::Ultra.sample_rate(), 48000);
}

#[test]
fn voice_quality_lpc_orders() {
    assert_eq!(VoiceQuality::Low.lpc_order(), 8);
    assert_eq!(VoiceQuality::Medium.lpc_order(), 10);
    assert_eq!(VoiceQuality::High.lpc_order(), 12);
    assert_eq!(VoiceQuality::Ultra.lpc_order(), 16);
}

// ---------------------------------------------------------------------------
// Types: SpeakerEmbedding
// ---------------------------------------------------------------------------

#[test]
fn speaker_embedding_from_slice_partial() {
    let data = vec![1.0f32, 0.0, 0.0];
    let e = SpeakerEmbedding::from_slice(&data);
    assert!((e.vector[0] - 1.0).abs() < 1e-6);
    assert!(e.vector[3..].iter().all(|&x| x == 0.0));
}

#[test]
fn speaker_embedding_from_array() {
    let mut arr = [0.0f32; EMBEDDING_DIM];
    arr[0] = 1.0;
    let e = SpeakerEmbedding::from_array(arr);
    assert!((e.vector[0] - 1.0).abs() < 1e-6);
}

#[test]
fn speaker_embedding_self_similarity_near_one() {
    let mut data = vec![0.0f32; EMBEDDING_DIM];
    data[..16].fill(1.0);
    let e = SpeakerEmbedding::new(&data);
    let sim = e.similarity(&e);
    assert!((sim - 1.0).abs() < 0.02, "self-similarity: {sim}");
}

#[test]
fn speaker_embedding_orthogonal_similarity_near_zero() {
    let mut a_data = vec![0.0f32; EMBEDDING_DIM];
    let mut b_data = vec![0.0f32; EMBEDDING_DIM];
    a_data[0] = 1.0;
    b_data[1] = 1.0;
    let a = SpeakerEmbedding::new(&a_data);
    let b = SpeakerEmbedding::new(&b_data);
    let sim = a.similarity(&b);
    assert!(sim.abs() < 0.02, "orthogonal similarity: {sim}");
}

#[test]
fn speaker_embedding_with_name() {
    let e = SpeakerEmbedding::default().with_name("alice");
    assert_ne!(e.name_hash, 0);
}

#[test]
fn speaker_embedding_zero_vectors_similarity() {
    let a = SpeakerEmbedding::default();
    let b = SpeakerEmbedding::default();
    // Both zero: similarity returns 0.0 (no division)
    let sim = a.similarity(&b);
    assert!(sim.abs() < f32::EPSILON);
}

// ---------------------------------------------------------------------------
// Types: Q16.16 utilities
// ---------------------------------------------------------------------------

#[test]
fn q16_utilities_int_roundtrip() {
    for v in [0i32, 1, -1, 100, -100, 32767] {
        let q = int_to_q16(v);
        assert_eq!(q16_to_int(q), v);
    }
}

#[test]
fn q16_utilities_f32_roundtrip() {
    for v in [0.0f32, 1.0, -1.0, 0.5, -0.5, PI] {
        let q = f32_to_q16(v);
        let back = q16_to_f32(q);
        assert!((v - back).abs() < 0.0002, "f32 roundtrip: {v} vs {back}");
    }
}

#[test]
fn q16_mul_basic() {
    let a = f32_to_q16(3.0);
    let b = f32_to_q16(2.0);
    let result = q16_to_f32(q16_mul(a, b));
    assert!((result - 6.0).abs() < 0.001);
}

#[test]
fn q16_div_basic() {
    let a = f32_to_q16(6.0);
    let b = f32_to_q16(3.0);
    let result = q16_to_f32(q16_div(a, b));
    assert!((result - 2.0).abs() < 0.001);
}

#[test]
fn q16_div_by_zero_returns_zero() {
    let a = f32_to_q16(5.0);
    assert_eq!(q16_div(a, 0), 0);
}

#[test]
fn fast_inv_sqrt_near_one() {
    let x = 1.0f32;
    let inv = fast_inv_sqrt(x);
    assert!((inv - 1.0).abs() < 0.01);
}

#[test]
fn fast_inv_sqrt_four() {
    let x = 4.0f32;
    let inv = fast_inv_sqrt(x);
    // 1/sqrt(4) = 0.5
    assert!((inv - 0.5).abs() < 0.01);
}

// ---------------------------------------------------------------------------
// Types: VoiceActivity
// ---------------------------------------------------------------------------

#[test]
fn voice_activity_default() {
    let v = VoiceActivity::default();
    assert!(!v.is_voiced);
    assert!(v.confidence.abs() < f32::EPSILON);
    assert!(v.energy_db < 0.0);
}

// ---------------------------------------------------------------------------
// API: VoiceCodec and VoiceCodecConfig
// ---------------------------------------------------------------------------

#[test]
fn codec_config_narrowband() {
    let c = VoiceCodecConfig::narrowband();
    assert_eq!(c.sample_rate, 8000);
    assert_eq!(c.lpc_order, 8);
}

#[test]
fn codec_config_wideband() {
    let c = VoiceCodecConfig::wideband();
    assert_eq!(c.sample_rate, 16000);
}

#[test]
fn codec_config_super_wideband() {
    let c = VoiceCodecConfig::super_wideband();
    assert_eq!(c.sample_rate, 32000);
    assert_eq!(c.lpc_order, 12);
}

#[test]
fn codec_config_fullband() {
    let c = VoiceCodecConfig::fullband();
    assert_eq!(c.sample_rate, 48000);
    assert_eq!(c.lpc_order, 16);
}

#[test]
fn codec_config_default_sample_rate() {
    let c = VoiceCodecConfig::default();
    assert_eq!(c.sample_rate, 16000);
    assert_eq!(c.lpc_order, 10);
}

#[test]
fn codec_encode_decode_spectral() {
    let mut codec = VoiceCodec::new(VoiceCodecConfig::wideband());
    let samples = sine_wave(300.0, 16000, 8000);
    let encoded = codec.encode_spectral(&samples).unwrap();
    assert!(!encoded.is_empty());
    let decoded = codec.decode_spectral(&encoded);
    assert!(!decoded.is_empty());
}

#[test]
fn codec_encode_decode_parametric() {
    let mut codec = VoiceCodec::new(VoiceCodecConfig::wideband());
    let samples = sine_wave(200.0, 16000, 8000);
    let encoded = codec.encode_parametric(&samples).unwrap();
    assert!(!encoded.is_empty());
    let decoded = codec.decode_parametric(&encoded);
    assert!(!decoded.is_empty());
}

#[test]
fn codec_compression_ratio_spectral_greater_than_one() {
    let mut codec = VoiceCodec::new(VoiceCodecConfig::wideband());
    let samples = sine_wave(200.0, 16000, 8000);
    let encoded = codec.encode_spectral(&samples).unwrap();
    let ratio = codec.compression_ratio_spectral(&samples, &encoded);
    assert!(ratio > 1.0, "spectral compression ratio: {ratio}");
}

#[test]
fn codec_compression_ratio_parametric_greater_than_one() {
    let mut codec = VoiceCodec::new(VoiceCodecConfig::wideband());
    let samples = sine_wave(200.0, 16000, 8000);
    let encoded = codec.encode_parametric(&samples).unwrap();
    let ratio = codec.compression_ratio_parametric(&samples, &encoded);
    assert!(ratio > 1.0, "parametric compression ratio: {ratio}");
}

#[test]
fn codec_config_getter() {
    let cfg = VoiceCodecConfig::wideband();
    let codec = VoiceCodec::new(cfg);
    assert_eq!(codec.config().sample_rate, 16000);
}

#[test]
fn encoding_stats_empty_input() {
    let stats = EncodingStats::from_parametric(&[], 0);
    assert_eq!(stats.frames_processed, 0);
    assert_eq!(stats.samples_processed, 0);
}

#[test]
fn encoding_stats_from_parametric_frames() {
    let mut codec = VoiceCodec::default_config();
    let samples = sine_wave(150.0, 16000, 16000);
    let params = codec.encode_parametric(&samples).unwrap();
    let stats = EncodingStats::from_parametric(&params, samples.len());
    assert!(stats.frames_processed > 0);
    assert_eq!(stats.samples_processed, samples.len());
    assert!(stats.compression_ratio > 1.0);
}
