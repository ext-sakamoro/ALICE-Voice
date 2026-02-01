//! ALICE-Voice Codec Benchmarks
//!
//! Measures performance of L1-L2 layers and SIMD operations.
//!
//! Note: L3 Semantic Layer benchmarks are available in the Commercial version.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use alice_voice::layers::{
    SpectralLayer, SpectralParams,
    ParametricLayer, ParametricParams,
};
use alice_voice::simd::arm::{
    q16_dot_product_neon,
    q16_cosine_similarity_neon,
    q16_lpc_filter_neon,
};
use alice_voice::types::EMBEDDING_DIM;

// ============================================
// Test Audio Generation
// ============================================

/// Generate synthetic voice-like audio (voiced speech simulation)
fn generate_voice_audio(samples: usize, sample_rate: u32) -> Vec<f32> {
    let mut audio = Vec::with_capacity(samples);
    let f0 = 150.0; // Fundamental frequency

    for i in 0..samples {
        let t = i as f32 / sample_rate as f32;
        // Fundamental + harmonics (simulates glottal pulse train)
        let fundamental = (2.0 * std::f32::consts::PI * f0 * t).sin();
        let h2 = (2.0 * std::f32::consts::PI * f0 * 2.0 * t).sin() * 0.5;
        let h3 = (2.0 * std::f32::consts::PI * f0 * 3.0 * t).sin() * 0.3;
        let h4 = (2.0 * std::f32::consts::PI * f0 * 4.0 * t).sin() * 0.2;
        audio.push((fundamental + h2 + h3 + h4) * 0.3);
    }
    audio
}

// ============================================
// L1: Spectral Layer Benchmarks
// ============================================

fn bench_spectral_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("L1_Spectral");

    // Test different frame sizes
    for frame_size in [256, 512, 1024].iter() {
        let hop_size = frame_size / 2;
        let sample_rate = 16000;
        let duration_sec = 1.0;
        let samples = (sample_rate as f32 * duration_sec) as usize;

        let audio = generate_voice_audio(samples, sample_rate);
        let mut layer = SpectralLayer::new(*frame_size, hop_size);

        group.throughput(Throughput::Elements(samples as u64));

        // Encode benchmark
        group.bench_with_input(
            BenchmarkId::new("encode", frame_size),
            &audio,
            |b, audio| {
                b.iter(|| {
                    layer.analyze_stream(black_box(audio)).unwrap()
                })
            },
        );

        // Get encoded data for decode benchmark
        let encoded = layer.analyze_stream(&audio).unwrap();

        // Decode benchmark
        group.bench_with_input(
            BenchmarkId::new("decode", frame_size),
            &encoded,
            |b, encoded| {
                b.iter(|| {
                    layer.synthesize_stream(black_box(encoded))
                })
            },
        );
    }

    group.finish();
}

// ============================================
// L2: Parametric Layer Benchmarks
// ============================================

fn bench_parametric_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("L2_Parametric");

    // Test different LPC orders
    for lpc_order in [10, 16, 24].iter() {
        let sample_rate = 16000;
        let frame_size = 1024; // 64ms for pitch detection
        let duration_sec = 1.0;
        let samples = (sample_rate as f32 * duration_sec) as usize;

        let audio = generate_voice_audio(samples, sample_rate);
        let mut layer = ParametricLayer::new(*lpc_order, frame_size, sample_rate);

        group.throughput(Throughput::Elements(samples as u64));

        // Encode benchmark
        group.bench_with_input(
            BenchmarkId::new("encode", lpc_order),
            &audio,
            |b, audio| {
                b.iter(|| {
                    layer.analyze_stream(black_box(audio), frame_size / 2).unwrap()
                })
            },
        );

        // Get encoded data for decode benchmark
        let encoded = layer.analyze_stream(&audio, frame_size / 2).unwrap();

        // Decode benchmark
        group.bench_with_input(
            BenchmarkId::new("decode", lpc_order),
            &encoded,
            |b, encoded| {
                b.iter(|| {
                    layer.synthesize_stream(black_box(encoded), frame_size / 2)
                })
            },
        );
    }

    group.finish();
}

// ============================================
// SIMD Function Benchmarks
// ============================================

fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD");

    // Q16 Dot Product
    for size in [64, 256, 1024, 4096].iter() {
        let a: Vec<i32> = (0..*size).map(|i| (i as f32 * 0.001 * 65536.0) as i32).collect();
        let b: Vec<i32> = (0..*size).map(|i| ((i as f32 * 0.002).sin() * 65536.0) as i32).collect();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("dot_product", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    q16_dot_product_neon(black_box(a), black_box(b))
                })
            },
        );
    }

    // Q16 Cosine Similarity (256-dim embeddings)
    {
        let mut a = [0i32; EMBEDDING_DIM];
        let mut b = [0i32; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            a[i] = ((i as f32 * 0.01).sin() * 65536.0) as i32;
            b[i] = ((i as f32 * 0.015).cos() * 65536.0) as i32;
        }

        group.bench_function("cosine_similarity_256", |bench| {
            bench.iter(|| {
                q16_cosine_similarity_neon(black_box(&a), black_box(&b))
            })
        });
    }

    // Q16 LPC Filter
    for order in [10, 16, 24].iter() {
        let coeffs: Vec<i32> = (0..*order).map(|i| ((i as f32 * 0.1).sin() * 32768.0) as i32).collect();
        let gain = 65536; // 1.0 in Q16
        let excitation: Vec<i32> = (0..1024).map(|i| if i % 80 == 0 { 65536 } else { 0 }).collect();
        let mut output = vec![0i32; 1024];

        group.throughput(Throughput::Elements(1024));

        group.bench_with_input(
            BenchmarkId::new("lpc_filter", order),
            &(&coeffs, &excitation),
            |bench, (coeffs, excitation)| {
                bench.iter(|| {
                    q16_lpc_filter_neon(
                        black_box(coeffs),
                        black_box(gain),
                        black_box(excitation),
                        black_box(&mut output),
                    )
                })
            },
        );
    }

    group.finish();
}

// ============================================
// End-to-End Pipeline Benchmarks
// ============================================

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pipeline");

    let sample_rate = 16000;
    let duration_sec = 1.0;
    let samples = (sample_rate as f32 * duration_sec) as usize;
    let audio = generate_voice_audio(samples, sample_rate);

    group.throughput(Throughput::Elements(samples as u64));

    // L1 Full Pipeline
    {
        let mut layer = SpectralLayer::new(512, 256);
        group.bench_function("L1_roundtrip", |b| {
            b.iter(|| {
                let encoded = layer.analyze_stream(black_box(&audio)).unwrap();
                layer.synthesize_stream(black_box(&encoded))
            })
        });
    }

    // L2 Full Pipeline
    {
        let mut layer = ParametricLayer::new(10, 1024, sample_rate);
        group.bench_function("L2_roundtrip", |b| {
            b.iter(|| {
                let encoded = layer.analyze_stream(black_box(&audio), 512).unwrap();
                layer.synthesize_stream(black_box(&encoded), 512)
            })
        });
    }

    // Note: L3 Semantic Layer benchmarks are in the Commercial version

    group.finish();
}

// ============================================
// Criterion Main
// ============================================

criterion_group!(
    benches,
    bench_spectral_layer,
    bench_parametric_layer,
    bench_simd_operations,
    bench_full_pipeline,
);

criterion_main!(benches);
