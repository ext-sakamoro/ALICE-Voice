# ALICE-Voice

**Voice-Specialized Procedural Codec** - "Don't send waveforms. Send the law of speech."

<p align="center">
  <em>Ultra-efficient voice transmission through parametric encoding</em>
</p>

## The Philosophy

Raw audio waveforms **never leave the device**. Instead, we extract the mathematical laws of speech production and transmit only the parameters.

```
Traditional Voice: 16kHz × 16bit × 1sec = 32KB transmitted
ALICE-Voice L2:    LPC(10) + Pitch + Gain = ~50 bytes/frame
                   Compression: 600x+
```

## What Can You Do?

ALICE-Voice is not just an audio compression tool. It's a next-generation voice codec library built on the philosophy of **"sending the blueprint of speech, not the waveform itself"** - ultra-lightweight and blazingly fast.

### 1. Dramatic Data Compression

| Layer | What It Does | Compression | License |
|-------|--------------|-------------|---------|
| **L1 Spectral** | High-quality voice/music transmission | 10-50x | MIT |
| **L2 Parametric** | Extract only vocal parameters (pitch, formants) | 100-600x | MIT |
| **L3 Semantic** | Convert to text + emotion + speaker ID | 1000x+ | [Commercial](https://github.com/ext-sakamoro/ALICE-Voice-Commercial) |

### 2. Runs Anywhere (Zero-Allocation Design)

Thanks to aggressive optimization:

- **Ultra-low latency**: No memory allocation in hot paths = no GC stuttering. Perfect for game voice chat and real-time communication.
- **Embedded-ready**: Q16.16 fixed-point + ARM NEON SIMD means it runs on cheap microcontrollers (Raspberry Pi Pico, STM32) via [ALICE-Edge](https://github.com/ext-sakamoro/ALICE-Edge).

### 3. Privacy Protection & Voice Transformation

Because we don't send waveforms:

- **Anonymization**: Only "speech parameters" are transmitted. The receiver can easily re-synthesize as a **different person's voice**. Your voiceprint never travels over the network.
- **Perfect Noise Removal**: Only voiced/unvoiced speech components are mathematically encoded. Background noise (construction, wind, crowds) is **physically not transmitted**.
- **Voice Accessibility**: People with speech impairments can communicate with a "normal voice" on the receiving end.

### Real-World Use Cases

With ALICE-Voice, you can have clear, uninterrupted voice calls on:

- A smartphone with data throttling
- Satellite links in space
- Battery-powered IoT toys
- Underground/underwater with minimal bandwidth

## Layer Architecture

| Layer | Name | Content | Typical Size | Compression | License |
|-------|------|---------|--------------|-------------|---------|
| L2 | Parametric | LPC coefficients + Formants + Pitch | 50-200 bytes/frame | 100-600x | MIT |
| L1 | Spectral | DCT coefficients | 200-1000 bytes/frame | 10-50x | MIT |

> **Note:** L3 Semantic Layer (1000x+ compression) is available under [Commercial License](https://github.com/ext-sakamoro/ALICE-Voice-Commercial).

## Quick Start

```rust
use alice_voice::{voice_to_params, params_to_voice};

// L2: Voice → Parametric representation
let params = voice_to_params(&audio_samples, 16000)?;
// Transmit only ~50 bytes per frame!

// L2: Parametric → Voice reconstruction
let reconstructed = params_to_voice(&params, 16000);
```

## Layer Details

### L1: Spectral Layer

DCT-based frequency-domain representation with pre-computed matrices.

```rust
use alice_voice::SpectralLayer;

// Create layer (frame_size=512, hop_size=256)
let mut layer = SpectralLayer::new(512, 256);

// Encode
let params = layer.analyze_stream(&audio)?;

// Decode
let reconstructed = layer.synthesize_stream(&params);

// Zero-allocation encode (for real-time)
let params_view = layer.analyze_into(&frame)?;
```

### L2: Parametric Layer (Primary)

Speech production model using Linear Predictive Coding (LPC).

```rust
use alice_voice::ParametricLayer;

// Create layer (lpc_order=10, frame_size=1024, sample_rate=16000)
let mut layer = ParametricLayer::new(10, 1024, 16000);

// Encode
let params = layer.analyze(&audio)?;
// params.lpc: LpcCoefficients (10 coeffs)
// params.pitch: PitchInfo (f0, voicing)
// params.formants: Vec<Formant>
// params.activity: VoiceActivity

// Decode
let reconstructed = layer.synthesize(&params);

// Zero-allocation API (for real-time processing)
let view = layer.analyze_into(&frame)?;  // Returns ParametricParamsView
let mut output = vec![0.0f32; 1024];
layer.synthesize_into(&view, &mut output)?;
```

## Unified Codec API

```rust
use alice_voice::{VoiceCodec, VoiceCodecConfig};

// Create codec with default config (16kHz, LPC order 10)
let mut codec = VoiceCodec::new(VoiceCodecConfig::default());

// Or use quality presets
let mut codec = VoiceCodec::new(VoiceCodecConfig::wideband());    // 16kHz
let mut codec = VoiceCodec::new(VoiceCodecConfig::fullband());    // 48kHz

// Encode/decode with L1-L2 layers
let l1_params = codec.encode_spectral(&audio)?;
let l2_params = codec.encode_parametric(&audio)?;

// Check compression ratio
let ratio = codec.compression_ratio_parametric(&audio, &l2_params);
println!("Compression: {:.0}x", ratio);  // e.g., "Compression: 150x"
```

## Q16.16 Fixed-Point (Embedded Systems)

For embedded systems without FPU, all LPC computations use Q16.16 fixed-point:

```rust
use alice_voice::ParametricParams;

// Convert to fixed-point
let fixed = params.to_fixed();
// fixed.lpc: LpcCoefficientsFixed (Q16.16)
// fixed.pitch_q16: i32 (Q16.16 Hz)
```

## ARM NEON SIMD

Optimized SIMD operations for ARM (Apple Silicon, Cortex-A series):

```rust
use alice_voice::simd::arm::{
    q16_dot_product_neon,
    q16_cosine_similarity_neon,
    q16_lpc_filter_neon,
};

// 256-dim embedding similarity
let similarity = q16_cosine_similarity_neon(&embedding_a, &embedding_b);

// LPC synthesis filter (zero-allocation)
q16_lpc_filter_neon(&coeffs, gain, &excitation, &mut output);
```

## Performance

Benchmarked on Apple Silicon (M-series), 1 second audio @ 16kHz.

### Layer Performance

| Layer | Encode | Decode | Throughput |
|-------|--------|--------|------------|
| L1 Spectral (frame=512) | 16.8 ms | 16.0 ms | 1.0 Melem/s |
| L2 Parametric (order=10) | 5.4 ms | 0.53 ms | 30 Melem/s |

### SIMD Operations (ARM NEON)

| Function | Size | Time | Throughput |
|----------|------|------|------------|
| q16_dot_product | 256 | 23.9 ns | 10.7 Gelem/s |
| q16_dot_product | 4096 | 518.8 ns | 7.9 Gelem/s |
| q16_cosine_similarity | 256-dim | 56.1 ns | 4.6M calls/s |
| q16_lpc_filter | order=10 | 4.5 µs | 226 Melem/s |
| q16_lpc_filter | order=24 | 7.3 µs | 140 Melem/s |

### Pipeline Roundtrip (Encode + Decode)

| Pipeline | Time | Real-time Factor |
|----------|------|------------------|
| L1 Spectral | 36.6 ms | 27x |
| L2 Parametric | 6.8 ms | 147x |

*Real-time factor = 1 second audio / processing time*

### Run Benchmarks

```bash
cargo bench --no-default-features --features std
```

## Installation

### Rust

```toml
[dependencies]
alice-voice = "0.1"
```

### Features

```toml
[dependencies]
alice-voice = { version = "0.1", default-features = false, features = ["std"] }
```

| Feature | Description |
|---------|-------------|
| `std` | Standard library (default) |
| `python` | Python bindings via PyO3 |
| `no_std` | Embedded systems support |
| `ml` | ALICE-ML bridge (ternary NN voice prediction) |
| `db` | ALICE-DB bridge (voice metrics persistence) |

### Python

```bash
cd ALICE-Voice
pip install maturin
maturin develop --release
```

```python
import alice_voice
import numpy as np

audio = np.random.randn(16000).astype(np.float32)  # 1 sec @ 16kHz
params = alice_voice.voice_to_params(audio, 16000)
reconstructed = alice_voice.params_to_voice(params, 16000)
```

## Build Optimization

For maximum performance, the release profile uses:

```toml
[profile.release]
lto = "fat"           # Full link-time optimization
codegen-units = 1     # Single codegen unit
opt-level = 3         # Maximum optimization
panic = "abort"       # No unwinding overhead
```

ARM NEON is automatically enabled on aarch64 targets via `.cargo/config.toml`.

## ASP Integration (ALICE-Streaming-Protocol)

ALICE-Voice is integrated into [ALICE-Streaming-Protocol](https://github.com/ext-sakamoro/ALICE-Streaming-Protocol) as the voice encoding backend via the `voice` feature flag.

```toml
# In ALICE-Streaming-Protocol's Cargo.toml
libasp = { version = "1.0", features = ["voice"] }
```

### How ASP Uses ALICE-Voice

ASP wraps ALICE-Voice's codec API for voice frame encoding within the ASP transport layer:

```
ASP Voice Encode:
  PCM f32 → VoiceCodec::encode_parametric (L2, 100-600x)
           or VoiceCodec::encode_spectral  (L1, 10-50x)
           → Serialize to AudioFrame bytes
           → ASP packet framing

ASP Voice Decode:
  ASP packet → Deserialize AudioFrame
             → VoiceCodec::decode_parametric / decode_spectral
             → PCM f32
```

- **Batch API**: `encode_batch()` for processing multiple voice frames efficiently
- **Pre-allocated serialization**: `Vec::with_capacity` for all serialize/deserialize paths
- **Python bindings**: `libasp.encode_voice()` / `libasp.decode_voice()` with GIL release + NumPy zero-copy

### Standalone vs ASP Usage

| Use Case | Recommended |
|----------|-------------|
| Direct voice processing / embedded | ALICE-Voice standalone |
| Voice within ASP video stream | ASP `media-stack` feature |
| Voice anonymization / transformation | ALICE-Voice standalone (parameter manipulation) |

## Cross-Crate Bridges

ALICE-Voice connects to other ALICE ecosystem crates via feature-gated bridge modules:

| Bridge | Feature | Target Crate | Description |
|--------|---------|--------------|-------------|
| `ml_bridge` | `ml` | [ALICE-ML](../ALICE-ML) | Ternary NN prediction for LPC coefficients and pitch parameters |
| `db_bridge` | `db` | [ALICE-DB](../ALICE-DB) | Voice metrics persistence (pitch/gain/voicing → time-series) |

### ML Bridge (feature: `ml`)

Uses ALICE-ML's ternary matrix-vector operations to predict LPC coefficients and pitch parameters from input feature vectors, enabling neural-enhanced voice coding.

```rust
use alice_voice::ml_bridge::VoicePredictor;

let predictor = VoicePredictor::new(input_dim, lpc_order);

// Predict LPC coefficients from feature vector
let lpc = predictor.predict_lpc(&features);

// Predict pitch parameters
let pitch = predictor.predict_pitch(&features);
```

### DB Bridge (feature: `db`)

Persists per-frame voice metrics (pitch F0, gain, voicing probability) into ALICE-DB time-series storage for analysis, monitoring, or offline tuning.

```rust
use alice_voice::db_bridge::VoiceMetricsSink;

let sink = VoiceMetricsSink::open("/tmp/voice_metrics")?;

// Record from ParametricParams
sink.record_frame(timestamp_ms, &params)?;

// Or record individual metrics
sink.record(timestamp_ms, pitch_f0, gain, voicing_prob)?;

// Query pitch history
let pitch_history = sink.query_pitch(start_ms, end_ms)?;
```

### ALICE-Text Bridge (feature: `text`)

Compressed voice transcript metadata via ALICE-Text exception-based compression.

- `VoiceTextMetadata` — Transcript hint, pitch, energy, duration
- `extract_voice_metadata()` — Extract features from PCM audio frames
- `compress_voice_transcript()` / `decompress_voice_transcript()` — Compress/decompress metadata

Enable: `alice-voice = { features = ["text"] }`

## Related Projects

| Project | Description |
|---------|-------------|
| [ALICE-Voice-Commercial](https://github.com/ext-sakamoro/ALICE-Voice-Commercial) | L3 Semantic Layer (Commercial License) |
| [ALICE-Edge](https://github.com/ext-sakamoro/ALICE-Edge) | Embedded model generator (LPC Q16.16) |
| [ALICE-Streaming-Protocol](https://github.com/ext-sakamoro/ALICE-Streaming-Protocol) | Video streaming (Spectral layer compatible) |
| [ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) | Procedural generation engine |

All projects share the core philosophy: **encode the generation process, not the data itself**.

## License

MIT License

**Note:** L3 Semantic Layer is available under Commercial License at [ALICE-Voice-Commercial](https://github.com/ext-sakamoro/ALICE-Voice-Commercial).

## Author

Moroya Sakamoto

---

*"The best voice codec is one where waveforms never travel."*
