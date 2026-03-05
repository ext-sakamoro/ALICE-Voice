# Changelog

All notable changes to ALICE-Voice will be documented in this file.

## [0.1.1] - 2026-03-04

### Added
- `ffi` — 20 `extern "C"` FFI functions (VoiceCodec, ParametricParams, SpeakerEmbedding)
- Unity C# bindings (`bindings/unity/AliceVoice.cs`) — 20 DllImport + RAII classes
- UE5 C++ header (`bindings/ue5/AliceVoice.h`) — 20 extern C + RAII wrappers

### Fixed
- `cargo fmt` trailing whitespace in source files

## [0.1.0] - 2026-02-23

### Added
- `codec/lpc` — LPC analysis (Levinson-Durbin), zero-copy view, Q16.16 fixed-point, DPS
- `codec/formant` — Formant extraction from LPC roots
- `codec/pitch` — Pitch detection (autocorrelation + harmonic product spectrum)
- `layers/spectral` — L1 Spectral layer (FFT/DCT coefficients, 10-50x compression)
- `layers/parametric` — L2 Parametric layer (LPC + Formants + Pitch, 100-600x compression)
- `api` — `VoiceCodec` unified L1/L2 encoding and decoding
- `types` — `SpeakerEmbedding` (fixed-size, Copy), `VoiceActivity`, `VoiceFrameHeader`
- `simd` — SIMD utility helpers
- `python` — PyO3 + NumPy bindings (feature-gated: `python`)
- `ml_bridge` — ALICE-ML ternary inference integration (feature-gated: `ml`)
- `codec_bridge` — ALICE-Codec wavelet + rANS integration (feature-gated: `codec`)
- `db_bridge` — ALICE-DB voice metrics persistence (feature-gated: `db`)
- `text_bridge` — ALICE-Text transcript metadata (feature-gated: `text`)
- Feature flags: `std`, `python`, `ml`, `codec`, `db`, `text`, `no_std`
- 43 unit tests
