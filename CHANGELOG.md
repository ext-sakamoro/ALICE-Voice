# Changelog

All notable changes to ALICE-Voice will be documented in this file.

## [Unreleased]

### Fixed
- `simd`: x86_64 など ARM 以外の target で `simd::scalar` が存在しない `arm` module を参照して compile できなかった問題を修正 (共通の整数演算 4 関数を `simd::common` へ移し両 backend から再 export)
- `simd`: scalar backend (ARM 以外) と 32-bit ARM の fallback の出力を aarch64 NEON と bit 一致させた 出力が変わるのは scalar の `q16_cosine_similarity_neon` (結果が 2^16 倍になっていた scale の誤りと整数平方根による差) と、scalar / 32-bit ARM の `q16_lpc_filter_neon` (order 4 以上で tap ごとに丸めていたため 1-2 LSB ずれていた) aarch64 の出力は変わらない (NEON の 64-bit 和の最後の加算を wrapping にしたが release build の値は同じで、debug build で overflow 時に panic しなくなる)

- `simd`: aarch64 の `q16_lpc_filter_neon` が order 33 以上で長さ 32 の stack 上の係数 buffer の範囲外を読み書きしていた問題を修正 (safe な公開 API から到達する範囲外書き込み) order 32 を超える場合は共通の定義で計算し、結果は他の order と同じ規則になる

### Changed
- `db` feature: `alice-db` を sibling の path 依存 (`0.2.0-beta.1`) から crates.io の `0.5` に変更し、`alice-db/fs` を有効にする (`AliceDB::open` は alice-db 0.5 で `fs` feature 配下) `fs` が無いと `--features db` が compile できない

### Security
- `pyo3` / `numpy` を 0.28 から 0.29 に更新 pyo3 0.28 の RUSTSEC-2026-0176 (`PyList` / `PyTuple` の iterator の `nth` / `nth_back` で範囲外を読む) と RUSTSEC-2026-0177 (`PyCFunction::new_closure` の closure に `Sync` 制約が無い) は 0.29 で修正済 この crate の `src/python.rs` はどちらの API も使っていないが、依存に脆弱な版を含めないために更新した `src/python.rs` の変更は無い

### Added
- CI: x86_64 と aarch64 の build / test job と、scalar と NEON の結果を bit 単位で比べる test
- `simd` の 4 kernel (mul_4x / dot / cosine / LPC) の出力を aarch64 NEON で記録した digest と照合する test (全 target で実行、x86_64 の scalar backend も固定する)

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
