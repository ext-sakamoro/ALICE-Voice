# Contributing to ALICE-Voice

## Build

```bash
cargo build --no-default-features --features std
cargo build                          # includes Python bindings
```

## Test

```bash
cargo test --no-default-features --features std
```

Note: Default `python` feature requires a compatible Python environment for linking.

## Lint

```bash
cargo clippy --no-default-features --features std -- -W clippy::all
cargo fmt -- --check
cargo doc --no-default-features --features std --no-deps 2>&1 | grep warning
```

## Design Constraints

- **Parametric codec**: L2 layer encodes voice as LPC + formants + pitch (100-600x compression vs raw PCM).
- **Zero-copy analysis**: `LpcCoefficientsView` returns references to internal buffers, no allocation.
- **Q16.16 fixed-point**: integer-only LPC path for embedded/edge targets.
- **Destination Passing Style**: `analyze_into()` writes directly to caller-provided buffers.
- **Speaker embedding**: fixed-size `[f32; 64]` with `Copy` semantics, FNV-1a name hashing.
- **Layered architecture**: L1 (spectral) and L2 (parametric) layers with configurable quality.
