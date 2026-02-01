"""
ALICE-Voice: Voice-Specialized Procedural Codec

"Don't send waveforms. Send the law of speech."

This package provides ultra-efficient voice transmission through parametric encoding.

Layers:
    L1 (Spectral): FFT/DCT coefficients - 10-50x compression
    L2 (Parametric): LPC + Formants + Pitch - 100-600x compression
    L3 (Semantic): Text + Emotion + Speaker - 1000x+ compression

Example:
    >>> import alice_voice
    >>> import numpy as np
    >>>
    >>> # Generate test audio
    >>> audio = np.sin(np.linspace(0, 100, 16000)).astype(np.float32)
    >>>
    >>> # L2: Parametric encoding
    >>> params = alice_voice.voice_to_params(audio, sample_rate=16000)
    >>> reconstructed = alice_voice.params_to_voice(params, sample_rate=16000)
    >>>
    >>> # L3: Semantic encoding
    >>> semantic = alice_voice.semantic_encode(audio, sample_rate=16000)
    >>> print(semantic['text'])

"""

__version__ = "0.1.0"
__author__ = "Moroya Sakamoto"

# Try to import the native Rust module
try:
    from .alice_voice import (
        VoiceCodec,
        voice_to_params,
        params_to_voice,
        semantic_encode,
        semantic_decode,
        version,
    )

    __all__ = [
        "VoiceCodec",
        "voice_to_params",
        "params_to_voice",
        "semantic_encode",
        "semantic_decode",
        "version",
    ]

except ImportError:
    # Fallback: provide Python-only implementation hints
    import warnings
    warnings.warn(
        "Native alice_voice module not found. "
        "Install with: cd ALICE-Voice && maturin develop --release",
        ImportWarning
    )

    def voice_to_params(audio, sample_rate=16000):
        """
        Encode voice to parametric representation.

        Note: Native module not loaded. Install with maturin.
        """
        raise NotImplementedError(
            "Native module not loaded. Install with: maturin develop --release"
        )

    def params_to_voice(params, sample_rate=16000):
        """
        Decode parametric representation to voice.

        Note: Native module not loaded. Install with maturin.
        """
        raise NotImplementedError(
            "Native module not loaded. Install with: maturin develop --release"
        )

    def semantic_encode(audio, sample_rate=16000):
        """
        Encode voice to semantic representation.

        Note: Native module not loaded. Install with maturin.
        """
        raise NotImplementedError(
            "Native module not loaded. Install with: maturin develop --release"
        )

    def semantic_decode(params, sample_rate=16000):
        """
        Decode semantic representation to voice.

        Note: Native module not loaded. Install with maturin.
        """
        raise NotImplementedError(
            "Native module not loaded. Install with: maturin develop --release"
        )

    def version():
        """Get library version."""
        return __version__

    class VoiceCodec:
        """
        Voice codec placeholder.

        Note: Native module not loaded. Install with maturin.
        """
        def __init__(self, sample_rate=16000, quality="medium"):
            raise NotImplementedError(
                "Native module not loaded. Install with: maturin develop --release"
            )

    __all__ = [
        "VoiceCodec",
        "voice_to_params",
        "params_to_voice",
        "semantic_encode",
        "semantic_decode",
        "version",
    ]
