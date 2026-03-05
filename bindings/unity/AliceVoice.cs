// ALICE-Voice Unity C# Bindings
// Auto-generated — 20 DllImport functions
// Author: Moroya Sakamoto

using System;
using System.Runtime.InteropServices;

namespace Alice.Voice
{
    // ========================================
    // VoiceCodec — L1/L2 voice encoding
    // ========================================

    public sealed class VoiceCodec : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        /// <summary>Create codec with default config (16kHz wideband).</summary>
        public VoiceCodec()
        {
            _handle = NativeMethods.alice_voice_codec_create();
        }

        /// <summary>Create codec with quality level (0=Low,1=Medium,2=High,3=Ultra).</summary>
        public VoiceCodec(byte quality)
        {
            _handle = NativeMethods.alice_voice_codec_create_quality(quality);
        }

        /// <summary>Sample rate in Hz.</summary>
        public uint SampleRate => NativeMethods.alice_voice_codec_sample_rate(_handle);

        /// <summary>Frame size in samples.</summary>
        public uint FrameSize => NativeMethods.alice_voice_codec_frame_size(_handle);

        /// <summary>Encode audio to L2 parametric params.</summary>
        public ParamsList EncodeParametric(float[] samples)
        {
            var ptr = NativeMethods.alice_voice_codec_encode_parametric(
                _handle, samples, (uint)samples.Length);
            if (ptr == IntPtr.Zero) return null;
            return new ParamsList(ptr);
        }

        /// <summary>Decode L2 parametric params to audio.</summary>
        public float[] DecodeParametric(ParamsList paramsList)
        {
            uint len = 0;
            var buf = NativeMethods.alice_voice_codec_decode_parametric(
                _handle, paramsList.Handle, ref len);
            if (buf == IntPtr.Zero) return Array.Empty<float>();
            return AudioBuffer.Extract(buf, len);
        }

        /// <summary>Encode audio through L1 spectral layer (round-trip).</summary>
        public float[] EncodeDecodeSpectral(float[] samples)
        {
            uint frames = 0;
            var buf = NativeMethods.alice_voice_codec_encode_spectral(
                _handle, samples, (uint)samples.Length, ref frames);
            if (buf == IntPtr.Zero) return Array.Empty<float>();
            uint len = 0;
            var ptr = NativeMethods.alice_voice_audio_ptr(buf, ref len);
            var result = new float[len];
            if (ptr != IntPtr.Zero && len > 0)
                Marshal.Copy(ptr, result, 0, (int)len);
            NativeMethods.alice_voice_data_free(buf);
            return result;
        }

        public void Dispose()
        {
            if (!_disposed && _handle != IntPtr.Zero)
            {
                NativeMethods.alice_voice_codec_destroy(_handle);
                _handle = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~VoiceCodec() { Dispose(); }
    }

    // ========================================
    // ParamsList — parametric params handle
    // ========================================

    public sealed class ParamsList : IDisposable
    {
        internal IntPtr Handle;
        private bool _disposed;

        internal ParamsList(IntPtr handle) { Handle = handle; }

        /// <summary>Number of frames.</summary>
        public uint Count => NativeMethods.alice_voice_params_count(Handle);

        public void Dispose()
        {
            if (!_disposed && Handle != IntPtr.Zero)
            {
                NativeMethods.alice_voice_params_destroy(Handle);
                Handle = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~ParamsList() { Dispose(); }
    }

    // ========================================
    // Convenience — voice_to_params / from_params
    // ========================================

    public static class VoiceConvert
    {
        /// <summary>Voice samples → parametric params.</summary>
        public static ParamsList ToParams(float[] samples, uint sampleRate)
        {
            var ptr = NativeMethods.alice_voice_to_params(
                samples, (uint)samples.Length, sampleRate);
            if (ptr == IntPtr.Zero) return null;
            return new ParamsList(ptr);
        }

        /// <summary>Parametric params → voice samples.</summary>
        public static float[] FromParams(ParamsList paramsList, uint sampleRate)
        {
            uint len = 0;
            var buf = NativeMethods.alice_voice_from_params(
                paramsList.Handle, sampleRate, ref len);
            if (buf == IntPtr.Zero) return Array.Empty<float>();
            return AudioBuffer.Extract(buf, len);
        }
    }

    // ========================================
    // EncodingStats — statistics helper
    // ========================================

    public static class VoiceStats
    {
        public struct Stats
        {
            public uint Frames;
            public uint VoicedFrames;
            public float AvgPitch;
            public float CompressionRatio;
        }

        /// <summary>Compute encoding statistics from params.</summary>
        public static Stats Compute(ParamsList paramsList, uint originalSamples)
        {
            var s = new Stats();
            NativeMethods.alice_voice_stats(
                paramsList.Handle, originalSamples,
                ref s.Frames, ref s.VoicedFrames,
                ref s.AvgPitch, ref s.CompressionRatio);
            return s;
        }
    }

    // ========================================
    // SpeakerEmbedding — speaker identification
    // ========================================

    public sealed class SpeakerEmbedding : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        /// <summary>Create from float vector (256-dim).</summary>
        public SpeakerEmbedding(float[] data)
        {
            _handle = NativeMethods.alice_voice_speaker_create(
                data, (uint)data.Length);
        }

        /// <summary>Cosine similarity to another embedding.</summary>
        public float Similarity(SpeakerEmbedding other)
        {
            return NativeMethods.alice_voice_speaker_similarity(
                _handle, other._handle);
        }

        public void Dispose()
        {
            if (!_disposed && _handle != IntPtr.Zero)
            {
                NativeMethods.alice_voice_speaker_destroy(_handle);
                _handle = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~SpeakerEmbedding() { Dispose(); }
    }

    // ========================================
    // Version
    // ========================================

    public static class Version
    {
        /// <summary>Library version string.</summary>
        public static string Get()
        {
            var ptr = NativeMethods.alice_voice_version();
            if (ptr == IntPtr.Zero) return "";
            var str = Marshal.PtrToStringAnsi(ptr);
            NativeMethods.alice_voice_string_free(ptr);
            return str ?? "";
        }
    }

    // ========================================
    // Internal AudioBuffer helper
    // ========================================

    internal static class AudioBuffer
    {
        internal static float[] Extract(IntPtr buf, uint knownLen)
        {
            uint len = knownLen;
            if (len == 0)
            {
                var ptr = NativeMethods.alice_voice_audio_ptr(buf, ref len);
                if (ptr == IntPtr.Zero || len == 0)
                {
                    NativeMethods.alice_voice_data_free(buf);
                    return Array.Empty<float>();
                }
            }
            var result = new float[len];
            var dataPtr = NativeMethods.alice_voice_audio_ptr(buf, ref len);
            if (dataPtr != IntPtr.Zero)
                Marshal.Copy(dataPtr, result, 0, (int)len);
            NativeMethods.alice_voice_data_free(buf);
            return result;
        }
    }

    // ========================================
    // P/Invoke declarations (20 functions)
    // ========================================

    internal static class NativeMethods
    {
        private const string Lib = "alice_voice";

        // 1
        [DllImport(Lib)] internal static extern IntPtr alice_voice_codec_create();
        // 2
        [DllImport(Lib)] internal static extern IntPtr alice_voice_codec_create_quality(byte quality);
        // 3
        [DllImport(Lib)] internal static extern void alice_voice_codec_destroy(IntPtr codec);
        // 4
        [DllImport(Lib)] internal static extern IntPtr alice_voice_codec_encode_parametric(
            IntPtr codec, float[] samples, uint len);
        // 5
        [DllImport(Lib)] internal static extern IntPtr alice_voice_codec_decode_parametric(
            IntPtr codec, IntPtr paramsList, ref uint outLen);
        // 6
        [DllImport(Lib)] internal static extern IntPtr alice_voice_codec_encode_spectral(
            IntPtr codec, float[] samples, uint len, ref uint outFrames);
        // 7
        [DllImport(Lib)] internal static extern IntPtr alice_voice_audio_ptr(
            IntPtr buf, ref uint outLen);
        // 8
        [DllImport(Lib)] internal static extern uint alice_voice_codec_sample_rate(IntPtr codec);
        // 9
        [DllImport(Lib)] internal static extern uint alice_voice_codec_frame_size(IntPtr codec);
        // 10
        [DllImport(Lib)] internal static extern uint alice_voice_params_count(IntPtr paramsList);
        // 11
        [DllImport(Lib)] internal static extern void alice_voice_params_destroy(IntPtr paramsList);
        // 12
        [DllImport(Lib)] internal static extern void alice_voice_stats(
            IntPtr paramsList, uint originalSamples,
            ref uint outFrames, ref uint outVoiced,
            ref float outAvgPitch, ref float outCompression);
        // 13
        [DllImport(Lib)] internal static extern IntPtr alice_voice_speaker_create(
            float[] data, uint len);
        // 14
        [DllImport(Lib)] internal static extern float alice_voice_speaker_similarity(
            IntPtr a, IntPtr b);
        // 15
        [DllImport(Lib)] internal static extern void alice_voice_speaker_destroy(IntPtr speaker);
        // 16
        [DllImport(Lib)] internal static extern IntPtr alice_voice_to_params(
            float[] samples, uint len, uint sampleRate);
        // 17
        [DllImport(Lib)] internal static extern IntPtr alice_voice_from_params(
            IntPtr paramsList, uint sampleRate, ref uint outLen);
        // 18
        [DllImport(Lib)] internal static extern void alice_voice_data_free(IntPtr buf);
        // 19
        [DllImport(Lib)] internal static extern void alice_voice_string_free(IntPtr s);
        // 20
        [DllImport(Lib)] internal static extern IntPtr alice_voice_version();
    }
}
