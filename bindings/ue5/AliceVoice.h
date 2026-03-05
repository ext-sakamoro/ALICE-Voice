// ALICE-Voice UE5 C++ Header
// Auto-generated — 20 extern C + RAII wrappers
// Author: Moroya Sakamoto

#pragma once

#include <cstdint>
#include <utility>

// ============================================
// C API (20 functions)
// ============================================

extern "C"
{
    // Opaque handles
    typedef struct VoiceCodec VoiceCodec;
    typedef struct ParamsList ParamsList;
    typedef struct AudioBuffer AudioBuffer;
    typedef struct SpeakerEmbedding SpeakerEmbedding;

    // 1. Create codec (default 16kHz wideband)
    VoiceCodec* alice_voice_codec_create();

    // 2. Create codec with quality (0=Low,1=Medium,2=High,3=Ultra)
    VoiceCodec* alice_voice_codec_create_quality(uint8_t quality);

    // 3. Destroy codec
    void alice_voice_codec_destroy(VoiceCodec* codec);

    // 4. Encode to L2 parametric
    ParamsList* alice_voice_codec_encode_parametric(
        VoiceCodec* codec, const float* samples, uint32_t len);

    // 5. Decode from L2 parametric
    AudioBuffer* alice_voice_codec_decode_parametric(
        const VoiceCodec* codec, const ParamsList* params, uint32_t* out_len);

    // 6. Encode L1 spectral (round-trip)
    AudioBuffer* alice_voice_codec_encode_spectral(
        VoiceCodec* codec, const float* samples, uint32_t len, uint32_t* out_frames);

    // 7. Get audio data pointer from buffer
    const float* alice_voice_audio_ptr(const AudioBuffer* buf, uint32_t* out_len);

    // 8. Get sample rate
    uint32_t alice_voice_codec_sample_rate(const VoiceCodec* codec);

    // 9. Get frame size
    uint32_t alice_voice_codec_frame_size(const VoiceCodec* codec);

    // 10. Get params count
    uint32_t alice_voice_params_count(const ParamsList* params);

    // 11. Destroy params
    void alice_voice_params_destroy(ParamsList* params);

    // 12. Compute encoding stats
    void alice_voice_stats(
        const ParamsList* params, uint32_t original_samples,
        uint32_t* out_frames, uint32_t* out_voiced,
        float* out_avg_pitch, float* out_compression);

    // 13. Create speaker embedding
    SpeakerEmbedding* alice_voice_speaker_create(const float* data, uint32_t len);

    // 14. Speaker similarity
    float alice_voice_speaker_similarity(
        const SpeakerEmbedding* a, const SpeakerEmbedding* b);

    // 15. Destroy speaker embedding
    void alice_voice_speaker_destroy(SpeakerEmbedding* speaker);

    // 16. Convenience: voice → params
    ParamsList* alice_voice_to_params(
        const float* samples, uint32_t len, uint32_t sample_rate);

    // 17. Convenience: params → voice
    AudioBuffer* alice_voice_from_params(
        const ParamsList* params, uint32_t sample_rate, uint32_t* out_len);

    // 18. Free audio buffer
    void alice_voice_data_free(AudioBuffer* buf);

    // 19. Free string
    void alice_voice_string_free(char* s);

    // 20. Get version
    char* alice_voice_version();
}

// ============================================
// RAII Wrappers
// ============================================

namespace Alice
{

/// RAII wrapper for VoiceCodec
class FVoiceCodec
{
    VoiceCodec* Handle;

public:
    /// Default constructor (16kHz wideband)
    FVoiceCodec()
        : Handle(alice_voice_codec_create())
    {
    }

    /// Quality constructor (0=Low,1=Medium,2=High,3=Ultra)
    explicit FVoiceCodec(uint8_t Quality)
        : Handle(alice_voice_codec_create_quality(Quality))
    {
    }

    ~FVoiceCodec()
    {
        if (Handle) alice_voice_codec_destroy(Handle);
    }

    // Move only
    FVoiceCodec(FVoiceCodec&& Other) noexcept : Handle(Other.Handle) { Other.Handle = nullptr; }
    FVoiceCodec& operator=(FVoiceCodec&& Other) noexcept
    {
        if (this != &Other)
        {
            if (Handle) alice_voice_codec_destroy(Handle);
            Handle = Other.Handle;
            Other.Handle = nullptr;
        }
        return *this;
    }
    FVoiceCodec(const FVoiceCodec&) = delete;
    FVoiceCodec& operator=(const FVoiceCodec&) = delete;

    VoiceCodec* Get() const { return Handle; }
    uint32_t SampleRate() const { return alice_voice_codec_sample_rate(Handle); }
    uint32_t FrameSize() const { return alice_voice_codec_frame_size(Handle); }
};

/// RAII wrapper for ParamsList
class FParamsList
{
    ParamsList* Handle;

public:
    explicit FParamsList(ParamsList* InHandle) : Handle(InHandle) {}

    ~FParamsList()
    {
        if (Handle) alice_voice_params_destroy(Handle);
    }

    FParamsList(FParamsList&& Other) noexcept : Handle(Other.Handle) { Other.Handle = nullptr; }
    FParamsList& operator=(FParamsList&& Other) noexcept
    {
        if (this != &Other)
        {
            if (Handle) alice_voice_params_destroy(Handle);
            Handle = Other.Handle;
            Other.Handle = nullptr;
        }
        return *this;
    }
    FParamsList(const FParamsList&) = delete;
    FParamsList& operator=(const FParamsList&) = delete;

    ParamsList* Get() const { return Handle; }
    uint32_t Count() const { return alice_voice_params_count(Handle); }
};

/// RAII wrapper for AudioBuffer
class FAudioBuffer
{
    AudioBuffer* Handle;

public:
    explicit FAudioBuffer(AudioBuffer* InHandle) : Handle(InHandle) {}

    ~FAudioBuffer()
    {
        if (Handle) alice_voice_data_free(Handle);
    }

    FAudioBuffer(FAudioBuffer&& Other) noexcept : Handle(Other.Handle) { Other.Handle = nullptr; }
    FAudioBuffer& operator=(FAudioBuffer&& Other) noexcept
    {
        if (this != &Other)
        {
            if (Handle) alice_voice_data_free(Handle);
            Handle = Other.Handle;
            Other.Handle = nullptr;
        }
        return *this;
    }
    FAudioBuffer(const FAudioBuffer&) = delete;
    FAudioBuffer& operator=(const FAudioBuffer&) = delete;

    AudioBuffer* Get() const { return Handle; }
    const float* Data(uint32_t* OutLen) const { return alice_voice_audio_ptr(Handle, OutLen); }
};

/// RAII wrapper for SpeakerEmbedding
class FSpeakerEmbedding
{
    SpeakerEmbedding* Handle;

public:
    FSpeakerEmbedding(const float* Data, uint32_t Len)
        : Handle(alice_voice_speaker_create(Data, Len))
    {
    }

    ~FSpeakerEmbedding()
    {
        if (Handle) alice_voice_speaker_destroy(Handle);
    }

    FSpeakerEmbedding(FSpeakerEmbedding&& Other) noexcept : Handle(Other.Handle) { Other.Handle = nullptr; }
    FSpeakerEmbedding& operator=(FSpeakerEmbedding&& Other) noexcept
    {
        if (this != &Other)
        {
            if (Handle) alice_voice_speaker_destroy(Handle);
            Handle = Other.Handle;
            Other.Handle = nullptr;
        }
        return *this;
    }
    FSpeakerEmbedding(const FSpeakerEmbedding&) = delete;
    FSpeakerEmbedding& operator=(const FSpeakerEmbedding&) = delete;

    float Similarity(const FSpeakerEmbedding& Other) const
    {
        return alice_voice_speaker_similarity(Handle, Other.Handle);
    }
};

/// Convenience: encode voice → params → voice round-trip
inline FParamsList VoiceToParams(const float* Samples, uint32_t Len, uint32_t SampleRate)
{
    return FParamsList(alice_voice_to_params(Samples, Len, SampleRate));
}

inline FAudioBuffer ParamsToVoice(const FParamsList& Params, uint32_t SampleRate)
{
    uint32_t Len = 0;
    return FAudioBuffer(alice_voice_from_params(Params.Get(), SampleRate, &Len));
}

} // namespace Alice
