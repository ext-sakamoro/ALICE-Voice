//! Python bindings for ALICE-Voice
//!
//! Provides NumPy-compatible interface for voice encoding/decoding.

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};

#[cfg(feature = "python")]
use crate::{
    VoiceCodec, VoiceCodecConfig, VoiceQuality,
    SpectralLayer, SpectralParams,
    ParametricLayer, ParametricParams,
    SemanticLayer, SemanticParams,
    EmotionType,
};

/// Python wrapper for VoiceCodec
#[cfg(feature = "python")]
#[pyclass(name = "VoiceCodec")]
pub struct PyVoiceCodec {
    inner: VoiceCodec,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyVoiceCodec {
    /// Create new voice codec
    ///
    /// Args:
    ///     sample_rate: Sample rate in Hz (default: 16000)
    ///     quality: Quality level ("low", "medium", "high", "ultra")
    #[new]
    #[pyo3(signature = (sample_rate=16000, quality="medium"))]
    fn new(sample_rate: u32, quality: &str) -> PyResult<Self> {
        let q = match quality.to_lowercase().as_str() {
            "low" => VoiceQuality::Low,
            "medium" => VoiceQuality::Medium,
            "high" => VoiceQuality::High,
            "ultra" => VoiceQuality::Ultra,
            _ => VoiceQuality::Medium,
        };

        let mut config = VoiceCodecConfig::for_quality(q);
        config.sample_rate = sample_rate;
        config.frame_size = (sample_rate as f32 * 0.032) as usize;
        config.hop_size = config.frame_size / 2;

        Ok(Self {
            inner: VoiceCodec::new(config),
        })
    }

    /// Get sample rate
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.config().sample_rate
    }

    /// Get frame size
    #[getter]
    fn frame_size(&self) -> usize {
        self.inner.config().frame_size
    }
}

/// Encode voice to parametric representation
///
/// Args:
///     audio: Audio samples as float32 numpy array
///     sample_rate: Sample rate in Hz
///
/// Returns:
///     List of parametric parameters (dict per frame)
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (audio, sample_rate=16000))]
fn voice_to_params<'py>(
    py: Python<'py>,
    audio: PyReadonlyArray1<'py, f32>,
    sample_rate: u32,
) -> PyResult<Vec<PyObject>> {
    let samples = audio.as_slice()?;

    let samples_owned: Vec<f32> = samples.to_vec();
    let params = py.allow_threads(|| {
        crate::layers::parametric::voice_to_params(&samples_owned, sample_rate)
    }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Convert to Python dicts
    let result: Vec<PyObject> = params
        .iter()
        .map(|p| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("lpc_coeffs", p.lpc.coeffs.clone()).unwrap();
            dict.set_item("gain", p.lpc.gain).unwrap();
            dict.set_item("pitch", p.pitch.f0).unwrap();
            dict.set_item("is_voiced", p.pitch.is_voiced).unwrap();
            dict.set_item("energy_db", p.activity.energy_db).unwrap();
            dict.into_py(py)
        })
        .collect();

    Ok(result)
}

/// Synthesize voice from parametric representation
///
/// Args:
///     params: List of parametric parameters (dict per frame)
///     sample_rate: Sample rate in Hz
///
/// Returns:
///     Audio samples as float32 numpy array
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (params, sample_rate=16000))]
fn params_to_voice<'py>(
    py: Python<'py>,
    params: Vec<PyObject>,
    sample_rate: u32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    use crate::codec::lpc::LpcCoefficients;
    use crate::codec::pitch::PitchInfo;
    use crate::types::VoiceActivity;

    // Convert Python dicts to ParametricParams
    let parametric_params: Vec<ParametricParams> = params
        .iter()
        .map(|obj| {
            let dict = obj.downcast_bound::<pyo3::types::PyDict>(py)?;

            let lpc_coeffs: Vec<f32> = dict
                .get_item("lpc_coeffs")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("lpc_coeffs"))?
                .extract()?;

            let gain: f32 = dict
                .get_item("gain")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("gain"))?
                .extract()?;

            let pitch: f32 = dict
                .get_item("pitch")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("pitch"))?
                .extract()?;

            let is_voiced: bool = dict
                .get_item("is_voiced")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("is_voiced"))?
                .extract()?;

            let energy_db: f32 = dict
                .get_item("energy_db")?
                .unwrap_or_else(|| pyo3::types::PyFloat::new(py, -30.0).into_any())
                .extract()?;

            let frame_size = (sample_rate as f32 * 0.032) as usize;

            Ok(ParametricParams {
                lpc: LpcCoefficients {
                    coeffs: lpc_coeffs,
                    gain,
                    reflection: vec![],
                    error: 0.0,
                },
                pitch: if is_voiced {
                    PitchInfo::voiced(pitch, 0.9, sample_rate)
                } else {
                    PitchInfo::unvoiced()
                },
                formants: vec![],
                activity: VoiceActivity {
                    is_voiced,
                    confidence: 0.9,
                    energy_db,
                },
                frame_size,
                sample_rate,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    let samples = py.allow_threads(|| {
        crate::layers::parametric::params_to_voice(&parametric_params, sample_rate)
    });
    Ok(samples.into_pyarray(py))
}

/// Encode voice to semantic representation
///
/// Args:
///     audio: Audio samples as float32 numpy array
///     sample_rate: Sample rate in Hz
///
/// Returns:
///     Semantic parameters as dict
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (audio, sample_rate=16000))]
fn semantic_encode<'py>(
    py: Python<'py>,
    audio: PyReadonlyArray1<'py, f32>,
    sample_rate: u32,
) -> PyResult<PyObject> {
    let samples = audio.as_slice()?;

    let samples_owned: Vec<f32> = samples.to_vec();
    let params = py.allow_threads(|| {
        crate::layers::semantic::semantic_encode(&samples_owned, sample_rate)
    }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("text", &params.text)?;
    dict.set_item("emotion", format!("{:?}", params.emotion))?;
    dict.set_item("emotion_confidence", params.emotion_confidence)?;
    dict.set_item("language", &params.language)?;
    dict.set_item("duration_ms", params.duration_ms)?;
    dict.set_item("speaking_rate", params.prosody.speaking_rate)?;
    dict.set_item("avg_pitch", params.prosody.avg_pitch)?;
    dict.set_item("avg_energy", params.prosody.avg_energy)?;
    dict.set_item("speaker_embedding", params.speaker.vector.clone())?;

    Ok(dict.into_py(py))
}

/// Decode semantic representation to voice
///
/// Args:
///     params: Semantic parameters as dict
///     sample_rate: Sample rate in Hz
///
/// Returns:
///     Audio samples as float32 numpy array
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (params, sample_rate=16000))]
fn semantic_decode<'py>(
    py: Python<'py>,
    params: PyObject,
    sample_rate: u32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    use crate::layers::semantic::Prosody;
    use crate::types::SpeakerEmbedding;

    let dict = params.downcast_bound::<pyo3::types::PyDict>(py)?;

    let text: String = dict
        .get_item("text")?
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("text"))?
        .extract()?;

    let emotion_str: String = dict
        .get_item("emotion")?
        .unwrap_or_else(|| pyo3::types::PyString::new(py, "Neutral").into_any())
        .extract()?;

    let emotion = match emotion_str.to_lowercase().as_str() {
        "happy" => EmotionType::Happy,
        "sad" => EmotionType::Sad,
        "angry" => EmotionType::Angry,
        "fearful" => EmotionType::Fearful,
        "surprised" => EmotionType::Surprised,
        "disgusted" => EmotionType::Disgusted,
        _ => EmotionType::Neutral,
    };

    let duration_ms: u32 = dict
        .get_item("duration_ms")?
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("duration_ms"))?
        .extract()?;

    let avg_pitch: f32 = dict
        .get_item("avg_pitch")?
        .unwrap_or_else(|| pyo3::types::PyFloat::new(py, 120.0).into_any())
        .extract()?;

    let avg_energy: f32 = dict
        .get_item("avg_energy")?
        .unwrap_or_else(|| pyo3::types::PyFloat::new(py, -20.0).into_any())
        .extract()?;

    let speaker_embedding: Vec<f32> = dict
        .get_item("speaker_embedding")?
        .map(|v| v.extract().ok())
        .flatten()
        .unwrap_or_else(|| vec![0.0; 256]);

    let semantic_params = SemanticParams {
        text,
        emotion,
        emotion_confidence: 1.0,
        speaker: SpeakerEmbedding::new(speaker_embedding),
        prosody: Prosody {
            speaking_rate: 150.0,
            avg_pitch,
            pitch_variance: 20.0,
            avg_energy,
            word_timings: vec![],
            stress_markers: vec![],
        },
        language: "en".to_string(),
        duration_ms,
    };

    let samples = py.allow_threads(|| {
        crate::layers::semantic::semantic_decode(&semantic_params, sample_rate)
    }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(samples.into_pyarray(py))
}

/// Get library version
#[cfg(feature = "python")]
#[pyfunction]
fn version() -> &'static str {
    crate::VERSION
}

/// Python module definition
#[cfg(feature = "python")]
#[pymodule]
fn alice_voice(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVoiceCodec>()?;
    m.add_function(wrap_pyfunction!(voice_to_params, m)?)?;
    m.add_function(wrap_pyfunction!(params_to_voice, m)?)?;
    m.add_function(wrap_pyfunction!(semantic_encode, m)?)?;
    m.add_function(wrap_pyfunction!(semantic_decode, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
