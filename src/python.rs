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
    ParametricParams,
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
    }).map_err(|e: crate::types::VoiceError| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Convert to Python dicts
    let result: Vec<PyObject> = params
        .iter()
        .map(|p| {
            let dict = pyo3::types::PyDict::new_bound(py);
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
                .unwrap_or_else(|| pyo3::types::PyFloat::new_bound(py, -30.0).into_any())
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
    Ok(samples.into_pyarray_bound(py))
}

/// Encode voice to emotion representation
///
/// Args:
///     audio: Audio samples as float32 numpy array
///     sample_rate: Sample rate in Hz
///
/// Returns:
///     Emotion parameters as dict
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (audio, sample_rate=16000))]
fn emotion_encode<'py>(
    py: Python<'py>,
    audio: PyReadonlyArray1<'py, f32>,
    sample_rate: u32,
) -> PyResult<PyObject> {
    let samples = audio.as_slice()?;
    let samples_owned: Vec<f32> = samples.to_vec();
    let num_samples = samples_owned.len();

    // Compute basic energy features in Rust (GIL-free)
    let (avg_pitch, avg_energy) = py.allow_threads(move || {
        let energy: f32 = samples_owned.iter().map(|&s| s * s).sum::<f32>()
            / samples_owned.len() as f32;
        let energy_db = 10.0 * (energy + 1e-10_f32).log10();
        let avg_pitch = sample_rate as f32 / 80.0; // placeholder: 80 Hz
        (avg_pitch, energy_db)
    });

    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("emotion", "Neutral")?;
    dict.set_item("emotion_confidence", 1.0_f32)?;
    dict.set_item("avg_pitch", avg_pitch)?;
    dict.set_item("avg_energy", avg_energy)?;
    dict.set_item("duration_ms", num_samples as u32 * 1000 / sample_rate)?;

    Ok(dict.into_py(py))
}

/// Decode emotion representation to voice
///
/// Args:
///     params: Emotion parameters as dict
///     sample_rate: Sample rate in Hz
///
/// Returns:
///     Audio samples as float32 numpy array
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (params, sample_rate=16000))]
fn emotion_decode<'py>(
    py: Python<'py>,
    params: PyObject,
    sample_rate: u32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let dict = params.downcast_bound::<pyo3::types::PyDict>(py)?;

    let emotion_str: String = dict
        .get_item("emotion")?
        .unwrap_or_else(|| pyo3::types::PyString::new_bound(py, "Neutral").into_any())
        .extract()?;

    let _emotion = match emotion_str.to_lowercase().as_str() {
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

    // Synthesize silence of the requested duration (placeholder)
    let num_samples = (sample_rate as u64 * duration_ms as u64 / 1000) as usize;
    let samples = py.allow_threads(move || vec![0.0_f32; num_samples]);

    Ok(samples.into_pyarray_bound(py))
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
    m.add_function(wrap_pyfunction!(emotion_encode, m)?)?;
    m.add_function(wrap_pyfunction!(emotion_decode, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
