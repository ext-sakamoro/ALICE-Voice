//! Python bindings for ALICE-Voice
//!
//! Provides NumPy-compatible interface for voice encoding/decoding.

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray1};

#[cfg(feature = "python")]
use crate::{EmotionType, ParametricParams, VoiceCodec, VoiceCodecConfig, VoiceQuality};

/// Python wrapper for `VoiceCodec`
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
    ///     `sample_rate`: Sample rate in Hz (default: 16000)
    ///     quality: Quality level ("low", "medium", "high", "ultra")
    #[new]
    #[pyo3(signature = (sample_rate=16000, quality="medium"))]
    #[allow(clippy::unnecessary_wraps)]
    fn new(sample_rate: u32, quality: &str) -> PyResult<Self> {
        let q = match quality.to_lowercase().as_str() {
            "low" => VoiceQuality::Low,
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
    const fn sample_rate(&self) -> u32 {
        self.inner.config().sample_rate
    }

    /// Get frame size
    #[getter]
    const fn frame_size(&self) -> usize {
        self.inner.config().frame_size
    }
}

/// Encode voice to parametric representation
///
/// Args:
///     audio: Audio samples as float32 numpy array
///     `sample_rate`: Sample rate in Hz
///
/// Returns:
///     List of parametric parameters (dict per frame)
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (audio, sample_rate=16000))]
#[allow(clippy::needless_pass_by_value)]
fn voice_to_params<'py>(
    py: Python<'py>,
    audio: PyReadonlyArray1<'py, f32>,
    sample_rate: u32,
) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
    let samples = audio.as_slice()?;
    let samples_owned: Vec<f32> = samples.to_vec();

    let params = crate::layers::parametric::voice_to_params(&samples_owned, sample_rate).map_err(
        |e: crate::types::VoiceError| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        },
    )?;

    // Convert to Python dicts
    let result: Vec<Py<pyo3::types::PyDict>> = params
        .iter()
        .map(|p| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("lpc_coeffs", p.lpc.coeffs.clone()).unwrap();
            dict.set_item("gain", p.lpc.gain).unwrap();
            dict.set_item("pitch", p.pitch.f0).unwrap();
            dict.set_item("is_voiced", p.pitch.is_voiced).unwrap();
            dict.set_item("energy_db", p.activity.energy_db).unwrap();
            dict.unbind()
        })
        .collect();

    Ok(result)
}

/// Synthesize voice from parametric representation
///
/// Args:
///     params: List of parametric parameters (dict per frame)
///     `sample_rate`: Sample rate in Hz
///
/// Returns:
///     Audio samples as float32 numpy array
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (params, sample_rate=16000))]
#[allow(clippy::needless_pass_by_value)]
fn params_to_voice(
    py: Python<'_>,
    params: Vec<Py<pyo3::types::PyDict>>,
    sample_rate: u32,
) -> PyResult<Bound<'_, PyArray1<f32>>> {
    use crate::codec::lpc::LpcCoefficients;
    use crate::codec::pitch::PitchInfo;
    use crate::types::VoiceActivity;

    // Convert Python dicts to ParametricParams
    let parametric_params: Vec<ParametricParams> = params
        .iter()
        .map(|obj| {
            let dict = obj.bind(py);

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

    let samples = crate::layers::parametric::params_to_voice(&parametric_params, sample_rate);
    Ok(PyArray1::from_vec(py, samples))
}

/// Encode voice to emotion representation
///
/// Args:
///     audio: Audio samples as float32 numpy array
///     `sample_rate`: Sample rate in Hz
///
/// Returns:
///     Emotion parameters as dict
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (audio, sample_rate=16000))]
#[allow(clippy::needless_pass_by_value)]
fn emotion_encode<'py>(
    py: Python<'py>,
    audio: PyReadonlyArray1<'py, f32>,
    sample_rate: u32,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let samples = audio.as_slice()?;
    let num_samples = samples.len();

    // Compute basic energy features
    let energy: f32 = samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32;
    let avg_energy = 10.0 * (energy + 1e-10_f32).log10();
    let avg_pitch = sample_rate as f32 / 80.0; // placeholder: 80 Hz

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("emotion", "Neutral")?;
    dict.set_item("emotion_confidence", 1.0_f32)?;
    dict.set_item("avg_pitch", avg_pitch)?;
    dict.set_item("avg_energy", avg_energy)?;
    dict.set_item("duration_ms", num_samples as u32 * 1000 / sample_rate)?;

    Ok(dict)
}

/// Decode emotion representation to voice
///
/// Args:
///     params: Emotion parameters as dict
///     `sample_rate`: Sample rate in Hz
///
/// Returns:
///     Audio samples as float32 numpy array
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (params, sample_rate=16000))]
#[allow(clippy::needless_pass_by_value)]
fn emotion_decode<'py>(
    py: Python<'py>,
    params: Bound<'py, pyo3::types::PyDict>,
    sample_rate: u32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let emotion_str: String = params
        .get_item("emotion")?
        .unwrap_or_else(|| pyo3::types::PyString::new(py, "Neutral").into_any())
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

    let duration_ms: u32 = params
        .get_item("duration_ms")?
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("duration_ms"))?
        .extract()?;

    // Synthesize silence of the requested duration (placeholder)
    let num_samples = (u64::from(sample_rate) * u64::from(duration_ms) / 1000) as usize;
    let samples = vec![0.0_f32; num_samples];

    Ok(PyArray1::from_vec(py, samples))
}

/// Get library version
#[cfg(feature = "python")]
#[pyfunction]
const fn version() -> &'static str {
    crate::VERSION
}

/// Python module definition
#[cfg(feature = "python")]
#[pymodule]
fn alice_voice(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyVoiceCodec>()?;
    m.add_function(wrap_pyfunction!(voice_to_params, m)?)?;
    m.add_function(wrap_pyfunction!(params_to_voice, m)?)?;
    m.add_function(wrap_pyfunction!(emotion_encode, m)?)?;
    m.add_function(wrap_pyfunction!(emotion_decode, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
