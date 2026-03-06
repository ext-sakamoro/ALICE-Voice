//! ノイズリダクションモジュール
//!
//! Spectral Subtraction によるノイズ除去を提供する。
//! 無音区間からノイズフロアを推定し、各フレームのスペクトルから
//! ノイズ成分を減算する。
//!
//! # アルゴリズム
//!
//! 1. 無音フレームの平均パワースペクトルをノイズプロファイルとして推定
//! 2. 各フレームを窓関数 (Hann) 適用後 DFT
//! 3. パワースペクトルからノイズプロファイルを減算 (spectral floor でクランプ)
//! 4. 位相を保持したまま逆 DFT で時間領域に復元
//!
//! Author: Moroya Sakamoto

use std::f32::consts::PI;

/// ノイズプロファイル: 周波数帯域別パワー推定値。
#[derive(Debug, Clone)]
pub struct NoiseProfile {
    /// 各周波数ビンのノイズパワー (|X[k]|^2)。
    pub power: Vec<f32>,
    /// 推定に使用したフレーム数。
    pub frame_count: usize,
}

impl NoiseProfile {
    /// 空のプロファイルを指定ビン数で作成。
    #[must_use]
    pub fn new(num_bins: usize) -> Self {
        Self {
            power: vec![0.0; num_bins],
            frame_count: 0,
        }
    }

    /// ビン数。
    #[must_use]
    pub const fn num_bins(&self) -> usize {
        self.power.len()
    }

    /// フレームのパワースペクトルを加算してプロファイルを更新。
    pub fn accumulate(&mut self, frame_power: &[f32]) {
        let len = self.power.len().min(frame_power.len());
        for (p, &fp) in self.power[..len].iter_mut().zip(&frame_power[..len]) {
            *p += fp;
        }
        self.frame_count += 1;
    }

    /// 累積パワーを平均化して最終プロファイルに確定。
    pub fn finalize(&mut self) {
        if self.frame_count > 0 {
            let inv = 1.0 / self.frame_count as f32;
            for p in &mut self.power {
                *p *= inv;
            }
        }
    }
}

/// Spectral Subtraction ノイズリダクタ。
pub struct SpectralSubtractor {
    /// フレームサイズ (DFT 点数)。
    frame_size: usize,
    /// 過減算係数 (1.0 = 標準、2.0-4.0 = 攻撃的)。
    over_subtraction: f32,
    /// 最小スペクトルフロア (0.0-1.0)。ミュージカルノイズ抑制。
    spectral_floor: f32,
    /// Hann 窓。
    window: Vec<f32>,
}

impl SpectralSubtractor {
    /// 指定フレームサイズで作成。
    #[must_use]
    pub fn new(frame_size: usize) -> Self {
        Self::with_params(frame_size, 2.0, 0.02)
    }

    /// パラメータ指定で作成。
    #[must_use]
    pub fn with_params(frame_size: usize, over_subtraction: f32, spectral_floor: f32) -> Self {
        let window = hann_window(frame_size);
        Self {
            frame_size,
            over_subtraction,
            spectral_floor,
            window,
        }
    }

    /// フレームサイズ。
    #[must_use]
    pub const fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// 過減算係数。
    #[must_use]
    pub const fn over_subtraction(&self) -> f32 {
        self.over_subtraction
    }

    /// スペクトルフロア。
    #[must_use]
    pub const fn spectral_floor(&self) -> f32 {
        self.spectral_floor
    }

    /// 無音フレーム列からノイズプロファイルを推定。
    ///
    /// `silence_frames`: 各フレームは `frame_size` サンプル。
    #[must_use]
    pub fn estimate_noise(&self, silence_frames: &[&[f32]]) -> NoiseProfile {
        let half = self.frame_size / 2 + 1;
        let mut profile = NoiseProfile::new(half);

        for frame in silence_frames {
            let power = self.frame_power(frame);
            profile.accumulate(&power);
        }
        profile.finalize();
        profile
    }

    /// 単一フレームにノイズ減算を適用。
    ///
    /// 入力フレーム長は `frame_size` でなければならない。
    /// 出力は同じ長さのクリーンフレーム。
    ///
    /// # Panics
    ///
    /// `frame.len() != frame_size` の場合パニック。
    #[must_use]
    pub fn subtract(&self, frame: &[f32], noise: &NoiseProfile) -> Vec<f32> {
        let n = self.frame_size;
        assert_eq!(frame.len(), n, "frame length must match frame_size");

        // 窓関数適用
        let windowed: Vec<f32> = frame
            .iter()
            .zip(&self.window)
            .map(|(s, w)| s * w)
            .collect();

        // DFT
        let (re, im) = dft(&windowed);
        let half = n / 2 + 1;

        // パワースペクトル計算 & ノイズ減算
        let mut out_re = vec![0.0f32; n];
        let mut out_im = vec![0.0f32; n];

        for k in 0..half {
            let power = re[k].mul_add(re[k], im[k] * im[k]);
            let noise_power = if k < noise.power.len() {
                noise.power[k]
            } else {
                0.0
            };

            // Spectral Subtraction with over-subtraction and floor
            let clean_power = self
                .over_subtraction
                .mul_add(-noise_power, power)
                .max(self.spectral_floor * power);

            // Gain = sqrt(clean_power / power)
            let gain = if power > 1e-10 {
                (clean_power / power).sqrt()
            } else {
                0.0
            };

            out_re[k] = re[k] * gain;
            out_im[k] = im[k] * gain;

            // ミラー (共役対称)
            if k > 0 && k < n - k {
                out_re[n - k] = out_re[k];
                out_im[n - k] = -out_im[k];
            }
        }

        // 逆 DFT
        idft(&out_re, &out_im)
    }

    /// フレームのパワースペクトル (半帯域) を計算。
    fn frame_power(&self, frame: &[f32]) -> Vec<f32> {
        let n = self.frame_size;
        let len = frame.len().min(n);
        let mut windowed = vec![0.0f32; n];
        for i in 0..len {
            windowed[i] = frame[i] * self.window[i];
        }

        let (re, im) = dft(&windowed);
        let half = n / 2 + 1;
        let mut power = Vec::with_capacity(half);
        for k in 0..half {
            power.push(re[k].mul_add(re[k], im[k] * im[k]));
        }
        power
    }
}

/// Hann 窓を生成。
fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let phase = 2.0 * PI * i as f32 / size as f32;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

/// 実数入力の DFT。
///
/// 出力: (実部, 虚部) の N 点。
fn dft(x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = x.len();
    let mut re = vec![0.0f32; n];
    let mut im = vec![0.0f32; n];
    let inv_n = 2.0 * PI / n as f32;

    for k in 0..n {
        let mut sum_re = 0.0f32;
        let mut sum_im = 0.0f32;
        for (j, &xj) in x.iter().enumerate() {
            let angle = inv_n * (k * j) as f32;
            sum_re += xj * angle.cos();
            sum_im -= xj * angle.sin();
        }
        re[k] = sum_re;
        im[k] = sum_im;
    }
    (re, im)
}

/// 逆 DFT。
///
/// 出力: 実部の N 点。
fn idft(re: &[f32], im: &[f32]) -> Vec<f32> {
    let n = re.len();
    let inv_n = 2.0 * PI / n as f32;
    let scale = 1.0 / n as f32;

    let mut out = vec![0.0f32; n];
    for (j, out_j) in out.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for k in 0..n {
            let angle = inv_n * (k * j) as f32;
            sum += re[k].mul_add(angle.cos(), -(im[k] * angle.sin()));
        }
        *out_j = sum * scale;
    }
    out
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: usize = 64;

    fn sine_frame(freq_hz: f32, sample_rate: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate).sin())
            .collect()
    }

    fn white_noise(n: usize, amplitude: f32) -> Vec<f32> {
        // 決定論的擬似ノイズ (テスト再現性)
        (0..n)
            .map(|i| {
                let x = (i as f32 * 1.618_034).fract() * 2.0 - 1.0;
                x * amplitude
            })
            .collect()
    }

    // --- NoiseProfile ---

    #[test]
    fn noise_profile_new() {
        let profile = NoiseProfile::new(33);
        assert_eq!(profile.num_bins(), 33);
        assert_eq!(profile.frame_count, 0);
        assert!(profile.power.iter().all(|&p| p == 0.0));
    }

    #[test]
    fn noise_profile_accumulate_and_finalize() {
        let mut profile = NoiseProfile::new(4);
        profile.accumulate(&[1.0, 2.0, 3.0, 4.0]);
        profile.accumulate(&[3.0, 4.0, 5.0, 6.0]);
        assert_eq!(profile.frame_count, 2);

        profile.finalize();
        assert!((profile.power[0] - 2.0).abs() < 0.01);
        assert!((profile.power[1] - 3.0).abs() < 0.01);
    }

    #[test]
    fn noise_profile_finalize_empty() {
        let mut profile = NoiseProfile::new(4);
        profile.finalize(); // Should not panic
        assert!(profile.power.iter().all(|&p| p == 0.0));
    }

    // --- Hann 窓 ---

    #[test]
    fn hann_window_endpoints() {
        let w = hann_window(8);
        assert!(w[0].abs() < 1e-6, "start should be ~0");
        // 中央付近が最大
        assert!(w[4] > 0.9);
    }

    // --- DFT / IDFT ---

    #[test]
    fn dft_idft_roundtrip() {
        let signal = sine_frame(440.0, 8000.0, FRAME);
        let (re, im) = dft(&signal);
        let recovered = idft(&re, &im);

        for (i, (&orig, &rec)) in signal.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-3,
                "sample {i}: orig={orig}, rec={rec}"
            );
        }
    }

    #[test]
    fn dft_dc_signal() {
        let signal = vec![1.0f32; 8];
        let (re, im) = dft(&signal);
        // DC bin should be N (= 8)
        assert!((re[0] - 8.0).abs() < 1e-4);
        assert!(im[0].abs() < 1e-4);
        // Other bins ~0
        for k in 1..8 {
            assert!(re[k].abs() < 1e-4, "re[{k}] = {}", re[k]);
            assert!(im[k].abs() < 1e-4, "im[{k}] = {}", im[k]);
        }
    }

    // --- SpectralSubtractor ---

    #[test]
    fn subtractor_new() {
        let sub = SpectralSubtractor::new(256);
        assert_eq!(sub.frame_size(), 256);
        assert!((sub.over_subtraction() - 2.0).abs() < 0.01);
        assert!((sub.spectral_floor() - 0.02).abs() < 0.01);
    }

    #[test]
    fn subtractor_with_params() {
        let sub = SpectralSubtractor::with_params(128, 3.0, 0.05);
        assert_eq!(sub.frame_size(), 128);
        assert!((sub.over_subtraction() - 3.0).abs() < 0.01);
    }

    #[test]
    fn estimate_noise_from_silence() {
        let sub = SpectralSubtractor::new(FRAME);
        let silence1 = white_noise(FRAME, 0.01);
        let silence2 = white_noise(FRAME, 0.01);
        let profile = sub.estimate_noise(&[&silence1, &silence2]);

        assert_eq!(profile.frame_count, 2);
        assert_eq!(profile.num_bins(), FRAME / 2 + 1);
        // ノイズパワーは小さいはず
        let max_power = profile
            .power
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        assert!(max_power < 1.0, "max noise power = {max_power}");
    }

    #[test]
    fn subtract_reduces_noise() {
        let sub = SpectralSubtractor::new(FRAME);

        // ノイズ推定
        let noise_frames: Vec<Vec<f32>> = (0..4).map(|_| white_noise(FRAME, 0.1)).collect();
        let noise_refs: Vec<&[f32]> = noise_frames.iter().map(|v| v.as_slice()).collect();
        let profile = sub.estimate_noise(&noise_refs);

        // 信号 + ノイズ
        let clean_signal = sine_frame(440.0, 8000.0, FRAME);
        let noise = white_noise(FRAME, 0.1);
        let noisy: Vec<f32> = clean_signal
            .iter()
            .zip(&noise)
            .map(|(s, n)| s + n)
            .collect();

        let denoised = sub.subtract(&noisy, &profile);
        assert_eq!(denoised.len(), FRAME);

        // エネルギーが減少していることを確認
        let noisy_energy: f32 = noisy.iter().map(|x| x * x).sum();
        let denoised_energy: f32 = denoised.iter().map(|x| x * x).sum();
        // Hann 窓適用のため必ずエネルギー減少
        assert!(
            denoised_energy < noisy_energy,
            "denoised_energy={denoised_energy} >= noisy_energy={noisy_energy}"
        );
    }

    #[test]
    fn subtract_silent_input() {
        let sub = SpectralSubtractor::new(FRAME);
        let silence = vec![0.0f32; FRAME];
        let profile = NoiseProfile::new(FRAME / 2 + 1);
        let result = sub.subtract(&silence, &profile);
        // 入力がゼロなら出力もゼロ
        assert!(result.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn subtract_preserves_length() {
        let sub = SpectralSubtractor::new(FRAME);
        let frame = sine_frame(200.0, 8000.0, FRAME);
        let profile = NoiseProfile::new(FRAME / 2 + 1);
        let result = sub.subtract(&frame, &profile);
        assert_eq!(result.len(), FRAME);
    }

    #[test]
    #[should_panic(expected = "frame length must match frame_size")]
    fn subtract_wrong_length_panics() {
        let sub = SpectralSubtractor::new(FRAME);
        let frame = vec![0.0f32; FRAME + 1];
        let profile = NoiseProfile::new(FRAME / 2 + 1);
        let _ = sub.subtract(&frame, &profile);
    }

    #[test]
    fn noise_profile_clone() {
        let mut profile = NoiseProfile::new(4);
        profile.accumulate(&[1.0, 2.0, 3.0, 4.0]);
        let cloned = profile.clone();
        assert_eq!(cloned.frame_count, profile.frame_count);
        assert_eq!(cloned.power, profile.power);
    }

    #[test]
    fn high_over_subtraction() {
        let sub = SpectralSubtractor::with_params(FRAME, 4.0, 0.01);
        let frame = white_noise(FRAME, 0.5);
        let mut profile = NoiseProfile::new(FRAME / 2 + 1);
        profile.accumulate(&vec![1.0; FRAME / 2 + 1]);
        profile.finalize();

        let result = sub.subtract(&frame, &profile);
        assert_eq!(result.len(), FRAME);
    }
}
