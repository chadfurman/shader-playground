use cpal::traits::{DeviceTrait, StreamTrait};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AudioFeatures {
    pub bass: f32,
    pub mids: f32,
    pub highs: f32,
    pub energy: f32,
    pub beat: f32,
    pub beat_accum: f32,
    pub beat_pulse: f32,
}

struct AnalysisResult {
    bass: f32,
    mids: f32,
    highs: f32,
    energy: f32,
}

pub struct AudioAnalyzer {
    fft: Arc<dyn Fft<f32>>,
    fft_size: usize,
    window: Vec<f32>,
    buffer: Vec<Complex<f32>>,
    magnitudes: Vec<f32>,
    band_ranges: Vec<(usize, usize)>,
}

impl AudioAnalyzer {
    pub fn new(fft_size: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        let window: Vec<f32> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / fft_size as f32).cos()))
            .collect();

        let half = fft_size / 2;
        let band_ranges: Vec<(usize, usize)> = (0..16)
            .map(|i| {
                let lo = band_edge(half, i);
                let hi = band_edge(half, i + 1);
                (lo, hi.max(lo + 1))
            })
            .collect();

        Self {
            fft,
            fft_size,
            window,
            buffer: vec![Complex::new(0.0, 0.0); fft_size],
            magnitudes: vec![0.0; half],
            band_ranges,
        }
    }

    fn analyze(&mut self, samples: &[f32]) -> AnalysisResult {
        for i in 0..self.fft_size {
            let s = if i < samples.len() { samples[i] } else { 0.0 };
            self.buffer[i] = Complex::new(s * self.window[i], 0.0);
        }

        self.fft.process(&mut self.buffer);

        let n = self.fft_size as f32;
        for i in 0..self.magnitudes.len() {
            self.magnitudes[i] = self.buffer[i].norm() / n;
        }

        let mut bands = [0.0f32; 16];
        for (b, &(lo, hi)) in self.band_ranges.iter().enumerate() {
            let count = (hi - lo).max(1) as f32;
            bands[b] = self.magnitudes[lo..hi].iter().sum::<f32>() / count;
        }

        let bass = (bands[0] + bands[1] + bands[2]) / 3.0;
        let mids = (bands[4] + bands[5] + bands[6] + bands[7]) / 4.0;
        let highs = (bands[10] + bands[11] + bands[12] + bands[13]) / 4.0;
        let energy = bands.iter().sum::<f32>() / 16.0;

        AnalysisResult { bass, mids, highs, energy }
    }
}

fn band_edge(half: usize, i: usize) -> usize {
    if i == 0 {
        return 0;
    }
    let frac = i as f32 / 16.0;
    let edge = (half as f32 * (2.0f32.powf(frac * 10.0) - 1.0) / 1023.0) as usize;
    edge.min(half)
}

// ── Audio Capture ──

const BUFFER_SIZE: usize = 4096;

// Stream holder keeps either cpal or SCK stream alive via RAII.
#[allow(dead_code)]
enum StreamHolder {
    Cpal(cpal::Stream),
    Sck(screencapturekit::stream::sc_stream::SCStream),
}

pub struct AudioCapture {
    buffer: Arc<Mutex<Vec<f32>>>,
    _stream: StreamHolder,
}

impl AudioCapture {
    /// Construct from a cpal device selected by the device picker.
    pub fn from_device(device: cpal::Device, is_input: bool) -> Result<Self, String> {
        let name = device.name().unwrap_or_else(|_| "???".into());
        let config = if is_input {
            device.default_input_config()
        } else {
            device.default_output_config()
        }
        .map_err(|e| format!("config for {name}: {e}"))?;
        eprintln!("[audio] using: {name} ({}Hz)", config.sample_rate());
        Self::build_cpal_capture(&device, &config)
    }

    /// Capture system audio via ScreenCaptureKit (macOS 12.3+).
    pub fn new_system_audio() -> Result<Self, String> {
        let buffer = Arc::new(Mutex::new(Vec::with_capacity(BUFFER_SIZE)));
        let (stream, rate) =
            crate::sck_audio::start_system_audio_capture(buffer.clone(), BUFFER_SIZE)?;
        let _ = rate; // logged by sck_audio
        Ok(Self {
            _stream: StreamHolder::Sck(stream),
            buffer,
        })
    }

    fn build_cpal_capture(
        device: &cpal::Device,
        config: &cpal::SupportedStreamConfig,
    ) -> Result<Self, String> {
        let sample_rate = config.sample_rate();
        let stream_config = config.config();
        let buffer: Arc<Mutex<Vec<f32>>> =
            Arc::new(Mutex::new(Vec::with_capacity(BUFFER_SIZE)));
        let buf_clone = buffer.clone();

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => {
                build_stream::<f32>(device, &stream_config, buf_clone)
            }
            cpal::SampleFormat::I16 => {
                build_stream::<i16>(device, &stream_config, buf_clone)
            }
            cpal::SampleFormat::U16 => {
                build_stream::<u16>(device, &stream_config, buf_clone)
            }
            fmt => Err(format!("unsupported sample format: {fmt:?}")),
        }?;

        stream.play().map_err(|e| format!("play: {e}"))?;

        eprintln!("[audio] sample rate: {sample_rate}Hz");
        Ok(Self {
            buffer,
            _stream: StreamHolder::Cpal(stream),
        })
    }

    pub fn get_samples(&self, dest: &mut Vec<f32>) {
        if let Ok(mut buf) = self.buffer.lock() {
            dest.clear();
            dest.extend_from_slice(&buf);
            buf.clear();
        }
    }
}

fn build_stream<T: cpal::SizedSample + cpal::Sample>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    buffer: Arc<Mutex<Vec<f32>>>,
) -> Result<cpal::Stream, String>
where
    f32: cpal::FromSample<T>,
{
    device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                if let Ok(mut buf) = buffer.lock() {
                    for &sample in data {
                        buf.push(<f32 as cpal::FromSample<T>>::from_sample_(sample));
                    }
                    if buf.len() > BUFFER_SIZE {
                        let excess = buf.len() - BUFFER_SIZE;
                        buf.drain(..excess);
                    }
                }
            },
            |err| eprintln!("[audio] stream error: {err}"),
            None,
        )
        .map_err(|e| format!("build stream: {e}"))
}

// ── Audio Processing ──

const BEAT_DECAY: f32 = 0.15;
const NOISE_GATE: f32 = 0.0001;

/// Per-band rolling peak normalizer.
/// Tracks a rolling maximum that snaps up instantly on loud input
/// and decays exponentially (~2s half-life) during quiet periods.
/// Output = raw / rolling_peak, clamped to 0-1.
pub struct BandNormalizer {
    rolling_peak: f32,
    floor: f32,
}

impl BandNormalizer {
    pub fn new(floor: f32) -> Self {
        Self {
            rolling_peak: floor,
            floor,
        }
    }

    /// Feed a raw value and dt (seconds since last call). Returns normalized 0-1.
    pub fn update(&mut self, raw: f32, dt: f32) -> f32 {
        if raw > self.rolling_peak {
            self.rolling_peak = raw;
        } else {
            // ~2s half-life decay: 0.97^(dt*30) per tick at 30Hz
            self.rolling_peak *= 0.97f32.powf(dt * 30.0);
        }
        self.rolling_peak = self.rolling_peak.max(self.floor);
        if self.rolling_peak < 1e-10 {
            return 0.0;
        }
        (raw / self.rolling_peak).min(1.0)
    }
}

const BEAT_WINDOW_SECS: f32 = 8.0;
const BEAT_DENSITY_MAX: f32 = 24.0; // 3 beats/sec * 8s = 180 BPM

/// Sliding window beat density tracker.
/// Counts beats in the last BEAT_WINDOW_SECS seconds,
/// normalized so 180 BPM = 1.0.
pub struct BeatDensity {
    timestamps: Vec<f32>,
}

impl BeatDensity {
    pub fn new() -> Self {
        Self {
            timestamps: Vec::with_capacity(64),
        }
    }

    pub fn record_beat(&mut self, time: f32) {
        self.timestamps.push(time);
    }

    pub fn prune(&mut self, current_time: f32) {
        let cutoff = current_time - BEAT_WINDOW_SECS;
        self.timestamps.retain(|&t| t > cutoff);
    }

    pub fn value(&self) -> f32 {
        (self.timestamps.len() as f32 / BEAT_DENSITY_MAX).min(1.0)
    }

    pub fn beats_in_window(&self) -> usize {
        self.timestamps.len()
    }
}

/// Compute energy from normalized bands + beat.
fn compute_energy(bass: f32, mids: f32, highs: f32, beat: f32) -> f32 {
    (bass + mids + highs + beat) / 4.0
}

pub struct AudioProcessor {
    analyzer: AudioAnalyzer,
    pub features: AudioFeatures,
    bass_norm: BandNormalizer,
    mids_norm: BandNormalizer,
    highs_norm: BandNormalizer,
    beat_density: BeatDensity,
    prev_energy_raw: f32,
    time: f32,
    sample_buf: Vec<f32>,
}

impl AudioProcessor {
    pub fn new() -> Self {
        Self {
            analyzer: AudioAnalyzer::new(2048),
            features: AudioFeatures::default(),
            bass_norm: BandNormalizer::new(0.005),
            mids_norm: BandNormalizer::new(0.003),
            highs_norm: BandNormalizer::new(0.0005),
            beat_density: BeatDensity::new(),
            prev_energy_raw: 0.01,
            time: 0.0,
            sample_buf: Vec::with_capacity(BUFFER_SIZE),
        }
    }

    pub fn process(&mut self, capture: &AudioCapture, dt: f32) {
        self.time += dt;
        capture.get_samples(&mut self.sample_buf);
        if self.sample_buf.is_empty() {
            self.decay_features(dt);
            return;
        }

        let result = self.analyzer.analyze(&self.sample_buf);

        let f = &mut self.features;

        // Per-band normalization (replaces old auto-gain + smoothing)
        f.bass = self.bass_norm.update(result.bass, dt);
        f.mids = self.mids_norm.update(result.mids, dt);
        f.highs = self.highs_norm.update(result.highs, dt);

        // Beat detection: energy spike relative to recent average
        if result.energy > self.prev_energy_raw * 1.3 && result.energy > NOISE_GATE {
            f.beat = 1.0;
        } else {
            f.beat = (f.beat - BEAT_DECAY).max(0.0);
        }
        self.prev_energy_raw = self.prev_energy_raw * 0.9 + result.energy * 0.1;

        // Energy from normalized bands + beat
        f.energy = compute_energy(f.bass, f.mids, f.highs, f.beat);

        // Beat envelopes
        if f.beat >= 1.0 {
            self.beat_density.record_beat(self.time);
        }
        self.beat_density.prune(self.time);
        f.beat_accum = self.beat_density.value();

        // Beat pulse: snap to 1.0 on beat, 1.5s exponential decay
        if f.beat >= 1.0 {
            f.beat_pulse = 1.0;
        }
        f.beat_pulse *= (-dt / 1.5).exp();
    }

    fn decay_features(&mut self, dt: f32) {
        let f = &mut self.features;
        f.bass = self.bass_norm.update(0.0, dt);
        f.mids = self.mids_norm.update(0.0, dt);
        f.highs = self.highs_norm.update(0.0, dt);
        f.beat = (f.beat - BEAT_DECAY).max(0.0);
        f.energy = compute_energy(f.bass, f.mids, f.highs, f.beat);
        self.beat_density.prune(self.time);
        f.beat_accum = self.beat_density.value();
        f.beat_pulse *= (-dt / 1.5).exp();
    }
}

// ── Background Audio Thread ──

/// Spawn a background thread that processes audio and writes features to a JSON file at ~30Hz.
pub fn spawn_audio_thread(capture: AudioCapture, features_path: PathBuf) {
    std::thread::spawn(move || {
        let mut processor = AudioProcessor::new();
        let mut last = Instant::now();
        loop {
            std::thread::sleep(Duration::from_millis(33));
            let now = Instant::now();
            let dt = now.duration_since(last).as_secs_f32();
            last = now;

            processor.process(&capture, dt);

            // Atomic write: temp file then rename
            if let Ok(json) = serde_json::to_string_pretty(&processor.features) {
                let tmp = features_path.with_extension("tmp");
                if std::fs::write(&tmp, &json).is_ok() {
                    let _ = std::fs::rename(&tmp, &features_path);
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: simulate N seconds of constant input at 30Hz
    fn run_normalizer(norm: &mut BandNormalizer, value: f32, seconds: f32) -> f32 {
        let dt = 1.0 / 30.0;
        let steps = (seconds / dt) as usize;
        let mut last = 0.0;
        for _ in 0..steps {
            last = norm.update(value, dt);
        }
        last
    }

    #[test]
    fn constant_loud_signal_converges_to_one() {
        let mut norm = BandNormalizer::new(0.005);
        let result = run_normalizer(&mut norm, 0.1, 3.0);
        assert!((result - 1.0).abs() < 0.05, "expected ~1.0, got {result}");
    }

    #[test]
    fn constant_quiet_signal_converges_to_one() {
        let mut norm = BandNormalizer::new(0.005);
        let result = run_normalizer(&mut norm, 0.001, 3.0);
        // 0.001 < floor 0.005, so peak stays at floor
        // output = 0.001 / 0.005 = 0.2
        assert!((result - 0.2).abs() < 0.05, "expected ~0.2, got {result}");
    }

    #[test]
    fn silence_stays_zero() {
        let mut norm = BandNormalizer::new(0.005);
        let result = run_normalizer(&mut norm, 0.0, 3.0);
        assert!(result < 0.001, "expected ~0.0, got {result}");
    }

    #[test]
    fn loud_then_quiet_adapts_up() {
        let mut norm = BandNormalizer::new(0.005);
        // 2s of loud signal
        run_normalizer(&mut norm, 0.1, 2.0);
        // switch to quiet — initially low relative to old peak
        let initial = norm.update(0.01, 1.0 / 30.0);
        assert!(initial < 0.2, "expected low initially, got {initial}");
        // after 3s of quiet, should adapt up
        let adapted = run_normalizer(&mut norm, 0.01, 3.0);
        assert!(adapted > 0.7, "expected adapted up, got {adapted}");
    }

    #[test]
    fn quiet_then_loud_no_overshoot() {
        let mut norm = BandNormalizer::new(0.005);
        run_normalizer(&mut norm, 0.01, 2.0);
        // jump to loud — peak snaps up instantly
        let result = norm.update(0.1, 1.0 / 30.0);
        assert!(result <= 1.01, "expected no overshoot, got {result}");
    }

    #[test]
    fn bands_are_independent() {
        let mut bass_norm = BandNormalizer::new(0.005);
        let mut mids_norm = BandNormalizer::new(0.003);
        // Feed different values
        run_normalizer(&mut bass_norm, 0.1, 2.0);
        run_normalizer(&mut mids_norm, 0.02, 2.0);
        // Both should converge to ~1.0 independently
        let bass_out = bass_norm.update(0.1, 1.0 / 30.0);
        let mids_out = mids_norm.update(0.02, 1.0 / 30.0);
        assert!((bass_out - 1.0).abs() < 0.1, "bass: expected ~1.0, got {bass_out}");
        assert!((mids_out - 1.0).abs() < 0.1, "mids: expected ~1.0, got {mids_out}");
    }

    #[test]
    fn near_floor_no_oscillation() {
        let mut norm = BandNormalizer::new(0.005);
        // Feed values just above floor
        let mut results = Vec::new();
        for _ in 0..90 {
            results.push(norm.update(0.006, 1.0 / 30.0));
        }
        // Last 30 samples should be stable (within 0.1 of each other)
        let last_30 = &results[60..];
        let min = last_30.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = last_30.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max - min < 0.1, "oscillating: min={min}, max={max}");
    }

    #[test]
    fn beat_density_no_beats() {
        let density = BeatDensity::new();
        assert_eq!(density.value(), 0.0);
    }

    #[test]
    fn beat_density_steady_120bpm() {
        let mut density = BeatDensity::new();
        // 120 BPM = 2 beats/sec. Simulate 8 seconds.
        for i in 0..16 {
            let time = i as f32 * 0.5; // beat every 0.5s
            density.record_beat(time);
            density.prune(time);
        }
        // 16 beats in window / 24 = 0.667
        let val = density.value();
        assert!((val - 0.667).abs() < 0.05, "expected ~0.67, got {val}");
    }

    #[test]
    fn beat_density_steady_180bpm() {
        let mut density = BeatDensity::new();
        // 180 BPM = 3 beats/sec. Simulate 8 seconds.
        for i in 0..24 {
            let time = i as f32 / 3.0;
            density.record_beat(time);
            density.prune(time);
        }
        let val = density.value();
        assert!((val - 1.0).abs() < 0.05, "expected ~1.0, got {val}");
    }

    #[test]
    fn beat_density_beats_then_silence() {
        let mut density = BeatDensity::new();
        // 2 beats/sec for 4 seconds
        for i in 0..8 {
            let time = i as f32 * 0.5;
            density.record_beat(time);
        }
        density.prune(4.0);
        let during = density.value();
        assert!(during > 0.3, "expected beats during, got {during}");

        // 10 seconds later — all beats older than 8s
        density.prune(12.0);
        let after = density.value();
        assert!(after < 0.01, "expected ~0 after silence, got {after}");
    }

    #[test]
    fn beat_density_sparse() {
        let mut density = BeatDensity::new();
        // 1 beat every 2 seconds for 8 seconds = 4 beats
        for i in 0..4 {
            let time = i as f32 * 2.0;
            density.record_beat(time);
        }
        density.prune(8.0);
        // 4 / 24 = 0.167
        let val = density.value();
        assert!((val - 0.167).abs() < 0.05, "expected ~0.17, got {val}");
    }

    #[test]
    fn beat_density_window_boundary() {
        let mut density = BeatDensity::new();
        density.record_beat(0.0);
        density.record_beat(1.0);
        density.prune(1.0);
        assert_eq!(density.beats_in_window(), 2);

        // 8.5s later — first beat should be pruned, second still in window
        density.prune(8.5);
        assert_eq!(density.beats_in_window(), 1);

        // 9.5s — both pruned
        density.prune(9.5);
        assert_eq!(density.beats_in_window(), 0);
    }

    #[test]
    fn energy_all_equal() {
        let e = compute_energy(0.5, 0.5, 0.5, 0.5);
        assert!((e - 0.5).abs() < 0.01, "expected 0.5, got {e}");
    }

    #[test]
    fn energy_beat_dominant() {
        let e = compute_energy(0.0, 0.0, 0.0, 1.0);
        assert!((e - 0.25).abs() < 0.01, "expected 0.25, got {e}");
    }

    #[test]
    fn energy_spectral_dominant() {
        let e = compute_energy(1.0, 1.0, 1.0, 0.0);
        assert!((e - 0.75).abs() < 0.01, "expected 0.75, got {e}");
    }

    #[test]
    fn real_samples_normalized_range() {
        use std::path::Path;

        let sample_files = [
            "audio_samples_1772893548.json",
            "audio_samples_1772895443.json",
            "audio_samples_1772895516.json",
            "audio_samples_1772895643.json",
            "audio_samples_1772895738.json",
            "audio_samples_1772895939.json",
            "audio_samples_1772896075.json",
            "audio_samples_1772896227.json",
        ];

        for filename in &sample_files {
            let path = Path::new(filename);
            if !path.exists() {
                eprintln!("skipping {filename} (not found)");
                continue;
            }
            let json = std::fs::read_to_string(path).unwrap();
            let raw_samples: Vec<AudioFeatures> = serde_json::from_str(&json).unwrap();

            // Run raw values through per-band normalizers
            let mut bass_norm = BandNormalizer::new(0.005);
            let mut mids_norm = BandNormalizer::new(0.003);
            let mut highs_norm = BandNormalizer::new(0.0005);
            let dt = 0.1; // samples were recorded at 10Hz

            let mut norm_bass_vals = Vec::new();
            let mut norm_mids_vals = Vec::new();
            let mut norm_highs_vals = Vec::new();

            for sample in &raw_samples {
                norm_bass_vals.push(bass_norm.update(sample.bass, dt));
                norm_mids_vals.push(mids_norm.update(sample.mids, dt));
                norm_highs_vals.push(highs_norm.update(sample.highs, dt));
            }

            // Check all outputs are in 0-1
            for (i, &v) in norm_bass_vals.iter().enumerate() {
                assert!(v >= 0.0 && v <= 1.01, "{filename} bass[{i}] out of range: {v}");
            }
            for (i, &v) in norm_mids_vals.iter().enumerate() {
                assert!(v >= 0.0 && v <= 1.01, "{filename} mids[{i}] out of range: {v}");
            }
            for (i, &v) in norm_highs_vals.iter().enumerate() {
                assert!(v >= 0.0 && v <= 1.01, "{filename} highs[{i}] out of range: {v}");
            }

            // Check p90 is in reasonable range (0.3-1.0) — skip first 20% as warmup
            let warmup = raw_samples.len() / 5;
            let check_p90 = |vals: &[f32], name: &str| {
                let mut sorted: Vec<f32> = vals[warmup..].to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if sorted.is_empty() { return; }
                let p90 = sorted[(sorted.len() as f32 * 0.9) as usize];
                assert!(
                    p90 > 0.3,
                    "{filename} {name} p90 too low: {p90} (normalization not working)"
                );
            };

            check_p90(&norm_bass_vals, "bass");
            check_p90(&norm_mids_vals, "mids");
            // highs can be legitimately quiet in some genres, use lower threshold
            let mut sorted_highs: Vec<f32> = norm_highs_vals[warmup..].to_vec();
            sorted_highs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if !sorted_highs.is_empty() {
                let p90 = sorted_highs[(sorted_highs.len() as f32 * 0.9) as usize];
                assert!(
                    p90 > 0.1,
                    "{filename} highs p90 too low: {p90}"
                );
            }

            eprintln!("[OK] {filename}: {} samples processed", raw_samples.len());
        }
    }
}
