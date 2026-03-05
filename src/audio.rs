use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::f32::consts::PI;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Default)]
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
    pub sample_rate: u32,
}

impl AudioCapture {
    /// List devices, let user pick. Option 0 = System Audio (SCK).
    pub fn select_and_start() -> Result<Self, String> {
        let host = cpal::default_host();
        let devices: Vec<_> = host
            .input_devices()
            .map_err(|e| format!("enumerate devices: {e}"))?
            .collect();

        eprintln!("\n── Audio Devices ──");
        eprintln!("  [0] System Audio (ScreenCaptureKit)");
        for (i, dev) in devices.iter().enumerate() {
            let name = dev.name().unwrap_or_else(|_| "???".into());
            let config_info = match dev.default_input_config() {
                Ok(c) => format!("{}Hz {:?}", c.sample_rate(), c.sample_format()),
                Err(_) => "no config".into(),
            };
            eprintln!("  [{}] {name} ({config_info})", i + 1);
        }
        eprintln!("  [Enter] = System Audio");
        eprint!("Select device: ");

        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(|e| format!("read stdin: {e}"))?;
        let input = input.trim();

        // Default or "0" → SCK system audio
        if input.is_empty() || input == "0" {
            return Self::new_system_audio();
        }

        let idx: usize = input
            .parse()
            .map_err(|_| format!("invalid number: {input}"))?;
        let dev_idx = idx - 1; // offset by 1 since [0] is SCK
        let device = devices
            .into_iter()
            .nth(dev_idx)
            .ok_or_else(|| format!("device {idx} not found"))?;

        let name = device.name().unwrap_or_else(|_| "???".into());
        let config = device
            .default_input_config()
            .map_err(|e| format!("input config for {name}: {e}"))?;

        eprintln!("[audio] using: {name} ({}Hz)", config.sample_rate());
        Self::build_cpal_capture(&device, &config)
    }

    /// Capture system audio via ScreenCaptureKit (macOS 12.3+).
    pub fn new_system_audio() -> Result<Self, String> {
        let buffer = Arc::new(Mutex::new(Vec::with_capacity(BUFFER_SIZE)));
        let (stream, rate) =
            crate::sck_audio::start_system_audio_capture(buffer.clone(), BUFFER_SIZE)?;
        Ok(Self {
            _stream: StreamHolder::Sck(stream),
            buffer,
            sample_rate: rate,
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

        Ok(Self {
            buffer,
            _stream: StreamHolder::Cpal(stream),
            sample_rate,
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
const SMOOTH_ATTACK: f32 = 0.85;
const SMOOTH_RELEASE: f32 = 0.97;
const TARGET_ENERGY: f32 = 0.05;
const MAX_GAIN: f32 = 50.0;
const GAIN_ATTACK: f32 = 0.05;
const GAIN_RELEASE: f32 = 0.003;
const NOISE_GATE: f32 = 0.0001;

pub struct AudioProcessor {
    analyzer: AudioAnalyzer,
    pub features: AudioFeatures,
    peak_energy: f32,
    auto_gain: f32,
    prev_energy_gained: f32,
    sample_buf: Vec<f32>,
}

impl AudioProcessor {
    pub fn new() -> Self {
        Self {
            analyzer: AudioAnalyzer::new(2048),
            features: AudioFeatures::default(),
            peak_energy: 0.01,
            auto_gain: 1.0,
            prev_energy_gained: 0.01,
            sample_buf: Vec::with_capacity(BUFFER_SIZE),
        }
    }

    pub fn process(&mut self, capture: &AudioCapture, dt: f32) {
        capture.get_samples(&mut self.sample_buf);
        if self.sample_buf.is_empty() {
            self.decay_features(dt);
            return;
        }

        let result = self.analyzer.analyze(&self.sample_buf);

        // Auto-gain: track peak energy and normalize
        let rate = if result.energy > self.peak_energy {
            GAIN_ATTACK
        } else {
            GAIN_RELEASE
        };
        self.peak_energy += (result.energy - self.peak_energy) * rate;
        self.peak_energy = self.peak_energy.max(0.001);
        let target_gain = (TARGET_ENERGY / self.peak_energy).clamp(1.0, MAX_GAIN);
        self.auto_gain += (target_gain - self.auto_gain) * 0.02;

        let gain = self.auto_gain;
        let gate = if result.energy > NOISE_GATE { 1.0 } else { 0.0 };

        // Asymmetric smoothing (fast attack, slow release)
        let f = &mut self.features;
        f.bass = smooth(f.bass, result.bass * gain * gate);
        f.mids = smooth(f.mids, result.mids * gain * gate);
        f.highs = smooth(f.highs, result.highs * gain * gate);
        f.energy = smooth(f.energy, result.energy * gain * gate);

        // Beat detection: energy spike relative to recent average
        let gained_energy = result.energy * gain;
        if gained_energy > self.prev_energy_gained * 1.3 && gained_energy > 0.008 {
            f.beat = 1.0;
        } else {
            f.beat = (f.beat - BEAT_DECAY).max(0.0);
        }
        self.prev_energy_gained = self.prev_energy_gained * 0.9 + gained_energy * 0.1;

        self.update_envelopes(dt);
    }

    fn decay_features(&mut self, dt: f32) {
        let f = &mut self.features;
        f.bass = smooth(f.bass, 0.0);
        f.mids = smooth(f.mids, 0.0);
        f.highs = smooth(f.highs, 0.0);
        f.energy = smooth(f.energy, 0.0);
        f.beat = (f.beat - BEAT_DECAY).max(0.0);
        self.update_envelopes(dt);
    }

    fn update_envelopes(&mut self, dt: f32) {
        let f = &mut self.features;

        // Beat accumulator: +0.05 on beat, 6s exponential decay
        if f.beat >= 1.0 {
            f.beat_accum = (f.beat_accum + 0.05).min(1.0);
        }
        f.beat_accum *= (-dt / 6.0).exp();

        // Beat pulse: snap to 1.0 on beat, 1.5s exponential decay
        if f.beat >= 1.0 {
            f.beat_pulse = 1.0;
        }
        f.beat_pulse *= (-dt / 1.5).exp();
    }
}

fn smooth(current: f32, target: f32) -> f32 {
    let retain = if target > current {
        SMOOTH_ATTACK
    } else {
        SMOOTH_RELEASE
    };
    current * retain + target * (1.0 - retain)
}
