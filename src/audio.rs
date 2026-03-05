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

pub struct AudioCapture {
    buffer: Arc<Mutex<Vec<f32>>>,
    _stream: cpal::Stream,
    pub sample_rate: u32,
}

impl AudioCapture {
    pub fn new() -> Result<Self, String> {
        let host = cpal::default_host();

        // Try loopback first (captures system audio on macOS via ProcessTap)
        if let Some(device) = host.default_output_device() {
            if let Ok(capture) = Self::from_output_device(&device) {
                eprintln!("[audio] loopback capture active");
                return Ok(capture);
            }
        }

        // Fall back to mic
        let device = host
            .default_input_device()
            .ok_or("no audio input device")?;
        eprintln!("[audio] mic capture active");
        Self::from_input_device(&device)
    }

    /// Build capture from an output device (loopback/system audio).
    /// Uses the device's output config since it has no input scope.
    fn from_output_device(device: &cpal::Device) -> Result<Self, String> {
        let config = device
            .default_output_config()
            .map_err(|e| format!("output config: {e}"))?;
        Self::build_capture(device, &config)
    }

    /// Build capture from an input device (microphone).
    fn from_input_device(device: &cpal::Device) -> Result<Self, String> {
        let config = device
            .default_input_config()
            .map_err(|e| format!("input config: {e}"))?;
        Self::build_capture(device, &config)
    }

    fn build_capture(
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
            _stream: stream,
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
