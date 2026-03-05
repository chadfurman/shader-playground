use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::f32::consts::PI;
use std::sync::Arc;

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
