# Audio Reactivity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add audio-reactive visuals via a weight matrix that maps audio features (bass/mids/highs/energy/beat/beat_accum) to flame and KIFS params, ported from silly-visualizer's battle-tested audio pipeline.

**Architecture:** Audio capture (cpal) runs via OS callbacks into a shared ring buffer. Each frame, the main thread reads samples, runs FFT analysis, applies auto-gain + smoothing + beat detection, then multiplies the resulting features by a hot-reloadable weight matrix to produce param offsets on top of the genome's base state.

**Tech Stack:** Rust, cpal 0.17, rustfft 6.4, serde_json, wgpu 28

---

### Task 1: Add audio dependencies

**Files:**
- Modify: `Cargo.toml`

**Step 1: Add cpal and rustfft**

Add to `[dependencies]`:
```toml
cpal = "0.17"
rustfft = "6.4"
```

**Step 2: Verify it compiles**

Run: `cargo build`
Expected: Compiles (cpal may pull in coreaudio-sys on macOS)

**Step 3: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "chore: add cpal and rustfft dependencies for audio reactivity"
```

---

### Task 2: Create AudioFeatures struct and AudioAnalyzer

**Files:**
- Create: `src/audio.rs`
- Modify: `src/main.rs` (add `mod audio;`)

**Step 1: Write src/audio.rs with FFT analyzer**

```rust
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::f32::consts::PI;
use std::sync::{Arc, Mutex};

/// Smoothed audio features extracted from FFT analysis.
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

/// Raw FFT analysis result (before smoothing/gain).
struct AnalysisResult {
    bands: [f32; 16],
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

        // Pre-compute Hann window
        let window: Vec<f32> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / fft_size as f32).cos()))
            .collect();

        // Pre-compute 16 logarithmic band ranges
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
        // Fill buffer with windowed samples, zero-pad if needed
        for i in 0..self.fft_size {
            let s = if i < samples.len() { samples[i] } else { 0.0 };
            self.buffer[i] = Complex::new(s * self.window[i], 0.0);
        }

        // FFT in-place
        self.fft.process(&mut self.buffer);

        // Magnitudes (half spectrum)
        let n = self.fft_size as f32;
        for i in 0..self.magnitudes.len() {
            self.magnitudes[i] = self.buffer[i].norm() / n;
        }

        // 16 bands
        let mut bands = [0.0f32; 16];
        for (b, &(lo, hi)) in self.band_ranges.iter().enumerate() {
            let count = (hi - lo).max(1) as f32;
            bands[b] = self.magnitudes[lo..hi].iter().sum::<f32>() / count;
        }

        // Band groups
        let bass = (bands[0] + bands[1] + bands[2]) / 3.0;
        let mids = (bands[4] + bands[5] + bands[6] + bands[7]) / 4.0;
        let highs = (bands[10] + bands[11] + bands[12] + bands[13]) / 4.0;
        let energy = bands.iter().sum::<f32>() / 16.0;

        AnalysisResult { bands, bass, mids, highs, energy }
    }
}

fn band_edge(half: usize, i: usize) -> usize {
    if i == 0 { return 0; }
    let frac = i as f32 / 16.0;
    let edge = (half as f32 * (2.0f32.powf(frac * 10.0) - 1.0) / 1023.0) as usize;
    edge.min(half)
}
```

**Step 2: Add mod declaration**

Add `mod audio;` in `src/main.rs` after `mod genome;`.

**Step 3: Verify it compiles**

Run: `cargo build`
Expected: Compiles (dead_code warnings expected)

**Step 4: Commit**

```bash
git add src/audio.rs src/main.rs
git commit -m "feat: AudioFeatures struct and FFT analyzer (16 log bands)"
```

---

### Task 3: Add audio capture (cpal ring buffer)

**Files:**
- Modify: `src/audio.rs`

**Step 1: Add capture to audio.rs**

Add after the AudioAnalyzer:

```rust
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

const BUFFER_SIZE: usize = 4096;

pub struct AudioCapture {
    buffer: Arc<Mutex<Vec<f32>>>,
    _stream: cpal::Stream,
    pub sample_rate: u32,
}

impl AudioCapture {
    /// Try loopback (system audio), fall back to mic input.
    pub fn new() -> Result<Self, String> {
        let host = cpal::default_host();

        // Try loopback first (captures system audio on macOS via ProcessTap)
        if let Some(device) = host.default_output_device() {
            if let Ok(capture) = Self::from_device(&device) {
                eprintln!("[audio] loopback capture active");
                return Ok(capture);
            }
        }

        // Fall back to mic
        let device = host.default_input_device()
            .ok_or("no audio input device")?;
        eprintln!("[audio] mic capture active");
        Self::from_device(&device)
    }

    fn from_device(device: &cpal::Device) -> Result<Self, String> {
        let config = device.default_input_config()
            .map_err(|e| format!("input config: {e}"))?;
        let sample_rate = config.sample_rate().0;
        let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::with_capacity(BUFFER_SIZE)));
        let buf_clone = buffer.clone();

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => build_stream::<f32>(device, &config.into(), buf_clone),
            cpal::SampleFormat::I16 => build_stream::<i16>(device, &config.into(), buf_clone),
            cpal::SampleFormat::U16 => build_stream::<u16>(device, &config.into(), buf_clone),
            fmt => Err(format!("unsupported sample format: {fmt:?}")),
        }?;

        stream.play().map_err(|e| format!("play: {e}"))?;

        Ok(Self { buffer, _stream: stream, sample_rate })
    }

    pub fn get_samples(&self, dest: &mut Vec<f32>) {
        if let Ok(mut buf) = self.buffer.lock() {
            dest.clear();
            dest.extend_from_slice(&buf);
            buf.clear();
        }
    }
}

fn build_stream<T: cpal::SizedSample + Into<f32>>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    buffer: Arc<Mutex<Vec<f32>>>,
) -> Result<cpal::Stream, String> {
    device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            if let Ok(mut buf) = buffer.lock() {
                for &sample in data {
                    buf.push(sample.into());
                }
                // Ring buffer: keep only last BUFFER_SIZE samples
                if buf.len() > BUFFER_SIZE {
                    let excess = buf.len() - BUFFER_SIZE;
                    buf.drain(..excess);
                }
            }
        },
        |err| eprintln!("[audio] stream error: {err}"),
        None,
    ).map_err(|e| format!("build stream: {e}"))
}
```

**Step 2: Verify it compiles**

Run: `cargo build`

**Step 3: Commit**

```bash
git add src/audio.rs
git commit -m "feat: audio capture via cpal (loopback + mic fallback)"
```

---

### Task 4: Add audio processing (gain, smoothing, beat detection)

**Files:**
- Modify: `src/audio.rs`

**Step 1: Add AudioProcessor with state + processing**

Add after AudioCapture:

```rust
// ── Processing Constants ──
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
    slow_energy: f32,
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
            slow_energy: 0.0,
            prev_energy_gained: 0.0,
            sample_buf: Vec::with_capacity(BUFFER_SIZE),
        }
    }

    /// Process one frame of audio. Call once per render frame.
    pub fn process(&mut self, capture: &AudioCapture, dt: f32) {
        capture.get_samples(&mut self.sample_buf);
        if self.sample_buf.is_empty() {
            // No new samples — just decay
            self.decay_features(dt);
            return;
        }

        let result = self.analyzer.analyze(&self.sample_buf);

        // Auto-gain
        let rate = if result.energy > self.peak_energy { GAIN_ATTACK } else { GAIN_RELEASE };
        self.peak_energy += (result.energy - self.peak_energy) * rate;
        self.peak_energy = self.peak_energy.max(0.001);
        let target_gain = (TARGET_ENERGY / self.peak_energy).clamp(1.0, MAX_GAIN);
        self.auto_gain += (target_gain - self.auto_gain) * 0.02;

        let gain = self.auto_gain;
        let gate = if result.energy > NOISE_GATE { 1.0 } else { 0.0 };

        // Asymmetric smoothing
        let f = &mut self.features;
        f.bass = smooth(f.bass, result.bass * gain * gate);
        f.mids = smooth(f.mids, result.mids * gain * gate);
        f.highs = smooth(f.highs, result.highs * gain * gate);
        f.energy = smooth(f.energy, result.energy * gain * gate);

        // Beat detection
        let gained_energy = result.energy * gain;
        if gained_energy > self.prev_energy_gained * 1.3 && gained_energy > 0.008 {
            f.beat = 1.0;
        } else {
            f.beat = (f.beat - BEAT_DECAY).max(0.0);
        }
        self.prev_energy_gained = self.prev_energy_gained * 0.9 + gained_energy * 0.1;

        // Envelopes
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

        // Slow energy: 7-second EMA
        let alpha = (dt / 7.0).min(1.0);
        self.slow_energy += (f.energy - self.slow_energy) * alpha;

        // Beat accumulator: +0.05 on beat, 6s decay
        if f.beat >= 1.0 {
            f.beat_accum = (f.beat_accum + 0.05).min(1.0);
        }
        f.beat_accum *= (-dt / 6.0).exp();

        // Beat pulse: snap to 1.0 on beat, 1.5s decay
        if f.beat >= 1.0 {
            f.beat_pulse = 1.0;
        }
        f.beat_pulse *= (-dt / 1.5).exp();
    }
}

fn smooth(current: f32, target: f32) -> f32 {
    let retain = if target > current { SMOOTH_ATTACK } else { SMOOTH_RELEASE };
    current * retain + target * (1.0 - retain)
}
```

**Step 2: Verify it compiles**

Run: `cargo build`

**Step 3: Commit**

```bash
git add src/audio.rs
git commit -m "feat: audio processing (auto-gain, smoothing, beat detection, envelopes)"
```

---

### Task 5: Create AudioWeights struct

**Files:**
- Create: `src/audio_weights.rs`
- Modify: `src/main.rs` (add `mod audio_weights;`)

**Step 1: Write audio_weights.rs**

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::fs;

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct SignalWeights {
    #[serde(default)]
    pub bass: f32,
    #[serde(default)]
    pub mids: f32,
    #[serde(default)]
    pub highs: f32,
    #[serde(default)]
    pub energy: f32,
    #[serde(default)]
    pub beat: f32,
    #[serde(default)]
    pub beat_accum: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct AudioWeights {
    #[serde(default)]
    pub targets: HashMap<String, SignalWeights>,
    #[serde(default)]
    pub caps: HashMap<String, [f32; 2]>,
}

impl AudioWeights {
    pub fn load(path: &Path) -> Result<Self, String> {
        let json = fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_str(&json)
            .map_err(|e| format!("parse {}: {e}", path.display()))
    }

    /// Compute the offset for a named target given current audio features.
    pub fn offset(&self, target: &str, features: &crate::audio::AudioFeatures) -> f32 {
        let Some(w) = self.targets.get(target) else { return 0.0 };
        w.bass * features.bass
            + w.mids * features.mids
            + w.highs * features.highs
            + w.energy * features.energy
            + w.beat * features.beat
            + w.beat_accum * features.beat_accum
    }

    /// Clamp a value to caps for a named target, if defined.
    pub fn clamp(&self, target: &str, value: f32) -> f32 {
        match self.caps.get(target) {
            Some(&[lo, hi]) => value.clamp(lo, hi),
            None => value,
        }
    }
}
```

**Step 2: Add mod declaration**

Add `mod audio_weights;` in `src/main.rs` after `mod audio;`.

**Step 3: Verify it compiles**

Run: `cargo build`

**Step 4: Commit**

```bash
git add src/audio_weights.rs src/main.rs
git commit -m "feat: AudioWeights struct with weight matrix and caps"
```

---

### Task 6: Add drift_speed and color_shift params to shaders

**Files:**
- Modify: `flame_compute.wgsl`
- Modify: `playground.wgsl`

**Step 1: Use param(7) as drift_speed in compute shader**

In `flame_compute.wgsl`, update the comment header to note param[7] = drift_speed.

Change the `apply_xform` function's angle drift line from:
```wgsl
let q = rot2(angle + t * 0.07 * f32(idx + 1)) * p * scale
```
to:
```wgsl
let drift = param(7);
let q = rot2(angle + t * 0.07 * drift * f32(idx + 1)) * p * scale
```

**Step 2: Add color_shift in fragment shader**

In `playground.wgsl`, after reading `kifs_bright`, add:
```wgsl
let color_shift = param(56);
```

Then update the flame color line to include color_shift:
```wgsl
let core_color = palette(avg_color + u.time * 0.02 + color_shift);
```

Update comment headers in both files.

**Step 3: Verify it compiles and runs**

Run: `cargo build`

**Step 4: Commit**

```bash
git add flame_compute.wgsl playground.wgsl
git commit -m "feat: drift_speed (param 7) and color_shift (param 56) shader params"
```

---

### Task 7: Integrate audio into App

**Files:**
- Modify: `src/main.rs`

**Step 1: Add audio fields to App struct**

```rust
use crate::audio::{AudioCapture, AudioProcessor};
use crate::audio_weights::AudioWeights;
```

Add to `App` struct:
```rust
audio_capture: Option<AudioCapture>,
audio_processor: AudioProcessor,
audio_weights: AudioWeights,
audio_enabled: bool,
mutation_accum: f32,
```

**Step 2: Initialize in App::new()**

```rust
audio_capture: None,  // initialized in resumed()
audio_processor: AudioProcessor::new(),
audio_weights: load_audio_weights(),
audio_enabled: true,
mutation_accum: 0.0,
```

Add helper:
```rust
fn audio_weights_path() -> PathBuf {
    project_dir().join("audio_weights.json")
}

fn load_audio_weights() -> AudioWeights {
    AudioWeights::load(&audio_weights_path()).unwrap_or_default()
}
```

**Step 3: Start audio capture in resumed()**

After the genome loading block:
```rust
match AudioCapture::new() {
    Ok(cap) => {
        eprintln!("[audio] capture started ({}Hz)", cap.sample_rate);
        self.audio_capture = Some(cap);
    }
    Err(e) => eprintln!("[audio] capture failed: {e} (visuals-only mode)"),
}
```

**Step 4: Add audio_weights.json to file watcher**

Add `audio_weights_path()` to the paths vec for FileWatcher.

In `check_file_changes()`, add:
```rust
if path.ends_with("audio_weights.json") {
    reload_weights = true;
}
```
And:
```rust
if reload_weights {
    self.audio_weights = load_audio_weights();
    eprintln!("[weights] reloaded");
}
```

**Step 5: Apply audio each frame in RedrawRequested**

After `check_file_changes()` and before the lerp, add:
```rust
// Process audio
if self.audio_enabled {
    if let Some(ref capture) = self.audio_capture {
        self.audio_processor.process(capture, dt);
    }

    let features = &self.audio_processor.features;
    let base = self.genome.flatten();
    let w = &self.audio_weights;

    // Apply weight matrix offsets to target_params
    self.target_params[2] = w.clamp("trail", base[2] + w.offset("trail", features));
    self.target_params[3] = w.clamp("flame_bright", base[3] + w.offset("flame_bright", features));
    self.target_params[4] = w.clamp("kifs_fold", base[4] + w.offset("kifs_fold", features));
    self.target_params[5] = w.clamp("kifs_scale", base[5] + w.offset("kifs_scale", features));
    self.target_params[6] = w.clamp("kifs_bright", base[6] + w.offset("kifs_bright", features));
    self.target_params[7] = 1.0 + w.offset("drift_speed", features); // drift default 1.0
    self.target_params[56] = w.offset("color_shift", features);

    // Transform weights
    for i in 0..4 {
        let name = format!("xf{}_weight", i);
        let base_idx = 8 + i * 12;
        self.target_params[base_idx] = (base[base_idx] + w.offset(&name, features)).max(0.0);
    }

    // Morph rate modulation (additive to current index rate)
    let morph_offset = w.offset("morph_rate", features);
    // Applied by scaling the lerp rate below

    // Mutation trigger
    self.mutation_accum += w.offset("mutation_trigger", features) * dt;
    if self.mutation_accum >= 1.0 {
        self.mutation_accum = 0.0;
        let old = self.genome.flatten();
        self.genome_history.push(old);
        if self.genome_history.len() > 10 {
            self.genome_history.remove(0);
        }
        self.genome = self.genome.mutate();
        // Don't set target_params here — let weight matrix keep controlling
        eprintln!("[auto-evolve] → {}", self.genome.name);
    }
}
```

**Step 6: Add 'A' key to toggle audio**

In the keyboard handler, add to the Character match:
```rust
"a" => {
    self.audio_enabled = !self.audio_enabled;
    eprintln!("[audio] {}", if self.audio_enabled { "enabled" } else { "disabled" });
}
```

**Step 7: Verify it compiles**

Run: `cargo build`

**Step 8: Commit**

```bash
git add src/main.rs
git commit -m "feat: integrate audio capture + weight matrix into render loop"
```

---

### Task 8: Create default audio_weights.json

**Files:**
- Create: `audio_weights.json`

**Step 1: Write default weights file**

```json
{
  "targets": {
    "kifs_fold": { "bass": 0.3, "mids": 0.0, "highs": 0.0, "energy": 0.0, "beat": 0.0, "beat_accum": 0.0 },
    "kifs_scale": { "bass": 0.0, "mids": 0.0, "highs": 0.0, "energy": 0.1, "beat": 0.0, "beat_accum": 0.0 },
    "kifs_bright": { "bass": 0.0, "mids": 0.0, "highs": 0.0, "energy": 0.2, "beat": 0.0, "beat_accum": 0.0 },
    "flame_bright": { "bass": 0.0, "mids": 0.0, "highs": 0.0, "energy": 0.15, "beat": 0.0, "beat_accum": 0.0 },
    "trail": { "bass": 0.0, "mids": 0.0, "highs": 0.0, "energy": 0.0, "beat": -0.1, "beat_accum": 0.0 },
    "drift_speed": { "bass": 0.0, "mids": 0.0, "highs": 0.0, "energy": 0.3, "beat": 0.0, "beat_accum": 0.0 },
    "color_shift": { "bass": -0.1, "mids": 0.0, "highs": 0.05, "energy": 0.0, "beat": 0.0, "beat_accum": 0.0 },
    "morph_rate": { "bass": 0.0, "mids": 0.0, "highs": 0.0, "energy": 0.0, "beat": 2.0, "beat_accum": 0.0 },
    "xf0_weight": { "bass": 0.0, "mids": 0.0, "highs": 0.0, "energy": 0.05, "beat": 0.0, "beat_accum": 0.0 },
    "xf1_weight": { "bass": 0.0, "mids": 0.05, "highs": 0.0, "energy": 0.0, "beat": 0.0, "beat_accum": 0.0 },
    "xf2_weight": { "bass": 0.0, "mids": 0.0, "highs": 0.05, "energy": 0.0, "beat": 0.0, "beat_accum": 0.0 },
    "xf3_weight": { "bass": 0.05, "mids": 0.0, "highs": 0.0, "energy": 0.0, "beat": 0.0, "beat_accum": 0.0 },
    "mutation_trigger": { "bass": 0.0, "mids": 0.0, "highs": 0.0, "energy": 0.0, "beat": 0.0, "beat_accum": 1.0 }
  },
  "caps": {
    "kifs_fold": [0.3, 0.9],
    "kifs_scale": [1.3, 2.5],
    "trail": [0.1, 0.6],
    "flame_bright": [0.1, 0.8],
    "drift_speed": [0.2, 5.0]
  }
}
```

**Step 2: Commit**

```bash
git add audio_weights.json
git commit -m "feat: default audio_weights.json with conservative starting weights"
```

---

### Task 9: Verify end-to-end and tag

**Step 1: Run the app with music playing**

Run: `cargo run`
Play any song on the system. Observe:
- Terminal shows `[audio] loopback capture active` or `[audio] mic capture active`
- KIFS should subtly pulse on bass hits
- Flame brightness should respond to energy
- Press `A` to toggle audio on/off

**Step 2: Test hot-reload of weights**

Edit `audio_weights.json` — change `kifs_fold.bass` from 0.3 to 0.8. Save. Observe stronger fold response. Terminal shows `[weights] reloaded`.

**Step 3: Test mutation trigger**

With beat_accum-driven mutation_trigger, sustained rhythmic music should trigger auto-evolve. Watch for `[auto-evolve]` messages in terminal.

**Step 4: Tag the release**

```bash
git tag -a v0.5.0-audio-reactive -m "Audio reactivity with weight matrix, cpal capture, FFT analysis"
```

---

## Summary

| Task | What | Key Files |
|------|------|-----------|
| 1 | Add cpal + rustfft deps | Cargo.toml |
| 2 | AudioFeatures + FFT analyzer | src/audio.rs |
| 3 | Audio capture (cpal loopback + mic) | src/audio.rs |
| 4 | Audio processing (gain, smoothing, beat) | src/audio.rs |
| 5 | AudioWeights struct | src/audio_weights.rs |
| 6 | drift_speed + color_shift in shaders | both .wgsl |
| 7 | Integrate audio into App render loop | src/main.rs |
| 8 | Default audio_weights.json | audio_weights.json |
| 9 | Verify end-to-end, tag v0.5.0 | — |
