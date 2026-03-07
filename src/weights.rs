use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// 1D smooth value noise (deterministic, no external crate).
fn value_noise(t: f32) -> f32 {
    let i = t.floor() as i32;
    let f = t - t.floor();
    let f = f * f * (3.0 - 2.0 * f); // smoothstep
    let a = hash_f32(i);
    let b = hash_f32(i + 1);
    a + (b - a) * f
}

fn hash_f32(n: i32) -> f32 {
    let n = (n as u32).wrapping_mul(1597334677);
    let n = n ^ (n >> 16);
    let n = n.wrapping_mul(2654435769);
    (n as f32) / 4294967295.0 * 2.0 - 1.0 // -1 to 1
}

pub struct TimeSignals {
    pub time: f32,           // raw elapsed time, steady linear ramp
    pub time_slow: f32,      // noise at 0.05 speed, ~20s wander
    pub time_med: f32,       // noise at 0.2 speed, ~5s wander
    pub time_fast: f32,      // noise at 0.8 speed, ~1.25s wander
    pub time_noise: f32,     // noise at 0.3 speed, organic wandering
    pub time_drift: f32,     // noise at 0.02 speed, ~50s glacial drift
    pub time_flutter: f32,   // noise at 1.5 speed, quick flicker
    pub time_walk: f32,      // random walk — accumulated noise, never reverses
    pub time_envelope: f32,  // time since last mutation, capped at 1.0
}

impl TimeSignals {
    pub fn compute(time: f32, time_since_mutation: f32, random_walk: f32) -> Self {
        Self {
            time,
            time_slow: value_noise(time * 0.05),
            time_med: value_noise(time * 0.2 + 100.0),
            time_fast: value_noise(time * 0.8 + 200.0),
            time_noise: value_noise(time * 0.3 + 300.0),
            time_drift: value_noise(time * 0.02 + 400.0),
            time_flutter: value_noise(time * 1.5 + 500.0),
            time_walk: random_walk,
            time_envelope: (time_since_mutation / 10.0).min(1.0),
        }
    }
}

/// Public wrapper for value_noise (used by main.rs for random walk accumulation).
pub fn value_noise_pub(t: f32) -> f32 {
    value_noise(t)
}

const AUDIO_SIGNAL_COUNT: f32 = 6.0;  // bass, mids, highs, energy, beat, beat_accum
const TIME_SIGNAL_COUNT: f32 = 9.0;   // time, time_slow, time_med, time_fast, time_noise, time_drift, time_flutter, time_walk, time_envelope
const PARAMS_PER_XF: usize = 32;

/// Per-transform field names in order (matching genome flatten layout).
const XF_FIELDS: [&str; PARAMS_PER_XF] = [
    "weight", "angle", "scale", "offset_x", "offset_y", "color",
    "linear", "sinusoidal", "spherical", "swirl", "horseshoe", "handkerchief",
    "julia", "polar", "disc", "rings", "bubble", "fisheye",
    "exponential", "spiral", "diamond", "bent", "waves", "popcorn",
    "fan", "eyefish", "cross", "tangent", "cosine", "blob",
    "noise", "curl",
];

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Weights {
    #[serde(default)]
    pub bass: HashMap<String, f32>,
    #[serde(default)]
    pub mids: HashMap<String, f32>,
    #[serde(default)]
    pub highs: HashMap<String, f32>,
    #[serde(default)]
    pub energy: HashMap<String, f32>,
    #[serde(default)]
    pub beat: HashMap<String, f32>,
    #[serde(default)]
    pub beat_accum: HashMap<String, f32>,
    #[serde(default)]
    pub time: HashMap<String, f32>,
    #[serde(default)]
    pub time_slow: HashMap<String, f32>,
    #[serde(default)]
    pub time_med: HashMap<String, f32>,
    #[serde(default)]
    pub time_fast: HashMap<String, f32>,
    #[serde(default)]
    pub time_noise: HashMap<String, f32>,
    #[serde(default)]
    pub time_drift: HashMap<String, f32>,
    #[serde(default)]
    pub time_flutter: HashMap<String, f32>,
    #[serde(default)]
    pub time_walk: HashMap<String, f32>,
    #[serde(default)]
    pub time_envelope: HashMap<String, f32>,
}

impl Weights {
    pub fn load(path: &Path) -> Result<Self, String> {
        let json =
            fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_str(&json).map_err(|e| format!("parse {}: {e}", path.display()))
    }

    /// Build the list of (weight-map, signal-value, divisor) tuples used by all apply methods.
    fn signal_list<'a>(
        &'a self,
        features: &crate::audio::AudioFeatures,
        time_signals: &TimeSignals,
    ) -> Vec<(&'a HashMap<String, f32>, f32, f32)> {
        vec![
            (&self.bass, features.bass, AUDIO_SIGNAL_COUNT),
            (&self.mids, features.mids, AUDIO_SIGNAL_COUNT),
            (&self.highs, features.highs, AUDIO_SIGNAL_COUNT),
            (&self.energy, features.energy, AUDIO_SIGNAL_COUNT),
            (&self.beat, features.beat, AUDIO_SIGNAL_COUNT),
            (&self.beat_accum, features.beat_accum, AUDIO_SIGNAL_COUNT),
            (&self.time, time_signals.time, TIME_SIGNAL_COUNT),
            (&self.time_slow, time_signals.time_slow, TIME_SIGNAL_COUNT),
            (&self.time_med, time_signals.time_med, TIME_SIGNAL_COUNT),
            (&self.time_fast, time_signals.time_fast, TIME_SIGNAL_COUNT),
            (&self.time_noise, time_signals.time_noise, TIME_SIGNAL_COUNT),
            (&self.time_drift, time_signals.time_drift, TIME_SIGNAL_COUNT),
            (&self.time_flutter, time_signals.time_flutter, TIME_SIGNAL_COUNT),
            (&self.time_walk, time_signals.time_walk, TIME_SIGNAL_COUNT),
            (&self.time_envelope, time_signals.time_envelope, TIME_SIGNAL_COUNT),
        ]
    }

    /// Apply weights to global params only.
    pub fn apply_globals(
        &self,
        base: &[f32; 12],
        features: &crate::audio::AudioFeatures,
        time_signals: &TimeSignals,
    ) -> [f32; 12] {
        let mut result = *base;
        let signals = self.signal_list(features, time_signals);
        for (_signal_idx, (weights, signal_val, divisor)) in signals.iter().enumerate() {
            for (name, &weight) in *weights {
                if name == "mutation_rate" { continue; }
                if name.starts_with("xf") { continue; } // transform params handled separately
                if let Some(idx) = global_index(name) {
                    result[idx] += weight * signal_val / divisor;
                }
            }
        }
        result
    }

    /// Apply weights to transform params.
    pub fn apply_transforms(
        &self,
        base: &[f32],
        num_transforms: usize,
        features: &crate::audio::AudioFeatures,
        time_signals: &TimeSignals,
    ) -> Vec<f32> {
        let mut result = base.to_vec();
        let signals = self.signal_list(features, time_signals);
        for (_signal_idx, (weights, signal_val, divisor)) in signals.iter().enumerate() {
            for (name, &weight) in *weights {
                if name == "mutation_rate" { continue; }
                // xfN_ wildcard
                if let Some(field) = name.strip_prefix("xfN_") {
                    if let Some(field_offset) = xf_field_index(field) {
                        for xf in 0..num_transforms {
                            let idx = xf * PARAMS_PER_XF + field_offset;
                            if idx < result.len() {
                                let r = transform_randomness(xf);
                                let m = transform_magnitude(xf);
                                result[idx] += weight * r * m * signal_val / divisor;
                            }
                        }
                    }
                }
                // Explicit xf0_, xf1_, etc. (no randomness — you targeted it specifically)
                else if let Some((xf, field_offset)) = try_parse_xf(name) {
                    if xf < num_transforms {
                        let idx = xf * PARAMS_PER_XF + field_offset;
                        if idx < result.len() {
                            result[idx] += weight * signal_val / divisor;
                        }
                    }
                }
            }
        }
        result
    }

    /// Compute mutation rate from weighted signals.
    pub fn mutation_rate(&self, features: &crate::audio::AudioFeatures, time_signals: &TimeSignals) -> f32 {
        let signals = self.signal_list(features, time_signals);
        let mut rate = 0.0;
        for (weights, signal_val, divisor) in &signals {
            if let Some(&w) = weights.get("mutation_rate") {
                rate += w * signal_val / divisor;
            }
        }
        rate
    }
}

/// Deterministic per-transform randomness (-1.0 to 1.0) seeded by transform index.
/// Each transform gets a fixed "personality" that scales (and potentially inverts)
/// how all weight signals affect it.
fn transform_randomness(xf: usize) -> f32 {
    let h = xf.wrapping_mul(2654435761);
    let frac = ((h & 0xFFFF) as f32) / 65535.0; // 0.0–1.0
    frac * 2.0 - 1.0 // -1.0–1.0
}

/// Deterministic per-transform magnitude (1.0 to 5.0) seeded by transform index.
/// Controls how strongly all weight signals affect this transform overall.
fn transform_magnitude(xf: usize) -> f32 {
    let h = xf.wrapping_mul(1597334677);
    let frac = ((h & 0xFFFF) as f32) / 65535.0; // 0.0–1.0
    1.0 + frac * 4.0 // 1.0–5.0
}

/// Map xfN_ field suffix to its offset within a transform block.
fn xf_field_index(field: &str) -> Option<usize> {
    XF_FIELDS.iter().position(|&f| f == field)
}

fn global_index(name: &str) -> Option<usize> {
    match name {
        "speed" => Some(0),
        "zoom" => Some(1),
        "trail" => Some(2),
        "flame_brightness" => Some(3),
        "kifs_fold" => Some(4),
        "kifs_scale" => Some(5),
        "kifs_brightness" => Some(6),
        "drift_speed" => Some(7),
        "color_shift" => Some(8),
        "vibrancy" => Some(9),
        "bloom_intensity" => Some(10),
        _ => None,
    }
}

/// Parse explicit `xf0_weight`, `xf3_angle`, etc. into (transform_index, field_offset).
fn try_parse_xf(name: &str) -> Option<(usize, usize)> {
    let rest = name.strip_prefix("xf")?;
    let (digit, field) = rest.split_once('_')?;
    let xf: usize = digit.parse().ok()?;
    let field_offset = xf_field_index(field)?;
    Some((xf, field_offset))
}
