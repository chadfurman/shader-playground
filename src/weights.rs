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
    pub time_slow: f32,     // sin(t * 0.1), ~60s cycle
    pub time_med: f32,      // sin(t * 0.5), ~12s cycle
    pub time_fast: f32,     // sin(t * 2.0), ~3s cycle
    pub time_noise: f32,    // smooth noise seeded by time
    pub time_envelope: f32, // time since last mutation, capped at 1.0
}

impl TimeSignals {
    pub fn compute(time: f32, time_since_mutation: f32) -> Self {
        Self {
            time_slow: (time * 0.1).sin(),
            time_med: (time * 0.5).sin(),
            time_fast: (time * 2.0).sin(),
            time_noise: value_noise(time * 0.3),
            time_envelope: (time_since_mutation / 10.0).min(1.0),
        }
    }
}

const SIGNAL_COUNT: f32 = 11.0; // bass, mids, highs, energy, beat, beat_accum, time_slow, time_med, time_fast, time_noise, time_envelope
const PARAMS_PER_XF: usize = 12;

/// Per-transform field names in order (matching genome flatten layout).
const XF_FIELDS: [&str; PARAMS_PER_XF] = [
    "weight", "angle", "scale", "offset_x", "offset_y", "color",
    "linear", "sinusoidal", "spherical", "swirl", "horseshoe", "handkerchief",
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
    pub time_slow: HashMap<String, f32>,
    #[serde(default)]
    pub time_med: HashMap<String, f32>,
    #[serde(default)]
    pub time_fast: HashMap<String, f32>,
    #[serde(default)]
    pub time_noise: HashMap<String, f32>,
    #[serde(default)]
    pub time_envelope: HashMap<String, f32>,
}

impl Weights {
    pub fn load(path: &Path) -> Result<Self, String> {
        let json =
            fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_str(&json).map_err(|e| format!("parse {}: {e}", path.display()))
    }

    /// Build the list of (weight-map, signal-value) pairs used by all apply methods.
    fn signal_list<'a>(
        &'a self,
        features: &crate::audio::AudioFeatures,
        time_signals: &TimeSignals,
    ) -> Vec<(&'a HashMap<String, f32>, f32)> {
        vec![
            (&self.bass, features.bass),
            (&self.mids, features.mids),
            (&self.highs, features.highs),
            (&self.energy, features.energy),
            (&self.beat, features.beat),
            (&self.beat_accum, features.beat_accum),
            (&self.time_slow, time_signals.time_slow),
            (&self.time_med, time_signals.time_med),
            (&self.time_fast, time_signals.time_fast),
            (&self.time_noise, time_signals.time_noise),
            (&self.time_envelope, time_signals.time_envelope),
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
        for (_signal_idx, (weights, signal_val)) in signals.iter().enumerate() {
            for (name, &weight) in *weights {
                if name == "mutation_rate" { continue; }
                if name.starts_with("xf") { continue; } // transform params handled separately
                if let Some(idx) = global_index(name) {
                    result[idx] += weight * signal_val / SIGNAL_COUNT;
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
        for (signal_idx, (weights, signal_val)) in signals.iter().enumerate() {
            for (name, &weight) in *weights {
                if name == "mutation_rate" { continue; }
                // xfN_ wildcard
                if let Some(field) = name.strip_prefix("xfN_") {
                    if let Some(field_offset) = xf_field_index(field) {
                        for xf in 0..num_transforms {
                            let idx = xf * PARAMS_PER_XF + field_offset;
                            if idx < result.len() {
                                let scale = per_transform_scale(xf, field_offset, signal_idx);
                                result[idx] += weight * scale * signal_val / SIGNAL_COUNT;
                            }
                        }
                    }
                }
                // Explicit xf0_, xf1_, etc.
                else if let Some((xf, field_offset)) = try_parse_xf(name) {
                    if xf < num_transforms {
                        let idx = xf * PARAMS_PER_XF + field_offset;
                        if idx < result.len() {
                            result[idx] += weight * signal_val / SIGNAL_COUNT;
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
        for (weights, signal_val) in &signals {
            if let Some(&w) = weights.get("mutation_rate") {
                rate += w * signal_val;
            }
        }
        rate / SIGNAL_COUNT
    }
}

/// Deterministic per-transform scale factor (0.5–1.5) so the same weight
/// produces varied expression across flames.
fn per_transform_scale(xf: usize, field: usize, signal: usize) -> f32 {
    // Simple hash: mix transform, field, and signal indices
    let h = (xf.wrapping_mul(2654435761)) ^ (field.wrapping_mul(40503)) ^ (signal.wrapping_mul(73));
    let frac = ((h & 0xFFFF) as f32) / 65535.0; // 0.0–1.0
    0.5 + frac // 0.5–1.5
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
