use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

const SIGNAL_COUNT: f32 = 7.0; // bass, mids, highs, energy, beat, beat_accum, time
const NUM_TRANSFORMS: usize = 4;
const PARAMS_PER_XF: usize = 12;
const XF_BASE: usize = 8;

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
    pub time: HashMap<String, f32>,
}

impl Weights {
    pub fn load(path: &Path) -> Result<Self, String> {
        let json =
            fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_str(&json).map_err(|e| format!("parse {}: {e}", path.display()))
    }

    /// Apply weight matrix: target[i] = base[i] + (sum of signal_weight * signal_value) / signal_count
    ///
    /// Supports `xfN_` wildcard keys that expand to all transforms with per-transform
    /// deterministic randomization (scale factor 0.5x–1.5x based on transform + field hash).
    pub fn apply(
        &self,
        base: &[f32; 64],
        features: &crate::audio::AudioFeatures,
        time: f32,
    ) -> [f32; 64] {
        let mut result = *base;
        let signals: &[(&HashMap<String, f32>, f32)] = &[
            (&self.bass, features.bass),
            (&self.mids, features.mids),
            (&self.highs, features.highs),
            (&self.energy, features.energy),
            (&self.beat, features.beat),
            (&self.beat_accum, features.beat_accum),
            (&self.time, time),
        ];
        for (signal_idx, (weights, signal_val)) in signals.iter().enumerate() {
            for (name, &weight) in *weights {
                if name == "mutation_rate" {
                    continue;
                }
                // xfN_ wildcard: expand to all transforms with per-transform randomization
                if let Some(field) = name.strip_prefix("xfN_") {
                    if let Some(field_offset) = xf_field_index(field) {
                        for xf in 0..NUM_TRANSFORMS {
                            let idx = XF_BASE + xf * PARAMS_PER_XF + field_offset;
                            let scale = per_transform_scale(xf, field_offset, signal_idx);
                            result[idx] += weight * scale * signal_val / SIGNAL_COUNT;
                        }
                    }
                } else if let Some(idx) = param_index(name) {
                    result[idx] += weight * signal_val / SIGNAL_COUNT;
                }
            }
        }
        result
    }

    /// Compute mutation rate from weighted signals.
    pub fn mutation_rate(&self, features: &crate::audio::AudioFeatures, time: f32) -> f32 {
        let signals: &[(&HashMap<String, f32>, f32)] = &[
            (&self.bass, features.bass),
            (&self.mids, features.mids),
            (&self.highs, features.highs),
            (&self.energy, features.energy),
            (&self.beat, features.beat),
            (&self.beat_accum, features.beat_accum),
            (&self.time, time),
        ];
        let mut rate = 0.0;
        for (weights, signal_val) in signals {
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

fn param_index(name: &str) -> Option<usize> {
    match name {
        // Global
        "speed" => Some(0),
        "zoom" => Some(1),
        "trail" => Some(2),
        "flame_brightness" => Some(3),
        // KIFS
        "kifs_fold" => Some(4),
        "kifs_scale" => Some(5),
        "kifs_brightness" => Some(6),
        // Misc
        "drift_speed" => Some(7),
        // Extra
        "color_shift" => Some(56),
        // Explicit per-transform (still supported alongside xfN_)
        _ => try_explicit_xf(name),
    }
}

/// Parse explicit `xf0_weight`, `xf3_angle`, etc.
fn try_explicit_xf(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("xf")?;
    let (digit, field) = rest.split_once('_')?;
    let xf: usize = digit.parse().ok()?;
    if xf >= NUM_TRANSFORMS {
        return None;
    }
    let field_offset = xf_field_index(field)?;
    Some(XF_BASE + xf * PARAMS_PER_XF + field_offset)
}
