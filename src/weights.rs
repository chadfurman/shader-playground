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
    pub time: f32,          // raw elapsed time, steady linear ramp
    pub time_slow: f32,     // noise at 0.05 speed, ~20s wander
    pub time_med: f32,      // noise at 0.2 speed, ~5s wander
    pub time_fast: f32,     // noise at 0.8 speed, ~1.25s wander
    pub time_noise: f32,    // noise at 0.3 speed, organic wandering
    pub time_drift: f32,    // noise at 0.02 speed, ~50s glacial drift
    pub time_flutter: f32,  // noise at 1.5 speed, quick flicker
    pub time_walk: f32,     // random walk — accumulated noise, never reverses
    pub time_envelope: f32, // time since last mutation, capped at 1.0
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

const AUDIO_SIGNAL_COUNT: f32 = 7.0; // bass, mids, highs, energy, beat, beat_accum, change
const TIME_SIGNAL_COUNT: f32 = 9.0; // time, time_slow, time_med, time_fast, time_noise, time_drift, time_flutter, time_walk, time_envelope
pub const PARAMS_PER_XF: usize = 50;

/// Per-transform field names in order (matching genome flatten layout).
/// Layout: weight, m00..m22 (9 affine), offset_x/y/z (3), color, 26 variations, 8 var params
const XF_FIELDS: [&str; PARAMS_PER_XF] = [
    "weight",       // 0
    "m00",          // 1
    "m01",          // 2
    "m02",          // 3
    "m10",          // 4
    "m11",          // 5
    "m12",          // 6
    "m20",          // 7
    "m21",          // 8
    "m22",          // 9
    "offset_x",     // 10
    "offset_y",     // 11
    "offset_z",     // 12
    "color",        // 13
    "linear",       // 14
    "sinusoidal",   // 15
    "spherical",    // 16
    "swirl",        // 17
    "horseshoe",    // 18
    "handkerchief", // 19
    "julia",        // 20
    "polar",        // 21
    "disc",         // 22
    "rings",        // 23
    "bubble",       // 24
    "fisheye",      // 25
    "exponential",  // 26
    "spiral",       // 27
    "diamond",      // 28
    "bent",         // 29
    "waves",        // 30
    "popcorn",      // 31
    "fan",          // 32
    "eyefish",      // 33
    "cross",        // 34
    "tangent",      // 35
    "cosine",       // 36
    "blob",         // 37
    "noise",        // 38
    "curl",         // 39
    "rings2_val",   // 40
    "blob_low",     // 41
    "blob_high",    // 42
    "blob_waves",   // 43
    "julian_power", // 44
    "julian_dist",  // 45
    "ngon_sides",   // 46
    "ngon_corners", // 47
    "spin_mod",     // 48
    "drift_mod",    // 49
];

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct RuntimeConfig {
    #[serde(default = "default_morph_duration")]
    pub morph_duration: f32,
    #[serde(default = "default_mutation_cooldown")]
    pub mutation_cooldown: f32,
    #[serde(default = "default_workgroups")]
    pub workgroups: u32,
    #[serde(default = "default_magnitude_min")]
    pub magnitude_min: f32,
    #[serde(default = "default_magnitude_max")]
    pub magnitude_max: f32,
    #[serde(default = "default_randomness_range")]
    pub randomness_range: f32,
    #[serde(default = "default_vibrancy")]
    pub vibrancy: f32,
    #[serde(default = "default_bloom_intensity")]
    pub bloom_intensity: f32,
    #[serde(default = "default_noise_displacement")]
    pub noise_displacement: f32,
    #[serde(default = "default_curl_displacement")]
    pub curl_displacement: f32,
    #[serde(default = "default_tangent_clamp")]
    pub tangent_clamp: f32,
    #[serde(default = "default_color_blend")]
    pub color_blend: f32,
    #[serde(default = "default_spin_speed_max")]
    pub spin_speed_max: f32,
    #[serde(default = "default_position_drift")]
    pub position_drift: f32,
    #[serde(default = "default_warmup_iters")]
    pub warmup_iters: f32,
    #[serde(default = "default_gamma")]
    pub gamma: f32,
    #[serde(default = "default_highlight_power")]
    pub highlight_power: f32,
    #[serde(default = "default_zoom_min")]
    pub zoom_min: f32,
    #[serde(default = "default_zoom_max")]
    pub zoom_max: f32,
    #[serde(default = "default_zoom_target")]
    pub zoom_target: f32,
    #[serde(default = "default_min_attractor_extent")]
    pub min_attractor_extent: f32,
    #[serde(default = "default_max_mutation_retries")]
    pub max_mutation_retries: u32,
    #[serde(default = "default_trail")]
    pub trail: f32,
    #[serde(default = "default_accumulation_decay")]
    pub accumulation_decay: f32,
    #[serde(default = "default_accumulation_cap")]
    pub accumulation_cap: f32,
    #[serde(default = "default_morph_burst_decay")]
    pub morph_burst_decay: f32,
    #[serde(default = "default_jacobian_weight_strength")]
    pub jacobian_weight_strength: f32,
    #[serde(default = "default_min_genome_fps")]
    pub min_genome_fps: f32,
    #[serde(default = "default_perf_good_fps")]
    pub perf_good_fps: f32,
    #[serde(default = "default_perf_weight")]
    pub perf_weight: f32,
    #[serde(default = "default_samples_per_frame")]
    pub samples_per_frame: u32,
    #[serde(default = "default_bloom_radius")]
    pub bloom_radius: f32,
    #[serde(default = "default_seed_mutation_bias")]
    pub seed_mutation_bias: f32,
    #[serde(default = "default_fitness_bias_strength")]
    pub fitness_bias_strength: f32,
    #[serde(default = "default_drift_speed")]
    pub drift_speed: f32,
    #[serde(default = "default_velocity_blur_max")]
    pub velocity_blur_max: f32,
    #[serde(default = "default_iterations_per_thread")]
    pub iterations_per_thread: u32,
    #[serde(default = "default_morph_speed")]
    pub morph_speed: f32,
    #[serde(default = "default_morph_stagger_count")]
    pub morph_stagger_count: u32,
    #[serde(default = "default_morph_stagger_min")]
    pub morph_stagger_min: f32,
    #[serde(default = "default_morph_stagger_max")]
    pub morph_stagger_max: f32,
    #[serde(default)]
    pub jitter_amount: f32,
    #[serde(default)]
    pub tonemap_mode: u32,
    #[serde(default)]
    pub histogram_equalization: f32,
    #[serde(default)]
    pub dof_strength: f32,
    #[serde(default)]
    pub dof_focal_distance: f32,
    #[serde(default)]
    pub spectral_rendering: bool,
    #[serde(default)]
    pub temporal_reprojection: f32,
    #[serde(default)]
    pub variation_scales: HashMap<String, f32>,
    #[serde(default = "default_parent_current_bias")]
    pub parent_current_bias: f32,
    #[serde(default = "default_parent_voted_bias")]
    pub parent_voted_bias: f32,
    #[serde(default = "default_parent_saved_bias")]
    pub parent_saved_bias: f32,
    #[serde(default = "default_parent_random_bias")]
    pub parent_random_bias: f32,
    #[serde(default = "default_vote_blacklist_threshold")]
    pub vote_blacklist_threshold: i32,
    #[serde(default = "default_min_breeding_distance")]
    pub min_breeding_distance: u32,
    #[serde(default = "default_max_lineage_depth")]
    pub max_lineage_depth: u32,
    #[serde(default)]
    pub taste_engine_enabled: bool,
    #[serde(default = "default_taste_min_votes")]
    pub taste_min_votes: u32,
    #[serde(default = "default_taste_strength")]
    pub taste_strength: f32,
    #[serde(default = "default_taste_exploration_rate")]
    pub taste_exploration_rate: f32,
    #[serde(default = "default_taste_diversity_penalty")]
    pub taste_diversity_penalty: f32,
    #[serde(default = "default_taste_candidates")]
    pub taste_candidates: u32,
    #[serde(default = "default_taste_recent_memory")]
    pub taste_recent_memory: usize,
    #[serde(default = "default_archive_threshold_mb")]
    pub archive_threshold_mb: u64,
    #[serde(default = "default_archive_on_startup")]
    pub archive_on_startup: bool,

    // Novelty search
    #[serde(default = "default_novelty_weight")]
    pub novelty_weight: f32,
    #[serde(default = "default_novelty_k_neighbors")]
    pub novelty_k_neighbors: u32,

    // Interpolative crossover
    #[serde(default = "default_interpolation_range_lo")]
    pub interpolation_range_lo: f32,
    #[serde(default = "default_interpolation_range_hi")]
    pub interpolation_range_hi: f32,

    // Taste engine
    #[serde(default = "default_igmm_activation_threshold")]
    pub igmm_activation_threshold: f32,
    #[serde(default = "default_igmm_decay_rate")]
    pub igmm_decay_rate: f32,
    #[serde(default = "default_igmm_min_weight")]
    pub igmm_min_weight: f32,
    #[serde(default = "default_igmm_max_clusters")]
    pub igmm_max_clusters: u32,
    #[serde(default = "default_igmm_learning_rate")]
    pub igmm_learning_rate: f32,

    // Proxy render (CPU chaos game for perceptual features)
    #[serde(default = "default_proxy_render_grid_size")]
    pub proxy_render_grid_size: u32,
    #[serde(default = "default_proxy_render_iterations")]
    pub proxy_render_iterations: u32,
    #[serde(default = "default_proxy_render_warmup")]
    pub proxy_render_warmup: u32,
    #[serde(default = "default_spatial_entropy_blocks")]
    pub spatial_entropy_blocks: u32,
    #[serde(default)]
    pub dist_lum_strength: f32,
    #[serde(default = "default_iter_lum_range")]
    pub iter_lum_range: f32,
    // 3D rendering
    #[serde(default)]
    pub camera_pitch: f32,
    #[serde(default = "default_camera_yaw")]
    pub camera_yaw: f32,
    #[serde(default = "default_camera_focal")]
    pub camera_focal: f32,
    #[serde(default = "default_z_mutation_rate")]
    pub z_mutation_rate: f32,
    #[serde(default = "default_window_width")]
    pub window_width: u32,
    #[serde(default = "default_window_height")]
    pub window_height: u32,
    // Transform count range for breeding
    #[serde(default = "default_transform_count_min")]
    pub transform_count_min: u32,
    #[serde(default = "default_transform_count_max")]
    pub transform_count_max: u32,
    #[serde(default = "default_spin_mod_max")]
    pub spin_mod_max: f32,
    #[serde(default = "default_drift_mod_max")]
    pub drift_mod_max: f32,
}

fn default_morph_duration() -> f32 {
    8.0
}
fn default_mutation_cooldown() -> f32 {
    3.0
}
fn default_workgroups() -> u32 {
    512
}
fn default_magnitude_min() -> f32 {
    1.0
}
fn default_magnitude_max() -> f32 {
    5.0
}
fn default_randomness_range() -> f32 {
    1.0
}
fn default_vibrancy() -> f32 {
    0.7
}
fn default_bloom_intensity() -> f32 {
    0.05
}
fn default_noise_displacement() -> f32 {
    0.08
}
fn default_curl_displacement() -> f32 {
    0.05
}
fn default_tangent_clamp() -> f32 {
    4.0
}
fn default_color_blend() -> f32 {
    0.4
} // lower = more color mixing between transforms
fn default_spin_speed_max() -> f32 {
    0.15
}
fn default_position_drift() -> f32 {
    0.08
}
fn default_warmup_iters() -> f32 {
    20.0
}
fn default_gamma() -> f32 {
    0.4545
} // 1/2.2 — standard display gamma
fn default_highlight_power() -> f32 {
    2.0
} // hot-spot glow intensity
fn default_zoom_min() -> f32 {
    1.0
}
fn default_zoom_max() -> f32 {
    5.0
}
fn default_zoom_target() -> f32 {
    3.0
}
fn default_min_attractor_extent() -> f32 {
    0.3
}
fn default_max_mutation_retries() -> u32 {
    5
}
fn default_trail() -> f32 {
    0.15
} // temporal AA only — accumulation is primary persistence
fn default_accumulation_decay() -> f32 {
    0.9
}
fn default_accumulation_cap() -> f32 {
    10_000_000.0
}
fn default_morph_burst_decay() -> f32 {
    0.85 // gentler than the old 0.7 — lets the old image persist during transition
}
fn default_jacobian_weight_strength() -> f32 {
    0.0
}
fn default_min_genome_fps() -> f32 {
    15.0
}
fn default_perf_good_fps() -> f32 {
    30.0
}
fn default_perf_weight() -> f32 {
    0.2
}
fn default_samples_per_frame() -> u32 {
    256
}
fn default_bloom_radius() -> f32 {
    3.0
}
fn default_seed_mutation_bias() -> f32 {
    0.7
}
fn default_fitness_bias_strength() -> f32 {
    0.5
}
fn default_drift_speed() -> f32 {
    0.5
}
fn default_velocity_blur_max() -> f32 {
    24.0
} // max directional blur length in pixels
fn default_iterations_per_thread() -> u32 {
    200
} // chaos game iterations per GPU thread per frame
fn default_morph_speed() -> f32 {
    1.0
} // global morph speed multiplier (affects all transforms)
fn default_morph_stagger_count() -> u32 {
    2
} // max transforms to randomize (0 = all same speed)
fn default_morph_stagger_min() -> f32 {
    0.3
} // slowest random morph rate
fn default_morph_stagger_max() -> f32 {
    0.6
} // fastest random morph rate
fn default_parent_current_bias() -> f32 {
    0.30
}
fn default_parent_voted_bias() -> f32 {
    0.25
}
fn default_parent_saved_bias() -> f32 {
    0.25
}
fn default_parent_random_bias() -> f32 {
    0.20
}
fn default_vote_blacklist_threshold() -> i32 {
    -2
}
fn default_min_breeding_distance() -> u32 {
    3
}
fn default_max_lineage_depth() -> u32 {
    8
}
fn default_taste_min_votes() -> u32 {
    10
}
fn default_taste_strength() -> f32 {
    0.5
}
fn default_taste_exploration_rate() -> f32 {
    0.1
}
fn default_taste_diversity_penalty() -> f32 {
    0.3
}
fn default_taste_candidates() -> u32 {
    20
}
fn default_taste_recent_memory() -> usize {
    5
}
fn default_archive_threshold_mb() -> u64 {
    100
}
fn default_archive_on_startup() -> bool {
    true
}
fn default_novelty_weight() -> f32 {
    0.3
}
fn default_novelty_k_neighbors() -> u32 {
    5
}
fn default_interpolation_range_lo() -> f32 {
    0.3
}
fn default_interpolation_range_hi() -> f32 {
    0.7
}
fn default_igmm_activation_threshold() -> f32 {
    2.0
}
fn default_igmm_decay_rate() -> f32 {
    0.95
}
fn default_igmm_min_weight() -> f32 {
    0.1
}
fn default_igmm_max_clusters() -> u32 {
    8
}
fn default_igmm_learning_rate() -> f32 {
    0.1
}
fn default_proxy_render_grid_size() -> u32 {
    64
}
fn default_proxy_render_iterations() -> u32 {
    500
}
fn default_proxy_render_warmup() -> u32 {
    50
}
fn default_spatial_entropy_blocks() -> u32 {
    8
}
fn default_iter_lum_range() -> f32 {
    0.5
} // 0.0 = uniform brightness, 0.5 = early iters 2x brighter than late
fn default_z_mutation_rate() -> f32 {
    0.05
}
fn default_camera_yaw() -> f32 {
    0.0
}
fn default_camera_focal() -> f32 {
    2.0
}
fn default_window_width() -> u32 {
    640
}
fn default_window_height() -> u32 {
    480
}
fn default_transform_count_min() -> u32 {
    3
}
fn default_transform_count_max() -> u32 {
    6
}
fn default_spin_mod_max() -> f32 {
    4.0
}
fn default_drift_mod_max() -> f32 {
    4.0
}

const VARIATION_START: usize = 14; // first variation field index in each 50-float transform block
pub const SPIN_MOD_FIELD: usize = 48; // per-transform spin modulation multiplier
pub const DRIFT_MOD_FIELD: usize = 49; // per-transform drift modulation multiplier

impl RuntimeConfig {
    /// Apply variation_scales to a flattened transform buffer.
    /// Multiplies each variation weight by the corresponding scale (default 1.0).
    pub fn apply_variation_scales(&self, xf_buf: &mut [f32], num_transforms: usize) {
        if self.variation_scales.is_empty() {
            return;
        }
        for xf in 0..num_transforms {
            for (fi, &name) in XF_FIELDS[VARIATION_START..PARAMS_PER_XF].iter().enumerate() {
                let field_idx = VARIATION_START + fi;
                if let Some(&scale) = self.variation_scales.get(name) {
                    let buf_idx = xf * PARAMS_PER_XF + field_idx;
                    if buf_idx < xf_buf.len() {
                        xf_buf[buf_idx] *= scale;
                    }
                }
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Weights {
    #[serde(default)]
    pub _config: RuntimeConfig,
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
    pub change: HashMap<String, f32>,
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
        let json = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
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
            (&self.change, features.change, AUDIO_SIGNAL_COUNT),
            (&self.time, time_signals.time, TIME_SIGNAL_COUNT),
            (&self.time_slow, time_signals.time_slow, TIME_SIGNAL_COUNT),
            (&self.time_med, time_signals.time_med, TIME_SIGNAL_COUNT),
            (&self.time_fast, time_signals.time_fast, TIME_SIGNAL_COUNT),
            (&self.time_noise, time_signals.time_noise, TIME_SIGNAL_COUNT),
            (&self.time_drift, time_signals.time_drift, TIME_SIGNAL_COUNT),
            (
                &self.time_flutter,
                time_signals.time_flutter,
                TIME_SIGNAL_COUNT,
            ),
            (&self.time_walk, time_signals.time_walk, TIME_SIGNAL_COUNT),
            (
                &self.time_envelope,
                time_signals.time_envelope,
                TIME_SIGNAL_COUNT,
            ),
        ]
    }

    /// Compute the aggregate mutation_rate contribution from all signals this frame.
    /// Returns a per-frame delta — caller accumulates and triggers mutation at >= 1.0.
    pub fn compute_mutation_rate(
        &self,
        features: &crate::audio::AudioFeatures,
        time_signals: &TimeSignals,
    ) -> f32 {
        let mut rate = 0.0;
        for (weights, signal_val, divisor) in self.signal_list(features, time_signals) {
            if let Some(&w) = weights.get("mutation_rate") {
                rate += w * signal_val / divisor;
            }
        }
        rate
    }

    /// Apply weights to global params only.
    pub fn apply_globals(
        &self,
        base: &[f32; 20],
        features: &crate::audio::AudioFeatures,
        time_signals: &TimeSignals,
    ) -> [f32; 20] {
        let mut result = *base;
        let signals = self.signal_list(features, time_signals);
        for (weights, signal_val, divisor) in signals.iter() {
            for (name, &weight) in *weights {
                if name == "mutation_rate" {
                    continue;
                }
                if name.starts_with("xf") {
                    continue;
                } // transform params handled separately
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
        for (weights, signal_val, divisor) in signals.iter() {
            for (name, &weight) in *weights {
                if name == "mutation_rate" {
                    continue;
                }
                // xfN_ wildcard
                if let Some(field) = name.strip_prefix("xfN_") {
                    if let Some(field_offset) = xf_field_index(field) {
                        for xf in 0..num_transforms {
                            let idx = xf * PARAMS_PER_XF + field_offset;
                            if idx < result.len() {
                                let r = transform_randomness(xf, self._config.randomness_range);
                                let m = transform_magnitude(
                                    xf,
                                    self._config.magnitude_min,
                                    self._config.magnitude_max,
                                );
                                result[idx] += weight * r * m * signal_val / divisor;
                            }
                        }
                    }
                }
                // Explicit xf0_, xf1_, etc. (no randomness — you targeted it specifically)
                else if let Some((xf, field_offset)) = try_parse_xf(name)
                    && xf < num_transforms
                {
                    let idx = xf * PARAMS_PER_XF + field_offset;
                    if idx < result.len() {
                        result[idx] += weight * signal_val / divisor;
                    }
                }
            }
        }
        result
    }
}

/// Deterministic per-transform randomness seeded by transform index.
/// Range: -randomness_range to +randomness_range.
fn transform_randomness(xf: usize, range: f32) -> f32 {
    let h = xf.wrapping_mul(2654435761);
    let frac = ((h & 0xFFFF) as f32) / 65535.0; // 0.0–1.0
    (frac * 2.0 - 1.0) * range
}

/// Deterministic per-transform magnitude seeded by transform index.
/// Range: magnitude_min to magnitude_max.
fn transform_magnitude(xf: usize, min: f32, max: f32) -> f32 {
    let h = xf.wrapping_mul(1597334677);
    let frac = ((h & 0xFFFF) as f32) / 65535.0; // 0.0–1.0
    min + frac * (max - min)
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
        "noise_displacement" => Some(12),
        "curl_displacement" => Some(13),
        "tangent_clamp" => Some(14),
        "color_blend" => Some(15),
        "spin_speed_max" => Some(16),
        "position_drift" => Some(17),
        "warmup_iters" => Some(18),
        "gamma" => Some(11),
        "velocity_blur_max" => Some(19),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_sane_values() {
        let cfg: RuntimeConfig = serde_json::from_str("{}").unwrap();
        assert!(cfg.morph_duration > 0.0);
        assert!(cfg.mutation_cooldown > 0.0);
        assert!(cfg.workgroups > 0);
        assert!(cfg.zoom_min > 0.0);
        assert!(cfg.zoom_max > cfg.zoom_min);
        assert!(cfg.max_mutation_retries > 0);
        assert!(cfg.min_breeding_distance > 0);
        assert!(cfg.max_lineage_depth > 0);
        assert!(cfg.taste_min_votes > 0);
        assert!(cfg.taste_candidates > 0);
        assert!(cfg.taste_recent_memory > 0);
    }

    #[test]
    fn config_serialization_roundtrip() {
        let cfg: RuntimeConfig = serde_json::from_str("{}").unwrap();
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: RuntimeConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.morph_duration, cfg2.morph_duration);
        assert_eq!(cfg.workgroups, cfg2.workgroups);
        assert_eq!(cfg.min_breeding_distance, cfg2.min_breeding_distance);
        assert_eq!(cfg.taste_min_votes, cfg2.taste_min_votes);
    }

    #[test]
    fn config_partial_json_uses_defaults() {
        let json = r#"{"morph_duration": 5.0}"#;
        let cfg: RuntimeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.morph_duration, 5.0);
        assert_eq!(cfg.workgroups, 512);
        assert_eq!(cfg.min_breeding_distance, 3);
        assert_eq!(cfg.taste_engine_enabled, false);
    }

    #[test]
    fn config_empty_json_uses_all_defaults() {
        let cfg: RuntimeConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.morph_duration, 8.0);
        assert_eq!(cfg.gamma, 0.4545);
        assert_eq!(cfg.vote_blacklist_threshold, -2);
    }

    #[test]
    fn weights_empty_json_is_valid() {
        let json = r#"{"_config": {}}"#;
        let weights: Weights = serde_json::from_str(json).unwrap();
        assert!(weights.bass.is_empty());
        assert_eq!(weights._config.morph_duration, 8.0);
    }

    #[test]
    fn global_index_known_params() {
        assert_eq!(global_index("speed"), Some(0));
        assert_eq!(global_index("zoom"), Some(1));
        assert_eq!(global_index("gamma"), Some(11));
        assert_eq!(global_index("nonexistent"), None);
    }

    #[test]
    fn xf_field_index_known_fields() {
        assert_eq!(xf_field_index("weight"), Some(0));
        assert_eq!(xf_field_index("linear"), Some(14));
        assert_eq!(xf_field_index("spherical"), Some(16));
        assert_eq!(xf_field_index("fake_field"), None);
    }

    #[test]
    fn try_parse_xf_valid() {
        assert_eq!(try_parse_xf("xf0_weight"), Some((0, 0)));
        assert_eq!(try_parse_xf("xf3_linear"), Some((3, 14)));
        assert_eq!(try_parse_xf("xf10_spherical"), Some((10, 16)));
    }

    #[test]
    fn try_parse_xf_invalid() {
        assert_eq!(try_parse_xf("speed"), None);
        assert_eq!(try_parse_xf("xfN_weight"), None);
        assert_eq!(try_parse_xf("xf_weight"), None);
    }

    #[test]
    fn archive_config_defaults() {
        let cfg: RuntimeConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.archive_threshold_mb, 100);
        assert!(cfg.archive_on_startup);
    }
}
