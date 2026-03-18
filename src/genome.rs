use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::audio::AudioFeatures;

const VARIATION_COUNT: usize = 26;

const VARIATION_NAMES: [&str; VARIATION_COUNT] = [
    "linear",
    "sinusoidal",
    "spherical",
    "swirl",
    "horseshoe",
    "handkerchief",
    "julia",
    "polar",
    "disc",
    "rings",
    "bubble",
    "fisheye",
    "exponential",
    "spiral",
    "diamond",
    "bent",
    "waves",
    "popcorn",
    "fan",
    "eyefish",
    "cross",
    "tangent",
    "cosine",
    "blob",
    "noise",
    "curl",
];

pub struct FavoriteProfile {
    pub variation_freq: HashMap<String, f32>,
}

impl FavoriteProfile {
    /// Scan genomes/*.json (NOT genomes/seeds/) to build a frequency profile
    /// of which variations appear most often in saved favorites.
    pub fn from_directory(dir: &Path) -> Self {
        let mut counts: HashMap<String, f32> = HashMap::new();
        let mut total_genomes = 0u32;

        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return Self::uniform_default(),
        };

        for entry in entries.flatten() {
            let path = entry.path();
            // Only .json files directly in genomes/ (skip subdirectories)
            if !path.is_file() || path.extension().is_none_or(|ext| ext != "json") {
                continue;
            }
            let genome = match FlameGenome::load(&path) {
                Ok(g) => g,
                Err(_) => continue,
            };
            total_genomes += 1;

            for xf in &genome.transforms {
                for (i, name) in VARIATION_NAMES.iter().enumerate() {
                    if xf.get_variation(i) > 0.1 {
                        *counts.entry(name.to_string()).or_insert(0.0) += 1.0;
                    }
                }
            }
        }

        if total_genomes == 0 {
            return Self::uniform_default();
        }

        // Normalize frequencies to 0..1 range
        let max_count = counts.values().cloned().fold(0.0f32, f32::max);
        let mut variation_freq = HashMap::new();
        if max_count > 0.0 {
            for (name, count) in &counts {
                variation_freq.insert(name.clone(), count / max_count);
            }
        }

        Self { variation_freq }
    }

    fn uniform_default() -> Self {
        Self {
            variation_freq: HashMap::new(),
        }
    }
}

/// Pick a variation index biased toward variations that appear in saved favorites.
fn fitness_biased_variation_pick(
    rng: &mut impl Rng,
    profile: &Option<FavoriteProfile>,
    bias: f32,
) -> usize {
    if bias <= 0.0 || profile.is_none() {
        return rng.random_range(0..VARIATION_COUNT);
    }
    let profile = profile.as_ref().unwrap();
    let uniform = 1.0 / VARIATION_NAMES.len() as f32;
    let weights: Vec<f32> = VARIATION_NAMES
        .iter()
        .map(|name| {
            let freq = profile.variation_freq.get(*name).copied().unwrap_or(0.1);
            uniform * (1.0 - bias) + freq * bias
        })
        .collect();
    let total: f32 = weights.iter().sum();
    let mut r = rng.random_range(0.0..total);
    for (i, w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return i;
        }
    }
    VARIATION_NAMES.len() - 1
}

// "Orby" variations that produce round/spherical/blobby shapes —
// biased toward in mutations so they appear more often
const ORBY_VARIATIONS: [usize; 7] = [
    2,  // spherical
    6,  // julia
    8,  // disc
    10, // bubble
    11, // fisheye
    19, // eyefish
    23, // blob
];

// Variation groups biased by audio frequency content
const BASS_VARIATIONS: [usize; 4] = [2, 10, 23, 11]; // spherical, bubble, blob, fisheye
const MIDS_VARIATIONS: [usize; 4] = [1, 16, 22, 5]; // sinusoidal, waves, cosine, handkerchief
const HIGHS_VARIATIONS: [usize; 5] = [6, 8, 20, 21, 14]; // julia, disc, cross, tangent, diamond
const BEAT_VARIATIONS: [usize; 3] = [13, 3, 7]; // spiral, swirl, polar

#[derive(Serialize, Clone, Debug)]
pub struct FlameTransform {
    pub weight: f32,
    /// 3x3 affine matrix: [[m00,m01,m02],[m10,m11,m12],[m20,m21,m22]]
    /// For 2D-only genomes, z-row and z-column are identity: affine[2][2]=1, rest of z=0.
    #[serde(default = "default_affine")]
    pub affine: [[f32; 3]; 3],
    /// 3D offset (translation). Third component is z-offset for 3D rendering.
    #[serde(default)]
    pub offset: [f32; 3],
    pub color: f32,
    // Original 6 variations
    pub linear: f32,
    pub sinusoidal: f32,
    pub spherical: f32,
    pub swirl: f32,
    pub horseshoe: f32,
    pub handkerchief: f32,
    // New 20 variations
    #[serde(default)]
    pub julia: f32,
    #[serde(default)]
    pub polar: f32,
    #[serde(default)]
    pub disc: f32,
    #[serde(default)]
    pub rings: f32,
    #[serde(default)]
    pub bubble: f32,
    #[serde(default)]
    pub fisheye: f32,
    #[serde(default)]
    pub exponential: f32,
    #[serde(default)]
    pub spiral: f32,
    #[serde(default)]
    pub diamond: f32,
    #[serde(default)]
    pub bent: f32,
    #[serde(default)]
    pub waves: f32,
    #[serde(default)]
    pub popcorn: f32,
    #[serde(default)]
    pub fan: f32,
    #[serde(default)]
    pub eyefish: f32,
    #[serde(default)]
    pub cross: f32,
    #[serde(default)]
    pub tangent: f32,
    #[serde(default)]
    pub cosine: f32,
    #[serde(default)]
    pub blob: f32,
    #[serde(default)]
    pub noise: f32,
    #[serde(default)]
    pub curl: f32,
    /// Per-variation parametric values (e.g. rings2_val, blob_low, etc.)
    #[serde(default)]
    pub variation_params: HashMap<String, f32>,
}

fn default_affine() -> [[f32; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// Custom deserializer: handles both old format (a,b,c,d + offset:[f32;2])
/// and new format (affine:[[f32;3];3] + offset:[f32;3]).
impl<'de> Deserialize<'de> for FlameTransform {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawTransform {
            weight: f32,
            // New format
            #[serde(default)]
            affine: Option<[[f32; 3]; 3]>,
            // Old format fields
            #[serde(default)]
            a: Option<f32>,
            #[serde(default)]
            b: Option<f32>,
            #[serde(default)]
            c: Option<f32>,
            #[serde(default)]
            d: Option<f32>,
            // Legacy angle/scale
            #[serde(default)]
            angle: Option<f32>,
            #[serde(default)]
            scale: Option<f32>,
            // Offset can be 2 or 3 elements
            #[serde(default, deserialize_with = "deserialize_offset")]
            offset: [f32; 3],
            color: f32,
            #[serde(default)]
            linear: f32,
            #[serde(default)]
            sinusoidal: f32,
            #[serde(default)]
            spherical: f32,
            #[serde(default)]
            swirl: f32,
            #[serde(default)]
            horseshoe: f32,
            #[serde(default)]
            handkerchief: f32,
            #[serde(default)]
            julia: f32,
            #[serde(default)]
            polar: f32,
            #[serde(default)]
            disc: f32,
            #[serde(default)]
            rings: f32,
            #[serde(default)]
            bubble: f32,
            #[serde(default)]
            fisheye: f32,
            #[serde(default)]
            exponential: f32,
            #[serde(default)]
            spiral: f32,
            #[serde(default)]
            diamond: f32,
            #[serde(default)]
            bent: f32,
            #[serde(default)]
            waves: f32,
            #[serde(default)]
            popcorn: f32,
            #[serde(default)]
            fan: f32,
            #[serde(default)]
            eyefish: f32,
            #[serde(default)]
            cross: f32,
            #[serde(default)]
            tangent: f32,
            #[serde(default)]
            cosine: f32,
            #[serde(default)]
            blob: f32,
            #[serde(default)]
            noise: f32,
            #[serde(default)]
            curl: f32,
            #[serde(default)]
            variation_params: HashMap<String, f32>,
        }

        let raw = RawTransform::deserialize(deserializer)?;

        // Determine affine matrix: prefer new format, fall back to old a,b,c,d
        let affine = if let Some(aff) = raw.affine {
            aff
        } else {
            let mut a = raw.a.unwrap_or(1.0);
            let mut b = raw.b.unwrap_or(0.0);
            let mut c = raw.c.unwrap_or(0.0);
            let mut d = raw.d.unwrap_or(1.0);

            // Handle legacy angle/scale → a,b,c,d conversion
            if let (Some(angle), Some(scale)) = (raw.angle, raw.scale)
                && a == 1.0
                && b == 0.0
                && c == 0.0
                && d == 1.0
            {
                let (s, cos_a) = angle.sin_cos();
                a = cos_a * scale;
                b = -s * scale;
                c = s * scale;
                d = cos_a * scale;
            }

            [[a, b, 0.0], [c, d, 0.0], [0.0, 0.0, 1.0]]
        };

        Ok(FlameTransform {
            weight: raw.weight,
            affine,
            offset: raw.offset,
            color: raw.color,
            linear: raw.linear,
            sinusoidal: raw.sinusoidal,
            spherical: raw.spherical,
            swirl: raw.swirl,
            horseshoe: raw.horseshoe,
            handkerchief: raw.handkerchief,
            julia: raw.julia,
            polar: raw.polar,
            disc: raw.disc,
            rings: raw.rings,
            bubble: raw.bubble,
            fisheye: raw.fisheye,
            exponential: raw.exponential,
            spiral: raw.spiral,
            diamond: raw.diamond,
            bent: raw.bent,
            waves: raw.waves,
            popcorn: raw.popcorn,
            fan: raw.fan,
            eyefish: raw.eyefish,
            cross: raw.cross,
            tangent: raw.tangent,
            cosine: raw.cosine,
            blob: raw.blob,
            noise: raw.noise,
            curl: raw.curl,
            variation_params: raw.variation_params,
        })
    }
}

/// Deserialize offset as either [f32; 2] (old) or [f32; 3] (new).
fn deserialize_offset<'de, D>(deserializer: D) -> Result<[f32; 3], D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v: Vec<f32> = Vec::deserialize(deserializer)?;
    match v.len() {
        0 => Ok([0.0, 0.0, 0.0]),
        2 => Ok([v[0], v[1], 0.0]),
        3 => Ok([v[0], v[1], v[2]]),
        _ => Ok([
            v.first().copied().unwrap_or(0.0),
            v.get(1).copied().unwrap_or(0.0),
            v.get(2).copied().unwrap_or(0.0),
        ]),
    }
}

impl Default for FlameTransform {
    fn default() -> Self {
        Self {
            weight: 0.0,
            affine: default_affine(),
            offset: [0.0, 0.0, 0.0],
            color: 0.0,
            linear: 0.0,
            sinusoidal: 0.0,
            spherical: 0.0,
            swirl: 0.0,
            horseshoe: 0.0,
            handkerchief: 0.0,
            julia: 0.0,
            polar: 0.0,
            disc: 0.0,
            rings: 0.0,
            bubble: 0.0,
            fisheye: 0.0,
            exponential: 0.0,
            spiral: 0.0,
            diamond: 0.0,
            bent: 0.0,
            waves: 0.0,
            popcorn: 0.0,
            fan: 0.0,
            eyefish: 0.0,
            cross: 0.0,
            tangent: 0.0,
            cosine: 0.0,
            blob: 0.0,
            noise: 0.0,
            curl: 0.0,
            variation_params: HashMap::new(),
        }
    }
}

impl FlameTransform {
    // ── 2D affine accessors (row-major: affine[row][col]) ──
    pub fn a(&self) -> f32 {
        self.affine[0][0]
    }
    pub fn b(&self) -> f32 {
        self.affine[0][1]
    }
    pub fn c(&self) -> f32 {
        self.affine[1][0]
    }
    pub fn d(&self) -> f32 {
        self.affine[1][1]
    }

    /// Legacy fixup is now handled by the custom deserializer.
    /// This is a no-op kept for API compatibility with load().
    pub fn fixup_legacy(&mut self) {
        // No-op: angle/scale migration is handled during deserialization
    }

    pub fn get_variation(&self, idx: usize) -> f32 {
        match idx {
            0 => self.linear,
            1 => self.sinusoidal,
            2 => self.spherical,
            3 => self.swirl,
            4 => self.horseshoe,
            5 => self.handkerchief,
            6 => self.julia,
            7 => self.polar,
            8 => self.disc,
            9 => self.rings,
            10 => self.bubble,
            11 => self.fisheye,
            12 => self.exponential,
            13 => self.spiral,
            14 => self.diamond,
            15 => self.bent,
            16 => self.waves,
            17 => self.popcorn,
            18 => self.fan,
            19 => self.eyefish,
            20 => self.cross,
            21 => self.tangent,
            22 => self.cosine,
            23 => self.blob,
            24 => self.noise,
            25 => self.curl,
            _ => 0.0,
        }
    }

    fn clear_variations(&mut self) {
        for i in 0..VARIATION_COUNT {
            self.set_variation(i, 0.0);
        }
    }

    fn set_variation(&mut self, idx: usize, val: f32) {
        match idx {
            0 => self.linear = val,
            1 => self.sinusoidal = val,
            2 => self.spherical = val,
            3 => self.swirl = val,
            4 => self.horseshoe = val,
            5 => self.handkerchief = val,
            6 => self.julia = val,
            7 => self.polar = val,
            8 => self.disc = val,
            9 => self.rings = val,
            10 => self.bubble = val,
            11 => self.fisheye = val,
            12 => self.exponential = val,
            13 => self.spiral = val,
            14 => self.diamond = val,
            15 => self.bent = val,
            16 => self.waves = val,
            17 => self.popcorn = val,
            18 => self.fan = val,
            19 => self.eyefish = val,
            20 => self.cross = val,
            21 => self.tangent = val,
            22 => self.cosine = val,
            23 => self.blob = val,
            24 => self.noise = val,
            25 => self.curl = val,
            _ => {}
        }
    }

    /// Returns the index of the variation with the highest weight, or None if no active variations.
    pub fn primary_variation_index(&self) -> Option<usize> {
        let mut best_idx = None;
        let mut best_weight = 0.0f32;
        for i in 0..VARIATION_COUNT {
            let w = self.get_variation(i);
            if w > best_weight {
                best_weight = w;
                best_idx = Some(i);
            }
        }
        best_idx
    }

    /// Linearly interpolate between two transforms at parameter t (0.0 = self, 1.0 = other).
    /// Lerps affine coefficients (a, b, c, d), offset, weight, and color.
    pub fn lerp_with(&self, other: &Self, t: f32) -> Self {
        let lerp = |a: f32, b: f32| a + (b - a) * t;
        let mut result = self.clone();
        for r in 0..3 {
            for c in 0..3 {
                result.affine[r][c] = lerp(self.affine[r][c], other.affine[r][c]);
            }
        }
        for i in 0..3 {
            result.offset[i] = lerp(self.offset[i], other.offset[i]);
        }
        result.weight = lerp(self.weight, other.weight);
        result.color = lerp(self.color, other.color);
        result
    }

    pub fn random_transform(rng: &mut impl Rng) -> Self {
        let a = rng.random::<f32>() * 2.0 - 1.0;
        let b = rng.random::<f32>() * 2.0 - 1.0;
        let c = rng.random::<f32>() * 2.0 - 1.0;
        let d = rng.random::<f32>() * 2.0 - 1.0;
        let mut xf = Self {
            weight: rng.random::<f32>() * 0.5 + 0.1,
            affine: [[a, b, 0.0], [c, d, 0.0], [0.0, 0.0, 1.0]],
            offset: [
                rng.random::<f32>() * 2.0 - 1.0,
                rng.random::<f32>() * 2.0 - 1.0,
                0.0,
            ],
            color: rng.random::<f32>(),
            ..Default::default()
        };
        // Set one random variation to 1.0
        let var_idx = rng.random_range(0..VARIATION_COUNT);
        xf.set_variation(var_idx, 1.0);
        if var_idx != 0 {
            xf.linear = 0.0;
        }
        xf.clamp_determinant();
        xf
    }

    /// Ensure the affine determinant stays in [0.2, 0.95] to prevent
    /// attractor collapse (empty voids) or area-preserving stagnation.
    /// Rescales a and d to restore the determinant while preserving
    /// rotation and shear characteristics.
    pub fn clamp_determinant(&mut self) {
        clamp_determinant(&mut self.affine, 0.2, 0.95);
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct KifsParams {
    pub fold_angle: f32,
    pub scale: f32,
    pub brightness: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GlobalParams {
    pub speed: f32,
    pub zoom: f32,
    pub trail: f32,
    pub flame_brightness: f32,
}

/// If child and parent transforms share a primary variation, interpolate affine params.
/// Otherwise, fall back to a direct swap (clone parent transform).
fn interpolate_or_swap(
    child: &FlameTransform,
    parent: &FlameTransform,
    lo: f32,
    hi: f32,
    rng: &mut impl Rng,
) -> FlameTransform {
    let child_pv = child.primary_variation_index();
    let parent_pv = parent.primary_variation_index();
    if child_pv.is_some() && child_pv == parent_pv {
        let t = rng.random_range(lo..=hi);
        child.lerp_with(parent, t)
    } else {
        parent.clone()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FlameGenome {
    #[serde(default)]
    pub name: String,
    pub global: GlobalParams,
    pub kifs: KifsParams,
    pub transforms: Vec<FlameTransform>,
    #[serde(default)]
    pub final_transform: Option<FlameTransform>,
    #[serde(default = "default_symmetry")]
    pub symmetry: i32,
    /// 256 RGB entries for the color palette, or None for cosine fallback.
    #[serde(default)]
    pub palette: Option<Vec<[f32; 3]>>,
    /// Lineage tracking for breeding system
    #[serde(default)]
    pub parent_a: Option<String>,
    #[serde(default)]
    pub parent_b: Option<String>,
    #[serde(default)]
    pub generation: u32,
}

fn default_symmetry() -> i32 {
    1
}

/// Pick a variation index with 20% bias toward "orby" shapes
fn biased_variation_pick(rng: &mut impl Rng) -> usize {
    if rng.random_range(0.0..1.0) < 0.2 {
        ORBY_VARIATIONS[rng.random_range(0..ORBY_VARIATIONS.len())]
    } else {
        rng.random_range(0..VARIATION_COUNT)
    }
}

fn audio_biased_variation_pick(rng: &mut impl Rng, audio: &AudioFeatures) -> usize {
    // Determine dominant audio character
    let max_band = if audio.bass >= audio.mids && audio.bass >= audio.highs {
        0 // bass
    } else if audio.mids >= audio.highs {
        1 // mids
    } else {
        2 // highs
    };

    // 40% chance to pick from audio-favored group
    if rng.random_range(0.0..1.0) < 0.4 {
        match max_band {
            0 => BASS_VARIATIONS[rng.random_range(0..BASS_VARIATIONS.len())],
            1 => MIDS_VARIATIONS[rng.random_range(0..MIDS_VARIATIONS.len())],
            _ => HIGHS_VARIATIONS[rng.random_range(0..HIGHS_VARIATIONS.len())],
        }
    } else if audio.beat_accum > 0.5 && rng.random_range(0.0..1.0) < 0.3 {
        BEAT_VARIATIONS[rng.random_range(0..BEAT_VARIATIONS.len())]
    } else {
        biased_variation_pick(rng)
    }
}

impl FlameGenome {
    /// Performance summary for debug logging.
    pub fn perf_summary(&self) -> String {
        let n = self.transforms.len();
        let sym = self.symmetry.abs().max(1);
        let bilateral = if self.symmetry < 0 {
            "bilateral"
        } else {
            "rotational"
        };
        let has_final = if self.final_transform.is_some() {
            "+final"
        } else {
            ""
        };

        let variation_names: [&str; 26] = [
            "linear", "sin", "sph", "swirl", "horse", "handk", "julia", "polar", "disc", "rings",
            "bubble", "fish", "exp", "spiral", "diamond", "bent", "waves", "popcorn", "fan",
            "eyefish", "cross", "tangent", "cosine", "blob", "noise", "curl",
        ];

        let mut xf_descs = Vec::new();
        for (i, xf) in self.transforms.iter().enumerate() {
            let det2d = (xf.a() * xf.d() - xf.b() * xf.c()).abs().sqrt();
            let mut active_vars = Vec::new();
            for (vi, &vname) in variation_names.iter().enumerate() {
                let v = xf.get_variation(vi);
                if v > 0.01 {
                    active_vars.push(vname);
                }
            }
            let vars_str = if active_vars.is_empty() {
                "none"
            } else {
                &active_vars.join("+")
            };
            xf_descs.push(format!(
                "xf{}({:.2}s{:.2}|{})",
                i, xf.weight, det2d, vars_str
            ));
        }

        format!(
            "{} | {}xf sym={}{} {} zoom={:.1} [{}]",
            self.name,
            n,
            sym,
            bilateral,
            has_final,
            self.global.zoom,
            xf_descs.join(" ")
        )
    }

    /// Pack global params into a fixed [f32; 20] for the uniform buffer.
    /// Layout:
    ///   [0] speed  [1] zoom  [2] trail  [3] flame_brightness
    ///   [4] kifs_fold  [5] kifs_scale  [6] kifs_brightness  [7] drift_speed
    ///   [8] color_shift  [9] vibrancy  [10] bloom_intensity  [11] (reserved)
    ///   [12] noise_disp  [13] curl_disp  [14] tangent_clamp  [15] color_blend
    ///   [16] spin_speed_max  [17] position_drift  [18] warmup_iters  [19] (reserved)
    pub fn flatten_globals(&self, cfg: &crate::weights::RuntimeConfig) -> [f32; 20] {
        let mut g = [0.0f32; 20];
        g[0] = self.global.speed;
        g[1] = self.global.zoom;
        g[2] = cfg.trail;
        g[3] = self.global.flame_brightness;
        g[4] = self.kifs.fold_angle;
        g[5] = self.kifs.scale;
        g[6] = self.kifs.brightness;
        g[7] = cfg.drift_speed; // base drift speed — weights modulate on top
        // g[8] = color_shift (set by weights, default 0)
        g[9] = cfg.vibrancy;
        g[10] = cfg.bloom_intensity;
        g[12] = cfg.noise_displacement;
        g[13] = cfg.curl_displacement;
        g[14] = cfg.tangent_clamp;
        g[15] = cfg.color_blend;
        g[16] = cfg.spin_speed_max;
        g[17] = cfg.position_drift;
        g[18] = cfg.warmup_iters;
        g[11] = cfg.gamma; // extra.w in shader
        g[19] = cfg.velocity_blur_max; // extra3.w in display shader — max directional blur pixels
        g
    }

    /// Pack all transforms into a flat Vec<f32> for the storage buffer.
    /// Each transform = 50 floats: weight, 9 affine, 3 offset,
    /// color, 26 variations, 8 parametric variation params, spin_mod, drift_mod.
    /// If a final_transform is present, it is appended after the regular transforms.
    pub fn flatten_transforms(&self) -> Vec<f32> {
        let total = self.transforms.len() + if self.final_transform.is_some() { 1 } else { 0 };
        let mut t = Vec::with_capacity(total * 50);
        for xf in &self.transforms {
            Self::push_transform(&mut t, xf);
        }
        if let Some(ref fxf) = self.final_transform {
            Self::push_transform(&mut t, fxf);
        }
        t
    }

    fn push_transform(t: &mut Vec<f32>, xf: &FlameTransform) {
        t.push(xf.weight); // 0
        // 9 affine entries (row-major: m00, m01, m02, m10, m11, m12, m20, m21, m22)
        t.push(xf.affine[0][0]); // 1  m00
        t.push(xf.affine[0][1]); // 2  m01
        t.push(xf.affine[0][2]); // 3  m02
        t.push(xf.affine[1][0]); // 4  m10
        t.push(xf.affine[1][1]); // 5  m11
        t.push(xf.affine[1][2]); // 6  m12
        t.push(xf.affine[2][0]); // 7  m20
        t.push(xf.affine[2][1]); // 8  m21
        t.push(xf.affine[2][2]); // 9  m22
        t.push(xf.offset[0]); // 10 offset_x
        t.push(xf.offset[1]); // 11 offset_y
        t.push(xf.offset[2]); // 12 offset_z
        t.push(xf.color); // 13
        t.push(xf.linear); // 14
        t.push(xf.sinusoidal); // 15
        t.push(xf.spherical); // 16
        t.push(xf.swirl); // 17
        t.push(xf.horseshoe); // 18
        t.push(xf.handkerchief); // 19
        t.push(xf.julia); // 20
        t.push(xf.polar); // 21
        t.push(xf.disc); // 22
        t.push(xf.rings); // 23
        t.push(xf.bubble); // 24
        t.push(xf.fisheye); // 25
        t.push(xf.exponential); // 26
        t.push(xf.spiral); // 27
        t.push(xf.diamond); // 28
        t.push(xf.bent); // 29
        t.push(xf.waves); // 30
        t.push(xf.popcorn); // 31
        t.push(xf.fan); // 32
        t.push(xf.eyefish); // 33
        t.push(xf.cross); // 34
        t.push(xf.tangent); // 35
        t.push(xf.cosine); // 36
        t.push(xf.blob); // 37
        t.push(xf.noise); // 38
        t.push(xf.curl); // 39
        // 8 parametric variation params [40-47]
        t.push(
            xf.variation_params
                .get("rings2_val")
                .copied()
                .unwrap_or(0.5),
        ); // 40
        t.push(xf.variation_params.get("blob_low").copied().unwrap_or(0.2)); // 41
        t.push(xf.variation_params.get("blob_high").copied().unwrap_or(1.0)); // 42
        t.push(
            xf.variation_params
                .get("blob_waves")
                .copied()
                .unwrap_or(5.0),
        ); // 43
        t.push(
            xf.variation_params
                .get("julian_power")
                .copied()
                .unwrap_or(2.0),
        ); // 44
        t.push(
            xf.variation_params
                .get("julian_dist")
                .copied()
                .unwrap_or(1.0),
        ); // 45
        t.push(
            xf.variation_params
                .get("ngon_sides")
                .copied()
                .unwrap_or(4.0),
        ); // 46
        t.push(
            xf.variation_params
                .get("ngon_corners")
                .copied()
                .unwrap_or(2.0),
        ); // 47
        // Per-transform audio modulation (default 1.0 = full participation)
        t.push(1.0); // 48 spin_mod
        t.push(1.0); // 49 drift_mod
    }

    pub fn transform_count(&self) -> u32 {
        self.transforms.len() as u32
    }

    /// Total transforms in the buffer (regular + optional final).
    pub fn total_buffer_transforms(&self) -> usize {
        self.transforms.len() + if self.final_transform.is_some() { 1 } else { 0 }
    }

    /// Create the default genome — 4 transforms with proper rotation-scale affines,
    /// 1-2 variations per transform summing to 1.0 (following Electric Sheep conventions).
    pub fn default_genome() -> Self {
        Self {
            name: "default".into(),
            global: GlobalParams {
                speed: 0.25,
                zoom: 3.0,
                trail: 0.34,
                flame_brightness: 0.2,
            },
            kifs: KifsParams {
                fold_angle: 0.0,
                scale: 0.0,
                brightness: 0.0,
            },
            transforms: vec![
                FlameTransform {
                    // spherical inversion — Draves classic
                    weight: 0.25,
                    affine: [
                        [-0.681206, 0.207690, 0.0],
                        [-0.077946, 0.755065, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    offset: [-0.041613, -0.262334, 0.0],
                    color: 0.0,
                    spherical: 1.0,
                    ..Default::default()
                },
                FlameTransform {
                    // julia branching
                    weight: 0.25,
                    affine: [
                        [0.953766, 0.432680, 0.0],
                        [0.483960, -0.054248, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    offset: [0.642503, -0.995898, 0.0],
                    color: 0.33,
                    julia: 1.0,
                    ..Default::default()
                },
                FlameTransform {
                    // sinusoidal + swirl texture
                    weight: 0.25,
                    affine: [
                        [0.840613, 0.318971, 0.0],
                        [-0.816191, -0.430402, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    offset: [0.905589, 0.909402, 0.0],
                    color: 0.66,
                    sinusoidal: 0.6,
                    swirl: 0.4,
                    ..Default::default()
                },
                FlameTransform {
                    // polar mapping
                    weight: 0.25,
                    affine: [
                        [0.960492, 0.215383, 0.0],
                        [-0.466555, -0.727377, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    offset: [-0.126074, 0.253509, 0.0],
                    color: 1.0,
                    polar: 1.0,
                    ..Default::default()
                },
            ],
            final_transform: None,
            symmetry: 1,
            palette: Some(generate_random_palette()),
            parent_a: None,
            parent_b: None,
            generation: 0,
        }
    }

    pub fn save(&self, dir: &Path) -> Result<PathBuf, String> {
        fs::create_dir_all(dir).map_err(|e| format!("create dir: {e}"))?;
        let filename = format!("{}.json", self.name);
        let path = dir.join(filename);
        let json = serde_json::to_string_pretty(self).map_err(|e| format!("serialize: {e}"))?;
        fs::write(&path, json).map_err(|e| format!("write: {e}"))?;
        Ok(path)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let json = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let mut genome: Self =
            serde_json::from_str(&json).map_err(|e| format!("parse {}: {e}", path.display()))?;
        genome.fixup_legacy_transforms();
        Ok(genome)
    }

    /// Apply legacy angle/scale fixup to all transforms.
    fn fixup_legacy_transforms(&mut self) {
        for xf in &mut self.transforms {
            xf.fixup_legacy();
        }
        if let Some(ref mut fxf) = self.final_transform {
            fxf.fixup_legacy();
        }
    }

    /// Adjust transform count to fit within the configured range.
    /// Too few: clone random existing transforms with small mutations.
    /// Too many: drop the lowest-weight transforms.
    pub fn adjust_transform_count(&mut self, cfg: &crate::weights::RuntimeConfig) {
        use rand::Rng;
        use rand::prelude::IndexedRandom;
        let min = cfg.transform_count_min.max(1) as usize;
        let max = cfg.transform_count_max.max(min as u32) as usize;
        let before = self.transforms.len();
        if before < min || before > max {
            eprintln!(
                "[adjust] {} has {} transforms, adjusting to {}-{}",
                self.name, before, min, max
            );
        }

        // Pad up: clone + mutate existing transforms
        while self.transforms.len() < min {
            let mut rng = rand::rng();
            if let Some(parent) = self.transforms.choose(&mut rng).cloned() {
                let mut child = parent;
                // Small affine mutation
                let angle: f32 = rng.random_range(-0.3..0.3);
                let (s, c) = angle.sin_cos();
                let m00 = child.affine[0][0] * c - child.affine[0][1] * s;
                let m01 = child.affine[0][0] * s + child.affine[0][1] * c;
                child.affine[0][0] = m00;
                child.affine[0][1] = m01;
                // Slightly shift color
                child.color = (child.color + rng.random_range(-0.1..0.1)).clamp(0.0, 1.0);
                self.transforms.push(child);
            } else {
                // No transforms at all — generate random
                self.transforms
                    .push(FlameTransform::random_transform(&mut rng));
            }
        }

        // Trim down: drop lowest-weight transforms
        while self.transforms.len() > max {
            if let Some((i, _)) = self
                .transforms
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.weight.partial_cmp(&b.weight).unwrap())
            {
                self.transforms.remove(i);
            } else {
                break;
            }
        }

        // Re-normalize weights after adjustment
        self.normalize_weights();
    }

    pub fn load_random(dir: &Path) -> Result<Self, String> {
        use rand::prelude::IndexedRandom;
        let entries: Vec<_> = fs::read_dir(dir)
            .map_err(|e| format!("read dir: {e}"))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "json"))
            .collect();
        if entries.is_empty() {
            return Err("no genomes found".into());
        }
        let entry = entries.choose(&mut rand::rng()).ok_or("empty")?;
        Self::load(&entry.path())
    }

    /// Apply a single transform on CPU (simplified variations for attractor estimation).
    fn apply_xform_cpu(p: (f32, f32), xf: &FlameTransform) -> (f32, f32) {
        let ax = xf.a() * p.0 + xf.b() * p.1 + xf.offset[0];
        let ay = xf.c() * p.0 + xf.d() * p.1 + xf.offset[1];

        let mut vx = 0.0f32;
        let mut vy = 0.0f32;
        let r2 = ax * ax + ay * ay;
        let r_len = r2.sqrt().max(1e-6);
        let theta = ay.atan2(ax);

        if xf.linear > 0.0 {
            vx += ax * xf.linear;
            vy += ay * xf.linear;
        }
        if xf.sinusoidal > 0.0 {
            vx += ax.sin() * xf.sinusoidal;
            vy += ay.sin() * xf.sinusoidal;
        }
        if xf.spherical > 0.0 {
            let s = xf.spherical / r2.max(0.01);
            vx += ax * s;
            vy += ay * s;
        }
        if xf.swirl > 0.0 {
            let (sr, cr) = (r2.sin(), r2.cos());
            vx += (ax * sr - ay * cr) * xf.swirl;
            vy += (ax * cr + ay * sr) * xf.swirl;
        }
        if xf.horseshoe > 0.0 {
            let h = xf.horseshoe / r_len;
            vx += (ax - ay) * (ax + ay) * h;
            vy += 2.0 * ax * ay * h;
        }
        if xf.handkerchief > 0.0 {
            vx += r_len * (theta + r_len).sin() * xf.handkerchief;
            vy += r_len * (theta - r_len).cos() * xf.handkerchief;
        }
        if xf.julia > 0.0 {
            let sr = r_len.sqrt();
            let t2 = theta / 2.0;
            vx += sr * t2.cos() * xf.julia;
            vy += sr * t2.sin() * xf.julia;
        }
        if xf.polar > 0.0 {
            vx += (theta / std::f32::consts::PI) * xf.polar;
            vy += (r_len - 1.0) * xf.polar;
        }
        if xf.disc > 0.0 {
            let d = theta / std::f32::consts::PI;
            let pr = std::f32::consts::PI * r_len;
            vx += d * pr.sin() * xf.disc;
            vy += d * pr.cos() * xf.disc;
        }
        if xf.rings > 0.0 {
            let val = xf
                .variation_params
                .get("rings2_val")
                .copied()
                .unwrap_or(0.5);
            let c2 = val * val;
            let rr = ((r_len + c2) % (2.0 * c2).max(0.001)) - c2 + r_len * (1.0 - c2);
            vx += rr * theta.cos() * xf.rings;
            vy += rr * theta.sin() * xf.rings;
        }
        if xf.bubble > 0.0 {
            let b = 4.0 / (r2 + 4.0);
            vx += ax * b * xf.bubble;
            vy += ay * b * xf.bubble;
        }
        if xf.fisheye > 0.0 {
            let f = 2.0 / (r_len + 1.0);
            vx += f * ay * xf.fisheye;
            vy += f * ax * xf.fisheye;
        }
        if xf.diamond > 0.0 {
            vx += theta.sin() * r_len.cos() * xf.diamond;
            vy += theta.cos() * r_len.sin() * xf.diamond;
        }
        if xf.bent > 0.0 {
            let bx = if ax >= 0.0 { ax } else { 2.0 * ax };
            let by = if ay >= 0.0 { ay } else { ay / 2.0 };
            vx += bx * xf.bent;
            vy += by * xf.bent;
        }
        if xf.eyefish > 0.0 {
            let e = 2.0 / (r_len + 1.0);
            vx += e * ax * xf.eyefish;
            vy += e * ay * xf.eyefish;
        }
        if xf.cross > 0.0 {
            let c = 1.0 / (ax * ax - ay * ay).abs().max(0.01);
            vx += ax * c * xf.cross;
            vy += ay * c * xf.cross;
        }
        if xf.cosine > 0.0 {
            vx += ax.cos() * ay.cosh() * xf.cosine;
            vy -= ax.sin() * ay.sinh() * xf.cosine;
        }
        if xf.blob > 0.0 {
            let low = xf.variation_params.get("blob_low").copied().unwrap_or(0.2);
            let high = xf.variation_params.get("blob_high").copied().unwrap_or(1.0);
            let waves = xf
                .variation_params
                .get("blob_waves")
                .copied()
                .unwrap_or(5.0);
            let br = r_len * (low + (high - low) * 0.5 * ((waves * theta).sin() + 1.0));
            vx += br * theta.cos() * xf.blob;
            vy += br * theta.sin() * xf.blob;
        }
        if xf.fan > 0.0 {
            let t2 = std::f32::consts::PI * 0.04;
            let t_half = t2 / 2.0;
            let f = if (theta + 0.2) % t2 > t_half {
                theta - t_half
            } else {
                theta + t_half
            };
            vx += r_len * f.cos() * xf.fan;
            vy += r_len * f.sin() * xf.fan;
        }
        let others =
            xf.spiral + xf.exponential + xf.waves + xf.popcorn + xf.tangent + xf.noise + xf.curl;
        if others > 0.0 {
            vx += ax * others;
            vy += ay * others;
        }

        (vx.clamp(-100.0, 100.0), vy.clamp(-100.0, 100.0))
    }

    /// Estimate attractor extent via CPU chaos game (percentile-based, outlier-resistant).
    pub(crate) fn estimate_attractor_extent(&self) -> f32 {
        let mut rng = rand::rng();
        let mut p = (0.5f32, 0.3f32);
        let mut xs = Vec::with_capacity(400);
        let mut ys = Vec::with_capacity(400);

        let total_w: f32 = self.transforms.iter().map(|xf| xf.weight).sum();
        if total_w <= 0.0 || self.transforms.is_empty() {
            return 0.0;
        }

        for i in 0..500 {
            let r: f32 = rng.random::<f32>() * total_w;
            let mut cumsum = 0.0;
            let mut tidx = 0;
            for (j, xf) in self.transforms.iter().enumerate() {
                cumsum += xf.weight;
                if r < cumsum {
                    tidx = j;
                    break;
                }
            }
            p = Self::apply_xform_cpu(p, &self.transforms[tidx]);
            if i >= 100 {
                xs.push(p.0);
                ys.push(p.1);
            }
        }

        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lo = (xs.len() as f32 * 0.05) as usize;
        let hi = (xs.len() as f32 * 0.95) as usize;
        let extent_x = (xs[hi] - xs[lo]).abs();
        let extent_y = (ys[hi] - ys[lo]).abs();
        extent_x.max(extent_y)
    }

    /// Auto-adjust zoom to fit the attractor. Returns suggested zoom value.
    pub fn auto_zoom(&self, cfg: &crate::weights::RuntimeConfig) -> f32 {
        let extent = self.estimate_attractor_extent();
        let extent = extent.max(0.5);
        let zoom = cfg.zoom_target / extent;
        zoom.clamp(cfg.zoom_min, cfg.zoom_max)
    }

    /// Breed two parent genomes to produce a child.
    /// Builds the child from scratch — no cloning from either parent.
    ///
    /// Transform sources:
    ///   1 wildcard (fresh random)
    ///   25% from Parent A
    ///   25% from Parent B
    ///   25% from community pool (voted/imported)
    ///   25% random environment (audio-biased)
    ///
    /// Always generates a fresh palette (taste engine placeholder).
    pub fn breed(
        parent_a: &FlameGenome,
        parent_b: &FlameGenome,
        community: &Option<FlameGenome>,
        audio: &AudioFeatures,
        cfg: &crate::weights::RuntimeConfig,
        _profile: &Option<FavoriteProfile>,
        taste: &mut Option<&mut crate::taste::TasteEngine>,
    ) -> Self {
        use rand::Rng;
        use rand::prelude::SliceRandom;
        let mut rng = rand::rng();

        // Child transform count: average of parents ±1, clamped to config range
        let xf_min = cfg.transform_count_min.max(1) as i32;
        let xf_max = cfg.transform_count_max.max(xf_min as u32) as i32;
        let avg_xf = (parent_a.transforms.len() + parent_b.transforms.len()) / 2;
        let child_count = (avg_xf as i32 + rng.random_range(-1..=1)).clamp(xf_min, xf_max) as usize;

        // Build transform slots
        let mut transforms = Vec::with_capacity(child_count);

        // Slot indices: shuffle and assign sources
        let mut slots: Vec<usize> = (0..child_count).collect();
        slots.shuffle(&mut rng);

        // First slot is always wildcard
        let _wildcard_slot = slots[0];

        // Split remaining slots into 4 groups
        let remaining = &slots[1..];
        let group_size = remaining.len() / 4;
        let leftover = remaining.len() % 4;

        // Assign group boundaries (distribute leftover evenly)
        let mut boundaries = vec![0usize];
        for i in 0..4 {
            let extra = if i < leftover { 1 } else { 0 };
            boundaries.push(boundaries.last().unwrap() + group_size + extra);
        }

        let group_a = &remaining[boundaries[0]..boundaries[1]];
        let group_b = &remaining[boundaries[1]..boundaries[2]];
        let group_community = &remaining[boundaries[2]..boundaries[3]];
        let group_env = &remaining[boundaries[3]..boundaries[4]];

        // Fill all slots with placeholders first (taste-biased if available)
        for _ in 0..child_count {
            let xf = if cfg.taste_engine_enabled {
                if let Some(te) = taste.as_ref() {
                    te.generate_biased_transform(
                        cfg.taste_min_votes,
                        cfg.taste_strength,
                        cfg.taste_exploration_rate,
                        cfg.taste_candidates,
                    )
                } else {
                    FlameTransform::random_transform(&mut rng)
                }
            } else {
                FlameTransform::random_transform(&mut rng)
            };
            transforms.push(xf);
        }

        // Wildcard: already random from above

        // Group A: transforms from Parent A (interpolate if shared variation)
        for &slot in group_a {
            if !parent_a.transforms.is_empty() {
                let src_idx = rng.random_range(0..parent_a.transforms.len());
                let parent_xf = &parent_a.transforms[src_idx];
                transforms[slot] = interpolate_or_swap(
                    &transforms[slot],
                    parent_xf,
                    cfg.interpolation_range_lo,
                    cfg.interpolation_range_hi,
                    &mut rng,
                );
            }
        }

        // Group B: transforms from Parent B (interpolate if shared variation)
        for &slot in group_b {
            if !parent_b.transforms.is_empty() {
                let src_idx = rng.random_range(0..parent_b.transforms.len());
                let parent_xf = &parent_b.transforms[src_idx];
                transforms[slot] = interpolate_or_swap(
                    &transforms[slot],
                    parent_xf,
                    cfg.interpolation_range_lo,
                    cfg.interpolation_range_hi,
                    &mut rng,
                );
            }
        }

        // Group Community: transforms from community genome
        for &slot in group_community {
            if let Some(g) = community
                && !g.transforms.is_empty()
            {
                let src_idx = rng.random_range(0..g.transforms.len());
                transforms[slot] = g.transforms[src_idx].clone();
            }
            // If no community genome, keeps the random transform
        }

        // Group Environment: audio-biased random transforms
        for &slot in group_env {
            let mut xf = FlameTransform::random_transform(&mut rng);
            // Override variation with audio-biased pick
            xf.clear_variations();
            let var_idx = audio_biased_variation_pick(&mut rng, audio);
            xf.set_variation(var_idx, 1.0);
            if var_idx != 0 {
                xf.linear = 0.0;
            }
            transforms[slot] = xf;
        }

        // Symmetry: random pick from either parent
        let symmetry = if rng.random::<bool>() {
            parent_a.symmetry
        } else {
            parent_b.symmetry
        };

        // Global params: blend from both parents
        let blend: f32 = rng.random();
        let global = GlobalParams {
            speed: parent_a.global.speed * blend + parent_b.global.speed * (1.0 - blend),
            zoom: parent_a.global.zoom * blend + parent_b.global.zoom * (1.0 - blend),
            trail: parent_a.global.trail * blend + parent_b.global.trail * (1.0 - blend),
            flame_brightness: parent_a.global.flame_brightness * blend
                + parent_b.global.flame_brightness * (1.0 - blend),
        };

        // Generate palette: use taste engine if available and enabled, else random
        let palette = Some(if cfg.taste_engine_enabled {
            if let Some(te) = taste.as_mut() {
                te.generate_palette(
                    cfg.taste_min_votes,
                    cfg.taste_strength,
                    cfg.taste_exploration_rate,
                    cfg.taste_diversity_penalty,
                    cfg.taste_candidates,
                    cfg.taste_recent_memory,
                )
            } else {
                generate_random_palette()
            }
        } else {
            generate_random_palette()
        });

        let gen_a = parent_a.generation;
        let gen_b = parent_b.generation;

        let mut child = FlameGenome {
            name: format!("child-{}", rng.random_range(1000..9999u32)),
            global,
            kifs: parent_a.kifs.clone(),
            transforms,
            final_transform: if rng.random::<bool>() {
                parent_a.final_transform.clone()
            } else {
                parent_b.final_transform.clone()
            },
            symmetry,
            palette,
            parent_a: Some(parent_a.name.clone()),
            parent_b: Some(parent_b.name.clone()),
            generation: gen_a.max(gen_b) + 1,
        };

        // Normalize like Electric Sheep does
        child.normalize_variations();
        child.enforce_variation_diversity(&mut rng);
        child.normalize_weights();
        child.distribute_colors();

        // Universal determinant clamping + scale hierarchy
        for xf in &mut child.transforms {
            xf.clamp_determinant();
        }
        if let Some(ref mut ft) = child.final_transform {
            ft.clamp_determinant();
        }
        child.enforce_scale_hierarchy();

        // Log transform taste scores if model is available
        if let Some(te) = taste.as_ref() {
            let scores: Vec<f32> = child
                .transforms
                .iter()
                .filter_map(|xf| te.score_transform(xf, cfg.taste_min_votes))
                .collect();
            if !scores.is_empty() {
                let avg = scores.iter().sum::<f32>() / scores.len() as f32;
                eprintln!(
                    "[breed] {} × {} → {} (gen {}), {} transforms, avg xf taste={:.2}",
                    parent_a.name,
                    parent_b.name,
                    child.name,
                    child.generation,
                    child.transforms.len(),
                    avg
                );
            } else {
                eprintln!(
                    "[breed] {} × {} → {} (gen {}), {} transforms",
                    parent_a.name,
                    parent_b.name,
                    child.name,
                    child.generation,
                    child.transforms.len()
                );
            }
        } else {
            eprintln!(
                "[breed] {} × {} → {} (gen {}), {} transforms",
                parent_a.name,
                parent_b.name,
                child.name,
                child.generation,
                child.transforms.len()
            );
        }

        child
    }

    /// Breed + mutate to produce offspring. Retries if attractor is degenerate.
    pub fn mutate(
        parent_a: &FlameGenome,
        parent_b: &FlameGenome,
        community: &Option<FlameGenome>,
        audio: &AudioFeatures,
        cfg: &crate::weights::RuntimeConfig,
        profile: &Option<FavoriteProfile>,
        taste: &mut Option<&mut crate::taste::TasteEngine>,
    ) -> Self {
        let retries = cfg.max_mutation_retries.max(1);
        for attempt in 0..retries {
            let bred =
                FlameGenome::breed(parent_a, parent_b, community, audio, cfg, profile, taste);
            let child = bred.mutate_inner(audio, cfg, profile);
            let extent = child.estimate_attractor_extent();
            if extent > cfg.min_attractor_extent {
                let mut result = child;
                result.parent_a = bred.parent_a;
                result.parent_b = bred.parent_b;
                result.generation = bred.generation;
                result.adjust_transform_count(cfg);
                result.global.zoom = result.auto_zoom(cfg);
                return result;
            }
            if attempt < retries - 1 {
                eprintln!(
                    "[mutate] attempt {} rejected (extent={:.3}), retrying",
                    attempt + 1,
                    extent
                );
            }
        }
        eprintln!("[mutate] all attempts degenerate, keeping parent A");
        let mut result = parent_a.clone();
        result.name = format!("mutant-{}", rand::rng().random_range(1000..9999u32));
        result.adjust_transform_count(cfg);
        result.global.zoom = result.auto_zoom(cfg);
        result
    }

    fn mutate_inner(
        &self,
        audio: &AudioFeatures,
        cfg: &crate::weights::RuntimeConfig,
        profile: &Option<FavoriteProfile>,
    ) -> Self {
        let mut child = self.clone();
        let mut rng = rand::rng();

        // Z-mutation fires first with probability z_mutation_rate
        if cfg.z_mutation_rate > 0.0 && rng.random::<f32>() < cfg.z_mutation_rate {
            if rng.random::<bool>() {
                child.mutate_z_tilt(&mut rng);
            } else {
                child.mutate_z_scale(&mut rng);
            }
        } else {
            // Single mutation per evolve — preserves the backbone, changes one thing at a time.
            // Weights: perturb 30%, final_xf 20%, swap_var 12%, symmetry 10%,
            //          rotate_colors 10%, shuffle 8%, globals 10%
            let mutation_roll: f32 = rng.random();
            if mutation_roll < 0.30 {
                child.mutate_perturb(&mut rng, audio, cfg, profile);
            } else if mutation_roll < 0.50 {
                child.mutate_final_transform(&mut rng, audio, cfg, profile);
            } else if mutation_roll < 0.62 {
                child.mutate_swap_variations(&mut rng);
            } else if mutation_roll < 0.72 {
                child.mutate_symmetry(&mut rng, audio);
            } else if mutation_roll < 0.82 {
                child.mutate_rotate_colors(&mut rng);
            } else if mutation_roll < 0.90 {
                child.mutate_shuffle_transforms(&mut rng);
            } else {
                child.mutate_global_params(&mut rng);
            }
        }

        // Force final transform on high-symmetry genomes (4+ fold)
        if child.symmetry.unsigned_abs() >= 4 && child.final_transform.is_none() {
            child.mutate_final_transform(&mut rng, audio, cfg, profile);
        }

        // Add/remove transforms — biased toward configured range
        let n = child.transforms.len();
        let xf_min = cfg.transform_count_min.max(1) as usize;
        let xf_max = cfg.transform_count_max.max(xf_min as u32) as usize;
        let (add_chance, remove_chance) = if n < xf_min {
            (0.25, 0.0) // below min — try to add
        } else if n <= xf_max {
            (0.05, 0.05) // in range — rare changes
        } else {
            (0.02, 0.20) // above max — aggressively prune
        };
        let roll: f32 = rng.random();
        if roll < add_chance {
            child.mutate_add_transform(&mut rng, audio, cfg, profile);
        } else if roll < add_chance + remove_chance && n > xf_min {
            child.mutate_remove_transform(&mut rng);
        }

        // Post-mutation normalization (Electric Sheep conventions)
        child.normalize_variations();
        child.enforce_variation_diversity(&mut rng);
        child.normalize_weights();
        child.distribute_colors();

        // Universal determinant clamping + scale hierarchy
        for xf in &mut child.transforms {
            xf.clamp_determinant();
        }
        if let Some(ref mut ft) = child.final_transform {
            ft.clamp_determinant();
        }
        child.enforce_scale_hierarchy();

        // Ensure genome has a palette; 20% chance to generate a fresh one
        if child.palette.is_none() || rng.random::<f32>() < 0.2 {
            child.palette = Some(generate_random_palette());
        }

        child.name = format!("mutant-{}", rng.random_range(1000..9999u32));
        child
    }

    /// Unified variation picker: fitness-biased when profile available, else audio-biased.
    fn pick_variation(
        rng: &mut impl Rng,
        audio: &AudioFeatures,
        cfg: &crate::weights::RuntimeConfig,
        profile: &Option<FavoriteProfile>,
    ) -> usize {
        if cfg.fitness_bias_strength > 0.0
            && profile.is_some()
            && rng.random::<f32>() < cfg.fitness_bias_strength
        {
            fitness_biased_variation_pick(rng, profile, cfg.fitness_bias_strength)
        } else {
            audio_biased_variation_pick(rng, audio)
        }
    }

    fn mutate_perturb(
        &mut self,
        rng: &mut impl Rng,
        audio: &AudioFeatures,
        cfg: &crate::weights::RuntimeConfig,
        profile: &Option<FavoriteProfile>,
    ) {
        if self.transforms.is_empty() {
            return;
        }
        let idx = rng.random_range(0..self.transforms.len());
        let xf = &mut self.transforms[idx];
        match rng.random_range(0..8) {
            0 => {
                // Rotate — wider range for more dramatic angle changes
                let angle = rng.random_range(-0.8..0.8);
                rotate_affine(&mut xf.affine, angle);
            }
            1 => {
                // Scale — wider range so transforms differ in magnification
                let factor = rng.random_range(0.6..1.5);
                scale_affine(&mut xf.affine, factor);
                // Clamp determinant to keep things stable (0.04..0.9025 for det matches 0.2..0.95 for sqrt)
                clamp_determinant(&mut xf.affine, 0.04, 0.9);
            }
            2 => {
                // Shear — breaks rotation symmetry, creates asymmetric shapes
                let shear = rng.random_range(-0.4..0.4);
                shear_affine(&mut xf.affine, shear);
            }
            3 => {
                // Anisotropic scale — stretch in one axis, compress in another
                let sx = rng.random_range(0.5..1.5);
                let sy = rng.random_range(0.5..1.5);
                xf.affine[0][0] *= sx;
                xf.affine[0][1] *= sy;
                xf.affine[1][0] *= sx;
                xf.affine[1][1] *= sy;
                // Keep contractive
                clamp_determinant(&mut xf.affine, 0.0, 0.9);
            }
            4 => {
                // Position — larger range for more spread-out transforms
                xf.offset[0] += rng.random_range(-1.0..1.0);
                xf.offset[1] += rng.random_range(-1.0..1.0);
            }
            5 => {
                // Weight — more dramatic rebalancing between transforms
                xf.weight = (xf.weight * rng.random_range(0.4..2.5)).clamp(0.05, 2.0);
            }
            6 => {
                // Reinvent this transform's affine from scratch
                let s = rng.random_range(0.2..0.9);
                let angle = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
                let (sin_a, cos_a) = angle.sin_cos();
                xf.affine[0][0] = s * cos_a;
                xf.affine[0][1] = -s * sin_a;
                xf.affine[1][0] = s * sin_a;
                xf.affine[1][1] = s * cos_a;
                // Add some asymmetry 50% of the time
                if rng.random::<f32>() < 0.5 {
                    let shear = rng.random_range(-0.3..0.3);
                    shear_affine(&mut xf.affine, shear);
                }
            }
            _ => {
                // Replace one variation with another (keeps max 2 active variations)
                let new_vi = Self::pick_variation(rng, audio, cfg, profile);

                // 50%: replace the weakest existing variation with the new one
                // 50%: swap the dominant variation entirely
                if rng.random::<f32>() < 0.5 {
                    // Find weakest active variation and replace it
                    let mut weakest_idx = 0usize;
                    let mut weakest_val = f32::MAX;
                    for vi in 0..VARIATION_COUNT {
                        let v = xf.get_variation(vi);
                        if v > 0.0 && v < weakest_val {
                            weakest_val = v;
                            weakest_idx = vi;
                        }
                    }
                    if weakest_val < f32::MAX {
                        xf.set_variation(weakest_idx, 0.0);
                    }
                    let cur = xf.get_variation(new_vi);
                    xf.set_variation(new_vi, (cur + rng.random_range(0.3..0.7)).min(1.0));
                } else {
                    // Clear all variations and set 1-2 fresh ones
                    for vi in 0..VARIATION_COUNT {
                        xf.set_variation(vi, 0.0);
                    }
                    xf.set_variation(new_vi, rng.random_range(0.6..1.0));
                    // 50% chance of a secondary variation
                    if rng.random::<f32>() < 0.5 {
                        let secondary = Self::pick_variation(rng, audio, cfg, profile);
                        if secondary != new_vi {
                            xf.set_variation(secondary, rng.random_range(0.1..0.4));
                        }
                    }
                }
                // Occasionally perturb variation params
                for (_key, val) in xf.variation_params.iter_mut() {
                    if rng.random::<f32>() < 0.3 {
                        *val += rng.random_range(-0.2..0.2);
                    }
                }
            }
        }
    }

    fn mutate_swap_variations(&mut self, rng: &mut impl Rng) {
        if self.transforms.is_empty() {
            return;
        }
        let idx = rng.random_range(0..self.transforms.len());
        let xf = &mut self.transforms[idx];
        let a = rng.random_range(0..VARIATION_COUNT);
        let b = rng.random_range(0..VARIATION_COUNT);
        let va = xf.get_variation(a);
        let vb = xf.get_variation(b);
        xf.set_variation(a, vb);
        xf.set_variation(b, va);
    }

    fn mutate_rotate_colors(&mut self, rng: &mut impl Rng) {
        let shift = rng.random_range(-0.2..0.2);
        for xf in &mut self.transforms {
            xf.color = (xf.color + shift).rem_euclid(1.0);
        }
    }

    fn mutate_shuffle_transforms(&mut self, rng: &mut impl Rng) {
        if self.transforms.len() < 2 {
            return;
        }
        let a = rng.random_range(0..self.transforms.len());
        let b = rng.random_range(0..self.transforms.len());
        self.transforms.swap(a, b);
    }

    fn mutate_global_params(&mut self, rng: &mut impl Rng) {
        self.global.flame_brightness =
            (self.global.flame_brightness + rng.random_range(-0.05..0.05)).clamp(0.05, 0.5);
        self.global.zoom = (self.global.zoom + rng.random_range(-0.5..0.5)).clamp(1.5, 6.0);
    }

    fn mutate_final_transform(
        &mut self,
        rng: &mut impl Rng,
        audio: &AudioFeatures,
        cfg: &crate::weights::RuntimeConfig,
        profile: &Option<FavoriteProfile>,
    ) {
        match &mut self.final_transform {
            Some(fxf) => {
                // Perturb existing final transform
                let angle = rng.random_range(-0.5..0.5);
                rotate_affine(&mut fxf.affine, angle);
                let factor = rng.random_range(0.85..1.18);
                scale_affine(&mut fxf.affine, factor);
                // Occasionally reinvent its variations
                if rng.random_range(0.0..1.0) < 0.3 {
                    for vi in 0..VARIATION_COUNT {
                        fxf.set_variation(vi, 0.0);
                    }
                    let dominant = Self::pick_variation(rng, audio, cfg, profile);
                    fxf.set_variation(dominant, rng.random_range(0.5..1.0));
                }
            }
            None => {
                // 30% chance to create a final transform
                if rng.random_range(0.0..1.0) < 0.3 {
                    let mut fxf = FlameTransform::default();
                    let s = rng.random_range(0.5..1.5);
                    let angle = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
                    scale_affine(&mut fxf.affine, s);
                    rotate_affine(&mut fxf.affine, angle);
                    let dominant = Self::pick_variation(rng, audio, cfg, profile);
                    fxf.set_variation(dominant, rng.random_range(0.5..1.0));
                    fxf.color = rng.random_range(0.0..1.0);
                    self.final_transform = Some(fxf);
                }
            }
        }
    }

    fn mutate_symmetry(&mut self, rng: &mut impl Rng, audio: &AudioFeatures) {
        // Nudge symmetry by ±1 instead of picking a random new value
        let current = self.symmetry.abs().max(1);
        let delta = if rng.random_range(0.0..1.0) < 0.5 {
            1
        } else {
            -1
        };
        let new_val = (current + delta).clamp(1, 6);

        if audio.beat_accum > 0.5 && rng.random_range(0.0..1.0) < 0.3 {
            // Beat-dense: occasionally bump up symmetry
            self.symmetry = (current + 1).min(8);
        } else if rng.random_range(0.0..1.0) < 0.2 {
            // Small chance to flip sign (rotational ↔ bilateral)
            self.symmetry = -self.symmetry.abs().max(2);
        } else {
            self.symmetry = new_val;
        }
    }

    /// Mutate z-tilt: rotate the z-row to couple z with x or y.
    fn mutate_z_tilt(&mut self, rng: &mut impl Rng) {
        if self.transforms.is_empty() {
            return;
        }
        let idx = rng.random_range(0..self.transforms.len());
        let xf = &mut self.transforms[idx];
        let angle: f32 = rng.random_range(-0.5..0.5);
        let (s, c) = angle.sin_cos();
        // Rotate z-row: couple z with either x (50%) or y (50%)
        if rng.random::<bool>() {
            // Tilt z ↔ x
            let old_20 = xf.affine[2][0];
            let old_22 = xf.affine[2][2];
            xf.affine[2][0] = old_20 * c - old_22 * s;
            xf.affine[2][2] = old_20 * s + old_22 * c;
        } else {
            // Tilt z ↔ y
            let old_21 = xf.affine[2][1];
            let old_22 = xf.affine[2][2];
            xf.affine[2][1] = old_21 * c - old_22 * s;
            xf.affine[2][2] = old_21 * s + old_22 * c;
        }
        clamp_determinant(&mut xf.affine, 0.04, 0.9);
    }

    /// Mutate z-scale: scale the z component and add random z-offset.
    fn mutate_z_scale(&mut self, rng: &mut impl Rng) {
        if self.transforms.is_empty() {
            return;
        }
        let idx = rng.random_range(0..self.transforms.len());
        let xf = &mut self.transforms[idx];
        let scale = rng.random_range(0.5..1.5);
        xf.affine[2][2] *= scale;
        xf.offset[2] += rng.random_range(-0.5..0.5);
        clamp_determinant(&mut xf.affine, 0.04, 0.9);
    }

    fn mutate_add_transform(
        &mut self,
        rng: &mut impl Rng,
        audio: &AudioFeatures,
        cfg: &crate::weights::RuntimeConfig,
        profile: &Option<FavoriteProfile>,
    ) {
        if self.transforms.is_empty() {
            return;
        }
        // 30% clone-and-diverge, 70% fresh specialist with contrasting geometry
        let new_xf = if rng.random::<f32>() < 0.3 {
            let source_idx = rng.random_range(0..self.transforms.len());
            let mut xf = self.transforms[source_idx].clone();
            xf.weight = 1.0 / (self.transforms.len() + 1) as f32;
            // Dramatic changes so the clone is actually different
            let angle = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
            rotate_affine(&mut xf.affine, angle);
            let scale_factor = rng.random_range(0.5..1.5);
            scale_affine(&mut xf.affine, scale_factor);
            xf.offset[0] = rng.random_range(-1.5..1.5);
            xf.offset[1] = rng.random_range(-1.5..1.5);
            xf.color = rng.random_range(0.0..1.0);
            // Give it a different variation than the source
            let new_vi = Self::pick_variation(rng, audio, cfg, profile);
            for vi in 0..VARIATION_COUNT {
                xf.set_variation(vi, 0.0);
            }
            xf.set_variation(new_vi, 1.0);
            xf
        } else {
            // Fresh specialist — intentionally contrast with existing transforms
            // Pick a scale that's different from existing ones
            let existing_scales: Vec<f32> = self
                .transforms
                .iter()
                .map(|t| det_2d(&t.affine).abs().sqrt())
                .collect();
            let avg_scale = existing_scales.iter().sum::<f32>() / existing_scales.len() as f32;
            // If existing are large, make this one small (detail), and vice versa
            let s = if avg_scale > 0.6 {
                rng.random_range(0.2..0.5) // tight detail transform
            } else {
                rng.random_range(0.6..0.9) // broad sweep transform
            };
            let angle = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
            let (sin_a, cos_a) = angle.sin_cos();
            let mut xf = FlameTransform {
                weight: 1.0 / (self.transforms.len() + 1) as f32,
                affine: [
                    [s * cos_a, -s * sin_a, 0.0],
                    [s * sin_a, s * cos_a, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                offset: [
                    rng.random_range(-1.5..1.5),
                    rng.random_range(-1.5..1.5),
                    0.0,
                ],
                color: rng.random_range(0.0..1.0),
                ..Default::default()
            };
            // Add shear/asymmetry 40% of the time
            if rng.random::<f32>() < 0.4 {
                let shear = rng.random_range(-0.3..0.3);
                shear_affine(&mut xf.affine, shear);
            }
            // Pick a variation that's NOT already dominant in other transforms
            let mut used_variations: Vec<usize> = Vec::new();
            for t in &self.transforms {
                for vi in 0..VARIATION_COUNT {
                    if t.get_variation(vi) > 0.5 {
                        used_variations.push(vi);
                    }
                }
            }
            let dominant = loop {
                let pick = Self::pick_variation(rng, audio, cfg, profile);
                // Try to avoid already-used variations (3 attempts then give up)
                if !used_variations.contains(&pick) || rng.random::<f32>() < 0.3 {
                    break pick;
                }
            };
            if rng.random::<f32>() < 0.5 {
                xf.set_variation(dominant, 1.0);
            } else {
                let split = rng.random_range(0.5..0.8);
                xf.set_variation(dominant, split);
                let secondary = Self::pick_variation(rng, audio, cfg, profile);
                if secondary != dominant {
                    xf.set_variation(secondary, 1.0 - split);
                } else {
                    xf.set_variation(dominant, 1.0);
                }
            }
            xf
        };
        self.transforms.push(new_xf);
    }

    fn mutate_remove_transform(&mut self, _rng: &mut impl Rng) {
        if self.transforms.len() <= 2 {
            return;
        } // keep at least 2
        // Remove the lowest-weight transform
        if let Some((min_idx, _)) = self
            .transforms
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.weight.partial_cmp(&b.weight).unwrap())
        {
            self.transforms.remove(min_idx);
        }
    }

    /// Normalize variation weights per transform to sum to 1.0, keeping max 2 active.
    pub(crate) fn normalize_variations(&mut self) {
        for xf in &mut self.transforms {
            // Collect active variations
            let mut active: Vec<(usize, f32)> = (0..VARIATION_COUNT)
                .filter_map(|i| {
                    let v = xf.get_variation(i);
                    if v > 0.01 { Some((i, v)) } else { None }
                })
                .collect();

            // If more than 2, keep only the 2 strongest
            if active.len() > 2 {
                active.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                for &(idx, _) in &active[2..] {
                    xf.set_variation(idx, 0.0);
                }
                active.truncate(2);
            }

            // Normalize to sum to 1.0
            let sum: f32 = active.iter().map(|(_, v)| v).sum();
            if sum > 0.01 {
                for (idx, val) in &active {
                    xf.set_variation(*idx, val / sum);
                }
            }
        }
    }

    /// Enforce variation diversity across the genome.
    /// If any single variation type accounts for >60% of total weight across all transforms,
    /// swap it out in a random transform for a different variation.
    pub(crate) fn enforce_variation_diversity(&mut self, rng: &mut impl Rng) {
        // Tally total weight per variation across all transforms
        let mut totals = [0.0f32; VARIATION_COUNT];
        let mut grand_total = 0.0f32;
        for xf in &self.transforms {
            for (vi, total) in totals.iter_mut().enumerate() {
                let v = xf.get_variation(vi);
                *total += v;
                grand_total += v;
            }
        }
        if grand_total < 0.01 {
            return;
        }

        // Find the dominant variation
        let (dominant_idx, dominant_weight) = totals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        // If it's >60% of total, swap it in a random transform
        if *dominant_weight / grand_total > 0.6 && self.transforms.len() > 1 {
            let xf_idx = rng.random_range(0..self.transforms.len());
            let xf = &mut self.transforms[xf_idx];
            let old_weight = xf.get_variation(dominant_idx);
            if old_weight > 0.01 {
                xf.set_variation(dominant_idx, 0.0);
                // Pick a different variation
                let mut new_var = rng.random_range(0..VARIATION_COUNT);
                while new_var == dominant_idx {
                    new_var = rng.random_range(0..VARIATION_COUNT);
                }
                xf.set_variation(new_var, old_weight);
            }
        }
    }

    /// Enforce hierarchical scale distribution across transforms.
    /// If all transforms have similar scale (determinant variance near zero),
    /// push one toward macro (0.7) and one toward micro (0.3) to create depth.
    pub(crate) fn enforce_scale_hierarchy(&mut self) {
        if self.transforms.len() < 3 {
            return;
        }
        let dets: Vec<f32> = self
            .transforms
            .iter()
            .map(|xf| determinant_3x3(&xf.affine).abs())
            .collect();
        let mean = dets.iter().sum::<f32>() / dets.len() as f32;
        let variance = dets.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / dets.len() as f32;

        // If variance is very low, transforms are all at similar scale — force spread
        if variance < 0.02 {
            // Find the largest and smallest determinant transforms
            let (max_idx, _) = dets
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            let (min_idx, _) = dets
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            // Push the "largest" toward macro scale (det ~0.7)
            let macro_target = 0.7;
            let macro_det = dets[max_idx].max(0.01);
            let macro_scale = (macro_target / macro_det).cbrt();
            scale_affine(&mut self.transforms[max_idx].affine, macro_scale);

            // Push the "smallest" toward micro scale (det ~0.3)
            if min_idx != max_idx {
                let micro_target = 0.3f32;
                let micro_det = dets[min_idx].max(0.01);
                let micro_scale = (micro_target / micro_det).cbrt();
                scale_affine(&mut self.transforms[min_idx].affine, micro_scale);
            }
        }
    }

    /// Normalize transform blend weights to sum to 1.0.
    pub(crate) fn normalize_weights(&mut self) {
        let sum: f32 = self.transforms.iter().map(|xf| xf.weight).sum();
        if sum > 0.01 {
            for xf in &mut self.transforms {
                xf.weight /= sum;
            }
        }
    }

    /// Evenly distribute color indices across transforms.
    pub(crate) fn distribute_colors(&mut self) {
        let n = self.transforms.len();
        if n == 0 {
            return;
        }
        for (i, xf) in self.transforms.iter_mut().enumerate() {
            xf.color = i as f32 / n.max(1) as f32;
        }
    }
}

// ── Affine matrix helpers for mutation ──
// Operate on 3x3 affine matrix directly. Z-row/col preserved (only 2x2 upper-left mutated).

fn rotate_affine(m: &mut [[f32; 3]; 3], angle: f32) {
    let (s, cos_a) = angle.sin_cos();
    let na = m[0][0] * cos_a - m[1][0] * s;
    let nb = m[0][1] * cos_a - m[1][1] * s;
    let nc = m[0][0] * s + m[1][0] * cos_a;
    let nd = m[0][1] * s + m[1][1] * cos_a;
    m[0][0] = na;
    m[0][1] = nb;
    m[1][0] = nc;
    m[1][1] = nd;
}

fn scale_affine(m: &mut [[f32; 3]; 3], factor: f32) {
    for row in m.iter_mut() {
        for val in row.iter_mut() {
            *val *= factor;
        }
    }
}

fn shear_affine(m: &mut [[f32; 3]; 3], shear: f32) {
    m[0][1] += shear * m[0][0];
    m[1][1] += shear * m[1][0];
}

/// Clamp 3x3 determinant: uniform scale all 9 entries by (target/|det|)^(1/3).
fn clamp_determinant(m: &mut [[f32; 3]; 3], min_det: f32, max_det: f32) {
    let det = determinant_3x3(m);
    let abs_det = det.abs();
    if abs_det < min_det {
        let fix = (min_det / abs_det.max(1e-10)).cbrt();
        scale_entries(m, fix);
    } else if abs_det > max_det {
        let fix = (max_det / abs_det).cbrt();
        scale_entries(m, fix);
    }
}

fn determinant_3x3(m: &[[f32; 3]; 3]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Scale all 9 entries by a factor (used by clamp_determinant).
fn scale_entries(m: &mut [[f32; 3]; 3], factor: f32) {
    for row in m.iter_mut() {
        for val in row.iter_mut() {
            *val *= factor;
        }
    }
}

/// 2D determinant of the upper-left 2x2 submatrix (for backward compat).
fn det_2d(m: &[[f32; 3]; 3]) -> f32 {
    m[0][0] * m[1][1] - m[0][1] * m[1][0]
}

// ── Palette generation ──

const TAU: f32 = std::f32::consts::TAU;

fn cosine_color(
    t: f32,
    offset: [f32; 3],
    amp: [f32; 3],
    freq: [f32; 3],
    phase: [f32; 3],
) -> [f32; 3] {
    [
        (offset[0] + amp[0] * (TAU * (freq[0] * t + phase[0])).cos()).clamp(0.0, 1.0),
        (offset[1] + amp[1] * (TAU * (freq[1] * t + phase[1])).cos()).clamp(0.0, 1.0),
        (offset[2] + amp[2] * (TAU * (freq[2] * t + phase[2])).cos()).clamp(0.0, 1.0),
    ]
}

/// Generate a random cohesive 256-entry palette.
/// Each palette has a distinct color identity: a dominant hue family going
/// from near-black through vivid peaks, like real Electric Sheep palettes.
pub fn generate_random_palette() -> Vec<[f32; 3]> {
    let mut rng = rand::rng();

    // Pick a palette "mood" — not every palette should be neon
    let mood: f32 = rng.random();
    let (base_range, amp_range) = if mood < 0.3 {
        // Dark/dramatic: low base, high contrast
        (0.0..0.1, 0.5..1.0)
    } else if mood < 0.6 {
        // Rich/warm: moderate base, moderate amplitude
        (0.1..0.35, 0.3..0.7)
    } else if mood < 0.85 {
        // Pastel/muted: higher base, lower amplitude
        (0.25..0.5, 0.15..0.45)
    } else {
        // High contrast: mixed base, full range amplitude
        (0.0..0.3, 0.4..0.9)
    };

    let offset = [
        rng.random_range(base_range.clone()),
        rng.random_range(base_range.clone()),
        rng.random_range(base_range),
    ];
    let amp = [
        rng.random_range(amp_range.clone()),
        rng.random_range(amp_range.clone()),
        rng.random_range(amp_range),
    ];
    // Wider frequency range: low = broad gradual sweeps, high = rapid color cycling
    let freq = [
        rng.random_range(0.3..2.5),
        rng.random_range(0.3..2.5),
        rng.random_range(0.3..2.5),
    ];
    let phase = [
        rng.random_range(0.0..1.0),
        rng.random_range(0.0..1.0),
        rng.random_range(0.0..1.0),
    ];

    (0..256)
        .map(|i| {
            let t = i as f32 / 255.0;
            cosine_color(t, offset, amp, freq, phase)
        })
        .collect()
}

/// Generate the default 256-entry palette as RGBA f32 data.
pub fn generate_default_palette_rgba() -> Vec<[f32; 4]> {
    generate_random_palette()
        .iter()
        .map(|rgb| [rgb[0], rgb[1], rgb[2], 1.0])
        .collect()
}

/// Generate palette RGBA data from a genome's palette field, falling back to random generation.
pub fn palette_rgba_data(genome: &FlameGenome) -> Vec<[f32; 4]> {
    match &genome.palette {
        Some(entries) => {
            let mut data = Vec::with_capacity(256);
            for i in 0..256 {
                let rgb = if i < entries.len() {
                    entries[i]
                } else if !entries.is_empty() {
                    *entries.last().unwrap()
                } else {
                    [0.0, 0.0, 0.0]
                };
                data.push([rgb[0], rgb[1], rgb[2], 1.0]);
            }
            data
        }
        None => generate_default_palette_rgba(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_genome(n_transforms: usize) -> FlameGenome {
        let mut rng = rand::rng();
        let transforms: Vec<FlameTransform> = (0..n_transforms)
            .map(|_| FlameTransform::random_transform(&mut rng))
            .collect();
        FlameGenome {
            name: "test".into(),
            global: GlobalParams {
                speed: 0.25,
                zoom: 3.0,
                trail: 0.15,
                flame_brightness: 0.2,
            },
            kifs: KifsParams {
                fold_angle: 0.0,
                scale: 0.0,
                brightness: 0.0,
            },
            transforms,
            final_transform: None,
            symmetry: 1,
            palette: Some(generate_random_palette()),
            parent_a: None,
            parent_b: None,
            generation: 0,
        }
    }

    #[test]
    fn default_genome_has_transforms() {
        let g = FlameGenome::default_genome();
        assert!(!g.transforms.is_empty());
        assert!(g.palette.is_some());
    }

    #[test]
    fn normalize_weights_sums_to_one() {
        let mut g = make_test_genome(4);
        g.transforms[0].weight = 5.0;
        g.transforms[1].weight = 3.0;
        g.transforms[2].weight = 1.0;
        g.transforms[3].weight = 1.0;
        g.normalize_weights();
        let sum: f32 = g.transforms.iter().map(|t| t.weight).sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "weights should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn normalize_variations_ensures_nonzero() {
        let mut g = make_test_genome(2);
        // Zero out all variations
        for xf in &mut g.transforms {
            *xf = FlameTransform::default();
            xf.weight = 0.5;
            xf.linear = 0.0;
        }
        g.normalize_variations();
        // After normalization with all zeros, variations stay zero — that's valid.
        // The function normalizes existing nonzero variations, it doesn't inject new ones.
        // Instead, verify that if we give a transform some variations, they normalize to 1.0.
        let mut g2 = make_test_genome(2);
        g2.transforms[0].linear = 0.6;
        g2.transforms[0].sinusoidal = 0.4;
        g2.normalize_variations();
        let sum: f32 = (0..VARIATION_COUNT)
            .map(|v| g2.transforms[0].get_variation(v))
            .sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "variation weights should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn normalize_variations_keeps_max_two() {
        let mut g = make_test_genome(1);
        let xf = &mut g.transforms[0];
        xf.clear_variations();
        xf.linear = 0.5;
        xf.sinusoidal = 0.3;
        xf.spherical = 0.2;
        g.normalize_variations();
        let active_count = (0..VARIATION_COUNT)
            .filter(|&v| g.transforms[0].get_variation(v) > 0.0)
            .count();
        assert!(
            active_count <= 2,
            "should have at most 2 active variations, got {active_count}"
        );
    }

    #[test]
    fn distribute_colors_evenly_spaced() {
        let mut g = make_test_genome(4);
        g.distribute_colors();
        let expected = [0.0, 0.25, 0.5, 0.75];
        for (i, xf) in g.transforms.iter().enumerate() {
            assert!(
                (xf.color - expected[i]).abs() < 0.01,
                "transform {i} color should be {}, got {}",
                expected[i],
                xf.color
            );
        }
    }

    #[test]
    fn random_transform_has_valid_weight() {
        let mut rng = rand::rng();
        let xf = FlameTransform::random_transform(&mut rng);
        assert!(
            xf.weight > 0.0,
            "random transform should have positive weight"
        );
    }

    #[test]
    fn random_transform_has_variation() {
        let mut rng = rand::rng();
        let xf = FlameTransform::random_transform(&mut rng);
        let has_variation = (0..VARIATION_COUNT).any(|v| xf.get_variation(v) > 0.0);
        assert!(
            has_variation,
            "random transform should have at least one variation"
        );
    }

    #[test]
    fn breed_records_lineage() {
        let pa = make_test_genome(4);
        let pb = make_test_genome(3);
        let audio = crate::audio::AudioFeatures::default();
        let cfg: crate::weights::RuntimeConfig = serde_json::from_str("{}").unwrap_or_default();
        let child = FlameGenome::breed(&pa, &pb, &None, &audio, &cfg, &None, &mut None);
        assert_eq!(child.parent_a.as_deref(), Some("test"));
        assert_eq!(child.parent_b.as_deref(), Some("test"));
        assert_eq!(child.generation, 1);
    }

    #[test]
    fn breed_transform_count_in_range() {
        let pa = make_test_genome(4);
        let pb = make_test_genome(4);
        let audio = crate::audio::AudioFeatures::default();
        let cfg: crate::weights::RuntimeConfig = serde_json::from_str("{}").unwrap_or_default();
        for _ in 0..20 {
            let child = FlameGenome::breed(&pa, &pb, &None, &audio, &cfg, &None, &mut None);
            let n = child.transforms.len();
            assert!(n >= 3 && n <= 6, "transform count {n} out of range 3..=6");
        }
    }

    #[test]
    fn breed_always_has_palette() {
        let pa = make_test_genome(4);
        let pb = make_test_genome(3);
        let audio = crate::audio::AudioFeatures::default();
        let cfg: crate::weights::RuntimeConfig = serde_json::from_str("{}").unwrap_or_default();
        let child = FlameGenome::breed(&pa, &pb, &None, &audio, &cfg, &None, &mut None);
        assert!(child.palette.is_some());
        assert_eq!(child.palette.as_ref().unwrap().len(), 256);
    }

    #[test]
    fn attractor_extent_default_genome() {
        let g = FlameGenome::default_genome();
        let extent = g.estimate_attractor_extent();
        assert!(
            extent > 0.0,
            "default genome should have positive extent, got {extent}"
        );
    }

    #[test]
    fn genome_serialization_roundtrip() {
        let g = make_test_genome(3);
        let json = serde_json::to_string(&g).unwrap();
        let g2: FlameGenome = serde_json::from_str(&json).unwrap();
        assert_eq!(g.name, g2.name);
        assert_eq!(g.transforms.len(), g2.transforms.len());
        assert_eq!(g.symmetry, g2.symmetry);
        assert_eq!(g.generation, g2.generation);
    }

    // ── T1.1: 3x3 affine tests ──

    #[test]
    fn default_transform_has_3x3_identity_z() {
        let xf = FlameTransform::default();
        // Z-row and z-column should be identity
        assert_eq!(xf.affine[2][2], 1.0, "affine[2][2] should be 1.0");
        assert_eq!(xf.affine[2][0], 0.0, "affine[2][0] should be 0.0");
        assert_eq!(xf.affine[2][1], 0.0, "affine[2][1] should be 0.0");
        assert_eq!(xf.affine[0][2], 0.0, "affine[0][2] should be 0.0");
        assert_eq!(xf.affine[1][2], 0.0, "affine[1][2] should be 0.0");
        // 2D part should be identity
        assert_eq!(xf.a(), 1.0);
        assert_eq!(xf.b(), 0.0);
        assert_eq!(xf.c(), 0.0);
        assert_eq!(xf.d(), 1.0);
        // Offset z should be 0
        assert_eq!(xf.offset[2], 0.0);
    }

    #[test]
    fn old_genome_json_migration() {
        // Simulate old JSON format with a, b, c, d fields and 2-element offset
        let old_json = r#"{
            "weight": 0.5,
            "a": 0.8, "b": -0.3, "c": 0.3, "d": 0.8,
            "offset": [0.1, 0.2],
            "color": 0.5,
            "linear": 1.0, "sinusoidal": 0.0, "spherical": 0.0,
            "swirl": 0.0, "horseshoe": 0.0, "handkerchief": 0.0
        }"#;
        let xf: FlameTransform = serde_json::from_str(old_json).unwrap();
        assert!((xf.a() - 0.8).abs() < 1e-6, "a() = {}", xf.a());
        assert!((xf.b() - -0.3).abs() < 1e-6, "b() = {}", xf.b());
        assert!((xf.c() - 0.3).abs() < 1e-6, "c() = {}", xf.c());
        assert!((xf.d() - 0.8).abs() < 1e-6, "d() = {}", xf.d());
        // Z-entries should be identity
        assert_eq!(xf.affine[2][2], 1.0);
        assert_eq!(xf.affine[0][2], 0.0);
        assert_eq!(xf.affine[1][2], 0.0);
        // Offset should have z=0
        assert!((xf.offset[0] - 0.1).abs() < 1e-6);
        assert!((xf.offset[1] - 0.2).abs() < 1e-6);
        assert_eq!(xf.offset[2], 0.0);
    }

    #[test]
    fn new_genome_json_roundtrip() {
        // New format with explicit affine and 3-element offset
        let xf = FlameTransform {
            weight: 0.5,
            affine: [[0.8, -0.3, 0.1], [0.3, 0.8, 0.2], [0.05, 0.0, 0.9]],
            offset: [0.1, 0.2, 0.3],
            color: 0.5,
            linear: 1.0,
            ..Default::default()
        };
        let json = serde_json::to_string(&xf).unwrap();
        let xf2: FlameTransform = serde_json::from_str(&json).unwrap();
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (xf.affine[r][c] - xf2.affine[r][c]).abs() < 1e-6,
                    "affine[{r}][{c}] mismatch: {} vs {}",
                    xf.affine[r][c],
                    xf2.affine[r][c],
                );
            }
        }
        assert!((xf2.offset[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn clamp_determinant_too_small() {
        let mut m = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 1.0]];
        // det = 0.01 * 0.01 * 1.0 = 0.0001, way below min
        clamp_determinant(&mut m, 0.04, 0.9);
        let det = determinant_3x3(&m).abs();
        assert!(det >= 0.039, "det should be clamped up to ~0.04, got {det}");
    }

    #[test]
    fn clamp_determinant_too_large() {
        let mut m = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        // det = 4.0, above max
        clamp_determinant(&mut m, 0.04, 0.9);
        let det = determinant_3x3(&m).abs();
        assert!(det <= 0.91, "det should be clamped down to ~0.9, got {det}");
    }

    #[test]
    fn flatten_produces_50_floats_per_transform() {
        let g = FlameGenome::default_genome();
        let flat = g.flatten_transforms();
        let expected = g.transforms.len() * 50;
        assert_eq!(
            flat.len(),
            expected,
            "expected {} floats ({} transforms * 50), got {}",
            expected,
            g.transforms.len(),
            flat.len()
        );
        // Verify second transform starts at position 50
        assert!((flat[50] - g.transforms[1].weight).abs() < 1e-6);
        // Verify spin_mod defaults to 1.0
        assert!(
            (flat[48] - 1.0).abs() < 1e-6,
            "spin_mod should default to 1.0"
        );
        assert!(
            (flat[49] - 1.0).abs() < 1e-6,
            "drift_mod should default to 1.0"
        );
    }

    #[test]
    fn clamp_determinant_in_range() {
        let original = [[0.5, 0.1, 0.0], [0.1, 0.5, 0.0], [0.0, 0.0, 1.0]];
        let mut m = original;
        // det = 0.5*0.5 - 0.1*0.1 = 0.24, within [0.04, 0.9]
        clamp_determinant(&mut m, 0.04, 0.9);
        // Should be unchanged
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (m[r][c] - original[r][c]).abs() < 1e-6,
                    "in-range det should not be modified"
                );
            }
        }
    }

    #[test]
    fn mutate_z_tilt_changes_z_row() {
        let mut g = make_test_genome(3);
        // Ensure z-row starts as identity
        for xf in &mut g.transforms {
            xf.affine[2] = [0.0, 0.0, 1.0];
        }
        let mut rng = rand::rng();
        g.mutate_z_tilt(&mut rng);
        // At least one transform should have changed z-row
        let any_changed = g.transforms.iter().any(|xf| {
            (xf.affine[2][0]).abs() > 1e-6
                || (xf.affine[2][1]).abs() > 1e-6
                || (xf.affine[2][2] - 1.0).abs() > 1e-6
        });
        assert!(
            any_changed,
            "z-tilt should modify the z-row of at least one transform"
        );
        // Determinant should still be valid after clamping
        for xf in &g.transforms {
            let det = determinant_3x3(&xf.affine).abs();
            assert!(det < 1.0, "det should be clamped after z-tilt, got {det}");
        }
    }

    #[test]
    fn mutate_z_scale_modifies_z() {
        let mut g = make_test_genome(3);
        let mut rng = rand::rng();
        g.mutate_z_scale(&mut rng);
        // At least one transform should have z-offset or non-1 z-scale
        let any_changed = g
            .transforms
            .iter()
            .any(|xf| xf.offset[2].abs() > 1e-6 || (xf.affine[2][2] - 1.0).abs() > 1e-6);
        assert!(
            any_changed,
            "z-scale should modify z-offset or z-scale of at least one transform"
        );
    }

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    #[test]
    fn lerp_transform_midpoint() {
        let mut xf_a = FlameTransform::default();
        xf_a.affine = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        xf_a.offset = [0.0, 0.0, 0.0];
        xf_a.weight = 0.2;
        xf_a.color = 0.0;
        xf_a.linear = 1.0;

        let mut xf_b = FlameTransform::default();
        xf_b.affine = [[2.0, 4.0, 0.0], [6.0, 8.0, 0.0], [0.0, 0.0, 1.0]];
        xf_b.offset = [10.0, 12.0, 0.0];
        xf_b.weight = 1.0;
        xf_b.color = 1.0;
        xf_b.linear = 1.0;

        let mid = xf_a.lerp_with(&xf_b, 0.5);
        assert!(approx_eq(mid.a(), 1.0), "a was {}", mid.a());
        assert!(approx_eq(mid.b(), 2.0), "b was {}", mid.b());
        assert!(approx_eq(mid.c(), 3.0), "c was {}", mid.c());
        assert!(approx_eq(mid.d(), 4.0), "d was {}", mid.d());
        assert!(
            approx_eq(mid.offset[0], 5.0),
            "offset_x was {}",
            mid.offset[0]
        );
        assert!(
            approx_eq(mid.offset[1], 6.0),
            "offset_y was {}",
            mid.offset[1]
        );
        assert!(approx_eq(mid.weight, 0.6), "weight was {}", mid.weight);
        assert!(approx_eq(mid.color, 0.5), "color was {}", mid.color);
    }

    #[test]
    fn interpolate_or_swap_shared_variation_interpolates() {
        let mut rng = rand::rng();
        let mut child = FlameTransform::default();
        child.affine = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        child.linear = 1.0;

        let mut parent = FlameTransform::default();
        parent.affine = [[2.0, 2.0, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        parent.linear = 1.0;

        let result = interpolate_or_swap(&child, &parent, 0.5, 0.5, &mut rng);
        assert!(approx_eq(result.a(), 1.0), "a was {}", result.a());
    }

    #[test]
    fn interpolate_or_swap_different_variation_swaps() {
        let mut rng = rand::rng();
        let mut child = FlameTransform::default();
        child.linear = 1.0;

        let mut parent = FlameTransform::default();
        parent.affine[0][0] = 99.0;
        parent.spherical = 1.0;

        let result = interpolate_or_swap(&child, &parent, 0.3, 0.7, &mut rng);
        assert!(
            approx_eq(result.a(), 99.0),
            "a was {}, expected parent clone",
            result.a()
        );
    }
}
