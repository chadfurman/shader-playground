use std::fs;
use std::path::{Path, PathBuf};

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::audio::AudioFeatures;

const VARIATION_COUNT: usize = 26;

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
const BASS_VARIATIONS: [usize; 4] = [2, 10, 23, 11];   // spherical, bubble, blob, fisheye
const MIDS_VARIATIONS: [usize; 4] = [1, 16, 22, 5];    // sinusoidal, waves, cosine, handkerchief
const HIGHS_VARIATIONS: [usize; 5] = [6, 8, 20, 21, 14]; // julia, disc, cross, tangent, diamond
const BEAT_VARIATIONS: [usize; 3] = [13, 3, 7];          // spiral, swirl, polar

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct FlameTransform {
    pub weight: f32,
    pub angle: f32,
    pub scale: f32,
    pub offset: [f32; 2],
    pub color: f32,
    // Original 6 variations
    pub linear: f32,
    pub sinusoidal: f32,
    pub spherical: f32,
    pub swirl: f32,
    pub horseshoe: f32,
    pub handkerchief: f32,
    // New 20 variations
    #[serde(default)] pub julia: f32,
    #[serde(default)] pub polar: f32,
    #[serde(default)] pub disc: f32,
    #[serde(default)] pub rings: f32,
    #[serde(default)] pub bubble: f32,
    #[serde(default)] pub fisheye: f32,
    #[serde(default)] pub exponential: f32,
    #[serde(default)] pub spiral: f32,
    #[serde(default)] pub diamond: f32,
    #[serde(default)] pub bent: f32,
    #[serde(default)] pub waves: f32,
    #[serde(default)] pub popcorn: f32,
    #[serde(default)] pub fan: f32,
    #[serde(default)] pub eyefish: f32,
    #[serde(default)] pub cross: f32,
    #[serde(default)] pub tangent: f32,
    #[serde(default)] pub cosine: f32,
    #[serde(default)] pub blob: f32,
    #[serde(default)] pub noise: f32,
    #[serde(default)] pub curl: f32,
}

impl FlameTransform {
    fn get_variation(&self, idx: usize) -> f32 {
        match idx {
            0 => self.linear, 1 => self.sinusoidal, 2 => self.spherical,
            3 => self.swirl, 4 => self.horseshoe, 5 => self.handkerchief,
            6 => self.julia, 7 => self.polar, 8 => self.disc,
            9 => self.rings, 10 => self.bubble, 11 => self.fisheye,
            12 => self.exponential, 13 => self.spiral, 14 => self.diamond,
            15 => self.bent, 16 => self.waves, 17 => self.popcorn,
            18 => self.fan, 19 => self.eyefish, 20 => self.cross,
            21 => self.tangent, 22 => self.cosine, 23 => self.blob,
            24 => self.noise, 25 => self.curl,
            _ => 0.0,
        }
    }

    fn set_variation(&mut self, idx: usize, val: f32) {
        match idx {
            0 => self.linear = val, 1 => self.sinusoidal = val, 2 => self.spherical = val,
            3 => self.swirl = val, 4 => self.horseshoe = val, 5 => self.handkerchief = val,
            6 => self.julia = val, 7 => self.polar = val, 8 => self.disc = val,
            9 => self.rings = val, 10 => self.bubble = val, 11 => self.fisheye = val,
            12 => self.exponential = val, 13 => self.spiral = val, 14 => self.diamond = val,
            15 => self.bent = val, 16 => self.waves = val, 17 => self.popcorn = val,
            18 => self.fan = val, 19 => self.eyefish = val, 20 => self.cross = val,
            21 => self.tangent = val, 22 => self.cosine = val, 23 => self.blob = val,
            24 => self.noise = val, 25 => self.curl = val,
            _ => {}
        }
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
}

fn default_symmetry() -> i32 { 1 }

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
        // Scale flame_brightness for accumulation depth: with decay 0.995, density is
        // ~200x higher than single-frame. Multiply by (1-decay) to normalize sensitivity.
        g[3] = self.global.flame_brightness * (1.0 - cfg.accumulation_decay);
        g[4] = self.kifs.fold_angle;
        g[5] = self.kifs.scale;
        g[6] = self.kifs.brightness;
        // g[7] = drift_speed (set by weights, default 0)
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
        g[11] = cfg.gamma;            // extra.w in shader
        g[19] = cfg.highlight_power;  // extra3.w in shader
        g
    }

    /// Pack all transforms into a flat Vec<f32> for the storage buffer.
    /// Each transform = 32 floats: weight, angle, scale, offset_x, offset_y,
    /// color, linear, sinusoidal, spherical, swirl, horseshoe, handkerchief,
    /// julia, polar, disc, rings, bubble, fisheye, exponential, spiral,
    /// diamond, bent, waves, popcorn, fan, eyefish, cross, tangent,
    /// cosine, blob, noise, curl.
    /// If a final_transform is present, it is appended after the regular transforms.
    pub fn flatten_transforms(&self) -> Vec<f32> {
        let total = self.transforms.len() + if self.final_transform.is_some() { 1 } else { 0 };
        let mut t = Vec::with_capacity(total * 32);
        for xf in &self.transforms {
            Self::push_transform(&mut t, xf);
        }
        if let Some(ref fxf) = self.final_transform {
            Self::push_transform(&mut t, fxf);
        }
        t
    }

    fn push_transform(t: &mut Vec<f32>, xf: &FlameTransform) {
        t.push(xf.weight);       // 0
        t.push(xf.angle);        // 1
        t.push(xf.scale);        // 2
        t.push(xf.offset[0]);    // 3
        t.push(xf.offset[1]);    // 4
        t.push(xf.color);        // 5
        t.push(xf.linear);       // 6
        t.push(xf.sinusoidal);   // 7
        t.push(xf.spherical);    // 8
        t.push(xf.swirl);        // 9
        t.push(xf.horseshoe);    // 10
        t.push(xf.handkerchief); // 11
        t.push(xf.julia);        // 12
        t.push(xf.polar);        // 13
        t.push(xf.disc);         // 14
        t.push(xf.rings);        // 15
        t.push(xf.bubble);       // 16
        t.push(xf.fisheye);      // 17
        t.push(xf.exponential);  // 18
        t.push(xf.spiral);       // 19
        t.push(xf.diamond);      // 20
        t.push(xf.bent);         // 21
        t.push(xf.waves);        // 22
        t.push(xf.popcorn);      // 23
        t.push(xf.fan);          // 24
        t.push(xf.eyefish);      // 25
        t.push(xf.cross);        // 26
        t.push(xf.tangent);      // 27
        t.push(xf.cosine);       // 28
        t.push(xf.blob);         // 29
        t.push(xf.noise);        // 30
        t.push(xf.curl);         // 31
    }

    pub fn transform_count(&self) -> u32 {
        self.transforms.len() as u32
    }

    /// Total transforms in the buffer (regular + optional final).
    pub fn total_buffer_transforms(&self) -> usize {
        self.transforms.len() + if self.final_transform.is_some() { 1 } else { 0 }
    }

    /// Create the default genome matching current hardcoded transforms.
    pub fn default_genome() -> Self {
        Self {
            name: "default".into(),
            global: GlobalParams {
                speed: 0.25,
                zoom: 3.0,
                trail: 0.34,
                flame_brightness: 0.4,
            },
            kifs: KifsParams {
                fold_angle: 0.62,
                scale: 1.8,
                brightness: 0.0, // KIFS disabled — boring kaleidoscope
            },
            // Showcase old + new variations — each transform is a specialist
            transforms: vec![
                FlameTransform { // sinusoidal tendrils
                    weight: 0.25, angle: 0.6, scale: 0.45,
                    offset: [2.5, 0.8], color: 0.0,
                    sinusoidal: 0.9, linear: 0.1,
                    ..Default::default()
                },
                FlameTransform { // spherical inversion orbs
                    weight: 0.15, angle: -1.3, scale: 0.30,
                    offset: [-1.8, 2.1], color: 0.25,
                    spherical: 0.9, swirl: 0.1,
                    ..Default::default()
                },
                FlameTransform { // julia bifurcation
                    weight: 0.15, angle: 2.2, scale: 0.55,
                    offset: [-0.3, -2.5], color: 0.50,
                    julia: 0.95,
                    ..Default::default()
                },
                FlameTransform { // bubble spheres
                    weight: 0.10, angle: -0.5, scale: 0.65,
                    offset: [1.2, -1.8], color: 0.60,
                    bubble: 0.85, fisheye: 0.15,
                    ..Default::default()
                },
                FlameTransform { // blob petals
                    weight: 0.08, angle: 1.1, scale: 0.50,
                    offset: [-2.0, 0.5], color: 0.40,
                    blob: 0.8, disc: 0.2,
                    ..Default::default()
                },
                FlameTransform { // spiral arms
                    weight: 0.10, angle: 0.0, scale: 0.70,
                    offset: [0.4, -0.2], color: 0.75,
                    spiral: 0.8, swirl: 0.2,
                    ..Default::default()
                },
                FlameTransform { // cosine curtains
                    weight: 0.08, angle: -0.7, scale: 0.35,
                    offset: [3.0, 1.5], color: 0.15,
                    cosine: 0.9, polar: 0.1,
                    ..Default::default()
                },
                FlameTransform { // curl smoke
                    weight: 0.09, angle: 1.5, scale: 0.25,
                    offset: [-2.8, -1.0], color: 0.90,
                    curl: 0.7, noise: 0.3,
                    ..Default::default()
                },
            ],
            final_transform: None,
            symmetry: 1,
        }
    }

    pub fn save(&self, dir: &Path) -> Result<PathBuf, String> {
        fs::create_dir_all(dir).map_err(|e| format!("create dir: {e}"))?;
        let filename = format!("{}.json", self.name);
        let path = dir.join(filename);
        let json =
            serde_json::to_string_pretty(self).map_err(|e| format!("serialize: {e}"))?;
        fs::write(&path, json).map_err(|e| format!("write: {e}"))?;
        Ok(path)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let json =
            fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_str(&json).map_err(|e| format!("parse {}: {e}", path.display()))
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
        let (sin_a, cos_a) = xf.angle.sin_cos();
        let rx = cos_a * p.0 - sin_a * p.1;
        let ry = sin_a * p.0 + cos_a * p.1;
        let ax = rx * xf.scale + xf.offset[0];
        let ay = ry * xf.scale + xf.offset[1];

        let mut vx = 0.0f32;
        let mut vy = 0.0f32;
        let r2 = ax * ax + ay * ay;
        let r_len = r2.sqrt().max(1e-6);
        let theta = ay.atan2(ax);

        if xf.linear > 0.0 { vx += ax * xf.linear; vy += ay * xf.linear; }
        if xf.sinusoidal > 0.0 { vx += ax.sin() * xf.sinusoidal; vy += ay.sin() * xf.sinusoidal; }
        if xf.spherical > 0.0 { let s = xf.spherical / r2.max(0.01); vx += ax * s; vy += ay * s; }
        if xf.swirl > 0.0 { let (sr, cr) = (r2.sin(), r2.cos()); vx += (ax * sr - ay * cr) * xf.swirl; vy += (ax * cr + ay * sr) * xf.swirl; }
        if xf.horseshoe > 0.0 { let h = xf.horseshoe / r_len; vx += (ax - ay) * (ax + ay) * h; vy += 2.0 * ax * ay * h; }
        if xf.handkerchief > 0.0 { vx += r_len * (theta + r_len).sin() * xf.handkerchief; vy += r_len * (theta - r_len).cos() * xf.handkerchief; }
        if xf.julia > 0.0 { let sr = r_len.sqrt(); let t2 = theta / 2.0; vx += sr * t2.cos() * xf.julia; vy += sr * t2.sin() * xf.julia; }
        if xf.polar > 0.0 { vx += (theta / std::f32::consts::PI) * xf.polar; vy += (r_len - 1.0) * xf.polar; }
        if xf.disc > 0.0 { let d = theta / std::f32::consts::PI; let pr = std::f32::consts::PI * r_len; vx += d * pr.sin() * xf.disc; vy += d * pr.cos() * xf.disc; }
        if xf.rings > 0.0 { let c2 = 0.04f32; let rr = ((r_len + c2) % (2.0 * c2)) - c2 + r_len * (1.0 - c2); vx += rr * theta.cos() * xf.rings; vy += rr * theta.sin() * xf.rings; }
        if xf.bubble > 0.0 { let b = 4.0 / (r2 + 4.0); vx += ax * b * xf.bubble; vy += ay * b * xf.bubble; }
        if xf.fisheye > 0.0 { let f = 2.0 / (r_len + 1.0); vx += f * ay * xf.fisheye; vy += f * ax * xf.fisheye; }
        if xf.diamond > 0.0 { vx += theta.sin() * r_len.cos() * xf.diamond; vy += theta.cos() * r_len.sin() * xf.diamond; }
        if xf.bent > 0.0 { let bx = if ax >= 0.0 { ax } else { 2.0 * ax }; let by = if ay >= 0.0 { ay } else { ay / 2.0 }; vx += bx * xf.bent; vy += by * xf.bent; }
        if xf.eyefish > 0.0 { let e = 2.0 / (r_len + 1.0); vx += e * ax * xf.eyefish; vy += e * ay * xf.eyefish; }
        if xf.cross > 0.0 { let c = 1.0 / (ax * ax - ay * ay).abs().max(0.01); vx += ax * c * xf.cross; vy += ay * c * xf.cross; }
        if xf.cosine > 0.0 { vx += ax.cos() * ay.cosh() * xf.cosine; vy -= ax.sin() * ay.sinh() * xf.cosine; }
        if xf.blob > 0.0 { let br = r_len * (0.5 + 0.5 * (3.0 * theta).sin()); vx += br * theta.cos() * xf.blob; vy += br * theta.sin() * xf.blob; }
        if xf.fan > 0.0 { let t2 = std::f32::consts::PI * 0.04; let t_half = t2 / 2.0; let f = if (theta + 0.2) % t2 > t_half { theta - t_half } else { theta + t_half }; vx += r_len * f.cos() * xf.fan; vy += r_len * f.sin() * xf.fan; }
        let others = xf.spiral + xf.exponential + xf.waves + xf.popcorn + xf.tangent + xf.noise + xf.curl;
        if others > 0.0 { vx += ax * others; vy += ay * others; }

        (vx.clamp(-100.0, 100.0), vy.clamp(-100.0, 100.0))
    }

    /// Estimate attractor extent via CPU chaos game (percentile-based, outlier-resistant).
    fn estimate_attractor_extent(&self) -> f32 {
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
                if r < cumsum { tidx = j; break; }
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

    pub fn mutate(&self, audio: &AudioFeatures, cfg: &crate::weights::RuntimeConfig) -> Self {
        // Seed-biased mutation: sometimes start from a random seed genome
        let mut rng = rand::rng();
        let base = if rng.random::<f32>() < cfg.seed_mutation_bias {
            let seeds_dir = crate::project_dir().join("genomes").join("seeds");
            match FlameGenome::load_random(&seeds_dir) {
                Ok(seed) => {
                    eprintln!("[mutate] starting from seed: {}", seed.name);
                    seed
                }
                Err(_) => self.clone(),
            }
        } else {
            self.clone()
        };

        let retries = cfg.max_mutation_retries.max(1);
        for attempt in 0..retries {
            let child = base.mutate_inner(audio);
            let extent = child.estimate_attractor_extent();
            if extent > cfg.min_attractor_extent {
                let mut result = child;
                result.global.zoom = result.auto_zoom(cfg);
                return result;
            }
            if attempt < retries - 1 {
                eprintln!("[mutate] attempt {} rejected (attractor extent={:.3}), retrying", attempt + 1, extent);
            }
        }
        eprintln!("[mutate] all attempts degenerate, keeping parent");
        let mut result = self.clone();
        result.name = format!("mutant-{}", rand::rng().random_range(1000..9999u32));
        result.global.zoom = result.auto_zoom(cfg);
        result
    }

    fn mutate_inner(&self, audio: &AudioFeatures) -> Self {
        let mut child = self.clone();
        let mut rng = rand::rng();

        // Single gentle mutation per evolve — keeps each generation similar
        match rng.random_range(0..8) {
            0 | 1 => child.mutate_perturb(&mut rng, audio),  // most common
            2 => child.mutate_swap_variations(&mut rng),
            3 => child.mutate_rotate_colors(&mut rng),
            4 => child.mutate_shuffle_transforms(&mut rng),
            5 => child.mutate_global_params(&mut rng),
            6 => child.mutate_final_transform(&mut rng, audio),
            _ => child.mutate_symmetry(&mut rng, audio),
        }

        // Add/remove transforms — rare, biased toward 6-16 sweet spot
        let n = child.transforms.len();
        let (add_chance, remove_chance) = if n < 6 {
            (0.20, 0.02)
        } else if n <= 16 {
            (0.08, 0.08)
        } else {
            (0.02, 0.20)
        };
        let energy_bias = audio.energy * 0.08;
        let add_chance = add_chance + energy_bias;
        let remove_chance = (remove_chance - energy_bias * 0.3).max(0.01);
        let roll: f32 = rng.random();
        if roll < add_chance {
            child.mutate_add_transform(&mut rng, audio);
        } else if roll < add_chance + remove_chance {
            child.mutate_remove_transform(&mut rng);
        }

        child.name = format!("mutant-{}", rng.random_range(1000..9999u32));
        child
    }

    fn mutate_perturb(&mut self, rng: &mut impl Rng, audio: &AudioFeatures) {
        if self.transforms.is_empty() { return; }
        let idx = rng.random_range(0..self.transforms.len());
        let xf = &mut self.transforms[idx];
        match rng.random_range(0..5) {
            0 => xf.angle += rng.random_range(-0.3..0.3),
            1 => xf.scale = (xf.scale + rng.random_range(-0.15..0.15)).clamp(0.05, 0.95),
            2 => {
                xf.offset[0] += rng.random_range(-0.5..0.5);
                xf.offset[1] += rng.random_range(-0.5..0.5);
            }
            3 => {
                // Gentle weight adjustment
                xf.weight = (xf.weight * rng.random_range(0.7..1.5)).clamp(0.01, 0.8);
            }
            _ => {
                // Nudge existing variations instead of reinventing
                // Pick an audio-biased variation and boost it slightly
                let vi = audio_biased_variation_pick(rng, audio);
                let cur = xf.get_variation(vi);
                xf.set_variation(vi, (cur + rng.random_range(0.05..0.25)).min(1.0));
                // Slightly reduce a random other variation to compensate
                let other = rng.random_range(0..VARIATION_COUNT);
                if other != vi {
                    let ocur = xf.get_variation(other);
                    xf.set_variation(other, (ocur - rng.random_range(0.0..0.15)).max(0.0));
                }
            }
        }
    }

    fn mutate_swap_variations(&mut self, rng: &mut impl Rng) {
        if self.transforms.is_empty() { return; }
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
        self.global.flame_brightness = (self.global.flame_brightness + rng.random_range(-0.1..0.1)).clamp(0.1, 1.0);
        self.global.zoom = (self.global.zoom + rng.random_range(-0.5..0.5)).clamp(1.5, 6.0);
    }

    fn mutate_final_transform(&mut self, rng: &mut impl Rng, audio: &AudioFeatures) {
        match &mut self.final_transform {
            Some(fxf) => {
                // Perturb existing final transform
                fxf.angle += rng.random_range(-0.5..0.5);
                fxf.scale = (fxf.scale + rng.random_range(-0.2..0.2)).clamp(0.3, 2.0);
                // Occasionally reinvent its variations
                if rng.random_range(0.0..1.0) < 0.3 {
                    for vi in 0..VARIATION_COUNT {
                        fxf.set_variation(vi, 0.0);
                    }
                    let dominant = audio_biased_variation_pick(rng, audio);
                    fxf.set_variation(dominant, rng.random_range(0.5..1.0));
                }
            }
            None => {
                // 30% chance to create a final transform
                if rng.random_range(0.0..1.0) < 0.3 {
                    let mut fxf = FlameTransform::default();
                    fxf.scale = rng.random_range(0.5..1.5);
                    fxf.angle = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
                    let dominant = audio_biased_variation_pick(rng, audio);
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
        let delta = if rng.random_range(0.0..1.0) < 0.5 { 1 } else { -1 };
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

    fn mutate_add_transform(&mut self, rng: &mut impl Rng, audio: &AudioFeatures) {
        if self.transforms.is_empty() { return; }
        // 50% clone-and-perturb, 50% fresh specialist
        let new_xf = if rng.random_range(0.0..1.0) < 0.5 {
            let source_idx = rng.random_range(0..self.transforms.len());
            let mut xf = self.transforms[source_idx].clone();
            xf.weight = 0.05;
            xf.angle += rng.random_range(-0.8..0.8);
            xf.scale = (xf.scale + rng.random_range(-0.3..0.3)).clamp(0.05, 0.95);
            xf.offset[0] += rng.random_range(-2.0..2.0);
            xf.offset[1] += rng.random_range(-2.0..2.0);
            xf.color = rng.random_range(0.0..1.0);
            xf
        } else {
            // Fresh specialist — audio-biased variation pick
            let mut xf = FlameTransform {
                weight: 0.05,
                angle: rng.random_range(-std::f32::consts::PI..std::f32::consts::PI),
                scale: rng.random_range(0.2..0.8),
                offset: [rng.random_range(-3.0..3.0), rng.random_range(-3.0..3.0)],
                color: rng.random_range(0.0..1.0),
                ..Default::default()
            };
            let dominant = audio_biased_variation_pick(rng, audio);
            xf.set_variation(dominant, rng.random_range(0.7..1.0));
            xf
        };
        self.transforms.push(new_xf);
    }

    fn mutate_remove_transform(&mut self, _rng: &mut impl Rng) {
        if self.transforms.len() <= 2 { return; } // keep at least 2
        // Remove the lowest-weight transform
        if let Some((min_idx, _)) = self.transforms.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.weight.partial_cmp(&b.weight).unwrap())
        {
            self.transforms.remove(min_idx);
        }
    }
}
