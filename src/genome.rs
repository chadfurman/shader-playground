use std::fs;
use std::path::{Path, PathBuf};

use rand::Rng;
use serde::{Deserialize, Serialize};

const VARIATION_COUNT: usize = 26;

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

impl FlameGenome {
    /// Pack global params into a fixed [f32; 12] for the uniform buffer.
    /// Layout:
    ///   [0] speed  [1] zoom  [2] trail  [3] flame_brightness
    ///   [4] kifs_fold  [5] kifs_scale  [6] kifs_brightness  [7] drift_speed
    ///   [8] color_shift  [9..11] reserved
    pub fn flatten_globals(&self) -> [f32; 12] {
        let mut g = [0.0f32; 12];
        g[0] = self.global.speed;
        g[1] = self.global.zoom;
        g[2] = self.global.trail;
        g[3] = self.global.flame_brightness;
        g[4] = self.kifs.fold_angle;
        g[5] = self.kifs.scale;
        g[6] = self.kifs.brightness;
        // g[7] = drift_speed (set by weights, default 0)
        // g[8] = color_shift (set by weights, default 0)
        g[9] = 0.7;  // vibrancy base
        g[10] = 0.15;  // bloom_intensity base
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
                    weight: 0.30, angle: 0.6, scale: 0.45,
                    offset: [2.5, 0.8], color: 0.0,
                    sinusoidal: 0.9, linear: 0.1,
                    ..Default::default()
                },
                FlameTransform { // spherical inversion
                    weight: 0.20, angle: -1.3, scale: 0.30,
                    offset: [-1.8, 2.1], color: 0.25,
                    spherical: 0.9, swirl: 0.1,
                    ..Default::default()
                },
                FlameTransform { // julia blobs
                    weight: 0.20, angle: 2.2, scale: 0.55,
                    offset: [-0.3, -2.5], color: 0.50,
                    julia: 0.85, bubble: 0.15,
                    ..Default::default()
                },
                FlameTransform { // spiral arms
                    weight: 0.10, angle: 0.0, scale: 0.70,
                    offset: [0.4, -0.2], color: 0.75,
                    spiral: 0.8, swirl: 0.2,
                    ..Default::default()
                },
                FlameTransform { // cosine curtains
                    weight: 0.10, angle: -0.7, scale: 0.35,
                    offset: [3.0, 1.5], color: 0.15,
                    cosine: 0.9, polar: 0.1,
                    ..Default::default()
                },
                FlameTransform { // curl smoke
                    weight: 0.10, angle: 1.5, scale: 0.25,
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

    pub fn mutate(&self) -> Self {
        let mut child = self.clone();
        let mut rng = rand::rng();
        let num_mutations = rng.random_range(1..=3);

        for _ in 0..num_mutations {
            match rng.random_range(0..7) {
                0 => child.mutate_perturb(&mut rng),
                1 => child.mutate_swap_variations(&mut rng),
                2 => child.mutate_rotate_colors(&mut rng),
                3 => child.mutate_shuffle_transforms(&mut rng),
                4 => child.mutate_global_params(&mut rng),
                5 => child.mutate_final_transform(&mut rng),
                _ => child.mutate_symmetry(&mut rng),
            }
        }

        // Add/remove transforms biased toward 6-16 sweet spot
        let n = child.transforms.len();
        let (add_chance, remove_chance) = if n < 6 {
            (0.40, 0.05)
        } else if n <= 16 {
            (0.15, 0.15)
        } else {
            (0.05, 0.40)
        };
        let roll: f32 = rng.random();
        if roll < add_chance {
            child.mutate_add_transform(&mut rng);
        } else if roll < add_chance + remove_chance {
            child.mutate_remove_transform(&mut rng);
        }

        child.name = format!("mutant-{}", rng.random_range(1000..9999u32));
        child
    }

    fn mutate_perturb(&mut self, rng: &mut impl Rng) {
        if self.transforms.is_empty() { return; }
        let idx = rng.random_range(0..self.transforms.len());
        let xf = &mut self.transforms[idx];
        match rng.random_range(0..5) {
            0 => xf.angle += rng.random_range(-0.8..0.8),
            1 => xf.scale = (xf.scale + rng.random_range(-0.3..0.3)).clamp(0.05, 0.95),
            2 => {
                // Wider offset jumps for spatial spread
                xf.offset[0] += rng.random_range(-1.5..1.5);
                xf.offset[1] += rng.random_range(-1.5..1.5);
            }
            3 => {
                // Weight redistribution — make some dominant, some subtle
                xf.weight = (xf.weight * rng.random_range(0.3..3.0)).clamp(0.01, 0.8);
            }
            _ => {
                // Reinvent as specialist — one dominant variation from all 26
                for vi in 0..VARIATION_COUNT {
                    xf.set_variation(vi, 0.0);
                }
                let dominant = rng.random_range(0..VARIATION_COUNT);
                xf.set_variation(dominant, rng.random_range(0.7..1.0));
                let secondary = rng.random_range(0..VARIATION_COUNT);
                if secondary != dominant {
                    xf.set_variation(secondary, rng.random_range(0.0..0.2));
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

    fn mutate_final_transform(&mut self, rng: &mut impl Rng) {
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
                    let dominant = rng.random_range(0..VARIATION_COUNT);
                    fxf.set_variation(dominant, rng.random_range(0.5..1.0));
                }
            }
            None => {
                // 30% chance to create a final transform
                if rng.random_range(0.0..1.0) < 0.3 {
                    let mut fxf = FlameTransform::default();
                    fxf.scale = rng.random_range(0.5..1.5);
                    fxf.angle = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
                    let dominant = rng.random_range(0..VARIATION_COUNT);
                    fxf.set_variation(dominant, rng.random_range(0.5..1.0));
                    fxf.color = rng.random_range(0.0..1.0);
                    self.final_transform = Some(fxf);
                }
            }
        }
    }

    fn mutate_symmetry(&mut self, rng: &mut impl Rng) {
        let roll: f32 = rng.random();
        if roll < 0.3 {
            self.symmetry = 1; // no symmetry
        } else if roll < 0.7 {
            self.symmetry = rng.random_range(2..=6); // rotational
        } else {
            self.symmetry = -rng.random_range(2..=4); // bilateral + rotational
        }
    }

    fn mutate_add_transform(&mut self, rng: &mut impl Rng) {
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
            // Fresh specialist with one dominant variation from all 26
            let mut xf = FlameTransform {
                weight: 0.05,
                angle: rng.random_range(-std::f32::consts::PI..std::f32::consts::PI),
                scale: rng.random_range(0.2..0.8),
                offset: [rng.random_range(-3.0..3.0), rng.random_range(-3.0..3.0)],
                color: rng.random_range(0.0..1.0),
                ..Default::default()
            };
            let dominant = rng.random_range(0..VARIATION_COUNT);
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
