use std::fs;
use std::path::{Path, PathBuf};

use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FlameTransform {
    pub weight: f32,
    pub angle: f32,
    pub scale: f32,
    pub offset: [f32; 2],
    pub color: f32,
    pub linear: f32,
    pub sinusoidal: f32,
    pub spherical: f32,
    pub swirl: f32,
    pub horseshoe: f32,
    pub handkerchief: f32,
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
}

impl FlameGenome {
    /// Flatten to [f32; 64] for the uniform buffer.
    /// Layout:
    ///   [0..=3]  global: speed, zoom, trail, flame_brightness
    ///   [4..=6]  kifs: fold_angle, scale, brightness
    ///   [7]      reserved (0.0)
    ///   [8..=19]  transform 0 (12 floats)
    ///   [20..=31] transform 1
    ///   [32..=43] transform 2
    ///   [44..=55] transform 3
    pub fn flatten(&self) -> [f32; 64] {
        let mut p = [0.0f32; 64];

        // Global
        p[0] = self.global.speed;
        p[1] = self.global.zoom;
        p[2] = self.global.trail;
        p[3] = self.global.flame_brightness;

        // KIFS
        p[4] = self.kifs.fold_angle;
        p[5] = self.kifs.scale;
        p[6] = self.kifs.brightness;

        // Transforms (up to 4)
        for (i, xf) in self.transforms.iter().enumerate().take(4) {
            let base = 8 + i * 12;
            p[base] = xf.weight;
            p[base + 1] = xf.angle;
            p[base + 2] = xf.scale;
            p[base + 3] = xf.offset[0];
            p[base + 4] = xf.offset[1];
            p[base + 5] = xf.color;
            p[base + 6] = xf.linear;
            p[base + 7] = xf.sinusoidal;
            p[base + 8] = xf.spherical;
            p[base + 9] = xf.swirl;
            p[base + 10] = xf.horseshoe;
            p[base + 11] = xf.handkerchief;
        }

        p
    }

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
        g
    }

    /// Pack all transforms into a flat Vec<f32> for the storage buffer.
    /// Each transform = 12 floats: weight, angle, scale, offset_x, offset_y,
    /// color, linear, sinusoidal, spherical, swirl, horseshoe, handkerchief.
    pub fn flatten_transforms(&self) -> Vec<f32> {
        let mut t = Vec::with_capacity(self.transforms.len() * 12);
        for xf in &self.transforms {
            t.push(xf.weight);
            t.push(xf.angle);
            t.push(xf.scale);
            t.push(xf.offset[0]);
            t.push(xf.offset[1]);
            t.push(xf.color);
            t.push(xf.linear);
            t.push(xf.sinusoidal);
            t.push(xf.spherical);
            t.push(xf.swirl);
            t.push(xf.horseshoe);
            t.push(xf.handkerchief);
        }
        t
    }

    pub fn transform_count(&self) -> u32 {
        self.transforms.len() as u32
    }

    /// Create the default genome matching current hardcoded transforms.
    pub fn default_genome() -> Self {
        Self {
            name: "default".into(),
            global: GlobalParams {
                speed: 0.25,
                zoom: 2.0,
                trail: 0.34,
                flame_brightness: 0.4,
            },
            kifs: KifsParams {
                fold_angle: 0.62,
                scale: 1.8,
                brightness: 0.3,
            },
            transforms: vec![
                FlameTransform {
                    weight: 0.20,
                    angle: 0.6,
                    scale: 0.65,
                    offset: [1.0, 0.4],
                    color: 0.0,
                    linear: 0.0,
                    sinusoidal: 0.5,
                    spherical: 0.0,
                    swirl: 0.5,
                    horseshoe: 0.0,
                    handkerchief: 0.0,
                },
                FlameTransform {
                    weight: 0.20,
                    angle: -1.0,
                    scale: 0.70,
                    offset: [-0.8, 0.9],
                    color: 0.33,
                    linear: 0.0,
                    sinusoidal: 0.0,
                    spherical: 0.6,
                    swirl: 0.0,
                    horseshoe: 0.4,
                    handkerchief: 0.0,
                },
                FlameTransform {
                    weight: 0.20,
                    angle: 1.7,
                    scale: 0.60,
                    offset: [-0.5, -1.0],
                    color: 0.67,
                    linear: 0.0,
                    sinusoidal: 0.3,
                    spherical: 0.0,
                    swirl: 0.0,
                    horseshoe: 0.0,
                    handkerchief: 0.7,
                },
                FlameTransform {
                    weight: 0.20,
                    angle: 0.0,
                    scale: 0.82,
                    offset: [0.6, -0.7],
                    color: 0.85,
                    linear: 0.6,
                    sinusoidal: 0.0,
                    spherical: 0.0,
                    swirl: 0.0,
                    horseshoe: 0.0,
                    handkerchief: 0.0,
                },
                FlameTransform {
                    weight: 0.20,
                    angle: -0.5,
                    scale: 0.55,
                    offset: [0.3, 1.2],
                    color: 0.15,
                    linear: 0.3,
                    sinusoidal: 0.0,
                    spherical: 0.4,
                    swirl: 0.0,
                    horseshoe: 0.3,
                    handkerchief: 0.0,
                },
                FlameTransform {
                    weight: 0.20,
                    angle: 2.1,
                    scale: 0.72,
                    offset: [-1.1, 0.2],
                    color: 0.50,
                    linear: 0.0,
                    sinusoidal: 0.2,
                    spherical: 0.0,
                    swirl: 0.6,
                    horseshoe: 0.0,
                    handkerchief: 0.2,
                },
            ],
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
            match rng.random_range(0..5) {
                0 => child.mutate_perturb(&mut rng),
                1 => child.mutate_swap_variations(&mut rng),
                2 => child.mutate_rotate_colors(&mut rng),
                3 => child.mutate_shuffle_transforms(&mut rng),
                _ => child.mutate_kifs_drift(&mut rng),
            }
        }

        child.name = format!("mutant-{}", rng.random_range(1000..9999u32));
        child
    }

    fn mutate_perturb(&mut self, rng: &mut impl Rng) {
        if self.transforms.is_empty() { return; }
        let idx = rng.random_range(0..self.transforms.len());
        let xf = &mut self.transforms[idx];
        match rng.random_range(0..3) {
            0 => xf.angle += rng.random_range(-0.4..0.4),
            1 => xf.scale = (xf.scale + rng.random_range(-0.15..0.15)).clamp(0.3, 0.95),
            _ => {
                xf.offset[0] += rng.random_range(-0.4..0.4);
                xf.offset[1] += rng.random_range(-0.4..0.4);
            }
        }
    }

    fn mutate_swap_variations(&mut self, rng: &mut impl Rng) {
        if self.transforms.is_empty() { return; }
        let idx = rng.random_range(0..self.transforms.len());
        let xf = &mut self.transforms[idx];
        let mut vars = [
            xf.linear,
            xf.sinusoidal,
            xf.spherical,
            xf.swirl,
            xf.horseshoe,
            xf.handkerchief,
        ];
        let a = rng.random_range(0..6);
        let b = rng.random_range(0..6);
        vars.swap(a, b);
        xf.linear = vars[0];
        xf.sinusoidal = vars[1];
        xf.spherical = vars[2];
        xf.swirl = vars[3];
        xf.horseshoe = vars[4];
        xf.handkerchief = vars[5];
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

    fn mutate_kifs_drift(&mut self, rng: &mut impl Rng) {
        self.kifs.fold_angle += rng.random_range(-0.1..0.1);
        self.kifs.scale = (self.kifs.scale + rng.random_range(-0.15..0.15)).clamp(1.3, 2.5);
    }
}
