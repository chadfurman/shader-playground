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
                zoom: 3.0,
                trail: 0.34,
                flame_brightness: 0.4,
            },
            kifs: KifsParams {
                fold_angle: 0.62,
                scale: 1.8,
                brightness: 0.0, // KIFS disabled — boring kaleidoscope
            },
            // Electric Sheep-inspired: each transform is a specialist
            // with one dominant variation, varied weights, and wide offsets
            transforms: vec![
                FlameTransform { // dominant sinusoidal — wispy tendrils
                    weight: 0.35,
                    angle: 0.6,
                    scale: 0.45,
                    offset: [2.5, 0.8],
                    color: 0.0,
                    linear: 0.1,
                    sinusoidal: 0.9,
                    spherical: 0.0,
                    swirl: 0.0,
                    horseshoe: 0.0,
                    handkerchief: 0.0,
                },
                FlameTransform { // dominant spherical — inversion, pulls inward
                    weight: 0.25,
                    angle: -1.3,
                    scale: 0.30,
                    offset: [-1.8, 2.1],
                    color: 0.25,
                    linear: 0.0,
                    sinusoidal: 0.0,
                    spherical: 0.95,
                    swirl: 0.05,
                    horseshoe: 0.0,
                    handkerchief: 0.0,
                },
                FlameTransform { // dominant swirl — spiral arms
                    weight: 0.15,
                    angle: 2.2,
                    scale: 0.70,
                    offset: [-0.3, -2.5],
                    color: 0.50,
                    linear: 0.0,
                    sinusoidal: 0.0,
                    spherical: 0.0,
                    swirl: 1.0,
                    horseshoe: 0.0,
                    handkerchief: 0.0,
                },
                FlameTransform { // dominant linear — contractive backbone
                    weight: 0.10,
                    angle: 0.0,
                    scale: 0.85,
                    offset: [0.4, -0.2],
                    color: 0.75,
                    linear: 1.0,
                    sinusoidal: 0.0,
                    spherical: 0.0,
                    swirl: 0.0,
                    horseshoe: 0.0,
                    handkerchief: 0.0,
                },
                FlameTransform { // dominant handkerchief — chaotic detail
                    weight: 0.05,
                    angle: -0.7,
                    scale: 0.20,
                    offset: [3.0, 1.5],
                    color: 0.15,
                    linear: 0.0,
                    sinusoidal: 0.0,
                    spherical: 0.0,
                    swirl: 0.0,
                    horseshoe: 0.0,
                    handkerchief: 1.0,
                },
                FlameTransform { // dominant horseshoe — distant structure
                    weight: 0.10,
                    angle: 1.5,
                    scale: 0.15,
                    offset: [-2.8, -1.0],
                    color: 0.90,
                    linear: 0.0,
                    sinusoidal: 0.0,
                    spherical: 0.0,
                    swirl: 0.0,
                    horseshoe: 1.0,
                    handkerchief: 0.0,
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
                // Reinvent as specialist — one dominant variation
                let vars = [0.0f32; 6];
                let dominant = rng.random_range(0..6);
                let mut v = vars;
                v[dominant] = rng.random_range(0.7..1.0);
                // Maybe a small secondary
                let secondary = rng.random_range(0..6);
                if secondary != dominant {
                    v[secondary] = rng.random_range(0.0..0.2);
                }
                xf.linear = v[0];
                xf.sinusoidal = v[1];
                xf.spherical = v[2];
                xf.swirl = v[3];
                xf.horseshoe = v[4];
                xf.handkerchief = v[5];
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
            // Fresh specialist with one dominant variation
            let mut vars = [0.0f32; 6];
            let dominant = rng.random_range(0..6);
            vars[dominant] = rng.random_range(0.7..1.0);
            FlameTransform {
                weight: 0.05,
                angle: rng.random_range(-std::f32::consts::PI..std::f32::consts::PI),
                scale: rng.random_range(0.1..0.85),
                offset: [rng.random_range(-3.0..3.0), rng.random_range(-3.0..3.0)],
                color: rng.random_range(0.0..1.0),
                linear: vars[0],
                sinusoidal: vars[1],
                spherical: vars[2],
                swirl: vars[3],
                horseshoe: vars[4],
                handkerchief: vars[5],
            }
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
