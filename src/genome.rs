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
    ///   [0..3]   global: speed, zoom, trail, flame_brightness
    ///   [4..6]   kifs: fold_angle, scale, brightness
    ///   [7]      reserved (0.0)
    ///   [8..19]  transform 0 (12 floats)
    ///   [20..31] transform 1
    ///   [32..43] transform 2
    ///   [44..55] transform 3
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
                    weight: 0.25,
                    angle: 0.6,
                    scale: 0.65,
                    offset: [0.4, 0.15],
                    color: 0.0,
                    linear: 0.0,
                    sinusoidal: 0.5,
                    spherical: 0.0,
                    swirl: 0.5,
                    horseshoe: 0.0,
                    handkerchief: 0.0,
                },
                FlameTransform {
                    weight: 0.25,
                    angle: -1.0,
                    scale: 0.70,
                    offset: [-0.3, 0.3],
                    color: 0.33,
                    linear: 0.0,
                    sinusoidal: 0.0,
                    spherical: 0.6,
                    swirl: 0.0,
                    horseshoe: 0.4,
                    handkerchief: 0.0,
                },
                FlameTransform {
                    weight: 0.25,
                    angle: 1.7,
                    scale: 0.60,
                    offset: [-0.1, -0.35],
                    color: 0.67,
                    linear: 0.0,
                    sinusoidal: 0.3,
                    spherical: 0.0,
                    swirl: 0.0,
                    horseshoe: 0.0,
                    handkerchief: 0.7,
                },
                FlameTransform {
                    weight: 0.25,
                    angle: 0.0,
                    scale: 0.82,
                    offset: [0.0, 0.0],
                    color: 0.85,
                    linear: 0.6,
                    sinusoidal: 0.0,
                    spherical: 0.0,
                    swirl: 0.0,
                    horseshoe: 0.0,
                    handkerchief: 0.0,
                },
            ],
        }
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
        let idx = rng.random_range(0..self.transforms.len());
        let xf = &mut self.transforms[idx];
        match rng.random_range(0..3) {
            0 => xf.angle += rng.random_range(-0.4..0.4),
            1 => xf.scale = (xf.scale + rng.random_range(-0.15..0.15)).clamp(0.3, 0.95),
            _ => {
                xf.offset[0] += rng.random_range(-0.2..0.2);
                xf.offset[1] += rng.random_range(-0.2..0.2);
            }
        }
    }

    fn mutate_swap_variations(&mut self, rng: &mut impl Rng) {
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
