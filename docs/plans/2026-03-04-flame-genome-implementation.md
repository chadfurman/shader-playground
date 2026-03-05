# Flame Genome & Evolution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace hardcoded flame transforms with a data-driven genome that supports mutation, crossfade, save/load, and keyboard control.

**Architecture:** A `FlameGenome` struct maps named fields to a flat `[f32; 64]` param array. The compute shader reads transform coefficients from uniforms instead of hardcoded constants. Mutation generates new genomes, the existing exponential lerp crossfades between them.

**Tech Stack:** Rust, wgpu 28, serde/serde_json, WGSL compute+fragment shaders, rand crate for mutation

---

### Task 1: Add rand dependency

**Files:**
- Modify: `Cargo.toml`

**Step 1: Add rand crate**

Add to `[dependencies]` in `Cargo.toml`:
```toml
rand = "0.9"
```

**Step 2: Verify it compiles**

Run: `cargo build`
Expected: Compiles with no errors

**Step 3: Commit**

```bash
git add Cargo.toml
git commit -m "chore: add rand dependency for genome mutation"
```

---

### Task 2: Create FlameGenome struct with serde + flatten

**Files:**
- Create: `src/genome.rs`
- Modify: `src/main.rs` (add `mod genome;`)

**Step 1: Write genome.rs with struct and flatten**

```rust
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
            p[base]     = xf.weight;
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
                    weight: 0.25, angle: 0.6, scale: 0.65,
                    offset: [0.4, 0.15], color: 0.0,
                    linear: 0.0, sinusoidal: 0.5, spherical: 0.0,
                    swirl: 0.5, horseshoe: 0.0, handkerchief: 0.0,
                },
                FlameTransform {
                    weight: 0.25, angle: -1.0, scale: 0.70,
                    offset: [-0.3, 0.3], color: 0.33,
                    linear: 0.0, sinusoidal: 0.0, spherical: 0.6,
                    swirl: 0.0, horseshoe: 0.4, handkerchief: 0.0,
                },
                FlameTransform {
                    weight: 0.25, angle: 1.7, scale: 0.60,
                    offset: [-0.1, -0.35], color: 0.67,
                    linear: 0.0, sinusoidal: 0.3, spherical: 0.0,
                    swirl: 0.0, horseshoe: 0.0, handkerchief: 0.7,
                },
                FlameTransform {
                    weight: 0.25, angle: 0.0, scale: 0.82,
                    offset: [0.0, 0.0], color: 0.85,
                    linear: 0.6, sinusoidal: 0.0, spherical: 0.0,
                    swirl: 0.0, horseshoe: 0.0, handkerchief: 0.0,
                },
            ],
        }
    }
}
```

**Step 2: Add mod declaration in main.rs**

Add `mod genome;` near the top of `src/main.rs` (after the `use` statements, before the Uniforms struct).

**Step 3: Verify it compiles**

Run: `cargo build`
Expected: Compiles with no errors

**Step 4: Commit**

```bash
git add src/genome.rs src/main.rs
git commit -m "feat: add FlameGenome struct with serde and flatten"
```

---

### Task 3: Add mutation logic to genome.rs

**Files:**
- Modify: `src/genome.rs`

**Step 1: Add mutation methods**

Add these methods to the `impl FlameGenome` block:

```rust
use rand::Rng;

impl FlameGenome {
    // ... existing methods ...

    /// Mutate this genome, returning a new child genome.
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
            xf.linear, xf.sinusoidal, xf.spherical,
            xf.swirl, xf.horseshoe, xf.handkerchief,
        ];
        // Shuffle weights
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
        if self.transforms.len() < 2 { return; }
        let a = rng.random_range(0..self.transforms.len());
        let b = rng.random_range(0..self.transforms.len());
        self.transforms.swap(a, b);
    }

    fn mutate_kifs_drift(&mut self, rng: &mut impl Rng) {
        self.kifs.fold_angle += rng.random_range(-0.1..0.1);
        self.kifs.scale = (self.kifs.scale + rng.random_range(-0.15..0.15)).clamp(1.3, 2.5);
    }
}
```

**Step 2: Verify it compiles**

Run: `cargo build`
Expected: Compiles

**Step 3: Commit**

```bash
git add src/genome.rs
git commit -m "feat: add genome mutation (perturb, swap, shuffle, drift)"
```

---

### Task 4: Add genome save/load

**Files:**
- Modify: `src/genome.rs`

**Step 1: Add save/load functions**

```rust
use std::path::{Path, PathBuf};
use std::fs;

impl FlameGenome {
    // ... existing methods ...

    pub fn save(&self, dir: &Path) -> Result<PathBuf, String> {
        fs::create_dir_all(dir).map_err(|e| format!("create dir: {e}"))?;
        let filename = format!("{}.json", self.name);
        let path = dir.join(filename);
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("serialize: {e}"))?;
        fs::write(&path, json).map_err(|e| format!("write: {e}"))?;
        Ok(path)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let json = fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_str(&json)
            .map_err(|e| format!("parse {}: {e}", path.display()))
    }

    pub fn load_random(dir: &Path) -> Result<Self, String> {
        use rand::seq::SliceRandom;
        let entries: Vec<_> = fs::read_dir(dir)
            .map_err(|e| format!("read dir: {e}"))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "json"))
            .collect();
        if entries.is_empty() {
            return Err("no genomes found".into());
        }
        let entry = entries.choose(&mut rand::rng())
            .ok_or("empty")?;
        Self::load(&entry.path())
    }
}
```

**Step 2: Verify it compiles**

Run: `cargo build`
Expected: Compiles

**Step 3: Commit**

```bash
git add src/genome.rs
git commit -m "feat: genome save/load/load_random"
```

---

### Task 5: Expand uniform buffer to 64 floats

**Files:**
- Modify: `src/main.rs` — Uniforms struct, param packing
- Modify: `flame_compute.wgsl` — Uniforms struct
- Modify: `playground.wgsl` — Uniforms struct

**Step 1: Update Rust Uniforms struct**

In `src/main.rs`, change the Uniforms struct (currently line ~17):
```rust
struct Uniforms {
    time: f32,
    frame: u32,
    resolution: [f32; 2],
    mouse: [f32; 2],
    _pad: [f32; 2],
    params: [[f32; 4]; 16], // 64 floats (was 4 vec4s, now 16)
}
```

**Step 2: Update App struct param arrays**

Change `params` and `target_params` from `[f32; 16]` to `[f32; 64]` in the App struct. Update `load_params` to return `[f32; 64]`. Update the param packing in `RedrawRequested` to pack all 16 vec4s:

```rust
let mut flat_params = [[0.0f32; 4]; 16];
for i in 0..16 {
    for j in 0..4 {
        flat_params[i][j] = self.params[i * 4 + j];
    }
}
```

**Step 3: Update WGSL Uniforms in both shaders**

In both `flame_compute.wgsl` and `playground.wgsl`, change:
```wgsl
params: array<vec4<f32>, 4>,
```
to:
```wgsl
params: array<vec4<f32>, 16>,
```

The `param()` function stays the same — it already indexes correctly.

**Step 4: Verify it compiles and runs**

Run: `cargo build && cargo run` (check window still renders)

**Step 5: Commit**

```bash
git add src/main.rs flame_compute.wgsl playground.wgsl
git commit -m "feat: expand uniform buffer to 64 floats (16 vec4s)"
```

---

### Task 6: Integrate genome into App — replace params.json with genome

**Files:**
- Modify: `src/main.rs`

**Step 1: Add genome fields to App struct**

```rust
use crate::genome::FlameGenome;

struct App {
    // ... existing fields ...
    params: [f32; 64],
    target_params: [f32; 64],
    genome: FlameGenome,
    genome_history: Vec<[f32; 64]>,
    morph_rate: f32,  // controls lerp speed (default 5.0)
}
```

**Step 2: Update App::new()**

```rust
fn new() -> Self {
    let genome = FlameGenome::default_genome();
    let initial = genome.flatten();
    Self {
        gpu: None,
        window: None,
        watcher: None,
        start: Instant::now(),
        frame: 0,
        mouse: [0.5, 0.5],
        params: initial,
        target_params: initial,
        last_frame_time: Instant::now(),
        genome,
        genome_history: Vec::new(),
        morph_rate: 5.0,
    }
}
```

**Step 3: On startup, try loading a random genome from genomes/**

In `resumed()`, after watcher setup:
```rust
let genomes_dir = project_dir().join("genomes");
if genomes_dir.exists() {
    if let Ok(g) = FlameGenome::load_random(&genomes_dir) {
        eprintln!("[genome] loaded: {}", g.name);
        self.genome = g;
        self.target_params = self.genome.flatten();
    }
}
```

**Step 4: Remove params.json loading** (genome replaces it)

Remove `load_params()` usage from `App::new()`. Keep the file watcher for params.json as a fallback but genome takes priority.

Actually — keep params.json loading as an override. If params.json exists and changes, it overrides the first 16 floats. This preserves backward compatibility and lets you still tweak global/kifs params by hand.

**Step 5: Verify it compiles and runs**

Run: `cargo build && cargo run`

**Step 6: Commit**

```bash
git add src/main.rs
git commit -m "feat: integrate FlameGenome into App, load on startup"
```

---

### Task 7: Make compute shader param-driven

**Files:**
- Modify: `flame_compute.wgsl`

**Step 1: Replace hardcoded transforms with param reads**

Replace the `apply_xform` function. The new version reads all coefficients from params:

```wgsl
fn apply_xform(p: vec2<f32>, idx: i32, t: f32) -> vec2<f32> {
    let base = 8 + idx * 12;
    let weight = param(base);           // not used here, used for selection
    let angle  = param(base + 1);
    let scale  = param(base + 2);
    let ox     = param(base + 3);
    let oy     = param(base + 4);
    // color is param(base + 5), used elsewhere
    let w_lin  = param(base + 6);
    let w_sin  = param(base + 7);
    let w_sph  = param(base + 8);
    let w_swi  = param(base + 9);
    let w_hor  = param(base + 10);
    let w_han  = param(base + 11);

    // Affine transform (angle drifts slowly with time)
    let q = rot2(angle + t * 0.07 * f32(idx + 1)) * p * scale
          + vec2(ox + 0.05 * sin(t * 0.3 * f32(idx + 1)),
                 oy + 0.05 * cos(t * 0.4 * f32(idx + 1)));

    // Weighted sum of variations
    var v = q * w_lin;
    v += V_sinusoidal(q)    * w_sin;
    v += V_spherical(q)     * w_sph;
    v += V_swirl(q)         * w_swi;
    v += V_horseshoe(q)     * w_hor;
    v += V_handkerchief(q)  * w_han;

    return v;
}
```

**Step 2: Update transform selection to use weights**

In the main loop, replace the equal-probability selection with weight-based:
```wgsl
let r = randf(&rng);
let w0 = param(8);            // transform 0 weight
let w1 = param(20);           // transform 1 weight
let w2 = param(32);           // transform 2 weight
let w3 = param(44);           // transform 3 weight
let total = w0 + w1 + w2 + w3;
let rn = r * total;
var tidx = 0;
if (rn > w0) { tidx = 1; }
if (rn > w0 + w1) { tidx = 2; }
if (rn > w0 + w1 + w2) { tidx = 3; }
```

**Step 3: Update color reading**

Replace `xform_color` function:
```wgsl
fn xform_color(idx: i32) -> f32 {
    return param(8 + idx * 12 + 5);
}
```

**Step 4: Remove old hardcoded switch statements**

Delete the old `apply_xform` switch and `xform_color` switch.

**Step 5: Verify it compiles and renders**

Run: `cargo run` — should look the same as before since default_genome() matches the old hardcoded values.

**Step 6: Commit**

```bash
git add flame_compute.wgsl
git commit -m "feat: param-driven flame transforms (no more hardcoded values)"
```

---

### Task 8: Update fragment shader param indices

**Files:**
- Modify: `playground.wgsl`

**Step 1: Update param reads to match new layout**

The fragment shader currently reads params at indices 0, 3-7. With the new layout:
- `param(0)` = speed (same)
- `param(2)` = trail (was param(3))
- `param(3)` = flame_brightness (was param(7))
- `param(4)` = kifs fold_angle (was param(4), same)
- `param(5)` = kifs scale (was param(5), same)
- `param(6)` = kifs brightness (was param(6), same)

Update the fragment shader reads:
```wgsl
let speed       = param(0);
let trail       = param(2);
let flame_bright = param(3);
let kifs_fold   = param(4);
let kifs_scale  = param(5);
let kifs_bright = param(6);
```

Also update the comment header to document the new layout.

**Step 2: Verify it renders correctly**

Run: `cargo run`

**Step 3: Commit**

```bash
git add playground.wgsl
git commit -m "feat: update fragment shader to new param layout"
```

---

### Task 9: Add keyboard controls

**Files:**
- Modify: `src/main.rs`

**Step 1: Add keyboard imports**

Add to the existing `use winit::event::` import:
```rust
use winit::event::KeyEvent;
use winit::keyboard::{Key, NamedKey};
```

**Step 2: Add keyboard handler in window_event match**

Add a new arm before the `_ => {}` catch-all:

```rust
WindowEvent::KeyboardInput {
    event: KeyEvent {
        logical_key,
        state: ElementState::Pressed,
        ..
    },
    ..
} => {
    match logical_key {
        Key::Named(NamedKey::Space) => {
            // Evolve: mutate and crossfade
            let old_params = self.genome.flatten();
            self.genome_history.push(old_params);
            if self.genome_history.len() > 10 {
                self.genome_history.remove(0);
            }
            self.genome = self.genome.mutate();
            self.target_params = self.genome.flatten();
            eprintln!("[evolve] → {}", self.genome.name);
        }
        Key::Named(NamedKey::Backspace) => {
            // Revert to previous genome
            if let Some(prev) = self.genome_history.pop() {
                self.target_params = prev;
                eprintln!("[revert] back to previous");
            }
        }
        Key::Character(c) => {
            match c.as_str() {
                "s" => {
                    let dir = project_dir().join("genomes");
                    match self.genome.save(&dir) {
                        Ok(p) => eprintln!("[save] {}", p.display()),
                        Err(e) => eprintln!("[save] error: {e}"),
                    }
                }
                "l" => {
                    let dir = project_dir().join("genomes");
                    match FlameGenome::load_random(&dir) {
                        Ok(g) => {
                            let old = self.genome.flatten();
                            self.genome_history.push(old);
                            eprintln!("[load] {}", g.name);
                            self.genome = g;
                            self.target_params = self.genome.flatten();
                        }
                        Err(e) => eprintln!("[load] error: {e}"),
                    }
                }
                "1" | "2" | "3" | "4" => {
                    let idx: usize = c.as_str().parse::<usize>().unwrap() - 1;
                    let base = 8 + idx * 12;
                    // Toggle weight: 0.0 ↔ 0.25
                    if self.target_params[base] < 0.01 {
                        self.target_params[base] = 0.25;
                    } else {
                        self.target_params[base] = 0.0;
                    }
                    eprintln!("[solo] transform {} = {}", idx, self.target_params[base]);
                }
                "=" | "+" => {
                    self.morph_rate = (self.morph_rate - 1.0).max(0.5);
                    eprintln!("[morph] rate = {} (slower)", self.morph_rate);
                }
                "-" => {
                    self.morph_rate = (self.morph_rate + 1.0).min(20.0);
                    eprintln!("[morph] rate = {} (faster)", self.morph_rate);
                }
                _ => {}
            }
        }
        _ => {}
    }
}
```

**Step 3: Use morph_rate in the interpolation**

In the RedrawRequested handler, change the lerp rate from hardcoded 5.0:
```rust
let rate = 1.0 - (-dt * self.morph_rate).exp();
```

**Step 4: Verify keyboard works**

Run: `cargo run`
- Press `Space` — should see "[evolve]" in terminal, flame morphs
- Press `Backspace` — should revert
- Press `S` — should save to genomes/
- Press `1` — should toggle transform 0

**Step 5: Commit**

```bash
git add src/main.rs
git commit -m "feat: keyboard controls (evolve/revert/save/load/solo/morph speed)"
```

---

### Task 10: Save default genome and verify full loop

**Files:**
- Create: `genomes/default.json` (via the app's save function)

**Step 1: Run the app, press S to save default genome**

Run: `cargo run`
Press `S`. Verify `genomes/default.json` is created.

**Step 2: Press Space a few times, then L to reload**

Verify evolve works and load brings back a saved genome.

**Step 3: Tag the release**

```bash
git add genomes/
git commit -m "feat: default genome preset"
git tag -a v0.4.0-genome-evolution -m "Flame genome with mutation, crossfade, save/load, keyboard controls"
```

---

## Summary

| Task | What | Key Files |
|------|------|-----------|
| 1 | Add rand dependency | Cargo.toml |
| 2 | FlameGenome struct + flatten | src/genome.rs |
| 3 | Mutation logic | src/genome.rs |
| 4 | Save/load | src/genome.rs |
| 5 | Expand uniform buffer to 64 floats | main.rs, both .wgsl |
| 6 | Integrate genome into App | main.rs |
| 7 | Param-driven compute shader | flame_compute.wgsl |
| 8 | Update fragment shader indices | playground.wgsl |
| 9 | Keyboard controls | main.rs |
| 10 | Verify full loop, tag release | genomes/ |
