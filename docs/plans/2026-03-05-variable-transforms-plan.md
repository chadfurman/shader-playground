# Variable Transforms & Time Signals Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand from fixed 4 transforms to dynamic N (sweet spot 6-16), with separate GPU storage buffer, mutation-driven birth/death, and 5 time signals.

**Architecture:** Split GPU data into uniform buffer (globals/KIFS/transform_count) and storage buffer (N*12 transform floats). Compute shader loops over N transforms. Weights system gets 5 time signals (slow/med/fast sine, perlin noise, envelope) replacing the single `time` channel, with `xfN_` expanding to actual transform count.

**Tech Stack:** Rust, wgpu 28, WGSL compute/fragment shaders, serde_json

**Design doc:** `docs/plans/2026-03-05-variable-transforms-design.md`

---

### Task 1: Genome API — split flatten into globals + transforms

**Files:**
- Modify: `src/genome.rs`

**Context:** Currently `flatten()` returns `[f32; 64]` packing everything into one array with `.take(4)` on transforms. We need two separate methods so globals go to the uniform buffer and transforms go to the storage buffer. We also need 6 transforms in the default genome.

**Step 1: Add `flatten_globals()` method**

Add this method to `FlameGenome` (after the existing `flatten()` — we'll remove `flatten()` later):

```rust
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
```

**Step 2: Add `flatten_transforms()` method**

```rust
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
```

**Step 3: Expand default genome to 6 transforms**

Update `default_genome()` to add 2 more transforms after the existing 4. The new transforms should have distinct parameters:

```rust
// Add after the 4th transform in the existing vec:
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
```

Also adjust the existing 4 transforms' weights from 0.25 to ~0.20 so total weight stays reasonable with 6.

**Step 4: Keep old `flatten()` temporarily**

Don't remove `flatten()` yet — main.rs still calls it. We'll remove it in Task 3 when we wire up the new split.

**Step 5: Verify build**

Run: `cargo build`
Expected: Compiles with no new errors (old `flatten()` still exists, new methods added alongside).

**Step 6: Commit**

```bash
git add src/genome.rs
git commit -m "feat: genome flatten_globals/flatten_transforms + 6-transform default"
```

---

### Task 2: GPU buffer refactor — new Uniforms + transform storage buffer

**Files:**
- Modify: `src/main.rs` (Uniforms struct, Gpu struct, bind group layouts, buffer creation)
- Modify: `flame_compute.wgsl`
- Modify: `playground.wgsl`

**Context:** This is the big coordinated change. The Uniforms struct, both shaders, the bind group layouts, and the buffer management all change together. Must be done in one commit or nothing compiles.

**Step 1: Update the Uniforms struct in `src/main.rs`**

Replace the current Uniforms with:

```rust
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    time: f32,
    frame: u32,
    resolution: [f32; 2],
    mouse: [f32; 2],
    transform_count: u32,
    _pad: u32,
    globals: [f32; 4],   // speed, zoom, trail, flame_brightness
    kifs: [f32; 4],       // fold_angle, scale, brightness, drift_speed
    extra: [f32; 4],      // color_shift, 0, 0, 0
}
```

**Step 2: Add transform buffer to Gpu struct**

Add `transform_buffer: wgpu::Buffer` field to the `Gpu` struct. Initialize it in `Gpu::create()`:

```rust
// Initial transform buffer (6 transforms * 12 floats * 4 bytes)
let transform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("transforms"),
    size: (6 * 12 * 4) as u64,
    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
});
```

**Step 3: Update compute bind group layout**

The compute bind group needs a 3rd entry for the transform storage buffer. In `Gpu::create()`, update `compute_bind_group_layout` to add:

```rust
wgpu::BindGroupLayoutEntry {
    binding: 2,
    visibility: wgpu::ShaderStages::COMPUTE,
    ty: wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
    },
    count: None,
},
```

**Step 4: Update `create_compute_bind_group()` helper**

Add the `transform_buffer` parameter and bind it at binding 2:

```rust
fn create_compute_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    histogram: &wgpu::Buffer,
    uniform_buffer: &wgpu::Buffer,
    transform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: histogram.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: transform_buffer.as_entire_binding(),
            },
        ],
    })
}
```

Update all call sites of `create_compute_bind_group` (in `Gpu::create()` and `rebuild_bind_groups()`) to pass `&self.transform_buffer`.

**Step 5: Add `resize_transform_buffer()` to Gpu**

```rust
fn resize_transform_buffer(&mut self, num_transforms: usize) {
    let size = (num_transforms.max(1) * 12 * 4) as u64;
    self.transform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("transforms"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    self.rebuild_bind_groups();
}
```

**Step 6: Update `flame_compute.wgsl`**

Replace the entire file with the new version. Key changes:
- New Uniforms struct (no more `params: array<vec4<f32>, 16>`)
- New `transforms` storage buffer binding
- `apply_xform` reads from `transforms` array instead of `param()`
- Selection loop replaces hardcoded 4-way check

```wgsl
// ── Fractal Flame Compute Shader ──

struct Uniforms {
    time: f32,
    frame: u32,
    resolution: vec2<f32>,
    mouse: vec2<f32>,
    transform_count: u32,
    _pad: u32,
    globals: vec4<f32>,   // speed, zoom, trail, flame_brightness
    kifs: vec4<f32>,      // fold_angle, scale, brightness, drift_speed
    extra: vec4<f32>,     // color_shift, 0, 0, 0
}

@group(0) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(1) var<uniform> u: Uniforms;
@group(0) @binding(2) var<storage, read> transforms: array<f32>;

fn xf(idx: u32, field: u32) -> f32 { return transforms[idx * 12u + field]; }

const PI: f32 = 3.14159265;

// ── PCG Random ──

fn pcg(state: ptr<function, u32>) -> u32 {
    let old = *state;
    *state = old * 747796405u + 2891336453u;
    let word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return (word >> 22u) ^ word;
}

fn randf(state: ptr<function, u32>) -> f32 {
    return f32(pcg(state)) / 4294967295.0;
}

fn rot2(a: f32) -> mat2x2<f32> {
    let c = cos(a); let s = sin(a);
    return mat2x2(c, -s, s, c);
}

// ── Flame Variations ──

fn V_sinusoidal(p: vec2<f32>) -> vec2<f32> {
    return vec2(sin(p.x), sin(p.y));
}

fn V_spherical(p: vec2<f32>) -> vec2<f32> {
    return p / (dot(p, p) + 1e-6);
}

fn V_swirl(p: vec2<f32>) -> vec2<f32> {
    let r2 = dot(p, p);
    return vec2(p.x * sin(r2) - p.y * cos(r2),
                p.x * cos(r2) + p.y * sin(r2));
}

fn V_horseshoe(p: vec2<f32>) -> vec2<f32> {
    let r = length(p) + 1e-6;
    return vec2((p.x - p.y) * (p.x + p.y), 2.0 * p.x * p.y) / r;
}

fn V_handkerchief(p: vec2<f32>) -> vec2<f32> {
    let r = length(p);
    let theta = atan2(p.y, p.x);
    return r * vec2(sin(theta + r), cos(theta - r));
}

// ── IFS Transform (reads from storage buffer) ──

fn apply_xform(p: vec2<f32>, idx: u32, t: f32) -> vec2<f32> {
    let angle  = xf(idx, 1u);
    let scale  = xf(idx, 2u);
    let ox     = xf(idx, 3u);
    let oy     = xf(idx, 4u);
    let w_lin  = xf(idx, 6u);
    let w_sin  = xf(idx, 7u);
    let w_sph  = xf(idx, 8u);
    let w_swi  = xf(idx, 9u);
    let w_hor  = xf(idx, 10u);
    let w_han  = xf(idx, 11u);

    let drift = u.kifs.w; // drift_speed
    let q = rot2(angle + t * 0.07 * drift * f32(idx + 1u)) * p * scale
          + vec2(ox + 0.05 * sin(t * 0.3 * drift * f32(idx + 1u)),
                 oy + 0.05 * cos(t * 0.4 * drift * f32(idx + 1u)));

    var v = q * w_lin;
    v += V_sinusoidal(q)    * w_sin;
    v += V_spherical(q)     * w_sph;
    v += V_swirl(q)         * w_swi;
    v += V_horseshoe(q)     * w_hor;
    v += V_handkerchief(q)  * w_han;

    return v;
}

fn xform_color(idx: u32) -> f32 {
    return xf(idx, 5u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let speed = u.globals.x;
    let zoom = u.globals.y;
    let t = u.time * speed;
    let num_xf = u.transform_count;

    var rng = gid.x * 2654435761u + u.frame * 7919u + 12345u;

    var p = vec2(randf(&rng) * 4.0 - 2.0, randf(&rng) * 4.0 - 2.0);
    var color_idx = randf(&rng);

    let w = u32(u.resolution.x);
    let h = u32(u.resolution.y);

    // Precompute total weight
    var total_weight = 0.0;
    for (var t_idx = 0u; t_idx < num_xf; t_idx++) {
        total_weight += xf(t_idx, 0u);
    }
    if (total_weight < 1e-6) { return; }

    for (var i = 0u; i < 200u; i++) {
        // Weighted random transform selection
        let r = randf(&rng) * total_weight;
        var tidx = 0u;
        var cumsum = 0.0;
        for (var ti = 0u; ti < num_xf; ti++) {
            cumsum += xf(ti, 0u);
            if (r < cumsum) {
                tidx = ti;
                break;
            }
        }

        p = apply_xform(p, tidx, t);
        color_idx = color_idx * 0.3 + xform_color(tidx) * 0.7;

        if (i < 20u) { continue; }

        let screen = (p / zoom + vec2(0.5, 0.5)) * vec2<f32>(f32(w), f32(h));
        let px_x = i32(screen.x);
        let px_y = i32(screen.y);

        if (px_x >= 0 && px_x < i32(w) && px_y >= 0 && px_y < i32(h)) {
            let buf_idx = (u32(px_y) * w + u32(px_x)) * 2u;
            atomicAdd(&histogram[buf_idx], 1u);
            atomicAdd(&histogram[buf_idx + 1u], u32(color_idx * 1000.0));
        }
    }
}
```

**Step 7: Update `playground.wgsl`**

Update the Uniforms struct and param reads. The fragment shader only reads globals/KIFS — no transform access needed:

Replace the `Uniforms` struct and the `param()` function + the reads in `fs_main`:

```wgsl
struct Uniforms {
    time: f32,
    frame: u32,
    resolution: vec2<f32>,
    mouse: vec2<f32>,
    transform_count: u32,
    _pad: u32,
    globals: vec4<f32>,
    kifs: vec4<f32>,
    extra: vec4<f32>,
}
```

In `fs_main`, replace the `param()` calls with direct uniform reads:

```wgsl
let speed        = u.globals.x;
let trail        = u.globals.z;
let flame_bright = u.globals.w;
let kifs_fold    = u.kifs.x;
let kifs_scale   = u.kifs.y;
let kifs_bright  = u.kifs.z;
let color_shift  = u.extra.x;
```

Remove the `fn param(i: i32) -> f32` function since it's no longer needed.

**Step 8: Verify build**

Run: `cargo build`
Expected: Compiles. Note: the app won't fully work yet because main.rs still uses the old params/flat approach. We'll fix that in Task 3.

**Step 9: Commit**

```bash
git add src/main.rs flame_compute.wgsl playground.wgsl
git commit -m "feat: separate uniform/storage buffers + shader N-transform loop"
```

---

### Task 3: Wire App to new split params

**Files:**
- Modify: `src/main.rs`
- Modify: `src/genome.rs` (remove old `flatten()`)

**Context:** Replace the flat `params: [f32; 64]` / `target_params: [f32; 64]` in App with split globals and transforms. Update the render loop to write both buffers.

**Step 1: Update App struct fields**

Replace:
```rust
params: [f32; 64],
target_params: [f32; 64],
```

With:
```rust
globals: [f32; 12],
target_globals: [f32; 12],
xf_params: Vec<f32>,
target_xf_params: Vec<f32>,
num_transforms: usize,
```

**Step 2: Update `App::new()`**

```rust
let genome = FlameGenome::default_genome();
let initial_globals = genome.flatten_globals();
let initial_xf = genome.flatten_transforms();
let num_transforms = genome.transforms.len();
// ...
globals: initial_globals,
target_globals: initial_globals,
xf_params: initial_xf.clone(),
target_xf_params: initial_xf,
num_transforms,
```

**Step 3: Update the render loop (RedrawRequested)**

Replace the old param packing with:

```rust
// Write uniforms
let uniforms = Uniforms {
    time: self.start.elapsed().as_secs_f32(),
    frame: self.frame,
    resolution: [gpu.config.width as f32, gpu.config.height as f32],
    mouse: self.mouse,
    transform_count: self.num_transforms as u32,
    _pad: 0,
    globals: [self.globals[0], self.globals[1], self.globals[2], self.globals[3]],
    kifs: [self.globals[4], self.globals[5], self.globals[6], self.globals[7]],
    extra: [self.globals[8], 0.0, 0.0, 0.0],
};

gpu.render(&uniforms, &self.xf_params);
```

**Step 4: Update `Gpu::render()` to write transform buffer**

Change `render()` signature to accept transform data and write it:

```rust
fn render(&mut self, uniforms: &Uniforms, transform_data: &[f32]) {
    self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    self.queue.write_buffer(&self.transform_buffer, 0, bytemuck::cast_slice(transform_data));
    // ... rest of render unchanged
}
```

**Step 5: Update morph interpolation**

The existing morph loop `for i in 0..64` becomes two loops:

```rust
// Morph globals
let rate = 1.0 - (-dt * MORPH_RATES[self.morph_rate_idx]).exp();
for i in 0..12 {
    self.globals[i] += (self.target_globals[i] - self.globals[i]) * rate;
}
// Morph transforms
let max_len = self.xf_params.len().max(self.target_xf_params.len());
self.xf_params.resize(max_len, 0.0);
self.target_xf_params.resize(max_len, 0.0);
for i in 0..max_len {
    self.xf_params[i] += (self.target_xf_params[i] - self.xf_params[i]) * rate;
}
```

**Step 6: Update weights apply call**

The weights system call in the render loop needs updating. For now, apply to globals only (transform weights will be wired in Task 7):

```rust
if self.audio_enabled {
    let time = self.start.elapsed().as_secs_f32();
    let base_globals = self.genome.flatten_globals();
    self.target_globals = self.weights.apply_globals(&base_globals, &self.audio_features, time);
    // Transform targets come from genome directly (weights applied in Task 7)
    self.target_xf_params = self.genome.flatten_transforms();
}
```

**Step 7: Update genome evolve/load/revert to use new targets**

Everywhere that currently sets `self.target_params = self.genome.flatten()`, change to:

```rust
self.target_globals = self.genome.flatten_globals();
self.target_xf_params = self.genome.flatten_transforms();
self.num_transforms = self.genome.transforms.len();
```

When transform count changes, also resize the GPU buffer:

```rust
if self.genome.transforms.len() != self.num_transforms {
    self.num_transforms = self.genome.transforms.len();
    if let Some(gpu) = &mut self.gpu {
        gpu.resize_transform_buffer(self.num_transforms);
    }
}
```

**Step 8: Update key handlers (1-4 solo toggle)**

The solo toggle currently modifies `self.target_params[base]`. Update to modify `self.target_xf_params`:

```rust
"1" | "2" | "3" | "4" => {
    let idx: usize = c.as_str().parse::<usize>().unwrap() - 1;
    if idx < self.num_transforms {
        let base = idx * 12; // weight is first field
        if self.target_xf_params[base] < 0.01 {
            self.target_xf_params[base] = 0.25;
        } else {
            self.target_xf_params[base] = 0.0;
        }
        eprintln!("[solo] transform {} = {}", idx, self.target_xf_params[base]);
    }
}
```

**Step 9: Remove old `flatten()` from genome.rs**

Delete the `flatten()` method and its doc comments.

**Step 10: Verify build and run**

Run: `cargo build && cargo run`
Expected: Compiles and runs. Should see the fractal flame rendering (now with 6 transforms instead of 4). The visual should look richer/more complex.

**Step 11: Commit**

```bash
git add src/main.rs src/genome.rs
git commit -m "feat: wire split globals/transforms through App + render loop"
```

---

### Task 4: Mutation add/remove transforms

**Files:**
- Modify: `src/genome.rs`

**Step 1: Add `mutate_add_transform` method**

```rust
fn mutate_add_transform(&mut self, rng: &mut impl Rng) {
    if self.transforms.is_empty() { return; }
    // Clone a random existing transform with perturbations
    let source_idx = rng.random_range(0..self.transforms.len());
    let mut new_xf = self.transforms[source_idx].clone();
    new_xf.weight = 0.05; // start small, fades in via morph
    new_xf.angle += rng.random_range(-0.3..0.3);
    new_xf.scale = (new_xf.scale + rng.random_range(-0.1..0.1)).clamp(0.3, 0.95);
    new_xf.offset[0] += rng.random_range(-0.3..0.3);
    new_xf.offset[1] += rng.random_range(-0.3..0.3);
    new_xf.color = rng.random_range(0.0..1.0);
    self.transforms.push(new_xf);
}
```

**Step 2: Add `mutate_remove_transform` method**

```rust
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
```

**Step 3: Update `mutate()` to include add/remove with sweet spot bias**

Update the `mutate()` method. After the existing mutation loop, add:

```rust
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
```

**Step 4: Verify build**

Run: `cargo build`
Expected: Compiles.

**Step 5: Commit**

```bash
git add src/genome.rs
git commit -m "feat: mutation add/remove transforms biased toward 6-16 sweet spot"
```

---

### Task 5: Time signals — 5 channels replacing single `time`

**Files:**
- Modify: `src/weights.rs`
- Modify: `src/main.rs`

**Context:** Replace the single `time` HashMap with 5 separate time signal channels. Add a simple 1D value noise function for `time_noise`. Add `last_mutation_time` to App for `time_envelope`.

**Step 1: Add value noise function to `src/weights.rs`**

Add at the top of the file:

```rust
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
```

**Step 2: Add `TimeSignals` struct**

```rust
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
```

**Step 3: Update `Weights` struct**

Replace the single `time` field with 5 fields:

```rust
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
```

Update `SIGNAL_COUNT` to `11.0`.

**Step 4: Update `apply()` and `mutation_rate()` signal lists**

In both methods, update the signals array to use all 11:

```rust
let signals: &[(&HashMap<String, f32>, f32)] = &[
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
];
```

Update method signatures to accept `&TimeSignals` instead of `time: f32`.

**Step 5: Update `src/main.rs` to track mutation time and compute TimeSignals**

Add `last_mutation_time: f32` to App struct (initialized to 0.0 in `new()`).

Set it whenever mutation happens:
```rust
self.last_mutation_time = self.start.elapsed().as_secs_f32();
```

In the render loop, compute time signals:
```rust
let time = self.start.elapsed().as_secs_f32();
let time_since_mutation = time - self.last_mutation_time;
let time_signals = weights::TimeSignals::compute(time, time_since_mutation);
```

Pass `&time_signals` to `weights.apply()` and `weights.mutation_rate()`.

**Step 6: Verify build**

Run: `cargo build`
Expected: Compiles.

**Step 7: Commit**

```bash
git add src/weights.rs src/main.rs
git commit -m "feat: 5 time signals (slow/med/fast sine, noise, envelope)"
```

---

### Task 6: Weights system — split apply for globals + transforms

**Files:**
- Modify: `src/weights.rs`
- Modify: `src/main.rs`

**Context:** The weights system needs separate methods for globals and transforms, since they live in different buffers. The `xfN_` wildcard needs to know the actual transform count.

**Step 1: Replace `apply()` with `apply_globals()` and `apply_transforms()`**

```rust
/// Apply weights to global params.
pub fn apply_globals(
    &self,
    base: &[f32; 12],
    features: &crate::audio::AudioFeatures,
    time_signals: &TimeSignals,
) -> [f32; 12] {
    let mut result = *base;
    let signals = self.signal_list(features, time_signals);
    for (signal_idx, (weights, signal_val)) in signals.iter().enumerate() {
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
```

**Step 2: Extract signal list helper**

```rust
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
```

**Step 3: Add `global_index()` function**

```rust
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
```

**Step 4: Add `try_parse_xf()` helper**

```rust
fn try_parse_xf(name: &str) -> Option<(usize, usize)> {
    let rest = name.strip_prefix("xf")?;
    let (digit, field) = rest.split_once('_')?;
    let xf: usize = digit.parse().ok()?;
    let field_offset = xf_field_index(field)?;
    Some((xf, field_offset))
}
```

**Step 5: Remove old `apply()` and `param_index()`**

Delete the old `apply()` method and the `param_index()` function, which are now replaced by `apply_globals()`, `apply_transforms()`, `global_index()`, and `try_parse_xf()`.

**Step 6: Update main.rs render loop**

```rust
if self.audio_enabled {
    let time = self.start.elapsed().as_secs_f32();
    let time_since_mutation = time - self.last_mutation_time;
    let time_signals = weights::TimeSignals::compute(time, time_since_mutation);

    let base_globals = self.genome.flatten_globals();
    let base_xf = self.genome.flatten_transforms();

    self.target_globals = self.weights.apply_globals(&base_globals, &self.audio_features, &time_signals);
    self.target_xf_params = self.weights.apply_transforms(&base_xf, self.num_transforms, &self.audio_features, &time_signals);

    // Auto-evolve
    let mr = self.weights.mutation_rate(&self.audio_features, &time_signals);
    // ... (existing mutation_accum logic)
}
```

**Step 7: Verify build and run**

Run: `cargo build && cargo run`
Expected: Compiles, runs, visuals work with audio weights + time signals affecting the fractal.

**Step 8: Commit**

```bash
git add src/weights.rs src/main.rs
git commit -m "feat: split weight application for globals + transforms, dynamic xfN_"
```

---

### Task 7: Update weights.json with time signals

**Files:**
- Modify: `weights.json`

**Step 1: Rewrite weights.json**

Update with the 5 time signal channels and updated `_doc`:

```json
{
  "_doc": {
    "_signals": "bass, mids, highs, energy, beat, beat_accum, time_slow, time_med, time_fast, time_noise, time_envelope — each maps param names to weights. Unspecified = 0.",
    "_formula": "final[i] = genome_base[i] + (sum of weight * signal for all 11 signals) / 11",
    "_wildcards": "xfN_ expands to all transforms with per-transform randomized scaling (0.5x-1.5x). xf0_, xf1_ etc. target specific transforms.",
    "_time_signals": {
      "time_slow": "sin(t * 0.1) — ~60s cycle, for glacial drift",
      "time_med": "sin(t * 0.5) — ~12s cycle, for breathing rhythms",
      "time_fast": "sin(t * 2.0) — ~3s cycle, for quick pulsing",
      "time_noise": "Smooth noise seeded by time — organic wandering, never repeats",
      "time_envelope": "0-1 ramp since last mutation (10s to reach 1.0) — settle then drift"
    },
    "_params": {
      "speed": "Global animation speed",
      "zoom": "Camera zoom level",
      "trail": "Feedback trail decay (lower = longer trails)",
      "flame_brightness": "Overall flame brightness",
      "kifs_fold": "KIFS folding angle",
      "kifs_scale": "KIFS scale factor",
      "kifs_brightness": "KIFS geometry brightness",
      "drift_speed": "Orbital drift speed multiplier",
      "color_shift": "Hue shift applied to flame palette",
      "xfN_weight": "All transforms blend weight",
      "xfN_angle": "All transforms rotation angle",
      "xfN_scale": "All transforms scale factor",
      "xfN_offset_x": "All transforms X translation",
      "xfN_offset_y": "All transforms Y translation",
      "xfN_color": "All transforms palette index",
      "xfN_linear": "All transforms linear variation",
      "xfN_sinusoidal": "All transforms sinusoidal variation",
      "xfN_spherical": "All transforms spherical variation",
      "xfN_swirl": "All transforms swirl variation",
      "xfN_horseshoe": "All transforms horseshoe variation",
      "xfN_handkerchief": "All transforms handkerchief variation",
      "mutation_rate": "Virtual — auto-evolve accumulator (triggers mutate at 1.0)"
    }
  },
  "bass": {
    "kifs_fold": 0.4,
    "xfN_weight": 0.5,
    "xfN_swirl": 0.3,
    "color_shift": -0.6
  },
  "mids": {
    "xfN_sinusoidal": 0.4,
    "xfN_angle": 0.3
  },
  "highs": {
    "xfN_spherical": 0.3,
    "color_shift": 0.3
  },
  "energy": {
    "flame_brightness": 0.9,
    "kifs_scale": 0.2,
    "kifs_brightness": 0.4,
    "drift_speed": 1.8,
    "xfN_weight": 0.3,
    "xfN_scale": 0.2,
    "mutation_rate": 0.1
  },
  "beat": {
    "xfN_horseshoe": 0.5,
    "xfN_handkerchief": 0.4
  },
  "beat_accum": {
    "trail": 0.6,
    "xfN_offset_x": 0.2,
    "xfN_offset_y": 0.2,
    "mutation_rate": 1.0
  },
  "time_slow": {
    "kifs_fold": 0.3,
    "kifs_scale": 0.2
  },
  "time_med": {
    "color_shift": 0.2,
    "xfN_angle": 0.1
  },
  "time_fast": {
    "xfN_scale": 0.05
  },
  "time_noise": {
    "xfN_offset_x": 0.1,
    "xfN_offset_y": 0.1
  },
  "time_envelope": {
    "drift_speed": 0.5
  }
}
```

**Step 2: Verify build**

Run: `cargo build`
Expected: Compiles (weights.json is read at runtime, not compile time).

**Step 3: Commit and tag**

```bash
git add weights.json
git commit -m "feat: weights.json with 5 time signal channels"
git tag -a v0.7.0-variable-transforms -m "Variable N transforms, 5 time signals, mutation add/remove"
```

---

## Summary

| Task | What | Files |
|---|---|---|
| 1 | Genome flatten split + 6-transform default | `genome.rs` |
| 2 | GPU buffer refactor (Uniforms + storage + shaders) | `main.rs`, `flame_compute.wgsl`, `playground.wgsl` |
| 3 | Wire App to split params + morph interpolation | `main.rs`, `genome.rs` |
| 4 | Mutation add/remove with sweet spot bias | `genome.rs` |
| 5 | 5 time signals (sine × 3, noise, envelope) | `weights.rs`, `main.rs` |
| 6 | Weights split apply (globals + transforms) | `weights.rs`, `main.rs` |
| 7 | Update weights.json with time signal channels | `weights.json` |

**Dependencies:** Task 2 depends on 1. Task 3 depends on 1+2. Tasks 4, 5, 6 depend on 3. Task 7 depends on 5+6.
