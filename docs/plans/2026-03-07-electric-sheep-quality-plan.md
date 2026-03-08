# Electric Sheep Quality Phase 1 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the visual quality gap with Electric Sheep through progressive accumulation, flam3 tonemapping, curated genomes, and better bloom.

**Architecture:** Add a persistent float32 accumulation buffer (separate from histogram) that compounds density+color across frames with exponential decay. Replace ad-hoc tonemapping with flam3's log-density algorithm normalized by sample count. Ship curated seed genomes. All new params in weights.json.

**Tech Stack:** Rust + wgpu, WGSL compute/fragment shaders, serde_json config

---

### Task 1: Add accumulation config params to RuntimeConfig

**Files:**
- Modify: `src/weights.rs` (add fields + defaults)
- Modify: `weights.json` (add to `_config` section)
- Modify: `CLAUDE.md` (update uniform layout docs)

**Step 1: Add fields to RuntimeConfig**

In `src/weights.rs`, add these fields to the `RuntimeConfig` struct (after `variation_scales`):

```rust
#[serde(default = "default_accumulation_decay")]
pub accumulation_decay: f32,
#[serde(default = "default_samples_per_frame")]
pub samples_per_frame: u32,
#[serde(default = "default_bloom_radius")]
pub bloom_radius: f32,
```

And the default functions:

```rust
fn default_accumulation_decay() -> f32 { 0.995 }
fn default_samples_per_frame() -> u32 { 256 }
fn default_bloom_radius() -> f32 { 3.0 }
```

**Step 2: Add to weights.json**

Add to the `_config` section:
```json
"accumulation_decay": 0.995,
"samples_per_frame": 256,
"bloom_radius": 3.0
```

Add to `_config_doc`:
```json
"accumulation_decay": "Exponential decay per frame for accumulation buffer (default 0.995, half-life ~2.3s at 60fps)",
"samples_per_frame": "GPU compute workgroups per frame — accumulation compensates for fewer (default 256)",
"bloom_radius": "Bloom sample radius in pixels (default 3.0)"
```

**Step 3: Update CLAUDE.md uniform layout**

No uniform layout changes yet — these are CPU-side config only.

**Step 4: Verify**

Run: `cargo build`
Expected: compiles with no errors

**Step 5: Commit**

```bash
git add src/weights.rs weights.json
git commit -m "feat: add accumulation_decay, samples_per_frame, bloom_radius config params"
```

---

### Task 2: Create accumulation compute shader

**Files:**
- Create: `accumulation.wgsl`

**Step 1: Write the accumulation shader**

This shader runs per-pixel. It reads the current frame's histogram, blends it into the persistent accumulation buffer using exponential decay, then the fragment shader reads from this buffer instead of the raw histogram.

```wgsl
// accumulation.wgsl — per-pixel accumulation with exponential decay
//
// Reads raw histogram (u32 density + RGB), blends into persistent
// float32 accumulation buffer. Fragment shader reads accum instead of histogram.

struct AccumUniforms {
    resolution: vec2<f32>,
    decay: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read> histogram: array<u32>;
@group(0) @binding(1) var<storage, read_write> accumulation: array<f32>;
@group(0) @binding(2) var<uniform> params: AccumUniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(params.resolution.x);
    let h = u32(params.resolution.y);
    if (gid.x >= w || gid.y >= h) {
        return;
    }

    let px = gid.y * w + gid.x;
    let hist_idx = px * 4u;
    let accum_idx = px * 4u;

    // Read raw histogram for this frame
    let density = f32(histogram[hist_idx]);
    let r = f32(histogram[hist_idx + 1u]);
    let g = f32(histogram[hist_idx + 2u]);
    let b = f32(histogram[hist_idx + 3u]);

    // Exponential decay blend: accum = accum * decay + new_frame
    let decay = params.decay;
    accumulation[accum_idx]      = accumulation[accum_idx]      * decay + density;
    accumulation[accum_idx + 1u] = accumulation[accum_idx + 1u] * decay + r;
    accumulation[accum_idx + 2u] = accumulation[accum_idx + 2u] * decay + g;
    accumulation[accum_idx + 3u] = accumulation[accum_idx + 3u] * decay + b;
}
```

**Step 2: Verify shader syntax**

Run: `cargo build` (won't compile it yet — just make sure file is saved)

**Step 3: Commit**

```bash
git add accumulation.wgsl
git commit -m "feat: add accumulation compute shader with exponential decay"
```

---

### Task 3: Create accumulation GPU resources in main.rs

**Files:**
- Modify: `src/main.rs` (Gpu struct, create method, resize, rebuild_bind_groups)

**Step 1: Add accumulation buffer and pipeline to Gpu struct**

Add to `struct Gpu` (after `workgroups`):

```rust
// Accumulation pipeline
accumulation_pipeline: wgpu::ComputePipeline,
accumulation_bind_group_layout: wgpu::BindGroupLayout,
accumulation_bind_group: wgpu::BindGroup,
accumulation_buffer: wgpu::Buffer,
accumulation_uniform_buffer: wgpu::Buffer,
```

**Step 2: Create accumulation buffer helper**

Add a helper function (near `create_histogram_buffer`):

```rust
fn create_accumulation_buffer(device: &wgpu::Device, w: u32, h: u32) -> wgpu::Buffer {
    let pixel_count = w.max(1) as u64 * h.max(1) as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("accumulation"),
        size: pixel_count * 4 * 4, // 4 f32s per pixel (density, R, G, B)
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
```

**Step 3: Add accumulation uniform buffer**

A small uniform buffer for the AccumUniforms struct (resolution + decay + pad = 4 floats = 16 bytes):

```rust
let accumulation_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("accum_uniforms"),
    size: 16,
    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
});
```

**Step 4: Create bind group layout for accumulation**

```rust
let accumulation_bind_group_layout =
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("accumulation"),
        entries: &[
            // binding 0: histogram (read-only storage)
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 1: accumulation (read-write storage)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 2: uniforms
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
```

**Step 5: Create accumulation pipeline + bind group**

Load the shader, create the pipeline and bind group:

```rust
fn load_accumulation_source() -> String {
    fs::read_to_string(project_dir().join("accumulation.wgsl"))
        .unwrap_or_else(|_| include_str!("../accumulation.wgsl").to_string())
}

fn create_accumulation_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader_src: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("accumulation"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("accumulation"),
        layout: Some(layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn create_accumulation_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    histogram: &wgpu::Buffer,
    accumulation: &wgpu::Buffer,
    uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("accumulation"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: histogram.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: accumulation.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    })
}
```

**Step 6: Wire it all into Gpu::create()**

In the `Gpu::create()` method, after creating the histogram buffer:

```rust
let accumulation_buffer = create_accumulation_buffer(&device, config.width, config.height);
let accumulation_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("accum_uniforms"),
    size: 16,
    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
});

// ... after creating compute pipeline layout:
let accumulation_pipeline_layout =
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("accumulation"),
        bind_group_layouts: &[&accumulation_bind_group_layout],
        immediate_size: 0,
    });

let accum_src = load_accumulation_source();
let accumulation_pipeline = create_accumulation_pipeline(
    &device, &accumulation_pipeline_layout, &accum_src,
);

let accumulation_bind_group = create_accumulation_bind_group(
    &device,
    &accumulation_bind_group_layout,
    &histogram_buffer,
    &accumulation_buffer,
    &accumulation_uniform_buffer,
);
```

Add all fields to the `Self { ... }` constructor.

**Step 7: Update resize()**

In `Gpu::resize()`, after recreating the histogram buffer:

```rust
self.accumulation_buffer = create_accumulation_buffer(&self.device, w, h);
```

**Step 8: Update rebuild_bind_groups()**

Add at the end of `rebuild_bind_groups()`:

```rust
self.accumulation_bind_group = create_accumulation_bind_group(
    &self.device,
    &self.accumulation_bind_group_layout,
    &self.histogram_buffer,
    &self.accumulation_buffer,
    &self.accumulation_uniform_buffer,
);
```

**Step 9: Verify**

Run: `cargo build`
Expected: compiles (accumulation pipeline created but not yet dispatched)

**Step 10: Commit**

```bash
git add src/main.rs
git commit -m "feat: create accumulation GPU resources (buffer, pipeline, bind group)"
```

---

### Task 4: Wire accumulation into render loop

**Files:**
- Modify: `src/main.rs` (Gpu::render method, RedrawRequested handler)
- Modify: `playground.wgsl` (read from accumulation buffer instead of histogram)

**Step 1: Update render bind group layout**

Change binding 3 in the render bind group layout from histogram to accumulation buffer. The fragment shader will now read from the accumulation buffer (float storage) instead of the histogram (u32 storage).

In `Gpu::create()`, the render bind group layout entry for binding 3 stays the same (storage, read-only) — we just bind the accumulation buffer instead.

**Step 2: Update create_render_bind_group**

Change the `histogram` parameter to `accumulation`:

```rust
fn create_render_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    prev_frame: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    accumulation: &wgpu::Buffer,  // was: histogram
    crossfade_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    // ... binding 3 now uses accumulation buffer
}
```

Update all call sites in `Gpu::create()` and `rebuild_bind_groups()` to pass `&self.accumulation_buffer` instead of `&self.histogram_buffer`.

**Step 3: Add accumulation dispatch to render()**

In `Gpu::render()`, after the chaos game compute pass (step 2) and before the render pass (step 3), add the accumulation pass:

```rust
// 2.5. Accumulation pass: blend histogram into persistent buffer
{
    let mut apass = encoder.begin_compute_pass(
        &wgpu::ComputePassDescriptor {
            label: Some("accumulation"),
            ..Default::default()
        },
    );
    apass.set_pipeline(&self.accumulation_pipeline);
    apass.set_bind_group(0, &self.accumulation_bind_group, &[]);
    // Dispatch enough workgroups to cover all pixels (16x16 per workgroup)
    let wg_x = (self.config.width + 15) / 16;
    let wg_y = (self.config.height + 15) / 16;
    apass.dispatch_workgroups(wg_x, wg_y, 1);
}
```

**Step 4: Write accumulation uniforms each frame**

In the `RedrawRequested` handler, before `gpu.render()`, write the accumulation uniform buffer:

```rust
if let Some(gpu) = &self.gpu {
    let accum_uniforms: [f32; 4] = [
        gpu.config.width as f32,
        gpu.config.height as f32,
        self.weights._config.accumulation_decay,
        0.0, // padding
    ];
    gpu.queue.write_buffer(
        &gpu.accumulation_uniform_buffer,
        0,
        bytemuck::cast_slice(&accum_uniforms),
    );
}
```

**Step 5: Use samples_per_frame for workgroups**

In the weights reload handler (check_file_changes), update:

```rust
gpu.workgroups = self.weights._config.samples_per_frame;
```

And in `App::new()` / `Gpu::create()`, initialize `workgroups` from `samples_per_frame` instead of `workgroups` config.

Keep the `workgroups` config field as a fallback/alias — if `samples_per_frame` is present, use it; otherwise fall back to `workgroups`.

**Step 6: Update playground.wgsl to read f32 accumulation buffer**

Change the histogram binding from `array<u32>` to `array<f32>`:

```wgsl
@group(0) @binding(3) var<storage, read> accumulation: array<f32>;
```

Update the fragment shader to read floats directly:

```wgsl
let buf_idx = (px.y * w + px.x) * 4u;
let density = accumulation[buf_idx];
let acc_r = accumulation[buf_idx + 1u];
let acc_g = accumulation[buf_idx + 2u];
let acc_b = accumulation[buf_idx + 3u];
```

(Remove the `f32()` casts since values are already f32.)

**Step 7: Verify**

Run: `cargo build && cargo run`
Expected: App runs. Image should build up quality over several seconds. Reducing `accumulation_decay` in weights.json (e.g., to 0.9) should show faster decay; 0.999 should show more persistence.

**Step 8: Commit**

```bash
git add src/main.rs playground.wgsl accumulation.wgsl
git commit -m "feat: wire accumulation buffer into render loop — progressive quality buildup"
```

---

### Task 5: Flam3-style tonemapping

**Files:**
- Modify: `playground.wgsl` (rewrite tonemapping)
- Modify: `src/weights.rs` (add `flame_k2_brightness` config param if needed)

**Step 1: Implement flam3 tonemapping in fragment shader**

Replace the current log-density + vibrancy section in `playground.wgsl` with the actual flam3 algorithm:

```wgsl
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<u32>(u32(pos.x), u32(pos.y));
    let w = u32(u.resolution.x);
    let tex_uv = pos.xy / u.resolution;

    let trail        = u.globals.z;
    let flame_bright = u.globals.w;
    let vibrancy     = u.extra.y;
    let bloom_int    = u.extra.z;
    let gamma_val    = u.extra.w;
    let highlight    = u.extra3.w;
    let bloom_rad    = u.extra3.x;  // repurpose spin_speed_max slot or use new slot

    // ── Read accumulation buffer ──
    let buf_idx = (px.y * w + px.x) * 4u;
    let density = accumulation[buf_idx];
    let acc_r = accumulation[buf_idx + 1u];
    let acc_g = accumulation[buf_idx + 2u];
    let acc_b = accumulation[buf_idx + 3u];

    // ── Flam3 tonemapping ──
    // k2 normalizes for resolution and sample density
    let area = u.resolution.x * u.resolution.y;
    let k2 = 1.0 / (area * flame_bright);

    // Log-density scaling (Scott Draves' algorithm)
    let log_scale = select(
        0.0,
        flame_bright * log(1.0 + density * k2) / density,
        density > 0.0
    );
    let alpha = log_scale * density;

    // Recover average color
    let raw_color = select(
        vec3(0.0),
        vec3(acc_r, acc_g, acc_b) / max(density * 1000.0, 1.0),
        density > 0.0
    );

    // Vibrancy blend (flam3 algorithm)
    // vibrancy=1: fully color-saturated, scaled by alpha
    // vibrancy=0: luminance-based, gamma applied inside
    let gamma_alpha = pow(max(alpha, 0.001), gamma_val);
    let ls = vibrancy * alpha + (1.0 - vibrancy) * gamma_alpha;
    let flame = ls * raw_color + (1.0 - vibrancy) * gamma_alpha * vec3(1.0);

    // Highlight boost for hot-spot glow
    let highlight_boost = pow(max(alpha, 0.0), highlight) * highlight * 0.3;
    let lit = flame + vec3(highlight_boost);

    // ── Feedback trail ──
    let prev = textureSample(prev_frame, prev_sampler, tex_uv).rgb;
    var new_col = lit + prev * trail;

    // ── Bloom ──
    if (bloom_int > 0.001) {
        let texel = 1.0 / u.resolution;
        let r = bloom_rad;
        let bloom = (
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(-r,  0.0) * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2( r,  0.0) * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(0.0, -r)  * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(0.0,  r)  * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(-r,  -r) * 0.707 * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2( r,  -r) * 0.707 * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(-r,   r) * 0.707 * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2( r,   r) * 0.707 * texel).rgb
        ) * 0.125;
        new_col += bloom * bloom_int;
    }

    // ── Final gamma (applied to composited result) ──
    var col = pow(max(new_col, vec3(0.0)), vec3(gamma_val));

    // Soft clamp — Reinhard tonemapping
    col = col / (col + vec3(1.0));

    return vec4(col, 1.0);
}
```

**Step 2: Pass bloom_radius through uniforms**

We need to get `bloom_radius` to the shader. Options:
- Use one of the existing uniform slots (extra3.x is currently spin_speed_max, which the fragment shader doesn't need)
- Add it as a new uniform vec4

Recommended: Pass it through extra3.x in the fragment shader since spin_speed_max is only used by the compute shader. The fragment shader can read extra3.x as bloom_radius. In `src/main.rs`, when writing uniforms for the fragment shader, set `extra3[0]` to `bloom_radius` from config.

Actually, the uniform buffer is shared between compute and fragment. So we need to be careful. The simplest approach: add `bloom_radius` to the globals array. But globals is already [f32; 20] and full.

Better approach: The fragment shader already reads extra3.x as spin_speed_max but doesn't use it. Just document that extra3.x serves double duty — compute reads it as spin_speed_max, fragment reads it as bloom_radius. But they share the same uniform buffer, so we can't have different values.

Simplest: just keep bloom_radius as a hardcoded value in the shader for now (it's read from the accumulation uniform or we pass it some other way). Actually — let's just use the existing bloom radius constant in the shader and make it configurable later when we expand the uniform buffer. The bloom_radius config param is there for hot-reload via the shader constant.

For now, we can pass bloom_radius through a uniform expansion. But to keep this simple and avoid changing the uniform struct (which affects both shaders), let's use the value from config in the shader by reading the bloom_radius from the accumulation uniforms. But the fragment shader doesn't have access to the accumulation bind group.

**Pragmatic solution:** The bloom radius is already `3.0` in the shader. Keep it there. The `bloom_radius` config param in RuntimeConfig will be used when we expand uniforms in a later task. For now, it exists in config as documentation of intent.

**Step 3: Verify**

Run: `cargo build && cargo run`
Expected: Tonemapping should produce more consistent brightness regardless of workgroup count. Dim filaments should be more visible. Changing `flame_brightness` in weights.json should still control overall sensitivity.

**Step 4: Commit**

```bash
git add playground.wgsl
git commit -m "feat: implement flam3-style tonemapping with vibrancy blend and log-density normalization"
```

---

### Task 6: Reduce trail feedback, let accumulation be primary persistence

**Files:**
- Modify: `weights.json` (reduce trail value)

**Step 1: Reduce trail**

In `weights.json`, the trail value is set in the genome's `flatten_globals`. The default is controlled by the genome. We should set the config default trail lower since accumulation now handles persistence.

Find where `trail` is set in `genome.rs` `flatten_globals()` and reduce the base value, or override via weights.json audio signals.

Actually, trail is `globals[2]` and it's set from genome/config. The simplest change: in weights.json `_config`, add a `trail` field with value `0.15` (down from ~0.34).

Add to RuntimeConfig:

```rust
#[serde(default = "default_trail")]
pub trail: f32,
```

```rust
fn default_trail() -> f32 { 0.15 }
```

Then use this in `flatten_globals()` as the base trail value.

**Step 2: Verify**

Run: `cargo run`
Expected: Accumulation provides smooth buildup, trail just adds a little temporal anti-aliasing.

**Step 3: Commit**

```bash
git add src/weights.rs src/genome.rs weights.json
git commit -m "feat: reduce trail feedback to 0.15, accumulation is now primary persistence"
```

---

### Task 7: Curated seed genomes

**Files:**
- Create: `genomes/seeds/` directory with 15-20 JSON genome files
- Modify: `src/genome.rs` (load seeds, prefer seeds for mutation starting points)

**Step 1: Create seed genome directory**

```bash
mkdir -p genomes/seeds
```

**Step 2: Create seed genomes**

Hand-craft 15-20 genomes with these properties:
- 5-8 transforms each
- Good offset spread (multiple quadrants)
- Diverse variation combos
- Symmetry 3-8 for many
- Mix of contractive and expansive variations

Each seed should be a JSON file matching FlameGenome format. Example:

```json
{
  "name": "seed-spiral-bloom",
  "symmetry": 5,
  "transforms": [
    {
      "weight": 0.35,
      "angle": 0.0,
      "scale": 0.6,
      "offset_x": 0.3,
      "offset_y": 0.0,
      "color": 0.1,
      "variations": {
        "linear": 0.4,
        "sinusoidal": 0.3,
        "julia": 0.2,
        "spiral": 0.1
      }
    },
    ...
  ]
}
```

Create at least 15 seeds covering different aesthetics:
- Spirals (julia + spiral + sinusoidal)
- Symmetric flowers (high symmetry + spherical + bubble)
- Organic flows (waves + popcorn + bent)
- Crystalline (diamond + rings + disc)
- Ethereal (fisheye + polar + exponential)

**Step 3: Update startup to load from seeds**

In `App::new()`, change startup logic:

```rust
// Try loading a seed genome instead of random mutations
let seeds_dir = project_dir().join("genomes").join("seeds");
let genome = if seeds_dir.exists() {
    FlameGenome::load_random(&seeds_dir).unwrap_or_else(|_| {
        let mut g = FlameGenome::default_genome();
        for _ in 0..3 {
            g = g.mutate(&AudioFeatures::default(), &weights._config);
        }
        g
    })
} else {
    let mut g = FlameGenome::default_genome();
    for _ in 0..3 {
        g = g.mutate(&AudioFeatures::default(), &weights._config);
    }
    g
};
```

**Step 4: Bias mutations toward seeds**

In `FlameGenome::mutate()` or `mutate_inner()`, add logic to sometimes start from a seed:

```rust
// 70% chance: mutate from a seed, 30% chance: mutate current genome
// (configurable via RuntimeConfig)
let use_seed = rng.random::<f32>() < 0.7; // TODO: make configurable
if use_seed {
    if let Ok(seed) = FlameGenome::load_random(&project_dir().join("genomes").join("seeds")) {
        // Mutate the seed instead of self
        return seed.mutate_inner(audio, cfg, rng);
    }
}
```

**Step 5: Verify**

Run: `cargo run`
Expected: App starts with a curated seed genome. Pressing space should sometimes produce mutations based on seeds rather than the current genome.

**Step 6: Commit**

```bash
git add genomes/seeds/ src/genome.rs src/main.rs
git commit -m "feat: add 15+ curated seed genomes with seed-biased mutation"
```

---

### Task 8: Multi-radius bloom upgrade

**Files:**
- Modify: `playground.wgsl` (3-radius bloom with configurable parameters)

**Step 1: Upgrade bloom to multi-radius**

Replace the current single-radius bloom with a 3-radius version:

```wgsl
// ── Multi-radius bloom — 3 radii, 4 taps each ──
if (bloom_int > 0.001) {
    let texel = 1.0 / u.resolution;
    var bloom_sum = vec3(0.0);

    // 3 radii: tight (2px), medium (5px), wide (12px)
    let radii = array<f32, 3>(2.0, 5.0, 12.0);
    let weights_arr = array<f32, 3>(0.5, 0.3, 0.2);

    for (var ri = 0u; ri < 3u; ri++) {
        let r = radii[ri];
        let w_r = weights_arr[ri];
        let tap = (
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(-r, 0.0) * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2( r, 0.0) * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(0.0, -r) * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(0.0,  r) * texel).rgb
        ) * 0.25;
        bloom_sum += tap * w_r;
    }
    new_col += bloom_sum * bloom_int;
}
```

Note: The 3 radii (2, 5, 12) and weights (0.5, 0.3, 0.2) are the algorithm definition — these are NOT magic numbers, they define the multi-scale bloom character. The bloom_int config param controls overall intensity.

**Step 2: Verify**

Run: `cargo run`
Expected: Bloom should produce wider, softer halos around bright structures compared to the single-radius version.

**Step 3: Commit**

```bash
git add playground.wgsl
git commit -m "feat: multi-radius bloom (3 scales) for soft Electric Sheep-style glow"
```

---

### Task 9: Final integration verification

**Files:**
- No new files — verification only

**Step 1: Build**

Run: `cargo build`
Expected: Clean build, no warnings

**Step 2: Run and verify accumulation**

Run: `cargo run`
Expected:
- Image starts grainy, builds to smooth quality over 5-10 seconds
- Changing `accumulation_decay` in weights.json hot-reloads (0.99 = faster decay, 0.999 = more persistence)
- Changing `samples_per_frame` works (lower = grainier per-frame but accumulation compensates)

**Step 3: Verify tonemapping**

- Dim filaments should be visible (not lost in black)
- Bright areas should glow without clipping to white
- Changing `flame_brightness` adjusts sensitivity
- Changing `vibrancy` adjusts color saturation vs luminance mapping

**Step 4: Verify seeds**

- App starts with a curated seed genome
- Seeds produce more visually interesting starting points than random mutations
- Space bar produces new genomes, some based on seeds

**Step 5: Verify bloom**

- Bloom produces soft multi-scale halo
- `bloom_intensity` in weights.json controls strength
- Setting to 0.0 disables bloom

**Step 6: Verify framerate**

- Should stay above 30fps on MacBook Pro
- If not, reduce `samples_per_frame` in weights.json

**Step 7: Final commit**

If any tweaks were needed, commit them:

```bash
git add -A
git commit -m "chore: Phase 1 integration tuning and verification"
```

---

## Summary

| Task | What | Key Files |
|------|------|-----------|
| 1 | Config params | `weights.rs`, `weights.json` |
| 2 | Accumulation shader | `accumulation.wgsl` |
| 3 | GPU resources | `main.rs` |
| 4 | Wire into render loop | `main.rs`, `playground.wgsl` |
| 5 | Flam3 tonemapping | `playground.wgsl` |
| 6 | Reduce trail | `weights.rs`, `genome.rs`, `weights.json` |
| 7 | Curated seeds | `genomes/seeds/`, `genome.rs`, `main.rs` |
| 8 | Multi-radius bloom | `playground.wgsl` |
| 9 | Integration test | verification only |

## Phase 2 Reminder

Phase 2 is COMMITTED — not optional. After Phase 1 is complete:
1. Full affine transforms (6-param matrix replaces angle+scale)
2. Flam3 XML importer
3. Per-genome 256-entry color palettes
4. Parametric variations
5. Fitness-biased mutation

See `docs/plans/2026-03-07-electric-sheep-quality-design.md` for Phase 2 design.
