# Rendering Pipeline v2 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modernize the fractal flame rendering pipeline with subgroup atomics, AgX tonemapping, Dual Kawase bloom, Quasi-Monte Carlo sampling, and Jacobian importance sampling — delivering dramatically higher visual quality and performance.

**Architecture:** Four independent phases, each shipping a working improvement. Phase 1 targets compute perf (subgroup atomics). Phase 2 targets visual quality (AgX + Dual Kawase bloom via compute). Phase 3 targets sampling efficiency (Sobol QMC + Jacobian importance sampling). Phase 4 is a stretch goal (ReSTIR reservoir accumulation).

**Tech Stack:** Rust, wgpu 28, WGSL (with `enable subgroups;` extension), winit 0.30

**Source Research:** `.plans/gemini-rendering-research-results.md` (81-citation Gemini deep research report)

**Project Rules:**
- NO MAGIC NUMBERS — every tunable goes in `weights.json` → `RuntimeConfig` → uniform buffer
- NO `#[allow(dead_code)]` — wire it up or remove it
- Update `docs/` alongside code changes
- All config fields need `default_*` functions in `src/weights.rs` and entries in `weights.json` `_config` + `_config_doc`

---

## File Structure

### New Files
- `bloom_downsample.wgsl` — Dual Kawase downsample compute shader
- `bloom_upsample.wgsl` — Dual Kawase upsample compute shader
- `post_process.wgsl` — Compute-based post-processing (velocity blur, DOF, tonemapping)
- `src/bloom.rs` — Bloom pipeline setup (mip chain textures, bind groups, dispatch)
- `src/post_process.rs` — Post-processing compute pipeline setup
- `src/sobol.rs` — Sobol sequence direction numbers and generator for QMC

### Modified Files
- `flame_compute.wgsl` — Add subgroup atomics to splatting, Sobol RNG, Jacobian weighting
- `playground.wgsl` — Replace inline post-processing with simple passthrough (reads from post_process output)
- `accumulation.wgsl` — No changes in Phase 1-3
- `src/main.rs` — Wire new pipelines, create bloom mip chain, add indirect dispatch buffer
- `src/weights.rs` — Add config fields for AgX, bloom mip levels, Sobol, Jacobian
- `weights.json` — Add corresponding config values
- `docs/rendering/tonemapping.md` — Update for AgX
- `docs/rendering/post-effects.md` — Update for Dual Kawase bloom
- `docs/rendering/chaos-game.md` — Update for subgroups, QMC, Jacobian

---

## Phase 1: Subgroup Atomic Reduction (Performance)

**Why:** The single biggest perf bottleneck is atomic contention on the histogram buffer. Subgroup ops reduce atomic traffic by 32x (Apple Silicon) by aggregating within the warp before writing.

**Dependency:** Requires wgpu subgroups support. Check `device.features()` for `Features::SUBGROUP` at runtime; fall back to current path if unavailable.

### Task 1: Add subgroup feature detection and config

**Files:**
- Modify: `src/main.rs` (GPU device creation, ~line 400-450)
- Modify: `src/weights.rs` (add `use_subgroup_atomics` config field)
- Modify: `weights.json` (add config + doc entries)

- [ ] **Step 1: Add config field to RuntimeConfig**

In `src/weights.rs`, add to `RuntimeConfig`:
```rust
#[serde(default = "default_use_subgroup_atomics")]
pub use_subgroup_atomics: bool,
```
And the default function:
```rust
fn default_use_subgroup_atomics() -> bool { true }
```

- [ ] **Step 2: Add to weights.json**

Add to `_config`:
```json
"use_subgroup_atomics": true
```
Add to `_config_doc`:
```json
"use_subgroup_atomics": "Use WGSL subgroup operations for atomic reduction in compute shader (default true, falls back if unsupported)"
```

- [ ] **Step 3: Check subgroup support at device creation**

In `src/main.rs` where the wgpu device is created, log whether subgroups are available:
```rust
let has_subgroups = adapter.features().contains(wgpu::Features::SUBGROUP);
log::info!("[gpu] Subgroup support: {has_subgroups}");
```
Store `has_subgroups` in the `Gpu` struct so the compute pipeline can branch on it.

- [ ] **Step 4: Build and run tests**

Run: `cargo test --release`
Expected: 89 tests pass, no compilation errors.

- [ ] **Step 5: Commit**

```bash
git add src/weights.rs src/main.rs weights.json
git commit -m "feat: add subgroup feature detection and config"
```

### Task 2: Implement subgroup atomic splatting in compute shader

**Files:**
- Modify: `flame_compute.wgsl` (splatting section, ~lines 340-395)
- Modify: `src/main.rs` (conditionally select shader variant or use `enable subgroups;`)

- [ ] **Step 1: Create subgroup-enabled splat function**

In `flame_compute.wgsl`, at the top of the file, add:
```wgsl
enable subgroups;
```

Replace the direct `atomicAdd` calls in the bilinear splatting section (~line 370-380) with a subgroup-reduced version. The pattern:

```wgsl
// Before: each thread writes independently
// atomicAdd(&histogram[bi], u32(wt * 1000.0));

// After: subgroup reduction first
fn subgroup_splat(bi: u32, val: u32) {
    // All threads in subgroup that target the same bin get reduced
    let sum = subgroupAdd(val);
    if (subgroupElect()) {
        atomicAdd(&histogram[bi], sum);
    }
}
```

However, `subgroupAdd` reduces ALL threads in the subgroup, not just those targeting the same bin. For bin-specific reduction, the approach is:

For each of the 7 channels (density, R, G, B, vx, vy, depth), and for each of the 4 bilinear neighbors:
1. Compute the target bin index `bi`
2. Use `subgroupAdd` on the value — this sums across the entire subgroup
3. `subgroupElect()` picks one thread to do the write

This works when threads in the same subgroup frequently target the same pixel (which they do in contractive IFS — the attractor core). When they don't, it's still correct because each thread contributes its own unique value to the sum at its own bin. The key insight: we need to group by bin index.

The correct implementation uses a loop over unique bin indices within the subgroup:

```wgsl
// For each bilinear neighbor pixel:
let target_bin = compute_bin_index(px, py);
let my_density = u32(wt * 1000.0);

// Broadcast each thread's target bin to the subgroup
// Threads with matching bins get their values summed
var remaining = subgroupBallot(true);  // all active threads
loop {
    if (all(remaining == vec4(0u))) { break; }
    // Pick the first active thread's bin as the "leader" bin
    let leader_bin = subgroupBroadcastFirst(target_bin);
    let match_mask = subgroupBallot(target_bin == leader_bin);

    // Sum values only from threads that match this bin
    let contrib = select(0u, my_density, target_bin == leader_bin);
    let total = subgroupAdd(contrib);

    // One thread writes the aggregated sum
    if (subgroupElect() && target_bin == leader_bin) {
        atomicAdd(&histogram[leader_bin], total);
    }

    // Remove matched threads from remaining set
    remaining = remaining & ~match_mask;
}
```

Apply this pattern to all 7 channels of the histogram write. The bilinear splatting writes to up to 4 pixel neighbors, so this loop runs for each neighbor.

**Important:** This is a significant shader rewrite. The `enable subgroups;` directive may cause compilation failure on hardware that doesn't support it. For graceful fallback:
- Keep two versions of the splatting function (with/without subgroups)
- Select the correct compute shader source string in Rust based on `has_subgroups`

- [ ] **Step 2: Add shader variant selection in Rust**

In `src/main.rs`, in `load_compute_source()` or wherever the compute shader is loaded:
```rust
fn load_compute_source(use_subgroups: bool) -> String {
    let mut src = fs::read_to_string(compute_path())
        .unwrap_or_else(|_| include_str!("../flame_compute.wgsl").to_string());
    if !use_subgroups {
        // Strip the "enable subgroups;" line for fallback
        src = src.replace("enable subgroups;", "// subgroups disabled");
        // Replace subgroup splat calls with direct atomicAdd
        // (Use a preprocessor-style flag in the shader)
    }
    src
}
```

A cleaner approach: use a `const USE_SUBGROUPS: bool` override injected at the top of the shader source.

- [ ] **Step 3: Build, run app, verify no visual regression**

Run: `cargo build --release && cargo run --release`
Expected: Compiles cleanly. Visual output identical to before (subgroups reduce to the same mathematical result). Performance improvement visible in frame time.

- [ ] **Step 4: Commit**

```bash
git add flame_compute.wgsl src/main.rs
git commit -m "feat: subgroup atomic reduction for histogram splatting"
```

### Task 3: Verify and measure performance improvement

**Files:**
- Modify: `src/main.rs` (add frame time logging)

- [ ] **Step 1: Add frame time measurement**

In the render loop, measure and log frame times every 60 frames:
```rust
if self.frame_count % 60 == 0 {
    let elapsed = self.last_fps_time.elapsed();
    let fps = 60.0 / elapsed.as_secs_f64();
    log::info!("[perf] {fps:.1} fps ({:.1}ms/frame)", elapsed.as_millis() as f64 / 60.0);
    self.last_fps_time = std::time::Instant::now();
}
```

- [ ] **Step 2: Test with subgroups enabled and disabled**

Toggle `use_subgroup_atomics` in `weights.json` between `true` and `false`. Compare frame times.

- [ ] **Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat: add frame time logging for perf measurement"
```

---

## Phase 2: AgX Tonemapping + Dual Kawase Bloom (Visual Quality)

**Why:** ACES causes "Notorious 6" color shifts on saturated fractal colors. AgX desaturates gracefully to white. Dual Kawase bloom replaces our expensive 3-radius Gaussian with a cascaded O(log N) approach that's smoother and faster.

### Task 4: Implement AgX tonemapping in display shader

**Files:**
- Modify: `playground.wgsl` (replace ACES function, ~line 34-38)
- Modify: `src/weights.rs` (add `tonemap_mode` value 2 for AgX)

- [ ] **Step 1: Add AgX function to playground.wgsl**

Add after the existing `aces_tonemap` function:

```wgsl
// AgX view transform (Troy Sobotka, Blender 4.0)
// Analytical polynomial approximation — no LUT needed.
// Gracefully desaturates to white at high exposure, avoiding "Notorious 6" color shifts.
const AGX_INSET: mat3x3<f32> = mat3x3<f32>(
    vec3(0.842479062253094,  0.0423282422610123, 0.0423756549057051),
    vec3(0.0784335999999992, 0.878468636469772,  0.0784336),
    vec3(0.0792237451477643, 0.0791661274605434, 0.879142973793104)
);

const AGX_OUTSET: mat3x3<f32> = mat3x3<f32>(
    vec3(1.19687900512017,   -0.0528968517574562, -0.0529716355144438),
    vec3(-0.0980208811401368, 1.15190312990417,   -0.0980434501171241),
    vec3(-0.0990297440797205, -0.0989611768448433, 1.15107367264116)
);

fn agx_tonemap(color: vec3<f32>) -> vec3<f32> {
    let agx = AGX_INSET * max(color, vec3(0.0));

    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var log_col = clamp(
        vec3(log2(max(agx.x, 1e-10)), log2(max(agx.y, 1e-10)), log2(max(agx.z, 1e-10))),
        vec3(min_ev), vec3(max_ev)
    );
    log_col = (log_col - min_ev) / (max_ev - min_ev);

    // 6th-order polynomial sigmoid approximation
    let x2 = log_col * log_col;
    let x4 = x2 * x2;
    let mapped = 15.5 * x4 * x2
               - 40.14 * x4 * log_col
               + 31.96 * x4
               - 6.868 * x2 * log_col
               + 0.4298 * x2
               + 0.1191 * log_col
               - 0.00232;

    return AGX_OUTSET * mapped;
}
```

- [ ] **Step 2: Wire tonemap_mode = 2 to AgX**

In the fragment shader's tonemapping section, add the AgX branch:
```wgsl
let tonemap_mode = u32(u.extra4.y);
var mapped: vec3<f32>;
if (tonemap_mode == 2u) {
    // AgX — apply to linear HDR color before gamma
    mapped = agx_tonemap(hdr_color);
} else if (tonemap_mode == 1u) {
    // ACES filmic
    mapped = vec3(aces_tonemap(hdr_color.x), aces_tonemap(hdr_color.y), aces_tonemap(hdr_color.z));
} else {
    // sqrt-log (default)
    mapped = sqrt(log_color);
}
```

- [ ] **Step 3: Update weights.json and docs**

Set default `tonemap_mode` to `2` (AgX) in `weights.json`.
Update `_config_doc`:
```json
"tonemap_mode": "Tonemapping mode: 0 = sqrt-log, 1 = ACES filmic, 2 = AgX (default 2)"
```
Update `docs/rendering/tonemapping.md` to document AgX and the pipeline ordering.

- [ ] **Step 4: Build, test, verify visually**

Run: `cargo build --release && cargo run --release`
Expected: Colors at high brightness gracefully desaturate to white instead of shifting. Toggle between modes 0/1/2 in weights.json to compare.

- [ ] **Step 5: Commit**

```bash
git add playground.wgsl src/weights.rs weights.json docs/rendering/tonemapping.md
git commit -m "feat: add AgX tonemapping (mode 2) as new default"
```

### Task 5: Create Dual Kawase bloom compute shaders

**Files:**
- Create: `bloom_downsample.wgsl`
- Create: `bloom_upsample.wgsl`

- [ ] **Step 1: Write downsample shader**

The Dual Kawase downsample samples the center pixel plus 4 diagonal neighbors offset by half-pixel, producing a 5-tap weighted average. Create `bloom_downsample.wgsl`:

```wgsl
// Dual Kawase Bloom — Downsample Pass
// Reads from source mip, writes to next (half-resolution) mip.
// Uses bilinear-filtered texture sampling at half-pixel offsets
// to get a free 4-pixel average per tap.

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;  // RGBA linear
@group(0) @binding(3) var<uniform> params: vec4<f32>;  // dst_width, dst_height, 0, 0

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_w = u32(params.x);
    let dst_h = u32(params.y);
    if (gid.x >= dst_w || gid.y >= dst_h) { return; }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2(params.x, params.y);
    let texel = 1.0 / vec2(params.x, params.y);

    // Center sample (weight 4/8)
    let center = textureSampleLevel(src_tex, src_sampler, uv, 0.0).rgb;
    // 4 diagonal samples at half-pixel offset (weight 1/8 each)
    let tl = textureSampleLevel(src_tex, src_sampler, uv + vec2(-1.0, -1.0) * texel, 0.0).rgb;
    let tr = textureSampleLevel(src_tex, src_sampler, uv + vec2( 1.0, -1.0) * texel, 0.0).rgb;
    let bl = textureSampleLevel(src_tex, src_sampler, uv + vec2(-1.0,  1.0) * texel, 0.0).rgb;
    let br = textureSampleLevel(src_tex, src_sampler, uv + vec2( 1.0,  1.0) * texel, 0.0).rgb;

    let result = center * 0.5 + (tl + tr + bl + br) * 0.125;

    let idx = (gid.y * dst_w + gid.x) * 4u;
    dst[idx]      = result.x;
    dst[idx + 1u] = result.y;
    dst[idx + 2u] = result.z;
    dst[idx + 3u] = 1.0;
}
```

- [ ] **Step 2: Write upsample shader**

The upsample pass reads the lower mip and blends with the current mip. Create `bloom_upsample.wgsl`:

```wgsl
// Dual Kawase Bloom — Upsample Pass
// Reads from lower (smaller) mip, samples 8 neighbors + center,
// blends additively into the current (larger) mip.

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<f32>;  // dst_width, dst_height, bloom_intensity, 0

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_w = u32(params.x);
    let dst_h = u32(params.y);
    if (gid.x >= dst_w || gid.y >= dst_h) { return; }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2(params.x, params.y);
    let texel = 1.0 / vec2(params.x, params.y);

    // 8 samples in a plus and diagonal pattern (Kawase tent filter)
    let s0 = textureSampleLevel(src_tex, src_sampler, uv + vec2(-1.0, -1.0) * texel, 0.0).rgb;
    let s1 = textureSampleLevel(src_tex, src_sampler, uv + vec2( 0.0, -1.0) * texel, 0.0).rgb;
    let s2 = textureSampleLevel(src_tex, src_sampler, uv + vec2( 1.0, -1.0) * texel, 0.0).rgb;
    let s3 = textureSampleLevel(src_tex, src_sampler, uv + vec2(-1.0,  0.0) * texel, 0.0).rgb;
    let s4 = textureSampleLevel(src_tex, src_sampler, uv + vec2( 1.0,  0.0) * texel, 0.0).rgb;
    let s5 = textureSampleLevel(src_tex, src_sampler, uv + vec2(-1.0,  1.0) * texel, 0.0).rgb;
    let s6 = textureSampleLevel(src_tex, src_sampler, uv + vec2( 0.0,  1.0) * texel, 0.0).rgb;
    let s7 = textureSampleLevel(src_tex, src_sampler, uv + vec2( 1.0,  1.0) * texel, 0.0).rgb;

    // Corners weight 1/12, edges weight 2/12
    let bloom = (s0 + s2 + s5 + s7) / 12.0 + (s1 + s3 + s4 + s6) / 6.0;
    let intensity = params.z;

    let idx = (gid.y * dst_w + gid.x) * 4u;
    // Additive blend with existing content
    dst[idx]      = dst[idx]      + bloom.x * intensity;
    dst[idx + 1u] = dst[idx + 1u] + bloom.y * intensity;
    dst[idx + 2u] = dst[idx + 2u] + bloom.z * intensity;
}
```

- [ ] **Step 3: Commit**

```bash
git add bloom_downsample.wgsl bloom_upsample.wgsl
git commit -m "feat: add Dual Kawase bloom downsample/upsample compute shaders"
```

### Task 6: Create bloom pipeline in Rust

**Files:**
- Create: `src/bloom.rs`
- Modify: `src/main.rs` (add `mod bloom;`, wire into render loop)
- Modify: `src/weights.rs` (add `bloom_mip_levels` config)
- Modify: `weights.json`

- [ ] **Step 1: Add config fields**

In `src/weights.rs`:
```rust
#[serde(default = "default_bloom_mip_levels")]
pub bloom_mip_levels: u32,
```
```rust
fn default_bloom_mip_levels() -> u32 { 5 }
```

In `weights.json` `_config`: `"bloom_mip_levels": 5`
In `_config_doc`: `"bloom_mip_levels": "Number of Dual Kawase bloom mip levels, 3-6 (default 5)"`

- [ ] **Step 2: Create `src/bloom.rs`**

This module manages:
- A chain of `bloom_mip_levels` textures at decreasing resolution (full → 1/2 → 1/4 → ... → 1/2^N)
- Corresponding storage buffers for compute shader output
- Downsample pipeline: dispatches `bloom_downsample.wgsl` for each mip level
- Upsample pipeline: dispatches `bloom_upsample.wgsl` in reverse order
- Takes the HDR framebuffer as input, produces bloom-added output

The struct:
```rust
pub struct BloomPipeline {
    downsample_pipeline: wgpu::ComputePipeline,
    upsample_pipeline: wgpu::ComputePipeline,
    mip_textures: Vec<wgpu::Texture>,
    mip_views: Vec<wgpu::TextureView>,
    mip_buffers: Vec<wgpu::Buffer>,
    params_buffers: Vec<wgpu::Buffer>,
    downsample_bind_groups: Vec<wgpu::BindGroup>,
    upsample_bind_groups: Vec<wgpu::BindGroup>,
    sampler: wgpu::Sampler,
    mip_count: u32,
}
```

Methods:
- `new(device, width, height, mip_count, bloom_intensity)` — creates all resources
- `resize(device, width, height)` — recreates mip chain on window resize
- `dispatch(encoder, source_texture)` — runs the full down→up cascade

- [ ] **Step 3: Wire into main.rs render loop**

In `src/main.rs`:
- Add `mod bloom;`
- After accumulation and CDF passes, before the fragment pass:
  1. Run bloom downsample cascade
  2. Run bloom upsample cascade
  3. The final mip0 buffer contains the bloom-composited image
- Pass the bloom output to the fragment shader (or read it as an additional binding)

- [ ] **Step 4: Remove old bloom from fragment shader**

In `playground.wgsl`, remove the inline 3-radius bloom sampling code. The bloom is now applied by the compute pipeline before the fragment pass.

- [ ] **Step 5: Build, test, compare bloom quality**

Run: `cargo build --release && cargo run --release`
Expected: Smoother, more natural bloom glow. No banding artifacts. Better performance.

- [ ] **Step 6: Commit**

```bash
git add src/bloom.rs src/main.rs bloom_downsample.wgsl bloom_upsample.wgsl playground.wgsl src/weights.rs weights.json
git commit -m "feat: Dual Kawase bloom via compute pipeline, replace 3-radius Gaussian"
```

---

## Phase 3: Quasi-Monte Carlo + Jacobian Importance Sampling (Efficiency)

**Why:** Sobol sequences improve convergence from O(1/√N) to ~O(1/N) — 2x quality at the same sample count. Jacobian weighting allocates more iterations to expansive transforms (visible tendrils) and fewer to contractive ones (subpixel singularities that waste compute).

### Task 7: Implement Sobol sequence generator in WGSL

**Files:**
- Create: `src/sobol.rs` (direction numbers, generates uniform buffer data)
- Modify: `flame_compute.wgsl` (add Sobol generator, use for transform selection)
- Modify: `src/main.rs` (create and bind Sobol direction numbers buffer)
- Modify: `src/weights.rs` (add `use_quasi_random` config)
- Modify: `weights.json`

- [ ] **Step 1: Add config field**

In `src/weights.rs`:
```rust
#[serde(default = "default_use_quasi_random")]
pub use_quasi_random: bool,
```
```rust
fn default_use_quasi_random() -> bool { true }
```

In `weights.json` `_config`: `"use_quasi_random": true`
In `_config_doc`: `"use_quasi_random": "Use Sobol quasi-random sequences for transform selection instead of PCG (default true)"`

- [ ] **Step 2: Create `src/sobol.rs`**

This module provides the 32 direction numbers for a 1D Sobol sequence. The GPU shader will use these to generate low-discrepancy values via bitwise XOR with `countTrailingZeros`:

```rust
/// Standard Sobol direction numbers for dimension 1.
/// The shader computes: sobol_value ^= direction_numbers[countTrailingZeros(index)]
pub const SOBOL_DIRECTION_NUMBERS: [u32; 32] = [
    0x80000000, 0x40000000, 0x20000000, 0x10000000,
    0x08000000, 0x04000000, 0x02000000, 0x01000000,
    0x00800000, 0x00400000, 0x00200000, 0x00100000,
    0x00080000, 0x00040000, 0x00020000, 0x00010000,
    0x00008000, 0x00004000, 0x00002000, 0x00001000,
    0x00000800, 0x00000400, 0x00000200, 0x00000100,
    0x00000080, 0x00000040, 0x00000020, 0x00000010,
    0x00000008, 0x00000004, 0x00000002, 0x00000001,
];

/// Create the GPU buffer for Sobol direction numbers.
pub fn create_direction_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sobol_directions"),
        contents: bytemuck::cast_slice(&SOBOL_DIRECTION_NUMBERS),
        usage: wgpu::BufferUsages::STORAGE,
    })
}
```

- [ ] **Step 3: Add Sobol generator to compute shader**

In `flame_compute.wgsl`, add a new binding and generator function:

```wgsl
@group(0) @binding(6) var<storage, read> sobol_dirs: array<u32>;

fn sobol_sample(index: u32) -> f32 {
    var result = 0u;
    var i = index;
    var dim = 0u;
    loop {
        if (i == 0u) { break; }
        if ((i & 1u) != 0u) {
            result = result ^ sobol_dirs[dim];
        }
        i = i >> 1u;
        dim = dim + 1u;
    }
    return f32(result) / 4294967296.0;
}
```

In the main iteration loop, replace `randf(&seed)` for transform selection with:
```wgsl
let sobol_idx = u.frame * 200u + u32(i);  // unique per frame+iteration
let r = sobol_sample(sobol_idx + gid);
```

Keep PCG for spatial jitter and other non-critical randomness.

- [ ] **Step 4: Bind the direction buffer in Rust**

In `src/main.rs`, add the Sobol direction buffer to the compute bind group at binding 6.

- [ ] **Step 5: Build, test, compare convergence**

Run: `cargo build --release && cargo run --release`
Expected: Fractal structures resolve faster (less noise at the same iteration count). Toggle `use_quasi_random` in weights.json to compare.

- [ ] **Step 6: Commit**

```bash
git add src/sobol.rs flame_compute.wgsl src/main.rs src/weights.rs weights.json
git commit -m "feat: Sobol quasi-random sequence for transform selection"
```

### Task 8: Implement Jacobian-weighted importance sampling

**Files:**
- Modify: `flame_compute.wgsl` (transform selection CDF)
- Modify: `src/main.rs` (compute Jacobian weights on CPU, pass via uniform or transform buffer)
- Modify: `src/weights.rs` (add `jacobian_weight_strength` config)
- Modify: `weights.json`

- [ ] **Step 1: Add config field**

In `src/weights.rs`:
```rust
#[serde(default = "default_jacobian_weight_strength")]
pub jacobian_weight_strength: f32,
```
```rust
fn default_jacobian_weight_strength() -> f32 { 0.5 }
```

In `weights.json` `_config`: `"jacobian_weight_strength": 0.5`
In `_config_doc`: `"jacobian_weight_strength": "Blend factor for Jacobian determinant in transform selection probability: 0.0 = weight-only, 1.0 = fully Jacobian-biased (default 0.5)"`

- [ ] **Step 2: Compute Jacobian-adjusted weights on CPU**

In `src/main.rs`, when writing the transform buffer, compute the Jacobian determinant for each transform and blend it with the genetic weight:

```rust
// For each transform: det = |a*d - b*c|
// Higher det = more expansive = more visible detail = deserves more samples
let det = (xf.a * xf.d - xf.b * xf.c).abs();
let jacobian_weight = xf.weight * (1.0 - strength) + det * strength;
```

Write the Jacobian-adjusted weight into the transform buffer (field 0 — the weight field). This is already read by the shader for transform selection.

**Important:** Normalize the adjusted weights to sum to 1.0 before writing.

- [ ] **Step 3: Build, test**

Run: `cargo build --release && cargo run --release`
Expected: Expansive transforms (tendrils, broad sweeps) render with less noise. Contractive transforms (dense cores) use fewer iterations but remain sharp due to existing density.

- [ ] **Step 4: Commit**

```bash
git add src/main.rs src/weights.rs weights.json flame_compute.wgsl
git commit -m "feat: Jacobian-weighted importance sampling for transform selection"
```

---

## Phase 4 (Stretch): ReSTIR Reservoir Accumulation

**Why:** ReSTIR replaces our exponential moving average with statistically rigorous reservoir sampling. Infinite refinement when static, instant response when parameters mutate. Eliminates ghosting.

**Note:** This is the most complex change. It requires a reservoir state buffer, temporal reuse logic, and spatial reuse. Implementation details are well-documented in the Gemini research (`.plans/gemini-rendering-research-results.md`, section 2). This phase should be planned in detail after Phases 1-3 are validated.

### Task 9: Design ReSTIR reservoir buffer and accumulation shader

This task is a **design-only** task. Read the ReSTIR section of the Gemini research, study the existing `accumulation.wgsl`, and produce a detailed design document at `.plans/restir-design.md` covering:

- [ ] **Step 1: Define reservoir buffer layout** (per-pixel: sample, weight_sum, count, target_weight)
- [ ] **Step 2: Define temporal reuse logic** (how to re-evaluate historical samples under new parameters)
- [ ] **Step 3: Define spatial reuse pattern** (3x3 or 5x5 neighborhood sampling)
- [ ] **Step 4: Define mutation detection** (how to detect parameter changes that invalidate history)
- [ ] **Step 5: Write the design doc**
- [ ] **Step 6: Commit design doc**

Implementation of ReSTIR will be a separate plan following this design.

---

## Phase Summary

| Phase | Tasks | Key Deliverable | Expected Impact |
|-------|-------|----------------|-----------------|
| 1 | 1-3 | Subgroup atomic splatting | 5-30x reduction in atomic contention |
| 2 | 4-6 | AgX tonemapping + Dual Kawase bloom | No more color shifts, smoother bloom, lower bandwidth |
| 3 | 7-8 | Sobol QMC + Jacobian weighting | ~2x convergence speed, smarter sample allocation |
| 4 | 9 | ReSTIR design doc | Foundation for ghost-free temporal accumulation |

**Execution order:** Phase 1 → Phase 2 → Phase 3 → Phase 4. Each phase is independently valuable and can ship separately. Within each phase, tasks are sequential.

**Tag after each phase:**
- Phase 1: `v0.5.0-subgroup-atomics`
- Phase 2: `v0.5.1-agx-bloom`
- Phase 3: `v0.5.2-qmc-jacobian`
- Phase 4: `v0.5.3-restir-design`
