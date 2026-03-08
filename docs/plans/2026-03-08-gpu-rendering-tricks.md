# GPU Rendering Tricks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add six GPU rendering techniques to improve visual quality of the fractal flame renderer.

**Architecture:** Each trick is independent and can be implemented/toggled separately. All are controlled via weights.json config parameters with sensible defaults (off or subtle). Shaders are in WGSL, app logic in Rust with wgpu.

**Tech Stack:** Rust, wgpu, WGSL compute/fragment shaders

---

## Current Architecture Summary

- **Buffers:** Histogram (6 x u32 per pixel: density, R, G, B, vx, vy) -> Accumulation (6 x f32 per pixel, same layout) -> Display shader reads accumulation
- **Uniforms struct:** `time, frame, resolution[2], mouse[2], transform_count, has_final_xform, globals[4], kifs[4], extra[4], extra2[4], extra3[4]` — total 24 floats + 2 u32 packed fields
- **Uniform slots:** `extra.w` = symmetry (not from globals array), `extra3.w` = velocity_blur_max. The `globals[11]` (gamma) goes into `extra.w` in the display shader but `extra.w` in compute is symmetry — the two shaders have different `extra` interpretations.
- **Pipeline order:** Clear histogram -> Compute (chaos game splatting) -> Accumulation (blend histogram into persistent buffer) -> Render (display shader reads accumulation + prev_frame feedback)

## Uniform Slot Availability

The Uniforms struct currently has 5 vec4s of payload (`globals`, `kifs`, `extra`, `extra2`, `extra3`). All 20 slots are used. New parameters need either:
1. A new `extra4` vec4 added to the Uniforms struct (preferred — clean and extensible), or
2. Repacking existing values

**Decision:** Add `extra4: [f32; 4]` and `extra5: [f32; 4]` to hold new config params. This gives us 8 new uniform slots.

---

## Task 1: Extend Uniforms with extra4/extra5

**Complexity:** Low | **Risk:** Low — purely additive

### Files to modify

**`src/main.rs`** — Add extra4/extra5 to Uniforms struct and populate them:

```rust
// In the Uniforms struct (line ~29):
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    time: f32,
    frame: u32,
    resolution: [f32; 2],
    mouse: [f32; 2],
    transform_count: u32,
    has_final_xform: u32,
    globals: [f32; 4],
    kifs: [f32; 4],
    extra: [f32; 4],
    extra2: [f32; 4],
    extra3: [f32; 4],
    extra4: [f32; 4],   // jitter_amount, tonemap_mode, histogram_equalization, dof_strength
    extra5: [f32; 4],   // dof_focal_distance, spectral_rendering, temporal_reprojection, _reserved
}
```

Where uniforms are constructed (line ~1643):

```rust
extra4: [
    self.weights._config.jitter_amount,
    self.weights._config.tonemap_mode as f32,
    self.weights._config.histogram_equalization,
    self.weights._config.dof_strength,
],
extra5: [
    self.weights._config.dof_focal_distance,
    if self.weights._config.spectral_rendering { 1.0 } else { 0.0 },
    self.weights._config.temporal_reprojection,
    0.0, // reserved
],
```

**`src/weights.rs`** — Add new RuntimeConfig fields:

```rust
// Add to RuntimeConfig struct:
#[serde(default)]
pub jitter_amount: f32,           // 0.0 = off, 1.0 = full pixel jitter
#[serde(default)]
pub tonemap_mode: u32,            // 0 = current sqrt(log) curve, 1 = ACES filmic
#[serde(default)]
pub histogram_equalization: f32,  // 0.0 = off, 1.0 = full equalization
#[serde(default)]
pub dof_strength: f32,            // 0.0 = off
#[serde(default)]
pub dof_focal_distance: f32,      // 0.0 = auto (center of mass)
#[serde(default)]
pub spectral_rendering: bool,     // false = RGB palette, true = spectral
#[serde(default)]
pub temporal_reprojection: f32,   // 0.0 = off, 1.0 = full
```

No default_* functions needed — serde `#[serde(default)]` uses `Default` for the type (0.0 for f32, 0 for u32, false for bool), which means "off" for every feature.

**`flame_compute.wgsl`** and **`playground.wgsl`** — Add extra4/extra5 to Uniforms struct:

```wgsl
// Add to the Uniforms struct in both shaders:
    extra4: vec4<f32>,   // jitter_amount, tonemap_mode, histogram_equalization, dof_strength
    extra5: vec4<f32>,   // dof_focal_distance, spectral_rendering, temporal_reprojection, reserved
```

**`weights.json`** — Add defaults to `_config`:

```json
"jitter_amount": 0.0,
"tonemap_mode": 0,
"histogram_equalization": 0.0,
"dof_strength": 0.0,
"dof_focal_distance": 0.0,
"spectral_rendering": false,
"temporal_reprojection": 0.0
```

Also add to `_config_doc`:

```json
"jitter_amount": "Sub-pixel jitter for free supersampling (0.0=off, 1.0=full pixel, default 0.0)",
"tonemap_mode": "Tonemapping curve (0=sqrt-log, 1=ACES filmic, default 0)",
"histogram_equalization": "Adaptive density equalization strength (0.0=off, 1.0=full, default 0.0)",
"dof_strength": "Depth of field blur strength (0.0=off, default 0.0)",
"dof_focal_distance": "Focal distance for DoF (0.0=auto center of mass, default 0.0)",
"spectral_rendering": "Use spectral wavelength color mixing instead of RGB (default false)",
"temporal_reprojection": "Motion-compensated frame blending during morphs (0.0=off, 1.0=full, default 0.0)"
```

### Verification

```bash
cargo build 2>&1 | head -5
```

Should compile with no errors. Run the app — visually identical since all new params default to off/zero.

### Commit message

```
feat: add extra4/extra5 uniform slots for GPU rendering tricks

All new config params default to off (0.0/false) so behavior is unchanged.
Prepares uniform plumbing for jitter, ACES tonemapping, histogram
equalization, DoF, spectral rendering, and temporal reprojection.
```

---

## Task 2: Jittered Accumulation (Free Supersampling)

**Complexity:** Low | **Risk:** Low — tiny change, big visual win

### Concept

Each frame, add a random sub-pixel offset (< 1 pixel) to the screen projection in the compute shader. Over many accumulated frames, this effectively supersamples, giving free anti-aliasing on edges and fine detail.

### Files to modify

**`flame_compute.wgsl`** — Add jitter to the screen projection. Modify the screen coordinate calculation inside the symmetry loop (around line 502):

Find this block (appears twice — once for the main plot point, once for the bilateral mirror):

```wgsl
            let screen = (sym_p / zoom + vec2(0.5, 0.5)) * vec2<f32>(f32(w), f32(h));
```

Replace with:

```wgsl
            // Sub-pixel jitter for free supersampling via accumulation averaging
            let jitter_amount = u.extra4.x;
            let jitter_seed = gid.x * 3u + u.frame * 17u + u32(si) * 7u;
            var jitter_rng = jitter_seed * 747796405u + 2891336453u;
            let jx = (f32(pcg(&jitter_rng)) / 4294967295.0 - 0.5) * jitter_amount;
            let jy = (f32(pcg(&jitter_rng)) / 4294967295.0 - 0.5) * jitter_amount;
            let screen = (sym_p / zoom + vec2(0.5, 0.5)) * vec2<f32>(f32(w), f32(h)) + vec2(jx, jy);
```

And the mirror version:

```wgsl
                let mscreen = (mir_p / zoom + vec2(0.5, 0.5)) * vec2<f32>(f32(w), f32(h));
```

Replace with:

```wgsl
                let mscreen = (mir_p / zoom + vec2(0.5, 0.5)) * vec2<f32>(f32(w), f32(h)) + vec2(jx, jy);
```

Note: the jitter reuses the same `jx, jy` for the mirror point since they share the same symmetry slot — this is intentional so mirrored pairs stay coherent within a frame.

### Verification

```bash
cargo build 2>&1 | head -5
```

Visual test:
1. Set `"jitter_amount": 0.0` — should look identical to before
2. Set `"jitter_amount": 0.5` — edges of flame structures should appear slightly smoother after a few seconds of accumulation
3. Set `"jitter_amount": 1.0` — maximum smoothing, fine detail may soften slightly but edges are very clean
4. Zoom in on a thin tendril — with jitter, it should look anti-aliased rather than pixelated

### Commit message

```
feat: add jittered accumulation for free supersampling

Per-frame sub-pixel jitter on screen projection. The accumulation buffer
averages it out, producing smooth anti-aliased edges at zero perf cost.
Controlled by jitter_amount in weights.json (0.0=off, 1.0=full pixel).
```

---

## Task 3: HDR Filmic Tonemapping (ACES)

**Complexity:** Low | **Risk:** Low — self-contained display shader change

### Concept

Replace the current `sqrt(log/(log+2))` tonemapping curve with the industry-standard ACES filmic curve. ACES produces better highlight rolloff, deeper blacks, and more saturated midtones. Toggle via `tonemap_mode`.

### Files to modify

**`playground.wgsl`** — Add ACES function and modify the tonemapping section.

Add the ACES function near the top of the file (after the Uniforms struct):

```wgsl
// ACES filmic tonemapping curve (Krzysztof Narkowicz fit)
// These constants define the curve shape — they ARE the algorithm, not magic numbers.
fn aces_tonemap(x: f32) -> f32 {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
```

Then modify the log-density tonemapping section (around line 112-116). Replace:

```wgsl
    // ── Log-density tonemapping with contrast preservation ──
    // Density is stored as fixed-point * 1000 (from bilinear splatting)
    let density_hits = blur_density / 1000.0;
    // Use sqrt of log to spread out the dynamic range instead of saturating
    let log_density = log(1.0 + density_hits * flame_bright);
    let alpha = sqrt(log_density / (log_density + 2.0));
```

With:

```wgsl
    // ── Log-density tonemapping with contrast preservation ──
    // Density is stored as fixed-point * 1000 (from bilinear splatting)
    let density_hits = blur_density / 1000.0;
    let tonemap_mode = u32(u.extra4.y);

    // Log-density mapping (common to both curves)
    let log_density = log(1.0 + density_hits * flame_bright);

    // Select tonemapping curve
    var alpha: f32;
    if (tonemap_mode == 1u) {
        // ACES filmic — better highlight rolloff, deeper blacks
        alpha = aces_tonemap(log_density);
    } else {
        // Default sqrt-log curve
        alpha = sqrt(log_density / (log_density + 2.0));
    }
```

### Verification

```bash
cargo build 2>&1 | head -5
```

Visual test:
1. Set `"tonemap_mode": 0` — identical to current rendering
2. Set `"tonemap_mode": 1` — ACES curve: highlights roll off more gracefully, blacks are deeper, colors feel more "filmic" and saturated. Dense core areas should show more contrast against background.
3. Compare side by side (toggle quickly): ACES should feel richer without blowing out bright areas

### Commit message

```
feat: add ACES filmic tonemapping as alternative curve

Toggle with tonemap_mode in weights.json (0=sqrt-log default, 1=ACES).
ACES provides better highlight rolloff and deeper blacks for a more
cinematic look without crushing bright flame cores.
```

---

## Task 4: Depth of Field Blur

**Complexity:** Medium | **Risk:** Medium — requires adding a 7th accumulation channel

### Concept

Points already have a radial distance from origin. Store this as a "depth" channel in the accumulation buffer (7th channel: density-weighted distance). In the display shader, compute focal distance and apply variable-radius blur based on |depth - focal|.

### Files to modify

**`src/main.rs`** — Change buffer sizes from 6 channels to 7 channels per pixel.

In `create_histogram_buffer` (line ~791):

```rust
fn create_histogram_buffer(
    device: &wgpu::Device,
    w: u32,
    h: u32,
) -> wgpu::Buffer {
    let pixel_count = w.max(1) as u64 * h.max(1) as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("histogram"),
        size: pixel_count * 7 * 4, // 7 u32s per pixel (density, R, G, B, vx, vy, depth)
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
```

In `create_accumulation_buffer` (line ~805):

```rust
fn create_accumulation_buffer(
    device: &wgpu::Device,
    w: u32,
    h: u32,
) -> wgpu::Buffer {
    let pixel_count = w.max(1) as u64 * h.max(1) as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("accumulation"),
        size: pixel_count * 7 * 4, // 7 f32s per pixel (density, R, G, B, vx, vy, depth)
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
```

**`flame_compute.wgsl`** — Change all `6u` channel strides to `7u` and add depth splatting.

Update `splat_pixel` to accept and write depth:

```wgsl
// Splat helper — writes density + color + velocity + depth to one pixel
fn splat_pixel(bi: u32, wt: f32, ic: u32, ig: u32, ib: u32, ivx: i32, ivy: i32, idepth: u32) {
    atomicAdd(&histogram[bi],      u32(wt * 1000.0));
    atomicAdd(&histogram[bi + 1u], u32(f32(ic) * wt));
    atomicAdd(&histogram[bi + 2u], u32(f32(ig) * wt));
    atomicAdd(&histogram[bi + 3u], u32(f32(ib) * wt));
    atomicAdd(&histogram[bi + 4u], u32(i32(f32(ivx) * wt)));
    atomicAdd(&histogram[bi + 5u], u32(i32(f32(ivy) * wt)));
    atomicAdd(&histogram[bi + 6u], u32(f32(idepth) * wt));
}
```

Update `splat_point` to accept and pass depth:

```wgsl
fn splat_point(cx: i32, cy: i32, fx: f32, fy: f32, col: vec3<f32>, vel: vec2<f32>, depth: f32, w: u32, h: u32) {
    let ic = u32(col.x * 1000.0);
    let ig = u32(col.y * 1000.0);
    let ib = u32(col.z * 1000.0);
    let ivx = i32(vel.x * 10000.0);
    let ivy = i32(vel.y * 10000.0);
    let idepth = u32(depth * 1000.0);

    // Center pixel (always valid — caller checked bounds)
    splat_pixel((u32(cy) * w + u32(cx)) * 7u, (1.0 - fx) * (1.0 - fy), ic, ig, ib, ivx, ivy, idepth);

    // Only splat neighbors when sub-pixel offset is significant
    let nx = cx + 1;
    let ny = cy + 1;
    if (fx > 0.1 && nx < i32(w)) {
        splat_pixel((u32(cy) * w + u32(nx)) * 7u, fx * (1.0 - fy), ic, ig, ib, ivx, ivy, idepth);
    }
    if (fy > 0.1 && ny < i32(h)) {
        splat_pixel((u32(ny) * w + u32(cx)) * 7u, (1.0 - fx) * fy, ic, ig, ib, ivx, ivy, idepth);
    }
    if (fx > 0.1 && fy > 0.1 && nx < i32(w) && ny < i32(h)) {
        splat_pixel((u32(ny) * w + u32(nx)) * 7u, fx * fy, ic, ig, ib, ivx, ivy, idepth);
    }
}
```

At the call sites for `splat_point`, compute depth from `plot_p` and pass it:

```wgsl
            // Depth = radial distance from origin (in world space)
            let point_depth = length(sym_p);

            if (px_x >= 0 && px_x < i32(w) && px_y >= 0 && px_y < i32(h)) {
                let frac_x = screen.x - f32(px_x);
                let frac_y = screen.y - f32(px_y);
                splat_point(px_x, px_y, frac_x, frac_y, base_col, vel, point_depth, w, h);
            }
```

And the bilateral mirror call site:

```wgsl
                    splat_point(mpx_x, mpx_y, mfrac_x, mfrac_y, base_col, mvel, point_depth, w, h);
```

**`accumulation.wgsl`** — Change stride from 6u to 7u and add depth channel:

```wgsl
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(params.resolution.x);
    let h = u32(params.resolution.y);
    if (gid.x >= w || gid.y >= h) {
        return;
    }

    let px = gid.y * w + gid.x;
    let hist_idx = px * 7u;
    let accum_idx = px * 7u;

    // Read raw histogram for this frame
    let density = f32(histogram[hist_idx]);
    let r = f32(histogram[hist_idx + 1u]);
    let g = f32(histogram[hist_idx + 2u]);
    let b = f32(histogram[hist_idx + 3u]);
    let vx = f32(i32(histogram[hist_idx + 4u]));
    let vy = f32(i32(histogram[hist_idx + 5u]));
    let depth = f32(histogram[hist_idx + 6u]);

    // Exponential decay blend
    let decay = params.decay;
    accumulation[accum_idx]      = accumulation[accum_idx]      * decay + density;
    accumulation[accum_idx + 1u] = accumulation[accum_idx + 1u] * decay + r;
    accumulation[accum_idx + 2u] = accumulation[accum_idx + 2u] * decay + g;
    accumulation[accum_idx + 3u] = accumulation[accum_idx + 3u] * decay + b;
    accumulation[accum_idx + 4u] = accumulation[accum_idx + 4u] * decay + vx;
    accumulation[accum_idx + 5u] = accumulation[accum_idx + 5u] * decay + vy;
    accumulation[accum_idx + 6u] = accumulation[accum_idx + 6u] * decay + depth;
}
```

**`playground.wgsl`** — Change stride from 6u to 7u everywhere, read depth, apply DoF blur.

Update all `* 6u` index calculations to `* 7u`. Then add DoF blur after the velocity blur section but before the log-density tonemapping:

```wgsl
    // ── Read accumulation buffer (7 channels: density, R, G, B, vx, vy, depth) ──
    let buf_idx = (px.y * w + px.x) * 7u;
    let density = accumulation[buf_idx];
    let acc_r = accumulation[buf_idx + 1u];
    let acc_g = accumulation[buf_idx + 2u];
    let acc_b = accumulation[buf_idx + 3u];
```

Update velocity reads similarly (buf_idx + 4u and + 5u stay the same).

Update all the neighbor reads for edge detection and velocity blur to use `* 7u` stride.

After the velocity blur section and before tonemapping, add depth-of-field blur:

```wgsl
    // ── Depth of Field blur ──
    let dof_strength = u.extra4.w;
    if (dof_strength > 0.001) {
        let dof_focal = u.extra5.x;
        // Read depth channel (density-weighted distance, same fixed-point scale as color)
        let raw_depth = accumulation[buf_idx + 6u];
        let avg_depth = select(0.0, raw_depth / max(density, 1.0), density > 0.0);

        // Focal distance: if dof_focal is 0, use a fixed default of 1.0
        let focal = select(dof_focal, 1.0, dof_focal < 0.001);

        // Circle of confusion: proportional to distance from focal plane
        let coc = abs(avg_depth - focal) * dof_strength;
        let blur_radius = clamp(coc * 8.0, 0.0, 16.0);  // max 16px blur

        if (blur_radius > 0.5) {
            // Simple disc blur: 8 samples at blur_radius
            var dof_col = vec3(0.0);
            var dof_weight = 0.0;
            let texel = 1.0 / u.resolution;
            for (var si = 0; si < 8; si++) {
                let angle = f32(si) * 0.785398;  // TAU/8
                let offset = vec2(cos(angle), sin(angle)) * blur_radius * texel;
                let sample_uv = tex_uv + offset;
                let sample_px = vec2<u32>(
                    u32(clamp(sample_uv.x * f32(w), 0.0, f32(w) - 1.0)),
                    u32(clamp(sample_uv.y * f32(h), 0.0, f32(h) - 1.0))
                );
                let si_idx = (sample_px.y * w + sample_px.x) * 7u;
                let sd = accumulation[si_idx];
                let sr = accumulation[si_idx + 1u];
                let sg = accumulation[si_idx + 2u];
                let sb = accumulation[si_idx + 3u];
                let s_depth = accumulation[si_idx + 6u];
                let s_avg_depth = select(0.0, s_depth / max(sd, 1.0), sd > 0.0);
                // Weight by how out-of-focus the sample is (prevent sharp foreground from bleeding)
                let s_coc = abs(s_avg_depth - focal) * dof_strength;
                let w_sample = select(1.0, s_coc / max(coc, 0.01), s_coc < coc);
                if (sd > 0.0) {
                    let s_col = vec3(sr, sg, sb) / sd;
                    dof_col += s_col * w_sample;
                    dof_weight += w_sample;
                }
            }
            if (dof_weight > 0.0) {
                let dof_blend = clamp(coc * 2.0, 0.0, 1.0);
                let blurred = dof_col / dof_weight;
                // Replace raw_color with DoF-blurred version (before tonemapping)
                // Note: raw_color was already computed above, we blend into it
                // This requires making raw_color a var instead of let
            }
        }
    }
```

**Important:** To make this work, `raw_color` must be declared as `var` instead of `let`, and the DoF blend writes back into it:

Change:
```wgsl
    let raw_color = select(
```
To:
```wgsl
    var raw_color = select(
```

Then at the end of the DoF block:
```wgsl
                raw_color = mix(raw_color, blurred, dof_blend);
```

### Verification

```bash
cargo build 2>&1 | head -5
```

Visual test:
1. Set `"dof_strength": 0.0` — identical to before
2. Set `"dof_strength": 0.3, "dof_focal_distance": 1.0` — points near radial distance 1.0 from origin stay sharp, points far away or very close get blurred
3. Set `"dof_strength": 0.8` — strong bokeh-like blur on out-of-focus regions
4. Set `"dof_focal_distance": 0.0` — auto-focuses at distance 1.0 (the default fallback)
5. Try with different fractal structures — spirals should show depth nicely

### Commit message

```
feat: add depth of field blur via 7th accumulation channel

Store density-weighted radial distance as depth. Display shader computes
circle of confusion per pixel and applies disc blur proportional to
distance from focal plane. Controlled by dof_strength and
dof_focal_distance in weights.json.
```

---

## Task 5: Adaptive Histogram Equalization

**Complexity:** High | **Risk:** Medium — requires new GPU buffers, new compute pass, and shader plumbing

### Concept

After accumulation, compute a density histogram (256 bins) via GPU reduction, build a CDF via prefix sum, then use the CDF in the display shader to remap density values. This reveals structure in both sparse and dense regions simultaneously.

### Files to create/modify

**Create `histogram_cdf.wgsl`** — Two-pass compute shader: bin densities, then prefix-sum into CDF.

```wgsl
// histogram_cdf.wgsl — Compute density histogram + CDF for adaptive equalization
//
// Pass 1 (entry: "bin_densities"): Read accumulation buffer, bin log-densities into 256 bins
// Pass 2 (entry: "prefix_sum"): Compute prefix sum to build CDF, then normalize

struct HistogramParams {
    resolution: vec2<f32>,
    flame_brightness: f32,
    total_pixels: f32,
}

@group(0) @binding(0) var<storage, read> accumulation: array<f32>;
@group(0) @binding(1) var<storage, read_write> hist_bins: array<atomic<u32>>;  // 256 bins
@group(0) @binding(2) var<storage, read_write> cdf: array<f32>;               // 256 floats (normalized CDF)
@group(0) @binding(3) var<uniform> params: HistogramParams;

@compute @workgroup_size(16, 16)
fn bin_densities(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(params.resolution.x);
    let h = u32(params.resolution.y);
    if (gid.x >= w || gid.y >= h) {
        return;
    }

    let px = gid.y * w + gid.x;
    let accum_idx = px * 7u;
    let density = accumulation[accum_idx] / 1000.0;  // convert from fixed-point

    if (density < 0.001) {
        return;  // skip empty pixels
    }

    // Map log-density to bin index [0, 255]
    let log_d = log(1.0 + density * params.flame_brightness);
    let bin = u32(clamp(log_d / (log_d + 4.0) * 255.0, 0.0, 255.0));
    atomicAdd(&hist_bins[bin], 1u);
}

// Prefix sum pass — single workgroup, 256 threads
@compute @workgroup_size(256)
fn prefix_sum(@builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = lid.x;
    let count = f32(atomicLoad(&hist_bins[idx]));

    // Store count temporarily in cdf
    cdf[idx] = count;
    workgroupBarrier();

    // Hillis-Steele prefix sum (inclusive)
    for (var stride = 1u; stride < 256u; stride = stride * 2u) {
        var val = cdf[idx];
        if (idx >= stride) {
            val += cdf[idx - stride];
        }
        workgroupBarrier();
        cdf[idx] = val;
        workgroupBarrier();
    }

    // Normalize to [0, 1]
    let total = cdf[255u];
    if (total > 0.0) {
        cdf[idx] = cdf[idx] / total;
    }
}
```

**`src/main.rs`** — Add histogram CDF buffers, pipeline, bind group, and dispatch.

Add to `Gpu` struct:

```rust
    // Histogram equalization
    histogram_cdf_bin_pipeline: wgpu::ComputePipeline,
    histogram_cdf_sum_pipeline: wgpu::ComputePipeline,
    histogram_cdf_bind_group_layout: wgpu::BindGroupLayout,
    histogram_cdf_bind_group: wgpu::BindGroup,
    histogram_cdf_pipeline_layout: wgpu::PipelineLayout,
    hist_bins_buffer: wgpu::Buffer,       // 256 u32s
    cdf_buffer: wgpu::Buffer,             // 256 f32s
    histogram_cdf_uniform_buffer: wgpu::Buffer,
```

Create buffers in `Gpu::create`:

```rust
        // Histogram equalization buffers
        let hist_bins_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hist_bins"),
            size: 256 * 4,  // 256 u32s
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cdf_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cdf"),
            size: 256 * 4,  // 256 f32s
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let histogram_cdf_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("histogram_cdf_uniforms"),
            size: 16,  // 4 f32s: resolution.xy, flame_brightness, total_pixels
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
```

Create bind group layout:

```rust
        let histogram_cdf_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("histogram_cdf"),
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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

Create two pipelines from the same shader module (two entry points):

```rust
        let histogram_cdf_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("histogram_cdf"),
                bind_group_layouts: &[&histogram_cdf_bind_group_layout],
                push_constant_ranges: &[],
            });

        let histogram_cdf_src = fs::read_to_string(project_dir().join("histogram_cdf.wgsl"))
            .unwrap_or_else(|_| include_str!("../histogram_cdf.wgsl").to_string());
        let histogram_cdf_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("histogram_cdf"),
            source: wgpu::ShaderSource::Wgsl(histogram_cdf_src.into()),
        });

        let histogram_cdf_bin_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("histogram_cdf_bin"),
                layout: Some(&histogram_cdf_pipeline_layout),
                module: &histogram_cdf_module,
                entry_point: Some("bin_densities"),
                compilation_options: Default::default(),
                cache: None,
            });

        let histogram_cdf_sum_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("histogram_cdf_sum"),
                layout: Some(&histogram_cdf_pipeline_layout),
                module: &histogram_cdf_module,
                entry_point: Some("prefix_sum"),
                compilation_options: Default::default(),
                cache: None,
            });
```

Create bind group:

```rust
        let histogram_cdf_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("histogram_cdf"),
            layout: &histogram_cdf_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: accumulation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hist_bins_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cdf_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: histogram_cdf_uniform_buffer.as_entire_binding(),
                },
            ],
        });
```

In the render function, add between accumulation pass and render pass:

```rust
        // 2.75. Histogram equalization pass (if enabled)
        // Clear histogram bins
        encoder.clear_buffer(&self.hist_bins_buffer, 0, None);

        // Pass 1: bin densities
        {
            let mut hpass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("histogram_bin"),
                    ..Default::default()
                },
            );
            hpass.set_pipeline(&self.histogram_cdf_bin_pipeline);
            hpass.set_bind_group(0, &self.histogram_cdf_bind_group, &[]);
            let wg_x = (self.config.width + 15) / 16;
            let wg_y = (self.config.height + 15) / 16;
            hpass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Pass 2: prefix sum -> CDF
        {
            let mut hpass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("histogram_cdf"),
                    ..Default::default()
                },
            );
            hpass.set_pipeline(&self.histogram_cdf_sum_pipeline);
            hpass.set_bind_group(0, &self.histogram_cdf_bind_group, &[]);
            hpass.dispatch_workgroups(1, 1, 1);  // single workgroup, 256 threads
        }
```

**`playground.wgsl`** — Add CDF buffer binding and use it for equalization.

Add new binding:

```wgsl
@group(0) @binding(5) var<storage, read> cdf: array<f32>;  // 256 normalized CDF values
```

After computing `alpha` (the tonemapped density), add equalization blend:

```wgsl
    // ── Adaptive histogram equalization ──
    let hist_eq = u.extra4.z;
    if (hist_eq > 0.001) {
        // Map current alpha to CDF bin and look up equalized value
        let bin = u32(clamp(alpha * 255.0, 0.0, 255.0));
        let equalized = cdf[bin];
        // Blend between linear (alpha) and equalized based on config
        alpha = mix(alpha, equalized, hist_eq);
    }
```

Add the cdf_buffer to the render bind group layout (binding 5) and bind group entries in `src/main.rs`:

In the render bind group layout, add:

```rust
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
```

In both `bind_group_a` and `bind_group_b` creation (and the resize recreation), add:

```rust
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cdf_buffer.as_entire_binding(),
                },
```

Write histogram CDF uniforms each frame (in the uniform writing section):

```rust
                let hist_cdf_uniforms: [f32; 4] = [
                    gpu.config.width as f32,
                    gpu.config.height as f32,
                    self.globals[3], // flame_brightness
                    (gpu.config.width * gpu.config.height) as f32,
                ];
                gpu.queue.write_buffer(
                    &gpu.histogram_cdf_uniform_buffer,
                    0,
                    bytemuck::cast_slice(&hist_cdf_uniforms),
                );
```

Don't forget to recreate the histogram CDF bind group in `resize()` since the accumulation buffer is recreated there.

### Verification

```bash
cargo build 2>&1 | head -5
```

Visual test:
1. Set `"histogram_equalization": 0.0` — identical to before
2. Set `"histogram_equalization": 0.3` — subtle: sparse outer regions become slightly more visible, dense cores are slightly tamed
3. Set `"histogram_equalization": 1.0` — full equalization: dramatic — sparse tendrils become as visible as dense cores, reveals hidden structure
4. Compare with a fractal that has both very dense cores and wispy outer regions — equalization should make both visible simultaneously

### Commit message

```
feat: add adaptive histogram equalization for density remapping

Two-pass GPU compute: bin log-densities into 256 bins, prefix-sum into
CDF, then remap density in display shader. Reveals structure in both
sparse and dense regions. Controlled by histogram_equalization in
weights.json (0.0=off, 1.0=full, blends with linear).
```

---

## Task 6: Spectral Color Rendering

**Complexity:** High | **Risk:** Medium — changes color pipeline, must be carefully toggled

### Concept

Instead of RGB palette lookup and mixing, define colors as dominant wavelengths (380-780nm). Use CIE XYZ color matching functions approximated analytically, then XYZ -> sRGB. Color mixing happens in spectral space (additive wavelength blending), which prevents muddy browns from complementary color mixing.

### Files to modify

**`flame_compute.wgsl`** — Add spectral color functions and modify palette lookup when spectral mode is on.

Add spectral helper functions:

```wgsl
// ── Spectral Color Rendering ──
// Attempt to convert palette index to dominant wavelength, then to XYZ -> sRGB.
// This prevents muddy browns from complementary color mixing.

// Approximate CIE XYZ color matching functions (Wyman et al. 2013)
// These are standard scientific approximations — the constants define the spectral response curves.
fn cie_x(wavelength: f32) -> f32 {
    let t1 = (wavelength - 442.0) * select(0.0624, 0.0374, wavelength < 442.0);
    let t2 = (wavelength - 599.8) * select(0.0264, 0.0323, wavelength < 599.8);
    let t3 = (wavelength - 501.1) * select(0.0490, 0.0382, wavelength < 501.1);
    return 0.362 * exp(-0.5 * t1 * t1) + 1.056 * exp(-0.5 * t2 * t2) - 0.065 * exp(-0.5 * t3 * t3);
}

fn cie_y(wavelength: f32) -> f32 {
    let t1 = (wavelength - 568.8) * select(0.0213, 0.0247, wavelength < 568.8);
    let t2 = (wavelength - 530.9) * select(0.0613, 0.0322, wavelength < 530.9);
    return 0.821 * exp(-0.5 * t1 * t1) + 0.286 * exp(-0.5 * t2 * t2);
}

fn cie_z(wavelength: f32) -> f32 {
    let t1 = (wavelength - 437.0) * select(0.0845, 0.0278, wavelength < 437.0);
    let t2 = (wavelength - 459.0) * select(0.0385, 0.0725, wavelength < 459.0);
    return 1.217 * exp(-0.5 * t1 * t1) + 0.681 * exp(-0.5 * t2 * t2);
}

// Map palette index [0,1] to wavelength [380,780]nm
fn palette_to_wavelength(t: f32) -> f32 {
    return 380.0 + fract(t) * 400.0;  // 380nm (violet) to 780nm (red)
}

// CIE XYZ to linear sRGB (D65 standard illuminant)
// These are the standard conversion matrix values — they define the sRGB color space.
fn xyz_to_srgb(xyz: vec3<f32>) -> vec3<f32> {
    return vec3(
         3.2406 * xyz.x - 1.5372 * xyz.y - 0.4986 * xyz.z,
        -0.9689 * xyz.x + 1.8758 * xyz.y + 0.0415 * xyz.z,
         0.0557 * xyz.x - 0.2040 * xyz.y + 1.0570 * xyz.z
    );
}

// Spectral palette: convert color index to visible spectrum via CIE XYZ
fn palette_spectral(t: f32) -> vec3<f32> {
    let wavelength = palette_to_wavelength(t);
    let xyz = vec3(cie_x(wavelength), cie_y(wavelength), cie_z(wavelength));
    return max(xyz_to_srgb(xyz), vec3(0.0));
}
```

Modify the color lookup section (around line 489). Replace:

```wgsl
        let base_col = palette(plot_color + u.extra.x) * lum;
```

With:

```wgsl
        var base_col: vec3<f32>;
        if (u.extra5.y > 0.5) {
            // Spectral rendering: wavelength-based color with CIE XYZ conversion
            base_col = palette_spectral(plot_color + u.extra.x) * lum;
        } else {
            // Standard RGB palette lookup
            base_col = palette(plot_color + u.extra.x) * lum;
        }
```

### Key insight for color mixing

The spectral advantage comes from how colors accumulate in the histogram. When two points with different color indices land on the same pixel:
- **RGB mode:** `red(1,0,0) + cyan(0,1,1) = gray(0.5,0.5,0.5)` — muddy
- **Spectral mode:** `650nm + 490nm` → each contributes its own CIE XYZ spectrum independently → the result looks like a warm pink-white, not gray

This works automatically because the histogram accumulates the splatted RGB values. In spectral mode, those RGB values come from wavelength conversion, so the additive mixing in the histogram produces physically-motivated color blends.

### Verification

```bash
cargo build 2>&1 | head -5
```

Visual test:
1. Set `"spectral_rendering": false` — identical to before
2. Set `"spectral_rendering": true` — colors shift to pure spectral hues (rainbow-like). Where different transforms overlap, colors should blend to whites/pastels instead of muddy browns.
3. Compare a fractal with many overlapping transforms:
   - RGB mode: overlapping areas tend toward gray/brown
   - Spectral mode: overlapping areas produce bright, saturated blends
4. Toggle back and forth to see the difference clearly

### Commit message

```
feat: add spectral color rendering via CIE XYZ wavelength conversion

Maps palette indices to dominant wavelengths (380-780nm), converts
through CIE color matching functions to XYZ, then to sRGB. Prevents
muddy browns from complementary color mixing since spectral addition
produces physically-motivated blends. Toggle with spectral_rendering
in weights.json.
```

---

## Task 7: Temporal Reprojection

**Complexity:** High | **Risk:** High — requires storing previous frame parameters and motion estimation

### Concept

During morph transitions, estimate per-pixel motion from genome parameter changes. Store previous frame's zoom/transform parameters, compute approximate screen-space motion vectors, and warp the previous frame before blending. This reduces ghosting during transitions.

### Files to modify

**`src/main.rs`** — Store previous frame's zoom parameter and pass it to the shader.

Add to `App` struct:

```rust
    prev_zoom: f32,
    prev_globals: [f32; 20],
```

Initialize in `App::new()`:

```rust
    prev_zoom: 3.0,
    prev_globals: [0.0; 20],
```

In the uniform writing section, pass the previous zoom via the reserved slot. We can use `extra5.z` (temporal_reprojection strength) and pack `prev_zoom` into `extra5.w` (was reserved):

Update the `extra5` construction:

```rust
extra5: [
    self.weights._config.dof_focal_distance,
    if self.weights._config.spectral_rendering { 1.0 } else { 0.0 },
    self.weights._config.temporal_reprojection,
    self.prev_zoom,  // previous frame's zoom for reprojection
],
```

After writing uniforms, save current values for next frame:

```rust
                self.prev_zoom = self.globals[1]; // zoom is globals[1]
                self.prev_globals = self.globals;
```

**`playground.wgsl`** — Add temporal reprojection to the feedback trail section.

Replace the simple trail blend:

```wgsl
    // ── Feedback trail — gentle temporal smoothing ──
    let prev = textureSample(prev_frame, prev_sampler, tex_uv).rgb;
    col = max(col, prev * trail);  // max-blend, not additive — prevents brightness buildup
```

With:

```wgsl
    // ── Feedback trail with temporal reprojection ──
    let temporal_reproj = u.extra5.z;
    let prev_zoom = u.extra5.w;
    let cur_zoom = u.globals.y;

    var prev_uv = tex_uv;
    if (temporal_reproj > 0.001 && prev_zoom > 0.001) {
        // Estimate motion from zoom change: pixels move radially from/toward center
        // In our projection: screen = (world / zoom + 0.5) * resolution
        // So world = (screen/resolution - 0.5) * zoom
        // Previous frame's UV for same world point:
        // prev_screen = (world / prev_zoom + 0.5) * resolution
        // prev_uv = world / prev_zoom + 0.5
        //         = (uv - 0.5) * cur_zoom / prev_zoom + 0.5

        let centered = tex_uv - vec2(0.5);
        let zoom_ratio = cur_zoom / prev_zoom;
        let reprojected = centered * zoom_ratio + vec2(0.5);

        // Blend between simple UV and reprojected UV based on strength
        prev_uv = mix(tex_uv, reprojected, temporal_reproj);
    }

    // Clamp UV to valid range
    prev_uv = clamp(prev_uv, vec2(0.001), vec2(0.999));
    let prev = textureSample(prev_frame, prev_sampler, prev_uv).rgb;
    col = max(col, prev * trail);  // max-blend, not additive
```

### Verification

```bash
cargo build 2>&1 | head -5
```

Visual test:
1. Set `"temporal_reprojection": 0.0` — identical to before
2. Set `"temporal_reprojection": 0.5` — during morph transitions where zoom changes, the feedback trail should track the fractal's movement instead of smearing. The previous frame warps to match the new zoom before blending.
3. Set `"temporal_reprojection": 1.0` — full reprojection: zoom transitions should feel smooth with minimal ghosting
4. Test with a sequence of mutations that change zoom significantly — without reprojection you see double images briefly, with it the transition is cleaner
5. Static frames (no zoom change): `zoom_ratio = 1.0`, so `reprojected = tex_uv` — no effect, as expected

### Commit message

```
feat: add temporal reprojection for smoother morph transitions

During zoom changes, warp the previous frame's feedback by the zoom
ratio before blending, reducing ghosting and double-images. Previous
zoom is passed through extra5.w uniform. Controlled by
temporal_reprojection in weights.json (0.0=off, 1.0=full).
```

---

## Task Order Summary

| # | Task | Complexity | Dependencies |
|---|------|-----------|-------------|
| 1 | Extend Uniforms (extra4/extra5) | Low | None |
| 2 | Jittered Accumulation | Low | Task 1 |
| 3 | ACES Tonemapping | Low | Task 1 |
| 4 | Depth of Field | Medium | Task 1 |
| 5 | Adaptive Histogram Equalization | High | Task 1, Task 4 (7-channel buffers) |
| 6 | Spectral Color Rendering | High | Task 1 |
| 7 | Temporal Reprojection | High | Task 1 |

Tasks 2, 3, 6, and 7 are independent of each other (only depend on Task 1). Task 4 must come before Task 5 because Task 4 changes the buffer stride from 6 to 7 channels, and Task 5's histogram_cdf shader needs to know the correct stride.

## Suggested weights.json for Testing

After all tasks are complete, use these values to see all features in action:

```json
"jitter_amount": 0.5,
"tonemap_mode": 1,
"histogram_equalization": 0.3,
"dof_strength": 0.2,
"dof_focal_distance": 1.0,
"spectral_rendering": false,
"temporal_reprojection": 0.5
```

Start conservative, then crank individual values to see their effect in isolation.
