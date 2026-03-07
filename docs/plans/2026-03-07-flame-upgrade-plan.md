# Flame System Upgrade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the visual gap with Electric Sheep — RGBA histogram, log-density tonemapping, vibrancy, bloom, 20 new variations, final transform, and N-fold symmetry.

**Architecture:** Four phases in dependency order: (1) rendering overhaul for immediate visual payoff, (2) 20 new variation functions, (3) final transform + symmetry, (4) weights/audio integration. Each phase builds on the last.

**Tech Stack:** Rust + wgpu + WGSL compute/fragment shaders. `cargo build` for verification (no unit tests for shader code). Rust-side genome/weights changes are testable.

**Design doc:** `docs/plans/2026-03-07-flame-upgrade-design.md`

---

## Phase 1: Rendering Overhaul

### Task 1: RGBA Histogram — Buffer + Compute Shader

**Files:**
- Modify: `src/main.rs` (buffer creation, binding, histogram clearing)
- Modify: `flame_compute.wgsl` (add palette function, write RGB to histogram)

**Context:** Currently the histogram stores 2 u32 per pixel (density + color_sum). We need 4 u32 per pixel (density + R + G + B) so the compute shader can accumulate actual RGB color during iteration.

**Step 1: Update histogram buffer size**

In `src/main.rs`, find `create_histogram_buffer` (around line 563):

```rust
fn create_histogram_buffer(device: &wgpu::Device, w: u32, h: u32) -> wgpu::Buffer {
    let pixel_count = w.max(1) as u64 * h.max(1) as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("histogram"),
        size: pixel_count * 4 * 4,  // 4 u32s per pixel (density, R, G, B)
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
```

Change `* 2 * 4` to `* 4 * 4`.

**Step 2: Add palette function to compute shader**

In `flame_compute.wgsl`, add after the `rot2` function (around line 39):

```wgsl
const TAU: f32 = 6.28318530;

fn palette(t: f32) -> vec3<f32> {
    let a = vec3(0.5, 0.5, 0.5);
    let b = vec3(0.5, 0.5, 0.5);
    let c = vec3(1.0, 1.0, 1.0);
    let d = vec3(0.00, 0.33, 0.67);
    return a + b * cos(TAU * (c * t + d));
}
```

This is the same cosine palette already in `playground.wgsl`.

**Step 3: Update compute shader histogram writes**

In `flame_compute.wgsl`, in the main function, replace the histogram write block (around lines 179-183):

```wgsl
        if (px_x >= 0 && px_x < i32(w) && px_y >= 0 && px_y < i32(h)) {
            let buf_idx = (u32(px_y) * w + u32(px_x)) * 4u;
            let col = palette(color_idx + u.extra.x);  // apply color_shift
            atomicAdd(&histogram[buf_idx], 1u);
            atomicAdd(&histogram[buf_idx + 1u], u32(col.x * 1000.0));
            atomicAdd(&histogram[buf_idx + 2u], u32(col.y * 1000.0));
            atomicAdd(&histogram[buf_idx + 3u], u32(col.z * 1000.0));
        }
```

Key changes: `* 4u` instead of `* 2u`, accumulate RGB via palette lookup, apply color_shift here.

**Step 4: Build and verify**

Run: `cargo build 2>&1`
Expected: Compiles. The fragment shader will still read the old 2-u32 layout so it will look broken — that's expected, fixed in Task 2.

**Step 5: Commit**

```bash
git add src/main.rs flame_compute.wgsl
git commit -m "feat: RGBA histogram — 4 u32/px with RGB accumulation in compute"
```

---

### Task 2: Fragment Shader — Log-Density Tonemapping + RGBA Read

**Files:**
- Modify: `playground.wgsl` (complete rewrite of flame reading + tonemapping)

**Context:** The fragment shader needs to read 4 values per pixel now and use proper log-density tonemapping. We're also removing the edge-detection volumetric effect (it doesn't match Electric Sheep's aesthetic).

**Step 1: Rewrite the flame section of the fragment shader**

In `playground.wgsl`, replace the entire flame reading + tonemapping section (from `// ── Flame foreground` through the `return` statement) with:

```wgsl
    // ── KIFS background layer — disabled ──
    let bg = vec3(0.0);

    // ── Flame foreground (from RGBA histogram) ──
    let buf_idx = (px.y * w + px.x) * 4u;
    let density = f32(histogram[buf_idx]);
    let acc_r = f32(histogram[buf_idx + 1u]);
    let acc_g = f32(histogram[buf_idx + 2u]);
    let acc_b = f32(histogram[buf_idx + 3u]);

    // Recover average color from accumulated RGB
    let avg_color = select(
        vec3(0.0),
        vec3(acc_r, acc_g, acc_b) / max(density * 1000.0, 1.0),
        density > 0.0
    );

    // Log-density alpha (no hard cap — natural falloff)
    let alpha = log(1.0 + density * flame_bright) / (log(1.0 + density * flame_bright) + 4.0);

    // Flame color
    let flame = avg_color * alpha;

    // ── Combine ──
    var col = flame + bg;

    // ── Feedback persistence ──
    let prev = textureSample(prev_frame, prev_sampler, tex_uv).rgb;
    col = col + prev * trail;

    // Tonemap: soft clamp to prevent blowout
    col = col / (col + vec3(1.0));  // Reinhard

    // Gamma
    col = pow(max(col, vec3(0.0)), vec3(1.0 / 2.2));

    return vec4(col, 1.0);
```

Key changes:
- Reads 4 u32s per pixel (`* 4u`)
- Accumulates RGB properly
- Log-density alpha with soft asymptote (no hard cap)
- Removed edge-detection/gradient logic
- Reinhard tonemap instead of log tonemap
- Standard gamma 2.2

**Step 2: Clean up unused variables**

Remove `color_shift` read from uniforms in the fragment shader since it's now applied in the compute shader. Also remove `kifs_fold`, `kifs_scale`, `kifs_bright` reads since KIFS is disabled. Clean up any warnings.

**Step 3: Build and run**

Run: `cargo build 2>&1`
Expected: Compiles. Run the app and verify flames are visible and colorful. The rendering should look immediately different — more dynamic range, actual color blending.

**Step 4: Commit**

```bash
git add playground.wgsl
git commit -m "feat: log-density tonemapping with RGBA color blending"
```

---

### Task 3: Vibrancy

**Files:**
- Modify: `playground.wgsl` (add vibrancy calculation)
- Modify: `src/main.rs` (pass vibrancy via uniform extra slot)
- Modify: `src/weights.rs` (add vibrancy param)

**Context:** Vibrancy controls how saturated colors become based on density. Low-density wisps stay muted; high-density cores get punchy.

**Step 1: Pass vibrancy through uniforms**

In `src/weights.rs`, in the `global_index()` function, add:

```rust
"vibrancy" => Some(9),
```

In `src/main.rs`, where uniforms are written (around line 1116), change:

```rust
extra: [self.globals[8], self.globals[9], 0.0, 0.0],
```

**Step 2: Add vibrancy default to genome**

In `src/genome.rs`, in the `globals_array()` method, ensure slot 9 has a sensible default. The genome already packs 12 floats — slot 9 maps to `extra[1]`. Add vibrancy base value of 0.7 to the default genome globals or set it via weights.

Actually, the cleanest approach: `vibrancy` starts at 0.0 in the globals array (meaning no vibrancy effect by default). The base value of 0.7 comes from weights.json as a constant weight on a signal that's always 1.0, or we just set the genome default.

Simplest: In `genome.rs` `globals_array()`, ensure `g[9] = 0.7` as base vibrancy. Check how globals_array works:

```rust
pub fn globals_array(&self) -> [f32; 12] {
    let mut g = [0.0f32; 12];
    g[0] = self.global.speed;
    g[1] = self.global.zoom;
    g[2] = self.global.trail;
    g[3] = self.global.flame_brightness;
    g[4] = self.kifs.fold_angle;
    g[5] = self.kifs.scale;
    g[6] = self.kifs.brightness;
    // g[7] = drift_speed (filled by weights)
    // g[8] = color_shift (filled by weights)
    // g[9] = vibrancy (new)
    g
}
```

The weights system adds to these base values. So set `g[9] = 0.7` as the base vibrancy.

**Step 3: Apply vibrancy in fragment shader**

In `playground.wgsl`, after computing `flame` and before combining:

```wgsl
    let vibrancy = u.extra.y;

    // Vibrancy: saturate colors based on density
    let lum = dot(avg_color, vec3(0.299, 0.587, 0.114));
    let vibrant_color = mix(vec3(lum), avg_color, pow(alpha, max(1.0 - vibrancy, 0.01)));
    let flame = vibrant_color * alpha;
```

Replace the earlier `let flame = avg_color * alpha;` with this block.

**Step 4: Build and verify**

Run: `cargo build 2>&1`
Expected: Compiles. Flames should show richer, more saturated colors in dense areas.

**Step 5: Commit**

```bash
git add playground.wgsl src/main.rs src/weights.rs src/genome.rs
git commit -m "feat: vibrancy parameter — density-based color saturation"
```

---

### Task 4: Bloom Post-Process

**Files:**
- Modify: `playground.wgsl` (add bloom via multi-tap from feedback texture)
- Modify: `src/main.rs` (pass bloom_intensity via uniform)
- Modify: `src/weights.rs` (add bloom_intensity param)

**Context:** For V1, we implement bloom as multi-tap sampling from the feedback texture in the fragment shader. This avoids adding new render passes/textures. The feedback texture already contains accumulated flame brightness, so blurring it creates a natural glow.

**Step 1: Add bloom_intensity param**

In `src/weights.rs`, `global_index()`, add:

```rust
"bloom_intensity" => Some(10),
```

In `src/main.rs` uniforms packing:

```rust
extra: [self.globals[8], self.globals[9], self.globals[10], 0.0],
```

In `src/genome.rs` `globals_array()`, set `g[10] = 0.15` as base bloom intensity.

**Step 2: Add bloom sampling in fragment shader**

In `playground.wgsl`, after the feedback persistence line (`col = col + prev * trail`), add bloom:

```wgsl
    // Bloom: sample feedback at offsets for glow effect
    let bloom_int = u.extra.z;
    if (bloom_int > 0.001) {
        let texel = 1.0 / u.resolution;
        var bloom = vec3(0.0);
        // 13-tap cross pattern
        let offsets = array<vec2<f32>, 12>(
            vec2(-3.0, 0.0), vec2(-2.0, 0.0), vec2(-1.0, 0.0),
            vec2( 1.0, 0.0), vec2( 2.0, 0.0), vec2( 3.0, 0.0),
            vec2(0.0, -3.0), vec2(0.0, -2.0), vec2(0.0, -1.0),
            vec2(0.0,  1.0), vec2(0.0,  2.0), vec2(0.0,  3.0),
        );
        let weights = array<f32, 12>(
            0.05, 0.1, 0.2,
            0.2, 0.1, 0.05,
            0.05, 0.1, 0.2,
            0.2, 0.1, 0.05,
        );
        for (var bi = 0u; bi < 12u; bi++) {
            bloom += textureSample(prev_frame, prev_sampler, tex_uv + offsets[bi] * texel * 4.0).rgb * weights[bi];
        }
        col += bloom * bloom_int;
    }
```

This samples the previous frame in a cross pattern with Gaussian-ish weights, scaled by 4 pixels per step. The result is a soft glow around bright areas.

**Step 3: Build and verify**

Run: `cargo build 2>&1`
Expected: Compiles. Bright flame cores should have a subtle glow halo.

**Step 4: Commit**

```bash
git add playground.wgsl src/main.rs src/weights.rs src/genome.rs
git commit -m "feat: bloom post-process via feedback texture multi-tap"
```

---

## Phase 2: New Variations

### Task 5: Expand FlameTransform — Struct + Storage Buffer

**Files:**
- Modify: `src/genome.rs` (FlameTransform struct, flatten_transforms, default genome, mutations)
- Modify: `src/main.rs` (transform buffer size, write size)
- Modify: `src/weights.rs` (XF_FIELDS array)

**Context:** FlameTransform grows from 12 fields (6 variations) to 32 fields (26 variations). Storage buffer changes from 12 to 32 floats per transform.

**Step 1: Expand FlameTransform struct**

In `src/genome.rs`, replace the FlameTransform struct:

```rust
#[derive(Serialize, Deserialize, Clone, Debug)]
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
```

The `#[serde(default)]` ensures backwards compatibility with existing genome JSON files — old files missing these fields get 0.0.

**Step 2: Update flatten_transforms**

Replace the `flatten_transforms` method to pack 32 floats per transform:

```rust
pub fn flatten_transforms(&self) -> Vec<f32> {
    let mut t = Vec::with_capacity(self.transforms.len() * 32);
    for xf in &self.transforms {
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
    t
}
```

**Step 3: Update storage buffer sizing**

In `src/main.rs`, find transform buffer creation (around line 167). Change `12` to `32`:

```rust
size: 6u64 * 32u64 * 4u64,  // 6 transforms * 32 floats * 4 bytes
```

Also find the transform write code (around line 1120). Change `12` to `32`:

```rust
let xf_write_len = self.num_transforms * 32;
```

Search for ALL instances of `* 12` related to transforms and change to `* 32`.

**Step 4: Update XF_FIELDS in weights.rs**

In `src/weights.rs`, find `XF_FIELDS` (around line 223):

```rust
const PARAMS_PER_XF: usize = 32;
const XF_FIELDS: [&str; PARAMS_PER_XF] = [
    "weight", "angle", "scale", "offset_x", "offset_y", "color",
    "linear", "sinusoidal", "spherical", "swirl", "horseshoe", "handkerchief",
    "julia", "polar", "disc", "rings", "bubble", "fisheye",
    "exponential", "spiral", "diamond", "bent", "waves", "popcorn",
    "fan", "eyefish", "cross", "tangent", "cosine", "blob",
    "noise", "curl",
];
```

**Step 5: Update compute shader accessor**

In `flame_compute.wgsl`, change the `xf` function stride:

```wgsl
fn xf(idx: u32, field: u32) -> f32 { return transforms[idx * 32u + field]; }
```

**Step 6: Build and verify**

Run: `cargo build 2>&1`
Expected: Compiles. Existing flames should look identical since all new variation weights default to 0.0.

**Step 7: Commit**

```bash
git add src/genome.rs src/main.rs src/weights.rs flame_compute.wgsl
git commit -m "feat: expand FlameTransform to 32 fields (26 variations)"
```

---

### Task 6: Variation Functions — Batch 1 (julia through spiral)

**Files:**
- Modify: `flame_compute.wgsl` (add 8 variation functions, update apply_xform)

**Context:** Add the first 8 new variations to the compute shader. Each is a `vec2<f32> -> vec2<f32>` function. Then wire them into `apply_xform`.

**Step 1: Add variation functions**

In `flame_compute.wgsl`, after the existing variation functions (after `V_handkerchief`, around line 66), add:

```wgsl
fn V_julia(p: vec2<f32>, rng: ptr<function, u32>) -> vec2<f32> {
    let r = sqrt(length(p));
    let theta = atan2(p.y, p.x) * 0.5;
    let k = f32(pcg(rng) & 1u) * PI;
    return r * vec2(cos(theta + k), sin(theta + k));
}

fn V_polar(p: vec2<f32>) -> vec2<f32> {
    let theta = atan2(p.y, p.x);
    let r = length(p);
    return vec2(theta / PI, r - 1.0);
}

fn V_disc(p: vec2<f32>) -> vec2<f32> {
    let theta = atan2(p.y, p.x);
    let r = length(p);
    let f = theta / PI;
    return f * vec2(sin(PI * r), cos(PI * r));
}

fn V_rings(p: vec2<f32>, c2: f32) -> vec2<f32> {
    let r = length(p);
    let theta = atan2(p.y, p.x);
    let k = c2 + 1e-6;
    let rr = ((r + k) % (2.0 * k)) - k + r * (1.0 - k);
    return rr * vec2(cos(theta), sin(theta));
}

fn V_bubble(p: vec2<f32>) -> vec2<f32> {
    let r2 = dot(p, p);
    return p * 4.0 / (r2 + 4.0);
}

fn V_fisheye(p: vec2<f32>) -> vec2<f32> {
    let r = length(p);
    return 2.0 * p.yx / (r + 1.0);
}

fn V_exponential(p: vec2<f32>) -> vec2<f32> {
    let e = exp(p.x - 1.0);
    return e * vec2(cos(PI * p.y), sin(PI * p.y));
}

fn V_spiral(p: vec2<f32>) -> vec2<f32> {
    let r = length(p) + 1e-6;
    let theta = atan2(p.y, p.x);
    return vec2(cos(theta) + sin(r), sin(theta) - cos(r)) / r;
}
```

**Step 2: Wire into apply_xform**

In `flame_compute.wgsl`, update `apply_xform` to read the new variation weights and apply them. After the existing variations block (around line 121-126), add:

```wgsl
    let w_jul  = xf(idx, 12u);
    let w_pol  = xf(idx, 13u);
    let w_dsc  = xf(idx, 14u);
    let w_rng  = xf(idx, 15u);
    let w_bub  = xf(idx, 16u);
    let w_fsh  = xf(idx, 17u);
    let w_exp  = xf(idx, 18u);
    let w_spi  = xf(idx, 19u);
```

And in the variation accumulation, after the existing lines:

```wgsl
    v += V_julia(q, &rng_copy) * w_jul;
    v += V_polar(q)             * w_pol;
    v += V_disc(q)              * w_dsc;
    v += V_rings(q, scale * scale) * w_rng;
    v += V_bubble(q)            * w_bub;
    v += V_fisheye(q)           * w_fsh;
    v += V_exponential(q)       * w_exp;
    v += V_spiral(q)            * w_spi;
```

Note: `V_julia` needs an RNG pointer. The `apply_xform` function needs to accept and pass through the RNG state. Update its signature:

```wgsl
fn apply_xform(p: vec2<f32>, idx: u32, t: f32, rng: ptr<function, u32>) -> vec2<f32> {
```

And update the call site in `main()`:

```wgsl
        p = apply_xform(p, tidx, t, &rng);
```

For `V_rings`, `c2` is derived from the transform's scale squared (`scale * scale`), which is the standard flam3 approach.

**Step 3: Build and verify**

Run: `cargo build 2>&1`
Expected: Compiles. Existing flames unchanged (new weights are 0). Manually test by editing a genome JSON to add `"julia": 0.8` to a transform.

**Step 4: Commit**

```bash
git add flame_compute.wgsl
git commit -m "feat: add 8 variation functions (julia through spiral)"
```

---

### Task 7: Variation Functions — Batch 2 (diamond through curl)

**Files:**
- Modify: `flame_compute.wgsl` (add 12 more variation functions, update apply_xform)

**Step 1: Add variation functions**

```wgsl
fn V_diamond(p: vec2<f32>) -> vec2<f32> {
    let r = length(p);
    let theta = atan2(p.y, p.x);
    return vec2(sin(theta) * cos(r), cos(theta) * sin(r));
}

fn V_bent(p: vec2<f32>) -> vec2<f32> {
    var q = p;
    if (q.x < 0.0) { q.x *= 2.0; }
    if (q.y < 0.0) { q.y *= 0.5; }
    return q;
}

fn V_waves(p: vec2<f32>, bx: f32, by: f32) -> vec2<f32> {
    return vec2(
        p.x + bx * sin(p.y * 4.0),
        p.y + by * sin(p.x * 4.0)
    );
}

fn V_popcorn(p: vec2<f32>, cx: f32, cy: f32) -> vec2<f32> {
    return vec2(
        p.x + cx * sin(tan(3.0 * p.y)),
        p.y + cy * sin(tan(3.0 * p.x))
    );
}

fn V_fan(p: vec2<f32>, fan_t: f32) -> vec2<f32> {
    let theta = atan2(p.y, p.x);
    let r = length(p);
    let t2 = PI * fan_t * fan_t + 1e-6;
    if ((theta + fan_t) % t2 > t2 * 0.5) {
        return r * vec2(cos(theta - t2 * 0.5), sin(theta - t2 * 0.5));
    } else {
        return r * vec2(cos(theta + t2 * 0.5), sin(theta + t2 * 0.5));
    }
}

fn V_eyefish(p: vec2<f32>) -> vec2<f32> {
    let r = length(p);
    return 2.0 * p / (r + 1.0);
}

fn V_cross(p: vec2<f32>) -> vec2<f32> {
    let s = p.x * p.x - p.y * p.y;
    return sqrt(1.0 / (s * s + 1e-6)) * p;
}

fn V_tangent(p: vec2<f32>) -> vec2<f32> {
    return vec2(sin(p.x) / (cos(p.y) + 1e-6), tan(p.y));
}

fn V_cosine(p: vec2<f32>) -> vec2<f32> {
    return vec2(
        cos(PI * p.x) * cosh(p.y),
        -sin(PI * p.x) * sinh(p.y)
    );
}

fn V_blob(p: vec2<f32>) -> vec2<f32> {
    let r = length(p);
    let theta = atan2(p.y, p.x);
    let blobr = r * (0.5 + 0.5 * sin(3.0 * theta));
    return blobr * vec2(cos(theta), sin(theta));
}

fn V_noise(p: vec2<f32>, seed: u32) -> vec2<f32> {
    let nx = vnoise(p.x * 3.0, seed + 500u);
    let ny = vnoise(p.y * 3.0, seed + 600u);
    return p + vec2(nx, ny) * 0.3;
}

fn V_curl(p: vec2<f32>, seed: u32) -> vec2<f32> {
    let eps = 0.01;
    let n00 = vnoise(p.x * 2.0, seed + 700u) + vnoise(p.y * 2.0, seed + 800u);
    let n10 = vnoise((p.x + eps) * 2.0, seed + 700u) + vnoise(p.y * 2.0, seed + 800u);
    let n01 = vnoise(p.x * 2.0, seed + 700u) + vnoise((p.y + eps) * 2.0, seed + 800u);
    let dx = (n10 - n00) / eps;
    let dy = (n01 - n00) / eps;
    return p + vec2(dy, -dx) * 0.15;
}
```

Note: WGSL doesn't have built-in `cosh`/`sinh`. Add helpers:

```wgsl
fn cosh(x: f32) -> f32 { return (exp(x) + exp(-x)) * 0.5; }
fn sinh(x: f32) -> f32 { return (exp(x) - exp(-x)) * 0.5; }
```

**Step 2: Wire into apply_xform**

Read weights for slots 20-31 and accumulate:

```wgsl
    let w_dia  = xf(idx, 20u);
    let w_bnt  = xf(idx, 21u);
    let w_wav  = xf(idx, 22u);
    let w_pop  = xf(idx, 23u);
    let w_fan  = xf(idx, 24u);
    let w_eye  = xf(idx, 25u);
    let w_crs  = xf(idx, 26u);
    let w_tan  = xf(idx, 27u);
    let w_cos  = xf(idx, 28u);
    let w_blb  = xf(idx, 29u);
    let w_noi  = xf(idx, 30u);
    let w_crl  = xf(idx, 31u);

    let seed = hash_u(idx * 31337u + 42u);

    v += V_diamond(q)              * w_dia;
    v += V_bent(q)                 * w_bnt;
    v += V_waves(q, ox * 0.5, oy * 0.5) * w_wav;
    v += V_popcorn(q, ox * 0.3, oy * 0.3) * w_pop;
    v += V_fan(q, angle)           * w_fan;
    v += V_eyefish(q)              * w_eye;
    v += V_cross(q)                * w_crs;
    v += V_tangent(q)              * w_tan;
    v += V_cosine(q)               * w_cos;
    v += V_blob(q)                 * w_blb;
    v += V_noise(q, seed)          * w_noi;
    v += V_curl(q, seed)           * w_crl;
```

For `V_waves` and `V_popcorn`, the per-transform parameters are derived from the transform's offset values (scaled down). For `V_fan`, the parameter is derived from the transform's angle. This avoids adding extra fields.

**Step 3: Build and verify**

Run: `cargo build 2>&1`
Expected: Compiles. Test by editing genome JSON with new variation names.

**Step 4: Commit**

```bash
git add flame_compute.wgsl
git commit -m "feat: add 12 more variations (diamond through curl) — 26 total"
```

---

### Task 8: Update Mutations for 26 Variations

**Files:**
- Modify: `src/genome.rs` (update mutate_perturb, mutate_swap_variations, mutate_add_transform)

**Context:** Mutations currently work with 6 variations. Update them to work with 26.

**Step 1: Define variation field names constant**

Add near the top of `genome.rs`:

```rust
const VARIATION_COUNT: usize = 26;
const VARIATION_NAMES: [&str; VARIATION_COUNT] = [
    "linear", "sinusoidal", "spherical", "swirl", "horseshoe", "handkerchief",
    "julia", "polar", "disc", "rings", "bubble", "fisheye",
    "exponential", "spiral", "diamond", "bent", "waves", "popcorn",
    "fan", "eyefish", "cross", "tangent", "cosine", "blob",
    "noise", "curl",
];
```

**Step 2: Add getter/setter helpers on FlameTransform**

```rust
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
```

**Step 3: Update mutate_perturb "reinvent as specialist"**

Replace the variation reinvention block (case 4) to use VARIATION_COUNT:

```rust
            _ => {
                // Reinvent as specialist — one dominant variation from all 26
                for vi in 0..VARIATION_COUNT {
                    xf.set_variation(vi, 0.0);
                }
                let dominant = rng.random_range(0..VARIATION_COUNT);
                xf.set_variation(dominant, rng.random_range(0.7..1.0));
                // Maybe a small secondary
                let secondary = rng.random_range(0..VARIATION_COUNT);
                if secondary != dominant {
                    xf.set_variation(secondary, rng.random_range(0.0..0.2));
                }
            }
```

**Step 4: Update mutate_swap_variations**

Replace the hardcoded 6-element array with dynamic variation access:

```rust
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
```

**Step 5: Update mutate_add_transform "fresh specialist"**

In `mutate_add_transform`, the fresh specialist branch creates a new transform. Update it to use VARIATION_COUNT:

```rust
        } else {
            let mut xf = FlameTransform {
                weight: 0.05,
                angle: rng.random_range(-PI..PI),
                scale: rng.random_range(0.2..0.8),
                offset: [rng.random_range(-3.0..3.0), rng.random_range(-3.0..3.0)],
                color: rng.random_range(0.0..1.0),
                ..Default::default()
            };
            let dominant = rng.random_range(0..VARIATION_COUNT);
            xf.set_variation(dominant, rng.random_range(0.7..1.0));
            xf
        };
```

For this to work, add `Default` derive to FlameTransform (all f32 fields default to 0.0, offset to [0.0, 0.0]).

Add above the struct:

```rust
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
```

Wait — `Default` for `[f32; 2]` is [0.0, 0.0], which is fine. But make sure the `#[serde(default)]` on new fields is compatible. It should be — serde default and Rust Default both give 0.0 for f32.

**Step 6: Replace mutate_kifs_drift with a more useful mutation**

Since KIFS is disabled, replace `mutate_kifs_drift` with `mutate_global_params`:

```rust
    fn mutate_global_params(&mut self, rng: &mut impl Rng) {
        self.global.flame_brightness = (self.global.flame_brightness + rng.random_range(-0.1..0.1)).clamp(0.1, 1.0);
        self.global.zoom = (self.global.zoom + rng.random_range(-0.5..0.5)).clamp(1.5, 6.0);
    }
```

Update the match in `mutate()`:

```rust
                _ => child.mutate_global_params(&mut rng),
```

**Step 7: Build and verify**

Run: `cargo build 2>&1 && cargo test 2>&1`
Expected: Compiles, all existing tests pass. Mutations now generate transforms using all 26 variations.

**Step 8: Commit**

```bash
git add src/genome.rs
git commit -m "feat: update mutations for 26 variations, add get/set_variation helpers"
```

---

## Phase 3: Final Transform + Symmetry

### Task 9: Final Transform — Genome + Storage

**Files:**
- Modify: `src/genome.rs` (add final_transform field, update flatten, update globals_array)
- Modify: `src/main.rs` (pack final transform into storage buffer, update transform_count)

**Context:** The final transform is a FlameTransform applied to every point after the IFS iteration, before screen projection. It's stored as an optional field in the genome and packed at the end of the transform storage buffer.

**Step 1: Add final_transform to FlameGenome**

In `src/genome.rs`, add to the FlameGenome struct:

```rust
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
```

**Step 2: Update flatten_transforms to include final transform**

The final transform gets appended after regular transforms. The compute shader will know it's there via a new `has_final_xform` flag.

```rust
pub fn flatten_transforms(&self) -> Vec<f32> {
    let total = self.transforms.len() + if self.final_transform.is_some() { 1 } else { 0 };
    let mut t = Vec::with_capacity(total * 32);
    for xf in &self.transforms {
        self.push_transform(&mut t, xf);
    }
    if let Some(ref fxf) = self.final_transform {
        self.push_transform(&mut t, fxf);
    }
    t
}

fn push_transform(&self, t: &mut Vec<f32>, xf: &FlameTransform) {
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
    t.push(xf.julia);
    t.push(xf.polar);
    t.push(xf.disc);
    t.push(xf.rings);
    t.push(xf.bubble);
    t.push(xf.fisheye);
    t.push(xf.exponential);
    t.push(xf.spiral);
    t.push(xf.diamond);
    t.push(xf.bent);
    t.push(xf.waves);
    t.push(xf.popcorn);
    t.push(xf.fan);
    t.push(xf.eyefish);
    t.push(xf.cross);
    t.push(xf.tangent);
    t.push(xf.cosine);
    t.push(xf.blob);
    t.push(xf.noise);
    t.push(xf.curl);
}
```

**Step 3: Pass has_final_xform + symmetry via uniforms**

Use the `_pad` slot in Uniforms for `has_final_xform`, and `extra[3]` for symmetry:

In `src/main.rs`, update uniform packing:

```rust
let has_final = if self.current_genome().final_transform.is_some() { 1u32 } else { 0u32 };
let uniforms = Uniforms {
    // ...
    _pad: has_final,
    // ...
    extra: [self.globals[8], self.globals[9], self.globals[10],
            self.current_genome().symmetry as f32],
};
```

Update buffer sizing to account for the possible final transform:

```rust
let total_xf = genome.transforms.len() + if genome.final_transform.is_some() { 1 } else { 0 };
```

**Step 4: Build and verify**

Run: `cargo build 2>&1`
Expected: Compiles. No visual change (final_transform defaults to None, symmetry defaults to 1).

**Step 5: Commit**

```bash
git add src/genome.rs src/main.rs
git commit -m "feat: final transform + symmetry fields in genome and uniforms"
```

---

### Task 10: Final Transform + Symmetry — Compute Shader

**Files:**
- Modify: `flame_compute.wgsl` (apply final transform, plot symmetry copies)

**Context:** After the IFS iteration loop converges, apply the optional final transform. Then plot the point + rotated/mirrored copies based on symmetry order.

**Step 1: Apply final transform after iteration**

In `flame_compute.wgsl`, after the iteration loop (after line 171 `color_idx = ...`), add:

```wgsl
    // Final transform (if present)
    let has_final = u._pad;
    if (has_final == 1u) {
        p = apply_xform(p, num_xf, t, &rng);  // final xform is at index num_xf
        color_idx = color_idx * 0.5 + xform_color(num_xf) * 0.5;
    }
```

The final transform is stored right after the regular transforms in the buffer, at index `num_xf`.

**Step 2: Plot symmetry copies**

Replace the existing screen projection + histogram write block with a loop that plots multiple copies:

```wgsl
        if (i < 20u) { continue; }

        let sym = i32(u.extra.w);
        let abs_sym = abs(sym);
        let bilateral = sym < 0;
        let sym_count = select(abs_sym, abs_sym, abs_sym > 0);
        let actual_sym = select(1, sym_count, sym_count > 0);

        for (var si = 0; si < actual_sym; si++) {
            let sym_angle = f32(si) * TAU / f32(actual_sym);
            let cp = cos(sym_angle);
            let sp = sin(sym_angle);
            let sym_p = vec2(p.x * cp - p.y * sp, p.x * sp + p.y * cp);

            // Plot this rotated copy
            let screen = (sym_p / zoom + vec2(0.5, 0.5)) * vec2<f32>(f32(w), f32(h));
            let px_x = i32(screen.x);
            let px_y = i32(screen.y);

            if (px_x >= 0 && px_x < i32(w) && px_y >= 0 && px_y < i32(h)) {
                let buf_idx = (u32(px_y) * w + u32(px_x)) * 4u;
                let col = palette(color_idx + u.extra.x);
                atomicAdd(&histogram[buf_idx], 1u);
                atomicAdd(&histogram[buf_idx + 1u], u32(col.x * 1000.0));
                atomicAdd(&histogram[buf_idx + 2u], u32(col.y * 1000.0));
                atomicAdd(&histogram[buf_idx + 3u], u32(col.z * 1000.0));
            }

            // Bilateral mirror
            if (bilateral) {
                let mir_p = vec2(-sym_p.x, sym_p.y);
                let mscreen = (mir_p / zoom + vec2(0.5, 0.5)) * vec2<f32>(f32(w), f32(h));
                let mpx_x = i32(mscreen.x);
                let mpx_y = i32(mscreen.y);
                if (mpx_x >= 0 && mpx_x < i32(w) && mpx_y >= 0 && mpx_y < i32(h)) {
                    let mbuf_idx = (u32(mpx_y) * w + u32(mpx_x)) * 4u;
                    let mcol = palette(color_idx + u.extra.x);
                    atomicAdd(&histogram[mbuf_idx], 1u);
                    atomicAdd(&histogram[mbuf_idx + 1u], u32(mcol.x * 1000.0));
                    atomicAdd(&histogram[mbuf_idx + 2u], u32(mcol.y * 1000.0));
                    atomicAdd(&histogram[mbuf_idx + 3u], u32(mcol.z * 1000.0));
                }
            }
        }
```

**Step 3: Update Uniforms struct in compute shader**

Make sure the compute shader's Uniforms struct reads `_pad` properly (it already has the field, just need to use it).

**Step 4: Build and verify**

Run: `cargo build 2>&1`
Expected: Compiles. Test by manually setting `"symmetry": 4` in a genome JSON. Should see 4-fold rotational symmetry.

**Step 5: Commit**

```bash
git add flame_compute.wgsl
git commit -m "feat: final transform + N-fold symmetry in compute shader"
```

---

### Task 11: Final Transform + Symmetry Mutations

**Files:**
- Modify: `src/genome.rs` (add mutation functions, update mutate())

**Step 1: Add final transform mutation**

```rust
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
                    fxf.angle = rng.random_range(-PI..PI);
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
            // No symmetry
            self.symmetry = 1;
        } else if roll < 0.7 {
            // Rotational 2-6
            self.symmetry = rng.random_range(2..=6);
        } else {
            // Bilateral + rotational
            self.symmetry = -rng.random_range(2..=4);
        }
    }
```

**Step 2: Wire into mutate()**

Update the match in `mutate()` to include new mutation types. Expand from 5 to 7 cases:

```rust
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
```

**Step 3: Update default genome with occasional symmetry**

In `default_genome()`, add:

```rust
            final_transform: None,
            symmetry: 1,
```

**Step 4: Build and verify**

Run: `cargo build 2>&1`
Expected: Compiles. Hit spacebar to mutate — should occasionally see symmetry and final transform effects.

**Step 5: Commit**

```bash
git add src/genome.rs
git commit -m "feat: mutations for final transform and symmetry"
```

---

## Phase 4: Weights & Audio Integration

### Task 12: Update weights.json

**Files:**
- Modify: `weights.json` (add bloom_intensity, vibrancy mappings)

**Step 1: Update weights.json**

Add the new params to the doc section and add audio mappings:

In the `_params` doc section, add:

```json
"bloom_intensity": "Bloom glow intensity",
"vibrancy":        "Color saturation based on density"
```

Update the signal sections:

```json
"energy": {
    "flame_brightness": 0.7,
    "drift_speed": 1.5,
    "mutation_rate": 0.3,
    "bloom_intensity": 0.3
},
"beat_pulse": {
    "bloom_intensity": 0.4
},
"bass": {
    "color_shift": -0.4,
    "flame_brightness": 0.3,
    "vibrancy": 0.15
}
```

**Step 2: Commit**

```bash
git add weights.json
git commit -m "feat: weights.json — bloom_intensity and vibrancy audio mappings"
```

---

### Task 13: Update Default Genome

**Files:**
- Modify: `src/genome.rs` (richer default genome with more variation diversity)

**Context:** The default genome should showcase the new system. Mix of old and new variations, moderate symmetry chance.

**Step 1: Update default_genome transforms**

Keep the 6 existing specialist transforms but make some of them use new variations. Replace 2-3 of the existing transforms with new-variation specialists:

```rust
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
```

**Step 2: Build and verify**

Run: `cargo build 2>&1`
Expected: Compiles. The default starting flame should show more variety.

**Step 3: Commit**

```bash
git add src/genome.rs
git commit -m "feat: richer default genome with julia, spiral, cosine, curl variations"
```

---

### Task 14: Cleanup + Tag

**Files:**
- Modify: any remaining warnings

**Step 1: Build clean**

Run: `cargo build 2>&1`
Fix any warnings related to the changes (unused variables, dead code, etc.).

**Step 2: Run tests**

Run: `cargo test 2>&1`
Expected: All tests pass.

**Step 3: Verify visually**

Run the app. Check:
- Flames are colorful with proper log-density tonemapping
- Bloom creates soft glow around bright areas
- Mutations produce varied shapes (julia, spiral, cosine, etc.)
- Symmetry appears occasionally on mutation
- No crashes or visual artifacts

**Step 4: Commit and tag**

```bash
git add -A
git commit -m "chore: cleanup warnings from flame upgrade"
git tag v1.0.0-flame-upgrade
```
