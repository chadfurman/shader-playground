# Electric Sheep Phase 2 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Match Electric Sheep visual fidelity by supporting full affine transforms, importing real `.flame` genomes, per-genome color palettes, and parametric variations.

**Architecture:** Expand FlameTransform from 4-param (angle/scale/offset) to 6-param affine matrix (a,b,c,d,e,f). Add 256-entry palette texture. Parse flam3 XML into our genome format. Add variation parameters to the GPU buffer. All changes maintain backward compatibility with existing genomes.

**Tech Stack:** Rust + wgpu, WGSL shaders, `quick-xml` for flam3 parsing, serde_json

**Known Issues to Fix Along the Way:**
- Darkening over time (accumulation + morph interaction)
- Degenerate genomes that collapse to wisps after evolve

---

### Task 1: Expand FlameTransform to full 6-param affine

**Files:**
- Modify: `src/genome.rs` — FlameTransform struct, flatten_transforms(), apply_xform_cpu(), mutate_inner()
- Modify: `flame_compute.wgsl` — transform application in apply_xform()
- Modify: `src/weights.rs` — XF_FIELDS array

**Step 1: Update FlameTransform struct**

Replace `angle: f32, scale: f32` with full affine coefficients:

```rust
pub struct FlameTransform {
    pub weight: f32,
    // Full 2x3 affine matrix: [a b; c d] * [x; y] + [e; f]
    // Replaces angle + scale. For backward compat, default is rotation matrix.
    pub a: f32,  // was: cos(angle) * scale
    pub b: f32,  // was: -sin(angle) * scale
    pub c: f32,  // was: sin(angle) * scale
    pub d: f32,  // was: cos(angle) * scale
    pub offset: [f32; 2],  // e, f — translation (unchanged)
    pub color: f32,
    // ... variations unchanged ...
}
```

Add a helper to construct from angle+scale for backward compat:

```rust
impl FlameTransform {
    pub fn from_angle_scale(angle: f32, scale: f32) -> (f32, f32, f32, f32) {
        let (s, c) = angle.sin_cos();
        (c * scale, -s * scale, s * scale, c * scale)
    }
}
```

Update `Default` impl and `default_genome()` to use a/b/c/d instead of angle/scale.

**Step 2: Update flatten_transforms()**

Change the 32-float layout. Currently:
```
[0] weight [1] angle [2] scale [3] offset_x [4] offset_y [5] color [6..31] variations
```

New layout:
```
[0] weight [1] a [2] b [3] c [4] d [5] offset_x [6] offset_y [7] color [8..33] variations
```

This expands from 32 to 34 floats per transform. Update `PARAMS_PER_XF` in weights.rs from 32 to 34. Update `XF_FIELDS` array to match.

**Step 3: Handle backward-compatible deserialization**

Add serde support so old genomes (with `angle`/`scale` fields) still load:

```rust
#[serde(default)]
pub a: f32,
#[serde(default)]
pub b: f32,
#[serde(default)]
pub c: f32,
#[serde(default)]
pub d: f32,
// Keep for deserialization compat only
#[serde(default, skip_serializing)]
pub angle: Option<f32>,
#[serde(default, skip_serializing)]
pub scale: Option<f32>,
```

In a post-deserialize step, if `angle`/`scale` are present and a/b/c/d are default, convert:
```rust
if let (Some(angle), Some(scale)) = (self.angle, self.scale) {
    let (a, b, c, d) = Self::from_angle_scale(angle, scale);
    self.a = a; self.b = b; self.c = c; self.d = d;
}
```

**Step 4: Update flame_compute.wgsl**

Change `xf()` reads and the affine application:

```wgsl
// Old: let angle = xf(idx, 1u); let scale = xf(idx, 2u);
//      let q = rot2(angle) * p * scale + offset;
// New:
let a = xf(idx, 1u);
let b = xf(idx, 2u);
let c = xf(idx, 3u);
let d = xf(idx, 4u);
let ox = xf(idx, 5u);
let oy = xf(idx, 6u);

// Full affine: [a b; c d] * [x; y] + [e; f]
let q = vec2(a * p.x + b * p.y + ox, c * p.x + d * p.y + oy);
```

Update all variation weight reads to start at index 8 instead of 6.

Update the `xform_color` function to read from index 7 instead of 5.

**Step 5: Update apply_xform_cpu()**

Replace the CPU angle/scale rotation with full affine matrix multiply:
```rust
let ax = xf.a * p.0 + xf.b * p.1 + xf.offset[0];
let ay = xf.c * p.0 + xf.d * p.1 + xf.offset[1];
```

**Step 6: Update mutation code**

Replace angle/scale mutations with affine mutations:
- Rotate: apply small rotation matrix to [a,b;c,d]
- Scale: multiply all of a,b,c,d by a factor
- Shear: add small off-diagonal perturbation to b,c
- Translate: same as before (offset)

**Step 7: Update weights.rs XF_FIELDS**

```rust
const PARAMS_PER_XF: usize = 34;
const XF_FIELDS: [&str; PARAMS_PER_XF] = [
    "weight", "a", "b", "c", "d", "offset_x", "offset_y", "color",
    "linear", "sinusoidal", "spherical", "swirl", ...
];
const VARIATION_START: usize = 8;
```

**Step 8: Update seed genomes**

Update all 15 seed genome JSON files to use a/b/c/d instead of angle/scale.

**Step 9: Verify**

Run: `cargo build && cargo run`
Expected: Existing genomes load (backward compat), mutations produce affine variety (shear, non-uniform scale), attractor estimator works with new affine.

**Step 10: Commit**

```bash
git add src/genome.rs src/weights.rs flame_compute.wgsl genomes/seeds/
git commit -m "feat: full 6-param affine transforms (a,b,c,d,e,f) replacing angle+scale"
```

---

### Task 2: Per-genome 256-entry color palettes

**Files:**
- Modify: `src/genome.rs` — add palette field to FlameGenome
- Modify: `src/main.rs` — create palette texture, bind to compute shader
- Modify: `flame_compute.wgsl` — sample palette texture instead of cosine function
- Modify: `playground.wgsl` — (no changes needed, reads accumulated color)

**Step 1: Add palette to FlameGenome**

```rust
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FlameGenome {
    // ... existing fields ...
    #[serde(default)]
    pub palette: Option<Vec<[f32; 3]>>,  // 256 RGB entries, or None for cosine fallback
}
```

**Step 2: Create palette texture on GPU**

In `Gpu` struct, add:
```rust
palette_texture: wgpu::Texture,
palette_view: wgpu::TextureView,
```

Create a 256x1 RGBA32Float texture. When a genome has a palette, upload it. When it doesn't, generate a default cosine palette and upload that.

```rust
fn create_palette_texture(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let desc = wgpu::TextureDescriptor {
        label: Some("palette"),
        size: wgpu::Extent3d { width: 256, height: 1, depth_or_array_layers: 1 },
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        // ...
    };
    // ...
}
```

**Step 3: Add palette to compute bind group**

Add a new binding (binding 3) to the compute bind group layout:
```rust
// binding 3: palette texture
wgpu::BindGroupLayoutEntry {
    binding: 3,
    visibility: wgpu::ShaderStages::COMPUTE,
    ty: wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: wgpu::TextureViewDimension::D1, // or D2 with height=1
    },
    count: None,
},
// binding 4: palette sampler
wgpu::BindGroupLayoutEntry {
    binding: 4,
    visibility: wgpu::ShaderStages::COMPUTE,
    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
    count: None,
},
```

**Step 4: Upload palette on genome change**

In the morph/evolve code, when a new genome is set:
```rust
let palette_data = genome.palette.as_ref()
    .map(|p| /* convert to [f32; 4] * 256 RGBA */)
    .unwrap_or_else(|| generate_cosine_palette());
gpu.queue.write_texture(/* palette_texture */, &palette_data, /* ... */);
```

**Step 5: Update flame_compute.wgsl**

Replace the cosine `palette()` function with texture sampling:

```wgsl
@group(0) @binding(3) var palette_tex: texture_2d<f32>;
@group(0) @binding(4) var palette_sampler: sampler;

fn palette(t: f32) -> vec3<f32> {
    let uv = vec2(fract(t), 0.5);
    return textureSampleLevel(palette_tex, palette_sampler, uv, 0.0).rgb;
}
```

Keep the old cosine palette functions as `cosine_palette()` fallback (called from CPU when generating default palette data).

**Step 6: Generate default cosine palette as texture data**

On the Rust side, generate the 4-palette cosine blend as 256 RGBA entries:
```rust
fn generate_cosine_palette() -> Vec<[f32; 4]> {
    (0..256).map(|i| {
        let t = i as f32 / 255.0;
        let rgb = cosine_palette_blend(t);
        [rgb[0], rgb[1], rgb[2], 1.0]
    }).collect()
}
```

**Step 7: Verify**

Run: `cargo build && cargo run`
Expected: Colors look the same as before (cosine palette loaded as texture). Genomes with custom palettes (coming from XML import in Task 3) will use their own colors.

**Step 8: Commit**

```bash
git add src/genome.rs src/main.rs flame_compute.wgsl
git commit -m "feat: per-genome 256-entry color palettes via GPU texture"
```

---

### Task 3: Flam3 XML importer

**Files:**
- Create: `src/flam3.rs` — XML parser module
- Modify: `src/main.rs` — add mod flam3, add 'f' keybind to load .flame files
- Modify: `Cargo.toml` — add `quick-xml` dependency

**Step 1: Add quick-xml dependency**

```bash
cargo add quick-xml
```

**Step 2: Create src/flam3.rs**

Parse the flam3 XML format:

```xml
<flame name="sheep_123" size="800 600" ...>
  <xform weight="0.5" color="0.2" symmetry="0.0"
         coefs="0.6 -0.3 0.3 0.6 0.1 -0.2"
         linear="0.7" sinusoidal="0.3" julia="0.0" ... />
  <xform weight="0.3" color="0.8"
         coefs="..." spherical="1.0" />
  <palette count="256" format="RGB">
    FF8800FF9900FFAA00...
  </palette>
</flame>
```

Implement:

```rust
pub struct Flam3File {
    pub flames: Vec<FlameGenome>,
}

impl Flam3File {
    pub fn parse(xml: &str) -> Result<Self, String> { ... }
}
```

**Key mapping:**
- `<flame>` → `FlameGenome`
- `<xform coefs="a b c d e f">` → `FlameTransform { a, b, c, d, e, f }`
  - Note: flam3 coefs order is `a d b e c f` (column-major), map to our `a b c d e f` (row-major)
- `<xform>` variation attributes → our variation weights (map by name)
- `<palette>` hex RGB → `Vec<[f32; 3]>` (256 entries)
- `<flame symmetry="N">` → our `symmetry` field

**Variation name mapping** (our 26 supported):
```rust
const VARIATION_MAP: &[(&str, &str)] = &[
    ("linear", "linear"),
    ("sinusoidal", "sinusoidal"),
    ("spherical", "spherical"),
    ("swirl", "swirl"),
    ("horseshoe", "horseshoe"),
    ("handkerchief", "handkerchief"),
    ("julia", "julia"),
    ("polar", "polar"),
    ("disc", "disc"),
    ("rings", "rings"),
    ("bubble", "bubble"),
    ("fisheye", "fisheye"),
    ("exponential", "exponential"),
    ("spiral", "spiral"),
    ("diamond", "diamond"),
    ("bent", "bent"),
    ("waves", "waves"),
    ("popcorn", "popcorn"),
    ("fan", "fan"),
    ("eyefish", "eyefish"),
    ("cross", "cross"),
    ("tangent", "tangent"),
    ("cosine", "cosine"),
    ("blob", "blob"),
    ("noise", "noise"),
    ("curl", "curl"),
];
```

Unsupported variations: log a warning and skip.

**Step 3: Parse palette hex data**

```rust
fn parse_palette(hex: &str) -> Vec<[f32; 3]> {
    hex.as_bytes()
        .chunks(6)
        .map(|chunk| {
            let hex = std::str::from_utf8(chunk).unwrap_or("000000");
            let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0) as f32 / 255.0;
            let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0) as f32 / 255.0;
            let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0) as f32 / 255.0;
            [r, g, b]
        })
        .collect()
}
```

**Step 4: Add keybind and flame directory**

Add 'f' keybind in main.rs to load a random `.flame` file from `genomes/flames/`:

```rust
"f" => {
    let flames_dir = project_dir().join("genomes").join("flames");
    match load_random_flame(&flames_dir) {
        Ok(g) => {
            self.genome_history.push(self.genome.clone());
            self.genome = g;
            self.begin_morph();
            // Upload palette if present
            eprintln!("[flame] loaded: {}", self.genome.name);
        }
        Err(e) => eprintln!("[flame] error: {e}"),
    }
}
```

**Step 5: Verify with a test flame file**

Download or create a simple `.flame` file in `genomes/flames/` for testing.

Run: `cargo build && cargo run`, press 'f'
Expected: Flame genome loads with proper affine transforms and palette.

**Step 6: Commit**

```bash
git add src/flam3.rs src/main.rs Cargo.toml
git commit -m "feat: flam3 XML importer — parse .flame files into our genome format"
```

---

### Task 4: Parametric variations

**Files:**
- Modify: `src/genome.rs` — add variation_params to FlameTransform
- Modify: `flame_compute.wgsl` — parametric variation implementations
- Modify: `src/flam3.rs` — import variation params from XML

**Step 1: Add variation params to FlameTransform**

```rust
#[serde(default)]
pub variation_params: HashMap<String, f32>,
```

Key params to support (matching flam3):
- `rings2_val` — rings2 variation parameter
- `fan2_x`, `fan2_y` — fan2 parameters
- `blob_low`, `blob_high`, `blob_waves` — blob parameters
- `julian_power`, `julian_dist` — julian variation
- `juliascope_power`, `juliascope_dist` — juliascope variation
- `ngon_power`, `ngon_sides`, `ngon_corners`, `ngon_circle` — ngon parameters

**Step 2: Pack params into GPU buffer**

Extend the per-transform buffer by 8 floats (34 → 42) for common variation params:

```
[34] rings2_val   [38] julian_power
[35] blob_low     [39] julian_dist
[36] blob_high    [40] ngon_sides
[37] blob_waves   [41] ngon_corners
```

Update `PARAMS_PER_XF` to 42. Update `flatten_transforms()` to pack these.

**Step 3: Update shader variations**

Example — parametric blob:
```wgsl
// Old: fixed blob
fn v_blob(p: vec2<f32>) -> vec2<f32> {
    let r = length(p);
    let theta = atan2(p.y, p.x);
    let rr = r * (0.5 + 0.5 * sin(3.0 * theta));  // hardcoded
    return rr * vec2(cos(theta), sin(theta));
}

// New: parametric blob
fn v_blob(p: vec2<f32>, low: f32, high: f32, waves: f32) -> vec2<f32> {
    let r = length(p);
    let theta = atan2(p.y, p.x);
    let rr = r * (low + (high - low) * 0.5 * (sin(waves * theta) + 1.0));
    return rr * vec2(cos(theta), sin(theta));
}
```

Similarly for rings2, julian, juliascope, ngon, fan2.

**Step 4: Import params from flam3 XML**

In `src/flam3.rs`, parse variation parameter attributes:
```xml
<xform blob="0.5" blob_low="0.2" blob_high="1.0" blob_waves="5" .../>
```

Map to `variation_params: { "blob_low": 0.2, "blob_high": 1.0, "blob_waves": 5.0 }`.

**Step 5: Update mutation code**

When mutating a transform that has parametric variations, occasionally perturb the params:
```rust
for (key, val) in xf.variation_params.iter_mut() {
    if rng.random::<f32>() < 0.3 {
        *val += rng.random_range(-0.2..0.2);
    }
}
```

**Step 6: Verify**

Run: `cargo build && cargo run`
Expected: Blob, rings, julian etc. now have richer behavior. Imported flames with params render more faithfully.

**Step 7: Commit**

```bash
git add src/genome.rs src/flam3.rs flame_compute.wgsl
git commit -m "feat: parametric variations (blob, rings2, julian, juliascope, ngon, fan2)"
```

---

### Task 5: Fitness-biased mutation

**Files:**
- Modify: `src/genome.rs` — analyze favorites, bias mutation
- Modify: `src/weights.rs` — fitness_bias_strength config

**Step 1: Add config param**

In RuntimeConfig:
```rust
#[serde(default = "default_fitness_bias_strength")]
pub fitness_bias_strength: f32,  // 0.0 = no bias, 1.0 = full bias toward favorites
```
Default: 0.5

**Step 2: Analyze saved favorites**

Create a function that scans `genomes/` (not seeds/) for saved genomes and builds a profile:

```rust
pub struct FavoriteProfile {
    pub variation_freq: HashMap<String, f32>,  // how often each variation appears
    pub avg_symmetry: f32,
    pub avg_transform_count: f32,
    pub avg_scale: f32,
}

impl FavoriteProfile {
    pub fn from_directory(dir: &Path) -> Self { ... }
}
```

**Step 3: Bias mutation toward favorites**

In `mutate_perturb()`, when boosting a variation:
- Without bias: pick random variation
- With bias: weight the pick toward variations common in favorites

```rust
let profile = FavoriteProfile::from_directory(&genomes_dir);
let variation_weights: Vec<f32> = VARIATION_NAMES.iter()
    .map(|name| {
        let freq = profile.variation_freq.get(*name).unwrap_or(&0.1);
        let uniform = 1.0 / VARIATION_NAMES.len() as f32;
        uniform * (1.0 - bias) + freq * bias
    })
    .collect();
// Weighted random pick from variation_weights
```

**Step 4: Cache the profile**

Don't re-scan the directory every mutation. Cache it in App state, refresh every 30 seconds or on 's' keypress.

**Step 5: Verify**

Save a few genomes with heavy julia/spiral usage. Mutations should start favoring julia/spiral over time.

**Step 6: Commit**

```bash
git add src/genome.rs src/weights.rs weights.json
git commit -m "feat: fitness-biased mutation — favorites influence variation picks"
```

---

### Task 6: Fix accumulation/morph darkening

**Files:**
- Modify: `accumulation.wgsl` — handle morph transitions
- Modify: `src/main.rs` — clear accumulation on genome change

**Step 1: Clear accumulation buffer on evolve**

When `begin_morph()` is called, clear the accumulation buffer so old patterns don't ghost:

```rust
fn begin_morph(&mut self) {
    // ... existing morph setup ...

    // Clear accumulation buffer — new genome starts fresh
    if let Some(gpu) = &self.gpu {
        gpu.queue.write_buffer(
            &gpu.accumulation_buffer,
            0,
            &vec![0u8; (gpu.config.width * gpu.config.height * 16) as usize],
        );
    }
}
```

Note: The design doc said "DO NOT clear on mutation." But in practice, the morph creates a dim intermediate period where old and new attractors overlap poorly. Clearing gives a clean start — the accumulation rebuilds in ~3 seconds which is within the morph duration.

Alternative (gentler): Instead of full clear, temporarily increase decay to 0.9 during morphs so old data fades faster. This preserves some visual continuity.

**Step 2: Verify**

Run: `cargo run`, evolve several times
Expected: No more darkening period between genomes. Brightness should be consistent.

**Step 3: Commit**

```bash
git add src/main.rs accumulation.wgsl
git commit -m "fix: clear accumulation buffer on evolve to prevent darkening"
```

---

### Task 7: Integration test with real flame files

**Files:**
- Create: `genomes/flames/` directory with downloaded .flame files

**Step 1: Download community flames**

Download 5-10 `.flame` files from the Electric Sheep archive or Apophysis community. Place in `genomes/flames/`.

**Step 2: Test import**

Run: `cargo run`, press 'f' to load flames
Expected:
- Affine transforms render correctly (shear, non-uniform scale visible)
- Custom palettes display (not just cosine colors)
- Parametric variations render (julian spirals, blob modulation)
- No crashes on unsupported variations (graceful skip)

**Step 3: Visual comparison**

Compare imported flames with their original renders (if available). Focus on:
- Overall structure matches
- Color palette matches
- Fine detail may differ (we skip unsupported variations)

**Step 4: Commit**

```bash
git add genomes/flames/
git commit -m "feat: add community flame files for testing import"
```

---

## Task Dependencies

```
Task 1 (affine) ──┬──→ Task 3 (XML import) ──→ Task 4 (parametric) ──→ Task 7 (integration)
                  │
Task 2 (palettes) ┘
Task 5 (fitness) ────→ (independent, can run anytime after Task 1)
Task 6 (fix dark) ───→ (independent, should do early)
```

**Recommended order:** 6, 1, 2, 3, 4, 5, 7

---

## Summary

| Task | What | Effort |
|------|------|--------|
| 1 | Full affine transforms (a,b,c,d,e,f) | Large — touches genome, shader, mutations, seeds |
| 2 | Per-genome 256-entry color palettes | Medium — new texture, bind group expansion |
| 3 | Flam3 XML importer | Medium — new module, XML parsing |
| 4 | Parametric variations | Medium — shader changes, buffer expansion |
| 5 | Fitness-biased mutation | Small — directory scan, weighted picks |
| 6 | Fix accumulation darkening | Small — buffer clear on evolve |
| 7 | Integration test with real flames | Small — download + verify |

## Success Criteria

- Can import any `.flame` file and render it recognizably
- Custom color palettes display per genome
- Parametric variations (julian, blob, rings2) render correctly
- Favorites influence mutation direction
- No darkening between evolves
- Visual quality approaches Electric Sheep in motion
