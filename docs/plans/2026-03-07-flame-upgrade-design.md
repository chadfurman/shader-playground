# Flame System Upgrade Design

**Goal:** Close the visual gap with Electric Sheep — fill the frame with deep, rich, varied fractal detail instead of sparse scattered clusters on black.

**Problem:** Current flames are thin and sparse compared to Electric Sheep. We have only 6 variations, no final transform, no symmetry, and tonemapping that clips dynamic range early. Color is a single averaged palette index rather than rich blended RGB.

**Approach:** Layered rollout — rendering first (instant visual payoff for existing flames), then variations (structural variety), then final transform + symmetry (global warping and density multiplication).

---

## Phase 1: Rendering Overhaul

### Histogram Format Change

Current: 2 x u32 per pixel (density, color_sum)
New: 4 x u32 per pixel (density, color_r, color_g, color_b)

During iteration, each point looks up its blended color_idx through the palette to get RGB, then accumulates those RGB values into the histogram. This produces rich color mixing — overlapping structures blend colors naturally instead of averaging a single index.

Buffer size: `width * height * 4 * sizeof(u32)` (4x current).

### Log-Density Tonemapping

Replace current `log(1 + density) / 5.0` capped at 1.2 with proper Electric Sheep log-density:

```
k = log(1 + density * brightness) / log(1 + max_density * brightness)
```

Where `max_density` is a rolling peak across the frame (same pattern as BandNormalizer — instant snap up, slow decay). This maps the full density range to 0-1 without clipping.

Final pixel color:
```
rgb = (accumulated_rgb / density) * k
```

The division recovers the average color, then `k` scales by log-density.

### Vibrancy

Electric Sheep's vibrancy parameter (0-1) blends between:
- The raw palette color (at vibrancy=0)
- A luminance-preserving saturated version (at vibrancy=1)

Applied per-pixel based on alpha (density). Low-density wisps stay muted; high-density cores get punchy saturated color. The formula:

```
lum = 0.299*r + 0.587*g + 0.114*b
color = mix(vec3(lum), color, pow(alpha, 1.0 - vibrancy))
```

Default vibrancy: 0.7 (fairly saturated). Weightable param driven by bass for audio reactivity.

### Bloom Post-Process

Two-pass separable Gaussian blur on the flame layer:
1. Horizontal blur pass → intermediate texture
2. Vertical blur pass → bloom texture
3. Composite: `final = flame + bloom * bloom_intensity`

Blur radius: ~8-12 pixels at 1080p. bloom_intensity default: 0.2.

This requires one additional texture and either a compute pass or fragment shader pass for the blur. Using the existing feedback texture pipeline as reference for the plumbing.

### New Weightable Params

- `bloom_intensity`: base 0.2, audio-driven by energy + beat_pulse
- `vibrancy`: base 0.7, audio-driven by bass

---

## Phase 2: New Variations (20)

Expand from 6 to 26 variation functions. Each is a `vec2<f32> -> vec2<f32>` function in the compute shader.

### Variation List

| # | Name | Formula (pseudocode) | Visual character |
|---|------|---------------------|------------------|
| 1 | julia | `sqrt(r) * (cos(theta/2 + k*PI), sin(theta/2 + k*PI))`, k random 0 or 1 | Bulbous spirals, organic blobs |
| 2 | polar | `(theta/PI, r - 1)` | Radial fans, spoke patterns |
| 3 | disc | `(theta/PI * sin(PI*r), theta/PI * cos(PI*r))` | Circular spreading |
| 4 | rings | `((r^2 mod c^2 + r) * cos(theta), ..sin)`, c = scale^2 | Concentric rings |
| 5 | bubble | `p * 4 / (dot(p,p) + 4)` | Glowing orbs |
| 6 | fisheye | `2 * p / (r + 1)` | Wide-angle warping |
| 7 | exponential | `exp(x-1) * (cos(PI*y), sin(PI*y))` | Dramatic curves |
| 8 | spiral | `(cos(theta) + sin(r)) / r, (sin(theta) - cos(r)) / r` | Nautilus spirals |
| 9 | diamond | `(sin(theta)*cos(r), cos(theta)*sin(r))` | Angular crystalline |
| 10 | bent | `if x<0: x*2; if y<0: y/2` | Asymmetric organic warping |
| 11 | waves | `(x + b*sin(y/c^2), y + e*sin(x/f^2))` with per-xform b,c,e,f | Rippled distortion |
| 12 | popcorn | `(x + c*sin(tan(3y)), y + f*sin(tan(3x)))` | Fine organic texture |
| 13 | fan | Angular slicing with phase offset | Fan/petal shapes |
| 14 | eyefish | `2*p / (r + 1)` (2D variant) | Dome warping |
| 15 | cross | `sqrt(1/(x^2 - y^2)^2) * p` | Sharp cross/star |
| 16 | tangent | `(sin(x)/cos(y), tan(y))` | Spiky vertical chaos |
| 17 | cosine | `(cos(PI*x)*cosh(y), -sin(PI*x)*sinh(y))` | Flowing curtains |
| 18 | blob | `r * (a + (b-a)/2 * (sin(c*theta)+1))` | Soft blobby forms |
| 19 | noise | `p + perlin_2d(p)` (hash-based, no texture) | Organic irregularity |
| 20 | curl | `p + curl_noise_2d(p)` (divergence-free noise) | Fluid smoke tendrils |

### Genome Impact

`FlameTransform` grows from 12 fields to 32 fields (6 old + 20 new variation weights + existing 6 non-variation fields).

Storage buffer layout: 12 floats/transform → 32 floats/transform. The `xf()` accessor in the shader updates accordingly.

`waves` and `popcorn` need per-transform parameters (b, c, e, f). These can be packed as extra fields or derived from the existing offset/angle values to avoid growing the struct further. Recommended: derive from hash of transform index + a genome-level "wave_freq" param.

### Mutation Impact

`mutate_perturb` "reinvent as specialist" now picks from 26 variations instead of 6. No other mutation changes needed — the existing swap/perturb logic generalizes.

### Weights Impact

The `xfN_` wildcard system already handles arbitrary variation names. New entries in weights.json `_params` doc: `xfN_julia`, `xfN_polar`, etc. No code change in the weights system.

---

## Phase 3: Final Transform + Symmetry

### Final Transform

A `FlameTransform` applied after the IFS iteration loop, before screen projection. It has the same 26 variation weights as regular transforms but is NOT in the weighted random selection — it always applies to every point.

Compute shader change:
```
// After iteration loop:
p = apply_final_xform(p, t);  // same apply_xform logic, different data source
// Then project to screen as before
```

Storage: The final transform is packed into the same storage buffer after the regular transforms. The compute shader reads `transform_count` regular transforms, then one more as the final.

Genome: `final_transform: Option<FlameTransform>`. None = no final transform (backwards compat). Serialized as `"final_transform": null` or `"final_transform": { ... }` in JSON.

### Symmetry

After computing a screen-space point, plot additional rotated copies.

```
symmetry_order from genome (passed via uniform)

for k in 0..abs(symmetry_order):
    rotated_p = rotate(p, 2*PI*k / abs(symmetry_order))
    plot(rotated_p)
    if symmetry_order < 0:  // bilateral
        plot((-rotated_p.x, rotated_p.y))
```

This multiplies effective sample density by N (or 2N for bilateral). A 4-fold symmetry fills 4x as many pixels from the same iteration count — directly combats sparseness.

Genome: `symmetry: i32`. Default 1 (none). Positive = rotational N-fold. Negative = bilateral + rotational.

Uniform: One extra u32 packed into the existing padding slot or extra vec4.

### Mutation

New mutation types added to the existing `mutate()` function:
- `mutate_final_transform`: perturb the final transform variations/angle/scale (same logic as perturb but targets final_xform)
- `mutate_symmetry`: occasionally change symmetry_order. Biased toward low values (1-4) with rare higher (5-8). ~20% chance of bilateral.

Fresh genomes: 20% chance of symmetry > 1 to get a mix of symmetric and asymmetric flames.

### Weights

New prefix `xf_final_` for targeting the final transform via weights. Time signals could slowly drift the final transform for evolving global warping.

`symmetry_order` is structural (integer), not continuously weightable.

---

## Phase 4: Weights & Audio Integration

### New Weightable Params

| Param | Default | Suggested Audio | Purpose |
|-------|---------|----------------|---------|
| bloom_intensity | 0.2 | energy 0.3, beat_pulse 0.5 | Glow amount |
| vibrancy | 0.7 | bass 0.2 | Color saturation |

### Updated weights.json Structure

```json
{
  "energy": {
    "flame_brightness": 0.7,
    "drift_speed": 1.5,
    "mutation_rate": 0.3,
    "bloom_intensity": 0.3
  },
  "beat_pulse": {
    "bloom_intensity": 0.5
  },
  "bass": {
    "color_shift": -0.4,
    "flame_brightness": 0.3,
    "vibrancy": 0.2
  }
}
```

### No New Signals Needed

The 7 audio signals (bass, mids, highs, energy, beat, beat_accum, beat_pulse) and 8 time signals are sufficient. The new params slot into the existing weights system.

---

## What Changes

- **flame_compute.wgsl**: 20 new variation functions, final transform application, symmetry point plotting, RGBA histogram accumulation, storage buffer layout (32 floats/xform)
- **playground.wgsl**: New tonemapping (log-density with rolling peak), vibrancy, bloom composite
- **New bloom pass**: Separable Gaussian blur (compute or fragment shader)
- **src/genome.rs**: FlameTransform grows to 32 fields, new final_transform + symmetry fields, new mutation types, updated default genome
- **src/main.rs**: Histogram buffer 4x size, bloom texture + pass plumbing, new uniform slots
- **src/weights.rs**: bloom_intensity + vibrancy params added
- **weights.json**: New audio mappings for bloom + vibrancy

## What Doesn't Change

- AudioFeatures struct, audio signals, audio normalization
- Time signals system
- The `xfN_` wildcard expansion logic
- Basic IFS chaos game loop structure
- Feedback/trail system

---

## Risk Notes

- **Histogram 4x larger**: At 1080p, goes from ~16MB to ~64MB. M3 has plenty of VRAM, but worth checking.
- **26 variation branches in shader**: GPU handles this fine — the branch is per-thread and each thread typically takes the same path (same transform).  The compiler will likely flatten the variation selection.
- **Bloom pass latency**: Extra texture copy + 2 blur passes. Should be <1ms on M3 at 1080p.
- **Symmetry * iterations**: 4-fold symmetry with 131K threads effectively becomes 524K points per frame. Free performance win.
