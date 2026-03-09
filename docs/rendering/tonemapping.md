# Tonemapping

The display shader (`playground.wgsl`) converts the raw accumulation buffer into
final pixel colors. This document covers every stage of that density-to-color
pipeline, from reading fixed-point accumulation data through to the final
clamped RGB output.

Source: `playground.wgsl` (fragment shader `fs_main`)

---

## Accumulation Buffer Reading

The accumulation buffer stores 7 `f32` channels per pixel, laid out
contiguously:

```
buf_idx = (py * width + px) * 7
```

| Offset | Channel   | Encoding |
|--------|-----------|----------|
| +0     | density   | hits * 1000 (fixed-point from bilinear splatting) |
| +1     | R         | red * 1000 (same fixed-point scale as density) |
| +2     | G         | green * 1000 |
| +3     | B         | blue * 1000 |
| +4     | vel_x     | velocity_x * 10000, weighted by splat (signed) |
| +5     | vel_y     | velocity_y * 10000, weighted by splat (signed) |
| +6     | depth     | iteration depth, weighted by splat |

All channels share the same bilinear-splat weighting, so ratios cancel
cleanly:

- **Color recovery**: `rgb = [R, G, B] / density` (fixed-point scales cancel)
- **Velocity recovery**: `vel_raw / (density * 10) * resolution` -- the extra
  factor of 10 accounts for the 10000/1000 ratio between velocity and density
  encoding, and multiplication by resolution converts from normalized to pixel
  space.
- **Depth recovery**: `depth / density`

The accumulation buffer itself is maintained by `accumulation.wgsl`, which
blends each new frame's histogram into the persistent buffer using exponential
decay (`accum = accum * decay + new_frame`). The decay rate is controlled by
`accumulation_decay` (default 0.94).

---

## Velocity Blur (Pre-Tonemapping)

Before tonemapping, the shader applies directional blur along the velocity
field. This happens on the raw accumulation values (density + RGB), not on
tonemapped color.

```wgsl
let avg_vel = vel_raw / (density * 10.0) * u.resolution;
let blur_len = clamp(length(avg_vel), 0.0, blur_max);
let tap_spacing = blur_len / 8.0;
```

- 8 taps forward + 8 taps backward along the velocity direction (16 total)
- Each tap is weighted by distance falloff: `weight = 1.0 / (1.0 + tap * 0.3)`
- Blur is skipped entirely when `blur_len < 0.5` pixels
- Maximum blur length is clamped to `velocity_blur_max` (default 24.0 pixels)

The blurred density and RGB (`blur_density`, `blur_r`, `blur_g`, `blur_b`)
are used for all subsequent tonemapping stages.

---

## Log-Density Mapping

The core of the flam3 rendering algorithm. Converts hit counts into
perceptually linear brightness values.

### Step 1: Convert to hit count

```wgsl
let density_hits = blur_density / 1000.0;
```

Removes the fixed-point encoding to get the actual number of chaos-game
hits at this pixel (after velocity blur accumulation).

### Step 2: Log compression

```wgsl
let log_density = log(1.0 + density_hits * flame_brightness);
```

The natural log compresses the enormous dynamic range of fractal density
(which can span several orders of magnitude) into a workable range.
`flame_brightness` scales the density before the log, controlling how
quickly the curve saturates. This is a signal-modulated parameter with
a base of ~0.4.

### Step 3: Per-image normalization

```wgsl
let max_density_bits = max_density_buf[0];
let max_density_val = bitcast<f32>(max_density_bits) / 1000.0;
let max_log = log(1.0 + max_density_val * flame_brightness);
```

The accumulation pass (`accumulation.wgsl`) tracks the maximum density
across all pixels using `atomicMax` on the bitcast `u32` representation
(IEEE 754 preserves ordering for positive floats). The display shader
reads this value and computes the same log transform on it to establish
the normalization ceiling.

This per-image normalization ensures every genome automatically uses the
full brightness range regardless of how many points land on screen.

### Step 4: Alpha computation (tonemapping curve)

The normalized log-density is mapped to a 0-1 alpha value using one of
two selectable curves. See the next section.

---

## Tonemapping Modes

Selected by `tonemap_mode` in `_config` (default 1). Stored in
`extra4.y`.

### Mode 0: Sqrt-Log (flam3 style)

```wgsl
alpha = sqrt(log_density / max(max_log, 0.001));
```

The classic Scott Draves algorithm. The square root further compresses
highlights and lifts shadows, producing the characteristic soft, luminous
look of traditional flame renders. Division by `max_log` normalizes to
[0, 1] before the sqrt.

### Mode 1: ACES Filmic

```wgsl
let normalized = log_density / max(max_log, 0.001);
alpha = aces_tonemap(normalized);
```

Uses the Krzysztof Narkowicz fit of the ACES (Academy Color Encoding
System) filmic curve:

```wgsl
fn aces_tonemap(x: f32) -> f32 {
    return clamp(
        (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
        0.0, 1.0
    );
}
```

This S-curve provides deeper blacks and a more cinematic highlight
rolloff compared to the sqrt curve. The five constants (2.51, 0.03,
2.43, 0.59, 0.14) define the ACES curve shape -- they are the algorithm
itself, not tunable parameters.

---

## Histogram Equalization

Adaptive density equalization that redistributes brightness to use the
full tonal range, reducing the "bright core, dark edges" problem common
in fractal flames.

### CDF Construction (`histogram_cdf.wgsl`)

A two-pass compute pipeline builds the equalization lookup:

**Pass 1 (`bin_densities`)**: Each pixel's log-density is mapped to one
of 256 bins using:

```wgsl
let log_d = log(1.0 + density * flame_brightness);
let bin = u32(clamp(log_d / (log_d + 4.0) * 255.0, 0.0, 255.0));
```

The `log_d / (log_d + 4.0)` mapping compresses the bin distribution so
that low-density pixels (the majority) get finer bin resolution. Empty
pixels are skipped.

**Pass 2 (`prefix_sum`)**: A Hillis-Steele parallel prefix sum computes
the cumulative distribution function across the 256 bins, then
normalizes to [0, 1] by dividing by the total count.

### Application in Display Shader

```wgsl
let hist_eq = u.extra4.z;  // histogram_equalization strength
if (hist_eq > 0.001) {
    let bin = u32(clamp(alpha * 255.0, 0.0, 255.0));
    let equalized = cdf[bin];
    alpha = mix(alpha, equalized, hist_eq);
}
```

The current alpha (from the tonemapping curve) is used as a bin index
into the CDF. The result is blended with the original alpha using
`histogram_equalization` as the mix factor:

- `0.0` = pure tonemapping curve (no equalization)
- `1.0` = full histogram equalization
- `0.3` (default) = subtle lift of dark regions while preserving the
  tonemapping character

### Post-Equalization Clamp

```wgsl
alpha = clamp(alpha, 0.0, 1.0);
```

Velocity blur can sum neighboring densities above `max_density`, pushing
alpha above 1.0. This clamp prevents bright white flashes.

---

## Color Recovery

After alpha is finalized, the average color is recovered from the
blurred accumulation:

```wgsl
let raw_color = select(
    vec3(0.0),
    vec3(blur_r, blur_g, blur_b) / max(blur_density, 1.0),
    blur_density > 0.0
);
```

Since RGB and density use the same fixed-point scale (* 1000), dividing
RGB by density yields the true average color at each pixel. Empty pixels
(zero density) return black.

---

## Edge Glow

Structure-aware edge detection that reveals boundaries between
overlapping transforms. Uses a simple finite-difference gradient on the
raw accumulation buffer (not the tonemapped result).

### Density Gradient

Samples the four cardinal neighbors (left, right, top, bottom) from the
accumulation buffer:

```wgsl
let grad_x = d_right - d_left;
let grad_y = d_bot - d_top;
let density_edge = sqrt(grad_x * grad_x + grad_y * grad_y);
```

This detects boundaries where density changes sharply -- the edges of
the fractal structure itself.

### Color Gradient

The same four neighbors are used to compute average color at each
position, then the magnitude of the color difference is measured:

```wgsl
let col_left  = vec3(R, G, B) / max(d_left, 1.0);
let col_right = vec3(R, G, B) / max(d_right, 1.0);
// ... same for top/bottom
let color_grad_x = length(col_right - col_left);
let color_grad_y = length(col_bot - col_top);
let color_edge = sqrt(color_grad_x * color_grad_x + color_grad_y * color_grad_y);
```

Color edges reveal where different transforms overlap -- even if density
is smooth, a shift in average color indicates a structural boundary.

### Combination

The two edge signals are combined into a single glow value:

```wgsl
let edge_glow = clamp(density_edge / max(blur_density * 0.3, 1.0), 0.0, 0.5)
              + clamp(color_edge * 2.0, 0.0, 0.3);
```

- **Density edge**: normalized by local density (`blur_density * 0.3`),
  so edges scale proportionally to local structure. This prevents sparse
  regions from dominating. Clamped to [0, 0.5].
- **Color edge**: scaled by 2x and clamped to [0, 0.3].
- **Maximum combined contribution**: 0.8 (additive into the color blend).

The density-proportional normalization is key -- it makes the glow
structure-aware rather than simply highlighting high-density boundaries.

---

## Vibrancy Color Blend

The flam3 algorithm's color application, combining alpha, gamma
correction, and edge glow into the final pre-bloom color.

### Gamma-Corrected Alpha

```wgsl
let gamma_alpha = pow(max(alpha, 0.001), gamma);
```

Raises alpha to the gamma power (default 0.4545, i.e., 1/2.2 for
standard sRGB display gamma). The `max(alpha, 0.001)` prevents `pow(0,
x)` edge cases.

### Vibrancy Blend

```wgsl
let ls = vibrancy * alpha + (1.0 - vibrancy) * gamma_alpha;
```

This is Scott Draves' vibrancy formula. It interpolates between:

- **Linear alpha** (`alpha`): preserves the raw density-to-brightness
  mapping. Colors in dense regions stay saturated because brightness
  tracks density directly.
- **Gamma-corrected alpha** (`gamma_alpha`): standard display gamma
  curve. Lifts shadows but can wash out color in bright regions.

At `vibrancy = 1.0`, brightness is purely linear (maximum saturation).
At `vibrancy = 0.0`, brightness is purely gamma-corrected (standard
display curve). The default of 0.7 leans toward saturated color while
still lifting shadows slightly.

### Final Color Application

```wgsl
var col = (ls + edge_glow) * raw_color;
```

The combined luminance scale (`ls` + edge glow) multiplies the recovered
average color. Edge glow adds extra brightness at structural boundaries,
making them subtly visible even when the base luminance is low.

---

## Post-Processing

After the vibrancy blend, two more stages modify the final color before
output.

### Feedback Trail

```wgsl
let prev = textureSample(prev_frame, prev_sampler, prev_uv).rgb;
col = max(col, prev * trail);
```

The previous frame is sampled (optionally with temporal reprojection to
correct for zoom changes) and blended using a `max` operation scaled by
`trail` (default 0.15). This creates persistence/afterglow without
additive blowout.

### Density-Aware Bloom

A 3-radius separable bloom (r=2, r=5, r=12 pixels) samples the previous
frame at cardinal offsets, weighted by sparsity:

```wgsl
let norm_density = clamp(density / max(max_d, 1.0), 0.0, 1.0);
let sparsity = 1.0 - sqrt(norm_density);
col += bloom_sum * bloom_int * sparsity;
```

- Sparse pixels (low density) get full bloom -- individual points glow
- Dense pixels get almost no bloom -- crisp structural detail preserved
- Three radii at decreasing weights (0.5, 0.3, 0.2) create a soft falloff

### Final Clamp

```wgsl
col = clamp(col, vec3(0.0), vec3(1.0));
```

Hard clamp to [0, 1]. No additional Reinhard or gamma is applied here
because the vibrancy blend already incorporated gamma correction.

---

## Config Reference

All `_config` fields used in the tonemapping pipeline:

| Field | Default | Uniform Slot | Description |
|-------|---------|--------------|-------------|
| `flame_brightness` | ~0.4 (signal-modulated) | `globals.w` | Density scale before log compression. Higher = brighter, faster saturation. |
| `vibrancy` | 0.7 | `extra.y` | Blend between linear alpha (saturated) and gamma alpha (lifted shadows). 0.0-1.0. |
| `bloom_intensity` | 0.05 | `extra.z` | Strength of density-aware bloom glow. 0.0 = off. |
| `gamma` | 0.4545 | `extra.w` | Display gamma exponent. 0.4545 = 1/2.2 (sRGB standard). |
| `velocity_blur_max` | 24.0 | `extra3.w` | Maximum directional blur length in pixels along velocity field. |
| `tonemap_mode` | 1 | `extra4.y` | Tonemapping curve: 0 = sqrt-log (flam3), 1 = ACES filmic. |
| `histogram_equalization` | 0.3 | `extra4.z` | Adaptive equalization strength. 0.0 = off, 1.0 = full equalization. |
| `dof_strength` | 0.2 | `extra4.w` | Depth of field blur strength. 0.0 = off. |
| `dof_focal_distance` | 1.0 | `extra5.x` | Focal plane distance for DoF. |
| `temporal_reprojection` | 0.5 | `extra5.z` | Zoom-corrected feedback trail blending. 0.0 = off. |
| `trail` | 0.15 | `globals.z` | Feedback trail decay -- previous frame retention via max blend. |
| `accumulation_decay` | 0.94 | (accumulation pass) | Exponential decay rate for the accumulation buffer. |

---

## Pipeline Summary

```
Accumulation Buffer (7-channel fixed-point)
    |
    v
Velocity Blur (16-tap directional, pre-tonemapping)
    |
    v
Depth of Field Blur (8-tap radial, if enabled)
    |
    v
Log-Density Compression (log(1 + hits * brightness))
    |
    v
Per-Image Normalization (divide by max_log from histogram reduce)
    |
    v
Tonemapping Curve (sqrt or ACES filmic)
    |
    v
Histogram Equalization (CDF lookup, blended)
    |
    v
Alpha Clamp [0, 1]
    |
    v
Color Recovery (RGB / density)
    |
    v
Edge Detection (density gradient + color gradient)
    |
    v
Vibrancy Blend (ls = vibrancy*alpha + (1-vibrancy)*gamma_alpha)
    |
    v
Final Color = (ls + edge_glow) * raw_color
    |
    v
Feedback Trail (max blend with previous frame)
    |
    v
Density-Aware Bloom (3-radius, sparsity-weighted)
    |
    v
Hard Clamp [0, 1] --> Output
```
