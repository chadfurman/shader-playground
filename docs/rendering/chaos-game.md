# Chaos Game

Reference-level documentation of the compute shader (`flame_compute.wgsl`).

## Overview

The chaos game is an iterated function system (IFS) renderer running entirely on
the GPU as a WebGPU compute shader. Each GPU thread represents one persistent
point that wanders through the attractor by repeatedly applying randomly
selected affine transforms + nonlinear variation functions. The resulting
positions are splatted into a shared histogram buffer that the display shader
reads for tonemapping and compositing.

The shader dispatches with `@workgroup_size(256)`. Each thread:
1. Loads its persistent state (position + color index)
2. Runs `max_iters` chaos-game iterations
3. Splats each post-warmup iteration into the histogram
4. Writes back its final state for the next frame

## Point Initialization

### Persistent State

Each thread owns 3 consecutive `f32` values in the `point_state` storage buffer,
indexed by `gid.x * 3`:

| Offset | Value       | Range        |
|--------|-------------|--------------|
| 0      | `p.x`       | unbounded    |
| 1      | `p.y`       | unbounded    |
| 2      | `color_idx` | [0, 1]       |

State persists across frames and genome changes, giving temporal continuity.

### Re-randomization

A point is re-initialized when any of these conditions hold:

| Condition | Test | Purpose |
|-----------|------|---------|
| First frame | `abs(x) < 1e-10 && abs(y) < 1e-10 && color < 1e-10` | Buffer starts zeroed |
| Escaped | `abs(x) > 10 || abs(y) > 10` | Point left the attractor |
| NaN | `x != x || y != y` | Numerical blowup |
| Random refresh | `randf() < 0.05` (5% per frame) | Maintain fresh coverage |

When re-initialized:
- Position: uniform random in `[-2, 2] x [-2, 2]`
- Color index: uniform random in `[0, 1]`

## Random Number Generator

Uses the PCG (Permuted Congruential Generator) algorithm seeded per-thread:

```
seed = gid.x * 2654435761 + frame * 7919 + 12345
```

The `pcg()` function advances state with multiply-add, then applies a
xorshift-multiply-xorshift output permutation. `randf()` maps the `u32` result
to `[0, 1]` by dividing by `2^32 - 1`.

## Iteration Loop

### Iteration Count

Packed into the upper 16 bits of the `has_final_xform` uniform:

```
max_iters = max(has_final_xform >> 16, 10)
```

The default comes from `iterations_per_thread` in `weights.json` `_config`
(typically 200). The floor of 10 prevents degenerate cases.

### Transform Selection

Each iteration selects a transform by weighted random sampling:

1. Precompute `total_weight` by summing `xf(t, 0)` (the `weight` field) across
   all transforms
2. Draw `r = randf() * total_weight`
3. Walk the cumulative sum until `r < cumsum` — that index wins

If `total_weight < 1e-6`, the thread returns early (degenerate genome).

### Per-Iteration Flow

```
for i in 0..max_iters:
    select transform tidx by weighted random
    save prev_p = p
    p = apply_xform(p, tidx, time, rng)
    blend color_idx toward xform color
    if i < warmup_iters: skip splatting (continue)
    compute velocity, final transform, luminosity
    splat to histogram
```

## Affine Transform Application

The `apply_xform()` function applies the full transform pipeline for a selected
transform index.

### Affine Matrix

The 6-parameter affine is stored as fields `[1..6]` in the per-transform buffer:

```
q.x = a * p.x + b * p.y + offset_x
q.y = c * p.x + d * p.y + offset_y
```

This is the 2x2 matrix `[[a, b], [c, d]]` plus translation `[offset_x, offset_y]`.

### Time-Varying Animation

Each transform gets unique animation derived from a hash of its index:

```
seed = hash(idx * 31337 + 42)
drift_amt = 0.3 + hash_f(seed) * 0.7
```

**Spin drift** (rotation over time):
- `spin_speed = (hash_f(seed+300) * 2 - 1) * spin_speed_max`
- `angle_drift = time * spin_speed * drift_speed * drift_amt`
- Applied by composing a rotation matrix with the affine:
  ```
  a' = a*cos(angle) - c*sin(angle)
  b' = b*cos(angle) - d*sin(angle)
  c' = a*sin(angle) + c*cos(angle)
  d' = b*sin(angle) + d*cos(angle)
  ```

**Position drift** (translation wobble via value noise):
- Only computed when `position_drift > 0.001`
- `ox_drift = vnoise(t * 0.03 * drift * drift_amt, seed+100) * position_drift`
- `oy_drift = vnoise(t * 0.04 * drift * drift_amt, seed+200) * position_drift`
- Added to the affine offsets

Both are gated on `drift_speed > 0.001` (from `kifs.w` uniform).

Config params: `spin_speed_max` (extra3.x), `position_drift` (extra3.y),
`drift_speed` (kifs.w).

## Variation Functions

After the affine, the point `q` is passed through a weighted sum of nonlinear
variation functions. The linear variation weight seeds the accumulator:
`v = q * w_linear`. Each active variation adds `V_i(q) * w_i`.

Variations are organized into three tiers for performance:

- **Tier 1** (cheap, always checked): linear through handkerchief
- **Tier 2** (moderate, atan2-based): julia through fisheye
- **Tier 3** (expensive/scatter-prone): guarded by `tier3_sum > 0` check

### Per-Transform Buffer Layout

Each transform occupies 42 floats. Fields 8-33 are variation weights:

| Offset | Field | Tier |
|--------|-------|------|
| 0 | weight | — |
| 1-4 | a, b, c, d | affine |
| 5-6 | offset_x, offset_y | affine |
| 7 | color | — |
| 8 | linear | 1 |
| 9 | sinusoidal | 1 |
| 10 | spherical | 1 |
| 11 | swirl | 1 |
| 12 | horseshoe | 1 |
| 13 | handkerchief | 1 |
| 14 | julia | 2 |
| 15 | polar | 2 |
| 16 | disc | 2 |
| 17 | rings | 2 |
| 18 | bubble | 2 |
| 19 | fisheye | 2 |
| 20 | exponential | 3 |
| 21 | spiral | 3 |
| 22 | diamond | 3 |
| 23 | bent | 3 |
| 24 | waves | 3 |
| 25 | popcorn | 3 |
| 26 | fan | 3 |
| 27 | eyefish | 3 |
| 28 | cross | 3 |
| 29 | tangent | 3 |
| 30 | cosine | 3 |
| 31 | blob | 3 |
| 32 | noise | 3 |
| 33 | curl | 3 |
| 34 | rings2_val (param) | — |
| 35 | blob_low (param) | — |
| 36 | blob_high (param) | — |
| 37 | blob_waves (param) | — |
| 38-41 | reserved params | — |

### Complete Variation Reference

| Idx | Name | Shader Function | Formula | Notes |
|-----|------|-----------------|---------|-------|
| 0 | Linear | (inline) | `q` | Identity; seeds the accumulator |
| 1 | Sinusoidal | `V_sinusoidal` | `(sin(x), sin(y))` | Element-wise sine |
| 2 | Spherical | `V_spherical` | `p / (dot(p,p) + 1e-6)` | Inversion; epsilon prevents division by zero |
| 3 | Swirl | `V_swirl` | `(x*sin(r2) - y*cos(r2), x*cos(r2) + y*sin(r2))` where `r2 = dot(p,p)` | Rotation by squared distance |
| 4 | Horseshoe | `V_horseshoe` | `((x-y)(x+y), 2xy) / r` i.e. `(x2-y2, 2xy) / r` | Conformal-like mapping |
| 5 | Handkerchief | `V_handkerchief` | `r * (sin(theta+r), cos(theta-r))` | Spiral wave pattern |
| 6 | Julia | `V_julia` | `sqrt(r) * (cos(theta/2 + k*pi), sin(theta/2 + k*pi))` where `k` is random 0 or 1 | Stochastic half-angle; uses RNG |
| 7 | Polar | `V_polar` | `(theta/pi, r - 1)` | Maps polar coords to Cartesian |
| 8 | Disc | `V_disc` | `(theta/pi) * (sin(pi*r), cos(pi*r))` | Disc-shaped mapping |
| 9 | Rings | `V_rings` | `rr * (cos(theta), sin(theta))` where `rr = r + c2 - 2*c2*floor((r+c2)/(2*c2)) + r*(1-c2)`, `c2 = rings2_val^2` | Modular radial pattern; uses `rings2_val` param (offset 34, default 0.5) |
| 10 | Bubble | `V_bubble` | `4p / (r2 + 4)` equivalently `p / (r2/4 + 1)` | Sphere-to-plane projection |
| 11 | Fisheye | `V_fisheye` | `2 * (y, x) / (r + 1)` | Wide-angle lens; swaps x and y |
| 12 | Exponential | `V_exponential` | `exp(x-1) * (cos(pi*y), sin(pi*y))` | Exponential radial mapping |
| 13 | Spiral | `V_spiral` | `(cos(theta) + sin(r), sin(theta) - cos(r)) / r` | Logarithmic spiral |
| 14 | Diamond | `V_diamond` | `(sin(theta)*cos(r), cos(theta)*sin(r))` | Cross-product of polar trig |
| 15 | Bent | `V_bent` | `if x<0: x*=2; if y<0: y*=0.5` | Piecewise asymmetric fold |
| 16 | Waves | `V_waves` | `(x + bx*sin(y*4), y + by*sin(x*4))` where `bx = offset_x*0.5`, `by = offset_y*0.5` | Sinusoidal displacement; amplitude derived from affine offsets |
| 17 | Popcorn | `V_popcorn` | `(x + cx*sin(tan(3y)), y + cy*sin(tan(3x)))` where `cx = offset_x*0.3`, `cy = offset_y*0.3` | Fractal displacement via nested trig; amplitude from affine offsets |
| 18 | Fan | `V_fan` | Radial fan: `fan_t = atan2(c, a)`, `t2 = pi*fan_t^2`. If `(theta+fan_t) mod t2 > t2/2`: rotate by `-t2/2`, else `+t2/2` | Fan blades; parameter derived from affine matrix angle |
| 19 | Eyefish | `V_eyefish` | `2p / (r + 1)` | Like fisheye but without x/y swap |
| 20 | Cross | `V_cross` | `sqrt(1 / ((x2 - y2)^2 + 1e-6)) * p` | Cross-shaped singularity pattern |
| 21 | Tangent | `V_tangent` | `clamp((sin(x)/cos(y), tan(y)), -tc, tc)` where `tc = tangent_clamp` | Clamped tangent mapping; `tangent_clamp` from extra2.z |
| 22 | Cosine | `V_cosine` | `(cos(pi*x)*cosh(y), -sin(pi*x)*sinh(y))` | Conformal cosine mapping |
| 23 | Blob | `V_blob` | `r * (low + (high-low)/2 * (sin(waves*theta) + 1)) * (cos(theta), sin(theta))` | Parametric: `blob_low` (offset 35, default 0.2), `blob_high` (offset 36, default 1.0), `blob_waves` (offset 37, default 5.0) |
| 24 | Noise | `V_noise` | `p + (vnoise(x*3, seed+500), vnoise(y*3, seed+600)) * noise_displacement` | Stochastic; amplitude from `noise_displacement` (extra2.x) |
| 25 | Curl | `V_curl` | `p + (dy, -dx) * curl_displacement` where dx/dy are finite-difference gradients of value noise | Rotational flow field; amplitude from `curl_displacement` (extra2.y) |

## Global Effects

These are applied via the variation functions themselves (noise and curl are
variation slots 24-25) rather than as post-processing. They use global uniform
values rather than per-transform params:

- **Noise displacement** (`extra2.x`): Adds smooth random displacement via 1D
  value noise evaluated at the point position. Scale controlled by
  `noise_displacement` config.

- **Curl displacement** (`extra2.y`): Adds divergence-free rotational flow.
  Computed as finite-difference curl of a 2D noise field (eps = 0.01). Scale
  controlled by `curl_displacement` config.

- **Tangent clamping** (`extra2.z`): The tangent variation clamps its output to
  `[-tangent_clamp, tangent_clamp]` to prevent extreme values from escaping the
  attractor. Controlled by `tangent_clamp` config.

## Color Blending

Each iteration blends the running `color_idx` toward the selected transform's
color:

```
pos_color_offset = sin(p.x * 3.0) * cos(p.y * 3.0) * 0.05
color_idx = color_idx * (1 - cb) + (xform_color + pos_color_offset) * cb
```

Where:
- `cb` = `color_blend` from extra2.w (default 0.4 from config)
- `xform_color` = field 7 of the selected transform (a palette position in [0,1])
- `pos_color_offset` = position-dependent variation so nearby points get
  slightly different palette lookups, adding visual richness

The final palette lookup adds `color_shift` (extra.x) as a global hue rotation.

### Palette Lookup

Color is resolved from `color_idx` via a 256x1 palette texture uploaded from
the CPU. The lookup uses `fract(t)` so the palette wraps. When spectral
rendering is enabled, the palette texture is bypassed in favor of physics-based
spectral conversion (see Spectral Rendering below).

## Luminosity

Three factors combine to modulate per-splat brightness:

### Iteration Depth

```
iter_lum = 1.0 - iter_lum_range * (i / max_iters)
```

- `iter_lum_range` from extra6.y (default 0.0 = uniform brightness)
- At range 0.5, early iterations are ~1.0 brightness, final iterations ~0.5

### Distance Falloff

```
dist_lum = 1.0 / (1.0 + length(plot_p) * dist_lum_strength)
```

- `dist_lum_strength` from extra6.x (default 0.0 = disabled)
- When nonzero, points far from origin are dimmer (radial vignette effect)

### Transform Weight

```
clamp(xf_weight * 3.0, 0.3, 1.0)
```

Low-weight transforms (rarely selected) produce dimmer splats. The `* 3.0`
means transforms with weight >= 0.33 are at full brightness.

### Combined

```
lum = iter_lum * dist_lum * clamp(xf_weight * 3.0, 0.3, 1.0)
```

The final color sent to the histogram is `palette_color * lum`.

## Final Transform

When `has_final_xform & 1 == 1`, an extra transform is applied after each
iteration but before splatting. The final transform is stored immediately after
the regular transforms in the buffer (at index `transform_count`).

```
plot_p = apply_xform(plot_p, final_idx, time, rng)
plot_color = plot_color * 0.5 + final_xform_color * 0.5
```

The final transform:
- Applies the full affine + variations pipeline (same as any other transform)
- Blends color 50/50 with its own color value (field 7)
- Does NOT affect the persistent point state — only the splatted position
- Is never selected by the random weighted sampling

## Splatting

### Screen Mapping

The point position is mapped to screen coordinates:

```
screen = (sym_p / zoom + 0.5) * resolution + jitter
```

Where:
- `zoom` = globals.y (camera zoom level)
- `0.5` offset centers the attractor on screen
- `jitter` = sub-pixel random offset for free supersampling (magnitude from
  `jitter_amount`, extra4.x)

Jitter uses a separate RNG seed per symmetry copy per frame:
```
jitter_seed = gid.x * 3 + frame * 17 + si * 7
```

### Bilinear Sub-pixel Distribution

Each splat is distributed across a 2x2 pixel quad weighted by the fractional
screen position:

| Pixel | Weight |
|-------|--------|
| (cx, cy) | `(1-fx) * (1-fy)` |
| (cx+1, cy) | `fx * (1-fy)` |
| (cx, cy+1) | `(1-fx) * fy` |
| (cx+1, cy+1) | `fx * fy` |

**Optimization**: neighbor pixels are skipped when their sub-pixel offset
is < 10% (`fx > 0.1` / `fy > 0.1`), saving atomic operations when the point
lands near a pixel center.

### Histogram Channels

Each pixel occupies 7 atomic `u32` channels:

| Channel | Offset | Scale | Content |
|---------|--------|-------|---------|
| Density | +0 | x1000 | Hit count (for log-density tonemapping) |
| Red | +1 | x1000 x lum | Color red channel |
| Green | +2 | x1000 x lum | Color green channel |
| Blue | +3 | x1000 x lum | Color blue channel |
| Velocity X | +4 | x10000 | Screen-space velocity (for motion blur) |
| Velocity Y | +5 | x10000 | Screen-space velocity |
| Depth | +6 | x1000 | Distance from origin (for depth of field) |

All values are stored as integers via `atomicAdd` for GPU-safe accumulation.
The display shader divides by the scale factors to recover floating-point values.

Velocity is computed as `(p - prev_p) / zoom` — the per-iteration displacement
in screen space.

Depth is `length(sym_p)` — Euclidean distance from origin after symmetry
rotation.

## Symmetry

### Rotational Symmetry

When `symmetry != 0` (from extra.w), each splatted point is duplicated as
`abs(symmetry)` rotational copies:

```
for si in 0..abs(symmetry):
    angle = si * 2*pi / abs(symmetry)
    sym_p = rotate(plot_p, angle)
    splat(sym_p, ...)
```

### Bilateral Symmetry

When `symmetry < 0`, each rotational copy also gets a Y-axis mirror:

```
mir_p = (-sym_p.x, sym_p.y)
```

The mirrored copy also negates the X velocity component for correct motion blur
direction.

So `symmetry = -4` produces 4 rotational copies, each with a bilateral mirror,
for 8 total splats per iteration.

## Spectral Rendering

When `spectral_rendering > 0.5` (extra5.y), palette lookup is replaced with
physically-based spectral color:

### Pipeline

1. **Palette index to wavelength**: `wavelength = 380 + fract(color_idx + color_shift) * 400` nm
   (visible spectrum: 380-780 nm)

2. **CIE XYZ color matching** (Wyman et al. 2013 Gaussian approximation):
   - `cie_x(λ)`: Sum of 3 Gaussians (peaks near 442, 600, 501 nm)
   - `cie_y(λ)`: Sum of 2 Gaussians (peaks near 569, 531 nm)
   - `cie_z(λ)`: Sum of 2 Gaussians (peaks near 437, 459 nm)
   - Each uses asymmetric widths (different sigma for wavelengths above/below peak)

3. **XYZ to linear sRGB** (D65 illuminant, standard 3x3 matrix):
   ```
   R =  3.2406*X - 1.5372*Y - 0.4986*Z
   G = -0.9689*X + 1.8758*Y + 0.0415*Z
   B =  0.0557*X - 0.2040*Y + 1.0570*Z
   ```

4. **Clamp**: `max(rgb, 0)` — negative values from out-of-gamut spectral colors
   are zeroed

This produces physically plausible rainbow-like color progressions rather than
the arbitrary palettes used in normal mode.

## Uniform Reference

Quick reference for all uniforms consumed by the compute shader:

| Uniform | Field | Config Key | Purpose |
|---------|-------|------------|---------|
| globals.x | speed | `speed` | Time multiplier |
| globals.y | zoom | `zoom` | Camera zoom |
| kifs.w | drift_speed | `drift_speed` | Master animation gate |
| extra.x | color_shift | `color_shift` | Global palette rotation |
| extra.w | symmetry | `symmetry` | Rotational/bilateral copies |
| extra2.x | noise_displacement | `noise_displacement` | Noise variation amplitude |
| extra2.y | curl_displacement | `curl_displacement` | Curl variation amplitude |
| extra2.z | tangent_clamp | `tangent_clamp` | Tangent variation clamp |
| extra2.w | color_blend | `color_blend` | Per-iteration color blend rate |
| extra3.x | spin_speed_max | `spin_speed_max` | Max per-transform rotation speed |
| extra3.y | position_drift | `position_drift` | Per-transform translation wobble |
| extra3.z | warmup_iters | `warmup_iters` | Iterations to skip before splatting |
| extra4.x | jitter_amount | `jitter_amount` | Sub-pixel jitter magnitude |
| extra5.y | spectral_rendering | `spectral_rendering` | Toggle spectral color mode |
| extra6.x | dist_lum_strength | `dist_lum_strength` | Distance-based luminosity falloff |
| extra6.y | iter_lum_range | `iter_lum_range` | Iteration-based luminosity range |
| has_final_xform & 1 | — | — | Final transform present flag |
| has_final_xform >> 16 | — | `iterations_per_thread` | Max iterations per thread |

## Helper Functions

| Function | Purpose |
|----------|---------|
| `pcg(state)` | PCG random number generator (advances state, returns u32) |
| `randf(state)` | Random float in [0, 1] via PCG |
| `rot2(angle)` | 2x2 rotation matrix |
| `cosh_f(x)` / `sinh_f(x)` | Hyperbolic trig (WGSL lacks builtins) |
| `hash_u(n)` / `hash_f(n)` | Integer/float hash for per-transform seeding |
| `vnoise(t, seed)` | 1D smooth value noise (smoothstep interpolation) |
| `palette(t)` | Texture-based palette lookup (256x1, wrapping) |
| `palette_spectral(t)` | Spectral palette via CIE XYZ |
| `xf(idx, field)` | Read field from transform buffer (`idx * 42 + field`) |
| `xf_param(idx, offset)` | Read parametric variation param (`idx * 42 + 34 + offset`) |
| `splat_pixel(...)` | Write 7 atomic channels to one histogram pixel |
| `splat_point(...)` | Bilinear 2x2 sub-pixel distribution |
