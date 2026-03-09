# Luminosity

Per-point brightness factors computed in the chaos game compute shader
(`flame_compute.wgsl`) before splatting into the accumulation buffer. These
modulate the base palette color to add depth and visual variation.

## Iteration Depth (`iter_lum`)

Controls brightness based on which iteration of the chaos game produced the
point.

```wgsl
iter_lum = 1.0 - iter_lum_range * (f32(i) / f32(max_iters))
```

- When `iter_lum_range = 0.5`: first iteration = 1.0, last = 0.5
- When `iter_lum_range = 0.0`: uniform brightness across all iterations
- When `iter_lum_range = 1.0`: first iteration = 1.0, last = 0.0

Early iterations tend to be closer to the initial random seed and more
scattered, while later iterations converge onto the attractor. Making early
iterations brighter gives a subtle glow around the attractor's edges.

### Config

| Parameter        | Default | Description                               |
|------------------|---------|-------------------------------------------|
| `iter_lum_range` | 0.5     | Brightness falloff from first to last iter |

Passed via `extra6.y`.

---

## Radial Falloff (`dist_lum`)

Dims points based on their distance from the origin in world space.

```wgsl
dist_lum = 1.0 / (1.0 + length(plot_p) * dist_lum_strength)
```

At distance 0, `dist_lum = 1.0` (full brightness). As distance increases,
brightness falls off hyperbolically. The strength parameter controls how
aggressive the falloff is:

- `dist_lum_strength = 0.0`: disabled (all points equal brightness)
- `dist_lum_strength = 0.3`: gentle falloff, points at distance 3 are ~50%
- `dist_lum_strength = 1.0`: strong falloff, points at distance 1 are 50%

### Known issue (fixed)

This parameter was previously hardcoded at 0.3, which caused a visible
center-hole artifact. The radial brightness bias meant points at a fixed
radius accumulated brightness faster than the trail decay could dissipate
it, creating a persistent bright ring. Setting the default to 0.0
(disabled) eliminated the artifact. The parameter remains available for
intentional artistic use.

### Config

| Parameter          | Default | Description                            |
|--------------------|---------|----------------------------------------|
| `dist_lum_strength`| 0.0     | Radial falloff strength (0 = disabled) |

Passed via `extra6.x`.

---

## Transform Weight

Prevents extreme brightness variation between transforms with very
different selection weights.

```wgsl
clamp(xf_weight * 3.0, 0.3, 1.0)
```

- A transform with weight 0.1 contributes brightness 0.3 (floor)
- A transform with weight 0.33 contributes brightness 1.0 (ceiling)
- A transform with weight 1.0 still contributes brightness 1.0

Without this clamp, low-weight transforms would be nearly invisible (they
get fewer points AND each point is dimmer), while high-weight transforms
would dominate both in density and per-point brightness.

---

## Combined Luminosity

All three factors multiply together before being applied to the base color:

```wgsl
lum = iter_lum * dist_lum * clamp(xf_weight * 3.0, 0.3, 1.0)
base_col = palette(plot_color + color_shift) * lum
```

This modulated color is what gets splatted into the accumulation buffer.
The density tonemapping in the fragment shader then applies log-density
mapping on top, so the final perceived brightness is a combination of:

1. **How many points land on a pixel** (density / log mapping)
2. **How bright each point was** (luminosity factors documented here)
