# Post Effects

Post-processing effects applied in the fragment shader (`playground.wgsl`)
after the accumulation buffer is read but before final output. These operate
in screen space on the per-pixel density and color data.

## Velocity Blur

Directional motion blur driven by the per-pixel velocity field accumulated
during the chaos game.

### Motion vector recovery

Each pixel in the accumulation buffer stores summed velocity components
(vx, vy) alongside density. The average velocity is recovered by dividing
out the density, then scaled to pixel space:

```wgsl
avg_vel = vel_raw / (density * 10.0) * resolution
```

The factor of 10 comes from the fixed-point encoding: velocity is stored as
`vel * 10000` and density as `val * 1000`, so their ratio yields `vel * 10`.

### Blur direction and length

The blur direction is the normalized average velocity vector. The blur
length is the velocity magnitude, clamped to a configurable maximum:

```wgsl
blur_dir = select(vec2(0.0), avg_vel / vel_len, vel_len > 0.5)
blur_len = clamp(vel_len, 0.0, blur_max)
```

Pixels with velocity magnitude below 0.5 are not blurred (avoids noise
from near-stationary points).

### Tap pattern

16 taps total: 8 forward and 8 backward along the motion vector. Tap
spacing is `blur_len / 8.0`, so all 8 taps in each direction span the full
blur length.

Each tap is weighted by distance falloff:

```wgsl
weight = 1.0 / (1.0 + f32(tap) * 0.3)
```

This gives the center pixel the most influence, with distant taps
contributing progressively less. The falloff prevents streaks from having
hard edges.

All accumulation channels (density, R, G, B) are blurred together so color
and density stay in sync.

### Config

| Parameter           | Default | Description                         |
|---------------------|---------|-------------------------------------|
| `velocity_blur_max` | 24.0    | Maximum blur length in pixels       |

Passed via `extra3.w`.

---

## Depth of Field

Simulates camera focus by blurring regions away from a focal plane, using
the per-pixel depth channel from the accumulation buffer.

### Circle of confusion

Average depth is recovered from the accumulated depth channel divided by
density. The circle of confusion (CoC) measures how far each pixel is from
the focal plane:

```wgsl
coc = abs(avg_depth - focal) * dof_strength
blur_radius = clamp(coc * 8.0, 0.0, 16.0)
```

If `dof_focal_distance` is set to 0, it defaults to 1.0.

### Blur kernel

8 taps in a radial pattern at equal angles (TAU/8 = ~0.785 radians apart),
forming a circle at the computed blur radius. Each tap samples the
accumulation buffer and averages the color.

Blur only activates when `blur_radius > 0.5` — sharp in-focus regions skip
the sampling entirely.

The DoF result is blended with the velocity-blurred color using
`dof_blend = clamp(coc * 2.0, 0.0, 1.0)`, so the transition from sharp to
blurred is smooth rather than binary.

### Config

| Parameter            | Default | Description                              |
|----------------------|---------|------------------------------------------|
| `dof_strength`       | 0.0     | CoC multiplier (0 = disabled)            |
| `dof_focal_distance` | 0.0     | Focal plane depth (0 defaults to 1.0)    |

Passed via `extra4.w` and `extra5.x`.

---

## Bloom

A density-aware bloom effect that adds soft glow to sparse regions while
keeping dense structure crisp.

### Sampling pattern

Three radii in a cross pattern (up/down/left/right), sampled from the
previous frame texture:

| Radius (px) | Weight per tap | Combined weight |
|-------------|----------------|-----------------|
| 2           | 0.25           | 0.5             |
| 5           | 0.25           | 0.3             |
| 12          | 0.25           | 0.2             |

Each radius samples 4 taps (cardinal directions), averaged with `* 0.25`,
then scaled by the radius weight (0.5, 0.3, 0.2). Closer samples
contribute more, giving a smooth falloff from the center.

### Sparsity weighting

Bloom intensity is modulated by how sparse the current pixel is relative to
the densest pixel in the frame:

```wgsl
norm_density = clamp(density / max_density, 0.0, 1.0)
sparsity = 1.0 - sqrt(norm_density)
```

The `sqrt` gives a gentler falloff — moderately dense regions still get
some bloom, while the densest regions get almost none. Sparse isolated
points (low density relative to max) get full bloom, creating the
characteristic glow around scattered points.

### Final blend

```wgsl
col += bloom_sum * bloom_intensity * sparsity
```

The bloom is purely additive, but the sparsity weighting prevents it from
washing out detailed structure.

### Config

| Parameter        | Default | Description                    |
|------------------|---------|--------------------------------|
| `bloom_intensity`| varies  | Overall bloom strength (0=off) |

Passed via `extra.z`.
