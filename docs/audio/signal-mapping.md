# Signal Mapping

How audio and time signals modulate shader parameters. Defined in `src/weights.rs`, configured in `weights.json`.

## TimeSignals

Nine time-based oscillators computed each frame via 1D value noise (smoothstep-interpolated hash):

| Signal | Speed | Period | Character |
|--------|-------|--------|-----------|
| `time` | 1.0 | linear | Raw elapsed seconds, unbounded |
| `time_slow` | 0.05 | ~20s | Glacial drift |
| `time_med` | 0.2 | ~5s | Medium wander |
| `time_fast` | 0.8 | ~1.25s | Quick oscillation |
| `time_noise` | 0.3 | ~3.3s | Organic wandering |
| `time_drift` | 0.02 | ~50s | Ultra-slow drift |
| `time_flutter` | 1.5 | ~0.67s | Quick flicker |
| `time_walk` | accumulated | persistent | Random walk, never reverses (externally accumulated) |
| `time_envelope` | triggered | 10s cap | `time_since_mutation / 10.0`, clamped to 1.0 |

All noise-based signals use offset seeds (e.g., `time * speed + offset`) to decorrelate them. Output range for noise signals is approximately -1.0 to 1.0.

## Signal Weight System

`weights.json` contains sections for each signal source. There are **6 audio signals** and **9 time signals**, for 15 total:

**Audio signals:** `bass`, `mids`, `highs`, `energy`, `beat`, `beat_accum`
**Time signals:** `time`, `time_slow`, `time_med`, `time_fast`, `time_noise`, `time_drift`, `time_flutter`, `time_walk`, `time_envelope`

Each section is a map of parameter names to modulation weights:

```json
{
  "bass": {
    "zoom": 0.3,
    "flame_brightness": 0.2
  },
  "time_slow": {
    "color_shift": 0.5
  }
}
```

### Modulation Formula

For global parameters:

```
param[i] = base_value + sum(weight * signal_value / divisor)
```

Where `divisor` is 6.0 for audio signals and 9.0 for time signals. This normalization prevents the total modulation from overwhelming base values when many signals are active.

### Transform Parameters

Weights can target per-transform fields using two naming conventions:

- **Wildcard** `xfN_<field>` -- applies to ALL transforms with per-transform randomness and magnitude scaling. Each transform gets a deterministic random factor seeded by its index, scaled by `randomness_range` and clamped between `magnitude_min` and `magnitude_max` from RuntimeConfig.
- **Explicit** `xf0_<field>`, `xf1_<field>`, etc. -- targets a specific transform with no randomness applied. The weight is used directly.

The 42 fields per transform block include: weight, a, b, c, d, offset_x, offset_y, color, and 34 variation weights (linear, sinusoidal, spherical, swirl, etc.).

## The globals[] Array

A 20-slot `[f32; 20]` array that carries all global shader parameters to the GPU each frame:

| Index | Parameter | Description |
|-------|-----------|-------------|
| 0 | speed | Iteration speed |
| 1 | zoom | Camera zoom level |
| 2 | trail | Temporal persistence |
| 3 | flame_brightness | Overall brightness |
| 4 | kifs_fold | KIFS fold parameter |
| 5 | kifs_scale | KIFS scale |
| 6 | kifs_brightness | KIFS brightness |
| 7 | drift_speed | Camera drift rate |
| 8 | color_shift | Hue rotation |
| 9 | vibrancy | Color saturation control |
| 10 | bloom_intensity | Glow strength |
| 11 | gamma | Display gamma (default 1/2.2) |
| 12 | noise_displacement | Noise-based distortion |
| 13 | curl_displacement | Curl noise distortion |
| 14 | tangent_clamp | Tangent variation clamp |
| 15 | color_blend | Color mixing between transforms |
| 16 | spin_speed_max | Max rotation speed |
| 17 | position_drift | Transform position wander |
| 18 | warmup_iters | Chaos game warmup iterations |
| 19 | velocity_blur_max | Max directional blur (pixels) |

### Per-Frame Flow

1. Base values are loaded from `params.json` or defaults
2. Signal modulation is applied additively via `Weights::apply_globals()`
3. The modulated array is packed into GPU uniform buffers as four `vec4`s plus extras
4. Additional parameters (jitter, tonemap, DOF, etc.) are passed through separate `extra4`/`extra5`/`extra6` uniform vecs

### Extra Uniform Vecs

Parameters that don't fit in the main 20 globals are passed directly from RuntimeConfig:

- **extra4**: `jitter_amount`, `tonemap_mode`, `histogram_equalization`, `dof_strength`
- **extra5**: `dof_focal_distance`, `spectral_rendering`, `temporal_reprojection`, `prev_zoom`
- **extra6**: `dist_lum_strength`, `iter_lum_range`, _reserved, _reserved
