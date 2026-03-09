# Signal Weights

The signal modulation system in detail.

## How It Works

Each parameter in the `globals[]` array and each per-transform field can be modulated by audio and time signals. The formula:

```
final[i] = base_config_value + sum(weight * signal_value / divisor)
```

- Audio signals are divided by 6 (there are 6 audio signals)
- Time signals are divided by 9 (there are 9 time signals)
- This normalization prevents the sum from overwhelming the base value

## weights.json Signal Sections

Each signal has its own top-level section mapping parameter names to modulation strengths:

```json
{
  "bass": {
    "flame_brightness": 0.02,
    "color_shift": -0.01,
    "xfN_spherical": 0.01
  },
  "energy": {
    "flame_brightness": 0.03,
    "mutation_rate": 0.08
  },
  "time_slow": {
    "color_shift": 0.15
  }
}
```

Positive weights make the parameter increase when the signal is active. Negative weights make it decrease. The weight magnitude controls how strong the effect is.

## Available Signals

### Audio signals (from AudioFeatures, range [0, 1])

| Signal | Description |
|--------|-------------|
| `bass` | Low frequency energy |
| `mids` | Mid frequency energy |
| `highs` | High frequency energy |
| `energy` | Overall audio energy |
| `beat` | Beat detection impulse |
| `beat_accum` | Accumulated beat energy (ramps up over time) |

### Time signals

| Signal | Range | Description |
|--------|-------|-------------|
| `time` | [0, inf) | Raw elapsed time, steady linear ramp |
| `time_slow` | [-1, 1] | Noise at 0.05 speed, ~20s wander |
| `time_med` | [-1, 1] | Noise at 0.2 speed, ~5s wander |
| `time_fast` | [-1, 1] | Noise at 0.8 speed, ~1.25s wander |
| `time_noise` | [-1, 1] | Noise at 0.3 speed, organic wandering |
| `time_drift` | [-1, 1] | Noise at 0.02 speed, ~50s glacial drift |
| `time_flutter` | [-1, 1] | Noise at 1.5 speed, quick flicker |
| `time_walk` | unbounded | Random walk -- accumulated noise, never reverses |
| `time_envelope` | [0, 1] | Time since last mutation, capped at 1.0 (ramps over 10s) |

Time noise signals use 1D smooth value noise with smoothstep interpolation.

## Modulatable Global Parameters

The `globals[f32; 20]` array. Use these names as keys in signal sections:

| Slot | Name | Description |
|------|------|-------------|
| 0 | `speed` | Global animation speed |
| 1 | `zoom` | Camera zoom level |
| 2 | `trail` | Feedback trail decay |
| 3 | `flame_brightness` | Overall flame brightness |
| 4 | `kifs_fold` | KIFS folding angle |
| 5 | `kifs_scale` | KIFS scale factor |
| 6 | `kifs_brightness` | KIFS geometry brightness |
| 7 | `drift_speed` | Orbital drift speed multiplier |
| 8 | `color_shift` | Hue shift applied to palette |
| 9 | `vibrancy` | Color saturation based on density |
| 10 | `bloom_intensity` | Bloom glow intensity |
| 11 | `gamma` | Display gamma correction |
| 12 | `noise_displacement` | Noise variation scatter strength |
| 13 | `curl_displacement` | Curl variation displacement |
| 14 | `tangent_clamp` | Tangent variation clamp range |
| 15 | `color_blend` | Per-iteration color blend toward xform color |
| 16 | `spin_speed_max` | Max per-transform spin speed |
| 17 | `position_drift` | Per-transform position drift amplitude |
| 18 | `warmup_iters` | Iterations to skip before plotting |
| 19 | `velocity_blur_max` | Max directional blur in pixels |

### Special parameter: mutation_rate

`mutation_rate` is a virtual parameter -- it doesn't map to a globals[] slot. Instead it accumulates over time as an auto-evolve trigger: when it reaches 1.0, a mutation fires. Used in `energy`, `beat`, and `beat_accum` sections to make audio drive evolution speed.

## Per-Transform Modulation

Each transform is a 42-float block. There are two ways to modulate transform fields:

### Wildcard: `xfN_fieldname`

Applies to ALL transforms with per-transform randomness and magnitude:

```
delta = weight * randomness(xf_index) * magnitude(xf_index) * signal / divisor
```

- `randomness` is deterministic per-transform, range [-randomness_range, +randomness_range]
- `magnitude` is deterministic per-transform, range [magnitude_min, magnitude_max]
- Both are seeded by transform index for consistency across frames

### Explicit: `xf0_fieldname`, `xf1_fieldname`, etc.

Targets a specific transform with no randomness applied:

```
delta = weight * signal / divisor
```

### Transform Field Names (42 fields per transform)

| Index | Field | Index | Field |
|-------|-------|-------|-------|
| 0 | weight | 21 | spiral |
| 1 | a | 22 | diamond |
| 2 | b | 23 | bent |
| 3 | c | 24 | waves |
| 4 | d | 25 | popcorn |
| 5 | offset_x | 26 | fan |
| 6 | offset_y | 27 | eyefish |
| 7 | color | 28 | cross |
| 8 | linear | 29 | tangent |
| 9 | sinusoidal | 30 | cosine |
| 10 | spherical | 31 | blob |
| 11 | swirl | 32 | noise |
| 12 | horseshoe | 33 | curl |
| 13 | handkerchief | 34 | rings2_val |
| 14 | julia | 35 | blob_low |
| 15 | polar | 36 | blob_high |
| 16 | disc | 37 | blob_waves |
| 17 | rings | 38 | julian_power |
| 18 | bubble | 39 | julian_dist |
| 19 | fisheye | 40 | ngon_sides |
| 20 | exponential | 41 | ngon_corners |

Fields 0-7 are affine/control parameters. Fields 8-41 are variation weights and variation parameters.
