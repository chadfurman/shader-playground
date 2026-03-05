# Variable Transforms & Time Signals Design

## Goal

Expand the fractal flame system from a fixed 4 transforms to a dynamic N transforms (sweet spot 6-16), with a storage buffer for GPU data, mutation-driven birth/death of transforms, and 5 time-based weight signals replacing the single `time` channel.

## Architecture

Separate globals from transforms into two GPU buffers. The uniform buffer stays small and fast (globals, KIFS, transform count). A new read-only storage buffer holds transform data at dynamic size (N * 12 floats). The compute shader loops over N transforms instead of hardcoding 4. The weights system expands to 11 signals (6 audio + 5 time) with `xfN_` wildcards that expand to all active transforms.

## Data Layout

### Uniform Buffer (fixed size, ~128 bytes)

```
time: f32
frame: u32
resolution: [f32; 2]
mouse: [f32; 2]
transform_count: u32
_pad: u32
globals: [f32; 4]    // speed, zoom, trail, flame_brightness
kifs: [f32; 4]        // kifs_fold, kifs_scale, kifs_brightness, drift_speed
extra: [f32; 4]       // color_shift, ...reserved
```

### Storage Buffer (dynamic size)

Flat `array<f32>` with N * 12 entries:

```
transform 0: weight, angle, scale, offset_x, offset_y, color, linear, sinusoidal, spherical, swirl, horseshoe, handkerchief
transform 1: ...
...
transform N-1: ...
```

Resized whenever the genome's transform count changes.

## Compute Shader

Replace hardcoded 4-way weight selection with a loop:

```wgsl
@group(0) @binding(2) var<storage, read> transforms: array<f32>;

let num_xf = uniforms.transform_count;

// Compute total weight
var total_weight = 0.0;
for (var t = 0u; t < num_xf; t++) {
    total_weight += transforms[t * 12u];
}

// Weighted random selection
let r = randf(&rng) * total_weight;
var tidx = 0u;
var cumsum = 0.0;
for (var t = 0u; t < num_xf; t++) {
    cumsum += transforms[t * 12u];
    if (r < cumsum) {
        tidx = t;
        break;
    }
}
```

`apply_xform()` reads from `transforms[idx * 12 + offset]` instead of `param()`. The fragment shader is unchanged (reads only globals/KIFS from the uniform).

## Genome & Mutation

### Structure

`FlameGenome.transforms` is already a `Vec<FlameTransform>` (unbounded). Changes:

- `flatten()` returns `(Vec<f32>, Vec<f32>)` — (globals, transforms) separately
- `default_genome()` starts with 6 transforms
- Genome JSON files naturally support any number of transforms

### Mutation: Add/Remove Bias

Biased toward the 6-16 sweet spot:

| Current Count | Add Chance | Remove Chance |
|---|---|---|
| < 6 | 40% | 5% |
| 6-16 | 15% | 15% |
| > 16 | 5% | 40% |

- **Adding:** Clone a random existing transform with small perturbations (angle, scale, offset jittered). New transform starts with low weight so it fades in.
- **Removing:** Drop the lowest-weight transform.

### Morph Interpolation

When genome changes transform count during evolution, the render loop must handle different-length arrays:

- Pad the shorter array with zero-weight transforms
- New transforms fade in from weight 0
- Dying transforms fade out to weight 0
- Once weight reaches 0, the transform is pruned from the next genome

## Weights System

### 11 Signals

| Signal | Value Range | Source |
|---|---|---|
| `bass` | 0 to 1 | Audio FFT low band |
| `mids` | 0 to 1 | Audio FFT mid band |
| `highs` | 0 to 1 | Audio FFT high band |
| `energy` | 0 to 1 | Audio overall energy |
| `beat` | 0 to 1 | Beat detection pulse |
| `beat_accum` | 0 to 1 | Accumulated beat intensity, 6s decay |
| `time_slow` | -1 to 1 | `sin(t * 0.1)` ~60s cycle |
| `time_med` | -1 to 1 | `sin(t * 0.5)` ~12s cycle |
| `time_fast` | -1 to 1 | `sin(t * 2.0)` ~3s cycle |
| `time_noise` | -1 to 1 | Perlin noise seeded by time |
| `time_envelope` | 0 to 1 | Time since last mutation, capped at 1.0 over ~10s |

### Formula

```
final[i] = genome_base[i] + (sum of weight * signal for all 11 signals) / 11
```

### xfN_ Wildcard

`xfN_weight`, `xfN_angle`, etc. expand to all active transforms in the current genome. Per-transform deterministic randomized scaling (0.5x-1.5x) still applies.

Explicit `xf0_`, `xf1_`, etc. still work for targeting specific transforms.

### weights.json Example

```json
{
  "bass": { "xfN_swirl": 0.5 },
  "energy": { "flame_brightness": 0.9, "xfN_weight": 0.3 },
  "beat_accum": { "mutation_rate": 1.0 },
  "time_slow": { "kifs_fold": 0.3, "kifs_scale": 0.2 },
  "time_med": { "color_shift": 0.2, "xfN_angle": 0.1 },
  "time_fast": { "xfN_scale": 0.05 },
  "time_noise": { "xfN_offset_x": 0.1, "xfN_offset_y": 0.1 },
  "time_envelope": { "drift_speed": 0.5 }
}
```

## What Doesn't Change

- Fragment shader (playground.wgsl) — reads globals/KIFS from uniform, histogram from storage. No transform references.
- Audio processing thread — still writes audio_features.json at ~30Hz.
- Device picker — unchanged.
- File watcher for weights.json — unchanged.

## Rust-Side Buffer Management

- On genome change (evolve, load, auto-evolve): check if transform count changed. If so, recreate the storage buffer at new size and rebuild the compute bind group.
- On every frame: write globals to uniform buffer, write transform params to storage buffer.
- The `params` / `target_params` arrays in App become `Vec<f32>` for transforms (variable length) plus a fixed globals array.
