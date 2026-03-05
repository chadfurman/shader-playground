# Audio Reactivity System Design

## Problem

The flame genome system produces beautiful evolving visuals, but they're disconnected from music. We want audio to influence the visuals without the rubber-banding issues that plagued silly-visualizer (where audio directly moved geometry).

## Solution

Port silly-visualizer's audio pipeline (cpal + rustfft + auto-gain + beat detection) and introduce a **weight matrix** that maps audio features to genome params. The genome is always the "home" state — audio pushes params away from home, and when audio goes quiet, params drift back naturally via the existing exponential lerp.

## Architecture

```
System Audio → cpal → FFT → AudioFeatures {bass, mids, highs, energy, beat, beat_accum}
                                   ↓
Genome.flatten() → base_params + (weight_matrix × audio_features) → target_params
                                   ↓
                          exponential lerp → params → GPU
```

Audio runs on its own thread, pushes AudioFeatures through a channel. Main loop reads features each frame, applies the weight matrix, clamps results, and sets target_params.

## Audio Features (ported from silly-visualizer)

| Feature | Description | Range |
|---------|-------------|-------|
| bass | Average of FFT bands 0-2 | 0.0-1.0 |
| mids | Average of FFT bands 4-7 | 0.0-1.0 |
| highs | Average of FFT bands 10-13 | 0.0-1.0 |
| energy | Average of all 16 bands | 0.0-1.0 |
| beat | Spike detection, decays at 0.15/frame | 0.0-1.0 |
| beat_accum | 6-second decay, +0.05 per beat, capped at 1.0 | 0.0-1.0 |

Processing: 2048-point FFT, Hann window, 16 log-spaced bands, auto-gain (target 0.05), noise gate, asymmetric smoothing (fast attack, slow release).

## Weight Matrix

`audio_weights.json` — hot-reloadable:

```json
{
  "targets": {
    "kifs_fold":    { "bass": 0.3, "mids": 0.0, "highs": 0.0, "energy": 0.0, "beat": 0.0, "beat_accum": 0.0 },
    "kifs_scale":   { "bass": 0.0, "mids": 0.0, "highs": 0.0, "energy": 0.1, "beat": 0.0, "beat_accum": 0.0 },
    "kifs_bright":  { "bass": 0.0, "mids": 0.0, "highs": 0.0, "energy": 0.2, "beat": 0.0, "beat_accum": 0.0 },
    "flame_bright": { "bass": 0.0, "mids": 0.0, "highs": 0.0, "energy": 0.15, "beat": 0.0, "beat_accum": 0.0 },
    "trail":        { "bass": 0.0, "mids": 0.0, "highs": 0.0, "energy": 0.0, "beat": -0.1, "beat_accum": 0.0 },
    "drift_speed":  { "bass": 0.0, "mids": 0.0, "highs": 0.0, "energy": 0.3, "beat": 0.0, "beat_accum": 0.0 },
    "color_shift":  { "bass": -0.1, "mids": 0.0, "highs": 0.05, "energy": 0.0, "beat": 0.0, "beat_accum": 0.0 },
    "morph_rate":   { "bass": 0.0, "mids": 0.0, "highs": 0.0, "energy": 0.0, "beat": 2.0, "beat_accum": 0.0 },
    "xf0_weight":   { "bass": 0.0, "mids": 0.0, "highs": 0.0, "energy": 0.05, "beat": 0.0, "beat_accum": 0.0 },
    "xf1_weight":   { "bass": 0.0, "mids": 0.05, "highs": 0.0, "energy": 0.0, "beat": 0.0, "beat_accum": 0.0 },
    "xf2_weight":   { "bass": 0.0, "mids": 0.0, "highs": 0.05, "energy": 0.0, "beat": 0.0, "beat_accum": 0.0 },
    "xf3_weight":   { "bass": 0.05, "mids": 0.0, "highs": 0.0, "energy": 0.0, "beat": 0.0, "beat_accum": 0.0 },
    "mutation_trigger": { "bass": 0.0, "mids": 0.0, "highs": 0.0, "energy": 0.0, "beat": 0.0, "beat_accum": 1.0 }
  },
  "caps": {
    "kifs_fold": [0.3, 0.9],
    "kifs_scale": [1.3, 2.5],
    "trail": [0.1, 0.6],
    "flame_bright": [0.1, 0.8],
    "morph_rate": [0.5, 20.0]
  }
}
```

### Target Descriptions

| Target | Param Index | What It Does |
|--------|-------------|-------------|
| kifs_fold | 4 | KIFS fold angle — bass spike = quick fractal warp |
| kifs_scale | 5 | KIFS iteration scale |
| kifs_bright | 6 | KIFS background brightness |
| flame_bright | 3 | Flame foreground brightness |
| trail | 2 | Feedback persistence |
| drift_speed | new (7) | Controls time multiplier for transform angle drift |
| color_shift | applied to palette sampling | Shifts palette offset |
| morph_rate | applied to lerp rate | Audio can speed/slow morphing |
| xf0-3_weight | 8, 20, 32, 44 | Per-transform contribution |
| mutation_trigger | accumulator | Crosses threshold → auto-mutate |

### How It Works

Each frame:
1. Read AudioFeatures from channel
2. For each target, compute: `offset = sum(feature[i] * weight[i])`
3. `effective_param = genome_base + offset`
4. Clamp to caps if defined
5. Set as target_params (lerp handles smoothing)

Special targets:
- `mutation_trigger`: accumulates each frame, when it crosses 1.0, triggers mutate() and resets to 0
- `morph_rate`: adjusts the lerp rate directly (doesn't go through params array)
- `color_shift`: added to the palette offset in the fragment shader (needs a new param slot)
- `drift_speed`: stored in param[7] (was reserved), read by compute shader

## New Param: drift_speed

Use the reserved slot param[7] for drift_speed. Default 1.0. The compute shader multiplies its time-based angle drift by this value:
```wgsl
let drift = param(7); // drift_speed
let q = rot2(angle + t * 0.07 * drift * f32(idx + 1)) * p * scale ...
```

## New Param: color_shift

Add to the unused param space (param[56] onwards). The fragment shader adds this to palette sampling:
```wgsl
let color_shift = param(56);
let flame = palette(avg_color + u.time * 0.02 + color_shift) * brightness * flame_bright;
```

## What To Port From silly-visualizer

Files to adapt:
- `src/audio_processing.rs` — FFT, band extraction, auto-gain, smoothing, beat detection
- `src/audio_capture.rs` — cpal setup, SCK fallback, sample ring buffer
- `AudioFeatures` struct (simplified from AudioUniforms — no GPU binding needed, just data)

Dependencies to add:
- `cpal = "0.17"`
- `rustfft = "6.4"`

## File Structure

```
src/audio.rs          — AudioFeatures struct, capture thread, FFT processing
src/audio_weights.rs  — WeightMatrix struct, load from JSON, apply to params
audio_weights.json    — hot-reloadable weight config
```

## What Changes

- main.rs: spawn audio thread, read features, apply weights before setting target_params
- flame_compute.wgsl: read drift_speed from param(7)
- playground.wgsl: read color_shift from param(56)
- Cargo.toml: add cpal, rustfft

## What Stays

- Genome system (mutation, save/load, keyboard controls)
- Compute pipeline, fragment shader structure
- Hot-reload, feedback, KIFS background
- Exponential lerp (audio just modifies target_params)

## Key Design Principle

**Genome is home. Audio is displacement.** When audio stops, everything drifts back to the genome's base state. This prevents the "washed out" problem from silly-visualizer where sustained audio permanently shifted the visual baseline.
