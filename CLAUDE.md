# Shader Playground

## Project Rules

### NO MAGIC NUMBERS
Every tunable numeric value MUST come from `weights.json` via `RuntimeConfig`.
- Rust side: Add field to `RuntimeConfig` in `src/weights.rs` with a `default_*` function
- Shader side: Pass through the globals/extra uniform slots
- The ONLY acceptable numeric literals are:
  - Mathematical constants (PI, TAU, 0.0, 1.0, 2.0 in math formulas)
  - Array indices and struct field offsets
  - Variation function definitions (these ARE the math — the constants are the algorithm)
- If you find yourself writing `0.3` or `4.0` or `100.0` as a threshold/clamp/scale, it MUST go in config

### NO `#[allow(dead_code)]`
Never suppress warnings with `#[allow(dead_code)]`. If code is unused, either hook it up or remove it.

### Architecture
- `weights.json` — single source of truth for ALL tunable parameters, hot-reloaded
- `RuntimeConfig` — Rust-side config params (morph, zoom, mutation, variation_scales, etc.)
- Signal weights — audio/time signal modulation of shader params
- `variation_scales` — CRISPR-style per-variation multipliers

### Key Files
- `src/weights.rs` — RuntimeConfig, Weights, signal mapping
- `src/genome.rs` — FlameGenome, mutation, attractor estimation
- `src/main.rs` — App state, render loop, uniform buffer writing
- `flame_compute.wgsl` — Chaos game compute shader
- `playground.wgsl` — Display/tonemapping fragment shader
- `weights.json` — All config and weights

### Uniform Layout (globals [f32; 20])
- [0] speed [1] zoom [2] trail [3] flame_brightness
- [4] kifs_fold [5] kifs_scale [6] kifs_brightness [7] drift_speed
- [8] color_shift [9] vibrancy [10] bloom_intensity [11] gamma
- [12] noise_disp [13] curl_disp [14] tangent_clamp [15] color_blend
- [16] spin_speed_max [17] position_drift [18] warmup_iters [19] highlight_power
