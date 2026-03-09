# Config

How `weights.json` works.

## File Structure

`weights.json` is the single source of truth for all tunable parameters. It has these top-level sections:

- **`_doc`** -- Human-readable documentation. Contains:
  - `_signals` -- List of all available signal names
  - `_formula` -- How signal modulation math works
  - `_wildcards` -- How `xfN_` and `xf0_` style keys expand
  - `_params` -- Description of every modulatable parameter
  - `_config_doc` -- Description of every RuntimeConfig field
- **`_config`** -- The actual `RuntimeConfig` values (morph timing, GPU settings, zoom, rendering options, breeding/taste parameters, etc.)
- **Signal sections** -- One section per signal, mapping parameter names to modulation weights:
  - Audio: `bass`, `mids`, `highs`, `energy`, `beat`, `beat_accum`
  - Time: `time`, `time_slow`, `time_med`, `time_fast`, `time_noise`, `time_drift`, `time_flutter`, `time_walk`, `time_envelope`

The `_doc` section is ignored by the parser -- it exists purely for humans reading the file.

## Hot-Reloading

- A file watcher monitors `weights.json` for changes
- On change: re-parses the file, updates `RuntimeConfig`, applies immediately
- No restart needed -- edit the file while the app is running
- Invalid JSON is rejected gracefully (the previous config stays active)

## No Magic Numbers Rule

Every tunable numeric value must live in `weights.json` via `RuntimeConfig`. No exceptions.

**Acceptable literals:**
- Mathematical constants (PI, TAU, 0.0, 1.0, 2.0 in math formulas)
- Array indices and struct field offsets
- Variation function definitions (the constants ARE the algorithm)

**Everything else goes in config.** If you find yourself writing `0.3` or `4.0` or `100.0` as a threshold, clamp, or scale factor, it must be a `RuntimeConfig` field.

## Adding a New Config Field

1. **Add field to `RuntimeConfig`** in `src/weights.rs` with `#[serde(default = "default_fieldname")]`
2. **Add default function** -- `fn default_fieldname() -> T { value }` with the default value
3. **Add to `weights.json`** `_config` section with the value
4. **Add to `_config_doc`** section with a description string
5. **If GPU-side:** wire through uniform buffer in `src/main.rs`:
   - Global params use `globals[]` (20 slots, see layout below)
   - Additional params use `extra4[]`, `extra5[]`, `extra6[]` vec4 slots
6. **If GPU-side:** add to the shader `Uniforms` struct in both `flame_compute.wgsl` and `playground.wgsl`

### Uniform Layout Reference

`globals[f32; 20]` slot assignments:

| Index | Parameter |
|-------|-----------|
| 0 | speed |
| 1 | zoom |
| 2 | trail |
| 3 | flame_brightness |
| 4 | kifs_fold |
| 5 | kifs_scale |
| 6 | kifs_brightness |
| 7 | drift_speed |
| 8 | color_shift |
| 9 | vibrancy |
| 10 | bloom_intensity |
| 11 | gamma |
| 12 | noise_displacement |
| 13 | curl_displacement |
| 14 | tangent_clamp |
| 15 | color_blend |
| 16 | spin_speed_max |
| 17 | position_drift |
| 18 | warmup_iters |
| 19 | velocity_blur_max |

Extra uniform vecs (passed directly from RuntimeConfig):

- `extra4`: jitter_amount, tonemap_mode, histogram_equalization, dof_strength
- `extra5`: dof_focal_distance, spectral_rendering, temporal_reprojection, prev_zoom
- `extra6`: dist_lum_strength, iter_lum_range, _reserved, _reserved

## variation_scales

CRISPR-style per-variation multipliers inside `_config`:

```json
"variation_scales": {
  "spherical": 0.5,
  "julia": 2.0,
  "noise": 0.0
}
```

- Halves spherical weight, doubles julia, disables noise entirely
- Applied during mutation/normalization via `apply_variation_scales()`
- Multiplies each variation weight in the flattened transform buffer by the corresponding scale
- Variations not listed default to 1.0 (unchanged)
- Covers all 34 variation fields (indices 8-41 in each 42-float transform block)
