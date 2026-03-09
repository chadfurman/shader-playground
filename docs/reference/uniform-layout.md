# Uniform Layout

How data flows from Rust config and genome state into the GPU uniform buffer.

## Rust Struct

The `Uniforms` struct is defined in `src/main.rs` as `#[repr(C)]` and written
to the GPU uniform buffer each frame via `bytemuck`.

```rust
struct Uniforms {
    time: f32,
    frame: u32,
    resolution: [f32; 2],
    mouse: [f32; 2],
    transform_count: u32,
    has_final_xform: u32,   // bit 0 = has final transform, bits 16..31 = iterations_per_thread
    globals: [f32; 4],
    kifs: [f32; 4],
    extra: [f32; 4],
    extra2: [f32; 4],
    extra3: [f32; 4],
    extra4: [f32; 4],
    extra5: [f32; 4],
    extra6: [f32; 4],
}
```

### Scalar Fields

| Field | Type | Source |
|---|---|---|
| `time` | f32 | Elapsed time in seconds |
| `frame` | u32 | Frame counter |
| `resolution` | [f32; 2] | Window width, height |
| `mouse` | [f32; 2] | Cursor position (pixels) |
| `transform_count` | u32 | Number of active transforms in the genome |
| `has_final_xform` | u32 | Packed: low bit = final transform present, upper 16 bits = `iterations_per_thread` (clamped 10..2000) |

## Uniform Slots

Values flow through the globals array (`[f32; 20]`), which is populated by
`FlameGenome::flatten_globals()` then modulated by audio/time signals via
`Weights::apply_globals()`. The 20-element array is split across the vec4
uniforms as follows.

### globals (vec4)

| Slot | Name | Source |
|---|---|---|
| globals[0] | speed | `genome.global.speed` + signal modulation |
| globals[1] | zoom | `genome.global.zoom` + signal modulation |
| globals[2] | trail | `RuntimeConfig.trail` |
| globals[3] | flame_brightness | `genome.global.flame_brightness` + signal modulation |

### kifs (vec4)

| Slot | Name | Source |
|---|---|---|
| kifs[0] | fold_angle | `genome.kifs.fold_angle` + signal modulation |
| kifs[1] | scale | `genome.kifs.scale` + signal modulation |
| kifs[2] | brightness | `genome.kifs.brightness` + signal modulation |
| kifs[3] | drift_speed | `RuntimeConfig.drift_speed` + signal modulation |

### extra (vec4)

| Slot | Name | Source |
|---|---|---|
| extra[0] | color_shift | Signal modulation only (base = 0.0) |
| extra[1] | vibrancy | `RuntimeConfig.vibrancy` + signal modulation |
| extra[2] | bloom_intensity | `RuntimeConfig.bloom_intensity` + signal modulation |
| extra[3] | symmetry | `genome.symmetry` (cast to f32, not from globals array) |

### extra2 (vec4)

| Slot | Name | Source |
|---|---|---|
| extra2[0] | noise_displacement | `RuntimeConfig.noise_displacement` + signal modulation |
| extra2[1] | curl_displacement | `RuntimeConfig.curl_displacement` + signal modulation |
| extra2[2] | tangent_clamp | `RuntimeConfig.tangent_clamp` + signal modulation |
| extra2[3] | color_blend | `RuntimeConfig.color_blend` + signal modulation |

### extra3 (vec4)

| Slot | Name | Source |
|---|---|---|
| extra3[0] | spin_speed_max | `RuntimeConfig.spin_speed_max` + signal modulation |
| extra3[1] | position_drift | `RuntimeConfig.position_drift` + signal modulation |
| extra3[2] | warmup_iters | `RuntimeConfig.warmup_iters` + signal modulation |
| extra3[3] | velocity_blur_max | `RuntimeConfig.velocity_blur_max` + signal modulation |

### extra4 (vec4)

Passed directly from `RuntimeConfig` — no signal modulation.

| Slot | Name | Source |
|---|---|---|
| extra4[0] | jitter_amount | `RuntimeConfig.jitter_amount` |
| extra4[1] | tonemap_mode | `RuntimeConfig.tonemap_mode` (cast to f32) |
| extra4[2] | histogram_equalization | `RuntimeConfig.histogram_equalization` |
| extra4[3] | dof_strength | `RuntimeConfig.dof_strength` |

### extra5 (vec4)

| Slot | Name | Source |
|---|---|---|
| extra5[0] | dof_focal_distance | `RuntimeConfig.dof_focal_distance` |
| extra5[1] | spectral_rendering | `RuntimeConfig.spectral_rendering` (bool -> 1.0 or 0.0) |
| extra5[2] | temporal_reprojection | `RuntimeConfig.temporal_reprojection` |
| extra5[3] | prev_zoom | Previous frame's zoom value (for reprojection) |

### extra6 (vec4)

| Slot | Name | Source |
|---|---|---|
| extra6[0] | dist_lum_strength | `RuntimeConfig.dist_lum_strength` |
| extra6[1] | iter_lum_range | `RuntimeConfig.iter_lum_range` |
| extra6[2] | _reserved | 0.0 |
| extra6[3] | _reserved | 0.0 |

## Globals Array Index Map

The internal `globals[0..20]` array uses this layout (mapped by `global_index()`
in `src/weights.rs`). Index 11 is gamma, and indices are non-contiguous because
gamma was added after the original layout.

| Index | Name | Index | Name |
|---|---|---|---|
| 0 | speed | 10 | bloom_intensity |
| 1 | zoom | 11 | gamma |
| 2 | trail | 12 | noise_displacement |
| 3 | flame_brightness | 13 | curl_displacement |
| 4 | kifs_fold | 14 | tangent_clamp |
| 5 | kifs_scale | 15 | color_blend |
| 6 | kifs_brightness | 16 | spin_speed_max |
| 7 | drift_speed | 17 | position_drift |
| 8 | color_shift | 18 | warmup_iters |
| 9 | vibrancy | 19 | velocity_blur_max |

## Data Flow

```
weights.json _config  -->  RuntimeConfig
genome.json           -->  FlameGenome
                              |
                    flatten_globals() produces [f32; 20]
                              |
                    Weights::apply_globals() adds signal modulation
                              |
                    globals[0..20] split into vec4 uniforms
                              |
                    extra4/5/6 read directly from RuntimeConfig
                              |
                    Uniforms struct written to GPU buffer
                              |
                    WGSL shaders read @group(0) @binding(0)
```
