# Config Fields

Every field in `RuntimeConfig` (`src/weights.rs`), set via the `_config` key in
`weights.json`. Fields use `#[serde(default)]` so any subset can be specified.

## Display

| Field | Type | Default | Description |
|---|---|---|---|
| `vibrancy` | f32 | 0.7 | Color saturation preservation in tonemapping (0 = desaturated, 1 = vivid) |
| `bloom_intensity` | f32 | 0.05 | Strength of bloom glow effect |
| `bloom_radius` | f32 | 3.0 | Pixel radius of bloom convolution |
| `gamma` | f32 | 0.4545 | Display gamma (1/2.2 = standard sRGB) |
| `highlight_power` | f32 | 2.0 | Hot-spot glow intensity |
| `trail` | f32 | 0.15 | Temporal AA trail persistence (accumulation is primary persistence) |
| `accumulation_decay` | f32 | 0.9 | Decay rate for the accumulation buffer between frames |

## Evolution

| Field | Type | Default | Description |
|---|---|---|---|
| `morph_duration` | f32 | 8.0 | Seconds to blend between genomes during transitions |
| `morph_speed` | f32 | 1.0 | Global morph speed multiplier (affects all transforms) |
| `morph_stagger_count` | u32 | 2 | Max transforms to randomize with different morph speeds (0 = all same speed) |
| `morph_stagger_min` | f32 | 0.3 | Slowest random morph rate for staggered transforms |
| `morph_stagger_max` | f32 | 0.6 | Fastest random morph rate for staggered transforms |
| `mutation_cooldown` | f32 | 3.0 | Minimum seconds between mutations |
| `max_mutation_retries` | u32 | 5 | Max attempts to find a valid mutation before giving up |
| `seed_mutation_bias` | f32 | 0.7 | Bias toward seed genome in mutation (vs fully random) |
| `fitness_bias_strength` | f32 | 0.5 | How strongly fitness scores influence mutation selection |
| `magnitude_min` | f32 | 1.0 | Min per-transform weight magnitude for signal modulation |
| `magnitude_max` | f32 | 5.0 | Max per-transform weight magnitude for signal modulation |
| `randomness_range` | f32 | 1.0 | Range of per-transform randomness applied to signal weights |

## Camera

| Field | Type | Default | Description |
|---|---|---|---|
| `zoom_min` | f32 | 1.0 | Minimum allowed zoom level |
| `zoom_max` | f32 | 5.0 | Maximum allowed zoom level |
| `zoom_target` | f32 | 3.0 | Target zoom level for auto-zoom |
| `min_attractor_extent` | f32 | 0.3 | Minimum bounding extent of the attractor before zoom compensation kicks in |

## Taste Engine

| Field | Type | Default | Description |
|---|---|---|---|
| `taste_engine_enabled` | bool | false | Enable learned aesthetic preference scoring |
| `taste_min_votes` | u32 | 10 | Minimum votes required before taste model activates |
| `taste_strength` | f32 | 0.5 | How strongly taste scores influence genome selection |
| `taste_exploration_rate` | f32 | 0.1 | Probability of ignoring taste to explore novel genomes |
| `taste_diversity_penalty` | f32 | 0.3 | Penalty for genomes similar to recently displayed ones |
| `taste_candidates` | u32 | 20 | Number of candidate genomes to evaluate per selection |
| `taste_recent_memory` | usize | 5 | Number of recent genomes to track for diversity penalty |

## Breeding

| Field | Type | Default | Description |
|---|---|---|---|
| `parent_current_bias` | f32 | 0.30 | Probability of selecting the current genome as a parent |
| `parent_voted_bias` | f32 | 0.25 | Probability of selecting a positively voted genome as a parent |
| `parent_saved_bias` | f32 | 0.25 | Probability of selecting a saved genome as a parent |
| `parent_random_bias` | f32 | 0.20 | Probability of selecting a random genome as a parent |
| `vote_blacklist_threshold` | i32 | -2 | Net vote score at or below which a genome is blacklisted |
| `min_breeding_distance` | u32 | 3 | Minimum generational distance between parents to allow crossover |
| `max_lineage_depth` | u32 | 8 | Maximum lineage depth before forcing a random parent |

## Archiving

| Field | Type | Default | Description |
|---|---|---|---|
| `archive_threshold_mb` | u64 | 100 | Genome archive size threshold in MB before pruning |
| `archive_on_startup` | bool | true | Whether to archive/prune genomes when the app starts |

## Luminosity

| Field | Type | Default | Description |
|---|---|---|---|
| `dist_lum_strength` | f32 | 0.0 | Distance-based luminosity modulation strength |
| `iter_lum_range` | f32 | 0.5 | Iteration-based luminosity range (0.0 = uniform, 0.5 = early iters 2x brighter) |

## Shader Parameters

These are passed through the globals array and modulated by audio/time signals.

| Field | Type | Default | Description |
|---|---|---|---|
| `noise_displacement` | f32 | 0.08 | Noise-based displacement strength in the compute shader |
| `curl_displacement` | f32 | 0.05 | Curl noise displacement strength |
| `tangent_clamp` | f32 | 4.0 | Maximum value for tangent variation output |
| `color_blend` | f32 | 0.4 | Color mixing between transforms (lower = more mixing) |
| `spin_speed_max` | f32 | 0.15 | Maximum rotational speed for transforms |
| `position_drift` | f32 | 0.08 | Spatial drift applied to transform offsets |
| `warmup_iters` | f32 | 20.0 | Number of initial chaos-game iterations discarded before plotting |
| `drift_speed` | f32 | 0.5 | Base speed of camera/attractor drift |
| `velocity_blur_max` | f32 | 24.0 | Maximum directional blur length in pixels |

## GPU / Performance

| Field | Type | Default | Description |
|---|---|---|---|
| `workgroups` | u32 | 512 | Number of compute workgroups dispatched per frame |
| `iterations_per_thread` | u32 | 200 | Chaos game iterations per GPU thread per frame |
| `samples_per_frame` | u32 | 256 | Number of samples accumulated per frame |

## Other

These fields are passed directly to the GPU via extra4/extra5 uniforms without
signal modulation.

| Field | Type | Default | Description |
|---|---|---|---|
| `jitter_amount` | f32 | 0.0 | Sub-pixel jitter for supersampling |
| `tonemap_mode` | u32 | 0 | Tonemapping algorithm selector |
| `histogram_equalization` | f32 | 0.0 | Strength of adaptive histogram equalization |
| `dof_strength` | f32 | 0.0 | Depth of field blur strength |
| `dof_focal_distance` | f32 | 0.0 | Focal distance for depth of field |
| `spectral_rendering` | bool | false | Enable spectral color rendering path |
| `temporal_reprojection` | f32 | 0.0 | Temporal reprojection blend factor |
| `variation_scales` | HashMap\<String, f32\> | {} | Per-variation CRISPR-style multipliers (e.g. `{"spherical": 0.0}` disables spherical) |
