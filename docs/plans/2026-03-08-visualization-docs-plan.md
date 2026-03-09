# Visualization Documentation System — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a Docsify-powered documentation site covering shader-playground's rendering pipeline, genetics, taste engine, audio, and config systems as living docs.

**Architecture:** Docsify v4 static site in `docs/`. Each subsystem gets a category directory with a README index and topic-specific markdown files. Content drawn from actual source code — no speculation. CLAUDE.md updated to require doc maintenance alongside code changes.

**Tech Stack:** Docsify v4, Markdown, ASCII diagrams

---

## Task 1: Docsify scaffolding

**Files:**
- Create: `docs/index.html`
- Create: `docs/README.md`
- Create: `docs/_sidebar.md`
- Create: `docs/package.json`

**Step 1: Create `docs/index.html`**

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Shader Playground</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/docsify-themeable@0/dist/css/theme-simple.min.css">
  <style>
    :root {
      --theme-color: #e66533;
    }
  </style>
</head>
<body>
  <div id="app"></div>
  <script>
    window.$docsify = {
      name: 'Shader Playground',
      repo: '',
      loadSidebar: true,
      subMaxLevel: 2,
      auto2top: true,
      relativePath: true,
      search: {
        maxAge: 86400000,
        paths: 'auto',
        depth: 3,
        placeholder: 'Search docs...',
      },
    };
  </script>
  <script src="https://cdn.jsdelivr.net/npm/docsify@4/lib/docsify.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/docsify@4/lib/plugins/search.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/docsify-pagination/dist/docsify-pagination.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/docsify-copy-code/dist/docsify-copy-code.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/prismjs@1/components/prism-rust.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/prismjs@1/components/prism-glsl.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/prismjs@1/components/prism-json.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/prismjs@1/components/prism-bash.min.js"></script>
</body>
</html>
```

**Step 2: Create `docs/README.md`**

Landing page with project overview, quick links to each category, and "What is this?" introduction covering: fractal flame renderer, genetic evolution, taste learning, audio reactivity. Keep to ~50 lines.

**Step 3: Create `docs/_sidebar.md`**

Full navigation tree matching the design doc structure:

```markdown
- [Home](/)
- **Reference**
  - [Overview](reference/README.md)
  - [Vocabulary](reference/vocabulary.md)
  - [Uniform Layout](reference/uniform-layout.md)
  - [Config Fields](reference/weights-config.md)
- **Rendering**
  - [Overview](rendering/README.md)
  - [Chaos Game](rendering/chaos-game.md)
  - [Tonemapping](rendering/tonemapping.md)
  - [Feedback & Trail](rendering/feedback-trail.md)
  - [Post Effects](rendering/post-effects.md)
  - [Luminosity](rendering/luminosity.md)
- **Genetics**
  - [Overview](genetics/README.md)
  - [Genome Format](genetics/genome-format.md)
  - [Breeding](genetics/breeding.md)
  - [Mutation](genetics/mutation.md)
  - [Persistence](genetics/persistence.md)
- **Taste Engine**
  - [Overview](taste-engine/README.md)
  - [Palette Model](taste-engine/palette-model.md)
  - [Transform Model](taste-engine/transform-model.md)
- **Audio**
  - [Overview](audio/README.md)
  - [Signal Mapping](audio/signal-mapping.md)
- **Config**
  - [Overview](config/README.md)
  - [Signal Weights](config/signal-weights.md)
```

**Step 4: Create `docs/package.json`**

```json
{
  "name": "shader-playground-docs",
  "scripts": {
    "serve": "npx docsify-cli serve ."
  }
}
```

**Step 5: Create directory structure**

```bash
mkdir -p docs/reference docs/rendering docs/genetics docs/taste-engine docs/audio docs/config
```

**Step 6: Verify Docsify serves**

Run: `cd docs && npx docsify-cli serve . &`
Open: `http://localhost:3000`
Expected: Landing page renders with sidebar navigation.

**Step 7: Commit**

```bash
git add docs/index.html docs/README.md docs/_sidebar.md docs/package.json
git commit -m "docs: scaffold Docsify site with navigation structure"
```

---

## Task 2: Reference — Vocabulary

**Files:**
- Create: `docs/reference/README.md`
- Create: `docs/reference/vocabulary.md`

**Step 1: Write `docs/reference/README.md`**

Quick Reference index with "When to Read" table:

| Working on... | Read |
|---|---|
| Unfamiliar term | [Vocabulary](vocabulary.md) |
| Uniform buffer layout | [Uniform Layout](uniform-layout.md) |
| Config field meanings | [Config Fields](weights-config.md) |

**Step 2: Write `docs/reference/vocabulary.md`**

Define all project-specific terms. Source definitions from actual code. Include:

- **Genome** — `FlameGenome` struct. Complete genetic specification of a fractal flame.
- **Transform** — `FlameTransform`. One affine map + variation weights in the chaos game.
- **Variation** — One of 26 nonlinear functions (linear, sinusoidal, spherical, swirl, horseshoe, handkerchief, julia, polar, disc, rings, bubble, fisheye, exponential, spiral, diamond, bent, waves, popcorn, fan, eyefish, cross, tangent, cosine, blob, noise, curl).
- **Chaos game** — Iterated function system: randomly select a transform, apply it, plot the point.
- **Attractor** — The set of points the chaos game converges to.
- **Palette** — 256-entry `Vec<[f32; 3]>` RGB color table. Each transform indexes into it via `color` field.
- **Breeding** — `breed()` in `genome.rs`. Combines two parent genomes via slot-based crossover.
- **Mutation** — Random perturbation of affine coefficients, variation weights, or parameters.
- **Taste model** — Gaussian centroid model (`TasteModel`) that learns preferred features from upvoted genomes.
- **Morph** — Smooth interpolation between two genomes over `morph_duration` frames.
- **Trail** — Temporal feedback: previous frame blended into current via `max(current, prev * trail)`.
- **Accumulation buffer** — 7-channel per-pixel buffer (density, R, G, B, vel_x, vel_y, depth).
- **Splatting** — Bilinear sub-pixel point deposition into the accumulation buffer.
- **Symmetry** — Rotational (positive) or bilateral (negative) copies of each plotted point.
- **Vibrancy** — Flam3 color blend parameter controlling saturation preservation.
- **Warmup** — Initial iterations skipped before plotting (lets points settle onto attractor).
- **Lineage** — Parent→child ancestry tree tracked in `lineage.json`.

Include an ASCII pipeline overview at the top:

```
┌─────────┐    ┌───────────┐    ┌──────────────┐    ┌──────────┐
│ Genome  │───▶│ Compute   │───▶│ Accumulation │───▶│ Display  │
│ (genes) │    │ (chaos    │    │ Buffer       │    │ (tonemap │
│         │    │  game)    │    │ (7 channels) │    │  + fx)   │
└─────────┘    └───────────┘    └──────────────┘    └──────────┘
     ▲                                                    │
     │              ┌───────────┐                         │
     └──────────────│ Breeding  │◀────── votes ───────────┘
                    │ + Taste   │
                    └───────────┘
```

**Step 3: Commit**

```bash
git add docs/reference/
git commit -m "docs: add reference vocabulary and index"
```

---

## Task 3: Reference — Uniform Layout + Config Fields

**Files:**
- Create: `docs/reference/uniform-layout.md`
- Create: `docs/reference/weights-config.md`

**Step 1: Write `docs/reference/uniform-layout.md`**

Document the complete uniform buffer layout. Source from `src/main.rs` Uniforms struct and CLAUDE.md. Include:

- Rust `Uniforms` struct field-by-field
- globals[0-3], kifs[0-3], extra[0-3], extra2-6 with field names and sources
- Which shader reads each field (compute, display, or both)
- How values flow: `weights.json` → `RuntimeConfig` → `globals[]` / `extraN[]` → GPU uniform

**Step 2: Write `docs/reference/weights-config.md`**

Every RuntimeConfig field documented. Source from `src/weights.rs`. Table format:

| Field | Default | Range | Description |
|---|---|---|---|
| `morph_duration` | 8.0 | 1-60 | Frames to blend between genomes |
| `mutation_cooldown` | 3.0 | 0-30 | Seconds between auto-evolves |
| ... | ... | ... | ... |

Include ALL fields from RuntimeConfig (there are ~50+). Group by category:
- Display (vibrancy, bloom, gamma, trail, etc.)
- Evolution (morph_duration, mutation_cooldown, magnitude_min/max, etc.)
- Camera (zoom_min/max/target, min_attractor_extent)
- Taste (taste_engine_enabled, taste_min_votes, taste_strength, etc.)
- Breeding (parent_*_bias, min_breeding_distance, max_lineage_depth)
- Archiving (archive_threshold_mb, archive_on_startup)
- Luminosity (dist_lum_strength, iter_lum_range)

**Step 3: Commit**

```bash
git add docs/reference/
git commit -m "docs: add uniform layout and config field reference"
```

---

## Task 4: Rendering — Pipeline Overview

**Files:**
- Create: `docs/rendering/README.md`

**Step 1: Write `docs/rendering/README.md`**

Pipeline overview with ASCII flow diagram and "When to Read" table. Cover the full frame lifecycle:

1. **Compute pass** (`flame_compute.wgsl`) — chaos game iteration, point splatting
2. **Histogram pass** — reduce accumulation buffer to density histogram
3. **CDF pass** — compute cumulative distribution for histogram equalization
4. **Fragment pass** (`playground.wgsl`) — tonemapping, effects, feedback, output

Include timing: what happens per-frame vs. what persists across frames (point state, accumulation via trail).

| Working on... | Read |
|---|---|
| Chaos game, transforms, variations | [Chaos Game](chaos-game.md) |
| Brightness, log-density, color | [Tonemapping](tonemapping.md) |
| Trail, temporal persistence | [Feedback & Trail](feedback-trail.md) |
| Bloom, DoF, velocity blur | [Post Effects](post-effects.md) |
| Per-point brightness factors | [Luminosity](luminosity.md) |

**Step 2: Commit**

```bash
git add docs/rendering/
git commit -m "docs: add rendering pipeline overview"
```

---

## Task 5: Rendering — Chaos Game (REFERENCE-LEVEL)

**Files:**
- Create: `docs/rendering/chaos-game.md`

**Step 1: Write `docs/rendering/chaos-game.md`** (300-500 lines)

Deep dive into `flame_compute.wgsl`. Source from actual shader code. Cover:

**Point Initialization**
- Persistent point state: 3 f32s per thread (x, y, color_idx) in `point_state` buffer
- Re-randomization conditions: first frame (all zeros), escaped (|x|>10 or |y|>10), NaN, 5% refresh rate
- Initial position range: [-2, 2] for both axes

**Iteration Loop**
- Weighted random transform selection (cumulative sum method)
- Affine application: `p' = [a b; c d] * p + [e, f]`
- Variation application: 26 functions applied with per-variation weights, accumulated
- Color blending: `color = color * (1-blend) + xform_color * blend + position_offset`
- Warmup skip: `if (i < warmup_iters) { continue; }`

**All 26 Variation Functions**
Table listing each variation with its formula, computational cost, and visual character:

| Index | Name | Formula | Character |
|---|---|---|---|
| 0 | Linear | `p` | Identity pass-through |
| 1 | Sinusoidal | `sin(p)` | Smooth wave distortion |
| 2 | Spherical | `p / |p|²` | Inversion, creates holes |
| 3 | Swirl | `rotate(p, |p|²)` | Spiral vortex |
| ... | ... | ... | ... |

**Splatting**
- Bilinear sub-pixel: distributes point across 2×2 pixel quad
- 7 channels: density (×1000), R, G, B, vel_x (×10000), vel_y (×10000), depth (×1000)
- Optimization: skips neighbor pixels when sub-pixel offset < 10%
- Uses `atomicAdd` on `u32` storage buffer (fixed-point encoding)

**Symmetry**
- Rotational: N copies at `2π/N` intervals
- Bilateral: mirrors across Y axis (negative symmetry values)
- Both applied before splatting

**Spectral Rendering**
- CIE XYZ color matching (Wyman et al. 2013 Gaussian approximation)
- Palette index → wavelength (380-780nm) → XYZ tristimulus → sRGB
- Toggled via `spectral_rendering` config flag

**Config Callouts**
- `warmup_iters` (default 20): iterations before plotting
- `color_blend` (default 0.4): transform color mixing rate
- `noise_displacement` (default 0.08): per-iteration noise perturbation
- `curl_displacement` (default 0.05): per-iteration curl perturbation
- `spin_speed_max` (default 0.15): per-transform rotation rate
- `position_drift` (default 0.08): per-transform position wandering
- `jitter_amount` (default 0.0): sub-pixel jitter for supersampling
- `iterations_per_thread` (default 200): chaos game iterations per GPU thread

**Step 2: Commit**

```bash
git add docs/rendering/chaos-game.md
git commit -m "docs: add chaos game reference documentation"
```

---

## Task 6: Rendering — Tonemapping (REFERENCE-LEVEL)

**Files:**
- Create: `docs/rendering/tonemapping.md`

**Step 1: Write `docs/rendering/tonemapping.md`** (300-400 lines)

Deep dive into density→color conversion in `playground.wgsl`. Cover:

**Accumulation Buffer Reading**
- 7-channel layout: `[density, R, G, B, vel_x, vel_y, depth]`
- Fixed-point decoding: divide by 1000 (density, depth) or 10000 (velocity)
- Per-pixel color recovery: `rgb = [R, G, B] / density`

**Log-Density Mapping**
- Core formula: `log_density = log(1 + hits * flame_brightness)`
- Normalization: `alpha = log_density / log(1 + max_density * flame_brightness)`
- Max density found by histogram reduction pass

**Tonemapping Modes** (`tonemap_mode` config)
- Mode 0 (default): `sqrt(alpha)` — Flam3-style square root compression
- Mode 1 (ACES): `x * (2.51x + 0.03) / (x * (2.43x + 0.59) + 0.14)` — filmic shoulder

**Histogram Equalization**
- CDF computation from density histogram
- Blend: `alpha = mix(alpha, cdf_lookup(alpha), histogram_equalization)`
- Controlled by `histogram_equalization` config (0.0 = off, 1.0 = full)

**Vibrancy Color Blend**
- Flam3 algorithm: `gamma_alpha = pow(alpha, gamma)`
- `ls = vibrancy * alpha + (1 - vibrancy) * gamma_alpha`
- Higher vibrancy → more saturated sparse regions

**Edge Glow**
- Density gradient: `grad = [density_right - density_left, density_bottom - density_top]`
- Color gradient: `|color_right - color_left|` across RGB
- Combined: `edge = sqrt(density_edge² + color_edge²)`
- Applied as additive glow on the base color

**Config Callouts**
- `flame_brightness` (default from genome global): accumulation scaling
- `gamma` (default 0.4545 = 1/2.2): display gamma correction
- `vibrancy` (default 0.05): color saturation preservation
- `tonemap_mode` (default 0): 0=sqrt, 1=ACES
- `histogram_equalization` (default 0.0): adaptive brightness mapping strength
- `highlight_power` (default 2.0): hot-spot glow intensity

**Step 2: Commit**

```bash
git add docs/rendering/tonemapping.md
git commit -m "docs: add tonemapping reference documentation"
```

---

## Task 7: Rendering — Feedback, Post Effects, Luminosity

**Files:**
- Create: `docs/rendering/feedback-trail.md`
- Create: `docs/rendering/post-effects.md`
- Create: `docs/rendering/luminosity.md`

**Step 1: Write `docs/rendering/feedback-trail.md`** (~120 lines)

Cover:
- Trail blending: `col = max(col, prev_frame * trail)` — uses `max`, not additive
- Why max: prevents brightness explosion, creates natural decay
- Temporal reprojection: when zoom changes, warp previous frame UVs by zoom ratio
- `prev_uv = mix(tex_uv, reprojected, temporal_reprojection)`
- Clamping: `prev_uv = clamp(prev_uv, 0.001, 0.999)` — prevents edge sampling artifacts
- Config: `trail` (default 0.15), `temporal_reprojection` (default 0.0)

**Step 2: Write `docs/rendering/post-effects.md`** (~150 lines)

Cover:
- **Velocity blur**: motion-aligned 16-tap blur (8 forward + 8 reverse), distance falloff, `velocity_blur_max` config
- **Depth of field**: 8-tap radial blur, circle of confusion from `|depth - focal| * dof_strength`, `dof_strength` and `dof_focal_distance` configs
- **Bloom**: 3-radius cross pattern (r=2, r=5, r=12 pixels), sparsity-weighted (dense regions get less bloom), `bloom_intensity` config

**Step 3: Write `docs/rendering/luminosity.md`** (~100 lines)

Cover the per-point brightness system (the thing we just fixed):
- `iter_lum = 1.0 - iter_lum_range * (iteration / max_iters)` — early iterations brighter
- `dist_lum = 1.0 / (1.0 + distance * dist_lum_strength)` — radial falloff from origin
- `xf_weight` clamped to [0.3, 1.0] — prevents dominant-transform brightness spikes
- Combined: `lum = iter_lum * dist_lum * clamp(xf_weight * 3.0, 0.3, 1.0)`
- **Known issue (fixed)**: `dist_lum_strength` was hardcoded at 0.3, causing center hole artifact via trail accumulation. Now defaults to 0.0 (disabled).
- Config: `dist_lum_strength` (default 0.0), `iter_lum_range` (default 0.5)

**Step 4: Commit**

```bash
git add docs/rendering/
git commit -m "docs: add feedback, post effects, and luminosity docs"
```

---

## Task 8: Genetics — All docs

**Files:**
- Create: `docs/genetics/README.md`
- Create: `docs/genetics/genome-format.md`
- Create: `docs/genetics/breeding.md`
- Create: `docs/genetics/mutation.md`
- Create: `docs/genetics/persistence.md`

**Step 1: Write `docs/genetics/README.md`** (~60 lines)

Overview + "When to Read" table. ASCII diagram of the evolution cycle:

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Parents  │───▶│ Breeding │───▶│ Mutation │───▶│ Offspring│
│ (voted   │    │ (slot    │    │ (affine, │    │ (render  │
│  pool)   │    │  crossvr)│    │  var, pal│    │  + vote) │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     ▲                                               │
     └───────────── upvote ──────────────────────────┘
```

**Step 2: Write `docs/genetics/genome-format.md`** (~150 lines)

Document `FlameGenome` and `FlameTransform` structs field by field. Source from `src/genome.rs`. Include:
- All FlameGenome fields with types and purposes
- All FlameTransform fields — especially the 26 variation weight fields by name
- GlobalParams and KifsParams sub-structs
- JSON serialization format (what a saved genome file looks like)
- Palette: 256 entries, RGB floats [0,1], how `color` field indexes into it

**Step 3: Write `docs/genetics/breeding.md`** (~180 lines)

Document `breed()` in `src/genome.rs`. Cover:
- Parent selection: voted pool → history → seeds → flames (priority order)
- Slot allocation strategy: wildcard (slot 0), then 4 groups split across remaining slots
- Group A: transforms from parent A
- Group B: transforms from parent B
- Group C: transforms from community genome (if available)
- Group D: environment transforms (random or taste-biased)
- Taste-biased wildcard: `generate_biased_transform()` when taste engine is active
- Post-breeding normalization: weights, variations, colors
- Config: `parent_current_bias`, `parent_voted_bias`, `parent_saved_bias`, `parent_random_bias`, `min_breeding_distance`, `max_lineage_depth`

**Step 4: Write `docs/genetics/mutation.md`** (~120 lines)

Document mutation operators. Source from `src/genome.rs`. Cover:
- Affine perturbation: small random deltas to a, b, c, d, offset
- Variation weight mutation: Gaussian noise, normalize after
- Determinant clamping: `det = |ad - bc|` kept in [0.2, 0.95]
- `normalize_variations()`: ensures at least one active variation, max 2 dominant
- `normalize_weights()`: transform selection weights sum to 1.0
- `distribute_colors()`: evenly spaces palette indices across transforms
- `estimate_attractor_extent()`: samples points to estimate visible size for auto-zoom
- Config: `magnitude_min`, `magnitude_max`, `randomness_range`, `max_mutation_retries`

**Step 5: Write `docs/genetics/persistence.md`** (~150 lines)

Document genome storage. Source from `src/votes.rs` and `src/main.rs`. Cover:
- Directory structure: `genomes/{voted/, history/, seeds/, flames/, votes.json, lineage.json}`
- Save flow: every genome → `history/`, upvoted → also `voted/`
- `votes.json`: `{filename: {score, file, last_seen}}`
- `lineage.json`: `{name: {parent_a, parent_b, generation, created}}`
- Archiving: generation-based, triggered when `history/` exceeds `archive_threshold_mb`
- Archive process: group by generation, compress below-median to tarball
- Migration: startup creates `voted/` if missing, moves positive-score genomes there
- Config: `archive_threshold_mb` (default 100), `archive_on_startup` (default true)

**Step 6: Commit**

```bash
git add docs/genetics/
git commit -m "docs: add genetics documentation (genome, breeding, mutation, persistence)"
```

---

## Task 9: Taste Engine — All docs

**Files:**
- Create: `docs/taste-engine/README.md`
- Create: `docs/taste-engine/palette-model.md`
- Create: `docs/taste-engine/transform-model.md`

**Step 1: Write `docs/taste-engine/README.md`** (~100 lines)

End-to-end overview of taste learning. Cover:
- Two models, same training data (upvoted genomes)
- Gaussian centroid scoring: lower score = closer to learned "good"
- Rebuild trigger: whenever votes change
- How taste influences breeding: biases random-source transforms only
- Novel combinations: transforms scored independently → can combine archetypes that never appeared together

**Step 2: Write `docs/taste-engine/palette-model.md`** (~150 lines)

Document `PaletteFeatures` (17 features) and palette generation. Source from `src/taste.rs`. Cover:
- 12-bin hue histogram (30° buckets, normalized)
- 5 stats: avg_saturation, saturation_spread, avg_brightness, brightness_range, hue_cluster_count
- `TasteModel::build()`: compute mean + stddev per feature, floor stddev at 0.01
- `TasteModel::score()`: sum of squared z-scores (Mahalanobis-like)
- `generate_palette()`: generate N candidates, score each, pick lowest. Diversity penalty against recent palettes.
- Config: `taste_candidates` (default 20), `taste_exploration_rate` (default 0.1), `taste_strength` (default 1.0), `taste_diversity_penalty` (default 0.3), `taste_recent_memory` (default 5), `taste_min_votes` (default 3)

**Step 3: Write `docs/taste-engine/transform-model.md`** (~150 lines)

Document `TransformFeatures` (8), `CompositionFeatures` (5), and biased generation. Cover:

Transform features table:

| Feature | Source | What it captures |
|---|---|---|
| `primary_variation_index` | Max-weight variation index | Which variation dominates (0-25) |
| `primary_dominance` | Max weight / total weight | How dominant is the primary (0-1) |
| `active_variation_count` | Count of weights > 0 | Complexity of the transform |
| `affine_determinant` | `|ad - bc|` | Contraction vs expansion |
| `affine_asymmetry` | `|a-d| + |b+c|` | Rotational asymmetry |
| `offset_magnitude` | `sqrt(ox² + oy²)` | How far from origin |
| `color_index` | `xf.color` | Palette position |
| `weight` | `xf.weight` | Selection probability |

Composition features table:

| Feature | Source | What it captures |
|---|---|---|
| `transform_count` | `genome.transforms.len()` | Structural complexity |
| `variation_diversity` | Unique active variation types | How varied the transform set is |
| `mean_determinant` | Average `|ad-bc|` | Overall contraction character |
| `determinant_contrast` | Stddev of determinants | Do transforms differ in scale? |
| `color_spread` | Stddev of color indices | Palette usage diversity |

`generate_biased_transform()`: generate N candidates, score each against transform model, pick lowest-scoring (closest to centroid). Exploration rate bypasses model for pure random.

**Step 4: Commit**

```bash
git add docs/taste-engine/
git commit -m "docs: add taste engine documentation (palette + transform models)"
```

---

## Task 10: Audio — All docs

**Files:**
- Create: `docs/audio/README.md`
- Create: `docs/audio/signal-mapping.md`

**Step 1: Write `docs/audio/README.md`** (~120 lines)

Audio signal pipeline overview. Source from `src/audio.rs`. Cover:
- Capture: macOS ScreenCaptureKit (system audio) or CPAL fallback
- FFT: 4096-sample window, Hann windowed
- Band splitting: exponential frequency spacing across 16 bands
- Bass (bands 0-2, ~0-250Hz), Mids (bands 4-7, ~250Hz-2kHz), Highs (bands 10-13, ~2kHz-20kHz)
- Normalization: adaptive floor, log compression
- Beat detection: spectral flux onset detection, accumulator with 0.95 decay
- `AudioFeatures` struct: bass, mids, highs, energy, beat, beat_accum, beat_pulse
- All values normalized to [0, 1]

**Step 2: Write `docs/audio/signal-mapping.md`** (~120 lines)

How audio and time signals modulate shader parameters. Source from `src/weights.rs`. Cover:
- `TimeSignals` struct: 9 oscillators at different speeds (time, time_slow through time_envelope)
- Signal weight mapping: `weights.json` maps signal names to parameter modulation strengths
- How it works: `param_value = base_value + signal_value * weight`
- Example mappings: bass → zoom, time_fast → spin, energy → bloom
- Available signals: all audio (bass, mids, highs, energy, beat, beat_accum) + all time (time, time_slow, time_med, time_fast, time_noise, time_drift, time_flutter, time_walk, time_envelope)

**Step 3: Commit**

```bash
git add docs/audio/
git commit -m "docs: add audio pipeline and signal mapping documentation"
```

---

## Task 11: Config — All docs

**Files:**
- Create: `docs/config/README.md`
- Create: `docs/config/signal-weights.md`

**Step 1: Write `docs/config/README.md`** (~100 lines)

How `weights.json` works. Source from `src/weights.rs` and `src/main.rs`. Cover:
- File structure: `_config_doc` (human docs), `_config` (values), signal sections (bass, mids, etc.)
- Hot-reloading: file watcher detects changes, re-parses, applies immediately
- RuntimeConfig deserialization: every field has `#[serde(default = "default_*")]`
- No magic numbers rule: every tunable value lives here
- `variation_scales`: per-variation multipliers (CRISPR-style), e.g. `{"spherical": 0.5}`
- How to add a new config field: add to RuntimeConfig with default function, add to weights.json, wire into uniform if GPU-side

**Step 2: Write `docs/config/signal-weights.md`** (~100 lines)

The signal modulation system in detail. Cover:
- Each parameter in `weights.json` has optional signal weight sections
- Signal names correspond to `TimeSignals` fields and `AudioFeatures` fields
- Mapping: `final_value = base + Σ(signal_i * weight_i)`
- The globals array: how `src/main.rs` combines base config + signal modulation into `globals[0..19]`
- Per-transform field mapping: `XF_FIELDS` array (42 fields per transform), `xf_field_index()` lookup

**Step 3: Commit**

```bash
git add docs/config/
git commit -m "docs: add config and signal weights documentation"
```

---

## Task 12: Update CLAUDE.md + Final verification

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add Documentation section to CLAUDE.md**

Add after the "Uniform Layout" section:

```markdown
### Documentation
- Docs live in `docs/` (Docsify site, `npx docsify-cli serve docs` to preview)
- When modifying rendering, genetics, taste, audio, or config systems, update the corresponding doc
- `docs/reference/vocabulary.md` is the single source of truth for terminology
- Living docs: update alongside code changes, not retroactively
```

**Step 2: Verify all docs render**

Run: `cd docs && npx docsify-cli serve . &`
Click through every sidebar link. Verify:
- All 22 markdown files render
- Sidebar navigation works
- Search returns results
- Code blocks have syntax highlighting
- Internal cross-references link correctly

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add documentation maintenance rule to CLAUDE.md"
```

**Step 4: Tag**

```bash
git tag -a v0.4.1-docs -m "Documentation system: Docsify site with full system docs"
```
