# Evolution Overhaul — Design Spec

## Overview

Overhaul the fractal flame evolution system across two parallel tracks:
- **Track 1 (3D Rendering):** Full Sdobnov 3D hack — 3x3 affines, camera, perspective, DOF
- **Track 2 (Taste Engine):** IGMM, perceptual features, MAP-Elites, novelty search, interpolative crossover

Tracks are mostly independent during development but merge when 3D proxy renders feed richer features into the IGMM. **Caveat:** `taste.rs` directly references `xf.a`, `xf.b`, `xf.c`, `xf.d` for feature extraction. When Track 1 replaces these with `affine: [[f32; 3]; 3]`, the taste feature extraction must be updated too. If Track 2 lands first, it uses the old field names; when Track 1 lands, a migration pass updates taste.rs.

## Decisions

- Full Sdobnov 3D (3x3 affines, camera pitch/yaw, perspective divide, Gaussian DOF)
- CPU proxy render (64x64, 500 iterations) for perceptual features — per-breed, ~2-5ms
- Auto-upgrade old 2x2 genomes to 3x3 by padding z with identity values
- Approach B: parallel development tracks, interleaved implementation

---

## Track 1: 3D Rendering

### 1.1 Genome Format Upgrade

**FlameTransform changes:**

Replace `a, b, c, d: f32` + `offset: [f32; 2]` with:
```
affine: [[f32; 3]; 3]  // 3x3 rotation/scale/shear
offset: [f32; 3]       // xyz translation
```

Top-left 2x2 of `affine` is the existing `a,b,c,d`. Third row/column handles z. `offset[2]` is z-translation.

**Migration:** On deserialization, if old `a,b,c,d` fields present and `affine` absent:
```
affine = [[a, b, 0], [c, d, 0], [0, 0, 1]]
offset = [ox, oy, 0]
```

**Determinant clamping:** Extends to 3x3 determinant. Same [0.2, 0.95] bounds. Clamping strategy: uniformly scale all 9 affine entries by `(target / |det|).cbrt()` to preserve rotational/shear character. The current 2x2 method of only adjusting `a` and `d` does not generalize to 3x3.

**New mutation operators:**
- `mutate_z_tilt` — rotate z-row to tilt attractor in depth
- `mutate_z_scale` — scale z component (controls depth spread)
- Combined ~5% initial mutation probability so evolution gradually discovers depth

**Compute shader transform buffer:** Grows from 42 to 48 floats per transform. Delta: 3x3 affine (9) replaces a,b,c,d (4) = +5, offset grows from 2 to 3 = +1. Total = 42 + 6 = 48. The `PARAMS_PER_XF` constant in `weights.rs` and all `* 42u` references in the shader must be updated to 48 in sync. The `XF_FIELDS` array in `weights.rs` must be extended with the new field names.

**Point state buffer:** Grows from 3 floats per thread `(x, y, color_idx)` to 7 floats per thread `(x, y, z, prev_x, prev_y, prev_z, color_idx)`. Buffer allocation in `main.rs` changes from `max_threads * 12` bytes to `max_threads * 28` bytes.

### 1.2 Compute Shader — 3D Chaos Game

**Point state:** Extends from `(x, y)` to `(x, y, z)`. Previous point also 3D for velocity.

**Iteration loop:** Applies full 3x3 affine to `vec3(x, y, z)` instead of 2x2 to `vec2(x, y)`.

**Velocity:** `vel = p.xyz - prev_p.xyz` — tracked as 3D vector.

**At splat time:** Both position and velocity are projected through the camera transform. The histogram receives 2D screen-space position + 2D projected velocity + z-depth. Histogram format stays at 7 channels per pixel (density, r, g, b, vx, vy, depth).

A point moving in z toward the camera produces radial zoom-blur in screen space naturally through the perspective projection of the velocity vector.

### 1.3 Camera & Perspective Projection

**New RuntimeConfig values (weights.json):**
- `camera_pitch` — x-axis rotation (default 0.0)
- `camera_yaw` — y-axis rotation (default 0.0)
- `camera_focal` — perspective divide distance (default 2.0, larger = flatter)

**Projection pipeline in compute shader:**
1. Camera rotation: apply pitch/yaw as 3x3 rotation matrix to `(x, y, z)`
2. Perspective divide: `screen_x = x / (z/focal + 1.0)`, `screen_y = y / (z/focal + 1.0)`

Higher z → points converge toward vanishing point. High focal values = subtle perspective, low = dramatic.

**Velocity projection:** Apply the same camera rotation + perspective Jacobian to the 3D velocity vector to get correct screen-space motion vectors.

### 1.4 Depth of Field

**RuntimeConfig values:**
- `dof_focal_plane` — z-distance in perfect focus (default 0.0)
- `dof_strength` — blur intensity (default 0.0 = disabled)

**Implementation:** DOF is applied as a post-process blur in the display shader, NOT at splat time. The histogram already stores z-depth per pixel (channel 7). The display shader reads each pixel's depth, computes `blur_radius = dof_strength * abs(depth - dof_focal_plane)`, and applies a depth-weighted blur. This avoids O(radius^2) atomic writes per point during splatting, which would negate the subgroup optimization gains.

**Uniform slots:** Current `extra4`–`extra6` are fully allocated. Add `extra7: vec4<f32>` for `(camera_pitch, camera_yaw, camera_focal, dof_focal_plane)` and `extra8: vec4<f32>` for `(dof_strength, z_mutation_rate, reserved, reserved)`. Requires extending the `Uniforms` struct in both Rust and WGSL.

**Audio modulation:** Camera params may drift slowly with audio signals — NOT beat-reactive (per rubber-banding lesson).

---

## Track 2: Taste Engine Overhaul

### 2.1 Perceptual Features

**CPU proxy render per candidate genome during breed/mutate:**
- 64x64 binary grid (hit/no-hit)
- 500 iterations, skip first 50 warmup
- Affine-only (no variation functions) — sufficient for FD/entropy/coverage at proxy resolution, avoids porting all 26 variation functions to CPU Rust
- ~1-3ms per evaluation (faster without variations)

**Three features extracted:**

1. **Fractal Dimension (FD)** — box-counting at sizes 2, 4, 8, 16, 32. FD = slope of log(count) vs log(1/box_size). Sweet spot: 1.3–1.5.
2. **Spatial entropy** — divide into 8x8 blocks, Shannon entropy of hit distribution. Low = collapsed attractor, high = space-filling.
3. **Coverage ratio** — fraction of 64x64 grid with hits. Catches degenerate (near 0) and overblown (near 1.0) genomes.

**Feature vector grows to 25:** 17 (palette) + 5 (composition) + 3 (perceptual). Transform features remain separate (8 per transform, scored individually).

### 2.2 IGMM Taste Engine

**Replace single Gaussian centroid with Incremental Gaussian Mixture Model.**

**TasteCluster struct:**
```rust
struct TasteCluster {
    mean: Vec<f32>,        // feature centroid
    variance: Vec<f32>,    // per-feature variance (diagonal covariance)
    weight: f32,           // reinforcement weight, decays over time
    sample_count: u32,     // votes that fed this cluster
}
```

**On upvote:**
1. Extract 25-dim feature vector from genome
2. Compute Mahalanobis distance to each existing cluster
3. If closest cluster within activation threshold (~2σ) → update that cluster's mean/variance via EMA
4. If no cluster close → spawn new cluster centered on this genome
5. Decay all cluster weights. Prune clusters below kill threshold.

**Scoring a candidate:**
- Distance to each cluster
- Score = minimum distance across all clusters
- Genome matching ANY style scores well (not forced to match the average)

**Config (weights.json):**
- `igmm_activation_threshold` (default 2.0)
- `igmm_decay_rate` (default 0.95)
- `igmm_min_weight` (default 0.1)
- `igmm_max_clusters` (default 8)
- `igmm_learning_rate` (default 0.1)

**Persistence:** IGMM state saved to `genomes/taste_model.json` (cluster means, variances, weights, sample counts). On startup, load saved model — no full rebuild. On new vote, incrementally update the relevant cluster and save. Full rebuild from `voted/` genomes only if model file is missing (first run after upgrade or manual reset).

**Cold start:** On first run (no saved model), scans all `voted/` genomes to bootstrap clusters. After that, behaves incrementally. Degrades to single-centroid behavior until divergent votes spawn 2+ clusters naturally.

**Scope:** IGMM replaces the genome-level taste model (palette + composition + perceptual features = 25 dims). The separate `transform_model` (8 features per transform) continues using the existing Gaussian centroid approach — it scores individual transforms, not genomes, so multimodal clustering is less critical there.

### 2.3 MAP-Elites Archive

**3D grid replacing the flat genome pool for parent selection:**

| Axis | Bins | Range | Captures |
|------|------|-------|----------|
| Symmetry | 6 | 1–6 fold (abs value, clamped) | Structured vs asymmetric |
| Fractal Dimension | 5 | 1.0–2.0 | Sparse vs dense |
| Color Entropy | 4 | Low–High | Monochromatic vs vibrant |

**Total: 120 cells.** Each holds at most one genome — the highest IGMM-scoring genome for that behavioral niche.

**Insertion (after breed/mutate):**
1. Compute behavioral characteristics (symmetry, FD, color entropy)
2. Map to grid cell
3. If empty → insert. If occupied → replace only if IGMM score is better.

**Parent selection:**
1. Pick random occupied cell (uniform across cells, not genomes)
2. Parent A = that cell's genome
3. Parent B = different occupied cell
4. Guarantees parents from diverse aesthetic niches

**Persistence:** `genomes/archive.json` — genome names + grid coordinates. Genome files stay in `history/`. Rebuilt on startup.

**Coexistence with existing selection:** MAP-Elites does NOT replace `pick_voted()`/`pick_random_saved()` entirely. Parent selection becomes: 50% chance pick from MAP-Elites archive (uniform across occupied cells), 50% chance use existing vote-weighted selection. This preserves the direct influence of votes while introducing archive diversity. Upvoted genomes also insert into the grid, so highly-voted genomes appear in both selection paths.

### 2.4 Novelty Search

**Blend novelty bonus into breeding fitness.**

Candidate scoring during breed/mutate:
- `taste_score` = IGMM distance (lower = closer to a style)
- `novelty_score` = average feature-space Euclidean distance to k=5 nearest neighbors in MAP-Elites archive
- `fitness = taste_score - novelty_weight * novelty_score`

Lower fitness = better. Subtracting novelty **rewards** novel genomes by lowering their fitness score. A genome that tastes good AND is novel gets the best (lowest) fitness.

**Config:**
- `novelty_weight` (default 0.3) — 0 = pure taste, 1 = aggressive exploration
- `novelty_k_neighbors` (default 5)

Cheap computation — archive has at most 120 entries, k=5 NN on 25-dim vectors. Fallback: if fewer than k occupied cells, use `min(k, occupied_cells - 1)`.

### 2.5 Interpolative Crossover

**Replace slot-swapping with smooth parameter blending.**

When Parent A and B share a variation type in a given slot:
- `child.affine = lerp(a.affine, b.affine, t)` — t random per-slot
- Blend variation weights, color index, transform weight, variation params
- t constrained to [0.3, 0.7] to avoid near-copies of either parent

When they don't share a variation type → fall back to current slot-swap.

**Genetic distance guardrail:** Existing LineageCache ancestry check stays. Add feature-space distance check — parents must be moderately different (not identical, not orthogonal).

**Config:**
- `interpolation_range` (default [0.3, 0.7])

---

## New Config Values Summary

All added to `RuntimeConfig` in `weights.rs`, exposed in `weights.json`:

### Track 1 (3D)
| Key | Default | Purpose |
|-----|---------|---------|
| `camera_pitch` | 0.0 | Camera x-rotation |
| `camera_yaw` | 0.0 | Camera y-rotation |
| `camera_focal` | 2.0 | Perspective divide distance |
| `dof_focal_plane` | 0.0 | Z-distance in focus |
| `dof_strength` | 0.0 | DOF blur intensity (0 = disabled) |
| `z_mutation_rate` | 0.05 | Probability of z-tilt/z-scale mutations |

### Track 2 (Taste)
| Key | Default | Purpose |
|-----|---------|---------|
| `igmm_activation_threshold` | 2.0 | Max distance to merge into cluster |
| `igmm_decay_rate` | 0.95 | Per-vote cluster weight decay |
| `igmm_min_weight` | 0.1 | Prune threshold |
| `igmm_max_clusters` | 8 | Max style clusters |
| `igmm_learning_rate` | 0.1 | EMA smoothing for updates |
| `novelty_weight` | 0.3 | Explore/exploit balance |
| `novelty_k_neighbors` | 5 | k for k-NN novelty calc |
| `interpolation_range_lo` | 0.3 | Crossover blend lower bound |
| `interpolation_range_hi` | 0.7 | Crossover blend upper bound |

---

## Implementation Order

### Track 1 (3D Rendering)
1. Genome format (3x3 affines + migration)
2. Compute shader (3D iteration + velocity)
3. Camera + perspective projection
4. DOF blur at splat time
5. Z-specific mutation operators

### Track 2 (Taste Engine)
1. Perceptual features (proxy render + FD + entropy + coverage)
2. IGMM (replace Gaussian centroid)
3. Interpolative crossover
4. MAP-Elites archive
5. Novelty search scoring

### Merge Point
Once both tracks are complete, the 3D proxy render feeds z-depth features into the IGMM, and MAP-Elites gains depth-related behavioral dimensions.

---

## Files Affected

### Track 1
- `src/genome.rs` — FlameTransform struct, affine fields, migration, z-mutations, serialization
- `src/weights.rs` — camera/DOF config fields
- `flame_compute.wgsl` — 3D iteration, 3x3 affine application, camera projection, velocity projection, DOF scatter
- `src/main.rs` — pass camera uniforms to shader

### Track 2
- `src/taste.rs` — IGMM model, perceptual features, proxy render, novelty scoring
- `src/genome.rs` — interpolative crossover in breed(), MAP-Elites parent selection
- `src/votes.rs` — MAP-Elites archive persistence
- `src/weights.rs` — IGMM/novelty/interpolation config fields
- `src/main.rs` — archive management, rebuild hooks

### Shared
- `weights.json` — all new config values
- Transform buffer layout (genome.rs ↔ flame_compute.wgsl)
