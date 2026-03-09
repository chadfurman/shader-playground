# Vocabulary

Single source of truth for shader-playground terminology.

## Pipeline Overview

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

## Terms

### Accumulation buffer

7-channel per-pixel buffer: density, R, G, B, vel_x, vel_y, depth.
Written by the compute shader via atomic ops. Fixed-point encoding
(×1000 for density/depth, ×10000 for velocity).

### Affine transform

2×3 matrix `[a b; c d] + [e, f]` applied to each point. Stored as
fields `a, b, c, d` plus `offset[2]` on `FlameTransform`.

### Attractor

The set of points the chaos game converges to. Shape determined by
transforms + variations.

### Breeding

`breed()` in `genome.rs`. Combines two parent genomes via slot-based
crossover producing offspring with traits from both.

### Chaos game

Core algorithm: randomly select a weighted transform, apply it to a
point, plot the result. Repeat millions of times per frame.

### Composition features

5 genome-level stats (transform_count, variation_diversity,
mean_determinant, determinant_contrast, color_spread) used by the taste
model.

### Determinant

`|ad - bc|` of an affine transform. Values <1 contract space, >1
expand. Clamped to [0.2, 0.95] during mutation.

### Genome

`FlameGenome` struct. Complete genetic specification: transforms,
palette, global params, symmetry, lineage.

### Lineage

Parent→child ancestry tree tracked in `lineage.json`. Maps genome
name → {parent_a, parent_b, generation, created}.

### Morph

Smooth interpolation between two genomes over `morph_duration` frames.
All parameters (transforms, palette, globals) linearly blended.

### Mutation

Random perturbation of a genome's affine coefficients, variation
weights, or parameters.

### Palette

256-entry `Vec<[f32; 3]>` RGB color table. Each transform's `color`
field (0–1) indexes into it.

### Splatting

Bilinear sub-pixel point deposition into the accumulation buffer.
Distributes point across 2×2 pixel quad weighted by sub-pixel position.

### Symmetry

Point replication. Positive values = rotational (N copies at 360°/N
intervals). Negative = bilateral mirror.

### Taste model

Gaussian centroid model (`TasteModel`). Learns feature means + stddevs
from upvoted genomes. Scores new genomes by distance from centroid.

### Trail

Temporal feedback: `col = max(current, prev_frame * trail)`. Creates
persistence/motion trails. Controlled by `trail` config.

### Transform

`FlameTransform`. One affine map + 26 variation weights + color index +
selection weight.

### Transform features

8 per-transform stats (primary_variation_index, primary_dominance,
active_variation_count, affine_determinant, affine_asymmetry,
offset_magnitude, color_index, weight) used by the taste model.

### Variation

Nonlinear function applied after affine transform. 26 available:
linear, sinusoidal, spherical, swirl, horseshoe, handkerchief, julia,
polar, disc, rings, bubble, fisheye, exponential, spiral, diamond,
bent, waves, popcorn, fan, eyefish, cross, tangent, cosine, blob,
noise, curl.

### Vibrancy

Flam3 color blend parameter. Higher values preserve saturation in
sparse regions. Formula:
`ls = vibrancy * alpha + (1 - vibrancy) * pow(alpha, gamma)`.

### Warmup

Initial chaos game iterations (default 20) skipped before plotting.
Lets points settle onto attractor.
