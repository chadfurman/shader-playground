# Phase 3: Transform Taste Engine — Design

## Goal

Extend the taste engine to learn transform-level preferences and use them to bias random transform generation during breeding. Novel combinations emerge naturally because each transform is scored independently.

## Two Models, Same Training Data

Both models rebuild from upvoted genomes whenever votes change (same trigger as current palette model rebuild).

### 1. Transform Taste Model (new)

- Training data: every individual transform from every upvoted genome, pooled together
- Gaussian centroid model (same `TasteModel` math as existing palette model)
- ~8 features per transform
- Used during breeding to bias random transform generation

### 2. Genome Composition Model (existing palette model, expanded)

- Training data: one feature vector per upvoted genome
- Existing 17 palette features + 5 new composition features = 22 total
- Used during breeding to bias palette generation (existing flow, richer features)

## Transform Features (8 per transform)

| Feature | What it captures |
|---------|-----------------|
| primary_variation_index | Which variation dominates (0-25) |
| primary_dominance | Top variation weight / total (0-1) |
| active_variation_count | How many variations have weight > 0 |
| affine_determinant | |ad - bc| — contraction/expansion |
| affine_asymmetry | |a-d| + |b+c| — how non-rotationally-symmetric |
| offset_magnitude | sqrt(ox^2 + oy^2) |
| color_index | Palette position (0-1) |
| weight | Transform selection probability |

## Composition Features (5, appended to genome-level feature vector)

| Feature | What it captures |
|---------|-----------------|
| transform_count | Number of transforms |
| variation_diversity | Unique active variation types across all transforms |
| mean_determinant | Average affine determinant |
| determinant_contrast | Stddev of determinants (do transforms differ?) |
| color_spread | Stddev of color indices across transforms |

## Breeding Integration

Only the "fresh random" source is biased (currently ~20% of transform slots). Other sources (current genome, voted pool, saved genome) are already user-curated.

When breeding assigns a slot to fresh random:

1. Generate `taste_candidates` candidate transforms
2. Score each against the transform taste model
3. Apply `taste_exploration_rate` — sometimes skip scoring and pick pure random
4. Pick the lowest-scoring candidate (closest to learned centroid)

## Config

Reuses existing taste config fields:

- `taste_candidates` — number of candidates to generate and score
- `taste_exploration_rate` — probability of bypassing taste model
- `taste_strength` — scoring weight
- `taste_min_votes` — minimum votes before model activates
- `taste_engine_enabled` — master gate

No new config fields needed.

## Architecture

### In taste.rs

- `TransformFeatures` struct with `extract(xf: &FlameTransform)` and `to_vec()`
- `CompositionFeatures` struct with `extract(genome: &FlameGenome)` and `to_vec()`
- Expand `TasteEngine` with a second `TasteModel` for transforms
- `TasteEngine::generate_biased_transform()` — generate and score candidates
- `TasteEngine::rebuild()` — now also builds the transform model from pooled transforms

### In genome.rs

- `breed()` calls `taste.generate_biased_transform()` for random-source slots when taste is enabled
- Falls back to `FlameTransform::random_transform()` when taste is inactive

## Novel Combinations

The transform model scores transforms independently. If you upvote a genome with spherical orbs and another with wispy waves, the model learns both archetypes are good. During breeding, it can generate a spherical orb for slot 1 and wispy waves for slot 2 — a novel combination, both individually vetted.

The composition model adds a soft preference for your structural tastes (variation diversity, affine contrast, transform count).

## Testing

- `TransformFeatures::extract` on known transforms
- `CompositionFeatures::extract` on known genomes
- Transform taste model build/score (same pattern as existing palette model tests)
- `generate_biased_transform` returns valid transforms
- Composition features have correct dimensionality
