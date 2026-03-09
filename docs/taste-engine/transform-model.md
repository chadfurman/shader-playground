# Transform Model

How the taste engine learns and applies transform preferences.

## TransformFeatures (8 per transform)

Extracted from individual `FlameTransform` instances via `TransformFeatures::extract()`.

| Feature | Source | Range | What it captures |
|---|---|---|---|
| `primary_variation_index` | Index of max-weight variation | 0-25 | Which variation dominates |
| `primary_dominance` | max_weight / total_weight | 0-1 | How dominant is the primary |
| `active_variation_count` | Count of weights > 0 | 0-26 | Transform complexity |
| `affine_determinant` | `abs(a*d - b*c)` | 0+ | Contraction (<1) vs expansion (>1) |
| `affine_asymmetry` | `abs(a-d) + abs(b+c)` | 0+ | How non-rotationally-symmetric |
| `offset_magnitude` | `sqrt(ox^2 + oy^2)` | 0+ | Distance from origin |
| `color_index` | `xf.color` | 0-1 | Palette position |
| `weight` | `xf.weight` | 0+ | Transform selection probability |

The variation index iterates over indices 0 through 25 (26 variation slots total). If no variation has weight > 0, `primary_dominance` is 0.0.

## Training

During `TasteEngine::rebuild()`:

```
for each upvoted genome:
    for each transform in genome.transforms:
        TransformFeatures::extract(transform)  -->  8-feature vector
        add to transform_features pool

TasteModel::build(transform_features)  -->  transform model
```

- Pools ALL individual transforms from ALL upvoted genomes
- Extracts 8 features from each
- Builds a single `TasteModel` from the pooled feature vectors
- Transform model is independent of the palette model

## Biased Transform Generation

`TasteEngine::generate_biased_transform(min_votes, strength, exploration_rate, candidates)`:

1. **Exploration roll**: `rand() < exploration_rate` -- return pure random transform
2. **Model check**: need `sample_count >= min_votes`, otherwise return random
3. **Generate** `candidates` random transforms via `FlameTransform::random_transform()`
4. **Score each**: `model.score(features) * strength`
5. **Pick** lowest-scoring (closest to centroid)

### Scoring Transforms

`TasteEngine::score_transform()` exposes direct transform scoring:

- Returns `None` if the transform model hasn't been built or `sample_count < min_votes`
- Otherwise returns the raw model score for the given transform's features

## Why Independent Scoring Enables Novel Combinations

The model learns what individual transforms look good, not which combinations work together.

**Example:**
- Upvoted genome A has: spherical orbs (transform 1) + linear fill (transform 2)
- Upvoted genome B has: wispy waves (transform 1) + julia swirls (transform 2)
- Model learns: spherical, linear, wispy, and julia all score well individually
- During breeding: can generate spherical orbs + julia swirls -- a combination that never appeared in any upvoted genome

This is a feature, not a bug. The taste engine biases toward individually good transforms while leaving combinatorial exploration to the genetic algorithm. The result is that novel pairings emerge naturally, constrained only by per-transform quality.

## Relationship to Palette Model

Both models share the same `TasteModel` implementation (Gaussian centroid with stddev floor). The difference is:

| | Palette Model | Transform Model |
|---|---|---|
| **Unit** | One feature vector per genome | One feature vector per transform |
| **Features** | 22 (17 palette + 5 composition) | 8 |
| **Training pool** | N vectors (one per upvoted genome) | M vectors (sum of transforms across all upvoted genomes) |
| **Used during** | `generate_palette()` | `generate_biased_transform()` |
