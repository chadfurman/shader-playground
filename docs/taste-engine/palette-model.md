# Palette Model

How the taste engine learns and applies palette preferences.

## PaletteFeatures (17 features)

Extracted from 256-entry RGB palettes via `PaletteFeatures::extract()`.

### Hue Histogram (12 features)

- 12 bins, 30 degrees each, covering the full hue wheel
- Only counts entries with saturation > 0.05 (skips near-gray)
- Normalized to sum to 1.0

### Statistics (5 features)

| Feature | Formula | What it captures |
|---|---|---|
| `avg_saturation` | mean(S) across palette | Overall color intensity |
| `saturation_spread` | stddev(S) | Mix of vivid and muted? |
| `avg_brightness` | mean(V) across palette | Overall lightness |
| `brightness_range` | max(V) - min(V) | Dynamic range |
| `hue_cluster_count` | contiguous non-empty hue bins | Number of distinct color groups |

Hue clusters are wraparound-aware: bins 0 and 11 being active counts as one cluster (not two). A bin is considered "non-empty" when its value exceeds 0.01.

## Composition Features (5 features, appended to palette model)

Genome-level structural features appended to the palette feature vector during model building, making the full palette model 22 features wide.

| Feature | Formula | What it captures |
|---|---|---|
| `transform_count` | `genome.transforms.len()` | Structural complexity |
| `variation_diversity` | unique active variation types across all transforms | How varied the transform set |
| `mean_determinant` | mean(`abs(a*d - b*c)`) across transforms | Overall contraction character |
| `determinant_contrast` | stddev of affine determinants | Do transforms differ in scale? |
| `color_spread` | stddev of color indices across transforms | Palette usage diversity |

Total genome-level model: 17 palette + 5 composition = 22 features.

## TasteModel

### Building (`TasteModel::build`)

- Input: `Vec` of feature vectors from upvoted genomes
- Compute mean per feature dimension
- Compute stddev per feature dimension
- Floor stddev at 0.01 (prevents collapse on low-variance features)
- Returns `None` if the input is empty

### Scoring (`TasteModel::score`)

- Sum of squared z-scores: `sum((val - mean) / stddev)^2`
- Lower = closer to centroid = more "tasteful"
- Scoring at the exact mean returns 0.0

## Palette Generation

`TasteEngine::generate_palette()` flow:

1. **Exploration check** -- roll `rand() < exploration_rate`, if true return a pure random palette
2. **Model readiness check** -- need `sample_count >= min_votes`, otherwise return random
3. **Generate candidates** -- create `taste_candidates` random palettes
4. **Score each candidate:**
   - Taste score: `model.score(features) * taste_strength`
   - Diversity penalty: for each recent palette, add `hue_overlap * diversity_penalty`
   - Total = taste score + diversity penalties
5. **Select** the lowest-scoring candidate
6. **Record** in recent memory (`VecDeque`, capped at `taste_recent_memory`)

### Hue Overlap

`PaletteFeatures::hue_overlap()` computes the histogram intersection between two palettes:

```
overlap = sum(min(bin_a[i], bin_b[i])) for i in 0..12
```

- Returns 0.0 for completely disjoint hue distributions
- Returns 1.0 for identical distributions
- Used by the diversity penalty to push new palettes away from recent ones

## Feature Extraction Path

During `TasteEngine::rebuild()`:

```
for each upvoted genome:
    PaletteFeatures::extract(genome)  -->  17 features
    CompositionFeatures::extract(genome)  -->  5 features
    concatenate  -->  22-feature vector
    add to good_features pool

TasteModel::build(good_features)  -->  palette model
```

The palette model and transform model are built independently in the same `rebuild()` call but from different feature sets.
