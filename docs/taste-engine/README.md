# Taste Engine

End-to-end overview of taste learning.

## How It Works

1. User upvotes genomes they like
2. Features extracted from all upvoted genomes
3. Two Gaussian centroid models built (palette + transform)
4. During breeding, new random elements scored against models
5. Lowest-scoring candidates (closest to "good" centroid) selected

## Two Models, Same Training Data

```
Upvoted Genomes
      |
      +-->  Palette Features (17 per genome)  -->  Palette Model   -->  bias palette generation
      |     + Composition Features (5)
      |
      +-->  Transform Features (8 per xform)  -->  Transform Model -->  bias random transforms
```

- Both models rebuild whenever votes change
- Same trigger as vote ledger save

## Novel Combinations

- Transform model scores transforms INDEPENDENTLY
- If you upvote spherical orbs AND wispy waves, model learns both are good
- During breeding: can generate spherical for slot 1, waves for slot 2
- Novel combination, both individually vetted

## Scoring

- Gaussian centroid: mean + stddev per feature, floor stddev at 0.01
- Score = sum of squared z-scores: `sum((val - mean) / stddev)^2`
- Lower = better (closer to centroid)

## Exploration

- `taste_exploration_rate`: probability of bypassing taste entirely (pure random)
- Prevents the model from collapsing to a narrow style

## Config

All taste config fields live in `RuntimeConfig` (`src/weights.rs`) and are hot-reloaded from `weights.json`.

| Field | Default | Description |
|---|---|---|
| `taste_engine_enabled` | `false` | Master gate |
| `taste_min_votes` | `10` | Minimum samples before model activates |
| `taste_strength` | `0.5` | Scoring weight multiplier |
| `taste_exploration_rate` | `0.1` | Probability of pure random (bypass model) |
| `taste_diversity_penalty` | `0.3` | Penalizes similarity to recent palettes |
| `taste_candidates` | `20` | Number of candidates to generate and score |
| `taste_recent_memory` | `5` | Recent palettes remembered for diversity |

## Key Files

| File | What it contains |
|---|---|
| `src/taste.rs` | `PaletteFeatures`, `TransformFeatures`, `CompositionFeatures`, `TasteModel`, `TasteEngine` |
| `src/weights.rs` | `RuntimeConfig` taste fields and defaults |
