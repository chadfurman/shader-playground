# Taste Engine

End-to-end overview of taste learning.

## How It Works

1. User upvotes genomes they like
2. Features extracted: palette (17) + composition (5) + perceptual (3) = 25 dimensions
3. IGMM (Incremental Gaussian Mixture Model) clusters votes into distinct style niches
4. During breeding, candidates scored against closest cluster (not a single average)
5. Novelty search rewards genomes that explore unexplored aesthetic territory
6. MAP-Elites archive ensures diverse parent selection across symmetry, fractal dimension, and color entropy

## IGMM vs Old Centroid

The old single Gaussian centroid averaged all votes into one "ideal" -- if you liked both sharp geometric fractals AND soft nebulous flows, it would learn a muddy middle ground. The IGMM maintains separate clusters per style:

```
Upvoted Genomes
      |
      +--> Feature extraction (25 dims)
      |
      +--> IGMM: find closest cluster
             |
             +-- Within threshold? --> Update cluster (EMA)
             +-- Too far? --> Spawn new cluster
             +-- Decay + prune old clusters
```

Scoring uses minimum distance across all clusters -- a genome matching ANY of your styles scores well.

## Perceptual Features

The taste engine evaluates what genomes actually *look like*, not just their parameters:

- **Fractal Dimension** -- box-counting FD on a 64x64 CPU proxy render. Humans prefer FD 1.3-1.5.
- **Spatial Entropy** -- Shannon entropy of hit distribution across 8x8 blocks. Low = collapsed, high = space-filling.
- **Coverage Ratio** -- fraction of grid cells hit. Catches degenerate and overblown genomes.

## Novelty Search

Breeding fitness combines taste and novelty:

```
fitness = taste_score - novelty_weight * novelty_score
```

Lower fitness = better. Subtracting novelty rewards novel genomes. The `novelty_score` is the average Euclidean distance to k-nearest neighbors in the MAP-Elites archive.

## MAP-Elites Archive

A 120-cell grid (6 symmetry bins x 5 FD bins x 4 color entropy bins) that stores the best genome per behavioral niche. Parent selection draws 50% from archive (uniform across occupied cells) and 50% from vote-weighted selection.

## Persistence

- IGMM state saved to `genomes/taste_model.json` -- loads on startup, updates incrementally on vote
- MAP-Elites archive saved to `genomes/archive.json`
- Cold start: if no saved model, bootstraps from all `voted/` genomes

## Config

| Field | Default | Description |
|---|---|---|
| `taste_engine_enabled` | `false` | Master gate |
| `taste_min_votes` | `10` | Minimum samples before model activates |
| `taste_strength` | `0.5` | Scoring weight multiplier |
| `taste_exploration_rate` | `0.1` | Probability of pure random (bypass model) |
| `taste_diversity_penalty` | `0.3` | Penalizes similarity to recent palettes |
| `taste_candidates` | `20` | Candidates to generate and score |
| `taste_recent_memory` | `5` | Recent palettes remembered for diversity |
| `igmm_activation_threshold` | `2.0` | Max distance to merge into existing cluster |
| `igmm_decay_rate` | `0.95` | Per-vote cluster weight decay |
| `igmm_min_weight` | `0.1` | Prune threshold for clusters |
| `igmm_max_clusters` | `8` | Max style clusters |
| `igmm_learning_rate` | `0.1` | EMA smoothing for cluster updates |
| `novelty_weight` | `0.3` | Explore/exploit balance (0 = pure taste, 1 = aggressive novelty) |
| `novelty_k_neighbors` | `5` | k for k-NN novelty calculation |

## Key Files

| File | What it contains |
|---|---|
| `src/taste.rs` | `TasteCluster`, `IgmmModel`, `PerceptualFeatures`, `proxy_render`, `novelty_score`, feature extraction |
| `src/archive.rs` | `MapElitesArchive`, `GridCoords`, persistence |
| `src/weights.rs` | `RuntimeConfig` taste/IGMM/novelty fields and defaults |
