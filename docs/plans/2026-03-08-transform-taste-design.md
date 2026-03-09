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

## Genome Persistence Restructure

All genomes are now saved to disk. Directory structure:

```
genomes/
  voted/          ← upvoted genomes copied here (breeding prefers these)
  history/        ← every genome ever generated (auto-evolve, breed, etc.)
  seeds/          ← hand-curated seeds (existing, unchanged)
  flames/         ← imported .flame files (existing, unchanged)
  votes.json      ← existing vote ledger
```

**Save flow:**
- Every auto-evolve/breed writes the new genome to `history/`
- Upvoting copies the genome to `voted/` (in addition to history)
- Downvoting does NOT copy to voted (stays only in history)
- `votes.json` file paths updated to point to `voted/` for upvoted genomes

**Breeding parent selection:**
- Prefers genomes from `voted/` (curated pool)
- Falls back to `history/` or `seeds/` or `flames/`

**Compression:**
- `history/` can be nuked/archived without losing curated genomes
- ~10KB/genome average, ~15:1 gzip compression ratio
- At 6000 genomes/day: ~60MB/day raw, ~4MB/day compressed

**Migration:**
- On startup, if `voted/` doesn't exist, create it
- Move any existing genomes in `genomes/*.json` that have positive votes into `voted/`
- Future genomes go to the new structure automatically

## Persistent Ancestry Tree

Replace the in-memory-only `LineageCache` (which rebuilds by scanning genome JSONs on startup) with a persistent `lineage.json` file.

```json
{
  "mutant-1234": {
    "parent_a": "mutant-0567",
    "parent_b": "seed-8901",
    "generation": 3,
    "created": "2026-03-08T14:30:00Z"
  }
}
```

**Properties:**
- Tiny file — just names, parent pointers, generation, timestamp. Grows ~100 bytes per genome.
- Updated on every breed (append new entry, flush to disk)
- Survives archiving — even after old genome JSONs are deleted, the lineage tree remains complete
- `LineageCache` loads from `lineage.json` on startup instead of scanning genome files
- Still falls back to scanning genome JSONs if `lineage.json` doesn't exist (migration)

## Archiving Strategy

Generation-based archiving with size fallback. The `voted/` directory is never archived — only `history/`.

**Trigger:** When `history/` exceeds `archive_threshold_mb` (config, default 100MB).

**Process:**
1. Scan all genomes in `history/`, group by generation
2. Find the median generation
3. Move all genomes below the median generation into `history/archive-YYYY-MM-DD.tar.gz`
4. Delete the archived JSON files from `history/`
5. `lineage.json` is untouched (ancestry survives)

**Config fields:**
- `archive_threshold_mb` — size at which archiving triggers (default 100)
- `archive_on_startup` — whether to check/archive on app start (default true)

**Manual trigger:** Could also be triggered by a keyboard shortcut or CLI flag later.

## Config

Reuses existing taste config fields:

- `taste_candidates` — number of candidates to generate and score
- `taste_exploration_rate` — probability of bypassing taste model
- `taste_strength` — scoring weight
- `taste_min_votes` — minimum votes before model activates
- `taste_engine_enabled` — master gate

New config fields for archiving:

- `archive_threshold_mb` — history size threshold before archiving (default 100)
- `archive_on_startup` — check and archive on app start (default true)

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
