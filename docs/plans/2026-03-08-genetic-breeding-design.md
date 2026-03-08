# Genetic Breeding + Taste Engine — Design

## Goal

Replace the current mutation system with proper sexual reproduction: two parent genomes breed to produce offspring, with lineage tracking, inbreeding avoidance, community gene influence, and a learned taste model for color/aesthetic quality.

## Overview

```
                    GENE POOL
    ┌─────────────────────────────────────┐
    │  Voted favorites (score > 0)        │
    │  Imported flames (community)        │
    │  Saved genomes (unvoted)            │
    │  Seed genomes                       │
    └─────────────┬───────────────────────┘
                  │
                  ▼
    ┌─ PARENT SELECTION ──────────────────┐
    │  Pick Parent A from gene pool       │
    │  Pick Parent B from gene pool       │
    │  Constraint: genetic_distance(A,B)  │
    │    must exceed inbreeding threshold │
    │  Fallback: imported flame if pool   │
    │    is too homogeneous               │
    └─────────────┬───────────────────────┘
                  │
                  ▼
    ┌─ BREEDING (per-transform crossover) ┐
    │                                      │
    │  1 transform → fresh random wildcard │
    │  Remaining N-1 split into 4 groups:  │
    │    25% → from Parent A               │
    │    25% → from Parent B               │
    │    25% → from community pool         │
    │    25% → random environment          │
    │           (audio + noise + seed)     │
    │                                      │
    │  Symmetry: random pick from A or B   │
    │  Transform count: avg of A and B ±1  │
    │                                      │
    └─────────────┬───────────────────────┘
                  │
                  ▼
    ┌─ PALETTE (taste engine) ────────────┐
    │  Extract color features from good   │
    │    genomes (voted + imported)        │
    │  Generate palette that scores well   │
    │    against taste model               │
    │  Diversity nudge: avoid repeating    │
    │    recent hue distributions          │
    └─────────────┬───────────────────────┘
                  │
                  ▼
    ┌─ MUTATION (single-gene tweak) ──────┐
    │  Apply one mutation type (existing   │
    │    mutate_inner logic)               │
    │  Normalize, distribute colors        │
    │  Attractor extent check              │
    └─────────────┬───────────────────────┘
                  │
                  ▼
              CHILD GENOME
         (parent_a, parent_b recorded)
```

## Part 1: Lineage Tracking

### Genome Identity

Every genome gets:
```rust
pub struct FlameGenome {
    pub name: String,
    pub parent_a: Option<String>,  // name of first parent
    pub parent_b: Option<String>,  // name of second parent
    pub generation: u32,           // how many breeding cycles deep
    // ... existing fields
}
```

Imported flames and seeds have `parent_a: None, parent_b: None, generation: 0`.

### Genetic Distance

Distance between two genomes = how far back you have to go to find a common ancestor.

```
distance(A, B):
  if A == B → 0
  if A.parent_a == B or A.parent_b == B → 1
  if they share a parent → 1
  if they share a grandparent → 2
  ... walk up both family trees
  if no common ancestor found within max_depth → max_depth (most diverse)
```

Implementation: store a `LineageCache` — a HashMap from genome name to (parent_a, parent_b). Walk both ancestor chains, find lowest common ancestor. Cap search at depth 6-8 to avoid expensive lookups.

Config:
```json
"min_breeding_distance": 3,
"max_lineage_depth": 8
```

### Parent Selection

```
1. Collect candidate pool:
   - Voted genomes with score > 0 (weighted by score)
   - Imported flames (weight 1 each)
   - Saved genomes with score >= 0 (weight 1 each)
   - Seed genomes (weight 1 each)

2. Pick Parent A (vote-weighted)

3. Pick Parent B candidates:
   - Filter pool to genetic_distance(A, candidate) >= min_breeding_distance
   - If no candidates pass: use an imported flame (maximum diversity)
   - If still none: generate a random seed genome
   - Pick from filtered pool (vote-weighted)
```

## Part 2: Breeding Mechanics

### Transform Crossover (reworked)

```
Input: Parent A, Parent B, community genome, config
Output: Child genome (not yet mutated)

1. Determine child transform count:
   avg = (A.transforms.len() + B.transforms.len()) / 2
   child_count = avg + random(-1, +1)  // clamped to 3..=6

2. Create empty child with child_count transform slots

3. Mark 1 random slot as "wildcard" → fill with fresh random transform

4. For remaining slots, shuffle and assign sources:
   Split into 4 roughly equal groups:
   - Group 1 (25%): copy transform from Parent A (by index, wrapping)
   - Group 2 (25%): copy transform from Parent B (by index, wrapping)
   - Group 3 (25%): copy transform from a community genome
     (randomly pick an imported flame or voted favorite,
      then pick a random transform from it)
   - Group 4 (25%): environment transform
     (random seed transform, optionally biased by audio features)

5. Symmetry: randomly pick A's or B's symmetry (50/50)

6. Palette: generated by taste engine (see Part 3)

7. Record lineage:
   child.parent_a = A.name
   child.parent_b = B.name
   child.generation = max(A.generation, B.generation) + 1
```

### Environment Transforms

The "random environment" group represents external influence — things that aren't inherited from any genome. Options:

- Pure random transform (current `random_transform()`)
- Audio-biased: if bass is high, bias toward spherical/bubble; if highs, bias toward julia/disc (using existing `audio_biased_variation_pick`)
- Noise-seeded: use a time-varying noise value to deterministically pick variation type, so the "environment" slowly shifts over time

Start with audio-biased random, since the audio→variation mapping already exists.

## Part 3: Taste Engine (Color)

### Feature Extraction

For each genome, extract a feature vector describing its color characteristics:

**Palette-level features (17):**
- Hue histogram: 12 bins (one per 30° of hue wheel)
- Average saturation
- Saturation spread (stddev)
- Average brightness
- Brightness range (max - min)
- Distinct hue cluster count (hues separated by > 30°)

**Transform-color features (per variation type, 26×2 = 52):**
- For each variation type: preferred hue (circular mean of color indices
  in transforms that use this variation, mapped through the palette)
- For each variation type: preferred saturation

**Pairing features (top 20 most common variation pairs):**
- For each frequently co-occurring variation pair (A, B):
  hue_difference(color_of_A_transform, color_of_B_transform)
  This captures "spherical in warm + swirl in cool = good"

Total: ~89 features. Plenty for a Gaussian model, not too many for
small sample sizes with regularization.

### Taste Model

**Phase 1: Gaussian centroid (start here)**

```
TasteModel {
    feature_means: Vec<f32>,     // mean of each feature across good genomes
    feature_stddevs: Vec<f32>,   // stddev of each feature
    sample_count: u32,
}

score(genome) → f32:
    features = extract_features(genome)
    distance = sum((f - mean)² / max(stddev², epsilon) for each feature)
    return distance  // lower = more "tasteful"
```

Good genomes = voted score > 0 + imported flames.
Rebuild model whenever votes change or on startup.

**Phase 2: Feature reduction (later, when we have data)**

Once we have 50+ voted genomes:
- Run PCA on the feature matrix
- Keep top N components that explain 90% of variance
- This tells us which features actually matter for taste
- Might discover: "hue spread and variation-color pairing explain 80%
  of your preferences, brightness doesn't matter"

**Phase 3: Discriminative model (much later)**

With 100+ votes, switch to a model that uses both positive and negative examples:
- Logistic regression or small SVM on reduced features
- Positive: upvoted + imported
- Negative: heavily downvoted (score <= -3, strong signal)
- Mildly downvoted genomes excluded from training (ambiguous signal)

### Palette Generation with Taste

```
generate_tasteful_palette(model, recent_palettes):
    candidates = []
    for _ in 0..20:  // generate 20 candidate palettes
        palette = generate_random_palette()
        score = model.score(palette_features(palette))
        candidates.push((score, palette))

    // Diversity nudge: penalize palettes similar to recent ones
    for (score, palette) in &mut candidates:
        for recent in recent_palettes.last(5):
            similarity = hue_histogram_overlap(palette, recent)
            *score += similarity * diversity_penalty  // higher = worse

    // Pick the best-scoring candidate
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0))
    return candidates[0].1
```

This is cheap: 20 random palettes, score each (~89 multiplies + additions),
pick the winner. Sub-millisecond even without optimization.

### Convergence Avoidance

1. **Recent palette memory**: track last 5 palette hue histograms,
   penalize new palettes that overlap too much
2. **Exploration rate**: 10% of the time, ignore the taste model and
   use a fully random palette (configurable)
3. **Feature drift**: slowly decay old votes' influence so the model
   adapts as your taste evolves
4. **Minimum variance**: if taste model stddev for any feature drops
   below a floor, widen it to prevent collapse

### Config

```json
"taste_engine_enabled": false,
"taste_engine_min_votes": 10,
"taste_engine_strength": 0.5,
"taste_engine_exploration_rate": 0.1,
"taste_engine_diversity_penalty": 0.3,
"taste_engine_candidates": 20,
"taste_engine_recent_memory": 5
```

## Implementation Order

### Phase 1: Breeding mechanics (no taste engine)
1. Add `parent_a`, `parent_b`, `generation` to FlameGenome
2. Build LineageCache for genetic distance computation
3. Implement parent selection with inbreeding avoidance
4. Rework crossover to build child from scratch (not clone)
5. Fresh random palette every time (placeholder until taste engine)

### Phase 2: Taste engine — color only
6. Feature extraction from palettes (17 palette-level features)
7. Build Gaussian taste model from voted + imported genomes
8. Generate palettes using taste model + diversity nudge
9. Add exploration rate and recent palette memory

### Phase 3: Taste engine — transforms + pairings
10. Add transform-color features (52)
11. Add pairing features (20)
12. Feature reduction via PCA when sample size is sufficient

### Phase 4: Discriminative model
13. Switch from Gaussian centroid to logistic regression / SVM
14. Use negative votes as negative training examples
15. Online learning: update model incrementally as votes come in

## Not In Scope (yet)
- Network sharing of genomes or taste models
- GPU-accelerated taste model inference
- Automatic vote inference from viewing duration
- Cross-session taste model persistence (auto-handled via votes.json)
