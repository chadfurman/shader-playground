# Taste Model — Design (Future)

## Goal

Learn aesthetic preferences from voting history to bias mutation toward features that produce genomes the user likes — not just "which genome" but "what makes a genome good."

## Prerequisites

- 50+ votes in the voting system to have meaningful signal
- The existing crossover/voting system working well as baseline

## Core Idea

Decompose every voted genome into a feature vector. Build a statistical profile of what upvoted genomes have in common vs downvoted ones. Use that profile to bias mutation parameters toward "good" ranges.

## Feature Extraction

Each genome gets decomposed into measurable features:

**Transform-level:**
- Variation type presence (26 booleans — which variations are active)
- Variation weight distribution (how dominant is the primary variation)
- Affine coefficient ranges (a, b, c, d, offset_x, offset_y)
- Transform count

**Genome-level:**
- Symmetry type (1, 2, 3, etc.)
- Palette hue spread (how many distinct hues)
- Palette brightness (average luminance)
- Palette warmth (warm vs cool dominant colors)
- Variation diversity (how many distinct variation types across all transforms)

**Combination features (the hard ones):**
- Variation pairings — which variations co-occur across transforms
  e.g. "spherical in xf0 + swirl in xf1" as a pair feature
- Affine contrast — how different are the transforms from each other
- Color spread — how far apart are transform color indices

## Architecture

### TasteProfile struct

```
TasteProfile {
    // Per-feature: mean and stddev from upvoted genomes
    feature_means: HashMap<String, f32>,
    feature_stddevs: HashMap<String, f32>,

    // Variation pairing scores: (var_a, var_b) -> frequency in upvoted
    variation_pairs: HashMap<(usize, usize), f32>,

    // Parameter ranges that correlate with upvotes
    param_ranges: HashMap<String, (f32, f32)>,  // (preferred_min, preferred_max)

    // Total votes analyzed
    sample_count: u32,
}
```

### How It's Built

1. On startup (and periodically), scan `votes.json` + load all voted genomes
2. Split into upvoted (score > 0) and downvoted (score < 0) sets
3. Extract features from both sets
4. For each feature: compute mean/stddev in upvoted set
5. For variation pairings: count co-occurrences in upvoted genomes, normalize
6. Save as `taste_profile.json` alongside `votes.json`

### How It's Used

Three integration points, from lightest to heaviest touch:

**1. Bias random seed generation (lightest)**
When generating the "fresh random seed" (20% of crossover sources), instead of uniform random:
- Sample transform count from upvoted distribution
- Pick variation types weighted by upvoted frequency
- Sample affine coefficients from upvoted ranges
- Pick palette characteristics from upvoted profiles

This is basically "random but in the neighborhood of what you like."

**2. Bias mutation parameters**
When `mutate_inner` perturbs values, constrain perturbation toward preferred ranges:
- If preferred affine range is [-0.8, 0.8], bias perturbation toward that range
- If preferred variation pairings exist, bias `mutate_swap_variations` toward those combos
- Soft bias, not hard clamp — still allow exploration

**3. Post-mutation fitness scoring (heaviest)**
After generating a mutation candidate, score it against the taste profile:
- Compute feature distance from upvoted centroid
- If score is very low (very far from preferred), regenerate with higher probability
- Like the existing attractor extent check but for aesthetic fitness

## Risks

**Convergence / boring-ification** — the biggest risk. If the model learns too well, everything looks the same. Mitigations:
- Keep the 20% pure random source in crossover (non-negotiable diversity)
- Taste bias should be soft (gaussian preference, not hard filter)
- Add a "novelty bonus" — occasionally prefer genomes that are far from the profile
- Decay old votes over time so preferences can evolve

**Small sample overfitting** — with 10 votes, any "pattern" is noise. Mitigations:
- Don't activate taste model until N votes threshold (50?)
- Use wide stddevs (conservative confidence) with small samples
- Laplace smoothing on variation pair counts

**Correlation vs causation** — "you upvoted 3 green genomes" doesn't mean you like green. Maybe you liked the structure and green was coincidental. Mitigations:
- Weight structural features (variations, affine, symmetry) higher than surface features (color)
- Need enough votes that real patterns emerge above noise

## Implementation Phases

**Phase 1: Feature extraction only (observational)**
- Extract features from all voted genomes
- Log the profile to terminal so user can see what the system thinks they like
- No mutation changes yet — just observe and report

**Phase 2: Bias random seeds**
- Use taste profile to generate better random seeds
- This is the safest integration — only affects 20% of crossover sources
- Easy to A/B test (disable with config flag)

**Phase 3: Mutation parameter biasing**
- Soft-constrain perturbation toward preferred ranges
- Only activate with sufficient vote corpus

**Phase 4: Post-mutation fitness scoring**
- Score candidates against profile, regenerate outliers
- Most aggressive — save for when model is well-calibrated

## Config Fields (weights.json)

```json
"taste_model_enabled": false,
"taste_model_min_votes": 50,
"taste_model_strength": 0.3,
"taste_model_novelty_bonus": 0.1
```

## Not In Scope

- Neural network / ML inference (too heavy for a real-time visualizer)
- Cross-user taste sharing (local only for now)
- Automatic vote inference from screen time (interesting idea but separate feature)
