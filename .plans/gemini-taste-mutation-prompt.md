# Deep Research: Improving Genetic Algorithm Aesthetics for Fractal Flame Evolution

## Context

I'm building a real-time fractal flame renderer with an interactive genetic algorithm — users vote on renders they like (thumbs up/down), and the system breeds and mutates genomes to produce more visually interesting results over time. The core rendering works well, but the genetic outputs are too homogeneous. I need help making the evolution produce more diverse, visually rich, and aesthetically interesting results.

## Current System

### Genome Format (FlameGenome)
Each genome contains:
- **3-6 transforms**, each with: 2x2 affine matrix (a,b,c,d + offset), blend weight, palette color index, and weights for 14 parametric variations (sinusoidal, spherical, swirl, horseshoe, handkerchief, julia, polar, disc, rings, bubble, fisheye, exponential, spiral, diamond)
- **Optional final transform** (post-processing transform applied after iteration)
- **256-entry color palette** (RGB)
- **Global params**: flame_brightness, zoom, symmetry mode (rotational or bilateral, 1-6 fold)

### Taste Engine (Gaussian Centroid Model)
When users upvote a genome, I extract 30 features:
- **PaletteFeatures (17D)**: hue histogram (12 bins, circular-aware), mean/stddev saturation, mean/stddev brightness
- **TransformFeatures (8D)**: per-transform scale/shear metrics, weight distribution
- **CompositionFeatures (5D)**: transform count, symmetry mode, final_transform weight, flame_brightness, zoom

The model computes feature-wise mean and standard deviation across all upvoted genomes. Scoring: `sum((feature_i - mean_i)^2 / stddev_i^2)` — lower = closer to user preference centroid.

**`generate_biased_transform()`**: Generates N=20 random candidate transforms, scores each against the taste model, returns the lowest-scoring (closest to learned preference). Used during breeding to fill transform slots.

### Breeding Algorithm
1. Two parents + optional community genome + optional environment genome
2. Child gets avg(parent_count) ± 1 transforms, clamped 3-6
3. Slots allocated proportionally to 4 groups (parent A, parent B, community, environment)
4. Each slot filled with taste-biased transform if taste engine is ready, else random
5. Groups A/B/C/E overwrite their slots with actual parent transforms
6. Post-processing: normalize_variations (max 2 active per transform), normalize_weights (sum to 1.0), distribute_colors (evenly space palette indices)

### Mutation (8 types, weighted probabilities)
1. **Perturb** (40%): rotate ±0.8 rad, scale 0.6-1.5x (det clamped 0.2-0.95), shear ±0.4, anisotropic scale
2. **Swap variations** (12%): swap two variation weights within a transform
3. **Rotate colors** (12%): shift all color indices by ±0.2
4. **Shuffle transforms** (12%): swap two transforms' positions
5. **Global params** (12%): adjust brightness ±0.05, zoom ±0.5
6. **Final transform** (6%): create/perturb/reinvent post-processing transform
7. **Symmetry** (3%): nudge symmetry ±1
8. **Add/remove transform** (conditional): add when n<3 (25%), remove when n>5 (20%)

### Variation Selection Biasing
- **Audio-biased**: dominant audio band maps to variation groups (bass→spherical/bubble, mids→sinusoidal/waves, highs→julia/disc)
- **Fitness-biased**: saved genome variation frequency profile blended with uniform distribution
- **Default fallback**: 20% bias toward "orby" variations (spherical, julia, disc, bubble, fisheye)

### Voting System
- Users press up/down arrows during rendering
- Scores accumulate in votes.json, hot-reloaded
- Positive-score genomes go to curated pool for breeding
- Negative-score genomes excluded from selection
- Lineage tracking via BFS genetic distance for diversity

## The Problems

### Problem 1: Empty Circles in the Center
Occasionally, a genome produces a visible circular void in the center of the render. This happens when:
- Affine determinant collapses toward zero (det → 0 means the transform maps to a line/point)
- The `mutate_perturb()` path clamps determinant to [0.2, 0.95], but other mutation paths (add_transform, final_transform creation) don't enforce this consistently
- When determinant is very small, the IFS attractor collapses and can't fill space around the origin

### Problem 2: Kaleidoscope Monotony
Most outputs look like kaleidoscopes — radially symmetric, repetitive, flat-feeling. They lack the dimensional depth and organic complexity that the best fractal flame art achieves. Root causes I've identified:
- **Variation homogeneity**: `normalize_variations()` limits each transform to max 2 variations, but doesn't enforce diversity *across* transforms. If 5 transforms all use julia+disc, you get julia-disc repeated 5 times
- **High symmetry amplifies repetition**: symmetry_mode = 6 with homogeneous transforms = pure kaleidoscope
- **Low final transform usage**: Only 6% mutation probability means most genomes never develop a final transform, which is the primary mechanism for breaking radial symmetry
- **Scale uniformity**: No mechanism ensures transforms operate at different scales — you can get 5 transforms all at ~0.5 scale, producing self-similar blobs
- **Taste model reinforcement loop**: If users upvote kaleidoscopes early (because that's what the system produces), the taste model learns to prefer kaleidoscopes, further narrowing diversity

### Problem 3: Lack of "Depth"
The renders feel 2D — flat patterns rather than the rich, volumetric, multi-layered compositions that characterize great fractal flame art (think Electric Sheep, Chaotica galleries). The chaos game *should* produce depth through overlapping attractors at different scales, but our breeding/mutation isn't producing the right transform combinations.

## What I Want to Understand

### 1. Genetic Algorithm Design for Aesthetic Evolution
- What does the academic literature (2020-2025) say about evolving aesthetic artifacts? Interactive evolutionary computation (IEC) best practices?
- How do successful aesthetic evolution systems (Electric Sheep, Picbreeder, EndlessForms) handle the exploration/exploitation tradeoff?
- Is there a better fitness model than Gaussian centroid scoring? Multi-objective? Novelty search? Quality-diversity (MAP-Elites)?
- How do you prevent the taste model from collapsing to a narrow preference region?

### 2. Fractal Flame Diversity Strategies
- What techniques do Apophysis, JWildfire, and Chaotica use to generate diverse, non-kaleidoscopic flames?
- Role of the final transform — how should it be designed and evolved to add visual complexity?
- Transform scale distribution — is there a principled approach to ensuring multi-scale composition?
- How important is variation diversity across transforms vs. within transforms?
- What makes the difference between a "flat kaleidoscope" and a flame with perceived depth and dimension?

### 3. Mutation Strategy Optimization
- Is our mutation probability distribution (40% perturb, 12/12/12/12, 6/3) well-balanced? What does the literature suggest?
- Should mutations be context-aware (e.g., if all transforms have similar scale, bias toward scale-divergent mutations)?
- Adaptive mutation rates based on population diversity or stagnation detection?
- Specific mutation operators that are known to produce visually interesting transitions in IFS systems?

### 4. Breeding / Crossover Improvements
- Is our slot-based crossover effective? Alternatives from GP/IEC literature?
- How should community/environment genomes be selected to maximize offspring diversity?
- Role of genetic distance in parent selection — should we enforce minimum distance between parents?
- Multi-parent crossover beyond 2 parents?

### 5. Feature Engineering for the Taste Model
- Are our 30 features (17 palette + 8 transform + 5 composition) the right ones for capturing aesthetic preference?
- What features correlate with "visual complexity" vs. "visual monotony"?
- Should we extract features from the rendered image rather than (or in addition to) the genome parameters?
- Perceptual features: fractal dimension, edge density, symmetry measures, color harmony metrics?

### 6. Novelty and Diversity Maintenance
- Novelty search: should we reward genomes that are *different* from what we've seen, not just close to the preference centroid?
- Quality-diversity algorithms (MAP-Elites, CVT-MAP-Elites) — how would these apply to fractal flame evolution?
- Niching strategies to maintain distinct genome species?
- Archive management: when to retire old "good" genomes to prevent taste model stagnation?

### 7. The "Depth" Problem Specifically
- From a fractal geometry perspective, what transform properties create the perception of depth and dimensionality?
- Role of overlapping attractors at different scales
- How does the log-density tonemapping interact with perceived depth?
- Color gradients across iteration depth as a depth cue
- Variation combinations that tend to produce volumetric vs. flat results

## Constraints
- Interactive system — users vote in real-time while watching renders evolve
- Taste model must update incrementally (can't retrain from scratch each time)
- Breeding/mutation must be fast (runs on CPU, <50ms per genome)
- The system runs for hours at a time; solutions must handle long sessions without quality degradation
- Need to balance "giving users what they want" with "showing users things they didn't know they'd like"

## Desired Output
I want a research-backed strategy for:
1. **Immediate fixes**: Concrete changes to mutation/breeding that will reduce kaleidoscope monotony and empty-circle artifacts
2. **Taste model evolution**: How to improve the scoring model to capture richer aesthetic preferences without collapsing to a narrow region
3. **Diversity mechanisms**: Specific algorithms or techniques to maintain population diversity over long sessions
4. **Feature engineering**: Better features for the taste model, especially perceptual/rendered-image features
5. **Literature pointers**: Key papers and systems to study, especially from the IEC, computational aesthetics, and fractal art communities (2020-2025 preferred)

Priority order: what changes will have the most visible impact on output diversity and quality?
