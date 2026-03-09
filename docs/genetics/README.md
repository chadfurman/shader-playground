# Genetics

The evolution system breeds, mutates, and selects flame fractals using an
interactive genetic algorithm. Users vote on rendered genomes; upvoted genomes
enter the breeding pool and produce offspring that inherit traits from both
parents plus environmental noise.

## Evolution Cycle

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Parents  │───>│ Breeding │───>│ Mutation  │───>│ Offspring│
│ (voted   │    │ (slot    │    │ (affine,  │    │ (render  │
│  pool)   │    │  crossvr)│    │  var, pal)│    │  + vote) │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     ^                                               │
     └───────────── upvote ──────────────────────────┘
```

1. **Parent selection** -- Two parents are drawn from the voted pool,
   history, seeds, or imported flames. A lineage cache enforces minimum
   genetic distance to prevent inbreeding.

2. **Breeding** -- Child transforms are assembled from four sources:
   parent A, parent B, a community genome, and audio-biased random
   transforms. A wildcard slot adds fresh genetic material.

3. **Mutation** -- One mutation operator fires per cycle (affine
   perturbation, variation swap, color rotation, symmetry nudge, etc.).
   The attractor is estimated on CPU; degenerate offspring are retried.

4. **Rendering + voting** -- The child is rendered via the chaos-game
   compute shader. Users upvote or downvote. Positive-score genomes
   enter the voted pool for future breeding.

## When to Read

| Topic | File | What it covers |
|---|---|---|
| Genome data model | [genome-format.md](genome-format.md) | `FlameGenome`, `FlameTransform`, palette |
| Crossover mechanics | [breeding.md](breeding.md) | `breed()`, parent selection, slot allocation |
| Mutation operators | [mutation.md](mutation.md) | `mutate_inner()`, normalization, attractor estimation |
| Storage and voting | [persistence.md](persistence.md) | Directory layout, `votes.json`, `lineage.json`, archiving |

## Key Source Files

- `src/genome.rs` -- `FlameGenome`, `FlameTransform`, `breed()`, mutation operators
- `src/votes.rs` -- `VoteLedger`, `LineageCache`, genetic distance
- `src/weights.rs` -- `RuntimeConfig` (all tunable breeding/mutation params)
- `src/main.rs` -- `pick_breeding_parents()`, `archive_history_if_needed()`
