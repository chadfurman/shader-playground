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

1. **Parent selection** -- Two parents drawn from the MAP-Elites archive
   (50% chance, uniform across occupied cells) or voted pool / history /
   seeds / imported flames (50% chance). Lineage cache enforces minimum
   genetic distance to prevent inbreeding.

2. **Breeding** -- Child transforms assembled from four sources:
   parent A, parent B, a community genome, and audio-biased random
   transforms. When parents share a variation type, **interpolative
   crossover** smoothly blends affine parameters instead of slot-swapping.
   A wildcard slot adds fresh genetic material.

3. **Mutation** -- Z-mutations fire first (5% probability: z-tilt or
   z-scale for 3D depth). Then one standard mutation operator fires
   (weighted: perturb 30%, final_xf 20%, swap_var 12%, symmetry 10%,
   rotate_colors 10%, shuffle 8%, globals 10%). High-symmetry genomes
   (4+ fold) force a final transform. The attractor is estimated on CPU;
   degenerate offspring are retried.

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

- `src/genome.rs` -- `FlameGenome`, `FlameTransform` (3x3 affine), `breed()`, `lerp_with()`, mutation operators (including z-tilt/z-scale)
- `src/archive.rs` -- `MapElitesArchive`, `GridCoords` (diversity-preserving parent selection)
- `src/votes.rs` -- `VoteLedger`, `LineageCache`, genetic distance
- `src/weights.rs` -- `RuntimeConfig` (all tunable breeding/mutation params)
- `src/main.rs` -- `pick_breeding_parents()`, `archive_genome()`, `morph_snapshot_or_current()`
