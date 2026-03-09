# Breeding

The `breed()` function in `src/genome.rs` combines two parent genomes and
a community genome to produce a child. The child is assembled from scratch
-- no direct cloning from either parent.

## Parent Selection

Parent selection happens in `pick_breeding_parents()` in `src/main.rs`.

### Priority chain

**Parent A** (prefer high-quality):
1. Vote-weighted pick from the voted pool (`pick_voted()`)
2. Random saved genome (any of voted/, history/, seeds/, flat genomes/)
3. Fallback: current on-screen genome

**Parent B** (prefer genetic diversity):
1. Random saved genome, checked against `min_breeding_distance`
2. Up to 10 attempts to find a sufficiently distant candidate
3. Fallback chain: imported flame -> seed -> generated default genome

**Community genome** (genetic diversity injection):
1. Random imported flame from `genomes/flames/`
2. Vote-weighted pick from voted pool
3. Random saved genome

### Config fields

| Config field | Default | Description |
|---|---|---|
| `parent_current_bias` | see weights.rs | Bias toward current on-screen genome |
| `parent_voted_bias` | see weights.rs | Bias toward voted pool |
| `parent_saved_bias` | see weights.rs | Bias toward saved genomes |
| `parent_random_bias` | see weights.rs | Bias toward random generation |
| `min_breeding_distance` | 3 | Minimum genetic distance between parents |

### Breeding distance

The `LineageCache` (in `src/votes.rs`) computes genetic distance as the
depth to the lowest common ancestor via BFS. If no common ancestor is
found within `max_lineage_depth`, the distance equals `max_lineage_depth`
(maximum diversity). Self-breeding (distance 0) is always rejected.

## Slot Allocation

The child's transform count is the average of both parents' transform
counts, +/-1, clamped to 3-6.

```
child_count = clamp(avg(parent_a.len, parent_b.len) + rand(-1..=1), 3, 6)
```

Slot indices are shuffled randomly, then assigned:

### Slot 0: Wildcard

The first shuffled slot gets a fresh transform:
- When `taste_engine_enabled`: uses `generate_biased_transform()` from
  the taste engine (learned user preferences)
- Otherwise: `FlameTransform::random_transform()`

### Remaining slots: four groups

The remaining slots are split into 4 roughly equal groups (leftover slots
distributed to earlier groups):

| Group | Source | Description |
|---|---|---|
| **A** | Parent A | Random transform copied from parent A |
| **B** | Parent B | Random transform copied from parent B |
| **Community** | Community genome | Random transform from voted/imported genome (keeps random placeholder if no community genome available) |
| **Environment** | Audio-biased random | Fresh `random_transform()` with variation chosen by `audio_biased_variation_pick()` |

For a typical 5-transform child: 1 wildcard + 1 from A + 1 from B +
1 community + 1 environment.

### Audio-biased variation picking

The environment group picks variations influenced by the current audio
analysis:

| Audio character | Favored variations |
|---|---|
| Bass-dominant | spherical, bubble, blob, fisheye |
| Mids-dominant | sinusoidal, waves, cosine, handkerchief |
| Highs-dominant | julia, disc, cross, tangent, diamond |
| Beat-dense | spiral, swirl, polar |

40% chance to pick from the dominant audio group, 30% chance for beat
group if `beat_accum > 0.5`, otherwise falls back to orby-biased random
(20% chance of spherical/julia/disc/bubble/fisheye/eyefish/blob).

## Post-Breeding Normalization

After assembling the child transforms, three normalization passes run
(following Electric Sheep conventions):

1. **`normalize_variations()`** -- For each transform: keep at most the 2
   strongest variation weights, zero out the rest, then scale the
   remaining weights to sum to 1.0.

2. **`normalize_weights()`** -- Scale all transform selection weights
   (`weight` field) to sum to 1.0 across all transforms.

3. **`distribute_colors()`** -- Evenly space palette indices across
   transforms: transform *i* gets `color = i / n`.

## Palette Generation

- When `taste_engine_enabled` and taste engine is active: generates a
  palette using `TasteEngine::generate_palette()` with learned preferences
- Otherwise: `generate_random_palette()` produces a cosine-gradient
  palette with a random "mood" (dark, rich, pastel, or high-contrast)

## Other Inherited Fields

| Field | Inheritance rule |
|---|---|
| `symmetry` | Random pick from either parent |
| `global` (speed, zoom, trail, brightness) | Linear blend with random factor (0.0-1.0) |
| `kifs` | Copied from parent A |
| `final_transform` | Random pick from either parent |
| `generation` | `max(parent_a.generation, parent_b.generation) + 1` |
| `name` | Generated as `"child-{random 4-digit}"` |
