# Mutation

After breeding, the `mutate()` function in `src/genome.rs` applies one
mutation operator to the child genome. It retries up to
`max_mutation_retries` times if the resulting attractor is degenerate
(extent below `min_attractor_extent`).

## Mutation Operators

Each call to `mutate_inner()` picks one of 8 operators at random:

| Weight | Operator | Description |
|---|---|---|
| 2/8 | `mutate_perturb` | Affine perturbation (most common -- see below) |
| 1/8 | `mutate_swap_variations` | Swap two variation weights within a random transform |
| 1/8 | `mutate_rotate_colors` | Shift all transform palette indices by a random amount |
| 1/8 | `mutate_shuffle_transforms` | Swap two transforms in the list |
| 1/8 | `mutate_global_params` | Perturb brightness and zoom |
| 1/8 | `mutate_final_transform` | Perturb or create a final transform |
| 1/8 | `mutate_symmetry` | Nudge symmetry order by +/-1 or flip rotational/bilateral |

### Affine Perturbation (`mutate_perturb`)

Picks one random transform, then applies one of 8 sub-operations:

| Sub-op | Description |
|---|---|
| Rotate | Random rotation angle (-0.8 to 0.8 radians) |
| Scale | Random scale factor (0.6 to 1.5), determinant clamped |
| Shear | Random shear (-0.4 to 0.4) |
| Anisotropic scale | Independent x/y scale (0.5-1.5 each), determinant clamped |
| Position | Offset perturbation (+/-1.0 in each axis) |
| Weight | Weight multiplied by random factor (0.4-2.5), clamped to [0.05, 2.0] |
| Reinvent affine | Generate fresh rotation+scale from scratch, optional shear |
| Replace variation | Swap or replace variation weights (see below) |

### Variation Replacement

When the "replace variation" sub-op fires:
- 50% chance: replace the weakest active variation with a new one
- 50% chance: clear all variations and set 1-2 fresh ones
- Variation picks use `pick_variation()`: fitness-biased when a
  `FavoriteProfile` is available, otherwise audio-biased
- 30% chance to perturb parametric variation params (+/-0.2)

### Transform Count Adjustment

After the main mutation, the transform count may change:

| Current count | Add chance | Remove chance |
|---|---|---|
| < 3 | 25% | 0% |
| 3-5 (sweet spot) | 5% | 5% |
| > 5 | 2% | 20% |

Adding a transform: 30% clone-and-diverge, 70% fresh specialist with
contrasting geometry and unused variations. Removing: drops the
lowest-weight transform (minimum 2 transforms kept).

## Determinant Clamping

After affine scale or anisotropic scale mutations, the determinant
`det = sqrt(|ad - bc|)` is checked. If outside [0.2, 0.95], the affine
coefficients are rescaled to bring the determinant into [0.4, 0.85].
This prevents:
- **Collapsed transforms** (det near 0): points map to a line or point
- **Exploding transforms** (det > 1): points diverge instead of converge

## Normalization Functions

Three normalization passes run after every mutation (same as post-breeding):

### `normalize_variations()`
- Collects all active variations (weight > 0.01) per transform
- Keeps at most the 2 strongest, zeros the rest
- Scales remaining weights to sum to 1.0

### `normalize_weights()`
- Sums all transform `weight` fields
- Divides each by the sum so they total 1.0

### `distribute_colors()`
- Sets each transform's `color` to `i / n` (evenly spaced palette indices)

## Attractor Estimation

`estimate_attractor_extent()` runs a 500-iteration CPU chaos game to
estimate the visible size of the attractor:

1. Start at point (0.5, 0.3)
2. For each iteration, select a transform weighted by `weight`
3. Apply the affine + variations (simplified CPU versions)
4. After 100 warmup iterations, collect 400 sample points
5. Sort x and y coordinates, take the 5th-95th percentile range
6. Return `max(extent_x, extent_y)`

### Auto-zoom

`auto_zoom()` uses the estimated extent to set the camera:
```
zoom = zoom_target / max(extent, 0.5)
```
Clamped to `[zoom_min, zoom_max]` from config.

If all mutation retry attempts produce degenerate attractors (extent <
`min_attractor_extent`, default 0.3), the system falls back to returning
a copy of parent A with a fresh name.

## Palette Mutation

After all other mutations, there is a 20% chance to regenerate the
palette entirely via `generate_random_palette()`. If the genome has no
palette (`None`), one is always generated.
