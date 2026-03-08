# Testing + Quality Gates — Design

## Goal

Add unit test coverage to all logic modules and gate commits behind formatting, linting, and test checks.

## Pre-commit Hook

A git pre-commit hook (`.git/hooks/pre-commit`) runs three checks, failing fast:

1. `cargo fmt --check` — formatting
2. `cargo clippy -- -D warnings` — lint warnings are errors
3. `cargo test` — all tests pass

Commit is rejected if any check fails.

## Test Strategy

- Tests live as `#[cfg(test)] mod tests` inside each source file
- No external test framework — standard `#[test]` + `assert!`/`assert_eq!`
- All test data is inline (no file dependencies)
- GPU-dependent code (main.rs, device_picker.rs, sck_audio.rs) is out of scope

## Test Modules

### taste.rs (~10-12 tests)
- `rgb_to_hsv` correctness (red, green, blue, white, black, gray)
- `PaletteFeatures::extract` on a known palette
- Hue cluster counting edge cases (wrap-around, single cluster, empty)
- `TasteModel::build` computes correct means/stddevs
- `TasteModel::score` — identical features score 0, distant features score high
- `generate_palette` returns 256 entries, falls back to random when model inactive
- Hue overlap: identical palettes = 1.0, disjoint = 0.0

### votes.rs (~8-10 tests)
- `LineageCache::genetic_distance` — self = 0, parent = 1, grandparent = 2
- Distance with no common ancestor = max_depth
- Distance with shared parent = 1
- `register` adds entries correctly
- `VoteLedger::vote` increments/decrements scores
- `pick_voted` respects score weighting (only positive scores)

### genome.rs (~10-12 tests)
- `breed` produces correct transform count (avg of parents +/-1, clamped 3-6)
- `breed` records lineage (parent_a, parent_b, generation)
- `breed` always produces a palette
- `normalize_weights` sums to 1.0
- `normalize_variations` — at least one variation nonzero per transform
- `distribute_colors` — evenly spaced 0..1
- `estimate_attractor_extent` — degenerate genome has small extent
- `random_transform` produces valid transform
- `mutate_inner` produces a different genome than input

### weights.rs (~4-5 tests)
- Default RuntimeConfig has sane values
- Serialization roundtrip (serialize then deserialize = equal)
- Missing fields use defaults (parse partial JSON)

### flam3.rs (~3-4 tests)
- Parse minimal flam3 XML string into FlameGenome
- Transforms have correct variation weights
- Palette parsing from hex colors
