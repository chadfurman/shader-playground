# Persistence

Genome storage, vote tracking, lineage, and archiving. Source lives in
`src/votes.rs` and `src/main.rs`.

## Directory Structure

```
genomes/
  voted/         <- upvoted genomes (breeding prefers these)
  history/       <- every genome ever generated
  seeds/         <- hand-curated seed genomes
  flames/        <- imported .flame XML files (converted to FlameGenome)
  votes.json     <- vote ledger (scores + file paths)
  lineage.json   <- ancestry tree (parent links + generation)
```

On startup, `voted/` and `history/` are created if missing via
`create_dir_all`.

## Save Flow

Every genome produced by auto-evolve or breeding is saved:

1. **Auto-save to history** -- When a genome is first voted on and has no
   existing entry in `votes.json`, it is saved to `history/` as
   `{name}.json`.

2. **Promotion to voted** -- When a genome's score crosses above 0
   (positive), it is additionally copied to `voted/`. The `votes.json`
   file path is updated to point to the voted copy.

3. **Downvoted genomes** -- Stay only in `history/`. They are excluded
   from `pick_voted()` (requires score > 0) and from `pick_random_saved()`
   (excludes score < 0).

## votes.json

The `VoteLedger` struct serializes as a flat JSON object:

```json
{
  "child-4821": {
    "score": 2,
    "file": "genomes/voted/child-4821.json",
    "last_seen": "2026-03-08"
  },
  "mutant-7193": {
    "score": -1,
    "file": "genomes/history/mutant-7193.json",
    "last_seen": "2026-03-07"
  }
}
```

### Fields

| Field | Type | Description |
|---|---|---|
| `score` | `i32` | Net vote count (upvotes minus downvotes) |
| `file` | `String` | Path to the genome JSON file |
| `last_seen` | `String` | Date of last vote (`YYYY-MM-DD` format) |

### Vote-weighted selection

`pick_voted()` selects from genomes with positive scores using
score-weighted random selection. A genome with score 3 is 3x more likely
to be picked than one with score 1.

### Random saved selection

`pick_random_saved()` scans voted/, history/, seeds/, and the flat
genomes directory. All qualifying genomes have equal probability
(unweighted). Genomes with negative vote scores are excluded.

## lineage.json

Tracks ancestry for genetic distance computation:

```json
{
  "child-4821": {
    "parent_a": "mutant-0567",
    "parent_b": "seed-8901",
    "generation": 3,
    "created": "2026-03-08"
  }
}
```

### Fields

| Field | Type | Description |
|---|---|---|
| `parent_a` | `Option<String>` | First parent's name |
| `parent_b` | `Option<String>` | Second parent's name |
| `generation` | `u32` | Breeding depth from original seed |
| `created` | `String` | Creation date |

### Behavior

- **Append-only** -- New entries are added after each breed via
  `register_and_save()`. Existing entries are never modified.
- **Grows ~100 bytes per genome** -- One JSON entry per genome ever bred.
- **Survives archiving** -- Lineage entries persist even after the
  corresponding genome JSON files are deleted from `history/`.
- **Migration fallback** -- If `lineage.json` doesn't exist on startup,
  the `LineageCache` is built by scanning genome JSON files in genomes/,
  seeds/, and flames/.

### Genetic distance

`genetic_distance()` computes the depth to the lowest common ancestor
using BFS from both genomes simultaneously. Used by
`pick_breeding_parents()` to enforce `min_breeding_distance`:

| Relationship | Distance |
|---|---|
| Self | 0 |
| Parent/child | 1 |
| Siblings (same parents) | 1 |
| Grandparent/grandchild | 2 |
| Unrelated (no common ancestor within max_depth) | max_depth |

## Archiving

When `history/` grows too large, old genomes are pruned.

### Trigger

Runs on startup when `archive_on_startup` is `true` (default) and the
total size of `history/*.json` exceeds `archive_threshold_mb` (default
100 MB).

### Process

1. Scan all JSON files in `history/`, read each genome's `generation`
2. Sort by generation, find the median generation number
3. Delete all genomes with generation below the median
4. `lineage.json` is not affected -- ancestry is preserved

### Config

| Config field | Default | Description |
|---|---|---|
| `archive_threshold_mb` | 100 | Size threshold in MB before archiving triggers |
| `archive_on_startup` | true | Whether to check and archive on app startup |

### What is never archived

- `voted/` -- Upvoted genomes are never deleted by archiving
- `seeds/` -- Hand-curated seeds are never touched
- `flames/` -- Imported flame files are never touched
- `lineage.json` -- Ancestry data persists indefinitely
- `votes.json` -- Vote data persists indefinitely
