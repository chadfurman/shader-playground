# Local Voting System — Design

## Goal

Electric Sheep-style local voting system where user votes shape which genomes survive, reproduce, and get screen time.

## Architecture

Votes influence evolution through **per-transform crossover** — when a mutation fires, each transform in the child genome independently picks its source from four weighted pools. Votes also control screen time (how long each genome is displayed before morphing).

## Key Concepts

### Four Parent Sources (per-transform crossover)

When mutation triggers, each transform independently rolls which source it comes from:

| Source | Default Weight | Purpose |
|--------|---------------|---------|
| Current genome | 30% | Continuity — "more of this" |
| Vote-weighted pool | 25% | Evolutionary pressure from explicit votes |
| Random from all saved | 25% | Unweighted pick from imported + voted + saved favorites — serendipity |
| Fresh random seed | 20% | Wild card diversity, prevents inbreeding |

All four weights configurable in `weights.json` `_config` as `parent_current_bias`, `parent_voted_bias`, `parent_saved_bias`, `parent_random_bias`.

### Voting Mechanics

- **Up arrow** = upvote current genome (+1 score)
- **Down arrow** = downvote current genome (-1 score)
- Genome auto-saved to `genomes/` on first vote if not already persisted
- Score of -2 or below = blacklisted (never shown, never used as parent)
- Unvoted genomes participate in "all saved" pool with neutral weight

### Screen Time Weighting

- After morph completes, next genome picked from pool weighted by `max(score + 1, 0)`
- Higher-scored genomes shown more often
- Blacklisted genomes (score <= -2) never picked
- If no voted genomes exist, falls back to current random mutation behavior

### Storage

- `genomes/` — individual genome JSON files (already exists for favorites)
- `votes.json` — vote ledger: `{ "genome_hash": { "score": 3, "file": "genomes/abc123.json", "last_seen": "2026-03-08" } }`
- Hot-reloaded alongside `weights.json` so scores can be hand-edited

### Data Flow

1. User presses up/down arrow -> genome auto-saved if needed -> score updated in `votes.json`
2. Mutation trigger -> for each transform, roll source (30/25/25/20) -> crossover child genome
3. Morph complete -> pick next genome from score-weighted pool
4. `FavoriteProfile` remains orthogonal — still biases variation type selection from saved genomes

### Edge Cases

- First run with no votes: identical to current system
- All genomes blacklisted: 100% fresh random seed
- Single genome with high score: gets picked often but 20% random seeds keep diversity
- Empty `genomes/` directory: falls back to current random mutation

### Future Extension (not in scope)

- Per-parameter crossover: individual params (color, weight) could also pull from random pool members
- Network voting: share votes/genomes across machines
