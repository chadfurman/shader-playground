# Local Voting System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an Electric Sheep-style local voting system where user votes shape genome evolution (parent selection via per-transform crossover) and screen time.

**Architecture:** A `VoteLedger` struct manages vote scores persisted to `votes.json`. Mutation parent selection uses per-transform crossover from 4 weighted sources: current genome (30%), vote-weighted pool (25%), random from all saved (25%), fresh random seed (20%). Screen time weighting picks the next genome from the vote-weighted pool after morph completes. Up/down arrow keys vote.

**Tech Stack:** Rust, serde_json, rand, winit keyboard events

---

### Task 1: Add voting config fields to RuntimeConfig

**Files:**
- Modify: `src/weights.rs:73-160` (RuntimeConfig struct + defaults)
- Modify: `weights.json:95-140` (_config section)

**Step 1: Add fields to RuntimeConfig**

In `src/weights.rs`, add these fields to the `RuntimeConfig` struct (after `temporal_reprojection`):

```rust
    #[serde(default = "default_parent_current_bias")]
    pub parent_current_bias: f32,
    #[serde(default = "default_parent_voted_bias")]
    pub parent_voted_bias: f32,
    #[serde(default = "default_parent_saved_bias")]
    pub parent_saved_bias: f32,
    #[serde(default = "default_parent_random_bias")]
    pub parent_random_bias: f32,
    #[serde(default = "default_vote_blacklist_threshold")]
    pub vote_blacklist_threshold: i32,
```

**Step 2: Add default functions**

After the existing default functions (around line 196):

```rust
fn default_parent_current_bias() -> f32 { 0.30 }
fn default_parent_voted_bias() -> f32 { 0.25 }
fn default_parent_saved_bias() -> f32 { 0.25 }
fn default_parent_random_bias() -> f32 { 0.20 }
fn default_vote_blacklist_threshold() -> i32 { -2 }
```

**Step 3: Add to weights.json _config**

Add to the `_config` section:
```json
    "parent_current_bias": 0.30,
    "parent_voted_bias": 0.25,
    "parent_saved_bias": 0.25,
    "parent_random_bias": 0.20,
    "vote_blacklist_threshold": -2
```

And to `_config_doc`:
```json
    "parent_current_bias": "Probability each transform comes from current genome during crossover (default 0.30)",
    "parent_voted_bias": "Probability each transform comes from vote-weighted pool (default 0.25)",
    "parent_saved_bias": "Probability each transform comes from random saved genome (default 0.25)",
    "parent_random_bias": "Probability each transform comes from fresh random seed (default 0.20)",
    "vote_blacklist_threshold": "Score at or below which a genome is blacklisted (default -2)"
```

**Step 4: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`
Expected: `Finished` with no errors

**Step 5: Commit**

```bash
git add src/weights.rs weights.json
git commit -m "feat(voting): add parent selection bias config fields"
```

---

### Task 2: Create VoteLedger module

**Files:**
- Create: `src/votes.rs`
- Modify: `src/main.rs:15-20` (add `mod votes;`)

**Step 1: Create `src/votes.rs`**

```rust
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use rand::prelude::IndexedRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::genome::FlameGenome;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VoteEntry {
    pub score: i32,
    pub file: String,
    pub last_seen: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct VoteLedger {
    #[serde(flatten)]
    pub entries: HashMap<String, VoteEntry>,
}

impl VoteLedger {
    /// Load votes.json from the given directory. Returns empty ledger if missing.
    pub fn load(dir: &Path) -> Self {
        let path = dir.join("votes.json");
        match fs::read_to_string(&path) {
            Ok(json) => serde_json::from_str(&json).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Save votes.json to the given directory.
    pub fn save(&self, dir: &Path) -> Result<(), String> {
        fs::create_dir_all(dir).map_err(|e| format!("create dir: {e}"))?;
        let path = dir.join("votes.json");
        let json = serde_json::to_string_pretty(self).map_err(|e| format!("serialize: {e}"))?;
        fs::write(&path, json).map_err(|e| format!("write: {e}"))?;
        Ok(())
    }

    /// Record a vote for the given genome. Auto-saves the genome if not already persisted.
    /// Returns the new score.
    pub fn vote(&mut self, genome: &FlameGenome, delta: i32, genomes_dir: &Path) -> i32 {
        let key = genome.name.clone();

        // Auto-save genome if no entry exists yet
        if !self.entries.contains_key(&key) {
            let file_path = match genome.save(genomes_dir) {
                Ok(p) => p.display().to_string(),
                Err(e) => {
                    eprintln!("[vote] auto-save failed: {e}");
                    format!("{}/{}.json", genomes_dir.display(), genome.name)
                }
            };
            self.entries.insert(key.clone(), VoteEntry {
                score: 0,
                file: file_path,
                last_seen: today(),
            });
        }

        let entry = self.entries.get_mut(&key).unwrap();
        entry.score += delta;
        entry.last_seen = today();
        let score = entry.score;

        // Persist immediately
        if let Err(e) = self.save(genomes_dir) {
            eprintln!("[vote] save error: {e}");
        }

        score
    }

    /// Is this genome blacklisted (score at or below threshold)?
    pub fn is_blacklisted(&self, name: &str, threshold: i32) -> bool {
        self.entries.get(name).is_some_and(|e| e.score <= threshold)
    }

    /// Pick a genome from the vote-weighted pool.
    /// Weight = max(score + 1, 0), so score 0 = weight 1, score 3 = weight 4.
    /// Excludes blacklisted genomes.
    pub fn pick_voted(&self, threshold: i32) -> Option<PathBuf> {
        let eligible: Vec<_> = self.entries.iter()
            .filter(|(_, e)| e.score > threshold)
            .collect();
        if eligible.is_empty() {
            return None;
        }

        let total_weight: f32 = eligible.iter()
            .map(|(_, e)| (e.score + 1).max(0) as f32)
            .sum();
        if total_weight <= 0.0 {
            return None;
        }

        let mut rng = rand::rng();
        let mut roll = rng.random::<f32>() * total_weight;
        for (_, entry) in &eligible {
            let w = (entry.score + 1).max(0) as f32;
            roll -= w;
            if roll <= 0.0 {
                return Some(PathBuf::from(&entry.file));
            }
        }

        // Fallback to last eligible
        eligible.last().map(|(_, e)| PathBuf::from(&e.file))
    }

    /// Pick a random genome from ALL saved genomes (unweighted).
    /// Includes voted and unvoted, excludes blacklisted.
    pub fn pick_random_saved(genomes_dir: &Path, threshold: i32, ledger: &VoteLedger) -> Option<PathBuf> {
        let entries: Vec<_> = fs::read_dir(genomes_dir)
            .ok()?
            .filter_map(|e| e.ok())
            .filter(|e| {
                let path = e.path();
                // Only .json files directly in genomes/ (skip subdirectories and votes.json)
                path.is_file()
                    && path.extension().is_some_and(|ext| ext == "json")
                    && path.file_name().is_some_and(|n| n != "votes.json")
            })
            .filter(|e| {
                // Exclude blacklisted genomes by checking filename against ledger
                let stem = e.path().file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                !ledger.is_blacklisted(&stem, threshold)
            })
            .collect();

        if entries.is_empty() {
            return None;
        }

        let entry = entries.choose(&mut rand::rng())?;
        Some(entry.path())
    }
}

fn today() -> String {
    // Simple date string without external crate
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = secs / 86400;
    // Approximate — good enough for a "last seen" stamp
    let year = 1970 + days / 365;
    let day_of_year = days % 365;
    let month = day_of_year / 30 + 1;
    let day = day_of_year % 30 + 1;
    format!("{year}-{month:02}-{day:02}")
}
```

**Step 2: Register module in main.rs**

In `src/main.rs`, add after `mod weights;` (line 20):

```rust
mod votes;
```

And add the import after the existing use statements (around line 23):

```rust
use crate::votes::VoteLedger;
```

**Step 3: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`
Expected: `Finished` with no errors (warnings about unused imports are OK for now)

**Step 4: Commit**

```bash
git add src/votes.rs src/main.rs
git commit -m "feat(voting): add VoteLedger module with vote/pick/blacklist"
```

---

### Task 3: Wire up keyboard voting (up/down arrows)

**Files:**
- Modify: `src/main.rs:1377-1412` (App struct — add `vote_ledger` field)
- Modify: `src/main.rs:1419-1470` (App::new — initialize ledger)
- Modify: `src/main.rs:1681-1797` (keyboard input handler — add arrow key handling)

**Step 1: Add vote_ledger to App struct**

In the `App` struct (around line 1411, before the closing `}`), add:

```rust
    vote_ledger: VoteLedger,
```

**Step 2: Initialize in App::new**

In `App::new()` (around line 1432-1470), add initialization. After `let favorite_profile = Self::scan_favorite_profile();`:

```rust
        let vote_ledger = VoteLedger::load(&project_dir().join("genomes"));
```

And in the `Self { ... }` constructor, add:

```rust
            vote_ledger,
```

**Step 3: Add arrow key handlers**

In the keyboard match block (`src/main.rs`, around line 1689), add before the `Key::Character(ref c)` arm:

```rust
                Key::Named(NamedKey::ArrowUp) => {
                    let dir = project_dir().join("genomes");
                    let score = self.vote_ledger.vote(&self.genome, 1, &dir);
                    self.favorite_profile = Self::scan_favorite_profile();
                    eprintln!("[vote] {} → +1 (score: {})", self.genome.name, score);
                }
                Key::Named(NamedKey::ArrowDown) => {
                    let dir = project_dir().join("genomes");
                    let score = self.vote_ledger.vote(&self.genome, -1, &dir);
                    self.favorite_profile = Self::scan_favorite_profile();
                    eprintln!("[vote] {} → -1 (score: {})", self.genome.name, score);
                }
```

**Step 4: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`
Expected: `Finished` with no errors

**Step 5: Manual test**

Run: `cargo run`
- Press up arrow → should see `[vote] <name> → +1 (score: 1)` in terminal
- Press down arrow → should see `[vote] <name> → -1 (score: 0)` in terminal
- Check that `genomes/votes.json` was created with the genome entry

**Step 6: Commit**

```bash
git add src/main.rs
git commit -m "feat(voting): wire up/down arrow keys for genome voting"
```

---

### Task 4: Per-transform crossover in mutation

**Files:**
- Modify: `src/genome.rs:628-662` (mutate method — replace parent selection with crossover)
- Modify: `src/genome.rs:1-8` (add imports if needed)

**Step 1: Add crossover method to FlameGenome**

Add this method to `impl FlameGenome` (before the existing `mutate` method):

```rust
    /// Per-transform crossover: each transform independently picks its source
    /// from 4 weighted pools: current, voted, saved, random.
    pub fn crossover(
        &self,
        voted: &Option<FlameGenome>,
        saved: &Option<FlameGenome>,
        cfg: &crate::weights::RuntimeConfig,
    ) -> Self {
        let mut rng = rand::rng();
        let mut child = self.clone();

        // Normalize biases to sum to 1.0
        let total = cfg.parent_current_bias + cfg.parent_voted_bias
            + cfg.parent_saved_bias + cfg.parent_random_bias;
        let (b_current, b_voted, b_saved) = if total > 0.0 {
            (
                cfg.parent_current_bias / total,
                cfg.parent_voted_bias / total,
                cfg.parent_saved_bias / total,
            )
        } else {
            (1.0, 0.0, 0.0) // fallback: 100% current
        };

        // Determine max transform count across all sources
        let n_current = self.transforms.len();
        let n_voted = voted.as_ref().map_or(0, |g| g.transforms.len());
        let n_saved = saved.as_ref().map_or(0, |g| g.transforms.len());
        let max_xf = n_current.max(n_voted).max(n_saved).max(3);

        // Resize child to max_xf transforms
        while child.transforms.len() < max_xf {
            child.transforms.push(FlameTransform::random_transform(&mut rng));
        }
        child.transforms.truncate(max_xf);

        for i in 0..child.transforms.len() {
            let roll: f32 = rng.random();

            if roll < b_current {
                // Current genome
                if i < self.transforms.len() {
                    child.transforms[i] = self.transforms[i].clone();
                }
            } else if roll < b_current + b_voted {
                // Vote-weighted pool genome
                if let Some(ref g) = voted {
                    if i < g.transforms.len() {
                        child.transforms[i] = g.transforms[i].clone();
                    }
                }
            } else if roll < b_current + b_voted + b_saved {
                // Random saved genome
                if let Some(ref g) = saved {
                    if i < g.transforms.len() {
                        child.transforms[i] = g.transforms[i].clone();
                    }
                }
            }
            // else: keep the random transform already in child (fresh random seed)
        }

        // Take palette from highest-priority source that has one
        if rng.random::<f32>() < b_voted {
            if let Some(ref g) = voted {
                if g.palette.is_some() {
                    child.palette = g.palette.clone();
                }
            }
        } else if rng.random::<f32>() < b_saved {
            if let Some(ref g) = saved {
                if g.palette.is_some() {
                    child.palette = g.palette.clone();
                }
            }
        }

        child.name = format!("crossover-{}", rng.random_range(1000..9999u32));
        child
    }
```

**Step 2: Check if `random_transform` exists; if not, add a helper**

Search for an existing random transform generator. If none exists, add to `impl FlameTransform`:

```rust
    pub fn random_transform(rng: &mut impl Rng) -> Self {
        let mut xf = Self::default();
        xf.weight = rng.random::<f32>() * 0.5 + 0.1;
        xf.a = rng.random::<f32>() * 2.0 - 1.0;
        xf.b = rng.random::<f32>() * 2.0 - 1.0;
        xf.c = rng.random::<f32>() * 2.0 - 1.0;
        xf.d = rng.random::<f32>() * 2.0 - 1.0;
        xf.offset = [
            rng.random::<f32>() * 2.0 - 1.0,
            rng.random::<f32>() * 2.0 - 1.0,
        ];
        xf.color = rng.random::<f32>();
        // Set one random variation to 1.0
        let var_idx = rng.random_range(0..VARIATION_COUNT);
        xf.set_variation(var_idx, 1.0);
        xf.linear = if var_idx != 0 { 0.0 } else { 1.0 };
        xf
    }
```

**Step 3: Update mutate() signature to accept crossover sources**

Change the `mutate` method signature in `src/genome.rs:628`:

```rust
    pub fn mutate(
        &self,
        audio: &AudioFeatures,
        cfg: &crate::weights::RuntimeConfig,
        profile: &Option<FavoriteProfile>,
        voted_parent: &Option<FlameGenome>,
        saved_parent: &Option<FlameGenome>,
    ) -> Self {
```

Replace the seed-biased parent selection (lines 630-642) with crossover:

```rust
        let mut rng = rand::rng();

        // Per-transform crossover from multiple sources
        let base = self.crossover(voted_parent, saved_parent, cfg);
```

The rest of `mutate()` stays the same — it applies `mutate_inner()` on the crossover child, does attractor extent checks, etc.

**Step 4: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`
Expected: Compile errors in main.rs where `mutate()` is called (next task fixes these)

**Step 5: Commit**

```bash
git add src/genome.rs
git commit -m "feat(voting): add per-transform crossover to mutation system"
```

---

### Task 5: Wire crossover sources into App mutation calls

**Files:**
- Modify: `src/main.rs:1690-1700` (Space key — manual mutate)
- Modify: `src/main.rs:1874-1886` (auto-evolve)

**Step 1: Add helper method to App for picking crossover parents**

Add to `impl App` (after `scan_favorite_profile`):

```rust
    /// Pick voted and saved parent genomes for crossover mutation.
    fn pick_crossover_parents(&self) -> (Option<FlameGenome>, Option<FlameGenome>) {
        let genomes_dir = project_dir().join("genomes");
        let threshold = self.weights._config.vote_blacklist_threshold;

        let voted = self.vote_ledger.pick_voted(threshold)
            .and_then(|p| FlameGenome::load(&p).ok());

        let saved = VoteLedger::pick_random_saved(&genomes_dir, threshold, &self.vote_ledger)
            .and_then(|p| FlameGenome::load(&p).ok());

        (voted, saved)
    }
```

**Step 2: Update Space key handler (manual mutate)**

Change line ~1696 from:
```rust
self.genome = self.genome.mutate(&self.audio_features, &self.weights._config, &self.favorite_profile);
```
to:
```rust
let (voted, saved) = self.pick_crossover_parents();
self.genome = self.genome.mutate(&self.audio_features, &self.weights._config, &self.favorite_profile, &voted, &saved);
```

**Step 3: Update auto-evolve (same pattern)**

Change line ~1882 from:
```rust
self.genome = self.genome.mutate(&self.audio_features, &self.weights._config, &self.favorite_profile);
```
to:
```rust
let (voted, saved) = self.pick_crossover_parents();
self.genome = self.genome.mutate(&self.audio_features, &self.weights._config, &self.favorite_profile, &voted, &saved);
```

**Step 4: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`
Expected: `Finished` with no errors

**Step 5: Manual test**

Run: `cargo run`
- Vote up a genome (up arrow), then press Space to mutate
- Terminal should show `[mutate] starting from crossover-NNNN` or similar
- Without any votes, behavior should be same as before (crossover falls back to current genome)

**Step 6: Commit**

```bash
git add src/main.rs
git commit -m "feat(voting): wire crossover parents into mutation calls"
```

---

### Task 6: Screen time weighting — vote-weighted next genome selection

**Files:**
- Modify: `src/main.rs:1871-1887` (auto-evolve section — pick next from weighted pool)

**Step 1: Update auto-evolve to prefer voted genomes for screen time**

Replace the auto-evolve block (lines ~1874-1886) with weighted next-genome selection:

```rust
                    if !self.flame_locked && all_morphed {
                        let time_since_last = time - self.last_mutation_time;
                        let cooldown = self.weights._config.mutation_cooldown;
                        if time_since_last >= cooldown {
                            self.genome_history.push(self.genome.clone());
                            if self.genome_history.len() > 10 {
                                self.genome_history.remove(0);
                            }

                            // Screen time weighting: try loading a vote-weighted genome
                            // to display, then mutate from it
                            let threshold = self.weights._config.vote_blacklist_threshold;
                            let genomes_dir = project_dir().join("genomes");
                            let screen_pick = self.vote_ledger.pick_voted(threshold)
                                .and_then(|p| FlameGenome::load(&p).ok());

                            let base = if let Some(picked) = screen_pick {
                                eprintln!("[screen-time] showing voted genome: {}", picked.name);
                                picked
                            } else {
                                self.genome.clone()
                            };

                            let (voted, saved) = self.pick_crossover_parents();
                            self.genome = base.mutate(
                                &self.audio_features,
                                &self.weights._config,
                                &self.favorite_profile,
                                &voted,
                                &saved,
                            );
                            self.last_mutation_time = self.start.elapsed().as_secs_f32();
                            self.begin_morph();
                            eprintln!("[auto-evolve] → {}", self.genome.name);
                        }
                    }
```

**Step 2: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`
Expected: `Finished` with no errors

**Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat(voting): add screen-time weighting to auto-evolve"
```

---

### Task 7: Hot-reload votes.json alongside weights.json

**Files:**
- Modify: `src/main.rs` (FileWatcher setup — watch votes.json)
- Modify: `src/main.rs` (reload handler — reload vote ledger on change)

**Step 1: Add votes.json to file watcher**

Find where `FileWatcher::new` is called (in `resumed()`). The watched paths list should include votes.json. Add:

```rust
project_dir().join("genomes").join("votes.json")
```

to the paths array passed to `FileWatcher::new`.

**Step 2: Handle votes.json reload**

In the file change handler (where `weights.json` triggers a reload), add a branch for votes.json:

```rust
if changed_path.file_name().is_some_and(|n| n == "votes.json") {
    self.vote_ledger = VoteLedger::load(&project_dir().join("genomes"));
    eprintln!("[reload] votes.json");
}
```

**Step 3: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`
Expected: `Finished` with no errors

**Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat(voting): hot-reload votes.json"
```

---

### Task 8: Skip blacklisted genomes in load_random

**Files:**
- Modify: `src/genome.rs:534-546` (load_random — accept optional ledger + threshold)

**Step 1: Add blacklist-aware variant**

Add a new method alongside `load_random`:

```rust
    /// Load a random genome from directory, excluding blacklisted names.
    pub fn load_random_filtered(
        dir: &Path,
        blacklist: &std::collections::HashSet<String>,
    ) -> Result<Self, String> {
        use rand::prelude::IndexedRandom;
        let entries: Vec<_> = fs::read_dir(dir)
            .map_err(|e| format!("read dir: {e}"))?
            .filter_map(|e| e.ok())
            .filter(|e| {
                let path = e.path();
                path.extension().is_some_and(|ext| ext == "json")
                    && path.file_name().is_some_and(|n| n != "votes.json")
                    && !blacklist.contains(
                        path.file_stem().and_then(|s| s.to_str()).unwrap_or("")
                    )
            })
            .collect();
        if entries.is_empty() {
            return Err("no eligible genomes found".into());
        }
        let entry = entries.choose(&mut rand::rng()).ok_or("empty")?;
        Self::load(&entry.path())
    }
```

**Step 2: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`
Expected: `Finished` with no errors

**Step 3: Commit**

```bash
git add src/genome.rs
git commit -m "feat(voting): add blacklist-filtered genome loading"
```

---

### Task 9: Integration test and final verification

**Step 1: Full build**

Run: `cargo build 2>&1 | tail -5`
Expected: `Finished` with no errors

**Step 2: Manual integration test**

Run: `cargo run`

Test sequence:
1. Let it run — auto-evolve should work as before (no votes yet)
2. Press up arrow 3 times → should see `[vote] ... → +1 (score: N)`
3. Check `genomes/votes.json` exists with correct score
4. Press Space → mutation should use crossover (`[mutate]` log)
5. Press down arrow on a genome 3 times → score should hit -2 (blacklisted)
6. That genome should never appear again in auto-evolve
7. Let auto-evolve run — should see `[screen-time] showing voted genome: ...` for upvoted genomes

**Step 3: Commit and tag**

```bash
git add -A
git commit -m "feat(voting): local Electric Sheep voting system with crossover mutation"
```
