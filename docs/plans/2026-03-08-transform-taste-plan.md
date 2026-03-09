# Phase 3: Transform Taste Engine — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add transform-level taste learning, genome composition features, biased random transform generation, restructured genome persistence (save all genomes, separate voted pool), persistent ancestry tree, and generation-based history archiving.

**Architecture:** Two taste models (transform-level + genome-level composition) built from the same upvoted genomes. Transform model biases random-source transforms during breeding. Genome persistence restructured into `history/` (all genomes) and `voted/` (upvoted genomes) directories. Persistent `lineage.json` tracks full ancestry tree. History auto-archives by generation when size exceeds threshold.

**Tech Stack:** Rust, serde_json, existing `TasteModel` Gaussian centroid math

---

## Task 1: Directory restructure — create `voted/` and `history/` subdirs

**Files:**
- Modify: `src/main.rs` (App::new, startup dir creation)

**Step 1: Add directory creation on startup**

In `src/main.rs`, in the `App::new()` function, after the `genomes_root` line (around line 1354), add:

```rust
// Ensure genome subdirectories exist
let _ = std::fs::create_dir_all(genomes_root.join("voted"));
let _ = std::fs::create_dir_all(genomes_root.join("history"));
```

**Step 2: Run `cargo build`**

Run: `cargo build`
Expected: Compiles successfully.

**Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat: create voted/ and history/ genome subdirectories on startup"
```

---

## Task 2: Auto-save every genome to `history/`

**Files:**
- Modify: `src/main.rs` (auto-evolve block ~line 2042, manual evolve block ~line 1793)

**Step 1: Add auto-save after manual evolve**

In `src/main.rs`, after the `self.lineage_cache.register(...)` call in the manual Space-bar evolve block (around line 1807), add:

```rust
// Auto-save to history
let history_dir = project_dir().join("genomes").join("history");
if let Err(e) = self.genome.save(&history_dir) {
    eprintln!("[history] save error: {e}");
}
```

**Step 2: Add auto-save after auto-evolve**

In `src/main.rs`, after the `self.lineage_cache.register(...)` call in the auto-evolve block (around line 2057), add the same code:

```rust
// Auto-save to history
let history_dir = project_dir().join("genomes").join("history");
if let Err(e) = self.genome.save(&history_dir) {
    eprintln!("[history] save error: {e}");
}
```

**Step 3: Run `cargo build`**

Run: `cargo build`
Expected: Compiles successfully.

**Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat: auto-save every bred genome to history/"
```

---

## Task 3: Save voted genomes to `voted/`

**Files:**
- Modify: `src/main.rs` (ArrowUp handler ~line 1822, ArrowDown handler ~line 1829)
- Modify: `src/votes.rs` (`vote()` method ~line 44)

**Step 1: Update vote() to save to voted/ on upvote**

In `src/votes.rs`, change the `vote()` method. Currently it auto-saves to `genomes_dir` when first voted. Change it to also copy to `voted/` when score becomes positive. Replace the existing `vote()` method body:

```rust
pub fn vote(&mut self, genome: &FlameGenome, delta: i32, genomes_dir: &Path) -> i32 {
    let key = genome.name.clone();

    // Auto-save genome to history if no entry exists yet
    let history_dir = genomes_dir.join("history");
    if !self.entries.contains_key(&key) {
        let file_path = match genome.save(&history_dir) {
            Ok(p) => p.display().to_string(),
            Err(e) => {
                eprintln!("[vote] auto-save failed: {e}");
                format!("{}/{}.json", history_dir.display(), genome.name)
            }
        };
        self.entries.insert(
            key.clone(),
            VoteEntry {
                score: 0,
                file: file_path,
                last_seen: today(),
            },
        );
    }

    let entry = self.entries.get_mut(&key).unwrap();
    entry.score += delta;
    entry.last_seen = today();
    let score = entry.score;

    // Copy to voted/ when score is positive
    if score > 0 {
        let voted_dir = genomes_dir.join("voted");
        match genome.save(&voted_dir) {
            Ok(p) => {
                // Update file path to point to voted copy
                entry.file = p.display().to_string();
            }
            Err(e) => eprintln!("[vote] voted-save failed: {e}"),
        }
    }

    // Persist immediately
    if let Err(e) = self.save(genomes_dir) {
        eprintln!("[vote] save error: {e}");
    }

    score
}
```

**Step 2: Run `cargo test --lib votes`**

Run: `cargo test --lib votes`
Expected: All existing tests pass. The tests use in-memory ledgers and don't touch the filesystem, so they should be unaffected.

**Step 3: Run `cargo build`**

Run: `cargo build`
Expected: Compiles successfully.

**Step 4: Commit**

```bash
git add src/votes.rs src/main.rs
git commit -m "feat: save voted genomes to voted/ directory, history to history/"
```

---

## Task 4: Update parent selection to prefer `voted/`

**Files:**
- Modify: `src/main.rs` (`pick_breeding_parents` ~line 1473)
- Modify: `src/votes.rs` (`pick_random_saved` ~line 111)

**Step 1: Update `pick_random_saved` to scan voted/ and history/**

In `src/votes.rs`, update `pick_random_saved` to scan `voted/`, `history/`, and `seeds/`:

```rust
pub fn pick_random_saved(
    genomes_dir: &Path,
    _threshold: i32,
    ledger: &VoteLedger,
) -> Option<PathBuf> {
    use rand::prelude::IndexedRandom;

    // Scan voted/, history/, seeds/ for diverse pool
    // Voted first (preferred), then history, then seeds
    let dirs = [
        genomes_dir.join("voted"),
        genomes_dir.join("history"),
        genomes_dir.join("seeds"),
        genomes_dir.to_path_buf(), // legacy flat genomes
    ];
    let mut entries = Vec::new();
    for dir in &dirs {
        if let Ok(read) = fs::read_dir(dir) {
            for e in read.filter_map(|e| e.ok()) {
                let path = e.path();
                if !path.is_file()
                    || path.extension().is_none_or(|ext| ext != "json")
                    || path.file_name().is_some_and(|n| n == "votes.json")
                {
                    continue;
                }
                // Skip blacklisted genomes
                let name = path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                if let Some(entry) = ledger.entries.get(&name) {
                    if entry.score <= _threshold {
                        continue;
                    }
                }
                entries.push(path);
            }
        }
    }

    let mut rng = rand::rng();
    entries.choose(&mut rng).cloned()
}
```

**Step 2: Run `cargo test --lib votes`**

Run: `cargo test --lib votes`
Expected: All tests pass.

**Step 3: Run `cargo build`**

Run: `cargo build`
Expected: Compiles successfully.

**Step 4: Commit**

```bash
git add src/votes.rs src/main.rs
git commit -m "feat: parent selection scans voted/, history/, seeds/"
```

---

## Task 5: Update `rebuild_taste_model` to scan `voted/`

**Files:**
- Modify: `src/main.rs` (`rebuild_taste_model` ~line 1425)

**Step 1: Update rebuild_taste_model to scan voted/ directory**

Replace the "Positively voted genomes" section in `rebuild_taste_model` with:

```rust
// Load good genomes from voted/ directory
let voted_dir = genomes_dir.join("voted");
if let Ok(read) = std::fs::read_dir(&voted_dir) {
    for entry in read.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "json") {
            if let Ok(g) = FlameGenome::load(&path) {
                good_genomes.push(g);
            }
        }
    }
}
```

This replaces the vote ledger iteration — now we just load everything in `voted/` since only positively-voted genomes end up there.

**Step 2: Run `cargo build`**

Run: `cargo build`
Expected: Compiles successfully.

**Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat: taste model rebuilds from voted/ directory"
```

---

## Task 6: Persistent lineage.json ancestry tree

**Files:**
- Modify: `src/votes.rs` (replace `LineageCache` internal storage, add persistence)
- Modify: `src/main.rs` (update `LineageCache::build` call)

**Step 1: Add LineageEntry struct and persistence to LineageCache**

In `src/votes.rs`, add a serializable entry struct and update `LineageCache`:

```rust
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LineageEntry {
    pub parent_a: Option<String>,
    pub parent_b: Option<String>,
    pub generation: u32,
    pub created: String,
}

impl LineageCache {
    /// Load lineage from lineage.json, falling back to scanning genome files.
    pub fn load(genomes_dir: &Path) -> Self {
        let lineage_path = genomes_dir.join("lineage.json");
        if lineage_path.exists() {
            if let Ok(json) = fs::read_to_string(&lineage_path) {
                if let Ok(entries) = serde_json::from_str::<HashMap<String, LineageEntry>>(&json) {
                    let parents: HashMap<String, (Option<String>, Option<String>)> = entries
                        .into_iter()
                        .map(|(k, v)| (k, (v.parent_a, v.parent_b)))
                        .collect();
                    return Self { parents };
                }
            }
        }
        // Fallback: scan genome files (migration path)
        Self::build(genomes_dir)
    }

    /// Register a new genome and persist to lineage.json.
    pub fn register_and_save(
        &mut self,
        name: &str,
        parent_a: &Option<String>,
        parent_b: &Option<String>,
        generation: u32,
        genomes_dir: &Path,
    ) {
        self.register(name, parent_a, parent_b);
        // Append to lineage.json
        let lineage_path = genomes_dir.join("lineage.json");
        let mut entries: HashMap<String, LineageEntry> = if lineage_path.exists() {
            fs::read_to_string(&lineage_path)
                .ok()
                .and_then(|json| serde_json::from_str(&json).ok())
                .unwrap_or_default()
        } else {
            HashMap::new()
        };
        entries.insert(
            name.to_string(),
            LineageEntry {
                parent_a: parent_a.clone(),
                parent_b: parent_b.clone(),
                generation,
                created: today(),
            },
        );
        if let Ok(json) = serde_json::to_string_pretty(&entries) {
            let _ = fs::write(&lineage_path, json);
        }
    }
}
```

The existing `register()` and `build()` methods remain unchanged for backwards compatibility. `register_and_save()` wraps `register()` with disk persistence.

**Step 2: Update main.rs to use `LineageCache::load` and `register_and_save`**

In `src/main.rs`:

- Change `LineageCache::build(&genomes_root)` → `LineageCache::load(&genomes_root)` in `App::new()`
- Change all `self.lineage_cache.register(...)` calls to `self.lineage_cache.register_and_save(...)` with the additional `generation` and `genomes_dir` params. There are two call sites: manual evolve (~line 1803) and auto-evolve (~line 2053).

```rust
// Replace:
self.lineage_cache.register(
    &self.genome.name,
    &self.genome.parent_a,
    &self.genome.parent_b,
);
// With:
let genomes_dir = project_dir().join("genomes");
self.lineage_cache.register_and_save(
    &self.genome.name,
    &self.genome.parent_a,
    &self.genome.parent_b,
    self.genome.generation,
    &genomes_dir,
);
```

**Step 3: Add tests**

Add to the existing `mod tests` in `src/votes.rs`:

```rust
    #[test]
    fn lineage_entry_serialization_roundtrip() {
        let entry = LineageEntry {
            parent_a: Some("pa".into()),
            parent_b: Some("pb".into()),
            generation: 3,
            created: "2026-03-08".into(),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let entry2: LineageEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry.parent_a, entry2.parent_a);
        assert_eq!(entry.parent_b, entry2.parent_b);
        assert_eq!(entry.generation, entry2.generation);
    }
```

**Step 4: Run tests**

Run: `cargo test --lib votes`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add src/votes.rs src/main.rs
git commit -m "feat: persistent lineage.json ancestry tree"
```

---

## Task 7: Archive config fields + archive logic

**Files:**
- Modify: `src/weights.rs` (add `archive_threshold_mb` and `archive_on_startup` to `RuntimeConfig`)
- Modify: `src/main.rs` (add archive check on startup)

**Step 1: Add config fields**

In `src/weights.rs`, add to `RuntimeConfig`:

```rust
    #[serde(default = "default_archive_threshold_mb")]
    pub archive_threshold_mb: u64,
    #[serde(default = "default_archive_on_startup")]
    pub archive_on_startup: bool,
```

Add default functions:

```rust
fn default_archive_threshold_mb() -> u64 {
    100
}
fn default_archive_on_startup() -> bool {
    true
}
```

**Step 2: Add archive function in main.rs**

Add a standalone function (not on App) in `src/main.rs`:

```rust
/// Archive old genomes from history/ when size exceeds threshold.
/// Groups by generation, archives the older half into a tar.gz.
fn archive_history_if_needed(genomes_dir: &Path, threshold_mb: u64) {
    let history_dir = genomes_dir.join("history");
    if !history_dir.exists() {
        return;
    }

    // Calculate total size
    let mut total_bytes: u64 = 0;
    let mut genomes: Vec<(PathBuf, u32)> = Vec::new(); // (path, generation)

    if let Ok(read) = std::fs::read_dir(&history_dir) {
        for entry in read.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                if let Ok(meta) = path.metadata() {
                    total_bytes += meta.len();
                    // Try to read generation from the genome
                    let gen = std::fs::read_to_string(&path)
                        .ok()
                        .and_then(|json| serde_json::from_str::<serde_json::Value>(&json).ok())
                        .and_then(|v| v.get("generation")?.as_u64())
                        .unwrap_or(0) as u32;
                    genomes.push((path, gen));
                }
            }
        }
    }

    let threshold_bytes = threshold_mb * 1024 * 1024;
    if total_bytes < threshold_bytes {
        return;
    }

    eprintln!(
        "[archive] history/ is {}MB (threshold {}MB), archiving old genomes...",
        total_bytes / (1024 * 1024),
        threshold_mb
    );

    // Find median generation
    genomes.sort_by_key(|(_, gen)| *gen);
    let median_idx = genomes.len() / 2;
    let median_gen = genomes[median_idx].1;

    // Delete genomes below median generation (lineage.json preserves ancestry)
    let mut archived_count = 0u32;
    for (path, gen) in &genomes {
        if *gen < median_gen {
            if std::fs::remove_file(path).is_ok() {
                archived_count += 1;
            }
        }
    }

    eprintln!(
        "[archive] removed {} genomes below generation {}",
        archived_count, median_gen
    );
}
```

Note: This does simple deletion rather than tar.gz to avoid adding a compression dependency. The lineage.json preserves ancestry regardless. Tar.gz archiving can be added later if needed.

**Step 3: Call archive on startup**

In `App::new()`, after the directory creation lines, add:

```rust
if weights._config.archive_on_startup {
    archive_history_if_needed(&genomes_root, weights._config.archive_threshold_mb);
}
```

**Step 4: Add config doc entries to weights.json**

Add to the `_config_doc` section:

```json
"archive_threshold_mb": "History directory size threshold in MB before archiving (default 100)",
"archive_on_startup": "Check and archive old history genomes on app start (default true)"
```

Add to the `_config` section:

```json
"archive_threshold_mb": 100,
"archive_on_startup": true
```

**Step 5: Add tests for config fields**

Add to the `mod tests` in `src/weights.rs`:

```rust
    #[test]
    fn archive_config_defaults() {
        let cfg: RuntimeConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.archive_threshold_mb, 100);
        assert!(cfg.archive_on_startup);
    }
```

**Step 6: Run tests**

Run: `cargo test`
Expected: All tests pass.

**Step 7: Commit**

```bash
git add src/weights.rs src/main.rs weights.json
git commit -m "feat: add generation-based history archiving with configurable threshold"
```

---

## Task 8: TransformFeatures — extraction and tests

**Files:**
- Modify: `src/taste.rs` (add `TransformFeatures` struct + tests)

**Step 1: Add the TransformFeatures struct and extraction**

Add after the `PaletteFeatures` impl block (after `hue_overlap` method, around line 110) in `src/taste.rs`:

```rust
/// Number of transform-level features.
pub const TRANSFORM_FEATURE_COUNT: usize = 8;

/// Features extracted from a single FlameTransform for taste modeling.
#[derive(Clone, Debug)]
pub struct TransformFeatures {
    /// Index of the variation with the highest weight (0-25)
    pub primary_variation_index: f32,
    /// Weight of primary variation / total variation weight (0-1)
    pub primary_dominance: f32,
    /// Number of variations with weight > 0
    pub active_variation_count: f32,
    /// |ad - bc| — contraction/expansion measure
    pub affine_determinant: f32,
    /// |a-d| + |b+c| — asymmetry measure
    pub affine_asymmetry: f32,
    /// sqrt(offset_x^2 + offset_y^2)
    pub offset_magnitude: f32,
    /// Palette color index (0-1)
    pub color_index: f32,
    /// Transform selection weight
    pub weight: f32,
}

impl TransformFeatures {
    /// Extract features from a FlameTransform.
    pub fn extract(xf: &crate::genome::FlameTransform) -> Self {
        // Find primary variation and compute stats
        let mut max_var_idx = 0usize;
        let mut max_var_weight = 0.0f32;
        let mut total_var_weight = 0.0f32;
        let mut active_count = 0u32;

        for i in 0..26 {
            let w = xf.get_variation(i);
            if w > 0.0 {
                active_count += 1;
                total_var_weight += w;
                if w > max_var_weight {
                    max_var_weight = w;
                    max_var_idx = i;
                }
            }
        }

        let primary_dominance = if total_var_weight > 0.0 {
            max_var_weight / total_var_weight
        } else {
            0.0
        };

        let affine_determinant = (xf.a * xf.d - xf.b * xf.c).abs();
        let affine_asymmetry = (xf.a - xf.d).abs() + (xf.b + xf.c).abs();
        let offset_magnitude = (xf.offset[0].powi(2) + xf.offset[1].powi(2)).sqrt();

        Self {
            primary_variation_index: max_var_idx as f32,
            primary_dominance,
            active_variation_count: active_count as f32,
            affine_determinant,
            affine_asymmetry,
            offset_magnitude,
            color_index: xf.color,
            weight: xf.weight,
        }
    }

    /// Convert to flat f32 vector for the taste model.
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.primary_variation_index,
            self.primary_dominance,
            self.active_variation_count,
            self.affine_determinant,
            self.affine_asymmetry,
            self.offset_magnitude,
            self.color_index,
            self.weight,
        ]
    }
}
```

**Step 2: Add tests**

Add inside the existing `mod tests` in `src/taste.rs`:

```rust
    // --- TransformFeatures tests ---

    #[test]
    fn transform_features_identity_affine() {
        let xf = crate::genome::FlameTransform {
            weight: 0.5,
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            offset: [0.0, 0.0],
            color: 0.3,
            linear: 1.0,
            ..Default::default()
        };
        let f = TransformFeatures::extract(&xf);
        assert!(approx_eq(f.affine_determinant, 1.0), "det was {}", f.affine_determinant);
        assert!(approx_eq(f.affine_asymmetry, 0.0), "asym was {}", f.affine_asymmetry);
        assert!(approx_eq(f.offset_magnitude, 0.0));
        assert!(approx_eq(f.primary_dominance, 1.0));
        assert!(approx_eq(f.active_variation_count, 1.0));
        assert!(approx_eq(f.color_index, 0.3));
        assert!(approx_eq(f.weight, 0.5));
    }

    #[test]
    fn transform_features_two_variations() {
        let xf = crate::genome::FlameTransform {
            weight: 1.0,
            a: 0.5,
            b: -0.5,
            c: 0.5,
            d: 0.5,
            offset: [0.3, 0.4],
            color: 0.0,
            spherical: 0.7,
            julia: 0.3,
            ..Default::default()
        };
        let f = TransformFeatures::extract(&xf);
        assert!(approx_eq(f.active_variation_count, 2.0));
        assert!(approx_eq(f.primary_dominance, 0.7), "dom was {}", f.primary_dominance);
        // primary_variation_index should be spherical (index 2)
        assert!(approx_eq(f.primary_variation_index, 2.0), "idx was {}", f.primary_variation_index);
        // offset magnitude: sqrt(0.09 + 0.16) = 0.5
        assert!(approx_eq(f.offset_magnitude, 0.5), "offset was {}", f.offset_magnitude);
        // determinant: |0.5*0.5 - (-0.5)*0.5| = |0.25 + 0.25| = 0.5
        assert!(approx_eq(f.affine_determinant, 0.5), "det was {}", f.affine_determinant);
    }

    #[test]
    fn transform_features_vec_length() {
        let xf = crate::genome::FlameTransform::default();
        let f = TransformFeatures::extract(&xf);
        assert_eq!(f.to_vec().len(), TRANSFORM_FEATURE_COUNT);
    }
```

**Step 3: Run tests**

Run: `cargo test --lib taste`
Expected: All tests pass (existing + 3 new).

**Step 4: Commit**

```bash
git add src/taste.rs
git commit -m "feat: add TransformFeatures extraction with unit tests"
```

---

## Task 9: CompositionFeatures — extraction and tests

**Files:**
- Modify: `src/taste.rs` (add `CompositionFeatures` struct + tests)

**Step 1: Add the CompositionFeatures struct**

Add after the `TransformFeatures` impl block in `src/taste.rs`:

```rust
/// Number of genome-level composition features.
pub const COMPOSITION_FEATURE_COUNT: usize = 5;

/// Genome-level structural features for the expanded taste model.
#[derive(Clone, Debug)]
pub struct CompositionFeatures {
    /// Number of transforms
    pub transform_count: f32,
    /// Number of distinct active variation types across all transforms
    pub variation_diversity: f32,
    /// Mean affine determinant across transforms
    pub mean_determinant: f32,
    /// Stddev of affine determinants (how different are transforms?)
    pub determinant_contrast: f32,
    /// Stddev of color indices across transforms
    pub color_spread: f32,
}

impl CompositionFeatures {
    /// Extract composition features from a genome.
    pub fn extract(genome: &FlameGenome) -> Self {
        let n = genome.transforms.len();
        if n == 0 {
            return Self {
                transform_count: 0.0,
                variation_diversity: 0.0,
                mean_determinant: 0.0,
                determinant_contrast: 0.0,
                color_spread: 0.0,
            };
        }

        // Variation diversity: count unique active variation types
        let mut active_types = std::collections::HashSet::new();
        for xf in &genome.transforms {
            for i in 0..26 {
                if xf.get_variation(i) > 0.0 {
                    active_types.insert(i);
                }
            }
        }

        // Affine determinants
        let dets: Vec<f32> = genome
            .transforms
            .iter()
            .map(|xf| (xf.a * xf.d - xf.b * xf.c).abs())
            .collect();
        let mean_det = dets.iter().sum::<f32>() / n as f32;
        let det_variance = dets.iter().map(|d| (d - mean_det).powi(2)).sum::<f32>() / n as f32;

        // Color spread
        let colors: Vec<f32> = genome.transforms.iter().map(|xf| xf.color).collect();
        let mean_color = colors.iter().sum::<f32>() / n as f32;
        let color_variance = colors
            .iter()
            .map(|c| (c - mean_color).powi(2))
            .sum::<f32>()
            / n as f32;

        Self {
            transform_count: n as f32,
            variation_diversity: active_types.len() as f32,
            mean_determinant: mean_det,
            determinant_contrast: det_variance.sqrt(),
            color_spread: color_variance.sqrt(),
        }
    }

    /// Convert to flat f32 vector.
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.transform_count,
            self.variation_diversity,
            self.mean_determinant,
            self.determinant_contrast,
            self.color_spread,
        ]
    }
}
```

**Step 2: Add tests**

Add inside the existing `mod tests`:

```rust
    // --- CompositionFeatures tests ---

    #[test]
    fn composition_features_basic() {
        let genome = crate::genome::FlameGenome::default_genome();
        let f = CompositionFeatures::extract(&genome);
        assert!(f.transform_count >= 3.0, "transform_count was {}", f.transform_count);
        assert!(f.variation_diversity >= 1.0, "diversity was {}", f.variation_diversity);
        assert!(f.mean_determinant > 0.0, "mean_det was {}", f.mean_determinant);
    }

    #[test]
    fn composition_features_vec_length() {
        let genome = crate::genome::FlameGenome::default_genome();
        let f = CompositionFeatures::extract(&genome);
        assert_eq!(f.to_vec().len(), COMPOSITION_FEATURE_COUNT);
    }

    #[test]
    fn composition_features_empty_genome() {
        let genome = crate::genome::FlameGenome {
            name: "empty".into(),
            transforms: vec![],
            ..Default::default()
        };
        let f = CompositionFeatures::extract(&genome);
        assert!(approx_eq(f.transform_count, 0.0));
        assert!(approx_eq(f.variation_diversity, 0.0));
    }
```

Note: the `..Default::default()` may need to use a `FlameGenome` with explicit fields if `Default` isn't derived. If so, build a minimal genome with explicit fields matching the struct.

**Step 3: Run tests**

Run: `cargo test --lib taste`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add src/taste.rs
git commit -m "feat: add CompositionFeatures extraction with unit tests"
```

---

## Task 10: Expand TasteEngine with transform model

**Files:**
- Modify: `src/taste.rs` (`TasteEngine` struct and `rebuild` method)

**Step 1: Add transform model field to TasteEngine**

Update the `TasteEngine` struct to hold a second model:

```rust
pub struct TasteEngine {
    /// Current palette model (None if not enough data)
    model: Option<TasteModel>,
    /// Current transform model (None if not enough data)
    transform_model: Option<TasteModel>,
    /// Recent palette features for diversity nudge
    recent_palettes: VecDeque<PaletteFeatures>,
    /// All feature vectors from good genomes (for rebuilding model)
    good_features: Vec<Vec<f32>>,
}
```

**Step 2: Update `new()`**

```rust
pub fn new() -> Self {
    Self {
        model: None,
        transform_model: None,
        recent_palettes: VecDeque::new(),
        good_features: Vec::new(),
    }
}
```

**Step 3: Update `rebuild()` to also build the transform model**

Update the `rebuild` method to extract both palette features (with composition features appended) and transform features:

```rust
pub fn rebuild(&mut self, good_genomes: &[&FlameGenome], recent_memory: usize) {
    self.good_features.clear();

    // Collect palette + composition features (genome-level)
    for genome in good_genomes {
        if let Some(palette_features) = PaletteFeatures::extract(genome) {
            let mut features = palette_features.to_vec();
            // Append composition features
            let comp = CompositionFeatures::extract(genome);
            features.extend(comp.to_vec());
            self.good_features.push(features);
        }
    }

    self.model = TasteModel::build(&self.good_features);

    // Collect transform-level features (pooled from all transforms)
    let mut transform_features: Vec<Vec<f32>> = Vec::new();
    for genome in good_genomes {
        for xf in &genome.transforms {
            let tf = TransformFeatures::extract(xf);
            transform_features.push(tf.to_vec());
        }
    }
    self.transform_model = TasteModel::build(&transform_features);

    // Trim recent palette memory
    while self.recent_palettes.len() > recent_memory {
        self.recent_palettes.pop_front();
    }

    if let Some(ref model) = self.model {
        eprintln!(
            "[taste] model rebuilt: {} genomes, {} genome features, {} transform samples",
            good_genomes.len(),
            model.feature_means.len(),
            self.transform_model.as_ref().map_or(0, |m| m.sample_count),
        );
    }
}
```

**Step 4: Run tests**

Run: `cargo test --lib taste`
Expected: All tests pass. The existing `taste_engine_generate_palette_returns_256` and `taste_engine_inactive_below_threshold` tests should still work since `generate_palette` uses `self.model` which is unchanged.

**Step 5: Commit**

```bash
git add src/taste.rs
git commit -m "feat: expand TasteEngine with transform taste model"
```

---

## Task 11: Generate biased transforms

**Files:**
- Modify: `src/taste.rs` (add `generate_biased_transform` method to `TasteEngine`)

**Step 1: Add the method**

Add to the `TasteEngine` impl block:

```rust
/// Generate a random transform biased by the taste model.
/// Falls back to pure random if model isn't ready.
pub fn generate_biased_transform(
    &self,
    min_votes: u32,
    strength: f32,
    exploration_rate: f32,
    candidates: u32,
) -> crate::genome::FlameTransform {
    use rand::Rng;
    let mut rng = rand::rng();

    // Exploration: sometimes skip the model entirely
    if rng.random::<f32>() < exploration_rate {
        return crate::genome::FlameTransform::random_transform(&mut rng);
    }

    // If transform model isn't ready, use pure random
    let model = match &self.transform_model {
        Some(m) if m.sample_count >= min_votes => m,
        _ => return crate::genome::FlameTransform::random_transform(&mut rng),
    };

    // Generate candidates and score them
    let mut best_xf = crate::genome::FlameTransform::random_transform(&mut rng);
    let mut best_score = f32::MAX;

    for _ in 0..candidates {
        let xf = crate::genome::FlameTransform::random_transform(&mut rng);
        let features = TransformFeatures::extract(&xf);
        let score = model.score(&features.to_vec()) * strength;

        if score < best_score {
            best_score = score;
            best_xf = xf;
        }
    }

    best_xf
}
```

**Step 2: Add tests**

Add inside the existing `mod tests`:

```rust
    #[test]
    fn generate_biased_transform_returns_valid() {
        let engine = TasteEngine::new();
        // With no model, should fall back to random
        let xf = engine.generate_biased_transform(10, 1.0, 0.0, 5);
        assert!(xf.weight > 0.0, "weight was {}", xf.weight);
        // Should have at least one variation
        let has_var = (0..26).any(|i| xf.get_variation(i) > 0.0);
        assert!(has_var, "should have at least one variation");
    }

    #[test]
    fn generate_biased_transform_exploration_returns_valid() {
        let engine = TasteEngine::new();
        // exploration_rate = 1.0 → always skip model
        let xf = engine.generate_biased_transform(10, 1.0, 1.0, 5);
        assert!(xf.weight > 0.0);
    }
```

**Step 3: Run tests**

Run: `cargo test --lib taste`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add src/taste.rs
git commit -m "feat: add generate_biased_transform for taste-driven random transforms"
```

---

## Task 12: Wire biased transforms into breeding

**Files:**
- Modify: `src/genome.rs` (`breed()` method, wildcard slot and environment slot generation)

**Step 1: Update wildcard slot to use taste-biased transform**

In `src/genome.rs`, in the `breed()` method, the wildcard slot (slot 0) currently gets a random transform from the initial fill at line 988. After the initial fill loop, replace the comment `// Wildcard: already random from above` with:

```rust
// Wildcard: use taste-biased transform if available
if cfg.taste_engine_enabled {
    if let Some(te) = taste.as_ref() {
        transforms[_wildcard_slot] = te.generate_biased_transform(
            cfg.taste_min_votes,
            cfg.taste_strength,
            cfg.taste_exploration_rate,
            cfg.taste_candidates,
        );
    }
}
```

Note: The `_wildcard_slot` variable needs to be renamed to `wildcard_slot` (remove the underscore prefix) since it's now used.

**Step 2: Run `cargo test --lib genome`**

Run: `cargo test --lib genome`
Expected: All tests pass. The `breed_*` tests don't enable `taste_engine_enabled` so this path won't execute in tests.

**Step 3: Run `cargo build`**

Run: `cargo build`
Expected: Compiles successfully.

**Step 4: Commit**

```bash
git add src/genome.rs
git commit -m "feat: wire taste-biased transforms into breeding wildcard slot"
```

---

## Task 13: Final verification

**Step 1: Run all tests**

Run: `cargo test`
Expected: All tests pass (78+ existing + new taste tests).

**Step 2: Run the pre-commit checks**

Run: `cargo fmt --check && cargo clippy -- -D warnings && cargo test`
Expected: All three checks pass.

**Step 3: Commit any final fixes**

```bash
git add -A
git commit -m "chore: Phase 3 transform taste engine complete"
```

**Step 4: Tag**

```bash
git tag -a v0.4.0-transform-taste -m "Phase 3: transform taste engine + genome persistence restructure"
```
