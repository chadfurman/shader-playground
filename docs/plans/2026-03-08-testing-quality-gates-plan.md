# Testing + Quality Gates Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add unit tests to all logic modules and a pre-commit hook that gates commits behind fmt, clippy, and test checks.

**Architecture:** Inline `#[cfg(test)] mod tests` in each source file. Pre-commit hook is a shell script at `.git/hooks/pre-commit`. All test data is constructed inline (no file dependencies). Private functions that need testing get `pub(crate)` visibility.

**Tech Stack:** Rust built-in test framework (`#[test]`, `assert!`, `assert_eq!`), `cargo fmt`, `cargo clippy`, shell script for git hook.

---

### Task 1: Pre-commit hook

**Files:**
- Create: `.git/hooks/pre-commit`

**Step 1: Create the pre-commit hook script**

```bash
#!/bin/sh
# Pre-commit hook: format, lint, test

echo "[pre-commit] Running cargo fmt --check..."
cargo fmt --check
if [ $? -ne 0 ]; then
    echo "[pre-commit] FAILED: cargo fmt. Run 'cargo fmt' to fix."
    exit 1
fi

echo "[pre-commit] Running cargo clippy..."
cargo clippy -- -D warnings
if [ $? -ne 0 ]; then
    echo "[pre-commit] FAILED: cargo clippy. Fix warnings above."
    exit 1
fi

echo "[pre-commit] Running cargo test..."
cargo test
if [ $? -ne 0 ]; then
    echo "[pre-commit] FAILED: tests. Fix failing tests above."
    exit 1
fi

echo "[pre-commit] All checks passed."
```

**Step 2: Make it executable**

Run: `chmod +x .git/hooks/pre-commit`

**Step 3: Run it manually to verify**

Run: `.git/hooks/pre-commit`
Expected: All three checks pass (or fmt/clippy issues surface that need fixing first).

**Step 4: Fix any cargo fmt issues**

Run: `cargo fmt`
Then run hook again to confirm clean.

**Step 5: Fix any clippy issues**

Run: `cargo clippy -- -D warnings`
Fix any warnings. Then run hook again.

**Step 6: Commit**

```bash
git add -A
git commit -m "chore: add pre-commit hook (fmt + clippy + test)"
```

Note: The hook lives in `.git/hooks/` which is not tracked by git. To persist it, we'll add a `scripts/install-hooks.sh` that copies it:

Create `scripts/install-hooks.sh`:
```bash
#!/bin/sh
cp scripts/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
echo "Git hooks installed."
```

Create `scripts/pre-commit` (same content as the hook above).

Then commit `scripts/` so the hook is version-controlled:
```bash
git add scripts/
git commit -m "chore: add pre-commit hook with install script"
```

---

### Task 2: taste.rs tests — RGB/HSV and feature extraction

**Files:**
- Modify: `src/taste.rs` (add `#[cfg(test)] mod tests` at end, make `rgb_to_hsv`, `count_hue_clusters`, `palette_features` pub(crate))

**Step 1: Make helper functions testable**

Change visibility of these functions in `src/taste.rs`:
- `fn rgb_to_hsv(...)` → `pub(crate) fn rgb_to_hsv(...)`
- `fn count_hue_clusters(...)` → `pub(crate) fn count_hue_clusters(...)`
- `fn palette_features(...)` → `pub(crate) fn palette_features(...)`

**Step 2: Write the tests**

Add at the end of `src/taste.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    #[test]
    fn rgb_to_hsv_pure_red() {
        let (h, s, v) = rgb_to_hsv([1.0, 0.0, 0.0]);
        assert!(approx_eq(h, 0.0), "hue should be 0, got {h}");
        assert!(approx_eq(s, 1.0), "saturation should be 1, got {s}");
        assert!(approx_eq(v, 1.0), "value should be 1, got {v}");
    }

    #[test]
    fn rgb_to_hsv_pure_green() {
        let (h, s, v) = rgb_to_hsv([0.0, 1.0, 0.0]);
        assert!(approx_eq(h, 120.0), "hue should be 120, got {h}");
        assert!(approx_eq(s, 1.0));
        assert!(approx_eq(v, 1.0));
    }

    #[test]
    fn rgb_to_hsv_pure_blue() {
        let (h, s, v) = rgb_to_hsv([0.0, 0.0, 1.0]);
        assert!(approx_eq(h, 240.0), "hue should be 240, got {h}");
        assert!(approx_eq(s, 1.0));
        assert!(approx_eq(v, 1.0));
    }

    #[test]
    fn rgb_to_hsv_white() {
        let (h, s, v) = rgb_to_hsv([1.0, 1.0, 1.0]);
        assert!(approx_eq(s, 0.0), "white has no saturation");
        assert!(approx_eq(v, 1.0));
    }

    #[test]
    fn rgb_to_hsv_black() {
        let (h, s, v) = rgb_to_hsv([0.0, 0.0, 0.0]);
        assert!(approx_eq(s, 0.0));
        assert!(approx_eq(v, 0.0));
    }

    #[test]
    fn rgb_to_hsv_gray() {
        let (_, s, v) = rgb_to_hsv([0.5, 0.5, 0.5]);
        assert!(approx_eq(s, 0.0), "gray has no saturation");
        assert!(approx_eq(v, 0.5));
    }

    #[test]
    fn hue_clusters_single_bin() {
        let mut hist = [0.0f32; HUE_BINS];
        hist[3] = 1.0;
        assert_eq!(count_hue_clusters(&hist), 1.0);
    }

    #[test]
    fn hue_clusters_two_separated() {
        let mut hist = [0.0f32; HUE_BINS];
        hist[1] = 0.5;
        hist[7] = 0.5;
        assert_eq!(count_hue_clusters(&hist), 2.0);
    }

    #[test]
    fn hue_clusters_adjacent_is_one() {
        let mut hist = [0.0f32; HUE_BINS];
        hist[3] = 0.3;
        hist[4] = 0.3;
        hist[5] = 0.4;
        assert_eq!(count_hue_clusters(&hist), 1.0);
    }

    #[test]
    fn hue_clusters_wrapping() {
        // Bins 0 and 11 are both filled — should count as one cluster (wrapping)
        let mut hist = [0.0f32; HUE_BINS];
        hist[0] = 0.5;
        hist[11] = 0.5;
        assert_eq!(count_hue_clusters(&hist), 1.0);
    }

    #[test]
    fn hue_clusters_empty() {
        let hist = [0.0f32; HUE_BINS];
        assert_eq!(count_hue_clusters(&hist), 0.0);
    }

    #[test]
    fn palette_features_uniform_red() {
        // 256 entries of pure red
        let palette: Vec<[f32; 3]> = vec![[1.0, 0.0, 0.0]; 256];
        let features = palette_features(&palette);
        // All hue in bin 0 (red = 0 degrees)
        assert!(approx_eq(features.hue_histogram[0], 1.0));
        assert!(approx_eq(features.avg_saturation, 1.0));
        assert_eq!(features.hue_cluster_count, 1.0);
    }

    #[test]
    fn palette_features_all_gray() {
        // Gray palette — no saturation, so hue histogram should be all zeros
        let palette: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5]; 256];
        let features = palette_features(&palette);
        assert!(features.hue_histogram.iter().all(|&h| h == 0.0));
        assert!(approx_eq(features.avg_saturation, 0.0));
        assert!(approx_eq(features.avg_brightness, 0.5));
    }

    #[test]
    fn hue_overlap_identical() {
        let palette: Vec<[f32; 3]> = vec![[1.0, 0.0, 0.0]; 256];
        let f1 = palette_features(&palette);
        let f2 = palette_features(&palette);
        assert!(approx_eq(f1.hue_overlap(&f2), 1.0));
    }

    #[test]
    fn hue_overlap_disjoint() {
        let red: Vec<[f32; 3]> = vec![[1.0, 0.0, 0.0]; 256];
        let blue: Vec<[f32; 3]> = vec![[0.0, 0.0, 1.0]; 256];
        let f1 = palette_features(&red);
        let f2 = palette_features(&blue);
        assert!(approx_eq(f1.hue_overlap(&f2), 0.0));
    }

    #[test]
    fn feature_vec_length() {
        let palette: Vec<[f32; 3]> = vec![[1.0, 0.0, 0.0]; 256];
        let features = palette_features(&palette);
        assert_eq!(features.to_vec().len(), PALETTE_FEATURE_COUNT);
    }
}
```

**Step 3: Run tests to verify they pass**

Run: `cargo test --lib taste`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add src/taste.rs
git commit -m "test: add taste.rs unit tests for RGB/HSV and feature extraction"
```

---

### Task 3: taste.rs tests — TasteModel

**Files:**
- Modify: `src/taste.rs` (add more tests to existing `mod tests`)

**Step 1: Add TasteModel tests**

Add inside the existing `mod tests`:

```rust
    #[test]
    fn taste_model_build_empty() {
        assert!(TasteModel::build(&[]).is_none());
    }

    #[test]
    fn taste_model_build_single_sample() {
        let features = vec![vec![0.5, 0.3, 0.7]];
        let model = TasteModel::build(&features).unwrap();
        assert_eq!(model.sample_count, 1);
        assert!(approx_eq(model.feature_means[0], 0.5));
        assert!(approx_eq(model.feature_means[1], 0.3));
        assert!(approx_eq(model.feature_means[2], 0.7));
        // Stddev of single sample = 0, floored to 0.01
        for s in &model.feature_stddevs {
            assert!(approx_eq(*s, 0.01));
        }
    }

    #[test]
    fn taste_model_build_two_samples() {
        let features = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ];
        let model = TasteModel::build(&features).unwrap();
        assert_eq!(model.sample_count, 2);
        assert!(approx_eq(model.feature_means[0], 0.5));
        assert!(approx_eq(model.feature_means[1], 0.5));
        assert!(approx_eq(model.feature_stddevs[0], 0.5));
        assert!(approx_eq(model.feature_stddevs[1], 0.5));
    }

    #[test]
    fn taste_model_score_at_mean_is_zero() {
        let features = vec![vec![0.5, 0.3]];
        let model = TasteModel::build(&features).unwrap();
        let score = model.score(&[0.5, 0.3]);
        assert!(approx_eq(score, 0.0), "score at mean should be 0, got {score}");
    }

    #[test]
    fn taste_model_score_far_from_mean_is_high() {
        let features = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
        ];
        let model = TasteModel::build(&features).unwrap();
        let score_at_mean = model.score(&[0.5, 0.5]);
        let score_far = model.score(&[5.0, 5.0]);
        assert!(score_far > score_at_mean, "far score ({score_far}) should exceed mean score ({score_at_mean})");
    }

    #[test]
    fn taste_engine_inactive_below_threshold() {
        let engine = TasteEngine::new();
        assert!(!engine.is_active(10));
    }

    #[test]
    fn taste_engine_generate_palette_returns_256() {
        let mut engine = TasteEngine::new();
        let palette = engine.generate_palette(10, 0.5, 0.0, 0.3, 5, 5);
        assert_eq!(palette.len(), 256);
    }
```

**Step 2: Run tests**

Run: `cargo test --lib taste`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add src/taste.rs
git commit -m "test: add TasteModel and TasteEngine unit tests"
```

---

### Task 4: votes.rs tests — LineageCache

**Files:**
- Modify: `src/votes.rs` (add `#[cfg(test)] mod tests` at end)

**Step 1: Write lineage cache tests**

Add at the end of `src/votes.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache() -> LineageCache {
        let mut cache = LineageCache::default();
        // Family tree:
        //   grandparent_a   grandparent_b
        //        \              /
        //         parent_a  parent_b (unrelated)
        //              \    /
        //              child
        cache.register("grandparent_a", &None, &None);
        cache.register("grandparent_b", &None, &None);
        cache.register("parent_a", &Some("grandparent_a".into()), &None);
        cache.register("parent_b", &Some("grandparent_b".into()), &None);
        cache.register("child", &Some("parent_a".into()), &Some("parent_b".into()));
        cache
    }

    #[test]
    fn distance_self_is_zero() {
        let cache = make_cache();
        assert_eq!(cache.genetic_distance("child", "child", 8), 0);
    }

    #[test]
    fn distance_to_parent_is_one() {
        let cache = make_cache();
        assert_eq!(cache.genetic_distance("child", "parent_a", 8), 1);
        assert_eq!(cache.genetic_distance("child", "parent_b", 8), 1);
    }

    #[test]
    fn distance_to_grandparent_is_two() {
        let cache = make_cache();
        assert_eq!(cache.genetic_distance("child", "grandparent_a", 8), 2);
    }

    #[test]
    fn distance_between_parents_sharing_child() {
        let cache = make_cache();
        // parent_a and parent_b share no common ancestor (besides child, which is a descendant)
        // They are unrelated — distance should be max_depth
        assert_eq!(cache.genetic_distance("parent_a", "parent_b", 8), 8);
    }

    #[test]
    fn distance_siblings_share_parent() {
        let mut cache = LineageCache::default();
        cache.register("dad", &None, &None);
        cache.register("mom", &None, &None);
        cache.register("sibling_a", &Some("dad".into()), &Some("mom".into()));
        cache.register("sibling_b", &Some("dad".into()), &Some("mom".into()));
        // Both siblings have dad and mom at depth 1 → distance = 1
        assert_eq!(cache.genetic_distance("sibling_a", "sibling_b", 8), 1);
    }

    #[test]
    fn distance_no_common_ancestor_returns_max_depth() {
        let mut cache = LineageCache::default();
        cache.register("island_a", &None, &None);
        cache.register("island_b", &None, &None);
        assert_eq!(cache.genetic_distance("island_a", "island_b", 8), 8);
    }

    #[test]
    fn distance_unknown_genome_returns_max_depth() {
        let cache = LineageCache::default();
        assert_eq!(cache.genetic_distance("unknown_a", "unknown_b", 8), 8);
    }

    #[test]
    fn register_updates_cache() {
        let mut cache = LineageCache::default();
        cache.register("new_genome", &Some("pa".into()), &Some("pb".into()));
        // Should now know about new_genome's parents
        assert_eq!(cache.genetic_distance("new_genome", "pa", 8), 1);
    }
}
```

**Step 2: Run tests**

Run: `cargo test --lib votes`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add src/votes.rs
git commit -m "test: add LineageCache genetic distance unit tests"
```

---

### Task 5: votes.rs tests — VoteLedger

**Files:**
- Modify: `src/votes.rs` (add more tests to existing `mod tests`)

**Step 1: Add VoteLedger tests**

Add inside the existing `mod tests`:

```rust
    #[test]
    fn vote_ledger_empty_pick_returns_none() {
        let ledger = VoteLedger::default();
        assert!(ledger.pick_voted(0).is_none());
    }

    #[test]
    fn vote_ledger_positive_score_is_pickable() {
        let mut ledger = VoteLedger::default();
        ledger.entries.insert("test_genome".into(), VoteEntry {
            score: 3,
            file: "/tmp/test_genome.json".into(),
            last_seen: "2026-01-01".into(),
        });
        // With only one positive entry, pick_voted should always return it
        let picked = ledger.pick_voted(0);
        assert!(picked.is_some());
        assert_eq!(picked.unwrap().to_str().unwrap(), "/tmp/test_genome.json");
    }

    #[test]
    fn vote_ledger_negative_score_not_pickable() {
        let mut ledger = VoteLedger::default();
        ledger.entries.insert("bad_genome".into(), VoteEntry {
            score: -5,
            file: "/tmp/bad.json".into(),
            last_seen: "2026-01-01".into(),
        });
        assert!(ledger.pick_voted(0).is_none());
    }

    #[test]
    fn vote_ledger_zero_score_not_pickable() {
        let mut ledger = VoteLedger::default();
        ledger.entries.insert("meh".into(), VoteEntry {
            score: 0,
            file: "/tmp/meh.json".into(),
            last_seen: "2026-01-01".into(),
        });
        assert!(ledger.pick_voted(0).is_none());
    }
```

**Step 2: Run tests**

Run: `cargo test --lib votes`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add src/votes.rs
git commit -m "test: add VoteLedger unit tests"
```

---

### Task 6: weights.rs tests

**Files:**
- Modify: `src/weights.rs` (add `#[cfg(test)] mod tests` at end)

**Step 1: Write tests**

Add at the end of `src/weights.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_sane_values() {
        let cfg = RuntimeConfig::default();
        assert!(cfg.morph_duration > 0.0);
        assert!(cfg.mutation_cooldown > 0.0);
        assert!(cfg.workgroups > 0);
        assert!(cfg.zoom_min > 0.0);
        assert!(cfg.zoom_max > cfg.zoom_min);
        assert!(cfg.max_mutation_retries > 0);
        assert!(cfg.min_breeding_distance > 0);
        assert!(cfg.max_lineage_depth > 0);
        assert!(cfg.taste_min_votes > 0);
        assert!(cfg.taste_candidates > 0);
        assert!(cfg.taste_recent_memory > 0);
    }

    #[test]
    fn config_serialization_roundtrip() {
        let cfg = RuntimeConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: RuntimeConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.morph_duration, cfg2.morph_duration);
        assert_eq!(cfg.workgroups, cfg2.workgroups);
        assert_eq!(cfg.min_breeding_distance, cfg2.min_breeding_distance);
        assert_eq!(cfg.taste_min_votes, cfg2.taste_min_votes);
    }

    #[test]
    fn config_partial_json_uses_defaults() {
        let json = r#"{"morph_duration": 5.0}"#;
        let cfg: RuntimeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.morph_duration, 5.0);
        // Everything else should be defaults
        assert_eq!(cfg.workgroups, 512);
        assert_eq!(cfg.min_breeding_distance, 3);
        assert_eq!(cfg.taste_engine_enabled, false);
    }

    #[test]
    fn config_empty_json_uses_all_defaults() {
        let cfg: RuntimeConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.morph_duration, 8.0);
        assert_eq!(cfg.gamma, 0.4545);
        assert_eq!(cfg.vote_blacklist_threshold, -2);
    }

    #[test]
    fn weights_empty_json_is_valid() {
        let json = r#"{"_config": {}}"#;
        let weights: Weights = serde_json::from_str(json).unwrap();
        assert!(weights.bass.is_empty());
        assert_eq!(weights._config.morph_duration, 8.0);
    }

    #[test]
    fn global_index_known_params() {
        assert_eq!(global_index("speed"), Some(0));
        assert_eq!(global_index("zoom"), Some(1));
        assert_eq!(global_index("gamma"), Some(11));
        assert_eq!(global_index("nonexistent"), None);
    }

    #[test]
    fn xf_field_index_known_fields() {
        assert_eq!(xf_field_index("weight"), Some(0));
        assert_eq!(xf_field_index("linear"), Some(8));
        assert_eq!(xf_field_index("spherical"), Some(10));
        assert_eq!(xf_field_index("fake_field"), None);
    }

    #[test]
    fn try_parse_xf_valid() {
        assert_eq!(try_parse_xf("xf0_weight"), Some((0, 0)));
        assert_eq!(try_parse_xf("xf3_linear"), Some((3, 8)));
        assert_eq!(try_parse_xf("xf10_spherical"), Some((10, 10)));
    }

    #[test]
    fn try_parse_xf_invalid() {
        assert_eq!(try_parse_xf("speed"), None);
        assert_eq!(try_parse_xf("xfN_weight"), None);
        assert_eq!(try_parse_xf("xf_weight"), None);
    }
}
```

**Step 2: Run tests**

Run: `cargo test --lib weights`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add src/weights.rs
git commit -m "test: add weights.rs config and parsing unit tests"
```

---

### Task 7: genome.rs tests

**Files:**
- Modify: `src/genome.rs` (make some private methods `pub(crate)`, add `#[cfg(test)] mod tests` at end)

**Step 1: Make necessary functions testable**

Change visibility in `src/genome.rs`:
- `fn normalize_variations(...)` → `pub(crate) fn normalize_variations(...)`
- `fn normalize_weights(...)` → `pub(crate) fn normalize_weights(...)`
- `fn distribute_colors(...)` → `pub(crate) fn distribute_colors(...)`
- `fn estimate_attractor_extent(...)` → `pub(crate) fn estimate_attractor_extent(...)`

**Step 2: Write the tests**

Add at the end of `src/genome.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_genome(n_transforms: usize) -> FlameGenome {
        let mut rng = rand::rng();
        let transforms: Vec<FlameTransform> = (0..n_transforms)
            .map(|_| FlameTransform::random_transform(&mut rng))
            .collect();
        FlameGenome {
            name: "test".into(),
            global: GlobalParams { speed: 0.25, zoom: 3.0, trail: 0.15, flame_brightness: 0.2 },
            kifs: KifsParams { fold_angle: 0.0, scale: 0.0, brightness: 0.0 },
            transforms,
            final_transform: None,
            symmetry: 1,
            palette: Some(generate_random_palette()),
            parent_a: None,
            parent_b: None,
            generation: 0,
        }
    }

    #[test]
    fn default_genome_has_transforms() {
        let g = FlameGenome::default_genome();
        assert!(!g.transforms.is_empty());
        assert!(g.palette.is_some());
    }

    #[test]
    fn normalize_weights_sums_to_one() {
        let mut g = make_test_genome(4);
        g.transforms[0].weight = 5.0;
        g.transforms[1].weight = 3.0;
        g.transforms[2].weight = 1.0;
        g.transforms[3].weight = 1.0;
        g.normalize_weights();
        let sum: f32 = g.transforms.iter().map(|t| t.weight).sum();
        assert!((sum - 1.0).abs() < 0.01, "weights should sum to 1.0, got {sum}");
    }

    #[test]
    fn normalize_variations_ensures_nonzero() {
        let mut g = make_test_genome(2);
        // Zero out all variations
        for xf in &mut g.transforms {
            *xf = FlameTransform::default();
            xf.weight = 0.5;
            xf.linear = 0.0;
        }
        g.normalize_variations();
        // Each transform should have at least one nonzero variation
        for (i, xf) in g.transforms.iter().enumerate() {
            let has_variation = (0..VARIATION_COUNT).any(|v| xf.get_variation(v) > 0.0);
            assert!(has_variation, "transform {i} should have at least one variation");
        }
    }

    #[test]
    fn distribute_colors_evenly_spaced() {
        let mut g = make_test_genome(4);
        g.distribute_colors();
        let expected = [0.0, 0.25, 0.5, 0.75];
        for (i, xf) in g.transforms.iter().enumerate() {
            assert!(
                (xf.color - expected[i]).abs() < 0.01,
                "transform {i} color should be {}, got {}",
                expected[i], xf.color
            );
        }
    }

    #[test]
    fn random_transform_has_valid_weight() {
        let mut rng = rand::rng();
        let xf = FlameTransform::random_transform(&mut rng);
        assert!(xf.weight > 0.0, "random transform should have positive weight");
    }

    #[test]
    fn random_transform_has_variation() {
        let mut rng = rand::rng();
        let xf = FlameTransform::random_transform(&mut rng);
        let has_variation = (0..VARIATION_COUNT).any(|v| xf.get_variation(v) > 0.0);
        assert!(has_variation, "random transform should have at least one variation");
    }

    #[test]
    fn breed_records_lineage() {
        let pa = make_test_genome(4);
        let pb = make_test_genome(3);
        let audio = crate::audio::AudioFeatures::default();
        let cfg = crate::weights::RuntimeConfig::default();
        let child = FlameGenome::breed(&pa, &pb, &None, &audio, &cfg, &None, &mut None);
        assert_eq!(child.parent_a.as_deref(), Some("test"));
        assert_eq!(child.parent_b.as_deref(), Some("test"));
        assert_eq!(child.generation, 1);
    }

    #[test]
    fn breed_transform_count_in_range() {
        let pa = make_test_genome(4);
        let pb = make_test_genome(4);
        let audio = crate::audio::AudioFeatures::default();
        let cfg = crate::weights::RuntimeConfig::default();
        // Run several times to test the clamped range
        for _ in 0..20 {
            let child = FlameGenome::breed(&pa, &pb, &None, &audio, &cfg, &None, &mut None);
            let n = child.transforms.len();
            assert!(n >= 3 && n <= 6, "transform count {n} out of range 3..=6");
        }
    }

    #[test]
    fn breed_always_has_palette() {
        let pa = make_test_genome(4);
        let pb = make_test_genome(3);
        let audio = crate::audio::AudioFeatures::default();
        let cfg = crate::weights::RuntimeConfig::default();
        let child = FlameGenome::breed(&pa, &pb, &None, &audio, &cfg, &None, &mut None);
        assert!(child.palette.is_some());
        assert_eq!(child.palette.as_ref().unwrap().len(), 256);
    }

    #[test]
    fn attractor_extent_default_genome() {
        let g = FlameGenome::default_genome();
        let extent = g.estimate_attractor_extent();
        assert!(extent > 0.0, "default genome should have positive extent, got {extent}");
    }

    #[test]
    fn genome_serialization_roundtrip() {
        let g = make_test_genome(3);
        let json = serde_json::to_string(&g).unwrap();
        let g2: FlameGenome = serde_json::from_str(&json).unwrap();
        assert_eq!(g.name, g2.name);
        assert_eq!(g.transforms.len(), g2.transforms.len());
        assert_eq!(g.symmetry, g2.symmetry);
        assert_eq!(g.generation, g2.generation);
    }
}
```

**Step 3: Run tests**

Run: `cargo test --lib genome`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add src/genome.rs
git commit -m "test: add genome.rs unit tests for breeding, normalization, transforms"
```

---

### Task 8: flam3.rs tests

**Files:**
- Modify: `src/flam3.rs` (add `#[cfg(test)] mod tests` at end)

**Step 1: Write the tests**

Add at the end of `src/flam3.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_FLAME: &str = r#"<flames>
<flame name="test_flame" symmetry="2">
    <xform weight="0.5" color="0.0" coefs="1 0 0 1 0 0" spherical="1.0"/>
    <xform weight="0.5" color="0.5" coefs="0.5 0.5 -0.5 0.5 0.1 0.2" linear="0.6" sinusoidal="0.4"/>
</flame>
</flames>"#;

    #[test]
    fn parse_minimal_flame() {
        let file = Flam3File::parse(MINIMAL_FLAME).unwrap();
        assert_eq!(file.flames.len(), 1);
        let flame = &file.flames[0];
        assert_eq!(flame.name, "test_flame");
        assert_eq!(flame.symmetry, 2);
    }

    #[test]
    fn parse_transforms() {
        let file = Flam3File::parse(MINIMAL_FLAME).unwrap();
        let flame = &file.flames[0];
        assert_eq!(flame.transforms.len(), 2);

        // First transform: spherical=1.0
        assert_eq!(flame.transforms[0].spherical, 1.0);
        assert_eq!(flame.transforms[0].weight, 0.5);
        assert_eq!(flame.transforms[0].color, 0.0);

        // Second transform: linear=0.6, sinusoidal=0.4
        assert!((flame.transforms[1].linear - 0.6).abs() < 0.01);
        assert!((flame.transforms[1].sinusoidal - 0.4).abs() < 0.01);
    }

    #[test]
    fn parse_affine_coefs() {
        let file = Flam3File::parse(MINIMAL_FLAME).unwrap();
        let xf = &file.flames[0].transforms[0];
        // coefs="1 0 0 1 0 0" → identity affine
        // flam3 layout: a d b e c f (column-major)
        // a=1, d=0, b=0, e=1, c=0, f=0
        assert!((xf.a - 1.0).abs() < 0.01);
        assert!((xf.d - 1.0).abs() < 0.01);
    }

    #[test]
    fn parse_palette_hex() {
        let xml = r#"<flames>
<flame name="pal_test">
    <xform weight="1.0" color="0.0" coefs="1 0 0 1 0 0" linear="1.0"/>
    <palette count="256">
FF0000FF000000FF00
    </palette>
</flame>
</flames>"#;
        let file = Flam3File::parse(xml).unwrap();
        let flame = &file.flames[0];
        assert!(flame.palette.is_some());
        let palette = flame.palette.as_ref().unwrap();
        assert!(!palette.is_empty());
        // First color should be red (FF0000)
        assert!((palette[0][0] - 1.0).abs() < 0.01, "red channel should be 1.0, got {}", palette[0][0]);
        assert!((palette[0][1] - 0.0).abs() < 0.01, "green channel should be 0.0");
    }

    #[test]
    fn parse_empty_xml() {
        let file = Flam3File::parse("<flames></flames>").unwrap();
        assert!(file.flames.is_empty());
    }

    #[test]
    fn parse_lineage_defaults() {
        let file = Flam3File::parse(MINIMAL_FLAME).unwrap();
        let flame = &file.flames[0];
        assert!(flame.parent_a.is_none());
        assert!(flame.parent_b.is_none());
        assert_eq!(flame.generation, 0);
    }
}
```

**Step 2: Run tests**

Run: `cargo test --lib flam3`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add src/flam3.rs
git commit -m "test: add flam3.rs XML parsing unit tests"
```

---

### Task 9: Final verification

**Step 1: Run all tests**

Run: `cargo test`
Expected: All tests pass (audio tests + all new tests).

**Step 2: Run the pre-commit hook**

Run: `.git/hooks/pre-commit` (or `scripts/pre-commit`)
Expected: All three checks pass (fmt, clippy, test).

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: all quality gates passing — fmt, clippy, 50+ tests"
```

**Step 4: Tag**

```bash
git tag -a v0.3.2-tested -m "Quality gates: pre-commit hook + unit tests across all logic modules"
```
