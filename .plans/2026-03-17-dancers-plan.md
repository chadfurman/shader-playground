# Dancers + Session Persistence Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ephemeral localized transforms ("dancers") that spawn during morphs, wander with audio reactivity, and dissolve — plus save/restore the current genome on exit.

**Architecture:** New `src/dancer.rs` module owns all dancer logic (DancerManager, breeding, bell curve, archive persistence). App holds a DancerManager and calls it during begin_morph (schedule), per-frame (tick), and FrameData construction (append transforms). Session persistence is a small addition to App's startup/shutdown paths.

**Tech Stack:** Rust, serde_json for archive/session persistence, existing FlameTransform and interpolate_or_swap from genome.rs

---

## File Structure

- **Create:** `src/dancer.rs` — DancerManager, Dancer, DancerAudioWeights, breeding, bell curve, archive persistence
- **Modify:** `src/weights.rs` — Add 12 dancer config fields to RuntimeConfig
- **Modify:** `src/main.rs` — Wire DancerManager into App (begin_morph, per-frame tick, FrameData, buffer sizing, session persistence, shutdown save)
- **Modify:** `src/genome.rs` — Make `push_transform` pub so dancer.rs can flatten transforms
- **Modify:** `weights.json` — Add dancer config values + session persistence docs

---

## Chunk 1: Core Dancer Module

### Task 1: Dancer config fields in RuntimeConfig

**Files:**
- Modify: `src/weights.rs`
- Modify: `weights.json`

- [ ] **Step 1: Add dancer config fields to RuntimeConfig**

Add before the closing `}` of RuntimeConfig (after `transform_count_max`):

```rust
    // Dancers
    #[serde(default = "default_dancer_enabled")]
    pub dancer_enabled: bool,
    #[serde(default = "default_dancer_count_min")]
    pub dancer_count_min: u32,
    #[serde(default = "default_dancer_count_max")]
    pub dancer_count_max: u32,
    #[serde(default = "default_dancer_lifetime_min")]
    pub dancer_lifetime_min: f32,
    #[serde(default = "default_dancer_lifetime_max")]
    pub dancer_lifetime_max: f32,
    #[serde(default = "default_dancer_scale_min")]
    pub dancer_scale_min: f32,
    #[serde(default = "default_dancer_scale_max")]
    pub dancer_scale_max: f32,
    #[serde(default = "default_dancer_offset_min")]
    pub dancer_offset_min: f32,
    #[serde(default = "default_dancer_offset_max")]
    pub dancer_offset_max: f32,
    #[serde(default = "default_dancer_drift_speed")]
    pub dancer_drift_speed: f32,
    #[serde(default = "default_dancer_audio_strength")]
    pub dancer_audio_strength: f32,
    #[serde(default = "default_dancer_archive_size")]
    pub dancer_archive_size: usize,
    #[serde(default = "default_dancer_fade_fraction")]
    pub dancer_fade_fraction: f32,
```

Add default functions:

```rust
fn default_dancer_enabled() -> bool { true }
fn default_dancer_count_min() -> u32 { 1 }
fn default_dancer_count_max() -> u32 { 5 }
fn default_dancer_lifetime_min() -> f32 { 4.0 }
fn default_dancer_lifetime_max() -> f32 { 8.0 }
fn default_dancer_scale_min() -> f32 { 0.05 }
fn default_dancer_scale_max() -> f32 { 0.2 }
fn default_dancer_offset_min() -> f32 { 1.0 }
fn default_dancer_offset_max() -> f32 { 3.0 }
fn default_dancer_drift_speed() -> f32 { 0.3 }
fn default_dancer_audio_strength() -> f32 { 0.5 }
fn default_dancer_archive_size() -> usize { 20 }
fn default_dancer_fade_fraction() -> f32 { 0.2 }
```

- [ ] **Step 2: Add dancer config to weights.json `_config` and `_config_doc`**

Add to `_config_doc`:
```json
"dancer_enabled": "Enable ephemeral dancer transforms (default true)",
"dancer_count_min": "Min dancers spawned per morph (default 1)",
"dancer_count_max": "Max dancers spawned per morph (default 5)",
"dancer_lifetime_min": "Min dancer lifetime in seconds (default 4)",
"dancer_lifetime_max": "Max dancer lifetime in seconds (default 8)",
"dancer_scale_min": "Min affine scale for dancers — smaller = more localized (default 0.05)",
"dancer_scale_max": "Max affine scale for dancers (default 0.2)",
"dancer_offset_min": "Min spawn distance from origin (default 1.0)",
"dancer_offset_max": "Max spawn distance from origin (default 3.0)",
"dancer_drift_speed": "How fast dancers wander (default 0.3)",
"dancer_audio_strength": "Max per-dancer audio weight magnitude (default 0.5)",
"dancer_archive_size": "Ring buffer size for dancer lineage (default 20)",
"dancer_fade_fraction": "Fraction of lifetime spent fading in/out (default 0.2)"
```

Add to `_config`:
```json
"dancer_enabled": true,
"dancer_count_min": 1,
"dancer_count_max": 5,
"dancer_lifetime_min": 4.0,
"dancer_lifetime_max": 8.0,
"dancer_scale_min": 0.05,
"dancer_scale_max": 0.2,
"dancer_offset_min": 1.0,
"dancer_offset_max": 3.0,
"dancer_drift_speed": 0.3,
"dancer_audio_strength": 0.5,
"dancer_archive_size": 20,
"dancer_fade_fraction": 0.2
```

- [ ] **Step 3: Build + test**

Run: `cargo test 2>&1 | tail -5`
Expected: All pass (config fields default correctly).

- [ ] **Step 4: Commit**

```bash
git add src/weights.rs weights.json
git commit -m "feat: add dancer config fields to RuntimeConfig"
```

---

### Task 2: Bell curve weight function + tests

**Files:**
- Create: `src/dancer.rs`
- Modify: `src/main.rs` (add `mod dancer;`)

- [ ] **Step 1: Write bell curve tests**

Create `src/dancer.rs` with tests first:

```rust
use crate::audio::AudioFeatures;
use crate::genome::FlameTransform;
use crate::weights::RuntimeConfig;
use std::collections::VecDeque;

/// Compute bell curve weight for a dancer at a given point in its lifetime.
/// Returns 0.0 at birth, ramps to 1.0 over fade_fraction of lifetime,
/// holds, then ramps back to 0.0 over the final fade_fraction.
fn bell_curve_weight(elapsed: f32, lifetime: f32, fade_fraction: f32) -> f32 {
    if lifetime <= 0.0 { return 0.0; }
    let t = (elapsed / lifetime).clamp(0.0, 1.0);
    let fade = fade_fraction.clamp(0.01, 0.49);
    if t < fade {
        t / fade
    } else if t > 1.0 - fade {
        (1.0 - t) / fade
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bell_curve_at_zero_is_zero() {
        assert_eq!(bell_curve_weight(0.0, 6.0, 0.2), 0.0);
    }

    #[test]
    fn bell_curve_at_fade_in_end_is_one() {
        // fade_fraction=0.2, lifetime=10 → fade-in ends at t=0.2 (2s)
        let w = bell_curve_weight(2.0, 10.0, 0.2);
        assert!((w - 1.0).abs() < 0.01);
    }

    #[test]
    fn bell_curve_at_midpoint_is_one() {
        assert_eq!(bell_curve_weight(3.0, 6.0, 0.2), 1.0);
    }

    #[test]
    fn bell_curve_at_fade_out_start_is_one() {
        // fade_fraction=0.2, lifetime=10 → fade-out starts at t=0.8 (8s)
        let w = bell_curve_weight(8.0, 10.0, 0.2);
        assert!((w - 1.0).abs() < 0.01);
    }

    #[test]
    fn bell_curve_at_end_is_zero() {
        assert_eq!(bell_curve_weight(6.0, 6.0, 0.2), 0.0);
    }

    #[test]
    fn bell_curve_at_halfway_fade_in() {
        // t=0.1 with fade=0.2 → 0.1/0.2 = 0.5
        let w = bell_curve_weight(1.0, 10.0, 0.2);
        assert!((w - 0.5).abs() < 0.01);
    }
}
```

- [ ] **Step 2: Add `mod dancer;` to main.rs**

Add after `mod weights;` in main.rs:
```rust
mod dancer;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test dancer 2>&1 | tail -10`
Expected: 6 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/dancer.rs src/main.rs
git commit -m "feat: add dancer module with bell curve weight function + tests"
```

---

### Task 3: DancerAudioWeights, Dancer struct, DancerManager

**Files:**
- Modify: `src/dancer.rs`

- [ ] **Step 1: Write archive ring buffer tests**

Add to tests module:

```rust
    #[test]
    fn archive_respects_capacity() {
        let mut mgr = DancerManager::new();
        let cap = 5;
        for i in 0..10 {
            let mut xf = FlameTransform::default();
            xf.weight = i as f32;
            mgr.archive_transform(xf, cap);
        }
        assert_eq!(mgr.archive.len(), cap);
        // Oldest should be evicted — first remaining is #5
        assert_eq!(mgr.archive.front().unwrap().weight, 5.0);
    }

    #[test]
    fn archive_empty_pick_returns_none() {
        let mgr = DancerManager::new();
        assert!(mgr.pick_archive_transform().is_none());
    }

    #[test]
    fn archive_pick_returns_some() {
        let mut mgr = DancerManager::new();
        mgr.archive_transform(FlameTransform::default(), 20);
        assert!(mgr.pick_archive_transform().is_some());
    }
```

- [ ] **Step 2: Implement structs and archive methods**

Add to `src/dancer.rs` (above tests module):

```rust
use rand::Rng;
use rand::prelude::{IndexedRandom, SliceRandom};
use serde::{Deserialize, Serialize};

pub struct DancerAudioWeights {
    pub drift_bass: f32,
    pub drift_mids: f32,
    pub drift_highs: f32,
    pub pulse_energy: f32,
    pub pulse_beat: f32,
}

pub struct Dancer {
    pub transform: FlameTransform,
    pub birth_time: f32,
    pub lifetime: f32,
    pub audio_weights: DancerAudioWeights,
    pub generation: u32,
}

pub struct DancerManager {
    pub active: Vec<Dancer>,
    pub scheduled: Vec<f32>,  // morph_progress trigger values
    pub archive: VecDeque<FlameTransform>,
}

impl DancerManager {
    pub fn new() -> Self {
        Self {
            active: Vec::new(),
            scheduled: Vec::new(),
            archive: VecDeque::new(),
        }
    }

    pub fn archive_transform(&mut self, xf: FlameTransform, capacity: usize) {
        if self.archive.len() >= capacity {
            self.archive.pop_front();
        }
        self.archive.push_back(xf);
    }

    pub fn pick_archive_transform(&self) -> Option<FlameTransform> {
        if self.archive.is_empty() {
            return None;
        }
        let mut rng = rand::rng();
        self.archive.make_contiguous().choose(&mut rng).cloned()
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test dancer 2>&1 | tail -10`
Expected: 9 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/dancer.rs
git commit -m "feat: DancerManager with archive ring buffer + tests"
```

---

### Task 4: Dancer breeding

**Files:**
- Modify: `src/dancer.rs`
- Modify: `src/genome.rs` (make `push_transform` pub)

- [ ] **Step 1: Make `FlameGenome::push_transform` public**

In `src/genome.rs`, change:
```rust
    fn push_transform(t: &mut Vec<f32>, xf: &FlameTransform) {
```
to:
```rust
    pub fn push_transform(t: &mut Vec<f32>, xf: &FlameTransform) {
```

- [ ] **Step 2: Write breeding tests**

Add to dancer tests:

```rust
    #[test]
    fn breed_dancer_from_genome_only() {
        let cfg = RuntimeConfig::default();
        let mut mgr = DancerManager::new();
        let genome_xfs = vec![
            FlameTransform::random_transform(&mut rand::rng()),
            FlameTransform::random_transform(&mut rand::rng()),
            FlameTransform::random_transform(&mut rand::rng()),
        ];
        let dancer = mgr.breed_dancer(&genome_xfs, &cfg);
        // Dancer should have small scale (det of affine)
        let det = dancer.transform.affine[0][0] * dancer.transform.affine[1][1]
            - dancer.transform.affine[0][1] * dancer.transform.affine[1][0];
        assert!(det.abs() < cfg.dancer_scale_max + 0.1);
        // Dancer offset should be pushed out
        let dist = (dancer.transform.offset[0].powi(2) + dancer.transform.offset[1].powi(2)).sqrt();
        assert!(dist >= cfg.dancer_offset_min * 0.5);
    }

    #[test]
    fn breed_dancer_uses_archive_when_available() {
        let cfg = RuntimeConfig::default();
        let mut mgr = DancerManager::new();
        // Add a distinctive transform to archive
        let mut archived = FlameTransform::default();
        archived.color = 0.99;
        mgr.archive_transform(archived, 20);
        let genome_xfs = vec![FlameTransform::random_transform(&mut rand::rng())];
        // Breed many times — archive influence should appear
        let mut saw_high_color = false;
        for _ in 0..20 {
            let d = mgr.breed_dancer(&genome_xfs, &cfg);
            if d.transform.color > 0.7 { saw_high_color = true; }
        }
        assert!(saw_high_color, "archive transform color should influence some dancers");
    }
```

- [ ] **Step 3: Implement breed_dancer**

Add to `impl DancerManager`:

```rust
    /// Breed a new dancer from genome transforms + archive.
    /// Returns a Dancer with small scale, pushed offset, and random audio weights.
    pub fn breed_dancer(
        &self,
        genome_xfs: &[FlameTransform],
        cfg: &RuntimeConfig,
    ) -> Dancer {
        use crate::genome::interpolate_or_swap;
        let mut rng = rand::rng();

        // Pick 2-3 source transforms from genome + archive
        let mut sources: Vec<FlameTransform> = Vec::new();
        // 1-2 from genome
        let n_genome = rng.random_range(1..=2usize).min(genome_xfs.len());
        let mut indices: Vec<usize> = (0..genome_xfs.len()).collect();
        indices.shuffle(&mut rng);
        for &i in indices.iter().take(n_genome) {
            sources.push(genome_xfs[i].clone());
        }
        // 0-1 from archive (if available)
        if let Some(archived) = self.pick_archive_transform() {
            sources.push(archived);
        }

        // Blend sources via lerp
        let mut result = sources[0].clone();
        for src in &sources[1..] {
            let t: f32 = rng.random_range(0.3..=0.7);
            result = result.lerp_with(src, t);
        }

        // Clamp scale small
        let target_scale: f32 = rng.random_range(cfg.dancer_scale_min..=cfg.dancer_scale_max);
        let det = (result.affine[0][0] * result.affine[1][1]
            - result.affine[0][1] * result.affine[1][0])
            .abs()
            .max(0.001);
        let scale_factor = (target_scale / det.sqrt()).min(1.0);
        result.affine[0][0] *= scale_factor;
        result.affine[0][1] *= scale_factor;
        result.affine[1][0] *= scale_factor;
        result.affine[1][1] *= scale_factor;

        // Push offset away from origin
        let angle: f32 = rng.random_range(0.0..std::f32::consts::TAU);
        let radius: f32 = rng.random_range(cfg.dancer_offset_min..=cfg.dancer_offset_max);
        result.offset[0] = angle.cos() * radius;
        result.offset[1] = angle.sin() * radius;

        // Mutate color slightly
        result.color = (result.color + rng.random_range(-0.15..0.15)).clamp(0.0, 1.0);

        // Random audio weights
        let s = cfg.dancer_audio_strength;
        let audio_weights = DancerAudioWeights {
            drift_bass: rng.random_range(-s..=s),
            drift_mids: rng.random_range(-s..=s),
            drift_highs: rng.random_range(-s..=s),
            pulse_energy: rng.random_range(0.0..=s),
            pulse_beat: rng.random_range(0.0..=s),
        };

        // Dancer generation from archive depth
        let generation = self.archive.len() as u32;

        let lifetime: f32 = rng.random_range(cfg.dancer_lifetime_min..=cfg.dancer_lifetime_max);

        Dancer {
            transform: result,
            birth_time: 0.0, // set at actual spawn time
            lifetime,
            audio_weights,
            generation,
        }
    }
```

**Note:** `lerp_with` is already pub on FlameTransform and simpler than `interpolate_or_swap` for blending arbitrary transforms.

- [ ] **Step 4: Run tests**

Run: `cargo test dancer 2>&1 | tail -10`
Expected: 11 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/dancer.rs src/genome.rs
git commit -m "feat: dancer breeding from genome transforms + archive"
```

---

### Task 5: Scheduling + tick (per-frame update)

**Files:**
- Modify: `src/dancer.rs`

- [ ] **Step 1: Write scheduling tests**

```rust
    #[test]
    fn schedule_dancers_creates_valid_triggers() {
        let cfg = RuntimeConfig::default();
        let mut mgr = DancerManager::new();
        let genome_xfs = vec![FlameTransform::random_transform(&mut rand::rng())];
        mgr.schedule_dancers(&genome_xfs, &cfg, 2.5); // morph_done_at = 2.5
        assert!(!mgr.scheduled.is_empty());
        assert!(mgr.scheduled.len() <= cfg.dancer_count_max as usize);
        for &trigger in &mgr.scheduled {
            assert!(trigger >= 0.0);
            assert!(trigger <= 2.5);
        }
    }

    #[test]
    fn tick_spawns_at_correct_progress() {
        let cfg = RuntimeConfig::default();
        let mut mgr = DancerManager::new();
        let genome_xfs = vec![FlameTransform::random_transform(&mut rand::rng())];
        // Manually set a schedule trigger at progress 0.5
        mgr.scheduled = vec![0.5];
        let audio = AudioFeatures::default();
        // Before trigger: no spawn
        mgr.tick(0.3, 1.0, 0.016, &audio, &genome_xfs, &cfg);
        assert_eq!(mgr.active.len(), 0);
        // After trigger: spawn
        mgr.tick(0.6, 1.1, 0.016, &audio, &genome_xfs, &cfg);
        assert_eq!(mgr.active.len(), 1);
        assert!(mgr.scheduled.is_empty());
    }

    #[test]
    fn tick_removes_expired_dancers() {
        let cfg = RuntimeConfig::default();
        let mut mgr = DancerManager::new();
        let genome_xfs = vec![FlameTransform::random_transform(&mut rand::rng())];
        // Manually insert a dancer that's already expired
        mgr.active.push(Dancer {
            transform: FlameTransform::default(),
            birth_time: 0.0,
            lifetime: 1.0,
            audio_weights: DancerAudioWeights {
                drift_bass: 0.0, drift_mids: 0.0, drift_highs: 0.0,
                pulse_energy: 0.0, pulse_beat: 0.0,
            },
            generation: 0,
        });
        let audio = AudioFeatures::default();
        mgr.tick(0.0, 5.0, 0.016, &audio, &genome_xfs, &cfg); // time=5, lifetime=1 → expired
        assert_eq!(mgr.active.len(), 0);
        assert_eq!(mgr.archive.len(), 1); // archived on death
    }
```

- [ ] **Step 2: Implement schedule_dancers and tick**

```rust
    /// Schedule dancers for the current morph.
    /// morph_done_at is the morph_progress value where the slowest transform finishes.
    pub fn schedule_dancers(
        &mut self,
        genome_xfs: &[FlameTransform],
        cfg: &RuntimeConfig,
        morph_done_at: f32,
    ) {
        if !cfg.dancer_enabled || genome_xfs.is_empty() || morph_done_at <= 0.0 {
            return;
        }
        let mut rng = rand::rng();
        let count = rng.random_range(cfg.dancer_count_min..=cfg.dancer_count_max) as usize;
        self.scheduled.clear();
        for _ in 0..count {
            let trigger: f32 = rng.random_range(0.0..=morph_done_at);
            self.scheduled.push(trigger);
        }
        self.scheduled.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    /// Per-frame update. Spawns scheduled dancers, updates active dancers, archives dead ones.
    pub fn tick(
        &mut self,
        morph_progress: f32,
        current_time: f32,
        dt: f32,
        audio: &AudioFeatures,
        genome_xfs: &[FlameTransform],
        cfg: &RuntimeConfig,
    ) {
        // Spawn any dancers whose trigger has been reached
        while let Some(&trigger) = self.scheduled.first() {
            if morph_progress >= trigger {
                self.scheduled.remove(0);
                let mut dancer = self.breed_dancer(genome_xfs, cfg);
                dancer.birth_time = current_time;
                self.active.push(dancer);
            } else {
                break;
            }
        }

        // Update active dancers
        let archive_cap = cfg.dancer_archive_size;
        let mut i = 0;
        while i < self.active.len() {
            let d = &mut self.active[i];
            let elapsed = current_time - d.birth_time;
            if elapsed >= d.lifetime {
                // Archive and remove
                let dead = self.active.remove(i);
                self.archive_transform(dead.transform, archive_cap);
            } else {
                // Apply audio drift to offset
                let drift = cfg.dancer_drift_speed;
                d.transform.offset[0] += (d.audio_weights.drift_bass * audio.bass
                    + d.audio_weights.drift_mids * audio.mids
                    + d.audio_weights.drift_highs * audio.highs)
                    * drift
                    * dt;
                d.transform.offset[1] += (d.audio_weights.drift_bass * audio.bass
                    - d.audio_weights.drift_mids * audio.mids
                    + d.audio_weights.drift_highs * audio.highs)
                    * drift
                    * dt;
                i += 1;
            }
        }
    }

    /// Flatten all active dancers into a transform buffer.
    /// Sets each dancer's weight to the bell curve value.
    pub fn flatten_active(&self, current_time: f32, cfg: &RuntimeConfig) -> Vec<f32> {
        let mut buf = Vec::new();
        for d in &self.active {
            let elapsed = current_time - d.birth_time;
            let weight = bell_curve_weight(elapsed, d.lifetime, cfg.dancer_fade_fraction);
            let mut xf = d.transform.clone();
            xf.weight = weight * xf.weight.max(0.01);
            // Apply energy/beat pulse to scale
            // (omitted from initial impl — can add in follow-up)
            crate::genome::FlameGenome::push_transform(&mut buf, &xf);
        }
        buf
    }

    pub fn active_count(&self) -> usize {
        self.active.len()
    }
```

- [ ] **Step 3: Add buffer sizing invariant test**

```rust
    #[test]
    fn flatten_active_fits_prealloc_budget() {
        let cfg = RuntimeConfig::default();
        let mut mgr = DancerManager::new();
        let genome_xfs = vec![FlameTransform::random_transform(&mut rand::rng()); 12];
        // Spawn max dancers
        for _ in 0..cfg.dancer_count_max {
            let mut d = mgr.breed_dancer(&genome_xfs, &cfg);
            d.birth_time = 0.0;
            mgr.active.push(d);
        }
        let flat = mgr.flatten_active(2.0, &cfg);
        let max_floats = (cfg.transform_count_max as usize + cfg.dancer_count_max as usize) * 48;
        assert!(
            flat.len() <= max_floats,
            "dancer buffer {} exceeds pre-alloc budget {}",
            flat.len(),
            max_floats
        );
    }
```

- [ ] **Step 4: Run tests**

Run: `cargo test dancer 2>&1 | tail -10`
Expected: 15 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/dancer.rs
git commit -m "feat: dancer scheduling, per-frame tick, and flatten"
```

---

### Task 6: Archive persistence (save/load)

**Files:**
- Modify: `src/dancer.rs`

- [ ] **Step 1: Write persistence test**

```rust
    #[test]
    fn archive_save_load_roundtrip() {
        let mut mgr = DancerManager::new();
        for i in 0..5 {
            let mut xf = FlameTransform::default();
            xf.color = i as f32 * 0.2;
            mgr.archive_transform(xf, 20);
        }
        let dir = std::env::temp_dir().join("dancer_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("dancer_archive.json");
        mgr.save_archive(&path).unwrap();

        let mut mgr2 = DancerManager::new();
        mgr2.load_archive(&path).unwrap();
        assert_eq!(mgr2.archive.len(), 5);
        assert!((mgr2.archive[2].color - 0.4).abs() < 0.01);
        // Verify affine survives roundtrip
        assert!((mgr2.archive[0].affine[0][0] - 1.0).abs() < 0.01); // default identity
        // Verify variation data survives
        assert_eq!(mgr2.archive[0].linear, 0.0); // default
        let _ = std::fs::remove_dir_all(&dir);
    }
```

- [ ] **Step 2: Implement save/load**

```rust
    pub fn save_archive(&self, path: &std::path::Path) -> Result<(), String> {
        let data: Vec<&FlameTransform> = self.archive.iter().collect();
        let json = serde_json::to_string_pretty(&data).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    pub fn load_archive(&mut self, path: &std::path::Path) -> Result<(), String> {
        let json = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let data: Vec<FlameTransform> = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        self.archive = data.into_iter().collect();
        Ok(())
    }
```

- [ ] **Step 3: Run tests**

Run: `cargo test dancer 2>&1 | tail -10`
Expected: 15 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/dancer.rs
git commit -m "feat: dancer archive persistence (save/load JSON)"
```

---

## Chunk 2: Wire Into App + Session Persistence

### Task 7: Wire DancerManager into App

**Files:**
- Modify: `src/main.rs`

This task connects the dancer module to the app. Key integration points:
1. Add `dancer_manager: DancerManager` to App struct
2. Call `schedule_dancers` in `begin_morph`
3. Call `tick` in RedrawRequested
4. Append dancer transforms to FrameData
5. Update `transform_count` in uniforms to include dancers
6. Account for dancers in buffer pre-allocation

- [ ] **Step 1: Add DancerManager to App struct + initialization**

In App struct, add:
```rust
    dancer_manager: crate::dancer::DancerManager,
```

In App::new(), add to the initialization:
```rust
            dancer_manager: crate::dancer::DancerManager::new(),
```

- [ ] **Step 2: Load dancer archive on startup**

In `resumed()`, after taste model rebuild, add:
```rust
        let archive_path = project_dir().join("genomes").join("dancer_archive.json");
        if archive_path.exists() {
            match self.dancer_manager.load_archive(&archive_path) {
                Ok(()) => eprintln!("[dancers] archive loaded ({} entries)", self.dancer_manager.archive.len()),
                Err(e) => eprintln!("[dancers] archive load error: {e}"),
            }
        }
```

- [ ] **Step 3: Schedule dancers in begin_morph**

At the **very end** of `begin_morph()` — AFTER `self.morph_xf_rates` is populated (after the stagger assignment block and `self.morph_burst_frames = 60`) and AFTER the palette upload. This is critical because `morph_xf_rates` must be fully set before we compute `morph_done_at`:

```rust
        // Schedule dancers for this morph (must be after morph_xf_rates is populated)
        let min_rate = self.morph_xf_rates.iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0)
            .max(0.1);
        let morph_done_at = 1.0 / min_rate;
        self.dancer_manager.schedule_dancers(
            &self.genome.transforms,
            &self.weights._config,
            morph_done_at,
        );
```

- [ ] **Step 4: Tick dancers in RedrawRequested**

In the RedrawRequested handler, after the morph/audio modulation block but before building FrameData, add:
```rust
                // Tick dancers
                let current_time = self.start.elapsed().as_secs_f32();
                self.dancer_manager.tick(
                    self.morph_progress,
                    current_time,
                    dt,
                    &self.audio_features,
                    &self.genome.transforms,
                    &self.weights._config,
                );
```

- [ ] **Step 5: Append dancer transforms to FrameData**

In the FrameData construction, after building `final_xf_params` (the Jacobian-adjusted genome transforms), append dancers:
```rust
                let dancer_xf = self.dancer_manager.flatten_active(
                    self.start.elapsed().as_secs_f32(),
                    &self.weights._config,
                );
                let active_dancer_count = self.dancer_manager.active_count();
                let mut combined_xf = final_xf_params;
                combined_xf.extend_from_slice(&dancer_xf);
```

Update `transform_count` in the **initial `uniforms` struct literal** (the one that feeds into `final_uniforms` via the adaptive workgroup patch). This is the `Uniforms { ... }` block that starts with `time: self.start.elapsed()...`. Change the `transform_count` field from:
```rust
                    transform_count: self.genome.transform_count(),
```
to:
```rust
                    transform_count: self.genome.transform_count() + active_dancer_count as u32,
```
This must happen AFTER the `dancer_manager.tick()` call (Step 4) so `active_dancer_count` is current. Compute `active_dancer_count` before the uniforms literal:
```rust
                let active_dancer_count = self.dancer_manager.active_count();
```

Use `combined_xf` instead of `final_xf_params` in the FrameData:
```rust
                        xf_params: combined_xf,
```

- [ ] **Step 6: Pre-allocate buffer for dancers**

Two resize paths need dancer headroom:

**In `begin_morph()`** — the `ResizeTransformBuffer` channel send:
```rust
let buffer_transforms = self.num_transforms + self.weights._config.dancer_count_max as usize;
// send ResizeTransformBuffer(buffer_transforms) instead of self.num_transforms
```

**In `resumed()`** — the direct `gpu.resize_transform_buffer()` call before spawning the render thread:
```rust
let buffer_transforms = self.num_transforms + self.weights._config.dancer_count_max as usize;
gpu.resize_transform_buffer(buffer_transforms);
```

**Important:** `self.num_transforms` on App continues to track genome-only count. The Jacobian loop in RedrawRequested uses `self.num_transforms` to bound its iteration — this is correct because Jacobian should only apply to genome transforms, not dancers. Dancers are appended AFTER the Jacobian-adjusted buffer.

- [ ] **Step 7: Build + test**

Run: `cargo build 2>&1 | tail -5 && cargo test 2>&1 | tail -5`
Expected: Clean build, all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/main.rs
git commit -m "feat: wire DancerManager into App — scheduling, tick, buffer append"
```

---

### Task 8: Session persistence

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Save session on shutdown**

In `WindowEvent::CloseRequested`, before sending `Shutdown`, add:
```rust
                // Save session state
                let genomes_dir = project_dir().join("genomes");
                let session = serde_json::json!({ "genome_name": self.genome.name });
                if let Err(e) = std::fs::write(
                    genomes_dir.join("session.json"),
                    serde_json::to_string_pretty(&session).unwrap_or_default(),
                ) {
                    eprintln!("[session] save error: {e}");
                }
                // Save dancer archive
                if let Err(e) = self.dancer_manager.save_archive(
                    &genomes_dir.join("dancer_archive.json"),
                ) {
                    eprintln!("[dancers] archive save error: {e}");
                }
                eprintln!("[session] saved genome={}", self.genome.name);
```

- [ ] **Step 2: Load session on startup**

In `resumed()`, replace the random genome loading block with:
```rust
        let genomes_dir = project_dir().join("genomes");
        let session_genome = std::fs::read_to_string(genomes_dir.join("session.json"))
            .ok()
            .and_then(|json| serde_json::from_str::<serde_json::Value>(&json).ok())
            .and_then(|v| v.get("genome_name")?.as_str().map(String::from))
            .and_then(|name| {
                // Try genomes/, then genomes/history/, then genomes/voted/
                for subdir in ["", "history", "voted"] {
                    let dir = if subdir.is_empty() { genomes_dir.clone() } else { genomes_dir.join(subdir) };
                    let path = dir.join(format!("{name}.json"));
                    if path.exists() {
                        if let Ok(g) = FlameGenome::load(&path) {
                            return Some(g);
                        }
                    }
                }
                None
            });
        if let Some(mut g) = session_genome {
            g.adjust_transform_count(&self.weights._config);
            eprintln!("[session] resumed: {}", g.name);
            self.genome = g;
            let g_globals = self.genome.flatten_globals(&self.weights._config);
            let g_xf = self.genome.flatten_transforms();
            self.globals = g_globals;
            self.xf_params = g_xf.clone();
            self.morph_base_globals = g_globals;
            self.morph_base_xf = g_xf.clone();
            self.morph_start_globals = g_globals;
            self.morph_start_xf = g_xf;
            self.morph_progress = 1.0;
            self.num_transforms = self.genome.total_buffer_transforms();
        } else if genomes_dir.exists() {
            // Fall back to random load (existing code stays as-is)
        }
```

- [ ] **Step 3: Periodic dancer archive save**

In the RedrawRequested handler, after the perf model save block, add:
```rust
                // Save dancer archive periodically (~30s)
                if self.frame.is_multiple_of(1800) {
                    let archive_path = project_dir().join("genomes").join("dancer_archive.json");
                    let _ = self.dancer_manager.save_archive(&archive_path);
                }
```

- [ ] **Step 4: Build + test**

Run: `cargo build 2>&1 | tail -5 && cargo test 2>&1 | tail -5`
Expected: Clean build, all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/main.rs
git commit -m "feat: session persistence — save/restore genome on exit + periodic dancer archive"
```

---

### Task 9: Remove debug logs + final verification

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Remove debug logs from render thread work**

Remove all `[debug]` eprintln lines added during render thread development.

- [ ] **Step 2: Build + clippy + test**

Run: `cargo clippy -- -D warnings 2>&1 | tail -5 && cargo test 2>&1 | tail -5`
Expected: Clean build, no warnings, all tests pass.

- [ ] **Step 3: Visual verification**

Run: `cargo run --release`
Verify:
- Dancers appear as small localized phenomena during morphs
- Dancers fade in, drift, fade out
- Multiple dancers active simultaneously
- Spacebar/arrows/save still work
- `[dancers] archive loaded` appears on restart
- `[session] resumed: <name>` appears on restart
- `[session] saved genome=<name>` appears on quit

- [ ] **Step 4: Commit + tag**

```bash
git add -A
git commit -m "feat: dancers + session persistence

Ephemeral localized transforms that spawn during morphs, wander the screen
with per-dancer audio reactivity, and dissolve. Breed from genome transforms
+ persistent dancer archive. Session saves current genome on exit."
git tag v0.8.0-dancers
```
