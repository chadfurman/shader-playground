use crate::audio::AudioFeatures;
use crate::genome::{FlameGenome, FlameTransform};
use crate::weights::RuntimeConfig;
use rand::Rng;
use rand::prelude::SliceRandom;
use std::collections::VecDeque;

/// Compute bell curve weight for a dancer at a given point in its lifetime.
/// Returns 0.0 at birth, ramps to 1.0 over fade_fraction of lifetime,
/// holds, then ramps back to 0.0 over the final fade_fraction.
fn bell_curve_weight(elapsed: f32, lifetime: f32, fade_fraction: f32) -> f32 {
    if lifetime <= 0.0 {
        return 0.0;
    }
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

pub struct DancerAudioWeights {
    pub drift_bass: f32,
    pub drift_mids: f32,
    pub drift_highs: f32,
    pub _pulse_energy: f32,
    pub _pulse_beat: f32,
}

pub struct Dancer {
    pub transform: FlameTransform,
    pub birth_time: f32,
    pub lifetime: f32,
    pub audio_weights: DancerAudioWeights,
    pub _generation: u32,
}

pub struct DancerManager {
    pub active: Vec<Dancer>,
    pub scheduled: Vec<f32>, // morph_progress trigger values
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
        let idx = rng.random_range(0..self.archive.len());
        self.archive.get(idx).cloned()
    }

    /// Breed a new dancer from genome transforms + archive.
    pub fn breed_dancer(&self, genome_xfs: &[FlameTransform], cfg: &RuntimeConfig) -> Dancer {
        let mut rng = rand::rng();

        // Pick 2-3 source transforms from genome + archive
        let mut sources: Vec<FlameTransform> = Vec::new();
        let n_genome = rng.random_range(1..=2usize).min(genome_xfs.len());
        let mut indices: Vec<usize> = (0..genome_xfs.len()).collect();
        indices.shuffle(&mut rng);
        for &i in indices.iter().take(n_genome) {
            sources.push(genome_xfs[i].clone());
        }
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
            _pulse_energy: rng.random_range(0.0..=s),
            _pulse_beat: rng.random_range(0.0..=s),
        };

        let generation = self.archive.len() as u32;
        let lifetime: f32 = rng.random_range(cfg.dancer_lifetime_min..=cfg.dancer_lifetime_max);

        Dancer {
            transform: result,
            birth_time: 0.0, // set at actual spawn time
            lifetime,
            audio_weights,
            _generation: generation,
        }
    }

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
            FlameGenome::push_transform(&mut buf, &xf);
        }
        buf
    }

    pub fn active_count(&self) -> usize {
        self.active.len()
    }

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
        let w = bell_curve_weight(2.0, 10.0, 0.2);
        assert!((w - 1.0).abs() < 0.01);
    }

    #[test]
    fn bell_curve_at_midpoint_is_one() {
        assert_eq!(bell_curve_weight(3.0, 6.0, 0.2), 1.0);
    }

    #[test]
    fn bell_curve_at_fade_out_start_is_one() {
        let w = bell_curve_weight(8.0, 10.0, 0.2);
        assert!((w - 1.0).abs() < 0.01);
    }

    #[test]
    fn bell_curve_at_end_is_zero() {
        assert_eq!(bell_curve_weight(6.0, 6.0, 0.2), 0.0);
    }

    #[test]
    fn bell_curve_at_halfway_fade_in() {
        let w = bell_curve_weight(1.0, 10.0, 0.2);
        assert!((w - 0.5).abs() < 0.01);
    }

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

    #[test]
    fn breed_dancer_from_genome_only() {
        let cfg: RuntimeConfig = serde_json::from_str("{}").unwrap();
        let mut rng = rand::rng();
        let mgr = DancerManager::new();
        let genome_xfs = vec![
            FlameTransform::random_transform(&mut rng),
            FlameTransform::random_transform(&mut rng),
            FlameTransform::random_transform(&mut rng),
        ];
        let dancer = mgr.breed_dancer(&genome_xfs, &cfg);
        let det = dancer.transform.affine[0][0] * dancer.transform.affine[1][1]
            - dancer.transform.affine[0][1] * dancer.transform.affine[1][0];
        assert!(det.abs() < cfg.dancer_scale_max + 0.1);
        let dist = (dancer.transform.offset[0].powi(2) + dancer.transform.offset[1].powi(2)).sqrt();
        assert!(dist >= cfg.dancer_offset_min * 0.5);
    }

    #[test]
    fn breed_dancer_uses_archive_when_available() {
        let cfg: RuntimeConfig = serde_json::from_str("{}").unwrap();
        let mut mgr = DancerManager::new();
        let mut archived = FlameTransform::default();
        archived.color = 0.99;
        mgr.archive_transform(archived, 20);
        let genome_xfs = vec![FlameTransform::random_transform(&mut rand::rng())];
        let mut saw_high_color = false;
        for _ in 0..20 {
            let d = mgr.breed_dancer(&genome_xfs, &cfg);
            if d.transform.color > 0.7 {
                saw_high_color = true;
            }
        }
        assert!(
            saw_high_color,
            "archive transform color should influence some dancers"
        );
    }

    #[test]
    fn schedule_dancers_creates_valid_triggers() {
        let cfg: RuntimeConfig = serde_json::from_str("{}").unwrap();
        let mut mgr = DancerManager::new();
        let genome_xfs = vec![FlameTransform::random_transform(&mut rand::rng())];
        mgr.schedule_dancers(&genome_xfs, &cfg, 2.5);
        assert!(!mgr.scheduled.is_empty());
        assert!(mgr.scheduled.len() <= cfg.dancer_count_max as usize);
        for &trigger in &mgr.scheduled {
            assert!(trigger >= 0.0);
            assert!(trigger <= 2.5);
        }
    }

    #[test]
    fn tick_spawns_at_correct_progress() {
        let cfg: RuntimeConfig = serde_json::from_str("{}").unwrap();
        let mut mgr = DancerManager::new();
        let genome_xfs = vec![FlameTransform::random_transform(&mut rand::rng())];
        mgr.scheduled = vec![0.5];
        let audio = AudioFeatures::default();
        mgr.tick(0.3, 1.0, 0.016, &audio, &genome_xfs, &cfg);
        assert_eq!(mgr.active.len(), 0);
        mgr.tick(0.6, 1.1, 0.016, &audio, &genome_xfs, &cfg);
        assert_eq!(mgr.active.len(), 1);
        assert!(mgr.scheduled.is_empty());
    }

    #[test]
    fn tick_removes_expired_dancers() {
        let cfg: RuntimeConfig = serde_json::from_str("{}").unwrap();
        let mut mgr = DancerManager::new();
        mgr.active.push(Dancer {
            transform: FlameTransform::default(),
            birth_time: 0.0,
            lifetime: 1.0,
            audio_weights: DancerAudioWeights {
                drift_bass: 0.0,
                drift_mids: 0.0,
                drift_highs: 0.0,
                _pulse_energy: 0.0,
                _pulse_beat: 0.0,
            },
            _generation: 0,
        });
        let audio = AudioFeatures::default();
        let genome_xfs = vec![FlameTransform::random_transform(&mut rand::rng())];
        mgr.tick(0.0, 5.0, 0.016, &audio, &genome_xfs, &cfg);
        assert_eq!(mgr.active.len(), 0);
        assert_eq!(mgr.archive.len(), 1);
    }

    #[test]
    fn flatten_active_fits_prealloc_budget() {
        let cfg: RuntimeConfig = serde_json::from_str("{}").unwrap();
        let mut mgr = DancerManager::new();
        let mut rng = rand::rng();
        let genome_xfs: Vec<_> = (0..12)
            .map(|_| FlameTransform::random_transform(&mut rng))
            .collect();
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
        assert!((mgr2.archive[0].affine[0][0] - 1.0).abs() < 0.01);
        assert_eq!(mgr2.archive[0].linear, 0.0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
