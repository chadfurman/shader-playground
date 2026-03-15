use std::path::Path;

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Grid dimensions for MAP-Elites archive.
const SYMMETRY_BINS: usize = 6;
const FD_BINS: usize = 5;
const COLOR_ENTROPY_BINS: usize = 4;
const TOTAL_CELLS: usize = SYMMETRY_BINS * FD_BINS * COLOR_ENTROPY_BINS; // 120

/// FD range for binning.
const FD_MIN: f32 = 1.0;
const FD_MAX: f32 = 2.0;

/// Discrete coordinates in the MAP-Elites grid.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridCoords {
    pub symmetry_bin: usize,
    pub fd_bin: usize,
    pub color_entropy_bin: usize,
}

impl GridCoords {
    /// Map behavioral traits to grid coordinates.
    /// Symmetry: abs(sym).clamp(1,6) - 1 => 6 bins [0..5]
    /// FD: 5 bins over [1.0, 2.0]
    /// Color entropy: 4 bins over [0, 1]
    pub fn from_traits(symmetry: i32, fractal_dim: f32, color_entropy: f32) -> Self {
        let symmetry_bin = (symmetry.unsigned_abs().clamp(1, 6) - 1) as usize;

        let fd_norm = ((fractal_dim - FD_MIN) / (FD_MAX - FD_MIN)).clamp(0.0, 1.0);
        let fd_bin = (fd_norm * FD_BINS as f32).floor() as usize;
        let fd_bin = fd_bin.min(FD_BINS - 1);

        let ce_norm = color_entropy.clamp(0.0, 1.0);
        let color_entropy_bin = (ce_norm * COLOR_ENTROPY_BINS as f32).floor() as usize;
        let color_entropy_bin = color_entropy_bin.min(COLOR_ENTROPY_BINS - 1);

        Self {
            symmetry_bin,
            fd_bin,
            color_entropy_bin,
        }
    }

    /// Linear index into the flat cell array.
    fn to_index(&self) -> usize {
        self.symmetry_bin * FD_BINS * COLOR_ENTROPY_BINS
            + self.fd_bin * COLOR_ENTROPY_BINS
            + self.color_entropy_bin
    }
}

/// An entry in the MAP-Elites archive.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchiveEntry {
    pub genome_name: String,
    pub score: f32,
    pub features: Vec<f32>,
}

/// MAP-Elites diversity archive for parent selection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MapElitesArchive {
    cells: Vec<Option<ArchiveEntry>>,
}

impl MapElitesArchive {
    pub fn new() -> Self {
        Self {
            cells: (0..TOTAL_CELLS).map(|_| None).collect(),
        }
    }

    /// Insert if cell is empty or the new score is better (lower).
    /// Returns true if the entry was inserted.
    pub fn insert(
        &mut self,
        coords: &GridCoords,
        name: String,
        score: f32,
        features: Vec<f32>,
    ) -> bool {
        let idx = coords.to_index();
        if idx >= TOTAL_CELLS {
            return false;
        }
        let should_insert = match &self.cells[idx] {
            None => true,
            Some(existing) => score < existing.score,
        };
        if should_insert {
            self.cells[idx] = Some(ArchiveEntry {
                genome_name: name,
                score,
                features,
            });
        }
        should_insert
    }

    /// Pick a random entry from occupied cells (uniform distribution).
    pub fn pick_random(&self, rng: &mut impl Rng) -> Option<&ArchiveEntry> {
        use rand::prelude::IndexedRandom;
        let occupied: Vec<&ArchiveEntry> = self.cells.iter().filter_map(|c| c.as_ref()).collect();
        occupied.choose(rng).copied()
    }

    /// Returns all stored feature vectors (for k-NN novelty scoring).
    pub fn all_features(&self) -> Vec<&Vec<f32>> {
        self.cells
            .iter()
            .filter_map(|c| c.as_ref().map(|e| &e.features))
            .collect()
    }

    /// Number of occupied cells.
    pub fn occupied_count(&self) -> usize {
        self.cells.iter().filter(|c| c.is_some()).count()
    }

    /// Save archive to a JSON file.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json =
            serde_json::to_string_pretty(self).map_err(|e| format!("serialize archive: {e}"))?;
        std::fs::write(path, json).map_err(|e| format!("write archive: {e}"))?;
        Ok(())
    }

    /// Load archive from a JSON file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| format!("read archive: {e}"))?;
        serde_json::from_str(&json).map_err(|e| format!("parse archive: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_coords_symmetry_clamping() {
        // abs(0) = 0 → clamp to 1 → bin 0
        assert_eq!(GridCoords::from_traits(0, 1.5, 0.5).symmetry_bin, 0);
        // abs(3) = 3 → bin 2
        assert_eq!(GridCoords::from_traits(3, 1.5, 0.5).symmetry_bin, 2);
        // abs(-6) = 6 → bin 5
        assert_eq!(GridCoords::from_traits(-6, 1.5, 0.5).symmetry_bin, 5);
        // abs(10) = 10 → clamp to 6 → bin 5
        assert_eq!(GridCoords::from_traits(10, 1.5, 0.5).symmetry_bin, 5);
    }

    #[test]
    fn grid_coords_fd_binning() {
        // FD 1.0 → bin 0
        assert_eq!(GridCoords::from_traits(1, 1.0, 0.5).fd_bin, 0);
        // FD 2.0 → bin 4 (max)
        assert_eq!(GridCoords::from_traits(1, 2.0, 0.5).fd_bin, 4);
        // FD 1.5 → bin 2
        assert_eq!(GridCoords::from_traits(1, 1.5, 0.5).fd_bin, 2);
    }

    #[test]
    fn insert_empty_cell() {
        let mut archive = MapElitesArchive::new();
        let coords = GridCoords::from_traits(1, 1.5, 0.5);
        let inserted = archive.insert(&coords, "test".into(), 1.0, vec![0.5]);
        assert!(inserted, "should insert into empty cell");
        assert_eq!(archive.occupied_count(), 1);
    }

    #[test]
    fn insert_replaces_with_better_score() {
        let mut archive = MapElitesArchive::new();
        let coords = GridCoords::from_traits(1, 1.5, 0.5);
        archive.insert(&coords, "worse".into(), 5.0, vec![]);
        let replaced = archive.insert(&coords, "better".into(), 2.0, vec![]);
        assert!(replaced, "should replace with better (lower) score");
        let entry = archive.cells[coords.to_index()].as_ref().unwrap();
        assert_eq!(entry.genome_name, "better");
    }

    #[test]
    fn insert_rejects_worse_score() {
        let mut archive = MapElitesArchive::new();
        let coords = GridCoords::from_traits(1, 1.5, 0.5);
        archive.insert(&coords, "good".into(), 2.0, vec![]);
        let rejected = archive.insert(&coords, "bad".into(), 5.0, vec![]);
        assert!(!rejected, "should reject worse score");
        let entry = archive.cells[coords.to_index()].as_ref().unwrap();
        assert_eq!(entry.genome_name, "good");
    }

    #[test]
    fn pick_random_from_empty_returns_none() {
        let archive = MapElitesArchive::new();
        let mut rng = rand::rng();
        assert!(archive.pick_random(&mut rng).is_none());
    }

    #[test]
    fn pick_random_from_occupied_returns_entry() {
        let mut archive = MapElitesArchive::new();
        let coords = GridCoords::from_traits(1, 1.5, 0.5);
        archive.insert(&coords, "test".into(), 1.0, vec![42.0]);
        let mut rng = rand::rng();
        let picked = archive.pick_random(&mut rng);
        assert!(picked.is_some());
        assert_eq!(picked.unwrap().genome_name, "test");
    }

    #[test]
    fn all_features_returns_occupied_features() {
        let mut archive = MapElitesArchive::new();
        archive.insert(
            &GridCoords::from_traits(1, 1.0, 0.0),
            "a".into(),
            1.0,
            vec![1.0],
        );
        archive.insert(
            &GridCoords::from_traits(2, 2.0, 1.0),
            "b".into(),
            2.0,
            vec![2.0],
        );
        let features = archive.all_features();
        assert_eq!(features.len(), 2);
    }

    #[test]
    fn archive_persistence_roundtrip() {
        let mut archive = MapElitesArchive::new();
        archive.insert(
            &GridCoords::from_traits(1, 1.5, 0.5),
            "test".into(),
            1.0,
            vec![0.5, 0.6],
        );

        let dir = std::env::temp_dir().join("archive_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_archive.json");
        archive.save(&path).expect("save should succeed");

        let loaded = MapElitesArchive::load(&path).expect("load should succeed");
        assert_eq!(loaded.occupied_count(), 1);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
