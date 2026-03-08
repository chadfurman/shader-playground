use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

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
    /// Only includes genomes with positive scores (score > 0).
    /// Weight = score, so score 1 = weight 1, score 3 = weight 3.
    pub fn pick_voted(&self, _threshold: i32) -> Option<PathBuf> {
        let eligible: Vec<_> = self.entries.iter()
            .filter(|(_, e)| e.score > 0)
            .collect();
        if eligible.is_empty() {
            return None;
        }

        let total_weight: f32 = eligible.iter()
            .map(|(_, e)| e.score.max(1) as f32)
            .sum();
        if total_weight <= 0.0 {
            return None;
        }

        let mut rng = rand::rng();
        let mut roll = rng.random::<f32>() * total_weight;
        for (_, entry) in &eligible {
            let w = entry.score.max(1) as f32;
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
        use rand::prelude::IndexedRandom;
        let entries: Vec<_> = fs::read_dir(genomes_dir)
            .ok()?
            .filter_map(|e| e.ok())
            .filter(|e| {
                let path = e.path();
                path.is_file()
                    && path.extension().is_some_and(|ext| ext == "json")
                    && path.file_name().is_some_and(|n| n != "votes.json")
            })
            .filter(|e| {
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
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = secs / 86400;
    let year = 1970 + days / 365;
    let day_of_year = days % 365;
    let month = day_of_year / 30 + 1;
    let day = day_of_year % 30 + 1;
    format!("{year}-{month:02}-{day:02}")
}
