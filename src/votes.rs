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

        // Persist immediately
        if let Err(e) = self.save(genomes_dir) {
            eprintln!("[vote] save error: {e}");
        }

        score
    }

    /// Pick a genome from the vote-weighted pool.
    /// Only includes genomes with positive scores (score > 0).
    /// Weight = score, so score 1 = weight 1, score 3 = weight 3.
    pub fn pick_voted(&self, _threshold: i32) -> Option<PathBuf> {
        let eligible: Vec<_> = self.entries.iter().filter(|(_, e)| e.score > 0).collect();
        if eligible.is_empty() {
            return None;
        }

        let total_weight: f32 = eligible.iter().map(|(_, e)| e.score.max(1) as f32).sum();
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
    /// Excludes any genome with a negative vote score.
    /// Pick a random genome from saved genomes + seeds (unweighted).
    /// Excludes any genome with a negative vote score.
    pub fn pick_random_saved(
        genomes_dir: &Path,
        _threshold: i32,
        ledger: &VoteLedger,
    ) -> Option<PathBuf> {
        use rand::prelude::IndexedRandom;

        // Scan both genomes/ and genomes/seeds/ for diverse pool
        let dirs = [genomes_dir.to_path_buf(), genomes_dir.join("seeds")];
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
                    // Exclude negatively-scored genomes
                    let stem = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_string();
                    if ledger.entries.get(&stem).is_some_and(|v| v.score < 0) {
                        continue;
                    }
                    entries.push(path);
                }
            }
        }

        if entries.is_empty() {
            return None;
        }

        entries.choose(&mut rand::rng()).cloned()
    }
}

// ── Lineage Cache ──

use std::collections::{HashSet, VecDeque};

/// Tracks genome ancestry for genetic distance computation.
/// Built by scanning saved genome files; rebuilt when genomes change.
#[derive(Debug, Default)]
pub struct LineageCache {
    /// genome name → (parent_a name, parent_b name)
    parents: HashMap<String, (Option<String>, Option<String>)>,
}

impl LineageCache {
    /// Build lineage cache by scanning all genome JSON files in the given directories.
    pub fn build(genomes_dir: &Path) -> Self {
        let mut parents = HashMap::new();
        let dirs = [
            genomes_dir.to_path_buf(),
            genomes_dir.join("seeds"),
            genomes_dir.join("flames"),
        ];
        for dir in &dirs {
            if let Ok(read) = fs::read_dir(dir) {
                for entry in read.filter_map(|e| e.ok()) {
                    let path = entry.path();
                    if !path.is_file()
                        || path.extension().is_none_or(|ext| ext != "json")
                        || path.file_name().is_some_and(|n| n == "votes.json")
                    {
                        continue;
                    }
                    if let Ok(genome) = FlameGenome::load(&path) {
                        parents.insert(genome.name.clone(), (genome.parent_a, genome.parent_b));
                    }
                }
            }
        }
        Self { parents }
    }

    /// Register a genome's lineage (call after breeding/saving a new genome).
    pub fn register(&mut self, name: &str, parent_a: &Option<String>, parent_b: &Option<String>) {
        self.parents
            .insert(name.to_string(), (parent_a.clone(), parent_b.clone()));
    }

    /// Compute genetic distance between two genomes.
    /// Distance = depth to lowest common ancestor. If no common ancestor
    /// found within max_depth, returns max_depth (maximum diversity).
    pub fn genetic_distance(&self, name_a: &str, name_b: &str, max_depth: u32) -> u32 {
        if name_a == name_b {
            return 0;
        }

        // BFS from both genomes simultaneously, looking for overlap
        let ancestors_a = self.collect_ancestors(name_a, max_depth);
        let ancestors_b = self.collect_ancestors(name_b, max_depth);

        // Find minimum combined depth to a shared ancestor
        let mut min_dist = max_depth;
        for (ancestor, depth_a) in &ancestors_a {
            if let Some(depth_b) = ancestors_b.get(ancestor) {
                let dist = (*depth_a).max(*depth_b);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }

        min_dist
    }

    /// Collect all ancestors of a genome up to max_depth.
    /// Returns HashMap<ancestor_name, depth>.
    fn collect_ancestors(&self, name: &str, max_depth: u32) -> HashMap<String, u32> {
        let mut result = HashMap::new();
        result.insert(name.to_string(), 0);

        let mut queue: VecDeque<(String, u32)> = VecDeque::new();
        queue.push_back((name.to_string(), 0));
        let mut visited = HashSet::new();
        visited.insert(name.to_string());

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            if let Some((pa, pb)) = self.parents.get(&current) {
                for parent in [pa, pb].iter().filter_map(|p| p.as_ref()) {
                    if visited.insert(parent.clone()) {
                        let parent_depth = depth + 1;
                        result.insert(parent.clone(), parent_depth);
                        queue.push_back((parent.clone(), parent_depth));
                    }
                }
            }
        }

        result
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache() -> LineageCache {
        let mut cache = LineageCache::default();
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
    fn distance_between_unrelated_is_max_depth() {
        let cache = make_cache();
        assert_eq!(cache.genetic_distance("parent_a", "parent_b", 8), 8);
    }

    #[test]
    fn distance_siblings_share_parent() {
        let mut cache = LineageCache::default();
        cache.register("dad", &None, &None);
        cache.register("mom", &None, &None);
        cache.register("sibling_a", &Some("dad".into()), &Some("mom".into()));
        cache.register("sibling_b", &Some("dad".into()), &Some("mom".into()));
        assert_eq!(cache.genetic_distance("sibling_a", "sibling_b", 8), 1);
    }

    #[test]
    fn distance_unknown_genomes_returns_max_depth() {
        let cache = LineageCache::default();
        assert_eq!(cache.genetic_distance("unknown_a", "unknown_b", 8), 8);
    }

    #[test]
    fn register_updates_cache() {
        let mut cache = LineageCache::default();
        cache.register("new_genome", &Some("pa".into()), &Some("pb".into()));
        assert_eq!(cache.genetic_distance("new_genome", "pa", 8), 1);
    }

    #[test]
    fn vote_ledger_empty_pick_returns_none() {
        let ledger = VoteLedger::default();
        assert!(ledger.pick_voted(0).is_none());
    }

    #[test]
    fn vote_ledger_positive_score_is_pickable() {
        let mut ledger = VoteLedger::default();
        ledger.entries.insert(
            "test_genome".into(),
            VoteEntry {
                score: 3,
                file: "/tmp/test_genome.json".into(),
                last_seen: "2026-01-01".into(),
            },
        );
        let picked = ledger.pick_voted(0);
        assert!(picked.is_some());
        assert_eq!(picked.unwrap().to_str().unwrap(), "/tmp/test_genome.json");
    }

    #[test]
    fn vote_ledger_negative_score_not_pickable() {
        let mut ledger = VoteLedger::default();
        ledger.entries.insert(
            "bad_genome".into(),
            VoteEntry {
                score: -5,
                file: "/tmp/bad.json".into(),
                last_seen: "2026-01-01".into(),
            },
        );
        assert!(ledger.pick_voted(0).is_none());
    }

    #[test]
    fn vote_ledger_zero_score_not_pickable() {
        let mut ledger = VoteLedger::default();
        ledger.entries.insert(
            "meh".into(),
            VoteEntry {
                score: 0,
                file: "/tmp/meh.json".into(),
                last_seen: "2026-01-01".into(),
            },
        );
        assert!(ledger.pick_voted(0).is_none());
    }
}
