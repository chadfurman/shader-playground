use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct SignalWeights {
    #[serde(default)]
    pub bass: f32,
    #[serde(default)]
    pub mids: f32,
    #[serde(default)]
    pub highs: f32,
    #[serde(default)]
    pub energy: f32,
    #[serde(default)]
    pub beat: f32,
    #[serde(default)]
    pub beat_accum: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct AudioWeights {
    #[serde(default)]
    pub targets: HashMap<String, SignalWeights>,
    #[serde(default)]
    pub caps: HashMap<String, [f32; 2]>,
}

impl AudioWeights {
    pub fn load(path: &Path) -> Result<Self, String> {
        let json =
            fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_str(&json).map_err(|e| format!("parse {}: {e}", path.display()))
    }

    /// Compute the offset for a named target given current audio features.
    pub fn offset(&self, target: &str, features: &crate::audio::AudioFeatures) -> f32 {
        let Some(w) = self.targets.get(target) else {
            return 0.0;
        };
        w.bass * features.bass
            + w.mids * features.mids
            + w.highs * features.highs
            + w.energy * features.energy
            + w.beat * features.beat
            + w.beat_accum * features.beat_accum
    }

    /// Clamp a value to caps for a named target, if defined.
    pub fn clamp(&self, target: &str, value: f32) -> f32 {
        match self.caps.get(target) {
            Some(&[lo, hi]) => value.clamp(lo, hi),
            None => value,
        }
    }
}
