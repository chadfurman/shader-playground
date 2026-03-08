use std::collections::VecDeque;

use crate::genome::FlameGenome;

/// Number of hue histogram bins (one per 30 degrees of hue wheel).
const HUE_BINS: usize = 12;

/// Total palette-level features extracted per genome.
pub const PALETTE_FEATURE_COUNT: usize = HUE_BINS + 5; // 12 hue bins + 5 stats = 17

/// Features extracted from a genome's palette for taste modeling.
#[derive(Clone, Debug)]
pub struct PaletteFeatures {
    /// 12-bin hue histogram (normalized, sums to 1.0)
    pub hue_histogram: [f32; HUE_BINS],
    /// Average saturation across palette entries
    pub avg_saturation: f32,
    /// Standard deviation of saturation
    pub saturation_spread: f32,
    /// Average brightness (value channel)
    pub avg_brightness: f32,
    /// Brightness range (max - min)
    pub brightness_range: f32,
    /// Number of distinct hue clusters (hues separated by > 30 degrees)
    pub hue_cluster_count: f32,
}

impl PaletteFeatures {
    /// Extract palette features from a genome.
    /// Returns None if genome has no palette.
    pub fn extract(genome: &FlameGenome) -> Option<Self> {
        let palette = genome.palette.as_ref()?;
        if palette.is_empty() {
            return None;
        }

        // Convert all palette entries to HSV
        let hsv: Vec<(f32, f32, f32)> = palette.iter().map(|rgb| rgb_to_hsv(*rgb)).collect();

        // Hue histogram (12 bins, 30 degrees each)
        let mut hue_histogram = [0.0f32; HUE_BINS];
        let mut saturated_count = 0u32;
        for &(h, s, _) in &hsv {
            // Only count entries with some saturation (skip near-gray)
            if s > 0.05 {
                let bin = ((h / 360.0) * HUE_BINS as f32).floor() as usize;
                let bin = bin.min(HUE_BINS - 1);
                hue_histogram[bin] += 1.0;
                saturated_count += 1;
            }
        }
        // Normalize histogram
        if saturated_count > 0 {
            let total = saturated_count as f32;
            for bin in &mut hue_histogram {
                *bin /= total;
            }
        }

        // Saturation stats
        let avg_saturation = hsv.iter().map(|(_, s, _)| s).sum::<f32>() / hsv.len() as f32;
        let saturation_spread = {
            let variance = hsv.iter()
                .map(|(_, s, _)| (s - avg_saturation).powi(2))
                .sum::<f32>() / hsv.len() as f32;
            variance.sqrt()
        };

        // Brightness stats
        let brightnesses: Vec<f32> = hsv.iter().map(|(_, _, v)| *v).collect();
        let avg_brightness = brightnesses.iter().sum::<f32>() / brightnesses.len() as f32;
        let brightness_range = brightnesses.iter().cloned().fold(0.0f32, f32::max)
            - brightnesses.iter().cloned().fold(1.0f32, f32::min);

        // Hue cluster count: count non-empty bins that are separated by at least one empty bin
        let hue_cluster_count = count_hue_clusters(&hue_histogram);

        Some(Self {
            hue_histogram,
            avg_saturation,
            saturation_spread,
            avg_brightness,
            brightness_range,
            hue_cluster_count,
        })
    }

    /// Convert features to a flat f32 vector for the taste model.
    pub fn to_vec(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(PALETTE_FEATURE_COUNT);
        v.extend_from_slice(&self.hue_histogram);
        v.push(self.avg_saturation);
        v.push(self.saturation_spread);
        v.push(self.avg_brightness);
        v.push(self.brightness_range);
        v.push(self.hue_cluster_count);
        v
    }

    /// Compute hue histogram overlap between two palettes (0 = no overlap, 1 = identical).
    pub fn hue_overlap(&self, other: &PaletteFeatures) -> f32 {
        self.hue_histogram.iter()
            .zip(other.hue_histogram.iter())
            .map(|(a, b)| a.min(*b))
            .sum()
    }
}

/// Gaussian centroid taste model.
/// Learns what palette features correlate with upvoted genomes.
#[derive(Clone, Debug)]
pub struct TasteModel {
    /// Mean of each feature across good genomes
    pub feature_means: Vec<f32>,
    /// Standard deviation of each feature
    pub feature_stddevs: Vec<f32>,
    /// Number of genomes used to build the model
    pub sample_count: u32,
}

impl TasteModel {
    /// Build taste model from a set of palette feature vectors.
    /// Returns None if fewer features than required.
    pub fn build(features: &[Vec<f32>]) -> Option<Self> {
        if features.is_empty() {
            return None;
        }
        let n = features.len() as f32;
        let dim = features[0].len();

        let mut means = vec![0.0f32; dim];
        for f in features {
            for (i, val) in f.iter().enumerate() {
                means[i] += val;
            }
        }
        for m in &mut means {
            *m /= n;
        }

        let mut stddevs = vec![0.0f32; dim];
        for f in features {
            for (i, val) in f.iter().enumerate() {
                stddevs[i] += (val - means[i]).powi(2);
            }
        }
        for s in &mut stddevs {
            *s = (*s / n).sqrt();
            // Floor to prevent collapse on low-variance features
            if *s < 0.01 {
                *s = 0.01;
            }
        }

        Some(Self {
            feature_means: means,
            feature_stddevs: stddevs,
            sample_count: features.len() as u32,
        })
    }

    /// Score a palette's features against the model.
    /// Lower score = closer to "good" centroid = more tasteful.
    pub fn score(&self, features: &[f32]) -> f32 {
        self.feature_means.iter()
            .zip(self.feature_stddevs.iter())
            .zip(features.iter())
            .map(|((mean, stddev), val)| {
                ((val - mean) / stddev).powi(2)
            })
            .sum()
    }
}

/// Manages taste learning and tasteful palette generation.
pub struct TasteEngine {
    /// Current model (None if not enough data)
    model: Option<TasteModel>,
    /// Recent palette features for diversity nudge
    recent_palettes: VecDeque<PaletteFeatures>,
    /// All feature vectors from good genomes (for rebuilding model)
    good_features: Vec<Vec<f32>>,
}

impl TasteEngine {
    pub fn new() -> Self {
        Self {
            model: None,
            recent_palettes: VecDeque::new(),
            good_features: Vec::new(),
        }
    }

    /// Rebuild the model from all voted/imported genomes.
    /// Call this on startup and whenever votes change.
    pub fn rebuild(
        &mut self,
        good_genomes: &[&FlameGenome],
        recent_memory: usize,
    ) {
        self.good_features.clear();
        for genome in good_genomes {
            if let Some(features) = PaletteFeatures::extract(genome) {
                self.good_features.push(features.to_vec());
            }
        }

        self.model = TasteModel::build(&self.good_features);

        // Trim recent palette memory
        while self.recent_palettes.len() > recent_memory {
            self.recent_palettes.pop_front();
        }

        if let Some(ref model) = self.model {
            eprintln!(
                "[taste] model rebuilt: {} samples, {} features",
                model.sample_count, model.feature_means.len()
            );
        }
    }

    /// Generate a palette biased by the taste model.
    /// Falls back to random palette if model isn't ready.
    pub fn generate_palette(
        &mut self,
        min_votes: u32,
        strength: f32,
        exploration_rate: f32,
        diversity_penalty: f32,
        candidates: u32,
        recent_memory: usize,
    ) -> Vec<[f32; 3]> {
        use rand::Rng;
        let mut rng = rand::rng();

        // Exploration: sometimes just go random
        if rng.random::<f32>() < exploration_rate {
            let palette = crate::genome::generate_random_palette();
            self.record_palette(&palette, recent_memory);
            return palette;
        }

        // If model isn't ready, use random
        let model = match &self.model {
            Some(m) if m.sample_count >= min_votes => m,
            _ => {
                let palette = crate::genome::generate_random_palette();
                self.record_palette(&palette, recent_memory);
                return palette;
            }
        };

        // Generate candidates and score them
        let mut best_palette = crate::genome::generate_random_palette();
        let mut best_score = f32::MAX;

        for _ in 0..candidates {
            let palette = crate::genome::generate_random_palette();

            // Build a temporary genome just for feature extraction
            let features = palette_features(&palette);
            let features_vec = features.to_vec();

            // Taste score (lower = better)
            let mut score = model.score(&features_vec) * strength;

            // Diversity penalty: penalize similarity to recent palettes
            for recent in &self.recent_palettes {
                let overlap = features.hue_overlap(recent);
                score += overlap * diversity_penalty;
            }

            if score < best_score {
                best_score = score;
                best_palette = palette;
            }
        }

        self.record_palette(&best_palette, recent_memory);
        best_palette
    }

    /// Record a palette in the recent memory for diversity tracking.
    fn record_palette(&mut self, palette: &[[f32; 3]], recent_memory: usize) {
        if let Some(features) = palette_features_from_slice(palette) {
            self.recent_palettes.push_back(features);
            while self.recent_palettes.len() > recent_memory {
                self.recent_palettes.pop_front();
            }
        }
    }

    /// Whether the model is active (has enough data to influence palettes).
    pub fn is_active(&self, min_votes: u32) -> bool {
        self.model.as_ref().is_some_and(|m| m.sample_count >= min_votes)
    }

    pub fn sample_count(&self) -> u32 {
        self.model.as_ref().map_or(0, |m| m.sample_count)
    }
}

/// Extract palette features from a raw palette (no genome wrapper needed).
fn palette_features(palette: &[[f32; 3]]) -> PaletteFeatures {
    // Convert to HSV
    let hsv: Vec<(f32, f32, f32)> = palette.iter().map(|rgb| rgb_to_hsv(*rgb)).collect();

    let mut hue_histogram = [0.0f32; HUE_BINS];
    let mut saturated_count = 0u32;
    for &(h, s, _) in &hsv {
        if s > 0.05 {
            let bin = ((h / 360.0) * HUE_BINS as f32).floor() as usize;
            let bin = bin.min(HUE_BINS - 1);
            hue_histogram[bin] += 1.0;
            saturated_count += 1;
        }
    }
    if saturated_count > 0 {
        let total = saturated_count as f32;
        for bin in &mut hue_histogram {
            *bin /= total;
        }
    }

    let avg_saturation = hsv.iter().map(|(_, s, _)| s).sum::<f32>() / hsv.len() as f32;
    let saturation_spread = {
        let variance = hsv.iter()
            .map(|(_, s, _)| (s - avg_saturation).powi(2))
            .sum::<f32>() / hsv.len() as f32;
        variance.sqrt()
    };

    let brightnesses: Vec<f32> = hsv.iter().map(|(_, _, v)| *v).collect();
    let avg_brightness = brightnesses.iter().sum::<f32>() / brightnesses.len() as f32;
    let brightness_range = brightnesses.iter().cloned().fold(0.0f32, f32::max)
        - brightnesses.iter().cloned().fold(1.0f32, f32::min);

    let hue_cluster_count = count_hue_clusters(&hue_histogram);

    PaletteFeatures {
        hue_histogram,
        avg_saturation,
        saturation_spread,
        avg_brightness,
        brightness_range,
        hue_cluster_count,
    }
}

/// Extract features from a palette slice (convenience wrapper).
fn palette_features_from_slice(palette: &[[f32; 3]]) -> Option<PaletteFeatures> {
    if palette.is_empty() {
        return None;
    }
    Some(palette_features(palette))
}

/// Count distinct hue clusters in the histogram.
/// A cluster is a contiguous group of non-empty bins (wrapping around).
fn count_hue_clusters(histogram: &[f32; HUE_BINS]) -> f32 {
    let mut clusters = 0u32;
    let mut in_cluster = false;
    // Check if the histogram wraps (last and first bins both non-zero)
    let wraps = histogram[0] > 0.01 && histogram[HUE_BINS - 1] > 0.01;

    for i in 0..HUE_BINS {
        if histogram[i] > 0.01 {
            if !in_cluster {
                clusters += 1;
                in_cluster = true;
            }
        } else {
            in_cluster = false;
        }
    }

    // If it wraps, the first and last clusters are actually one
    if wraps && clusters > 1 {
        clusters -= 1;
    }

    clusters as f32
}

/// Convert RGB [0..1] to HSV (hue in degrees 0..360, s/v in 0..1).
fn rgb_to_hsv(rgb: [f32; 3]) -> (f32, f32, f32) {
    let r = rgb[0];
    let g = rgb[1];
    let b = rgb[2];

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;
    let s = if max > 0.0 { delta / max } else { 0.0 };

    let h = if delta < 1e-6 {
        0.0
    } else if (max - r).abs() < 1e-6 {
        60.0 * (((g - b) / delta) % 6.0)
    } else if (max - g).abs() < 1e-6 {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };

    let h = if h < 0.0 { h + 360.0 } else { h };

    (h, s, v)
}
