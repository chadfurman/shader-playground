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
            let variance = hsv
                .iter()
                .map(|(_, s, _)| (s - avg_saturation).powi(2))
                .sum::<f32>()
                / hsv.len() as f32;
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
        self.hue_histogram
            .iter()
            .zip(other.hue_histogram.iter())
            .map(|(a, b)| a.min(*b))
            .sum()
    }
}

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
        let v = vec![
            self.primary_variation_index,
            self.primary_dominance,
            self.active_variation_count,
            self.affine_determinant,
            self.affine_asymmetry,
            self.offset_magnitude,
            self.color_index,
            self.weight,
        ];
        debug_assert_eq!(v.len(), TRANSFORM_FEATURE_COUNT);
        v
    }
}

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
    pub fn extract(genome: &crate::genome::FlameGenome) -> Self {
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
        let color_variance =
            colors.iter().map(|c| (c - mean_color).powi(2)).sum::<f32>() / n as f32;

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
        let v = vec![
            self.transform_count,
            self.variation_diversity,
            self.mean_determinant,
            self.determinant_contrast,
            self.color_spread,
        ];
        debug_assert_eq!(v.len(), COMPOSITION_FEATURE_COUNT);
        v
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
        self.feature_means
            .iter()
            .zip(self.feature_stddevs.iter())
            .zip(features.iter())
            .map(|((mean, stddev), val)| ((val - mean) / stddev).powi(2))
            .sum()
    }
}

/// Manages taste learning and tasteful palette generation.
pub struct TasteEngine {
    /// Current model (None if not enough data)
    model: Option<TasteModel>,
    /// Transform-level taste model (None if not enough data)
    transform_model: Option<TasteModel>,
    /// Recent palette features for diversity nudge
    recent_palettes: VecDeque<PaletteFeatures>,
    /// All feature vectors from good genomes (for rebuilding model)
    good_features: Vec<Vec<f32>>,
}

impl TasteEngine {
    pub fn new() -> Self {
        Self {
            model: None,
            transform_model: None,
            recent_palettes: VecDeque::new(),
            good_features: Vec::new(),
        }
    }

    /// Rebuild the model from all voted/imported genomes.
    /// Call this on startup and whenever votes change.
    pub fn rebuild(&mut self, good_genomes: &[&FlameGenome], recent_memory: usize) {
        self.good_features.clear();
        let mut transform_features: Vec<Vec<f32>> = Vec::new();

        for genome in good_genomes {
            if let Some(palette_feats) = PaletteFeatures::extract(genome) {
                let mut features_vec = palette_feats.to_vec();
                let comp = CompositionFeatures::extract(genome);
                features_vec.extend(comp.to_vec());
                self.good_features.push(features_vec);
            }
            for xf in &genome.transforms {
                transform_features.push(TransformFeatures::extract(xf).to_vec());
            }
        }

        self.model = TasteModel::build(&self.good_features);
        self.transform_model = TasteModel::build(&transform_features);

        // Trim recent palette memory
        while self.recent_palettes.len() > recent_memory {
            self.recent_palettes.pop_front();
        }

        if let Some(ref model) = self.model {
            eprintln!(
                "[taste] model rebuilt: {} samples, {} features",
                model.sample_count,
                model.feature_means.len()
            );
        }
        if let Some(ref tm) = self.transform_model {
            eprintln!(
                "[taste] transform model rebuilt: {} samples, {} features",
                tm.sample_count,
                tm.feature_means.len()
            );
        }
    }

    /// Score a transform against the transform taste model.
    /// Returns None if the model isn't ready.
    pub fn score_transform(
        &self,
        xf: &crate::genome::FlameTransform,
        min_votes: u32,
    ) -> Option<f32> {
        let model = self.transform_model.as_ref()?;
        if model.sample_count < min_votes {
            return None;
        }
        let features = TransformFeatures::extract(xf).to_vec();
        Some(model.score(&features))
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
        self.model
            .as_ref()
            .is_some_and(|m| m.sample_count >= min_votes)
    }

    pub fn sample_count(&self) -> u32 {
        self.model.as_ref().map_or(0, |m| m.sample_count)
    }
}

/// Extract palette features from a raw palette (no genome wrapper needed).
pub(crate) fn palette_features(palette: &[[f32; 3]]) -> PaletteFeatures {
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
        let variance = hsv
            .iter()
            .map(|(_, s, _)| (s - avg_saturation).powi(2))
            .sum::<f32>()
            / hsv.len() as f32;
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
pub(crate) fn count_hue_clusters(histogram: &[f32; HUE_BINS]) -> f32 {
    let mut clusters = 0u32;
    let mut in_cluster = false;
    // Check if the histogram wraps (last and first bins both non-zero)
    let wraps = histogram[0] > 0.01 && histogram[HUE_BINS - 1] > 0.01;

    for bin in histogram {
        if *bin > 0.01 {
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
pub(crate) fn rgb_to_hsv(rgb: [f32; 3]) -> (f32, f32, f32) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    // --- RGB/HSV tests ---

    #[test]
    fn rgb_to_hsv_pure_red() {
        let (h, s, v) = rgb_to_hsv([1.0, 0.0, 0.0]);
        assert!(approx_eq(h, 0.0), "hue was {h}");
        assert!(approx_eq(s, 1.0), "sat was {s}");
        assert!(approx_eq(v, 1.0), "val was {v}");
    }

    #[test]
    fn rgb_to_hsv_pure_green() {
        let (h, _s, _v) = rgb_to_hsv([0.0, 1.0, 0.0]);
        assert!(approx_eq(h, 120.0), "hue was {h}");
    }

    #[test]
    fn rgb_to_hsv_pure_blue() {
        let (h, _s, _v) = rgb_to_hsv([0.0, 0.0, 1.0]);
        assert!(approx_eq(h, 240.0), "hue was {h}");
    }

    #[test]
    fn rgb_to_hsv_white() {
        let (_h, s, v) = rgb_to_hsv([1.0, 1.0, 1.0]);
        assert!(approx_eq(s, 0.0), "sat was {s}");
        assert!(approx_eq(v, 1.0), "val was {v}");
    }

    #[test]
    fn rgb_to_hsv_black() {
        let (_h, s, v) = rgb_to_hsv([0.0, 0.0, 0.0]);
        assert!(approx_eq(s, 0.0), "sat was {s}");
        assert!(approx_eq(v, 0.0), "val was {v}");
    }

    #[test]
    fn rgb_to_hsv_gray() {
        let (_h, s, v) = rgb_to_hsv([0.5, 0.5, 0.5]);
        assert!(approx_eq(s, 0.0), "sat was {s}");
        assert!(approx_eq(v, 0.5), "val was {v}");
    }

    // --- Hue cluster tests ---

    #[test]
    fn hue_clusters_single_bin() {
        let mut histogram = [0.0f32; HUE_BINS];
        histogram[3] = 1.0;
        assert!(approx_eq(count_hue_clusters(&histogram), 1.0));
    }

    #[test]
    fn hue_clusters_two_separated() {
        let mut histogram = [0.0f32; HUE_BINS];
        histogram[1] = 0.5;
        histogram[7] = 0.5;
        assert!(approx_eq(count_hue_clusters(&histogram), 2.0));
    }

    #[test]
    fn hue_clusters_adjacent_is_one() {
        let mut histogram = [0.0f32; HUE_BINS];
        histogram[3] = 0.3;
        histogram[4] = 0.4;
        histogram[5] = 0.3;
        assert!(approx_eq(count_hue_clusters(&histogram), 1.0));
    }

    #[test]
    fn hue_clusters_wrapping() {
        let mut histogram = [0.0f32; HUE_BINS];
        histogram[0] = 0.5;
        histogram[11] = 0.5;
        assert!(approx_eq(count_hue_clusters(&histogram), 1.0));
    }

    #[test]
    fn hue_clusters_empty() {
        let histogram = [0.0f32; HUE_BINS];
        assert!(approx_eq(count_hue_clusters(&histogram), 0.0));
    }

    // --- Palette feature tests ---

    #[test]
    fn palette_features_uniform_red() {
        let palette: Vec<[f32; 3]> = vec![[1.0, 0.0, 0.0]; 256];
        let features = palette_features(&palette);
        assert!(
            approx_eq(features.hue_histogram[0], 1.0),
            "hue bin 0 was {}",
            features.hue_histogram[0]
        );
        assert!(
            approx_eq(features.avg_saturation, 1.0),
            "sat was {}",
            features.avg_saturation
        );
        assert!(
            approx_eq(features.hue_cluster_count, 1.0),
            "clusters was {}",
            features.hue_cluster_count
        );
    }

    #[test]
    fn palette_features_all_gray() {
        let palette: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5]; 256];
        let features = palette_features(&palette);
        for (i, &bin) in features.hue_histogram.iter().enumerate() {
            assert!(approx_eq(bin, 0.0), "hue bin {i} was {bin}");
        }
        assert!(
            approx_eq(features.avg_saturation, 0.0),
            "sat was {}",
            features.avg_saturation
        );
        assert!(
            approx_eq(features.avg_brightness, 0.5),
            "brightness was {}",
            features.avg_brightness
        );
    }

    #[test]
    fn hue_overlap_identical() {
        let palette: Vec<[f32; 3]> = vec![[1.0, 0.0, 0.0]; 256];
        let features = palette_features(&palette);
        let overlap = features.hue_overlap(&features);
        assert!(approx_eq(overlap, 1.0), "overlap was {overlap}");
    }

    #[test]
    fn hue_overlap_disjoint() {
        let red_palette: Vec<[f32; 3]> = vec![[1.0, 0.0, 0.0]; 256];
        let blue_palette: Vec<[f32; 3]> = vec![[0.0, 0.0, 1.0]; 256];
        let red_features = palette_features(&red_palette);
        let blue_features = palette_features(&blue_palette);
        let overlap = red_features.hue_overlap(&blue_features);
        assert!(approx_eq(overlap, 0.0), "overlap was {overlap}");
    }

    #[test]
    fn feature_vec_length() {
        let palette: Vec<[f32; 3]> = vec![[1.0, 0.0, 0.0]; 256];
        let features = palette_features(&palette);
        assert_eq!(features.to_vec().len(), PALETTE_FEATURE_COUNT);
    }

    // --- TasteModel tests ---

    #[test]
    fn taste_model_build_empty() {
        let features: Vec<Vec<f32>> = vec![];
        assert!(TasteModel::build(&features).is_none());
    }

    #[test]
    fn taste_model_build_single_sample() {
        let sample = vec![0.5, 0.3, 0.8];
        let model = TasteModel::build(&[sample.clone()]).unwrap();
        assert!(approx_eq(model.feature_means[0], 0.5));
        assert!(approx_eq(model.feature_means[1], 0.3));
        assert!(approx_eq(model.feature_means[2], 0.8));
        // stddevs floored to 0.01
        for s in &model.feature_stddevs {
            assert!(approx_eq(*s, 0.01), "stddev was {s}");
        }
    }

    #[test]
    fn taste_model_build_two_samples() {
        let a = vec![0.0, 1.0];
        let b = vec![1.0, 0.0];
        let model = TasteModel::build(&[a, b]).unwrap();
        assert!(
            approx_eq(model.feature_means[0], 0.5),
            "mean0 was {}",
            model.feature_means[0]
        );
        assert!(
            approx_eq(model.feature_means[1], 0.5),
            "mean1 was {}",
            model.feature_means[1]
        );
        assert!(
            approx_eq(model.feature_stddevs[0], 0.5),
            "std0 was {}",
            model.feature_stddevs[0]
        );
        assert!(
            approx_eq(model.feature_stddevs[1], 0.5),
            "std1 was {}",
            model.feature_stddevs[1]
        );
    }

    #[test]
    fn taste_model_score_at_mean_is_zero() {
        let sample = vec![0.5, 0.3, 0.8];
        let model = TasteModel::build(&[sample.clone()]).unwrap();
        let score = model.score(&sample);
        assert!(approx_eq(score, 0.0), "score was {score}");
    }

    #[test]
    fn taste_model_score_far_from_mean_is_high() {
        let sample = vec![0.5, 0.5];
        let model = TasteModel::build(&[sample]).unwrap();
        let near = vec![0.51, 0.51];
        let far = vec![1.0, 1.0];
        let near_score = model.score(&near);
        let far_score = model.score(&far);
        assert!(
            far_score > near_score,
            "far={far_score} should be > near={near_score}"
        );
    }

    #[test]
    fn taste_engine_inactive_below_threshold() {
        let engine = TasteEngine::new();
        assert!(!engine.is_active(10));
    }

    #[test]
    fn taste_engine_generate_palette_returns_256() {
        let mut engine = TasteEngine::new();
        let palette = engine.generate_palette(10, 1.0, 1.0, 0.0, 1, 10);
        assert_eq!(palette.len(), 256, "palette len was {}", palette.len());
    }

    // --- TransformFeatures tests ---

    #[test]
    fn transform_features_identity_affine() {
        let mut xf = crate::genome::FlameTransform::default();
        xf.weight = 0.5;
        xf.a = 1.0;
        xf.b = 0.0;
        xf.c = 0.0;
        xf.d = 1.0;
        xf.offset = [0.0, 0.0];
        xf.color = 0.3;
        xf.linear = 1.0;
        let f = TransformFeatures::extract(&xf);
        assert!(
            approx_eq(f.affine_determinant, 1.0),
            "det was {}",
            f.affine_determinant
        );
        assert!(
            approx_eq(f.affine_asymmetry, 0.0),
            "asym was {}",
            f.affine_asymmetry
        );
        assert!(approx_eq(f.offset_magnitude, 0.0));
        assert!(approx_eq(f.primary_dominance, 1.0));
        assert!(approx_eq(f.active_variation_count, 1.0));
        assert!(approx_eq(f.color_index, 0.3));
        assert!(approx_eq(f.weight, 0.5));
    }

    #[test]
    fn transform_features_two_variations() {
        let mut xf = crate::genome::FlameTransform::default();
        xf.weight = 1.0;
        xf.a = 0.5;
        xf.b = -0.5;
        xf.c = 0.5;
        xf.d = 0.5;
        xf.offset = [0.3, 0.4];
        xf.color = 0.0;
        xf.spherical = 0.7;
        xf.julia = 0.3;
        let f = TransformFeatures::extract(&xf);
        assert!(approx_eq(f.active_variation_count, 2.0));
        assert!(
            approx_eq(f.primary_dominance, 0.7),
            "dom was {}",
            f.primary_dominance
        );
        // primary_variation_index should be spherical (index 2)
        assert!(
            approx_eq(f.primary_variation_index, 2.0),
            "idx was {}",
            f.primary_variation_index
        );
        // offset magnitude: sqrt(0.09 + 0.16) = 0.5
        assert!(
            approx_eq(f.offset_magnitude, 0.5),
            "offset was {}",
            f.offset_magnitude
        );
        // determinant: |0.5*0.5 - (-0.5)*0.5| = |0.25 + 0.25| = 0.5
        assert!(
            approx_eq(f.affine_determinant, 0.5),
            "det was {}",
            f.affine_determinant
        );
    }

    #[test]
    fn transform_features_vec_length() {
        let xf = crate::genome::FlameTransform::default();
        let f = TransformFeatures::extract(&xf);
        assert_eq!(f.to_vec().len(), TRANSFORM_FEATURE_COUNT);
    }

    #[test]
    fn score_transform_returns_none_without_model() {
        let engine = TasteEngine::new();
        let xf = crate::genome::FlameTransform::default();
        assert!(engine.score_transform(&xf, 1).is_none());
    }

    // --- CompositionFeatures tests ---

    #[test]
    fn composition_features_basic() {
        let genome = crate::genome::FlameGenome::default_genome();
        let f = CompositionFeatures::extract(&genome);
        assert!(
            f.transform_count >= 3.0,
            "transform_count was {}",
            f.transform_count
        );
        assert!(
            f.variation_diversity >= 1.0,
            "diversity was {}",
            f.variation_diversity
        );
        assert!(
            f.mean_determinant > 0.0,
            "mean_det was {}",
            f.mean_determinant
        );
    }

    #[test]
    fn composition_features_vec_length() {
        let genome = crate::genome::FlameGenome::default_genome();
        let f = CompositionFeatures::extract(&genome);
        assert_eq!(f.to_vec().len(), COMPOSITION_FEATURE_COUNT);
    }

    #[test]
    fn score_transform_scores_after_rebuild() {
        let mut engine = TasteEngine::new();
        // Build a minimal genome with palette and transforms
        let mut xf = crate::genome::FlameTransform::default();
        xf.linear = 1.0;
        xf.color = 0.5;
        xf.weight = 1.0;
        let genome = crate::genome::FlameGenome {
            name: String::new(),
            global: crate::genome::GlobalParams {
                speed: 1.0,
                zoom: 1.0,
                trail: 0.9,
                flame_brightness: 1.0,
            },
            kifs: crate::genome::KifsParams {
                fold_angle: 0.0,
                scale: 1.0,
                brightness: 1.0,
            },
            transforms: vec![xf],
            final_transform: None,
            symmetry: 1,
            palette: Some(vec![[1.0, 0.0, 0.0]; 256]),
            parent_a: None,
            parent_b: None,
            generation: 0,
        };

        engine.rebuild(&[&genome], 10);

        let score = engine.score_transform(&genome.transforms[0], 1);
        assert!(score.is_some(), "should have a score after rebuild");
        // Scoring the same transform used to build the model should give ~0
        assert!(
            approx_eq(score.unwrap(), 0.0),
            "score was {}",
            score.unwrap()
        );
    }
}
