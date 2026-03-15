use std::collections::VecDeque;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::genome::FlameGenome;
use crate::weights::RuntimeConfig;

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
    /// IGMM model for multi-style taste learning
    igmm: IgmmModel,
    /// Recent palette features for diversity nudge
    recent_palettes: VecDeque<PaletteFeatures>,
    /// All feature vectors from good genomes (for rebuilding model)
    good_features: Vec<Vec<f32>>,
    /// Config for proxy render parameters
    config: RuntimeConfig,
}

impl TasteEngine {
    pub fn new() -> Self {
        // Use serde defaults (not derive Default which zeros everything)
        let config: RuntimeConfig = serde_json::from_str("{}").unwrap_or_default();
        Self {
            model: None,
            transform_model: None,
            igmm: IgmmModel::new(),
            recent_palettes: VecDeque::new(),
            good_features: Vec::new(),
            config,
        }
    }

    /// Update the runtime config (called when weights are reloaded).
    pub fn set_config(&mut self, cfg: &RuntimeConfig) {
        self.config = cfg.clone();
    }

    /// Build a full feature vector from a genome (palette + composition + perceptual).
    pub fn extract_full_features(&self, genome: &FlameGenome) -> Option<Vec<f32>> {
        let palette_feats = PaletteFeatures::extract(genome)?;
        let mut features_vec = palette_feats.to_vec();
        features_vec.extend(CompositionFeatures::extract(genome).to_vec());
        features_vec.extend(PerceptualFeatures::from_genome(genome, &self.config).to_vec());
        Some(features_vec)
    }

    /// Rebuild with an explicit IGMM save path.
    /// Tries loading from disk first; if missing, bootstraps from genomes.
    pub fn rebuild_with_igmm_path(
        &mut self,
        good_genomes: &[&FlameGenome],
        recent_memory: usize,
        igmm_path: Option<&Path>,
    ) {
        self.good_features.clear();
        let mut transform_features: Vec<Vec<f32>> = Vec::new();

        for genome in good_genomes {
            if let Some(features_vec) = self.extract_full_features(genome) {
                self.good_features.push(features_vec);
            }
            for xf in &genome.transforms {
                transform_features.push(TransformFeatures::extract(xf).to_vec());
            }
        }

        self.model = TasteModel::build(&self.good_features);
        self.transform_model = TasteModel::build(&transform_features);

        // IGMM: try loading from disk, fall back to cold-start bootstrap
        let loaded = igmm_path.and_then(|p| IgmmModel::load(p).ok());
        if let Some(model) = loaded {
            self.igmm = model;
            eprintln!(
                "[taste] IGMM loaded from disk: {} clusters",
                self.igmm.clusters.len()
            );
        } else {
            // Cold-start: feed all good features through on_upvote
            self.igmm = IgmmModel::new();
            for features in &self.good_features {
                self.igmm.on_upvote(features, &self.config);
            }
            if !self.igmm.clusters.is_empty() {
                eprintln!(
                    "[taste] IGMM cold-start: {} clusters from {} genomes",
                    self.igmm.clusters.len(),
                    self.good_features.len()
                );
            }
        }

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

    /// Handle an upvote: extract features, update IGMM, save model.
    pub fn on_upvote(&mut self, genome: &FlameGenome, save_path: Option<&Path>) {
        if let Some(features) = self.extract_full_features(genome) {
            self.igmm.on_upvote(&features, &self.config);
            if let Some(path) = save_path
                && let Err(e) = self.igmm.save(path)
            {
                eprintln!("[taste] IGMM save error: {e}");
            }
        }
    }

    /// Score a genome using the IGMM model (lower = better).
    /// Falls back to Gaussian centroid if IGMM has no clusters.
    pub fn score_genome(&self, genome: &FlameGenome) -> Option<f32> {
        let features = self.extract_full_features(genome)?;
        if !self.igmm.clusters.is_empty() {
            Some(self.igmm.score(&features))
        } else {
            self.model.as_ref().map(|m| m.score(&features))
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

// ── Perceptual Features ──

/// Grid size for proxy render (always 64x64 for box counting).
const PROXY_GRID: usize = 64;

/// Number of perceptual features.
pub const PERCEPTUAL_FEATURE_COUNT: usize = 3;

/// Perceptual features extracted from a CPU proxy render of a genome.
#[derive(Clone, Debug)]
pub struct PerceptualFeatures {
    pub fractal_dimension: f32,
    pub spatial_entropy: f32,
    pub coverage_ratio: f32,
}

impl PerceptualFeatures {
    /// Extract perceptual features from a genome using a CPU proxy render.
    pub fn from_genome(genome: &FlameGenome, cfg: &RuntimeConfig) -> Self {
        let grid = proxy_render(genome, cfg);
        Self {
            fractal_dimension: box_counting_fd(&grid),
            spatial_entropy: spatial_entropy(&grid, cfg.spatial_entropy_blocks as usize),
            coverage_ratio: coverage_ratio(&grid),
        }
    }

    /// Convert to flat f32 vector.
    pub fn to_vec(&self) -> Vec<f32> {
        let v = vec![
            self.fractal_dimension,
            self.spatial_entropy,
            self.coverage_ratio,
        ];
        debug_assert_eq!(v.len(), PERCEPTUAL_FEATURE_COUNT);
        v
    }
}

/// Compute the linear regression slope of y vs x.
/// Returns 0.0 if fewer than 2 points.
fn linear_regression_slope(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return 0.0;
    }
    let n_f = n as f32;
    let sum_x: f32 = xs[..n].iter().sum();
    let sum_y: f32 = ys[..n].iter().sum();
    let sum_xy: f32 = xs[..n].iter().zip(ys[..n].iter()).map(|(x, y)| x * y).sum();
    let sum_xx: f32 = xs[..n].iter().map(|x| x * x).sum();
    let denom = n_f * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    (n_f * sum_xy - sum_x * sum_y) / denom
}

/// Box-counting fractal dimension estimate.
/// Uses box sizes [2, 4, 8, 16, 32] on a 64x64 boolean grid.
/// Returns slope of log(count) vs log(1/size).
pub fn box_counting_fd(grid: &[[bool; PROXY_GRID]; PROXY_GRID]) -> f32 {
    let box_sizes: [usize; 5] = [2, 4, 8, 16, 32];
    let mut log_inv_sizes = Vec::with_capacity(box_sizes.len());
    let mut log_counts = Vec::with_capacity(box_sizes.len());

    for &size in &box_sizes {
        let mut count = 0u32;
        let blocks = PROXY_GRID / size;
        for by in 0..blocks {
            for bx in 0..blocks {
                let mut has_point = false;
                'search: for dy in 0..size {
                    for dx in 0..size {
                        if grid[by * size + dy][bx * size + dx] {
                            has_point = true;
                            break 'search;
                        }
                    }
                }
                if has_point {
                    count += 1;
                }
            }
        }
        if count > 0 {
            log_inv_sizes.push((1.0 / size as f32).ln());
            log_counts.push((count as f32).ln());
        }
    }

    linear_regression_slope(&log_inv_sizes, &log_counts)
}

/// Shannon entropy of spatial block hit counts.
/// Divides the grid into blocks_per_side x blocks_per_side blocks.
pub fn spatial_entropy(grid: &[[bool; PROXY_GRID]; PROXY_GRID], blocks_per_side: usize) -> f32 {
    let blocks_per_side = blocks_per_side.max(1);
    let block_size = PROXY_GRID / blocks_per_side;
    if block_size == 0 {
        return 0.0;
    }
    let total_blocks = blocks_per_side * blocks_per_side;
    let mut block_hits = vec![0u32; total_blocks];

    for (y, row) in grid.iter().enumerate() {
        for (x, &cell) in row.iter().enumerate() {
            if cell {
                let bx = (x / block_size).min(blocks_per_side - 1);
                let by = (y / block_size).min(blocks_per_side - 1);
                block_hits[by * blocks_per_side + bx] += 1;
            }
        }
    }

    let total_hits: u32 = block_hits.iter().sum();
    if total_hits == 0 {
        return 0.0;
    }
    let total_f = total_hits as f32;
    let mut entropy = 0.0f32;
    for &count in &block_hits {
        if count > 0 {
            let p = count as f32 / total_f;
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// Fraction of 64x64 grid cells that are hit.
pub fn coverage_ratio(grid: &[[bool; PROXY_GRID]; PROXY_GRID]) -> f32 {
    let total = (PROXY_GRID * PROXY_GRID) as f32;
    let hits: u32 = grid
        .iter()
        .map(|row| row.iter().filter(|&&c| c).count() as u32)
        .sum();
    hits as f32 / total
}

/// Lightweight CPU chaos game — affine-only (no variation functions).
/// Renders into a 64x64 boolean grid. Skips warmup iterations.
pub fn proxy_render(genome: &FlameGenome, cfg: &RuntimeConfig) -> [[bool; PROXY_GRID]; PROXY_GRID] {
    let mut grid = [[false; PROXY_GRID]; PROXY_GRID];
    if genome.transforms.is_empty() {
        return grid;
    }

    let iterations = cfg.proxy_render_iterations as usize;
    let warmup = cfg.proxy_render_warmup as usize;

    // Build weight-based CDF for transform selection
    let total_weight: f32 = genome.transforms.iter().map(|xf| xf.weight.max(0.0)).sum();
    if total_weight <= 0.0 {
        return grid;
    }
    let mut cdf = Vec::with_capacity(genome.transforms.len());
    let mut accum = 0.0f32;
    for xf in &genome.transforms {
        accum += xf.weight.max(0.0) / total_weight;
        cdf.push(accum);
    }

    // Simple LCG PRNG (deterministic, no external state needed)
    let mut seed: u32 = 0xDEAD_BEEF;
    let lcg_next = |s: &mut u32| -> f32 {
        *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (*s as f32) / u32::MAX as f32
    };

    let mut x = 0.0f32;
    let mut y = 0.0f32;

    for i in 0..(warmup + iterations) {
        // Pick transform by weight
        let r = lcg_next(&mut seed);
        let xf_idx = cdf.iter().position(|&c| r <= c).unwrap_or(cdf.len() - 1);
        let xf = &genome.transforms[xf_idx];

        // Apply affine: [a b; c d] * [x; y] + offset
        let nx = xf.a * x + xf.b * y + xf.offset[0];
        let ny = xf.c * x + xf.d * y + xf.offset[1];

        // NaN/infinity escape check
        if !nx.is_finite() || !ny.is_finite() {
            x = 0.0;
            y = 0.0;
            continue;
        }
        x = nx;
        y = ny;

        // Plot after warmup
        if i >= warmup {
            // Map [-2, 2] to [0, 64)
            let gx = ((x + 2.0) / 4.0 * PROXY_GRID as f32) as i32;
            let gy = ((y + 2.0) / 4.0 * PROXY_GRID as f32) as i32;
            if gx >= 0 && gx < PROXY_GRID as i32 && gy >= 0 && gy < PROXY_GRID as i32 {
                grid[gy as usize][gx as usize] = true;
            }
        }
    }

    grid
}

// ── IGMM Taste Model ──

/// A single cluster in the Incremental Gaussian Mixture Model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TasteCluster {
    pub mean: Vec<f32>,
    pub variance: Vec<f32>,
    pub weight: f32,
    pub sample_count: u32,
}

impl TasteCluster {
    /// Create a new cluster from an initial feature vector.
    pub fn new(features: &[f32]) -> Self {
        Self {
            mean: features.to_vec(),
            variance: vec![1.0; features.len()],
            weight: 1.0,
            sample_count: 1,
        }
    }

    /// Mahalanobis distance (diagonal covariance) from features to this cluster.
    pub fn mahalanobis_distance(&self, features: &[f32]) -> f32 {
        self.mean
            .iter()
            .zip(self.variance.iter())
            .zip(features.iter())
            .map(|((m, v), f)| {
                let diff = f - m;
                diff * diff / v.max(1e-6)
            })
            .sum::<f32>()
            .sqrt()
    }

    /// Update cluster mean and variance via exponential moving average.
    pub fn update(&mut self, features: &[f32], learning_rate: f32) {
        self.sample_count += 1;
        for (m, (v, f)) in self
            .mean
            .iter_mut()
            .zip(self.variance.iter_mut().zip(features.iter()))
        {
            let diff = f - *m;
            *m += learning_rate * diff;
            *v += learning_rate * (diff * diff - *v);
        }
    }
}

/// Incremental Gaussian Mixture Model for taste scoring.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IgmmModel {
    pub clusters: Vec<TasteCluster>,
}

impl IgmmModel {
    pub fn new() -> Self {
        Self {
            clusters: Vec::new(),
        }
    }

    /// Process an upvoted genome's features.
    /// Finds closest cluster and merges, or spawns a new one.
    pub fn on_upvote(&mut self, features: &[f32], cfg: &RuntimeConfig) {
        if self.clusters.is_empty() {
            self.clusters.push(TasteCluster::new(features));
            return;
        }

        // Find closest cluster
        let (closest_idx, closest_dist) = self
            .clusters
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.mahalanobis_distance(features)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        if closest_dist < cfg.igmm_activation_threshold {
            // Merge into existing cluster
            self.clusters[closest_idx].update(features, cfg.igmm_learning_rate);
            self.clusters[closest_idx].weight += 1.0;
        } else if (self.clusters.len() as u32) < cfg.igmm_max_clusters {
            // Spawn new cluster
            self.clusters.push(TasteCluster::new(features));
        } else {
            // At max clusters: merge into closest anyway
            self.clusters[closest_idx].update(features, cfg.igmm_learning_rate);
            self.clusters[closest_idx].weight += 1.0;
        }

        // Decay all cluster weights
        for cluster in &mut self.clusters {
            cluster.weight *= cfg.igmm_decay_rate;
        }

        // Prune clusters below minimum weight
        self.clusters.retain(|c| c.weight >= cfg.igmm_min_weight);
    }

    /// Score features against the IGMM model.
    /// Returns minimum Mahalanobis distance across all clusters (lower = better).
    pub fn score(&self, features: &[f32]) -> f32 {
        if self.clusters.is_empty() {
            return f32::MAX;
        }
        self.clusters
            .iter()
            .map(|c| c.mahalanobis_distance(features))
            .fold(f32::MAX, f32::min)
    }

    /// Save IGMM model to a JSON file.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json =
            serde_json::to_string_pretty(self).map_err(|e| format!("serialize igmm: {e}"))?;
        std::fs::write(path, json).map_err(|e| format!("write igmm: {e}"))?;
        Ok(())
    }

    /// Load IGMM model from a JSON file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| format!("read igmm: {e}"))?;
        serde_json::from_str(&json).map_err(|e| format!("parse igmm: {e}"))
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

        engine.rebuild_with_igmm_path(&[&genome], 10, None);

        let score = engine.score_transform(&genome.transforms[0], 1);
        assert!(score.is_some(), "should have a score after rebuild");
        // Scoring the same transform used to build the model should give ~0
        assert!(
            approx_eq(score.unwrap(), 0.0),
            "score was {}",
            score.unwrap()
        );
    }

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

    // --- Linear regression tests ---

    #[test]
    fn linear_regression_slope_perfect_line() {
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys = vec![2.0, 4.0, 6.0, 8.0]; // y = 2x
        let slope = linear_regression_slope(&xs, &ys);
        assert!(approx_eq(slope, 2.0), "slope was {slope}");
    }

    #[test]
    fn linear_regression_slope_too_few_points() {
        assert!(approx_eq(linear_regression_slope(&[1.0], &[1.0]), 0.0));
        assert!(approx_eq(linear_regression_slope(&[], &[]), 0.0));
    }

    // --- Box-counting FD tests ---

    #[test]
    fn box_counting_fd_empty_grid() {
        let grid = [[false; PROXY_GRID]; PROXY_GRID];
        let fd = box_counting_fd(&grid);
        // Empty grid: no occupied boxes at any scale → FD = 0
        assert!(approx_eq(fd, 0.0), "FD was {fd}");
    }

    #[test]
    fn box_counting_fd_full_grid() {
        let grid = [[true; PROXY_GRID]; PROXY_GRID];
        let fd = box_counting_fd(&grid);
        // Full grid: FD should be close to 2.0 (fills 2D space)
        assert!(fd > 1.8 && fd < 2.2, "FD was {fd}, expected ~2.0");
    }

    #[test]
    fn box_counting_fd_diagonal_line() {
        let mut grid = [[false; PROXY_GRID]; PROXY_GRID];
        for i in 0..PROXY_GRID {
            grid[i][i] = true;
        }
        let fd = box_counting_fd(&grid);
        // Diagonal line: FD should be close to 1.0
        assert!(fd > 0.7 && fd < 1.3, "FD was {fd}, expected ~1.0");
    }

    // --- Spatial entropy tests ---

    #[test]
    fn spatial_entropy_uniform_grid() {
        // Every block has equal hits → maximum entropy
        let mut grid = [[false; PROXY_GRID]; PROXY_GRID];
        // Place one hit in each 8x8 block
        for by in 0..8 {
            for bx in 0..8 {
                grid[by * 8][bx * 8] = true;
            }
        }
        let entropy = spatial_entropy(&grid, 8);
        // Maximum entropy for 64 equally-occupied blocks = ln(64) ≈ 4.16
        let max_entropy = (64.0f32).ln();
        assert!(
            (entropy - max_entropy).abs() < 0.1,
            "entropy was {entropy}, expected ~{max_entropy}"
        );
    }

    #[test]
    fn spatial_entropy_single_block() {
        // All hits in a single block → zero entropy
        let mut grid = [[false; PROXY_GRID]; PROXY_GRID];
        for y in 0..8 {
            for x in 0..8 {
                grid[y][x] = true;
            }
        }
        let entropy = spatial_entropy(&grid, 8);
        assert!(
            approx_eq(entropy, 0.0),
            "entropy was {entropy}, expected 0.0"
        );
    }

    #[test]
    fn spatial_entropy_empty_grid() {
        let grid = [[false; PROXY_GRID]; PROXY_GRID];
        let entropy = spatial_entropy(&grid, 8);
        assert!(approx_eq(entropy, 0.0), "entropy was {entropy}");
    }

    // --- Coverage ratio tests ---

    #[test]
    fn coverage_ratio_empty() {
        let grid = [[false; PROXY_GRID]; PROXY_GRID];
        assert!(approx_eq(coverage_ratio(&grid), 0.0));
    }

    #[test]
    fn coverage_ratio_full() {
        let grid = [[true; PROXY_GRID]; PROXY_GRID];
        assert!(approx_eq(coverage_ratio(&grid), 1.0));
    }

    // --- Proxy render tests ---

    #[test]
    fn proxy_render_degenerate_genome_no_panic() {
        // Genome with diverging transforms (large coefficients) should not panic
        let mut xf = crate::genome::FlameTransform::default();
        xf.a = 100.0;
        xf.b = 100.0;
        xf.c = 100.0;
        xf.d = 100.0;
        xf.offset = [50.0, 50.0];
        xf.weight = 1.0;
        xf.linear = 1.0;
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
            palette: None,
            parent_a: None,
            parent_b: None,
            generation: 0,
        };
        let cfg = default_test_config();
        let _grid = proxy_render(&genome, &cfg);
        // Just checking it doesn't panic
    }

    #[test]
    fn proxy_render_contractive_produces_hits() {
        // Identity-ish transform with small contraction should produce hits
        let mut xf = crate::genome::FlameTransform::default();
        xf.a = 0.5;
        xf.b = 0.0;
        xf.c = 0.0;
        xf.d = 0.5;
        xf.offset = [0.5, 0.5];
        xf.weight = 1.0;
        xf.linear = 1.0;
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
            palette: None,
            parent_a: None,
            parent_b: None,
            generation: 0,
        };
        let cfg = default_test_config();
        let grid = proxy_render(&genome, &cfg);
        let hits: u32 = grid
            .iter()
            .map(|row| row.iter().filter(|&&c| c).count() as u32)
            .sum();
        assert!(hits > 0, "contractive transform should produce hits");
    }

    // --- PerceptualFeatures tests ---

    #[test]
    fn perceptual_features_vec_length() {
        let genome = crate::genome::FlameGenome::default_genome();
        let cfg = default_test_config();
        let f = PerceptualFeatures::from_genome(&genome, &cfg);
        assert_eq!(f.to_vec().len(), PERCEPTUAL_FEATURE_COUNT);
    }

    fn default_test_config() -> crate::weights::RuntimeConfig {
        serde_json::from_str("{}").unwrap()
    }

    // --- TasteCluster tests ---

    #[test]
    fn taste_cluster_from_features() {
        let features = vec![1.0, 2.0, 3.0];
        let cluster = TasteCluster::new(&features);
        assert_eq!(cluster.mean, vec![1.0, 2.0, 3.0]);
        assert_eq!(cluster.sample_count, 1);
        assert!(approx_eq(cluster.weight, 1.0));
        // Distance from own mean should be 0
        let dist = cluster.mahalanobis_distance(&features);
        assert!(approx_eq(dist, 0.0), "dist was {dist}");
    }

    // --- IGMM tests ---

    #[test]
    fn igmm_update_merges_nearby_vote() {
        let cfg = default_test_config();
        let mut model = IgmmModel::new();
        let f1 = vec![1.0, 2.0, 3.0];
        let f2 = vec![1.1, 2.1, 3.1]; // very close to f1
        model.on_upvote(&f1, &cfg);
        model.on_upvote(&f2, &cfg);
        assert_eq!(
            model.clusters.len(),
            1,
            "close features should merge into 1 cluster, got {}",
            model.clusters.len()
        );
    }

    #[test]
    fn igmm_spawns_new_cluster_for_distant_vote() {
        let cfg = default_test_config();
        let mut model = IgmmModel::new();
        let f1 = vec![0.0, 0.0, 0.0];
        let f2 = vec![100.0, 100.0, 100.0]; // very far from f1
        model.on_upvote(&f1, &cfg);
        model.on_upvote(&f2, &cfg);
        assert_eq!(
            model.clusters.len(),
            2,
            "distant features should spawn 2 clusters, got {}",
            model.clusters.len()
        );
    }

    #[test]
    fn igmm_score_picks_closest_cluster() {
        let cfg = default_test_config();
        let mut model = IgmmModel::new();
        let f1 = vec![0.0, 0.0, 0.0];
        let f2 = vec![100.0, 100.0, 100.0];
        model.on_upvote(&f1, &cfg);
        model.on_upvote(&f2, &cfg);

        // Point near f1 should score lower than point far from both
        let near_f1 = vec![0.1, 0.1, 0.1];
        let far = vec![50.0, 50.0, 50.0];
        let score_near = model.score(&near_f1);
        let score_far = model.score(&far);
        assert!(
            score_near < score_far,
            "near={score_near} should be < far={score_far}"
        );
    }

    #[test]
    fn igmm_persistence_roundtrip() {
        let cfg = default_test_config();
        let mut model = IgmmModel::new();
        model.on_upvote(&vec![1.0, 2.0, 3.0], &cfg);
        model.on_upvote(&vec![100.0, 200.0, 300.0], &cfg);

        let dir = std::env::temp_dir().join("taste_igmm_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_igmm.json");
        model.save(&path).expect("save should succeed");

        let loaded = IgmmModel::load(&path).expect("load should succeed");
        assert_eq!(model.clusters.len(), loaded.clusters.len());
        for (orig, load) in model.clusters.iter().zip(loaded.clusters.iter()) {
            assert_eq!(orig.mean.len(), load.mean.len());
            for (a, b) in orig.mean.iter().zip(load.mean.iter()) {
                assert!(approx_eq(*a, *b), "mean mismatch: {a} vs {b}");
            }
        }

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
