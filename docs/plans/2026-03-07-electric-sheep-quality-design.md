# Electric Sheep Quality Upgrade — Design

## Vision

Close the visual quality gap with Electric Sheep through progressive accumulation, proper flam3-style tonemapping, curated genomes, and cinematic post-processing. Audio reactivity is a differentiator — Electric Sheep doesn't have it.

## Phase 1: "Make It Beautiful"

### 1. Accumulation System

**Problem:** We render ~7 samples/pixel/frame. Electric Sheep renders 10,000-100,000 spp. Ours looks grainy.

**Solution:** Persistent float32 accumulation buffer with exponential decay.

**Architecture:**
- New GPU buffer: `accumulation_buffer: array<f32>` — float32x4 per pixel (density, R, G, B). ~7.7MB at 800x600.
- New compute shader: `accumulation.wgsl` — per-pixel: `accum[px] = accum[px] * decay + histogram[px]`
- Fragment shader reads from accumulation buffer instead of raw histogram
- DO NOT clear on mutation — the morph + decay handles transitions naturally (0.995^480 = 9% after 8s morph)
- Reduce workgroups from 512 to 128-256 (configurable) — accumulation compensates

**Effective quality:** At decay 0.995, ~200 frame effective window. At 128 workgroups (~820K pts/frame), steady-state ~342 spp. At 256 workgroups, ~684 spp. 50-100x improvement over current.

**Config params (weights.json):**
- `accumulation_decay` — exponential decay per frame (default 0.995, half-life ~2.3s at 60fps)
- `samples_per_frame` — workgroups, replaces current `workgroups` param

**Interaction with audio:** Audio-driven parameter shifts create ~2% spatial blur in accumulation — intentional, creates gentle breathing rather than jitter. Accumulation naturally dampens audio reactivity into something organic.

**Trail feedback:** Becomes mostly redundant. Keep a small trail value (0.1-0.15) in the display shader for temporal anti-aliasing, but the accumulation buffer is the primary persistence mechanism.

### 2. Flam3-Style Tonemapping

**Problem:** Our tonemapping is ad-hoc. Dim structures are invisible, bright areas clip, exposure changes with sample count.

**Solution:** Implement the actual flam3 tonemapping algorithm.

**The flam3 algorithm:**
1. Compute `k2 = 1.0 / (area * sample_density * brightness)` — normalizes for resolution and sample count
2. Per pixel: `logscale = brightness * log(1 + density * k2) / density`
3. `alpha = logscale * density`
4. Vibrancy blend: `ls = vibrancy * alpha + (1 - vibrancy) * pow(alpha, 1/gamma)`
5. Final color: `pixel = ls * color + (1 - vibrancy) * pow(alpha, 1/gamma)`

**Key difference from current:** Normalization by sample count means consistent brightness regardless of workgroup count or accumulation depth. Gamma is applied INSIDE the vibrancy calculation, not as a blanket final step.

**Config params:**
- `flame_brightness` — sensitivity control (existing, reinterpreted)
- `vibrancy` — color saturation control (existing, algorithm changes)
- `gamma` — contrast curve (existing, moves into vibrancy calc)

### 3. Curated Seed Genomes

**Problem:** Our mutations start from a random default genome. Electric Sheep starts from artist-curated genomes.

**Solution:** Hand-craft 15-20 beautiful seed genomes.

**Structure:**
- `genomes/seeds/` — curated library, read-only, ships with the project
- `genomes/favorites/` — user saves (existing save behavior, redirected here)
- Startup loads random seed instead of `default_genome()` + 3 mutations
- Mutations prefer seeds as starting points (70% seed, 30% current genome — configurable)

**Seed design guidelines:**
- At least 5-8 transforms per seed
- Good offset spread (multiple quadrants)
- Diverse variation combos (not all linear/sinusoidal)
- Symmetry 3-8 for many seeds (rotational symmetry produces stunning patterns)
- Mix of contractive and expansive variations

### 4. Bloom + Post-Processing

**Bloom:**
- Multi-radius temporal feedback: 4 taps each at 3 configurable radii
- The feedback loop compounds bloom across frames into a wide soft halo
- Config: `bloom_intensity`, `bloom_radius`

**Available at zero default:**
- Chromatic aberration: `chromatic_strength: 0.0`
- Vignette: `vignette_strength: 0.0`
- Film grain: `film_grain: 0.0`

### 5. Config

ALL numeric values in `weights.json` `_config` section. No magic numbers anywhere.

---

## Phase 2: "Match Electric Sheep" (COMMITTED — not optional)

### 1. Full Affine Transforms

**Problem:** Our transforms use angle+scale+offset (4 params). Flam3 uses full 2x2 affine matrix + translation (6 params: a, b, c, d, e, f). Many beautiful flames require shear and non-uniform scale.

**Changes:**
- Expand `FlameTransform` struct: replace `angle: f32, scale: f32` with `affine: [f32; 6]`
- Update GPU buffer layout (32 → 34 floats per transform, or repack)
- Update compute shader: matrix multiply instead of angle/scale rotation
- Update all mutation code for 6-param affine
- Update CPU attractor estimator

### 2. Flam3 XML Importer

**Problem:** Thousands of beautiful artist-curated genomes exist in `.flame` XML format. We can't use them.

**Solution:** Parse flam3 XML → our FlameGenome JSON.

**Mapping:**
- `<flame>` → `FlameGenome`
- `<xform>` → `FlameTransform` with full affine
- `<color>` → palette entries
- `<symmetry>` → our symmetry field
- Variation names map 1:1 for the ~26 we support

### 3. Per-Genome Color Palettes

**Problem:** Cosine palettes are generic. Electric Sheep uses 256-entry color lookup tables per genome. Each sheep has a unique color identity.

**Solution:**
- Add `palette: Vec<[f32; 3]>` (256 entries) to FlameGenome
- Upload as 1D texture (256x1 RGBA)
- Compute shader samples palette texture instead of computing cosine function
- Import palette from flam3 `<color>` elements
- Keep cosine palette as fallback for genomes without custom palettes

### 4. Parametric Variations

**Problem:** Our variations are fixed functions. Flam3 variations have parameters (e.g., `julian_power`, `blob_low/high/waves`, `rings2_val`).

**Solution:**
- Add `variation_params: HashMap<String, f32>` to FlameTransform
- Pack into GPU buffer alongside variation weights
- Implement parametric versions: `julian(power, dist)`, `juliascope(power, dist)`, `ngon(power, sides, corners, circle)`, `blob(low, high, waves)`, `fan2(x, y)`, `rings2(val)`
- Import params from flam3 XML

### 5. Fitness-Biased Mutation

**Problem:** Random mutations explore blindly. User taste is wasted.

**Solution:**
- Analyze saved favorites: which variations, symmetry, transform count, scale distributions
- When mutating, bias variation picks toward combos seen in favorites
- Soft bias (70/30), not hard constraint — keeps exploration alive
- Config: `fitness_bias_strength` in weights.json

---

## Success Criteria

**Phase 1 complete when:**
- Accumulation buffer working, image builds to smooth quality over 10-15 seconds
- Tonemapping matches flam3 algorithm — dim filaments visible, highlights preserved
- 15+ curated seed genomes ship with the project
- Bloom produces soft halo around bright structures
- All params in weights.json, no magic numbers
- Audio reactivity affects atmosphere (color, bloom, vibrancy), not geometry
- Framerate stays above 30fps on MacBook Pro

**Phase 2 complete when:**
- Can import any `.flame` file from the community and render it faithfully
- Custom color palettes per genome
- Parametric variations working
- Favorites influence mutation direction
- Visual quality approaches Electric Sheep in motion (won't match still frames due to real-time constraint)
