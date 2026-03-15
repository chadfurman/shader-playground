# Evolution Overhaul Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development to implement this plan. Track 1 and Track 2 are independent and MUST run as parallel subagents in separate worktrees. The merge task runs after both complete.

**Goal:** Overhaul the fractal flame evolution system with 3D rendering (Sdobnov hack) and an intelligent multi-style taste engine (IGMM + MAP-Elites + novelty search).

**Architecture:** Two parallel tracks. Track 1 extends the genome and compute shader into 3D space. Track 2 replaces the single-centroid taste model with a clustering-based IGMM, adds perceptual features from a CPU proxy render, and introduces a MAP-Elites diversity archive. Tracks merge at the end when 3D features feed into the IGMM.

**Tech Stack:** Rust, wgpu 28, WGSL compute shaders, serde_json for persistence

**Spec:** `.plans/2026-03-14-evolution-overhaul-design.md`

---

## File Map

### Track 1 (3D Rendering) — branch: `feat/3d-rendering`
| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/genome.rs` | FlameTransform 3x3 affine fields, migration, z-mutations, flatten_transforms, clamp_determinant_3x3 |
| Modify | `src/weights.rs` | Camera/DOF config fields, PARAMS_PER_XF update (42→48), XF_FIELDS extension |
| Modify | `flame_compute.wgsl` | 3D iteration loop, 3x3 affine application, camera projection, velocity projection |
| Modify | `playground.wgsl` | DOF post-process blur using z-depth channel |
| Modify | `src/main.rs` | Point state buffer resize (3→7 floats), camera uniform passing (extra7/extra8), Uniforms struct |
| Modify | `weights.json` | Camera/DOF default values |

### Track 2 (Taste Engine) — branch: `feat/taste-engine-v2`
| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/taste.rs` | IGMM model (TasteCluster, clustering logic), perceptual features (proxy render, FD, entropy, coverage), novelty scoring |
| Create | `src/archive.rs` | MAP-Elites archive grid, persistence, parent selection |
| Modify | `src/genome.rs` | Interpolative crossover in breed(), affine lerp helper |
| Modify | `src/weights.rs` | IGMM/novelty/interpolation config fields |
| Modify | `src/main.rs` | Archive initialization, archive insertion after breed, parent selection from archive, IGMM persistence on vote |
| Modify | `src/main.rs` (top) | `mod archive;` declaration (no `lib.rs` — this is a binary crate) |
| Modify | `weights.json` | IGMM/novelty/interpolation default values |

### Merge Task — branch: `main`
| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/taste.rs` | Update feature extraction to use 3x3 affine fields |
| Modify | `src/genome.rs` | Resolve any conflicts between 3D affine + interpolative crossover |
| Modify | `src/weights.rs` | Combine both tracks' config additions |

---

## Track 1: 3D Rendering

### Task T1.1: Genome Format — 3x3 Affines

**Files:**
- Modify: `src/genome.rs` (FlameTransform struct, serialization, clamp_determinant)

- [ ] **Step 1: Write failing test for 3x3 affine default**

In `src/genome.rs` test module, add:
```rust
#[test]
fn default_transform_has_3x3_identity_z() {
    let xf = FlameTransform::default();
    // z-row and z-column should be identity
    assert_eq!(xf.affine[2][2], 1.0);
    assert_eq!(xf.affine[2][0], 0.0);
    assert_eq!(xf.affine[2][1], 0.0);
    assert_eq!(xf.affine[0][2], 0.0);
    assert_eq!(xf.affine[1][2], 0.0);
    assert_eq!(xf.offset[2], 0.0);
}
```

- [ ] **Step 2: Run test, verify it fails**

Run: `cargo test default_transform_has_3x3_identity_z -- --nocapture`
Expected: FAIL — `affine` field doesn't exist yet

- [ ] **Step 3: Replace a,b,c,d with affine + migrate offset**

In `FlameTransform`:
- Remove fields `a, b, c, d: f32` and `offset: [f32; 2]`
- Add `affine: [[f32; 3]; 3]` and `offset: [f32; 3]`
- Add `#[serde(default)]` on both for migration
- Add custom deserializer that checks for old `a,b,c,d` fields and constructs `affine = [[a,b,0],[c,d,0],[0,0,1]]`
- Update `Default` impl: `affine: [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]`, `offset: [0.0,0.0,0.0]`
- Add helper accessors `fn a(&self) -> f32 { self.affine[0][0] }` etc. for code that still uses 2D semantics

- [ ] **Step 4: Update all references to a,b,c,d throughout genome.rs**

Every `xf.a`, `xf.b`, `xf.c`, `xf.d` → use the accessor helpers. Every `xf.offset[0]`, `xf.offset[1]` stays the same (array grew but indices unchanged).

**IMPORTANT:** `rotate_affine`, `scale_affine`, `shear_affine` take `&mut f32` references to individual fields — these must be rewritten to operate on `&mut [[f32;3];3]` directly. Also update `apply_xform_cpu` which is used by `estimate_attractor_extent`.

- [ ] **Step 5: Update clamp_determinant for 3x3**

```rust
pub fn clamp_determinant(&mut self) {
    let m = &self.affine;
    let det = (m[0][0] * (m[1][1]*m[2][2] - m[1][2]*m[2][1])
             - m[0][1] * (m[1][0]*m[2][2] - m[1][2]*m[2][0])
             + m[0][2] * (m[1][0]*m[2][1] - m[1][1]*m[2][0])).abs();
    if !(0.2..=0.95).contains(&det) {
        let target = det.clamp(0.2, 0.95);
        let scale = (target / det.max(1e-10)).cbrt();
        for row in &mut self.affine {
            for val in row {
                *val *= scale;
            }
        }
    }
}
```

- [ ] **Step 6: Write test for 3x3 determinant clamping**

```rust
#[test]
fn clamp_determinant_3x3_too_small() {
    let mut xf = FlameTransform::default();
    // Scale down to make det < 0.2
    xf.affine = [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]];
    xf.clamp_determinant();
    let det = compute_3x3_det(&xf.affine);
    assert!(det >= 0.19, "det={det} should be >= 0.2");
}
```

- [ ] **Step 7: Write test for old genome migration (a,b,c,d → affine)**

```rust
#[test]
fn genome_migration_abcd_to_affine() {
    let json = r#"{"a":0.5,"b":0.1,"c":-0.1,"d":0.5,"offset":[1.0,2.0],"weight":1.0}"#;
    let xf: FlameTransform = serde_json::from_str(json).unwrap();
    assert!((xf.affine[0][0] - 0.5).abs() < 1e-6);
    assert!((xf.affine[0][1] - 0.1).abs() < 1e-6);
    assert!((xf.affine[1][0] - -0.1).abs() < 1e-6);
    assert!((xf.affine[1][1] - 0.5).abs() < 1e-6);
    assert_eq!(xf.affine[2][2], 1.0); // z identity
    assert!((xf.offset[0] - 1.0).abs() < 1e-6);
    assert!((xf.offset[1] - 2.0).abs() < 1e-6);
    assert_eq!(xf.offset[2], 0.0);
}
```

- [ ] **Step 8: Run all tests, fix any breakage**

Run: `cargo test`
Expected: All 89+ tests pass. Many existing tests reference `xf.a` etc — these need updating to use the accessors.

- [ ] **Step 9: Commit**

```
git add src/genome.rs
git commit -m "feat: migrate FlameTransform to 3x3 affine matrix"
```

---

### Task T1.2: Transform Buffer Layout Update

**Files:**
- Modify: `src/weights.rs` (PARAMS_PER_XF, XF_FIELDS)
- Modify: `src/genome.rs` (flatten_transforms)
- Modify: `flame_compute.wgsl` (transform stride)

- [ ] **Step 1: Update PARAMS_PER_XF from 42 to 48**

In `src/weights.rs`, change `PARAMS_PER_XF` constant. Update `XF_FIELDS` array to include the new fields: `az, bz, cz, dz, ez, fz` (the z-row and z-column entries, however named). Total 48 entries.

- [ ] **Step 2: Update flatten_transforms in genome.rs**

The transform flattening must now write 9 affine values + 3 offset values instead of 4+2. Ensure the layout matches what the shader expects. Write the 3x3 affine row-major, then offset xyz, then variation weights and params as before.

- [ ] **Step 3: Write test for flatten_transforms buffer layout**

```rust
#[test]
fn flatten_produces_48_floats_per_transform() {
    let mut g = FlameGenome::default_genome();
    g.transforms.truncate(3);
    let flat = g.flatten_transforms();
    assert_eq!(flat.len(), 3 * 48, "expected 48 floats per transform");
}
```

- [ ] **Step 4: Update flame_compute.wgsl transform access**

Change `idx * 42u` to `idx * 48u` everywhere. Update `xf()` helper and all field offset indices.

**Exact new layout per transform (48 floats):**
`weight, m00, m01, m02, m10, m11, m12, m20, m21, m22, offset_x, offset_y, offset_z, color, [26 variation weights], [8 variation params]`

Update `XF_FIELDS` array in `weights.rs` with these exact names. Update `playground.wgsl` Uniforms struct to also include `extra7`/`extra8` (both shaders must match).

- [ ] **Step 5: Run cargo test + cargo build --release**

All Rust tests pass. Shader compiles at runtime.

- [ ] **Step 5: Visual verification — run the app, confirm it looks normal**

The z-components are all identity/zero, so rendering should be identical to before.

- [ ] **Step 6: Commit**

```
git commit -m "feat: update transform buffer layout from 42 to 48 floats"
```

---

### Task T1.3: 3D Chaos Game Iteration

**Files:**
- Modify: `flame_compute.wgsl` (iteration loop, point state)
- Modify: `src/main.rs` (point state buffer size)

- [ ] **Step 1: Expand point state buffer**

In `src/main.rs`, find point state buffer creation. Change from `max_threads * 12` bytes (3 floats: x,y,color) to `max_threads * 28` bytes (7 floats: x,y,z,prev_x,prev_y,prev_z,color_idx).

- [ ] **Step 2: Update flame_compute.wgsl point state read/write**

Change point state indexing from `thread_id * 3u` to `thread_id * 7u`. Read/write x,y,z,prev_x,prev_y,prev_z,color_idx.

- [ ] **Step 3: Apply 3x3 affine in iteration loop**

Replace the 2x2 affine application:
```wgsl
// Old: p = vec2(a*p.x + b*p.y + ox, c*p.x + d*p.y + oy)
// New:
let px = m00*p.x + m01*p.y + m02*p.z + ox;
let py = m10*p.x + m11*p.y + m12*p.z + oy;
let pz = m20*p.x + m21*p.y + m22*p.z + oz;
p = vec3(px, py, pz);
```

- [ ] **Step 4: Compute 3D velocity for motion blur**

```wgsl
let vel = p.xyz - prev_p.xyz;
// prev_p is read from point state at start of iteration
// updated at end: store current p as prev_p for next frame
```

- [ ] **Step 5: Visual verification — run app, confirm identical rendering**

Z starts at 0, z-row is identity, so all z-values remain 0. Output should be identical.

- [ ] **Step 6: Commit**

```
git commit -m "feat: 3D chaos game iteration with xyz point state"
```

---

### Task T1.4: Camera & Perspective Projection

**Files:**
- Modify: `src/weights.rs` (camera config fields)
- Modify: `src/main.rs` (Uniforms struct, extra7/extra8)
- Modify: `flame_compute.wgsl` (camera rotation, perspective divide)
- Modify: `weights.json` (camera defaults)

- [ ] **Step 1: Add camera config to RuntimeConfig**

**NOTE:** `dof_strength` and `dof_focal_distance` already exist in RuntimeConfig. Only add the new camera fields:
```rust
#[serde(default = "default_camera_pitch")]
pub camera_pitch: f32,      // default 0.0
#[serde(default = "default_camera_yaw")]
pub camera_yaw: f32,        // default 0.0
#[serde(default = "default_camera_focal")]
pub camera_focal: f32,      // default 2.0
```
Reuse existing `dof_strength` and `dof_focal_distance` fields for DOF.

- [ ] **Step 2: Add extra7/extra8 to Uniforms struct (Rust + WGSL)**

In `src/main.rs` Uniforms struct and `flame_compute.wgsl` Uniforms struct, add:
```
extra7: vec4<f32>,  // camera_pitch, camera_yaw, camera_focal, dof_focal_plane
extra8: vec4<f32>,  // dof_strength, reserved, reserved, reserved
```

Wire up uniform writing in the render loop.

- [ ] **Step 3: Implement camera projection in compute shader**

After chaos game iteration produces `(x, y, z)`, before splatting:
```wgsl
let pitch = u.extra7.x;
let yaw = u.extra7.y;
let focal = u.extra7.z;

// Camera rotation (pitch around X, yaw around Y)
let cp = cos(pitch); let sp = sin(pitch);
let cy = cos(yaw); let sy = sin(yaw);
let rx = vec3(cy, 0.0, sy);
let ry = vec3(sp*sy, cp, -sp*cy);
let rz = vec3(-cp*sy, sp, cp*cy);
let cam_p = vec3(dot(rx, p), dot(ry, p), dot(rz, p));

// Perspective divide
let persp = 1.0 / (cam_p.z / focal + 1.0);
let screen = vec2(cam_p.x * persp, cam_p.y * persp);

// Project velocity through same transform
let cam_v = vec3(dot(rx, vel), dot(ry, vel), dot(rz, vel));
let screen_vel = vec2(cam_v.x * persp, cam_v.y * persp);
```

Use `screen` for pixel position, `screen_vel` for motion blur, `cam_p.z` for depth channel.

- [ ] **Step 4: Add camera defaults to weights.json**

```json
"camera_pitch": 0.0,
"camera_yaw": 0.0,
"camera_focal": 2.0,
"dof_focal_plane": 0.0,
"dof_strength": 0.0
```

- [ ] **Step 5: Visual verification — with defaults (0,0,2.0), rendering should be unchanged**

Then try `camera_pitch: 0.1` in weights.json — should see a subtle tilt.

- [ ] **Step 6: Commit**

```
git commit -m "feat: camera rotation and perspective projection"
```

---

### Task T1.5: Update DOF to Use Real Z-Depth

**Files:**
- Modify: `playground.wgsl` (existing DOF blur at lines 156-180)
- Modify: `flame_compute.wgsl` (update `point_depth` computation)

**NOTE:** DOF already exists in `playground.wgsl`. Currently it uses `length(sym_p)` (2D distance from origin) as a depth proxy. Update it to use the real z-depth from the 3D camera projection.

- [ ] **Step 1: Update compute shader `point_depth` to use camera z**

In `flame_compute.wgsl`, change the depth channel written to the histogram from `length(sym_p)` to `cam_p.z` (the z-coordinate after camera projection, computed in T1.4).

- [ ] **Step 2: Verify existing DOF shader reads the depth channel correctly**

The existing DOF code in `playground.wgsl` should already read from the depth channel. Confirm it uses the correct histogram channel index (channel 6, the z-depth).

- [ ] **Step 3: Visual verification — DOF should now blur based on actual 3D depth**

With `dof_strength > 0`, points far from the camera focal plane blur. Previously this was approximated by 2D distance; now it uses real z.

- [ ] **Step 4: Commit**

```
git commit -m "feat: update DOF to use real z-depth from 3D projection"
```

---

### Task T1.6: Z-Mutation Operators

**Files:**
- Modify: `src/genome.rs` (new mutation functions, mutation probability table)
- Modify: `src/weights.rs` (z_mutation_rate config)

- [ ] **Step 1: Add z_mutation_rate config**

In `src/weights.rs`:
```rust
#[serde(default = "default_z_mutation_rate")]
pub z_mutation_rate: f32,  // default 0.05
```

- [ ] **Step 2: Write test for z-tilt mutation**

```rust
#[test]
fn mutate_z_tilt_changes_z_row() {
    let mut xf = FlameTransform::default();
    let original_z_row = xf.affine[2];
    xf.mutate_z_tilt(&mut rand::rng());
    assert_ne!(xf.affine[2], original_z_row);
    // Should still have valid determinant
    let det = compute_3x3_det(&xf.affine);
    assert!(det > 0.1);
}
```

- [ ] **Step 3: Implement z-tilt and z-scale mutations**

```rust
pub fn mutate_z_tilt(&mut self, rng: &mut impl Rng) {
    let angle = rng.random_range(-0.3..0.3);
    let (s, c) = angle.sin_cos();
    // Rotate z-row: mix z with x or y
    if rng.random_bool(0.5) {
        let old_x = self.affine[2][0];
        let old_z = self.affine[2][2];
        self.affine[2][0] = old_x * c - old_z * s;
        self.affine[2][2] = old_x * s + old_z * c;
    } else {
        let old_y = self.affine[2][1];
        let old_z = self.affine[2][2];
        self.affine[2][1] = old_y * c - old_z * s;
        self.affine[2][2] = old_y * s + old_z * c;
    }
    self.clamp_determinant();
}

pub fn mutate_z_scale(&mut self, rng: &mut impl Rng) {
    let scale = rng.random_range(0.5..1.5);
    self.affine[2][2] *= scale;
    self.offset[2] += rng.random_range(-0.5..0.5);
    self.clamp_determinant();
}
```

- [ ] **Step 4: Add z-mutations to mutation probability table**

In `mutate_inner()`, add to the weighted distribution:
```rust
// z_mutation_rate (default 5%) split between tilt and scale
("z_tilt", cfg.z_mutation_rate / 2.0),
("z_scale", cfg.z_mutation_rate / 2.0),
```
Reduce `mutate_perturb` weight proportionally to keep total at 1.0.

- [ ] **Step 5: Run tests, verify all pass**

- [ ] **Step 6: Visual verification — let it evolve, z-mutations should gradually introduce depth**

- [ ] **Step 7: Commit + tag**

```
git commit -m "feat: z-tilt and z-scale mutation operators"
git tag v0.6.0-3d-rendering
```

---

## Track 2: Taste Engine Overhaul

### Task T2.1: Perceptual Features — CPU Proxy Render

**Files:**
- Modify: `src/taste.rs` (ProxyRender struct, FD/entropy/coverage functions)

- [ ] **Step 1: Write test for box-counting fractal dimension**

```rust
#[test]
fn fractal_dimension_empty_grid() {
    let grid = [[false; 64]; 64];
    let fd = box_counting_fd(&grid);
    assert!(fd < 0.1, "empty grid should have near-zero FD");
}

#[test]
fn fractal_dimension_full_grid() {
    let grid = [[true; 64]; 64];
    let fd = box_counting_fd(&grid);
    assert!((fd - 2.0).abs() < 0.2, "full grid FD should be near 2.0, got {fd}");
}

#[test]
fn fractal_dimension_diagonal_line() {
    let mut grid = [[false; 64]; 64];
    for i in 0..64 { grid[i][i] = true; }
    let fd = box_counting_fd(&grid);
    assert!((fd - 1.0).abs() < 0.2, "diagonal line FD should be near 1.0, got {fd}");
}
```

- [ ] **Step 2: Run tests, verify they fail**

- [ ] **Step 3: Implement box_counting_fd**

```rust
fn box_counting_fd(grid: &[[bool; 64]; 64]) -> f32 {
    let sizes = [2u32, 4, 8, 16, 32];
    let mut log_counts = Vec::new();
    let mut log_sizes = Vec::new();
    for &size in &sizes {
        let mut count = 0u32;
        let step = size as usize;
        for y in (0..64).step_by(step) {
            for x in (0..64).step_by(step) {
                'box_check: for dy in 0..step.min(64 - y) {
                    for dx in 0..step.min(64 - x) {
                        if grid[y + dy][x + dx] {
                            count += 1;
                            break 'box_check;
                        }
                    }
                }
            }
        }
        if count > 0 {
            log_counts.push((count as f32).ln());
            log_sizes.push((1.0 / size as f32).ln());
        }
    }
    if log_counts.len() < 2 { return 0.0; }
    // Linear regression slope
    linear_regression_slope(&log_sizes, &log_counts)
}
```

- [ ] **Step 4: Write test for spatial entropy**

```rust
#[test]
fn spatial_entropy_uniform_is_high() {
    let grid = [[true; 64]; 64];
    let e = spatial_entropy(&grid);
    assert!(e > 5.0, "uniform grid should have high entropy, got {e}");
}

#[test]
fn spatial_entropy_single_block_is_low() {
    let mut grid = [[false; 64]; 64];
    for y in 0..8 { for x in 0..8 { grid[y][x] = true; } }
    let e = spatial_entropy(&grid);
    assert!(e < 1.0, "single block should have low entropy, got {e}");
}
```

- [ ] **Step 5: Implement spatial_entropy**

```rust
fn spatial_entropy(grid: &[[bool; 64]; 64]) -> f32 {
    let mut block_hits = [0u32; 64]; // 8x8 blocks
    for y in 0..64 {
        for x in 0..64 {
            if grid[y][x] {
                let bx = x / 8;
                let by = y / 8;
                block_hits[by * 8 + bx] += 1;
            }
        }
    }
    let total: u32 = block_hits.iter().sum();
    if total == 0 { return 0.0; }
    let total_f = total as f32;
    block_hits.iter()
        .filter(|&&c| c > 0)
        .map(|&c| { let p = c as f32 / total_f; -p * p.ln() })
        .sum()
}
```

- [ ] **Step 6: Implement coverage_ratio**

```rust
fn coverage_ratio(grid: &[[bool; 64]; 64]) -> f32 {
    let hits: u32 = grid.iter().flat_map(|row| row.iter()).filter(|&&b| b).count() as u32;
    hits as f32 / (64.0 * 64.0)
}
```

- [ ] **Step 7: Implement CPU proxy render (affine-only)**

**NOTE:** Track 2 branch still has the old `a,b,c,d` fields. Use those here. The merge task will update to 3x3 accessors. Can also reuse `apply_xform_cpu` if it exists.

```rust
pub fn proxy_render(genome: &FlameGenome) -> [[bool; 64]; 64] {
    let mut grid = [[false; 64]; 64];
    let mut rng = rand::rng();
    let mut x = 0.0f32;
    let mut y = 0.0f32;
    let transforms = &genome.transforms;
    if transforms.is_empty() { return grid; }
    let weights: Vec<f32> = transforms.iter().map(|t| t.weight).collect();
    for i in 0..500 {
        let idx = weighted_random_pick(&weights, &mut rng);
        let xf = &transforms[idx];
        let nx = xf.a*x + xf.b*y + xf.offset[0];
        let ny = xf.c*x + xf.d*y + xf.offset[1];
        x = nx; y = ny;
        // Escape check — prevent NaN/infinity from degenerate transforms
        if x.is_nan() || y.is_nan() || x.abs() > 1e6 || y.abs() > 1e6 {
            x = rng.random_range(-2.0..2.0);
            y = rng.random_range(-2.0..2.0);
            continue;
        }
        if i < 50 { continue; } // warmup
        let gx = ((x + 2.0) / 4.0 * 64.0) as i32;
        let gy = ((y + 2.0) / 4.0 * 64.0) as i32;
        if gx >= 0 && gx < 64 && gy >= 0 && gy < 64 {
            grid[gy as usize][gx as usize] = true;
        }
    }
    grid
}
```

- [ ] **Step 7b: Write test for proxy_render with degenerate genome**

```rust
#[test]
fn proxy_render_degenerate_genome_no_panic() {
    let mut g = FlameGenome::default_genome();
    // Make a transform that diverges
    g.transforms[0].a = 10.0;
    g.transforms[0].d = 10.0;
    let grid = proxy_render(&g); // should not panic
    // Coverage will be low since points escape
    let coverage = coverage_ratio(&grid);
    assert!(coverage < 0.5);
}
```

- [ ] **Step 8: Write PerceptualFeatures struct and extraction**

```rust
pub struct PerceptualFeatures {
    pub fractal_dimension: f32,
    pub spatial_entropy: f32,
    pub coverage: f32,
}

impl PerceptualFeatures {
    pub fn from_genome(genome: &FlameGenome) -> Self {
        let grid = proxy_render(genome);
        Self {
            fractal_dimension: box_counting_fd(&grid),
            spatial_entropy: spatial_entropy(&grid),
            coverage: coverage_ratio(&grid),
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        vec![self.fractal_dimension, self.spatial_entropy, self.coverage]
    }
}
```

- [ ] **Step 9: Run all tests**

- [ ] **Step 10: Commit**

```
git commit -m "feat: perceptual features — CPU proxy render with FD, entropy, coverage"
```

---

### Task T2.2: IGMM Taste Model

**Files:**
- Modify: `src/taste.rs` (TasteCluster, IGMM logic, persistence)
- Modify: `src/weights.rs` (IGMM config fields)

- [ ] **Step 1: Add IGMM config to RuntimeConfig**

```rust
#[serde(default = "default_igmm_activation_threshold")]
pub igmm_activation_threshold: f32,  // 2.0
#[serde(default = "default_igmm_decay_rate")]
pub igmm_decay_rate: f32,            // 0.95
#[serde(default = "default_igmm_min_weight")]
pub igmm_min_weight: f32,            // 0.1
#[serde(default = "default_igmm_max_clusters")]
pub igmm_max_clusters: u32,          // 8
#[serde(default = "default_igmm_learning_rate")]
pub igmm_learning_rate: f32,         // 0.1
```

- [ ] **Step 2: Write test for TasteCluster creation**

```rust
#[test]
fn taste_cluster_from_features() {
    let features = vec![1.0, 2.0, 3.0];
    let cluster = TasteCluster::new(&features);
    assert_eq!(cluster.mean, features);
    assert_eq!(cluster.variance.len(), 3);
    assert!(cluster.weight > 0.0);
}
```

- [ ] **Step 3: Implement TasteCluster**

```rust
#[derive(Clone, Serialize, Deserialize)]
pub struct TasteCluster {
    pub mean: Vec<f32>,
    pub variance: Vec<f32>,
    pub weight: f32,
    pub sample_count: u32,
}

impl TasteCluster {
    pub fn new(features: &[f32]) -> Self {
        Self {
            mean: features.to_vec(),
            variance: vec![1.0; features.len()], // start with unit variance
            weight: 1.0,
            sample_count: 1,
        }
    }

    pub fn mahalanobis_distance(&self, features: &[f32]) -> f32 {
        self.mean.iter().zip(features).zip(&self.variance)
            .map(|((&m, &f), &v)| (f - m).powi(2) / v.max(0.01))
            .sum::<f32>()
            .sqrt()
    }

    pub fn update(&mut self, features: &[f32], learning_rate: f32) {
        self.sample_count += 1;
        self.weight += 1.0;
        for i in 0..self.mean.len() {
            let diff = features[i] - self.mean[i];
            self.mean[i] += learning_rate * diff;
            self.variance[i] += learning_rate * (diff * diff - self.variance[i]);
        }
    }
}
```

- [ ] **Step 4: Write test for IGMM update (merge into existing cluster)**

```rust
#[test]
fn igmm_update_merges_nearby_vote() {
    let mut igmm = IgmmModel::new();
    let f1 = vec![1.0, 2.0, 3.0];
    let f2 = vec![1.1, 2.1, 3.1]; // close to f1
    igmm.on_upvote(&f1, 2.0, 0.1, 8, 0.95, 0.1);
    igmm.on_upvote(&f2, 2.0, 0.1, 8, 0.95, 0.1);
    assert_eq!(igmm.clusters.len(), 1, "nearby votes should merge");
    assert_eq!(igmm.clusters[0].sample_count, 2);
}
```

- [ ] **Step 5: Write test for IGMM spawning new cluster**

```rust
#[test]
fn igmm_spawns_new_cluster_for_distant_vote() {
    let mut igmm = IgmmModel::new();
    let f1 = vec![1.0, 2.0, 3.0];
    let f2 = vec![100.0, 200.0, 300.0]; // very far
    igmm.on_upvote(&f1, 2.0, 0.1, 8, 0.95, 0.1);
    igmm.on_upvote(&f2, 2.0, 0.1, 8, 0.95, 0.1);
    assert_eq!(igmm.clusters.len(), 2, "distant votes should spawn new cluster");
}
```

- [ ] **Step 6: Implement IgmmModel**

```rust
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct IgmmModel {
    pub clusters: Vec<TasteCluster>,
}

impl IgmmModel {
    pub fn new() -> Self { Self { clusters: Vec::new() } }

    pub fn on_upvote(&mut self, features: &[f32],
        activation_threshold: f32, min_weight: f32,
        max_clusters: u32, decay_rate: f32, learning_rate: f32,
    ) {
        // Find closest cluster
        let closest = self.clusters.iter_mut()
            .enumerate()
            .map(|(i, c)| (i, c.mahalanobis_distance(features)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        match closest {
            Some((idx, dist)) if dist < activation_threshold => {
                self.clusters[idx].update(features, learning_rate);
            }
            _ if self.clusters.len() < max_clusters as usize => {
                self.clusters.push(TasteCluster::new(features));
            }
            _ => {
                // At max clusters — merge into closest anyway
                if let Some((idx, _)) = closest {
                    self.clusters[idx].update(features, learning_rate);
                }
            }
        }

        // Decay all weights, prune
        for c in &mut self.clusters { c.weight *= decay_rate; }
        self.clusters.retain(|c| c.weight >= min_weight);
    }

    pub fn score(&self, features: &[f32]) -> f32 {
        if self.clusters.is_empty() { return f32::MAX; }
        self.clusters.iter()
            .map(|c| c.mahalanobis_distance(features))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(f32::MAX)
    }
}
```

- [ ] **Step 7: Write test for IGMM scoring (min distance across clusters)**

```rust
#[test]
fn igmm_score_picks_closest_cluster() {
    let mut igmm = IgmmModel::new();
    igmm.clusters.push(TasteCluster::new(&[0.0, 0.0]));
    igmm.clusters.push(TasteCluster::new(&[10.0, 10.0]));
    let near_first = igmm.score(&[0.1, 0.1]);
    let near_second = igmm.score(&[9.9, 9.9]);
    assert!(near_first < near_second, "point near cluster 0 should score lower");
}
```

- [ ] **Step 8: Add persistence (save/load to genomes/taste_model.json)**

```rust
impl IgmmModel {
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    pub fn load(path: &Path) -> Option<Self> {
        std::fs::read_to_string(path).ok()
            .and_then(|s| serde_json::from_str(&s).ok())
    }
}
```

- [ ] **Step 9: Write test for IGMM persistence roundtrip**

```rust
#[test]
fn igmm_persistence_roundtrip() {
    let mut igmm = IgmmModel::new();
    igmm.on_upvote(&[1.0, 2.0, 3.0], 2.0, 0.1, 8, 0.95, 0.1);
    igmm.on_upvote(&[100.0, 200.0, 300.0], 2.0, 0.1, 8, 0.95, 0.1);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_model.json");
    igmm.save(&path).unwrap();
    let loaded = IgmmModel::load(&path).unwrap();
    assert_eq!(loaded.clusters.len(), igmm.clusters.len());
    assert!((loaded.clusters[0].mean[0] - igmm.clusters[0].mean[0]).abs() < 1e-6);
}
```

- [ ] **Step 10: Integrate IGMM into TasteEngine**

Replace the single `TasteModel` with `IgmmModel` in `TasteEngine`. Update `rebuild()`:
1. First, try loading saved model from `genomes/taste_model.json`
2. If no saved model, **cold-start bootstrap**: scan all `voted/` genomes, extract features from each, feed through `on_upvote()` one by one to build initial clusters
3. Save the bootstrapped model

Update `score` methods to use IGMM. Update the `on_upvote` flow: extract features → IGMM update → save.

Include perceptual features in the 25-dim feature vector:
```rust
let mut features = palette_features.to_vec();  // 17
features.extend(composition_features.to_vec()); // +5
features.extend(perceptual_features.to_vec());  // +3 = 25
```

The separate `transform_model` continues using the existing Gaussian centroid approach (per spec).

- [ ] **Step 11: Run all tests**

- [ ] **Step 12: Commit**

```
git commit -m "feat: IGMM taste model with multi-style clustering"
```

---

### Task T2.3: Interpolative Crossover

**Files:**
- Modify: `src/genome.rs` (breed function, affine lerp)
- Modify: `src/weights.rs` (interpolation config)

- [ ] **Step 1: Add interpolation config**

```rust
#[serde(default = "default_interpolation_range_lo")]
pub interpolation_range_lo: f32,  // 0.3
#[serde(default = "default_interpolation_range_hi")]
pub interpolation_range_hi: f32,  // 0.7
```

- [ ] **Step 2: Write test for affine interpolation**

**NOTE:** Track 2 still has `a,b,c,d` fields. Write `lerp_transform` using those. Merge task upgrades to 3x3.

```rust
#[test]
fn lerp_transform_midpoint() {
    let mut a = FlameTransform::default();
    a.a = 1.0; a.d = 1.0;
    let mut b = FlameTransform::default();
    b.a = 0.5; b.d = 0.5; b.b = 0.1;
    let result = lerp_transform(&a, &b, 0.5);
    assert!((result.a - 0.75).abs() < 1e-6);
    assert!((result.b - 0.05).abs() < 1e-6);
}
```

- [ ] **Step 3: Implement lerp_transform**

```rust
fn lerp_transform(a: &FlameTransform, b: &FlameTransform, t: f32) -> FlameTransform {
    let mut result = a.clone();
    result.a = a.a * (1.0-t) + b.a * t;
    result.b = a.b * (1.0-t) + b.b * t;
    result.c = a.c * (1.0-t) + b.c * t;
    result.d = a.d * (1.0-t) + b.d * t;
    result.offset[0] = a.offset[0]*(1.0-t) + b.offset[0]*t;
    result.offset[1] = a.offset[1]*(1.0-t) + b.offset[1]*t;
    result.weight = a.weight * (1.0-t) + b.weight * t;
    result.color = a.color * (1.0-t) + b.color * t;
    // Blend variation weights for matching variations
    // (handled in breed() logic, not here)
    result
}
```

- [ ] **Step 4: Modify breed() to use interpolative crossover**

In the slot-filling logic of `breed()`, when copying a transform from a parent:
- Check if the child slot and parent transform share a primary variation type
- If yes: interpolate affine, offset, variation weights, color, weight with random t in [lo, hi]
- If no: fall back to current slot-swap behavior (copy whole transform)

- [ ] **Step 5: Write test for breed producing interpolated offspring**

```rust
#[test]
fn breed_interpolates_matching_variations() {
    // Create two parents with same variation (linear)
    let pa = make_test_genome_with_variation("linear");
    let pb = make_test_genome_with_variation("linear");
    let cfg = RuntimeConfig::default();
    let child = FlameGenome::breed(&pa, &pb, &None, ...);
    // Child's affine should be between parent values, not identical to either
    let ca = child.transforms[0].affine[0][0];
    let pa_a = pa.transforms[0].affine[0][0];
    let pb_a = pb.transforms[0].affine[0][0];
    // Should be interpolated (between parents, not equal to either)
    // This is probabilistic — just check it's in a reasonable range
    assert!(ca >= pa_a.min(pb_a) - 0.5 && ca <= pa_a.max(pb_a) + 0.5);
}
```

- [ ] **Step 6: Run all tests**

- [ ] **Step 7: Commit**

```
git commit -m "feat: interpolative crossover for matching variation types"
```

---

### Task T2.4: MAP-Elites Archive

**Files:**
- Create: `src/archive.rs`
- Modify: `src/main.rs` (mod declaration, archive init, insertion, parent selection)
- Modify: `src/genome.rs` (behavioral characteristic extraction)

- [ ] **Step 1: Write test for grid coordinate mapping**

```rust
// In src/archive.rs
#[test]
fn grid_coords_map_correctly() {
    let coords = GridCoords::from_traits(3, 1.4, 0.6);
    assert_eq!(coords.symmetry_bin, 2);  // 0-indexed: sym 3 → bin 2
    assert_eq!(coords.fd_bin, 2);        // FD 1.4 in [1.0,2.0] with 5 bins → bin 2
    assert_eq!(coords.color_bin, 2);     // entropy 0.6 in [0,1] with 4 bins → bin 2
}
```

- [ ] **Step 2: Implement GridCoords and MapElitesArchive**

```rust
use serde::{Deserialize, Serialize};

const SYMMETRY_BINS: usize = 6;
const FD_BINS: usize = 5;
const COLOR_BINS: usize = 4;
const TOTAL_CELLS: usize = SYMMETRY_BINS * FD_BINS * COLOR_BINS; // 120

#[derive(Clone, Debug)]
pub struct GridCoords {
    pub symmetry_bin: usize,
    pub fd_bin: usize,
    pub color_bin: usize,
}

impl GridCoords {
    pub fn from_traits(symmetry: i32, fractal_dim: f32, color_entropy: f32) -> Self {
        let symmetry_bin = (symmetry.unsigned_abs() as usize).clamp(1, 6) - 1;
        let fd_bin = ((fractal_dim - 1.0) / 0.2).clamp(0.0, 4.99) as usize;
        let color_bin = (color_entropy * COLOR_BINS as f32).clamp(0.0, 3.99) as usize;
        Self { symmetry_bin, fd_bin, color_bin }
    }

    pub fn to_index(&self) -> usize {
        self.symmetry_bin * FD_BINS * COLOR_BINS
        + self.fd_bin * COLOR_BINS
        + self.color_bin
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ArchiveEntry {
    pub genome_name: String,
    pub score: f32,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct MapElitesArchive {
    cells: Vec<Option<ArchiveEntry>>,
}

impl MapElitesArchive {
    pub fn new() -> Self {
        Self { cells: vec![None; TOTAL_CELLS] }
    }

    pub fn insert(&mut self, coords: &GridCoords, name: String, score: f32) -> bool {
        let idx = coords.to_index();
        match &self.cells[idx] {
            None => { self.cells[idx] = Some(ArchiveEntry { genome_name: name, score }); true }
            Some(existing) if score < existing.score => {
                self.cells[idx] = Some(ArchiveEntry { genome_name: name, score });
                true
            }
            _ => false
        }
    }

    pub fn pick_random(&self, rng: &mut impl rand::Rng) -> Option<&str> {
        let occupied: Vec<_> = self.cells.iter().filter_map(|c| c.as_ref()).collect();
        if occupied.is_empty() { return None; }
        Some(&occupied[rng.random_range(0..occupied.len())].genome_name)
    }

    pub fn occupied_count(&self) -> usize {
        self.cells.iter().filter(|c| c.is_some()).count()
    }

    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    pub fn load(path: &std::path::Path) -> Option<Self> {
        std::fs::read_to_string(path).ok()
            .and_then(|s| serde_json::from_str(&s).ok())
    }
}
```

- [ ] **Step 3: Write tests for archive insert/replace/pick**

```rust
#[test]
fn archive_insert_empty_cell() {
    let mut archive = MapElitesArchive::new();
    let coords = GridCoords { symmetry_bin: 0, fd_bin: 0, color_bin: 0 };
    assert!(archive.insert(&coords, "genome-1".into(), 5.0));
    assert_eq!(archive.occupied_count(), 1);
}

#[test]
fn archive_replace_with_better_score() {
    let mut archive = MapElitesArchive::new();
    let coords = GridCoords { symmetry_bin: 0, fd_bin: 0, color_bin: 0 };
    archive.insert(&coords, "genome-1".into(), 5.0);
    assert!(archive.insert(&coords, "genome-2".into(), 3.0)); // lower = better
    assert_eq!(archive.occupied_count(), 1);
}

#[test]
fn archive_reject_worse_score() {
    let mut archive = MapElitesArchive::new();
    let coords = GridCoords { symmetry_bin: 0, fd_bin: 0, color_bin: 0 };
    archive.insert(&coords, "genome-1".into(), 3.0);
    assert!(!archive.insert(&coords, "genome-2".into(), 5.0)); // worse
}
```

- [ ] **Step 4: Integrate archive into main.rs**

- Add `archive: MapElitesArchive` to App state
- Load from `genomes/archive.json` on startup (or create new)
- After every breed/mutate: compute behavioral traits, insert into archive
- In `pick_breeding_parents()`: 50% chance pick from archive, 50% existing vote-based selection
- Save archive after modifications

- [ ] **Step 5: Add `mod archive;` declaration**

- [ ] **Step 6: Run all tests**

- [ ] **Step 7: Commit**

```
git commit -m "feat: MAP-Elites archive for structured diversity"
```

---

### Task T2.5: Novelty Search Scoring

**Files:**
- Modify: `src/taste.rs` (novelty scoring function)
- Modify: `src/archive.rs` (feature vectors for k-NN)
- Modify: `src/weights.rs` (novelty config)
- Modify: `src/genome.rs` (integrate novelty into breed scoring)

- [ ] **Step 1: Add novelty config**

```rust
#[serde(default = "default_novelty_weight")]
pub novelty_weight: f32,        // 0.3
#[serde(default = "default_novelty_k_neighbors")]
pub novelty_k_neighbors: u32,   // 5
```

- [ ] **Step 2: Write test for novelty score**

```rust
#[test]
fn novelty_score_higher_for_distant_point() {
    let archive_features = vec![
        vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.2],
        vec![0.3, 0.3], vec![0.4, 0.4], vec![0.5, 0.5],
    ];
    let near = novelty_score(&[0.05, 0.05], &archive_features, 5);
    let far = novelty_score(&[10.0, 10.0], &archive_features, 5);
    assert!(far > near, "distant point should have higher novelty");
}
```

- [ ] **Step 3: Implement novelty_score**

```rust
pub fn novelty_score(features: &[f32], archive_features: &[Vec<f32>], k: usize) -> f32 {
    if archive_features.is_empty() { return 0.0; }
    let k = k.min(archive_features.len());
    let mut distances: Vec<f32> = archive_features.iter()
        .map(|af| {
            features.iter().zip(af).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt()
        })
        .collect();
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    distances[..k].iter().sum::<f32>() / k as f32
}
```

- [ ] **Step 4: Extend archive to store feature vectors**

Add `features: Vec<f32>` to `ArchiveEntry`. Update `insert()` to accept features. Add `all_features()` method that returns `Vec<Vec<f32>>` for k-NN.

- [ ] **Step 5: Integrate novelty into breed candidate scoring**

In `generate_biased_transform` or wherever candidates are ranked:
```rust
let taste = igmm.score(&features);
let novelty = novelty_score(&features, &archive_features, cfg.novelty_k_neighbors);
let fitness = taste - cfg.novelty_weight * novelty;
```

- [ ] **Step 6: Run all tests**

- [ ] **Step 7: Visual verification — evolution should produce more diverse outputs**

- [ ] **Step 8: Commit + tag**

```
git commit -m "feat: novelty search scoring for exploration"
git tag v0.6.0-taste-engine-v2
```

---

## Merge Task

### Task M1: Merge Tracks + Cross-Integration

**Prerequisite:** Both Track 1 and Track 2 branches are complete and tested independently.

- [ ] **Step 1: Merge Track 1 into main**

```bash
git checkout main
git merge feat/3d-rendering
```

- [ ] **Step 2: Merge Track 2 into main**

```bash
git merge feat/taste-engine-v2
```

Resolve conflicts in shared files (`genome.rs`, `weights.rs`, `main.rs`, `weights.json`). Conflicts should be mostly additive (different sections of same files).

- [ ] **Step 3: Update taste.rs feature extraction for 3x3 affines**

Replace all `xf.a`, `xf.b`, `xf.c`, `xf.d` references in `TransformFeatures` and `CompositionFeatures` with the new accessor helpers or direct `xf.affine[i][j]` access.

- [ ] **Step 4: Update proxy_render to use 3x3 affine accessors**

`proxy_render()` in taste.rs uses `xf.a`, `xf.b`, etc. Update to use `xf.a()` accessors (or `xf.affine[i][j]`).

- [ ] **Step 5: Update lerp_transform to use 3x3 affine**

Replace the `a,b,c,d` field interpolation with full `affine` matrix interpolation + 3D offset lerp.

- [ ] **Step 6: Run all tests from both tracks**

```bash
cargo test
```

- [ ] **Step 7: Visual verification — full system running with both tracks**

- [ ] **Step 8: Commit merge + tag**

```
git commit -m "feat: merge 3D rendering + taste engine v2"
git tag v0.6.0-evolution-overhaul
```

---

## Dependency Graph

```
Track 1 (worktree 1):    T1.1 → T1.2 → T1.3 → T1.4 → T1.5 → T1.6
                                                                    ↘
Track 2 (worktree 2):    T2.1 → T2.2 → T2.3 → T2.4 → T2.5 ------→ M1
```

Both tracks run in parallel. M1 waits for both to complete.
