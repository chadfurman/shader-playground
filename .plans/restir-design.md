# ReSTIR Reservoir Accumulation — Design Document

## Overview

This document describes replacing the current exponential moving average (EMA)
accumulation pass with Spatiotemporal Reservoir Resampling (ReSTIR). The goal is
to eliminate ghosting when transform parameters mutate, enable infinite refinement
when parameters are static, and propagate high-quality samples to sparse neighbors
via spatial reuse.

The current EMA in `accumulation.wgsl` applies `accum = accum * decay + new_frame`
per pixel per channel. It is fast but has two fatal properties for this renderer:
stale history never fully disappears (ghosting), and convergence speed is capped by
the decay constant regardless of how much GPU time is available. ReSTIR fixes both.

---

## 1. Reservoir Buffer Layout

### What a "Sample" Is in IFS Context

In the chaos game (`flame_compute.wgsl`), a **sample** is a single point that lands
on a pixel after a chain of affine + variation transforms. The data attached to that
point at the moment it is splatted is:

- `pos`: the 2D attractor position in world space (before zoom/screen mapping)
- `color_idx`: the blended palette index accumulated along the transform chain
- `transform_sequence`: which transform was last applied (encoded as `tidx`)

For reservoir purposes, a sample needs to be re-evaluable under new parameters. The
minimum representation that supports re-evaluation of `p_hat` is:

```
struct IFSSample {
    pos:   vec2<f32>,   // world-space attractor position
    color: f32,         // palette index [0, 1]
    tidx:  u32,         // last transform index applied
}
```

`pos` lets us re-run the color/luminosity function under mutated transforms.
`tidx` lets us re-evaluate the transform weight (used in the luminosity formula).
`color` carries the blended palette index so color contribution is reproducible.

### Per-Pixel Reservoir Struct

```wgsl
struct Reservoir {
    y_pos:   vec2<f32>,  // sample attractor position (world space)
    y_color: f32,        // sample palette index
    y_tidx:  u32,        // sample transform index
    p_hat:   f32,        // target function weight: luminosity(y) under current params
    w_sum:   f32,        // sum of candidate weights seen so far
    M:       f32,        // total sample count (f32 to avoid integer overflow over time)
    _pad:    f32,        // alignment to 32 bytes
}
```

**Field sizes:**
- `y_pos` (vec2<f32>): 8 bytes
- `y_color` (f32): 4 bytes
- `y_tidx` (u32): 4 bytes
- `p_hat` (f32): 4 bytes
- `w_sum` (f32): 4 bytes
- `M` (f32): 4 bytes
- `_pad` (f32): 4 bytes

**Total: 32 bytes per pixel**

### Buffer Sizes

At 1920×1080:
- Pixel count: 2,073,600
- Reservoir buffer: 2,073,600 × 32 bytes = **63.0 MB**
- Two buffers needed (ping-pong for temporal reuse): **126.1 MB total**

At 1280×720 (common development resolution):
- Reservoir buffer: 921,600 × 32 bytes = **28.1 MB** × 2 = **56.2 MB total**

Both are well within the GPU memory budget on Apple Silicon M-series (unified 18–36 GB).

The existing `accumulation_buffer` is `width × height × 7 × 4 bytes` = ~56 MB at 1080p.
The reservoir buffer is slightly larger but replaces `accumulation_buffer` entirely.

### Alignment Note

WGSL requires struct fields to be aligned to their size. The layout above is
naturally aligned: all f32/u32 fields are 4-byte aligned, `y_pos` as a
`vec2<f32>` is 8-byte aligned at offset 0, total struct size 32 bytes is a
power of two. No extra padding needed beyond the explicit `_pad`.

---

## 2. Temporal Reuse Logic

### The Core Idea

At the start of each frame, the reservoir from the previous frame holds `y` — a
sample that was good under the *old* parameters. Before deciding whether the
current frame's new sample should replace it, we re-evaluate `p_hat` for `y`
under the *current* parameters. This single step is what eliminates ghosting.

### Target Function `p_hat`

`p_hat` for a sample `y` is the luminosity contribution that point would produce
under the current transform set. Concretely:

```
fn p_hat(pos: vec2<f32>, tidx: u32) -> f32 {
    let xf_weight = transforms[tidx * 42u + 0u];
    let dist = length(pos);
    let dist_lum = 1.0 / (1.0 + dist * dist * config.dist_lum_strength);
    let lum = dist_lum * clamp(xf_weight * 3.0, config.lum_weight_min, config.lum_weight_max);
    return max(lum, 0.0);
}
```

This mirrors the luminosity computation in `flame_compute.wgsl` (`iter_lum`,
`dist_lum`, `xf_weight`), but stripped of per-iteration factors since we only
have the final point. All scale/clamp constants must come from `RuntimeConfig`
(no magic numbers).

### Weighted Reservoir Sampling (WRS) Update Rule

Given the existing reservoir `R` from frame `N-1` and a new candidate `c`
from frame `N`:

```
fn update_reservoir(R: ptr<Reservoir>, c: IFSSample, w_c: f32, rng: ptr<RNG>) {
    R.w_sum += w_c;
    R.M     += 1.0;
    // Accept with probability w_c / w_sum
    if (randf(rng) < w_c / R.w_sum) {
        R.y_pos   = c.pos;
        R.y_color = c.color;
        R.y_tidx  = c.tidx;
        R.p_hat   = p_hat(c.pos, c.tidx);
    }
}
```

The unbiased contribution weight `W` for the pixel is:

```
W = (1.0 / p_hat(R.y)) * (R.w_sum / R.M)
```

The fragment shader multiplies the sample's color by `W` to get the final pixel
contribution. When `p_hat` drops to zero (stale sample under new params), `W`
goes to infinity — but this is handled by the invalidation step below.

### Temporal Re-evaluation (Anti-Ghosting)

Before executing WRS for the current frame's candidate, re-evaluate the stored
sample's `p_hat` under current parameters:

```
fn temporal_reuse_pass(px: u32, prev_reservoir: Reservoir,
                       current_candidate: IFSSample, rng: ptr<RNG>) -> Reservoir {
    var R = prev_reservoir;

    // Re-evaluate stored sample under current parameters
    let new_p_hat = p_hat(R.y_pos, R.y_tidx);

    if (new_p_hat < config.reservoir_zero_threshold) {
        // Historical sample is invalid under new params — reset reservoir
        R.w_sum = 0.0;
        R.M     = 0.0;
        R.p_hat = 0.0;
    } else {
        // Scale w_sum to reflect updated p_hat (keeps statistics consistent)
        let scale = new_p_hat / max(R.p_hat, 1e-6);
        R.w_sum *= scale;
        R.p_hat  = new_p_hat;
    }

    // Stream in current frame's candidate
    let w_c = p_hat(current_candidate.pos, current_candidate.tidx);
    update_reservoir(&R, current_candidate, w_c, rng);
    return R;
}
```

`reservoir_zero_threshold` lives in `RuntimeConfig._config` (e.g., `0.001`).

### Why This Eliminates Ghosting

When a transform mutates significantly, `p_hat` for the stored position drops
toward zero because the new transforms no longer produce high density at that
location. The reset branch fires immediately. The reservoir is rebuilt from
scratch using only current-frame candidates. Ghosting is structurally impossible:
old samples cannot survive if their target weight is zero under new rules.

When parameters are static, `new_p_hat ≈ R.p_hat`, `R.w_sum` and `R.M` keep
growing, `W` converges, and the image refines indefinitely.

### M Clamping

To prevent `M` from growing without bound (which would make new samples
essentially weightless after many static frames), cap `M` at a configurable
maximum:

```
R.M = min(R.M, config.reservoir_m_cap);
```

`reservoir_m_cap` in `RuntimeConfig._config` (suggested starting value: `512.0`).
This trades some statistical purity for responsiveness to slow parameter drift.

---

## 3. Spatial Reuse Pattern

### Motivation

The chaos game distributes samples stochastically. Some pixels receive many hits
per frame (dense attractor regions); others receive zero hits for many consecutive
frames (thin filaments, outer structures). Without spatial reuse, thin regions
converge very slowly. With spatial reuse, a neighbor pixel that hit a good sample
can propagate it to zero-hit neighbors instantly.

### Neighborhood Size

Use a **3×3 neighborhood** (8 neighbors) for Phase B. This reads 8 extra
reservoir values per pixel — 256 bytes of additional bandwidth per pixel, or
~0.5 GB/s at 60fps at 1080p. That is acceptable. A 5×5 neighborhood (24
neighbors) can be considered later if convergence is still slow, but 3×3 is the
right starting point.

### Spatial RIS Pass

After temporal reuse, execute a second pass that treats neighbor reservoirs as
additional candidates:

```
fn spatial_reuse_pass(px_coord: vec2<u32>, temporal_reservoir: Reservoir,
                      rng: ptr<RNG>) -> Reservoir {
    var R = temporal_reservoir;

    for each neighbor offset in 3x3 grid (skip center):
        let n_coord = px_coord + offset;
        if out of bounds: continue;
        let N = reservoir_buffer[n_coord];  // from previous spatial pass or temporal

        // Re-evaluate neighbor's stored sample at THIS pixel's p_hat
        let p_hat_here = p_hat(N.y_pos, N.y_tidx);
        // Merge N into R using MIS weight
        let m_n = N.M;
        let w_n = p_hat_here * (N.w_sum / max(N.M, 1.0));
        update_reservoir(&R, N.y_as_sample(), w_n * m_n, rng);

    // After merging, recalculate M for bias correction
    // R.M was already incremented by each update_reservoir call
    return R;
}
```

The key: we evaluate the neighbor's sample through *our* p_hat, not the
neighbor's. This is standard biased ReSTIR — unbias correction is Phase C.

### How Many Spatial Candidates Per Pixel

With a 3×3 grid, we merge up to 8 spatial candidates per pixel per frame.
Combined with the 1 temporal candidate, each pixel considers up to 9 samples
per frame. Even at 60fps with 1 new chaos-game sample per pixel per frame, this
gives the convergence rate of ~540 samples/second per pixel — comparable to an
offline renderer.

### Convergence in Sparse Regions

Consider a thin filament pixel that receives 0 hits from the chaos game for 20
consecutive frames. Without spatial reuse, its reservoir stays at `M=0` and the
pixel is black. With spatial reuse, its 3×3 neighborhood likely contains pixels
that *do* receive chaos-game hits. The spatial pass merges those hits into the
filament pixel's reservoir. Convergence time for sparse regions drops from
O(1/hit_rate) to O(1/neighborhood_hit_rate), which is roughly 9× faster for
uniform-density neighborhoods.

---

## 4. Mutation Detection

### The Problem

Audio-driven modulation changes transform parameters every frame. ReSTIR needs to
distinguish between:

1. **Continuous small modulation** — zoom, drift, color shift: the attractor
   moves slowly. Historical samples remain valid; re-evaluate `p_hat` and accept
   graceful weight scaling.

2. **Discrete genome mutation** — new transforms, variation weights changed: the
   attractor jumps. Historical samples are instantly invalid; reservoir must reset.

### Transform Parameter Hash

Maintain a 64-bit hash of the current transform buffer on the Rust side.
Recompute every frame before uploading. Store as two `u32` fields in a new
`ReSTIRUniforms` struct passed to the accumulation shader:

```rust
struct ReSTIRUniforms {
    resolution:       [f32; 2],
    transform_hash_lo: u32,
    transform_hash_hi: u32,
    mutation_threshold: f32,
    m_cap:             f32,
    zero_threshold:    f32,
    _pad:              f32,
}
```

Hash function: FNV-1a over the raw `[f32]` bytes of the transform buffer.
Fast, deterministic, no dependencies.

Store `prev_transform_hash` in `ReSTIRUniforms` (or as a separate uniform).
If `current_hash != prev_hash`, it is a discrete mutation frame.

### Frame-to-Frame Delta Threshold

For continuous modulation that does NOT trigger a hash change (same genome,
parameters audio-modulated within one frame), use the per-pixel `p_hat`
re-evaluation to detect local invalidity. No extra detection needed — the
temporal reuse logic handles it naturally via `reservoir_zero_threshold`.

For large but continuous parameter changes (e.g., rapid zoom), `p_hat` will
decrease smoothly rather than jump to zero. The reservoir degrades gracefully
rather than resetting.

### Discrete Mutation — Full Reset Strategy

When `current_hash != prev_hash`:

Option A: **CPU-side buffer clear.** On mutation, Rust calls
`encoder.clear_buffer(&reservoir_buffer_a, 0, None)` before submitting the
frame. Zeros out all `M`, `w_sum`, `p_hat` fields. The shader then treats every
pixel as a fresh reservoir. Simple, correct, one GPU command.

Option B: **Shader-side conditional reset.** Pass `is_mutation_frame: u32` in
`ReSTIRUniforms`. The shader reads this flag and zeroes the reservoir inline.
Avoids a separate GPU command but adds a branch per pixel.

**Recommendation: Option A for Phase C.** It is explicit, zero-cost when there
is no mutation, and easy to reason about. Option B is useful if partial
invalidation is needed (only invalidate pixels whose `p_hat` delta exceeds a
threshold — see below).

### Audio-Driven Modulation — Continuous Small Changes

Audio modulation (bass → zoom, mids → color_shift, etc.) modifies uniform
values, not the transform buffer directly. The transform hash does not change.
`p_hat` re-evaluation handles this correctly: small parameter changes produce
small `p_hat` changes, and the reservoir updates smoothly.

For audio-driven affine matrix modulation (if the genome ever drives `a/b/c/d`
offsets via audio signals), the hash *will* change every frame. In that case,
`M` clamping via `reservoir_m_cap` provides graceful degradation: the reservoir
forgets old samples at a rate proportional to `1/m_cap` per frame, acting like
a controlled EMA with mathematically correct weighting.

### Partial Invalidation

For Phase C, implement per-pixel partial invalidation:

```
let p_hat_ratio = new_p_hat / max(old_p_hat, 1e-6);
if p_hat_ratio < config.partial_invalidation_threshold:
    R.M = R.M * p_hat_ratio  // shrink M proportional to change
    R.w_sum = R.w_sum * p_hat_ratio
```

`partial_invalidation_threshold` in `RuntimeConfig._config` (suggested: `0.1`).
This allows pixels that changed a lot to effectively reset while pixels that
changed a little retain most of their history.

---

## 5. Integration with Current Pipeline

### Which Stage Is Replaced

The accumulation pass (`accumulation.wgsl`, `accumulation_pipeline`) is
**replaced** by the ReSTIR temporal pass. The spatial reuse pass is a new
additional compute stage inserted after temporal reuse.

The four-stage pipeline becomes five stages:

```
1. Clear histogram
2. Chaos game compute (flame_compute.wgsl) — unchanged
3. ReSTIR temporal pass (restir_temporal.wgsl) — replaces accumulation.wgsl
4. ReSTIR spatial pass (restir_spatial.wgsl) — NEW
5. Histogram equalization (unchanged)
6. Bloom (unchanged)
7. Fragment display pass (playground.wgsl) — reads reservoir instead of accumulation
```

### Buffer Changes in Rust (`src/main.rs`)

**Remove:**
- `accumulation_buffer: wgpu::Buffer`
- `accumulation_uniform_buffer: wgpu::Buffer`
- `accumulation_pipeline: wgpu::ComputePipeline`
- `accumulation_bind_group_layout: wgpu::BindGroupLayout`
- `accumulation_bind_group: wgpu::BindGroup`

**Add:**
- `reservoir_buffer_a: wgpu::Buffer` — ping buffer (current frame writes here)
- `reservoir_buffer_b: wgpu::Buffer` — pong buffer (previous frame's reservoir)
- `restir_uniform_buffer: wgpu::Buffer` — `ReSTIRUniforms` struct
- `restir_temporal_pipeline: wgpu::ComputePipeline`
- `restir_temporal_bind_group_layout: wgpu::BindGroupLayout`
- `restir_temporal_bind_group: wgpu::BindGroup`
- `restir_spatial_pipeline: wgpu::ComputePipeline`
- `restir_spatial_bind_group_layout: wgpu::BindGroupLayout`
- `restir_spatial_bind_group: wgpu::BindGroup`
- `reservoir_ping: bool` — tracks which buffer is current

**Buffer creation:**

```rust
fn create_reservoir_buffer(device: &wgpu::Device, w: u32, h: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reservoir"),
        size: (w as u64) * (h as u64) * 32, // 32 bytes per reservoir
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
```

Call twice (for A and B).

**Temporal pass bind group layout:**

```
binding 0: histogram (read) — array<u32>
binding 1: reservoir_prev (read) — array<Reservoir>
binding 2: reservoir_curr (read_write) — array<Reservoir>
binding 3: uniforms (uniform) — ReSTIRUniforms
binding 4: transforms (read) — array<f32>  [needed for p_hat re-evaluation]
binding 5: max_density (read_write) — array<atomic<u32>>
```

**Spatial pass bind group layout:**

```
binding 0: reservoir_temporal (read) — array<Reservoir>  [output of temporal pass]
binding 1: reservoir_spatial (read_write) — array<Reservoir>  [final reservoir]
binding 2: uniforms (uniform) — ReSTIRUniforms
binding 3: transforms (read) — array<f32>
```

The spatial pass reads the temporal reservoir and writes to the final reservoir.
The final reservoir is what the fragment shader reads.

### Bind Group Changes for Fragment Shader

In `playground.wgsl`, replace the `accumulation` storage buffer binding with
the final reservoir buffer:

```wgsl
// Old
@group(0) @binding(3) var<storage, read> accumulation: array<f32>;

// New
@group(0) @binding(3) var<storage, read> reservoirs: array<Reservoir>;
```

The fragment shader reads `reservoirs[px].y_color` for color and computes
luminosity as:

```wgsl
let R = reservoirs[px];
let W = (1.0 / max(R.p_hat, 1e-6)) * (R.w_sum / max(R.M, 1.0));
let display_density = R.p_hat * W;  // = w_sum / M, the unbiased estimator
let display_color = palette(R.y_color + color_shift);
```

The log-density tonemapping, histogram equalization, and AgX transform all
operate on `display_density` exactly as before — they are downstream of the
accumulation abstraction and require no changes.

### Max Density Tracking

The `max_density_buffer` (single `atomic<u32>`) currently gets updated inside
the accumulation pass. In the ReSTIR design, update it in the **temporal pass**:
after writing the new reservoir, compute the unbiased estimator `W * p_hat` for
the pixel and update `atomicMax(&max_density[0], bitcast<u32>(max(estimator, 0.0)))`.
This preserves the per-image normalization used by the tonemapper.

### Resize Handling

In `Gpu::resize()`, recreate both reservoir buffers alongside the histogram
buffer:

```rust
self.reservoir_buffer_a = create_reservoir_buffer(&self.device, w, h);
self.reservoir_buffer_b = create_reservoir_buffer(&self.device, w, h);
```

Clearing them on resize is correct (full reset on resolution change).

### Ping-Pong Swap in Render Loop

Each frame:

```
1. temporal_pass reads: histogram, reservoir_{prev}, transforms
   temporal_pass writes: reservoir_{curr}
2. spatial_pass reads: reservoir_{curr}
   spatial_pass writes: reservoir_final  (can be same as reservoir_{curr} or a 3rd buffer)
3. Swap prev/curr: reservoir_ping = !reservoir_ping
4. Fragment reads: reservoir_final
```

For Phase A (temporal only), there is no spatial pass and `reservoir_final` is
the same as `reservoir_curr`.

For Phase B (spatial added), a third buffer is needed for the spatial output, OR
the spatial pass can be run in-place if reads and writes are to the same buffer
(requires care — only safe if each pixel reads only its own value after the
temporal write). For a 3×3 pass reading neighbors, in-place is NOT safe.
Use a dedicated `reservoir_spatial_out` buffer (a third reservoir buffer, same
size, ~63 MB at 1080p).

---

## 6. Implementation Phases

### Phase A: Temporal-Only Reservoir

**Goal:** Replace EMA with reservoir sampling. No spatial reuse. Eliminates
ghosting, enables infinite refinement.

**Scope:**
- Write `restir_temporal.wgsl` implementing WRS update + `p_hat` re-evaluation
- Create two reservoir buffers (ping-pong)
- Create `ReSTIRUniforms` struct in Rust and WGSL
- Add `reservoir_m_cap`, `reservoir_zero_threshold` to `RuntimeConfig._config`
  in `weights.json`
- Remove `accumulation.wgsl` and its pipeline/bind group from `main.rs`
- Update `playground.wgsl` fragment shader to read reservoir
- Update `rebuild_bind_groups()` and `resize()` in `main.rs`
- Add transform hash computation in Rust (before each upload)
- Add mutation detection: clear reservoir buffers on hash change

**Expected outcome:** Static parameters → image keeps refining forever.
Parameter mutation → instant response, zero ghosting.

**Estimated code change:** ~300 lines new WGSL, ~150 lines Rust changes
(remove old accumulation code, add reservoir management).

### Phase B: Add Spatial Reuse

**Goal:** Accelerate convergence in sparse regions; propagate high-quality
samples across neighborhoods.

**Scope:**
- Write `restir_spatial.wgsl` implementing 3×3 neighborhood RIS
- Add a third reservoir buffer (`reservoir_spatial_out`)
- Add spatial pass pipeline and bind group to `Gpu` struct
- Insert spatial dispatch in the render loop between temporal pass and CDF pass
- Add `restir_spatial_radius` (value `1` for 3×3) to `RuntimeConfig._config`

**Expected outcome:** Thin filament regions that previously took seconds to
resolve converge in 1–3 frames. Overall visual quality significantly higher
at the same sample count.

**Estimated code change:** ~150 lines new WGSL, ~80 lines Rust additions.

### Phase C: Adaptive Invalidation Based on Parameter Change Magnitude

**Goal:** Handle audio-driven continuous modulation gracefully — neither full
ghosting (EMA) nor excessive temporal thrashing (full reset on every audio beat).

**Scope:**
- Implement partial invalidation in temporal pass: scale `M` and `w_sum` by
  `p_hat_ratio` when ratio is between `partial_invalidation_threshold` and `1.0`
- Add `partial_invalidation_threshold` to `RuntimeConfig._config`
- Implement per-pixel `p_hat_delta` accumulation: track how much `p_hat` has
  changed over recent frames and use it to smoothly blend between "keep history"
  and "reset" — this is the MIS weight approach from ReSTIR DI
- Optionally: differentiate between uniform mutations (affect all pixels equally,
  e.g., zoom change) and local mutations (affect only some pixels, e.g., a
  variation weight change) using per-pixel delta vs. global delta

**Expected outcome:** Audio-reactive rendering that responds instantly to beats
(discrete mutations) while retaining temporal coherence during slow modulation
(zoom glides, color drifts). The reservoir acts like an adaptive history buffer
whose length is proportional to local parameter stability.

**Estimated code change:** ~100 lines additional WGSL, ~40 lines Rust for
delta tracking.

---

## Config Fields Required in `weights.json` `_config`

All of the following must be added to `RuntimeConfig` in `src/weights.rs` with
corresponding `default_*` functions. No magic numbers in shader code.

| Field | Type | Suggested Default | Purpose |
|---|---|---|---|
| `reservoir_m_cap` | f32 | 512.0 | Maximum M before clamping (prevents stale lock-in) |
| `reservoir_zero_threshold` | f32 | 0.001 | p_hat below this → reset reservoir |
| `partial_invalidation_threshold` | f32 | 0.1 | p_hat ratio below this → partial reset |
| `lum_weight_min` | f32 | 0.3 | Clamp min for xf_weight luminosity contribution |
| `lum_weight_max` | f32 | 1.0 | Clamp max for xf_weight luminosity contribution |
| `restir_spatial_radius` | u32 | 1 | Neighborhood radius (1 = 3×3, 2 = 5×5) |

---

## Open Questions

1. **Bias correction for spatial reuse.** The spatial RIS pass as described is
   biased (it uses a neighbor's `M` count without accounting for the different
   target distribution). Unbiased correction requires a second visibility check.
   For Phase B, biased ReSTIR is acceptable — bias manifests as slightly
   over-brightened pixels near high-density neighbors, which is a very minor
   artifact compared to EMA ghosting. Unbias correction can be added in Phase C.

2. **Reservoir for symmetry copies.** The chaos game plots multiple symmetry
   copies of each point. The reservoir buffer is 1:1 with screen pixels. If a
   pixel receives hits from both the direct and mirrored path, the histogram
   currently adds them. With ReSTIR, each pixel maintains one reservoir — multiple
   hits from different symmetry arms should each be treated as separate candidates
   fed into the same reservoir WRS update loop.

3. **Sample rate per pixel.** Currently, one chaos-game thread may contribute
   to many pixels per frame (via bilinear splat to 2×2 neighbors). The reservoir
   model assumes each pixel gets independent candidates. The bilinear splat creates
   correlations between neighboring reservoir updates. This is acceptable for
   Phase A/B and can be addressed later by having each pixel pull from a
   per-pixel candidate buffer rather than the shared histogram.
