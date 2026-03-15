# Rendering Pipeline v2 — Surgical Approach

**Lesson learned:** Never ship multiple shader/pipeline changes without visual verification between each. One change, one visual check, one commit.

**Baseline:** `feat/rendering-v2-surgical` branch, forked from main after the bundle commit. All original rendering quality intact.

## Stack-Ranked Improvements

### Tier 1: High Impact, Low Risk

**1. AgX tonemapping (done right)**
- Replace sqrt-log alpha computation when tonemap_mode=2, NOT stack on top
- AgX takes raw linear color and handles full HDR→display compression itself
- No new bindings, no pipeline changes, shader-only
- Test: toggle tonemap_mode 0/1/2 in weights.json, visually compare
- CHECKPOINT: Does it look as good or better than sqrt-log? Colors richer at high brightness?

**2. Jacobian importance sampling (gentle, strength 0.1)**
- CPU-only: blend |a*d - b*c| with genetic weight at 10% strength
- No shader changes, no new bindings
- Adjusts transform selection probability so expansive transforms get slightly more budget
- Test: toggle jacobian_weight_strength 0.0 vs 0.1 in weights.json
- CHECKPOINT: Tendrils slightly sharper? Dense areas still filled in?

**3. Sobol QMC for transform selection**
- New binding (slot 6) for 32 direction numbers
- Replace randf() with sobol_sample() for transform selection only
- Everything else (jitter, color noise) stays PCG
- Pass flag via extra6.z uniform
- Test: toggle use_quasi_random in weights.json
- CHECKPOINT: Structures converge faster? No visible patterns/banding?

### Tier 2: Medium Impact, Medium Risk

**4. Dual Kawase bloom (proper architecture)**
- Create a dedicated pre-bloom render target (separate from trail feedback)
- Bloom reads pre-trail tonemapped image → no feedback loop
- Compute shaders already written (bloom_downsample.wgsl, bloom_upsample.wgsl)
- CHECKPOINT: Smooth glow without whiteout? Better than inline 3-radius?

**5. Compute-based post-processing**
- Move velocity blur + DOF into dedicated compute passes
- Better cache utilization via workgroup shared memory
- Fragment shader becomes a simple compositor
- CHECKPOINT: Same visual quality, better frame times?

### Tier 3: High Impact, High Risk

**6. ReSTIR reservoir accumulation**
- Replace exponential moving average with statistical reservoir sampling
- Design doc at .plans/restir-design.md
- Infinite refinement when static, instant response on mutation
- CHECKPOINT: No ghosting? Converges to sharp image over time?

**7. Subgroup atomics (blocked by Naga)**
- Code is written and ready on feat/rendering-pipeline-v2
- Activates automatically when wgpu/Naga ships enable subgroups support
- Track: https://github.com/gfx-rs/wgpu/issues/5555
- CHECKPOINT: Same visual output, better frame times (check [perf] log)

## Process

For each improvement:
1. Implement on feat/rendering-v2-surgical
2. `cargo run --release` — Chad visually verifies
3. If good: commit and move to next
4. If bad: revert immediately, discuss what went wrong
5. If subtle: leave running for a few mutations to see how it behaves over time
