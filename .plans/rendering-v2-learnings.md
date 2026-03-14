# Rendering Pipeline v2 — Learnings

What worked, what didn't, and why. Reference for future optimization work.

## Attempted Improvements

### Jacobian Importance Sampling — SHIPPED
**What:** Blend |a*d - b*c| (affine determinant) into transform selection weights on CPU side.
**Result:** Higher contrast, crisper tendrils. At strength 0.5 it starves contractive transforms (dots everywhere). At 0.1 it's a clear win.
**Shipped config:** `jacobian_weight_strength: 0.1`

### Shoulder Highlight Compression — SHIPPED
**What:** Replace hard `clamp(col, 0, 1)` with a knee curve: linear below 0.8, soft Reinhard rolloff above.
**Result:** Eliminates flat white blowouts while keeping midtones punchy. Full Reinhard `col/(1+col)` was too muted — the knee approach preserves punch below 0.8.
**Shipped in:** playground.wgsl final display transform

### AgX Tonemapping — REJECTED (doesn't fit pipeline)
**What:** Troy Sobotka's view transform from Blender 4.0. Polynomial approximation of a filmic curve that gracefully desaturates highlights to white.
**Why it failed:** Our pipeline produces display-ready [0,1] values through sqrt-log + vibrancy blend. AgX expects scene-referred linear HDR (where 1.0 = mid-gray, 10+ = bright). Applying AgX to already-tonemapped values just brightens everything to white. First attempt stacked it on top of sqrt-log (double tonemapping → whiteout). Second attempt applied it at the final clamp (values already in [0,1] → just brightens).
**What would make it work:** Full pipeline restructure: skip sqrt-log, pass raw linear density-weighted color to AgX, let it handle the entire HDR→display compression. Major refactor, not a drop-in.

### Sobol Quasi-Monte Carlo — REJECTED (breaks chaos game)
**What:** Replace PCG random number generator with Sobol low-discrepancy sequence for transform selection. Theory: QMC converges at O(1/N) vs O(1/√N) for pure Monte Carlo.
**Why it failed:** The chaos game is a Markov chain that fundamentally requires true randomness to properly explore the attractor. Sobol forces too-uniform sampling, which breaks the stochastic exploration — threads end up correlated, creating visible structured patterns and sparse coverage. The IFS attractor is not a standard integration domain where QMC theory applies.
**Gemini was wrong here:** The research suggested QMC would help, but IFS rendering is a special case where the random process IS the algorithm, not just a sampling strategy.

### Subgroup Atomic Reduction — BLOCKED (Naga limitation)
**What:** Use WGSL `enable subgroups;` + `subgroupBallot/subgroupAdd/subgroupElect` to aggregate atomic writes within the warp before hitting the histogram buffer. Would reduce atomic contention by up to 32x.
**Why blocked:** Naga (wgpu's shader compiler) doesn't support `enable subgroups;` yet despite the adapter reporting the feature. Tracked at https://github.com/gfx-rs/wgpu/issues/5555
**Code is ready:** The subgroup splatting implementation exists on `feat/rendering-pipeline-v2` branch. When Naga ships support, flip `use_subgroups = false` back to the real check.

### Dual Kawase Bloom — REJECTED (feedback loop)
**What:** Replace 3-radius inline bloom with cascaded downsample/upsample compute passes. Industry standard O(log N) bloom.
**Why it failed:** The bloom pipeline read from the previous rendered frame (which already has bloom baked in from the trail feedback). Each frame adds more bloom on top of existing bloom → exponential brightness → whiteout within seconds.
**What would fix it:** Dedicated pre-bloom render target. Render the tonemapped image to an intermediate texture BEFORE trail feedback, run Kawase on that, then composite the bloom-only contribution. Requires an additional render target and careful pipeline ordering.
**Shaders are ready:** `bloom_downsample.wgsl` and `bloom_upsample.wgsl` exist and are correct.

## Key Principles Discovered

1. **The chaos game is not a standard Monte Carlo integrator.** QMC doesn't help because the random process IS the algorithm. Don't try to make it deterministic.

2. **AgX and similar scene-referred tonemappers need a full pipeline designed around them.** You can't drop them into a pipeline that already does its own density→display mapping. The vibrancy/gamma/log-density chain in flam3 is a complete tonemapping pipeline — AgX would replace ALL of it, not supplement it.

3. **Bloom feedback loops are insidious.** Any time a post-processing effect reads from a buffer that includes its own output from the previous frame, you get exponential accumulation. Always verify the data flow: source → process → destination, where source ≠ destination.

4. **Jacobian weighting works but is sensitive.** 0.1 is gentle and helpful. 0.5 is destructive. The sweet spot is low because contractive transforms are important for filling in dense structure — you want to bias slightly toward expansive transforms, not aggressively.

5. **Shoulder curves > Reinhard > hard clamp.** Full Reinhard `x/(1+x)` compresses the entire range and feels muted. A knee curve that only activates above a threshold preserves midtone punch while taming highlights.
