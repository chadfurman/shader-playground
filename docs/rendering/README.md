# Rendering Pipeline

## When to Read

| Working on... | Read |
|---|---|
| Chaos game, transforms, variations | [Chaos Game](chaos-game.md) |
| Brightness, log-density, color | [Tonemapping](tonemapping.md) |
| Trail, temporal persistence | [Feedback & Trail](feedback-trail.md) |
| Bloom, DoF, velocity blur | [Post Effects](post-effects.md) |
| Per-point brightness factors | [Luminosity](luminosity.md) |

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Per Frame                                │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────┐    ┌───────────────┐ │
│  │ Compute  │───▶│Histogram │───▶│ CDF  │───▶│   Fragment    │ │
│  │ Pass     │    │ Reduce   │    │ Pass │    │   Pass        │ │
│  └──────────┘    └──────────┘    └──────┘    └───────────────┘ │
│       │                                            │           │
│  chaos game                                   tonemapping      │
│  iterations                                   + effects        │
│  splatting                                    + feedback       │
│       │                                            │           │
│       ▼                                            ▼           │
│  ┌──────────┐                                ┌──────────┐     │
│  │ Accum    │                                │  Screen  │     │
│  │ Buffer   │                                │  Output  │     │
│  │(7-chan)  │                                │          │     │
│  └──────────┘                                └──────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## Pass Descriptions

### 1. Compute Pass (`flame_compute.wgsl`)

- Launches `workgroups × 256` threads
- Each thread: persistent point state (7 floats: x, y, z, prev_x, prev_y, prev_z, color_idx) → iterate 3x3 affine transforms → splat to accumulation buffer
- 3D iteration: full 3x3 affine applied per transform, z-component propagates through chaos game
- Camera projection: pitch/yaw rotation + perspective divide before splatting
- Subgroup atomics: `subgroupBroadcastFirst` + `subgroupAdd` reduce atomic contention for threads targeting the same pixel
- Warmup iterations skipped before splatting
- Wavefront regeneration: escaped/NaN points recycled immediately
- Writes 7 channels: density, RGB color, velocity XY (projected through camera), depth (camera-space z)

### 2. Histogram Reduction

- Scans accumulation buffer to find max density
- Used for per-image normalization in tonemapping

### 3. CDF Pass

- Builds cumulative distribution function from density histogram
- Enables adaptive histogram equalization in the fragment shader

### 4. Fragment Pass (`playground.wgsl`)

- Full-screen quad, one fragment per pixel
- Reads accumulation buffer → log-density tonemapping → vibrancy blend
- Applies velocity blur, depth of field, bloom, edge glow
- Temporal feedback: blends with previous frame via trail

## What Persists Across Frames

- **Point state**: each thread's (x, y, z, prev_x, prev_y, prev_z, color_idx) survives frame-to-frame for attractor continuity
- **Trail feedback**: previous frame's output blended into current via `max(current, prev * trail)`
- **Accumulation**: cleared each frame (no multi-frame accumulation)
