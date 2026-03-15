# Shader Playground

A real-time fractal flame renderer with interactive evolution, audio reactivity, and 3D depth.

Built with Rust, wgpu 28, and WGSL compute shaders. Runs on macOS (Metal), Linux (Vulkan), and Windows (DX12).

## Features

- **Chaos game rendering** -- GPU compute shader splatting ~50M points/frame into a 7-channel histogram
- **3D depth** -- Full Sdobnov hack: 3x3 affine matrices, camera pitch/yaw, perspective projection, depth-of-field blur
- **Subgroup atomics** -- wgpu `Features::SUBGROUP` reduces atomic contention up to 32x in dense attractor regions
- **Interactive evolution** -- upvote/downvote genomes, breed offspring from voted parents
- **IGMM taste engine** -- multi-style Gaussian mixture model learns your aesthetic preferences without averaging them into mush
- **MAP-Elites archive** -- 120-cell diversity grid (symmetry x fractal dimension x color entropy) ensures broad aesthetic exploration
- **Novelty search** -- rewards genomes that explore unexplored aesthetic territory
- **Perceptual features** -- CPU proxy render evaluates fractal dimension, spatial entropy, and coverage for each genome
- **Audio reactivity** -- system audio capture drives color, glow, and atmospheric effects (not geometry -- no rubber-banding)
- **Morph capture** -- save/vote mid-transition to preserve aesthetics that only exist during morphs
- **Hot-reload** -- edit `weights.json` or shader files live, changes apply instantly

## Quick Start

```bash
cargo run --release
```

## Controls

| Key | Action |
|-----|--------|
| Space | Evolve (breed + mutate) |
| Up Arrow | Upvote current genome |
| Down Arrow | Downvote current genome |
| Backspace | Revert to previous genome |
| S | Save genome (captures morph state if mid-transition) |
| L | Load random saved genome |
| F | Load random imported flame |
| A | Toggle audio reactivity |
| 1-4 | Solo/unsolo transform N |

All controls are morph-aware -- voting or saving mid-morph captures the interpolated state.

## Configuration

All tunable values live in `weights.json` and are hot-reloaded. Key sections:

- `_config` -- rendering, genetics, taste engine, camera, DOF parameters
- Signal weights -- audio/time signal modulation of shader params
- `_comments` -- documentation for each parameter

## Architecture

```
src/
  main.rs       -- app state, render loop, uniform buffer, keybinds
  genome.rs     -- FlameGenome, 3x3 affine transforms, breed/mutate
  taste.rs      -- IGMM model, perceptual features, proxy render, novelty scoring
  archive.rs    -- MAP-Elites diversity archive
  votes.rs      -- VoteLedger, LineageCache, genetic distance
  weights.rs    -- RuntimeConfig, signal weight matrix
  audio.rs      -- system audio capture + feature extraction
  bloom.rs      -- bloom texture placeholder
  flam3.rs      -- Apophysis .flame XML import
  device_picker.rs -- audio device selection

flame_compute.wgsl  -- chaos game compute shader (3D iteration, subgroup atomics)
playground.wgsl     -- display/tonemapping fragment shader (DOF, bloom, feedback)
accumulation.wgsl   -- accumulation buffer management
```

## Docs

Serve locally with `npx docsify-cli serve docs`.
