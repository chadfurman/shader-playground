# Flame Genome & Evolution System Design

## Problem

Flame transforms are hardcoded in WGSL. To evolve, mutate, and crossfade between flames, transform coefficients need to be data — not code.

## Solution

A "flame genome" is a structured JSON file that flattens to a float array in the uniform buffer. Mutation generates new genomes by perturbing coefficients. The existing smooth param interpolation handles crossfades automatically.

## Genome Format

```json
{
  "name": "ember-spiral",
  "transforms": [
    {
      "weight": 0.25,
      "angle": 0.6, "scale": 0.65,
      "offset": [0.4, 0.15],
      "variations": {
        "linear": 0.2, "sinusoidal": 0.4, "spherical": 0.0,
        "swirl": 0.4, "horseshoe": 0.0, "handkerchief": 0.0,
        "polar": 0.0, "heart": 0.0
      },
      "color": 0.0
    }
  ],
  "kifs": {
    "fold_angle": 0.62,
    "scale": 1.8,
    "brightness": 0.3
  },
  "global": {
    "speed": 0.25,
    "zoom": 2.0,
    "trail": 0.34,
    "flame_brightness": 0.4
  }
}
```

4 transforms, each with: weight, angle, scale, offset (2), color, 6 variation weights = 12 floats per transform.

## Param Layout (flat array)

```
[0..3]   global: speed, zoom, trail, flame_brightness
[4..6]   kifs: fold_angle, scale, brightness
[7]      reserved

Per transform (4 transforms x 12 floats = 48):
[8..19]  transform 0: weight, angle, scale, ox, oy, color,
                       linear, sinusoidal, spherical, swirl, horseshoe, handkerchief
[20..31] transform 1
[32..43] transform 2
[44..55] transform 3

Total: 56 floats -> 14 vec4s (uniform buffer: array<vec4<f32>, 16>)
```

## Mutation

On `Space` press, generate new genome by applying 1-3 random mutations:

- **Perturb**: Gaussian noise on random coefficient (angle +/- 0.3, scale +/- 0.1, etc.)
- **Swap variations**: Redistribute variation weights within a transform
- **Rotate colors**: Shift all transform color values by random offset
- **Shuffle transform**: Swap two transforms' parameter blocks
- **KIFS drift**: Perturb fold angle and scale

Crossfade: mutated genome becomes `target_params`, existing exponential lerp morphs over ~2-3 seconds.

## Keyboard Controls

| Key | Action |
|-----|--------|
| Space | Evolve: mutate current genome, crossfade |
| Backspace | Revert: previous genome from history |
| S | Save: write genome to `genomes/<timestamp>.json` |
| L | Load: random genome from `genomes/` |
| 1-4 | Solo: toggle transform weight to 0 |
| +/- | Morph speed: adjust crossfade duration (1-10s) |

## History

Ring buffer of last 10 genomes in memory. Backspace walks backward.

## File Structure

```
genomes/              — saved genome JSON files (auto-created)
src/genome.rs         — FlameGenome struct, serde, flatten, mutation
src/main.rs           — expanded uniforms, keyboard, genome loading
flame_compute.wgsl    — param-driven transforms (no hardcoded values)
playground.wgsl       — expanded uniform struct, same rendering
```

## What Changes

- Uniform buffer: 16 floats -> 64 floats (4 vec4s -> 16 vec4s)
- Compute shader: hardcoded transforms -> param-driven
- main.rs: genome management, keyboard input, history
- New: src/genome.rs

## What Stays

- Compute pipeline (histogram, atomics, chaos game)
- Fragment shader (KIFS + flame display)
- Hot-reload of WGSL files
- Ping-pong feedback
- Smooth param interpolation

## Not In Scope

- Audio reactivity (future)
- Multiple background modes (KIFS only)
- GUI / sliders
- Networking
