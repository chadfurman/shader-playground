# Audio Reactivity v2 — Design

## Problem

The flame system has audio reactivity but it's barely perceptible. Audio features (bass/mids/highs/energy/beat) drive subtle color shifts, brightness, and mutation timing, but the actual flame *structure* doesn't respond to music. Transitions between genomes are also jarring — mutations cause hard trajectory changes mid-morph.

## Design Principles

- **Audio shapes evolution, not oscillation.** No mechanical pumping, no sinusoidal scale effects. Audio biases what the flame *becomes*, not what it does frame-to-frame.
- **Genome is home, audio is displacement.** Real-time modulation pushes transforms gently; when audio stops, everything drifts back.
- **Every transition is a dissolve.** Crossfade blending means no hard cuts ever, even with rapid mutations.
- **Organic over mechanical.** Per-transform randomness ensures no two transforms respond the same way. Nothing feels uniform or predictable.

## 1. Crossfade Transition System

### New GPU Resource
One `crossfade_texture` (same format/size as render target). Created at startup alongside feedback texture.

### On Mutation
1. GPU copy: current feedback texture -> crossfade texture (`encoder.copy_texture_to_texture()`)
2. Clear histogram buffer
3. Set `crossfade_alpha = 0.0`
4. Each frame: `crossfade_alpha += dt / CROSSFADE_DURATION` (6-8 seconds)

### Fragment Shader
```wgsl
let old_flame = textureSample(crossfade_tex, crossfade_sampler, tex_uv).rgb;
let new_flame = /* existing tonemapping pipeline */;
let crossfade = clamp(u.crossfade_alpha, 0.0, 1.0);
var col = mix(old_flame, new_flame, crossfade);

// Trail only applies proportionally to crossfade progress
let prev = textureSample(prev_frame, prev_sampler, tex_uv).rgb;
col = mix(col, col + prev * trail, crossfade);
```

### Chained Mutations
If a mutation fires mid-crossfade (alpha < 1.0):
1. Snapshot *current blended output* into crossfade texture
2. Clear histogram
3. Restart crossfade at alpha = 0.0
4. Visually: continuous flowing evolution, never a hard cut

## 2. Audio-Biased Mutations

When `mutate()` fires, current `AudioFeatures` bias what kind of flame is produced.

### Audio -> Variation Personality

| Audio State | Favored Variations | Character |
|---|---|---|
| Bass-heavy | spherical, bubble, blob, fisheye | Round, massive, gravitational |
| Mids-heavy | sinusoidal, waves, cosine, handkerchief | Flowing, organic, curtains |
| Highs-heavy | julia, disc, cross, tangent, diamond | Sharp, crystalline, detailed |
| High energy | more transforms, multi-variation combos, symmetry | Complex |
| Low energy | fewer transforms, simpler specialists | Meditative |
| Beat-dense | spiral, swirl, polar | Rotational, rhythmic |

### Implementation
- `mutate()` takes `AudioFeatures` parameter
- When picking variations for specialists/blends, 40-50% chance to pick from audio-favored group
- Transform count add/remove biased by energy level
- Symmetry mutations biased by beat density
- Always probabilistic — never deterministic

## 3. Real-Time Audio Transform Modulation

Subtle, organic pushing of current genome's variation weights based on audio.

### New weights.json Mappings
- `bass -> xfN_spherical` (+0.2-0.3), `bass -> xfN_bubble` (+0.2)
- `highs -> xfN_julia` (+0.15-0.2), `highs -> xfN_disc` (+0.15)
- `mids -> xfN_sinusoidal` (+0.15), `mids -> xfN_waves` (+0.15)

### Divisor Fix
Current weights are divided by SIGNAL_COUNT (15), making audio effects nearly invisible. Either:
- Increase raw weights to compensate
- Or split divisor: use smaller divisor (~5) for audio signals, keep 15 for time signals

### Per-Transform Randomness
Existing `transform_randomness()` (-1 to +1) and `transform_magnitude()` (1 to 5) already ensure each transform responds differently. Some get +bass boost to spherical, others get -bass. Organic, not uniform.

### Explicitly Not Added
- No bass -> zoom/scale
- No energy -> scale
- No sinusoidal oscillation effects

## 4. Mutation Timing

### Keep Current Rates
`energy x 0.3 + beat x 0.2 + beat_accum x 1.5` stays. Fast mutations during intense music are desirable now that crossfade handles smoothness.

### Minimum Cooldown: 3 Seconds
Prevents degenerate rapid-fire (multiple mutations in <1 second). Everything else handled by crossfade.

### Silence Holds Naturally
Low energy/no beats means accumulator barely moves. Flame holds its form during quiet sections without any explicit logic.

## 5. Files Changed

| File | Changes |
|---|---|
| `src/main.rs` | Crossfade state, texture copy on mutation, alpha ramp, 3s cooldown, pass AudioFeatures to mutate() |
| `src/genome.rs` | `mutate()` takes AudioFeatures, biases variation picks + transform count + symmetry |
| `playground.wgsl` | Crossfade blend, trail proportional to crossfade progress |
| `weights.json` | Variation weight modulations, strengthen textural weights |
| `src/weights.rs` | Separate audio vs time signal divisors |

| File | No Changes |
|---|---|
| `flame_compute.wgsl` | Compute shader unaware of crossfade |
| `src/audio.rs` | Audio pipeline unchanged |

## 6. What It Feels Like

- Bass drop -> flame drifts toward round, massive, gravitational forms
- Bright treble section -> crystalline, sharp structures emerge
- Calm passage -> flame holds, breathes gently
- Intense section -> flame is liquid, constantly flowing between forms via dissolves
- Every transition smooth, every evolution music-driven
- Variation weights subtly shift with the music in real-time
- No pumping, no oscillation, no mechanical feel
