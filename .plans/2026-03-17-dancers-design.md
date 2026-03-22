# Dancers + Session Persistence — Design Spec

## Overview

**Dancers** are ephemeral, localized transforms that spawn during genome transitions, wander the screen with audio-reactive movement, and dissolve after a configurable lifetime. They breed from a mix of the current genome's transforms and a persistent dancer archive.

**Session persistence** saves the current genome, morph state, and dancer archive on exit so the app resumes where it left off.

---

## Dancers

### What They Are

A dancer is a regular IFS transform with:
- **Small affine scale** (0.05-0.2) — creates a tight, localized cluster of points
- **Large offset** from origin (1.0-3.0) — appears at a specific screen position
- **Bell curve weight** over its lifetime — fades in, full presence, fades out
- **Drifting position** — wanders the screen, modulated by audio
- **Per-dancer audio weight vector** — each dancer reacts to music uniquely

The shader doesn't know dancers exist. They're appended to the transform buffer alongside genome transforms — just transforms with specific parameter ranges.

### Lifecycle

1. When **any** morph begins (spacebar, auto-evolve, load, or flame import), schedule 1-5 dancers (random, configurable) at random `morph_progress` values between 0.0 and `1.0 / min_stagger_rate` (the morph completion point for the slowest transform)
2. Each frame, if `morph_progress >= scheduled_progress` for a pending dancer, spawn it
3. Each dancer lives for a configurable duration (4-8s wall-clock time)
4. Weight follows a bell curve with configurable fade fraction: ramp up over `fade_fraction * lifetime`, hold at peak, ramp down over `fade_fraction * lifetime`
5. When weight reaches 0, the dancer is removed and its transform is archived
6. In-progress dancers are **not** killed when a new morph begins — they finish their lifetime naturally, overlapping with the next morph's dancers

### Breeding

Dancers breed from two sources: the **current genome's transforms** and the **dancer archive** (a ring buffer of past dancer transforms).

- **First generation** (empty archive): pick 2-3 random transforms from the current genome, blend via interpolative crossover, clamp scale small, push offset out
- **Later generations**: pick 1-2 transforms from the current genome + 1 from the dancer archive, blend, clamp, offset
- Each dancer picks a **different random subset** of the genome's transforms, so dancers express different facets of the genome
- On spawn, apply small random mutations to variation weights, color index, and offset direction
- Assign a random per-dancer audio weight vector: small random weights for bass/mids/highs/energy/beat that modulate position drift and scale pulse

### Dancer Archive

- Ring buffer of the last ~20 dancer transforms (configurable via `dancer_archive_size`)
- When a dancer dies, its final transform state goes into the archive
- Persisted to `genomes/dancer_archive.json`
- Loaded on startup — dancers resume their lineage across sessions
- Archive entries include: transform data + dancer generation counter (no parent_genome — YAGNI)

### Buffer Strategy

**Pre-allocate the transform buffer** for `transform_count_max + dancer_count_max` at startup and on `ResizeTransformBuffer`. This avoids mid-frame resizes when dancers spawn/die. With max 12 genome transforms + 5 dancers = 17 transforms × 48 floats × 4 bytes = 3.2KB — trivially bounded.

The `ResizeTransformBuffer` command already handles reallocation. We just ensure it accounts for the dancer headroom by adding `dancer_count_max` to the requested size.

### Transform Count and Uniform Integration

Three sites in the per-frame path reference transform count. All must include active dancers:

1. **`transform_count` in Uniforms** — the shader iterates this many transforms. Set to `genome.transform_count() + active_dancer_count`.
2. **`num_transforms` on App** — drives Jacobian loop bounds and buffer write length. Track genome-only count in `num_transforms` and add `active_dancer_count` when building `FrameData.xf_params`.
3. **`apply_variation_scales`** — CRISPR scaling runs on `self.xf_params` before dancers are appended. Dancers skip CRISPR intentionally — they inherit their parent's variation character, not the global scale overrides.

### Adaptive Workgroup Budget

Dancers are excluded from the `effective_xforms` budget calculation. Rationale: dancers are lightweight by construction (max 5, tiny scale, few points per dancer). Including them would over-penalize complex genomes that happen to have dancers active. If FPS drops are observed with many dancers, `dancer_count_max` is the correct tuning knob.

### Runtime (per frame)

1. Check if any scheduled dancers should spawn (compare `morph_progress >= scheduled_value`)
2. For each active dancer:
   - Advance lifetime (`elapsed = current_time - birth_time`)
   - Compute bell curve weight from `elapsed / lifetime` and `dancer_fade_fraction`
   - Apply per-dancer audio modulation to offset (drift) and scale (pulse)
   - If lifetime expired: archive transform, remove dancer
3. Flatten active dancers' transforms and append to the genome's transform buffer
4. Set `transform_count` in uniforms to `genome.transform_count() + active_dancer_count`

### Data Structures

```rust
struct Dancer {
    transform: Transform,              // the IFS transform
    birth_time: f32,                   // wall-clock app time when spawned
    lifetime: f32,                     // total seconds to live
    audio_weights: DancerAudioWeights, // per-dancer audio reactivity
    generation: u32,                   // dancer lineage generation
}

struct DancerAudioWeights {
    drift_bass: f32,
    drift_mids: f32,
    drift_highs: f32,
    pulse_energy: f32,
    pulse_beat: f32,
}

struct DancerManager {
    active: Vec<Dancer>,               // currently alive dancers
    scheduled: Vec<f32>,               // spawn triggers as morph_progress values
    archive: VecDeque<Transform>,      // ring buffer of past dancer transforms
}
```

### Storage

- `DancerManager` lives on `App` (not in genome, not in Gpu)
- Archive saved to `genomes/dancer_archive.json` periodically (~30s) + on exit
- Active dancers are NOT saved (they're ephemeral by nature)
- `scheduled` is rebuilt on each morph start

### Config (weights.json `_config`)

```json
{
  "dancer_enabled": true,
  "dancer_count_min": 1,
  "dancer_count_max": 5,
  "dancer_lifetime_min": 4.0,
  "dancer_lifetime_max": 8.0,
  "dancer_scale_min": 0.05,
  "dancer_scale_max": 0.2,
  "dancer_offset_min": 1.0,
  "dancer_offset_max": 3.0,
  "dancer_drift_speed": 0.3,
  "dancer_audio_strength": 0.5,
  "dancer_archive_size": 20,
  "dancer_fade_fraction": 0.2
}
```

---

## Session Persistence

### What It Does

On exit (or periodically), save enough state to resume on next launch.

### Saved State (`genomes/session.json`)

```json
{
  "genome_name": "mutant-1234"
}
```

### Behavior

- **On exit**: save session.json + dancer_archive.json
- **On startup**: if `session.json` exists, load the named genome instead of picking random. Load dancer archive. Start fresh morph from the saved genome (no mid-morph resume — stagger rates are randomized at morph start, so resuming mid-morph would cause a visual jump. Snapping to the saved genome is simpler and visually clean.)
- **Fallback**: if the saved genome file is missing or corrupted, fall back to normal random load
- **Periodic save**: save dancer_archive.json every ~30s; session.json on exit only

### Known Limitation

Session persistence does not restore mid-morph state, dancer positions, or auto-evolve timing. It restores *which genome* you were looking at so you don't start from scratch. Full state serialization is complex and the visual impact of snapping vs. mid-morph resume is minimal.

---

## What Doesn't Change

- `FlameGenome` struct — dancers are not part of the genome
- Shader code — dancers are regular transforms
- Render thread — just gets a longer transform buffer some frames
- Mutation / crossover / voting / taste engine — unaware of dancers
- `transform_count_min/max` in breeding — independent of dancer count
- Adaptive workgroup budget — excludes dancers (see rationale above)

---

## Testing Strategy

- **Bell curve weight**: unit test weight computation at 0%, 20%, 50%, 80%, 100% of lifetime with various fade fractions
- **Spawn scheduling**: unit test that scheduled values fall within valid morph_progress range
- **Archive ring buffer**: unit test insert, capacity limit, oldest-evicted behavior
- **Dancer breeding**: unit test transform subset selection from genome + archive, interpolative crossover
- **Buffer sizing**: unit test that pre-allocated buffer fits max genome + max dancers
- **Session persistence**: roundtrip test for session.json and dancer_archive.json
- **Integration**: visual verification — dancers should appear as small localized phenomena that fade in/out during morphs
