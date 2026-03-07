# Audio Reactivity v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Crossfade transitions between genomes, audio-biased mutations, and real-time variation modulation for a music-driven flame visualizer.

**Architecture:** Three independent features: (1) GPU crossfade system — snapshot feedback texture on mutation, blend old/new in fragment shader; (2) Audio-biased mutations — pass AudioFeatures to mutate(), bias variation picks by frequency content; (3) Weights tuning — add variation modulations to weights.json, fix signal divisor.

**Tech Stack:** wgpu (Metal backend), WGSL shaders, Rust

---

### Task 1: Add crossfade_alpha to Uniforms

**Files:**
- Modify: `src/main.rs:27-37` (Uniforms struct)
- Modify: `playground.wgsl:7-17` (Uniforms struct)

**Step 1: Update Rust Uniforms struct**

In `src/main.rs`, add `crossfade_alpha` to the Uniforms struct. Add a new `extra2` field (vec4 for alignment):

```rust
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    time: f32,
    frame: u32,
    resolution: [f32; 2],
    mouse: [f32; 2],
    transform_count: u32,
    has_final_xform: u32,
    globals: [f32; 4],
    kifs: [f32; 4],
    extra: [f32; 4],
    extra2: [f32; 4],  // crossfade_alpha, reserved, reserved, reserved
}
```

**Step 2: Update WGSL Uniforms struct**

In `playground.wgsl`, add matching field:

```wgsl
struct Uniforms {
    time: f32,
    frame: u32,
    resolution: vec2<f32>,
    mouse: vec2<f32>,
    transform_count: u32,
    _pad: u32,
    globals: vec4<f32>,
    kifs: vec4<f32>,
    extra: vec4<f32>,
    extra2: vec4<f32>,  // crossfade_alpha, reserved, reserved, reserved
}
```

**Step 3: Update uniform writing**

In `src/main.rs` where the Uniforms struct is constructed (~line 1108), add:

```rust
extra2: [0.0, 0.0, 0.0, 0.0], // crossfade_alpha — will be set properly in Task 3
```

**Step 4: Also update the compute shader Uniforms**

In `flame_compute.wgsl`, the Uniforms struct must match layout. Add `extra2` there too:

```wgsl
struct Uniforms {
    time: f32,
    frame: u32,
    resolution: vec2<f32>,
    mouse: vec2<f32>,
    transform_count: u32,
    has_final_xform: u32,
    globals: vec4<f32>,
    kifs: vec4<f32>,
    extra: vec4<f32>,
    extra2: vec4<f32>,
}
```

**Step 5: Build and verify**

Run: `cargo build 2>&1`
Expected: compiles with no errors

**Step 6: Commit**

```bash
git add src/main.rs playground.wgsl flame_compute.wgsl
git commit -m "feat: add extra2 uniform field for crossfade_alpha"
```

---

### Task 2: Create crossfade texture and update bind groups

**Files:**
- Modify: `src/main.rs:89-112` (Gpu struct)
- Modify: `src/main.rs:577-601` (create_frame_textures)
- Modify: `src/main.rs:175-222` (bind group layout)
- Modify: `src/main.rs:603-633` (create_render_bind_group)
- Modify: `src/main.rs:294-340` (Gpu::create)
- Modify: `src/main.rs:343-380` (Gpu::resize)
- Modify: `src/main.rs:394-418` (rebuild_bind_groups)

**Step 1: Modify create_frame_textures to return Textures too**

The frame textures need `COPY_SRC` for the crossfade snapshot copy. Change the function to return `(Texture, TextureView, Texture, TextureView)`:

```rust
fn create_frame_textures(
    device: &wgpu::Device,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Texture, wgpu::TextureView) {
    let desc = wgpu::TextureDescriptor {
        label: Some("frame"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    };
    let a = device.create_texture(&desc);
    let a_view = a.create_view(&Default::default());
    let b = device.create_texture(&desc);
    let b_view = b.create_view(&Default::default());
    (a, a_view, b, b_view)
}
```

**Step 2: Add create_crossfade_texture function**

```rust
fn create_crossfade_texture(
    device: &wgpu::Device,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let desc = wgpu::TextureDescriptor {
        label: Some("crossfade"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    let tex = device.create_texture(&desc);
    let view = tex.create_view(&Default::default());
    (tex, view)
}
```

**Step 3: Update Gpu struct to store Texture objects**

```rust
struct Gpu {
    // ... existing fields ...
    frame_a_tex: wgpu::Texture,
    frame_a: wgpu::TextureView,
    frame_b_tex: wgpu::Texture,
    frame_b: wgpu::TextureView,
    // ... remove old frame_a, frame_b TextureView-only fields ...
    crossfade_tex: wgpu::Texture,
    crossfade_view: wgpu::TextureView,
    // ... rest ...
}
```

**Step 4: Add binding 4 to render bind group layout**

Add a new entry for the crossfade texture at binding 4:

```rust
wgpu::BindGroupLayoutEntry {
    binding: 4,
    visibility: wgpu::ShaderStages::FRAGMENT,
    ty: wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float {
            filterable: true,
        },
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    },
    count: None,
},
```

**Step 5: Update create_render_bind_group to include crossfade texture**

Add `crossfade_view: &wgpu::TextureView` parameter and a new entry:

```rust
wgpu::BindGroupEntry {
    binding: 4,
    resource: wgpu::BindingResource::TextureView(crossfade_view),
},
```

**Step 6: Update Gpu::create to use new signatures**

Update the `create_frame_textures` call, create crossfade texture, update all bind group creation calls.

**Step 7: Update Gpu::resize to recreate crossfade texture**

In the resize method, also recreate the crossfade texture at the new size.

**Step 8: Update rebuild_bind_groups to pass crossfade_view**

Both bind_group_a and bind_group_b need the crossfade_view passed in.

**Step 9: Build and verify**

Run: `cargo build 2>&1`
Expected: compiles with no errors

**Step 10: Commit**

```bash
git add src/main.rs
git commit -m "feat: crossfade texture + updated bind groups for dissolve transitions"
```

---

### Task 3: Crossfade state + snapshot logic in main.rs

**Files:**
- Modify: `src/main.rs:720-745` (App struct)
- Modify: `src/main.rs:1062-1072` (mutation trigger)
- Modify: `src/main.rs:1107-1121` (uniform writing)
- Modify: `src/main.rs:420-500` (Gpu::render)

**Step 1: Add crossfade state to App**

```rust
struct App {
    // ... existing fields ...
    crossfade_alpha: f32,       // 0.0 = fully old, 1.0 = fully new
    crossfade_active: bool,
}
```

Initialize in `App::new()`: `crossfade_alpha: 1.0, crossfade_active: false`.

**Step 2: Snapshot + reset on mutation**

Where the auto-evolve mutation fires (~line 1062), add snapshot logic:

```rust
if self.mutation_accum >= 1.0 {
    self.mutation_accum = 0.0;

    // Snapshot current frame for crossfade
    if let Some(gpu) = &mut self.gpu {
        gpu.snapshot_crossfade();
    }
    self.crossfade_alpha = 0.0;
    self.crossfade_active = true;

    self.genome_history.push(self.genome.clone());
    if self.genome_history.len() > 10 {
        self.genome_history.remove(0);
    }
    self.genome = self.genome.mutate();
    self.last_mutation_time = self.start.elapsed().as_secs_f32();
    self.apply_genome_targets();
    eprintln!("[auto-evolve] → {}", self.genome.name);
}
```

**Step 3: Add Gpu::snapshot_crossfade method**

This creates a command buffer that copies the current feedback frame to the crossfade texture:

```rust
fn snapshot_crossfade(&mut self) {
    // Copy whichever frame was most recently rendered to
    let source_tex = if self.ping { &self.frame_b_tex } else { &self.frame_a_tex };
    // Note: ping was toggled AFTER rendering, so the most recent render
    // went to frame_a if ping is now true (it was false during render),
    // or frame_b if ping is now false. Actually: ping toggles at end of render().
    // If ping==true now, last render wrote to frame_a. If ping==false, wrote to frame_b.
    // Wait — need to check the actual ping-pong logic. The previous frame's
    // feedback texture (the one being sampled, not rendered to) is the snapshot source.
    // Actually, the simplest: copy the texture that was RENDERED TO last frame.
    // ping toggles at end of render. Before toggle: target = frame_a if ping, frame_b if !ping.
    // After toggle (current state): if ping==true, last target was frame_a (ping was true).
    // No wait — let me just check. If ping is currently true, then during the LAST render(),
    // ping was true, so target was frame_a. So we copy frame_a.
    let source = if self.ping { &self.frame_a_tex } else { &self.frame_b_tex };

    let mut encoder = self.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_texture(
        source.as_image_copy(),
        self.crossfade_tex.as_image_copy(),
        wgpu::Extent3d {
            width: self.config.width,
            height: self.config.height,
            depth_or_array_layers: 1,
        },
    );
    self.queue.submit(std::iter::once(encoder.finish()));
}
```

Note: verify the ping-pong logic by reading the render() method carefully. The comment above captures the intent — copy whichever frame was most recently rendered to.

**Step 4: Ramp crossfade_alpha each frame**

In the update section, before writing uniforms:

```rust
if self.crossfade_active {
    self.crossfade_alpha += dt / 7.0; // 7-second dissolve
    if self.crossfade_alpha >= 1.0 {
        self.crossfade_alpha = 1.0;
        self.crossfade_active = false;
    }
}
```

**Step 5: Write crossfade_alpha to uniforms**

Update the Uniforms construction:

```rust
extra2: [self.crossfade_alpha, 0.0, 0.0, 0.0],
```

**Step 6: Also snapshot on keyboard-triggered mutations**

Search for all places where `self.genome = self.genome.mutate()` or genome changes happen (keyboard 'm' key, undo 'z' key, etc.) and add the same snapshot logic. Extract a helper:

```rust
fn trigger_mutation(&mut self) {
    if let Some(gpu) = &mut self.gpu {
        gpu.snapshot_crossfade();
    }
    self.crossfade_alpha = 0.0;
    self.crossfade_active = true;
    self.genome_history.push(self.genome.clone());
    if self.genome_history.len() > 10 {
        self.genome_history.remove(0);
    }
}
```

**Step 7: Build and verify**

Run: `cargo build 2>&1`
Expected: compiles (crossfade won't be visible yet — shader not updated)

**Step 8: Commit**

```bash
git add src/main.rs
git commit -m "feat: crossfade snapshot + alpha ramp on genome mutation"
```

---

### Task 4: Fragment shader crossfade blend

**Files:**
- Modify: `playground.wgsl:19-23` (bindings)
- Modify: `playground.wgsl:103-195` (fragment function)

**Step 1: Add crossfade texture binding**

After the existing bindings, add:

```wgsl
@group(0) @binding(4) var crossfade_tex: texture_2d<f32>;
```

It reuses `prev_sampler` (binding 2) since it's the same filtering mode.

**Step 2: Update fragment shader to blend**

Replace the combine + feedback section with crossfade-aware logic:

```wgsl
// ── Crossfade blend ──
let crossfade = clamp(u.extra2.x, 0.0, 1.0);
let old_flame = textureSample(crossfade_tex, prev_sampler, tex_uv).rgb;

// New flame from histogram (existing tonemapping)
let new_col = flame + bg;

// Feedback trail — proportional to crossfade progress
let prev = textureSample(prev_frame, prev_sampler, tex_uv).rgb;
let new_with_trail = new_col + prev * trail;

// Blend: old snapshot dissolves out, new flame with trail fades in
var col = mix(old_flame * (1.0 - trail * 0.5), new_with_trail, crossfade);
```

The `old_flame * (1.0 - trail * 0.5)` gently dims the frozen snapshot over time so it doesn't persist forever even during slow crossfades.

Keep the existing bloom and Reinhard tonemap after this blend.

**Step 3: Run and visually verify**

Run: `cargo run`
Expected: When the flame mutates (press 'm' or wait for auto-evolve), the old flame should freeze and gradually dissolve while the new flame builds up. No hard cuts.

**Step 4: Commit**

```bash
git add playground.wgsl
git commit -m "feat: crossfade dissolve blend in fragment shader"
```

---

### Task 5: Audio-biased mutations

**Files:**
- Modify: `src/genome.rs` (mutate function signature + internals)
- Modify: `src/main.rs` (all mutate() call sites)

**Step 1: Define audio variation groups**

Add to `src/genome.rs` near the top, after ORBY_VARIATIONS:

```rust
// Variation groups biased by audio frequency content
const BASS_VARIATIONS: [usize; 4] = [2, 10, 23, 11]; // spherical, bubble, blob, fisheye
const MIDS_VARIATIONS: [usize; 4] = [1, 16, 22, 5];  // sinusoidal, waves, cosine=28 wait...
```

Actually, use the correct indices:
```rust
const BASS_VARIATIONS: [usize; 4] = [2, 10, 23, 11]; // spherical, bubble, blob, fisheye
const MIDS_VARIATIONS: [usize; 4] = [1, 16, 22, 5];  // sinusoidal, waves, cosine, handkerchief — wait
```

Correct index mapping (from FlameTransform field order):
- 0=linear, 1=sinusoidal, 2=spherical, 3=swirl, 4=horseshoe, 5=handkerchief
- 6=julia, 7=polar, 8=disc, 9=rings, 10=bubble, 11=fisheye
- 12=exponential, 13=spiral, 14=diamond, 15=bent, 16=waves, 17=popcorn
- 18=fan, 19=eyefish, 20=cross, 21=tangent, 22=cosine, 23=blob
- 24=noise, 25=curl

```rust
const BASS_VARIATIONS: [usize; 4] = [2, 10, 23, 11];  // spherical, bubble, blob, fisheye
const MIDS_VARIATIONS: [usize; 4] = [1, 16, 22, 5];   // sinusoidal, waves, cosine, handkerchief
const HIGHS_VARIATIONS: [usize; 5] = [6, 8, 20, 21, 14]; // julia, disc, cross, tangent, diamond
const BEAT_VARIATIONS: [usize; 3] = [13, 3, 7];        // spiral, swirl, polar
```

**Step 2: Add audio-biased variation picker**

```rust
use crate::audio::AudioFeatures;

fn audio_biased_variation_pick(rng: &mut impl Rng, audio: &AudioFeatures) -> usize {
    // Determine dominant audio character
    let max_band = if audio.bass >= audio.mids && audio.bass >= audio.highs {
        0 // bass
    } else if audio.mids >= audio.highs {
        1 // mids
    } else {
        2 // highs
    };

    // 40% chance to pick from audio-favored group, 60% random (with orby bias)
    if rng.random_range(0.0..1.0) < 0.4 {
        match max_band {
            0 => BASS_VARIATIONS[rng.random_range(0..BASS_VARIATIONS.len())],
            1 => MIDS_VARIATIONS[rng.random_range(0..MIDS_VARIATIONS.len())],
            _ => HIGHS_VARIATIONS[rng.random_range(0..HIGHS_VARIATIONS.len())],
        }
    } else if audio.beat_accum > 0.5 && rng.random_range(0.0..1.0) < 0.3 {
        // Beat-dense: favor rotational variations
        BEAT_VARIATIONS[rng.random_range(0..BEAT_VARIATIONS.len())]
    } else {
        biased_variation_pick(rng)
    }
}
```

**Step 3: Change mutate() signature**

```rust
pub fn mutate(&self, audio: &AudioFeatures) -> Self {
```

Replace all calls to `biased_variation_pick(rng)` inside mutate/sub-methods with `audio_biased_variation_pick(rng, audio)`. This means the sub-methods also need the audio parameter passed through.

Alternatively, store audio on `self` temporarily or pass it through. The cleanest approach: add `audio: &AudioFeatures` parameter to the private mutation methods that pick variations: `mutate_perturb`, `mutate_add_transform`, `mutate_final_transform`.

**Step 4: Bias transform count by energy**

In the add/remove transform section of `mutate()`:

```rust
let (add_chance, remove_chance) = if n < 6 {
    (0.40, 0.05)
} else if n <= 16 {
    (0.15, 0.15)
} else {
    (0.05, 0.40)
};
// Energy biases toward more/fewer transforms
let energy_bias = audio.energy * 0.15; // up to +0.15 add chance at max energy
let add_chance = add_chance + energy_bias;
let remove_chance = (remove_chance - energy_bias * 0.5).max(0.02);
```

**Step 5: Bias symmetry by beat density**

In `mutate_symmetry`:

```rust
fn mutate_symmetry(&mut self, rng: &mut impl Rng, audio: &AudioFeatures) {
    let roll: f32 = rng.random();
    if audio.beat_accum > 0.5 {
        // Beat-dense: favor higher rotational symmetry
        if roll < 0.15 {
            self.symmetry = 1;
        } else if roll < 0.65 {
            self.symmetry = rng.random_range(3..=8);
        } else {
            self.symmetry = -rng.random_range(3..=6);
        }
    } else {
        // Original logic
        if roll < 0.3 {
            self.symmetry = 1;
        } else if roll < 0.7 {
            self.symmetry = rng.random_range(2..=6);
        } else {
            self.symmetry = -rng.random_range(2..=4);
        }
    }
}
```

**Step 6: Update all mutate() call sites in main.rs**

Search for `.mutate()` calls and pass `&self.audio_features`:

- `App::new()`: `genome.mutate()` → use `AudioFeatures::default()` (no audio at startup)
- Auto-evolve: `self.genome.mutate()` → `self.genome.mutate(&self.audio_features)`
- Keyboard 'm': same
- Any other call sites

**Step 7: Build and verify**

Run: `cargo build 2>&1`
Expected: compiles with no errors

**Step 8: Commit**

```bash
git add src/genome.rs src/main.rs
git commit -m "feat: audio-biased mutations — frequency content shapes flame evolution"
```

---

### Task 6: Real-time variation modulation via weights.json

**Files:**
- Modify: `weights.json`

**Step 1: Add variation weight modulations**

Add audio → variation weight mappings:

```json
{
  "bass": {
    "color_shift": -0.4,
    "flame_brightness": 0.3,
    "vibrancy": 0.15,
    "xfN_spherical": 0.3,
    "xfN_bubble": 0.25,
    "xfN_blob": 0.2
  },
  "mids": {
    "color_shift": 0.15,
    "xfN_sinusoidal": 0.2,
    "xfN_waves": 0.15,
    "xfN_cosine": 0.15
  },
  "highs": {
    "color_shift": 0.3,
    "xfN_julia": 0.2,
    "xfN_disc": 0.15,
    "xfN_cross": 0.1
  }
}
```

Keep all other existing mappings (energy, beat, beat_pulse, beat_accum, time signals).

**Step 2: Strengthen textural weights**

Increase existing weights that are currently too subtle:

```json
{
  "energy": {
    "flame_brightness": 1.0,
    "drift_speed": 2.0,
    "mutation_rate": 0.3,
    "bloom_intensity": 0.5
  },
  "beat_pulse": {
    "bloom_intensity": 0.6
  },
  "beat": {
    "color_shift": -0.5,
    "trail": -0.3,
    "mutation_rate": 0.2
  }
}
```

**Step 3: Run and visually verify**

Run: `cargo run`
Play music. Verify: bass-heavy passages should subtly boost spherical/bubble variation weights. The effect should be organic due to per-transform randomness.

**Step 4: Commit**

```bash
git add weights.json
git commit -m "feat: audio → variation weight modulations + stronger textural weights"
```

---

### Task 7: Fix signal divisor for audio signals

**Files:**
- Modify: `src/weights.rs:56` (SIGNAL_COUNT)
- Modify: `src/weights.rs:136-206` (apply_globals, apply_transforms)

**Step 1: Split the divisor**

Currently all 15 signals share `SIGNAL_COUNT = 15.0` as divisor, making each signal contribute only 1/15th of its weight. Audio signals (6 of them) should use a smaller divisor so they're more impactful. Time signals (9 of them) can keep a larger divisor since they're ambient.

Replace the single SIGNAL_COUNT with two:

```rust
const AUDIO_SIGNAL_COUNT: f32 = 6.0;  // bass, mids, highs, energy, beat, beat_accum
const TIME_SIGNAL_COUNT: f32 = 9.0;   // time, time_slow, ..., time_envelope
```

**Step 2: Update signal_list to return divisor per signal**

Change `signal_list` to return `Vec<(&HashMap<String, f32>, f32, f32)>` — (weights, signal_value, divisor):

```rust
fn signal_list<'a>(
    &'a self,
    features: &crate::audio::AudioFeatures,
    time_signals: &TimeSignals,
) -> Vec<(&'a HashMap<String, f32>, f32, f32)> {
    vec![
        (&self.bass, features.bass, AUDIO_SIGNAL_COUNT),
        (&self.mids, features.mids, AUDIO_SIGNAL_COUNT),
        (&self.highs, features.highs, AUDIO_SIGNAL_COUNT),
        (&self.energy, features.energy, AUDIO_SIGNAL_COUNT),
        (&self.beat, features.beat, AUDIO_SIGNAL_COUNT),
        (&self.beat_accum, features.beat_accum, AUDIO_SIGNAL_COUNT),
        (&self.time, time_signals.time, TIME_SIGNAL_COUNT),
        (&self.time_slow, time_signals.time_slow, TIME_SIGNAL_COUNT),
        (&self.time_med, time_signals.time_med, TIME_SIGNAL_COUNT),
        (&self.time_fast, time_signals.time_fast, TIME_SIGNAL_COUNT),
        (&self.time_noise, time_signals.time_noise, TIME_SIGNAL_COUNT),
        (&self.time_drift, time_signals.time_drift, TIME_SIGNAL_COUNT),
        (&self.time_flutter, time_signals.time_flutter, TIME_SIGNAL_COUNT),
        (&self.time_walk, time_signals.time_walk, TIME_SIGNAL_COUNT),
        (&self.time_envelope, time_signals.time_envelope, TIME_SIGNAL_COUNT),
    ]
}
```

**Step 3: Update apply_globals and apply_transforms**

Replace `/ SIGNAL_COUNT` with the per-signal divisor:

```rust
// In apply_globals:
for (weights, signal_val, divisor) in &signals {
    for (name, &weight) in *weights {
        if name == "mutation_rate" { continue; }
        if name.starts_with("xf") { continue; }
        if let Some(idx) = global_index(name) {
            result[idx] += weight * signal_val / divisor;
        }
    }
}

// Same pattern in apply_transforms and mutation_rate
```

**Step 4: Update mutation_rate method too**

```rust
pub fn mutation_rate(&self, features: &crate::audio::AudioFeatures, time_signals: &TimeSignals) -> f32 {
    let signals = self.signal_list(features, time_signals);
    let mut rate = 0.0;
    for (weights, signal_val, divisor) in &signals {
        if let Some(&w) = weights.get("mutation_rate") {
            rate += w * signal_val / divisor;
        }
    }
    rate
}
```

**Step 5: Build and run tests**

Run: `cargo build 2>&1`
Run: `cargo test 2>&1`
Expected: all pass (audio tests don't depend on SIGNAL_COUNT)

**Step 6: Commit**

```bash
git add src/weights.rs
git commit -m "feat: split signal divisor — audio signals 2.5x stronger than time signals"
```

---

### Task 8: 3-second mutation cooldown

**Files:**
- Modify: `src/main.rs` (mutation trigger section)

**Step 1: Add cooldown constant and check**

At the mutation trigger (~line 1062):

```rust
const MUTATION_COOLDOWN: f32 = 3.0;

// Auto-evolve via mutation_rate
let mr = self.weights.mutation_rate(&self.audio_features, &time_signals);
self.mutation_accum += mr * dt;
let time_since_last = time - self.last_mutation_time;
if self.mutation_accum >= 1.0 && time_since_last >= MUTATION_COOLDOWN {
    self.mutation_accum = 0.0;
    // ... rest of mutation logic
}
```

The accumulator still fills up during the cooldown — it just can't fire until 3s have passed. This means after the cooldown, the next mutation fires immediately if the accumulator is full.

**Step 2: Build and verify**

Run: `cargo build 2>&1`
Expected: compiles

**Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat: 3-second mutation cooldown to prevent degenerate rapid-fire"
```

---

### Task 9: Integration test + tag

**Files:**
- All modified files from tasks 1-8

**Step 1: Run full test suite**

Run: `cargo test 2>&1`
Expected: all tests pass

**Step 2: Run and visually verify the full experience**

Run: `cargo run`

Verify:
- [ ] Mutation transitions show smooth crossfade dissolve (no hard cuts)
- [ ] Chained mutations dissolve smoothly (A → B → C all smooth)
- [ ] During bass-heavy music, mutations favor round/blobby shapes
- [ ] During treble-heavy music, mutations favor sharp/detailed shapes
- [ ] Variation weights subtly shift with audio in real-time
- [ ] Bloom and brightness respond more visibly to energy/beats
- [ ] No mutations fire within 3 seconds of each other
- [ ] Silence holds the flame steady
- [ ] Press 'm' for manual mutation — also crossfades smoothly

**Step 3: Tag the release**

```bash
git tag -a v1.2.0-audio-reactivity-v2 -m "Crossfade transitions, audio-biased mutations, real-time variation modulation"
```
