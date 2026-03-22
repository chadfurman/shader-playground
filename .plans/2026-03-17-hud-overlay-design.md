# HUD Overlay + Per-Transform Audio + Config Panel — Design Spec

## Overview

Three interconnected features that add a visual interface layer to the fractal renderer:

1. **HUD overlay** — real-time debug/info overlay shown on mouse hover, fades out after inactivity
2. **Per-transform audio reactivity** — `spin_mod` and `drift_mod` fields in the transform buffer so audio can drive individual transforms differently
3. **Interactive panels** — vote feedback, audio device selector, and live tabbed config editor

All GUI is rendered via `egui` on the render thread, overlaid on the fractal.

---

## egui Integration Architecture

### Dependencies

- `egui 0.31`, `egui-winit 0.31`, `egui-wgpu 0.31`

### Thread Ownership Split

`egui_winit::State` requires the winit `Window` for IME/clipboard integration on macOS — it **must** live on the main thread. `egui_wgpu::Renderer` needs `&wgpu::Device` and `&wgpu::Queue` — it **must** live on the render thread. `egui::Context` is thread-safe (`Send + Sync`) and lives on the render thread alongside the renderer.

### Data Flow

```
Main thread                            Render thread
┌────────────────────┐                ┌────────────────────┐
│ winit events       │                │                    │
│    ↓               │                │                    │
│ egui_winit::State  │                │                    │
│    ↓               │  RawInput      │                    │
│ produce RawInput ──────────────────→│ egui::Context      │
│                    │                │    ↓               │
│ build FrameData    │  FrameData     │ build HUD panels   │
│  + HudData ────────────────────────→│ egui_wgpu::Render  │
│                    │                │                    │
│ poll UiCommand ←───── UiCommand ────│ (reverse channel)  │
│                    │                │                    │
│                    │                │ fractal pass       │
│                    │                │ egui pass          │
│                    │                │ present            │
└────────────────────┘                └────────────────────┘
```

- **Main thread** owns `egui_winit::State`, feeds winit events to it, produces `egui::RawInput` each frame
- **Main thread** sends `RawInput` to render thread via new `RenderCommand::EguiInput(RawInput)` variant
- **Render thread** owns `egui::Context` + `egui_wgpu::Renderer`, runs UI builder, tessellates, renders
- **Reverse channel** (`mpsc::channel`): render thread sends `UiCommand` back to main thread for config changes, device switching, mutation pause
- Render pipeline: fractal compute → fractal display → egui overlay → present

---

## HUD Data

Split into static (changes on mutation only) and per-frame (changes every frame) to avoid per-frame heap allocations at 60+ fps.

```rust
/// Sent once on genome change (mutation, manual evolve, flame load).
struct HudStaticData {
    genome_name: String,
    generation: u32,
    parent_a: String,
    parent_b: String,
    num_transforms: usize,
    /// Per-transform: (weight, palette_rgb, active_variation_indices).
    /// palette_rgb sampled from CPU-side palette array, not GPU readback.
    transform_info: Vec<TransformStaticInfo>,
}

struct TransformStaticInfo {
    weight: f32,
    palette_rgb: [f32; 3],               // sampled from palette_rgba_data() at color_idx
    active_variations: Vec<(u8, f32)>,    // (variation_index, weight) for non-zero variations
}

/// Sent every frame as part of FrameData. No heap allocations — fixed-size fields only.
struct HudFrameData {
    fps: f32,
    fps_history: [f32; 60],
    mutation_accum: f32,
    time_since_mutation: f32,
    cooldown: f32,
    morph_progress: f32,
    morph_xf_rates: [f32; 12],           // fixed-size, pad with 0.0 (max 12 transforms)
    audio: AudioFeatures,
    time_signals: TimeSignalSnapshot,
}

struct TimeSignalSnapshot {
    time: f32,
    slow: f32,
    med: f32,
    fast: f32,
    noise: f32,
    drift: f32,
    flutter: f32,
    walk: f32,
    envelope: f32,
}
```

Note: `time_since_mutation` and `time_envelope` are related (`envelope = (time_since_mutation / 10.0).min(1.0)`) but both are useful — raw seconds for the cooldown display, envelope for the signal bar.

---

## HUD Panels

### Layout

| Panel | Position | Content |
|-------|----------|---------|
| Identity | top-left | FPS + sparkline, genome name, gen #, parents |
| Progress | top-right | mutation_accum bar, cooldown timer, per-transform morph bars |
| Audio | left, below identity | 7 signal bars (bass→change), animated fill |
| Time | left, below audio | 9 signal bars, bipolar for noise-based signals (center-zero) |
| Transforms | right, below progress | All transforms: index, palette color dot, weight, variation icons |
| Hotkeys | bottom, centered | Keyboard shortcuts in compact row |

### Fade Behavior

All fade timing values come from `RuntimeConfig` (no magic numbers):
- `hud_fade_delay` (default 3.0) — seconds of no mouse movement before fade starts
- `hud_fade_duration` (default 0.5) — seconds to lerp from fully visible to fully hidden

Behavior:
- Track `last_mouse_move: Instant` on the render thread (from forwarded RawInput)
- `hud_opacity = 1.0` when `elapsed < hud_fade_delay`
- Lerp from 1.0 → 0.0 over `hud_fade_duration` after delay
- When opacity reaches 0.0, skip egui entirely (no paint jobs)
- Any mouse movement resets the timer

### Variation Icons

24 tiny shapes drawn via egui's `Painter` API:
- Each variation gets a ~10-line draw function producing a distinctive icon (line for linear, sine wave for sinusoidal, overlapping circles for julia, concentric rings for rings, spiral for spiral, etc.)
- Icons rendered inline in each transform row
- `response.on_hover_text("sinusoidal")` for tooltip on hover
- No separate legend panel — tooltips are sufficient

### Signal Bar Animations

Bars reflect current signal values each frame — natural animation since values change every frame.
- Audio bars: left-aligned fill (0→1 range), glow effect on beat bar when `> beat_glow_threshold` (RuntimeConfig, default 0.8)
- Time noise signals (slow, med, fast, noise, drift, flutter): bipolar bars with center zero line, fill extends left for negative, right for positive
- time_walk: special display (monotonically growing, show as capped bar + numeric value)
- time_envelope: 0→1 bar, ramps up after mutation

---

## Interactive Panels

### Vote Feedback

- On up/down arrow vote, pop a small egui text input overlay
- Prompt: "What do you like?" (upvote) or "What don't you like?" (downvote)
- User types a note and presses Enter, or presses Enter/Escape with empty text to skip
- Note saved alongside the vote in votes.json
- Requires adding `note: Option<String>` field to `VoteEntry` with `#[serde(default)]` for backward compatibility
- Panel auto-dismisses after submission

### Audio Device Selector

- Dropdown in the HUD listing available audio capture devices
- Current device shown with a highlight
- Selecting a new device sends `UiCommand::SwitchAudioDevice(device_id)` via reverse channel
- Main thread picks it up next frame and switches the audio capture stream
- During device switch latency (may drop for several frames), audio bars gracefully show silence (zeroed `AudioFeatures`)
- Replaces (or supplements) the terminal-based device picker at startup

### Live Config Panel (hotkey `C`)

- Tabbed panel with sliders for live editing
- Tabs:
  - **Config** — morph_duration, mutation_cooldown, spin_speed_max, position_drift, drift_speed, trail, bloom_intensity, accumulation_decay, etc.
  - **Signals** — audio + time signal weight mappings (which param each signal drives, magnitude)
  - **Breeding** — parent biases, transform count range, min_breeding_distance, crossover settings
  - **Rendering** — tonemap_mode, jitter_amount, histogram_equalization, DoF, velocity_blur, gamma, etc.
- All changes apply immediately via `UiCommand::UpdateConfig` / `UiCommand::UpdateSignalWeights` on the reverse channel
- "Save" button sends `UiCommand::SaveConfig` — main thread writes current config to weights.json
- **Mutation paused** while config panel is open — render thread sends `UiCommand::PauseMutation(true)` on open, `PauseMutation(false)` on close. One-frame delay is tolerable given mutation cooldowns are 10s+.
- **File watcher suppressed** while config panel is open — main thread ignores weights.json reload events when mutation is paused, preventing the watcher from overwriting slider changes before Save

### Reverse Channel (render → main)

`mpsc::channel` in the reverse direction. Main thread polls it each frame alongside file watcher events.

```rust
enum UiCommand {
    UpdateConfig(RuntimeConfig),
    UpdateSignalWeights(Weights),
    SwitchAudioDevice(String),
    SaveConfig,
    PauseMutation(bool),
}
```

---

## Per-Transform Audio Reactivity

### Buffer Expansion

- `PARAMS_PER_XF`: 48 → 50
- New fields per transform:
  - Index 48: `spin_mod` (default 1.0) — multiplier on per-transform spin speed
  - Index 49: `drift_mod` (default 1.0) — multiplier on per-transform position drift

### Clamping

To prevent runaway values from strong audio signals, `spin_mod` and `drift_mod` are clamped to `[0.0, spin_mod_max]` and `[0.0, drift_mod_max]` respectively. These are `RuntimeConfig` fields:
- `spin_mod_max` (default 4.0) — max spin modulation multiplier
- `drift_mod_max` (default 4.0) — max drift modulation multiplier

Clamping is applied after `apply_transforms()` in the per-frame path, before sending xf_params to the render thread.

### Rust Changes

- `PARAMS_PER_XF` constant: 48 → 50
- `XF_FIELDS`: add `"spin_mod"` at index 48, `"drift_mod"` at index 49
- `push_transform()`: push 1.0 for both new fields
- `FlameTransform`: no struct change needed — these are modulation-only fields, not part of the genome. They default to 1.0 and only deviate via weight signals.

**CRITICAL: Search for all literal `48` values in the codebase.** The following are known sites that use `48` instead of `PARAMS_PER_XF` and MUST be updated:
- `main.rs` initial transform buffer allocation (`size: (6 * 48 * 4)`)
- `main.rs` `resize_transform_buffer()` (`num_transforms.max(1) * 48 * 4`)
- `main.rs` morph buffer sizing (`/ 48`, `* 48`)
- `main.rs` morph interpolation loop (`let base = idx * 48; for j in 0..48`)
- `genome.rs` `flatten_transforms()` doc comment and `Vec::with_capacity(total * 48)`
- `genome.rs` test `flatten_produces_48_floats_per_transform` (rename + update assertion)
- All other occurrences found via `grep -rn '48' src/` filtered for transform-related contexts

### Shader Changes

In `flame_compute.wgsl`:
```wgsl
fn xf(idx: u32, field: u32) -> f32 { return transforms[idx * 50u + field]; }
```

Spin line (~285):
```wgsl
let spin_speed = (hash_f(seed + 300u) * 2.0 - 1.0) * spin_max * xf(idx, 48u);
```

Drift lines (~289-290):
```wgsl
let drift_scale = xf(idx, 49u);
ox_drift = vnoise(...) * pos_drift * drift_scale;
oy_drift = vnoise(...) * pos_drift * drift_scale;
```

### Weight System

`xfN_spin_mod` and `xfN_drift_mod` become valid weight keys. Example:
```json
"energy": { "xfN_spin_mod": 0.3 },
"beat": { "xfN_drift_mod": 0.2 }
```

With `randomness_range: 0.5` and `magnitude_min/max: [1.0, 3.0]`, each transform gets a stochastic response — some spin a lot with energy, others barely. Some drift hard on beats, others hold still.

---

## New RuntimeConfig Fields

All new tunable values for this feature set:

```rust
// HUD fade timing
#[serde(default = "default_hud_fade_delay")]
pub hud_fade_delay: f32,           // default 3.0
#[serde(default = "default_hud_fade_duration")]
pub hud_fade_duration: f32,        // default 0.5

// Signal bar glow threshold
#[serde(default = "default_beat_glow_threshold")]
pub beat_glow_threshold: f32,      // default 0.8

// Per-transform modulation clamps
#[serde(default = "default_spin_mod_max")]
pub spin_mod_max: f32,             // default 4.0
#[serde(default = "default_drift_mod_max")]
pub drift_mod_max: f32,            // default 4.0
```

---

## What Doesn't Change

- Fractal rendering pipeline (compute + display shaders, except the stride change)
- Genome serialization format (spin_mod/drift_mod are not persisted in genomes)
- Mutation / crossover / breeding logic (only paused when config panel is open)
- Accumulation / bloom / tonemap shaders
- weights.json format (only new valid keys added)
- Audio capture system (device selector wraps existing functionality)

---

## Implementation Order

1. **Per-transform buffer expansion** — expand PARAMS_PER_XF 48→50, grep and fix ALL literal `48` sites, update shader stride, add spin_mod/drift_mod. Self-contained, no new deps.
2. **egui integration** — add deps, set up `egui_winit::State` on main thread and `egui::Context` + `egui_wgpu::Renderer` on render thread. Get a blank overlay rendering on top of the fractal. RawInput forwarding from main thread.
3. **Reverse channel** — `mpsc::channel` render→main, `UiCommand` enum. Wire up polling on main thread.
4. **HUD panels** — build out each panel one at a time: identity → progress → signals → transforms → hotkeys. One commit per panel, visual verification after each.
5. **Fade system** — mouse tracking + alpha lerp using RuntimeConfig timing values.
6. **Variation icons** — 24 painter draw functions + hover tooltips.
7. **Vote feedback** — add `note` field to `VoteEntry`, text input popup on vote.
8. **Audio device selector** — dropdown + UiCommand for device switching.
9. **Config panel** — tabbed slider panel, UiCommand for config updates, Save button, mutation pause + watcher suppression.

---

## Testing Strategy

- **Per-transform buffer**: unit test that flatten produces 50 floats per transform, spin_mod/drift_mod default to 1.0. Grep for any remaining literal `48`.
- **HudStaticData construction**: unit test that TransformStaticInfo correctly extracts non-zero variations and samples palette RGB
- **Variation icons**: visual verification — each icon should be visually distinct at 14x14px
- **Fade timing**: unit test the opacity lerp function with configurable delay/duration values
- **Config panel save**: roundtrip test — modify config in memory, save, reload, verify values match
- **VoteEntry backward compat**: roundtrip test — old votes.json without `note` field loads correctly
- **Signal bars**: visual verification — bars should animate smoothly with audio playing
- **Reverse channel**: unit test UiCommand send/recv, verify PauseMutation suppresses evolution
- **Integration**: run app, hover mouse, verify HUD appears and fades. Press C, verify config panel. Vote, verify feedback prompt.
