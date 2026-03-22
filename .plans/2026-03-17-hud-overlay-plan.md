# HUD Overlay + Per-Transform Audio + Config Panel — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an egui overlay with real-time signal meters, transform info, mutation progress, and live config editing. Also expand the transform buffer for per-transform audio reactivity.

**Architecture:** egui_winit::State on main thread produces RawInput, forwarded to render thread which owns egui::Context + egui_wgpu::Renderer. HUD panels overlay the fractal render. Reverse channel (render→main) enables config editing and device switching. Transform buffer expands from 48→50 floats for spin_mod/drift_mod per-transform multipliers.

**Tech Stack:** Rust, wgpu, winit, egui 0.31 + egui-winit + egui-wgpu

**Spec:** `.plans/2026-03-17-hud-overlay-design.md`

---

## Task 1: Per-Transform Buffer Expansion (48→50)

Expand `PARAMS_PER_XF` from 48 to 50, adding `spin_mod` and `drift_mod` per-transform fields. Self-contained, no new deps.

**Files:**
- Modify: `src/weights.rs:58` (PARAMS_PER_XF), `src/weights.rs:62-111` (XF_FIELDS)
- Modify: `src/genome.rs:768,778-859` (flatten + push_transform)
- Modify: `src/genome.rs:2556-2576` (test)
- Modify: `src/main.rs` (14 sites with literal `48` — see list below)
- Modify: `flame_compute.wgsl:32,35` (xf and xf_param function strides)
- Modify: `flame_compute.wgsl:285-290` (spin_speed and drift lines)

### Step 1.1: Update PARAMS_PER_XF and XF_FIELDS

- [ ] In `src/weights.rs:58`, change `PARAMS_PER_XF` from 48 to 50.
- [ ] In `src/weights.rs:62`, update the array size: `const XF_FIELDS: [&str; PARAMS_PER_XF] = [`
- [ ] After `"ngon_corners"` (line 110), add two new entries before the closing `];`:

```rust
    "spin_mod",     // 48
    "drift_mod",    // 49
```

### Step 1.2: Add RuntimeConfig fields for clamping

- [ ] In `src/weights.rs`, add to `RuntimeConfig` struct (after `transform_count_max`):

```rust
    #[serde(default = "default_spin_mod_max")]
    pub spin_mod_max: f32,
    #[serde(default = "default_drift_mod_max")]
    pub drift_mod_max: f32,
```

- [ ] Add default functions:

```rust
fn default_spin_mod_max() -> f32 { 4.0 }
fn default_drift_mod_max() -> f32 { 4.0 }
```

### Step 1.3: Update push_transform in genome.rs

- [ ] In `src/genome.rs`, at the end of `push_transform()` (before the closing `}` at ~line 859), add:

```rust
        // Per-transform audio modulation (default 1.0 = full participation)
        t.push(1.0); // 48 spin_mod
        t.push(1.0); // 49 drift_mod
```

### Step 1.4: Update flatten_transforms doc and capacity

- [ ] In `src/genome.rs:763`, update doc comment: `"Each transform = 50 floats"`
- [ ] In `src/genome.rs:768`, update capacity: `Vec::with_capacity(total * 50)`

### Step 1.5: Replace all literal 48s in main.rs

Replace every literal `48` related to transform buffer math. Use `PARAMS_PER_XF` constant — import it at the top of main.rs.

- [ ] Add import at top of `src/main.rs`: `use crate::weights::PARAMS_PER_XF;`
- [ ] Make `PARAMS_PER_XF` pub in `src/weights.rs:58`: `pub const PARAMS_PER_XF: usize = 50;`

Replace each site (all in `src/main.rs`):

- [ ] Line 256: Update comment `// 6 transforms * 50 floats * 4 bytes`
- [ ] Line 259: `size: (6 * PARAMS_PER_XF * 4) as u64,`
- [ ] Line 744: `let size = (num_transforms.max(1) * PARAMS_PER_XF * 4) as u64;`
- [ ] Line 1763: `let base = i * PARAMS_PER_XF;`
- [ ] Line 2012: `let max_xf = (self.xf_params.len().max(target_xf.len())) / PARAMS_PER_XF;`
- [ ] Line 2020: `self.morph_start_xf.resize(max_xf * PARAMS_PER_XF, 0.0);`
- [ ] Line 2445: `let base = idx * PARAMS_PER_XF;`
- [ ] Line 2574: `let num_xf = max_len / PARAMS_PER_XF;`
- [ ] Line 2579: `let base = xi * PARAMS_PER_XF;`
- [ ] Line 2580: `for j in 0..48 {` — NOTE: keep 48 here intentionally! Morph interpolation must NOT lerp spin_mod/drift_mod (indices 48-49). These are audio-driven modulation-only fields that default to 1.0 from flatten_transforms and get overwritten by apply_transforms each frame. Add a comment: `// Only morph genome-owned fields (0..48), not audio modulation fields (48-49)`
- [ ] Line 2805: `let xf_write_len = self.num_transforms * PARAMS_PER_XF;`
- [ ] Line 2811: `let n = (xf_len / PARAMS_PER_XF).min(self.num_transforms);`
- [ ] Line 2814: `let base = i * PARAMS_PER_XF;`
- [ ] Line 2826: `adjusted[i * PARAMS_PER_XF] = jw / total;`

### Step 1.6: Update shader

- [ ] In `flame_compute.wgsl:32`, change stride:

```wgsl
fn xf(idx: u32, field: u32) -> f32 { return transforms[idx * 50u + field]; }
```

- [ ] In `flame_compute.wgsl:35`, also update `xf_param` stride:

```wgsl
fn xf_param(idx: u32, param_offset: u32) -> f32 {
    return transforms[idx * 50u + 40u + param_offset];
}
```

- [ ] In `flame_compute.wgsl:285`, multiply by spin_mod:

```wgsl
let spin_speed = (hash_f(seed + 300u) * 2.0 - 1.0) * spin_max * xf(idx, 48u);
```

- [ ] In `flame_compute.wgsl:289-290`, multiply by drift_mod:

```wgsl
            ox_drift = vnoise(t * 0.03 * drift * drift_amt, seed + 100u) * pos_drift * xf(idx, 49u);
            oy_drift = vnoise(t * 0.04 * drift * drift_amt, seed + 200u) * pos_drift * xf(idx, 49u);
```

### Step 1.7: Add spin_mod/drift_mod clamping in per-frame path

- [ ] In `src/main.rs`, after `apply_variation_scales` (~line 2608), add clamping:

```rust
                    // Clamp per-transform spin_mod and drift_mod
                    let cfg = &self.weights._config;
                    for xf in 0..self.num_transforms {
                        let spin_idx = xf * PARAMS_PER_XF + 48;
                        let drift_idx = xf * PARAMS_PER_XF + 49;
                        if spin_idx < self.xf_params.len() {
                            self.xf_params[spin_idx] = self.xf_params[spin_idx].clamp(0.0, cfg.spin_mod_max);
                        }
                        if drift_idx < self.xf_params.len() {
                            self.xf_params[drift_idx] = self.xf_params[drift_idx].clamp(0.0, cfg.drift_mod_max);
                        }
                    }
```

### Step 1.8: Update test

- [ ] In `src/genome.rs`, rename test to `flatten_produces_50_floats_per_transform` and update:

```rust
    #[test]
    fn flatten_produces_50_floats_per_transform() {
        let g = FlameGenome::random_seed();
        let flat = g.flatten_transforms();
        let expected = g.transforms.len() * 50;
        assert_eq!(
            flat.len(),
            expected,
            "expected {} floats ({} transforms * 50), got {}",
            expected,
            g.transforms.len(),
            flat.len()
        );
        // Verify second transform starts at position 50
        assert!((flat[50] - g.transforms[1].weight).abs() < 1e-6);
        // Verify spin_mod defaults to 1.0
        assert!((flat[48] - 1.0).abs() < 1e-6, "spin_mod should default to 1.0");
        assert!((flat[49] - 1.0).abs() < 1e-6, "drift_mod should default to 1.0");
    }
```

### Step 1.9: Build, test, verify

- [ ] Run: `cargo test -- flatten_produces_50 -v` — should PASS
- [ ] Run: `cargo build --release` — should compile clean
- [ ] Run: `grep -rn '48' src/main.rs src/genome.rs flame_compute.wgsl | grep -v '//' | grep -v PARAMS | grep -v 'line\|Line'` — verify no remaining literal 48s in transform math (including shader)
- [ ] Visual verification: run the app, confirm it renders normally

### Step 1.10: Update weights.json docs

- [ ] Add `spin_mod_max` and `drift_mod_max` to `_config_doc` in `weights.json`
- [ ] Add `xfN_spin_mod` and `xfN_drift_mod` to `_params` in `weights.json`
- [ ] Add example to energy/beat signals:

```json
"energy": {
    "bloom_intensity": 0.02,
    "xfN_spin_mod": 0.3
},
"beat": {
    "position_drift": 0.001,
    "xfN_drift_mod": 0.2
}
```

### Step 1.11: Commit

- [ ] `git add -A && git commit -m "feat: per-transform spin_mod/drift_mod for audio reactivity (48→50 buffer)"`

---

## Task 2: Add egui Dependencies

**Files:**
- Modify: `Cargo.toml`

- [ ] Add to `[dependencies]` in `Cargo.toml`:

```toml
egui = "0.31"
egui-winit = "0.31"
egui-wgpu = "0.31"
```

- [ ] Run: `cargo check` — should resolve deps and compile
- [ ] Commit: `git commit -m "deps: add egui, egui-winit, egui-wgpu for HUD overlay"`

---

## Task 3: egui Scaffolding on Render Thread

Set up egui_winit::State on main thread, egui::Context + egui_wgpu::Renderer on render thread. Get a blank transparent overlay rendering.

**Files:**
- Modify: `src/main.rs` — App struct, RenderCommand enum, FrameData, render_thread_loop, resumed(), RedrawRequested

### Step 3.1: Add RenderCommand variants

- [ ] In `src/main.rs`, add to `RenderCommand` enum (before `Shutdown`):

```rust
    /// Forward egui input from main thread.
    EguiInput(egui::RawInput),
    /// Static HUD data (sent on genome change).
    UpdateHudStatic(Box<HudStaticData>),
```

### Step 3.2: Add HUD data structs

- [ ] In `src/main.rs`, add after the `RenderCommand` enum:

```rust
/// Static HUD data — sent once per genome change.
struct HudStaticData {
    genome_name: String,
    generation: u32,
    parent_a: String,
    parent_b: String,
    num_transforms: usize,
    transform_info: Vec<TransformStaticInfo>,
}

struct TransformStaticInfo {
    weight: f32,
    palette_rgb: [f32; 3],
    active_variations: Vec<(u8, f32)>,
}

/// Per-frame HUD data — no heap allocations.
#[derive(Copy, Clone)]
struct HudFrameData {
    fps: f32,
    fps_history: [f32; 60],   // sparkline ring buffer
    fps_history_idx: usize,
    mutation_accum: f32,
    time_since_mutation: f32,
    cooldown: f32,
    morph_progress: f32,
    morph_xf_rates: [f32; 12],
    audio: AudioFeatures,     // use the existing Copy struct directly
    time_signals: [f32; 9],   // slow, med, fast, noise, drift, flutter, walk, envelope, time
}
```

### Step 3.3: Add HudFrameData to FrameData

- [ ] Add field to `FrameData` struct (`src/main.rs:32-39`):

```rust
    hud: HudFrameData,
```

### Step 3.4: Add egui state to App struct

- [ ] Add to App struct fields (`src/main.rs:1565-1611`):

```rust
    egui_state: Option<egui_winit::State>,
    fps_history: [f32; 60],
    fps_history_idx: usize,
```

- [ ] Initialize in App constructor (in `new()` or wherever App is built):

```rust
    egui_state: None,  // initialized in resumed() when window is available
    fps_history: [0.0; 60],
    fps_history_idx: 0,
```

### Step 3.5: Initialize egui_winit::State in resumed()

- [ ] In `resumed()`, after the window is created, initialize egui state:

```rust
        let egui_ctx = egui::Context::default();
        self.egui_state = Some(egui_winit::State::new(
            egui_ctx,
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,  // theme
            None,  // max_texture_side
        ));
```

### Step 3.6: Set up egui_wgpu::Renderer in render_thread_loop

- [ ] In `render_thread_loop` (`src/main.rs:916`), after GPU setup, create egui renderer:

```rust
    let mut egui_renderer = egui_wgpu::Renderer::new(
        &gpu.device,
        gpu.config.format,  // surface format from wgpu config
        None, // depth format
        1,    // msaa samples
        false,
    );
    let egui_ctx = egui::Context::default();
    let mut last_raw_input = egui::RawInput::default();
    let mut hud_static: Option<HudStaticData> = None;
```

### Step 3.7: Handle EguiInput and UpdateHudStatic in render loop

- [ ] In the render thread's `match` on `RenderCommand`, add handlers:

```rust
            RenderCommand::EguiInput(raw_input) => {
                last_raw_input = raw_input;
            }
            RenderCommand::UpdateHudStatic(data) => {
                hud_static = Some(*data);
            }
```

### Step 3.8: Run egui pass after fractal render

- [ ] In the render thread's `Render` handler, after the fractal display pass, add:

```rust
                // egui overlay pass
                let egui_output = egui_ctx.run(last_raw_input.take(), |ctx| {
                    egui::Area::new(egui::Id::new("hud_test"))
                        .fixed_pos(egui::pos2(10.0, 10.0))
                        .show(ctx, |ui| {
                            ui.label("HUD active");
                        });
                });
                let paint_jobs = egui_ctx.tessellate(egui_output.shapes, egui_output.pixels_per_point);
                for (id, delta) in &egui_output.textures_delta.set {
                    egui_renderer.update_texture(&gpu.device, &gpu.queue, *id, delta);
                }
                let screen = egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [gpu.width, gpu.height],
                    pixels_per_point: egui_output.pixels_per_point,
                };
                egui_renderer.update_buffers(&gpu.device, &gpu.queue, &mut encoder, &paint_jobs, &screen);
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("egui"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load, // preserve fractal
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        ..Default::default()
                    });
                    egui_renderer.render(&mut rpass, &paint_jobs, &screen);
                }
                for id in &egui_output.textures_delta.free {
                    egui_renderer.free_texture(id);
                }
```

### Step 3.9: Forward winit events to egui on main thread

- [ ] In the `WindowEvent` handler in main.rs, before the existing key/mouse matching, add:

```rust
                // Feed events to egui
                if let Some(egui_state) = &mut self.egui_state {
                    let response = egui_state.on_window_event(&self.window.as_ref().unwrap(), &event);
                    if response.consumed {
                        return; // egui ate this event (e.g., typing in text field)
                    }
                }
```

### Step 3.10: Send RawInput each frame

- [ ] In the RedrawRequested handler, before sending FrameData, extract and send egui input:

```rust
                if let (Some(egui_state), Some(window), Some(tx)) =
                    (&mut self.egui_state, &self.window, &self.render_tx)
                {
                    let raw_input = egui_state.take_egui_input(window);
                    let _ = tx.try_send(RenderCommand::EguiInput(raw_input));
                }
```

### Step 3.11: Build HudFrameData and include in FrameData

- [ ] Where FrameData is constructed in RedrawRequested, add the hud field:

```rust
                    let hud = HudFrameData {
                        fps: 1.0 / dt,
                        mutation_accum: self.mutation_accum,
                        time_since_mutation: time - self.last_mutation_time,
                        cooldown: self.weights._config.mutation_cooldown,
                        morph_progress: self.morph_progress,
                        morph_xf_rates: {
                            let mut rates = [0.0f32; 12];
                            for (i, r) in self.morph_xf_rates.iter().take(12).enumerate() {
                                rates[i] = *r;
                            }
                            rates
                        },
                        audio: [
                            self.audio_features.bass, self.audio_features.mids,
                            self.audio_features.highs, self.audio_features.energy,
                            self.audio_features.beat, self.audio_features.beat_accum,
                            self.audio_features.change,
                        ],
                        time_signals: [
                            time_signals.time_slow, time_signals.time_med,
                            time_signals.time_fast, time_signals.time_noise,
                            time_signals.time_drift, time_signals.time_flutter,
                            time_signals.time_walk, time_signals.time_envelope,
                            time_signals.time,
                        ],
                    };
```

### Step 3.12: Build, test, visual verify

- [ ] Run: `cargo build --release`
- [ ] Run the app — should see "HUD active" text in top-left corner overlaid on the fractal
- [ ] Commit: `git commit -m "feat: egui scaffolding — blank overlay rendering on fractal"`

---

## Task 4: Reverse Channel (render → main)

**Files:**
- Modify: `src/main.rs` — add UiCommand enum, reverse channel setup, polling

### Step 4.1: Add UiCommand enum

- [ ] In `src/main.rs`, after the HUD data structs:

```rust
enum UiCommand {
    PauseMutation(bool),
    SaveConfig,
}
```

(Start minimal — expand when config panel is built.)

### Step 4.2: Wire up channels

- [ ] Add `ui_rx: Option<mpsc::Receiver<UiCommand>>` to App struct
- [ ] In `resumed()`, create the reverse channel and pass `tx` to render thread:

```rust
        let (ui_tx, ui_rx) = mpsc::channel();
        self.ui_rx = Some(ui_rx);
        // Pass ui_tx to render_thread_loop (add parameter)
```

- [ ] Update `render_thread_loop` signature to accept `ui_tx: mpsc::Sender<UiCommand>`
- [ ] Update the `thread::spawn` closure (~line 2231) to pass `ui_tx` into `render_thread_loop`
- [ ] Store `ui_tx` in the render loop for later use by egui panels

### Step 4.3: Poll UiCommand on main thread

- [ ] In RedrawRequested, before the evolution logic:

```rust
                // Poll UI commands from render thread
                if let Some(rx) = &self.ui_rx {
                    while let Ok(cmd) = rx.try_recv() {
                        match cmd {
                            UiCommand::PauseMutation(paused) => {
                                self.mutation_paused = paused;
                            }
                            UiCommand::SaveConfig => {
                                // Will be implemented with config panel
                            }
                        }
                    }
                }
```

- [ ] Add `mutation_paused: bool` field to App struct (default false)
- [ ] Gate evolution logic: add `&& !self.mutation_paused` to both signal_trigger and time_trigger checks

### Step 4.4: Build and commit

- [ ] Run: `cargo build --release`
- [ ] Commit: `git commit -m "feat: reverse channel render→main for UI commands"`

---

## Task 5: HUD Identity Panel (top-left)

**Files:**
- Modify: `src/main.rs` — render_thread_loop egui builder

### Step 5.1: Send HudStaticData on genome change

- [ ] Create a helper method `send_hud_static(&self)` on App that builds `HudStaticData` from the current genome and sends it via `RenderCommand::UpdateHudStatic`.
- [ ] Call it after every genome change: in spacebar handler, auto-evolve, signal-driven evolve, flame load, and `resumed()`.

### Step 5.2: Build identity panel in egui

- [ ] Replace the "HUD active" test label with:

```rust
egui::Area::new(egui::Id::new("identity"))
    .fixed_pos(egui::pos2(12.0, 12.0))
    .show(ctx, |ui| {
        egui::Frame::none()
            .fill(egui::Color32::from_rgba_unmultiplied(0, 0, 0, 153))
            .rounding(6.0)
            .inner_margin(8.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(egui::Color32::from_rgb(119, 255, 119),
                        format!("{:.0} fps", frame_data.hud.fps));
                    // FPS sparkline would go here (later)
                });
                if let Some(s) = &hud_static {
                    ui.label(egui::RichText::new(
                        format!("{} gen {}", s.genome_name, s.generation))
                        .size(10.0).color(egui::Color32::from_rgb(170, 170, 170)));
                    ui.label(egui::RichText::new(
                        format!("parents: {} x {}", s.parent_a, s.parent_b))
                        .size(9.0).color(egui::Color32::from_rgb(85, 85, 85)));
                }
            });
    });
```

### Step 5.3: Visual verify and commit

- [ ] Run the app — should see FPS, genome name, generation, parents in top-left
- [ ] Commit: `git commit -m "feat: HUD identity panel — FPS, genome name, parents"`

---

## Task 6: HUD Progress Panel (top-right)

Mutation accumulator bar, cooldown timer, per-transform morph bars.

- [ ] Build `egui::Area` at top-right with mutation progress bar, cooldown text, morph bars (one colored bar per transform, width = morph completion %)
- [ ] Visual verify — mutation bar should fill as signals accumulate
- [ ] Commit: `git commit -m "feat: HUD progress panel — mutation + morph bars"`

---

## Task 7: HUD Signal Panels (left side)

Audio signals and time signals with animated bars.

- [ ] Build Audio panel with 7 signal bars (bass→change), each showing name, value, colored fill bar
- [ ] Build Time panel with 9 signal bars, bipolar bars for noise-based signals (center zero line)
- [ ] Add `beat_glow_threshold` to RuntimeConfig (default 0.8), apply glow effect on beat bar
- [ ] Visual verify — bars should animate in real-time with audio playing
- [ ] Commit: `git commit -m "feat: HUD signal panels — audio + time signal meters"`

---

## Task 8: HUD Transforms Panel (right side)

All transforms with index, palette color dot, weight, variation icons.

- [ ] Build transforms panel showing all transforms from HudStaticData
- [ ] Each row: index number, colored circle (palette_rgb), weight value, text abbreviations of active variations (icons come in Task 10)
- [ ] Visual verify — should show all 8-12 transforms
- [ ] Commit: `git commit -m "feat: HUD transforms panel — weight, color, variations"`

---

## Task 9: HUD Hotkeys Bar + Fade System

Bottom hotkey bar and mouse-driven fade.

### Step 9.1: Hotkey bar

- [ ] Build bottom `egui::Area` with centered row of key+label pairs (Space=evolve, Up/Down=vote, S=save, F=flame, A=audio, C=config, I=info, Esc=quit)

### Step 9.2: Add fade config to RuntimeConfig

- [ ] Add `hud_fade_delay: f32` (default 3.0) and `hud_fade_duration: f32` (default 0.5) to RuntimeConfig
- [ ] Add `_config_doc` entries in weights.json

### Step 9.3: Implement fade

- [ ] Track `last_mouse_move: std::time::Instant` on render thread, reset on each `RawInput` with pointer movement
- [ ] Compute `hud_opacity` from elapsed time vs delay/duration
- [ ] Apply opacity to all panel `Frame::fill()` alpha values
- [ ] When opacity is 0, skip egui UI building entirely

### Step 9.4: Verify and commit

- [ ] Run: move mouse over window → HUD appears. Stop moving → fades after 3s.
- [ ] Commit: `git commit -m "feat: HUD hotkey bar + mouse-driven fade"`

---

## Task 10: Variation Icons with Tooltips

24 tiny shapes via egui Painter API.

- [ ] Create a function `draw_variation_icon(ui: &mut Ui, variation_idx: u8, size: f32) -> Response` that draws a distinctive shape for each of the 26 variations and returns the response for tooltip attachment
- [ ] Each icon is a small (14x14) shape: line (linear), sine wave (sinusoidal), circle+dot (spherical), curl (swirl), horseshoe shape, diamond, overlapping circles (julia), crosshairs+circle (polar), etc.
- [ ] Attach `response.on_hover_text("variation_name")` to each icon
- [ ] Replace text abbreviations in transforms panel (Task 8) with icons
- [ ] Visual verify — hover over icons, tooltips should appear
- [ ] Commit: `git commit -m "feat: variation icons with hover tooltips"`

---

## Task 11: Vote Feedback

Text input popup on up/down vote.

### Step 11.1: Add note field to VoteEntry

- [ ] In `src/votes.rs:11-15`, add to `VoteEntry`:

```rust
    #[serde(default)]
    pub note: Option<String>,
```

### Step 11.2: Build vote feedback UI

- [ ] Add `vote_feedback_state: Option<(i32, String)>` to render thread state (score, text buffer)
- [ ] When main thread sends a vote event (new `RenderCommand::VoteCast(score)`), set `vote_feedback_state = Some((score, String::new()))`
- [ ] In egui builder, if `vote_feedback_state.is_some()`, show a centered text input panel with prompt
- [ ] On Enter: send `UiCommand::VoteNote(score, note)` via reverse channel, clear state
- [ ] On Escape: send `UiCommand::VoteNote(score, None)`, clear state

### Step 11.3: Wire up vote note in main thread

- [ ] Add `VoteNote(i32, Option<String>)` to `UiCommand`
- [ ] In main thread UiCommand handler, save note with the vote in VoteLedger

### Step 11.4: Verify and commit

- [ ] Press Up/Down → popup appears, type note, Enter → saved in votes.json
- [ ] Commit: `git commit -m "feat: vote feedback — optional text note on up/down vote"`

---

## Task 12: Audio Device Selector

Dropdown in HUD for switching audio capture device.

- [ ] Add `UiCommand::SwitchAudioDevice(String)` to UiCommand enum
- [ ] Add `RenderCommand::AudioDeviceList(Vec<(String, String)>)` — send available devices to render thread on startup and periodically
- [ ] Build egui dropdown (ComboBox) showing device names, current highlighted
- [ ] On selection, send `UiCommand::SwitchAudioDevice(device_id)` via reverse channel
- [ ] Main thread handles by restarting audio capture with new device
- [ ] Commit: `git commit -m "feat: audio device selector in HUD"`

---

## Task 13: Live Config Panel (hotkey C)

Tabbed slider panel with live editing and Save button.

### Step 13.1: Config panel toggle

- [ ] Add hotkey `C` handler: sends `RenderCommand::ToggleConfigPanel` to render thread
- [ ] Render thread toggles `config_panel_open: bool`
- [ ] On open: send `UiCommand::PauseMutation(true)` via reverse channel
- [ ] On close: send `UiCommand::PauseMutation(false)`
- [ ] Main thread suppresses evolution while paused: add `&& !self.mutation_paused` to signal_trigger and time_trigger
- [ ] Main thread suppresses file watcher reloads while paused: in the `reload_weights` section (~line 2112), add `if self.mutation_paused { continue; }` or skip the reload block when paused

### Step 13.2: Config tab

- [ ] Build egui Window with tab bar (egui::TopBottomPanel or custom tabs)
- [ ] Config tab: sliders for morph_duration, mutation_cooldown, spin_speed_max, position_drift, drift_speed, trail, bloom_intensity, accumulation_decay, samples_per_frame, iterations_per_thread
- [ ] On slider change: clone the current RuntimeConfig, modify the field, send `UiCommand::UpdateConfig(RuntimeConfig)` — main thread replaces `self.weights._config` immediately

### Step 13.3: Signals tab

- [ ] Show each signal map (bass, mids, highs, etc.) with editable param→weight entries
- [ ] Add/remove entries
- [ ] On change: send `UiCommand::UpdateSignalWeights(signal_name, param, weight)`

### Step 13.4: Breeding tab

- [ ] Sliders for parent biases, transform count range, min_breeding_distance, crossover settings

### Step 13.5: Rendering tab

- [ ] Sliders/dropdowns for tonemap_mode, jitter_amount, histogram_equalization, dof_strength, velocity_blur_max, gamma, highlight_power

### Step 13.6: Save button

- [ ] "Save" button at bottom of panel
- [ ] Sends `UiCommand::SaveConfig` — main thread writes current Weights to weights.json
- [ ] Show brief "Saved!" confirmation text that fades after 2s

### Step 13.7: Verify and commit

- [ ] Press C → config panel appears, mutation pauses
- [ ] Adjust sliders → see immediate visual effect
- [ ] Press Save → weights.json updated on disk
- [ ] Press C again → panel closes, mutation resumes
- [ ] Commit: `git commit -m "feat: live config panel with tabs, sliders, and Save"`

---

## Task 14: Unit Tests + Final Polish + Tag

- [ ] Add unit test: `TransformStaticInfo` correctly extracts non-zero variations from a `FlameTransform`
- [ ] Add unit test: opacity lerp function — `elapsed=0 → 1.0`, `elapsed=3.0 → 1.0`, `elapsed=3.25 → 0.5`, `elapsed=3.5 → 0.0` (with default config values)
- [ ] Add unit test: `VoteEntry` backward compat — deserialize old votes.json without `note` field, verify it loads with `note: None`
- [ ] Add unit test: config save roundtrip — modify RuntimeConfig in memory, serialize to JSON, deserialize, verify values match
- [ ] Run full test suite: `cargo test`
- [ ] Run release build: `cargo build --release`
- [ ] Full visual verification: all HUD panels, fade, config panel, vote feedback, device selector
- [ ] Update weights.json `_config_doc` with all new RuntimeConfig fields
- [ ] Commit any final tweaks
- [ ] Tag: `git tag v0.8.0-hud-overlay`
