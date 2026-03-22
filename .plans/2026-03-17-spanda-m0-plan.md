# Spanda M0: Fork & Fly — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the Spanda repo as a fork of shader-playground with basic WASD flight through fractal flame space.

**Architecture:** Full copy of shader-playground. Strip macOS-only audio capture (`sck_audio.rs`), HUD overlay (egui), and dev tooling. Add a player camera with WASD movement that flies through the existing 3D fractal renderer. Keep the morph system, audio reactivity, and genome evolution running as-is — we're just adding a camera you can steer.

**Tech Stack:** Rust, wgpu, winit, cpal (existing stack minus screencapturekit/egui)

**Spec:** `.plans/2026-03-17-spanda-game-design.md`

---

### Task 1: Fork shader-playground into spanda repo

**Files:**
- Create: `/Users/chadfurman/projects/spanda/` (full copy)
- Modify: `Cargo.toml` (rename package)
- Modify: `CLAUDE.md` (update project context)

- [ ] **Step 1: Copy the repo**

```bash
cp -r /Users/chadfurman/projects/shader-playground /Users/chadfurman/projects/spanda
cd /Users/chadfurman/projects/spanda
rm -rf target/ .git .plans/.superpowers/ perf.log audio_samples_*.json audio_features.json
git init
```

- [ ] **Step 2: Update Cargo.toml**

Change package name from `shader-playground` to `spanda`. Update description.

```toml
[package]
name = "spanda"
version = "0.1.0"
edition = "2024"
```

Update `[package.metadata.packager]`:
```toml
product-name = "Spanda"
identifier = "com.chadfurman.spanda"
version = "0.1.0"
description = "Music-reactive fractal flight — the divine creative pulse"
```

- [ ] **Step 3: Update CLAUDE.md**

Replace shader-playground project context with Spanda context. Reference the game design spec.

- [ ] **Step 4: Build and verify**

```bash
cargo build --release 2>&1 | tail -5
```

Expected: successful build.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock CLAUDE.md src/ *.wgsl weights.json params.json genomes/default.json build.rs LICENSE README.md .gitignore
git commit -m "init: fork shader-playground as spanda"
```

---

### Task 1.5: Game-tuned config & remove pre-render guard

**Files:**
- Modify: `weights.json` — lower compute demands for game responsiveness
- Modify: `src/main.rs` — remove frame-skip guard and lower frame latency

- [ ] **Step 1: Tune weights.json for game performance**

Update `_config` section:
```json
"workgroups": 128,
"samples_per_frame": 256,
"iterations_per_thread": 200,
"transform_count_min": 3,
"transform_count_max": 6,
"desired_maximum_frame_latency": 2
```

These can all be tuned up later via hot-reload. Start lean.

- [ ] **Step 2: Remove pre-render frame skip**

In `src/main.rs` line ~3403, change:
```rust
run_compute: self.frame >= 3,
```
to:
```rust
run_compute: true,
```

This was a pre-render-thread workaround to let the pipeline stabilize. The render thread handles this now — compute should run from frame 1.

- [ ] **Step 3: Lower surface frame latency**

In `src/main.rs` line ~259, change:
```rust
desired_maximum_frame_latency: 3,
```
to:
```rust
desired_maximum_frame_latency: 1,
```

Lower = less input lag. Important for a game where you're steering.

- [ ] **Step 4: Remove debug frame logging**

Remove the `eprintln!` guards at lines ~1268 and ~2969 that log the first 3 frames. These are shader-playground dev noise.

- [ ] **Step 5: Build and run**

```bash
cargo build --release && cargo run --release
```

Should start faster (no 3-frame delay) and run lighter.

- [ ] **Step 6: Commit**

```bash
git add weights.json src/main.rs
git commit -m "perf: game-tuned config — lower compute demands, remove frame-skip guard"
```

---

### Task 2: Strip macOS-only and unnecessary dependencies

**Files:**
- Remove: `src/sck_audio.rs`
- Remove: `src/device_picker.rs` (screencapturekit device picker)
- Modify: `src/main.rs` — extensive removal (see details below)
- Modify: `Cargo.toml` — remove `screencapturekit`, `objc2`, `objc2-app-kit`, `objc2-foundation`, `egui`, `egui-winit`, `egui-wgpu`, `crossterm`

**⚠️ egui removal is extensive.** `main.rs` has ~142 lines of egui code including:
- HUD panel functions: `hud_frame`, `signal_bar`, `bipolar_bar`, `morph_color`, `hud_panel_identity`, `hud_panel_progress`, `hud_panel_audio`, `hud_panel_time`, `hud_panel_transforms`, `hud_panel_hotkeys`
- `egui_state` field in the App struct
- egui renderer setup on the render thread (~lines 1258-1370)
- `RenderCommand::EguiInput` variant and egui input forwarding in event loop
- egui rendering section in the render thread loop

**⚠️ device_picker::run() replacement.** `device_picker::run()` handles audio device selection. Replace with: default to cpal's default input device. If no input device available, start without audio (audio features stay at zero).

- [ ] **Step 1: Remove sck_audio.rs and device_picker.rs**

```bash
rm src/sck_audio.rs src/device_picker.rs
```

- [ ] **Step 2: Remove mod declarations and usage from main.rs**

Remove these lines from `src/main.rs`:
- `mod sck_audio;`
- `mod device_picker;`
- All code that references `sck_audio` or `device_picker` (search for these strings)
- Replace `device_picker::run()` call with: use cpal default input device, fall back to no audio if unavailable
- All egui code: HUD panel functions, `egui_state` in App, `RenderCommand::EguiInput`, egui renderer setup on render thread, egui input forwarding, egui rendering in render loop
- All crossterm references

- [ ] **Step 3: Remove dependencies from Cargo.toml**

Remove from main `[dependencies]`:
- `screencapturekit` (line 19 — this is in the main deps block, NOT in target-cfg)
- `egui`, `egui-winit`, `egui-wgpu`
- `crossterm`

Remove the entire `[target.'cfg(target_os = "macos")'.dependencies]` section (objc2, objc2-app-kit, objc2-foundation).

Remove `[package.metadata.packager.macos]` section.

- [ ] **Step 4: Build and verify**

```bash
cargo build --release 2>&1 | tail -5
```

Fix any remaining references to removed modules until it compiles clean.

- [ ] **Step 5: Run tests**

```bash
cargo test 2>&1 | tail -10
```

All existing tests should still pass (none depend on sck_audio/device_picker/egui).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "strip: remove macOS-only audio capture, egui HUD, crossterm"
```

---

### Task 3: Add player camera with WASD flight

This is the core of M0. We add a `PlayerCamera` struct that tracks position and orientation, reads WASD + mouse input, and feeds into the existing `camera_pitch`/`camera_yaw`/`camera_focal` uniforms.

**Files:**
- Create: `src/player.rs` — PlayerCamera struct, input handling, position update
- Modify: `src/main.rs` — add `mod player`, wire input events, feed camera into uniforms
- Test: Unit tests in `src/player.rs`

- [ ] **Step 1: Write failing tests for PlayerCamera**

Create `src/player.rs` with tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_camera_at_origin() {
        let cam = PlayerCamera::new();
        assert_eq!(cam.position, [0.0, 0.0, 0.0]);
        assert_eq!(cam.yaw, 0.0);
        assert_eq!(cam.pitch, 0.0);
    }

    #[test]
    fn move_forward_changes_position() {
        let mut cam = PlayerCamera::new();
        cam.set_input(CameraInput { forward: true, ..Default::default() });
        cam.update(1.0);
        assert!(cam.position[2] != 0.0, "forward should change z");
    }

    #[test]
    fn move_right_changes_position() {
        let mut cam = PlayerCamera::new();
        cam.set_input(CameraInput { right: true, ..Default::default() });
        cam.update(1.0);
        assert!(cam.position[0] != 0.0, "right should change x");
    }

    #[test]
    fn mouse_delta_changes_yaw_pitch() {
        let mut cam = PlayerCamera::new();
        cam.apply_mouse_delta(10.0, 5.0);
        assert!(cam.yaw != 0.0);
        assert!(cam.pitch != 0.0);
    }

    #[test]
    fn pitch_clamped_to_bounds() {
        let mut cam = PlayerCamera::new();
        cam.apply_mouse_delta(0.0, 10000.0);
        assert!(cam.pitch <= std::f32::consts::FRAC_PI_2);
        assert!(cam.pitch >= -std::f32::consts::FRAC_PI_2);
    }

    #[test]
    fn uniforms_reflect_position() {
        let mut cam = PlayerCamera::new();
        cam.position = [1.0, 2.0, 3.0];
        cam.yaw = 0.5;
        cam.pitch = 0.2;
        let u = cam.uniforms();
        assert_eq!(u.pitch, 0.2);
        assert_eq!(u.yaw, 0.5);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test --lib player 2>&1 | tail -5
```

Expected: compilation error (PlayerCamera not defined yet).

- [ ] **Step 3: Implement PlayerCamera**

```rust
use std::f32::consts::FRAC_PI_2;

const MOVE_SPEED: f32 = 2.0;
const MOUSE_SENSITIVITY: f32 = 0.003;

#[derive(Default)]
pub struct CameraInput {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
}

pub struct CameraUniforms {
    pub pitch: f32,
    pub yaw: f32,
    pub focal: f32,
    pub position: [f32; 3],
}

pub struct PlayerCamera {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub focal: f32,
    input: CameraInput,
}

impl PlayerCamera {
    pub fn new() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            focal: 1.0,
            input: CameraInput::default(),
        }
    }

    pub fn set_input(&mut self, input: CameraInput) {
        self.input = input;
    }

    pub fn apply_mouse_delta(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * MOUSE_SENSITIVITY;
        self.pitch = (self.pitch + dy * MOUSE_SENSITIVITY).clamp(-FRAC_PI_2, FRAC_PI_2);
    }

    pub fn update(&mut self, dt: f32) {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, _cp) = self.pitch.sin_cos();

        // Forward/back along yaw direction
        let mut dz = 0.0f32;
        let mut dx = 0.0f32;
        let mut dy = 0.0f32;

        if self.input.forward { dz -= 1.0; }
        if self.input.backward { dz += 1.0; }
        if self.input.right { dx += 1.0; }
        if self.input.left { dx -= 1.0; }
        if self.input.up { dy += 1.0; }
        if self.input.down { dy -= 1.0; }

        let speed = MOVE_SPEED * dt;
        self.position[0] += (dx * cy - dz * sy) * speed;
        self.position[1] += dy * speed;
        self.position[2] += (dx * sy + dz * cy) * speed;
    }

    pub fn uniforms(&self) -> CameraUniforms {
        CameraUniforms {
            pitch: self.pitch,
            yaw: self.yaw,
            focal: self.focal,
            position: self.position,
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test --lib player 2>&1 | tail -10
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/player.rs
git commit -m "feat: PlayerCamera with WASD + mouse look"
```

---

### Task 4: Wire PlayerCamera into the main loop

**Files:**
- Modify: `src/main.rs` — add `mod player`, create PlayerCamera in App, handle input events, feed uniforms

- [ ] **Step 1: Add mod player to main.rs**

Add `mod player;` to the module declarations at the top.
Add `use crate::player::{PlayerCamera, CameraInput};` to imports.

- [ ] **Step 2: Add PlayerCamera to App struct**

Add `camera: PlayerCamera` field to the App struct. Initialize in `App::new()` with `PlayerCamera::new()`.

- [ ] **Step 3: Handle keyboard input for camera**

In the `KeyEvent` handler (where keys like Space, Up, Down, etc. are handled), add WASD/QE mapping:

- W → forward, S → backward, A → left, D → right, Q → down, E → up
- Track key press/release state and call `camera.set_input(...)` each frame

- [ ] **Step 4: Handle mouse input for camera look**

Use `WindowEvent::CursorMoved` with previous-position tracking to compute deltas (simpler than DeviceEvent which requires implementing the `device_event` trait method). Store `last_cursor_pos: Option<[f64; 2]>` in App. On each CursorMoved event, compute delta from previous, call `camera.apply_mouse_delta(dx, dy)`, update stored position. Reset to None on `CursorLeft`.

- [ ] **Step 4b: Grab cursor for FPS-style mouse look**

On window focus / mouse click, grab the cursor for smooth look:
```rust
window.set_cursor_grab(winit::window::CursorGrabMode::Confined).ok();
window.set_cursor_visible(false);
```
Add Escape key to release cursor (toggle grab). Without this, mouse hits screen edges and look breaks.

- [ ] **Step 5: Feed camera uniforms into the render pipeline**

In the frame update (where uniforms are built), use `camera.uniforms()` to set:
- `extra7[0]` = camera_pitch (already mapped)
- `extra7[1]` = camera_yaw (already mapped)
- `extra7[2]` = camera_focal (already mapped)
- `extra8[1]` = camera position X (currently reserved/zero)
- `extra8[2]` = camera position Y (currently reserved/zero)
- `extra8[3]` = camera position Z (currently reserved/zero)

Update the uniform layout comments in both `main.rs` and WGSL shaders to document the new extra8 mapping.

**Note:** Forward movement is yaw-only (horizontal plane). Pitch does not affect movement direction. This is intentional for M0 — "fly forward" means "drift forward." Pitch-influenced flight is a later refinement.

- [ ] **Step 6: Call camera.update(dt) each frame**

In the main update loop, call `camera.update(dt)` where `dt` is the frame delta time.

- [ ] **Step 7: Build and run**

```bash
cargo run --release
```

Expected: the fractal flame renders and you can look around with mouse + move with WASD.

- [ ] **Step 8: Commit**

```bash
git add src/main.rs
git commit -m "feat: wire PlayerCamera into main loop — WASD flight through fractal space"
```

---

### Task 5: First flight polish

**Files:**
- Modify: `src/player.rs` — add smooth damping, speed from config
- Modify: `weights.json` — add camera_speed, camera_sensitivity

- [ ] **Step 1: Add camera config to weights.json**

Add to `weights.json`:
```json
"camera_speed": 2.0,
"camera_sensitivity": 0.003,
"camera_damping": 0.85
```

- [ ] **Step 2: Add fields to RuntimeConfig**

In `src/weights.rs`, add `camera_speed`, `camera_sensitivity`, `camera_damping` with `default_*` functions.

- [ ] **Step 3: Update PlayerCamera to use config values**

Pass speed/sensitivity/damping from RuntimeConfig instead of hardcoded constants. Add velocity damping so movement feels smooth, not jerky.

- [ ] **Step 4: Build and test**

```bash
cargo test && cargo run --release
```

Fly around. Adjust values in weights.json (hot-reloaded!) until it feels good.

- [ ] **Step 5: Commit and tag**

```bash
git add -A
git commit -m "feat: smooth camera with configurable speed/sensitivity/damping"
git tag v0.1.0-m0
```

**M0 complete.** You can fly through fractal flame space with WASD + mouse. The world evolves and reacts to music around you as you drift.
