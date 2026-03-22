# Render Thread Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all GPU work to a dedicated render thread so the main thread stays responsive to macOS window events (drag, resize, click).

**Architecture:** Main thread computes frame data (uniforms, transform params, config) and sends it via `mpsc::channel` to a render thread that owns the `Gpu` struct and handles all wgpu calls. This is the Chrome compositor pattern — the main/UI thread never touches the GPU.

**Tech Stack:** Rust, wgpu (Device/Queue/Surface are all Send), std::sync::mpsc, std::thread

**Why:** `get_current_texture()` blocks the main thread 15-32ms on macOS, starving AppKit event processing. Window drag/click events queue behind GPU blocking, causing 3-4 second response delays.

---

## File Structure

- **Modify:** `src/main.rs` — All changes are in this file. We're restructuring the thread boundary, not adding new modules.

The key structural changes:
1. New `RenderCommand` enum + `FrameData` struct (channel protocol)
2. New `render_thread_loop()` function (extracted from current `Gpu::render()` + buffer writes)
3. `App` struct: replace `gpu: Option<Gpu>` with `render_tx: Option<Sender<RenderCommand>>` + cached dimensions
4. All `gpu.` accesses in App become channel sends

## Critical Constraints

- **NO logic changes.** Every uniform value, buffer write, and compute dispatch must produce identical results.
- **NO new config values.** This is pure threading infrastructure.
- **131 existing tests must pass unchanged.** They test genome/taste/archive/weights, not the render loop.
- **Visual verification:** After each task that compiles, run the app and confirm fractals render correctly.

---

### Task 1: Define RenderCommand enum and FrameData struct

**Files:**
- Modify: `src/main.rs:1-50` (add after imports, before Uniforms)

This is the channel protocol. Every GPU operation App currently does becomes a command variant.

- [ ] **Step 1: Add the RenderCommand enum and FrameData struct**

Add after the imports block (line 27), before the `// ── Uniforms ──` comment:

```rust
// ── Render Thread Protocol ──

/// Data needed to render one frame. Sent from main thread to render thread.
struct FrameData {
    uniforms: Uniforms,
    xf_params: Vec<f32>,
    accum_uniforms: [f32; 4],
    hist_cdf_uniforms: [f32; 4],
    workgroups: u32,
    run_compute: bool,
}

/// Commands sent from the main thread to the render thread.
enum RenderCommand {
    /// Render one frame with the given data.
    Render(FrameData),
    /// Window was resized.
    Resize { width: u32, height: u32 },
    /// Upload a new 256-color palette (256 RGBA tuples).
    UpdatePalette(Vec<[f32; 4]>),
    /// Transform count changed — recreate the transform buffer.
    ResizeTransformBuffer(usize),
    /// Hot-reload the display shader.
    ReloadShader(String),
    /// Hot-reload the compute shader.
    ReloadComputeShader(String),
    /// Shut down the render thread.
    Shutdown,
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`
Expected: Warnings about unused types (fine for now), no errors.

- [ ] **Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat: add RenderCommand enum and FrameData struct for render thread protocol"
```

---

### Task 2: Implement render_thread_loop

**Files:**
- Modify: `src/main.rs` (add function after `Gpu` impl block, before helper functions)

This function owns the `Gpu` and loops on the channel receiver. It handles each command variant. The `Render` variant does all the buffer writes + the existing `Gpu::render()` logic.

- [ ] **Step 1: Add the render_thread_loop function**

Add after the closing `}` of `impl Gpu` (after line 911), before `// ── Helper Functions ──`:

```rust
// ── Render Thread ──

fn render_thread_loop(rx: mpsc::Receiver<RenderCommand>, mut gpu: Gpu) {
    loop {
        let cmd = match rx.recv() {
            Ok(cmd) => cmd,
            Err(_) => break, // channel closed
        };
        match cmd {
            RenderCommand::Render(data) => {
                // Write uniforms
                gpu.queue
                    .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&data.uniforms));

                // Write transform params
                gpu.queue.write_buffer(
                    &gpu.transform_buffer,
                    0,
                    bytemuck::cast_slice(&data.xf_params),
                );

                // Write accumulation uniforms
                gpu.queue.write_buffer(
                    &gpu.accumulation_uniform_buffer,
                    0,
                    bytemuck::cast_slice(&data.accum_uniforms),
                );

                // Write histogram CDF uniforms
                gpu.queue.write_buffer(
                    &gpu.histogram_cdf_uniform_buffer,
                    0,
                    bytemuck::cast_slice(&data.hist_cdf_uniforms),
                );

                // Set workgroups and render
                gpu.workgroups = data.workgroups;
                gpu.render(data.run_compute);
            }
            RenderCommand::Resize { width, height } => {
                gpu.resize(width, height);
            }
            RenderCommand::UpdatePalette(rgba_data) => {
                upload_palette_texture(&gpu.queue, &gpu.palette_texture, &rgba_data);
            }
            RenderCommand::ResizeTransformBuffer(num_transforms) => {
                gpu.resize_transform_buffer(num_transforms);
            }
            RenderCommand::ReloadShader(src) => {
                gpu.pipeline = create_render_pipeline(
                    &gpu.device,
                    &gpu.pipeline_layout,
                    &src,
                    gpu.config.format,
                );
                eprintln!("[render-thread] shader reloaded");
            }
            RenderCommand::ReloadComputeShader(src) => {
                gpu.compute_pipeline = create_compute_pipeline(
                    &gpu.device,
                    &gpu.compute_pipeline_layout,
                    &src,
                );
                eprintln!("[render-thread] compute shader reloaded");
            }
            RenderCommand::Shutdown => break,
        }
    }
    eprintln!("[render-thread] shutdown");
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`
Expected: Warnings about unused function, no errors.

- [ ] **Step 3: Run tests**

Run: `cargo test 2>&1 | tail -5`
Expected: 131 passed, 0 failed.

- [ ] **Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat: add render_thread_loop that owns Gpu and processes RenderCommands"
```

---

### Task 3: Migrate App struct — replace gpu with channel

**Files:**
- Modify: `src/main.rs` — App struct definition + App::new()

Replace `gpu: Option<Gpu>` with the channel sender and cached GPU dimensions that App needs for mouse normalization and uniform construction.

- [ ] **Step 1: Update App struct fields**

In the `App` struct (line 1493), replace:
```rust
    gpu: Option<Gpu>,
```
with:
```rust
    render_tx: Option<mpsc::Sender<RenderCommand>>,
    gpu_width: u32,
    gpu_height: u32,
```

- [ ] **Step 2: Update App::new() initialization**

In `App::new()` (around line 1602), replace:
```rust
            gpu: None,
```
with:
```rust
            render_tx: None,
            gpu_width: 1,
            gpu_height: 1,
```

- [ ] **Step 3: Verify it compiles (expect errors)**

Run: `cargo build 2>&1 | grep "error" | head -20`

This will produce many errors where `self.gpu` is referenced. That's expected — we'll fix those in Tasks 4-7.

- [ ] **Step 4: Commit (WIP — won't compile yet)**

```bash
git add src/main.rs
git commit -m "wip: replace App.gpu with render channel + cached dimensions"
```

---

### Task 4: Migrate resumed() — spawn the render thread

**Files:**
- Modify: `src/main.rs` — `ApplicationHandler::resumed()` impl

Create the Gpu, spawn the render thread, store the channel sender. Also do initial palette upload and transform buffer resize via channel instead of direct gpu access.

- [ ] **Step 1: Rewrite resumed()**

Replace the GPU creation and palette upload section (lines 2092-2155) with:

```rust
        let t = Instant::now();
        let gpu = Gpu::create(window.clone());
        let initial_width = gpu.config.width;
        let initial_height = gpu.config.height;
        eprintln!(
            "[boot] GPU initialized ({:.0}ms)",
            t.elapsed().as_secs_f64() * 1000.0
        );

        // Load a random genome and set up initial state (before gpu moves to thread)
        let genomes_dir = project_dir().join("genomes");
        if genomes_dir.exists() {
            if let Ok(g) = FlameGenome::load_random(&genomes_dir) {
                eprintln!("[genome] loaded: {}", g.name);
                self.genome = g;
                let g_globals = self.genome.flatten_globals(&self.weights._config);
                let g_xf = self.genome.flatten_transforms();
                self.globals = g_globals;
                self.xf_params = g_xf.clone();
                self.morph_base_globals = g_globals;
                self.morph_base_xf = g_xf.clone();
                self.morph_start_globals = g_globals;
                self.morph_start_xf = g_xf;
                self.morph_progress = 1.0;
                self.num_transforms = self.genome.total_buffer_transforms();
            }
        }

        // Upload palette + resize transform buffer while we still own gpu directly
        let palette_data = crate::genome::palette_rgba_data(&self.genome);
        upload_palette_texture(&gpu.queue, &gpu.palette_texture, &palette_data);
        gpu.resize_transform_buffer(self.num_transforms);

        // Spawn render thread — gpu moves into it permanently
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("render".into())
            .spawn(move || render_thread_loop(rx, gpu))
            .expect("failed to spawn render thread");

        self.render_tx = Some(tx);
        self.gpu_width = initial_width;
        self.gpu_height = initial_height;
        self.window = Some(window);
```

Remove the old genome loading block (lines 2114-2135) and palette upload block (lines 2137-2141) since they're now integrated above.

Keep the file watcher setup, taste model rebuild, and boot timing logs exactly as-is.

- [ ] **Step 2: Verify errors are reduced**

Run: `cargo build 2>&1 | grep "error" | wc -l`
Expected: Fewer errors than before (remaining ones are in window_event/RedrawRequested).

- [ ] **Step 3: Commit (WIP)**

```bash
git add src/main.rs
git commit -m "wip: spawn render thread in resumed(), move Gpu ownership"
```

---

### Task 5: Migrate window_event — Resized + CursorMoved

**Files:**
- Modify: `src/main.rs` — window_event match arms

- [ ] **Step 1: Fix Resized handler**

Replace (line 2166-2169):
```rust
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size.width, size.height);
                }
            }
```
with:
```rust
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    self.gpu_width = size.width;
                    self.gpu_height = size.height;
                    if let Some(tx) = &self.render_tx {
                        let _ = tx.send(RenderCommand::Resize {
                            width: size.width,
                            height: size.height,
                        });
                    }
                }
            }
```

- [ ] **Step 2: Fix CursorMoved handler**

Replace (lines 2171-2178):
```rust
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(gpu) = &self.gpu {
                    self.mouse = [
                        position.x as f32 / gpu.config.width as f32,
                        position.y as f32 / gpu.config.height as f32,
                    ];
                }
            }
```
with:
```rust
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse = [
                    position.x as f32 / self.gpu_width.max(1) as f32,
                    position.y as f32 / self.gpu_height.max(1) as f32,
                ];
            }
```

- [ ] **Step 3: Commit (WIP)**

```bash
git add src/main.rs
git commit -m "wip: migrate Resized and CursorMoved to use render channel"
```

---

### Task 6: Migrate check_file_changes + begin_morph

**Files:**
- Modify: `src/main.rs` — `check_file_changes()` and `begin_morph()` methods

- [ ] **Step 1: Fix check_file_changes — shader reloads via channel**

Replace the shader/compute reload blocks (lines 2013-2021):
```rust
        if reload_shader {
            let src = load_shader_source();
            self.gpu.as_mut().unwrap().reload_shader(&src);
            eprintln!("[shader] reloaded");
        }
        if reload_compute {
            let src = load_compute_source();
            self.gpu.as_mut().unwrap().reload_compute_shader(&src);
            eprintln!("[compute] reloaded");
        }
```
with:
```rust
        if reload_shader {
            let src = load_shader_source();
            if let Some(tx) = &self.render_tx {
                let _ = tx.send(RenderCommand::ReloadShader(src));
            }
            eprintln!("[shader] reloaded");
        }
        if reload_compute {
            let src = load_compute_source();
            if let Some(tx) = &self.render_tx {
                let _ = tx.send(RenderCommand::ReloadComputeShader(src));
            }
            eprintln!("[compute] reloaded");
        }
```

- [ ] **Step 2: Fix check_file_changes — workgroups update via cached field**

Replace (lines 2033-2035):
```rust
            if let Some(gpu) = &mut self.gpu {
                gpu.workgroups = self.weights._config.samples_per_frame;
            }
```
with just removing those lines — workgroups are set per-frame in the FrameData now.

- [ ] **Step 3: Fix begin_morph — transform buffer resize via channel**

Replace (lines 1938-1940):
```rust
            if let Some(gpu) = &mut self.gpu {
                gpu.resize_transform_buffer(self.num_transforms);
            }
```
with:
```rust
            if let Some(tx) = &self.render_tx {
                let _ = tx.send(RenderCommand::ResizeTransformBuffer(self.num_transforms));
            }
```

- [ ] **Step 4: Fix begin_morph — palette upload via channel**

Replace (lines 1972-1975):
```rust
        if let Some(gpu) = &self.gpu {
            let palette_data = crate::genome::palette_rgba_data(&self.genome);
            upload_palette_texture(&gpu.queue, &gpu.palette_texture, &palette_data);
        }
```
with:
```rust
        if let Some(tx) = &self.render_tx {
            let palette_data = crate::genome::palette_rgba_data(&self.genome);
            let _ = tx.send(RenderCommand::UpdatePalette(palette_data));
        }
```

- [ ] **Step 5: Commit (WIP)**

```bash
git add src/main.rs
git commit -m "wip: migrate file watcher and morph to use render channel"
```

---

### Task 7: Migrate RedrawRequested — the big one

**Files:**
- Modify: `src/main.rs` — `WindowEvent::RedrawRequested` handler (lines 2394-2895)

This is where ALL the buffer writes and render calls happen. Instead of writing directly to gpu buffers, we build a `FrameData` and send it.

- [ ] **Step 1: Replace gpu access block with FrameData construction**

Replace the entire block from `let gpu = match &mut self.gpu { ... }` (line 2561) through `gpu.render(self.frame >= 3)` (line 2763) with the following. This is the same logic but building FrameData instead of writing to gpu buffers directly.

Key substitutions throughout:
- `gpu.config.width` → `self.gpu_width`
- `gpu.config.height` → `self.gpu_height`

```rust
                // ── Build uniforms (same logic, using cached gpu dimensions) ──
                let uniforms = Uniforms {
                    time: self.start.elapsed().as_secs_f32(),
                    frame: self.frame,
                    resolution: [self.gpu_width as f32, self.gpu_height as f32],
                    mouse: self.mouse,
                    transform_count: self.genome.transform_count(),
                    has_final_xform: (if self.genome.final_transform.is_some() {
                        1u32
                    } else {
                        0u32
                    }) | (self
                        .weights
                        ._config
                        .iterations_per_thread
                        .clamp(10, 2000)
                        << 16),
                    globals: [
                        self.globals[0],
                        self.globals[1],
                        self.globals[2],
                        self.globals[3],
                    ],
                    kifs: [
                        self.globals[4],
                        self.globals[5],
                        self.globals[6],
                        self.globals[7],
                    ],
                    extra: [
                        self.globals[8],
                        self.globals[9],
                        self.globals[10],
                        self.genome.symmetry as f32,
                    ],
                    extra2: [
                        self.globals[12],
                        self.globals[13],
                        self.globals[14],
                        self.globals[15],
                    ],
                    extra3: [
                        self.globals[16],
                        self.globals[17],
                        self.globals[18],
                        self.globals[19],
                    ],
                    extra4: [
                        self.weights._config.jitter_amount,
                        self.weights._config.tonemap_mode as f32,
                        self.weights._config.histogram_equalization,
                        self.weights._config.dof_strength,
                    ],
                    extra5: [
                        self.weights._config.dof_focal_distance,
                        if self.weights._config.spectral_rendering {
                            1.0
                        } else {
                            0.0
                        },
                        self.weights._config.temporal_reprojection,
                        self.prev_zoom,
                    ],
                    extra6: [
                        self.weights._config.dist_lum_strength,
                        self.weights._config.iter_lum_range,
                        0.0,
                        0.0,
                    ],
                    extra7: [
                        self.weights._config.camera_pitch,
                        self.weights._config.camera_yaw,
                        self.weights._config.camera_focal,
                        self.weights._config.dof_focal_distance,
                    ],
                    extra8: [self.weights._config.dof_strength, 0.0, 0.0, 0.0],
                };

                // IMPORTANT: update prev_zoom AFTER building uniforms (captures this frame's
                // zoom for next frame's temporal reprojection)
                self.prev_zoom = self.globals[1];

                // ── Adaptive workgroups + possible uniforms patch ──
                let effective_xforms =
                    self.genome.transform_count() * (self.genome.symmetry.unsigned_abs().max(1));
                let budget_baseline = 4u32;
                let base_wg = self.weights._config.samples_per_frame;
                let base_iters = self.weights._config.iterations_per_thread;

                let (final_uniforms, computed_workgroups) = if effective_xforms > budget_baseline {
                    let ratio = budget_baseline as f32 / effective_xforms as f32;
                    let sqrt_ratio = ratio.sqrt();
                    let wg = (base_wg as f32 * sqrt_ratio).max(256.0) as u32;
                    let scaled_iters = (base_iters as f32 * sqrt_ratio).max(40.0) as u32;
                    let has_final = if self.genome.final_transform.is_some() {
                        1u32
                    } else {
                        0u32
                    };
                    let patched = Uniforms {
                        has_final_xform: has_final | (scaled_iters.clamp(10, 2000) << 16),
                        ..uniforms
                    };
                    (patched, wg)
                } else {
                    (uniforms, base_wg)
                };

                // ── Jacobian importance sampling (CPU work, stays on main thread) ──
                let xf_write_len = self.num_transforms * 48;
                let xf_len = xf_write_len.min(self.xf_params.len());

                let final_xf_params: Vec<f32> = {
                    let jac_strength = self.weights._config.jacobian_weight_strength;
                    if jac_strength > 0.001 && self.num_transforms > 0 {
                        let mut adjusted: Vec<f32> = self.xf_params[..xf_len].to_vec();
                        let n = (xf_len / 48).min(self.num_transforms);
                        let mut weights: Vec<f32> = Vec::with_capacity(n);
                        for i in 0..n {
                            let base = i * 48;
                            let w = adjusted[base];
                            let a = adjusted[base + 1];
                            let b = adjusted[base + 2];
                            let c = adjusted[base + 4];
                            let d = adjusted[base + 5];
                            let det = (a * d - b * c).abs();
                            weights.push(w * (1.0 - jac_strength) + det * jac_strength);
                        }
                        let total: f32 = weights.iter().sum();
                        if total > 0.0 {
                            for (i, jw) in weights.iter().enumerate() {
                                adjusted[i * 48] = jw / total;
                            }
                        }
                        adjusted
                    } else {
                        self.xf_params[..xf_len].to_vec()
                    }
                };

                // ── Accumulation uniforms ──
                let base_decay = self.weights._config.accumulation_decay;
                let morph_burst_decay = self.weights._config.morph_burst_decay;
                let decay = if self.morph_burst_frames > 0 {
                    let burst_t = self.morph_burst_frames as f32 / 60.0;
                    base_decay + (morph_burst_decay - base_decay) * burst_t
                } else {
                    base_decay
                };
                let accum_uniforms: [f32; 4] = [
                    self.gpu_width as f32,
                    self.gpu_height as f32,
                    decay,
                    self.weights._config.accumulation_cap,
                ];

                // ── Histogram CDF uniforms ──
                let hist_cdf_uniforms: [f32; 4] = [
                    self.gpu_width as f32,
                    self.gpu_height as f32,
                    self.globals[3], // flame_brightness
                    (self.gpu_width * self.gpu_height) as f32,
                ];

                // Tick down morph burst
                if self.morph_burst_frames > 0 {
                    self.morph_burst_frames -= 1;
                }

                // ── Send to render thread ──
                let t_pre_render = frame_start.elapsed();
                if let Some(tx) = &self.render_tx {
                    let frame_data = FrameData {
                        uniforms: final_uniforms,
                        xf_params: final_xf_params,
                        accum_uniforms,
                        hist_cdf_uniforms,
                        workgroups: computed_workgroups,
                        run_compute: self.frame >= 3,
                    };
                    let _ = tx.send(RenderCommand::Render(frame_data));
                }
                let t_render = frame_start.elapsed();
                self.frame += 1;
```

Note: `t_render - t_pre_render` now measures channel send time (~microseconds) instead of GPU render time. That's expected — render timing is now internal to the render thread.

- [ ] **Step 2: Fix perf logging that references gpu.workgroups**

The perf log (line ~2850) uses `gpu.workgroups`. Replace with `computed_workgroups` which is now a local variable in scope. Specifically, change:
```rust
                        gpu.workgroups,
```
to:
```rust
                        computed_workgroups,
```

- [ ] **Step 3: Build and verify**

Run: `cargo build 2>&1 | tail -10`
Expected: Clean compile (possibly with warnings about unused old Gpu methods).

- [ ] **Step 4: Run tests**

Run: `cargo test 2>&1 | tail -5`
Expected: 131 passed, 0 failed.

- [ ] **Step 5: Commit**

```bash
git add src/main.rs
git commit -m "feat: migrate RedrawRequested to send FrameData via render channel"
```

---

### Task 8: Cleanup — remove hacks and dead code

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Remove device.poll() hack from Gpu::render()**

In `Gpu::render()` (line 839), remove:
```rust
        let _ = self.device.poll(wgpu::PollType::Poll);
```
The render thread doesn't need this — it's not competing with the main thread for CPU time anymore.

- [ ] **Step 2: Remove old Gpu reload methods (now handled by render_thread_loop)**

Remove `Gpu::reload_shader()` (lines 713-716) and `Gpu::reload_compute_shader()` (lines 718-721) since the render thread handles these directly.

- [ ] **Step 3: Clean up any remaining warnings**

Run: `cargo build 2>&1 | grep "warning"`
Fix any dead code warnings by removing unused code.

- [ ] **Step 4: Send Shutdown on exit**

In `WindowEvent::CloseRequested` (line 2165), before `event_loop.exit()`:
```rust
            WindowEvent::CloseRequested => {
                if let Some(tx) = self.render_tx.take() {
                    let _ = tx.send(RenderCommand::Shutdown);
                }
                event_loop.exit();
            }
```

- [ ] **Step 5: Build + test**

Run: `cargo build 2>&1 | tail -5 && cargo test 2>&1 | tail -5`
Expected: Clean build, 131 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/main.rs
git commit -m "cleanup: remove device.poll hack, dead gpu methods, add graceful shutdown"
```

---

### Task 9: Visual verification + final commit

- [ ] **Step 1: Run the app**

Run: `cargo run`
Verify:
- Fractals render correctly
- Window drag is responsive (no 3-4 second delay)
- Resize works
- Space (mutate), arrows (vote), s (save), l (load) all work
- Hot-reload weights.json works
- Audio reactivity works (if audio device available)
- Console shows `[render-thread] shutdown` on quit

- [ ] **Step 2: Delete perf_model.json (retrain with new timing profile)**

```bash
rm -f genomes/perf_model.json
```

The render thread changes frame timing characteristics, so the old perf model is stale.

- [ ] **Step 3: Final commit + tag**

```bash
git add -A
git commit -m "feat: render thread — move all GPU work off main thread

Chrome-style compositor pattern: main thread computes frame data and sends
via mpsc::channel to a dedicated render thread that owns the Gpu.

Fixes macOS window drag/click lag caused by get_current_texture() blocking
the main thread 15-32ms per frame."
git tag v0.7.0
```
