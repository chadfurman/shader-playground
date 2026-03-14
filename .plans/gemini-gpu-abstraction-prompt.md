# Deep Research: Cross-Platform GPU Compute with Platform-Specific Shader Backends in Rust

## Context

I'm building a real-time fractal flame renderer in Rust. It's a GPU-heavy application — the core is a chaos game compute shader that runs ~2M point iterations per frame, splatting results to a histogram buffer via atomic operations. The display pipeline is simpler: accumulation, histogram equalization, and a fullscreen fragment shader for tonemapping/post-processing.

Currently the entire GPU stack uses **wgpu 28** (WebGPU abstraction) with **WGSL** shaders. This works cross-platform (Metal on macOS, Vulkan on Linux/Windows, DX12 on Windows) but has a critical limitation: **wgpu's shader compiler (Naga) does not support WGSL subgroup operations** (`enable subgroups;`), even though the underlying hardware and APIs all support them. Subgroup operations would reduce atomic contention on our histogram buffer by up to 32x.

I want to explore architectures that allow **platform-specific GPU optimizations** while maintaining a single Rust codebase. The compute shader is the hot path — that's where platform-specific optimizations matter most. The rest of the pipeline (buffer management, render passes, swapchain) is boilerplate that doesn't need platform-specific code.

## Current Architecture

### Pipeline (all GPU)
1. **Chaos Game Compute** — 256-1024 workgroups × 256 threads. Each thread maintains persistent state, iterates IFS transforms, splats to histogram via `atomicAdd`. This is the bottleneck.
2. **Accumulation Compute** — Exponential moving average blend, per-pixel.
3. **Histogram CDF Compute** — Parallel prefix sum over 256 bins.
4. **Fragment Render** — Reads accumulation buffer, applies tonemapping, velocity blur, DOF, bloom, trail feedback.

### Buffer Layout
- Histogram: width × height × 28 bytes (7 u32 atomics per pixel)
- Point state: 2M × 12 bytes (persistent across frames)
- Accumulation: width × height × 28 bytes
- Transform params: up to 6 transforms × 42 f32s
- Uniforms: ~20 f32 globals + 6 vec4 extras

### What I Need from Platform-Specific Code
The primary motivation is **subgroup/SIMD operations in the compute shader**:
- Metal: `simd_group` functions (`simd_sum`, `simd_broadcast`, `simd_is_first`)
- Vulkan: `subgroupAdd`, `subgroupBroadcastFirst`, `subgroupElect` (via `GL_KHR_shader_subgroup`)
- DX12: Wave intrinsics (`WaveActiveSum`, `WaveReadLaneFirst`, `WaveIsFirstLane`)

Secondary: platform-specific memory optimizations (Metal shared storage mode on Apple Silicon UMA, Vulkan memory type selection).

### Current Dependencies
- `wgpu 28` — GPU abstraction
- `winit 0.30` — windowing
- `objc2`, `objc2-app-kit`, `objc2-foundation` — already used for macOS activation policy
- `bytemuck` — buffer casting
- Rust edition 2024

## What I Want to Understand

### 1. Hybrid wgpu + Native Compute
Can I keep wgpu for the rendering pipeline (fragment shader, accumulation, CDF, render passes, swapchain) but use native APIs for just the compute dispatch?

- **Metal**: Use `objc2-metal` or `metal-rs` to create a compute pipeline with MSL shader, dispatch it, and have wgpu read the output buffer. How do you share `wgpu::Buffer` with a Metal `MTLBuffer`? Can you extract the underlying Metal buffer handle from a wgpu buffer?
- **Vulkan**: Use `ash` to create a compute pipeline with SPIR-V shader containing subgroup ops. How do you share Vulkan buffers/command queues with wgpu?
- How does command synchronization work when mixing wgpu and native API dispatches in the same frame?

### 2. wgpu Unsafe SPIR-V Passthrough
wgpu has `Features::SPIRV_SHADER_PASSTHROUGH` which accepts pre-compiled SPIR-V bypassing Naga.

- Does this work on the **Metal backend**? Or only Vulkan? If Metal, how does it translate SPIR-V subgroup ops to MSL `simd_group`?
- If it works, can I write GLSL with `GL_KHR_shader_subgroup`, compile to SPIR-V with `glslc` or `naga-cli`, and pass that through?
- What are the risks and limitations of SPIR-V passthrough (validation, portability, debugging)?

### 3. Native API Abstraction in Rust
If I were to write a thin abstraction over native APIs instead of wgpu:

- What Rust crates provide the best Metal and Vulkan bindings? (`metal-rs` vs `objc2-metal`, `ash` vs `vulkano`)
- How much code is the "boilerplate" (device creation, swapchain, buffer management, command encoding) vs the "interesting" code (shader dispatch, bind groups)?
- Is there a minimal abstraction that gives me native API access with less boilerplate than raw `ash`/`metal-rs`? Something like `gpu-allocator` + thin dispatch wrapper?
- How do projects like Bevy, wgpu itself, and gfx-hal handle this internally?

### 4. Shader Cross-Compilation
If I write shaders per-platform (MSL, GLSL, HLSL):

- What's the best toolchain for maintaining shader parity? Write once in one language and cross-compile?
- **GLSL → SPIR-V → MSL** via `spirv-cross`: does this preserve subgroup operations?
- **Slang** (shader language by NVIDIA): does it support all three targets with subgroup ops?
- **Rust-GPU** (`rust-gpu`): write shaders in Rust, compile to SPIR-V. Does it support subgroups? Does the SPIR-V work on Metal via translation?
- How do professional game engines (Unreal, Unity) handle per-platform shader variants?

### 5. Buffer Sharing and Interop
The critical question for a hybrid approach:

- Can a `wgpu::Buffer` created on the Metal backend be safely cast to/from an `MTLBuffer` for use with a native Metal compute pipeline?
- Does wgpu expose the underlying API handles? (`as_hal` methods?)
- On Apple Silicon UMA, is there a way to ensure both wgpu and native Metal operate on the same physical memory without copies?
- What about command queue synchronization — can I insert a native compute dispatch into wgpu's command stream, or do I need to synchronize externally?

### 6. Build and Distribution
- How to handle platform-specific shader compilation in the Rust build pipeline? `build.rs`?
- Conditional compilation (`#[cfg(target_os = "macos")]`) for the compute backend selection?
- Impact on cross-compilation (e.g., building macOS targets from Linux CI)?
- Does cargo-packager handle platform-specific shader assets correctly?

## Constraints
- Primary target: Apple Silicon (M1-M4) with Metal
- Secondary: Windows (Vulkan or DX12), Linux (Vulkan)
- Must remain a single Rust binary — no separate processes or IPC
- Interactive frame rates (15-60 fps)
- The compute shader changes transforms every few seconds (genetic algorithm mutations)
- Already using `objc2` for macOS integration — familiar with Objective-C bridging

## Desired Output
1. **Recommended architecture** — which approach gives the best balance of platform optimization vs maintenance cost?
2. **Concrete implementation path** for the recommended approach — crate dependencies, code structure, key API calls
3. **Risk assessment** — what could go wrong with each approach? What's the fallback?
4. **Example code or pseudocode** for the critical pieces: buffer sharing, native compute dispatch, command synchronization
5. **Comparison table** of approaches with columns: perf gain, code complexity, cross-platform coverage, maintenance burden
