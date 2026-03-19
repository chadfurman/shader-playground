use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};
use std::{fs, mem};

use bytemuck::{Pod, Zeroable};
use notify::{Event, EventKind, RecursiveMode, Watcher};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes};

mod archive;
mod audio;
mod device_picker;
mod flam3;
mod genome;
mod sck_audio;
mod taste;
mod votes;
mod weights;
use crate::audio::{AudioCapture, AudioFeatures};
use crate::genome::{FavoriteProfile, FlameGenome};
use crate::votes::VoteLedger;
use crate::weights::{DRIFT_MOD_FIELD, PARAMS_PER_XF, SPIN_MOD_FIELD, Weights};

// ── Render Thread Protocol ──

/// Data needed to render one frame. Sent from main thread to render thread.
struct FrameData {
    uniforms: Uniforms,
    xf_params: Vec<f32>,
    accum_uniforms: [f32; 4],
    hist_cdf_uniforms: [f32; 4],
    workgroups: u32,
    run_compute: bool,
    /// Pre-tessellated egui paint jobs (built + tessellated on main thread).
    egui_primitives: Vec<egui::ClippedPrimitive>,
    egui_textures_delta: egui::TexturesDelta,
    egui_pixels_per_point: f32,
}

/// Commands sent from the main thread to the render thread.
enum RenderCommand {
    /// Render one frame with the given data.
    Render(Box<FrameData>),
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

// ── UI Thread Protocol ──

/// Events sent from the main (winit) thread to the UI thread.
enum UiEvent {
    /// egui raw input collected on the main thread — triggers a tick.
    EguiInput(egui::RawInput),
    /// Cursor moved — tracked for HUD fade.
    CursorMoved { x: f32, y: f32 },
    /// Key pressed (logical key name string).
    KeyPressed(String),
    /// Window resized.
    Resize { width: u32, height: u32 },
    /// Shut down the UI thread.
    Shutdown,
}

/// Per-frame HUD data — sent with each FrameData.
#[derive(Clone, Default)]
struct HudFrameData {
    fps: f32,
    mutation_accum: f32,
    time_since_mutation: f32,
    cooldown: f32,
    morph_progress: f32,
    morph_xf_rates: [f32; 12],
    num_transforms: usize,
    // Audio signals
    audio_bass: f32,
    audio_mids: f32,
    audio_highs: f32,
    audio_energy: f32,
    audio_beat: f32,
    audio_beat_accum: f32,
    audio_change: f32,
    // Time signals
    time_slow: f32,
    time_med: f32,
    time_fast: f32,
    time_noise: f32,
    time_drift: f32,
    time_flutter: f32,
    time_walk: f32,
    time_envelope: f32,
    // Per-transform weights
    transform_weights: [f32; 12],
    // Per-transform variation weights: [transform_idx][variation_idx]
    transform_variations: [[f32; 26]; 12],
    // HUD fade config
    hud_fade_delay: f32,
    hud_fade_duration: f32,
}

// ── Uniforms ──

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    time: f32,
    frame: u32,
    resolution: [f32; 2],
    mouse: [f32; 2],
    transform_count: u32,
    has_final_xform: u32, // low bit = has_final, upper 16 bits = iterations_per_thread
    globals: [f32; 4],    // speed, zoom, trail, flame_brightness
    kifs: [f32; 4],       // fold_angle, scale, brightness, drift_speed
    extra: [f32; 4],      // color_shift, vibrancy, bloom_intensity, symmetry
    extra2: [f32; 4],     // noise_disp, curl_disp, tangent_clamp, color_blend
    extra3: [f32; 4],     // spin_speed_max, position_drift, warmup_iters, velocity_blur_max
    extra4: [f32; 4],     // jitter_amount, tonemap_mode, histogram_equalization, dof_strength
    extra5: [f32; 4], // dof_focal_distance, spectral_rendering, temporal_reprojection, prev_zoom
    extra6: [f32; 4], // dist_lum_strength, iter_lum_range, _reserved, _reserved
    extra7: [f32; 4], // camera_pitch, camera_yaw, camera_focal, dof_focal_distance
    extra8: [f32; 4], // dof_strength, _reserved, _reserved, _reserved
}

// ── File Watcher ──

struct FileWatcher {
    rx: mpsc::Receiver<PathBuf>,
    _watcher: notify::RecommendedWatcher,
}

impl FileWatcher {
    fn new(paths: &[PathBuf]) -> Result<Self, String> {
        let (tx, rx) = mpsc::channel();
        let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            if let Ok(event) = res
                && matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_))
            {
                for path in &event.paths {
                    let _ = tx.send(path.clone());
                }
            }
        })
        .map_err(|e| format!("watcher: {e}"))?;

        for path in paths {
            if let Some(parent) = path.parent() {
                watcher
                    .watch(parent, RecursiveMode::NonRecursive)
                    .map_err(|e| format!("watch {}: {e}", parent.display()))?;
            }
        }
        Ok(Self {
            rx,
            _watcher: watcher,
        })
    }

    fn changed_files(&self) -> Vec<PathBuf> {
        let mut files = Vec::new();
        while let Ok(path) = self.rx.try_recv() {
            files.push(path);
        }
        files
    }
}

// ── GPU State ──

struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    // Render pipeline
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group_a: wgpu::BindGroup,
    bind_group_b: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    frame_a_tex: wgpu::Texture,
    frame_a: wgpu::TextureView,
    frame_b_tex: wgpu::Texture,
    frame_b: wgpu::TextureView,
    ping: bool,
    pipeline_layout: wgpu::PipelineLayout,
    sampler: wgpu::Sampler,
    crossfade_tex: wgpu::Texture,
    crossfade_view: wgpu::TextureView,
    // Compute pipeline (fractal flame)
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    compute_bind_group: wgpu::BindGroup,
    compute_pipeline_layout: wgpu::PipelineLayout,
    histogram_buffer: wgpu::Buffer,
    transform_buffer: wgpu::Buffer,
    palette_texture: wgpu::Texture,
    palette_view: wgpu::TextureView,
    palette_sampler: wgpu::Sampler,
    workgroups: u32,
    render_frame_count: u32,
    // Accumulation pipeline
    accumulation_pipeline: wgpu::ComputePipeline,
    accumulation_bind_group_layout: wgpu::BindGroupLayout,
    accumulation_bind_group: wgpu::BindGroup,
    accumulation_buffer: wgpu::Buffer,
    accumulation_uniform_buffer: wgpu::Buffer,
    max_density_buffer: wgpu::Buffer,
    // Persistent point state for chaos game continuity
    point_state_buffer: wgpu::Buffer,
    // Histogram equalization
    histogram_cdf_bin_pipeline: wgpu::ComputePipeline,
    histogram_cdf_sum_pipeline: wgpu::ComputePipeline,
    histogram_cdf_bind_group_layout: wgpu::BindGroupLayout,
    histogram_cdf_bind_group: wgpu::BindGroup,
    hist_bins_buffer: wgpu::Buffer,
    cdf_buffer: wgpu::Buffer,
    histogram_cdf_uniform_buffer: wgpu::Buffer,
}

impl Gpu {
    fn create(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("no GPU adapter");

        let adapter_limits = adapter.limits();
        let adapter_features = adapter.features();
        let has_subgroups = adapter_features.contains(wgpu::Features::SUBGROUP);
        eprintln!("[gpu] Adapter: {:?}", adapter.get_info().name);
        eprintln!("[gpu] Subgroup support: {}", has_subgroups);

        let mut required_features = wgpu::Features::empty();
        if has_subgroups {
            required_features |= wgpu::Features::SUBGROUP;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features,
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                max_buffer_size: adapter_limits.max_buffer_size,
                ..wgpu::Limits::default()
            },
            ..Default::default()
        }))
        .expect("device creation failed");

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
                wgpu::PresentMode::Mailbox
            } else {
                wgpu::PresentMode::Fifo
            },
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 3,
        };
        surface.configure(&device, &config);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Persistent point state: 7 f32s per thread (x, y, z, prev_x, prev_y, prev_z, color_idx)
        // Max 8192 workgroups * 256 threads = 2M threads
        let max_threads: u64 = 8192 * 256;
        let point_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("point_state"),
            size: max_threads * 28, // 7 f32s * 4 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Histogram buffer: 6 u32s per pixel (density + R + G + B + vx + vy)
        let histogram_buffer = create_histogram_buffer(&device, config.width, config.height);

        // Initial transform buffer (6 transforms * 50 floats * 4 bytes)
        let transform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transforms"),
            size: (6 * PARAMS_PER_XF * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Palette texture (256x1 Rgba32Float) ──
        let (palette_texture, palette_view) = create_palette_texture(&device);
        let palette_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("palette_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // Upload default cosine palette
        {
            let palette_data = crate::genome::generate_default_palette_rgba();
            upload_palette_texture(&queue, &palette_texture, &palette_data);
        }

        // ── Render bind group layout (uniform + prev_frame + sampler + histogram read) ──
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // ── Compute bind group layout (histogram rw + uniform + transforms) ──
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("compute"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute"),
                bind_group_layouts: &[&compute_bind_group_layout],
                immediate_size: 0,
            });

        let (frame_a_tex, frame_a, frame_b_tex, frame_b) =
            create_frame_textures(&device, config.width, config.height, format);
        let (crossfade_tex, crossfade_view) =
            create_crossfade_texture(&device, config.width, config.height, format);

        let shader_src = load_shader_source();
        let pipeline = create_render_pipeline(&device, &pipeline_layout, &shader_src, format);

        let compute_src = load_compute_source();
        let compute_pipeline =
            create_compute_pipeline(&device, &compute_pipeline_layout, &compute_src);

        // ── Accumulation buffer (needed by render bind groups) ──
        let accumulation_buffer = create_accumulation_buffer(&device, config.width, config.height);

        // Single u32 for atomicMax — tracks max density for per-image normalization
        let max_density_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("max_density"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // CDF buffer (256 floats) — needed by render bind group at binding 6
        let cdf_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cdf"),
            size: 256 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bind_group_a = create_render_bind_group(
            &device,
            &bind_group_layout,
            &uniform_buffer,
            &frame_b,
            &sampler,
            &accumulation_buffer,
            &crossfade_view,
            &max_density_buffer,
            &cdf_buffer,
        );
        let bind_group_b = create_render_bind_group(
            &device,
            &bind_group_layout,
            &uniform_buffer,
            &frame_a,
            &sampler,
            &accumulation_buffer,
            &crossfade_view,
            &max_density_buffer,
            &cdf_buffer,
        );

        let compute_bind_group = create_compute_bind_group(
            &device,
            &compute_bind_group_layout,
            &histogram_buffer,
            &uniform_buffer,
            &transform_buffer,
            &palette_view,
            &palette_sampler,
            &point_state_buffer,
        );

        // 16 bytes: vec2f resolution + f32 decay + f32 pad
        let accumulation_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accumulation_uniforms"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let accumulation_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("accumulation"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let accumulation_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("accumulation"),
                bind_group_layouts: &[&accumulation_bind_group_layout],
                immediate_size: 0,
            });

        let accumulation_src = load_accumulation_source();
        let accumulation_pipeline =
            create_accumulation_pipeline(&device, &accumulation_pipeline_layout, &accumulation_src);

        let accumulation_bind_group = create_accumulation_bind_group(
            &device,
            &accumulation_bind_group_layout,
            &histogram_buffer,
            &accumulation_buffer,
            &accumulation_uniform_buffer,
            &max_density_buffer,
        );

        // ── Histogram CDF pipelines ──
        let hist_bins_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hist_bins"),
            size: 256 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let histogram_cdf_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("histogram_cdf_uniforms"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let histogram_cdf_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("histogram_cdf"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let histogram_cdf_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("histogram_cdf"),
                bind_group_layouts: &[&histogram_cdf_bind_group_layout],
                immediate_size: 0,
            });

        let histogram_cdf_src = fs::read_to_string(resource_dir().join("histogram_cdf.wgsl"))
            .unwrap_or_else(|_| include_str!("../histogram_cdf.wgsl").to_string());
        let histogram_cdf_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("histogram_cdf"),
            source: wgpu::ShaderSource::Wgsl(histogram_cdf_src.into()),
        });
        let histogram_cdf_bin_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("histogram_cdf_bin"),
                layout: Some(&histogram_cdf_pipeline_layout),
                module: &histogram_cdf_module,
                entry_point: Some("bin_densities"),
                compilation_options: Default::default(),
                cache: None,
            });
        let histogram_cdf_sum_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("histogram_cdf_sum"),
                layout: Some(&histogram_cdf_pipeline_layout),
                module: &histogram_cdf_module,
                entry_point: Some("prefix_sum"),
                compilation_options: Default::default(),
                cache: None,
            });

        let histogram_cdf_bind_group = create_histogram_cdf_bind_group(
            &device,
            &histogram_cdf_bind_group_layout,
            &accumulation_buffer,
            &hist_bins_buffer,
            &cdf_buffer,
            &histogram_cdf_uniform_buffer,
        );

        Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            bind_group_layout,
            bind_group_a,
            bind_group_b,
            uniform_buffer,
            frame_a_tex,
            frame_a,
            frame_b_tex,
            frame_b,
            ping: true,
            pipeline_layout,
            sampler,
            crossfade_tex,
            crossfade_view,
            compute_pipeline,
            compute_bind_group_layout,
            compute_bind_group,
            compute_pipeline_layout,
            histogram_buffer,
            transform_buffer,
            palette_texture,
            palette_view,
            palette_sampler,
            workgroups: 256, // default samples_per_frame
            render_frame_count: 0,
            accumulation_pipeline,
            accumulation_bind_group_layout,
            accumulation_bind_group,
            accumulation_buffer,
            accumulation_uniform_buffer,
            max_density_buffer,
            point_state_buffer,
            histogram_cdf_bin_pipeline,
            histogram_cdf_sum_pipeline,
            histogram_cdf_bind_group_layout,
            histogram_cdf_bind_group,
            hist_bins_buffer,
            cdf_buffer,
            histogram_cdf_uniform_buffer,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);

        let (a_tex, a, b_tex, b) = create_frame_textures(&self.device, w, h, self.config.format);
        let (cf_tex, cf_view) = create_crossfade_texture(&self.device, w, h, self.config.format);
        self.frame_a_tex = a_tex;
        self.frame_a = a;
        self.frame_b_tex = b_tex;
        self.frame_b = b;
        self.crossfade_tex = cf_tex;
        self.crossfade_view = cf_view;

        // Recreate histogram for new dimensions
        self.histogram_buffer = create_histogram_buffer(&self.device, w, h);
        self.accumulation_buffer = create_accumulation_buffer(&self.device, w, h);

        self.rebuild_bind_groups();
    }

    fn resize_transform_buffer(&mut self, num_transforms: usize) {
        let size = (num_transforms.max(1) * PARAMS_PER_XF * 4) as u64;
        self.transform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transforms"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.rebuild_bind_groups();
    }

    fn rebuild_bind_groups(&mut self) {
        self.bind_group_a = create_render_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.uniform_buffer,
            &self.frame_b,
            &self.sampler,
            &self.accumulation_buffer,
            &self.crossfade_view,
            &self.max_density_buffer,
            &self.cdf_buffer,
        );
        self.bind_group_b = create_render_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.uniform_buffer,
            &self.frame_a,
            &self.sampler,
            &self.accumulation_buffer,
            &self.crossfade_view,
            &self.max_density_buffer,
            &self.cdf_buffer,
        );
        self.compute_bind_group = create_compute_bind_group(
            &self.device,
            &self.compute_bind_group_layout,
            &self.histogram_buffer,
            &self.uniform_buffer,
            &self.transform_buffer,
            &self.palette_view,
            &self.palette_sampler,
            &self.point_state_buffer,
        );
        self.accumulation_bind_group = create_accumulation_bind_group(
            &self.device,
            &self.accumulation_bind_group_layout,
            &self.histogram_buffer,
            &self.accumulation_buffer,
            &self.accumulation_uniform_buffer,
            &self.max_density_buffer,
        );
        self.histogram_cdf_bind_group = create_histogram_cdf_bind_group(
            &self.device,
            &self.histogram_cdf_bind_group_layout,
            &self.accumulation_buffer,
            &self.hist_bins_buffer,
            &self.cdf_buffer,
            &self.histogram_cdf_uniform_buffer,
        );
    }

    /// Runs compute + display passes, returns the surface texture for overlay rendering.
    /// Caller must call `.present()` on the returned texture after any overlay passes.
    fn render(&mut self, run_compute: bool) -> Option<wgpu::SurfaceTexture> {
        // Phase 1: Submit compute work BEFORE acquiring swapchain image.
        let mut compute_encoder = self.device.create_command_encoder(&Default::default());
        compute_encoder.clear_buffer(&self.histogram_buffer, 0, None);
        if run_compute {
            let mut cpass = compute_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flame"),
                ..Default::default()
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups(self.workgroups, 1, 1);
        }
        compute_encoder.clear_buffer(&self.max_density_buffer, 0, None);
        {
            let mut apass = compute_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("accumulation"),
                ..Default::default()
            });
            apass.set_pipeline(&self.accumulation_pipeline);
            apass.set_bind_group(0, &self.accumulation_bind_group, &[]);
            let wg_x = self.config.width.div_ceil(16);
            let wg_y = self.config.height.div_ceil(16);
            apass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        compute_encoder.clear_buffer(&self.hist_bins_buffer, 0, None);
        {
            let mut hpass = compute_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("histogram_bin"),
                ..Default::default()
            });
            hpass.set_pipeline(&self.histogram_cdf_bin_pipeline);
            hpass.set_bind_group(0, &self.histogram_cdf_bind_group, &[]);
            let wg_x = self.config.width.div_ceil(16);
            let wg_y = self.config.height.div_ceil(16);
            hpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        {
            let mut hpass = compute_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("histogram_cdf"),
                ..Default::default()
            });
            hpass.set_pipeline(&self.histogram_cdf_sum_pipeline);
            hpass.set_bind_group(0, &self.histogram_cdf_bind_group, &[]);
            hpass.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit(std::iter::once(compute_encoder.finish()));

        // Phase 2: Acquire swapchain image (may block on vsync)
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                return None;
            }
        };
        let screen_view = frame.texture.create_view(&Default::default());

        let (target_view, bind_group) = if self.ping {
            (&self.frame_a, &self.bind_group_a)
        } else {
            (&self.frame_b, &self.bind_group_b)
        };

        // Phase 3: Fragment pass (reads accumulation buffer, writes to screen)
        let mut display_encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = display_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("feedback"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = display_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &screen_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(display_encoder.finish()));
        self.ping = !self.ping;
        self.render_frame_count += 1;
        Some(frame)
    }
}

// ── HUD Panel Helpers ──

/// Semi-transparent dark background frame for HUD panels.
fn hud_frame(opacity: f32) -> egui::Frame {
    let alpha = (210.0 * opacity) as u8;
    egui::Frame::NONE
        .fill(egui::Color32::from_rgba_unmultiplied(0, 0, 0, alpha))
        .corner_radius(6.0)
        .inner_margin(egui::Margin::same(8))
}

/// Draw a horizontal signal bar: label | colored fill | numeric value.
fn signal_bar(
    ui: &mut egui::Ui,
    label: &str,
    value: f32,
    color: egui::Color32,
    max: f32,
    opacity: f32,
) {
    let dim = fade_color(egui::Color32::from_rgb(190, 190, 190), opacity);
    let bg = fade_color(egui::Color32::from_rgb(34, 34, 34), opacity);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).size(11.0).color(dim));
        let (rect, _) = ui.allocate_exact_size(egui::vec2(100.0, 4.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 2.0, bg);
        let fill_frac = (value / max).clamp(0.0, 1.0);
        let fill_rect = egui::Rect::from_min_size(
            rect.min,
            egui::vec2(fill_frac * rect.width(), rect.height()),
        );
        ui.painter()
            .rect_filled(fill_rect, 2.0, fade_color(color, opacity));
        ui.label(
            egui::RichText::new(format!("{:.2}", value))
                .size(11.0)
                .color(dim),
        );
    });
}

/// Draw a bipolar signal bar (center-zero): fills left for negative, right for positive.
fn bipolar_bar(ui: &mut egui::Ui, label: &str, value: f32, color: egui::Color32, opacity: f32) {
    let dim = fade_color(egui::Color32::from_rgb(190, 190, 190), opacity);
    let bg = fade_color(egui::Color32::from_rgb(34, 34, 34), opacity);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).size(11.0).color(dim));
        let (rect, _) = ui.allocate_exact_size(egui::vec2(100.0, 4.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 2.0, bg);
        let center_x = rect.min.x + rect.width() * 0.5;
        let frac = value.clamp(-1.0, 1.0);
        let fill_rect = if frac >= 0.0 {
            egui::Rect::from_min_max(
                egui::pos2(center_x, rect.min.y),
                egui::pos2(center_x + frac * rect.width() * 0.5, rect.max.y),
            )
        } else {
            egui::Rect::from_min_max(
                egui::pos2(center_x + frac * rect.width() * 0.5, rect.min.y),
                egui::pos2(center_x, rect.max.y),
            )
        };
        ui.painter()
            .rect_filled(fill_rect, 2.0, fade_color(color, opacity));
        ui.label(
            egui::RichText::new(format!("{:+.2}", value))
                .size(11.0)
                .color(dim),
        );
    });
}

/// Morph progress color: red→yellow→green based on completion.
fn morph_color(completion: f32) -> egui::Color32 {
    if completion >= 1.0 {
        egui::Color32::from_rgb(68, 238, 68) // green
    } else if completion >= 0.5 {
        egui::Color32::from_rgb(238, 238, 68) // yellow
    } else {
        egui::Color32::from_rgb(238, 68, 68) // red
    }
}

/// Apply opacity to a Color32.
fn fade_color(c: egui::Color32, opacity: f32) -> egui::Color32 {
    let [r, g, b, a] = c.to_array();
    egui::Color32::from_rgba_unmultiplied(r, g, b, (a as f32 * opacity) as u8)
}

/// Create a HUD window with a minimal title-bar drag handle.
/// When `override_pos` is Some, the window is forced to that position for one frame.
fn hud_window(
    name: &str,
    default_pos: egui::Pos2,
    opacity: f32,
    override_pos: Option<egui::Pos2>,
) -> egui::Window<'static> {
    let mut win = egui::Window::new(
        egui::RichText::new("≡")
            .size(8.0)
            .color(fade_color(egui::Color32::from_rgb(100, 100, 100), opacity)),
    )
    .id(egui::Id::new(name))
    .collapsible(false)
    .resizable(false)
    .default_pos(default_pos)
    .frame(hud_frame(opacity));
    if let Some(pos) = override_pos {
        win = win.current_pos(pos);
    }
    win
}

/// Disable text selection inside a UI (so dragging works on text).
fn disable_text_selection(ui: &mut egui::Ui) {
    ui.style_mut().interaction.selectable_labels = false;
}

/// Top-left: Identity panel — FPS, transform count, mutation info.
fn hud_panel_identity(
    ctx: &egui::Context,
    hud: &HudFrameData,
    opacity: f32,
    override_pos: Option<egui::Pos2>,
) {
    let default_pos = egui::pos2(10.0, 10.0);
    hud_window("hud_identity", default_pos, opacity, override_pos).show(ctx, |ui| {
        disable_text_selection(ui);
        ui.label(
            egui::RichText::new(format!("{:.0} fps", hud.fps))
                .color(fade_color(egui::Color32::from_rgb(119, 255, 119), opacity))
                .size(16.0),
        );
        let dim = fade_color(egui::Color32::from_rgb(210, 210, 210), opacity);
        ui.label(
            egui::RichText::new(format!("{} transforms", hud.num_transforms))
                .color(dim)
                .size(12.0),
        );
        ui.label(
            egui::RichText::new(format!(
                "cd:{:.0}/{:.0}s",
                hud.time_since_mutation, hud.cooldown
            ))
            .color(dim)
            .size(12.0),
        );
    });
}

/// Top-right: Progress panel — mutation progress bar, cooldown, per-transform morph bars.
fn hud_panel_progress(
    ctx: &egui::Context,
    hud: &HudFrameData,
    screen_w: f32,
    opacity: f32,
    override_pos: Option<egui::Pos2>,
) {
    let default_pos = egui::pos2(screen_w - 230.0, 10.0);
    hud_window("hud_progress", default_pos, opacity, override_pos).show(ctx, |ui| {
        disable_text_selection(ui);
        let dim = fade_color(egui::Color32::from_rgb(190, 190, 190), opacity);
        let bg = fade_color(egui::Color32::from_rgb(34, 34, 34), opacity);

        // Next evolve — signal-driven trigger bar
        ui.label(
            egui::RichText::new("next evolve (signal)")
                .size(11.0)
                .color(dim),
        );
        let (rect, _) = ui.allocate_exact_size(egui::vec2(180.0, 6.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 3.0, bg);
        let fill_frac = hud.mutation_accum.clamp(0.0, 1.0);
        let gate_met = hud.time_since_mutation >= 10.0;
        let bar_color = fade_color(
            if fill_frac >= 1.0 && gate_met {
                egui::Color32::from_rgb(68, 238, 68) // green = ready to fire
            } else if fill_frac >= 1.0 {
                egui::Color32::from_rgb(238, 238, 68) // yellow = full but gated
            } else {
                egui::Color32::from_rgb(238, 153, 34) // orange = filling
            },
            opacity,
        );
        let fill_rect = egui::Rect::from_min_size(
            rect.min,
            egui::vec2(fill_frac * rect.width(), rect.height()),
        );
        ui.painter().rect_filled(fill_rect, 3.0, bar_color);
        let gate_label = if gate_met { "ready" } else { "wait" };
        ui.label(
            egui::RichText::new(format!(
                "{:.0}% gate:{} cd:{:.0}/{:.0}s",
                fill_frac * 100.0,
                gate_label,
                hud.time_since_mutation,
                hud.cooldown,
            ))
            .size(8.0)
            .color(dim),
        );

        ui.add_space(4.0);
        ui.label(egui::RichText::new("morph").size(11.0).color(dim));

        // Per-transform morph bars
        for i in 0..hud.num_transforms.min(12) {
            let rate = hud.morph_xf_rates[i];
            let completion = (hud.morph_progress * rate).min(1.0);
            let bar_color = fade_color(morph_color(completion), opacity);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("{i}")).size(8.0).color(dim));
                let (bar_rect, _) =
                    ui.allocate_exact_size(egui::vec2(150.0, 3.0), egui::Sense::hover());
                ui.painter().rect_filled(bar_rect, 1.5, bg);
                let fill = egui::Rect::from_min_size(
                    bar_rect.min,
                    egui::vec2(completion * bar_rect.width(), bar_rect.height()),
                );
                ui.painter().rect_filled(fill, 1.5, bar_color);
            });
        }
    });
}

/// Left side: Audio signals panel.
fn hud_panel_audio(
    ctx: &egui::Context,
    hud: &HudFrameData,
    opacity: f32,
    override_pos: Option<egui::Pos2>,
) {
    let default_pos = egui::pos2(10.0, 80.0);
    hud_window("hud_audio", default_pos, opacity, override_pos).show(ctx, |ui| {
        disable_text_selection(ui);
        let dim = fade_color(egui::Color32::from_rgb(190, 190, 190), opacity);
        ui.label(egui::RichText::new("audio").size(12.0).color(dim));
        signal_bar(
            ui,
            "bass ",
            hud.audio_bass,
            egui::Color32::from_rgb(238, 68, 68),
            1.0,
            opacity,
        );
        signal_bar(
            ui,
            "mids ",
            hud.audio_mids,
            egui::Color32::from_rgb(238, 136, 68),
            1.0,
            opacity,
        );
        signal_bar(
            ui,
            "highs",
            hud.audio_highs,
            egui::Color32::from_rgb(238, 170, 68),
            1.0,
            opacity,
        );
        signal_bar(
            ui,
            "enrgy",
            hud.audio_energy,
            egui::Color32::from_rgb(68, 238, 136),
            1.0,
            opacity,
        );
        // Beat bar — glow when > 0.8
        let beat_color = if hud.audio_beat > 0.8 {
            egui::Color32::from_rgb(255, 102, 102)
        } else {
            egui::Color32::from_rgb(238, 68, 68)
        };
        signal_bar(ui, "beat ", hud.audio_beat, beat_color, 1.0, opacity);
        signal_bar(
            ui,
            "b.acc",
            hud.audio_beat_accum,
            egui::Color32::from_rgb(136, 68, 238),
            1.0,
            opacity,
        );
        signal_bar(
            ui,
            "chng ",
            hud.audio_change,
            egui::Color32::from_rgb(255, 136, 68),
            1.0,
            opacity,
        );
    });
}

/// Left side: Time signals panel (below audio).
fn hud_panel_time(
    ctx: &egui::Context,
    hud: &HudFrameData,
    opacity: f32,
    override_pos: Option<egui::Pos2>,
) {
    let default_pos = egui::pos2(10.0, 300.0);
    hud_window("hud_time", default_pos, opacity, override_pos).show(ctx, |ui| {
        disable_text_selection(ui);
        let dim = fade_color(egui::Color32::from_rgb(190, 190, 190), opacity);
        let bipolar_color = egui::Color32::from_rgb(68, 136, 170);
        ui.label(egui::RichText::new("time").size(12.0).color(dim));
        // Bipolar signals (-1 to 1): slow, med, fast, noise, drift, flutter
        bipolar_bar(ui, "slow ", hud.time_slow, bipolar_color, opacity);
        bipolar_bar(ui, "med  ", hud.time_med, bipolar_color, opacity);
        bipolar_bar(ui, "fast ", hud.time_fast, bipolar_color, opacity);
        bipolar_bar(ui, "noise", hud.time_noise, bipolar_color, opacity);
        bipolar_bar(ui, "drift", hud.time_drift, bipolar_color, opacity);
        bipolar_bar(ui, "flutr", hud.time_flutter, bipolar_color, opacity);
        // Walk: monotonically growing, show capped bar + numeric value
        signal_bar(
            ui,
            "walk ",
            hud.time_walk.abs(),
            egui::Color32::from_rgb(136, 68, 170),
            10.0,
            opacity,
        );
        // Envelope: 0→1 range
        signal_bar(
            ui,
            "envlp",
            hud.time_envelope,
            egui::Color32::from_rgb(170, 136, 68),
            1.0,
            opacity,
        );
    });
}

/// Variation names for tooltips (indices 0-25).
const VARIATION_NAMES: [&str; 26] = [
    "linear",
    "sinusoidal",
    "spherical",
    "swirl",
    "horseshoe",
    "handkerchief",
    "julia",
    "polar",
    "disc",
    "rings",
    "bubble",
    "fisheye",
    "exponential",
    "spiral",
    "diamond",
    "bent",
    "waves",
    "popcorn",
    "fan",
    "eyefish",
    "cross",
    "tangent",
    "cosine",
    "blob",
    "noise",
    "curl",
];

/// Draw a tiny distinctive icon for a variation type.
fn draw_variation_icon(ui: &mut egui::Ui, var_idx: u8, size: f32, opacity: f32) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    let p = ui.painter();
    let c = rect.center();
    let r = size * 0.4;
    let stroke = egui::Stroke {
        width: 1.0,
        color: fade_color(egui::Color32::from_rgb(200, 200, 200), opacity),
    };
    let fill = fade_color(egui::Color32::from_rgb(200, 200, 200), opacity);
    draw_variation_shape(p, var_idx, c, r, stroke, fill);
    response
}

/// Draw the shape for a given variation index at the given center and radius.
fn draw_variation_shape(
    p: &egui::Painter,
    var_idx: u8,
    c: egui::Pos2,
    r: f32,
    stroke: egui::Stroke,
    fill: egui::Color32,
) {
    match var_idx {
        0 => {
            // linear: diagonal line /
            p.line_segment(
                [egui::pos2(c.x - r, c.y + r), egui::pos2(c.x + r, c.y - r)],
                stroke,
            );
        }
        1 => {
            // sinusoidal: sine wave ~
            let pts: Vec<egui::Pos2> = (0..=8)
                .map(|i| {
                    let t = i as f32 / 8.0;
                    egui::pos2(
                        c.x - r + t * 2.0 * r,
                        c.y - (t * std::f32::consts::TAU).sin() * r * 0.5,
                    )
                })
                .collect();
            p.line(pts, stroke);
        }
        2 => {
            // spherical: circle with center dot
            p.circle_stroke(c, r, stroke);
            p.circle_filled(c, 1.5, fill);
        }
        3 => {
            // swirl: spiral curve
            let pts: Vec<egui::Pos2> = (0..=16)
                .map(|i| {
                    let t = i as f32 / 16.0;
                    let angle = t * std::f32::consts::TAU * 1.5;
                    let rad = t * r;
                    egui::pos2(c.x + rad * angle.cos(), c.y + rad * angle.sin())
                })
                .collect();
            p.line(pts, stroke);
        }
        4 => {
            // horseshoe: U shape
            let pts: Vec<egui::Pos2> = (0..=8)
                .map(|i| {
                    let t = i as f32 / 8.0;
                    let angle = std::f32::consts::PI * t;
                    egui::pos2(c.x + r * angle.cos(), c.y + r * angle.sin().abs())
                })
                .collect();
            p.line(pts, stroke);
        }
        5 => {
            // handkerchief: diamond outline
            let d = r;
            let pts = vec![
                egui::pos2(c.x, c.y - d),
                egui::pos2(c.x + d, c.y),
                egui::pos2(c.x, c.y + d),
                egui::pos2(c.x - d, c.y),
                egui::pos2(c.x, c.y - d),
            ];
            p.line(pts, stroke);
        }
        6 => {
            // julia: two overlapping circles
            p.circle_stroke(egui::pos2(c.x - r * 0.3, c.y), r * 0.6, stroke);
            p.circle_stroke(egui::pos2(c.x + r * 0.3, c.y), r * 0.6, stroke);
        }
        7 => {
            // polar: crosshair + circle
            p.circle_stroke(c, r * 0.7, stroke);
            p.line_segment([egui::pos2(c.x - r, c.y), egui::pos2(c.x + r, c.y)], stroke);
            p.line_segment([egui::pos2(c.x, c.y - r), egui::pos2(c.x, c.y + r)], stroke);
        }
        8 => {
            // disc: circle with vertical line
            p.circle_stroke(c, r, stroke);
            p.line_segment([egui::pos2(c.x, c.y - r), egui::pos2(c.x, c.y + r)], stroke);
        }
        9 => {
            // rings: concentric circles
            p.circle_stroke(c, r, stroke);
            p.circle_stroke(c, r * 0.5, stroke);
        }
        10 => {
            // bubble: dashed circle (approx with arcs)
            for i in 0..6 {
                let a0 = i as f32 * std::f32::consts::TAU / 6.0;
                let a1 = a0 + std::f32::consts::TAU / 12.0;
                p.line_segment(
                    [
                        egui::pos2(c.x + r * a0.cos(), c.y + r * a0.sin()),
                        egui::pos2(c.x + r * a1.cos(), c.y + r * a1.sin()),
                    ],
                    stroke,
                );
            }
        }
        11 => {
            // fisheye: ellipse with dot
            let pts: Vec<egui::Pos2> = (0..=12)
                .map(|i| {
                    let a = i as f32 * std::f32::consts::TAU / 12.0;
                    egui::pos2(c.x + r * a.cos(), c.y + r * 0.6 * a.sin())
                })
                .collect();
            p.line(pts, stroke);
            p.circle_filled(c, 1.5, fill);
        }
        12 => {
            // exponential: exponential curve
            let pts: Vec<egui::Pos2> = (0..=8)
                .map(|i| {
                    let t = i as f32 / 8.0;
                    let y = (t * 3.0).exp() / (3.0_f32).exp();
                    egui::pos2(c.x - r + t * 2.0 * r, c.y + r - y * 2.0 * r)
                })
                .collect();
            p.line(pts, stroke);
        }
        13 => {
            // spiral
            let pts: Vec<egui::Pos2> = (0..=20)
                .map(|i| {
                    let t = i as f32 / 20.0;
                    let angle = t * std::f32::consts::TAU * 2.0;
                    let rad = t * r;
                    egui::pos2(c.x + rad * angle.cos(), c.y + rad * angle.sin())
                })
                .collect();
            p.line(pts, stroke);
        }
        14 => {
            // diamond: filled diamond
            let d = r * 0.8;
            let pts = vec![
                egui::pos2(c.x, c.y - d),
                egui::pos2(c.x + d, c.y),
                egui::pos2(c.x, c.y + d),
                egui::pos2(c.x - d, c.y),
            ];
            p.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
        15 => {
            // bent: bent line
            p.line_segment([egui::pos2(c.x - r, c.y), egui::pos2(c.x, c.y)], stroke);
            p.line_segment([egui::pos2(c.x, c.y), egui::pos2(c.x + r, c.y - r)], stroke);
        }
        16 => {
            // waves: double wave
            for offset in [-r * 0.3, r * 0.3] {
                let pts: Vec<egui::Pos2> = (0..=8)
                    .map(|i| {
                        let t = i as f32 / 8.0;
                        egui::pos2(
                            c.x - r + t * 2.0 * r,
                            c.y + offset - (t * std::f32::consts::TAU).sin() * r * 0.3,
                        )
                    })
                    .collect();
                p.line(pts, stroke);
            }
        }
        17 => {
            // popcorn: scattered dots
            for (dx, dy) in [
                (-0.5, -0.5),
                (0.3, -0.2),
                (-0.2, 0.4),
                (0.5, 0.3),
                (0.0, -0.6),
            ] {
                p.circle_filled(egui::pos2(c.x + dx * r, c.y + dy * r), 1.0, fill);
            }
        }
        18 => {
            // fan: triangle
            let pts = vec![
                egui::pos2(c.x, c.y - r),
                egui::pos2(c.x + r, c.y + r),
                egui::pos2(c.x - r, c.y + r),
                egui::pos2(c.x, c.y - r),
            ];
            p.line(pts, stroke);
        }
        19 => {
            // eyefish: eye shape
            let top: Vec<egui::Pos2> = (0..=8)
                .map(|i| {
                    let t = i as f32 / 8.0;
                    let x = c.x - r + t * 2.0 * r;
                    let y = c.y - (t * std::f32::consts::PI).sin() * r * 0.6;
                    egui::pos2(x, y)
                })
                .collect();
            let bot: Vec<egui::Pos2> = (0..=8)
                .map(|i| {
                    let t = i as f32 / 8.0;
                    let x = c.x - r + t * 2.0 * r;
                    let y = c.y + (t * std::f32::consts::PI).sin() * r * 0.6;
                    egui::pos2(x, y)
                })
                .collect();
            p.line(top, stroke);
            p.line(bot, stroke);
        }
        20 => {
            // cross: + cross
            p.line_segment([egui::pos2(c.x - r, c.y), egui::pos2(c.x + r, c.y)], stroke);
            p.line_segment([egui::pos2(c.x, c.y - r), egui::pos2(c.x, c.y + r)], stroke);
        }
        21 => {
            // tangent: vertical asymptote
            p.line_segment([egui::pos2(c.x, c.y - r), egui::pos2(c.x, c.y + r)], stroke);
            let pts: Vec<egui::Pos2> = (0..=8)
                .map(|i| {
                    let t = (i as f32 / 8.0 - 0.5) * 2.0;
                    egui::pos2(c.x + t * r, c.y - t.atan() * r * 0.8)
                })
                .collect();
            p.line(pts, stroke);
        }
        22 => {
            // cosine: cosine curve
            let pts: Vec<egui::Pos2> = (0..=8)
                .map(|i| {
                    let t = i as f32 / 8.0;
                    egui::pos2(
                        c.x - r + t * 2.0 * r,
                        c.y - (t * std::f32::consts::TAU).cos() * r * 0.5,
                    )
                })
                .collect();
            p.line(pts, stroke);
        }
        23 => {
            // blob: filled ellipse
            let pts: Vec<egui::Pos2> = (0..=12)
                .map(|i| {
                    let a = i as f32 * std::f32::consts::TAU / 12.0;
                    egui::pos2(c.x + r * a.cos(), c.y + r * 0.6 * a.sin())
                })
                .collect();
            p.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
        24 => {
            // noise: random dots (deterministic positions)
            for (dx, dy) in [
                (-0.7, 0.2),
                (0.4, -0.6),
                (-0.3, -0.4),
                (0.6, 0.5),
                (0.1, 0.1),
                (-0.5, 0.6),
            ] {
                p.circle_filled(egui::pos2(c.x + dx * r, c.y + dy * r), 1.0, fill);
            }
        }
        _ => {
            // curl (25) or unknown: S-curve
            let pts: Vec<egui::Pos2> = (0..=10)
                .map(|i| {
                    let t = i as f32 / 10.0 * 2.0 - 1.0;
                    egui::pos2(c.x + t * r, c.y - (t * 1.5).tanh() * r)
                })
                .collect();
            p.line(pts, stroke);
        }
    }
}

/// Right side: Transforms panel (below progress).
fn hud_panel_transforms(
    ctx: &egui::Context,
    hud: &HudFrameData,
    screen_w: f32,
    opacity: f32,
    override_pos: Option<egui::Pos2>,
) {
    let default_pos = egui::pos2(screen_w - 230.0, 260.0);
    hud_window("hud_transforms", default_pos, opacity, override_pos).show(ctx, |ui| {
        disable_text_selection(ui);
        let dim = fade_color(egui::Color32::from_rgb(190, 190, 190), opacity);
        ui.label(egui::RichText::new("transforms").size(12.0).color(dim));
        for i in 0..hud.num_transforms.min(12) {
            let w = hud.transform_weights[i];
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("xf{i:>2} w:{w:.3}"))
                        .size(11.0)
                        .color(dim)
                        .family(egui::FontFamily::Monospace),
                );
                // Show variation icons for non-zero variations
                for (v, &vw) in hud.transform_variations[i].iter().enumerate() {
                    if vw.abs() > 0.001 {
                        let resp = draw_variation_icon(ui, v as u8, 12.0, opacity);
                        resp.on_hover_text(VARIATION_NAMES[v]);
                    }
                }
            });
        }
    });
}

/// Bottom: Hotkey bar — centered row of key+label pairs.
fn hud_panel_hotkeys(
    ctx: &egui::Context,
    screen_w: f32,
    screen_h: f32,
    opacity: f32,
    override_pos: Option<egui::Pos2>,
) {
    let default_pos = egui::pos2(screen_w * 0.5 - 250.0, screen_h - 40.0);
    let win = hud_window("hud_hotkeys", default_pos, opacity, override_pos)
        .frame(hud_frame(opacity).inner_margin(egui::Margin::symmetric(10, 4)));
    win.show(ctx, |ui| {
        disable_text_selection(ui);
        let key_color = fade_color(egui::Color32::from_rgb(187, 187, 187), opacity);
        let label_color = fade_color(egui::Color32::from_rgb(102, 102, 102), opacity);
        ui.horizontal(|ui| {
            for (key, desc) in [
                ("Space", "evolve"),
                ("\u{2191}/\u{2193}", "vote"),
                ("S", "save"),
                ("F", "flame"),
                ("A", "audio"),
                ("C", "config"),
                ("I", "info"),
                ("Esc", "quit"),
            ] {
                ui.label(egui::RichText::new(key).size(11.0).color(key_color));
                ui.label(egui::RichText::new(desc).size(11.0).color(label_color));
                ui.add_space(6.0);
            }
        });
    });
}

// ── Config Panel ──

/// Config panel actions.
struct ConfigAction {
    changed: bool,
    save: bool,
    reset: bool,
    cancel: bool,
}

/// Config panel state passed in from the App.
struct ConfigPanelState<'a> {
    weights: &'a mut crate::weights::Weights,
    tab: &'a mut usize,
    reposition_panels: &'a mut bool,
    cached_audio_devices: &'a [String],
    selected_audio_device: &'a mut String,
    refresh_audio_devices: &'a mut bool,
}

/// Main config panel UI.
fn config_panel_ui(
    ctx: &egui::Context,
    state: &mut ConfigPanelState<'_>,
    screen_w: f32,
    screen_h: f32,
) -> ConfigAction {
    let mut changed = false;
    let mut save = false;
    let mut reset = false;
    let mut cancel = false;
    let panel_w = 360.0;
    let panel_h = 480.0;
    let x = (screen_w - panel_w) * 0.5;
    let y = (screen_h - panel_h) * 0.5;

    egui::Window::new("Config")
        .id(egui::Id::new("config_panel"))
        .collapsible(false)
        .resizable(false)
        .default_pos(egui::pos2(x, y))
        .frame(
            egui::Frame::NONE
                .fill(egui::Color32::from_rgba_premultiplied(15, 15, 15, 210))
                .corner_radius(8.0)
                .inner_margin(egui::Margin::same(12)),
        )
        .show(ctx, |ui| {
            ui.set_min_width(panel_w - 24.0);
            ui.label(
                egui::RichText::new("Config Panel")
                    .size(14.0)
                    .color(egui::Color32::WHITE),
            );
            ui.add_space(4.0);

            // Tab buttons
            ui.horizontal(|ui| {
                for (i, name) in ["Config", "Rendering", "Breeding", "Signals", "Audio"]
                    .iter()
                    .enumerate()
                {
                    let color = if *state.tab == i {
                        egui::Color32::from_rgb(100, 200, 255)
                    } else {
                        egui::Color32::from_rgb(140, 140, 140)
                    };
                    if ui
                        .add(egui::Button::new(
                            egui::RichText::new(*name).size(11.0).color(color),
                        ))
                        .clicked()
                    {
                        *state.tab = i;
                    }
                }
            });
            ui.add_space(4.0);

            // Tab content in a scroll area
            egui::ScrollArea::vertical()
                .max_height(380.0)
                .show(ui, |ui| match *state.tab {
                    0 => changed = config_tab_main(ui, &mut state.weights._config),
                    1 => changed = config_tab_rendering(ui, &mut state.weights._config),
                    2 => changed = config_tab_breeding(ui, &mut state.weights._config),
                    3 => changed = config_tab_signals(ui, state.weights),
                    4 => config_tab_audio(
                        ui,
                        state.cached_audio_devices,
                        state.selected_audio_device,
                        state.refresh_audio_devices,
                    ),
                    _ => {}
                });

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui
                    .add(egui::Button::new(
                        egui::RichText::new("Save")
                            .size(11.0)
                            .color(egui::Color32::from_rgb(100, 255, 100)),
                    ))
                    .clicked()
                {
                    save = true;
                }
                if ui
                    .add(egui::Button::new(
                        egui::RichText::new("Reset")
                            .size(11.0)
                            .color(egui::Color32::from_rgb(255, 200, 100)),
                    ))
                    .clicked()
                {
                    reset = true;
                }
                if ui
                    .add(egui::Button::new(
                        egui::RichText::new("Cancel")
                            .size(11.0)
                            .color(egui::Color32::from_rgb(200, 100, 100)),
                    ))
                    .clicked()
                {
                    cancel = true;
                }
                ui.add_space(8.0);
                if ui
                    .add(egui::Button::new(
                        egui::RichText::new("Reset positions")
                            .size(11.0)
                            .color(egui::Color32::from_rgb(180, 180, 255)),
                    ))
                    .clicked()
                {
                    *state.reposition_panels = true;
                }
            });
        });
    ConfigAction {
        changed,
        save,
        reset,
        cancel,
    }
}

/// Config tab: morph, mutation, motion, and accumulation sliders.
fn config_tab_main(ui: &mut egui::Ui, cfg: &mut crate::weights::RuntimeConfig) -> bool {
    let mut changed = false;
    changed |= config_slider_f32(ui, "morph_duration", &mut cfg.morph_duration, 1.0, 120.0);
    changed |= config_slider_f32(
        ui,
        "mutation_cooldown",
        &mut cfg.mutation_cooldown,
        1.0,
        300.0,
    );
    changed |= config_slider_f32(ui, "spin_speed_max", &mut cfg.spin_speed_max, 0.0, 2.0);
    changed |= config_slider_f32(ui, "position_drift", &mut cfg.position_drift, 0.0, 2.0);
    changed |= config_slider_f32(ui, "drift_speed", &mut cfg.drift_speed, 0.0, 10.0);
    changed |= config_slider_f32(ui, "trail", &mut cfg.trail, 0.0, 1.0);
    changed |= config_slider_f32(ui, "bloom_intensity", &mut cfg.bloom_intensity, 0.0, 0.5);
    changed |= config_slider_f32(
        ui,
        "accumulation_decay",
        &mut cfg.accumulation_decay,
        0.5,
        0.999,
    );
    changed |= config_slider_f32(
        ui,
        "mutation_accum_decay",
        &mut cfg.mutation_accum_decay,
        0.0,
        2.0,
    );
    changed
}

/// Rendering tab: tonemap, jitter, histogram, DoF, blur, gamma, bloom.
fn config_tab_rendering(ui: &mut egui::Ui, cfg: &mut crate::weights::RuntimeConfig) -> bool {
    let mut changed = false;
    // tonemap_mode as u32 selector
    let mut mode = cfg.tonemap_mode as usize;
    let labels = ["Reinhard (0)", "ACES (1)", "Filmic (2)"];
    ui.horizontal(|ui| {
        let dim = egui::Color32::from_rgb(210, 210, 210);
        ui.label(egui::RichText::new("tonemap_mode").size(12.0).color(dim));
        for (i, label) in labels.iter().enumerate() {
            let color = if mode == i {
                egui::Color32::from_rgb(100, 200, 255)
            } else {
                egui::Color32::from_rgb(100, 100, 100)
            };
            if ui
                .add(egui::Button::new(
                    egui::RichText::new(*label).size(11.0).color(color),
                ))
                .clicked()
            {
                mode = i;
                changed = true;
            }
        }
    });
    cfg.tonemap_mode = mode as u32;

    changed |= config_slider_f32(ui, "jitter_amount", &mut cfg.jitter_amount, 0.0, 1.0);
    changed |= config_slider_f32(
        ui,
        "histogram_equalization",
        &mut cfg.histogram_equalization,
        0.0,
        1.0,
    );
    changed |= config_slider_f32(ui, "dof_strength", &mut cfg.dof_strength, 0.0, 2.0);
    changed |= config_slider_f32(
        ui,
        "velocity_blur_max",
        &mut cfg.velocity_blur_max,
        0.0,
        50.0,
    );
    changed |= config_slider_f32(ui, "gamma", &mut cfg.gamma, 0.1, 2.0);
    changed |= config_slider_f32(ui, "bloom_radius", &mut cfg.bloom_radius, 0.0, 10.0);
    changed
}

/// Breeding tab: transform counts, parent biases, breeding distance.
fn config_tab_breeding(ui: &mut egui::Ui, cfg: &mut crate::weights::RuntimeConfig) -> bool {
    let mut changed = false;
    changed |= config_slider_u32(
        ui,
        "transform_count_min",
        &mut cfg.transform_count_min,
        1,
        20,
    );
    changed |= config_slider_u32(
        ui,
        "transform_count_max",
        &mut cfg.transform_count_max,
        1,
        20,
    );
    changed |= config_slider_f32(
        ui,
        "parent_current_bias",
        &mut cfg.parent_current_bias,
        0.0,
        1.0,
    );
    changed |= config_slider_f32(
        ui,
        "parent_voted_bias",
        &mut cfg.parent_voted_bias,
        0.0,
        1.0,
    );
    changed |= config_slider_f32(
        ui,
        "parent_saved_bias",
        &mut cfg.parent_saved_bias,
        0.0,
        1.0,
    );
    changed |= config_slider_f32(
        ui,
        "parent_random_bias",
        &mut cfg.parent_random_bias,
        0.0,
        1.0,
    );
    changed |= config_slider_u32(
        ui,
        "min_breeding_distance",
        &mut cfg.min_breeding_distance,
        0,
        20,
    );
    changed
}

/// Signals tab: show and edit audio/time signal weight mappings.
fn config_tab_signals(ui: &mut egui::Ui, weights: &mut crate::weights::Weights) -> bool {
    let mut changed = false;
    let dim = egui::Color32::from_rgb(210, 210, 210);

    // Signal groups in order matching Weights struct
    let signal_names = [
        "bass",
        "mids",
        "highs",
        "energy",
        "beat",
        "beat_accum",
        "change",
        "time",
        "time_slow",
        "time_med",
        "time_fast",
        "time_noise",
        "time_drift",
        "time_flutter",
        "time_walk",
        "time_envelope",
    ];

    for signal_name in &signal_names {
        let map = match *signal_name {
            "bass" => &mut weights.bass,
            "mids" => &mut weights.mids,
            "highs" => &mut weights.highs,
            "energy" => &mut weights.energy,
            "beat" => &mut weights.beat,
            "beat_accum" => &mut weights.beat_accum,
            "change" => &mut weights.change,
            "time" => &mut weights.time,
            "time_slow" => &mut weights.time_slow,
            "time_med" => &mut weights.time_med,
            "time_fast" => &mut weights.time_fast,
            "time_noise" => &mut weights.time_noise,
            "time_drift" => &mut weights.time_drift,
            "time_flutter" => &mut weights.time_flutter,
            "time_walk" => &mut weights.time_walk,
            "time_envelope" => &mut weights.time_envelope,
            _ => continue,
        };

        let header_text = if map.is_empty() {
            format!("{signal_name} (empty)")
        } else {
            format!("{signal_name} ({})", map.len())
        };

        egui::CollapsingHeader::new(egui::RichText::new(header_text).size(11.0).color(dim))
            .default_open(!map.is_empty())
            .show(ui, |ui| {
                let mut to_remove: Option<String> = None;
                // Sort keys for stable display order
                let mut keys: Vec<String> = map.keys().cloned().collect();
                keys.sort();
                for key in &keys {
                    if let Some(val) = map.get_mut(key) {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(key)
                                    .size(11.0)
                                    .color(dim)
                                    .family(egui::FontFamily::Monospace),
                            );
                            let before = *val;
                            ui.add(egui::DragValue::new(val).range(-10.0..=10.0).speed(0.01));
                            if (*val - before).abs() > f32::EPSILON {
                                changed = true;
                            }
                            if ui
                                .add(egui::Button::new(
                                    egui::RichText::new("X")
                                        .size(11.0)
                                        .color(egui::Color32::from_rgb(200, 100, 100)),
                                ))
                                .clicked()
                            {
                                to_remove = Some(key.clone());
                            }
                        });
                    }
                }
                if let Some(key) = to_remove {
                    map.remove(&key);
                    changed = true;
                }

                // Add new entry
                ui.horizontal(|ui| {
                    let add_id = egui::Id::new(format!("signals_add_{signal_name}"));
                    let mut new_param =
                        ui.data_mut(|d| d.get_temp::<String>(add_id).unwrap_or_default());
                    ui.label(egui::RichText::new("+").size(12.0).color(dim));
                    ui.add(
                        egui::TextEdit::singleline(&mut new_param)
                            .desired_width(120.0)
                            .hint_text("param name"),
                    );
                    if ui
                        .add(egui::Button::new(
                            egui::RichText::new("Add")
                                .size(11.0)
                                .color(egui::Color32::from_rgb(100, 200, 100)),
                        ))
                        .clicked()
                        && !new_param.trim().is_empty()
                        && !map.contains_key(new_param.trim())
                    {
                        map.insert(new_param.trim().to_string(), 0.0);
                        changed = true;
                        new_param.clear();
                    }
                    ui.data_mut(|d| d.insert_temp(add_id, new_param));
                });
            });
    }

    changed
}

/// Audio tab: dropdown device selector + refresh.
fn config_tab_audio(
    ui: &mut egui::Ui,
    cached_devices: &[String],
    selected_device: &mut String,
    refresh: &mut bool,
) {
    let dim = egui::Color32::from_rgb(210, 210, 210);
    ui.label(
        egui::RichText::new("Audio Device")
            .size(12.0)
            .color(egui::Color32::WHITE),
    );
    ui.add_space(4.0);

    // Build device options: System Audio first, then cached devices
    let system_audio = "System Audio (ScreenCaptureKit)".to_string();
    let mut all_devices = vec![system_audio.clone()];
    all_devices.extend_from_slice(cached_devices);

    // Default to system audio if nothing selected
    if selected_device.is_empty() {
        *selected_device = system_audio;
    }

    egui::ComboBox::from_label(egui::RichText::new("Device").size(12.0).color(dim))
        .width(280.0)
        .selected_text(
            egui::RichText::new(selected_device.as_str())
                .size(12.0)
                .color(egui::Color32::from_rgb(100, 200, 255)),
        )
        .show_ui(ui, |ui| {
            for device in &all_devices {
                ui.selectable_value(
                    selected_device,
                    device.clone(),
                    egui::RichText::new(device).size(12.0),
                );
            }
        });

    ui.add_space(4.0);
    if ui
        .add(egui::Button::new(
            egui::RichText::new("Refresh devices")
                .size(12.0)
                .color(egui::Color32::from_rgb(100, 200, 255)),
        ))
        .clicked()
    {
        *refresh = true;
    }

    ui.add_space(8.0);
    ui.label(
        egui::RichText::new("Device switching requires restart.")
            .size(11.0)
            .color(egui::Color32::from_rgb(180, 140, 80)),
    );
}

/// Helper: f32 slider with label. Returns true if value changed.
fn config_slider_f32(ui: &mut egui::Ui, label: &str, value: &mut f32, min: f32, max: f32) -> bool {
    let before = *value;
    ui.add(egui::Slider::new(value, min..=max).text(label));
    (*value - before).abs() > f32::EPSILON
}

/// Helper: u32 slider with label. Returns true if value changed.
fn config_slider_u32(ui: &mut egui::Ui, label: &str, value: &mut u32, min: u32, max: u32) -> bool {
    let before = *value;
    let mut v = *value as i32;
    ui.add(egui::Slider::new(&mut v, min as i32..=max as i32).text(label));
    *value = v as u32;
    *value != before
}

/// Enumerate audio devices once (used for caching — avoids per-frame cpal calls).
fn enumerate_audio_devices() -> Vec<String> {
    use cpal::traits::{DeviceTrait, HostTrait};
    let host = cpal::default_host();
    let mut names = Vec::new();

    names.push("Input Devices:".to_string());
    if let Ok(inputs) = host.input_devices() {
        for d in inputs {
            let name = d
                .description()
                .map(|desc| desc.name().to_string())
                .unwrap_or_else(|_| "???".into());
            names.push(format!("  {name}"));
        }
    }

    names.push(String::new());
    names.push("Output Devices (loopback):".to_string());
    if let Ok(outputs) = host.output_devices() {
        for d in outputs {
            let name = d
                .description()
                .map(|desc| desc.name().to_string())
                .unwrap_or_else(|_| "???".into());
            names.push(format!("  {name}"));
        }
    }

    names
}

// ── Render Thread ──

/// Compute HUD opacity based on time since last mouse move.
fn compute_hud_opacity(elapsed: f32, fade_delay: f32, fade_duration: f32) -> f32 {
    if elapsed < fade_delay {
        return 1.0;
    }
    let fade_elapsed = elapsed - fade_delay;
    if fade_duration <= 0.0 {
        return 0.0;
    }
    (1.0 - fade_elapsed / fade_duration).clamp(0.0, 1.0)
}

fn render_thread_loop(rx: mpsc::Receiver<RenderCommand>, mut gpu: Gpu) {
    // egui renderer lives on the render thread (needs GPU device/queue)
    let mut egui_renderer = egui_wgpu::Renderer::new(
        &gpu.device,
        gpu.config.format,
        egui_wgpu::RendererOptions::default(),
    );
    while let Ok(cmd) = rx.recv() {
        match cmd {
            RenderCommand::Render(data) => {
                if gpu.render_frame_count < 3 {
                    eprintln!(
                        "[debug] render thread got frame {}, wg={}, xf_len={}",
                        gpu.render_frame_count,
                        data.workgroups,
                        data.xf_params.len()
                    );
                }
                gpu.queue
                    .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&data.uniforms));
                gpu.queue.write_buffer(
                    &gpu.transform_buffer,
                    0,
                    bytemuck::cast_slice(&data.xf_params),
                );
                gpu.queue.write_buffer(
                    &gpu.accumulation_uniform_buffer,
                    0,
                    bytemuck::cast_slice(&data.accum_uniforms),
                );
                gpu.queue.write_buffer(
                    &gpu.histogram_cdf_uniform_buffer,
                    0,
                    bytemuck::cast_slice(&data.hist_cdf_uniforms),
                );
                gpu.workgroups = data.workgroups;

                // Run fractal compute + display passes; get surface for overlay
                if let Some(surface_texture) = gpu.render(data.run_compute) {
                    let screen_view = surface_texture.texture.create_view(&Default::default());

                    // egui overlay — execute pre-built paint jobs from main thread
                    let screen_descriptor = egui_wgpu::ScreenDescriptor {
                        size_in_pixels: [gpu.config.width, gpu.config.height],
                        pixels_per_point: data.egui_pixels_per_point,
                    };

                    for (id, image_delta) in &data.egui_textures_delta.set {
                        egui_renderer.update_texture(&gpu.device, &gpu.queue, *id, image_delta);
                    }

                    let mut egui_encoder = gpu.device.create_command_encoder(&Default::default());
                    let egui_cmd_bufs = egui_renderer.update_buffers(
                        &gpu.device,
                        &gpu.queue,
                        &mut egui_encoder,
                        &data.egui_primitives,
                        &screen_descriptor,
                    );

                    // Render egui overlay — LoadOp::Load preserves the fractal
                    {
                        let egui_pass =
                            egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("egui"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &screen_view,
                                    resolve_target: None,
                                    depth_slice: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                ..Default::default()
                            });
                        let mut egui_pass = egui_pass.forget_lifetime();
                        egui_renderer.render(
                            &mut egui_pass,
                            &data.egui_primitives,
                            &screen_descriptor,
                        );
                    }

                    let mut cmd_bufs: Vec<wgpu::CommandBuffer> = egui_cmd_bufs;
                    cmd_bufs.push(egui_encoder.finish());
                    gpu.queue.submit(cmd_bufs);
                    surface_texture.present();

                    for id in &data.egui_textures_delta.free {
                        egui_renderer.free_texture(id);
                    }
                }
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
                gpu.compute_pipeline =
                    create_compute_pipeline(&gpu.device, &gpu.compute_pipeline_layout, &src);
                eprintln!("[render-thread] compute shader reloaded");
            }
            RenderCommand::Shutdown => {
                eprintln!("[render-thread] shutdown");
                return;
            }
        }
    }
    eprintln!("[render-thread] channel closed");
}

// ── Helper Functions ──

/// Returns the directory where read-only resources (shaders, default config) live.
/// In a .app bundle this is Contents/Resources; in dev mode, the repo root.
pub fn resource_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe()
        && let Some(macos_dir) = exe.parent()
    {
        let resources = macos_dir.with_file_name("Resources");
        if resources.join("weights.json").exists() {
            return resources;
        }
    }
    // Dev mode: walk up from cwd looking for Cargo.toml
    let mut dir = std::env::current_dir().unwrap();
    loop {
        if dir.join("Cargo.toml").exists() {
            return dir;
        }
        if !dir.pop() {
            return std::env::current_dir().unwrap();
        }
    }
}

/// Returns the writable data directory for user content (genomes, votes, config overrides).
///
/// - Dev mode: same as resource_dir() (the repo root)
/// - Bundle mode: ~/Library/Application Support/Shader Playground/ (macOS)
///   or equivalent XDG dir on Linux, %APPDATA% on Windows.
///
/// On first launch, seeds resources from the bundle into the data dir.
pub fn data_dir() -> PathBuf {
    let res = resource_dir();
    // If resource_dir has a Cargo.toml, we're in dev mode — use it directly
    if res.join("Cargo.toml").exists() {
        return res;
    }
    // Bundle mode: use platform-appropriate app data directory
    let home = std::env::var("HOME").ok().map(PathBuf::from);
    let app_data = if cfg!(target_os = "macos") {
        home.as_ref()
            .map(|h| h.join("Library/Application Support/Shader Playground"))
    } else if cfg!(target_os = "windows") {
        std::env::var("APPDATA")
            .ok()
            .map(|a| PathBuf::from(a).join("Shader Playground"))
    } else {
        // Linux: XDG_DATA_HOME or ~/.local/share
        std::env::var("XDG_DATA_HOME")
            .ok()
            .map(PathBuf::from)
            .or_else(|| home.as_ref().map(|h| h.join(".local/share")))
            .map(|d| d.join("shader-playground"))
    };
    let data = app_data.unwrap_or_else(|| res.clone());
    // Seed on first launch: copy writable resources from bundle
    if !data.join("weights.json").exists() {
        let _ = std::fs::create_dir_all(&data);
        // Copy weights.json and params.json as user-editable copies
        for f in &["weights.json", "params.json"] {
            let src = res.join(f);
            if src.exists() {
                let _ = std::fs::copy(&src, data.join(f));
            }
        }
        // Copy seed genomes — in a bundle, these are flat in Resources/
        let dst_genomes = data.join("genomes");
        let _ = std::fs::create_dir_all(&dst_genomes);
        let src_genomes = res.join("genomes");
        if src_genomes.exists() {
            // Dev-like layout: Resources/genomes/ exists
            copy_dir_recursive(&src_genomes, &dst_genomes);
        } else {
            // Bundle layout: default.json and flames/ are flat in Resources/
            let default_src = res.join("default.json");
            if default_src.exists() {
                let _ = std::fs::copy(&default_src, dst_genomes.join("default.json"));
            }
            let flames_src = res.join("flames");
            if flames_src.exists() {
                copy_dir_recursive(&flames_src, &dst_genomes.join("flames"));
            }
        }
    }
    data
}

fn copy_dir_recursive(src: &Path, dst: &Path) {
    let _ = std::fs::create_dir_all(dst);
    if let Ok(entries) = std::fs::read_dir(src) {
        for entry in entries.flatten() {
            let path = entry.path();
            let dest = dst.join(entry.file_name());
            if path.is_dir() {
                copy_dir_recursive(&path, &dest);
            } else {
                let _ = std::fs::copy(&path, &dest);
            }
        }
    }
}

/// project_dir() returns the writable data directory — this is the primary
/// directory the app reads/writes genomes, votes, etc.
/// For read-only resources (shaders), use resource_dir().
pub fn project_dir() -> PathBuf {
    data_dir()
}

fn shader_path() -> PathBuf {
    resource_dir().join("playground.wgsl")
}

fn compute_path() -> PathBuf {
    resource_dir().join("flame_compute.wgsl")
}

fn params_path() -> PathBuf {
    project_dir().join("params.json")
}

fn load_shader_source() -> String {
    fs::read_to_string(shader_path())
        .unwrap_or_else(|_| include_str!("../playground.wgsl").to_string())
}

fn load_compute_source() -> String {
    fs::read_to_string(compute_path())
        .unwrap_or_else(|_| include_str!("../flame_compute.wgsl").to_string())
}

fn accumulation_path() -> PathBuf {
    resource_dir().join("accumulation.wgsl")
}

fn load_accumulation_source() -> String {
    fs::read_to_string(accumulation_path())
        .unwrap_or_else(|_| include_str!("../accumulation.wgsl").to_string())
}

fn load_params() -> Vec<f32> {
    if let Ok(json) = fs::read_to_string(params_path())
        && let Ok(vals) = serde_json::from_str::<Vec<f32>>(&json)
    {
        return vals;
    }
    Vec::new()
}

fn weights_path() -> PathBuf {
    project_dir().join("weights.json")
}

fn audio_features_path() -> PathBuf {
    project_dir().join("audio_features.json")
}

fn load_weights() -> Weights {
    match Weights::load(&weights_path()) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("[weights] ERROR: {e} — using defaults");
            Weights::default()
        }
    }
}

/// Archive old genomes from history/ when size exceeds threshold.
/// Groups by generation, removes the older half.
/// lineage.json preserves ancestry regardless.
fn archive_history_if_needed(genomes_dir: &std::path::Path, threshold_mb: u64) {
    let history_dir = genomes_dir.join("history");
    if !history_dir.exists() {
        return;
    }

    // Calculate total size
    let mut total_bytes: u64 = 0;
    let mut genomes: Vec<(std::path::PathBuf, u32)> = Vec::new();

    if let Ok(read) = std::fs::read_dir(&history_dir) {
        for entry in read.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json")
                && let Ok(meta) = path.metadata()
            {
                total_bytes += meta.len();
                let generation = std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|json| serde_json::from_str::<serde_json::Value>(&json).ok())
                    .and_then(|v| v.get("generation")?.as_u64())
                    .unwrap_or(0) as u32;
                genomes.push((path, generation));
            }
        }
    }

    let threshold_bytes = threshold_mb * 1024 * 1024;
    if total_bytes < threshold_bytes {
        return;
    }

    eprintln!(
        "[archive] history/ is {}MB (threshold {}MB), archiving old genomes...",
        total_bytes / (1024 * 1024),
        threshold_mb
    );

    // Find median generation
    genomes.sort_by_key(|(_, generation)| *generation);
    let median_idx = genomes.len() / 2;
    let median_gen = genomes[median_idx].1;

    // Delete genomes below median generation (lineage.json preserves ancestry)
    let mut archived_count = 0u32;
    for (path, generation) in &genomes {
        if *generation < median_gen && std::fs::remove_file(path).is_ok() {
            archived_count += 1;
        }
    }

    eprintln!(
        "[archive] removed {} genomes below generation {}",
        archived_count, median_gen
    );
}

fn create_histogram_buffer(device: &wgpu::Device, w: u32, h: u32) -> wgpu::Buffer {
    let pixel_count = w.max(1) as u64 * h.max(1) as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("histogram"),
        size: pixel_count * 7 * 4, // 7 u32s per pixel (density, R, G, B, vx, vy, depth)
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_accumulation_buffer(device: &wgpu::Device, w: u32, h: u32) -> wgpu::Buffer {
    let pixel_count = w.max(1) as u64 * h.max(1) as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("accumulation"),
        size: pixel_count * 7 * 4, // 7 f32s per pixel (density, R, G, B, vx, vy, depth)
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_accumulation_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    histogram: &wgpu::Buffer,
    accumulation: &wgpu::Buffer,
    uniform_buffer: &wgpu::Buffer,
    max_density: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("accumulation"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: histogram.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: accumulation.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: max_density.as_entire_binding(),
            },
        ],
    })
}

fn create_accumulation_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader_src: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("accumulation"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("accumulation"),
        layout: Some(layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn create_frame_textures(
    device: &wgpu::Device,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
) {
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

#[allow(clippy::too_many_arguments)]
fn create_render_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    prev_frame: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    accumulation: &wgpu::Buffer,
    crossfade_view: &wgpu::TextureView,
    max_density: &wgpu::Buffer,
    cdf_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(prev_frame),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: accumulation.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(crossfade_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: max_density.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: cdf_buffer.as_entire_binding(),
            },
        ],
    })
}

fn create_histogram_cdf_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    accumulation: &wgpu::Buffer,
    hist_bins: &wgpu::Buffer,
    cdf: &wgpu::Buffer,
    uniforms: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("histogram_cdf"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: accumulation.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: hist_bins.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cdf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniforms.as_entire_binding(),
            },
        ],
    })
}

#[allow(clippy::too_many_arguments)]
fn create_compute_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    histogram: &wgpu::Buffer,
    uniform_buffer: &wgpu::Buffer,
    transform_buffer: &wgpu::Buffer,
    palette_view: &wgpu::TextureView,
    palette_sampler: &wgpu::Sampler,
    point_state: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: histogram.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: transform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(palette_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(palette_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: point_state.as_entire_binding(),
            },
        ],
    })
}

fn create_palette_texture(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("palette"),
        size: wgpu::Extent3d {
            width: 256,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn upload_palette_texture(queue: &wgpu::Queue, texture: &wgpu::Texture, data: &[[f32; 4]]) {
    // Convert f32 → f16 for Rgba16Float texture
    let f16_data: Vec<u16> = data
        .iter()
        .flat_map(|px| px.iter().map(|&v| half::f16::from_f32(v).to_bits()))
        .collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&f16_data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(256 * 8), // 256 pixels * 4 f16 * 2 bytes
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: 256,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader_src: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("playground"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("main"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader_src: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("flame_compute"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("flame"),
        layout: Some(layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

// ── UiState — all heavy state, lives on the UI thread ──

struct UiState {
    render_tx: mpsc::SyncSender<RenderCommand>,
    egui_ctx: egui::Context,
    // Core state
    start: Instant,
    frame: u32,
    fps_frame_count: u32,
    last_fps_time: Instant,
    mouse: [f32; 2],
    gpu_width: u32,
    gpu_height: u32,
    globals: [f32; 20],
    xf_params: Vec<f32>,
    num_transforms: usize,
    last_frame_time: Instant,
    genome: FlameGenome,
    genome_history: Vec<FlameGenome>,
    audio_features: AudioFeatures,
    weights: Weights,
    audio_enabled: bool,
    audio_info: bool,
    audio_info_timer: f32,
    audio_samples: Vec<AudioFeatures>,
    mutation_accum: f32,
    last_mutation_time: f32,
    flame_locked: bool,
    morph_burst_frames: u32,
    random_walk: f32,
    // Morph state
    morph_start_globals: [f32; 20],
    morph_start_xf: Vec<f32>,
    morph_base_globals: [f32; 20],
    morph_base_xf: Vec<f32>,
    morph_progress: f32,
    morph_xf_rates: Vec<f32>,
    // Genetics / taste / archive
    favorite_profile: Option<FavoriteProfile>,
    vote_ledger: VoteLedger,
    lineage_cache: crate::votes::LineageCache,
    taste_engine: crate::taste::TasteEngine,
    perf_model: crate::taste::PerfModel,
    archive: crate::archive::MapElitesArchive,
    last_profile_scan: f32,
    perf_log: Option<std::fs::File>,
    prev_zoom: f32,
    genome_frame_count: u32,
    genome_start_time: Instant,
    // HUD / egui state
    last_cursor_move: Instant,
    pending_egui_textures: egui::TexturesDelta,
    vote_feedback: Option<(i32, String, String, Option<String>)>,
    config_panel_open: bool,
    config_edit: Option<crate::weights::RuntimeConfig>,
    config_edit_weights: Option<crate::weights::Weights>,
    config_tab: usize,
    prev_screen_size: (f32, f32),
    reposition_panels: bool,
    cached_audio_devices: Vec<String>,
    selected_audio_device: String,
    // File watcher
    watcher: Option<FileWatcher>,
}

// ── App — paper-thin, lives on the main (winit) thread ──

struct App {
    window: Option<Arc<Window>>,
    egui_state: Option<egui_winit::State>,
    ui_tx: Option<mpsc::Sender<UiEvent>>,
    gpu_width: u32,
    gpu_height: u32,
    // Held temporarily during init before UI thread spawns
    init_state: Option<UiInitState>,
}

/// Temporary state held on App before the UI thread is spawned.
struct UiInitState {
    weights: Weights,
    genome: FlameGenome,
    initial_globals: [f32; 20],
    initial_xf: Vec<f32>,
    num_transforms: usize,
    favorite_profile: Option<FavoriteProfile>,
    vote_ledger: VoteLedger,
    lineage_cache: crate::votes::LineageCache,
    taste_engine: crate::taste::TasteEngine,
    perf_model: crate::taste::PerfModel,
    archive: crate::archive::MapElitesArchive,
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl App {
    fn new() -> Self {
        let t = Instant::now();
        let weights = load_weights();
        eprintln!(
            "[boot] weights loaded ({:.0}ms)",
            t.elapsed().as_secs_f64() * 1000.0
        );

        let t = Instant::now();
        let favorite_profile = scan_favorite_profile();
        eprintln!(
            "[boot] favorite profile scanned ({:.0}ms)",
            t.elapsed().as_secs_f64() * 1000.0
        );

        let genomes_root = project_dir().join("genomes");
        let _ = std::fs::create_dir_all(genomes_root.join("voted"));
        let _ = std::fs::create_dir_all(genomes_root.join("history"));
        if weights._config.archive_on_startup {
            let t = Instant::now();
            archive_history_if_needed(&genomes_root, weights._config.archive_threshold_mb);
            eprintln!(
                "[boot] archive check ({:.0}ms)",
                t.elapsed().as_secs_f64() * 1000.0
            );
        }

        let t = Instant::now();
        let vote_ledger = VoteLedger::load(&genomes_root);
        eprintln!(
            "[boot] vote ledger loaded ({:.0}ms)",
            t.elapsed().as_secs_f64() * 1000.0
        );

        let t = Instant::now();
        let lineage_cache = crate::votes::LineageCache::load(&genomes_root);
        eprintln!(
            "[boot] lineage cache built ({:.0}ms)",
            t.elapsed().as_secs_f64() * 1000.0
        );

        let t = Instant::now();
        let flames_dir = project_dir().join("genomes").join("flames");
        let seeds_dir = project_dir().join("genomes").join("seeds");
        let mut genome = crate::flam3::load_random_flame(&flames_dir)
            .or_else(|_| FlameGenome::load_random(&seeds_dir))
            .unwrap_or_else(|_| FlameGenome::default_genome());
        genome.adjust_transform_count(&weights._config);
        eprintln!(
            "[boot] genome loaded: {} ({:.0}ms)",
            genome.name,
            t.elapsed().as_secs_f64() * 1000.0
        );
        let initial_globals = genome.flatten_globals(&weights._config);
        let initial_xf = genome.flatten_transforms();
        let num_transforms = genome.total_buffer_transforms();

        Self {
            window: None,
            egui_state: None,
            ui_tx: None,
            gpu_width: 1,
            gpu_height: 1,
            init_state: Some(UiInitState {
                weights,
                genome,
                initial_globals,
                initial_xf,
                num_transforms,
                favorite_profile,
                vote_ledger,
                lineage_cache,
                taste_engine: crate::taste::TasteEngine::new(),
                perf_model: crate::taste::PerfModel::load(
                    &project_dir().join("genomes").join("perf_model.json"),
                )
                .unwrap_or_default(),
                archive: {
                    let archive_path = genomes_root.join("archive.json");
                    crate::archive::MapElitesArchive::load(&archive_path)
                        .unwrap_or_else(|_| crate::archive::MapElitesArchive::new())
                },
            }),
        }
    }
}

/// Scan genomes/ directory for favorite profile (excludes seeds/).
fn scan_favorite_profile() -> Option<FavoriteProfile> {
    let genomes_dir = project_dir().join("genomes");
    if !genomes_dir.exists() {
        return None;
    }
    let profile = FavoriteProfile::from_directory(&genomes_dir);
    if profile.variation_freq.is_empty() {
        None
    } else {
        Some(profile)
    }
}

impl UiState {
    fn new(
        render_tx: mpsc::SyncSender<RenderCommand>,
        egui_ctx: egui::Context,
        init: UiInitState,
        watcher: Option<FileWatcher>,
        gpu_width: u32,
        gpu_height: u32,
    ) -> Self {
        Self {
            render_tx,
            egui_ctx,
            start: Instant::now(),
            frame: 0,
            fps_frame_count: 0,
            last_fps_time: Instant::now(),
            mouse: [0.5, 0.5],
            gpu_width,
            gpu_height,
            globals: init.initial_globals,
            xf_params: init.initial_xf.clone(),
            num_transforms: init.num_transforms,
            last_frame_time: Instant::now(),
            genome: init.genome,
            genome_history: Vec::new(),
            audio_features: AudioFeatures::default(),
            weights: init.weights,
            audio_enabled: true,
            audio_info: false,
            audio_info_timer: 0.0,
            audio_samples: Vec::new(),
            mutation_accum: 0.0,
            last_mutation_time: 0.0,
            flame_locked: false,
            morph_burst_frames: 0,
            random_walk: 0.0,
            morph_start_globals: init.initial_globals,
            morph_start_xf: init.initial_xf.clone(),
            morph_base_globals: init.initial_globals,
            morph_base_xf: init.initial_xf,
            morph_progress: 1.0,
            morph_xf_rates: Vec::new(),
            favorite_profile: init.favorite_profile,
            vote_ledger: init.vote_ledger,
            lineage_cache: init.lineage_cache,
            taste_engine: init.taste_engine,
            perf_model: init.perf_model,
            archive: init.archive,
            last_profile_scan: 0.0,
            perf_log: std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("perf.log")
                .ok(),
            prev_zoom: 3.0,
            genome_frame_count: 0,
            genome_start_time: Instant::now(),
            last_cursor_move: Instant::now(),
            pending_egui_textures: egui::TexturesDelta::default(),
            vote_feedback: None,
            config_panel_open: false,
            config_edit: None,
            config_edit_weights: None,
            config_tab: 0,
            prev_screen_size: (1.0, 1.0),
            reposition_panels: true,
            cached_audio_devices: enumerate_audio_devices(),
            selected_audio_device: String::new(),
            watcher,
        }
    }

    /// Handle cursor movement from the main thread.
    fn on_cursor_moved(&mut self, x: f32, y: f32) {
        self.last_cursor_move = Instant::now();
        self.mouse = [
            x / self.gpu_width.max(1) as f32,
            y / self.gpu_height.max(1) as f32,
        ];
    }

    /// Handle window resize from the main thread.
    fn on_resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.gpu_width = width;
            self.gpu_height = height;
        }
    }

    /// Handle a key press forwarded from the main thread.
    fn on_key(&mut self, key: &str) {
        // If vote feedback popup is open, keys are handled by egui
        if self.vote_feedback.is_some() {
            return;
        }
        match key {
            "Space" => {
                self.genome_history.push(self.genome.clone());
                if self.genome_history.len() > 10 {
                    self.genome_history.remove(0);
                }
                self.flame_locked = false;
                self.mutation_accum = 0.0;
                let (pa, pb, community) = self.pick_breeding_parents();
                self.genome = FlameGenome::mutate(
                    &pa,
                    &pb,
                    &community,
                    &self.audio_features,
                    &self.weights._config,
                    &self.favorite_profile,
                    &mut Some(&mut self.taste_engine),
                );
                let genomes_dir = project_dir().join("genomes");
                self.lineage_cache.register_and_save(
                    &self.genome.name,
                    &self.genome.parent_a,
                    &self.genome.parent_b,
                    self.genome.generation,
                    &genomes_dir,
                );
                let history_dir = genomes_dir.join("history");
                if let Err(e) = self.genome.save(&history_dir) {
                    eprintln!("[history] save error: {e}");
                }
                self.archive_genome();
                self.last_mutation_time = self.start.elapsed().as_secs_f32();
                self.begin_morph();
                if let Some(taste_score) = self.taste_engine.score_genome(&self.genome) {
                    eprintln!(
                        "[evolve] t={:.1}s → {} (gen {}, taste={:.2})",
                        self.start.elapsed().as_secs_f32(),
                        self.genome.name,
                        self.genome.generation,
                        taste_score
                    );
                } else {
                    eprintln!(
                        "[evolve] t={:.1}s → {} (gen {})",
                        self.start.elapsed().as_secs_f32(),
                        self.genome.name,
                        self.genome.generation
                    );
                }
            }
            "Backspace" => {
                if let Some(prev) = self.genome_history.pop() {
                    self.genome = prev;
                    self.begin_morph();
                    eprintln!("[revert] back to previous");
                }
            }
            "ArrowUp" => {
                let dir = project_dir().join("genomes");
                let vote_genome = self.morph_snapshot_or_current();
                let score = self.vote_ledger.vote(&vote_genome, 1, &dir);
                self.favorite_profile = scan_favorite_profile();
                let igmm_path = dir.join("taste_model.json");
                self.taste_engine.on_upvote(&self.genome, Some(&igmm_path));
                self.rebuild_taste_model();
                let existing_note = self
                    .vote_ledger
                    .entries
                    .get(&vote_genome.name)
                    .and_then(|e| e.note.clone())
                    .unwrap_or_default();
                let prev_name = self.genome_history.last().map(|g| g.name.clone());
                self.vote_feedback = Some((1, vote_genome.name.clone(), existing_note, prev_name));
                if vote_genome.name != self.genome.name {
                    eprintln!(
                        "[vote] captured morph {:.0}% → {} +1 (score: {})",
                        self.morph_progress * 100.0,
                        vote_genome.name,
                        score
                    );
                } else {
                    eprintln!("[vote] {} → +1 (score: {})", self.genome.name, score);
                }
            }
            "ArrowDown" => {
                let dir = project_dir().join("genomes");
                let vote_genome = self.morph_snapshot_or_current();
                let score = self.vote_ledger.vote(&vote_genome, -1, &dir);
                self.favorite_profile = scan_favorite_profile();
                self.rebuild_taste_model();
                let existing_note = self
                    .vote_ledger
                    .entries
                    .get(&vote_genome.name)
                    .and_then(|e| e.note.clone())
                    .unwrap_or_default();
                let prev_name = self.genome_history.last().map(|g| g.name.clone());
                self.vote_feedback = Some((-1, vote_genome.name.clone(), existing_note, prev_name));
                if vote_genome.name != self.genome.name {
                    eprintln!(
                        "[vote] captured morph {:.0}% → {} -1 (score: {})",
                        self.morph_progress * 100.0,
                        vote_genome.name,
                        score
                    );
                } else {
                    eprintln!("[vote] {} → -1 (score: {})", self.genome.name, score);
                }
            }
            "s" => {
                let save_genome = self.morph_snapshot_or_current();
                let dir = project_dir().join("genomes");
                match save_genome.save(&dir) {
                    Ok(p) => {
                        if save_genome.name != self.genome.name {
                            eprintln!(
                                "[save] morph snapshot at {:.0}% → {}",
                                self.morph_progress * 100.0,
                                p.display()
                            );
                        } else {
                            eprintln!("[save] {}", p.display());
                        }
                    }
                    Err(e) => eprintln!("[save] error: {e}"),
                }
                self.favorite_profile = scan_favorite_profile();
                self.last_profile_scan = self.start.elapsed().as_secs_f32();
            }
            "l" => {
                let dir = project_dir().join("genomes");
                match FlameGenome::load_random(&dir) {
                    Ok(mut g) => {
                        g.adjust_transform_count(&self.weights._config);
                        self.genome_history.push(self.genome.clone());
                        if self.genome_history.len() > 10 {
                            self.genome_history.remove(0);
                        }
                        eprintln!("[load] {}", g.name);
                        self.genome = g;
                        self.begin_morph();
                    }
                    Err(e) => eprintln!("[load] error: {e}"),
                }
            }
            "1" | "2" | "3" | "4" => {
                let idx: usize = key.parse::<usize>().unwrap() - 1;
                if idx < self.num_transforms {
                    let base = idx * PARAMS_PER_XF;
                    if base < self.xf_params.len() {
                        if self.xf_params[base] < 0.01 {
                            self.xf_params[base] = 0.25;
                        } else {
                            self.xf_params[base] = 0.0;
                        }
                        eprintln!("[solo] transform {} = {}", idx, self.xf_params[base]);
                    }
                }
            }
            "f" => {
                let flames_dir = project_dir().join("genomes").join("flames");
                match crate::flam3::load_random_flame(&flames_dir) {
                    Ok(mut g) => {
                        g.adjust_transform_count(&self.weights._config);
                        self.genome_history.push(self.genome.clone());
                        if self.genome_history.len() > 10 {
                            self.genome_history.remove(0);
                        }
                        self.genome = g;
                        self.flame_locked = true;
                        self.mutation_accum = 0.0;
                        self.begin_morph();
                        eprintln!("[flame] loaded (locked): {}", self.genome.name);
                    }
                    Err(e) => eprintln!("[flame] error: {e}"),
                }
            }
            "a" => {
                self.audio_enabled = !self.audio_enabled;
                eprintln!(
                    "[audio] {}",
                    if self.audio_enabled {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
            }
            "c" => {
                self.config_panel_open = !self.config_panel_open;
                if self.config_panel_open {
                    self.config_edit = Some(self.weights._config.clone());
                    self.config_edit_weights = Some(self.weights.clone());
                } else {
                    self.config_edit = None;
                    self.config_edit_weights = None;
                }
                eprintln!(
                    "[config] panel {}",
                    if self.config_panel_open {
                        "opened"
                    } else {
                        "closed"
                    }
                );
            }
            "i" => {
                self.audio_info = !self.audio_info;
                if self.audio_info {
                    self.audio_samples.clear();
                    eprintln!("[info] ON — recording audio features (press i again to save)");
                } else {
                    let count = self.audio_samples.len();
                    if count > 0 {
                        let timestamp = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs();
                        let path = format!("audio_samples_{}.json", timestamp);
                        match serde_json::to_string_pretty(&self.audio_samples) {
                            Ok(json) => {
                                std::fs::write(&path, json).ok();
                                eprintln!("[info] OFF — saved {count} samples to {path}");
                            }
                            Err(e) => eprintln!("[info] OFF — serialize error: {e}"),
                        }
                    } else {
                        eprintln!("[info] OFF — no samples collected");
                    }
                }
            }
            _ => {}
        }
    }

    /// Build the egui UI on the UI thread using pre-collected RawInput.
    fn build_egui_frame(
        &mut self,
        hud: &HudFrameData,
        raw_input: egui::RawInput,
    ) -> (Vec<egui::ClippedPrimitive>, egui::TexturesDelta, f32) {
        let egui_ctx = &self.egui_ctx;

        let screen_w = self.gpu_width as f32;
        let screen_h = self.gpu_height as f32;

        // HUD fade — compute opacity from cursor idle time
        let hud_opacity = compute_hud_opacity(
            self.last_cursor_move.elapsed().as_secs_f32(),
            hud.hud_fade_delay,
            hud.hud_fade_duration,
        );

        // --- Resize-aware panel repositioning ---
        // Detect window size change and compute per-panel position overrides so
        // panels maintain their distance from the nearest edge.
        let size_changed = (screen_w - self.prev_screen_size.0).abs() > 0.5
            || (screen_h - self.prev_screen_size.1).abs() > 0.5;
        let dw = screen_w - self.prev_screen_size.0;
        let dh = screen_h - self.prev_screen_size.1;
        let old_w = self.prev_screen_size.0;
        let old_h = self.prev_screen_size.1;
        self.prev_screen_size = (screen_w, screen_h);

        // Build per-panel override positions for this frame.
        // - On resize: shift panels that are right-anchored / bottom-anchored
        // - On reposition (reset button): snap to default non-overlapping layout
        // - Otherwise: None (let egui manage drag state)
        let panel_ids: [&str; 6] = [
            "hud_identity",
            "hud_progress",
            "hud_audio",
            "hud_time",
            "hud_transforms",
            "hud_hotkeys",
        ];

        let override_positions: [Option<egui::Pos2>; 6] = if self.reposition_panels {
            // One-shot reset to default non-overlapping positions
            self.reposition_panels = false;
            [
                Some(egui::pos2(10.0, 10.0)),              // identity: top-left
                Some(egui::pos2(screen_w - 230.0, 10.0)),  // progress: top-right
                Some(egui::pos2(10.0, 80.0)),              // audio: left, below identity
                Some(egui::pos2(10.0, 300.0)),             // time: left, below audio
                Some(egui::pos2(screen_w - 230.0, 260.0)), // transforms: right, below progress
                Some(egui::pos2(screen_w * 0.5 - 250.0, screen_h - 40.0)), // hotkeys: bottom center
            ]
        } else if size_changed {
            // Shift panels based on their anchor edge
            let mut positions: [Option<egui::Pos2>; 6] = [None; 6];
            for (i, name) in panel_ids.iter().enumerate() {
                let id = egui::Id::new(*name);
                if let Some(rect) = egui_ctx.memory(|m| m.area_rect(id)) {
                    let mut pos = rect.min;
                    // Right-anchored if panel center is past 40% of old width
                    if pos.x > old_w * 0.4 {
                        pos.x += dw;
                    }
                    // Bottom-anchored if panel center is past 60% of old height
                    if pos.y > old_h * 0.6 {
                        pos.y += dh;
                    }
                    positions[i] = Some(pos);
                }
            }
            positions
        } else {
            [None; 6]
        };

        // Track vote/config actions to apply after the egui closure
        let mut vote_submit: Option<(String, Option<String>)> = None;
        let mut config_action = ConfigAction {
            changed: false,
            save: false,
            reset: false,
            cancel: false,
        };

        let mut refresh_audio_devices = false;
        let egui_output = egui_ctx.run_ui(raw_input, |ui| {
            // HUD panels — skip when fully faded out
            if hud_opacity > 0.0 {
                let ctx = ui.ctx();
                hud_panel_identity(ctx, hud, hud_opacity, override_positions[0]);
                hud_panel_progress(ctx, hud, screen_w, hud_opacity, override_positions[1]);
                hud_panel_audio(ctx, hud, hud_opacity, override_positions[2]);
                hud_panel_time(ctx, hud, hud_opacity, override_positions[3]);
                hud_panel_transforms(ctx, hud, screen_w, hud_opacity, override_positions[4]);
                hud_panel_hotkeys(ctx, screen_w, screen_h, hud_opacity, override_positions[5]);
            }

            // Config panel — always visible when open (ignores HUD fade)
            if self.config_panel_open
                && let Some(ref mut edit_weights) = self.config_edit_weights
            {
                let mut panel_state = ConfigPanelState {
                    weights: edit_weights,
                    tab: &mut self.config_tab,
                    reposition_panels: &mut self.reposition_panels,
                    cached_audio_devices: &self.cached_audio_devices,
                    selected_audio_device: &mut self.selected_audio_device,
                    refresh_audio_devices: &mut refresh_audio_devices,
                };
                config_action = config_panel_ui(ui.ctx(), &mut panel_state, screen_w, screen_h);
            }

            // Vote feedback popup — always visible when active
            if let Some((score, ref genome_name, ref mut text, ref prev_name)) = self.vote_feedback
            {
                let ctx = ui.ctx();
                let prompt = if score > 0 {
                    "What do you like about this?"
                } else {
                    "What don't you like?"
                };
                let popup_w = 400.0;
                let popup_h = 120.0;
                egui::Area::new(egui::Id::new("vote_feedback"))
                    .fixed_pos(egui::pos2(
                        (screen_w - popup_w) * 0.5,
                        (screen_h - popup_h) * 0.5,
                    ))
                    .show(ctx, |ui| {
                        egui::Frame::NONE
                            .fill(egui::Color32::from_rgba_premultiplied(20, 20, 20, 200))
                            .corner_radius(8.0)
                            .inner_margin(egui::Margin::same(16))
                            .show(ui, |ui| {
                                ui.set_min_width(popup_w - 32.0);
                                let arrow = if score > 0 { "\u{2191}" } else { "\u{2193}" };
                                ui.label(
                                    egui::RichText::new(format!("{arrow} {prompt}"))
                                        .size(14.0)
                                        .color(egui::Color32::WHITE),
                                );
                                // Show genome context
                                let dim = egui::Color32::from_rgb(120, 120, 120);
                                ui.label(
                                    egui::RichText::new(format!("current: {genome_name}"))
                                        .size(11.0)
                                        .color(dim),
                                );
                                if let Some(prev) = prev_name {
                                    ui.label(
                                        egui::RichText::new(format!("previous: {prev}"))
                                            .size(11.0)
                                            .color(dim),
                                    );
                                }
                                ui.add_space(4.0);
                                let response = ui.add(
                                    egui::TextEdit::singleline(text)
                                        .desired_width(popup_w - 32.0)
                                        .hint_text(
                                            "optional \u{2014} press Enter to submit, Esc to skip",
                                        ),
                                );
                                // Auto-focus the text input
                                if response.gained_focus() || !response.has_focus() {
                                    response.request_focus();
                                }
                                // Enter = submit note, Esc = dismiss
                                if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                    let note = if text.trim().is_empty() {
                                        None
                                    } else {
                                        Some(text.trim().to_string())
                                    };
                                    vote_submit = Some((genome_name.clone(), note));
                                }
                                if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                                    vote_submit = Some((genome_name.clone(), None));
                                }
                            });
                    });
            }
        });

        // Note: handle_platform_output (cursor changes) requires Window, which lives
        // on the main thread. We skip it here — cursor shape changes are cosmetic.

        // Apply vote feedback actions
        if let Some((genome_name, note)) = vote_submit {
            if let Some(note_text) = note {
                let dir = project_dir().join("genomes");
                self.vote_ledger.attach_note(&genome_name, note_text, &dir);
                eprintln!("[vote] note attached to {genome_name}");
            }
            self.vote_feedback = None;
        }

        // Apply config actions
        if config_action.changed
            && let Some(ref edit_weights) = self.config_edit_weights
        {
            self.weights = edit_weights.clone();
        }
        if config_action.save {
            // Save applies edit to live weights AND writes to disk
            if let Some(ref edit_weights) = self.config_edit_weights {
                self.weights = edit_weights.clone();
            }
            let path = weights_path();
            match serde_json::to_string_pretty(&self.weights) {
                Ok(json) => {
                    if let Err(e) = std::fs::write(&path, json) {
                        eprintln!("[config] save error: {e}");
                    } else {
                        eprintln!("[config] saved to {}", path.display());
                    }
                }
                Err(e) => eprintln!("[config] serialize error: {e}"),
            }
            self.config_panel_open = false;
            self.config_edit = None;
            self.config_edit_weights = None;
        }
        if config_action.reset {
            // Reset to serde defaults
            let default_weights: crate::weights::Weights =
                serde_json::from_str(r#"{"_config": {}}"#).unwrap();
            self.config_edit = Some(default_weights._config.clone());
            self.config_edit_weights = Some(default_weights.clone());
            self.weights = default_weights;
            eprintln!("[config] reset to defaults");
        }
        if config_action.cancel {
            // Revert to what was on disk before editing
            self.config_edit = None;
            self.config_edit_weights = None;
            self.config_panel_open = false;
            // Reload from disk
            if let Ok(w) = crate::weights::Weights::load(&weights_path()) {
                self.weights = w;
                eprintln!("[config] cancelled — reverted to saved config");
            }
        }

        // Refresh audio device cache if requested
        if refresh_audio_devices {
            self.cached_audio_devices = enumerate_audio_devices();
        }

        // Tessellate on main thread — send paint jobs to render thread
        let pixels_per_point = egui_output.pixels_per_point;
        let primitives = egui_ctx.tessellate(egui_output.shapes, pixels_per_point);
        (primitives, egui_output.textures_delta, pixels_per_point)
    }

    /// If mid-morph, snapshot the interpolated state as a new genome. Otherwise return current.
    fn morph_snapshot_or_current(&self) -> FlameGenome {
        if self.morph_progress >= 1.0 {
            return self.genome.clone();
        }
        let mut snapshot = self.genome.clone();
        snapshot.name = format!(
            "captured-{}",
            rand::Rng::random_range(&mut rand::rng(), 1000u32..9999)
        );
        let xf_data = &self.morph_base_xf;
        for (i, xf) in snapshot.transforms.iter_mut().enumerate() {
            let base = i * PARAMS_PER_XF;
            if base + 13 < xf_data.len() {
                xf.weight = xf_data[base];
                // 3x3 affine: 9 values at base+1..base+9
                for r in 0..3 {
                    for c in 0..3 {
                        xf.affine[r][c] = xf_data[base + 1 + r * 3 + c];
                    }
                }
                xf.offset[0] = xf_data[base + 10];
                xf.offset[1] = xf_data[base + 11];
                xf.offset[2] = xf_data[base + 12];
                xf.color = xf_data[base + 13];
            }
        }
        snapshot
    }

    /// Rebuild the taste engine model from all positively-voted + imported genomes.
    fn rebuild_taste_model(&mut self) {
        let genomes_dir = project_dir().join("genomes");
        let flames_dir = genomes_dir.join("flames");

        // Collect good genomes: positively voted + imported flames
        let mut good_genomes: Vec<FlameGenome> = Vec::new();

        // Load good genomes from voted/ directory
        let voted_dir = genomes_dir.join("voted");
        if let Ok(read) = std::fs::read_dir(&voted_dir) {
            for entry in read.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "json")
                    && let Ok(g) = FlameGenome::load(&path)
                {
                    good_genomes.push(g);
                }
            }
        }

        // Imported flames (always considered "good")
        if let Ok(read) = std::fs::read_dir(&flames_dir) {
            for entry in read.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path
                    .extension()
                    .is_some_and(|ext| ext == "flam3" || ext == "flame")
                    && let Ok(g) = crate::flam3::load_random_flame(&flames_dir)
                {
                    good_genomes.push(g);
                    break; // just get a sample, not all
                }
            }
        }

        let refs: Vec<&FlameGenome> = good_genomes.iter().collect();
        let recent_memory = self.weights._config.taste_recent_memory;
        self.taste_engine.set_config(&self.weights._config);
        let igmm_path = genomes_dir.join("taste_model.json");
        self.taste_engine
            .rebuild_with_igmm_path(&refs, recent_memory, Some(&igmm_path));

        if self
            .taste_engine
            .is_active(self.weights._config.taste_min_votes)
        {
            eprintln!(
                "[taste] model active with {} samples",
                self.taste_engine.sample_count()
            );
        }
    }

    /// Insert the current genome into the MAP-Elites archive.
    /// Computes behavioral traits from the genome and proxy render.
    fn archive_genome(&mut self) {
        let cfg = &self.weights._config;
        let perceptual = crate::taste::PerceptualFeatures::from_genome(&self.genome, cfg);
        let color_entropy = if let Some(palette) = &self.genome.palette {
            let pf = crate::taste::palette_features(palette);
            // Normalize hue cluster count to [0, 1] range (max ~6 clusters)
            (pf.hue_cluster_count / 6.0).min(1.0)
        } else {
            0.0
        };

        let coords = crate::archive::GridCoords::from_traits(
            self.genome.symmetry,
            perceptual.fractal_dimension,
            color_entropy,
        );

        let archive_features = self.archive.all_features();
        let score = self
            .taste_engine
            .score_genome_with_novelty(
                &self.genome,
                &archive_features,
                cfg.novelty_weight,
                cfg.novelty_k_neighbors,
                Some(&self.perf_model),
                cfg.perf_weight,
            )
            .unwrap_or(f32::MAX);
        let features = self
            .taste_engine
            .extract_full_features(&self.genome)
            .unwrap_or_default();

        if self
            .archive
            .insert(&coords, self.genome.name.clone(), score, features)
        {
            eprintln!(
                "[archive] inserted {} ({} cells occupied, {} feature vecs)",
                self.genome.name,
                self.archive.occupied_count(),
                self.archive.all_features().len()
            );
            let archive_path = project_dir().join("genomes").join("archive.json");
            if let Err(e) = self.archive.save(&archive_path) {
                eprintln!("[archive] save error: {e}");
            }
        }
    }

    /// Pick two parents for breeding + a community genome.
    /// Uses lineage cache to enforce minimum genetic distance between parents.
    fn pick_breeding_parents(&self) -> (FlameGenome, FlameGenome, Option<FlameGenome>) {
        use rand::Rng;
        let mut rng = rand::rng();

        let genomes_dir = project_dir().join("genomes");
        let threshold = self.weights._config.vote_blacklist_threshold;
        let min_distance = self.weights._config.min_breeding_distance;
        let max_depth = self.weights._config.max_lineage_depth;

        // 50% chance: pick parent A from MAP-Elites archive (uniform across occupied cells)
        let archive_pick = if rng.random::<f32>() < 0.5 {
            self.archive.pick_random(&mut rng).and_then(|entry| {
                let path = genomes_dir
                    .join("history")
                    .join(format!("{}.json", entry.genome_name));
                FlameGenome::load(&path).ok().or_else(|| {
                    let voted = genomes_dir
                        .join("voted")
                        .join(format!("{}.json", entry.genome_name));
                    FlameGenome::load(&voted).ok()
                })
            })
        } else {
            None
        };

        // Parent A: archive pick, or vote-weighted, or random saved
        let parent_a = archive_pick
            .or_else(|| {
                self.vote_ledger
                    .pick_voted(threshold)
                    .and_then(|p| FlameGenome::load(&p).ok())
            })
            .or_else(|| {
                VoteLedger::pick_random_saved(&genomes_dir, threshold, &self.vote_ledger)
                    .and_then(|p| FlameGenome::load(&p).ok())
            })
            .unwrap_or_else(|| self.genome.clone());

        // Parent B: try multiple candidates, pick one with sufficient genetic distance
        let max_attempts = 10u32;
        let mut parent_b: Option<FlameGenome> = None;

        for _ in 0..max_attempts {
            let candidate =
                VoteLedger::pick_random_saved(&genomes_dir, threshold, &self.vote_ledger)
                    .and_then(|p| FlameGenome::load(&p).ok())
                    .or_else(|| {
                        self.vote_ledger
                            .pick_voted(threshold)
                            .and_then(|p| FlameGenome::load(&p).ok())
                    });

            if let Some(c) = candidate {
                if c.name == parent_a.name {
                    continue; // skip self-breeding
                }
                let dist = self
                    .lineage_cache
                    .genetic_distance(&parent_a.name, &c.name, max_depth);
                if dist >= min_distance {
                    parent_b = Some(c);
                    break;
                }
                // Keep as fallback if nothing better found
                if parent_b.is_none() {
                    parent_b = Some(c);
                }
            }
        }

        // Fallback chain: imported flame → seed → random genome
        let parent_b = parent_b
            .or_else(|| {
                let flames_dir = project_dir().join("genomes").join("flames");
                crate::flam3::load_random_flame(&flames_dir).ok()
            })
            .or_else(|| {
                let seeds_dir = project_dir().join("genomes").join("seeds");
                FlameGenome::load_random(&seeds_dir).ok()
            })
            .unwrap_or_else(|| {
                let mut g = FlameGenome::default_genome();
                g.name = format!(
                    "seed-{}",
                    rand::Rng::random_range(&mut rand::rng(), 1000..9999u32)
                );
                g
            });

        // Community genome: pick from imported flames or voted pool
        let community = {
            let flames_dir = project_dir().join("genomes").join("flames");
            crate::flam3::load_random_flame(&flames_dir)
                .ok()
                .or_else(|| {
                    self.vote_ledger
                        .pick_voted(threshold)
                        .and_then(|p| FlameGenome::load(&p).ok())
                })
                .or_else(|| {
                    VoteLedger::pick_random_saved(&genomes_dir, threshold, &self.vote_ledger)
                        .and_then(|p| FlameGenome::load(&p).ok())
                })
        };

        (parent_a, parent_b, community)
    }

    /// Begin morphing toward the current genome. Captures current base as start point.
    fn begin_morph(&mut self) {
        // Reset per-genome performance tracking
        self.genome_frame_count = 0;
        self.genome_start_time = Instant::now();

        // Snapshot wherever the morph currently is as our new start
        self.morph_start_globals = self.morph_base_globals;
        self.morph_start_xf = self.morph_base_xf.clone();
        self.morph_progress = 0.0;

        // Ensure buffers can hold max of current and target transforms
        let target_xf = self.genome.flatten_transforms();
        let max_xf = (self.xf_params.len().max(target_xf.len())) / PARAMS_PER_XF;
        if max_xf != self.num_transforms {
            self.num_transforms = max_xf;
            let _ = self
                .render_tx
                .send(RenderCommand::ResizeTransformBuffer(self.num_transforms));
        }
        // Pad start vectors to match
        self.morph_start_xf.resize(max_xf * PARAMS_PER_XF, 0.0);

        // Generate per-transform morph rates: some fast (2x), some slow (0.4x)
        // This makes transforms arrive at different times for organic transitions.
        // Per-transform morph rates: base speed from config, with optional stagger.
        // morph_speed scales all transforms. morph_stagger_count=0 disables randomness.
        let cfg = &self.weights._config;
        let base_speed = cfg.morph_speed;
        self.morph_xf_rates = vec![base_speed; max_xf];
        if cfg.morph_stagger_count > 0 && max_xf > 0 {
            use rand::Rng;
            let mut rng = rand::rng();
            let stagger_min = cfg.morph_stagger_min;
            let stagger_max = cfg.morph_stagger_max;
            let num_slow = rng
                .random_range(1..=cfg.morph_stagger_count as usize)
                .min(max_xf);
            for _ in 0..num_slow {
                let idx = rng.random_range(0..max_xf);
                self.morph_xf_rates[idx] = base_speed * rng.random_range(stagger_min..=stagger_max);
            }
        }

        // Don't clear accumulation buffer — let old density fade naturally.
        // Front-load compute: run extra passes for the first ~60 frames so the
        // new genome fills in faster. Also use faster decay to dissolve the old.
        self.morph_burst_frames = 60;

        // Upload palette for the new genome
        let palette_data = crate::genome::palette_rgba_data(&self.genome);
        let _ = self
            .render_tx
            .send(RenderCommand::UpdatePalette(palette_data));
    }

    fn check_file_changes(&mut self) {
        let watcher = match &self.watcher {
            Some(w) => w,
            None => return,
        };
        let changed = watcher.changed_files();
        let mut reload_shader = false;
        let mut reload_compute = false;
        let mut reload_params = false;
        let mut reload_weights = false;
        let mut reload_features = false;
        let mut reload_votes = false;
        let shader = shader_path();
        let compute = compute_path();
        let params = params_path();
        for path in &changed {
            if path.ends_with(shader.file_name().unwrap()) {
                reload_shader = true;
            }
            if path.ends_with(compute.file_name().unwrap()) {
                reload_compute = true;
            }
            if path.ends_with(params.file_name().unwrap()) {
                reload_params = true;
            }
            if path.ends_with("weights.json") {
                reload_weights = true;
            }
            if path.ends_with("audio_features.json") {
                reload_features = true;
            }
            if path.ends_with("votes.json") {
                reload_votes = true;
            }
        }
        if reload_shader {
            let src = load_shader_source();
            let _ = self.render_tx.send(RenderCommand::ReloadShader(src));
            eprintln!("[shader] reloaded");
        }
        if reload_compute {
            let src = load_compute_source();
            let _ = self.render_tx.send(RenderCommand::ReloadComputeShader(src));
            eprintln!("[compute] reloaded");
        }
        if reload_params {
            let override_params = load_params();
            // Override globals from params.json (first 12 entries map to globals)
            let n = 20.min(override_params.len());
            self.globals[..n].copy_from_slice(&override_params[..n]);
            self.morph_base_globals[..n].copy_from_slice(&override_params[..n]);
            eprintln!("[params] reloaded (overriding globals)");
        }
        if reload_weights {
            match Weights::load(&weights_path()) {
                Ok(w) => {
                    self.weights = w;
                    eprintln!(
                        "[weights] reloaded (samples_per_frame={}, morph={}s, cooldown={}s)",
                        self.weights._config.samples_per_frame,
                        self.weights._config.morph_duration,
                        self.weights._config.mutation_cooldown
                    );
                }
                Err(e) => {
                    eprintln!("[weights] ERROR: {e} — keeping current config");
                }
            }
        }
        if reload_features
            && let Ok(json) = fs::read_to_string(audio_features_path())
            && let Ok(f) = serde_json::from_str::<AudioFeatures>(&json)
        {
            self.audio_features = f;
        }
        if reload_votes {
            self.vote_ledger = VoteLedger::load(&project_dir().join("genomes"));
            self.rebuild_taste_model();
            eprintln!("[reload] votes.json");
        }
    }

    /// Main per-frame tick. Called on the UI thread with pre-collected egui RawInput.
    /// This is the body of what used to be RedrawRequested on the main thread.
    fn tick(&mut self, raw_input: egui::RawInput) {
        if self.frame < 3 {
            eprintln!("[debug] UI tick frame={}", self.frame);
        }
        self.check_file_changes();

        let now = Instant::now();
        let dt = now
            .duration_since(self.last_frame_time)
            .as_secs_f32()
            .max(0.001);
        self.last_frame_time = now;

        // Apply weight matrix (audio features from background thread, plus time)
        if self.audio_enabled {
            let time = self.start.elapsed().as_secs_f32();
            let time_since_mutation = time - self.last_mutation_time;
            self.random_walk += crate::weights::value_noise_pub(time * 0.3) * dt * 0.5;
            let time_signals =
                crate::weights::TimeSignals::compute(time, time_since_mutation, self.random_walk);

            // Advance morph progress
            let morph_dur = self.weights._config.morph_duration;
            let min_rate = self
                .morph_xf_rates
                .iter()
                .copied()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(1.0)
                .max(0.1);
            let morph_done_at = 1.0 / min_rate;
            if self.morph_progress < morph_done_at {
                self.morph_progress = (self.morph_progress + dt / morph_dur).min(morph_done_at);
            }
            let t = smoothstep(self.morph_progress.min(1.0));

            // Interpolate genome base values (ease-in-out)
            let genome_globals = self.genome.flatten_globals(&self.weights._config);
            let genome_xf = self.genome.flatten_transforms();

            for (base, (start, target)) in self
                .morph_base_globals
                .iter_mut()
                .zip(self.morph_start_globals.iter().zip(genome_globals.iter()))
            {
                *base = start + (target - start) * t;
            }
            let max_len = self.morph_start_xf.len().max(genome_xf.len());
            self.morph_base_xf.resize(max_len, 0.0);
            let mut padded_start = self.morph_start_xf.clone();
            padded_start.resize(max_len, 0.0);
            let mut padded_genome = genome_xf;
            padded_genome.resize(max_len, 0.0);
            let num_xf = max_len / PARAMS_PER_XF;
            for xi in 0..num_xf {
                let rate = self.morph_xf_rates.get(xi).copied().unwrap_or(1.0);
                let xf_t = smoothstep((self.morph_progress * rate).min(1.0));
                let base = xi * PARAMS_PER_XF;
                for j in 0..48 {
                    let idx = base + j;
                    if idx < max_len {
                        self.morph_base_xf[idx] =
                            padded_start[idx] + (padded_genome[idx] - padded_start[idx]) * xf_t;
                    }
                }
            }

            // Apply audio modulation on top of morphed base
            let modulated_globals = self.weights.apply_globals(
                &self.morph_base_globals,
                &self.audio_features,
                &time_signals,
            );
            let modulated_xf = self.weights.apply_transforms(
                &self.morph_base_xf,
                self.num_transforms,
                &self.audio_features,
                &time_signals,
            );

            self.globals = modulated_globals;
            self.xf_params = modulated_xf;

            // Apply variation CRISPR
            self.weights
                ._config
                .apply_variation_scales(&mut self.xf_params, self.num_transforms);

            // Clamp per-transform spin_mod and drift_mod
            let cfg = &self.weights._config;
            for xf in 0..self.num_transforms {
                let spin_idx = xf * PARAMS_PER_XF + SPIN_MOD_FIELD;
                let drift_idx = xf * PARAMS_PER_XF + DRIFT_MOD_FIELD;
                if spin_idx < self.xf_params.len() {
                    self.xf_params[spin_idx] =
                        self.xf_params[spin_idx].clamp(0.0, cfg.spin_mod_max);
                }
                if drift_idx < self.xf_params.len() {
                    self.xf_params[drift_idx] =
                        self.xf_params[drift_idx].clamp(0.0, cfg.drift_mod_max);
                }
            }

            // Accumulate signal-driven mutation_rate
            let mr = self
                .weights
                .compute_mutation_rate(&self.audio_features, &time_signals);
            let decay = self.weights._config.mutation_accum_decay;
            self.mutation_accum =
                ((self.mutation_accum + mr * dt) * (1.0 - decay * dt)).clamp(0.0, 1.0);

            // Evolution logic
            let time_since_last = time - self.last_mutation_time;
            let all_morphed = self
                .morph_xf_rates
                .iter()
                .all(|r| self.morph_progress * r >= 1.0)
                || self.morph_xf_rates.is_empty();

            let signal_trigger =
                !self.flame_locked && self.mutation_accum >= 1.0 && time_since_last >= 10.0;
            let cooldown = self.weights._config.mutation_cooldown;
            let time_trigger = !self.flame_locked && all_morphed && time_since_last >= cooldown;

            if signal_trigger || time_trigger {
                self.mutation_accum = 0.0;
                let reason = if signal_trigger { "signal" } else { "auto" };
                self.genome_history.push(self.genome.clone());
                if self.genome_history.len() > 10 {
                    self.genome_history.remove(0);
                }
                let (pa, pb, community) = self.pick_breeding_parents();
                self.genome = FlameGenome::mutate(
                    &pa,
                    &pb,
                    &community,
                    &self.audio_features,
                    &self.weights._config,
                    &self.favorite_profile,
                    &mut Some(&mut self.taste_engine),
                );
                let genomes_dir = project_dir().join("genomes");
                self.lineage_cache.register_and_save(
                    &self.genome.name,
                    &self.genome.parent_a,
                    &self.genome.parent_b,
                    self.genome.generation,
                    &genomes_dir,
                );
                let history_dir = genomes_dir.join("history");
                if let Err(e) = self.genome.save(&history_dir) {
                    eprintln!("[history] save error: {e}");
                }
                self.archive_genome();
                self.last_mutation_time = self.start.elapsed().as_secs_f32();
                self.begin_morph();
                eprintln!(
                    "[evolve:{reason}] t={:.1}s → {} (gen {})",
                    self.start.elapsed().as_secs_f32(),
                    self.genome.name,
                    self.genome.generation,
                );
            }

            // Periodic favorite profile refresh (every 30s)
            let profile_scan_interval = 30.0;
            if time - self.last_profile_scan >= profile_scan_interval {
                self.favorite_profile = scan_favorite_profile();
                self.last_profile_scan = time;
            }

            // Info mode
            if self.audio_info {
                self.audio_info_timer += dt;
                if self.audio_info_timer >= 0.1 {
                    self.audio_info_timer = 0.0;
                    self.audio_samples.push(self.audio_features.clone());
                    let f = &self.audio_features;
                    eprintln!(
                        "[info] bass={:.3} mids={:.3} highs={:.3} energy={:.3} beat={:.2} accum={:.2}",
                        f.bass, f.mids, f.highs, f.energy, f.beat, f.beat_accum
                    );
                }
            }
        }

        // Build uniforms
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
            }) | (self.weights._config.iterations_per_thread.clamp(10, 2000)
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

        self.prev_zoom = self.globals[1];

        // Adaptive compute budget
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

        // Jacobian importance sampling
        let xf_write_len = self.num_transforms * PARAMS_PER_XF;
        let xf_len = xf_write_len.min(self.xf_params.len());
        let final_xf_params: Vec<f32> = {
            let jac_strength = self.weights._config.jacobian_weight_strength;
            if jac_strength > 0.001 && self.num_transforms > 0 {
                let mut adjusted: Vec<f32> = self.xf_params[..xf_len].to_vec();
                let n = (xf_len / PARAMS_PER_XF).min(self.num_transforms);
                let mut weights: Vec<f32> = Vec::with_capacity(n);
                for i in 0..n {
                    let base = i * PARAMS_PER_XF;
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
                        adjusted[i * PARAMS_PER_XF] = jw / total;
                    }
                }
                adjusted
            } else {
                self.xf_params[..xf_len].to_vec()
            }
        };

        // Accumulation uniforms
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

        // Histogram CDF uniforms
        let hist_cdf_uniforms: [f32; 4] = [
            self.gpu_width as f32,
            self.gpu_height as f32,
            self.globals[3],
            (self.gpu_width * self.gpu_height) as f32,
        ];

        if self.morph_burst_frames > 0 {
            self.morph_burst_frames -= 1;
        }

        // Build HUD data
        let time = self.start.elapsed().as_secs_f32();
        let hud_time_signals = crate::weights::TimeSignals::compute(
            time,
            time - self.last_mutation_time,
            self.random_walk,
        );
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
            num_transforms: self.num_transforms,
            audio_bass: self.audio_features.bass,
            audio_mids: self.audio_features.mids,
            audio_highs: self.audio_features.highs,
            audio_energy: self.audio_features.energy,
            audio_beat: self.audio_features.beat,
            audio_beat_accum: self.audio_features.beat_accum,
            audio_change: self.audio_features.change,
            time_slow: hud_time_signals.time_slow,
            time_med: hud_time_signals.time_med,
            time_fast: hud_time_signals.time_fast,
            time_noise: hud_time_signals.time_noise,
            time_drift: hud_time_signals.time_drift,
            time_flutter: hud_time_signals.time_flutter,
            time_walk: hud_time_signals.time_walk,
            time_envelope: hud_time_signals.time_envelope,
            transform_weights: {
                let mut w = [0.0f32; 12];
                for (i, slot) in w.iter_mut().enumerate().take(self.num_transforms.min(12)) {
                    let idx = i * PARAMS_PER_XF;
                    if idx < self.xf_params.len() {
                        *slot = self.xf_params[idx];
                    }
                }
                w
            },
            transform_variations: {
                let mut vars = [[0.0f32; 26]; 12];
                for (i, xf_vars) in vars
                    .iter_mut()
                    .enumerate()
                    .take(self.num_transforms.min(12))
                {
                    let base = i * PARAMS_PER_XF + 14;
                    for (v, slot) in xf_vars.iter_mut().enumerate() {
                        let idx = base + v;
                        if idx < self.xf_params.len() {
                            *slot = self.xf_params[idx];
                        }
                    }
                }
                vars
            },
            hud_fade_delay: self.weights._config.hud_fade_delay,
            hud_fade_duration: self.weights._config.hud_fade_duration,
        };

        // Build egui UI and tessellate
        let (egui_primitives, egui_textures_delta, egui_pixels_per_point) =
            self.build_egui_frame(&hud, raw_input);

        // Merge pending texture deltas from dropped frames
        let mut egui_textures_delta = egui_textures_delta;
        if !self.pending_egui_textures.set.is_empty() || !self.pending_egui_textures.free.is_empty()
        {
            let mut merged = std::mem::take(&mut self.pending_egui_textures);
            merged.set.append(&mut egui_textures_delta.set);
            merged.free.append(&mut egui_textures_delta.free);
            egui_textures_delta = merged;
        }

        let frame_data = FrameData {
            uniforms: final_uniforms,
            xf_params: final_xf_params,
            accum_uniforms,
            hist_cdf_uniforms,
            workgroups: computed_workgroups,
            run_compute: self.frame >= 3,
            egui_primitives,
            egui_textures_delta,
            egui_pixels_per_point,
        };
        match self
            .render_tx
            .try_send(RenderCommand::Render(Box::new(frame_data)))
        {
            Ok(()) => {}
            Err(mpsc::TrySendError::Full(RenderCommand::Render(dropped))) => {
                self.pending_egui_textures = dropped.egui_textures_delta;
            }
            Err(_) => {}
        }
        self.frame += 1;

        // Per-genome performance tracking
        self.genome_frame_count += 1;
        let genome_elapsed = self.genome_start_time.elapsed().as_secs_f32();
        let mut perf_skip = false;
        if genome_elapsed > 3.0 && self.genome_frame_count > 30 {
            let genome_fps = self.genome_frame_count as f32 / genome_elapsed;
            let min_fps = self.weights._config.min_genome_fps;
            let good_fps = self.weights._config.perf_good_fps;

            let comp = crate::taste::CompositionFeatures::extract(&self.genome);
            let mut perf_features = comp.to_vec();
            for xf in &self.genome.transforms {
                let tf = crate::taste::TransformFeatures::extract(xf);
                perf_features.extend(tf.to_vec());
            }
            perf_features.resize(5 + 6 * 8, 0.0);
            perf_features.push(self.genome.symmetry.unsigned_abs() as f32);
            if genome_fps < min_fps && !self.flame_locked {
                self.perf_model.record_slow(&perf_features, 0.1);
                eprintln!(
                    "[perf-skip] {} avg {genome_fps:.1}fps < {min_fps}fps \
                     (xf={}, sym={}, frames={})",
                    self.genome.name,
                    self.genome.transform_count(),
                    self.genome.symmetry,
                    self.genome_frame_count,
                );
                perf_skip = true;
            } else if genome_fps >= good_fps {
                self.perf_model.record_fast(&perf_features, 0.1);
            }

            if self.perf_model.is_active() && self.genome_frame_count.is_multiple_of(600) {
                let _ = self
                    .perf_model
                    .save(&project_dir().join("genomes").join("perf_model.json"));
            }
        }

        // FPS logging
        self.fps_frame_count += 1;
        if self.fps_frame_count >= 60 {
            let elapsed = self.last_fps_time.elapsed();
            let fps = self.fps_frame_count as f64 / elapsed.as_secs_f64();
            let ms = elapsed.as_millis() as f64 / self.fps_frame_count as f64;
            log::info!("[perf] {fps:.1} fps ({ms:.1}ms/frame)");
            self.fps_frame_count = 0;
            self.last_fps_time = Instant::now();
        }

        // Perf log
        let ms_per_frame = dt * 1000.0;
        let is_slow = ms_per_frame > 33.0;
        let is_periodic = self.frame.is_multiple_of(300);
        if (is_slow || is_periodic)
            && let Some(ref mut log) = self.perf_log
        {
            let tag = if is_slow { "SLOW" } else { "ok" };
            let _ = writeln!(
                log,
                "[{}] f={} {:.1}ms/f {:.0}fps wg={} morph={:.2} burst={} decay={:.3} | {}",
                tag,
                self.frame,
                ms_per_frame,
                1.0 / dt.max(0.001),
                computed_workgroups,
                self.morph_progress,
                self.morph_burst_frames,
                decay,
                self.genome.perf_summary()
            );
        }

        // Perf-skip: auto-evolve away from expensive genomes
        if perf_skip {
            self.genome_history.push(self.genome.clone());
            if self.genome_history.len() > 10 {
                self.genome_history.remove(0);
            }
            let (pa, pb, community) = self.pick_breeding_parents();
            self.genome = FlameGenome::mutate(
                &pa,
                &pb,
                &community,
                &self.audio_features,
                &self.weights._config,
                &self.favorite_profile,
                &mut Some(&mut self.taste_engine),
            );
            let genomes_dir = project_dir().join("genomes");
            self.lineage_cache.register_and_save(
                &self.genome.name,
                &self.genome.parent_a,
                &self.genome.parent_b,
                self.genome.generation,
                &genomes_dir,
            );
            self.last_mutation_time = self.start.elapsed().as_secs_f32();
            self.begin_morph();
            eprintln!(
                "[auto-evolve] t={:.1}s → {} (gen {}) [perf skip]",
                self.start.elapsed().as_secs_f32(),
                self.genome.name,
                self.genome.generation,
            );
        }
    }
}

// ── UI Thread Loop ──

fn ui_thread_loop(rx: mpsc::Receiver<UiEvent>, mut state: UiState) {
    let mut last_tick = Instant::now();
    loop {
        match rx.recv() {
            Ok(UiEvent::EguiInput(raw_input)) => {
                // 120fps cap — skip if too soon since last tick
                if state.frame >= 3 && last_tick.elapsed() < Duration::from_micros(8333) {
                    continue;
                }
                last_tick = Instant::now();
                state.tick(raw_input);
            }
            Ok(UiEvent::CursorMoved { x, y }) => {
                state.on_cursor_moved(x, y);
            }
            Ok(UiEvent::KeyPressed(key)) => {
                state.on_key(&key);
            }
            Ok(UiEvent::Resize { width, height }) => {
                state.on_resize(width, height);
                // Forward resize to render thread too
                let _ = state
                    .render_tx
                    .send(RenderCommand::Resize { width, height });
            }
            Ok(UiEvent::Shutdown) | Err(_) => {
                let _ = state.render_tx.send(RenderCommand::Shutdown);
                eprintln!("[ui-thread] shutdown");
                break;
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let init = self.init_state.take().expect("init_state consumed twice");

        let attrs = WindowAttributes::default()
            .with_title("Shader Playground")
            .with_inner_size(winit::dpi::LogicalSize::new(
                init.weights._config.window_width,
                init.weights._config.window_height,
            ))
            .with_resizable(true);
        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        // Register as a proper macOS application so it appears in window pickers / Dock
        #[cfg(target_os = "macos")]
        {
            use objc2::msg_send;
            use objc2::runtime::AnyObject;
            use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};
            use objc2_foundation::{MainThreadMarker, NSString};

            let mtm = MainThreadMarker::new().expect("must be on main thread");
            let ns_app = NSApplication::sharedApplication(mtm);
            ns_app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
            #[allow(deprecated)]
            ns_app.activateIgnoringOtherApps(true);

            let process_info: *mut AnyObject =
                unsafe { msg_send![objc2::class!(NSProcessInfo), processInfo] };
            let name = NSString::from_str("Shader Playground");
            let _: () = unsafe { msg_send![process_info, setProcessName: &*name] };
        }

        let t = Instant::now();
        let mut gpu = Gpu::create(window.clone());
        let initial_width = gpu.config.width;
        let initial_height = gpu.config.height;
        eprintln!(
            "[boot] GPU initialized ({:.0}ms)",
            t.elapsed().as_secs_f64() * 1000.0
        );

        // File watcher
        let paths = vec![
            shader_path(),
            compute_path(),
            params_path(),
            weights_path(),
            audio_features_path(),
            project_dir().join("genomes").join("votes.json"),
        ];
        let watcher = match FileWatcher::new(&paths) {
            Ok(w) => Some(w),
            Err(e) => {
                eprintln!("warning: file watcher failed: {e}");
                None
            }
        };

        // Try loading a random genome (before gpu moves to render thread)
        let mut init = init;
        let genomes_dir = project_dir().join("genomes");
        if genomes_dir.exists()
            && let Ok(mut g) = FlameGenome::load_random(&genomes_dir)
        {
            g.adjust_transform_count(&init.weights._config);
            eprintln!("[genome] loaded: {}", g.name);
            let g_globals = g.flatten_globals(&init.weights._config);
            let g_xf = g.flatten_transforms();
            init.initial_globals = g_globals;
            init.initial_xf = g_xf;
            init.num_transforms = g.total_buffer_transforms();
            init.genome = g;
        }

        // Upload palette + resize transform buffer while we still own gpu
        let palette_data = crate::genome::palette_rgba_data(&init.genome);
        upload_palette_texture(&gpu.queue, &gpu.palette_texture, &palette_data);
        gpu.resize_transform_buffer(init.num_transforms);

        // Spawn render thread
        let (render_tx, render_rx) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("render".into())
            .spawn(move || render_thread_loop(render_rx, gpu))
            .expect("failed to spawn render thread");

        // Use configured dimensions
        self.gpu_width = init.weights._config.window_width.max(1);
        self.gpu_height = init.weights._config.window_height.max(1);

        eprintln!(
            "[debug] window surface={}x{}, gpu_dims={}x{}, samples_per_frame={}, xf_range={}-{}, iters={}",
            initial_width,
            initial_height,
            self.gpu_width,
            self.gpu_height,
            init.weights._config.samples_per_frame,
            init.weights._config.transform_count_min,
            init.weights._config.transform_count_max,
            init.weights._config.iterations_per_thread,
        );

        // Initialize egui — State stays on main thread, Context goes to UI thread
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        self.egui_state = Some(egui_state);

        // Create UiState and spawn UI thread
        let gpu_w = self.gpu_width;
        let gpu_h = self.gpu_height;
        let mut ui_state = UiState::new(render_tx, egui_ctx, init, watcher, gpu_w, gpu_h);

        // Build initial taste model before spawning
        let t = Instant::now();
        ui_state.rebuild_taste_model();
        eprintln!(
            "[boot] taste model rebuilt ({:.0}ms)",
            t.elapsed().as_secs_f64() * 1000.0
        );

        let (ui_tx, ui_rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("ui".into())
            .spawn(move || ui_thread_loop(ui_rx, ui_state))
            .expect("failed to spawn UI thread");
        self.ui_tx = Some(ui_tx);

        self.window = Some(window.clone());
        window.request_redraw();

        eprintln!(
            "[boot] total startup: {:.0}ms",
            Instant::now().elapsed().as_secs_f64() * 1000.0
        );
        eprintln!("shader playground running — edit playground.wgsl or flame_compute.wgsl");
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        // Forward cursor position to UI thread for HUD fade
        if let WindowEvent::CursorMoved { position, .. } = &event
            && let Some(tx) = &self.ui_tx
        {
            let _ = tx.send(UiEvent::CursorMoved {
                x: position.x as f32,
                y: position.y as f32,
            });
        }

        // Feed ALL events to egui — never skip (UI thread needs them via take_egui_input)
        if let Some(egui_state) = &mut self.egui_state
            && let Some(window) = &self.window
        {
            let _ = egui_state.on_window_event(window, &event);
        }

        match event {
            WindowEvent::CloseRequested => {
                if let Some(tx) = self.ui_tx.take() {
                    let _ = tx.send(UiEvent::Shutdown);
                }
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    self.gpu_width = size.width;
                    self.gpu_height = size.height;
                    if let Some(tx) = &self.ui_tx {
                        let _ = tx.send(UiEvent::Resize {
                            width: size.width,
                            height: size.height,
                        });
                    }
                }
            }
            WindowEvent::CursorMoved { .. } => {} // forwarded above
            WindowEvent::MouseInput { .. } => {}
            // Forward key presses to UI thread
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                let key_str = match &logical_key {
                    Key::Named(NamedKey::Space) => Some("Space"),
                    Key::Named(NamedKey::Backspace) => Some("Backspace"),
                    Key::Named(NamedKey::ArrowUp) => Some("ArrowUp"),
                    Key::Named(NamedKey::ArrowDown) => Some("ArrowDown"),
                    Key::Character(c) => match c.as_str() {
                        "s" | "l" | "1" | "2" | "3" | "4" | "f" | "a" | "c" | "i" => {
                            Some(c.as_str())
                        }
                        _ => None,
                    },
                    _ => None,
                };
                if let Some(key) = key_str
                    && let Some(tx) = &self.ui_tx
                {
                    let _ = tx.send(UiEvent::KeyPressed(key.to_string()));
                }
            }
            WindowEvent::RedrawRequested => {
                // Collect egui input and send to UI thread — the UI thread does all heavy work
                if let Some(egui_state) = &mut self.egui_state
                    && let Some(window) = &self.window
                {
                    let raw = egui_state.take_egui_input(window);
                    if let Some(tx) = &self.ui_tx {
                        let _ = tx.send(UiEvent::EguiInput(raw));
                    }
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}
// ── Main ──

fn main() {
    env_logger::init();

    // Auto-select System Audio (ScreenCaptureKit) — skip terminal picker
    eprintln!("[audio] Auto-selecting System Audio (ScreenCaptureKit)");
    let capture = match AudioCapture::new_system_audio() {
        Ok(cap) => Some(cap),
        Err(e) => {
            eprintln!("[audio] SCK capture failed: {e} — falling back to device picker");
            // Fall back to interactive picker if SCK fails
            match device_picker::run() {
                device_picker::Selection::SystemAudio => match AudioCapture::new_system_audio() {
                    Ok(cap) => Some(cap),
                    Err(e2) => {
                        eprintln!("[audio] SCK retry failed: {e2} (visuals-only mode)");
                        None
                    }
                },
                device_picker::Selection::CpalDevice(device, is_input) => {
                    match AudioCapture::from_device(device, is_input) {
                        Ok(cap) => Some(cap),
                        Err(e2) => {
                            eprintln!("[audio] capture failed: {e2} (visuals-only mode)");
                            None
                        }
                    }
                }
                device_picker::Selection::Cancelled => {
                    eprintln!("[audio] cancelled — visuals-only mode");
                    None
                }
            }
        }
    };

    // Spawn background audio processing thread
    if let Some(capture) = capture {
        audio::spawn_audio_thread(capture, audio_features_path());
    }

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
