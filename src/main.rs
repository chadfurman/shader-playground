use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::time::Instant;
use std::{fs, mem};

use bytemuck::{Pod, Zeroable};
use notify::{Event, EventKind, RecursiveMode, Watcher};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes};

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
use crate::weights::Weights;

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
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
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
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
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

        // Persistent point state: 3 f32s per thread (x, y, color_idx)
        // Max 8192 workgroups * 256 threads = 2M threads
        let max_threads: u64 = 8192 * 256;
        let point_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("point_state"),
            size: max_threads * 12, // 3 f32s * 4 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Histogram buffer: 6 u32s per pixel (density + R + G + B + vx + vy)
        let histogram_buffer = create_histogram_buffer(&device, config.width, config.height);

        // Initial transform buffer (6 transforms * 42 floats * 4 bytes)
        let transform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transforms"),
            size: (6 * 42 * 4) as u64,
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

        let histogram_cdf_src = fs::read_to_string(project_dir().join("histogram_cdf.wgsl"))
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

    fn reload_shader(&mut self, src: &str) {
        self.pipeline =
            create_render_pipeline(&self.device, &self.pipeline_layout, src, self.config.format);
    }

    fn reload_compute_shader(&mut self, src: &str) {
        self.compute_pipeline =
            create_compute_pipeline(&self.device, &self.compute_pipeline_layout, src);
    }

    fn resize_transform_buffer(&mut self, num_transforms: usize) {
        let size = (num_transforms.max(1) * 42 * 4) as u64;
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

    fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let screen_view = frame.texture.create_view(&Default::default());

        let (target_view, bind_group) = if self.ping {
            (&self.frame_a, &self.bind_group_a)
        } else {
            (&self.frame_b, &self.bind_group_b)
        };

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // 1. Clear histogram
        encoder.clear_buffer(&self.histogram_buffer, 0, None);

        // 2. Compute pass: run the chaos game
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flame"),
                ..Default::default()
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups(self.workgroups, 1, 1);
        }

        // 2.4. Clear max density for per-image normalization
        encoder.clear_buffer(&self.max_density_buffer, 0, None);

        // 2.5. Accumulation pass: blend histogram into persistent buffer
        {
            let mut apass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("accumulation"),
                ..Default::default()
            });
            apass.set_pipeline(&self.accumulation_pipeline);
            apass.set_bind_group(0, &self.accumulation_bind_group, &[]);
            let wg_x = self.config.width.div_ceil(16);
            let wg_y = self.config.height.div_ceil(16);
            apass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // 2.75. Histogram equalization: bin densities + prefix sum CDF
        encoder.clear_buffer(&self.hist_bins_buffer, 0, None);
        {
            let mut hpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
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
            let mut hpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("histogram_cdf"),
                ..Default::default()
            });
            hpass.set_pipeline(&self.histogram_cdf_sum_pipeline);
            hpass.set_bind_group(0, &self.histogram_cdf_bind_group, &[]);
            hpass.dispatch_workgroups(1, 1, 1);
        }

        // 3. Render to feedback texture
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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

        // 4. Copy to screen
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        self.ping = !self.ping;
    }
}

// ── Helper Functions ──

pub fn project_dir() -> PathBuf {
    // When running as a .app bundle, use Contents/Resources
    if let Ok(exe) = std::env::current_exe()
        && let Some(macos_dir) = exe.parent()
    {
        let resources = macos_dir.with_file_name("Resources");
        if resources.join("weights.json").exists() {
            return resources;
        }
    }
    // Development mode: walk up from cwd looking for Cargo.toml
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

fn shader_path() -> PathBuf {
    project_dir().join("playground.wgsl")
}

fn compute_path() -> PathBuf {
    project_dir().join("flame_compute.wgsl")
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
    project_dir().join("accumulation.wgsl")
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
    Weights::load(&weights_path()).unwrap_or_default()
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

// ── App ──

struct App {
    gpu: Option<Gpu>,
    window: Option<Arc<Window>>,
    watcher: Option<FileWatcher>,
    start: Instant,
    frame: u32,
    mouse: [f32; 2],
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
    flame_locked: bool,      // true = imported flame, skip auto-evolve/mutate
    morph_burst_frames: u32, // extra compute passes after mutation for faster fill
    random_walk: f32,
    // Ease-in-out genome morph
    morph_start_globals: [f32; 20],
    morph_start_xf: Vec<f32>,
    morph_base_globals: [f32; 20], // current interpolated genome base (no audio)
    morph_base_xf: Vec<f32>,       // current interpolated xf base (no audio)
    morph_progress: f32,           // 0.0 → 1.0
    morph_xf_rates: Vec<f32>,      // per-transform morph speed multiplier (0.5 - 2.0)
    favorite_profile: Option<FavoriteProfile>,
    vote_ledger: VoteLedger,
    lineage_cache: crate::votes::LineageCache,
    taste_engine: crate::taste::TasteEngine,
    last_profile_scan: f32,
    perf_log: Option<std::fs::File>,
    prev_zoom: f32,
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl App {
    fn new() -> Self {
        // Try loading a flame file first, then fall back to seeds, then default
        let weights = load_weights();
        let favorite_profile = Self::scan_favorite_profile();
        let genomes_root = project_dir().join("genomes");
        // Ensure genome subdirectories exist
        let _ = std::fs::create_dir_all(genomes_root.join("voted"));
        let _ = std::fs::create_dir_all(genomes_root.join("history"));
        if weights._config.archive_on_startup {
            archive_history_if_needed(&genomes_root, weights._config.archive_threshold_mb);
        }
        let vote_ledger = VoteLedger::load(&genomes_root);
        let lineage_cache = crate::votes::LineageCache::load(&genomes_root);
        let flames_dir = project_dir().join("genomes").join("flames");
        let seeds_dir = project_dir().join("genomes").join("seeds");
        let genome = crate::flam3::load_random_flame(&flames_dir)
            .or_else(|_| FlameGenome::load_random(&seeds_dir))
            .unwrap_or_else(|_| FlameGenome::default_genome());
        let initial_globals = genome.flatten_globals(&weights._config);
        let initial_xf = genome.flatten_transforms();
        let num_transforms = genome.total_buffer_transforms();
        Self {
            gpu: None,
            window: None,
            watcher: None,
            start: Instant::now(),
            frame: 0,
            mouse: [0.5, 0.5],
            globals: initial_globals,
            xf_params: initial_xf.clone(),
            num_transforms,
            last_frame_time: Instant::now(),
            genome,
            genome_history: Vec::new(),
            audio_features: AudioFeatures::default(),
            weights,
            audio_enabled: true,
            audio_info: false,
            audio_info_timer: 0.0,
            audio_samples: Vec::new(),
            mutation_accum: 0.0,
            last_mutation_time: 0.0,
            flame_locked: false,
            morph_burst_frames: 0,
            random_walk: 0.0,
            // Start fully morphed (no transition at launch)
            morph_start_globals: initial_globals,
            morph_start_xf: initial_xf.clone(),
            morph_base_globals: initial_globals,
            morph_base_xf: initial_xf,
            morph_progress: 1.0,
            morph_xf_rates: Vec::new(),
            favorite_profile,
            vote_ledger,
            lineage_cache,
            taste_engine: crate::taste::TasteEngine::new(),
            last_profile_scan: 0.0,
            perf_log: std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("perf.log")
                .ok(),
            prev_zoom: 3.0,
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
        self.taste_engine.rebuild(&refs, recent_memory);

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

    /// Pick two parents for breeding + a community genome.
    /// Uses lineage cache to enforce minimum genetic distance between parents.
    fn pick_breeding_parents(&self) -> (FlameGenome, FlameGenome, Option<FlameGenome>) {
        let genomes_dir = project_dir().join("genomes");
        let threshold = self.weights._config.vote_blacklist_threshold;
        let min_distance = self.weights._config.min_breeding_distance;
        let max_depth = self.weights._config.max_lineage_depth;

        // Parent A: prefer voted genome, fallback to random saved
        let parent_a = self
            .vote_ledger
            .pick_voted(threshold)
            .and_then(|p| FlameGenome::load(&p).ok())
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
        // Snapshot wherever the morph currently is as our new start
        self.morph_start_globals = self.morph_base_globals;
        self.morph_start_xf = self.morph_base_xf.clone();
        self.morph_progress = 0.0;

        // Ensure buffers can hold max of current and target transforms
        let target_xf = self.genome.flatten_transforms();
        let max_xf = (self.xf_params.len().max(target_xf.len())) / 42;
        if max_xf != self.num_transforms {
            self.num_transforms = max_xf;
            if let Some(gpu) = &mut self.gpu {
                gpu.resize_transform_buffer(self.num_transforms);
            }
        }
        // Pad start vectors to match
        self.morph_start_xf.resize(max_xf * 42, 0.0);

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
        if let Some(gpu) = &self.gpu {
            let palette_data = crate::genome::palette_rgba_data(&self.genome);
            upload_palette_texture(&gpu.queue, &gpu.palette_texture, &palette_data);
        }
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
            self.gpu.as_mut().unwrap().reload_shader(&src);
            eprintln!("[shader] reloaded");
        }
        if reload_compute {
            let src = load_compute_source();
            self.gpu.as_mut().unwrap().reload_compute_shader(&src);
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
            self.weights = load_weights();
            if let Some(gpu) = &mut self.gpu {
                gpu.workgroups = self.weights._config.samples_per_frame;
            }
            eprintln!(
                "[weights] reloaded (samples_per_frame={}, morph={}s, cooldown={}s)",
                self.weights._config.samples_per_frame,
                self.weights._config.morph_duration,
                self.weights._config.mutation_cooldown
            );
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
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title("Shader Playground")
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.weights._config.window_width,
                self.weights._config.window_height,
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

            // Set the process name so window pickers show "Shader Playground"
            let process_info: *mut AnyObject =
                unsafe { msg_send![objc2::class!(NSProcessInfo), processInfo] };
            let name = NSString::from_str("Shader Playground");
            let _: () = unsafe { msg_send![process_info, setProcessName: &*name] };
        }

        self.gpu = Some(Gpu::create(window.clone()));
        self.window = Some(window);

        // Watch all shader files + params
        let paths = vec![
            shader_path(),
            compute_path(),
            params_path(),
            weights_path(),
            audio_features_path(),
            project_dir().join("genomes").join("votes.json"),
        ];
        match FileWatcher::new(&paths) {
            Ok(w) => self.watcher = Some(w),
            Err(e) => eprintln!("warning: file watcher failed: {e}"),
        }

        // Try loading a random genome
        let genomes_dir = project_dir().join("genomes");
        if genomes_dir.exists()
            && let Ok(g) = FlameGenome::load_random(&genomes_dir)
        {
            eprintln!("[genome] loaded: {}", g.name);
            self.genome = g;
            // Snap (no morph) on initial load
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
            if let Some(gpu) = &mut self.gpu {
                gpu.resize_transform_buffer(self.num_transforms);
            }
        }

        // Upload palette for the initial genome
        if let Some(gpu) = &self.gpu {
            let palette_data = crate::genome::palette_rgba_data(&self.genome);
            upload_palette_texture(&gpu.queue, &gpu.palette_texture, &palette_data);
        }

        // Build initial taste model from voted/imported genomes
        self.rebuild_taste_model();

        eprintln!("shader playground running — edit playground.wgsl or flame_compute.wgsl");
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(gpu) = &self.gpu {
                    self.mouse = [
                        position.x as f32 / gpu.config.width as f32,
                        position.y as f32 / gpu.config.height as f32,
                    ];
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {}
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match logical_key {
                Key::Named(NamedKey::Space) => {
                    self.genome_history.push(self.genome.clone());
                    if self.genome_history.len() > 10 {
                        self.genome_history.remove(0);
                    }
                    self.flame_locked = false; // unlock on manual mutate
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
                    // Auto-save to history
                    let history_dir = genomes_dir.join("history");
                    if let Err(e) = self.genome.save(&history_dir) {
                        eprintln!("[history] save error: {e}");
                    }
                    self.last_mutation_time = self.start.elapsed().as_secs_f32();
                    self.begin_morph();
                    eprintln!(
                        "[evolve] → {} (gen {})",
                        self.genome.name, self.genome.generation
                    );
                }
                Key::Named(NamedKey::Backspace) => {
                    if let Some(prev) = self.genome_history.pop() {
                        self.genome = prev;
                        self.begin_morph();
                        eprintln!("[revert] back to previous");
                    }
                }
                Key::Named(NamedKey::ArrowUp) => {
                    let dir = project_dir().join("genomes");
                    let score = self.vote_ledger.vote(&self.genome, 1, &dir);
                    self.favorite_profile = Self::scan_favorite_profile();
                    self.rebuild_taste_model();
                    eprintln!("[vote] {} → +1 (score: {})", self.genome.name, score);
                }
                Key::Named(NamedKey::ArrowDown) => {
                    let dir = project_dir().join("genomes");
                    let score = self.vote_ledger.vote(&self.genome, -1, &dir);
                    self.favorite_profile = Self::scan_favorite_profile();
                    self.rebuild_taste_model();
                    eprintln!("[vote] {} → -1 (score: {})", self.genome.name, score);
                }
                Key::Character(ref c) => match c.as_str() {
                    "s" => {
                        let dir = project_dir().join("genomes");
                        match self.genome.save(&dir) {
                            Ok(p) => eprintln!("[save] {}", p.display()),
                            Err(e) => eprintln!("[save] error: {e}"),
                        }
                        // Refresh favorite profile after saving
                        self.favorite_profile = Self::scan_favorite_profile();
                        self.last_profile_scan = self.start.elapsed().as_secs_f32();
                    }
                    "l" => {
                        let dir = project_dir().join("genomes");
                        match FlameGenome::load_random(&dir) {
                            Ok(g) => {
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
                        let idx: usize = c.as_str().parse::<usize>().unwrap() - 1;
                        if idx < self.num_transforms {
                            let base = idx * 42; // weight is first field
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
                            Ok(g) => {
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
                    "i" => {
                        self.audio_info = !self.audio_info;
                        if self.audio_info {
                            self.audio_samples.clear();
                            eprintln!(
                                "[info] ON — recording audio features (press i again to save)"
                            );
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
                },
                _ => {}
            },
            WindowEvent::RedrawRequested => {
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
                    let time_signals = crate::weights::TimeSignals::compute(
                        time,
                        time_since_mutation,
                        self.random_walk,
                    );

                    // Advance morph progress
                    let morph_dur = self.weights._config.morph_duration;
                    // morph_progress can exceed 1.0 so slow transforms finish
                    let min_rate = self
                        .morph_xf_rates
                        .iter()
                        .copied()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap_or(1.0)
                        .max(0.1);
                    let morph_done_at = 1.0 / min_rate; // e.g. 2.5 for rate=0.4
                    if self.morph_progress < morph_done_at {
                        self.morph_progress =
                            (self.morph_progress + dt / morph_dur).min(morph_done_at);
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
                    let num_xf = max_len / 42;
                    for xi in 0..num_xf {
                        // Each transform morphs at its own rate
                        let rate = self.morph_xf_rates.get(xi).copied().unwrap_or(1.0);
                        let xf_t = smoothstep((self.morph_progress * rate).min(1.0));
                        let base = xi * 42;
                        for j in 0..42 {
                            let idx = base + j;
                            if idx < max_len {
                                self.morph_base_xf[idx] = padded_start[idx]
                                    + (padded_genome[idx] - padded_start[idx]) * xf_t;
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

                    // Set final values directly (morph handles smoothing)
                    self.globals = modulated_globals;
                    self.xf_params = modulated_xf;

                    // Apply variation CRISPR — scale/zero out variations from config
                    self.weights
                        ._config
                        .apply_variation_scales(&mut self.xf_params, self.num_transforms);

                    // Auto-evolve when ALL transforms finish morphing (disabled when flame_locked)
                    let all_morphed = self
                        .morph_xf_rates
                        .iter()
                        .all(|r| self.morph_progress * r >= 1.0)
                        || self.morph_xf_rates.is_empty();
                    if !self.flame_locked && all_morphed {
                        let time_since_last = time - self.last_mutation_time;
                        let cooldown = self.weights._config.mutation_cooldown;
                        if time_since_last >= cooldown {
                            self.genome_history.push(self.genome.clone());
                            if self.genome_history.len() > 10 {
                                self.genome_history.remove(0);
                            }

                            // Breed two parents to produce offspring
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
                            // Auto-save to history
                            let history_dir = genomes_dir.join("history");
                            if let Err(e) = self.genome.save(&history_dir) {
                                eprintln!("[history] save error: {e}");
                            }
                            self.last_mutation_time = self.start.elapsed().as_secs_f32();
                            self.begin_morph();
                            eprintln!(
                                "[auto-evolve] → {} (gen {})",
                                self.genome.name, self.genome.generation
                            );
                        }
                    }

                    // Periodic favorite profile refresh (every 30s)
                    let profile_scan_interval = 30.0;
                    if time - self.last_profile_scan >= profile_scan_interval {
                        self.favorite_profile = Self::scan_favorite_profile();
                        self.last_profile_scan = time;
                    }

                    // Info mode — collect samples + print
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

                let gpu = match &mut self.gpu {
                    Some(g) => g,
                    None => return,
                };

                // Write uniforms
                let uniforms = Uniforms {
                    time: self.start.elapsed().as_secs_f32(),
                    frame: self.frame,
                    resolution: [gpu.config.width as f32, gpu.config.height as f32],
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
                };

                gpu.queue
                    .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

                self.prev_zoom = self.globals[1]; // zoom is globals[1]

                // Adaptive compute budget: target ~4 effective transforms worth of work.
                // Scale both workgroups AND iterations to keep frame time manageable
                // while preserving point density (fewer iterations = same points, less work).
                let effective_xforms =
                    self.genome.transform_count() * (self.genome.symmetry.unsigned_abs().max(1));
                let budget_baseline = 4u32;
                let base_wg = self.weights._config.samples_per_frame;
                let base_iters = self.weights._config.iterations_per_thread;
                if effective_xforms > budget_baseline {
                    let ratio = budget_baseline as f32 / effective_xforms as f32;
                    // Split the scaling: sqrt on each so neither goes too low
                    let sqrt_ratio = ratio.sqrt();
                    gpu.workgroups = (base_wg as f32 * sqrt_ratio).max(256.0) as u32;
                    // Pack scaled iterations into has_final_xform upper bits
                    let scaled_iters = (base_iters as f32 * sqrt_ratio).max(40.0) as u32;
                    let has_final = if self.genome.final_transform.is_some() {
                        1u32
                    } else {
                        0u32
                    };
                    let uniforms_patched = Uniforms {
                        has_final_xform: has_final | (scaled_iters.clamp(10, 2000) << 16),
                        ..uniforms
                    };
                    gpu.queue.write_buffer(
                        &gpu.uniform_buffer,
                        0,
                        bytemuck::bytes_of(&uniforms_patched),
                    );
                } else {
                    gpu.workgroups = base_wg;
                };

                let xf_write_len = self.num_transforms * 42;
                let xf_slice = &self.xf_params[..xf_write_len.min(self.xf_params.len())];
                gpu.queue
                    .write_buffer(&gpu.transform_buffer, 0, bytemuck::cast_slice(xf_slice));

                // Write accumulation uniforms — faster decay during morph transition
                let base_decay = self.weights._config.accumulation_decay;
                let decay = if self.morph_burst_frames > 0 {
                    // Lerp from fast decay (0.7) back to normal (0.95) over burst period
                    let burst_t = self.morph_burst_frames as f32 / 60.0;
                    0.95 + (0.7 - 0.95) * burst_t
                } else {
                    base_decay
                };
                let accum_uniforms: [f32; 4] = [
                    gpu.config.width as f32,
                    gpu.config.height as f32,
                    decay,
                    0.0,
                ];
                gpu.queue.write_buffer(
                    &gpu.accumulation_uniform_buffer,
                    0,
                    bytemuck::cast_slice(&accum_uniforms),
                );

                // Write histogram CDF uniforms
                let hist_cdf_uniforms: [f32; 4] = [
                    gpu.config.width as f32,
                    gpu.config.height as f32,
                    self.globals[3], // flame_brightness
                    (gpu.config.width * gpu.config.height) as f32,
                ];
                gpu.queue.write_buffer(
                    &gpu.histogram_cdf_uniform_buffer,
                    0,
                    bytemuck::cast_slice(&hist_cdf_uniforms),
                );

                // Tick down morph burst (drives faster decay ramp)
                if self.morph_burst_frames > 0 {
                    self.morph_burst_frames -= 1;
                }

                gpu.render();
                self.frame += 1;

                // Log to perf.log: every slow frame (<30fps) + periodic baseline every 300 frames
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
                        gpu.workgroups,
                        self.morph_progress,
                        self.morph_burst_frames,
                        decay,
                        self.genome.perf_summary()
                    );
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

    // Interactive device picker before opening the window
    let capture = match device_picker::run() {
        device_picker::Selection::SystemAudio => match AudioCapture::new_system_audio() {
            Ok(cap) => Some(cap),
            Err(e) => {
                eprintln!("[audio] SCK capture failed: {e} (visuals-only mode)");
                None
            }
        },
        device_picker::Selection::CpalDevice(device, is_input) => {
            match AudioCapture::from_device(device, is_input) {
                Ok(cap) => Some(cap),
                Err(e) => {
                    eprintln!("[audio] capture failed: {e} (visuals-only mode)");
                    None
                }
            }
        }
        device_picker::Selection::Cancelled => {
            eprintln!("[audio] cancelled — visuals-only mode");
            None
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
