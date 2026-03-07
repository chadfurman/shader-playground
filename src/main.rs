use std::path::PathBuf;
use std::sync::{mpsc, Arc};
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
mod genome;
mod sck_audio;
mod weights;
use crate::genome::FlameGenome;
use crate::audio::{AudioCapture, AudioFeatures};
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
    has_final_xform: u32,
    globals: [f32; 4],   // speed, zoom, trail, flame_brightness
    kifs: [f32; 4],       // fold_angle, scale, brightness, drift_speed
    extra: [f32; 4],      // color_shift, vibrancy, bloom_intensity, symmetry
    extra2: [f32; 4],     // crossfade_alpha, reserved, reserved, reserved
}

// ── File Watcher ──

struct FileWatcher {
    rx: mpsc::Receiver<PathBuf>,
    _watcher: notify::RecommendedWatcher,
}

impl FileWatcher {
    fn new(paths: &[PathBuf]) -> Result<Self, String> {
        let (tx, rx) = mpsc::channel();
        let mut watcher = notify::recommended_watcher(
            move |res: Result<Event, notify::Error>| {
                if let Ok(event) = res {
                    if matches!(
                        event.kind,
                        EventKind::Modify(_) | EventKind::Create(_)
                    ) {
                        for path in &event.paths {
                            let _ = tx.send(path.clone());
                        }
                    }
                }
            },
        )
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
}

impl Gpu {
    fn create(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            },
        ))
        .expect("no GPU adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor::default(),
        ))
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

        // Histogram buffer: 4 u32s per pixel (density + R + G + B)
        let histogram_buffer = create_histogram_buffer(
            &device,
            config.width,
            config.height,
        );

        // Initial transform buffer (6 transforms * 32 floats * 4 bytes)
        let transform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transforms"),
            size: (6 * 32 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Render bind group layout (uniform + prev_frame + sampler + histogram read) ──
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("render"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT
                            | wgpu::ShaderStages::VERTEX,
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
                            sample_type: wgpu::TextureSampleType::Float {
                                filterable: true,
                            },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(
                            wgpu::SamplerBindingType::Filtering,
                        ),
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
                ],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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
        let pipeline = create_render_pipeline(
            &device,
            &pipeline_layout,
            &shader_src,
            format,
        );

        let compute_src = load_compute_source();
        let compute_pipeline = create_compute_pipeline(
            &device,
            &compute_pipeline_layout,
            &compute_src,
        );

        let bind_group_a = create_render_bind_group(
            &device,
            &bind_group_layout,
            &uniform_buffer,
            &frame_b,
            &sampler,
            &histogram_buffer,
            &crossfade_view,
        );
        let bind_group_b = create_render_bind_group(
            &device,
            &bind_group_layout,
            &uniform_buffer,
            &frame_a,
            &sampler,
            &histogram_buffer,
            &crossfade_view,
        );

        let compute_bind_group = create_compute_bind_group(
            &device,
            &compute_bind_group_layout,
            &histogram_buffer,
            &uniform_buffer,
            &transform_buffer,
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
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);

        let (a_tex, a, b_tex, b) = create_frame_textures(
            &self.device,
            w,
            h,
            self.config.format,
        );
        let (cf_tex, cf_view) = create_crossfade_texture(
            &self.device,
            w,
            h,
            self.config.format,
        );
        self.frame_a_tex = a_tex;
        self.frame_a = a;
        self.frame_b_tex = b_tex;
        self.frame_b = b;
        self.crossfade_tex = cf_tex;
        self.crossfade_view = cf_view;

        // Recreate histogram for new dimensions
        self.histogram_buffer = create_histogram_buffer(&self.device, w, h);

        self.rebuild_bind_groups();
    }

    fn reload_shader(&mut self, src: &str) {
        self.pipeline = create_render_pipeline(
            &self.device,
            &self.pipeline_layout,
            src,
            self.config.format,
        );
    }

    fn reload_compute_shader(&mut self, src: &str) {
        self.compute_pipeline = create_compute_pipeline(
            &self.device,
            &self.compute_pipeline_layout,
            src,
        );
    }

    fn resize_transform_buffer(&mut self, num_transforms: usize) {
        let size = (num_transforms.max(1) * 32 * 4) as u64;
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
            &self.histogram_buffer,
            &self.crossfade_view,
        );
        self.bind_group_b = create_render_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.uniform_buffer,
            &self.frame_a,
            &self.sampler,
            &self.histogram_buffer,
            &self.crossfade_view,
        );
        self.compute_bind_group = create_compute_bind_group(
            &self.device,
            &self.compute_bind_group_layout,
            &self.histogram_buffer,
            &self.uniform_buffer,
            &self.transform_buffer,
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
            let mut cpass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("flame"),
                    ..Default::default()
                },
            );
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups(512, 1, 1); // 512 * 256 = 131K threads
        }

        // 3. Render to feedback texture
        {
            let mut pass =
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("feedback"),
                    color_attachments: &[Some(
                        wgpu::RenderPassColorAttachment {
                            view: target_view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        },
                    )],
                    ..Default::default()
                });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // 4. Copy to screen
        {
            let mut pass =
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("blit"),
                    color_attachments: &[Some(
                        wgpu::RenderPassColorAttachment {
                            view: &screen_view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        },
                    )],
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

fn project_dir() -> PathBuf {
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

fn load_params() -> Vec<f32> {
    if let Ok(json) = fs::read_to_string(params_path()) {
        if let Ok(vals) = serde_json::from_str::<Vec<f32>>(&json) {
            return vals;
        }
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

fn create_histogram_buffer(
    device: &wgpu::Device,
    w: u32,
    h: u32,
) -> wgpu::Buffer {
    let pixel_count = w.max(1) as u64 * h.max(1) as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("histogram"),
        size: pixel_count * 4 * 4, // 4 u32s per pixel
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

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

fn create_render_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    prev_frame: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    histogram: &wgpu::Buffer,
    crossfade_view: &wgpu::TextureView,
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
                resource: histogram.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(crossfade_view),
            },
        ],
    })
}

fn create_compute_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    histogram: &wgpu::Buffer,
    uniform_buffer: &wgpu::Buffer,
    transform_buffer: &wgpu::Buffer,
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
        ],
    })
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
    globals: [f32; 12],
    target_globals: [f32; 12],
    xf_params: Vec<f32>,
    target_xf_params: Vec<f32>,
    num_transforms: usize,
    last_frame_time: Instant,
    genome: FlameGenome,
    genome_history: Vec<FlameGenome>,
    morph_rate_idx: usize,
    audio_features: AudioFeatures,
    weights: Weights,
    audio_enabled: bool,
    audio_info: bool,
    audio_info_timer: f32,
    audio_samples: Vec<AudioFeatures>,
    mutation_accum: f32,
    last_mutation_time: f32,
    random_walk: f32,
}

const MORPH_RATES: [f32; 15] = [
    0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 20.0, 35.0, 50.0,
];

impl App {
    fn new() -> Self {
        // Randomize startup — mutate default genome so every launch is unique
        let mut genome = FlameGenome::default_genome();
        for _ in 0..3 {
            genome = genome.mutate();
        }
        let initial_globals = genome.flatten_globals();
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
            target_globals: initial_globals,
            xf_params: initial_xf.clone(),
            target_xf_params: initial_xf,
            num_transforms,
            last_frame_time: Instant::now(),
            genome,
            genome_history: Vec::new(),
            morph_rate_idx: 2, // MORPH_RATES[2] = 0.05 — gradual, smooth crossfade
            audio_features: AudioFeatures::default(),
            weights: load_weights(),
            audio_enabled: true,
            audio_info: false,
            audio_info_timer: 0.0,
            audio_samples: Vec::new(),
            mutation_accum: 0.0,
            last_mutation_time: 0.0,
            random_walk: 0.0,
        }
    }

    /// Set morph targets from current genome, resize GPU buffer if needed.
    fn apply_genome_targets(&mut self) {
        self.target_globals = self.genome.flatten_globals();
        self.target_xf_params = self.genome.flatten_transforms();
        // num_transforms = max of current and target so dying transforms fade out
        let max_xf = (self.xf_params.len().max(self.target_xf_params.len())) / 32;
        if max_xf != self.num_transforms {
            self.num_transforms = max_xf;
            if let Some(gpu) = &mut self.gpu {
                gpu.resize_transform_buffer(self.num_transforms);
            }
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
            for i in 0..12.min(override_params.len()) {
                self.target_globals[i] = override_params[i];
            }
            eprintln!("[params] reloaded (overriding globals)");
        }
        if reload_weights {
            self.weights = load_weights();
            eprintln!("[weights] reloaded");
        }
        if reload_features {
            if let Ok(json) = fs::read_to_string(audio_features_path()) {
                if let Ok(f) = serde_json::from_str::<AudioFeatures>(&json) {
                    self.audio_features = f;
                }
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title("shader playground")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.gpu = Some(Gpu::create(window.clone()));
        self.window = Some(window);

        // Watch all shader files + params
        let paths = vec![shader_path(), compute_path(), params_path(), weights_path(), audio_features_path()];
        match FileWatcher::new(&paths) {
            Ok(w) => self.watcher = Some(w),
            Err(e) => eprintln!("warning: file watcher failed: {e}"),
        }

        // Try loading a random genome
        let genomes_dir = project_dir().join("genomes");
        if genomes_dir.exists() {
            if let Ok(g) = FlameGenome::load_random(&genomes_dir) {
                eprintln!("[genome] loaded: {}", g.name);
                self.genome = g;
                // Snap (no morph) on initial load
                self.globals = self.genome.flatten_globals();
                self.xf_params = self.genome.flatten_transforms();
                self.apply_genome_targets();
            }
        }

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
                    // Evolve: mutate and crossfade
                    self.genome_history.push(self.genome.clone());
                    if self.genome_history.len() > 10 {
                        self.genome_history.remove(0);
                    }
                    self.genome = self.genome.mutate();
                    self.last_mutation_time = self.start.elapsed().as_secs_f32();
                    self.apply_genome_targets();
                    eprintln!("[evolve] → {}", self.genome.name);
                }
                Key::Named(NamedKey::Backspace) => {
                    // Revert to previous genome
                    if let Some(prev) = self.genome_history.pop() {
                        self.genome = prev;
                        self.apply_genome_targets();
                        eprintln!("[revert] back to previous");
                    }
                }
                Key::Character(ref c) => match c.as_str() {
                    "s" => {
                        let dir = project_dir().join("genomes");
                        match self.genome.save(&dir) {
                            Ok(p) => eprintln!("[save] {}", p.display()),
                            Err(e) => eprintln!("[save] error: {e}"),
                        }
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
                                self.apply_genome_targets();
                            }
                            Err(e) => eprintln!("[load] error: {e}"),
                        }
                    }
                    "1" | "2" | "3" | "4" => {
                        let idx: usize = c.as_str().parse::<usize>().unwrap() - 1;
                        if idx < self.num_transforms {
                            let base = idx * 32; // weight is first field
                            if self.target_xf_params[base] < 0.01 {
                                self.target_xf_params[base] = 0.25;
                            } else {
                                self.target_xf_params[base] = 0.0;
                            }
                            eprintln!(
                                "[solo] transform {} = {}",
                                idx, self.target_xf_params[base]
                            );
                        }
                    }
                    "=" | "+" => {
                        if self.morph_rate_idx < MORPH_RATES.len() - 1 {
                            self.morph_rate_idx += 1;
                        }
                        eprintln!("[morph] rate = {} (faster)", MORPH_RATES[self.morph_rate_idx]);
                    }
                    "-" => {
                        if self.morph_rate_idx > 0 {
                            self.morph_rate_idx -= 1;
                        }
                        eprintln!("[morph] rate = {} (slower)", MORPH_RATES[self.morph_rate_idx]);
                    }
                    "a" => {
                        self.audio_enabled = !self.audio_enabled;
                        eprintln!("[audio] {}", if self.audio_enabled { "enabled" } else { "disabled" });
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
                },
                _ => {}
            },
            WindowEvent::RedrawRequested => {
                self.check_file_changes();

                let now = Instant::now();
                let dt = now.duration_since(self.last_frame_time).as_secs_f32().max(0.001);
                self.last_frame_time = now;

                // Apply weight matrix (audio features from background thread, plus time)
                if self.audio_enabled {
                    let time = self.start.elapsed().as_secs_f32();
                    let time_since_mutation = time - self.last_mutation_time;
                    // Accumulate random walk: integrate noise over time
                    self.random_walk += crate::weights::value_noise_pub(time * 0.3) * dt * 0.5;
                    let time_signals = crate::weights::TimeSignals::compute(time, time_since_mutation, self.random_walk);

                    let base_globals = self.genome.flatten_globals();
                    let base_xf = self.genome.flatten_transforms();

                    self.target_globals = self.weights.apply_globals(&base_globals, &self.audio_features, &time_signals);
                    self.target_xf_params = self.weights.apply_transforms(&base_xf, self.num_transforms, &self.audio_features, &time_signals);

                    // Auto-evolve via mutation_rate
                    let mr = self.weights.mutation_rate(&self.audio_features, &time_signals);
                    self.mutation_accum += mr * dt;
                    if self.mutation_accum >= 1.0 {
                        self.mutation_accum = 0.0;
                        self.genome_history.push(self.genome.clone());
                        if self.genome_history.len() > 10 {
                            self.genome_history.remove(0);
                        }
                        self.genome = self.genome.mutate();
                        self.last_mutation_time = self.start.elapsed().as_secs_f32();
                        self.apply_genome_targets();
                        eprintln!("[auto-evolve] → {}", self.genome.name);
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

                // Morph globals
                let rate = 1.0 - (-dt * MORPH_RATES[self.morph_rate_idx]).exp();
                for i in 0..12 {
                    self.globals[i] += (self.target_globals[i] - self.globals[i]) * rate;
                }
                // Morph transforms (handle different lengths)
                let max_len = self.xf_params.len().max(self.target_xf_params.len());
                self.xf_params.resize(max_len, 0.0);
                self.target_xf_params.resize(max_len, 0.0);
                for i in 0..max_len {
                    self.xf_params[i] += (self.target_xf_params[i] - self.xf_params[i]) * rate;
                }

                // Write uniforms
                let uniforms = Uniforms {
                    time: self.start.elapsed().as_secs_f32(),
                    frame: self.frame,
                    resolution: [
                        gpu.config.width as f32,
                        gpu.config.height as f32,
                    ],
                    mouse: self.mouse,
                    transform_count: self.genome.transform_count(),
                    has_final_xform: if self.genome.final_transform.is_some() { 1 } else { 0 },
                    globals: [self.globals[0], self.globals[1], self.globals[2], self.globals[3]],
                    kifs: [self.globals[4], self.globals[5], self.globals[6], self.globals[7]],
                    extra: [self.globals[8], self.globals[9], self.globals[10], self.genome.symmetry as f32],
                    extra2: [1.0, 0.0, 0.0, 0.0],  // crossfade_alpha=1.0 (fully new, no crossfade active)
                };

                gpu.queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
                let xf_write_len = self.num_transforms * 32;
                let xf_slice = &self.xf_params[..xf_write_len.min(self.xf_params.len())];
                gpu.queue.write_buffer(&gpu.transform_buffer, 0, bytemuck::cast_slice(xf_slice));

                gpu.render();
                self.frame += 1;

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
        device_picker::Selection::SystemAudio => {
            match AudioCapture::new_system_audio() {
                Ok(cap) => Some(cap),
                Err(e) => {
                    eprintln!("[audio] SCK capture failed: {e} (visuals-only mode)");
                    None
                }
            }
        }
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
