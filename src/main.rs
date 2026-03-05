use std::path::PathBuf;
use std::sync::{mpsc, Arc};
use std::time::Instant;
use std::{fs, mem};

use bytemuck::{Pod, Zeroable};
use notify::{Event, EventKind, RecursiveMode, Watcher};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowAttributes};

// ── Uniforms ──

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    time: f32,
    frame: u32,
    resolution: [f32; 2],
    mouse: [f32; 2],
    _pad: [f32; 2],
    params: [[f32; 4]; 4], // 16 floats as 4 vec4s
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
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group_a: wgpu::BindGroup,
    bind_group_b: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    frame_a: wgpu::TextureView,
    frame_b: wgpu::TextureView,
    ping: bool,
    pipeline_layout: wgpu::PipelineLayout,
    sampler: wgpu::Sampler,
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

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("main"),
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
                ],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("main"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

        let (frame_a, frame_b) =
            create_frame_textures(&device, config.width, config.height, format);

        let shader_src = load_shader_source();
        let pipeline = create_pipeline(
            &device,
            &pipeline_layout,
            &shader_src,
            format,
        );

        let bind_group_a = create_bind_group(
            &device,
            &bind_group_layout,
            &uniform_buffer,
            &frame_b, // read B when rendering to A
            &sampler,
        );
        let bind_group_b = create_bind_group(
            &device,
            &bind_group_layout,
            &uniform_buffer,
            &frame_a, // read A when rendering to B
            &sampler,
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
            frame_a,
            frame_b,
            ping: true,
            pipeline_layout,
            sampler,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        let (a, b) = create_frame_textures(
            &self.device,
            w,
            h,
            self.config.format,
        );
        self.frame_a = a;
        self.frame_b = b;
        self.rebuild_bind_groups();
    }

    fn reload_shader(&mut self, src: &str) {
        self.pipeline = create_pipeline(
            &self.device,
            &self.pipeline_layout,
            src,
            self.config.format,
        );
    }

    fn rebuild_bind_groups(&mut self) {
        self.bind_group_a = create_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.uniform_buffer,
            &self.frame_b,
            &self.sampler,
        );
        self.bind_group_b = create_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.uniform_buffer,
            &self.frame_a,
            &self.sampler,
        );
    }

    fn render(&mut self, uniforms: &Uniforms) {
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(uniforms),
        );

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let screen_view = frame.texture.create_view(&Default::default());

        // Pick which feedback texture to render to and which bind group to use
        let (target_view, bind_group) = if self.ping {
            (&self.frame_a, &self.bind_group_a)
        } else {
            (&self.frame_b, &self.bind_group_b)
        };

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Render to feedback texture
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

        // Copy feedback texture to screen
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
    // Find the directory containing Cargo.toml
    let mut dir = std::env::current_dir().unwrap();
    loop {
        if dir.join("Cargo.toml").exists() {
            return dir;
        }
        if !dir.pop() {
            // Fallback to cwd
            return std::env::current_dir().unwrap();
        }
    }
}

fn shader_path() -> PathBuf {
    project_dir().join("playground.wgsl")
}

fn params_path() -> PathBuf {
    project_dir().join("params.json")
}

fn load_shader_source() -> String {
    fs::read_to_string(shader_path())
        .unwrap_or_else(|_| include_str!("../playground.wgsl").to_string())
}

fn load_params() -> [f32; 16] {
    let mut params = [0.0f32; 16];
    if let Ok(json) = fs::read_to_string(params_path()) {
        if let Ok(vals) = serde_json::from_str::<Vec<f32>>(&json) {
            for (i, v) in vals.iter().enumerate().take(16) {
                params[i] = *v;
            }
        }
    }
    params
}

fn create_frame_textures(
    device: &wgpu::Device,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::TextureView, wgpu::TextureView) {
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
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    let a = device.create_texture(&desc).create_view(&Default::default());
    let b = device.create_texture(&desc).create_view(&Default::default());
    (a, b)
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    prev_frame: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("main"),
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
        ],
    })
}

fn create_pipeline(
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

// ── App ──

struct App {
    gpu: Option<Gpu>,
    window: Option<Arc<Window>>,
    watcher: Option<FileWatcher>,
    start: Instant,
    frame: u32,
    mouse: [f32; 2],
    params: [f32; 16],
    target_params: [f32; 16],
    last_frame_time: Instant,
}

impl App {
    fn new() -> Self {
        let initial = load_params();
        Self {
            gpu: None,
            window: None,
            watcher: None,
            start: Instant::now(),
            frame: 0,
            mouse: [0.5, 0.5],
            params: initial,
            target_params: initial,
            last_frame_time: Instant::now(),
        }
    }

    fn check_file_changes(&mut self) {
        let watcher = match &self.watcher {
            Some(w) => w,
            None => return,
        };
        let changed = watcher.changed_files();
        let mut reload_shader = false;
        let mut reload_params = false;
        let shader = shader_path();
        let params = params_path();
        for path in &changed {
            if path.ends_with(shader.file_name().unwrap()) {
                reload_shader = true;
            }
            if path.ends_with(params.file_name().unwrap()) {
                reload_params = true;
            }
        }
        if reload_shader {
            let src = load_shader_source();
            self.gpu.as_mut().unwrap().reload_shader(&src);
            eprintln!("[shader] reloaded");
        }
        if reload_params {
            self.target_params = load_params();
            eprintln!("[params] reloaded (morphing)");
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

        // Start file watcher
        let paths = vec![shader_path(), params_path()];
        match FileWatcher::new(&paths) {
            Ok(w) => self.watcher = Some(w),
            Err(e) => eprintln!("warning: file watcher failed: {e}"),
        }

        eprintln!("shader playground running — edit playground.wgsl");
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
            } => {
                // Could use for interaction later
            }
            WindowEvent::RedrawRequested => {
                self.check_file_changes();

                let gpu = match &mut self.gpu {
                    Some(g) => g,
                    None => return,
                };

                // Smooth param interpolation (~0.5s to reach target)
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame_time).as_secs_f32();
                self.last_frame_time = now;
                let rate = 1.0 - (-dt * 5.0_f32).exp(); // exponential ease, ~5x/sec
                for i in 0..16 {
                    self.params[i] += (self.target_params[i] - self.params[i]) * rate;
                }

                let flat_params: [[f32; 4]; 4] = [
                    [self.params[0], self.params[1], self.params[2], self.params[3]],
                    [self.params[4], self.params[5], self.params[6], self.params[7]],
                    [self.params[8], self.params[9], self.params[10], self.params[11]],
                    [self.params[12], self.params[13], self.params[14], self.params[15]],
                ];

                let uniforms = Uniforms {
                    time: self.start.elapsed().as_secs_f32(),
                    frame: self.frame,
                    resolution: [
                        gpu.config.width as f32,
                        gpu.config.height as f32,
                    ],
                    mouse: self.mouse,
                    _pad: [0.0; 2],
                    params: flat_params,
                };

                gpu.render(&uniforms);
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
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
