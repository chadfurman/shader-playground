use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::{cursor, event, execute, queue, style, terminal};
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// What the user selected.
pub enum Selection {
    SystemAudio,
    CpalDevice(cpal::Device, bool), // (device, is_input)
    Cancelled,
}

struct LiveDevice {
    name: String,
    kind: &'static str,
    is_input: bool,
    is_sck: bool,
    level: Arc<Mutex<f32>>,
    _stream: Option<cpal::Stream>,
    device: Option<cpal::Device>,
}

/// Run the interactive device picker with live level meters.
pub fn run() -> Selection {
    let mut devices = build_device_list();
    if devices.is_empty() {
        eprintln!("No audio devices found.");
        return Selection::Cancelled;
    }
    start_level_captures(&mut devices);
    let result = interactive_loop(&mut devices);
    stop_all(&mut devices);
    result
}

fn build_device_list() -> Vec<LiveDevice> {
    let host = cpal::default_host();
    let mut devices = Vec::new();

    // SCK entry first
    devices.push(LiveDevice {
        name: "System Audio (ScreenCaptureKit)".into(),
        kind: "system",
        is_input: false,
        is_sck: true,
        level: Arc::new(Mutex::new(0.0)),
        _stream: None,
        device: None,
    });

    // Input devices
    if let Ok(inputs) = host.input_devices() {
        for d in inputs {
            let name = d
                .description()
                .map(|d| d.name().to_string())
                .unwrap_or_else(|_| "???".into());
            devices.push(LiveDevice {
                name,
                kind: "input",
                is_input: true,
                is_sck: false,
                level: Arc::new(Mutex::new(0.0)),
                _stream: None,
                device: Some(d),
            });
        }
    }

    // Output devices (loopback)
    if let Ok(outputs) = host.output_devices() {
        for d in outputs {
            let name = d
                .description()
                .map(|d| d.name().to_string())
                .unwrap_or_else(|_| "???".into());
            devices.push(LiveDevice {
                name,
                kind: "loopback",
                is_input: false,
                is_sck: false,
                level: Arc::new(Mutex::new(0.0)),
                _stream: None,
                device: Some(d),
            });
        }
    }

    devices
}

fn start_level_captures(devices: &mut [LiveDevice]) {
    for dev in devices.iter_mut() {
        if dev.is_sck {
            continue;
        }
        let Some(ref device) = dev.device else {
            continue;
        };
        let config = if dev.is_input {
            device.default_input_config().ok()
        } else {
            device.default_output_config().ok()
        };
        let Some(config) = config else { continue };
        let level = dev.level.clone();
        let stream_config: cpal::StreamConfig = config.into();
        let stream = device
            .build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let max = data.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                    let mut lvl = level.lock().unwrap();
                    if max > *lvl {
                        *lvl = max;
                    } else {
                        *lvl *= 0.9;
                    }
                },
                |_| {},
                None,
            )
            .ok();
        if let Some(ref s) = stream {
            let _ = s.play();
        }
        dev._stream = stream;
    }
}

fn stop_all(devices: &mut [LiveDevice]) {
    for dev in devices.iter_mut() {
        dev._stream = None;
    }
}

fn interactive_loop(devices: &mut [LiveDevice]) -> Selection {
    let mut stdout = std::io::stdout();
    let _ = terminal::enable_raw_mode();
    let _ = execute!(stdout, terminal::EnterAlternateScreen, cursor::Hide);

    let mut cursor_pos = 0usize;
    let mut last_draw = Instant::now();
    let result = loop {
        if last_draw.elapsed() >= Duration::from_millis(50) {
            draw(&mut stdout, devices, cursor_pos);
            last_draw = Instant::now();
        }
        if let Some(action) = poll_input() {
            match action {
                Action::Up => cursor_pos = cursor_pos.saturating_sub(1),
                Action::Down => cursor_pos = (cursor_pos + 1).min(devices.len() - 1),
                Action::Confirm => {
                    let dev = &mut devices[cursor_pos];
                    if dev.is_sck {
                        break Selection::SystemAudio;
                    } else if let Some(d) = dev.device.take() {
                        break Selection::CpalDevice(d, dev.is_input);
                    }
                }
                Action::Cancel => break Selection::Cancelled,
            }
        }
    };

    let _ = execute!(stdout, terminal::LeaveAlternateScreen, cursor::Show);
    let _ = terminal::disable_raw_mode();
    result
}

enum Action {
    Up,
    Down,
    Confirm,
    Cancel,
}

fn poll_input() -> Option<Action> {
    if !event::poll(Duration::from_millis(30)).unwrap_or(false) {
        return None;
    }
    let Ok(event::Event::Key(k)) = event::read() else {
        return None;
    };
    match k.code {
        event::KeyCode::Up | event::KeyCode::Char('k') => Some(Action::Up),
        event::KeyCode::Down | event::KeyCode::Char('j') => Some(Action::Down),
        event::KeyCode::Enter | event::KeyCode::Char(' ') => Some(Action::Confirm),
        event::KeyCode::Esc | event::KeyCode::Char('q') => Some(Action::Cancel),
        _ => None,
    }
}

const HEADER_LINES: u16 = 4;

fn draw(stdout: &mut impl Write, devices: &[LiveDevice], cursor: usize) {
    let (w, _) = terminal::size().unwrap_or((80, 24));
    let _ = queue!(
        stdout,
        cursor::MoveTo(0, 0),
        terminal::Clear(terminal::ClearType::All)
    );

    // Header
    let _ = queue!(stdout, cursor::MoveTo(1, 0));
    let _ = queue!(stdout, style::SetAttribute(style::Attribute::Bold));
    let _ = queue!(stdout, style::Print("Audio Device Picker"));
    let _ = queue!(stdout, style::SetAttribute(style::Attribute::Reset));
    let _ = queue!(stdout, cursor::MoveTo(1, 1));
    let _ = queue!(
        stdout,
        style::SetForegroundColor(style::Color::DarkGrey),
        style::Print("↑↓=navigate  Enter=select  Esc=cancel"),
        style::SetForegroundColor(style::Color::Reset)
    );
    let _ = queue!(stdout, cursor::MoveTo(1, 2));
    let rule_len = (w as usize).saturating_sub(2).min(72);
    let _ = queue!(stdout, style::Print("─".repeat(rule_len)));

    // Device rows
    for (i, dev) in devices.iter().enumerate() {
        let row = HEADER_LINES + i as u16;
        let is_cur = i == cursor;

        if is_cur {
            let _ = queue!(stdout, style::SetAttribute(style::Attribute::Bold));
        }

        // Arrow + name
        let arrow = if is_cur { "▶" } else { " " };
        let _ = queue!(stdout, cursor::MoveTo(1, row), style::Print(arrow));

        let tag = format!("({})", dev.kind);
        let bar_w = bar_width(w);
        let name_end = (w as usize).saturating_sub(bar_w + 1);
        let tag_start = name_end.saturating_sub(tag.len());
        let name_budget = tag_start.saturating_sub(4);
        let name = truncate(&dev.name, name_budget);

        let _ = queue!(stdout, cursor::MoveTo(3, row), style::Print(name));
        let _ = queue!(
            stdout,
            cursor::MoveTo(tag_start as u16, row),
            style::SetForegroundColor(style::Color::DarkGrey),
            style::Print(&tag),
            style::SetForegroundColor(style::Color::Reset)
        );

        // Level meter
        let col = (w as usize).saturating_sub(bar_w) as u16;
        let _ = queue!(stdout, cursor::MoveTo(col, row));
        if dev.is_sck {
            let _ = queue!(
                stdout,
                style::SetForegroundColor(style::Color::Cyan),
                style::Print("[SCK]"),
                style::SetForegroundColor(style::Color::Reset)
            );
        } else {
            let level = *dev.level.lock().unwrap();
            draw_level_bar(stdout, level, bar_w);
        }

        if is_cur {
            let _ = queue!(stdout, style::SetAttribute(style::Attribute::Reset));
        }
    }

    let _ = stdout.flush();
}

fn draw_level_bar(stdout: &mut impl Write, level: f32, width: usize) {
    let db = if level > 0.0 {
        (20.0 * level.log10()).max(-60.0)
    } else {
        -60.0
    };
    let filled = (((db + 60.0) / 60.0) * width as f32).clamp(0.0, width as f32) as usize;
    let color = if filled * 3 > width * 2 {
        style::Color::Red
    } else if filled * 3 > width {
        style::Color::Yellow
    } else {
        style::Color::Green
    };
    let _ = queue!(
        stdout,
        style::SetForegroundColor(color),
        style::Print("█".repeat(filled)),
        style::SetForegroundColor(style::Color::DarkGrey),
        style::Print("░".repeat(width - filled)),
        style::SetForegroundColor(style::Color::Reset)
    );
}

fn bar_width(term_w: u16) -> usize {
    if term_w >= 100 {
        25
    } else if term_w >= 70 {
        15
    } else {
        10
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    if max <= 3 {
        return s.chars().take(max).collect();
    }
    let mut t: String = s.chars().take(max - 1).collect();
    t.push('…');
    t
}
