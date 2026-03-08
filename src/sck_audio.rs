use screencapturekit::cm::CMSampleBuffer;
use screencapturekit::shareable_content::SCShareableContent;
use screencapturekit::stream::configuration::SCStreamConfiguration;
use screencapturekit::stream::content_filter::SCContentFilter;
use screencapturekit::stream::output_type::SCStreamOutputType;
use screencapturekit::stream::sc_stream::SCStream;
use std::sync::{Arc, Mutex};

const SAMPLE_RATE: u32 = 48000;

/// Start capturing system audio via ScreenCaptureKit.
///
/// Runs SCK initialization on a background thread to avoid blocking
/// the main run loop (macOS dispatches SCK callbacks via GCD and will
/// trap if the main thread is blocked waiting for them).
pub fn start_system_audio_capture(
    buffer: Arc<Mutex<Vec<f32>>>,
    buffer_cap: usize,
) -> Result<(SCStream, u32), String> {
    std::thread::spawn(move || init_and_start(buffer, buffer_cap))
        .join()
        .map_err(|_| "SCK init thread panicked".to_string())?
}

fn init_and_start(
    buffer: Arc<Mutex<Vec<f32>>>,
    buffer_cap: usize,
) -> Result<(SCStream, u32), String> {
    let (filter, config) = build_sck_filter_and_config()?;
    let mut stream = SCStream::new(&filter, &config);
    attach_audio_handler(&mut stream, buffer, buffer_cap);
    stream
        .start_capture()
        .map_err(|e| format!("start_capture failed: {e}"))?;
    eprintln!("[audio] SCK system audio capture started ({SAMPLE_RATE} Hz)");
    Ok((stream, SAMPLE_RATE))
}

fn build_sck_filter_and_config() -> Result<(SCContentFilter, SCStreamConfiguration), String> {
    let content =
        SCShareableContent::get().map_err(|e| format!("SCShareableContent::get failed: {e}"))?;
    let display = content
        .displays()
        .into_iter()
        .next()
        .ok_or("no displays available")?;
    let filter = SCContentFilter::create()
        .with_display(&display)
        .with_excluding_windows(&[])
        .build();
    let config = SCStreamConfiguration::new()
        .with_width(2)
        .with_height(2)
        .with_captures_audio(true)
        .with_sample_rate(SAMPLE_RATE as i32)
        .with_channel_count(2)
        .with_excludes_current_process_audio(true);
    Ok((filter, config))
}

fn attach_audio_handler(stream: &mut SCStream, buffer: Arc<Mutex<Vec<f32>>>, cap: usize) {
    stream.add_output_handler(
        move |sample: CMSampleBuffer, output_type: SCStreamOutputType| {
            if output_type != SCStreamOutputType::Audio {
                return;
            }
            let Some(audio_list) = sample.audio_buffer_list() else {
                return;
            };
            let mut buf = buffer.lock().unwrap();
            for audio_buf in &audio_list {
                push_f32_from_bytes(&mut buf, audio_buf.data(), cap);
            }
        },
        SCStreamOutputType::Audio,
    );
}

fn push_f32_from_bytes(buf: &mut Vec<f32>, data: &[u8], cap: usize) {
    for chunk in data.chunks_exact(4) {
        buf.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    if buf.len() > cap {
        let drain = buf.len() - cap;
        buf.drain(..drain);
    }
}
