# Audio

Audio signal pipeline overview for shader-playground.

## Capture

Two capture backends, chosen at startup:

- **ScreenCaptureKit** (preferred) -- captures system audio on macOS 12.3+ via `sck_audio::start_system_audio_capture`. Grabs whatever is playing through the system output.
- **CPAL fallback** -- captures from a specific input or output device selected by the device picker. Supports F32, I16, and U16 sample formats.

Both backends write into a shared ring buffer (`Arc<Mutex<Vec<f32>>>`) capped at 4096 samples. Excess samples are drained from the front to keep the buffer current.

## Processing Pipeline

```
Audio Input --> Ring Buffer (4096 samples)
                    |
            Hann-windowed FFT (2048-point)
                    |
            Magnitude spectrum (1024 bins)
                    |
            16 logarithmic bands (band_edge fn)
                    |
            Bass / Mids / Highs aggregation
                    |
            Per-band rolling peak normalization
                    |
                AudioFeatures
```

Processing runs on a background thread at ~30 Hz. Features are written atomically to a JSON file (temp + rename).

## Band Splitting

The FFT output is divided into **16 logarithmic frequency bands** using an exponential edge function:

```
edge(i) = half_spectrum * (2^(i/16 * 10) - 1) / 1023
```

The 16 bands are then grouped into three aggregate signals:

| Signal | Bands | Approximate Range |
|--------|-------|-------------------|
| **bass** | 0, 1, 2 | Sub-bass through low-mids (~0-170 Hz at 44.1kHz) |
| **mids** | 4, 5, 6, 7 | Mid frequencies (~340-1400 Hz) |
| **highs** | 10, 11, 12, 13 | Upper frequencies (~4.5-14 kHz) |

Each aggregate is the mean of its constituent bands. Band 3, bands 8-9, and bands 14-15 are not directly mapped but contribute to overall energy.

## Normalization

Each band (bass, mids, highs) has an independent `BandNormalizer` with its own noise floor:

| Band | Floor |
|------|-------|
| bass | 0.005 |
| mids | 0.003 |
| highs | 0.0005 |

The normalizer tracks a **rolling peak** that:
- Snaps up instantly when input exceeds the current peak
- Decays exponentially during quiet periods (~2s half-life, `0.97^(dt*30)`)
- Never drops below the configured floor

Output is `raw / rolling_peak`, clamped to [0, 1]. This means a constant signal converges toward 1.0 regardless of absolute level, while silence stays near 0.0.

## Beat Detection

Beats are detected via **energy spike detection**:

1. A running average of raw spectral energy is maintained: `prev = prev * 0.9 + current * 0.1`
2. A beat fires when current energy exceeds `prev * 1.3` AND exceeds the noise gate (`0.0001`)
3. On beat: `beat = 1.0`. Otherwise: `beat -= 0.15` per frame (BEAT_DECAY), floored at 0.0

### beat_accum (Beat Density)

A sliding window tracks beat timestamps over the last **8 seconds**. The density is normalized so that 180 BPM (24 beats in 8s) maps to 1.0:

```
beat_accum = min(beats_in_window / 24.0, 1.0)
```

### beat_pulse

Snaps to 1.0 on each beat, then decays with a **1.5-second exponential envelope**:

```
beat_pulse *= exp(-dt / 1.5)
```

Useful for smooth, punchy visual responses that ring out after each hit.

## Energy

Energy is a composite of all normalized signals plus beat:

```
energy = (bass + mids + highs + beat) / 4.0
```

This uses the already-normalized band values, so energy reflects perceptual loudness rather than raw amplitude.

## AudioFeatures Struct

| Field | Range | Description |
|-------|-------|-------------|
| `bass` | 0.0 - 1.0 | Normalized low-frequency energy (bands 0-2) |
| `mids` | 0.0 - 1.0 | Normalized mid-frequency energy (bands 4-7) |
| `highs` | 0.0 - 1.0 | Normalized high-frequency energy (bands 10-13) |
| `energy` | 0.0 - 1.0 | Composite of bass + mids + highs + beat |
| `beat` | 0.0 - 1.0 | Beat impulse, snaps to 1.0 then decays at 0.15/frame |
| `beat_accum` | 0.0 - 1.0 | Beat density over 8s window, 180 BPM = 1.0 |
| `beat_pulse` | 0.0 - 1.0 | Beat impulse with 1.5s exponential decay envelope |

All fields are serialized to JSON at ~30 Hz for consumption by the render loop.
