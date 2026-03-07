# Audio Signal Normalization Design

**Goal:** Rescale all audio signals so they use the full 0-1 range regardless of genre, making the weights system behave consistently across lofi, hard techno, ambient, and everything in between.

**Problem:** Raw FFT-derived signals (bass, mids, highs, energy) never exceed 0.12 even at peak loudness. Highs max at 0.005. Meanwhile beat_accum saturates near 1.0 and stays there. The weights system multiplies these signals by weights in the 0.1-1.5 range, so the frequency signals produce negligible visual effect while beat signals dominate.

**Data:** 8 sample recordings across genres (song transitions, funky electro, hard techno x2, slow chill, lofi x2). Analysis shows:
- bass P90: 0.08-0.18 across genres (needs ~7x gain)
- mids P90: 0.06-0.10 (needs ~10x gain)
- highs P90: 0.003-0.013 (needs ~55-230x gain, 4.3x genre variation)
- energy P90: 0.05-0.08 (redundant with bass+mids, r=0.97)
- beat_accum: saturates to 0.9+ identically across all genres

---

## Approach: Per-Band Rolling Peak Normalization

### Per-Band Normalizer

Each of bass, mids, highs maintains a `rolling_peak` — an exponentially-weighted recent maximum.

```
if raw > rolling_peak:
    rolling_peak = raw                      // instant snap up
else:
    rolling_peak *= 0.97^(dt * 30)          // ~2s half-life decay at 30Hz

rolling_peak = max(rolling_peak, floor)     // prevent noise amplification
normalized = raw / rolling_peak             // output: 0-1
```

Floor values (below this is silence, don't amplify):
- bass: 0.005
- mids: 0.003
- highs: 0.0005

### Energy Recomputation

Energy is recomputed from normalized signals including beat:

```
energy = (norm_bass + norm_mids + norm_highs + beat) / 4
```

This makes energy reflect both spectral content and rhythmic activity.

### Beat Density (replaces beat_accum)

Current beat_accum adds +0.05 per beat with 6s decay — saturates near 1.0 whenever any beat exists.

New approach: sliding window beat density over 8 seconds.

```
on beat detected:
    push timestamp to ring buffer
    prune entries older than 8 seconds

beat_accum = min(beats_in_window / 24.0, 1.0)
```

24 beats in 8 seconds = 3 beats/sec = 180 BPM = ~1.0. This gives:
- Lofi gentle beats: ~0.2-0.4
- Four-on-the-floor at 120 BPM: ~0.67
- Hard techno at 150 BPM: ~0.83
- Breakdown/silence: drops to 0

### beat_pulse

Unchanged — impulse to 1.0 on beat, 1.5s exponential decay. Useful for triggering momentary effects.

---

## Signal Flow

```
Raw FFT → bass_raw, mids_raw, highs_raw
              ↓
Per-band normalizer (rolling peak, 2s adapt)
              ↓
         bass, mids, highs  (0-1, genre-adaptive)
              ↓
         energy = (bass + mids + highs + beat) / 4
              ↓
Beat detector → beat (0-1 stepped, unchanged)
              ↓
         beat_accum = sliding window density (0-1)
         beat_pulse = impulse with 1.5s decay (unchanged)
```

## Output Signals

| Signal | Range | Behavior |
|--------|-------|----------|
| bass | 0-1 | Normalized low-frequency energy |
| mids | 0-1 | Normalized mid-frequency energy |
| highs | 0-1 | Normalized high-frequency energy |
| energy | 0-1 | Combined spectral + rhythmic activity |
| beat | 0-1 | Stepped beat confidence (instantaneous) |
| beat_accum | 0-1 | Beat density over 8s window |
| beat_pulse | 0-1 | Beat impulse with fast decay |

## What Changes

- `AudioProcessor` gains per-band rolling peak state (3 floats + 3 floor constants)
- `update_envelopes` replaces beat_accum logic with ring buffer density
- Energy computation moves after normalization

## What Doesn't Change

- Time signals, weights system, shader, weights.json format
- beat and beat_pulse behavior
- AudioFeatures struct fields (same 7 fields, just better scaled)

---

## Test Plan

### Per-Band Normalizer Tests
1. Constant loud signal (0.1 for 3s) → output converges to ~1.0
2. Constant quiet signal (0.001 for 3s) → output converges to ~1.0
3. Silence (0.0) → output stays 0.0
4. Loud then quiet (0.1→0.01) → initially low, adapts up within ~2s
5. Quiet then loud (0.01→0.1) → instant snap, no overshoot past 1.0
6. Bands are independent — bass normalization doesn't affect mids/highs
7. Values near floor — clean behavior, no oscillation

### Energy Tests
8. All bands equal (0.5 each + beat 0.5) → energy = 0.5
9. Beat-dominant (bands 0, beat 1.0) → energy = 0.25
10. Spectral-dominant (bands 1.0, beat 0) → energy = 0.75

### Beat Density Tests
11. No beats → beat_accum = 0.0
12. Steady 120 BPM (2 beats/sec, 8s) → beat_accum ≈ 0.67
13. Steady 180 BPM (3 beats/sec) → beat_accum ≈ 1.0
14. Beats then silence → decays to 0 as beats fall out of window
15. Sparse beats (1 per 2s) → beat_accum ≈ 0.17
16. Window boundary — beats older than 8s pruned

### Integration Tests
17. Feed real sample data from collected JSON files, verify all outputs 0-1, p90 of bass/mids/highs in 0.5-0.9 range across genres
