# Feedback & Trail

The feedback trail system creates persistent luminous trails by blending the
current frame with the previous frame. Two mechanisms work together: a
max-based trail blend and optional temporal reprojection to compensate for
zoom changes.

## Trail Blending

The trail combines the current frame color with the previous frame using
`max()`, not additive blending:

```wgsl
col = max(col, prev * trail);
```

### Why `max()` instead of additive

Additive blending (`col + prev * trail`) would cause brightness to
accumulate without bound — every frame adds energy, and dense regions
quickly blow out to white.

`max()` avoids this entirely. The previous frame decays by the `trail`
factor each frame (e.g., `prev * 0.15`), so any pixel that stops receiving
new points fades exponentially toward black. Meanwhile, actively-hit pixels
stay bright because `max(new_color, fading_old)` simply keeps whichever is
brighter.

The result: bright pixels persist and decay naturally, dim pixels fade
quickly, and brightness never explodes.

### Decay rate

A trail value of 0.15 means a pixel retains 15% of its previous brightness
each frame. After *n* frames without new hits, brightness is
`original * 0.15^n` — roughly halving every 4 frames, and effectively gone
within 15-20 frames.

## Temporal Reprojection

When the zoom level changes between frames, the previous frame's pixels no
longer line up with the current frame. Without correction, trails leave
ghost images at the wrong scale.

Temporal reprojection warps the previous frame's UV coordinates to account
for the zoom change:

```wgsl
let centered = tex_uv - vec2(0.5);
let zoom_ratio = cur_zoom / prev_zoom;
let reprojected = centered * zoom_ratio + vec2(0.5);
prev_uv = mix(tex_uv, reprojected, temporal_reproj);
```

### Derivation

Screen position maps to world position as:
`screen = (world / zoom + 0.5) * resolution`

For the same world point to appear at the correct position in the warped
previous frame, the UV must be scaled by the ratio of current to previous
zoom. `centered * (cur_zoom / prev_zoom)` shifts the UV outward when
zooming in (cur > prev) and inward when zooming out.

The `temporal_reproj` parameter controls how much of this correction to
apply. At 0.0, no reprojection happens (raw previous frame). At 1.0, full
geometric correction is applied. Values in between blend linearly.

### UV Clamping

After reprojection, UVs are clamped to prevent sampling beyond the texture
edges:

```wgsl
prev_uv = clamp(prev_uv, vec2(0.001), vec2(0.999));
```

The 0.001/0.999 margins prevent edge bleeding artifacts from the texture
sampler's filtering kernel.

## Config

| Parameter                | Default | Description                                |
|--------------------------|---------|--------------------------------------------|
| `trail`                  | 0.15    | Previous frame retention factor (0-1)      |
| `temporal_reprojection`  | 0.0     | Zoom-compensation strength (0=off, 1=full) |

Both values are passed through the uniform buffer — `trail` via
`globals.z`, `temporal_reprojection` via `extra5.z`. The previous frame's
zoom level is stored in `extra5.w` (`prev_zoom`) and compared against the
current zoom in `globals.y`.
