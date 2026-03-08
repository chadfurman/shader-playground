// accumulation.wgsl — per-pixel accumulation with exponential decay
//
// Reads raw histogram (u32 density + RGB), blends into persistent
// float32 accumulation buffer. Fragment shader reads accum instead of histogram.

struct AccumUniforms {
    resolution: vec2<f32>,
    decay: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read> histogram: array<u32>;
@group(0) @binding(1) var<storage, read_write> accumulation: array<f32>;
@group(0) @binding(2) var<uniform> params: AccumUniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(params.resolution.x);
    let h = u32(params.resolution.y);
    if (gid.x >= w || gid.y >= h) {
        return;
    }

    let px = gid.y * w + gid.x;
    let hist_idx = px * 4u;
    let accum_idx = px * 4u;

    // Read raw histogram for this frame
    let density = f32(histogram[hist_idx]);
    let r = f32(histogram[hist_idx + 1u]);
    let g = f32(histogram[hist_idx + 2u]);
    let b = f32(histogram[hist_idx + 3u]);

    // Exponential decay blend: accum = accum * decay + new_frame
    let decay = params.decay;
    accumulation[accum_idx]      = accumulation[accum_idx]      * decay + density;
    accumulation[accum_idx + 1u] = accumulation[accum_idx + 1u] * decay + r;
    accumulation[accum_idx + 2u] = accumulation[accum_idx + 2u] * decay + g;
    accumulation[accum_idx + 3u] = accumulation[accum_idx + 3u] * decay + b;
}
