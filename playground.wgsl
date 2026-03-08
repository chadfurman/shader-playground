// ── Fractal Flame Display Shader ──
//
// Flam3-faithful rendering:
// 1. Log-density mapping (Scott Draves' algorithm)
// 2. Vibrancy-aware color blending
// 3. Single gamma correction
// 4. Soft bloom halo

struct Uniforms {
    time: f32,
    frame: u32,
    resolution: vec2<f32>,
    mouse: vec2<f32>,
    transform_count: u32,
    _pad: u32,
    globals: vec4<f32>,     // speed, zoom, trail, flame_brightness
    kifs: vec4<f32>,        // fold_angle, scale, brightness, drift_speed
    extra: vec4<f32>,       // color_shift, vibrancy, bloom_intensity, gamma
    extra2: vec4<f32>,      // noise_disp, curl_disp, tangent_clamp, color_blend
    extra3: vec4<f32>,      // spin_speed_max, position_drift, warmup_iters, highlight_power
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var prev_frame: texture_2d<f32>;
@group(0) @binding(2) var prev_sampler: sampler;
@group(0) @binding(3) var<storage, read> accumulation: array<f32>;
@group(0) @binding(4) var crossfade_tex: texture_2d<f32>;

// ── Vertex ──

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) % 2) * 4.0 - 1.0;
    return vec4(x, y, 0.0, 1.0);
}

// ── Fragment ──

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<u32>(u32(pos.x), u32(pos.y));
    let w = u32(u.resolution.x);
    let tex_uv = pos.xy / u.resolution;

    let trail        = u.globals.z;
    let flame_bright = u.globals.w;
    let vibrancy     = u.extra.y;
    let bloom_int    = u.extra.z;
    let gamma        = u.extra.w;

    // ── Read accumulation buffer ──
    let buf_idx = (px.y * w + px.x) * 4u;
    let density = accumulation[buf_idx];
    let acc_r = accumulation[buf_idx + 1u];
    let acc_g = accumulation[buf_idx + 2u];
    let acc_b = accumulation[buf_idx + 3u];

    // ── Flam3 log-density tonemapping ──
    // alpha maps density to brightness via log curve.
    // Self-normalizing: alpha → 0 for empty, → 1 for bright, no max-density needed.
    let log_density = log(1.0 + density * flame_bright);
    let alpha = log_density / (log_density + 1.0);

    // Recover average color (RGB stored as fixed-point * 1000 in histogram)
    let raw_color = select(
        vec3(0.0),
        vec3(acc_r, acc_g, acc_b) / max(density * 1000.0, 1.0),
        density > 0.0
    );

    // ── Flam3 vibrancy color blend ──
    // vibrancy=1: color * alpha (preserves saturation, linear brightness)
    // vibrancy=0: color * alpha^(1/gamma) (gamma-corrected brightness)
    // This is the ONLY place gamma is applied — not again at the end.
    let gamma_alpha = pow(max(alpha, 0.001), gamma);
    let ls = vibrancy * alpha + (1.0 - vibrancy) * gamma_alpha;
    var col = ls * raw_color;

    // ── Feedback trail — gentle temporal smoothing ──
    let prev = textureSample(prev_frame, prev_sampler, tex_uv).rgb;
    col = max(col, prev * trail);  // max-blend, not additive — prevents brightness buildup

    // ── Multi-radius bloom ──
    if (bloom_int > 0.001) {
        let texel = 1.0 / u.resolution;
        var bloom_sum = vec3(0.0);

        let r1 = 2.0;
        bloom_sum += (
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(-r1, 0.0) * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2( r1, 0.0) * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(0.0, -r1) * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(0.0,  r1) * texel).rgb
        ) * 0.25 * 0.5;

        let r2 = 5.0;
        bloom_sum += (
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(-r2, 0.0) * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2( r2, 0.0) * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(0.0, -r2) * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(0.0,  r2) * texel).rgb
        ) * 0.25 * 0.3;

        let r3 = 12.0;
        bloom_sum += (
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(-r3, 0.0) * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2( r3, 0.0) * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(0.0, -r3) * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(0.0,  r3) * texel).rgb
        ) * 0.25 * 0.2;

        col += bloom_sum * bloom_int;
    }

    // ── Soft clamp to prevent >1.0 without crushing dynamic range ──
    // No Reinhard, no extra gamma — the vibrancy blend already did gamma.
    col = clamp(col, vec3(0.0), vec3(1.0));

    return vec4(col, 1.0);
}
