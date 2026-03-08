// ── Fractal Flame + KIFS Background ──
//
// Flame: reads compute histogram, log-density tonemapping
// Background: KIFS fractal runs per-pixel in fragment shader
// Both blend with feedback for persistence.

struct Uniforms {
    time: f32,
    frame: u32,
    resolution: vec2<f32>,
    mouse: vec2<f32>,
    transform_count: u32,
    _pad: u32,
    globals: vec4<f32>,
    kifs: vec4<f32>,
    extra: vec4<f32>,
    extra2: vec4<f32>,  // crossfade_alpha, reserved, reserved, reserved
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var prev_frame: texture_2d<f32>;
@group(0) @binding(2) var prev_sampler: sampler;
@group(0) @binding(3) var<storage, read> histogram: array<u32>;
@group(0) @binding(4) var crossfade_tex: texture_2d<f32>;

const PI: f32  = 3.14159265;
const TAU: f32 = 6.28318530;

fn rot2(a: f32) -> mat2x2<f32> {
    let c = cos(a); let s = sin(a);
    return mat2x2(c, -s, s, c);
}

// Cosine palette (Inigo Quilez)
fn palette(t: f32) -> vec3<f32> {
    let a = vec3(0.5, 0.5, 0.5);
    let b = vec3(0.5, 0.5, 0.5);
    let c = vec3(1.0, 1.0, 1.0);
    let d = vec3(0.00, 0.33, 0.67);
    return a + b * cos(TAU * (c * t + d));
}

// Second palette (warmer, for KIFS background)
fn palette_warm(t: f32) -> vec3<f32> {
    let a = vec3(0.5, 0.5, 0.5);
    let b = vec3(0.5, 0.5, 0.5);
    let c = vec3(1.0, 0.7, 0.4);
    let d = vec3(0.00, 0.15, 0.20);
    return a + b * cos(TAU * (c * t + d));
}

// ── KIFS Background ──

fn kifs_fractal(uv: vec2<f32>, t: f32, fold_param: f32, ifs_scale: f32) -> vec3<f32> {
    var p = uv;
    var min_trap = 100.0;
    var trap_idx = 0.0;
    var line_trap = 100.0;
    var total_scale = 1.0;

    let base_angle = fold_param * PI;

    for (var i = 0; i < 18; i++) {
        p = abs(p);
        if (p.x < p.y) { p = p.yx; }

        let angle = base_angle + f32(i) * 0.18 + sin(t * 0.3 + f32(i) * 0.4) * 0.15;
        p = rot2(angle) * p;

        let offset = vec2(
            1.0 + 0.15 * sin(t * 0.2 + f32(i) * 0.5),
            0.8 + 0.1 * cos(t * 0.25 + f32(i) * 0.3)
        );
        p = p * ifs_scale - offset;
        total_scale *= ifs_scale;

        let d = length(p) / total_scale;
        if (d < min_trap) {
            min_trap = d;
            trap_idx = f32(i) + d * 3.0;
        }
        line_trap = min(line_trap, abs(p.x) / total_scale);
    }

    let point_glow = exp(-min_trap * 6.0);
    let line_glow  = exp(-line_trap * 10.0) * 0.7;
    let hue = fract(trap_idx * 0.07 + t * 0.04);
    let line_hue = fract(trap_idx * 0.12 + 0.5 + t * 0.03);

    return palette_warm(hue) * point_glow + palette_warm(line_hue) * line_glow;
}

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
    let uv = (pos.xy - u.resolution * 0.5) / u.resolution.y * 2.5;
    let tex_uv = pos.xy / u.resolution;

    let speed        = u.globals.x;
    let trail        = u.globals.z;
    let flame_bright = u.globals.w;
    let t = u.time * speed;

    // ── KIFS background layer — disabled ──
    let bg = vec3(0.0);

    // ── Flame foreground (direct histogram read — trail provides temporal smoothing) ──
    let h = u32(u.resolution.y);
    let buf_idx = (px.y * w + px.x) * 4u;
    let density = f32(histogram[buf_idx]);
    let acc_r = f32(histogram[buf_idx + 1u]);
    let acc_g = f32(histogram[buf_idx + 2u]);
    let acc_b = f32(histogram[buf_idx + 3u]);

    // Recover average color from accumulated RGB
    let avg_color = select(
        vec3(0.0),
        vec3(acc_r, acc_g, acc_b) / max(density * 1000.0, 1.0),
        density > 0.0
    );

    // Log-density alpha (no hard cap — natural falloff)
    let alpha = log(1.0 + density * flame_bright) / (log(1.0 + density * flame_bright) + 4.0);

    // Vibrancy: saturate colors based on density
    let vibrancy = u.extra.y;
    let lum = dot(avg_color, vec3(0.299, 0.587, 0.114));
    let vibrant_color = mix(vec3(lum), avg_color, pow(alpha, max(1.0 - vibrancy, 0.01)));
    let flame = vibrant_color * alpha;

    // ── Combine ──
    var new_col = flame + bg;

    // Feedback trail — temporal accumulation smooths grain across frames
    let prev = textureSample(prev_frame, prev_sampler, tex_uv).rgb;
    new_col = new_col + prev * trail;

    // Bloom — cheap 5-tap cross
    let bloom_int = u.extra.z;
    if (bloom_int > 0.001) {
        let texel = 1.0 / u.resolution;
        let bloom = (
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(-2.0, 0.0) * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2( 2.0, 0.0) * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(0.0, -2.0) * texel).rgb +
            textureSample(prev_frame, prev_sampler, tex_uv + vec2(0.0,  2.0) * texel).rgb
        ) * 0.25;
        new_col += bloom * bloom_int;
    }

    // Tonemap: soft clamp to prevent blowout
    var col = new_col / (new_col + vec3(1.0));  // Reinhard

    return vec4(col, 1.0);
}
