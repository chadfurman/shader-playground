// ── Fractal Flame Compute Shader ──
//
// Runs the chaos game: each thread iterates a random point
// through IFS transforms, plotting to an atomic histogram.
//
// params[0] = evolution speed
// params[1] = variation amount (0=affine, 1=full nonlinear)
// params[2] = zoom
// params[3] = (reserved for display shader)

struct Uniforms {
    time: f32,
    frame: u32,
    resolution: vec2<f32>,
    mouse: vec2<f32>,
    _pad: vec2<f32>,
    params: array<vec4<f32>, 16>,
}

@group(0) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(1) var<uniform> u: Uniforms;

fn param(i: i32) -> f32 { return u.params[i / 4][i % 4]; }

const PI: f32 = 3.14159265;

// ── PCG Random ──

fn pcg(state: ptr<function, u32>) -> u32 {
    let old = *state;
    *state = old * 747796405u + 2891336453u;
    let word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return (word >> 22u) ^ word;
}

fn randf(state: ptr<function, u32>) -> f32 {
    return f32(pcg(state)) / 4294967295.0;
}

fn rot2(a: f32) -> mat2x2<f32> {
    let c = cos(a); let s = sin(a);
    return mat2x2(c, -s, s, c);
}

// ── Flame Variations ──

fn V_sinusoidal(p: vec2<f32>) -> vec2<f32> {
    return vec2(sin(p.x), sin(p.y));
}

fn V_spherical(p: vec2<f32>) -> vec2<f32> {
    return p / (dot(p, p) + 1e-6);
}

fn V_swirl(p: vec2<f32>) -> vec2<f32> {
    let r2 = dot(p, p);
    return vec2(p.x * sin(r2) - p.y * cos(r2),
                p.x * cos(r2) + p.y * sin(r2));
}

fn V_horseshoe(p: vec2<f32>) -> vec2<f32> {
    let r = length(p) + 1e-6;
    return vec2((p.x - p.y) * (p.x + p.y), 2.0 * p.x * p.y) / r;
}

fn V_handkerchief(p: vec2<f32>) -> vec2<f32> {
    let r = length(p);
    let theta = atan2(p.y, p.x);
    return r * vec2(sin(theta + r), cos(theta - r));
}

fn V_polar(p: vec2<f32>) -> vec2<f32> {
    return vec2(atan2(p.y, p.x) / PI, length(p) - 1.0);
}

// ── IFS Transforms ──
// 4 transforms, each with different affine + variation blend

fn apply_xform(p: vec2<f32>, idx: i32, t: f32, v: f32) -> vec2<f32> {
    var q: vec2<f32>;

    switch (idx) {
        case 0 {
            // "Flame tendril" — sinusoidal + swirl
            q = rot2(0.6 + t * 0.07) * p * 0.65 + vec2(0.4, 0.15 + 0.1 * sin(t * 0.3));
            let varied = V_sinusoidal(q) * 0.5 + V_swirl(q) * 0.5;
            return mix(q, varied, v);
        }
        case 1 {
            // "Spiral arm" — spherical + horseshoe
            q = rot2(-1.0 + t * 0.11) * p * 0.70 + vec2(-0.3 + 0.05 * cos(t * 0.4), 0.3);
            let varied = V_spherical(q) * 0.6 + V_horseshoe(q) * 0.4;
            return mix(q, varied, v);
        }
        case 2 {
            // "Organic curl" — handkerchief
            q = rot2(1.7 + t * 0.09) * p * 0.60 + vec2(-0.1, -0.35 + 0.08 * sin(t * 0.5));
            let varied = V_handkerchief(q) * 0.7 + V_sinusoidal(q) * 0.3;
            return mix(q, varied, v);
        }
        default {
            // "Background weave" — polar + linear
            q = rot2(t * 0.04) * p * 0.82 + vec2(0.05 * sin(t * 0.2), 0.0);
            let varied = V_polar(q);
            return mix(q, varied, v * 0.4);
        }
    }
}

fn xform_color(idx: i32) -> f32 {
    switch (idx) {
        case 0  { return 0.0; }
        case 1  { return 0.33; }
        case 2  { return 0.67; }
        default { return 0.85; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let speed = param(0);
    let var_amt = param(1);
    let zoom = param(2);
    let t = u.time * speed;

    var rng = gid.x * 2654435761u + u.frame * 7919u + 12345u;

    var p = vec2(randf(&rng) * 4.0 - 2.0, randf(&rng) * 4.0 - 2.0);
    var color_idx = randf(&rng);

    let w = u32(u.resolution.x);
    let h = u32(u.resolution.y);

    for (var i = 0u; i < 200u; i++) {
        // Pick random transform
        let r = randf(&rng);
        var tidx = 0;
        if (r > 0.25) { tidx = 1; }
        if (r > 0.5) { tidx = 2; }
        if (r > 0.75) { tidx = 3; }

        p = apply_xform(p, tidx, t, var_amt);

        // Standard flame color blending
        color_idx = (color_idx + xform_color(tidx)) * 0.5;

        // Skip transient
        if (i < 20u) { continue; }

        // Map to screen and plot
        let screen = (p / zoom + vec2(0.5, 0.5)) * vec2<f32>(f32(w), f32(h));
        let px_x = i32(screen.x);
        let px_y = i32(screen.y);

        if (px_x >= 0 && px_x < i32(w) && px_y >= 0 && px_y < i32(h)) {
            let buf_idx = (u32(px_y) * w + u32(px_x)) * 2u;
            atomicAdd(&histogram[buf_idx], 1u);
            atomicAdd(&histogram[buf_idx + 1u], u32(color_idx * 1000.0));
        }
    }
}
