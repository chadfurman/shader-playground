// ── Fractal Flame Compute Shader ──

struct Uniforms {
    time: f32,
    frame: u32,
    resolution: vec2<f32>,
    mouse: vec2<f32>,
    transform_count: u32,
    _pad: u32,
    globals: vec4<f32>,   // speed, zoom, trail, flame_brightness
    kifs: vec4<f32>,      // fold_angle, scale, brightness, drift_speed
    extra: vec4<f32>,     // color_shift, 0, 0, 0
}

@group(0) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(1) var<uniform> u: Uniforms;
@group(0) @binding(2) var<storage, read> transforms: array<f32>;

fn xf(idx: u32, field: u32) -> f32 { return transforms[idx * 12u + field]; }

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

// ── IFS Transform (reads from storage buffer) ──

fn apply_xform(p: vec2<f32>, idx: u32, t: f32) -> vec2<f32> {
    let angle  = xf(idx, 1u);
    let scale  = xf(idx, 2u);
    let ox     = xf(idx, 3u);
    let oy     = xf(idx, 4u);
    let w_lin  = xf(idx, 6u);
    let w_sin  = xf(idx, 7u);
    let w_sph  = xf(idx, 8u);
    let w_swi  = xf(idx, 9u);
    let w_hor  = xf(idx, 10u);
    let w_han  = xf(idx, 11u);

    let drift = u.kifs.w; // drift_speed
    let q = rot2(angle + t * 0.07 * drift * f32(idx + 1u)) * p * scale
          + vec2(ox + 0.05 * sin(t * 0.3 * drift * f32(idx + 1u)),
                 oy + 0.05 * cos(t * 0.4 * drift * f32(idx + 1u)));

    var v = q * w_lin;
    v += V_sinusoidal(q)    * w_sin;
    v += V_spherical(q)     * w_sph;
    v += V_swirl(q)         * w_swi;
    v += V_horseshoe(q)     * w_hor;
    v += V_handkerchief(q)  * w_han;

    return v;
}

fn xform_color(idx: u32) -> f32 {
    return xf(idx, 5u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let speed = u.globals.x;
    let zoom = u.globals.y;
    let t = u.time * speed;
    let num_xf = u.transform_count;

    var rng = gid.x * 2654435761u + u.frame * 7919u + 12345u;

    var p = vec2(randf(&rng) * 4.0 - 2.0, randf(&rng) * 4.0 - 2.0);
    var color_idx = randf(&rng);

    let w = u32(u.resolution.x);
    let h = u32(u.resolution.y);

    // Precompute total weight
    var total_weight = 0.0;
    for (var t_idx = 0u; t_idx < num_xf; t_idx++) {
        total_weight += xf(t_idx, 0u);
    }
    if (total_weight < 1e-6) { return; }

    for (var i = 0u; i < 200u; i++) {
        // Weighted random transform selection
        let r = randf(&rng) * total_weight;
        var tidx = 0u;
        var cumsum = 0.0;
        for (var ti = 0u; ti < num_xf; ti++) {
            cumsum += xf(ti, 0u);
            if (r < cumsum) {
                tidx = ti;
                break;
            }
        }

        p = apply_xform(p, tidx, t);
        color_idx = color_idx * 0.3 + xform_color(tidx) * 0.7;

        if (i < 20u) { continue; }

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
