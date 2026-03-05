// ── Fractal Flame Compute Shader ──
//
// Runs the chaos game: each thread iterates a random point
// through IFS transforms, plotting to an atomic histogram.
//
// Param layout (from FlameGenome):
//   [0]  speed
//   [1]  zoom
//   [2]  trail (used by fragment shader)
//   [3]  flame_brightness (used by fragment shader)
//   [4-6] kifs params (used by fragment shader)
//   [7]  drift_speed
//   [8..19]  transform 0: weight, angle, scale, ox, oy, color, w_lin, w_sin, w_sph, w_swi, w_hor, w_han
//   [20..31] transform 1
//   [32..43] transform 2
//   [44..55] transform 3

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

// ── IFS Transforms (param-driven) ──
// Each transform occupies 12 floats starting at param(8 + idx * 12):
//   weight, angle, scale, ox, oy, color, w_lin, w_sin, w_sph, w_swi, w_hor, w_han

fn apply_xform(p: vec2<f32>, idx: i32, t: f32) -> vec2<f32> {
    let base = 8 + idx * 12;
    let angle  = param(base + 1);
    let scale  = param(base + 2);
    let ox     = param(base + 3);
    let oy     = param(base + 4);
    let w_lin  = param(base + 6);
    let w_sin  = param(base + 7);
    let w_sph  = param(base + 8);
    let w_swi  = param(base + 9);
    let w_hor  = param(base + 10);
    let w_han  = param(base + 11);

    // Affine transform (angle drifts slowly with time)
    let drift = param(7);
    let q = rot2(angle + t * 0.07 * drift * f32(idx + 1)) * p * scale
          + vec2(ox + 0.05 * sin(t * 0.3 * drift * f32(idx + 1)),
                 oy + 0.05 * cos(t * 0.4 * drift * f32(idx + 1)));

    // Weighted sum of variations
    var v = q * w_lin;
    v += V_sinusoidal(q)    * w_sin;
    v += V_spherical(q)     * w_sph;
    v += V_swirl(q)         * w_swi;
    v += V_horseshoe(q)     * w_hor;
    v += V_handkerchief(q)  * w_han;

    return v;
}

fn xform_color(idx: i32) -> f32 {
    return param(8 + idx * 12 + 5);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let speed = param(0);
    let zoom = param(1);
    let t = u.time * speed;

    var rng = gid.x * 2654435761u + u.frame * 7919u + 12345u;

    var p = vec2(randf(&rng) * 4.0 - 2.0, randf(&rng) * 4.0 - 2.0);
    var color_idx = randf(&rng);

    let w = u32(u.resolution.x);
    let h = u32(u.resolution.y);

    for (var i = 0u; i < 200u; i++) {
        // Pick transform weighted by genome weights
        let r = randf(&rng);
        let w0 = param(8);
        let w1 = param(20);
        let w2 = param(32);
        let w3 = param(44);
        let total = w0 + w1 + w2 + w3;
        let rn = r * total;
        var tidx = 0;
        if (rn > w0) { tidx = 1; }
        if (rn > w0 + w1) { tidx = 2; }
        if (rn > w0 + w1 + w2) { tidx = 3; }

        p = apply_xform(p, tidx, t);

        // Color blending: weight toward current transform for distinct tendrils
        color_idx = color_idx * 0.3 + xform_color(tidx) * 0.7;

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
