// ── Kaleidoscopic IFS Fractal with Flow Trails ──
//
// Fold + rotate + scale iterations create fractal structure.
// Orbit trap coloring + cosine palette for rich color.
// Feedback with warped sampling for flowing trails.
//
// params[0] = evolution speed (0.15)
// params[1] = fold angle, radians/PI (0.62)
// params[2] = iteration scale (1.8)
// params[3] = trail persistence (0.92)

struct Uniforms {
    time: f32,
    frame: u32,
    resolution: vec2<f32>,
    mouse: vec2<f32>,
    _pad: vec2<f32>,
    params: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var prev_frame: texture_2d<f32>;
@group(0) @binding(2) var prev_sampler: sampler;

const PI: f32  = 3.14159265;
const TAU: f32 = 6.28318530;

fn param(i: i32) -> f32 { return u.params[i / 4][i % 4]; }

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

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) % 2) * 4.0 - 1.0;
    return vec4(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (pos.xy - u.resolution * 0.5) / u.resolution.y * 2.5;
    let tex_uv = pos.xy / u.resolution;

    let speed      = param(0);
    let fold_param = param(1);
    let ifs_scale  = param(2);
    let trail      = param(3);

    let t = u.time * speed;

    // ── KIFS iteration ──
    var p = uv;
    var min_trap = 100.0;       // orbit trap: closest approach
    var trap_idx = 0.0;         // which iteration hit the trap
    var line_trap = 100.0;      // distance to y-axis trap
    var total_scale = 1.0;

    let base_angle = fold_param * PI;

    for (var i = 0; i < 22; i++) {
        // Absolute fold — creates kaleidoscopic symmetry
        p = abs(p);

        // Diagonal fold — ensures consistent orientation
        if (p.x < p.y) { p = p.yx; }

        // Rotating fold axis — this is where the magic happens
        let angle = base_angle + f32(i) * 0.18 + sin(t * 0.3 + f32(i) * 0.4) * 0.15;
        p = rot2(angle) * p;

        // Scale and translate — drives the self-similarity
        let offset = vec2(
            1.0 + 0.15 * sin(t * 0.2 + f32(i) * 0.5),
            0.8 + 0.1 * cos(t * 0.25 + f32(i) * 0.3)
        );
        p = p * ifs_scale - offset;

        total_scale *= ifs_scale;

        // Orbit trap: track closest approach to origin
        let d = length(p) / total_scale;
        if (d < min_trap) {
            min_trap = d;
            trap_idx = f32(i) + d * 3.0;
        }

        // Line trap: distance to y-axis (creates ribbon-like structures)
        let ld = abs(p.x) / total_scale;
        line_trap = min(line_trap, ld);
    }

    // ── Coloring ──
    // Combine point trap and line trap for rich detail
    let point_glow = exp(-min_trap * 6.0);
    let line_glow  = exp(-line_trap * 10.0) * 0.7;

    let hue = fract(trap_idx * 0.07 + t * 0.04);
    let base_color = palette(hue);

    let line_hue = fract(trap_idx * 0.12 + 0.5 + t * 0.03);
    let line_color = palette(line_hue);

    var col = base_color * point_glow + line_color * line_glow;

    // ── Feedback: additive accumulation like real flame density ──
    let flow = vec2(
        sin(uv.y * 3.0 + t * 2.5) * 0.002,
        cos(uv.x * 3.0 + t * 2.5) * 0.002
    );
    let prev = textureSample(prev_frame, prev_sampler, tex_uv + flow).rgb;

    // Additive: new fractal adds to decaying history (density builds up)
    col = col * 0.35 + prev * trail;

    // Log-density tonemap (like real fractal flames)
    col = log(1.0 + col * 4.0) / log(5.0);

    return vec4(col, 1.0);
}
