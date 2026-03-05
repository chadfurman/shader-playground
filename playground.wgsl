// ── Shader Playground ──
// Edit this file, save, see changes instantly.
//
// Available inputs:
//   u.time       - seconds since start
//   u.resolution - window size in pixels
//   u.mouse      - mouse position (0-1)
//   u.frame      - frame counter
//   u.params[0..15] - tweakable floats from params.json
//   prev_frame   - texture of last frame (for feedback/trails)

struct Uniforms {
    time: f32,
    frame: u32,
    resolution: vec2<f32>,
    mouse: vec2<f32>,
    _pad: vec2<f32>,
    params: array<vec4<f32>, 4>,  // 16 floats as 4 vec4s
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var prev_frame: texture_2d<f32>;
@group(0) @binding(2) var prev_sampler: sampler;

// ── Helpers ──

const PI: f32 = 3.14159265;

fn rot2(a: f32) -> mat2x2<f32> {
    let c = cos(a); let s = sin(a);
    return mat2x2(c, -s, s, c);
}

fn param(i: i32) -> f32 {
    return u.params[i / 4][i % 4];
}

// ── SDF Primitives ──

fn sd_cylinder(p: vec3<f32>, r: f32, h: f32) -> f32 {
    let d = vec2(length(p.xz) - r, abs(p.y) - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0)));
}

fn sd_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sd_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sd_torus(p: vec3<f32>, r1: f32, r2: f32) -> f32 {
    let q = vec2(length(p.xz) - r1, p.y);
    return length(q) - r2;
}

// ── Scene ──
// This is where you define what to render.
// Start simple, add effects one at a time.

fn map(p: vec3<f32>) -> f32 {
    // Rotate with time
    var q = p;
    let xy = rot2(u.time * 0.3) * q.xy;
    q = vec3(xy, q.z);
    let xz = rot2(u.time * 0.2) * q.xz;
    q = vec3(xz.x, q.y, xz.y);

    // A cylinder — params[0] = radius, params[1] = height
    let r = param(0);
    let h = param(1);
    return sd_cylinder(q, r, h);
}

// ── Rendering ──

fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let e = 0.001;
    let d = map(p);
    return normalize(vec3(
        map(p + vec3(e, 0.0, 0.0)) - d,
        map(p + vec3(0.0, e, 0.0)) - d,
        map(p + vec3(0.0, 0.0, e)) - d,
    ));
}

fn raymarch(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
    var t = 0.0;
    for (var i = 0; i < 80; i++) {
        let p = ro + rd * t;
        let d = map(p);
        if (d < 0.001) { return vec2(t, f32(i)); }
        if (t > 20.0) { break; }
        t += d * 0.9;
    }
    return vec2(-1.0, 0.0);
}

// ── Vertex Shader ──

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Fullscreen triangle
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) % 2) * 4.0 - 1.0;
    return vec4(x, y, 0.0, 1.0);
}

// ── Fragment Shader ──

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (pos.xy - u.resolution * 0.5) / u.resolution.y;

    // Camera
    let cam_dist = 4.0;
    let ro = vec3(0.0, 0.0, cam_dist);
    let rd = normalize(vec3(uv, -1.5));

    // Raymarch
    let hit = raymarch(ro, rd);

    var col = vec3(0.02, 0.02, 0.05); // background

    if (hit.x > 0.0) {
        let p = ro + rd * hit.x;
        let n = calc_normal(p);

        // Simple lighting
        let light = normalize(vec3(1.0, 2.0, 3.0));
        let diff = max(dot(n, light), 0.0);
        let spec = pow(max(dot(reflect(-light, n), -rd), 0.0), 32.0);
        let amb = 0.15;

        // Color based on normal
        let base = 0.5 + 0.5 * n;
        col = base * (amb + diff * 0.7) + vec3(1.0) * spec * 0.3;
    }

    // Optional: blend with previous frame for trails
    // let prev = textureSample(prev_frame, prev_sampler, pos.xy / u.resolution);
    // col = mix(prev.rgb, col, 0.1);

    return vec4(col, 1.0);
}
