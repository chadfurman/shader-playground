// ── Fractal Flame: IFS Feedback ──
//
// Each frame: transform pixel coords through IFS functions,
// sample previous frame at those positions, accumulate.
// A seed feeds energy; the fractal attractor emerges over time.
//
// params[0] = evolution speed (0.2)
// params[1] = nonlinear variation amount (0.3)
// params[2] = feedback gain per transform (0.4)
// params[3] = color cycle speed (0.1)

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

const PI: f32 = 3.14159265;

fn param(i: i32) -> f32 { return u.params[i / 4][i % 4]; }

fn rot2(a: f32) -> mat2x2<f32> {
    let c = cos(a); let s = sin(a);
    return mat2x2(c, -s, s, c);
}

// Map centered coords back to texture UV (0-1)
fn to_tex(p: vec2<f32>) -> vec2<f32> {
    let aspect = u.resolution.x / u.resolution.y;
    return p * vec2(1.0 / aspect, 1.0) * 0.5 + 0.5;
}

// HSV to RGB
fn hsv(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let x = c * (1.0 - abs(((h * 6.0) % 2.0) - 1.0));
    let m = v - c;
    let hi = i32(floor(h * 6.0)) % 6;
    var rgb = vec3(0.0);
    switch (hi) {
        case 0  { rgb = vec3(c, x, 0.0); }
        case 1  { rgb = vec3(x, c, 0.0); }
        case 2  { rgb = vec3(0.0, c, x); }
        case 3  { rgb = vec3(0.0, x, c); }
        case 4  { rgb = vec3(x, 0.0, c); }
        default { rgb = vec3(c, 0.0, x); }
    }
    return rgb + m;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) % 2) * 4.0 - 1.0;
    return vec4(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (pos.xy - u.resolution * 0.5) / u.resolution.y;

    let speed     = param(0);
    let var_amt   = param(1);
    let gain      = param(2);
    let color_spd = param(3);

    let t = u.time * speed;

    // ── Transform 1: affine + sinusoidal variation ──
    let a1 = rot2(0.7 + t * 0.13) * uv * 0.55
           + vec2(0.3 * cos(t * 0.7), 0.2 * sin(t * 0.5));
    let s1 = mix(a1, vec2(sin(a1.x * PI), sin(a1.y * PI)), var_amt);

    // ── Transform 2: affine + swirl variation ──
    let a2 = rot2(-0.9 + t * 0.17) * uv * 0.6
           + vec2(-0.25 + 0.1 * sin(t * 0.3), 0.3 * cos(t * 0.4));
    let r2 = dot(a2, a2);
    let sw = vec2(a2.x * sin(r2) - a2.y * cos(r2),
                  a2.x * cos(r2) + a2.y * sin(r2));
    let s2 = mix(a2, sw, var_amt);

    // ── Transform 3: affine + spherical variation ──
    let a3 = rot2(2.1 + t * 0.09) * uv * 0.5
           + vec2(0.15 * sin(t * 0.6), -0.25 + 0.1 * cos(t * 0.8));
    let s3 = mix(a3, a3 / (dot(a3, a3) + 1e-4), var_amt);

    // ── Sample previous frame at transformed positions ──
    let c1 = textureSample(prev_frame, prev_sampler, to_tex(s1)).rgb;
    let c2 = textureSample(prev_frame, prev_sampler, to_tex(s2)).rgb;
    let c3 = textureSample(prev_frame, prev_sampler, to_tex(s3)).rgb;

    // ── Accumulate brightness ──
    var col = (c1 + c2 + c3) * gain;

    // ── Seed: orbiting dot that feeds energy into the system ──
    let seed_pos = vec2(0.1 * cos(t * 0.5), 0.1 * sin(t * 0.7));
    let d = length(uv - seed_pos);
    col += vec3(smoothstep(0.02, 0.0, d) * 0.4);

    // ── Color from intensity ──
    // Use log-density like real fractal flames
    let density = length(col);
    let log_d = log(1.0 + density * 10.0) / log(11.0);
    let hue = fract(log_d * 2.0 + u.time * color_spd);
    let final_col = hsv(hue, 0.75, log_d);

    // Soft tonemap to prevent blowout
    let mapped = final_col / (1.0 + final_col);

    return vec4(mapped, 1.0);
}
