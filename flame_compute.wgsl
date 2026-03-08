// ── Fractal Flame Compute Shader ──

struct Uniforms {
    time: f32,
    frame: u32,
    resolution: vec2<f32>,
    mouse: vec2<f32>,
    transform_count: u32,
    has_final_xform: u32,
    globals: vec4<f32>,   // speed, zoom, trail, flame_brightness
    kifs: vec4<f32>,      // fold_angle, scale, brightness, drift_speed
    extra: vec4<f32>,     // color_shift, vibrancy, bloom_intensity, symmetry
    extra2: vec4<f32>,     // noise_disp, curl_disp, tangent_clamp, color_blend
    extra3: vec4<f32>,     // spin_speed_max, position_drift, warmup_iters, reserved
}

@group(0) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(1) var<uniform> u: Uniforms;
@group(0) @binding(2) var<storage, read> transforms: array<f32>;
@group(0) @binding(3) var palette_tex: texture_2d<f32>;
@group(0) @binding(4) var palette_sampler: sampler;
@group(0) @binding(5) var<storage, read_write> point_state: array<f32>;

fn xf(idx: u32, field: u32) -> f32 { return transforms[idx * 42u + field]; }

fn xf_param(idx: u32, param_offset: u32) -> f32 {
    return transforms[idx * 42u + 34u + param_offset];
}

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

fn cosh_f(x: f32) -> f32 { return (exp(x) + exp(-x)) * 0.5; }
fn sinh_f(x: f32) -> f32 { return (exp(x) - exp(-x)) * 0.5; }

// ── Per-transform hash noise (non-repeating drift) ──

fn hash_u(n: u32) -> u32 {
    var x = n;
    x = x ^ (x >> 16u);
    x = x * 2654435769u;
    x = x ^ (x >> 16u);
    return x;
}

fn hash_f(n: u32) -> f32 {
    return f32(hash_u(n) & 0xFFFFu) / 65535.0;
}

// 1D smooth value noise
fn vnoise(t: f32, seed: u32) -> f32 {
    let i = i32(floor(t));
    let f = t - floor(t);
    let s = f * f * (3.0 - 2.0 * f); // smoothstep
    let a = hash_f(seed + u32(i) * 7919u) * 2.0 - 1.0;
    let b = hash_f(seed + u32(i + 1) * 7919u) * 2.0 - 1.0;
    return a + (b - a) * s;
}

const TAU: f32 = 6.28318530;

// Palette lookup via 256x1 texture (uploaded from CPU)
fn palette(t: f32) -> vec3<f32> {
    let uv = vec2(fract(t), 0.5);
    return textureSampleLevel(palette_tex, palette_sampler, uv, 0.0).rgb;
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
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

fn V_julia(p: vec2<f32>, rng: ptr<function, u32>) -> vec2<f32> {
    let r = sqrt(length(p));
    let theta = atan2(p.y, p.x) * 0.5;
    let k = f32(pcg(rng) & 1u) * PI;
    return r * vec2(cos(theta + k), sin(theta + k));
}

fn V_polar(p: vec2<f32>) -> vec2<f32> {
    let theta = atan2(p.y, p.x);
    let r = length(p);
    return vec2(theta / PI, r - 1.0);
}

fn V_disc(p: vec2<f32>) -> vec2<f32> {
    let theta = atan2(p.y, p.x);
    let r = length(p);
    let f = theta / PI;
    return f * vec2(sin(PI * r), cos(PI * r));
}

fn V_rings(p: vec2<f32>, idx: u32) -> vec2<f32> {
    let r = length(p);
    let theta = atan2(p.y, p.x);
    let val = xf_param(idx, 0u);  // rings2_val [34]
    let val2 = val * val + 1e-6;
    let rr = r + val2 - 2.0 * val2 * floor((r + val2) / (2.0 * val2)) + r * (1.0 - val2);
    return rr * vec2(cos(theta), sin(theta));
}

fn V_bubble(p: vec2<f32>) -> vec2<f32> {
    let r2 = dot(p, p);
    return p * 4.0 / (r2 + 4.0);
}

fn V_fisheye(p: vec2<f32>) -> vec2<f32> {
    let r = length(p);
    return 2.0 * p.yx / (r + 1.0);
}

fn V_exponential(p: vec2<f32>) -> vec2<f32> {
    let e = exp(p.x - 1.0);
    return e * vec2(cos(PI * p.y), sin(PI * p.y));
}

fn V_spiral(p: vec2<f32>) -> vec2<f32> {
    let r = length(p) + 1e-6;
    let theta = atan2(p.y, p.x);
    return vec2(cos(theta) + sin(r), sin(theta) - cos(r)) / r;
}

fn V_diamond(p: vec2<f32>) -> vec2<f32> {
    let r = length(p);
    let theta = atan2(p.y, p.x);
    return vec2(sin(theta) * cos(r), cos(theta) * sin(r));
}

fn V_bent(p: vec2<f32>) -> vec2<f32> {
    var q = p;
    if (q.x < 0.0) { q.x *= 2.0; }
    if (q.y < 0.0) { q.y *= 0.5; }
    return q;
}

fn V_waves(p: vec2<f32>, bx: f32, by: f32) -> vec2<f32> {
    return vec2(
        p.x + bx * sin(p.y * 4.0),
        p.y + by * sin(p.x * 4.0)
    );
}

fn V_popcorn(p: vec2<f32>, cx: f32, cy: f32) -> vec2<f32> {
    return vec2(
        p.x + cx * sin(tan(3.0 * p.y)),
        p.y + cy * sin(tan(3.0 * p.x))
    );
}

fn V_fan(p: vec2<f32>, fan_t: f32) -> vec2<f32> {
    let theta = atan2(p.y, p.x);
    let r = length(p);
    let t2 = PI * fan_t * fan_t + 1e-6;
    if ((theta + fan_t) % t2 > t2 * 0.5) {
        return r * vec2(cos(theta - t2 * 0.5), sin(theta - t2 * 0.5));
    } else {
        return r * vec2(cos(theta + t2 * 0.5), sin(theta + t2 * 0.5));
    }
}

fn V_eyefish(p: vec2<f32>) -> vec2<f32> {
    let r = length(p);
    return 2.0 * p / (r + 1.0);
}

fn V_cross(p: vec2<f32>) -> vec2<f32> {
    let s = p.x * p.x - p.y * p.y;
    return sqrt(1.0 / (s * s + 1e-6)) * p;
}

fn V_tangent(p: vec2<f32>) -> vec2<f32> {
    let tc = u.extra2.z;  // tangent_clamp from weights
    return clamp(vec2(sin(p.x) / (cos(p.y) + 1e-6), tan(p.y)), vec2(-tc), vec2(tc));
}

fn V_cosine(p: vec2<f32>) -> vec2<f32> {
    return vec2(
        cos(PI * p.x) * cosh_f(p.y),
        -sin(PI * p.x) * sinh_f(p.y)
    );
}

fn V_blob(p: vec2<f32>, idx: u32) -> vec2<f32> {
    let r = length(p);
    let theta = atan2(p.y, p.x);
    let low = xf_param(idx, 1u);    // blob_low [35]
    let high = xf_param(idx, 2u);   // blob_high [36]
    let waves = xf_param(idx, 3u);  // blob_waves [37]
    let rr = r * (low + (high - low) * 0.5 * (sin(waves * theta) + 1.0));
    return rr * vec2(cos(theta), sin(theta));
}

fn V_noise(p: vec2<f32>, seed: u32) -> vec2<f32> {
    let nx = vnoise(p.x * 3.0, seed + 500u);
    let ny = vnoise(p.y * 3.0, seed + 600u);
    return p + vec2(nx, ny) * u.extra2.x;  // noise_displacement from weights
}

fn V_curl(p: vec2<f32>, seed: u32) -> vec2<f32> {
    let eps = 0.01;
    let n00 = vnoise(p.x * 2.0, seed + 700u) + vnoise(p.y * 2.0, seed + 800u);
    let n10 = vnoise((p.x + eps) * 2.0, seed + 700u) + vnoise(p.y * 2.0, seed + 800u);
    let n01 = vnoise(p.x * 2.0, seed + 700u) + vnoise((p.y + eps) * 2.0, seed + 800u);
    let dx = (n10 - n00) / eps;
    let dy = (n01 - n00) / eps;
    return p + vec2(dy, -dx) * u.extra2.y;  // curl_displacement from weights
}

// ── IFS Transform (reads from storage buffer) ──

fn apply_xform(p: vec2<f32>, idx: u32, t: f32, rng: ptr<function, u32>) -> vec2<f32> {
    let af_a   = xf(idx, 1u);
    let af_b   = xf(idx, 2u);
    let af_c   = xf(idx, 3u);
    let af_d   = xf(idx, 4u);
    let ox     = xf(idx, 5u);
    let oy     = xf(idx, 6u);
    let w_lin  = xf(idx, 8u);
    let w_sin  = xf(idx, 9u);
    let w_sph  = xf(idx, 10u);
    let w_swi  = xf(idx, 11u);
    let w_hor  = xf(idx, 12u);
    let w_han  = xf(idx, 13u);

    let drift = u.kifs.w; // drift_speed

    // Per-transform seeded drift — all multipliers from uniforms
    let seed = hash_u(idx * 31337u + 42u);
    let drift_amt = 0.3 + hash_f(seed) * 0.7; // 0.3–1.0 per transform
    let spin_max = u.extra3.x;  // spin_speed_max from weights
    let pos_drift = u.extra3.y; // position_drift from weights
    let spin_speed = (hash_f(seed + 300u) * 2.0 - 1.0) * spin_max;
    let angle_drift = t * spin_speed * drift * drift_amt;
    let ox_drift = vnoise(t * 0.03 * drift * drift_amt, seed + 100u) * pos_drift;
    let oy_drift = vnoise(t * 0.04 * drift * drift_amt, seed + 200u) * pos_drift;

    // Apply spin drift as a rotation on top of the affine matrix
    let spin_cos = cos(angle_drift);
    let spin_sin = sin(angle_drift);
    let a2 = af_a * spin_cos - af_c * spin_sin;
    let b2 = af_b * spin_cos - af_d * spin_sin;
    let c2 = af_a * spin_sin + af_c * spin_cos;
    let d2 = af_b * spin_sin + af_d * spin_cos;

    let q = vec2(a2 * p.x + b2 * p.y + ox + ox_drift,
                 c2 * p.x + d2 * p.y + oy + oy_drift);

    // Compute effective scale for variations that need it (rings, etc.)
    let eff_scale = sqrt(abs(af_a * af_d - af_b * af_c));

    // Skip zero-weight variations to save compute
    var v = q * w_lin;
    if (w_sin > 0.0) { v += V_sinusoidal(q)    * w_sin; }
    if (w_sph > 0.0) { v += V_spherical(q)     * w_sph; }
    if (w_swi > 0.0) { v += V_swirl(q)         * w_swi; }
    if (w_hor > 0.0) { v += V_horseshoe(q)     * w_hor; }
    if (w_han > 0.0) { v += V_handkerchief(q)  * w_han; }

    let w_jul  = xf(idx, 14u);
    let w_pol  = xf(idx, 15u);
    let w_dsc  = xf(idx, 16u);
    let w_rng  = xf(idx, 17u);
    let w_bub  = xf(idx, 18u);
    let w_fsh  = xf(idx, 19u);
    let w_exp  = xf(idx, 20u);
    let w_spi  = xf(idx, 21u);

    if (w_jul > 0.0) { v += V_julia(q, rng)                * w_jul; }
    if (w_pol > 0.0) { v += V_polar(q)                     * w_pol; }
    if (w_dsc > 0.0) { v += V_disc(q)                      * w_dsc; }
    if (w_rng > 0.0) { v += V_rings(q, idx) * w_rng; }
    if (w_bub > 0.0) { v += V_bubble(q)                    * w_bub; }
    if (w_fsh > 0.0) { v += V_fisheye(q)                   * w_fsh; }
    if (w_exp > 0.0) { v += V_exponential(q)               * w_exp; }
    if (w_spi > 0.0) { v += V_spiral(q)                    * w_spi; }

    let w_dia  = xf(idx, 22u);
    let w_bnt  = xf(idx, 23u);
    let w_wav  = xf(idx, 24u);
    let w_pop  = xf(idx, 25u);
    let w_fan  = xf(idx, 26u);
    let w_eye  = xf(idx, 27u);
    let w_crs  = xf(idx, 28u);
    let w_tan  = xf(idx, 29u);
    let w_cos  = xf(idx, 30u);
    let w_blb  = xf(idx, 31u);
    let w_noi  = xf(idx, 32u);
    let w_crl  = xf(idx, 33u);

    // Compute effective angle for fan variation
    let eff_angle = atan2(af_c, af_a);

    if (w_dia > 0.0) { v += V_diamond(q)                        * w_dia; }
    if (w_bnt > 0.0) { v += V_bent(q)                           * w_bnt; }
    if (w_wav > 0.0) { v += V_waves(q, ox * 0.5, oy * 0.5)     * w_wav; }
    if (w_pop > 0.0) { v += V_popcorn(q, ox * 0.3, oy * 0.3)   * w_pop; }
    if (w_fan > 0.0) { v += V_fan(q, eff_angle)                 * w_fan; }
    if (w_eye > 0.0) { v += V_eyefish(q)                        * w_eye; }
    if (w_crs > 0.0) { v += V_cross(q)                          * w_crs; }
    if (w_tan > 0.0) { v += V_tangent(q)                        * w_tan; }
    if (w_cos > 0.0) { v += V_cosine(q)                         * w_cos; }
    if (w_blb > 0.0) { v += V_blob(q, idx)                       * w_blb; }
    if (w_noi > 0.0) { v += V_noise(q, seed)                    * w_noi; }
    if (w_crl > 0.0) { v += V_curl(q, seed)                     * w_crl; }

    return v;
}

fn xform_color(idx: u32) -> f32 {
    return xf(idx, 7u);
}

// Soft-splat a point using bilinear weighting to the 4 nearest pixels.
// This turns each chaos-game hit into a smooth sub-pixel contribution,
// filling gaps and producing curved-looking surfaces instead of stipple dots.
fn splat_point(cx: i32, cy: i32, fx: f32, fy: f32, col: vec3<f32>, vel: vec2<f32>, w: u32, h: u32) {
    // Bilinear weights: distribute the point's contribution across a 2x2 quad
    // based on sub-pixel position (fx, fy are the fractional offsets 0-1)
    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let ic = u32(col.x * 1000.0);
    let ig = u32(col.y * 1000.0);
    let ib = u32(col.z * 1000.0);
    let ivx = i32(vel.x * 10000.0);
    let ivy = i32(vel.y * 10000.0);

    // Center pixel (always valid — caller checked bounds)
    let idx00 = (u32(cy) * w + u32(cx)) * 6u;
    let d00 = u32(w00 * 1000.0);
    atomicAdd(&histogram[idx00], d00);
    atomicAdd(&histogram[idx00 + 1u], u32(f32(ic) * w00));
    atomicAdd(&histogram[idx00 + 2u], u32(f32(ig) * w00));
    atomicAdd(&histogram[idx00 + 3u], u32(f32(ib) * w00));
    atomicAdd(&histogram[idx00 + 4u], u32(i32(f32(ivx) * w00)));
    atomicAdd(&histogram[idx00 + 5u], u32(i32(f32(ivy) * w00)));

    // Right neighbor
    let nx = cx + 1;
    if (nx < i32(w)) {
        let idx10 = (u32(cy) * w + u32(nx)) * 6u;
        let d10 = u32(w10 * 1000.0);
        atomicAdd(&histogram[idx10], d10);
        atomicAdd(&histogram[idx10 + 1u], u32(f32(ic) * w10));
        atomicAdd(&histogram[idx10 + 2u], u32(f32(ig) * w10));
        atomicAdd(&histogram[idx10 + 3u], u32(f32(ib) * w10));
        atomicAdd(&histogram[idx10 + 4u], u32(i32(f32(ivx) * w10)));
        atomicAdd(&histogram[idx10 + 5u], u32(i32(f32(ivy) * w10)));
    }

    // Bottom neighbor
    let ny = cy + 1;
    if (ny < i32(h)) {
        let idx01 = (u32(ny) * w + u32(cx)) * 6u;
        let d01 = u32(w01 * 1000.0);
        atomicAdd(&histogram[idx01], d01);
        atomicAdd(&histogram[idx01 + 1u], u32(f32(ic) * w01));
        atomicAdd(&histogram[idx01 + 2u], u32(f32(ig) * w01));
        atomicAdd(&histogram[idx01 + 3u], u32(f32(ib) * w01));
        atomicAdd(&histogram[idx01 + 4u], u32(i32(f32(ivx) * w01)));
        atomicAdd(&histogram[idx01 + 5u], u32(i32(f32(ivy) * w01)));
    }

    // Bottom-right corner
    if (nx < i32(w) && ny < i32(h)) {
        let idx11 = (u32(ny) * w + u32(nx)) * 6u;
        let d11 = u32(w11 * 1000.0);
        atomicAdd(&histogram[idx11], d11);
        atomicAdd(&histogram[idx11 + 1u], u32(f32(ic) * w11));
        atomicAdd(&histogram[idx11 + 2u], u32(f32(ig) * w11));
        atomicAdd(&histogram[idx11 + 3u], u32(f32(ib) * w11));
        atomicAdd(&histogram[idx11 + 4u], u32(i32(f32(ivx) * w11)));
        atomicAdd(&histogram[idx11 + 5u], u32(i32(f32(ivy) * w11)));
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let speed = u.globals.x;
    let zoom = u.globals.y;
    let t = u.time * speed;
    let num_xf = u.transform_count;

    var rng = gid.x * 2654435761u + u.frame * 7919u + 12345u;

    // Read persistent point state (survives across frames/genome changes)
    let state_idx = gid.x * 3u;
    var p = vec2(point_state[state_idx], point_state[state_idx + 1u]);
    var color_idx = point_state[state_idx + 2u];

    // Re-randomize if: first frame (all zeros), escaped, NaN, or periodic refresh
    // 5% of threads re-randomize each frame to maintain fresh attractor coverage
    let refresh = randf(&rng) < 0.05;
    let needs_init = (abs(p.x) < 1e-10 && abs(p.y) < 1e-10 && color_idx < 1e-10)
                   || abs(p.x) > 10.0 || abs(p.y) > 10.0  // tighter bound — well outside any zoom range
                   || p.x != p.x || p.y != p.y  // NaN check
                   || refresh;
    if (needs_init) {
        p = vec2(randf(&rng) * 4.0 - 2.0, randf(&rng) * 4.0 - 2.0);
        color_idx = randf(&rng);
    }

    let w = u32(u.resolution.x);
    let h = u32(u.resolution.y);

    // Precompute total weight
    var total_weight = 0.0;
    for (var t_idx = 0u; t_idx < num_xf; t_idx++) {
        total_weight += xf(t_idx, 0u);
    }
    if (total_weight < 1e-6) { return; }

    var prev_p = p;

    for (var i = 0u; i < 500u; i++) {
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

        prev_p = p;
        p = apply_xform(p, tidx, t, &rng);
        let cb = u.extra2.w;  // color_blend from weights
        color_idx = color_idx * (1.0 - cb) + xform_color(tidx) * cb;

        if (i < u32(u.extra3.z)) { continue; }

        // Velocity in screen space (for directional blur)
        let vel = (p - prev_p) / zoom;

        // Final transform (if present)
        var plot_p = p;
        var plot_color = color_idx;
        if (u.has_final_xform == 1u) {
            let final_idx = u.transform_count;  // final xform is right after regular xforms
            plot_p = apply_xform(plot_p, final_idx, t, &rng);
            plot_color = plot_color * 0.5 + xf(final_idx, 7u) * 0.5;  // blend with final xform's color
        }

        // Symmetry: plot rotated/mirrored copies
        let sym = i32(u.extra.w);
        let abs_sym = max(abs(sym), 1);
        let bilateral = sym < 0;

        for (var si = 0; si < abs_sym; si++) {
            let sym_angle = f32(si) * TAU / f32(abs_sym);
            let cp = cos(sym_angle);
            let sp = sin(sym_angle);
            let sym_p = vec2(plot_p.x * cp - plot_p.y * sp, plot_p.x * sp + plot_p.y * cp);

            let screen = (sym_p / zoom + vec2(0.5, 0.5)) * vec2<f32>(f32(w), f32(h));
            let px_x = i32(screen.x);
            let px_y = i32(screen.y);

            if (px_x >= 0 && px_x < i32(w) && px_y >= 0 && px_y < i32(h)) {
                let col = palette(plot_color + u.extra.x);
                // Sub-pixel offset for soft splatting
                let frac_x = screen.x - f32(px_x);
                let frac_y = screen.y - f32(px_y);
                splat_point(px_x, px_y, frac_x, frac_y, col, vel, w, h);
            }

            // Bilateral mirror
            if (bilateral) {
                let mir_p = vec2(-sym_p.x, sym_p.y);
                let mscreen = (mir_p / zoom + vec2(0.5, 0.5)) * vec2<f32>(f32(w), f32(h));
                let mpx_x = i32(mscreen.x);
                let mpx_y = i32(mscreen.y);
                if (mpx_x >= 0 && mpx_x < i32(w) && mpx_y >= 0 && mpx_y < i32(h)) {
                    let mcol = palette(plot_color + u.extra.x);
                    let mfrac_x = mscreen.x - f32(mpx_x);
                    let mfrac_y = mscreen.y - f32(mpx_y);
                    let mvel = vec2(-vel.x, vel.y);
                    splat_point(mpx_x, mpx_y, mfrac_x, mfrac_y, mcol, mvel, w, h);
                }
            }
        }
    }

    // Write back persistent point state for next frame
    point_state[state_idx] = p.x;
    point_state[state_idx + 1u] = p.y;
    point_state[state_idx + 2u] = color_idx;
}
