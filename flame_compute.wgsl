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
    extra2: vec4<f32>,     // crossfade_alpha, reserved, reserved, reserved
}

@group(0) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(1) var<uniform> u: Uniforms;
@group(0) @binding(2) var<storage, read> transforms: array<f32>;

fn xf(idx: u32, field: u32) -> f32 { return transforms[idx * 32u + field]; }

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

fn palette(t: f32) -> vec3<f32> {
    let a = vec3(0.5, 0.5, 0.5);
    let b = vec3(0.5, 0.5, 0.5);
    let c = vec3(1.0, 1.0, 1.0);
    let d = vec3(0.00, 0.33, 0.67);
    return a + b * cos(TAU * (c * t + d));
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

fn V_rings(p: vec2<f32>, c2: f32) -> vec2<f32> {
    let r = length(p);
    let theta = atan2(p.y, p.x);
    let k = c2 + 1e-6;
    let rr = ((r + k) % (2.0 * k)) - k + r * (1.0 - k);
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

fn V_blob(p: vec2<f32>) -> vec2<f32> {
    let r = length(p);
    let theta = atan2(p.y, p.x);
    let blobr = r * (0.5 + 0.5 * sin(3.0 * theta));
    return blobr * vec2(cos(theta), sin(theta));
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

    // Per-transform seeded drift
    let seed = hash_u(idx * 31337u + 42u);
    let drift_amt = 0.3 + hash_f(seed) * 0.7; // 0.3–1.0 per transform
    // Steady spin — each transform rotates at its own constant speed, never reverses
    let spin_speed = (hash_f(seed + 300u) * 2.0 - 1.0) * 0.15; // ±0.15 rad/s
    let angle_drift = t * spin_speed * drift * drift_amt;
    // Position: gentle bounded noise wander
    let ox_drift = vnoise(t * 0.03 * drift * drift_amt, seed + 100u) * 0.08;
    let oy_drift = vnoise(t * 0.04 * drift * drift_amt, seed + 200u) * 0.08;

    let q = rot2(angle + angle_drift) * p * scale
          + vec2(ox + ox_drift, oy + oy_drift);

    // Skip zero-weight variations to save compute
    var v = q * w_lin;
    if (w_sin > 0.0) { v += V_sinusoidal(q)    * w_sin; }
    if (w_sph > 0.0) { v += V_spherical(q)     * w_sph; }
    if (w_swi > 0.0) { v += V_swirl(q)         * w_swi; }
    if (w_hor > 0.0) { v += V_horseshoe(q)     * w_hor; }
    if (w_han > 0.0) { v += V_handkerchief(q)  * w_han; }

    let w_jul  = xf(idx, 12u);
    let w_pol  = xf(idx, 13u);
    let w_dsc  = xf(idx, 14u);
    let w_rng  = xf(idx, 15u);
    let w_bub  = xf(idx, 16u);
    let w_fsh  = xf(idx, 17u);
    let w_exp  = xf(idx, 18u);
    let w_spi  = xf(idx, 19u);

    if (w_jul > 0.0) { v += V_julia(q, rng)           * w_jul; }
    if (w_pol > 0.0) { v += V_polar(q)                * w_pol; }
    if (w_dsc > 0.0) { v += V_disc(q)                 * w_dsc; }
    if (w_rng > 0.0) { v += V_rings(q, scale * scale) * w_rng; }
    if (w_bub > 0.0) { v += V_bubble(q)               * w_bub; }
    if (w_fsh > 0.0) { v += V_fisheye(q)              * w_fsh; }
    if (w_exp > 0.0) { v += V_exponential(q)          * w_exp; }
    if (w_spi > 0.0) { v += V_spiral(q)               * w_spi; }

    let w_dia  = xf(idx, 20u);
    let w_bnt  = xf(idx, 21u);
    let w_wav  = xf(idx, 22u);
    let w_pop  = xf(idx, 23u);
    let w_fan  = xf(idx, 24u);
    let w_eye  = xf(idx, 25u);
    let w_crs  = xf(idx, 26u);
    let w_tan  = xf(idx, 27u);
    let w_cos  = xf(idx, 28u);
    let w_blb  = xf(idx, 29u);
    let w_noi  = xf(idx, 30u);
    let w_crl  = xf(idx, 31u);

    if (w_dia > 0.0) { v += V_diamond(q)                        * w_dia; }
    if (w_bnt > 0.0) { v += V_bent(q)                           * w_bnt; }
    if (w_wav > 0.0) { v += V_waves(q, ox * 0.5, oy * 0.5)     * w_wav; }
    if (w_pop > 0.0) { v += V_popcorn(q, ox * 0.3, oy * 0.3)   * w_pop; }
    if (w_fan > 0.0) { v += V_fan(q, angle)                     * w_fan; }
    if (w_eye > 0.0) { v += V_eyefish(q)                        * w_eye; }
    if (w_crs > 0.0) { v += V_cross(q)                          * w_crs; }
    if (w_tan > 0.0) { v += V_tangent(q)                        * w_tan; }
    if (w_cos > 0.0) { v += V_cosine(q)                         * w_cos; }
    if (w_blb > 0.0) { v += V_blob(q)                           * w_blb; }
    if (w_noi > 0.0) { v += V_noise(q, seed)                    * w_noi; }
    if (w_crl > 0.0) { v += V_curl(q, seed)                     * w_crl; }

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

    for (var i = 0u; i < 100u; i++) {
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

        p = apply_xform(p, tidx, t, &rng);
        color_idx = color_idx * 0.3 + xform_color(tidx) * 0.7;

        if (i < 20u) { continue; }

        // Final transform (if present)
        var plot_p = p;
        var plot_color = color_idx;
        if (u.has_final_xform == 1u) {
            let final_idx = u.transform_count;  // final xform is right after regular xforms
            plot_p = apply_xform(plot_p, final_idx, t, &rng);
            plot_color = plot_color * 0.5 + xf(final_idx, 5u) * 0.5;  // blend with final xform's color
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
                let buf_idx = (u32(px_y) * w + u32(px_x)) * 4u;
                let col = palette(plot_color + u.extra.x);
                atomicAdd(&histogram[buf_idx], 1u);
                atomicAdd(&histogram[buf_idx + 1u], u32(col.x * 1000.0));
                atomicAdd(&histogram[buf_idx + 2u], u32(col.y * 1000.0));
                atomicAdd(&histogram[buf_idx + 3u], u32(col.z * 1000.0));
            }

            // Bilateral mirror
            if (bilateral) {
                let mir_p = vec2(-sym_p.x, sym_p.y);
                let mscreen = (mir_p / zoom + vec2(0.5, 0.5)) * vec2<f32>(f32(w), f32(h));
                let mpx_x = i32(mscreen.x);
                let mpx_y = i32(mscreen.y);
                if (mpx_x >= 0 && mpx_x < i32(w) && mpx_y >= 0 && mpx_y < i32(h)) {
                    let mbuf_idx = (u32(mpx_y) * w + u32(mpx_x)) * 4u;
                    let mcol = palette(plot_color + u.extra.x);
                    atomicAdd(&histogram[mbuf_idx], 1u);
                    atomicAdd(&histogram[mbuf_idx + 1u], u32(mcol.x * 1000.0));
                    atomicAdd(&histogram[mbuf_idx + 2u], u32(mcol.y * 1000.0));
                    atomicAdd(&histogram[mbuf_idx + 3u], u32(mcol.z * 1000.0));
                }
            }
        }
    }
}
