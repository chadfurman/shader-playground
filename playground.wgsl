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
    extra3: vec4<f32>,      // spin_speed_max, position_drift, warmup_iters, velocity_blur_max
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var prev_frame: texture_2d<f32>;
@group(0) @binding(2) var prev_sampler: sampler;
@group(0) @binding(3) var<storage, read> accumulation: array<f32>;
@group(0) @binding(4) var crossfade_tex: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> max_density_buf: array<u32>;

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
    let h = u32(u.resolution.y);
    let tex_uv = pos.xy / u.resolution;

    let trail        = u.globals.z;
    let flame_bright = u.globals.w;
    let vibrancy     = u.extra.y;
    let bloom_int    = u.extra.z;
    let gamma        = u.extra.w;
    let blur_max     = u.extra3.w;  // velocity_blur_max from config

    // ── Read accumulation buffer (6 channels: density, R, G, B, vx, vy) ──
    let buf_idx = (px.y * w + px.x) * 6u;
    let density = accumulation[buf_idx];
    let acc_r = accumulation[buf_idx + 1u];
    let acc_g = accumulation[buf_idx + 2u];
    let acc_b = accumulation[buf_idx + 3u];

    // Read accumulated velocity (signed fixed-point * 10000, weighted by bilinear splat)
    let vel_raw = vec2(accumulation[buf_idx + 4u], accumulation[buf_idx + 5u]);
    // Average velocity: vel stored as vel*10000*w, density as 1000*w, ratio = vel*10
    // So divide by density * 10 to recover velocity, then scale to pixel space
    let avg_vel = select(
        vec2(0.0),
        vel_raw / (density * 10.0) * u.resolution,
        density > 1.0
    );
    // Blur length proportional to velocity, clamped to configurable max
    let vel_len = length(avg_vel);
    let blur_dir = select(vec2(0.0), avg_vel / vel_len, vel_len > 0.5);
    let blur_len = clamp(vel_len, 0.0, blur_max);
    // Scale tap spacing so we always cover the full blur length with 8 taps
    let tap_spacing = blur_len / 8.0;

    // ── Directional blur along velocity field ──
    // 8 taps forward + 8 taps backward for smooth velocity streaks
    var blur_density = density;
    var blur_r = acc_r;
    var blur_g = acc_g;
    var blur_b = acc_b;

    if (blur_len > 0.5) {
        let step = blur_dir * tap_spacing;
        for (var tap = 1; tap <= 8; tap++) {
            let offset = step * f32(tap);
            let sx = clamp(i32(f32(px.x) + offset.x), 0, i32(w) - 1);
            let sy = clamp(i32(f32(px.y) + offset.y), 0, i32(h) - 1);
            let si = (u32(sy) * w + u32(sx)) * 6u;
            let sd = accumulation[si];
            // Weight taps by distance falloff
            let weight = 1.0 / (1.0 + f32(tap) * 0.3);
            blur_density += sd * weight;
            blur_r += accumulation[si + 1u] * weight;
            blur_g += accumulation[si + 2u] * weight;
            blur_b += accumulation[si + 3u] * weight;

            // Also sample in reverse direction
            let rx = clamp(i32(f32(px.x) - offset.x), 0, i32(w) - 1);
            let ry = clamp(i32(f32(px.y) - offset.y), 0, i32(h) - 1);
            let ri = (u32(ry) * w + u32(rx)) * 6u;
            let rd = accumulation[ri];
            blur_density += rd * weight;
            blur_r += accumulation[ri + 1u] * weight;
            blur_g += accumulation[ri + 2u] * weight;
            blur_b += accumulation[ri + 3u] * weight;
        }
    }

    // ── Per-image normalized log-density tonemapping ──
    // Density is stored as fixed-point * 1000 (from bilinear splatting)
    let density_hits = blur_density / 1000.0;
    let log_density = log(1.0 + density_hits * flame_bright);

    // Normalize against max density found by the accumulation pass.
    // This ensures every genome uses the full brightness range automatically.
    let max_density_bits = max_density_buf[0];
    let max_density_val = bitcast<f32>(max_density_bits) / 1000.0;
    let max_log = log(1.0 + max_density_val * flame_bright);
    let alpha = sqrt(log_density / max(max_log, 0.001));

    // Recover average color (RGB and density both stored as fixed-point * 1000)
    let raw_color = select(
        vec3(0.0),
        vec3(blur_r, blur_g, blur_b) / max(blur_density, 1.0),
        blur_density > 0.0
    );

    // ── Density + color edge detection — reveals structure in overlapping regions ──
    let lx = clamp(i32(px.x) - 1, 0, i32(w) - 1);
    let rx2 = clamp(i32(px.x) + 1, 0, i32(w) - 1);
    let ty = clamp(i32(px.y) - 1, 0, i32(h) - 1);
    let by = clamp(i32(px.y) + 1, 0, i32(h) - 1);

    // Density gradient
    let li = (u32(px.y) * w + u32(lx)) * 6u;
    let ri = (u32(px.y) * w + u32(rx2)) * 6u;
    let ti = (u32(ty) * w + px.x) * 6u;
    let bi = (u32(by) * w + px.x) * 6u;
    let d_left  = accumulation[li];
    let d_right = accumulation[ri];
    let d_top   = accumulation[ti];
    let d_bot   = accumulation[bi];
    let grad_x = d_right - d_left;
    let grad_y = d_bot - d_top;
    let density_edge = sqrt(grad_x * grad_x + grad_y * grad_y);

    // Color gradient — where different transforms overlap, average color shifts
    // Both density and color use same fixed-point scale, so divide color by density directly
    let col_left  = vec3(accumulation[li + 1u], accumulation[li + 2u], accumulation[li + 3u]) / max(d_left, 1.0);
    let col_right = vec3(accumulation[ri + 1u], accumulation[ri + 2u], accumulation[ri + 3u]) / max(d_right, 1.0);
    let col_top   = vec3(accumulation[ti + 1u], accumulation[ti + 2u], accumulation[ti + 3u]) / max(d_top, 1.0);
    let col_bot   = vec3(accumulation[bi + 1u], accumulation[bi + 2u], accumulation[bi + 3u]) / max(d_bot, 1.0);
    let color_grad_x = length(col_right - col_left);
    let color_grad_y = length(col_bot - col_top);
    let color_edge = sqrt(color_grad_x * color_grad_x + color_grad_y * color_grad_y);

    // Combine: density edges show structure boundaries, color edges show transform overlaps
    let edge_glow = clamp(density_edge / max(blur_density * 0.3, 1.0), 0.0, 0.5)
                  + clamp(color_edge * 2.0, 0.0, 0.3);

    // ── Flam3 vibrancy color blend ──
    let gamma_alpha = pow(max(alpha, 0.001), gamma);
    let ls = vibrancy * alpha + (1.0 - vibrancy) * gamma_alpha;
    var col = (ls + edge_glow) * raw_color;

    // ── Feedback trail — gentle temporal smoothing ──
    let prev = textureSample(prev_frame, prev_sampler, tex_uv).rgb;
    col = max(col, prev * trail);  // max-blend, not additive — prevents brightness buildup

    // ── Density-aware bloom — soft glow on sparse points, crisp dense structure ──
    if (bloom_int > 0.001) {
        // Bloom weight inversely proportional to density: sparse pixels get full bloom,
        // dense pixels get almost none. Uses normalized density (relative to max).
        let max_d = bitcast<f32>(max_density_buf[0]);
        let norm_density = clamp(density / max(max_d, 1.0), 0.0, 1.0);
        let sparsity = 1.0 - sqrt(norm_density);  // sqrt for gentler falloff

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

        col += bloom_sum * bloom_int * sparsity;
    }

    // ── Soft clamp to prevent >1.0 without crushing dynamic range ──
    // No Reinhard, no extra gamma — the vibrancy blend already did gamma.
    col = clamp(col, vec3(0.0), vec3(1.0));

    return vec4(col, 1.0);
}
