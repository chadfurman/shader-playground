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
    extra4: vec4<f32>,   // jitter_amount, tonemap_mode, histogram_equalization, dof_strength
    extra5: vec4<f32>,   // dof_focal_distance, spectral_rendering, temporal_reprojection, prev_zoom
    extra6: vec4<f32>,   // dist_lum_strength, iter_lum_range, reserved, reserved
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var prev_frame: texture_2d<f32>;
@group(0) @binding(2) var prev_sampler: sampler;
@group(0) @binding(3) var<storage, read> accumulation: array<f32>;
@group(0) @binding(4) var crossfade_tex: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> max_density_buf: array<u32>;
@group(0) @binding(6) var<storage, read> cdf: array<f32>;

// ACES filmic tonemapping curve (Krzysztof Narkowicz fit)
// These constants define the curve shape — they ARE the algorithm, not magic numbers.
fn aces_tonemap(x: f32) -> f32 {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

// AgX view transform (Troy Sobotka, Blender 4.0)
// Analytical polynomial approximation — gracefully desaturates to white at high exposure.
// These matrices and polynomial coefficients ARE the algorithm, not magic numbers.
const AGX_INSET: mat3x3<f32> = mat3x3<f32>(
    vec3(0.842479062253094,  0.0423282422610123, 0.0423756549057051),
    vec3(0.0784335999999992, 0.878468636469772,  0.0784336),
    vec3(0.0792237451477643, 0.0791661274605434, 0.879142973793104)
);
const AGX_OUTSET: mat3x3<f32> = mat3x3<f32>(
    vec3(1.19687900512017,   -0.0528968517574562, -0.0529716355144438),
    vec3(-0.0980208811401368, 1.15190312990417,   -0.0980434501171241),
    vec3(-0.0990297440797205, -0.0989611768448433, 1.15107367264116)
);
fn agx_view_transform(color: vec3<f32>) -> vec3<f32> {
    let agx = AGX_INSET * max(color, vec3(0.0));
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var log_col = clamp(
        vec3(log2(max(agx.x, 1e-10)), log2(max(agx.y, 1e-10)), log2(max(agx.z, 1e-10))),
        vec3(min_ev), vec3(max_ev)
    );
    log_col = (log_col - min_ev) / (max_ev - min_ev);
    let x2 = log_col * log_col;
    let x4 = x2 * x2;
    let mapped = 15.5 * x4 * x2
               - 40.14 * x4 * log_col
               + 31.96 * x4
               - 6.868 * x2 * log_col
               + 0.4298 * x2
               + 0.1191 * log_col
               - 0.00232;
    return AGX_OUTSET * mapped;
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
    let h = u32(u.resolution.y);
    let tex_uv = pos.xy / u.resolution;

    let trail        = u.globals.z;
    let flame_bright = u.globals.w;
    let vibrancy     = u.extra.y;
    let bloom_int    = u.extra.z;
    let gamma        = u.extra.w;
    let blur_max     = u.extra3.w;  // velocity_blur_max from config

    // ── Read accumulation buffer (7 channels: density, R, G, B, vx, vy, depth) ──
    let buf_idx = (px.y * w + px.x) * 7u;
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
            let si = (u32(sy) * w + u32(sx)) * 7u;
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
            let ri = (u32(ry) * w + u32(rx)) * 7u;
            let rd = accumulation[ri];
            blur_density += rd * weight;
            blur_r += accumulation[ri + 1u] * weight;
            blur_g += accumulation[ri + 2u] * weight;
            blur_b += accumulation[ri + 3u] * weight;
        }
    }

    // ── Depth of Field blur ──
    let dof_strength = u.extra4.w;
    if (dof_strength > 0.001) {
        let dof_focal = u.extra5.x;
        let raw_depth = accumulation[buf_idx + 6u];
        let avg_depth = select(0.0, raw_depth / max(density, 1.0), density > 0.0);

        // Focal distance: if dof_focal is 0, use default of 1.0
        let focal = select(dof_focal, 1.0, dof_focal < 0.001);

        // Circle of confusion proportional to distance from focal plane
        let coc = abs(avg_depth - focal) * dof_strength;
        let blur_radius = clamp(coc * 8.0, 0.0, 16.0);

        if (blur_radius > 0.5) {
            var dof_col = vec3(0.0);
            var dof_weight = 0.0;
            let dof_texel = 1.0 / u.resolution;
            for (var di = 0; di < 8; di++) {
                let angle = f32(di) * 0.785398;  // TAU/8
                let d_offset = vec2(cos(angle), sin(angle)) * blur_radius * dof_texel;
                let sample_uv = tex_uv + d_offset;
                let sample_px = vec2<u32>(
                    u32(clamp(sample_uv.x * f32(w), 0.0, f32(w) - 1.0)),
                    u32(clamp(sample_uv.y * f32(h), 0.0, f32(h) - 1.0))
                );
                let si_idx = (sample_px.y * w + sample_px.x) * 7u;
                let sd = accumulation[si_idx];
                if (sd > 0.0) {
                    let s_col = vec3(
                        accumulation[si_idx + 1u],
                        accumulation[si_idx + 2u],
                        accumulation[si_idx + 3u]
                    ) / sd;
                    dof_col += s_col;
                    dof_weight += 1.0;
                }
            }
            if (dof_weight > 0.0) {
                let dof_blend = clamp(coc * 2.0, 0.0, 1.0);
                blur_r = blur_r * (1.0 - dof_blend) + (dof_col.x / dof_weight) * blur_density * dof_blend;
                blur_g = blur_g * (1.0 - dof_blend) + (dof_col.y / dof_weight) * blur_density * dof_blend;
                blur_b = blur_b * (1.0 - dof_blend) + (dof_col.z / dof_weight) * blur_density * dof_blend;
            }
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
    // Select tonemapping curve
    let tonemap_mode = u32(u.extra4.y);
    var alpha: f32;
    if (tonemap_mode == 1u) {
        // ACES filmic — better highlight rolloff, deeper blacks
        let normalized = log_density / max(max_log, 0.001);
        alpha = aces_tonemap(normalized);
    } else {
        // Default sqrt-log curve with per-image normalization
        alpha = sqrt(log_density / max(max_log, 0.001));
    }

    // ── Adaptive histogram equalization ──
    let hist_eq = u.extra4.z;
    if (hist_eq > 0.001) {
        let bin = u32(clamp(alpha * 255.0, 0.0, 255.0));
        let equalized = cdf[bin];
        alpha = mix(alpha, equalized, hist_eq);
    }

    // Velocity blur can sum neighboring densities above max_density,
    // pushing alpha > 1.0 and causing bright white flashes.
    alpha = clamp(alpha, 0.0, 1.0);

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
    let li = (u32(px.y) * w + u32(lx)) * 7u;
    let ri = (u32(px.y) * w + u32(rx2)) * 7u;
    let ti = (u32(ty) * w + px.x) * 7u;
    let bi = (u32(by) * w + px.x) * 7u;
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

    // ── Feedback trail with temporal reprojection ──
    let temporal_reproj = u.extra5.z;
    let prev_zoom = u.extra5.w;
    let cur_zoom = u.globals.y;

    var prev_uv = tex_uv;
    if (temporal_reproj > 0.001 && prev_zoom > 0.001) {
        // Warp previous frame by zoom ratio to reduce ghosting during zoom transitions
        // Derivation: screen = (world / zoom + 0.5) * resolution
        // So for same world point: prev_uv = (uv - 0.5) * cur_zoom / prev_zoom + 0.5
        let centered = tex_uv - vec2(0.5);
        let zoom_ratio = cur_zoom / prev_zoom;
        let reprojected = centered * zoom_ratio + vec2(0.5);
        prev_uv = mix(tex_uv, reprojected, temporal_reproj);
    }

    prev_uv = clamp(prev_uv, vec2(0.001), vec2(0.999));
    let prev = textureSample(prev_frame, prev_sampler, prev_uv).rgb;
    col = max(col, prev * trail);

    // ── Accumulation-sourced bloom — reads clean density data, no feedback loop ──
    // Samples neighbor pixels from the accumulation buffer (not prev_frame) to avoid
    // exponential bloom accumulation. Converts raw density+color to approximate luminance
    // and adds a soft glow halo around bright regions.
    if (bloom_int > 0.001) {
        let max_d = max(max_density_val, 0.001);
        let norm_density = clamp(density / max_d, 0.0, 1.0);
        let sparsity = 1.0 - sqrt(norm_density);

        var bloom_sum = vec3(0.0);
        let radii = array<i32, 3>(2, 5, 12);
        let rweights = array<f32, 3>(0.5, 0.3, 0.2);

        for (var ri = 0u; ri < 3u; ri++) {
            let r = radii[ri];
            let rw = rweights[ri];
            // 4 cardinal taps per radius
            var ring = vec3(0.0);
            let offsets = array<vec2<i32>, 4>(
                vec2(-r, 0), vec2(r, 0), vec2(0, -r), vec2(0, r)
            );
            for (var d = 0u; d < 4u; d++) {
                let sp = clamp(vec2<i32>(px) + offsets[d], vec2(0i), vec2<i32>(vec2(u.resolution)) - 1);
                let si = (u32(sp.y) * w + u32(sp.x)) * 7u;
                let sd = accumulation[si];
                if (sd > 0.0) {
                    let sc = vec3(accumulation[si + 1u], accumulation[si + 2u], accumulation[si + 3u]) / sd;
                    // Same tonemapping as the main pixel: log density → sqrt normalized
                    let neighbor_log = log(1.0 + sd / 1000.0 * flame_bright);
                    let neighbor_alpha = sqrt(neighbor_log / max(max_log, 0.001));
                    ring += sc / 255.0 * neighbor_alpha;
                }
            }
            bloom_sum += ring * rw * 0.25;
        }
        col += bloom_sum * bloom_int * sparsity;
    }

    // ── Final display transform ──
    let tonemap_final = u32(u.extra4.y);
    if (tonemap_final == 2u) {
        // AgX: graceful HDR→display compression, desaturates highlights to white
        col = agx_view_transform(col);
    }
    // Shoulder compression — linear below knee, soft rolloff above.
    // Preserves punchy midtones while taming hot highlights.
    let knee = 0.8;
    col = vec3(
        select(col.x, knee + (1.0 - knee) * (col.x - knee) / (col.x - knee + (1.0 - knee)), col.x > knee),
        select(col.y, knee + (1.0 - knee) * (col.y - knee) / (col.y - knee + (1.0 - knee)), col.y > knee),
        select(col.z, knee + (1.0 - knee) * (col.z - knee) / (col.z - knee + (1.0 - knee)), col.z > knee)
    );
    col = clamp(col, vec3(0.0), vec3(1.0));

    return vec4(col, 1.0);
}
