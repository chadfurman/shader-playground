// histogram_cdf.wgsl — Compute density histogram + CDF for adaptive equalization
//
// Pass 1 (entry: "bin_densities"): Read accumulation buffer, bin log-densities into 256 bins
// Pass 2 (entry: "prefix_sum"): Compute prefix sum to build CDF, then normalize

struct HistogramParams {
    resolution: vec2<f32>,
    flame_brightness: f32,
    total_pixels: f32,
}

@group(0) @binding(0) var<storage, read> accumulation: array<f32>;
@group(0) @binding(1) var<storage, read_write> hist_bins: array<atomic<u32>>;  // 256 bins
@group(0) @binding(2) var<storage, read_write> cdf: array<f32>;               // 256 floats (normalized CDF)
@group(0) @binding(3) var<uniform> params: HistogramParams;

@compute @workgroup_size(16, 16)
fn bin_densities(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(params.resolution.x);
    let h = u32(params.resolution.y);
    if (gid.x >= w || gid.y >= h) {
        return;
    }

    let px = gid.y * w + gid.x;
    let accum_idx = px * 7u;
    let density = accumulation[accum_idx] / 1000.0;  // convert from fixed-point

    if (density < 0.001) {
        return;  // skip empty pixels
    }

    // Map log-density to bin index [0, 255]
    let log_d = log(1.0 + density * params.flame_brightness);
    let bin = u32(clamp(log_d / (log_d + 4.0) * 255.0, 0.0, 255.0));
    atomicAdd(&hist_bins[bin], 1u);
}

// Prefix sum pass — single workgroup, 256 threads
@compute @workgroup_size(256)
fn prefix_sum(@builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = lid.x;
    let count = f32(atomicLoad(&hist_bins[idx]));

    // Store count temporarily in cdf
    cdf[idx] = count;
    workgroupBarrier();

    // Hillis-Steele prefix sum (inclusive)
    for (var stride = 1u; stride < 256u; stride = stride * 2u) {
        var val = cdf[idx];
        if (idx >= stride) {
            val += cdf[idx - stride];
        }
        workgroupBarrier();
        cdf[idx] = val;
        workgroupBarrier();
    }

    // Normalize to [0, 1]
    let total = cdf[255u];
    if (total > 0.0) {
        cdf[idx] = cdf[idx] / total;
    }
}
