#include <metal_stdlib>
using namespace metal;

// Layouts below must match the Swift structs in Registration/ICP.swift.
struct Intrinsics {
    float fx, fy, cx, cy;
};

struct DepthFilterParams {
    Intrinsics k;
    float4 plane;       // camera space (n, d): points with n·p + d < planeMargin are masked; n == 0 disables
    float4 region;      // camera space sphere (center, radius): points outside are masked; radius <= 0 disables
    float minDepth, maxDepth;
    float sigmaSpace;   // pixels
    float sigmaDepth;   // meters
    float planeMargin;
};

struct DownsampleParams {
    uint srcWidth, srcHeight;
    float maxJump;
};

struct MapParams {
    Intrinsics k;
    uint width, height;
    float maxJump;
};

struct ICPParams {
    float4x4 transform;   // source camera -> reference camera
    Intrinsics k;         // reference camera, at this pyramid level
    uint width, height;
    float maxDistance;
    float minNormalDot;
};

static inline float3 unproject(uint2 px, float z, Intrinsics k) {
    return float3((float(px.x) - k.cx) * z / k.fx, (float(px.y) - k.cy) * z / k.fy, z);
}

// Bit test because MTL_FAST_MATH lets the compiler assume NaN never occurs, but TrueDepth holes are NaN.
static inline bool inRange(float z, float lo, float hi) {
    return (as_type<uint>(z) & 0x7f800000u) != 0x7f800000u && z > lo && z < hi;
}

// 5x5 bilateral filter so normals aren't dominated by sensor noise. Output encoding, which TSDF fusion relies on:
//   z > 0      valid depth
//   z == -1000 no return from the sensor (nothing in range, or a hole)
//   z < 0      real surface at -z that is masked out (or beyond maxDepth): the ray is known free up to -z
//   z == 0     unknown (too near)
kernel void bilateralDepth(texture2d<float, access::read> depth [[texture(0)]],
                           constant DepthFilterParams &p       [[buffer(0)]],
                           device float *out                   [[buffer(1)]],
                           device float *raw                   [[buffer(2)]],
                           uint2 gid                           [[thread_position_in_grid]])
{
    uint w = depth.get_width(), h = depth.get_height();
    if (gid.x >= w || gid.y >= h) return;
    uint idx = gid.y * w + gid.x;

    float zc = depth.read(gid).r;
    if (!inRange(zc, 0.0, 1e6)) { out[idx] = raw[idx] = -1000.0; return; }
    if (zc <= p.minDepth) { out[idx] = raw[idx] = 0.0; return; }
    if (zc >= p.maxDepth) { out[idx] = raw[idx] = -p.maxDepth; return; }
    float3 pc = unproject(gid, zc, p.k);
    bool belowPlane = any(p.plane.xyz != 0.0) && dot(p.plane.xyz, pc) + p.plane.w < p.planeMargin;
    bool outsideRegion = p.region.w > 0.0 && distance(pc, p.region.xyz) > p.region.w;
    if (belowPlane || outsideRegion) { out[idx] = raw[idx] = -zc; return; }
    // Fusion averages many frames itself; smoothing here would only blur fine detail like a nose's edges.
    raw[idx] = zc;

    float invSpace = 0.5 / (p.sigmaSpace * p.sigmaSpace);
    float invDepth = 0.5 / (p.sigmaDepth * p.sigmaDepth);
    float sum = 0.0, weightSum = 0.0;
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int2 q = int2(gid) + int2(dx, dy);
            if (q.x < 0 || q.y < 0 || q.x >= int(w) || q.y >= int(h)) continue;
            float z = depth.read(uint2(q)).r;
            if (!inRange(z, p.minDepth, p.maxDepth)) continue;
            float dz = z - zc;
            float wt = exp(-float(dx * dx + dy * dy) * invSpace - dz * dz * invDepth);
            sum += wt * z;
            weightSum += wt;
        }
    }
    out[idx] = sum / weightSum;
}

// Halves resolution, averaging only samples on the same surface so silhouettes stay sharp.
kernel void downsampleDepth(device const float *src         [[buffer(0)]],
                            constant DownsampleParams &p    [[buffer(1)]],
                            device float *dst               [[buffer(2)]],
                            uint2 gid                       [[thread_position_in_grid]])
{
    uint dw = p.srcWidth / 2, dh = p.srcHeight / 2;
    if (gid.x >= dw || gid.y >= dh) return;

    float ref = 0.0, sum = 0.0, n = 0.0;
    for (uint dy = 0; dy < 2; dy++) {
        for (uint dx = 0; dx < 2; dx++) {
            float z = src[(gid.y * 2 + dy) * p.srcWidth + gid.x * 2 + dx];
            if (z <= 0.0) continue;
            if (ref == 0.0) ref = z;
            if (fabs(z - ref) < p.maxJump) { sum += z; n += 1.0; }
        }
    }
    dst[gid.y * dw + gid.x] = n > 0.0 ? sum / n : 0.0;
}

// Camera-space vertex and normal per pixel. w == 0 marks invalid; normals face the camera.
kernel void vertexNormalMap(device const float *depth   [[buffer(0)]],
                            constant MapParams &p       [[buffer(1)]],
                            device float4 *vertices     [[buffer(2)]],
                            device float4 *normals      [[buffer(3)]],
                            uint2 gid                   [[thread_position_in_grid]])
{
    uint w = p.width, h = p.height;
    if (gid.x >= w || gid.y >= h) return;
    uint idx = gid.y * w + gid.x;

    float z = depth[idx];
    if (z <= 0.0) { vertices[idx] = float4(0.0); normals[idx] = float4(0.0); return; }
    float3 v = unproject(gid, z, p.k);
    vertices[idx] = float4(v, 1.0);

    normals[idx] = float4(0.0);
    if (gid.x == 0 || gid.y == 0 || gid.x + 1 >= w || gid.y + 1 >= h) return;
    float zl = depth[idx - 1], zr = depth[idx + 1], zu = depth[idx - w], zd = depth[idx + w];
    if (zl <= 0.0 || zr <= 0.0 || zu <= 0.0 || zd <= 0.0) return;
    if (fabs(zr - zl) > p.maxJump || fabs(zd - zu) > p.maxJump) return;

    float3 dx = unproject(gid + uint2(1, 0), zr, p.k) - unproject(gid - uint2(1, 0), zl, p.k);
    float3 dy = unproject(gid + uint2(0, 1), zd, p.k) - unproject(gid - uint2(0, 1), zu, p.k);
    float3 n = cross(dy, dx);
    float len = length(n);
    if (len < 1e-12) return;
    n /= len;
    if (dot(n, v) > 0.0) n = -n;
    normals[idx] = float4(n, 1.0);
}

constant constexpr uint kICPThreads = 128;   // threadgroup is 16x8
constant constexpr uint kICPTerms = 29;      // 21 upper JᵀJ + 6 Jᵀr + r² + inlier count

// Point-to-plane ICP with projective association. Each thread handles a 2x2 pixel block (accumulating in registers,
// which keeps the expensive threadgroup reduction to a quarter), then each threadgroup writes its partial sums of
// the normal equations; icpSumPartials adds those up.
kernel void icpReduce(device const float4 *srcVertices [[buffer(0)]],
                      device const float4 *srcNormals  [[buffer(1)]],
                      device const float4 *refVertices [[buffer(2)]],
                      device const float4 *refNormals  [[buffer(3)]],
                      constant ICPParams &p            [[buffer(4)]],
                      device float *partials           [[buffer(5)]],
                      uint2 gid                        [[thread_position_in_grid]],
                      uint tid                         [[thread_index_in_threadgroup]],
                      uint2 group                      [[threadgroup_position_in_grid]],
                      uint2 groupCount                 [[threadgroups_per_grid]])
{
    threadgroup float shared[kICPTerms][kICPThreads];
    float acc[kICPTerms];
    for (uint k = 0; k < kICPTerms; k++) acc[k] = 0.0;

    // No early return: every thread must reach the barriers below.
    for (uint b = 0; b < 4; b++) {
        uint2 px = gid * 2 + uint2(b & 1, b >> 1);
        if (px.x >= p.width || px.y >= p.height) continue;
        uint i = px.y * p.width + px.x;
        float4 ns = srcNormals[i];
        float3 ps = (p.transform * float4(srcVertices[i].xyz, 1.0)).xyz;
        if (ns.w <= 0.0 || ps.z <= 0.0) continue;
        int2 q = int2(round(float2(p.k.fx * ps.x / ps.z + p.k.cx, p.k.fy * ps.y / ps.z + p.k.cy)));
        if (q.x < 0 || q.y < 0 || q.x >= int(p.width) || q.y >= int(p.height)) continue;
        uint j = uint(q.y) * p.width + uint(q.x);
        float4 nr = refNormals[j];
        float3 qr = refVertices[j].xyz;
        float3 nsr = (p.transform * float4(ns.xyz, 0.0)).xyz;
        if (nr.w <= 0.0 || distance(ps, qr) >= p.maxDistance || dot(nsr, nr.xyz) <= p.minNormalDot) continue;

        // J = [p × n, n], r = n · (p - q)
        float3 c = cross(ps, nr.xyz);
        float row[7] = { c.x, c.y, c.z, nr.x, nr.y, nr.z, dot(nr.xyz, ps - qr) };
        uint t = 0;
        for (uint a = 0; a < 6; a++)
            for (uint b2 = a; b2 < 6; b2++)
                acc[t++] += row[a] * row[b2];
        for (uint a = 0; a < 6; a++)
            acc[t++] += row[a] * row[6];
        acc[t++] += row[6] * row[6];
        acc[t] += 1.0;
    }

    for (uint k = 0; k < kICPTerms; k++) shared[k][tid] = acc[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kICPThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            for (uint k = 0; k < kICPTerms; k++)
                shared[k][tid] += shared[k][tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < kICPTerms)
        partials[(group.y * groupCount.x + group.x) * kICPTerms + tid] = shared[tid][0];
}

// Adds up icpReduce's per-threadgroup partials: threadgroup t sums term t over every group (tree reduction, so
// float rounding stays small), leaving the CPU 29 numbers to read per iteration.
kernel void icpSumPartials(device const float *partials [[buffer(0)]],
                           constant uint &groupCount    [[buffer(1)]],
                           device float *totals         [[buffer(2)]],
                           uint term                    [[threadgroup_position_in_grid]],
                           uint tid                     [[thread_index_in_threadgroup]])
{
    threadgroup float shared[256];
    float sum = 0.0;
    for (uint g = tid; g < groupCount; g += 256) sum += partials[g * kICPTerms + term];
    shared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) totals[term] = shared[0];
}
