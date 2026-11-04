#include <metal_stdlib>
using namespace metal;

// Must match VolumeParams in Fusion/TSDFVolume.swift.
struct VolumeParams {
    float4x4 worldToCamera;
    float4x4 cameraToWorld;
    float4 origin;              // xyz: world position of the volume's min corner
    float fx, fy, cx, cy;       // depth-map intrinsics
    uint resolution;            // voxels per side
    uint width, height;         // depth-map size
    float voxelSize;
    float truncation;           // meters
    float maxWeight;
    float minDepth, maxDepth;
};

// Must match MarchingCubesParams in Fusion/MarchingCubes.swift.
struct MarchingCubesParams {
    float4 origin;
    float4 plane;           // world (n, d): everything with n·p + d < 0 is outside; n == 0 disables
    float4 back;            // a second such plane: the flat back of a one-sided scan
    float4 region;          // world sphere (center, radius): everything outside it is outside; radius <= 0 disables
    uint resolution;
    float voxelSize;
    uint tableStride;
    uint maxTriangles;
};

static inline uint voxelIndex(uint3 q, uint n) {
    return (q.z * n + q.y) * n + q.x;
}

// Voxels store (tsdf / truncation in [-1, 1], weight). Weight 0 means never directly observed; then the tsdf
// slot counts inside/outside votes: -1 each time the voxel is hidden behind a surface, +1 each time a view
// shows it as empty or doesn't contain it. Negative means unseen interior or underside.
kernel void resetVolume(device half2 *volume [[buffer(0)]],
                        constant uint &count [[buffer(1)]],
                        uint gid             [[thread_position_in_grid]])
{
    if (gid < count) volume[gid] = half2(0.0h, 0.0h);
}

// Half floats count integers exactly up to 2048; votes must not saturate or the latest views would win.
constant float kVote = 1.0;
constant float kMaxVotes = 2000.0;
// Least weight a surface measurement gets (edge-on or at a depth jump).
constant float kMinWeight = 0.05;

static inline void vote(device half2 *volume, uint idx, float2 old, float delta) {
    if (old.y == 0.0) volume[idx] = half2(half(clamp(old.x + delta, -kMaxVotes, kMaxVotes)), 0.0h);
}

// Projective TSDF fusion: each voxel compares its camera depth with the measured depth at its pixel.
// Masked pixels (negative depth, see bilateralDepth) still carve the free space in front of them.
kernel void integrateVolume(device half2 *volume        [[buffer(0)]],
                            device const float *depth   [[buffer(1)]],
                            constant VolumeParams &p    [[buffer(2)]],
                            device const float4 *normals [[buffer(3)]],
                            uint3 gid                   [[thread_position_in_grid]])
{
    uint n = p.resolution;
    if (any(gid >= n)) return;

    float3 pw = p.origin.xyz + (float3(gid) + 0.5) * p.voxelSize;
    float3 pc = (p.worldToCamera * float4(pw, 1.0)).xyz;
    uint idx = voxelIndex(gid, n);
    int2 px = int2(round(float2(p.fx * pc.x / pc.z + p.cx, p.fy * pc.y / pc.z + p.cy)));
    // A view that doesn't contain the voxel votes "outside": only space hidden from most viewpoints gets filled,
    // not space that merely sat outside the frame whenever it wasn't behind the subject.
    if (pc.z <= 0.0 || px.x < 0 || px.y < 0 || px.x >= int(p.width) || px.y >= int(p.height)) {
        vote(volume, idx, float2(volume[idx]), kVote);
        return;
    }

    float d = depth[uint(px.y) * p.width + uint(px.x)];
    if (d == 0.0) return;
    float2 old = float2(volume[idx]);
    // Only a vote, never an observation: random sensor holes on an object must not tunnel through it.
    if (d <= -999.0) { vote(volume, idx, old, kVote); return; }

    float tsdf;
    float weight = 1.0;
    if (d < 0.0) {
        float surface = -d;
        if (pc.z < surface - p.truncation) {
            tsdf = 1.0;
        } else {
            // Beyond-range pixels say nothing about what lies past maxDepth.
            if (pc.z > surface + p.truncation && surface < p.maxDepth - 1e-4) vote(volume, idx, old, -kVote);
            return;
        }
    } else {
        float sdf = d - pc.z;
        if (sdf < -p.truncation) { vote(volume, idx, old, -kVote); return; }
        tsdf = min(1.0, sdf / p.truncation);
        // Grazing views are noisy and their projective distance is stretched; pixels at depth jumps (no normal)
        // are often flying pixels between foreground and background. Both may only refine, not override.
        float4 nrm = normals[uint(px.y) * p.width + uint(px.x)];
        float facing = nrm.w > 0.0 ? -dot(nrm.xyz, normalize(pc)) : 0.0;
        weight = max(facing * facing, kMinWeight);
    }
    volume[idx] = half2(half((old.x * old.y + tsdf * weight) / (old.y + weight)), half(min(old.y + weight, p.maxWeight)));
}

// Trilinear TSDF lookup; invalid if any of the 8 neighbours is unobserved or outside the volume.
static float sampleVolume(device const half2 *volume, constant VolumeParams &p, float3 pw, thread bool &valid)
{
    int n = int(p.resolution);
    float3 g = (pw - p.origin.xyz) / p.voxelSize - 0.5;
    int3 i0 = int3(floor(g));
    valid = false;
    if (any(i0 < 0) || any(i0 + 1 >= n)) return 1.0;
    float3 f = g - float3(i0);
    float v[8];
    for (int c = 0; c < 8; c++) {
        uint3 q = uint3(i0 + int3(c & 1, (c >> 1) & 1, (c >> 2) & 1));
        float2 s = float2(volume[voxelIndex(q, uint(n))]);
        if (s.y <= 0.0) return 1.0;
        v[c] = s.x;
    }
    valid = true;
    float y0 = mix(mix(v[0], v[1], f.x), mix(v[2], v[3], f.x), f.y);
    float y1 = mix(mix(v[4], v[5], f.x), mix(v[6], v[7], f.x), f.y);
    return mix(y0, y1, f.z);
}

// Renders the fused model's depth from a camera pose, so ICP can align new frames to the model instead of
// the previous frame. Output is 0 where the ray hits no surface.
kernel void raycastVolume(device const half2 *volume  [[buffer(0)]],
                          constant VolumeParams &p    [[buffer(1)]],
                          device float *depthOut      [[buffer(2)]],
                          uint2 gid                   [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;
    uint idx = gid.y * p.width + gid.x;
    depthOut[idx] = 0.0;

    float3 rayCam = float3((float(gid.x) - p.cx) / p.fx, (float(gid.y) - p.cy) / p.fy, 1.0);
    float rayScale = length(rayCam);   // meters travelled per unit of camera depth
    float3 eye = p.cameraToWorld[3].xyz;
    float3 dir = (p.cameraToWorld * float4(rayCam, 0.0)).xyz;

    // Fast math assumes no infinities, so keep 1/dir finite.
    float3 safeDir = select(dir, float3(1e-9), fabs(dir) < 1e-9);
    float3 lo = p.origin.xyz, hi = lo + float(p.resolution) * p.voxelSize;
    float3 ta = (lo - eye) / safeDir, tb = (hi - eye) / safeDir;
    float3 tMin = min(ta, tb), tMax = max(ta, tb);
    float tNear = max(max(max(tMin.x, tMin.y), tMin.z), p.minDepth);
    float tFar = min(min(min(tMax.x, tMax.y), tMax.z), p.maxDepth);

    float t = tNear, prevT = tNear, prevF = 0.0;
    bool prevValid = false;
    for (uint i = 0; i < 2048 && t < tFar; i++) {
        bool valid;
        float f = sampleVolume(volume, p, eye + dir * t, valid);
        if (!valid) {
            prevValid = false;
            t += p.truncation / rayScale;
            continue;
        }
        if (prevValid && prevF > 0.0 && f < 0.0) {
            depthOut[idx] = prevT + (t - prevT) * prevF / (prevF - f);
            return;
        }
        prevValid = true;
        prevF = f;
        prevT = t;
        // The TSDF bounds the distance to the surface; 0.8 guards against projective overestimates.
        t += max(f * p.truncation * 0.8, p.voxelSize * 0.5) / rayScale;
    }
}

// Value used for meshing. The mesh is made watertight by treating as outside: the volume's border, the far side
// of the cut plane, and anything outside the region; unobserved voxels count as inside only if they were
// hidden behind a surface (the unseen interior / underside).
static float closedValue(device const half2 *volume, constant MarchingCubesParams &p, uint3 q)
{
    uint n = p.resolution;
    if (any(q == 0u) || any(q == n - 1)) return 1.0;
    float3 pw = p.origin.xyz + (float3(q) + 0.5) * p.voxelSize;
    if (any(p.plane.xyz != 0.0) && dot(p.plane.xyz, pw) + p.plane.w < 0.0) return 1.0;
    if (any(p.back.xyz != 0.0) && dot(p.back.xyz, pw) + p.back.w < 0.0) return 1.0;
    if (p.region.w > 0.0 && distance(pw, p.region.xyz) > p.region.w) return 1.0;
    float2 s = float2(volume[voxelIndex(q, n)]);
    if (s.y > 0.0) return s.x;
    return s.x < 0.0 ? -1.0 : 1.0;
}

// Corner k of a cube sits at offset (k & 1, (k >> 1) & 1, (k >> 2) & 1). Bit k is set when corner k is inside.
static uint cubeCase(device const half2 *volume, constant MarchingCubesParams &p, uint3 c, thread float *v)
{
    uint m = 0;
    for (uint k = 0; k < 8; k++) {
        v[k] = closedValue(volume, p, c + uint3(k & 1, (k >> 1) & 1, (k >> 2) & 1));
        if (v[k] < 0.0) m |= 1u << k;
    }
    return m;
}

kernel void marchingCubesCount(device const half2 *volume        [[buffer(0)]],
                               constant MarchingCubesParams &p   [[buffer(1)]],
                               device const uchar *triCounts     [[buffer(2)]],
                               device atomic_uint *total         [[buffer(3)]],
                               uint3 gid                         [[thread_position_in_grid]])
{
    if (any(gid + 1 >= p.resolution)) return;
    float v[8];
    uint count = triCounts[cubeCase(volume, p, gid, v)];
    if (count > 0) atomic_fetch_add_explicit(total, count, memory_order_relaxed);
}

// Emits a triangle soup plus, per vertex, a key for its grid edge so the CPU can weld shared vertices.
kernel void marchingCubesEmit(device const half2 *volume        [[buffer(0)]],
                              constant MarchingCubesParams &p   [[buffer(1)]],
                              device const uchar *triCounts     [[buffer(2)]],
                              device atomic_uint *cursor        [[buffer(3)]],
                              device const uchar *triEdges      [[buffer(4)]],
                              device const uchar2 *edgeCorners  [[buffer(5)]],
                              device float4 *positions          [[buffer(6)]],
                              device uint *edgeKeys             [[buffer(7)]],
                              uint3 gid                         [[thread_position_in_grid]])
{
    uint n = p.resolution;
    if (any(gid + 1 >= n)) return;
    float v[8];
    uint m = cubeCase(volume, p, gid, v);
    uint count = triCounts[m];
    if (count == 0) return;
    uint first = atomic_fetch_add_explicit(cursor, count, memory_order_relaxed);
    if (first + count > p.maxTriangles) return;

    for (uint i = 0; i < count * 3; i++) {
        uint e = triEdges[m * p.tableStride + i];
        uchar2 ends = edgeCorners[e];
        uint3 a = uint3(ends.x & 1, (ends.x >> 1) & 1, (ends.x >> 2) & 1);
        uint3 b = uint3(ends.y & 1, (ends.y >> 1) & 1, (ends.y >> 2) & 1);
        float t = v[ends.x] / (v[ends.x] - v[ends.y]);
        float3 g = float3(gid + a) + (float3(b) - float3(a)) * t;
        positions[first * 3 + i] = float4(p.origin.xyz + (g + 0.5) * p.voxelSize, 1.0);
        // Edge e runs along axis e / 4 from corner `a`, so (corner voxel, axis) names it uniquely in the grid.
        edgeKeys[first * 3 + i] = voxelIndex(gid + a, n) * 3 + e / 4;
    }
}
