#include <metal_stdlib>
using namespace metal;

// Must match UnprojectParams in PointCloudGenerator.swift.
struct UnprojectParams {
    float fx, fy, cx, cy;
    float minDepth, maxDepth;
};

// One thread per depth pixel -> camera-space point (x right, y down, z forward, meters). w == 0 marks invalid.
kernel void unprojectDepth(texture2d<float, access::read> depth [[texture(0)]],
                           constant UnprojectParams &p         [[buffer(0)]],
                           device float4 *points               [[buffer(1)]],
                           uint2 gid                           [[thread_position_in_grid]])
{
    uint w = depth.get_width();
    uint h = depth.get_height();
    if (gid.x >= w || gid.y >= h) return;

    uint idx = gid.y * w + gid.x;
    float z = depth.read(gid).r;
    // Bit test because MTL_FAST_MATH lets the compiler assume NaN never occurs, but TrueDepth holes are NaN.
    bool finite = (as_type<uint>(z) & 0x7f800000u) != 0x7f800000u;
    if (!(finite && z > p.minDepth && z < p.maxDepth)) {
        points[idx] = float4(0.0);
        return;
    }
    points[idx] = float4((float(gid.x) - p.cx) * z / p.fx,
                         (float(gid.y) - p.cy) * z / p.fy,
                         z,
                         1.0);
}
