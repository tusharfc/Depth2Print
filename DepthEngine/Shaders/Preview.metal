#include <metal_stdlib>
using namespace metal;

// Shaded render of the fused model for the live preview. BGRA, premultiplied, transparent where there is no surface
// or below `hide` (camera-space plane n·p + d < 0; n == 0 disables), so the table doesn't swamp the object.
kernel void shadePreview(device const float4 *normals  [[buffer(0)]],
                         device const float4 *vertices [[buffer(1)]],
                         device uchar4 *pixels         [[buffer(2)]],
                         constant uint2 &size          [[buffer(3)]],
                         constant float4 &hide         [[buffer(4)]],
                         uint2 gid                     [[thread_position_in_grid]])
{
    if (gid.x >= size.x || gid.y >= size.y) return;
    uint i = gid.y * size.x + gid.x;
    float4 n = normals[i];
    bool hidden = any(hide.xyz != 0.0) && dot(hide.xyz, vertices[i].xyz) + hide.w < 0.0;
    if (n.w <= 0.0 || hidden) { pixels[i] = uchar4(0); return; }
    // Headlight shading: normals face the camera (-z), so -n.z is the cosine to the viewer.
    float shade = 0.3 + 0.7 * saturate(-n.z);
    float alpha = 0.85;
    float3 rgb = float3(0.35, 0.85, 1.0) * shade * alpha;
    pixels[i] = uchar4(uchar(rgb.b * 255.0), uchar(rgb.g * 255.0), uchar(rgb.r * 255.0), uchar(alpha * 255.0));
}
