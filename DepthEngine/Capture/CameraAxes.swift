import CoreGraphics
import simd

public enum CameraAxes {
    /// The depth camera's axes (image right, image down, forward) in CoreMotion device axes, worked out from where a
    /// portrait UI's preview layer draws the depth buffer's +x and +y directions. Taken from the live preview rather
    /// than hardcoded because sensor mounting is a hardware detail; the preview's selfie mirroring is undone by
    /// requiring a right-handed frame whose forward axis leaves the screen (front camera) or the back (rear camera).
    public static func deviceFromCamera(bufferXOnScreen: CGVector, bufferYOnScreen: CGVector,
                                        frontCamera: Bool = true) -> simd_float3x3 {
        // Portrait UIKit points: screen x is device x, screen y (down) is device -y.
        func device(_ v: CGVector, mirrored: Bool) -> SIMD3<Float> {
            normalize(SIMD3(Float(mirrored ? -v.dx : v.dx), Float(-v.dy), 0))
        }
        let forward = SIMD3<Float>(0, 0, frontCamera ? 1 : -1)
        var x = device(bufferXOnScreen, mirrored: false)
        var y = device(bufferYOnScreen, mirrored: false)
        if dot(cross(x, y), forward) < 0 {
            x = device(bufferXOnScreen, mirrored: true)
            y = device(bufferYOnScreen, mirrored: true)
        }
        return simd_float3x3(x, y, cross(x, y))
    }
}
