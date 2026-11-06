import simd

/// Where the head is in one depth frame, from a face tracker (ARKit). Head coordinates follow ARFaceAnchor: origin
/// inside the head, +y toward the crown, +z out of the face, meters.
public struct HeadPose: Sendable {
    /// Head coordinates → this frame's depth-camera coordinates (x image right, y image down, z forward).
    public var transform: simd_float4x4
    /// In head coordinates: the bottom of the chin and the tip of the nose (from the fitted face mesh).
    public var chin: SIMD3<Float>
    public var noseTip: SIMD3<Float>

    public init(transform: simd_float4x4, chin: SIMD3<Float>, noseTip: SIMD3<Float>) {
        self.transform = transform
        self.chin = chin
        self.noseTip = noseTip
    }

    /// The skull's middle: ~11.5 cm above the chin, ~10 cm behind the nose tip.
    var center: SIMD3<Float> { SIMD3(0, chin.y + 0.115, noseTip.z - 0.10) }
}
