/// How the scan is captured. The reconstruction core is the same; modes differ in what gets masked out,
/// what motion is allowed, and how the mesh is closed for printing.
public enum ScanMode: String, CaseIterable, Sendable {
    /// Phone on a tripod, object on a turntable.
    case turntable
    /// Phone moved slowly around an object standing on a table.
    case handheld
    /// Phone held still at arm's length while the person slowly turns their head (or circles a still head).
    case face
    /// Someone circles a still person. TrueDepth reaches ~1.5 m, so this suits a bust, not a full standing body.
    case body
}

struct ScanProfile {
    var icp = ICP.Config()
    var volume = TSDFVolume.Config()
    /// Detect the surface the object stands on; the mesh is cut flat there and oriented Z-up.
    var findsSupportPlane = false
    /// Also hide that surface from tracking. Only right when the camera is static (turntable).
    var masksSupportPlane = false
    var locksTurntableAxis = false
    /// Keep only the head: a sphere found from the crown down, plus a cut below the chin.
    var isolatesHead = false
    /// Cut flat (perpendicular to gravity) where the camera's view ended, so a bust stands on its base.
    var cutsBelowView = false

    init(_ mode: ScanMode) {
        switch mode {
        case .turntable:
            icp.depthRange = 0.15...0.7
            volume.size = 0.3
            findsSupportPlane = true
            masksSupportPlane = true
            locksTurntableAxis = true
        case .handheld:
            // Hand motion is jerkier and frames get dropped, so allow larger jumps between frames.
            icp.depthRange = 0.1...0.8
            icp.maxDistance = [0.02, 0.03, 0.04]
            icp.maxStepRotation = 0.2
            icp.maxStepTranslation = 0.05
            volume.size = 0.35
            findsSupportPlane = true
        case .face:
            // Selfie distance is often under 20 cm (the nose at 18 cm in a real capture); TrueDepth still measures there.
            icp.depthRange = 0.1...0.8
            icp.maxDistance = [0.015, 0.02, 0.03]
            icp.maxStepRotation = 0.2
            // Finer than the other modes (0.94 mm voxels, 128 MiB); 0.30 m still holds a whole head.
            volume.size = 0.30
            volume.resolution = 320
            isolatesHead = true
        case .body:
            icp.depthRange = 0.3...1.5
            icp.maxDistance = [0.03, 0.045, 0.06]
            icp.maxDepthJump = 0.04
            icp.maxStepRotation = 0.2
            icp.maxStepTranslation = 0.08
            volume.size = 1.0
            cutsBelowView = true
        }
    }
}
