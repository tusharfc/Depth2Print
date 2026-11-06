import ARKit
import AVFoundation
import CoreMotion
import simd

/// Face mode capture through ARKit face tracking: TrueDepth depth plus where the head is on every frame, which lets
/// the tracker follow quick turns in any posture (the same tracking Face ID enrollment relies on).
public final class FaceTrackingCaptureManager: NSObject, ARSessionDelegate, @unchecked Sendable {
    public static var isSupported: Bool { ARFaceTrackingConfiguration.isSupported }

    public weak var delegate: TrueDepthCaptureDelegate?
    public let session = ARSession()

    private let queue = DispatchQueue(label: "depth2print.facetracking", qos: .userInitiated)
    private let motion = CMMotionManager()
    // State below is only touched on `queue`.
    private var lastDepthTime: TimeInterval = -1
    /// Recent head poses: colour frames come at ~60 Hz, depth at ~15 Hz, so match each depth map to the pose
    /// measured closest to it.
    private var recentHeads: [(time: TimeInterval, pose: HeadPose)] = []

    public override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = queue
    }

    public func start() throws {
        guard Self.isSupported else { throw DepthEngineError.trueDepthUnavailable }
        let configuration = ARFaceTrackingConfiguration()
        configuration.isLightEstimationEnabled = false
        configuration.maximumNumberOfTrackedFaces = 1
        // Prefer the depth map's 4:3 shape, so colour and depth cover the same view, at the highest frame rate.
        let formats = ARFaceTrackingConfiguration.supportedVideoFormats
        if let format = formats.filter({ abs($0.imageResolution.width * 3 - $0.imageResolution.height * 4) < 1 })
            .max(by: { $0.framesPerSecond < $1.framesPerSecond }) {
            configuration.videoFormat = format
        }
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        if motion.isDeviceMotionAvailable && !motion.isDeviceMotionActive {
            motion.deviceMotionUpdateInterval = 1.0 / 60
            motion.startDeviceMotionUpdates()
        }
    }

    public func stop() {
        session.pause()
        motion.stopDeviceMotionUpdates()
        queue.async { [self] in
            recentHeads.removeAll()
            lastDepthTime = -1
        }
    }

    public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if let anchor = frame.anchors.lazy.compactMap({ $0 as? ARFaceAnchor }).first(where: \.isTracked) {
            recentHeads.append((frame.timestamp, Self.headPose(anchor, camera: frame.camera)))
            if recentHeads.count > 12 { recentHeads.removeFirst(recentHeads.count - 12) }
        }

        // ARKit repeats the latest depth map on the colour frames in between; only pass on new ones.
        guard let depthData = frame.capturedDepthData, frame.capturedDepthDataTimestamp > lastDepthTime,
              let calibration = depthData.cameraCalibrationData else { return }
        lastDepthTime = frame.capturedDepthDataTimestamp
        let float16 = depthData.depthDataType == kCVPixelFormatType_DepthFloat16
            ? depthData : depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat16)
        let depthMap = float16.depthDataMap

        // Within ~1.5 colour frames; otherwise the face wasn't tracked at that moment.
        let head = recentHeads.min { abs($0.time - lastDepthTime) < abs($1.time - lastDepthTime) }
            .flatMap { abs($0.time - lastDepthTime) < 0.025 ? $0.pose : nil }

        // Colour sampling assumes colour and depth show the same view, which holds when their shapes match.
        let image = frame.capturedImage
        let sameShape = CVPixelBufferGetWidth(image) * CVPixelBufferGetHeight(depthMap)
            == CVPixelBufferGetHeight(image) * CVPixelBufferGetWidth(depthMap)
        let gravity = motion.deviceMotion.map { SIMD3(Float($0.gravity.x), Float($0.gravity.y), Float($0.gravity.z)) }
        delegate?.captureDidOutput(DepthFrame(
            depth: depthMap,
            color: sameShape ? image : nil,
            intrinsics: TrueDepthCaptureManager.scaledIntrinsics(calibration, to: depthMap),
            calibration: calibration,
            timestamp: CMTime(seconds: lastDepthTime, preferredTimescale: 1_000_000),
            gravity: gravity,
            head: head
        ))
    }

    /// The anchor in this frame's depth-camera axes. ARKit's camera looks down -z with +y up (sensor orientation);
    /// the depth camera looks down +z with +y down the image, so y and z flip.
    private static func headPose(_ anchor: ARFaceAnchor, camera: ARCamera) -> HeadPose {
        let flip = simd_float4x4(diagonal: SIMD4(1, -1, -1, 1))
        let transform = flip * camera.transform.inverse * anchor.transform
        // Landmarks from the fitted face mesh, along the face's midline.
        var chin = SIMD3<Float>(0, -0.1, 0.03), nose = SIMD3<Float>(0, -0.01, 0.06)
        var lowest = Float.infinity, frontmost = -Float.infinity
        for v in anchor.geometry.vertices where abs(v.x) < 0.015 {
            if v.y < lowest { lowest = v.y; chin = v }
            if v.z > frontmost { frontmost = v.z; nose = v }
        }
        return HeadPose(transform: transform, chin: chin, noseTip: nose)
    }
}
