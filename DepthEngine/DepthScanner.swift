import ARKit
import AVFoundation
import simd

/// Public entry point: TrueDepth camera feed in, triangle mesh out. No UI dependencies.
public final class DepthScanner {
    /// Exposed so a host app can attach its own preview layer (every mode but face tracking).
    public var captureSession: AVCaptureSession { capture.session }
    /// Face mode's camera feed when `usesFaceTracking` (show it with an ARSCNView).
    public var faceSession: ARSession { faceCapture.session }
    /// Whether the running preview is ARKit face tracking rather than the plain TrueDepth session.
    public private(set) var usesFaceTracking = false

    private let capture = TrueDepthCaptureManager()
    private let faceCapture = FaceTrackingCaptureManager()
    private let frames = FrameRouter()

    /// Called on a background queue for each frame tracked during a scan.
    public var onTrackingUpdate: ((TrackingStatus) -> Void)? {
        get { frames.onTrackingUpdate }
        set { frames.onTrackingUpdate = newValue }
    }

    public init() {
        capture.delegate = frames
        faceCapture.delegate = frames
    }

    /// Starts the camera for `mode`: ARKit face tracking for faces where the device supports it (it follows the head
    /// through quick turns in any posture), the plain TrueDepth session otherwise. Only one can own the camera.
    public func startPreview(mode: ScanMode = .turntable) async throws {
        if mode == .face && FaceTrackingCaptureManager.isSupported {
            await capture.stopAndWait()
            try faceCapture.start()
            usesFaceTracking = true
        } else {
            faceCapture.stop()
            usesFaceTracking = false
            try await capture.start()
        }
    }

    public func stop() {
        capture.stop()
        faceCapture.stop()
    }

    /// `deviceFromCamera` (see `CameraAxes`) lets gravity orient the export and pick the support surface;
    /// without it, orientation falls back to geometry alone. With `recordingTo`, every raw depth frame is saved
    /// there for replay on a Mac (see `CaptureRecorder`); it's kept even if the scan fails.
    public func beginScan(mode: ScanMode, deviceFromCamera: simd_float3x3? = nil, recordingTo directory: URL? = nil) throws {
        let recorder = try directory.map { try CaptureRecorder(directory: $0, mode: mode, deviceFromCamera: deviceFromCamera) }
        frames.start(try FrameTracker(mode: mode, deviceFromCamera: deviceFromCamera), recorder: recorder)
        if !usesFaceTracking { capture.setImagingLocked(true) }
    }

    public func finishScan() async throws -> Mesh {
        if !usesFaceTracking { capture.setImagingLocked(false) }
        let mesh = try await frames.finishTracking()
        guard mesh.triangleCount > 0 else { throw DepthEngineError.emptyScan }
        return mesh
    }
}

private final class FrameRouter: TrueDepthCaptureDelegate {
    private let lock = NSLock()
    private var tracker: FrameTracker?
    private var recorder: CaptureRecorder?
    private var trackerBusy = false
    private var trackingHandler: ((TrackingStatus) -> Void)?
    private let trackingQueue = DispatchQueue(label: "depth2print.tracking", qos: .userInitiated)

    var onTrackingUpdate: ((TrackingStatus) -> Void)? {
        get { lock.withLock { trackingHandler } }
        set { lock.withLock { trackingHandler = newValue } }
    }

    func start(_ tracker: FrameTracker, recorder: CaptureRecorder?) {
        lock.withLock {
            self.tracker = tracker
            self.recorder = recorder
        }
    }

    /// Stops tracking, then meshes the fused model once any in-flight frame has been fused.
    func finishTracking() async throws -> Mesh {
        let (tracker, recorder) = lock.withLock {
            defer {
                self.tracker = nil
                self.recorder = nil
            }
            return (self.tracker, self.recorder)
        }
        guard let tracker else { throw DepthEngineError.noDepthData }
        return try await withCheckedThrowingContinuation { continuation in
            trackingQueue.async {
                recorder?.finish()
                continuation.resume(with: Result { try tracker.extractMesh() })
            }
        }
    }

    func captureDidOutput(_ frame: DepthFrame) {
        let (tracker, recorder): (FrameTracker?, CaptureRecorder?) = lock.withLock {
            // Skip this frame if the previous one is still aligning; the velocity prior covers the gap.
            guard !trackerBusy, let tracker = self.tracker else { return (nil, self.recorder) }
            trackerBusy = true
            return (tracker, self.recorder)
        }
        recorder?.append(frame)
        guard let tracker else { return }
        trackingQueue.async {
            let status = try? tracker.track(frame)
            let handler: ((TrackingStatus) -> Void)? = self.lock.withLock {
                self.trackerBusy = false
                return self.trackingHandler
            }
            if let status {
                recorder?.log(status, time: frame.timestamp.seconds)
                handler?(status)
            }
        }
    }
}
