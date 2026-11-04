import AVFoundation
import simd

/// Public entry point: TrueDepth camera feed in, triangle mesh out. No UI dependencies.
public final class DepthScanner {
    /// Exposed so a host app can attach its own preview layer.
    public var captureSession: AVCaptureSession { capture.session }

    private let capture = TrueDepthCaptureManager()
    private let frames = FrameRouter()

    /// Called on a background queue for each frame tracked during a scan.
    public var onTrackingUpdate: ((TrackingStatus) -> Void)? {
        get { frames.onTrackingUpdate }
        set { frames.onTrackingUpdate = newValue }
    }

    public init() {
        capture.delegate = frames
    }

    public func startPreview() async throws {
        try await capture.start()
    }

    public func stop() {
        capture.stop()
    }

    /// `deviceFromCamera` (see `CameraAxes`) lets gravity orient the export and pick the support surface;
    /// without it, orientation falls back to geometry alone. With `recordingTo`, every raw depth frame is saved
    /// there for replay on a Mac (see `CaptureRecorder`); it's kept even if the scan fails.
    public func beginScan(mode: ScanMode, deviceFromCamera: simd_float3x3? = nil, recordingTo directory: URL? = nil) throws {
        let recorder = try directory.map { try CaptureRecorder(directory: $0, mode: mode, deviceFromCamera: deviceFromCamera) }
        frames.start(try FrameTracker(mode: mode, deviceFromCamera: deviceFromCamera), recorder: recorder)
        capture.setImagingLocked(true)
    }

    public func finishScan() async throws -> Mesh {
        capture.setImagingLocked(false)
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

    func captureManager(_ manager: TrueDepthCaptureManager, didOutput frame: DepthFrame) {
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
