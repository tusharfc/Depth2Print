import AVFoundation
import CoreMotion
import CoreVideo
import simd

public struct DepthFrame {
    /// kCVPixelFormatType_DepthFloat16, meters. Invalid pixels are NaN or 0.
    public let depth: CVPixelBuffer
    /// kCVPixelFormatType_32BGRA, aligned to `depth`. Nil if the color frame was dropped.
    public let color: CVPixelBuffer?
    /// Pinhole intrinsics scaled to the depth map's resolution (column-major: fx=[0][0], fy=[1][1], cx=[2][0], cy=[2][1]).
    public let intrinsics: simd_float3x3
    public let calibration: AVCameraCalibrationData
    public let timestamp: CMTime
    /// Gravity in CoreMotion device axes (x right, y toward the top edge, z out of the screen), in g.
    public let gravity: SIMD3<Float>?
}

public protocol TrueDepthCaptureDelegate: AnyObject {
    /// Called on the capture data queue. Return immediately and don't hold more than ~2 frames.
    func captureManager(_ manager: TrueDepthCaptureManager, didOutput frame: DepthFrame)
}

// State is only touched on sessionQueue (configuration) or dataQueue (delivery).
public final class TrueDepthCaptureManager: NSObject, @unchecked Sendable {

    public weak var delegate: TrueDepthCaptureDelegate?
    public let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "depth2print.capture.session")
    private let dataQueue = DispatchQueue(label: "depth2print.capture.data",
                                          qos: .userInitiated,
                                          autoreleaseFrequency: .workItem)

    private let videoOutput = AVCaptureVideoDataOutput()
    private let depthOutput = AVCaptureDepthDataOutput()
    private var synchronizer: AVCaptureDataOutputSynchronizer?
    private var device: AVCaptureDevice?
    private let motion = CMMotionManager()

    private let preferredFPS: Int32
    private var maxSupportedFPS: Int32 = 30
    private var isConfigured = false
    private var pressureObservation: NSKeyValueObservation?
    private var runtimeErrorObserver: NSObjectProtocol?

    public init(preferredFPS: Int32 = 60) {
        self.preferredFPS = preferredFPS
        super.init()
    }

    deinit {
        if let runtimeErrorObserver { NotificationCenter.default.removeObserver(runtimeErrorObserver) }
    }

    // MARK: - Lifecycle

    public func start() async throws {
        guard await AVCaptureDevice.requestAccess(for: .video) else {
            throw DepthEngineError.cameraAccessDenied
        }
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            sessionQueue.async {
                do {
                    if !self.isConfigured {
                        try self.configureSession()
                        self.isConfigured = true
                    }
                    if !self.session.isRunning { self.session.startRunning() }
                    if self.motion.isDeviceMotionAvailable && !self.motion.isDeviceMotionActive {
                        self.motion.deviceMotionUpdateInterval = 1.0 / 60
                        self.motion.startDeviceMotionUpdates()
                    }
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    public func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
            self.motion.stopDeviceMotionUpdates()
        }
    }

    /// Lock once auto-exposure has settled so texture color doesn't drift as the turntable rotates.
    public func setImagingLocked(_ locked: Bool) {
        sessionQueue.async {
            guard let device = self.device else { return }
            do { try device.lockForConfiguration() } catch { return }
            defer { device.unlockForConfiguration() }
            let exposure: AVCaptureDevice.ExposureMode = locked ? .locked : .continuousAutoExposure
            let whiteBalance: AVCaptureDevice.WhiteBalanceMode = locked ? .locked : .continuousAutoWhiteBalance
            let focus: AVCaptureDevice.FocusMode = locked ? .locked : .continuousAutoFocus
            if device.isExposureModeSupported(exposure) { device.exposureMode = exposure }
            if device.isWhiteBalanceModeSupported(whiteBalance) { device.whiteBalanceMode = whiteBalance }
            if device.isFocusModeSupported(focus) { device.focusMode = focus }
        }
    }

    // MARK: - Configuration (sessionQueue only)

    private func configureSession() throws {
        guard let device = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) else {
            throw DepthEngineError.trueDepthUnavailable
        }
        self.device = device

        session.beginConfiguration()
        defer { session.commitConfiguration() }

        // Required so the manually chosen activeFormat isn't overridden by a preset.
        session.sessionPreset = .inputPriority

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw DepthEngineError.cannotAdd("TrueDepth input") }
        session.addInput(input)

        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        guard session.canAddOutput(videoOutput) else { throw DepthEngineError.cannotAdd("video output") }
        session.addOutput(videoOutput)

        // TSDF fusion does its own temporal averaging; Apple's filter smears edges on a rotating object.
        depthOutput.isFilteringEnabled = false
        depthOutput.alwaysDiscardsLateDepthData = true
        guard session.canAddOutput(depthOutput) else { throw DepthEngineError.cannotAdd("depth output") }
        session.addOutput(depthOutput)

        try configureFormats(device)

        // Keep both streams unmirrored in native sensor orientation so depth and color stay aligned.
        for connection in [videoOutput.connection(with: .video), depthOutput.connection(with: .depthData)] {
            guard let connection else { continue }
            connection.isEnabled = true
            if connection.isVideoMirroringSupported {
                connection.automaticallyAdjustsVideoMirroring = false   // Must precede isVideoMirrored or it throws.
                connection.isVideoMirrored = false
            }
        }

        let sync = AVCaptureDataOutputSynchronizer(dataOutputs: [depthOutput, videoOutput])
        sync.setDelegate(self, queue: dataQueue)
        synchronizer = sync

        observeSystemPressure(device)
        observeRuntimeErrors()
    }

    private func configureFormats(_ device: AVCaptureDevice) throws {
        func dims(_ f: AVCaptureDevice.Format) -> CMVideoDimensions {
            CMVideoFormatDescriptionGetDimensions(f.formatDescription)
        }
        func area(_ f: AVCaptureDevice.Format) -> Int32 { let d = dims(f); return d.width * d.height }
        func maxFPS(_ f: AVCaptureDevice.Format) -> Int32 {
            Int32(f.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0)
        }

        typealias Candidate = (video: AVCaptureDevice.Format, depth: AVCaptureDevice.Format, fps: Int32)
        var candidates: [Candidate] = []

        for video in device.formats {
            let float16Depth = video.supportedDepthDataFormats.filter {
                CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat16
            }
            guard let depth = float16Depth.max(by: { area($0) < area($1) }) else { continue }
            let v = dims(video), d = dims(depth)
            guard v.width * d.height == v.height * d.width else { continue }
            candidates.append((video, depth, min(maxFPS(video), maxFPS(depth))))
        }

        // Largest depth map, then highest fps, then smallest color frame (less bandwidth and heat).
        guard let best = candidates.min(by: { a, b in
            if area(a.depth) != area(b.depth) { return area(a.depth) > area(b.depth) }
            if a.fps != b.fps { return a.fps > b.fps }
            return area(a.video) < area(b.video)
        }) else {
            throw DepthEngineError.noFloat16DepthFormat
        }

        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        // Order matters: changing activeFormat resets the depth format and frame durations.
        device.activeFormat = best.video
        device.activeDepthDataFormat = best.depth
        maxSupportedFPS = best.fps
        applyFrameRate(min(preferredFPS, best.fps), on: device)
    }

    /// Caller must hold lockForConfiguration.
    private func applyFrameRate(_ fps: Int32, on device: AVCaptureDevice) {
        let duration = CMTime(value: 1, timescale: fps)
        device.activeVideoMinFrameDuration = duration
        device.activeVideoMaxFrameDuration = duration
        device.activeDepthDataMinFrameDuration = duration
    }

    // MARK: - Thermal / runtime resilience

    private func observeSystemPressure(_ device: AVCaptureDevice) {
        pressureObservation = device.observe(\.systemPressureState, options: [.new]) { [weak self] device, _ in
            let level = device.systemPressureState.level
            self?.sessionQueue.async { self?.handlePressure(level) }
        }
    }

    private func handlePressure(_ level: AVCaptureDevice.SystemPressureState.Level) {
        guard let device else { return }
        let fps: Int32
        switch level {
        case .shutdown:
            session.stopRunning()
            return
        case .serious, .critical:
            fps = min(15, maxSupportedFPS)
        default:
            fps = min(preferredFPS, maxSupportedFPS)
        }
        do { try device.lockForConfiguration() } catch { return }
        applyFrameRate(fps, on: device)
        device.unlockForConfiguration()
    }

    private func observeRuntimeErrors() {
        runtimeErrorObserver = NotificationCenter.default.addObserver(
            forName: AVCaptureSession.runtimeErrorNotification, object: session, queue: nil
        ) { [weak self] note in
            guard let self,
                  let error = note.userInfo?[AVCaptureSessionErrorKey] as? AVError,
                  error.code == .mediaServicesWereReset else { return }
            self.sessionQueue.async { self.session.startRunning() }
        }
    }

    private static func scaledIntrinsics(_ calibration: AVCameraCalibrationData,
                                         to depthMap: CVPixelBuffer) -> simd_float3x3 {
        var k = calibration.intrinsicMatrix
        let ref = calibration.intrinsicMatrixReferenceDimensions
        let sx = Float(CVPixelBufferGetWidth(depthMap)) / Float(ref.width)
        let sy = Float(CVPixelBufferGetHeight(depthMap)) / Float(ref.height)
        k[0][0] *= sx; k[2][0] *= sx
        k[1][1] *= sy; k[2][1] *= sy
        return k
    }
}

// MARK: - Synchronized delivery (dataQueue)

extension TrueDepthCaptureManager: AVCaptureDataOutputSynchronizerDelegate {
    public func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                       didOutput collection: AVCaptureSynchronizedDataCollection) {
        guard let syncedDepth = collection.synchronizedData(for: depthOutput) as? AVCaptureSynchronizedDepthData,
              !syncedDepth.depthDataWasDropped,
              let calibration = syncedDepth.depthData.cameraCalibrationData else { return }

        let depthMap = syncedDepth.depthData.depthDataMap

        var color: CVPixelBuffer?
        if let syncedVideo = collection.synchronizedData(for: videoOutput) as? AVCaptureSynchronizedSampleBufferData,
           !syncedVideo.sampleBufferWasDropped {
            color = CMSampleBufferGetImageBuffer(syncedVideo.sampleBuffer)
        }

        let gravity = motion.deviceMotion.map { SIMD3(Float($0.gravity.x), Float($0.gravity.y), Float($0.gravity.z)) }
        delegate?.captureManager(self, didOutput: DepthFrame(
            depth: depthMap,
            color: color,
            intrinsics: Self.scaledIntrinsics(calibration, to: depthMap),
            calibration: calibration,
            timestamp: syncedDepth.timestamp,
            gravity: gravity
        ))
    }
}
