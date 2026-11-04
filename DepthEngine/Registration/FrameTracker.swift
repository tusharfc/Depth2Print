import CoreGraphics
import CoreMedia
import CoreVideo
import Metal
import simd

public struct TrackingStatus {
    /// Current camera pose in the first frame's camera space, i.e. the object's frame on a turntable.
    public let pose: simd_float4x4
    /// Net rotation tracked so far; ~360 after one full turn of the turntable.
    public let rotationDegrees: Float
    /// Signed turn about gravity since the start (left/right for a face). 0 while gravity is unknown.
    public let headingDegrees: Float
    public let degreesPerSecond: Float
    /// False when the latest frame could not be aligned.
    public let isTracking: Bool
    public let framesTracked: Int
    public let framesLost: Int
    /// Point-to-plane residual of the last alignment, meters.
    public let rmse: Float
    /// Why the latest frame couldn't be aligned (nil while tracking).
    public let failureReason: String?
    /// Time the phone spent on this frame. Frames arriving meanwhile are skipped, so this sets the tracking rate.
    public let processingMilliseconds: Double
    /// Where that time went, e.g. "depth 3 · icp 12 · fuse 8 · model 5" (ms).
    public let stageTimings: String
    /// The fused model rendered from the current viewpoint, on the depth buffer's pixel grid (sensor-native,
    /// unmirrored). Refreshed about 12 times a second; nil on the frames in between.
    public let preview: CGImage?
}

/// KinectFusion loop: align each frame to the fused model with ICP, then fuse it into the TSDF.
/// Aligning to the model rather than the previous frame stops errors from accumulating over a turn.
/// World space is the first frame's camera space. Not thread-safe: use from one serial queue.
final class FrameTracker: @unchecked Sendable {
    private let device: MTLDevice
    private let library: MTLLibrary
    private let profile: ScanProfile
    private let icp: ICP
    private let textureCache: CVMetalTextureCache

    private var volume: TSDFVolume?
    private var current: FramePyramid?
    /// The model's surface raycast from the last tracked pose; ICP's reference.
    private var model: FramePyramid?
    private var modelIsCurrent = false
    private var lastTrackedTime = 0.0
    private var pose = matrix_identity_float4x4
    // Motion is roughly smooth, so the last motion is a good initial guess (scaled by elapsed time).
    private var angularVelocity = SIMD3<Float>(repeating: 0)
    private var linearVelocity = SIMD3<Float>(repeating: 0)
    private var netRotation = SIMD3<Float>(repeating: 0)
    private var tracked = 0
    private var lost = 0

    private var supportPlane: Plane?
    /// Where the mesh is cut flat and which side is kept; its normal becomes "up" for printing.
    private var cutPlane: Plane?
    private var headRegion: SIMD4<Float>?
    private var neckPlane: Plane?
    private var turntableAxis: RotationAxis?
    /// Poses tracked freely, used to fit the turntable axis; with their residuals as the "healthy" baseline.
    private var axisSamples: [simd_float4x4] = []
    private var freeRMSE: [Float] = []
    private var sampledRotation: Float = 0
    private var badLockedFrames = 0

    private let deviceFromCamera: simd_float3x3?
    /// Sum of per-frame "up" (against gravity) in world space; averaged rather than taken from one shaky sample.
    private var upSum = SIMD3<Float>(repeating: 0)
    private var heading: Float = 0
    private struct View { var origin: SIMD3<Float>; var edges: [SIMD3<Float>] }
    private var views: [View] = []
    /// Where the camera was, in world space, for every tracked frame.
    private var cameraPositions: [SIMD3<Float>] = []

    private let previewRenderer: PreviewRenderer
    private var colorizer = VertexColorizer()
    private var lastPreviewTime = -Double.infinity
    private var preview: CGImage?
    private var failure: String?
    private var processingStarted: TimeInterval = 0
    private var stages: [(name: String, ms: Double)] = []

    private func timed<T>(_ name: String, _ body: () throws -> T) rethrows -> T {
        let start = ProcessInfo.processInfo.systemUptime
        defer { stages.append((name, (ProcessInfo.processInfo.systemUptime - start) * 1000)) }
        return try body()
    }

    init(device: MTLDevice, library: MTLLibrary, profile: ScanProfile, deviceFromCamera: simd_float3x3? = nil) throws {
        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(nil, nil, device, nil, &cache) == kCVReturnSuccess, let cache else {
            throw DepthEngineError.metalUnavailable
        }
        self.device = device
        self.library = library
        self.profile = profile
        self.icp = try ICP(device: device, library: library, config: profile.icp)
        self.textureCache = cache
        self.deviceFromCamera = deviceFromCamera
        self.previewRenderer = try PreviewRenderer(device: device, library: library)
    }

    convenience init(mode: ScanMode, deviceFromCamera: simd_float3x3?) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw DepthEngineError.metalUnavailable }
        let library = try device.makeDefaultLibrary(bundle: Bundle(for: FrameTracker.self))
        try self.init(device: device, library: library, profile: ScanProfile(mode), deviceFromCamera: deviceFromCamera)
    }

    func track(_ frame: DepthFrame) throws -> TrackingStatus {
        let width = CVPixelBufferGetWidth(frame.depth)
        let height = CVPixelBufferGetHeight(frame.depth)
        var cvTexture: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, textureCache, frame.depth, nil, .r16Float,
                                                  width, height, 0, &cvTexture)
        guard let cvTexture, let texture = CVMetalTextureGetTexture(cvTexture) else {
            throw DepthEngineError.textureCreationFailed
        }
        return try withExtendedLifetime(cvTexture) {
            let gravity = deviceFromCamera.flatMap { m in frame.gravity.map { m.transpose * $0 } }
            let color = frame.color.map { buffer in { VertexColorizer.rgb(buffer, width: $0, height: $1) } }
            return try track(depth: texture, intrinsics: CameraIntrinsics(frame.intrinsics), time: frame.timestamp.seconds,
                             gravity: gravity, color: color)
        }
    }

    /// `gravity` is in this frame's camera space; `color` gives the matching colour image if there is one.
    func track(depth: MTLTexture, intrinsics: CameraIntrinsics, time: Double,
               gravity: SIMD3<Float>? = nil, color: ColorSampler? = nil) throws -> TrackingStatus {
        processingStarted = ProcessInfo.processInfo.systemUptime
        stages.removeAll()
        failure = nil
        preview = nil
        if current?.width != depth.width || current?.height != depth.height {
            current = try icp.makePyramid(width: depth.width, height: depth.height)
            model = try icp.makePyramid(width: depth.width, height: depth.height)
            modelIsCurrent = false
        }
        let source = current!, model = model!
        guard let volume else {
            return try start(source, model, depth: depth, intrinsics: intrinsics, time: time, gravity: gravity, color: color)
        }

        let dt = Float(time - lastTrackedTime)
        let axis = turntableAxis.map(referenceSpace)
        let guess = axis.map { $0.rotation(by: dot(angularVelocity, $0.direction) * dt) }
            ?? ICP.rigid(rotation: angularVelocity * dt, translation: linearVelocity * dt)
        try timed("depth") { try icp.build(source, depth: depth, intrinsics: intrinsics, mask: mask(cameraToWorld: pose * guess)) }
        if !modelIsCurrent { try render(volume, into: model, intrinsics: intrinsics) }

        // If the motion prior is off (a jolt, or the user paused), retry from the last good pose: that is also how
        // tracking recovers once the user moves back to where it was lost.
        guard let result = try timed("icp", {
            try icp.align(source: source, reference: model, initial: guess, axis: axis)
                ?? icp.align(source: source, reference: model, initial: matrix_identity_float4x4, axis: axis)
        }) else {
            failure = icp.lastFailure
            lost += 1
            return status(isTracking: false, rmse: 0)
        }

        let motion = result.transform
        let rotation = Self.rotationVector(motion)
        if dt > 0 {
            // Smoothed so hand tremor in one frame doesn't throw off the next frame's guess.
            let blend: Float = 0.3
            angularVelocity += (rotation / dt - angularVelocity) * blend
            linearVelocity += (SIMD3(motion.columns.3.x, motion.columns.3.y, motion.columns.3.z) / dt - linearVelocity) * blend
        }
        netRotation += Self.upperLeft(pose) * rotation
        if let up { heading += dot(Self.upperLeft(pose) * rotation, up) }
        pose = Self.orthonormalized(pose * motion)
        try fuse(source, into: volume, model: model, intrinsics: intrinsics, time: time)
        record(gravity: gravity, intrinsics: intrinsics, width: depth.width, height: depth.height)
        if let color { colorizer.offer(cameraToWorld: pose, level: source.levels[1], speed: length(angularVelocity), color: color) }
        tracked += 1

        if profile.locksTurntableAxis { updateTurntableAxis(volume, rmse: result.rmse, turned: length(rotation)) }
        return status(isTracking: true, rmse: result.rmse)
    }

    /// Locks rotation to the turntable axis once 45° of free tracking pins it down, and unlocks again if the
    /// fit stops explaining the motion (residual well above what free tracking achieved).
    private func updateTurntableAxis(_ volume: TSDFVolume, rmse: Float, turned: Float) {
        guard turntableAxis != nil else {
            axisSamples.append(pose)
            freeRMSE.append(rmse)
            sampledRotation += turned
            if sampledRotation > 45 * .pi / 180 && axisSamples.count >= 10 { turntableAxis = fitTurntableAxis(volume) }
            return
        }
        let baseline = freeRMSE.sorted()[freeRMSE.count / 2]
        badLockedFrames = rmse > max(0.001, 2.5 * baseline) ? badLockedFrames + 1 : 0
        if badLockedFrames >= 5 {
            turntableAxis = nil
            badLockedFrames = 0
            // Re-fit only from what comes next; the poses that led to the bad fit are suspect.
            axisSamples.removeAll()
            freeRMSE.removeAll()
            sampledRotation = 0
        }
    }

    /// Watertight mesh of everything fused so far, cut flat and placed for slicers: Z up (against gravity, or off
    /// the support plane; a face relief lies on its back), base on z = 0, centered in X/Y.
    func extractMesh() throws -> Mesh {
        if profile.locksTurntableAxis && length(netRotation) < .pi / 2 { throw DepthEngineError.turntableRotationNotSeen }
        let layout = printLayout()
        var mesh = Self.placed(try worldMesh(), z: layout.z, y: layout.y)
        // Matches printLayout: a face, or a one-sided scan with no table, is laid on its flat back.
        mesh.liesOnBack = headRegion != nil || (cutPlane == nil && flatBack() != nil)
        return mesh
    }

    func worldMesh() throws -> Mesh {
        guard let volume else { throw DepthEngineError.noDepthData }
        return colorizer.paint(try MarchingCubes(device: device, library: library)
            .extract(volume, cuts: printLayout().cuts.map(\.vector), region: headRegion)
            .largestComponent())
    }

    var detectedSupportPlane: Plane? { supportPlane }
    var detectedHead: (region: SIMD4<Float>?, neck: Plane?) { (headRegion, neckPlane) }
    var isAxisLocked: Bool { turntableAxis != nil }
    /// Tests turn this off to track frames against the model without adding them to it.
    var fusesFrames = true

    /// World "up", once gravity has been seen.
    var up: SIMD3<Float>? { length(upSum) > 0 ? normalize(upSum) : nil }

    /// Cuts, the direction that becomes +Z, and a hint for +Y (the first camera's forward, or head-up for a relief).
    private func printLayout() -> (cuts: [Plane], z: SIMD3<Float>?, y: SIMD3<Float>?) {
        let forward = SIMD3<Float>(0, 0, 1)
        if let plane = cutPlane, headRegion != nil { return ([plane] + [neckPlane].compactMap { $0 }, plane.normal, up) }
        let back = flatBack()
        if let plane = cutPlane { return ([plane] + [back].compactMap { $0 }, plane.normal, forward) }
        if let back {
            // A relief lies on its flat back, top of the subject along +Y.
            let bottom = profile.cutsBelowView ? up.flatMap(lowestCommonView) : nil
            return ([back] + [bottom].compactMap { $0 }, back.normal, up ?? forward)
        }
        guard let up else { return ([], nil, nil) }
        return ([profile.cutsBelowView ? lowestCommonView(up) : nil].compactMap { $0 }, up, forward)
    }

    /// For a scan seen from one side only, a flat back through the volume's center, facing the cameras. Without it,
    /// space that was never seen from behind gets filled in, turning a front view into a solid block. "One side" means
    /// the camera directions are concentrated: mean resultant length above 0.5, roughly less than ±105° of orbit.
    private func flatBack() -> Plane? {
        guard let volume, cameraPositions.count >= 2 else { return nil }
        let center = volume.origin + volume.config.size / 2
        let directions = cameraPositions.map { normalize($0 - center) }
        var facing = directions.reduce(SIMD3<Float>(repeating: 0), +) / Float(directions.count)
        guard length(facing) > 0.5 else { return nil }
        // Objects standing on a table keep a vertical back, so they still sit flat on their base.
        if supportPlane != nil, let up { facing -= dot(facing, up) * up }
        let normal = normalize(facing)
        return Plane(normal: normal, offset: -dot(normal, center))
    }

    /// Horizontal cut where most views stopped seeing the subject; below it the shape was only guessed from a few.
    private func lowestCommonView(_ up: SIMD3<Float>) -> Plane? {
        guard let volume, !views.isEmpty else { return nil }
        let center = volume.origin + volume.config.size / 2
        let bottoms = views.map { view -> Float in
            let reach = length(center - view.origin)
            return view.edges.map { dot(up, view.origin + $0 * reach) }.min()!
        }.sorted()
        return Plane(normal: up, offset: -(bottoms[bottoms.count / 2] + 0.02))
    }

    private func record(gravity: SIMD3<Float>?, intrinsics k: CameraIntrinsics, width: Int, height: Int) {
        let rotation = Self.upperLeft(pose)
        cameraPositions.append(SIMD3(pose.columns.3.x, pose.columns.3.y, pose.columns.3.z))
        if let gravity, length(gravity) > 0.5 { upSum += rotation * normalize(-gravity) }
        if profile.cutsBelowView {
            var edges: [SIMD3<Float>] = []
            for v in [0, height / 2, height - 1] {
                for u in [0, width / 2, width - 1] where u != width / 2 || v != height / 2 {
                    edges.append(normalize(rotation * SIMD3((Float(u) - k.cx) / k.fx, (Float(v) - k.cy) / k.fy, 1)))
                }
            }
            views.append(View(origin: SIMD3(pose.columns.3.x, pose.columns.3.y, pose.columns.3.z), edges: edges))
        }
    }

    private func start(_ source: FramePyramid, _ model: FramePyramid, depth: MTLTexture, intrinsics: CameraIntrinsics,
                       time: Double, gravity: SIMD3<Float>?, color: ColorSampler?) throws -> TrackingStatus {
        try icp.build(source, depth: depth, intrinsics: intrinsics)
        let points = SceneSetup.points(source.levels[1])
        let upHere = gravity.flatMap { length($0) > 0.5 ? normalize(-$0) : nil }
        if profile.findsSupportPlane { supportPlane = SceneSetup.supportPlane(points, up: upHere) }
        if profile.isolatesHead, let head = SceneSetup.head(points, up: upHere) {
            headRegion = head.region
            neckPlane = head.neck
        }

        let center = headRegion.map { SIMD3($0.x, $0.y, $0.z) }
            ?? SceneSetup.volumeCenter(points, plane: supportPlane, size: profile.volume.size)
        guard let center, !profile.isolatesHead || headRegion != nil else {
            failure = profile.isolatesHead ? "no face in range" : "nothing in range"
            lost += 1
            return status(isTracking: false, rmse: 0)
        }
        // A few mm above the support plane so table noise doesn't leave a film under the object.
        cutPlane = supportPlane.map { Plane(normal: $0.normal, offset: $0.offset - 0.003) }
        if headRegion != nil {
            // The back of the head is never seen; keep the front half as a flat-backed relief.
            cutPlane = Plane(normal: SIMD3(0, 0, -1), offset: center.z)
        }

        let firstMask = mask(cameraToWorld: pose)
        if firstMask.plane != nil || firstMask.region != nil {
            try icp.build(source, depth: depth, intrinsics: intrinsics, mask: firstMask)
        }
        // A face keeps only its front half, so shift the volume toward the camera to spend voxels there.
        let volumeCenter = headRegion == nil ? center : center - normalize(center) * 0.02
        let volume = try TSDFVolume(device: device, library: library, center: volumeCenter, config: profile.volume)
        self.volume = volume
        try fuse(source, into: volume, model: model, intrinsics: intrinsics, time: time)
        record(gravity: gravity, intrinsics: intrinsics, width: depth.width, height: depth.height)
        if let color { colorizer.offer(cameraToWorld: pose, level: source.levels[1], speed: 0, color: color) }
        tracked += 1
        return status(isTracking: true, rmse: 0)
    }

    private func mask(cameraToWorld: simd_float4x4) -> DepthMask {
        var mask = DepthMask()
        // The camera doesn't move on a turntable, so the first frame's plane stays put in camera space.
        if profile.masksSupportPlane { mask.plane = supportPlane?.vector }
        if let head = headRegion {
            let c = cameraToWorld.inverse * SIMD4(head.x, head.y, head.z, 1)
            mask.region = SIMD4(c.x, c.y, c.z, head.w)
        }
        // The chest stays still while the head turns, so it must not take part in tracking either.
        if let neckPlane { mask.plane = Self.cameraSpace(neckPlane, cameraToWorld: cameraToWorld) }
        return mask
    }

    /// A world plane in the camera's space: n_c = Rᵀ n_w, d_c = n_w · t + d_w.
    private static func cameraSpace(_ plane: Plane, cameraToWorld: simd_float4x4) -> SIMD4<Float> {
        let t = SIMD3(cameraToWorld.columns.3.x, cameraToWorld.columns.3.y, cameraToWorld.columns.3.z)
        return SIMD4(upperLeft(cameraToWorld).transpose * plane.normal, dot(plane.normal, t) + plane.offset)
    }

    private func referenceSpace(_ axis: RotationAxis) -> RotationAxis {
        let worldToCamera = pose.inverse
        let p = worldToCamera * SIMD4(axis.point, 1)
        return RotationAxis(direction: Self.upperLeft(worldToCamera) * axis.direction, point: SIMD3(p.x, p.y, p.z))
    }

    /// Axis direction from the turntable plane when it agrees with the measured rotation (it is fitted to thousands
    /// of points, so it is steadier). The point on the axis is the least-squares solution of (I − Rₖ)c = tₖ over
    /// all free poses, which averages out per-frame noise that a single pose would carry straight into the lock.
    private func fitTurntableAxis(_ volume: TSDFVolume) -> RotationAxis {
        var n = normalize(Self.rotationVector(pose))
        if let plane = supportPlane, abs(dot(plane.normal, n)) > cos(10 * Float.pi / 180) { n = plane.normal }
        // (I − R) can't see along the axis; n nᵀ pins that component to the volume's center.
        let anchor = volume.origin + volume.config.size / 2
        let weight = Float(axisSamples.count)
        var normal = simd_float3x3(n * n.x, n * n.y, n * n.z) * weight
        var rhs = n * dot(n, anchor) * weight
        for sample in axisSamples {
            let a = matrix_identity_float3x3 - Self.upperLeft(sample)
            normal += a.transpose * a
            rhs += a.transpose * SIMD3(sample.columns.3.x, sample.columns.3.y, sample.columns.3.z)
        }
        return RotationAxis(direction: n, point: normal.inverse * rhs)
    }

    private static func placed(_ mesh: Mesh, z: SIMD3<Float>?, y yHint: SIMD3<Float>?) -> Mesh {
        guard !mesh.vertices.isEmpty else { return mesh }
        var rotation = matrix_identity_float3x3
        if let z = z.map({ normalize($0) }) {
            var y = yHint.map { $0 - dot($0, z) * z } ?? .zero
            if length(y) < 0.1 { y = abs(z.x) < 0.9 ? SIMD3(1, 0, 0) - z.x * z : SIMD3(0, 1, 0) - z.y * z }
            y = normalize(y)
            rotation = simd_float3x3(rows: [cross(y, z), y, z])
        }
        let rotated = mesh.vertices.map { rotation * $0 }
        let lo = rotated.reduce(rotated[0]) { simd_min($0, $1) }
        let hi = rotated.reduce(rotated[0]) { simd_max($0, $1) }
        let mid = (lo + hi) / 2
        let shift = SIMD3(mid.x, mid.y, z == nil ? mid.z : lo.z)
        return Mesh(vertices: rotated.map { $0 - shift }, indices: mesh.indices, colors: mesh.colors)
    }

    private func fuse(_ source: FramePyramid, into volume: TSDFVolume, model: FramePyramid,
                      intrinsics: CameraIntrinsics, time: Double) throws {
        let level = source.levels[0]
        try timed("fuse") {
            guard fusesFrames else { return }
            try volume.integrate(depth: source.rawDepth, normals: level.normals, width: level.width, height: level.height,
                                 intrinsics: intrinsics, cameraToWorld: pose, depthRange: icp.config.depthRange)
        }
        try timed("model") { try render(volume, into: model, intrinsics: intrinsics) }
        lastTrackedTime = time
        if time - lastPreviewTime >= 1.0 / 12 {
            // The cut plane moved into this camera's space: n_c = Rᵀ n_w, d_c = n_w · t + d_w.
            // Only the table is hidden; the face cut would hide the side of a turned head.
            let hide = supportPlane == nil ? nil : cutPlane.map { plane -> SIMD4<Float> in
                let t = SIMD3(pose.columns.3.x, pose.columns.3.y, pose.columns.3.z)
                return SIMD4(Self.upperLeft(pose).transpose * plane.normal, dot(plane.normal, t) + plane.offset)
            }
            preview = try timed("preview") { try previewRenderer.image(model.levels[1], hiding: hide) }
            lastPreviewTime = time
        }
    }

    private func render(_ volume: TSDFVolume, into model: FramePyramid, intrinsics: CameraIntrinsics) throws {
        let level = model.levels[0]
        try volume.raycast(into: level.depth, width: level.width, height: level.height, intrinsics: intrinsics,
                           cameraToWorld: pose, depthRange: icp.config.depthRange)
        try icp.buildFromLevel0(model, intrinsics: intrinsics)
        modelIsCurrent = true
    }

    private func status(isTracking: Bool, rmse: Float) -> TrackingStatus {
        TrackingStatus(pose: pose, rotationDegrees: length(netRotation) * 180 / .pi, headingDegrees: heading * 180 / .pi,
                       degreesPerSecond: length(angularVelocity) * 180 / .pi, isTracking: isTracking,
                       framesTracked: tracked, framesLost: lost, rmse: rmse, failureReason: failure,
                       processingMilliseconds: (ProcessInfo.processInfo.systemUptime - processingStarted) * 1000,
                       stageTimings: stages.map { String(format: "%@ %.0f", $0.name, $0.ms) }.joined(separator: " · "),
                       preview: preview)
    }

    private static func upperLeft(_ m: simd_float4x4) -> simd_float3x3 {
        simd_float3x3(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
                      SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
                      SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z))
    }

    /// Axis-angle vector with angle in [0, π].
    static func rotationVector(_ m: simd_float4x4) -> SIMD3<Float> {
        var q = simd_quatf(upperLeft(m))
        if q.real < 0 { q = simd_quatf(vector: -q.vector) }
        let angle = q.angle
        return angle > 1e-9 ? q.axis * angle : SIMD3(repeating: 0)
    }

    /// Removes float drift that accumulates when thousands of rigid transforms are chained.
    private static func orthonormalized(_ m: simd_float4x4) -> simd_float4x4 {
        let r = simd_float3x3(simd_normalize(simd_quatf(upperLeft(m))))
        return simd_float4x4(columns: (SIMD4(r.columns.0, 0), SIMD4(r.columns.1, 0),
                                       SIMD4(r.columns.2, 0), m.columns.3))
    }
}
