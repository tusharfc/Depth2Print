import Metal
import simd

struct CameraIntrinsics {
    var fx, fy, cx, cy: Float

    init(fx: Float, fy: Float, cx: Float, cy: Float) {
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    }

    init(_ k: simd_float3x3) {
        self.init(fx: k[0][0], fy: k[1][1], cx: k[2][0], cy: k[2][1])
    }

    /// For a map downsampled 2x where each output pixel averages a 2x2 block.
    var halved: CameraIntrinsics {
        CameraIntrinsics(fx: fx / 2, fy: fy / 2, cx: (cx - 0.5) / 2, cy: (cy - 0.5) / 2)
    }
}

// Layouts must match the structs in ICP.metal.
private struct DepthFilterParams {
    var k: CameraIntrinsics
    var plane: SIMD4<Float>
    var region: SIMD4<Float>
    var minDepth, maxDepth, sigmaSpace, sigmaDepth, planeMargin: Float
}

private struct DownsampleParams {
    var srcWidth, srcHeight: UInt32
    var maxJump: Float
}

private struct MapParams {
    var k: CameraIntrinsics
    var width, height: UInt32
    var maxJump: Float
}

private struct ICPParams {
    var transform: simd_float4x4
    var k: CameraIntrinsics
    var width, height: UInt32
    var maxDistance, minNormalDot: Float
}

/// GPU-resident vertex/normal maps at successively halved resolutions; level 0 is full resolution.
final class FramePyramid {
    struct Level {
        let width: Int
        let height: Int
        let depth: MTLBuffer
        let vertices: MTLBuffer
        let normals: MTLBuffer
        var intrinsics = CameraIntrinsics(fx: 1, fy: 1, cx: 0, cy: 0)
    }

    fileprivate(set) var levels: [Level]
    /// Full-resolution depth with the same masking but no smoothing; this is what gets fused.
    let rawDepth: MTLBuffer

    var width: Int { levels[0].width }
    var height: Int { levels[0].height }

    fileprivate init(device: MTLDevice, width: Int, height: Int, levelCount: Int) throws {
        guard let rawDepth = device.makeBuffer(length: width * height * MemoryLayout<Float>.stride,
                                               options: .storageModeShared) else { throw DepthEngineError.metalUnavailable }
        self.rawDepth = rawDepth
        levels = []
        var w = width, h = height
        for _ in 0..<levelCount {
            let pixels = w * h
            // Shared (unified memory on Apple GPUs) so the CPU can read the coarse level to place the volume.
            guard let depth = device.makeBuffer(length: pixels * MemoryLayout<Float>.stride, options: .storageModeShared),
                  let vertices = device.makeBuffer(length: pixels * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared),
                  let normals = device.makeBuffer(length: pixels * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared)
            else { throw DepthEngineError.metalUnavailable }
            levels.append(Level(width: w, height: h, depth: depth, vertices: vertices, normals: normals))
            w /= 2
            h /= 2
        }
    }
}

struct ICPResult {
    /// Maps source-camera points into the reference camera.
    var transform: simd_float4x4
    var inliers: Int
    var rmse: Float
}

/// Parts of the current depth frame to hide from tracking and fusion, in that frame's camera space.
struct DepthMask {
    /// (n, d) with n pointing to the kept side; points with n·p + d below the margin are dropped.
    var plane: SIMD4<Float>?
    /// (center, radius); points outside the sphere are dropped.
    var region: SIMD4<Float>?
}

/// A line the motion is constrained to rotate about, in reference-camera space.
struct RotationAxis {
    var direction: SIMD3<Float>
    var point: SIMD3<Float>

    func rotation(by angle: Float) -> simd_float4x4 {
        let r = ICP.rigid(rotation: direction * angle, translation: .zero)
        let c = SIMD4(point, 1)
        var m = r
        m.columns.3 = c - r * c + SIMD4(0, 0, 0, 1)
        return m
    }
}

/// Coarse-to-fine point-to-plane ICP with projective data association (KinectFusion-style).
/// The reference can be the previous frame or, once TSDF lands, a raycast of the model.
final class ICP {
    struct Config {
        /// Interim object mask: anything outside this range (background, far table) is ignored.
        var depthRange: ClosedRange<Float> = 0.15...0.7
        var levelCount = 3
        /// Indexed fine to coarse; the coarsest level runs first. Full resolution only polishes, so it gets few.
        var iterations = [2, 4, 8]
        var maxDistance: [Float] = [0.01, 0.015, 0.02]
        var minInliers = [1000, 250, 60]
        var minNormalDot: Float = cos(30 * .pi / 180)
        var maxDepthJump: Float = 0.02
        /// Support-plane points this close above the plane are masked too, so the table never fuses into the mesh.
        var planeMargin: Float = 0.005
        /// Larger single steps mean degenerate geometry or bad correspondences, not real motion.
        var maxStepRotation: Float = 0.1
        var maxStepTranslation: Float = 0.03
    }

    let config: Config
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let bilateral: MTLComputePipelineState
    private let downsample: MTLComputePipelineState
    private let vertexNormal: MTLComputePipelineState
    private let reduce: MTLComputePipelineState
    private let sumPartials: MTLComputePipelineState
    private var partials: MTLBuffer?
    private let totals: MTLBuffer?

    private static let terms = 29
    private static let reduceGroup = MTLSize(width: 16, height: 8, depth: 1)
    private static let imageGroup = MTLSize(width: 16, height: 16, depth: 1)

    init(device: MTLDevice, library: MTLLibrary, config: Config = Config()) throws {
        precondition(config.iterations.count == config.levelCount
                     && config.maxDistance.count == config.levelCount
                     && config.minInliers.count == config.levelCount)
        guard let queue = device.makeCommandQueue() else { throw DepthEngineError.metalUnavailable }
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw DepthEngineError.shaderMissing(name) }
            return try device.makeComputePipelineState(function: function)
        }
        self.config = config
        self.device = device
        self.queue = queue
        self.bilateral = try pipeline("bilateralDepth")
        self.downsample = try pipeline("downsampleDepth")
        self.vertexNormal = try pipeline("vertexNormalMap")
        self.reduce = try pipeline("icpReduce")
        self.sumPartials = try pipeline("icpSumPartials")
        self.totals = device.makeBuffer(length: Self.terms * MemoryLayout<Float>.stride, options: .storageModeShared)
    }

    func makePyramid(width: Int, height: Int) throws -> FramePyramid {
        try FramePyramid(device: device, width: width, height: height, levelCount: config.levelCount)
    }

    /// Fills `pyramid` from a `.r16Float`/`.r32Float` depth texture in meters.
    func build(_ pyramid: FramePyramid, depth: MTLTexture, intrinsics: CameraIntrinsics,
               mask: DepthMask = DepthMask()) throws {
        precondition(depth.width == pyramid.width && depth.height == pyramid.height)
        guard let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder() else { throw DepthEngineError.metalUnavailable }

        var filter = DepthFilterParams(k: intrinsics, plane: mask.plane ?? .zero, region: mask.region ?? .zero,
                                       minDepth: config.depthRange.lowerBound, maxDepth: config.depthRange.upperBound,
                                       sigmaSpace: 1.5, sigmaDepth: 0.005, planeMargin: config.planeMargin)
        encoder.setComputePipelineState(bilateral)
        encoder.setTexture(depth, index: 0)
        encoder.setBytes(&filter, length: MemoryLayout<DepthFilterParams>.stride, index: 0)
        encoder.setBuffer(pyramid.levels[0].depth, offset: 0, index: 1)
        encoder.setBuffer(pyramid.rawDepth, offset: 0, index: 2)
        Self.dispatch(encoder, pyramid.width, pyramid.height, Self.imageGroup)

        try finish(encoder, commands, pyramid, intrinsics)
    }

    /// Fills the rest of `pyramid` from depth already written to level 0 (e.g. a TSDF raycast).
    func buildFromLevel0(_ pyramid: FramePyramid, intrinsics: CameraIntrinsics) throws {
        guard let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder() else { throw DepthEngineError.metalUnavailable }
        try finish(encoder, commands, pyramid, intrinsics)
    }

    private func finish(_ encoder: MTLComputeCommandEncoder, _ commands: MTLCommandBuffer,
                        _ pyramid: FramePyramid, _ intrinsics: CameraIntrinsics) throws {
        var k = intrinsics
        for level in pyramid.levels.indices {
            pyramid.levels[level].intrinsics = k
            k = k.halved
        }
        let levels = pyramid.levels

        encoder.setComputePipelineState(downsample)
        for level in levels.indices.dropFirst() {
            let src = levels[level - 1], dst = levels[level]
            var params = DownsampleParams(srcWidth: UInt32(src.width), srcHeight: UInt32(src.height),
                                          maxJump: config.maxDepthJump)
            encoder.setBuffer(src.depth, offset: 0, index: 0)
            encoder.setBytes(&params, length: MemoryLayout<DownsampleParams>.stride, index: 1)
            encoder.setBuffer(dst.depth, offset: 0, index: 2)
            Self.dispatch(encoder, dst.width, dst.height, Self.imageGroup)
        }

        encoder.setComputePipelineState(vertexNormal)
        for level in levels {
            var params = MapParams(k: level.intrinsics, width: UInt32(level.width), height: UInt32(level.height),
                                   maxJump: config.maxDepthJump)
            encoder.setBuffer(level.depth, offset: 0, index: 0)
            encoder.setBytes(&params, length: MemoryLayout<MapParams>.stride, index: 1)
            encoder.setBuffer(level.vertices, offset: 0, index: 2)
            encoder.setBuffer(level.normals, offset: 0, index: 3)
            Self.dispatch(encoder, level.width, level.height, Self.imageGroup)
        }

        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw error }
    }

    /// Why the last `align` returned nil, for diagnostics.
    private(set) var lastFailure: String?

    /// Returns nil when tracking fails (too few matches or a runaway step). With `axis`, only rotation about it
    /// is solved for (turntable).
    func align(source: FramePyramid, reference: FramePyramid, initial: simd_float4x4,
               axis: RotationAxis? = nil) throws -> ICPResult? {
        precondition(source.width == reference.width && source.height == reference.height)
        var transform = initial
        var inliers = 0
        var error = 0.0
        lastFailure = nil

        for level in stride(from: config.levelCount - 1, through: 0, by: -1) {
            for _ in 0..<config.iterations[level] {
                let system = try normalEquations(source.levels[level], reference.levels[level],
                                                 transform, level: level)
                guard system.inliers >= config.minInliers[level] else {
                    lastFailure = "few matches (level \(level): \(system.inliers) < \(config.minInliers[level]))"
                    return nil
                }
                inliers = system.inliers
                error = system.error

                let step = axis.map { Self.solveAboutAxis(system, $0) }
                    ?? Self.solveDegeneracyAware(system.ata, system.atb.map { -$0 })

                let rotation = SIMD3<Float>(Float(step[0]), Float(step[1]), Float(step[2]))
                let translation = SIMD3<Float>(Float(step[3]), Float(step[4]), Float(step[5]))
                guard length(rotation) < config.maxStepRotation,
                      length(translation) < config.maxStepTranslation else {
                    lastFailure = String(format: "jump too large (level %d: %.1f°, %.0f mm)", level,
                                         length(rotation) * 180 / .pi, length(translation) * 1000)
                    return nil
                }

                transform = Self.rigid(rotation: rotation, translation: translation) * transform
                if length(rotation) < 1e-5 && length(translation) < 1e-6 { break }
            }
        }
        return ICPResult(transform: transform, inliers: inliers, rmse: Float((error / Double(inliers)).squareRoot()))
    }

    // MARK: - Internals

    private struct NormalEquations {
        var ata = [Double](repeating: 0, count: 36)
        var atb = [Double](repeating: 0, count: 6)
        var error = 0.0
        var inliers = 0
    }

    private func normalEquations(_ src: FramePyramid.Level, _ ref: FramePyramid.Level,
                                 _ transform: simd_float4x4, level: Int) throws -> NormalEquations {
        // Each thread covers a 2x2 pixel block.
        let groupsX = ((src.width + 1) / 2 + Self.reduceGroup.width - 1) / Self.reduceGroup.width
        let groupsY = ((src.height + 1) / 2 + Self.reduceGroup.height - 1) / Self.reduceGroup.height
        let groupCount = groupsX * groupsY
        let length = groupCount * Self.terms * MemoryLayout<Float>.stride
        if (partials?.length ?? 0) < length {
            partials = device.makeBuffer(length: length, options: .storageModePrivate)
        }
        guard let partials, let totals,
              let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder() else { throw DepthEngineError.metalUnavailable }

        var params = ICPParams(transform: transform, k: ref.intrinsics,
                               width: UInt32(src.width), height: UInt32(src.height),
                               maxDistance: config.maxDistance[level], minNormalDot: config.minNormalDot)
        encoder.setComputePipelineState(reduce)
        encoder.setBuffer(src.vertices, offset: 0, index: 0)
        encoder.setBuffer(src.normals, offset: 0, index: 1)
        encoder.setBuffer(ref.vertices, offset: 0, index: 2)
        encoder.setBuffer(ref.normals, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<ICPParams>.stride, index: 4)
        encoder.setBuffer(partials, offset: 0, index: 5)
        encoder.dispatchThreadgroups(MTLSize(width: groupsX, height: groupsY, depth: 1),
                                     threadsPerThreadgroup: Self.reduceGroup)

        var count = UInt32(groupCount)
        encoder.setComputePipelineState(sumPartials)
        encoder.setBuffer(partials, offset: 0, index: 0)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBuffer(totals, offset: 0, index: 2)
        encoder.dispatchThreadgroups(MTLSize(width: Self.terms, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw error }

        let values = totals.contents().bindMemory(to: Float.self, capacity: Self.terms)
        let sums = (0..<Self.terms).map { Double(values[$0]) }

        var system = NormalEquations()
        var t = 0
        for a in 0..<6 {
            for b in a..<6 {
                system.ata[a * 6 + b] = sums[t]
                system.ata[b * 6 + a] = sums[t]
                t += 1
            }
        }
        for a in 0..<6 { system.atb[a] = sums[21 + a] }
        system.error = sums[27]
        system.inliers = Int(sums[28])
        return system
    }

    private static func dispatch(_ encoder: MTLComputeCommandEncoder, _ width: Int, _ height: Int, _ group: MTLSize) {
        encoder.dispatchThreadgroups(MTLSize(width: (width + group.width - 1) / group.width,
                                             height: (height + group.height - 1) / group.height, depth: 1),
                                     threadsPerThreadgroup: group)
    }

    /// Solves A x = b, leaving out directions the geometry doesn't constrain (e.g. spinning a vase about its
    /// own axis) so they keep the prior instead of being driven by noise.
    static func solveDegeneracyAware(_ a: [Double], _ b: [Double]) -> [Double] {
        // Translation is rescaled to ~object size so rotation and translation eigenvalues are comparable.
        let scale = 0.1
        let s = [1, 1, 1, scale, scale, scale]
        var scaled = a
        for i in 0..<6 { for j in 0..<6 { scaled[i * 6 + j] *= s[i] * s[j] } }
        let (values, vectors) = symmetricEigen(scaled, n: 6)
        // Only truly unconstrained directions: a big table plane makes object-only directions look weak by
        // comparison, and freezing those stalls tracking.
        let floor = 1e-8 * values.max()!
        var x = [Double](repeating: 0, count: 6)
        for k in 0..<6 where values[k] > floor {
            var projection = 0.0
            for i in 0..<6 { projection += vectors[i * 6 + k] * b[i] * s[i] }
            for i in 0..<6 { x[i] += vectors[i * 6 + k] * projection / values[k] }
        }
        return (0..<6).map { x[$0] * s[$0] }
    }

    /// One-parameter solve: rotation by dθ about `axis`, i.e. twist (n, c × n) dθ.
    private static func solveAboutAxis(_ system: NormalEquations, _ axis: RotationAxis) -> [Double] {
        let n = axis.direction, m = cross(axis.point, axis.direction)
        let g = [n.x, n.y, n.z, m.x, m.y, m.z].map(Double.init)
        var a = 0.0, b = 0.0
        for i in 0..<6 {
            b += g[i] * system.atb[i]
            for j in 0..<6 { a += g[i] * system.ata[i * 6 + j] * g[j] }
        }
        // A rotationally symmetric object gives no rotation cue; keep the prior rather than chase noise.
        let angle = a > 1e-6 * Double(system.inliers) ? -b / a : 0
        return g.map { $0 * angle }
    }

    /// Jacobi eigen-decomposition of a symmetric row-major matrix. Eigenvectors are the columns of `vectors`.
    static func symmetricEigen(_ matrix: [Double], n: Int) -> (values: [Double], vectors: [Double]) {
        var a = matrix
        var v = [Double](repeating: 0, count: n * n)
        for i in 0..<n { v[i * n + i] = 1 }
        for _ in 0..<50 {
            var off = 0.0, diagonal = 0.0
            for p in 0..<n {
                diagonal += a[p * n + p] * a[p * n + p]
                for q in (p + 1)..<n { off += a[p * n + q] * a[p * n + q] }
            }
            if off <= 1e-24 * diagonal { break }
            for p in 0..<(n - 1) {
                for q in (p + 1)..<n where a[p * n + q] != 0 {
                    let theta = (a[q * n + q] - a[p * n + p]) / (2 * a[p * n + q])
                    let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + (theta * theta + 1).squareRoot())
                    let c = 1 / (t * t + 1).squareRoot(), s = t * c
                    for k in 0..<n {
                        let kp = a[k * n + p], kq = a[k * n + q]
                        a[k * n + p] = c * kp - s * kq
                        a[k * n + q] = s * kp + c * kq
                    }
                    for k in 0..<n {
                        let pk = a[p * n + k], qk = a[q * n + k]
                        a[p * n + k] = c * pk - s * qk
                        a[q * n + k] = s * pk + c * qk
                    }
                    for k in 0..<n {
                        let kp = v[k * n + p], kq = v[k * n + q]
                        v[k * n + p] = c * kp - s * kq
                        v[k * n + q] = s * kp + c * kq
                    }
                }
            }
        }
        return ((0..<n).map { a[$0 * n + $0] }, v)
    }

    /// Rotation by axis-angle vector, then translation.
    static func rigid(rotation: SIMD3<Float>, translation: SIMD3<Float>) -> simd_float4x4 {
        let angle = length(rotation)
        let r = angle > 0 ? simd_float3x3(simd_quatf(angle: angle, axis: rotation / angle)) : matrix_identity_float3x3
        return simd_float4x4(columns: (SIMD4(r.columns.0, 0), SIMD4(r.columns.1, 0),
                                       SIMD4(r.columns.2, 0), SIMD4(translation, 1)))
    }
}
