import Metal
import simd

// Layout must match VolumeParams in TSDF.metal.
private struct VolumeParams {
    var worldToCamera: simd_float4x4
    var cameraToWorld: simd_float4x4
    var origin: SIMD4<Float>
    var fx, fy, cx, cy: Float
    var resolution, width, height: UInt32
    var voxelSize, truncation, maxWeight, minDepth, maxDepth: Float
}

/// Dense truncated signed distance field on the GPU. Each voxel is half2 (distance / truncation, weight),
/// so 256³ voxels take 64 MiB instead of the 512 MiB of a 512³ float grid.
final class TSDFVolume {
    struct Config {
        var resolution = 256
        /// Edge length of the cubic volume in meters; 0.3 m at 256³ gives ~1.2 mm voxels.
        var size: Float = 0.3
        var truncationVoxels: Float = 4
        /// Caps the running average so the model can still adapt to new observations.
        var maxWeight: Float = 64
    }

    let config: Config
    let origin: SIMD3<Float>
    let voxelSize: Float
    let truncation: Float
    let buffer: MTLBuffer

    private let queue: MTLCommandQueue
    private let integratePipeline: MTLComputePipelineState
    private let raycastPipeline: MTLComputePipelineState

    private static let volumeGroup = MTLSize(width: 8, height: 8, depth: 4)
    private static let imageGroup = MTLSize(width: 16, height: 16, depth: 1)

    init(device: MTLDevice, library: MTLLibrary, center: SIMD3<Float>, config: Config = Config()) throws {
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw DepthEngineError.shaderMissing(name) }
            return try device.makeComputePipelineState(function: function)
        }
        let voxels = config.resolution * config.resolution * config.resolution
        guard let queue = device.makeCommandQueue(),
              let buffer = device.makeBuffer(length: voxels * MemoryLayout<UInt32>.stride,   // half2
                                             options: .storageModePrivate)
        else { throw DepthEngineError.metalUnavailable }

        self.config = config
        self.voxelSize = config.size / Float(config.resolution)
        self.truncation = config.truncationVoxels * voxelSize
        self.origin = center - SIMD3(repeating: config.size / 2)
        self.buffer = buffer
        self.queue = queue
        self.integratePipeline = try pipeline("integrateVolume")
        self.raycastPipeline = try pipeline("raycastVolume")

        let reset = try pipeline("resetVolume")
        try run { encoder in
            var count = UInt32(voxels)
            encoder.setComputePipelineState(reset)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.dispatchThreadgroups(MTLSize(width: (voxels + 255) / 256, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    /// `depth` is a masked Float32 depth map in meters (0 = invalid); `normals` its camera-space normals (w 0 = none).
    func integrate(depth: MTLBuffer, normals: MTLBuffer, width: Int, height: Int, intrinsics: CameraIntrinsics,
                   cameraToWorld: simd_float4x4, depthRange: ClosedRange<Float>) throws {
        var params = makeParams(width: width, height: height, intrinsics: intrinsics,
                                cameraToWorld: cameraToWorld, depthRange: depthRange)
        let n = config.resolution
        try run { encoder in
            encoder.setComputePipelineState(integratePipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBuffer(depth, offset: 0, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<VolumeParams>.stride, index: 2)
            encoder.setBuffer(normals, offset: 0, index: 3)
            let g = Self.volumeGroup
            encoder.dispatchThreadgroups(MTLSize(width: (n + g.width - 1) / g.width, height: (n + g.height - 1) / g.height,
                                                 depth: (n + g.depth - 1) / g.depth),
                                         threadsPerThreadgroup: g)
        }
    }

    /// Writes the model's depth as seen from `cameraToWorld` into `depth` (0 where nothing is hit).
    func raycast(into depth: MTLBuffer, width: Int, height: Int, intrinsics: CameraIntrinsics,
                 cameraToWorld: simd_float4x4, depthRange: ClosedRange<Float>) throws {
        var params = makeParams(width: width, height: height, intrinsics: intrinsics,
                                cameraToWorld: cameraToWorld, depthRange: depthRange)
        try run { encoder in
            encoder.setComputePipelineState(raycastPipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&params, length: MemoryLayout<VolumeParams>.stride, index: 1)
            encoder.setBuffer(depth, offset: 0, index: 2)
            let g = Self.imageGroup
            encoder.dispatchThreadgroups(MTLSize(width: (width + g.width - 1) / g.width,
                                                 height: (height + g.height - 1) / g.height, depth: 1),
                                         threadsPerThreadgroup: g)
        }
    }

    private func makeParams(width: Int, height: Int, intrinsics k: CameraIntrinsics,
                            cameraToWorld: simd_float4x4, depthRange: ClosedRange<Float>) -> VolumeParams {
        VolumeParams(worldToCamera: cameraToWorld.inverse, cameraToWorld: cameraToWorld,
                     origin: SIMD4(origin, 1), fx: k.fx, fy: k.fy, cx: k.cx, cy: k.cy,
                     resolution: UInt32(config.resolution), width: UInt32(width), height: UInt32(height),
                     voxelSize: voxelSize, truncation: truncation, maxWeight: config.maxWeight,
                     minDepth: depthRange.lowerBound, maxDepth: depthRange.upperBound)
    }

    private func run(_ encode: (MTLComputeCommandEncoder) -> Void) throws {
        guard let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder() else { throw DepthEngineError.metalUnavailable }
        encode(encoder)
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw error }
    }
}
