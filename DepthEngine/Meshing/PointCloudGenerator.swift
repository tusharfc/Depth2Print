import CoreVideo
import Metal
import simd

struct PointGrid {
    let width: Int
    let height: Int
    /// Row-major, one point per depth pixel. w == 0 marks invalid.
    let points: [SIMD4<Float>]
}

private struct UnprojectParams {
    var fx, fy, cx, cy, minDepth, maxDepth: Float
}

final class PointCloudGenerator {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let textureCache: CVMetalTextureCache

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw DepthEngineError.metalUnavailable
        }
        // Shaders compile into DepthEngine.framework's default.metallib, not the app's.
        let library = try device.makeDefaultLibrary(bundle: Bundle(for: PointCloudGenerator.self))
        guard let function = library.makeFunction(name: "unprojectDepth") else {
            throw DepthEngineError.shaderMissing("unprojectDepth")
        }
        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(nil, nil, device, nil, &cache) == kCVReturnSuccess, let cache else {
            throw DepthEngineError.metalUnavailable
        }
        self.device = device
        self.queue = queue
        self.pipeline = try device.makeComputePipelineState(function: function)
        self.textureCache = cache
    }

    func unproject(_ frame: DepthFrame, depthRange: ClosedRange<Float> = 0.1...1.0) throws -> PointGrid {
        let width = CVPixelBufferGetWidth(frame.depth)
        let height = CVPixelBufferGetHeight(frame.depth)

        var cvTexture: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, textureCache, frame.depth, nil, .r16Float,
                                                  width, height, 0, &cvTexture)
        guard let cvTexture, let texture = CVMetalTextureGetTexture(cvTexture) else {
            throw DepthEngineError.textureCreationFailed
        }

        let count = width * height
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<SIMD4<Float>>.stride,
                                             options: .storageModeShared),
              let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder() else {
            throw DepthEngineError.metalUnavailable
        }

        let k = frame.intrinsics
        var params = UnprojectParams(fx: k[0][0], fy: k[1][1], cx: k[2][0], cy: k[2][1],
                                     minDepth: depthRange.lowerBound, maxDepth: depthRange.upperBound)
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setBytes(&params, length: MemoryLayout<UnprojectParams>.stride, index: 0)
        encoder.setBuffer(buffer, offset: 0, index: 1)
        encoder.dispatchThreadgroups(MTLSize(width: (width + 15) / 16, height: (height + 15) / 16, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw error }

        let pointer = buffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        return PointGrid(width: width, height: height,
                         points: Array(UnsafeBufferPointer(start: pointer, count: count)))
    }
}
