import CoreGraphics
import Foundation
import Metal

/// Turns the model's raycast normals into a small image the app lays over the camera feed.
final class PreviewRenderer {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private var pixels: MTLBuffer?

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let queue = device.makeCommandQueue() else { throw DepthEngineError.metalUnavailable }
        guard let function = library.makeFunction(name: "shadePreview") else {
            throw DepthEngineError.shaderMissing("shadePreview")
        }
        self.device = device
        self.queue = queue
        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    /// Same pixel grid as the depth buffer at this level (sensor-native, unmirrored). `hiding` is a camera-space
    /// plane (n, d); surfaces below it are left out.
    func image(_ level: FramePyramid.Level, hiding plane: SIMD4<Float>? = nil) throws -> CGImage? {
        let count = level.width * level.height
        if (pixels?.length ?? 0) < count * 4 {
            pixels = device.makeBuffer(length: count * 4, options: .storageModeShared)
        }
        guard let pixels, let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder() else { throw DepthEngineError.metalUnavailable }
        var size = SIMD2<UInt32>(UInt32(level.width), UInt32(level.height))
        var hide = plane ?? .zero
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(level.normals, offset: 0, index: 0)
        encoder.setBuffer(level.vertices, offset: 0, index: 1)
        encoder.setBuffer(pixels, offset: 0, index: 2)
        encoder.setBytes(&size, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 3)
        encoder.setBytes(&hide, length: MemoryLayout<SIMD4<Float>>.stride, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: (level.width + 15) / 16, height: (level.height + 15) / 16, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw error }

        // Copied so the image stays valid while the next frame overwrites the buffer.
        let data = Data(bytes: pixels.contents(), count: count * 4) as CFData
        guard let provider = CGDataProvider(data: data) else { return nil }
        return CGImage(width: level.width, height: level.height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: level.width * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Little.rawValue
                                                | CGImageAlphaInfo.premultipliedFirst.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}
