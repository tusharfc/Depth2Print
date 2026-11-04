import CoreVideo
import Dispatch
import simd

/// Returns the colour frame as RGB bytes at `width` × `height`, registered to the depth map; nil if there is none.
typealias ColorSampler = (_ width: Int, _ height: Int) -> [UInt8]?

/// Keeps a spread of colour views during a scan, then paints each mesh vertex from the views that saw it most
/// squarely (sharpest, least foreshortened), skipping views where something else was in front of it.
struct VertexColorizer {
    private struct View {
        var worldToCamera: simd_float4x4
        var eye: SIMD3<Float>
        var forward: SIMD3<Float>
        var k: CameraIntrinsics
        var width: Int
        var height: Int
        var depth: [Float16]
        var rgb: [UInt8]
    }

    private var views: [View] = []
    private static let maxViews = 80
    /// A new view is kept once the camera has turned this far from every kept one.
    private static let spacing = cos(6 * Float.pi / 180)
    /// Faster than this (rad/s) the colour frame is likely motion-blurred.
    private static let maxSpeed: Float = 1.2
    /// Colour for surface no view saw directly (cut faces, the flat back of a relief).
    static let unseen = SIMD3<Float>(repeating: 0.7)

    /// `level` is the frame's depth pyramid level the colour is sampled at (masked pixels hold 0).
    mutating func offer(cameraToWorld pose: simd_float4x4, level: FramePyramid.Level, speed: Float, color: ColorSampler) {
        guard views.count < Self.maxViews, speed < Self.maxSpeed else { return }
        let eye = SIMD3(pose.columns.3.x, pose.columns.3.y, pose.columns.3.z)
        let forward = SIMD3(pose.columns.2.x, pose.columns.2.y, pose.columns.2.z)
        if views.contains(where: { dot($0.forward, forward) > Self.spacing && distance($0.eye, eye) < 0.02 }) { return }
        let w = level.width, h = level.height
        guard let rgb = color(w, h), rgb.count == w * h * 3 else { return }
        let depth = level.depth.contents().bindMemory(to: Float.self, capacity: w * h)
        views.append(View(worldToCamera: pose.inverse, eye: eye, forward: forward, k: level.intrinsics, width: w,
                          height: h, depth: (0..<(w * h)).map { Float16(depth[$0]) }, rgb: rgb))
    }

    func paint(_ mesh: Mesh) -> Mesh {
        guard !views.isEmpty, !mesh.vertices.isEmpty else { return mesh }
        var normals = [SIMD3<Float>](repeating: .zero, count: mesh.vertices.count)
        for t in stride(from: 0, to: mesh.indices.count, by: 3) {
            let i = (Int(mesh.indices[t]), Int(mesh.indices[t + 1]), Int(mesh.indices[t + 2]))
            let n = cross(mesh.vertices[i.1] - mesh.vertices[i.0], mesh.vertices[i.2] - mesh.vertices[i.0])
            normals[i.0] += n
            normals[i.1] += n
            normals[i.2] += n
        }
        var colors = [SIMD3<Float>](repeating: Self.unseen, count: mesh.vertices.count)
        let chunk = 4096
        colors.withUnsafeMutableBufferPointer { out in
            DispatchQueue.concurrentPerform(iterations: (mesh.vertices.count + chunk - 1) / chunk) { c in
                for i in (c * chunk)..<min((c + 1) * chunk, mesh.vertices.count) {
                    out[i] = color(at: mesh.vertices[i], normal: normals[i])
                }
            }
        }
        var painted = mesh
        painted.colors = colors
        return painted
    }

    private func color(at p: SIMD3<Float>, normal: SIMD3<Float>) -> SIMD3<Float> {
        guard length(normal) > 0 else { return Self.unseen }
        let n = normalize(normal)
        var sum = SIMD3<Float>(repeating: 0)
        var total: Float = 0
        for view in views {
            let facing = dot(n, normalize(view.eye - p))
            guard facing > 0.3 else { continue }
            let c = view.worldToCamera * SIMD4(p, 1)
            guard c.z > 0 else { continue }
            let x = view.k.fx * c.x / c.z + view.k.cx, y = view.k.fy * c.y / c.z + view.k.cy
            let u = Int(x.rounded()), v = Int(y.rounded())
            guard u >= 1, v >= 1, u + 1 < view.width, v + 1 < view.height else { continue }
            // Visible here, and not at a silhouette, where colour and depth registration bleed the background in.
            let d = Float(view.depth[v * view.width + u])
            guard d > 0, abs(d - c.z) < 0.005 else { continue }
            var edge = false
            for dv in -1...1 { for du in -1...1 where abs(Float(view.depth[(v + dv) * view.width + u + du]) - d) > 0.01 { edge = true } }
            guard !edge else { continue }
            let weight = facing * facing * facing * facing
            sum += sample(view, x, y) * weight
            total += weight
        }
        return total > 0 ? sum / total : Self.unseen
    }

    private func sample(_ view: View, _ x: Float, _ y: Float) -> SIMD3<Float> {
        let x0 = max(0, min(Int(x.rounded(.down)), view.width - 2)), y0 = max(0, min(Int(y.rounded(.down)), view.height - 2))
        let fx = min(max(x - Float(x0), 0), 1), fy = min(max(y - Float(y0), 0), 1)
        func rgb(_ u: Int, _ v: Int) -> SIMD3<Float> {
            let i = (v * view.width + u) * 3
            return SIMD3(Float(view.rgb[i]), Float(view.rgb[i + 1]), Float(view.rgb[i + 2])) / 255
        }
        let top = rgb(x0, y0) * (1 - fx) + rgb(x0 + 1, y0) * fx
        let bottom = rgb(x0, y0 + 1) * (1 - fx) + rgb(x0 + 1, y0 + 1) * fx
        return top * (1 - fy) + bottom * fy
    }

    /// Box-filters a BGRA camera frame down to `width` × `height` RGB.
    static func rgb(_ buffer: CVPixelBuffer, width: Int, height: Int) -> [UInt8]? {
        guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA else { return nil }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        let sw = CVPixelBufferGetWidth(buffer), sh = CVPixelBufferGetHeight(buffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        var out = [UInt8](repeating: 0, count: width * height * 3)
        for v in 0..<height {
            let y0 = v * sh / height, y1 = max(y0 + 1, (v + 1) * sh / height)
            for u in 0..<width {
                let x0 = u * sw / width, x1 = max(x0 + 1, (u + 1) * sw / width)
                var sum = SIMD3<UInt32>(repeating: 0)
                for y in y0..<y1 {
                    for x in x0..<x1 {
                        let p = bytes + y * rowBytes + x * 4
                        sum &+= SIMD3(UInt32(p[2]), UInt32(p[1]), UInt32(p[0]))
                    }
                }
                let mean = sum / UInt32((y1 - y0) * (x1 - x0))
                out[(v * width + u) * 3] = UInt8(mean.x)
                out[(v * width + u) * 3 + 1] = UInt8(mean.y)
                out[(v * width + u) * 3 + 2] = UInt8(mean.z)
            }
        }
        return out
    }
}
