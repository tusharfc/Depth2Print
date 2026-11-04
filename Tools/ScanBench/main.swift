// ScanBench: runs DepthEngine on a Mac, no iPhone needed. Not part of the app (outside both Xcode targets).
//   Tools/ScanBench/run.sh                 synthetic scenarios with known ground truth
//   Tools/ScanBench/run.sh replay <dir>    a raw capture recorded on an iPhone ("Record raw capture" in the app)
// Env: ONLY=<scenario>  TRACE=1 (per-frame log)
import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import Metal
import simd

setvbuf(stdout, nil, _IOLBF, 0)

/// Stand-in for the capture type, which needs AVFoundation's iOS-only camera classes.
struct DepthFrame {
    let depth: CVPixelBuffer
    let color: CVPixelBuffer?
    let intrinsics: simd_float3x3
    let timestamp: CMTime
    let gravity: SIMD3<Float>?
}

let device = MTLCreateSystemDefaultDevice()!
let shaderDir = URL(fileURLWithPath: "DepthEngine/Shaders")
let shaderSource = try FileManager.default.contentsOfDirectory(at: shaderDir, includingPropertiesForKeys: nil)
    .filter { $0.pathExtension == "metal" }
    .map { try String(contentsOf: $0, encoding: .utf8) }
    .joined(separator: "\n")
let compileOptions = MTLCompileOptions()
compileOptions.mathMode = .fast   // matches MTL_FAST_MATH = YES in the project
let library = try device.makeLibrary(source: shaderSource, options: compileOptions)
let environment = ProcessInfo.processInfo.environment
let tracing = environment["TRACE"] != nil

/// ICP_ITERS=fine,mid,coarse overrides the per-level iteration counts, for tuning.
func profile(_ mode: ScanMode) -> ScanProfile {
    var profile = ScanProfile(mode)
    if let override = environment["ICP_ITERS"] {
        profile.icp.iterations = override.split(separator: ",").compactMap { Int($0) }
    }
    if let trunc = environment["TRUNC"].flatMap(Float.init) { profile.volume.truncationVoxels = trunc }
    return profile
}

func depthTexture(_ depth: [Float], width: Int, height: Int) -> MTLTexture {
    let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: width, height: height, mipmapped: false)
    let texture = device.makeTexture(descriptor: desc)!
    depth.withUnsafeBytes {
        texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                        withBytes: $0.baseAddress!, bytesPerRow: width * 4)
    }
    return texture
}

func meshReport(_ mesh: Mesh) -> String {
    var directed: [SIMD2<UInt32>: Int] = [:]
    for t in stride(from: 0, to: mesh.indices.count, by: 3) {
        for i in 0..<3 { directed[SIMD2(mesh.indices[t + i], mesh.indices[t + (i + 1) % 3]), default: 0] += 1 }
    }
    let open = directed.keys.filter { directed[SIMD2($0.y, $0.x)] == nil }.count + directed.values.filter { $0 > 1 }.count
    var volume: Float = 0
    for t in stride(from: 0, to: mesh.indices.count, by: 3) {
        let a = mesh.vertices[Int(mesh.indices[t])], b = mesh.vertices[Int(mesh.indices[t + 1])]
        volume += dot(a, cross(b, mesh.vertices[Int(mesh.indices[t + 2])])) / 6
    }
    return String(format: "%d tris, open/non-manifold edges %d, volume %.1f cm³", mesh.triangleCount, open, volume * 1e6)
}

func percentiles(_ values: [Float]) -> String {
    guard !values.isEmpty else { return "n/a" }
    let s = values.sorted()
    return String(format: "median %.2f mm, 90%% %.2f mm, 99%% %.2f mm", s[s.count / 2], s[s.count * 9 / 10], s[s.count * 99 / 100])
}

// MARK: - Synthetic scenes (local frames are y-up)

let W = 640, H = 480
let K = CameraIntrinsics(fx: 450, fy: 450, cx: 319.5, cy: 239.5)

enum Shape {
    case box(SIMD3<Float>, SIMD3<Float>)
    case ellipsoid(SIMD3<Float>, SIMD3<Float>)
    case cylinder(radius: Float, y0: Float, y1: Float)
    case plane(SIMD3<Float>, Float)

    static func sphere(_ c: SIMD3<Float>, _ r: Float) -> Shape { .ellipsoid(c, SIMD3(repeating: r)) }

    var isSupport: Bool {
        if case .plane = self { return true }
        if case .cylinder(let r, _, _) = self, r >= 0.1 { return true }
        return false
    }

    func hit(_ o: SIMD3<Float>, _ d: SIMD3<Float>) -> Float? {
        switch self {
        case let .box(lo, hi):
            let t0 = (lo - o) / d, t1 = (hi - o) / d
            let near = simd_reduce_max(simd_min(t0, t1)), far = simd_reduce_min(simd_max(t0, t1))
            return near <= far && near > 0 ? near : nil
        case let .ellipsoid(c, r):
            let oc = (o - c) / r, dd = d / r
            let a = dot(dd, dd), b = dot(oc, dd), disc = b * b - a * (dot(oc, oc) - 1)
            guard disc >= 0 else { return nil }
            let t = (-b - disc.squareRoot()) / a
            return t > 0 ? t : nil
        case let .cylinder(r, y0, y1):
            var best: Float?
            let a = d.x * d.x + d.z * d.z, b = o.x * d.x + o.z * d.z, c = o.x * o.x + o.z * o.z - r * r
            if a > 0, b * b - a * c >= 0 {
                let t = (-b - (b * b - a * c).squareRoot()) / a, y = o.y + t * d.y
                if t > 0 && y >= y0 && y <= y1 { best = t }
            }
            for y in [y0, y1] where d.y != 0 {
                let t = (y - o.y) / d.y, p = o + d * t
                if t > 0 && p.x * p.x + p.z * p.z <= r * r { best = min(best ?? t, t) }
            }
            return best
        case let .plane(n, offset):
            let t = -(dot(n, o) + offset) / dot(n, d)
            return t.isFinite && t > 0 ? t : nil
        }
    }

    func sdf(_ p: SIMD3<Float>) -> Float {
        switch self {
        case let .box(lo, hi):
            let q = abs(p - (lo + hi) / 2) - (hi - lo) / 2
            return length(simd_max(q, .zero)) + min(simd_reduce_max(q), 0)
        case let .ellipsoid(c, r):
            return (length((p - c) / r) - 1) * r.min()
        case let .cylinder(r, y0, y1):
            let q = SIMD2<Float>(length(SIMD2(p.x, p.z)) - r, max(y0 - p.y, p.y - y1))
            return length(simd_max(q, .zero)) + min(max(q.x, q.y), 0)
        case let .plane(n, offset):
            return dot(n, p) + offset
        }
    }
}

struct Group {
    var shapes: [Shape]
    var toCamera: simd_float4x4
}

var rng = SystemRandomNumberGenerator()
func gaussian() -> Float {
    let u1 = Float.random(in: Float.ulpOfOne..<1, using: &rng), u2 = Float.random(in: 0..<1, using: &rng)
    return (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
}

/// Depth along unit-z camera rays, with 1 mm noise at 40 cm growing with distance², and 2% dropouts.
func render(_ groups: [Group], noise: Bool = true) -> [Float] {
    let locals = groups.map { $0.toCamera.inverse }
    var depth = [Float](repeating: .nan, count: W * H)
    for v in 0..<H {
        for u in 0..<W {
            let ray = SIMD4<Float>((Float(u) - K.cx) / K.fx, (Float(v) - K.cy) / K.fy, 1, 0)
            var t = Float.infinity
            for (group, inverse) in zip(groups, locals) {
                let o = SIMD3(inverse.columns.3.x, inverse.columns.3.y, inverse.columns.3.z)
                let d4 = inverse * ray
                for shape in group.shapes { if let h = shape.hit(o, SIMD3(d4.x, d4.y, d4.z)) { t = min(t, h) } }
            }
            guard t.isFinite else { continue }
            if noise {
                guard Float.random(in: 0..<1, using: &rng) > 0.02 else { continue }
                t += gaussian() * 0.001 * (t / 0.4) * (t / 0.4)
            }
            depth[v * W + u] = t
        }
    }
    return depth
}

/// Smooth colour pattern painted on every subject, in its local frame (3 cm period per channel).
func pattern(_ p: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3(0.5 + 0.4 * sin(2 * .pi * p.x / 0.03), 0.5 + 0.4 * sin(2 * .pi * p.y / 0.03), 0.5 + 0.4 * sin(2 * .pi * p.z / 0.03))
}

/// Colour image at width × height (a 2x-downsampled grid of the depth camera): the pattern on the subject, grey elsewhere.
func renderColor(_ groups: [Group], width: Int, height: Int) -> [UInt8] {
    let locals = groups.map { $0.toCamera.inverse }
    let scale = Float(W) / Float(width)
    var rgb = [UInt8](repeating: 0, count: width * height * 3)
    for v in 0..<height {
        for u in 0..<width {
            let ray = SIMD4<Float>(((Float(u) + 0.5) * scale - 0.5 - K.cx) / K.fx, ((Float(v) + 0.5) * scale - 0.5 - K.cy) / K.fy, 1, 0)
            var t = Float.infinity, hitGroup = -1
            for (g, (group, inverse)) in zip(groups, locals).enumerated() {
                let o = SIMD3(inverse.columns.3.x, inverse.columns.3.y, inverse.columns.3.z)
                let d4 = inverse * ray
                for shape in group.shapes {
                    if let h = shape.hit(o, SIMD3(d4.x, d4.y, d4.z)), h < t { t = h; hitGroup = g }
                }
            }
            var c = SIMD3<Float>(repeating: 0.3)
            if hitGroup == 0 {
                let p = locals[0] * SIMD4(ray.x * t, ray.y * t, t, 1)
                c = pattern(SIMD3(p.x, p.y, p.z))
            }
            let i = (v * width + u) * 3
            rgb[i] = UInt8(c.x * 255); rgb[i + 1] = UInt8(c.y * 255); rgb[i + 2] = UInt8(c.z * 255)
        }
    }
    return rgb
}

func rotY(_ angle: Float) -> simd_float4x4 { ICP.rigid(rotation: SIMD3(0, angle, 0), translation: .zero) }
func translate(_ t: SIMD3<Float>) -> simd_float4x4 { ICP.rigid(rotation: .zero, translation: t) }

/// A y-up local frame placed in camera space.
func frame(origin: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
    let y = normalize(up), x = normalize(cross(y, SIMD3(0, 0, -1)))
    return simd_float4x4(columns: (SIMD4(x, 0), SIMD4(y, 0), SIMD4(cross(x, y), 0), SIMD4(origin, 1)))
}

struct Region {
    var name: String
    var contains: (SIMD3<Float>) -> Bool
}

/// `scene(k)` gives the frame's groups (first = the subject) and the subject's local→camera transform.
func runScenario(_ name: String, mode: ScanMode, frames: Int, scene: (Int) -> (groups: [Group], subject: simd_float4x4),
                 truth: @escaping (SIMD3<Float>) -> Float, evaluate: @escaping (SIMD3<Float>) -> Bool = { _ in true },
                 details: [Region] = []) throws {
    let tracker = try FrameTracker(device: device, library: library, profile: profile(mode))
    var first = matrix_identity_float4x4
    var maxPoseError: Float = 0
    var seconds = 0.0
    var previewIoU: Double?
    var last: TrackingStatus?
    for k in 0...frames {
        let s = scene(k)
        if k == 0 { first = s.subject }
        let depth = render(s.groups)
        let up = s.subject * SIMD4<Float>(0, 1, 0, 0)
        let gravity = -SIMD3(up.x, up.y, up.z) + SIMD3(gaussian(), gaussian(), gaussian()) * 0.01
        let start = Date()
        let status = try tracker.track(depth: depthTexture(depth, width: W, height: H), intrinsics: K,
                                       time: Double(k) / 30, gravity: gravity,
                                       color: { renderColor(s.groups, width: $0, height: $1) })
        seconds += Date().timeIntervalSince(start)
        last = status
        let truthPose = first * s.subject.inverse
        if status.isTracking {
            maxPoseError = max(maxPoseError, length(FrameTracker.rotationVector(status.pose.inverse * truthPose)) * 180 / .pi)
        }
        if tracing { print(String(format: "  k=%d %@ rot=%.1f rmse=%.2fmm %@", k, status.isTracking ? "ok  " : "LOST",
                                  status.rotationDegrees, status.rmse * 1000, status.failureReason ?? "")) }
        if k >= frames / 3, previewIoU == nil, let image = status.preview {
            previewIoU = overlap(image, subject: s.groups[0], full: depth)
        }
    }

    print("== \(name) (\(mode))")
    if let region = tracker.detectedHead.region {
        // Ground truth in the subject's frame: head center at the origin, crown at y = 0.11, chin at y = -0.11.
        let c = first.inverse * SIMD4(region.x, region.y, region.z, 1)
        print(String(format: "  head region: center (%.0f, %.0f, %.0f) mm in head frame, radius %.0f mm",
                     c.x * 1000, c.y * 1000, c.z * 1000, region.w * 1000))
        if let neck = tracker.detectedHead.neck {
            let n = first.inverse * SIMD4(neck.normal, 0)
            let p = first.inverse * SIMD4(neck.normal * -neck.offset, 1)
            print(String(format: "  neck cut: normal (%.2f, %.2f, %.2f), at head-frame y = %.0f mm",
                         n.x, n.y, n.z, p.y * 1000))
        }
    }
    if let s = last {
        print(String(format: "  tracking: rotation %.1f°, max pose error %.2f°, lost %d, %.1f ms/frame on this Mac",
                     s.rotationDegrees, maxPoseError, s.framesLost, seconds / Double(frames + 1) * 1000))
    }
    let world: Mesh
    do { world = try tracker.worldMesh(); _ = try tracker.extractMesh() } catch {
        print("  mesh: \(error)")
        return
    }
    let local = world.vertices.map { v -> SIMD3<Float> in let p = first.inverse * SIMD4(v, 1); return SIMD3(p.x, p.y, p.z) }
    print("  mesh: \(meshReport(world))")
    print("  surface error: \(percentiles(local.filter(evaluate).map { abs(truth($0)) * 1000 }))")
    for region in details {
        let inside = local.filter(region.contains)
        print("  \(region.name): \(inside.count) vertices, error \(percentiles(inside.map { abs(truth($0)) * 1000 }))")
    }
    if let colors = world.colors {
        // Largest channel error, in 0-255 steps, on the true surface; unseen (cut) faces are counted separately.
        var errors: [Float] = []
        var unseen = 0
        for (p, c) in zip(local, colors) where evaluate(p) && abs(truth(p)) < 0.001 {
            if c == VertexColorizer.unseen { unseen += 1; continue }
            errors.append(simd_reduce_max(abs(c - pattern(p))) * 255)
        }
        let s = errors.sorted()
        if !s.isEmpty {
            print(String(format: "  colour error (of 255): median %.1f, 90%% %.1f, 99%% %.1f; %d vertices uncoloured",
                         s[s.count / 2], s[s.count * 9 / 10], s[s.count * 99 / 100], unseen))
        }
    }
    if let previewIoU { print(String(format: "  live preview overlap with the real subject: %.0f%%", previewIoU * 100)) }
}

/// Fraction of pixels where the preview and the visible subject agree (intersection over union).
func overlap(_ image: CGImage, subject: Group, full: [Float]) -> Double {
    let alone = render([Group(shapes: subject.shapes.filter { !$0.isSupport }, toCamera: subject.toCamera)], noise: false)
    let w = image.width, h = image.height
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    var both = 0, either = 0
    for v in 0..<h {
        for u in 0..<w {
            let i = (2 * v) * W + 2 * u
            let visible = alone[i].isFinite && full[i].isFinite && abs(alone[i] - full[i]) < 0.01
            let shown = rgba[(v * w + u) * 4 + 3] > 0
            if visible && shown { both += 1 }
            if visible || shown { either += 1 }
        }
    }
    return Double(both) / Double(max(either, 1))
}

func runSynthetic() throws {
    let only = environment["ONLY"]
    func enabled(_ name: String) -> Bool { only == nil || only == name }

    let turntableBase = frame(origin: SIMD3(0, 0.06, 0.42), up: SIMD3(0, -1, -0.35))
    let object: [Shape] = [
        .box(SIMD3(-0.04, 0, -0.025), SIMD3(0.04, 0.06, 0.025)),
        .sphere(SIMD3(-0.02, 0.085, 0), 0.03),
        .box(SIMD3(0.04, 0.02, -0.01), SIMD3(0.07, 0.04, 0.01)),
    ]
    let platter: [Shape] = [.cylinder(radius: 0.1, y0: -0.015, y1: 0), .plane(SIMD3(0, 1, 0), 0.015)]
    // Meshes are cut 3 mm above the support plane.
    func clipped(_ shapes: [Shape]) -> (SIMD3<Float>) -> Float { { p in max(shapes.map { $0.sdf(p) }.min()!, 0.003 - p.y) } }
    let wall = Group(shapes: [.plane(SIMD3(0, 0, -1), 1.0)], toCamera: matrix_identity_float4x4)

    if enabled("turntable") {
        try runScenario("turntable", mode: .turntable, frames: 360, scene: { k in
            let m = turntableBase * rotY(Float(k) * .pi / 180)
            return ([Group(shapes: object + platter, toCamera: m), wall], m)
        }, truth: clipped(object))
    }
    if enabled("vase") {
        let vase: [Shape] = [.cylinder(radius: 0.04, y0: 0, y1: 0.12)]
        try runScenario("vase", mode: .turntable, frames: 360, scene: { k in
            let m = turntableBase * rotY(Float(k) * .pi / 180)
            return ([Group(shapes: vase + platter, toCamera: m), wall], m)
        }, truth: clipped(vase))
    }
    if enabled("handheld") {
        try runScenario("handheld", mode: .handheld, frames: 360, scene: { k in
            let jitter = ICP.rigid(rotation: SIMD3(gaussian(), gaussian(), gaussian()) * 0.2 * .pi / 180,
                                   translation: SIMD3(gaussian(), gaussian(), gaussian()) * 0.0015)
            let m = jitter * turntableBase * rotY(Float(k) * .pi / 180)
            return ([Group(shapes: object + [.plane(SIMD3(0, 1, 0), 0)], toCamera: m)], m)
        }, truth: clipped(object))
    }
    if enabled("body") {
        let person: [Shape] = [
            .ellipsoid(.zero, SIMD3(0.18, 0.3, 0.11)), .cylinder(radius: 0.05, y0: 0.25, y1: 0.38),
            .sphere(SIMD3(0, 0.45, 0.01), 0.1),
        ]
        let standing = frame(origin: SIMD3(0, 0, 0.9), up: SIMD3(0, -1, 0))
        try runScenario("body", mode: .body, frames: 360, scene: { k in
            let jitter = ICP.rigid(rotation: SIMD3(gaussian(), gaussian(), gaussian()) * 0.2 * .pi / 180,
                                   translation: SIMD3(gaussian(), gaussian(), gaussian()) * 0.002)
            let m = jitter * standing * rotY(Float(k) * .pi / 180)
            return ([Group(shapes: person, toCamera: m)], m)
        }, truth: { p in person.map { $0.sdf(p) }.min()! })
    }
    if enabled("face") || enabled("face-low") {
        // Head turns ±60° in front of a still phone; shoulders stay put. Cheek bumps 6 mm wide and 4 mm high
        // measure how much fine detail survives.
        let bumps = [SIMD3<Float>(0.04, -0.02, 0.0745), SIMD3<Float>(-0.04, -0.02, 0.0745)]
        let nose = SIMD3<Float>(0, -0.01, 0.09)
        let head: [Shape] = [
            .ellipsoid(.zero, SIMD3(0.075, 0.11, 0.095)),
            .ellipsoid(nose, SIMD3(0.012, 0.025, 0.022)),
            .cylinder(radius: 0.05, y0: -0.2, y1: -0.06),
        ] + bumps.map { .sphere($0, 0.006) }
        let neck = SIMD3<Float>(0, 0, -0.02)
        let details = [
            Region(name: "nose", contains: { distance($0, nose + SIMD3(0, 0, 0.02)) < 0.02 }),
            Region(name: "cheek bumps", contains: { p in bumps.contains { distance(p, $0) < 0.01 } }),
            Region(name: "chin", contains: { p in p.y < -0.085 && p.y > -0.11 && p.z > 0.03 && abs(p.x) < 0.03 }),
        ]
        // "face": phone level with the eyes. "face-low": portrait phone held 8 cm below the eyes and tilted up, so
        // the chest is in view and nearer to it than the nose (the case that used to scan the shirt).
        let setups: [(String, SIMD3<Float>, Float, Float, Float)] = [("face", SIMD3(0, 0, 0.4), 0, 0, 0.1),
                                                                    ("face-low", SIMD3(0, -0.08, 0.35), 0.15, .pi / 2, 0.13)]
        for (name, origin, tilt, roll, chest) in setups where enabled(name) {
            let torso: [Shape] = [.box(SIMD3(-0.18, -0.45, -0.1), SIMD3(0.18, -0.17, chest))]
            let level = simd_float4x4(columns: (SIMD4(1, 0, 0, 0), SIMD4(0, -1, 0, 0), SIMD4(0, 0, -1, 0), SIMD4(origin, 1)))
            // Tilting the phone up rotates the scene about the camera's x axis; portrait rolls it about z.
            let headBase = ICP.rigid(rotation: SIMD3(0, 0, roll), translation: .zero)
                * ICP.rigid(rotation: SIMD3(-tilt, 0, 0), translation: .zero) * level
            try runScenario(name, mode: .face, frames: 240, scene: { k in
                let angle = 60 * Float.pi / 180 * sin(2 * .pi * Float(k) / 240)
                let m = headBase * translate(neck) * rotY(angle) * translate(-neck)
                return ([Group(shapes: head, toCamera: m), Group(shapes: torso, toCamera: headBase)], m)
            }, truth: { p in head.map { $0.sdf(p) }.min()! },
               evaluate: { p in p.z > 0.04 && p.y > -0.06 },   // what the camera saw, not the flat back
               details: details)
        }
    }
}

if CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "replay" {
    try replay(URL(fileURLWithPath: CommandLine.arguments[2]))
} else {
    try runSynthetic()
}
