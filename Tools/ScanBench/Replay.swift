import Foundation
import Metal
import simd

/// Replays a raw capture from the app (format: DepthEngine/Capture/CaptureRecorder.swift) through the same engine.
/// DEVICE_FRAMES=1 feeds only the frames the phone managed to process, to reproduce its tracking exactly.
func replay(_ directory: URL) throws {
    let header = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("capture.json")))
        as! [String: Any]
    let mode = ScanMode(rawValue: header["mode"] as! String)!
    let width = header["width"] as! Int, height = header["height"] as! Int
    let deviceFromCamera = (header["deviceFromCamera"] as? [Double]).map { v -> simd_float3x3 in
        let f = v.map(Float.init)
        return simd_float3x3(SIMD3(f[0], f[1], f[2]), SIMD3(f[3], f[4], f[5]), SIMD3(f[6], f[7], f[8]))
    }

    struct Frame { var time: Double; var k: CameraIntrinsics; var gravity: SIMD3<Float>?; var depth: Data }
    let blob = try Data(contentsOf: directory.appendingPathComponent("frames.bin"))
    var frames: [Frame] = []
    var offset = 0
    blob.withUnsafeBytes { bytes in
        func float(_ at: Int) -> Float { Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: at, as: UInt32.self))) }
        while offset + 40 <= bytes.count {
            let time = Double(bitPattern: UInt64(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt64.self)))
            let k = CameraIntrinsics(fx: float(offset + 8), fy: float(offset + 12), cx: float(offset + 16), cy: float(offset + 20))
            let g = SIMD3(float(offset + 24), float(offset + 28), float(offset + 32))
            let count = Int(UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: offset + 36, as: UInt32.self)))
            guard offset + 40 + count <= bytes.count else { break }
            frames.append(Frame(time: time, k: k, gravity: g.x.isNaN ? nil : g,
                                depth: Data(bytes[(offset + 40)..<(offset + 40 + count)])))
            offset += 40 + count
        }
    }

    // What happened on the phone.
    let logLines = (try? String(contentsOf: directory.appendingPathComponent("log.csv"), encoding: .utf8))?
        .split(separator: "\n").dropFirst().map { $0.split(separator: ",", omittingEmptySubsequences: false) } ?? []
    let processedTimes = Set(logLines.compactMap { Double($0[0]) }.map { ($0 * 1000).rounded() })
    let deviceMs = logLines.compactMap { Double($0[2]) }.sorted()
    let deviceLost = logLines.filter { $0[1] == "0" }
    print("== capture \(directory.lastPathComponent): mode \(mode), \(width)x\(height), \(frames.count) frames over "
          + String(format: "%.1f s", (frames.last?.time ?? 0) - (frames.first?.time ?? 0))
          + ", gravity \(deviceFromCamera == nil ? "unmapped" : "mapped")")
    if !deviceMs.isEmpty {
        let span = (frames.last?.time ?? 0) - (frames.first?.time ?? 0)
        print(String(format: "  on the phone: processed %d frames (%.1f fps), %.0f ms median / %.0f ms 90%%, lost %d",
                     deviceMs.count, Double(deviceMs.count) / max(span, 0.001), deviceMs[deviceMs.count / 2],
                     deviceMs[deviceMs.count * 9 / 10], deviceLost.count))
        printReasons(deviceLost.map { String($0[4]) })
        printStages(logLines.compactMap { $0.count > 5 ? String($0[5]) : nil }, label: "phone")
    }

    // The same engine on this Mac.
    let tracker = try FrameTracker(device: device, library: library, profile: profile(mode), deviceFromCamera: deviceFromCamera)
    let onlyDeviceFrames = environment["DEVICE_FRAMES"] != nil
    // FRAMES=first,count feeds only that stretch (indices into the recording).
    let window = environment["FRAMES"].map { s -> Range<Int> in
        let p = s.split(separator: ",").compactMap { Int($0) }
        return p[0]..<(p[0] + p[1])
    }
    var reasons: [String] = []
    var stageLines: [String] = []
    var samples: [(time: Double, pose: simd_float4x4, k: CameraIntrinsics, depth: [Float], heading: Float)] = []
    // REFERENCE=first,count: average those still frames' raw depth into a low-noise reference for the fused mesh.
    let referenceRange = environment["REFERENCE"].map { s -> Range<Int> in
        let p = s.split(separator: ",").compactMap { Int($0) }
        return p[0]..<(p[0] + p[1])
    }
    var referenceFrames: [(pose: simd_float4x4, k: CameraIntrinsics, depth: [Float])] = []
    var fed = 0
    var last: TrackingStatus?
    for (index, frame) in frames.enumerated() where !onlyDeviceFrames || processedTimes.contains((frame.time * 1000).rounded()) {
        if let window, !window.contains(index) { continue }
        // HOLDOUT=first,count tracks those frames without fusing them (pair with REFERENCE for a fair test).
        if let holdout = environment["HOLDOUT"]?.split(separator: ",").compactMap({ Int($0) }) {
            tracker.fusesFrames = !(holdout[0]..<(holdout[0] + holdout[1])).contains(index)
        }
        guard var raw = try? (frame.depth as NSData).decompressed(using: .lzfse) as Data else { continue }
        if let k = environment["DISTORTION"].flatMap(Float.init) {
            raw = undistort(raw, width: width, height: height, center: SIMD2(frame.k.cx, frame.k.cy), strength: k)
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: width, height: height, mipmapped: false)
        let texture = device.makeTexture(descriptor: desc)!
        raw.withUnsafeBytes { texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                                              withBytes: $0.baseAddress!, bytesPerRow: width * 2) }
        let gravity = deviceFromCamera.flatMap { m in frame.gravity.map { m.transpose * $0 } }
        let status = try tracker.track(depth: texture, intrinsics: frame.k, time: frame.time, gravity: gravity)
        fed += 1
        last = status
        if let reason = status.failureReason { reasons.append(reason) }
        stageLines.append(status.stageTimings)
        if environment["CONSISTENCY"] != nil, status.isTracking, fed % 15 == 0 {
            let depth: [Float] = raw.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)).map(Float.init) }
            samples.append((frame.time - frames[0].time, status.pose, frame.k, depth, status.headingDegrees))
        }
        if let referenceRange, referenceRange.contains(fed - 1), status.isTracking {
            let depth: [Float] = raw.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)).map(Float.init) }
            referenceFrames.append((status.pose, frame.k, depth))
        }
        if tracing {
            print(String(format: "  t=%.3f %@ rot=%.1f rmse=%.2fmm %@", frame.time - frames[0].time,
                         status.isTracking ? "ok  " : "LOST", status.rotationDegrees, status.rmse * 1000, status.failureReason ?? ""))
        }
    }
    if let last {
        print(String(format: "  replayed here: fed %d frames, lost %d, rotation %.1f°%@", fed, last.framesLost,
                     last.rotationDegrees, onlyDeviceFrames ? " (phone's frames only)" : " (every frame)"))
    }
    printReasons(reasons)
    printStages(stageLines, label: "this Mac")
    if let region = tracker.detectedHead.region, let k = frames.first?.k {
        // World space is the first camera's, so this is the head's position in the first image.
        print(String(format: "  head region: pixel (%.0f, %.0f) of %dx%d, %.0f mm away, radius %.0f mm",
                     k.fx * region.x / region.z + k.cx, k.fy * region.y / region.z + k.cy, width, height,
                     region.z * 1000, region.w * 1000))
    }

    let mesh = try tracker.extractMesh()
    print("  mesh: \(meshReport(mesh))")
    if !samples.isEmpty { try consistency(samples, world: tracker.worldMesh(), range: profile(mode).icp.depthRange) }
    if !referenceFrames.isEmpty { try compareToReference(referenceFrames, world: tracker.worldMesh(), range: profile(mode).icp.depthRange) }
    let out = directory.appendingPathComponent("replay.obj")
    try MeshExporter.write(mesh, format: .obj, to: out)
    print("  wrote \(out.path)")
}

/// Resamples a Float16 depth map to a pinhole image, assuming Apple's radial model with magnification
/// strength · (r / rMax)²: each undistorted pixel p comes from center + (p - center) · (1 + magnification).
private func undistort(_ raw: Data, width: Int, height: Int, center: SIMD2<Float>, strength: Float) -> Data {
    let src: [Float] = raw.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)).map(Float.init) }
    let corners = [SIMD2<Float>(0, 0), SIMD2(Float(width), 0), SIMD2(0, Float(height)), SIMD2(Float(width), Float(height))]
    let rMax = corners.map { distance($0, center) }.max()!
    var out = [Float16](repeating: .nan, count: width * height)
    for y in 0..<height {
        for x in 0..<width {
            let d = SIMD2<Float>(Float(x), Float(y)) - center
            let r = length(d) / rMax
            let s = center + d * (1 + strength * r * r)
            let x0 = Int(s.x.rounded(.down)), y0 = Int(s.y.rounded(.down))
            guard x0 >= 0, y0 >= 0, x0 + 1 < width, y0 + 1 < height else { continue }
            let f = s - SIMD2(Float(x0), Float(y0))
            let z = [src[y0 * width + x0], src[y0 * width + x0 + 1], src[(y0 + 1) * width + x0], src[(y0 + 1) * width + x0 + 1]]
            let nearest = z[(f.y < 0.5 ? 0 : 2) + (f.x < 0.5 ? 0 : 1)]
            // Bilinear only within one surface; otherwise the nearest sample, so edges don't get bridged.
            if z.allSatisfy({ $0.isFinite && $0 > 0 }), z.max()! - z.min()! < 0.005 {
                out[y * width + x] = Float16((z[0] * (1 - f.x) + z[1] * f.x) * (1 - f.y) + (z[2] * (1 - f.x) + z[3] * f.x) * f.y)
            } else {
                out[y * width + x] = Float16(nearest)
            }
        }
    }
    return out.withUnsafeBufferPointer { Data(buffer: $0) }
}

/// How far each sampled frame's raw depth lands from the fused surface under its tracked pose. Consistent poses keep
/// every angle near sensor noise (~1 mm); a band of angles far off means those frames were misregistered.
private func consistency(_ samples: [(time: Double, pose: simd_float4x4, k: CameraIntrinsics, depth: [Float], heading: Float)],
                         world: Mesh, range: ClosedRange<Float>) throws {
    // Vertex hash on a 2 mm grid; nearest vertex within the 3x3x3 neighbourhood approximates surface distance.
    let cell: Float = 0.002
    func key(_ p: SIMD3<Float>) -> SIMD3<Int32> { SIMD3<Int32>((p / cell).rounded(.down)) }
    var grid: [SIMD3<Int32>: [SIMD3<Float>]] = [:]
    for v in world.vertices { grid[key(v), default: []].append(v) }
    print("  consistency of raw depth with the fused surface (only points within 6 mm of it count as 'on' the subject):")
    for s in samples {
        var distances: [Float] = []
        for y in stride(from: 0, to: 480, by: 4) {
            for x in stride(from: 0, to: 640, by: 4) {
                let z = s.depth[y * 640 + x]
                guard z.isFinite, range.contains(z) else { continue }
                let pc = SIMD4<Float>((Float(x) - s.k.cx) * z / s.k.fx, (Float(y) - s.k.cy) * z / s.k.fy, z, 1)
                let pw4 = s.pose * pc
                let pw = SIMD3(pw4.x, pw4.y, pw4.z)
                let c = key(pw)
                var best = Float.infinity
                for dz in -3...3 { for dy in -3...3 { for dx in -3...3 {
                    for v in grid[c &+ SIMD3(Int32(dx), Int32(dy), Int32(dz))] ?? [] { best = min(best, distance(v, pw)) }
                }}}
                if best < 0.006 { distances.append(best * 1000) }
            }
        }
        guard distances.count > 50 else { continue }
        distances.sort()
        print(String(format: "    t=%5.1fs heading %6.1f°: %5d points, median %.2f mm, 90%% %.2f mm", s.time, s.heading,
                     distances.count, distances[distances.count / 2], distances[distances.count * 9 / 10]))
    }
}

/// Exact distance from points to a mesh surface (closest point on triangles), via a 3 mm grid of triangles.
struct SurfaceDistance {
    private let mesh: Mesh
    private let cell: Float = 0.003
    private var grid: [SIMD3<Int32>: [Int]] = [:]

    init(_ mesh: Mesh) {
        self.mesh = mesh
        for t in 0..<mesh.triangleCount {
            let (a, b, c) = corners(t)
            let lo = key(simd_min(a, simd_min(b, c))), hi = key(simd_max(a, simd_max(b, c)))
            for z in lo.z...hi.z { for y in lo.y...hi.y { for x in lo.x...hi.x { grid[SIMD3(x, y, z), default: []].append(t) } } }
        }
    }

    private func key(_ p: SIMD3<Float>) -> SIMD3<Int32> { SIMD3<Int32>((p / cell).rounded(.down)) }

    private func corners(_ t: Int) -> (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>) {
        (mesh.vertices[Int(mesh.indices[3 * t])], mesh.vertices[Int(mesh.indices[3 * t + 1])],
         mesh.vertices[Int(mesh.indices[3 * t + 2])])
    }

    /// Nil if nothing is within ~3 mm.
    func callAsFunction(_ p: SIMD3<Float>) -> Float? {
        closestPoint(p).map { distance($0, p) }
    }

    func closestPoint(_ p: SIMD3<Float>) -> SIMD3<Float>? {
        let c = key(p)
        var best = Float.infinity
        var point: SIMD3<Float>?
        for dz in -1...1 { for dy in -1...1 { for dx in -1...1 {
            for t in grid[c &+ SIMD3(Int32(dx), Int32(dy), Int32(dz))] ?? [] {
                let (a, b, c) = corners(t)
                let q = closest(p, a, b, c)
                let d = distance(p, q)
                if d < best { best = d; point = q }
            }
        }}}
        return point
    }

    /// Closest point on triangle abc (Ericson, Real-Time Collision Detection 5.1.5).
    private func closest(_ p: SIMD3<Float>, _ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>) -> SIMD3<Float> {
        let ab = b - a, ac = c - a, ap = p - a
        let d1 = dot(ab, ap), d2 = dot(ac, ap)
        if d1 <= 0 && d2 <= 0 { return a }
        let bp = p - b, d3 = dot(ab, bp), d4 = dot(ac, bp)
        if d3 >= 0 && d4 <= d3 { return b }
        let vc = d1 * d4 - d3 * d2
        if vc <= 0 && d1 >= 0 && d3 <= 0 { return a + ab * (d1 / (d1 - d3)) }
        let cp = p - c, d5 = dot(ab, cp), d6 = dot(ac, cp)
        if d6 >= 0 && d5 <= d6 { return c }
        let vb = d5 * d2 - d1 * d6
        if vb <= 0 && d2 >= 0 && d6 <= 0 { return a + ac * (d2 / (d2 - d6)) }
        let va = d3 * d6 - d5 * d4
        if va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0 { return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))) }
        let denom = 1 / (va + vb + vc)
        return a + ab * (vb * denom) + ac * (vc * denom)
    }
}

/// Averages still raw frames pixel by pixel (cutting sensor noise by √N) and measures the fused mesh against that:
/// what is left is detail or accuracy the pipeline lost, not sensor noise.
private func compareToReference(_ frames: [(pose: simd_float4x4, k: CameraIntrinsics, depth: [Float])], world: Mesh,
                                range: ClosedRange<Float>) throws {
    let surface = SurfaceDistance(world)
    let middle = frames[frames.count / 2]
    var distances: [Float] = []
    var spread: [Float] = []
    var points: [SIMD3<Float>] = []
    var blocks: [Int: [Float]] = [:]   // signed error (mm, + = mesh nearer the camera) per 32x32 pixel block
    let eye = SIMD3(middle.pose.columns.3.x, middle.pose.columns.3.y, middle.pose.columns.3.z)
    // FACING=c keeps only pixels whose averaged surface faces the reference camera with cos > c.
    let minFacing = environment["FACING"].flatMap(Float.init) ?? -1
    var average = [Float](repeating: 0, count: 640 * 480)
    for i in average.indices {
        let values = frames.map { $0.depth[i] }.filter { $0.isFinite && range.contains($0) }
        if values.count == frames.count { average[i] = values.reduce(0, +) / Float(values.count) }
    }
    func facing(_ x: Int, _ y: Int) -> Float {
        guard x >= 3, y >= 3, x + 3 < 640, y + 3 < 480 else { return 0 }
        let k = middle.k
        func at(_ u: Int, _ v: Int) -> SIMD3<Float>? {
            let z = average[v * 640 + u]
            return z > 0 ? SIMD3((Float(u) - k.cx) * z / k.fx, (Float(v) - k.cy) * z / k.fy, z) : nil
        }
        guard let c = at(x, y), let l = at(x - 3, y), let r = at(x + 3, y), let u = at(x, y - 3), let d = at(x, y + 3)
        else { return 0 }
        let n = normalize(cross(d - u, r - l))
        return abs(dot(n, normalize(c)))
    }
    for y in stride(from: 0, to: 480, by: 2) {
        for x in stride(from: 0, to: 640, by: 2) {
            let values = frames.map { $0.depth[y * 640 + x] }.filter { $0.isFinite && range.contains($0) }
            guard values.count == frames.count else { continue }
            let z = values.reduce(0, +) / Float(values.count)
            guard values.allSatisfy({ abs($0 - z) < 0.005 }), facing(x, y) > minFacing else { continue }
            spread.append((values.map { ($0 - z) * ($0 - z) }.reduce(0, +) / Float(values.count)).squareRoot() * 1000)
            let k = middle.k
            let p4 = middle.pose * SIMD4<Float>((Float(x) - k.cx) * z / k.fx, (Float(y) - k.cy) * z / k.fy, z, 1)
            let p = SIMD3(p4.x, p4.y, p4.z)
            if let d = surface(p), d < 0.006 {
                distances.append(d * 1000); points.append(p)
                let q = surface.closestPoint(p)!
                let sign: Float = distance(q, eye) < distance(p, eye) ? 1 : -1
                blocks[(y / 32) * 20 + x / 32, default: []].append(sign * d * 1000)
            }
        }
    }
    print("  signed error map (mm ×10, + = mesh in front of raw; rows = image y / 32):")
    for by in 0..<15 {
        var line = "    "
        for bx in 0..<20 {
            guard var v = blocks[by * 20 + bx], v.count > 20 else { line += "   ."; continue }
            v.sort()
            line += String(format: "%4d", Int((v[v.count / 2] * 10).rounded()))
        }
        print(line)
    }
    distances.sort(); spread.sort()
    guard !distances.isEmpty else { return }
    print(String(format: "  fused mesh vs %d-frame averaged raw depth (%d points; raw per-frame spread median %.2f mm):",
                 frames.count, distances.count, spread[spread.count / 2]))
    print(String(format: "    median %.2f mm, 75%% %.2f mm, 90%% %.2f mm", distances[distances.count / 2],
                 distances[distances.count * 3 / 4], distances[distances.count * 9 / 10]))

    // Best rigid fit (ICP with Kabsch): what remains is shape error, not the reference frames' pose.
    var moved = points
    for _ in 0..<15 {
        let pairs = moved.compactMap { p in surface.closestPoint(p).map { (p, $0) } }
        let n = Float(pairs.count)
        let ca = pairs.reduce(SIMD3<Float>()) { $0 + $1.0 } / n, cb = pairs.reduce(SIMD3<Float>()) { $0 + $1.1 } / n
        var h = simd_double3x3()
        for (a, b) in pairs {
            let x = SIMD3<Double>(a - ca), y = SIMD3<Double>(b - cb)
            h += simd_double3x3(columns: (x * y.x, x * y.y, x * y.z))
        }
        // Rotation maximizing tr(R H): polar decomposition by Newton iteration (H is near identity-like here).
        var r = h.transpose
        for _ in 0..<30 { r = (r + r.inverse.transpose) * 0.5 }
        let rf = simd_float3x3(SIMD3<Float>(r.columns.0), SIMD3<Float>(r.columns.1), SIMD3<Float>(r.columns.2))
        moved = moved.map { rf * ($0 - ca) + cb }
    }
    var fitted = moved.compactMap { surface($0) }.map { $0 * 1000 }.filter { $0 < 6 }
    fitted.sort()
    guard !fitted.isEmpty else { return }
    print(String(format: "    after rigid fit: median %.2f mm, 75%% %.2f mm, 90%% %.2f mm", fitted[fitted.count / 2],
                 fitted[fitted.count * 3 / 4], fitted[fitted.count * 9 / 10]))
}

/// Averages "depth 3 · icp 12 · …" lines per stage.
private func printStages(_ lines: [String], label: String) {
    var total: [String: Double] = [:], count: [String: Int] = [:]
    for line in lines {
        for part in line.components(separatedBy: " · ") {
            let pieces = part.split(separator: " ")
            guard pieces.count == 2, let ms = Double(pieces[1]) else { continue }
            total[String(pieces[0]), default: 0] += ms
            count[String(pieces[0]), default: 0] += 1
        }
    }
    guard !total.isEmpty else { return }
    let order = ["depth", "icp", "fuse", "model", "preview"]
    let summary = order.compactMap { name in total[name].map { String(format: "%@ %.1f", name, $0 / Double(count[name]!)) } }
    print("  average ms per stage (\(label)): " + summary.joined(separator: " · "))
}

private func printReasons(_ reasons: [String]) {
    // Group "few matches (level 2: 41 < 60)" and similar by their wording, not their numbers.
    var counts: [String: Int] = [:]
    for reason in reasons where !reason.isEmpty {
        let key = reason.prefix { $0 != "(" }.trimmingCharacters(in: .whitespaces)
        let level = reason.range(of: "level \\d", options: .regularExpression).map { String(reason[$0]) } ?? ""
        counts["\(key) \(level)".trimmingCharacters(in: .whitespaces), default: 0] += 1
    }
    for (reason, count) in counts.sorted(by: { $0.value > $1.value }) { print("    \(count)× \(reason)") }
}
