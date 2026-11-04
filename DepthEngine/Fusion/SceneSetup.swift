import simd

/// A plane n·p + d = 0 with n pointing to the camera's side.
struct Plane {
    var normal: SIMD3<Float>
    var offset: Float

    var vector: SIMD4<Float> { SIMD4(normal, offset) }

    func distance(_ p: SIMD3<Float>) -> Float { dot(normal, p) + offset }
}

/// First-frame analysis: what to keep, where to put the volume, and where to cut the mesh.
enum SceneSetup {
    struct Points {
        var positions: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
    }

    /// Valid points with normals from one pyramid level, thinned to at most `maxCount`.
    static func points(_ level: FramePyramid.Level, maxCount: Int = 6000) -> Points {
        let count = level.width * level.height
        let vertices = level.vertices.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        let normals = level.normals.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        var valid: [Int] = []
        for i in 0..<count where normals[i].w > 0 { valid.append(i) }
        let stride = max(1, valid.count / maxCount)
        var points = Points()
        for i in Swift.stride(from: 0, to: valid.count, by: stride) {
            let v = vertices[valid[i]], n = normals[valid[i]]
            points.positions.append(SIMD3(v.x, v.y, v.z))
            points.normals.append(SIMD3(n.x, n.y, n.z))
        }
        return points
    }

    /// The surface the object stands on. A candidate must have part of the scene resting on its camera side and
    /// nothing visible beneath its own footprint (which rules out an object's top face), and must face up when
    /// gravity (`up`, camera space) is known. Of those, the one nearest the camera's height wins: a turntable top
    /// rather than the table beneath it.
    static func supportPlane(_ points: Points, up: SIMD3<Float>? = nil) -> Plane? {
        let total = points.positions.count
        guard total >= 200 else { return nil }
        var rng = SplitMix(state: 0x9E37_79B9_7F4A_7C15)
        var remaining = Array(points.positions.indices)
        var candidates: [Plane] = []
        for _ in 0..<5 {
            guard let (plane, inliers) = ransac(points, remaining, &rng), inliers.count >= max(100, total / 40) else { break }
            let above = points.positions.filter { plane.distance($0) > 0.01 }.count
            let level = up.map { dot($0, plane.normal) > cos(25 * Float.pi / 180) } ?? true
            if level && above >= total / 20 && underneath(plane, inliers, points) <= max(20, inliers.count / 20) {
                candidates.append(plane)
            }
            let removed = Set(inliers)
            remaining.removeAll { removed.contains($0) }
        }
        return candidates.min { $0.offset < $1.offset }
    }

    /// Points more than 2 cm below `plane` that sit within its inliers' footprint (1 cm grid, dilated one cell).
    private static func underneath(_ plane: Plane, _ inliers: [Int], _ points: Points) -> Int {
        let u = normalize(cross(plane.normal, abs(plane.normal.x) < 0.9 ? SIMD3(1, 0, 0) : SIMD3(0, 1, 0)))
        let v = cross(plane.normal, u)
        func cell(_ p: SIMD3<Float>) -> SIMD2<Int32> {
            SIMD2(Int32((dot(p, u) / 0.01).rounded(.down)), Int32((dot(p, v) / 0.01).rounded(.down)))
        }
        var footprint = Set<SIMD2<Int32>>()
        for i in inliers {
            let c = cell(points.positions[i])
            for dx in -1...1 { for dy in -1...1 { footprint.insert(c &+ SIMD2(Int32(dx), Int32(dy))) } }
        }
        return points.positions.filter { plane.distance($0) < -0.02 && footprint.contains(cell($0)) }.count
    }

    struct Head {
        /// Sphere (center, radius) that contains the whole head, chin included.
        var region: SIMD4<Float>
        /// Horizontal cut below the chin; the side toward the crown is kept. Nil without gravity.
        var neck: Plane?
    }

    /// Finds the head on the first frame. With gravity (`up`, camera space) it works down from the crown: the head
    /// is the top of the person, whereas the nearest surface is often the chest when the phone is held low.
    /// Without gravity it falls back to the nearest surface.
    static func head(_ points: Points, up: SIMD3<Float>?) -> Head? {
        guard points.positions.count >= 200 else { return nil }
        func median(_ values: [Float]) -> Float { values.sorted()[values.count / 2] }
        let byDepth = points.positions.sorted { $0.z < $1.z }
        let nearest = byDepth[byDepth.count / 50]

        guard let up else {
            // Points within 12 cm of the nearest one: the face, but not the shoulders.
            let face = points.positions.filter { distance($0, nearest) < 0.12 }
            let direction = SIMD3(median(face.map { $0.x / $0.z }), median(face.map { $0.y / $0.z }), 1)
            return Head(region: SIMD4(direction * (nearest.z + 0.09), 0.14), neck: nil)
        }

        // The person is everything within 30 cm behind the nearest surface; the room is further back.
        let person = byDepth.prefix { $0.z < nearest.z + 0.3 }
        let heights = person.map { dot($0, up) }.sorted()
        // Nearly the top: the skull's cap is a tiny share of the person's points, so a plain percentile lands
        // centimetres low; dropping only the top ~0.2% still rejects stray pixels.
        let crown = heights[heights.count - max(3, heights.count / 500)]
        // Crown to chin is ~23 cm on an adult; the band below that is neck and chest.
        let band = person.filter { dot($0, up) > crown - 0.23 }
        guard band.count >= 100 else { return nil }
        let front = band.map(\.z).sorted()[band.count / 50]
        let direction = SIMD3(median(band.map { $0.x / $0.z }), median(band.map { $0.y / $0.z }), 1)
        var center = direction * (front + 0.09)
        // The skull's middle is ~11 cm below the crown, whatever the camera's tilt.
        center += up * ((crown - 0.11) - dot(center, up))
        return Head(region: SIMD4(center, 0.17), neck: Plane(normal: up, offset: -(crown - 0.27)))
    }

    static func volumeCenter(_ points: Points, plane: Plane?, size: Float) -> SIMD3<Float>? {
        let object = plane.map { plane in points.positions.filter { plane.distance($0) > 0.01 } } ?? points.positions
        guard object.count >= 50 else { return nil }
        let surface = object.reduce(SIMD3<Float>(repeating: 0), +) / Float(object.count)
        // Only the front of the object is visible, so its middle lies further along the view ray.
        var center = surface + normalize(surface) * size / 4
        if let plane {
            // Rest the volume 2 cm below the plane so its height isn't wasted on the table.
            center += plane.normal * (size / 2 - 0.02 - plane.distance(center))
        }
        return center
    }

    private static func ransac(_ points: Points, _ indices: [Int],
                               _ rng: inout SplitMix) -> (Plane, [Int])? {
        guard indices.count >= 3 else { return nil }
        let tolerance: Float = 0.004
        func inliers(of plane: Plane) -> [Int] {
            indices.filter {
                abs(plane.distance(points.positions[$0])) < tolerance && abs(dot(points.normals[$0], plane.normal)) > 0.9
            }
        }

        var best: Plane?
        var bestCount = 0
        for _ in 0..<300 {
            let a = points.positions[indices[Int.random(in: 0..<indices.count, using: &rng)]]
            let b = points.positions[indices[Int.random(in: 0..<indices.count, using: &rng)]]
            let c = points.positions[indices[Int.random(in: 0..<indices.count, using: &rng)]]
            let n = cross(b - a, c - a)
            guard length(n) > 1e-7 else { continue }
            let plane = Plane(normal: normalize(n), offset: -dot(normalize(n), a))
            let count = inliers(of: plane).count
            if count > bestCount {
                best = plane
                bestCount = count
            }
        }
        guard let best else { return nil }

        // Least-squares refit: the normal is the covariance's smallest eigenvector.
        let fit = inliers(of: best)
        let mean = fit.reduce(SIMD3<Float>(repeating: 0)) { $0 + points.positions[$1] } / Float(fit.count)
        var covariance = [Double](repeating: 0, count: 9)
        for i in fit {
            let d = SIMD3<Double>(points.positions[i] - mean)
            for r in 0..<3 { for c in 0..<3 { covariance[r * 3 + c] += d[r] * d[c] } }
        }
        let (values, vectors) = ICP.symmetricEigen(covariance, n: 3)
        let k = values.indices.min { values[$0] < values[$1] }!
        var normal = normalize(SIMD3<Float>(Float(vectors[k]), Float(vectors[3 + k]), Float(vectors[6 + k])))
        var offset = -dot(normal, mean)
        if offset < 0 {
            normal = -normal
            offset = -offset
        }
        let refined = Plane(normal: normal, offset: offset)
        return (refined, inliers(of: refined))
    }
}

/// Deterministic so the same frame always yields the same plane.
private struct SplitMix: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
