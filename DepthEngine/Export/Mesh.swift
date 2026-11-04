public struct Mesh: Sendable {
    /// Meters.
    public var vertices: [SIMD3<Float>]
    /// Triangle list, counter-clockwise when viewed from outside.
    public var indices: [UInt32]
    /// Per-vertex RGB in 0...1, when the scan had colour.
    public var colors: [SIMD3<Float>]?
    /// A relief lying on its flat back: +Z faces the viewer and +Y is the subject's top. Otherwise +Z is up.
    public var liesOnBack = false

    public var triangleCount: Int { indices.count / 3 }

    public init(vertices: [SIMD3<Float>], indices: [UInt32], colors: [SIMD3<Float>]? = nil) {
        self.vertices = vertices
        self.indices = indices
        self.colors = colors
    }

    /// Drops everything not connected to the largest piece: sensor-noise blobs, or a turntable marker.
    func largestComponent() -> Mesh {
        var parent = Array(vertices.indices)
        func root(_ x: Int) -> Int {
            var x = x
            while parent[x] != x {
                parent[x] = parent[parent[x]]
                x = parent[x]
            }
            return x
        }
        for t in stride(from: 0, to: indices.count, by: 3) {
            let a = root(Int(indices[t]))
            parent[root(Int(indices[t + 1]))] = a
            parent[root(Int(indices[t + 2]))] = a
        }
        var triangles: [Int: Int] = [:]
        for t in stride(from: 0, to: indices.count, by: 3) { triangles[root(Int(indices[t])), default: 0] += 1 }
        guard triangles.count > 1, let keep = triangles.max(by: { $0.value < $1.value })?.key else { return self }

        var remap = [UInt32](repeating: .max, count: vertices.count)
        var kept = Mesh(vertices: [], indices: [], colors: colors == nil ? nil : [])
        for t in stride(from: 0, to: indices.count, by: 3) where root(Int(indices[t])) == keep {
            for i in indices[t..<(t + 3)] {
                if remap[Int(i)] == .max {
                    remap[Int(i)] = UInt32(kept.vertices.count)
                    kept.vertices.append(vertices[Int(i)])
                    if let colors { kept.colors?.append(colors[Int(i)]) }
                }
                kept.indices.append(remap[Int(i)])
            }
        }
        return kept
    }
}
