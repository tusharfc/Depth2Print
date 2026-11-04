import Metal
import simd

// Layout must match MarchingCubesParams in TSDF.metal.
private struct MarchingCubesParams {
    var origin: SIMD4<Float>
    var plane: SIMD4<Float>
    var back: SIMD4<Float>
    var region: SIMD4<Float>
    var resolution: UInt32
    var voxelSize: Float
    var tableStride: UInt32
    var maxTriangles: UInt32
}

/// Marching Cubes lookup tables, derived from first principles instead of hand-copied.
/// On ambiguous faces inside corners are always kept apart; neighbouring cubes see the same face and make the
/// same choice, so the mesh has no cracks.
enum MarchingCubesTables {
    struct Table {
        /// Triangles per case (case bit k = corner k is inside).
        let counts: [UInt8]
        /// Per case, `stride` entries: edge indices, three per triangle.
        let edges: [UInt8]
        let stride: Int
    }

    /// Corner k sits at (k & 1, (k >> 1) & 1, (k >> 2) & 1). Edge e runs along axis e / 4, lower corner first.
    static let edgeCorners: [SIMD2<UInt8>] = {
        var edges: [SIMD2<UInt8>] = []
        for axis in 0..<3 {
            for c in 0..<8 where c & (1 << axis) == 0 {
                edges.append(SIMD2(UInt8(c), UInt8(c | 1 << axis)))
            }
        }
        return edges
    }()

    static let table: Table = {
        func edgeIndex(_ a: Int, _ b: Int) -> Int {
            edgeCorners.firstIndex(of: SIMD2(UInt8(min(a, b)), UInt8(max(a, b))))!
        }

        // Corners of each face, counter-clockwise when viewed from outside the cube.
        var faces: [[Int]] = []
        for axis in 0..<3 {
            let u = (axis + 1) % 3, v = (axis + 2) % 3
            for side in 0..<2 {
                var cycle = [(0, 0), (1, 0), (1, 1), (0, 1)].map { side << axis | $0.0 << u | $0.1 << v }
                if side == 0 { cycle.reverse() }
                faces.append(cycle)
            }
        }

        func onFace(_ e: Int, _ face: [Int]) -> Bool {
            face.contains(Int(edgeCorners[e].x)) && face.contains(Int(edgeCorners[e].y))
        }
        func shareFace(_ a: Int, _ b: Int) -> Bool {
            faces.contains { onFace(a, $0) && onFace(b, $0) }
        }

        // Splits a loop into triangles (kept in loop order, which faces outward) without any diagonal between
        // two points on the same cube face: the neighbouring cube could pick that diagonal too, making it
        // non-manifold.
        func triangulate(_ loop: [Int]) -> [Int] {
            let n = loop.count
            func allowed(_ i: Int, _ j: Int) -> Bool {
                j - i == 1 || (i == 0 && j == n - 1) || !shareFace(loop[i], loop[j])
            }
            var memo: [Int: [Int]?] = [:]
            func solve(_ i: Int, _ j: Int) -> [Int]? {
                if j - i < 2 { return [] }
                if let cached = memo[i * 16 + j] { return cached }
                var result: [Int]?
                for k in (i + 1)..<j where allowed(i, k) && allowed(k, j) {
                    if let left = solve(i, k), let right = solve(k, j) {
                        result = left + [loop[i], loop[k], loop[j]] + right
                        break
                    }
                }
                memo[i * 16 + j] = result
                return result
            }
            guard let triangles = solve(0, n - 1) else { preconditionFailure("No manifold triangulation for loop \(loop)") }
            return triangles
        }

        var perCase: [[Int]] = []
        for m in 0..<256 {
            func inside(_ corner: Int) -> Bool { m >> corner & 1 == 1 }

            // On each face, link the edge where the walk enters the inside to the edge where it next leaves.
            // Faces are walked consistently, so the links chain into closed loops around the cube.
            var next = [Int](repeating: -1, count: 12)
            for face in faces {
                var crossings: [(edge: Int, leaves: Bool)] = []
                for i in 0..<4 {
                    let a = face[i], b = face[(i + 1) % 4]
                    if inside(a) != inside(b) { crossings.append((edgeIndex(a, b), inside(a))) }
                }
                for (j, crossing) in crossings.enumerated() where crossing.leaves {
                    let entry = crossings[(j + crossings.count - 1) % crossings.count]
                    next[entry.edge] = crossing.edge
                }
            }

            var visited = [Bool](repeating: false, count: 12)
            var triangles: [Int] = []
            for start in 0..<12 where next[start] >= 0 && !visited[start] {
                var loop: [Int] = []
                var e = start
                while !visited[e] {
                    visited[e] = true
                    loop.append(e)
                    e = next[e]
                }
                triangles += triangulate(loop)
            }
            perCase.append(triangles)
        }

        let stride = perCase.map(\.count).max()!
        var edges = [UInt8](repeating: 0, count: 256 * stride)
        for (m, triangles) in perCase.enumerated() {
            for (i, e) in triangles.enumerated() { edges[m * stride + i] = UInt8(e) }
        }
        return Table(counts: perCase.map { UInt8($0.count / 3) }, edges: edges, stride: stride)
    }()
}

/// GPU Marching Cubes over a `TSDFVolume`, producing a closed, welded (shared-vertex) mesh in world space.
final class MarchingCubes {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let countPipeline: MTLComputePipelineState
    private let emitPipeline: MTLComputePipelineState
    private let triangleCounts: MTLBuffer
    private let triangleEdges: MTLBuffer
    private let edgeCorners: MTLBuffer

    private static let group = MTLSize(width: 8, height: 8, depth: 4)

    init(device: MTLDevice, library: MTLLibrary) throws {
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw DepthEngineError.shaderMissing(name) }
            return try device.makeComputePipelineState(function: function)
        }
        let table = MarchingCubesTables.table
        guard let queue = device.makeCommandQueue(),
              let counts = device.makeBuffer(bytes: table.counts, length: table.counts.count),
              let edges = device.makeBuffer(bytes: table.edges, length: table.edges.count),
              let corners = device.makeBuffer(bytes: MarchingCubesTables.edgeCorners,
                                              length: MarchingCubesTables.edgeCorners.count * MemoryLayout<SIMD2<UInt8>>.stride)
        else { throw DepthEngineError.metalUnavailable }
        self.device = device
        self.queue = queue
        self.countPipeline = try pipeline("marchingCubesCount")
        self.emitPipeline = try pipeline("marchingCubesEmit")
        self.triangleCounts = counts
        self.triangleEdges = edges
        self.edgeCorners = corners
    }

    /// Always watertight. Everything below the `cuts` planes (world (n, d), at most two) and outside `region`
    /// (world sphere) is removed and the cuts are capped flat.
    func extract(_ volume: TSDFVolume, cuts: [SIMD4<Float>] = [], region: SIMD4<Float>? = nil) throws -> Mesh {
        precondition(cuts.count <= 2)
        let n = volume.config.resolution
        var params = MarchingCubesParams(origin: SIMD4(volume.origin, 1), plane: cuts.first ?? .zero,
                                         back: cuts.count > 1 ? cuts[1] : .zero,
                                         region: region ?? .zero, resolution: UInt32(n),
                                         voxelSize: volume.voxelSize,
                                         tableStride: UInt32(MarchingCubesTables.table.stride), maxTriangles: 0)
        guard let counter = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            throw DepthEngineError.metalUnavailable
        }

        // Pass 1 counts triangles so pass 2 can write into an exactly sized buffer.
        counter.contents().storeBytes(of: 0, as: UInt32.self)
        try run(n) { encoder in
            encoder.setComputePipelineState(countPipeline)
            encoder.setBuffer(volume.buffer, offset: 0, index: 0)
            encoder.setBytes(&params, length: MemoryLayout<MarchingCubesParams>.stride, index: 1)
            encoder.setBuffer(triangleCounts, offset: 0, index: 2)
            encoder.setBuffer(counter, offset: 0, index: 3)
        }
        let total = Int(counter.contents().load(as: UInt32.self))
        guard total > 0 else { return Mesh(vertices: [], indices: []) }

        params.maxTriangles = UInt32(total)
        guard let positions = device.makeBuffer(length: total * 3 * MemoryLayout<SIMD4<Float>>.stride,
                                                options: .storageModeShared),
              let keys = device.makeBuffer(length: total * 3 * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else { throw DepthEngineError.metalUnavailable }
        counter.contents().storeBytes(of: 0, as: UInt32.self)
        try run(n) { encoder in
            encoder.setComputePipelineState(emitPipeline)
            encoder.setBuffer(volume.buffer, offset: 0, index: 0)
            encoder.setBytes(&params, length: MemoryLayout<MarchingCubesParams>.stride, index: 1)
            encoder.setBuffer(triangleCounts, offset: 0, index: 2)
            encoder.setBuffer(counter, offset: 0, index: 3)
            encoder.setBuffer(triangleEdges, offset: 0, index: 4)
            encoder.setBuffer(edgeCorners, offset: 0, index: 5)
            encoder.setBuffer(positions, offset: 0, index: 6)
            encoder.setBuffer(keys, offset: 0, index: 7)
        }

        let vertexCount = total * 3
        let soup = positions.contents().bindMemory(to: SIMD4<Float>.self, capacity: vertexCount)
        let edgeKeys = keys.contents().bindMemory(to: UInt32.self, capacity: vertexCount)
        var lookup = [UInt32: UInt32](minimumCapacity: vertexCount / 2)
        var vertices: [SIMD3<Float>] = []
        vertices.reserveCapacity(vertexCount / 2)
        var indices: [UInt32] = []
        indices.reserveCapacity(vertexCount)
        for i in 0..<vertexCount {
            if let index = lookup[edgeKeys[i]] {
                indices.append(index)
                continue
            }
            let index = UInt32(vertices.count)
            lookup[edgeKeys[i]] = index
            let p = soup[i]
            vertices.append(SIMD3(p.x, p.y, p.z))
            indices.append(index)
        }
        return Mesh(vertices: vertices, indices: indices)
    }

    private func run(_ n: Int, _ encode: (MTLComputeCommandEncoder) -> Void) throws {
        guard let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder() else { throw DepthEngineError.metalUnavailable }
        encode(encoder)
        let g = Self.group
        encoder.dispatchThreadgroups(MTLSize(width: (n + g.width - 1) / g.width, height: (n + g.height - 1) / g.height,
                                             depth: (n + g.depth - 1) / g.depth),
                                     threadsPerThreadgroup: g)
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw error }
    }
}
