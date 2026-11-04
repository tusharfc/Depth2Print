/// Triangulates a single depth view by connecting neighboring pixels. Stand-in until TSDF + Marching Cubes.
enum GridMesher {
    static func mesh(_ grid: PointGrid, maxDepthJump: Float = 0.01) -> Mesh {
        let invalid = UInt32.max
        var remap = [UInt32](repeating: invalid, count: grid.points.count)
        var vertices: [SIMD3<Float>] = []
        vertices.reserveCapacity(grid.points.count)
        for (i, p) in grid.points.enumerated() where p.w > 0 {
            remap[i] = UInt32(vertices.count)
            vertices.append(SIMD3(p.x, p.y, p.z))
        }

        var indices: [UInt32] = []
        indices.reserveCapacity(grid.points.count * 6)

        // Skip triangles spanning a silhouette edge, otherwise foreground and background get stitched together.
        func connect(_ a: Int, _ b: Int, _ c: Int) {
            let ia = remap[a], ib = remap[b], ic = remap[c]
            guard ia != invalid, ib != invalid, ic != invalid else { return }
            let za = grid.points[a].z, zb = grid.points[b].z, zc = grid.points[c].z
            guard max(za, zb, zc) - min(za, zb, zc) < maxDepthJump else { return }
            indices.append(ia)
            indices.append(ib)
            indices.append(ic)
        }

        guard grid.width > 1, grid.height > 1 else { return Mesh(vertices: vertices, indices: []) }
        for y in 0..<(grid.height - 1) {
            for x in 0..<(grid.width - 1) {
                let i00 = y * grid.width + x
                let i10 = i00 + 1
                let i01 = i00 + grid.width
                let i11 = i01 + 1
                // Wound so normals face the camera (-z in camera space).
                connect(i00, i01, i10)
                connect(i10, i01, i11)
            }
        }
        return Mesh(vertices: vertices, indices: indices)
    }
}
