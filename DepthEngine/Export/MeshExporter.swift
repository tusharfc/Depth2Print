import Foundation
import simd

public enum MeshExportFormat: String, CaseIterable, Sendable {
    case stl
    case obj

    public var fileExtension: String { rawValue }
}

public enum MeshExporter {
    /// `scale` converts meters to millimeters, which slicers assume for unitless STL/OBJ.
    public static func write(_ mesh: Mesh, format: MeshExportFormat, to url: URL, scale: Float = 1000) throws {
        let data: Data
        switch format {
        case .stl: data = binarySTL(mesh, scale: scale)
        case .obj: data = Data(obj(mesh, scale: scale).utf8)
        }
        try data.write(to: url, options: .atomic)
    }

    static func binarySTL(_ mesh: Mesh, scale: Float) -> Data {
        let triangleCount = mesh.triangleCount
        var data = Data(capacity: 84 + triangleCount * 50)
        var header = Data("Depth2Print binary STL".utf8)
        header.append(Data(count: 80 - header.count))
        data.append(header)
        append(UInt32(triangleCount), to: &data)

        for t in 0..<triangleCount {
            let a = mesh.vertices[Int(mesh.indices[3 * t])] * scale
            let b = mesh.vertices[Int(mesh.indices[3 * t + 1])] * scale
            let c = mesh.vertices[Int(mesh.indices[3 * t + 2])] * scale
            let cross = simd_cross(b - a, c - a)
            let length = simd_length(cross)
            let normal = length > 0 ? cross / length : .zero
            for v in [normal, a, b, c] {
                append(v.x.bitPattern, to: &data)
                append(v.y.bitPattern, to: &data)
                append(v.z.bitPattern, to: &data)
            }
            append(UInt16(0), to: &data)
        }
        return data
    }

    /// Colours, if any, go on the vertex lines ("v x y z r g b"), which MeshLab and Blender read.
    static func obj(_ mesh: Mesh, scale: Float) -> String {
        var out = "# Depth2Print\n"
        out.reserveCapacity(mesh.vertices.count * (mesh.colors == nil ? 32 : 52) + mesh.indices.count * 8)
        for (i, v) in mesh.vertices.enumerated() {
            let s = v * scale
            out += "v \(s.x) \(s.y) \(s.z)"
            if let c = mesh.colors?[i] {
                let rgb = (simd_clamp(c, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1)) * 1000).rounded(.toNearestOrEven) / 1000
                out += " \(rgb.x) \(rgb.y) \(rgb.z)"
            }
            out += "\n"
        }
        for t in stride(from: 0, to: mesh.indices.count, by: 3) {
            out += "f \(mesh.indices[t] + 1) \(mesh.indices[t + 1] + 1) \(mesh.indices[t + 2] + 1)\n"
        }
        return out
    }

    private static func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }
}
