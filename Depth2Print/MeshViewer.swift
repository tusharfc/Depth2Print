import DepthEngine
import SceneKit
import SwiftUI

/// Full-screen orbit view of a finished scan, with or without its colours.
struct MeshViewer: View {
    let mesh: Mesh
    @Environment(\.dismiss) private var dismiss
    @State private var geometry: SceneMesh?
    @State private var showsColor = true

    var body: some View {
        NavigationStack {
            Group {
                if let geometry {
                    SceneMeshView(mesh: geometry, showsColor: showsColor && geometry.colored != nil)
                        .ignoresSafeArea(edges: .bottom)
                } else {
                    ProgressView("Preparing 3D view…")
                }
            }
            .navigationTitle("Scan")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
                ToolbarItem(placement: .primaryAction) {
                    Toggle(isOn: $showsColor) { Label("Colour", systemImage: "paintpalette") }
                        .toggleStyle(.button)
                        .disabled(mesh.colors == nil)
                }
            }
        }
        .task {
            let mesh = mesh
            geometry = await Task.detached(priority: .userInitiated) { SceneMesh(mesh) }.value
        }
    }
}

/// The scan as SceneKit geometry, Y-up and centered on the origin, its front facing the initial camera (+Z).
struct SceneMesh: @unchecked Sendable {
    let plain: SCNGeometry
    let colored: SCNGeometry?
    let radius: Float

    init(_ mesh: Mesh) {
        let lo = mesh.vertices.reduce(SIMD3<Float>(repeating: .infinity)) { simd_min($0, $1) }
        let hi = mesh.vertices.reduce(SIMD3<Float>(repeating: -.infinity)) { simd_max($0, $1) }
        let center = (lo + hi) / 2
        // Standing exports are Z-up with the first camera's side toward -Y.
        let liesOnBack = mesh.liesOnBack
        func yUp(_ v: SIMD3<Float>) -> SCNVector3 { liesOnBack ? SCNVector3(v.x, v.y, v.z) : SCNVector3(v.x, v.z, -v.y) }

        var normals = [SIMD3<Float>](repeating: .zero, count: mesh.vertices.count)
        for t in stride(from: 0, to: mesh.indices.count, by: 3) {
            let a = Int(mesh.indices[t]), b = Int(mesh.indices[t + 1]), c = Int(mesh.indices[t + 2])
            let n = cross(mesh.vertices[b] - mesh.vertices[a], mesh.vertices[c] - mesh.vertices[a])
            normals[a] += n
            normals[b] += n
            normals[c] += n
        }
        let positions = SCNGeometrySource(vertices: mesh.vertices.map { yUp($0 - center) })
        let normalSource = SCNGeometrySource(normals: normals.map { yUp(simd_length($0) > 0 ? simd_normalize($0) : $0) })
        let element = SCNGeometryElement(data: mesh.indices.withUnsafeBufferPointer { Data(buffer: $0) },
                                         primitiveType: .triangles, primitiveCount: mesh.triangleCount,
                                         bytesPerIndex: MemoryLayout<UInt32>.size)

        plain = SCNGeometry(sources: [positions, normalSource], elements: [element])
        plain.firstMaterial = Self.material(UIColor(white: 0.8, alpha: 1))

        if let colors = mesh.colors {
            // SceneKit shades in linear space; the scan's colours are sRGB.
            let linear = colors.map { c in SIMD3<Float>(Self.linear(c.x), Self.linear(c.y), Self.linear(c.z)) }
            let colorSource = SCNGeometrySource(data: linear.withUnsafeBufferPointer { Data(buffer: $0) }, semantic: .color,
                                                vectorCount: linear.count, usesFloatComponents: true,
                                                componentsPerVector: 3, bytesPerComponent: MemoryLayout<Float>.size,
                                                dataOffset: 0, dataStride: MemoryLayout<SIMD3<Float>>.stride)
            let geometry = SCNGeometry(sources: [positions, normalSource, colorSource], elements: [element])
            geometry.firstMaterial = Self.material(.white)
            colored = geometry
        } else {
            colored = nil
        }
        radius = simd_length(hi - lo) / 2
    }

    private static func material(_ color: UIColor) -> SCNMaterial {
        let material = SCNMaterial()
        material.lightingModel = .lambert
        material.diffuse.contents = color
        return material
    }

    private static func linear(_ c: Float) -> Float {
        c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
    }
}

/// Drag to orbit, pinch to zoom, two fingers to pan.
private struct SceneMeshView: UIViewRepresentable {
    let mesh: SceneMesh
    let showsColor: Bool

    func makeUIView(context: Context) -> SCNView {
        let scene = SCNScene()
        let node = SCNNode(geometry: mesh.plain)
        node.name = "scan"
        scene.rootNode.addChildNode(node)

        let camera = SCNCamera()
        camera.zNear = Double(mesh.radius) * 0.01
        camera.zFar = Double(mesh.radius) * 100
        let cameraNode = SCNNode()
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(0, 0, mesh.radius * 2.8)
        // A light that moves with the camera, so whatever faces you is lit.
        let headlight = SCNLight()
        headlight.type = .directional
        headlight.intensity = 800
        cameraNode.light = headlight
        scene.rootNode.addChildNode(cameraNode)

        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 400
        scene.rootNode.addChildNode(ambient)

        let view = SCNView()
        view.scene = scene
        view.pointOfView = cameraNode
        view.allowsCameraControl = true
        view.defaultCameraController.interactionMode = .orbitTurntable
        view.defaultCameraController.target = SCNVector3Zero
        view.backgroundColor = .systemBackground
        view.antialiasingMode = .multisampling4X
        return view
    }

    func updateUIView(_ view: SCNView, context: Context) {
        view.scene?.rootNode.childNode(withName: "scan", recursively: false)?.geometry =
            showsColor ? mesh.colored : mesh.plain
    }
}
