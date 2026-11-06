import ARKit
import DepthEngine
import SceneKit
import SwiftUI
import simd

/// Camera feed of an ARKit face-tracking session, with the live model preview drawn over it.
struct FaceTrackingPreview: UIViewRepresentable {
    let session: ARSession
    /// Image on the depth buffer's pixel grid (sensor-native, unmirrored), drawn exactly over the camera feed.
    var overlay: CGImage?
    var onCameraAxes: (simd_float3x3) -> Void = { _ in }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView(frame: .zero, options: nil)
        view.session = session
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.onCameraAxes = onCameraAxes
        uiView.setOverlay(overlay)
    }

    final class PreviewView: ARSCNView {
        var onCameraAxes: (simd_float3x3) -> Void = { _ in }
        private let overlayLayer = CALayer()

        override init(frame: CGRect, options: [String: Any]? = nil) {
            super.init(frame: frame, options: options)
            scene = SCNScene()
            automaticallyUpdatesLighting = false
            overlayLayer.anchorPoint = .zero
            overlayLayer.bounds = CGRect(x: 0, y: 0, width: 1, height: 1)
            overlayLayer.contentsGravity = .resize
            overlayLayer.actions = ["contents": NSNull(), "transform": NSNull()]
            layer.addSublayer(overlayLayer)
        }

        required init?(coder: NSCoder) { fatalError("init(coder:) is not used") }

        func setOverlay(_ image: CGImage?) {
            overlayLayer.contents = image
            overlayLayer.isHidden = image == nil
            alignOverlay()
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            alignOverlay()
        }

        /// Where the camera image's corners land on screen, found by projecting their rays through the view's own
        /// camera, so the view's rotation, selfie mirroring and aspect-fill cropping are all matched.
        private func alignOverlay() {
            guard let camera = session.currentFrame?.camera else { return }
            let size = camera.imageResolution, k = camera.intrinsics
            func screen(_ nx: Float, _ ny: Float) -> CGPoint {
                let px = nx * Float(size.width), py = ny * Float(size.height)
                // ARKit's camera looks down -z with +y up.
                let ray = SIMD3((px - k[2][0]) / k[0][0], -(py - k[2][1]) / k[1][1], -1) * 0.5
                let world = camera.transform * SIMD4(ray, 1)
                let p = projectPoint(SCNVector3(world.x, world.y, world.z))
                return CGPoint(x: CGFloat(p.x), y: CGFloat(p.y))
            }
            let origin = screen(0, 0), xEnd = screen(1, 0), yEnd = screen(0, 1)
            let x = CGVector(dx: xEnd.x - origin.x, dy: xEnd.y - origin.y)
            let y = CGVector(dx: yEnd.x - origin.x, dy: yEnd.y - origin.y)
            // Before the first frame is drawn the projection is degenerate.
            guard [x.dx, x.dy, y.dx, y.dy].allSatisfy(\.isFinite), abs(x.dx * y.dy - x.dy * y.dx) > 1 else { return }

            overlayLayer.position = .zero
            overlayLayer.transform = CATransform3DMakeAffineTransform(
                CGAffineTransform(a: x.dx, b: x.dy, c: y.dx, d: y.dy, tx: origin.x, ty: origin.y))
            onCameraAxes(CameraAxes.deviceFromCamera(bufferXOnScreen: x, bufferYOnScreen: y))
        }
    }
}
