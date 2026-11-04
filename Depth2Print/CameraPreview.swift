import AVFoundation
import DepthEngine
import SwiftUI
import simd

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    /// Image on the depth buffer's pixel grid (sensor-native, unmirrored), drawn exactly over the camera feed.
    var overlay: CGImage?
    var onCameraAxes: (simd_float3x3) -> Void = { _ in }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.onCameraAxes = onCameraAxes
        uiView.setOverlay(overlay)
    }

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
        var onCameraAxes: (simd_float3x3) -> Void = { _ in }

        private let overlayLayer = CALayer()

        override init(frame: CGRect) {
            super.init(frame: frame)
            overlayLayer.anchorPoint = .zero
            overlayLayer.bounds = CGRect(x: 0, y: 0, width: 1, height: 1)
            overlayLayer.contentsGravity = .resize
            overlayLayer.actions = ["contents": NSNull(), "transform": NSNull()]
            previewLayer.addSublayer(overlayLayer)
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

        /// Maps the buffer's normalized grid onto the screen the same way the preview layer maps the video, so
        /// rotation, mirroring and aspect-fill cropping all match. The same mapping tells the engine how the
        /// camera sits in the phone.
        private func alignOverlay() {
            let origin = previewLayer.layerPointConverted(fromCaptureDevicePoint: .zero)
            let xEnd = previewLayer.layerPointConverted(fromCaptureDevicePoint: CGPoint(x: 1, y: 0))
            let yEnd = previewLayer.layerPointConverted(fromCaptureDevicePoint: CGPoint(x: 0, y: 1))
            let x = CGVector(dx: xEnd.x - origin.x, dy: xEnd.y - origin.y)
            let y = CGVector(dx: yEnd.x - origin.x, dy: yEnd.y - origin.y)
            // Before the session's connection exists the conversions are degenerate.
            guard [x.dx, x.dy, y.dx, y.dy].allSatisfy(\.isFinite), abs(x.dx * y.dy - x.dy * y.dx) > 1 else { return }

            overlayLayer.position = .zero
            overlayLayer.transform = CATransform3DMakeAffineTransform(
                CGAffineTransform(a: x.dx, b: x.dy, c: y.dx, d: y.dy, tx: origin.x, ty: origin.y))
            onCameraAxes(CameraAxes.deviceFromCamera(bufferXOnScreen: x, bufferYOnScreen: y))
        }
    }
}
