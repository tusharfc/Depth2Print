import DepthEngine
import SwiftUI

struct ContentView: View {
    @StateObject private var model = ScanModel()
    @State private var showsViewer = false

    var body: some View {
        ZStack(alignment: .bottom) {
            Group {
                if model.usesFaceTracking {
                    FaceTrackingPreview(session: model.scanner.faceSession, overlay: model.preview,
                                        onCameraAxes: { model.cameraAxes = $0 })
                } else {
                    CameraPreview(session: model.scanner.captureSession, overlay: model.preview,
                                  onCameraAxes: { model.cameraAxes = $0 })
                }
            }
            .ignoresSafeArea()

            controls
                .frame(maxWidth: .infinity)
                .padding()
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
                .padding()
        }
        .task { await model.startPreview() }
        .fullScreenCover(isPresented: $showsViewer) {
            if let mesh = model.mesh { MeshViewer(mesh: mesh) }
        }
    }

    @ViewBuilder
    private var controls: some View {
        VStack(spacing: 12) {
            switch model.phase {
            case .starting:
                ProgressView("Starting TrueDepth camera…")

            case .ready:
                Picker("Mode", selection: $model.mode) {
                    ForEach(ScanMode.allCases, id: \.self) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                Picker("Format", selection: $model.format) {
                    ForEach(MeshExportFormat.allCases, id: \.self) { format in
                        Text(format.rawValue.uppercased()).tag(format)
                    }
                }
                .pickerStyle(.segmented)
                Text(model.format == .obj ? "OBJ keeps colour (view in MeshLab or Blender)."
                                          : "STL is shape only; choose OBJ for colour.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Toggle("Voice & vibration guidance", isOn: $model.cuesEnabled)
                Toggle("Record raw capture (for debugging)", isOn: $model.recordsCapture)
                Button("Start Scan") { model.startScan() }
                    .buttonStyle(.borderedProminent)

            case .scanning:
                Text(model.mode.instructions)
                    .multilineTextAlignment(.center)
                if let tracking = model.tracking {
                    Text("Rotation \(Int(tracking.rotationDegrees.rounded()))°")
                        .font(.title2.monospacedDigit())
                    if !tracking.isTracking {
                        Text("Tracking lost. Slow down or move back to where tracking was last good.")
                            .foregroundStyle(.orange)
                            .multilineTextAlignment(.center)
                    }
                    Text(diagnostics(tracking))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Button("Finish Scan") { Task { await model.finishScan() } }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)

            case .processing:
                ProgressView("Building mesh…")

            case .done(let url):
                if model.mesh != nil {
                    Button { showsViewer = true } label: {
                        Label("View in 3D", systemImage: "cube")
                    }
                    .buttonStyle(.bordered)
                }
                if model.savedScan == nil {
                    Button { model.saveScan() } label: {
                        Label("Save to Files", systemImage: "square.and.arrow.down")
                    }
                    .buttonStyle(.borderedProminent)
                } else {
                    Label("Saved in Files › Depth2Print", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                }
                ShareLink(item: url) {
                    Label("Share \(url.lastPathComponent)", systemImage: "square.and.arrow.up")
                }
                captureShareLink
                Button("New Scan") { model.reset() }
                if model.savedScan == nil {
                    Text("Unsaved scans are deleted when you start a new one.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

            case .failed(let message):
                Text(message)
                    .multilineTextAlignment(.center)
                captureShareLink
                Button("Retry") { Task { await model.startPreview() } }
            }
        }
    }

    @ViewBuilder
    private var captureShareLink: some View {
        if let archive = model.captureArchive {
            ShareLink(item: archive) {
                Label("Share raw capture", systemImage: "waveform.path.ecg")
            }
        }
    }

    private func diagnostics(_ status: TrackingStatus) -> String {
        let ms = status.processingMilliseconds
        var text = String(format: "%.0f ms/frame (≤%.0f fps) · lost %d of %d", ms, 1000 / max(ms, 1),
                          status.framesLost, status.framesTracked + status.framesLost)
        text += "\n" + status.stageTimings
        if status.followsFace { text += " · face tracking" }
        if let reason = status.failureReason { text += "\n\(reason)" }
        return text
    }
}

private extension ScanMode {
    var title: String {
        switch self {
        case .turntable: return "Turntable"
        case .handheld: return "Handheld"
        case .face: return "Face"
        case .body: return "Body"
        }
    }

    var instructions: String {
        switch self {
        case .turntable: return "Turn the turntable slowly, one full turn."
        case .handheld: return "Walk the phone slowly around the object, 20–60 cm away."
        case .face: return "Hold the phone still. Turn your head left, then right, until each ear faces the phone, then tilt your chin up, then down. Move naturally; keep a neutral face."
        case .body: return "Stand still while someone slowly circles you with the phone, 50–120 cm away."
        }
    }
}
