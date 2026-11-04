import DepthEngine
import UIKit
import simd

@MainActor
final class ScanModel: ObservableObject {
    enum Phase: Equatable {
        case starting
        case ready
        case scanning
        case processing
        case done(URL)
        case failed(String)
    }

    @Published private(set) var phase: Phase = .starting
    @Published private(set) var tracking: TrackingStatus?
    @Published private(set) var preview: CGImage?
    @Published var format: MeshExportFormat = .stl
    @Published var mode: ScanMode = .turntable
    @Published var cuesEnabled = true
    @Published var recordsCapture = false
    /// Zipped raw capture of the last scan, for replaying it on a Mac (Tools/ScanBench).
    @Published private(set) var captureArchive: URL?
    /// Set once the current scan has been kept with Save to Files.
    @Published private(set) var savedScan: URL?
    /// The finished scan, for the in-app 3D viewer.
    @Published private(set) var mesh: Mesh?

    let scanner = DepthScanner()
    /// Set by the camera preview; not published since it never drives the UI.
    var cameraAxes: simd_float3x3?
    private let cues = ScanCues()
    private var captureDirectory: URL?

    init() {
        scanner.onTrackingUpdate = { [weak self] status in
            Task { @MainActor in self?.apply(status) }
        }
        Self.clearWork()
        Self.removeLegacyAutoSaves()
    }

    private func apply(_ status: TrackingStatus) {
        // Frames still in flight when the scan ends are ignored.
        guard phase == .scanning else { return }
        tracking = status
        if let image = status.preview { preview = image }
        cues.update(status)
    }

    func startPreview() async {
        phase = .starting
        do {
            try await scanner.startPreview()
            phase = .ready
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func startScan() {
        // Whatever the last scan left behind and wasn't saved goes now.
        Self.clearWork()
        captureArchive = nil
        savedScan = nil
        mesh = nil
        do {
            captureDirectory = recordsCapture
                ? try Self.work().appendingPathComponent("Capture-\(Self.timestamp())", isDirectory: true)
                : nil
            try scanner.beginScan(mode: mode, deviceFromCamera: cameraAxes, recordingTo: captureDirectory)
        } catch {
            phase = .failed(error.localizedDescription)
            return
        }
        tracking = nil
        preview = nil
        cues.isEnabled = cuesEnabled
        cues.begin(mode)
        // Phone sits untouched on a tripod; don't let it auto-lock mid-scan.
        UIApplication.shared.isIdleTimerDisabled = true
        phase = .scanning
    }

    func finishScan() async {
        phase = .processing
        cues.end()
        preview = nil
        UIApplication.shared.isIdleTimerDisabled = false
        let format = format
        do {
            let mesh = try await scanner.finishScan()
            let url = try Self.work().appendingPathComponent("Scan-\(Self.timestamp()).\(format.fileExtension)")
            try await Task.detached(priority: .userInitiated) {
                try MeshExporter.write(mesh, format: format, to: url)
            }.value
            self.mesh = mesh
            phase = .done(url)
        } catch {
            phase = .failed(error.localizedDescription)
        }
        // Kept for failed scans too: those are the ones most worth replaying.
        if let captureDirectory {
            captureArchive = try? await Task.detached(priority: .utility) {
                let archive = try Self.zip(captureDirectory)
                // The zip holds everything; keeping the folder too would double the space used.
                try? FileManager.default.removeItem(at: captureDirectory)
                return archive
            }.value
        }
    }

    /// Moves the finished scan into Documents, where the Files app shows it under Depth2Print.
    func saveScan() {
        guard case .done(let url) = phase, savedScan == nil else { return }
        do {
            let destination = try Self.documents().appendingPathComponent(url.lastPathComponent)
            try FileManager.default.moveItem(at: url, to: destination)
            savedScan = destination
            phase = .done(destination)
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func reset() {
        Self.clearWork()
        captureArchive = nil
        savedScan = nil
        mesh = nil
        phase = .ready
    }

    /// Scratch space for exports and raw captures until the user decides to keep them.
    nonisolated private static func work() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("Scans", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    nonisolated private static func clearWork() {
        try? FileManager.default.removeItem(at: FileManager.default.temporaryDirectory.appendingPathComponent("Scans"))
    }

    /// Earlier builds saved every scan and raw capture straight into Documents; clear those out once.
    private static func removeLegacyAutoSaves() {
        let key = "removedLegacyAutoSaves"
        guard !UserDefaults.standard.bool(forKey: key), let documents = try? documents() else { return }
        let items = (try? FileManager.default.contentsOfDirectory(at: documents, includingPropertiesForKeys: nil)) ?? []
        for item in items where item.lastPathComponent.hasPrefix("Scan-") || item.lastPathComponent == "Captures" {
            try? FileManager.default.removeItem(at: item)
        }
        UserDefaults.standard.set(true, forKey: key)
    }

    // Documents is exposed in the Files app via UIFileSharingEnabled + LSSupportsOpeningDocumentsInPlace.
    private static func documents() throws -> URL {
        try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
    }

    private static func timestamp() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter.string(from: Date())
    }

    /// Uses the system's zip-for-upload (no third-party code); the temporary archive is copied next to the folder.
    nonisolated private static func zip(_ directory: URL) throws -> URL {
        let destination = directory.appendingPathExtension("zip")
        var coordinationError: NSError?
        var copyError: Error?
        NSFileCoordinator().coordinate(readingItemAt: directory, options: .forUploading, error: &coordinationError) { zipped in
            do {
                try? FileManager.default.removeItem(at: destination)
                try FileManager.default.copyItem(at: zipped, to: destination)
            } catch {
                copyError = error
            }
        }
        if let error = coordinationError ?? copyError { throw error }
        return destination
    }
}
