import CoreMedia
import CoreVideo
import Foundation
import simd

/// Saves every depth frame of a scan, plus gravity and the tracker's per-frame verdicts, so the scan can be replayed
/// on a Mac with Tools/ScanBench. Layout of the directory:
///   capture.json  {"version":1, "mode", "width", "height", "deviceFromCamera": [9 floats, column-major] | null}
///   frames.bin    per frame, little-endian: Float64 time, Float32 fx fy cx cy, Float32 gravity xyz (device axes,
///                 NaN if unknown), UInt32 byte count, then LZFSE-compressed Float16 depth rows (meters)
///   log.csv       time,tracking,ms,rotation,reason,stages — one line per frame the phone actually processed
final class CaptureRecorder: @unchecked Sendable {
    let directory: URL
    private let mode: ScanMode
    private let deviceFromCamera: simd_float3x3?
    private let queue = DispatchQueue(label: "depth2print.recorder", qos: .utility)
    private let lock = NSLock()
    private var backlog = 0
    private var frames: FileHandle?
    private var log: FileHandle?

    init(directory: URL, mode: ScanMode, deviceFromCamera: simd_float3x3?) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        self.directory = directory
        self.mode = mode
        self.deviceFromCamera = deviceFromCamera
        for name in ["frames.bin", "log.csv"] {
            FileManager.default.createFile(atPath: directory.appendingPathComponent(name).path, contents: nil)
        }
        frames = try FileHandle(forWritingTo: directory.appendingPathComponent("frames.bin"))
        log = try FileHandle(forWritingTo: directory.appendingPathComponent("log.csv"))
        log?.write(Data("time,tracking,ms,rotation,reason,stages\n".utf8))
    }

    /// Called on the capture queue. Copies the depth right away so the camera can recycle its buffer.
    func append(_ frame: DepthFrame) {
        // About 2 s of frames; beyond that the disk can't keep up, so drop rather than grow memory.
        let accepted = lock.withLock { () -> Bool in
            guard backlog < 60 else { return false }
            backlog += 1
            return true
        }
        guard accepted else { return }
        let buffer = frame.depth
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        var depth = Data(capacity: width * height * 2)
        if let base = CVPixelBufferGetBaseAddress(buffer) {
            for row in 0..<height { depth.append(base.advanced(by: row * rowBytes).assumingMemoryBound(to: UInt8.self), count: width * 2) }
        }
        CVPixelBufferUnlockBaseAddress(buffer, .readOnly)

        let k = frame.intrinsics
        let gravity = frame.gravity ?? SIMD3(repeating: .nan)
        let time = frame.timestamp.seconds
        queue.async { [self] in
            defer { lock.withLock { backlog -= 1 } }
            if !FileManager.default.fileExists(atPath: directory.appendingPathComponent("capture.json").path) {
                writeHeader(width: width, height: height)
            }
            guard let compressed = try? (depth as NSData).compressed(using: .lzfse) as Data else { return }
            var record = Data()
            withUnsafeBytes(of: time.bitPattern.littleEndian) { record.append(contentsOf: $0) }
            for value in [k[0][0], k[1][1], k[2][0], k[2][1], gravity.x, gravity.y, gravity.z] {
                withUnsafeBytes(of: value.bitPattern.littleEndian) { record.append(contentsOf: $0) }
            }
            withUnsafeBytes(of: UInt32(compressed.count).littleEndian) { record.append(contentsOf: $0) }
            record.append(compressed)
            frames?.write(record)
        }
    }

    func log(_ status: TrackingStatus, time: Double) {
        let reason = (status.failureReason ?? "").replacingOccurrences(of: ",", with: ";")
        let line = String(format: "%.4f,%d,%.1f,%.1f,%@,%@\n", time, status.isTracking ? 1 : 0,
                          status.processingMilliseconds, status.rotationDegrees, reason, status.stageTimings)
        queue.async { [self] in log?.write(Data(line.utf8)) }
    }

    func finish() {
        queue.sync {
            try? frames?.close()
            try? log?.close()
            frames = nil
            log = nil
        }
    }

    private func writeHeader(width: Int, height: Int) {
        var header: [String: Any] = ["version": 1, "mode": mode.rawValue, "width": width, "height": height]
        if let m = deviceFromCamera {
            header["deviceFromCamera"] = [m.columns.0, m.columns.1, m.columns.2].flatMap { [$0.x, $0.y, $0.z] }
        }
        if let json = try? JSONSerialization.data(withJSONObject: header, options: [.prettyPrinted, .sortedKeys]) {
            try? json.write(to: directory.appendingPathComponent("capture.json"))
        }
    }
}
