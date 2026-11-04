import Foundation

public enum DepthEngineError: LocalizedError {
    case cameraAccessDenied
    case trueDepthUnavailable
    case noFloat16DepthFormat
    case cannotAdd(String)
    case metalUnavailable
    case shaderMissing(String)
    case textureCreationFailed
    case noDepthData
    case emptyScan
    case turntableRotationNotSeen

    public var errorDescription: String? {
        switch self {
        case .cameraAccessDenied: return "Camera access was denied. Enable it in Settings > Privacy > Camera."
        case .trueDepthUnavailable: return "This device has no TrueDepth camera."
        case .noFloat16DepthFormat: return "The TrueDepth camera offers no Float16 depth format."
        case .cannotAdd(let what): return "Could not add \(what) to the capture session."
        case .metalUnavailable: return "Metal is unavailable on this device."
        case .shaderMissing(let name): return "Metal shader '\(name)' is missing from DepthEngine."
        case .textureCreationFailed: return "Could not wrap the depth map in a Metal texture."
        case .noDepthData: return "No depth frames were received. Is the object within ~20–100 cm?"
        case .emptyScan: return "No surface was reconstructed. Keep the object 15–70 cm from the camera and rotate it slowly."
        case .turntableRotationNotSeen:
            return "The turntable's rotation couldn't be seen, which happens with round objects like vases. Put a small item (e.g. a LEGO brick) near the turntable's edge as a marker and scan again."
        }
    }
}
