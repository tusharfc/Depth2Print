import AVFoundation
import AudioToolbox
import DepthEngine
import UIKit

/// Spoken prompts, ticks and vibration, so a scan can be guided while the screen faces away from the user.
@MainActor
final class ScanCues {
    var isEnabled = true

    private let speech = AVSpeechSynthesizer()
    private let haptics = UINotificationFeedbackGenerator()
    private var mode = ScanMode.turntable
    private var wasTracking = true
    private var nextMilestone: Float = 90
    private var faceStage = 0
    private var firstSide: Float = 0
    private var firstTilt: Float = 0
    private var lastSaid: [String: Date] = [:]

    // On a tripod a vibration would shake the camera mid-scan.
    private var vibrates: Bool { mode != .turntable }

    private var maxDegreesPerSecond: Float {
        switch mode {
        case .turntable: return 35
        case .handheld: return 25
        case .face: return 40
        case .body: return 20
        }
    }

    func begin(_ mode: ScanMode) {
        self.mode = mode
        wasTracking = true
        nextMilestone = 90
        faceStage = 0
        lastSaid = [:]
        guard isEnabled else { return }
        // Playback so prompts are heard with the ring switch on silent; others' audio is ducked, not stopped.
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .voicePrompt, options: [.duckOthers])
        try? AVAudioSession.sharedInstance().setActive(true)
        haptics.prepare()
        switch mode {
        case .turntable: say("Start turning slowly.")
        case .handheld: say("Walk slowly around the object.")
        case .face: say("Hold still, then slowly turn your head to one side.")
        case .body: say("Walk slowly around the person.")
        }
    }

    func update(_ status: TrackingStatus) {
        guard isEnabled else { return }
        guard status.isTracking else {
            if wasTracking { vibrate(.warning) }
            say(mode == .turntable ? "Tracking lost. Turn back a little." : "Tracking lost. Go back slowly.",
                key: "lost", every: 4)
            wasTracking = false
            return
        }
        if !wasTracking {
            tick()
            say("Tracking again.", key: "regained", every: 2)
        }
        wasTracking = true

        if status.degreesPerSecond > maxDegreesPerSecond { say("Slow down.", key: "slow", every: 3) }

        if mode == .face {
            guideFace(status)
        } else if status.rotationDegrees >= nextMilestone {
            tick()
            switch nextMilestone {
            case 90: say("Quarter turn.")
            case 180: say("Halfway.")
            case 270: say("Three quarters.")
            default:
                say("Full turn. You can finish.")
                vibrate(.success)
            }
            nextMilestone += 90
        }
    }

    func end() {
        speech.stopSpeaking(at: .immediate)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func guideFace(_ status: TrackingStatus) {
        // Without gravity the heading stays 0; fall back to unsigned rotation for the first side only.
        let heading = status.headingDegrees != 0 ? status.headingDegrees : status.rotationDegrees
        switch faceStage {
        case 0 where abs(heading) >= 45:
            firstSide = heading < 0 ? -1 : 1
            tick()
            say("Now slowly turn to the other side.")
            faceStage = 1
        case 1 where heading * firstSide <= -45:
            tick()
            say("Now face the camera and slowly tilt your chin up.")
            faceStage = 2
        case 2 where abs(status.tiltDegrees) >= 20:
            firstTilt = status.tiltDegrees < 0 ? -1 : 1
            tick()
            say("Now slowly tilt your chin down.")
            faceStage = 3
        case 3 where status.tiltDegrees * firstTilt <= -20:
            tick()
            say("Good. Face the camera, then tap Finish.")
            vibrate(.success)
            faceStage = 4
        default:
            break
        }
    }

    private func say(_ text: String, key: String? = nil, every interval: TimeInterval = 0) {
        let key = key ?? text
        if let last = lastSaid[key], Date().timeIntervalSince(last) < interval { return }
        lastSaid[key] = Date()
        speech.stopSpeaking(at: .word)
        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 1.05
        speech.speak(utterance)
    }

    private func tick() {
        AudioServicesPlaySystemSound(1104)
    }

    private func vibrate(_ type: UINotificationFeedbackGenerator.FeedbackType) {
        guard vibrates else { return }
        haptics.notificationOccurred(type)
        haptics.prepare()
    }
}
