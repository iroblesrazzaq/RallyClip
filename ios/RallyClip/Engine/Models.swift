import Foundation

/// The five pipeline stages surfaced in the progress UI — same set and order as
/// the desktop (`script.js` `steps` / backend ProgressEvent stages).
enum ProgressStage: String, CaseIterable, Codable {
    case pose, preprocess, feature, inference, output

    var label: String {
        switch self {
        case .pose: return "Pose extraction"
        case .preprocess: return "Preprocessing"
        case .feature: return "Features"
        case .inference: return "Inference"
        case .output: return "Output"
        }
    }
    /// The active-stage headline verbs mirror `script.js` stepLabels.
    var activeLabel: String {
        switch self {
        case .pose: return "Extracting pose"
        case .preprocess: return "Preprocessing"
        case .feature: return "Building features"
        case .inference: return "Finding points"
        case .output: return "Saving match"
        }
    }
}

enum StageStatus: String, Codable { case waiting, inProgress, completed, failed, cancelled }

struct ProgressEvent {
    var stage: ProgressStage
    var progress: Int          // 0–100 within the stage
    var status: StageStatus
}

/// Run parameters — the Advanced-panel fields. Defaults come from the manifest
/// (`postprocess.params`); device nil = auto.
struct AnalysisConfig {
    var device: PoseDevice?
    var low: Double
    var high: Double
    var minDurSec: Double
    var sigma: Double
    var outputName: String?

    static func defaults(_ c: ModelContract) -> AnalysisConfig {
        AnalysisConfig(device: nil, low: c.low, high: c.high, minDurSec: c.minDurSec, sigma: c.sigma, outputName: nil)
    }
    var resolvedDevice: PoseDevice { device ?? PoseDevice.auto }
}

/// Persisted per-match metadata (`meta.json`).
struct MatchMeta: Codable, Identifiable, Equatable {
    var id: String
    var name: String
    var createdISO: String
    var durationS: Double
    var sourceFilename: String
    var nSegments: Int
    var pointDurationS: Double
    var hasEdits: Bool

    /// "3 points · 42s points · 300s video · <date>" — mirrors `cardMeta`.
    var metaLine: String {
        var parts: [String] = []
        parts.append("\(nSegments) point\(nSegments == 1 ? "" : "s")")
        parts.append("\(Int(pointDurationS.rounded()))s points")
        parts.append("\(Int(durationS.rounded()))s video")
        return parts.joined(separator: " · ")
    }
}
