import Foundation

/// The model contract, read from the bundled `manifest.json` — never hardcoded.
///
/// Ports the invariant enforced across the desktop runtime: pipeline parameters
/// (imgsz / fps / seq_len / thresholds) come from the artifact manifest, not code
/// (`rallyclip_core/pipelines.py`, `models/rallyclip_v0.3.1/manifest.json`,
/// AGENTS.md "model contract values come from the manifest").
struct ModelContract: Sendable {
    // feature_pipeline
    let conf: Double
    let featureDim: Int
    let imgsz: Int
    let numKeypoints: Int
    let sampleFps: Double
    let screenWidth: Int
    let screenHeight: Int
    let poseModelName: String

    // inference
    let inputName: String
    let outputName: String
    let seqLen: Int
    let overlap: Int

    // postprocess (hysteresis) — user-overridable defaults in the Advanced panel
    let low: Double
    let high: Double
    let minDurSec: Double
    let sigma: Double

    static func loadBundled() throws -> ModelContract {
        guard let url = Bundle.main.url(forResource: "manifest", withExtension: "json") else {
            throw PipelineError.missingAsset("manifest.json")
        }
        let data = try Data(contentsOf: url)
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let fp = root["feature_pipeline"] as? [String: Any] ?? [:]
        let inf = root["inference"] as? [String: Any] ?? [:]
        let pp = (root["postprocess"] as? [String: Any])?["params"] as? [String: Any] ?? [:]

        func d(_ dict: [String: Any], _ k: String, _ fallback: Double) -> Double {
            (dict[k] as? NSNumber)?.doubleValue ?? fallback
        }
        func i(_ dict: [String: Any], _ k: String, _ fallback: Int) -> Int {
            (dict[k] as? NSNumber)?.intValue ?? fallback
        }

        return ModelContract(
            conf: d(fp, "conf", 0.25),
            featureDim: i(fp, "feature_dim", 362),
            imgsz: i(fp, "imgsz", 960),
            numKeypoints: i(fp, "num_keypoints", 17),
            sampleFps: d(fp, "sample_fps", 5.0),
            screenWidth: i(fp, "screen_width", 1280),
            screenHeight: i(fp, "screen_height", 720),
            poseModelName: (fp["yolo_model"] as? String) ?? "yolov8n-pose-960-dynamic.onnx",
            inputName: (inf["input_name"] as? String) ?? "features",
            outputName: (inf["output_name"] as? String) ?? "logits",
            seqLen: i(inf, "seq_len_frames", 100),
            overlap: i(inf, "overlap_frames", 50),
            low: d(pp, "low", 0.45),
            high: d(pp, "high", 0.7),
            minDurSec: d(pp, "min_dur_sec", 1.0),
            sigma: d(pp, "sigma", 1.0)
        )
    }
}

enum PipelineError: LocalizedError {
    case missingAsset(String)
    case shortVideo
    case sessionInit(String)
    case decode(String)
    case export(String)
    case cancelled

    var errorDescription: String? {
        switch self {
        case .missingAsset(let n): return "Missing bundled asset: \(n)"
        case .shortVideo: return "This video is too short to analyze."
        case .sessionInit(let m): return "Could not initialize the model runtime: \(m)"
        case .decode(let m): return "Could not read the video: \(m)"
        case .export(let m): return "Could not export the clip: \(m)"
        case .cancelled: return "Cancelled."
        }
    }
}
