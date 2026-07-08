import Foundation

/// JSON StandardScaler: `(x - mean) / max(scale, 1e-12)`.
/// Ports `infer/inference.py` `JsonStandardScaler` + `load_scaler_asset`.
struct StandardScaler {
    let mean: [Float]
    let scale: [Float]

    static func loadBundled() throws -> StandardScaler {
        guard let url = Bundle.main.url(forResource: "scaler", withExtension: "json") else {
            throw PipelineError.missingAsset("scaler.json")
        }
        let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
        let mean = (root["mean"] as? [NSNumber])?.map { $0.floatValue } ?? []
        let scale = (root["scale"] as? [NSNumber])?.map { $0.floatValue } ?? []
        guard !mean.isEmpty, !scale.isEmpty else { throw PipelineError.missingAsset("scaler.json (empty)") }
        return StandardScaler(mean: mean, scale: scale)
    }

    /// In-place transform of one feature row (length == mean.count).
    func transform(_ row: [Float]) -> [Float] {
        var out = row
        let n = min(out.count, mean.count)
        for i in 0..<n {
            let s = scale[i] > 1e-12 ? scale[i] : 1e-12
            out[i] = (out[i] - mean[i]) / s
        }
        return out
    }
}
