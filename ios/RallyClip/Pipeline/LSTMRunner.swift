import Foundation
import onnxruntime

/// TennisPointLSTM on onnxruntime (CPU — the LSTM is slower on CoreML, so the
/// desktop keeps it on CPU; see the v0.3.1 manifest / CoreML spike note).
/// Ports `run_windowed_inference_average_onnx`'s per-window step: run → squeeze
/// → sigmoid. The averaging over overlapping windows lives in `Postprocess`.
final class LSTMRunner {
    private let env: ORTEnv
    private let session: ORTSession
    private let inputName: String
    private let outputName: String
    let seqLen: Int
    let featureDim: Int

    init(contract: ModelContract) throws {
        self.env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
        guard let url = Bundle.main.url(forResource: "model", withExtension: "onnx") else {
            throw PipelineError.missingAsset("model.onnx")
        }
        self.session = try ORTSession(env: env, modelPath: url.path, sessionOptions: try ORTSessionOptions())
        self.inputName = (try? session.inputNames().first) ?? contract.inputName
        self.outputName = (try? session.outputNames().first) ?? contract.outputName
        self.seqLen = contract.seqLen
        self.featureDim = contract.featureDim
    }

    /// window: seqLen × featureDim (already scaled) → seqLen probabilities.
    func runWindow(_ window: [[Float]]) throws -> [Float] {
        var flat = [Float](); flat.reserveCapacity(seqLen * featureDim)
        for row in window { flat.append(contentsOf: row) }
        let data = NSMutableData(bytes: flat, length: flat.count * MemoryLayout<Float>.size)
        let shape: [NSNumber] = [1, NSNumber(value: seqLen), NSNumber(value: featureDim)]
        let input = try ORTValue(tensorData: data, elementType: .float, shape: shape)
        let outputs = try session.run(withInputs: [inputName: input],
                                      outputNames: [outputName], runOptions: nil)
        guard let out = outputs[outputName] else { throw PipelineError.decode("lstm: no output") }
        let raw = try out.tensorData() as Data
        let count = raw.count / MemoryLayout<Float>.size
        var logits = [Float](repeating: 0, count: count)
        _ = logits.withUnsafeMutableBytes { raw.copyBytes(to: $0) }
        if logits.count != seqLen, logits.count % seqLen == 0 {
            logits = Array(logits.prefix(seqLen))   // squeeze leading batch dim
        }
        return Postprocess.sigmoid(logits)
    }
}
