import Foundation
import UIKit
import CoreImage

/// Runs the full on-device pipeline for one match and emits stage progress.
/// Ports `rallyclip_engine/models.py` `FrameProbabilityHysteresisModel.run`
/// (preprocess → infer → postprocess) plus the CLI's save-to-library step.
///
/// Cancellation: cooperative — checks `Task.isCancelled` at frame/window
/// boundaries and throws `PipelineError.cancelled`.
final class AnalysisJob {
    let sourceURL: URL
    let displayName: String
    let config: AnalysisConfig
    let contract: ModelContract

    init(sourceURL: URL, displayName: String, config: AnalysisConfig, contract: ModelContract) {
        self.sourceURL = sourceURL
        self.displayName = displayName
        self.config = config
        self.contract = contract
    }

    /// Returns the saved match, or nil if no points were detected.
    func run(progress: @escaping (ProgressEvent) -> Void) async throws -> MatchMeta? {
        func emit(_ s: ProgressStage, _ p: Int, _ st: StageStatus = .inProgress) { progress(ProgressEvent(stage: s, progress: p, status: st)) }

        let decoder = try await VideoDecoder(url: sourceURL)
        let pose = try PoseRunner(contract: contract, device: config.resolvedDevice)
        let scaler = try StandardScaler.loadBundled()
        let lstm = try LSTMRunner(contract: contract)

        // --- court detection (front-loaded, like the desktop) ---
        emit(.pose, 1)
        let courtMask = await CourtDetectorDriver(decoder: decoder, pose: pose).computeCourtMask()
        try Task.checkCancellation()
        emit(.pose, 3)

        // --- pose → preprocess → features (streamed per sampled frame) ---
        let fps = contract.sampleFps
        let approxTotal = max(1, Int((decoder.durationSeconds * fps).rounded()) - 1)
        let fe = FeatureEngineer(targetFps: fps)
        let pre = Preprocessor(screenWidth: contract.screenWidth, screenHeight: contract.screenHeight)
        var scaledRows: [[Float]] = []
        scaledRows.reserveCapacity(approxTotal)
        let srcW = decoder.sourceWidth, srcH = decoder.sourceHeight

        var poseError: Error?
        try await decoder.sampledFrames(fps: fps) { pb, _, index in
            if Task.isCancelled { return false }
            do {
                let dets = try pose.infer(pixelBuffer: pb)
                let (near, far) = pre.process(dets, mask: courtMask, srcWidth: srcW, srcHeight: srcH)
                let feat = fe.build(near: near, far: far)
                scaledRows.append(scaler.transform(feat))
            } catch { poseError = error; return false }
            if index % 8 == 0 {
                let frac = min(1.0, Double(index) / Double(approxTotal))
                emit(.pose, Int(3 + frac * 96))
                emit(.preprocess, Int(1 + frac * 94))
                emit(.feature, Int(1 + frac * 94))
            }
            return true
        }
        if let poseError { throw poseError }
        try Task.checkCancellation()
        emit(.pose, 100, .completed); emit(.preprocess, 100, .completed); emit(.feature, 100, .completed)

        guard scaledRows.count >= contract.seqLen else { throw PipelineError.shortVideo }

        // --- windowed LSTM inference + hysteresis postprocess ---
        emit(.inference, 5)
        let starts = try Postprocess.startIndices(numFrames: scaledRows.count, seqLen: contract.seqLen, overlap: contract.overlap)
        var fired = 0
        let probs = try Postprocess.windowedAverage(
            features: scaledRows, seqLen: contract.seqLen, overlap: contract.overlap
        ) { window in
            if Task.isCancelled { throw PipelineError.cancelled }
            let out = try lstm.runWindow(window)
            fired += 1
            emit(.inference, Int(5 + Double(fired) / Double(max(1, starts.count)) * 90))
            return out
        }
        emit(.inference, 100, .completed)

        let segments = Postprocess.segments(from: probs, contract: contract, fps: fps,
                                             low: config.low, high: config.high,
                                             minDurSec: config.minDurSec, sigma: config.sigma)

        // --- save to library ---
        emit(.output, 20)
        guard !segments.isEmpty else { emit(.output, 100, .completed); return nil }
        let thumb = await thumbnail(decoder: decoder, at: segments.first?.start ?? 0)
        emit(.output, 70)
        let name = config.outputName?.isEmpty == false
            ? config.outputName!
            : sourceURL.deletingPathExtension().lastPathComponent
        let meta = MatchStore.shared.create(sourceTempURL: sourceURL, name: name,
                                            durationS: decoder.durationSeconds, segments: segments,
                                            thumbnail: thumb)
        emit(.output, 100, .completed)
        return meta
    }

    private func thumbnail(decoder: VideoDecoder, at seconds: Double) async -> UIImage? {
        guard let pb = await decoder.frame(at: seconds) else { return nil }
        let ci = CIImage(cvPixelBuffer: pb)
        let ctx = CIContext()
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return nil }
        return UIImage(cgImage: cg)
    }
}
