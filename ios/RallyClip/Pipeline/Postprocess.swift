import Foundation

/// Windowed-average inference driver + hysteresis postprocess.
/// Ports `infer/inference.py`: `generate_start_indices`,
/// `run_windowed_inference_average_onnx`, `gaussian_filter1d`,
/// `hysteresis_threshold`, `extract_segments_from_binary`.
enum Postprocess {

    /// Overlapping window starts. Matches `generate_start_indices` exactly,
    /// including the end-anchored trailing window.
    static func startIndices(numFrames: Int, seqLen: Int, overlap: Int) throws -> [Int] {
        guard seqLen > 0 else { throw PipelineError.decode("sequence_length must be > 0") }
        guard overlap >= 0, overlap < seqLen else { throw PipelineError.decode("bad overlap") }
        guard numFrames >= seqLen else { throw PipelineError.shortVideo }
        let step = seqLen - overlap
        var starts: [Int] = []
        var idx = 0
        while idx + seqLen <= numFrames { starts.append(idx); idx += step }
        if let last = starts.last, last + seqLen < numFrames { starts.append(numFrames - seqLen) }
        return starts
    }

    /// Averaged per-frame probability over overlapping windows.
    /// `runWindow` maps a `seqLen × featureDim` window to `seqLen` probabilities
    /// (the LSTM forward + sigmoid) — supplied by `LSTMRunner`.
    static func windowedAverage(
        features: [[Float]],
        seqLen: Int,
        overlap: Int,
        runWindow: ([[Float]]) throws -> [Float]
    ) throws -> [Float] {
        let n = features.count
        let starts = try startIndices(numFrames: n, seqLen: seqLen, overlap: overlap)
        var summed = [Float](repeating: 0, count: n)
        var counts = [Int](repeating: 0, count: n)
        for start in starts {
            let window = Array(features[start..<(start + seqLen)])
            let probs = try runWindow(window)
            guard probs.count == seqLen else {
                throw PipelineError.decode("window output length \(probs.count) != \(seqLen)")
            }
            for k in 0..<seqLen { summed[start + k] += probs[k]; counts[start + k] += 1 }
        }
        return (0..<n).map { summed[$0] / Float(max(counts[$0], 1)) }
    }

    static func sigmoid(_ x: [Float]) -> [Float] { x.map { 1.0 / (1.0 + expf(-$0)) } }

    /// 1D Gaussian smoothing with edge padding. Matches `gaussian_filter1d`.
    static func gaussianFilter1d(_ data: [Float], sigma: Double) -> [Float] {
        guard sigma > 0, !data.isEmpty else { return data }
        let radius = Int(3.0 * sigma + 0.5)
        var kernel = [Float](repeating: 0, count: 2 * radius + 1)
        var sum: Float = 0
        for (k, x) in stride(from: -radius, through: radius, by: 1).enumerated() {
            let v = expf(-0.5 * Float(Double(x) / sigma) * Float(Double(x) / sigma))
            kernel[k] = v; sum += v
        }
        for k in 0..<kernel.count { kernel[k] /= sum }
        let n = data.count
        func padded(_ i: Int) -> Float { data[min(max(i, 0), n - 1)] }
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var acc: Float = 0
            for (k, off) in stride(from: -radius, through: radius, by: 1).enumerated() {
                acc += padded(i + off) * kernel[k]
            }
            out[i] = acc
        }
        return out
    }

    /// Two-threshold hysteresis with a minimum active duration (frames).
    /// Matches `hysteresis_threshold`.
    static func hysteresis(_ values: [Float], low: Float, high: Float, minDuration: Int) -> [Int8] {
        let n = values.count
        var pred = [Int8](repeating: 0, count: n)
        var active = false
        var startIdx: Int? = nil
        for i in 0..<n {
            let v = values[i]
            if !active {
                if v >= high { active = true; startIdx = i }
            } else if v < low {
                if let s = startIdx, i - s >= max(0, minDuration) {
                    for k in s..<i { pred[k] = 1 }
                }
                active = false; startIdx = nil
            }
        }
        if active, let s = startIdx, n - s >= max(0, minDuration) {
            for k in s..<n { pred[k] = 1 }
        }
        return pred
    }

    /// Contiguous 1-runs → (startFrame, endFrame). Matches `extract_segments_from_binary`.
    static func extractSegments(_ pred: [Int8]) -> [(Int, Int)] {
        var segs: [(Int, Int)] = []
        var inSeg = false
        var start = 0
        for i in 0..<pred.count {
            if !inSeg, pred[i] == 1 { inSeg = true; start = i }
            else if inSeg, pred[i] == 0 { segs.append((start, i)); inSeg = false }
        }
        if inSeg { segs.append((start, pred.count)) }
        return segs
    }

    /// Full postprocess: smooth → hysteresis → segments → seconds intervals.
    static func segments(from probs: [Float], contract: ModelContract, fps: Double,
                         low: Double, high: Double, minDurSec: Double, sigma: Double) -> [Segment] {
        let smoothed = gaussianFilter1d(probs, sigma: sigma)
        let minFrames = Int((max(0.0, minDurSec) * fps).rounded())
        let pred = hysteresis(smoothed, low: Float(low), high: Float(high), minDuration: minFrames)
        return extractSegments(pred).map {
            Segment(start: Double($0.0) / fps, end: Double($0.1) / fps)
        }
    }
}
