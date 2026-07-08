import Foundation
import CoreVideo
import CoreGraphics

/// Swift side of court detection: picks sample timestamps, runs the pose model
/// as the person detector, chooses a clean reference frame, and hands frames to
/// `RCCourtDetector`. Ports `data_preprocessor.compute_court_mask` +
/// `court_detector_impl.extract_clean_frame` (reference-frame search).
///
/// PARITY: person boxes here come from the main pose runner (imgsz 960), whereas
/// the desktop court detector runs the dynamic model at its default imgsz 640.
/// This only affects occlusion coverage for the homography clean-frame, not the
/// line geometry — verify the resulting masks against the court fixtures.
struct CourtDetectorDriver {
    let decoder: VideoDecoder
    let pose: PoseRunner

    func computeCourtMask() async -> CourtMask {
        let duration = decoder.durationSeconds
        for t in sampleTimes(duration: duration) {
            guard let base = await decoder.frame(at: Double(t)) else { continue }
            let baseBoxes = personRects(base)
            let ref = await findReferenceFrame(targetTime: t, baseBoxes: baseBoxes, duration: duration)
            let result = RCCourtDetector.detect(withBaseFrame: base, baseBoxes: baseBoxes, referenceFrame: ref)
            if result.success { return mask(from: result) }
        }
        // Fallback: the empirical default mask, resized to the source frame.
        let path = Bundle.main.path(forResource: "default_court_mask", ofType: "png") ?? ""
        let def = RCCourtDetector.defaultMask(fromPNGPath: path,
                                              width: Int32(decoder.sourceWidth),
                                              height: Int32(decoder.sourceHeight))
        return mask(from: def)
    }

    // Timestamps to try, spread across the video. Mirrors `_court_sample_times`.
    private func sampleTimes(duration: Double) -> [Int] {
        if duration > 10 {
            let fracs = [0.2, 0.35, 0.5, 0.65, 0.8]
            return Array(Set(fracs.map { max(1, Int(duration * $0)) })).sorted()
        }
        return [60, 90, 45, 120, 30]
    }

    /// Search ±15 s for a frame where players don't occlude the base occlusions.
    /// Mirrors the reference-frame search in `extract_clean_frame`.
    private func findReferenceFrame(targetTime t: Int, baseBoxes: [NSValue], duration: Double) async -> CVPixelBuffer? {
        for st in [t - 15, t + 15] {
            guard st >= 0, Double(st) < duration, let cand = await decoder.frame(at: Double(st)) else { continue }
            let candBoxes = personRects(cand)
            if isSuitable(baseBoxes: baseBoxes, candBoxes: candBoxes) { return cand }
        }
        return nil
    }

    /// Suitable if no candidate box overlaps > 30% of any base occlusion box.
    private func isSuitable(baseBoxes: [NSValue], candBoxes: [NSValue]) -> Bool {
        for bv in baseBoxes {
            let b = bv.cgRectValue
            let baseArea = b.width * b.height
            for cv in candBoxes {
                let c = cv.cgRectValue
                let ox = max(0, min(b.maxX, c.maxX) - max(b.minX, c.minX))
                let oy = max(0, min(b.maxY, c.maxY) - max(b.minY, c.minY))
                if ox * oy > 0.3 * baseArea { return false }
            }
        }
        return true
    }

    private func personRects(_ pb: CVPixelBuffer) -> [NSValue] {
        guard let dets = try? pose.infer(pixelBuffer: pb) else { return [] }
        return (0..<dets.count).map {
            let b = dets.boxes[$0]
            return NSValue(cgRect: CGRect(x: CGFloat(b[0]), y: CGFloat(b[1]),
                                          width: CGFloat(b[2] - b[0]), height: CGFloat(b[3] - b[1])))
        }
    }

    private func mask(from r: RCCourtResult) -> CourtMask {
        let data = [UInt8](r.mask)
        return CourtMask(width: Int(r.width), height: Int(r.height), data: data)
    }
}
