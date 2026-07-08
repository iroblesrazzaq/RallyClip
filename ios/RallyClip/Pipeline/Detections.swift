import Foundation

/// One person detection in reference-resolution pixels.
/// Ports the per-player dict used across `preprocessing/data_preprocessor.py`
/// and `features/feature_engineer.py` (`box`, `keypoints`, `conf`, `box_conf`).
struct Player: Equatable {
    var box: [Float]        // 4: x1, y1, x2, y2
    var keypoints: [Float]  // 34: x0,y0, x1,y1, ... x16,y16  (17 keypoints, flattened)
    var conf: [Float]       // 17: per-keypoint confidence
    var boxConf: Float

    static func kp(_ p: [Float], _ i: Int) -> (Float, Float) { (p[2 * i], p[2 * i + 1]) }
}

/// Raw per-frame pose output — the four parallel arrays `yolo_onnx_runner.YOLO`
/// hands back (boxes / box_conf / keypoints xy / keypoint conf).
struct PoseDetections {
    var boxes: [[Float]]        // N x 4
    var boxConf: [Float]        // N
    var keypoints: [[Float]]    // N x 34
    var kptConf: [[Float]]      // N x 17

    var count: Int { boxes.count }
    static let empty = PoseDetections(boxes: [], boxConf: [], keypoints: [], kptConf: [])
}

/// A kept point interval, in source seconds. Mirrors the `{start, end}` objects
/// the frontend timeline works in.
struct Segment: Equatable, Codable, Identifiable {
    var start: Double
    var end: Double
    var id: String { "\(start)-\(end)" }
}
