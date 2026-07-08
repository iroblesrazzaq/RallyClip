import Foundation

/// Binary "out" mask in source-frame pixels (255 = outside the playable court).
/// Produced by `RCCourtDetector` (or the bundled default), consumed by the court
/// filter exactly as `data_preprocessor.filter_frame_by_court`.
struct CourtMask {
    let width: Int
    let height: Int
    let data: [UInt8]   // row-major, length width*height

    /// True where the pixel is "out" (255) — i.e. a detection centered here is dropped.
    func isOut(x: Int, y: Int) -> Bool {
        guard x >= 0, x < width, y >= 0, y < height else { return false }
        return data[y * width + x] != 0
    }
}

/// Court filter → clump merge → near/far assignment → reference rescale.
/// Ports the runtime path in `preprocessing/data_preprocessor.py`
/// (`iter_preprocess_frames` → `filter_frame_by_court` → `assign_players`
/// → `rescale_player_to_reference`).
///
/// NOTE (faithful quirk): assignment/merge compare source-pixel detections
/// against reference-space constants (`screenWidth/2`, `0.10/0.90/0.80` zones)
/// because rescale happens *after* assignment — identical to the desktop, where
/// it's a no-op at 720p (reference == source).
struct Preprocessor {
    let screenWidth: Int
    let screenHeight: Int
    let mergeIouThresh: Float

    var screenCenterX: Float { Float(screenWidth) / 2 }
    var leftZoneX: Float { Float(screenWidth) * 0.10 }
    var rightZoneX: Float { Float(screenWidth) * 0.90 }
    var bottomZoneY: Float { Float(screenHeight) * 0.80 }

    init(screenWidth: Int, screenHeight: Int, mergeIouThresh: Float = 0.6) {
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.mergeIouThresh = mergeIouThresh
    }

    /// One frame: filter by court, assign players, rescale to reference.
    func process(_ dets: PoseDetections, mask: CourtMask?, srcWidth: Int, srcHeight: Int) -> (near: Player?, far: Player?) {
        let filtered = filterByCourt(dets, mask: mask)
        let (near, far) = assignPlayers(filtered)
        return (
            rescale(near, srcWidth: srcWidth, srcHeight: srcHeight),
            rescale(far, srcWidth: srcWidth, srcHeight: srcHeight)
        )
    }

    // MARK: - court filter

    private func filterByCourt(_ dets: PoseDetections, mask: CourtMask?) -> PoseDetections {
        guard let mask else { return dets }
        var out = PoseDetections.empty
        for i in 0..<dets.count {
            let box = dets.boxes[i]
            let cx = Int((box[0] + box[2]) / 2)
            let cy = Int((box[1] + box[3]) / 2)
            // Keep detections whose center is inside the court (mask == 0).
            if cy >= 0, cy < mask.height, cx >= 0, cx < mask.width, !mask.isOut(x: cx, y: cy) {
                out.boxes.append(box)
                out.boxConf.append(dets.boxConf[i])
                out.keypoints.append(dets.keypoints[i])
                out.kptConf.append(dets.kptConf[i])
            }
        }
        return out
    }

    // MARK: - assignment (with clump merge)

    private func assignPlayers(_ dets: PoseDetections) -> (near: Player?, far: Player?) {
        let merged = conditionalMergeBoxes(dets)
        guard merged.count > 0 else { return (nil, nil) }
        var candidates: [Player] = (0..<merged.count).map {
            Player(box: merged.boxes[$0], keypoints: merged.keypoints[$0],
                   conf: merged.kptConf[$0], boxConf: merged.boxConf[$0])
        }
        // near = detection with the greatest bottom-y (box[3]).
        var nearIdx = 0
        for i in 1..<candidates.count where candidates[i].box[3] > candidates[nearIdx].box[3] { nearIdx = i }
        let near = candidates.remove(at: nearIdx)
        var far: Player?
        if !candidates.isEmpty {
            // far = remaining detection whose center-x is closest to screen center.
            var farIdx = 0
            func centerDist(_ p: Player) -> Float { abs((p.box[0] + p.box[2]) / 2 - screenCenterX) }
            for i in 1..<candidates.count where centerDist(candidates[i]) < centerDist(candidates[farIdx]) { farIdx = i }
            far = candidates[farIdx]
        }
        return (near, far)
    }

    private func iou(_ a: [Float], _ b: [Float]) -> Float {
        let x1 = max(a[0], b[0]), y1 = max(a[1], b[1])
        let x2 = min(a[2], b[2]), y2 = min(a[3], b[3])
        let inter = max(0, x2 - x1) * max(0, y2 - y1)
        let areaA = (a[2] - a[0]) * (a[3] - a[1])
        let areaB = (b[2] - b[0]) * (b[3] - b[1])
        let union = areaA + areaB - inter
        return union > 0 ? inter / union : 0
    }

    /// Clump detections by IoU; merge only clumps in the edge zones (partial
    /// bodies at the frame margins). Ports `_conditional_merge_boxes`.
    private func conditionalMergeBoxes(_ dets: PoseDetections) -> PoseDetections {
        guard dets.count > 1 else { return dets }
        var clumpId = [Int](repeating: -1, count: dets.count)
        var clumpCount = 0
        for i in 0..<dets.count where clumpId[i] == -1 {
            clumpId[i] = clumpCount
            for j in (i + 1)..<dets.count where iou(dets.boxes[i], dets.boxes[j]) > mergeIouThresh {
                clumpId[j] = clumpCount
            }
            clumpCount += 1
        }
        if clumpCount == dets.count { return dets }

        var out = PoseDetections.empty
        for c in 0..<clumpCount {
            let members = (0..<dets.count).filter { clumpId[$0] == c }
            if members.count == 1 {
                let m = members[0]
                out.boxes.append(dets.boxes[m]); out.keypoints.append(dets.keypoints[m])
                out.kptConf.append(dets.kptConf[m]); out.boxConf.append(dets.boxConf[m])
                continue
            }
            let minX1 = members.map { dets.boxes[$0][0] }.min()!
            let minY1 = members.map { dets.boxes[$0][1] }.min()!
            let maxX2 = members.map { dets.boxes[$0][2] }.max()!
            let maxY2 = members.map { dets.boxes[$0][3] }.max()!
            let centerX = (minX1 + maxX2) / 2
            let inEdgeZone = centerX < leftZoneX || centerX > rightZoneX || maxY2 > bottomZoneY
            if inEdgeZone {
                // Merge to the bounding box; keep the largest member's pose.
                let best = members.max {
                    let ba = dets.boxes[$0], bb = dets.boxes[$1]
                    return (ba[2] - ba[0]) * (ba[3] - ba[1]) < (bb[2] - bb[0]) * (bb[3] - bb[1])
                }!
                out.boxes.append([minX1, minY1, maxX2, maxY2])
                out.keypoints.append(dets.keypoints[best])
                out.kptConf.append(dets.kptConf[best])
                out.boxConf.append(dets.boxConf[best])
            } else {
                for m in members {
                    out.boxes.append(dets.boxes[m]); out.keypoints.append(dets.keypoints[m])
                    out.kptConf.append(dets.kptConf[m]); out.boxConf.append(dets.boxConf[m])
                }
            }
        }
        return out
    }

    // MARK: - reference rescale

    /// Map a player from native source pixels into the model's reference
    /// resolution. Identity at 720p. Ports `rescale_player_to_reference`.
    private func rescale(_ player: Player?, srcWidth: Int, srcHeight: Int) -> Player? {
        guard let p = player else { return nil }
        guard srcWidth > 0, srcHeight > 0 else { return p }
        let sx = Float(screenWidth) / Float(srcWidth)
        let sy = Float(screenHeight) / Float(srcHeight)
        if sx == 1.0 && sy == 1.0 { return p }
        var box = p.box
        box[0] *= sx; box[1] *= sy; box[2] *= sx; box[3] *= sy
        var kp = p.keypoints
        for k in 0..<(kp.count / 2) { kp[2 * k] *= sx; kp[2 * k + 1] *= sy }
        return Player(box: box, keypoints: kp, conf: p.conf, boxConf: p.boxConf)
    }
}
