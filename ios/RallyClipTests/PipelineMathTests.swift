import Testing
@testable import RallyClip

/// Deterministic ports — no models, no simulator services. Fast; validates the
/// math against the Python originals (infer/inference.py, v1.py, data_preprocessor.py).
@Suite("Pipeline math")
struct PipelineMathTests {

    // MARK: generate_start_indices

    @Test func startIndicesRegular() throws {
        #expect(try Postprocess.startIndices(numFrames: 250, seqLen: 100, overlap: 50) == [0, 50, 100, 150])
    }
    @Test func startIndicesEndAnchored() throws {
        // 0 fits (0..100); 50 would exceed → end-anchored tail window at n-seqLen.
        #expect(try Postprocess.startIndices(numFrames: 120, seqLen: 100, overlap: 50) == [0, 20])
    }
    @Test func startIndicesTooShortThrows() {
        #expect(throws: PipelineError.self) {
            _ = try Postprocess.startIndices(numFrames: 50, seqLen: 100, overlap: 50)
        }
    }

    // MARK: gaussian / sigmoid

    @Test func gaussianConstantIsIdentity() {
        let out = Postprocess.gaussianFilter1d([5, 5, 5, 5, 5], sigma: 1.0)
        #expect(out.allSatisfy { abs($0 - 5) < 1e-4 })
    }
    @Test func gaussianZeroSigmaIsIdentity() {
        #expect(Postprocess.gaussianFilter1d([1, 2, 3], sigma: 0) == [1, 2, 3])
    }
    @Test func sigmoidMidpoint() {
        #expect(abs(Postprocess.sigmoid([0])[0] - 0.5) < 1e-6)
    }

    // MARK: hysteresis + segment extraction

    @Test func hysteresisBump() {
        let pred = Postprocess.hysteresis([0, 0, 0.8, 0.8, 0.8, 0, 0], low: 0.45, high: 0.7, minDuration: 0)
        #expect(pred == [0, 0, 1, 1, 1, 0, 0])
    }
    @Test func hysteresisRespectsMinDuration() {
        let pred = Postprocess.hysteresis([0, 0, 0.8, 0.8, 0.8, 0, 0], low: 0.45, high: 0.7, minDuration: 5)
        #expect(pred.allSatisfy { $0 == 0 })
    }
    @Test func extractSegmentsRuns() {
        let segs = Postprocess.extractSegments([0, 1, 1, 0, 1])
        #expect(segs.map { [$0.0, $0.1] } == [[1, 3], [4, 5]])
    }

    // MARK: windowed average

    @Test func windowedAverageConstant() throws {
        let features = Array(repeating: [Float](repeating: 0, count: 2), count: 120)
        var windowSizes: Set<Int> = []
        let out = try Postprocess.windowedAverage(features: features, seqLen: 100, overlap: 50) { window in
            windowSizes.insert(window.count)
            return [Float](repeating: 0.5, count: 100)
        }
        #expect(out.count == 120)
        #expect(out.allSatisfy { abs($0 - 0.5) < 1e-6 })
        #expect(windowSizes == [100])
    }

    // MARK: StandardScaler

    @Test func scalerTransform() {
        let s = StandardScaler(mean: [0, 10], scale: [1, 2])
        let out = s.transform([2, 14])
        #expect(out == [2, 2])
    }
    @Test func scalerZeroScaleGuarded() {
        let s = StandardScaler(mean: [0], scale: [0])
        #expect(s.transform([1])[0].isFinite)   // divided by 1e-12, not 0
    }

    // MARK: FeatureSetV1 (362-dim layout)

    @Test func featureDimAndAbsentPlayers() {
        let fe = FeatureEngineer(targetFps: 5)
        let v = fe.build(near: nil, far: nil)
        #expect(v.count == 362)
        #expect(v[0] == 0)         // near "exists" = 0
        #expect(v[1] == -1)        // near box[0] = -1
        #expect(v[181] == 0)       // far "exists" = 0
    }

    @Test func featurePresentPlayerFirstFrame() {
        let fe = FeatureEngineer(targetFps: 5)
        let p = Player(box: [10, 20, 30, 40], keypoints: Array(repeating: 0, count: 34),
                       conf: Array(repeating: 0.9, count: 17), boxConf: 0.8)
        let v = fe.build(near: p, far: nil)
        #expect(v[0] == 1)                 // exists
        #expect(Array(v[1...4]) == [10, 20, 30, 40])
        #expect(v[5] == 20 && v[6] == 30)  // centroid
        #expect(v[7] == 0 && v[8] == 0)    // velocity (no prev)
        #expect(v[11] == -1)               // speed default (no prev)
        #expect(v[12] == -1)               // accel_mag default
        #expect(v[180] == 0.8)             // box_conf is the last per-player slot
    }

    @Test func featureVelocityAcrossFrames() {
        let fe = FeatureEngineer(targetFps: 5)   // dt = 0.2
        let p1 = Player(box: [0, 0, 10, 10], keypoints: Array(repeating: 0, count: 34),
                        conf: Array(repeating: 0.5, count: 17), boxConf: 0.5)
        let p2 = Player(box: [10, 10, 20, 20], keypoints: Array(repeating: 0, count: 34),
                        conf: Array(repeating: 0.5, count: 17), boxConf: 0.5)
        _ = fe.build(near: p1, far: nil)
        let v = fe.build(near: p2, far: nil)
        // centroid 5,5 -> 15,15 over dt 0.2 = 50,50
        #expect(abs(v[7] - 50) < 1e-3 && abs(v[8] - 50) < 1e-3)
        #expect(abs(v[11] - (50 * 50 + 50 * 50).squareRoot()) < 1e-2)  // speed
        #expect(v[9] == 0 && v[10] == 0)   // accel: no stored prev-velocity yet
        #expect(v[12] == -1)               // accel_mag
    }

    // MARK: Preprocessor (assignment + court filter)

    @Test func assignNearFar() {
        let pre = Preprocessor(screenWidth: 1280, screenHeight: 720)
        let dets = PoseDetections(
            boxes: [[100, 100, 200, 600], [600, 100, 700, 300]],
            boxConf: [0.9, 0.8],
            keypoints: [Array(repeating: 0, count: 34), Array(repeating: 0, count: 34)],
            kptConf: [Array(repeating: 0, count: 17), Array(repeating: 0, count: 17)])
        let (near, far) = pre.process(dets, mask: nil, srcWidth: 1280, srcHeight: 720)
        #expect(near?.box[3] == 600)   // near = greatest bottom-y
        #expect(far?.box[3] == 300)    // far = remaining, closest to center-x
    }

    @Test func courtFilterDropsOutsidePlayers() {
        let pre = Preprocessor(screenWidth: 1280, screenHeight: 720)
        let dets = PoseDetections(
            boxes: [[100, 100, 200, 600]], boxConf: [0.9],
            keypoints: [Array(repeating: 0, count: 34)], kptConf: [Array(repeating: 0, count: 17)])
        let allOut = CourtMask(width: 800, height: 700, data: Array(repeating: 255, count: 800 * 700))
        let (near, far) = pre.process(dets, mask: allOut, srcWidth: 1280, srcHeight: 720)
        #expect(near == nil && far == nil)   // centroid (150,350) is "out"
    }

    // MARK: pose decode + NMS

    @Test func decodeSingleAnchor() throws {
        var pred = [Float](repeating: 0, count: 56)  // dims [1,56,1] -> row-major = pred[row]
        pred[0] = 100; pred[1] = 100; pred[2] = 40; pred[3] = 60; pred[4] = 0.9  // cx,cy,w,h,conf
        pred[5] = 100; pred[6] = 100; pred[7] = 0.8                              // kpt0 x,y,conf
        let out = try PoseRunner.decode(pred: pred, dims: [1, 56, 1], ratio: 1, padLeft: 0, padTop: 0,
                                        origW: 200, origH: 200, confThr: 0.25, iouThr: 0.7, maxDet: 300)
        #expect(out.count == 1)
        #expect(out.boxes[0] == [80, 70, 120, 130])
        #expect(abs(out.boxConf[0] - 0.9) < 1e-6)
        #expect(out.keypoints[0][0] == 100 && out.keypoints[0][1] == 100)
    }

    @Test func nmsSuppressesOverlap() {
        let keep = PoseRunner.nms(boxes: [[0, 0, 10, 10], [0, 0, 10, 9]], scores: [0.9, 0.8], iouThr: 0.7)
        #expect(keep == [0])          // IoU 0.9 > 0.7
    }
    @Test func nmsKeepsDisjoint() {
        let keep = PoseRunner.nms(boxes: [[0, 0, 10, 10], [100, 100, 110, 110]], scores: [0.9, 0.8], iouThr: 0.7)
        #expect(Set(keep) == [0, 1])
    }
}
