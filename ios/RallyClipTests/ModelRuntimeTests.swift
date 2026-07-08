import Testing
import CoreVideo
@testable import RallyClip

/// Exercises the bundled model assets + ONNX/OpenCV runtimes on the simulator
/// (CPU path). Confirms sessions build, shapes are sane, and nothing crashes —
/// numerical parity is covered by the end-to-end suite.
@Suite("Model runtime")
struct ModelRuntimeTests {

    @Test func contractMatchesManifest() throws {
        let c = try ModelContract.loadBundled()
        #expect(c.imgsz == 960)
        #expect(c.seqLen == 100)
        #expect(c.overlap == 50)
        #expect(c.featureDim == 362)
        #expect(c.numKeypoints == 17)
        #expect(abs(c.conf - 0.25) < 1e-9)
    }

    @Test func scalerBundled() throws {
        let s = try StandardScaler.loadBundled()
        #expect(s.mean.count == 362)
        #expect(s.scale.count == 362)
    }

    @Test func lstmProducesProbabilities() throws {
        let c = try ModelContract.loadBundled()
        let lstm = try LSTMRunner(contract: c)
        let window = Array(repeating: [Float](repeating: 0, count: c.featureDim), count: c.seqLen)
        let probs = try lstm.runWindow(window)
        #expect(probs.count == c.seqLen)
        #expect(probs.allSatisfy { $0 >= 0 && $0 <= 1 })
    }

    @Test func poseRunsOnCPU() throws {
        let c = try ModelContract.loadBundled()
        let pose = try PoseRunner(contract: c, device: .cpu)
        #expect(pose.effectiveDevice == .cpu)
        // A flat green field: the detector must run and return a valid (possibly
        // empty) result rather than crashing.
        let pb = T.solidPixelBuffer(width: 1280, height: 720, b: 60, g: 130, r: 60)
        let dets = try pose.infer(pixelBuffer: pb)
        #expect(dets.count >= 0)
        #expect(dets.boxes.count == dets.boxConf.count)
        #expect(dets.keypoints.count == dets.count)
    }

    @Test func defaultCourtMaskLoads() throws {
        // Bundled in the app (test host) as default_court_mask.png.
        let path = try #require(Bundle.main.path(forResource: "default_court_mask", ofType: "png"))
        let result = RCCourtDetector.defaultMask(fromPNGPath: path, width: 128, height: 72)
        #expect(result.success)
        #expect(result.width == 128 && result.height == 72)
        #expect(result.mask.count == 128 * 72)
    }

    @Test func courtDetectOnSyntheticFrameDoesNotCrash() throws {
        // No court lines in a solid frame → detection fails gracefully, still
        // returns a correctly sized (empty) mask.
        let frame = T.solidPixelBuffer(width: 1280, height: 720, b: 40, g: 90, r: 40)
        let result = RCCourtDetector.detect(withBaseFrame: frame, baseBoxes: [], referenceFrame: nil)
        #expect(result.width == 1280 && result.height == 720)
        #expect(result.mask.count == 1280 * 720)
    }
}
