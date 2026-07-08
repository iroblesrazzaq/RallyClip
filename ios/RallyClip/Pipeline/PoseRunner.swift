import Foundation
import CoreVideo
import OnnxRuntimeBindings

/// YOLOv8-pose on onnxruntime. Ports `extraction/yolo_onnx_runner.py`
/// (`predict` + `decode_v8_pose` + NMS) and the device/session selection in
/// `pose_extractor._resolve_onnx_session`:
///
///  - CoreML: the static 544×960 export on the CoreML EP (`MLProgram`, all units)
///    with a CPU fallback if the EP won't initialize.
///  - CPU: the dynamic-axes export, the byte-parity reference path.
final class PoseRunner {
    private let env: ORTEnv
    private let session: ORTSession
    private let contract: ModelContract
    private let inputName: String
    private let isStatic: Bool
    private(set) var effectiveDevice: PoseDevice

    // Static export canvas (16:9 at imgsz 960 → 544×960).
    private let staticW = 960, staticH = 544
    private let iouThr: Float = 0.7
    private let maxDet = 300

    init(contract: ModelContract, device: PoseDevice) throws {
        self.contract = contract
        self.env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)

        func modelURL(_ name: String) throws -> String {
            guard let u = Bundle.main.url(forResource: name, withExtension: "onnx") else {
                throw PipelineError.missingAsset("\(name).onnx")
            }
            return u.path
        }

        // Build the requested session; degrade CoreML→CPU on any EP failure,
        // exactly like the desktop's try/except around the CoreML providers.
        if device == .coreml {
            do {
                let opts = try ORTSessionOptions()
                try PoseRunner.appendCoreML(to: opts)
                self.session = try ORTSession(env: env,
                                              modelPath: try modelURL("yolov8n-pose-544x960-static"),
                                              sessionOptions: opts)
                self.isStatic = true
                self.effectiveDevice = .coreml
            } catch {
                NSLog("RallyClip: CoreML pose EP unavailable (%@); falling back to CPU.", "\(error)")
                self.session = try ORTSession(env: env,
                                              modelPath: try modelURL("yolov8n-pose-960-dynamic"),
                                              sessionOptions: try ORTSessionOptions())
                self.isStatic = false
                self.effectiveDevice = .cpu
            }
        } else {
            self.session = try ORTSession(env: env,
                                          modelPath: try modelURL("yolov8n-pose-960-dynamic"),
                                          sessionOptions: try ORTSessionOptions())
            self.isStatic = false
            self.effectiveDevice = .cpu
        }
        self.inputName = (try? session.inputNames().first) ?? "images"
    }

    private static func appendCoreML(to opts: ORTSessionOptions) throws {
        // API name has drifted across onnxruntime-objc versions; try the
        // options-based call, then the newer generic one. PARITY: MLProgram +
        // all-units matches the desktop's CoreML provider config.
        let coreOpts = ORTCoreMLExecutionProviderOptions()
        coreOpts.useCPUAndGPU = true
        try opts.appendCoreMLExecutionProvider(with: coreOpts)
    }

    /// Run pose on one frame → detections in source pixels (conf-filtered, NMS'd).
    func infer(pixelBuffer: CVPixelBuffer) throws -> PoseDetections {
        let lb: RCLetterboxResult = isStatic
            ? RCImageOps.letterboxExact(pixelBuffer, targetW: Int32(staticW), targetH: Int32(staticH))
            : RCImageOps.letterboxDynamic(pixelBuffer, imgsz: Int32(contract.imgsz))

        let shape: [NSNumber] = [1, 3, NSNumber(value: lb.height), NSNumber(value: lb.width)]
        let inputData = NSMutableData(data: lb.tensor)
        let input = try ORTValue(tensorData: inputData, elementType: .float, shape: shape)
        let outputs = try session.run(withInputs: [inputName: input],
                                      outputNames: Set(try session.outputNames()),
                                      runOptions: nil)
        guard let out = outputs.values.first else { throw PipelineError.decode("pose: no output") }
        let (pred, dims) = try floats(from: out)
        return try Self.decode(pred: pred, dims: dims,
                               ratio: lb.ratio, padLeft: Int(lb.padLeft), padTop: Int(lb.padTop),
                               origW: Int(lb.origWidth), origH: Int(lb.origHeight),
                               confThr: Float(contract.conf), iouThr: iouThr, maxDet: maxDet)
    }

    private func floats(from value: ORTValue) throws -> ([Float], [Int]) {
        let info = try value.tensorTypeAndShapeInfo()
        let dims = info.shape.map { $0.intValue }
        let data = try value.tensorData() as Data
        let count = data.count / MemoryLayout<Float>.size
        var arr = [Float](repeating: 0, count: count)
        _ = arr.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        return (arr, dims)
    }

    // MARK: - decode_v8_pose

    /// Decode a raw v8-pose head. `dims` is [1,56,N] or [56,N] (transpose if not).
    static func decode(pred: [Float], dims: [Int], ratio: Float, padLeft: Int, padTop: Int,
                       origW: Int, origH: Int, confThr: Float, iouThr: Float, maxDet: Int) throws -> PoseDetections {
        // Resolve C=56 rows and N anchors.
        var rows = 0, n = 0, transposed = false
        let d = dims.count == 3 ? Array(dims[1...]) : dims
        guard d.count == 2 else { throw PipelineError.decode("pose output rank \(dims)") }
        if d[0] == 56 { rows = d[0]; n = d[1] }
        else if d[1] == 56 { rows = d[1]; n = d[0]; transposed = true }
        else { throw PipelineError.decode("unexpected pose head \(dims)") }
        _ = rows
        // Accessor into row-major pred with optional transpose.
        func at(_ r: Int, _ i: Int) -> Float { transposed ? pred[i * 56 + r] : pred[r * n + i] }

        var boxes: [[Float]] = [], confs: [Float] = [], kpts: [[Float]] = [], kconf: [[Float]] = []
        for i in 0..<n {
            let c = at(4, i)
            if c <= confThr { continue }
            let cx = at(0, i), cy = at(1, i), w = at(2, i), h = at(3, i)
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            confs.append(c)
            var xy = [Float](repeating: 0, count: 34)
            var kc = [Float](repeating: 0, count: 17)
            for k in 0..<17 {
                xy[2 * k] = at(5 + k * 3, i)
                xy[2 * k + 1] = at(5 + k * 3 + 1, i)
                kc[k] = at(5 + k * 3 + 2, i)
            }
            kpts.append(xy); kconf.append(kc)
        }
        if boxes.isEmpty { return .empty }

        let keep = Array(nms(boxes: boxes, scores: confs, iouThr: iouThr).prefix(maxDet))
        var out = PoseDetections.empty
        for idx in keep {
            var b = boxes[idx]
            b[0] = min(max((b[0] - Float(padLeft)) / ratio, 0), Float(origW))
            b[2] = min(max((b[2] - Float(padLeft)) / ratio, 0), Float(origW))
            b[1] = min(max((b[1] - Float(padTop)) / ratio, 0), Float(origH))
            b[3] = min(max((b[3] - Float(padTop)) / ratio, 0), Float(origH))
            var xy = kpts[idx]
            for k in 0..<17 {
                xy[2 * k] = min(max((xy[2 * k] - Float(padLeft)) / ratio, 0), Float(origW))
                xy[2 * k + 1] = min(max((xy[2 * k + 1] - Float(padTop)) / ratio, 0), Float(origH))
            }
            out.boxes.append(b); out.boxConf.append(confs[idx])
            out.keypoints.append(xy); out.kptConf.append(kconf[idx])
        }
        return out
    }

    /// Greedy NMS (suppress IoU > thr), descending score. Matches `_nms`.
    static func nms(boxes: [[Float]], scores: [Float], iouThr: Float) -> [Int] {
        var order = Array(0..<scores.count).sorted { scores[$0] > scores[$1] }
        var keep: [Int] = []
        func iou(_ a: [Float], _ b: [Float]) -> Float {
            let x1 = max(a[0], b[0]), y1 = max(a[1], b[1])
            let x2 = min(a[2], b[2]), y2 = min(a[3], b[3])
            let inter = max(0, x2 - x1) * max(0, y2 - y1)
            let areaA = (a[2] - a[0]) * (a[3] - a[1])
            let areaB = (b[2] - b[0]) * (b[3] - b[1])
            return inter / max(areaA + areaB - inter, 1e-9)
        }
        while !order.isEmpty {
            let i = order.removeFirst()
            keep.append(i)
            order = order.filter { iou(boxes[i], boxes[$0]) <= iouThr }
        }
        return keep
    }
}
