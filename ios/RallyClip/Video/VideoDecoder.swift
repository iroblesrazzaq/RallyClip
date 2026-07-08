import Foundation
import AVFoundation
import CoreVideo

/// AVFoundation decode. Two roles, matching the desktop's PyAV usage:
///  - `sampledFrames`: the pose loop, sampling ~`fps` frames/sec (0.2 s at fps 5).
///  - `frame(at:)`: random-access single frames for court detection.
///
/// PARITY: PyAV's exact sampled-frame indices at fps 5 may differ slightly from
/// this pts-threshold sampler; verify frame counts against the golden run.
final class VideoDecoder {
    let asset: AVURLAsset
    private(set) var sourceWidth = 0
    private(set) var sourceHeight = 0
    private(set) var durationSeconds: Double = 0
    private(set) var nominalFrameRate: Float = 30

    init(url: URL) async throws {
        self.asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration)
        self.durationSeconds = CMTimeGetSeconds(duration)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw PipelineError.decode("no video track")
        }
        let size = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let oriented = size.applying(transform)
        self.sourceWidth = Int(abs(oriented.width).rounded())
        self.sourceHeight = Int(abs(oriented.height).rounded())
        self.nominalFrameRate = try await track.load(.nominalFrameRate)
    }

    /// Stream frames sampled at `fps`. `handler` returns false to stop (cancel).
    /// `progress` is called with (framesEmitted, approxTotalFrames).
    func sampledFrames(fps: Double,
                       startSeconds: Double = 0,
                       durationSeconds dur: Double = 0,
                       handler: (CVPixelBuffer, Double, Int) throws -> Bool) async throws {
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw PipelineError.decode("no video track")
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        reader.add(output)

        let start = max(0, startSeconds)
        let end = dur > 0 ? start + dur : durationSeconds
        if start > 0 || dur > 0 {
            reader.timeRange = CMTimeRange(start: CMTime(seconds: start, preferredTimescale: 600),
                                           duration: CMTime(seconds: max(0.001, end - start), preferredTimescale: 600))
        }
        guard reader.startReading() else { throw PipelineError.decode(reader.error?.localizedDescription ?? "reader") }

        let step = 1.0 / fps
        var nextSample = start
        var index = 0
        while let sample = output.copyNextSampleBuffer() {
            let t = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
            if t + 1e-6 >= nextSample, let pb = CMSampleBufferGetImageBuffer(sample) {
                let keepGoing = try handler(pb, t, index)
                index += 1
                nextSample += step
                if !keepGoing { reader.cancelReading(); break }
            }
        }
        if reader.status == .failed { throw PipelineError.decode(reader.error?.localizedDescription ?? "read failed") }
    }

    /// Single frame at `seconds` as a 32BGRA pixel buffer (court detection).
    func frame(at seconds: Double) async -> CVPixelBuffer? {
        let gen = AVAssetImageGenerator(asset: asset)
        gen.requestedTimeToleranceBefore = .zero
        gen.requestedTimeToleranceAfter = .zero
        gen.appliesPreferredTrackTransform = true
        let time = CMTime(seconds: max(0, seconds), preferredTimescale: 600)
        guard let cg = try? gen.copyCGImage(at: time, actualTime: nil) else { return nil }
        return VideoDecoder.pixelBuffer(from: cg)
    }

    static func pixelBuffer(from image: CGImage) -> CVPixelBuffer? {
        let w = image.width, h = image.height
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        ]
        var pb: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA,
                                  attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                  width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                      | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }
}
