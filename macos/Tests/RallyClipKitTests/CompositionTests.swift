import AVFoundation
import CoreVideo
import XCTest
@testable import RallyClipKit

final class CompositionTests: XCTestCase {
    func clip(at url: URL, color: UInt8) async throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: 64, AVVideoHeightKey: 48
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: 64, kCVPixelBufferHeightKey as String: 48
        ])
        writer.add(input); XCTAssertTrue(writer.startWriting()); writer.startSession(atSourceTime: .zero)
        for index in 0..<60 {
            while !input.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(2)) }
            var buffer: CVPixelBuffer?
            XCTAssertEqual(CVPixelBufferCreate(nil, 64, 48, kCVPixelFormatType_32BGRA, nil, &buffer), kCVReturnSuccess)
            let pixel = try XCTUnwrap(buffer)
            CVPixelBufferLockBaseAddress(pixel, [])
            memset(CVPixelBufferGetBaseAddress(pixel), Int32(color), CVPixelBufferGetDataSize(pixel))
            CVPixelBufferUnlockBaseAddress(pixel, [])
            XCTAssertTrue(adaptor.append(pixel, withPresentationTime: CMTime(value: Int64(index), timescale: 30)))
        }
        input.markAsFinished(); await writer.finishWriting()
        XCTAssertEqual(writer.status, .completed, "\(String(describing: writer.error))")
    }

    func testPointCrossingCameraBoundaryHasContinuousTimeline() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let a = root.appendingPathComponent("a.mp4"), b = root.appendingPathComponent("b.mp4")
        try await clip(at: a, color: 50); try await clip(at: b, color: 200)
        let sources = [Source(path: a.path, start: 0, duration: 2), Source(path: b.path, start: 2, duration: 2)]
        let points = [Point(start: 1.5, end: 2.5), Point(start: 3, end: 3.5)]
        let composition = try await CompositionBuilder.build(sources: sources, points: points)
        XCTAssertEqual(composition.duration.seconds, 1.5, accuracy: 0.002)
        let track = try XCTUnwrap(composition.tracks(withMediaType: .video).first)
        XCTAssertEqual(track.segments.filter { !$0.isEmpty }.count, 3)
        let export = root.appendingPathComponent("output.mp4")
        try await VideoExporter().export(sources: sources, points: points, to: export) { _ in }
        let asset = AVURLAsset(url: export)
        let duration = try await asset.load(.duration)
        XCTAssertEqual(duration.seconds, 1.5, accuracy: 0.05)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.requestedTimeToleranceBefore = .zero; generator.requestedTimeToleranceAfter = .zero
        let first = try await generator.image(at: CMTime(seconds: 0.2, preferredTimescale: 600)).image
        let second = try await generator.image(at: CMTime(seconds: 0.8, preferredTimescale: 600)).image
        func brightness(_ image: CGImage) -> Double {
            var bytes = [UInt8](repeating: 0, count: 4)
            let context = CGContext(data: &bytes, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                                    space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            context.draw(image, in: CGRect(x: 0, y: 0, width: 1, height: 1))
            return Double(bytes[0])
        }
        XCTAssertGreaterThan(brightness(second) - brightness(first), 100, "Export must cross to the second original clip")
    }

    func testMissingFootageAndUncoveredIntervalsFail() async throws {
        do {
            _ = try await CompositionBuilder.build(sources: [Source(path: "/missing.mp4", start: 0, duration: 2)], points: [Point(start: 0, end: 1)])
            XCTFail("Missing sources must not produce a silent empty composition")
        } catch { XCTAssertTrue(error.localizedDescription.contains("Reconnect")) }

        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let a = root.appendingPathComponent("a.mp4"), b = root.appendingPathComponent("b.mp4")
        try await clip(at: a, color: 50); try await clip(at: b, color: 200)
        // A 2 s gap between the clips: the point is 3 s long but only 1 s is on disk.
        let gapped = [Source(path: a.path, start: 0, duration: 2), Source(path: b.path, start: 4, duration: 2)]
        do {
            _ = try await CompositionBuilder.build(sources: gapped, points: [Point(start: 1.5, end: 4.5)])
            XCTFail("A point the clips do not cover must not produce a shorter composition")
        } catch { XCTAssertTrue(error.localizedDescription.contains("do not cover")) }
    }
}
