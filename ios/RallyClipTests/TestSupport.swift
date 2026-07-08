import Foundation
import CoreVideo
import Testing
@testable import RallyClip

/// Resolves the test bundle (Swift Testing has no XCTestCase to hang `Bundle(for:)` on).
final class BundleToken {}

extension Tag {
    /// Heavy, model-loading end-to-end tests. Exclude with
    /// `-skip-testing:RallyClipTests/EndToEndParityTests` for a fast run.
    @Tag static var e2e: Self
}

enum T {
    static var testBundle: Bundle { Bundle(for: BundleToken.self) }

    /// Solid-color 32BGRA pixel buffer for feeding the pose/court paths.
    static func solidPixelBuffer(width: Int, height: Int, b: UInt8, g: UInt8, r: UInt8) -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferCGImageCompatibilityKey: true,
                             kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary, &pb)
        let buf = pb!
        CVPixelBufferLockBaseAddress(buf, [])
        let base = CVPixelBufferGetBaseAddress(buf)!.assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(buf)
        for y in 0..<height {
            for x in 0..<width {
                let p = y * stride + x * 4
                base[p] = b; base[p + 1] = g; base[p + 2] = r; base[p + 3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(buf, [])
        return buf
    }

    /// Parse a `start_time,end_time` CSV (golden format).
    static func parseSegmentsCSV(_ url: URL) throws -> [Segment] {
        let text = try String(contentsOf: url, encoding: .utf8)
        // Split on any newline flavor and trim stray \r / whitespace per field.
        let lines = text.split(whereSeparator: \.isNewline)
        return lines.enumerated().compactMap { (i, line) in
            if i == 0 { return nil }   // header
            let parts = line.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            guard parts.count >= 2, let a = Double(parts[0]), let b = Double(parts[1]) else { return nil }
            return Segment(start: a, end: b)
        }
    }
}
