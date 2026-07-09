import Foundation
import AVFoundation
import Testing
@testable import RallyClip

/// The two per-point export flows, cutting real clips out of the bundled golden
/// video (same fixture the E2E parity test uses). Mirrors the desktop
/// `test_api_point_export` contract: one clip per point in order, and a
/// highlight whose length is the sum of the selected points.
@Suite("Clip export")
struct ExportTests {
    // Two points well inside the golden clip (which runs past 24s).
    private let segments = [Segment(start: 5.8, end: 17.0), Segment(start: 18.6, end: 24.0)]

    private func goldenClip() throws -> URL {
        try #require(T.testBundle.url(forResource: "clip", withExtension: "mp4"))
    }

    private func duration(_ url: URL) async throws -> Double {
        try await AVURLAsset(url: url).load(.duration).seconds
    }

    @Test func individualClipsAreOnePerPointInOrder() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("rc_points_\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let urls = try await ClipExporter.exportIndividual(sourceURL: goldenClip(), segments: segments, to: dir)

        #expect(urls.map { $0.lastPathComponent } == ["point_01.mp4", "point_02.mp4"])
        for (url, seg) in zip(urls, segments) {
            let d = try await duration(url)
            #expect(abs(d - (seg.end - seg.start)) < 0.6, "clip \(url.lastPathComponent) was \(d)s")
        }
    }

    @Test func highlightLengthIsSumOfSelectedPoints() async throws {
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("rc_hl_\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: out) }

        try await ClipExporter.export(sourceURL: goldenClip(), segments: segments, to: out)

        let expected = segments.reduce(0.0) { $0 + ($1.end - $1.start) }   // 11.2 + 5.4 = 16.6s
        let d = try await duration(out)
        #expect(abs(d - expected) < 0.8, "highlight was \(d)s, expected ~\(expected)s")
    }

    @Test func exportRejectsEmptySelection() async throws {
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("rc_empty.mp4")
        await #expect(throws: PipelineError.self) {
            try await ClipExporter.export(sourceURL: goldenClip(), segments: [], to: out)
        }
    }
}
