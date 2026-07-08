import Testing
import Foundation
@testable import RallyClip

/// The full on-device pipeline vs. the desktop golden. Runs `AnalysisJob` (CPU,
/// like the golden was generated) on the committed `clip.mp4` and asserts the
/// segments match `golden_segments.csv` — same rule as the desktop's
/// `test_cli_golden_parity`: identical count, boundaries within one 0.2 s hop
/// (0.25 s tolerance).
///
/// Heavy (loads both ONNX models + OpenCV, runs pose over the whole clip). Tagged
/// `.e2e`; skip with `-skip-testing:RallyClipTests/EndToEndParityTests`.
@Suite("End-to-end parity", .tags(.e2e))
struct EndToEndParityTests {

    @Test(.timeLimit(.minutes(5)))
    func segmentsMatchGolden() async throws {
        let clip = try #require(T.testBundle.url(forResource: "clip", withExtension: "mp4"))
        let goldenURL = try #require(T.testBundle.url(forResource: "golden_segments", withExtension: "csv"))
        let want = try T.parseSegmentsCSV(goldenURL)

        let contract = try ModelContract.loadBundled()
        var cfg = AnalysisConfig.defaults(contract)
        cfg.device = .cpu   // match the golden (CPU / dynamic export)

        let job = AnalysisJob(sourceURL: clip, displayName: "golden", config: cfg, contract: contract)
        let meta = try await job.run { _ in }
        let saved = try #require(meta, "pipeline detected no points on the golden clip")
        defer { MatchStore.shared.delete(saved.id) }

        let got = MatchStore.shared.segments(saved.id)

        #expect(got.count == want.count, "segment count \(got.count) != golden \(want.count): \(got)")
        for (g, w) in zip(got.sorted { $0.start < $1.start }, want) {
            #expect(abs(g.start - w.start) <= 0.25, "start \(g.start) vs golden \(w.start)")
            #expect(abs(g.end - w.end) <= 0.25, "end \(g.end) vs golden \(w.end)")
        }
    }
}
