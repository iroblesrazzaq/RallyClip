import XCTest
@testable import RallyClipKit

final class PointMathTests: XCTestCase {
    let points = [Point(start: 10, end: 15), Point(start: 40, end: 44)]
    func testSourceAndCompositionMappingAcrossGap() {
        XCTAssertEqual(PointMath.sourceTime(at: 4.9, points: points), 14.9, accuracy: 0.001)
        XCTAssertEqual(PointMath.sourceTime(at: 5, points: points), 40)
        XCTAssertEqual(PointMath.playbackTime(at: 42, points: points), 7)
        XCTAssertEqual(PointMath.playbackTime(at: 25, points: points), 5)
        XCTAssertEqual(PointMath.sourceTime(at: 20, points: points), 44)
    }
    func testAPointsOnlySeekIntoAGapLandsAtTheNextPoint() {
        let points = [Point(start: 0, end: 10), Point(start: 20, end: 30)]
        XCTAssertEqual(PointMath.landedTime(at: 15, points: points), 20)
        XCTAssertEqual(PointMath.landedTime(at: 25, points: points), 25)  // inside a point: no move
        XCTAssertEqual(PointMath.landedTime(at: 5, points: points), 5)
        XCTAssertEqual(PointMath.landedTime(at: 40, points: points), 30)  // after the last point
        // The window then reveals 20 s, so the playhead stays on screen.
        var window = TimelineWindow()
        window.zoom(by: 30, anchor: 0.5, total: 30)  // 1 s around 15 s
        window.reveal(PointMath.landedTime(at: 15, points: points), total: 30)
        XCTAssertNotNil(window.playheadX(20, total: 30, width: 500))
    }

    func testRejectInvalidEditsAndSortValidOnes() throws {
        XCTAssertEqual(try PointMath.validate(points.reversed(), duration: 50), points)
        for invalid in [[Point(start: .nan, end: 2)], [Point(start: 1, end: .infinity)],
                        [Point(start: -1, end: 2)], [Point(start: 2, end: 1)],
                        [Point(start: 0, end: 51)], [Point(start: 1, end: 5), Point(start: 4, end: 8)]] {
            XCTAssertThrowsError(try PointMath.validate(invalid, duration: 50))
        }
    }
    func testEmptyAndTouchingPoints() throws {
        XCTAssertEqual(PointMath.sourceTime(at: 3, points: []), 0)
        XCTAssertEqual(try PointMath.validate([Point(start: 0, end: 1), Point(start: 1, end: 2)], duration: 2).count, 2)
    }
}
