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
