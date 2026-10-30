import XCTest
@testable import RallyClipKit

final class TimelineWindowTests: XCTestCase {
    func testStartsAtTheWholeMatchAndFitsALateLength() {
        let window = TimelineWindow()
        XCTAssertEqual(window.visible(total: 600).span, 600)
        XCTAssertEqual(window.zoom(total: 600), 1)
        XCTAssertEqual(window.visible(total: 7200).span, 7200)  // the sources loaded a longer match
    }

    func testZoomKeepsTheAnchorTimeUnderThePointer() {
        var window = TimelineWindow()
        window.zoom(by: 4, anchor: 0.25, total: 400)  // the pointer is at 100 s
        let shown = window.visible(total: 400)
        XCTAssertEqual(shown.span, 100, accuracy: 1e-9)
        XCTAssertEqual(window.time(at: 0.25, total: 400), 100, accuracy: 1e-9)
        window.zoom(by: 1.7, anchor: 0.9, total: 400)
        let pinned = shown.start + 0.9 * shown.span
        XCTAssertEqual(window.time(at: 0.9, total: 400), pinned, accuracy: 1e-9)
    }

    func testZoomStopsAtOneSecondAndAtTheWholeMatch() {
        var window = TimelineWindow()
        XCTAssertEqual(TimelineWindow.maxZoom(total: 3600), 3600)
        for _ in 0..<40 { window.zoom(by: 2, anchor: 0.5, total: 3600) }
        XCTAssertEqual(window.visible(total: 3600).span, 1, accuracy: 1e-9)
        XCTAssertEqual(window.zoom(total: 3600), 3600, accuracy: 1e-6)
        for _ in 0..<40 { window.zoom(by: 0.5, anchor: 0.1, total: 3600) }
        XCTAssertEqual(window.visible(total: 3600).start, 0)
        XCTAssertEqual(window.zoom(total: 3600), 1)
        // A clip shorter than one second cannot zoom at all.
        XCTAssertEqual(TimelineWindow.maxZoom(total: 0.5), 1)
        var short = TimelineWindow()
        short.zoom(by: 8, anchor: 0.5, total: 0.5)
        XCTAssertEqual(short.visible(total: 0.5).span, 0.5)
    }

    func testPanAndRevealStayInsideTheMatch() {
        var window = TimelineWindow()
        window.zoom(by: 10, anchor: 0, total: 1000)  // 0 to 100 s
        window.pan(by: -50, total: 1000)
        XCTAssertEqual(window.visible(total: 1000).start, 0)
        window.pan(by: 5000, total: 1000)
        XCTAssertEqual(window.visible(total: 1000).start, 900)
        window.reveal(950, total: 1000)  // already on screen
        XCTAssertEqual(window.visible(total: 1000).start, 900)
        window.reveal(300, total: 1000)  // pages so the time is a tenth in
        XCTAssertEqual(window.visible(total: 1000).start, 290, accuracy: 1e-9)
        XCTAssertEqual(window.fraction(of: 300, total: 1000), 0.1, accuracy: 1e-9)
        window.zoom(by: .nan, anchor: 0.5, total: 1000)
        window.pan(by: .infinity, total: 1000)
        XCTAssertEqual(window.visible(total: 1000).start, 290, accuracy: 1e-9)
    }

    func testPlayheadIsHiddenOffScreenAndPinnedInsideTheBar() {
        var window = TimelineWindow()
        XCTAssertEqual(window.playheadX(0, total: 100, width: 500), 1)  // whole match: the ends stay visible
        XCTAssertEqual(window.playheadX(100, total: 100, width: 500), 499)
        window.zoom(by: 10, anchor: 0, total: 100)  // 0 to 10 s
        XCTAssertEqual(window.playheadX(5, total: 100, width: 500)!, 250, accuracy: 1e-9)
        XCTAssertNil(window.playheadX(10.5, total: 100, width: 500))  // panned away while paused
        window.pan(by: 50, total: 100)
        XCTAssertNil(window.playheadX(5, total: 100, width: 500))
        XCTAssertNil(window.playheadX(.nan, total: 100, width: 500))
    }

    func testFineLabelsShowHoursPastAnHour() {
        XCTAssertEqual(TimelineWindow.fineLabel(65.04), "1:05.0")
        XCTAssertEqual(TimelineWindow.fineLabel(3599.96), "1:00:00.0")
        XCTAssertEqual(TimelineWindow.fineLabel(3725.25), "1:02:05.3")  // 3725.25 rounds to the nearest tenth
        XCTAssertEqual(TimelineWindow.fineLabel(-3), "0:00.0")
    }
}

final class TimelineHitTests: XCTestCase {
    let points = [Point(start: 10, end: 20), Point(start: 20, end: 30), Point(start: 50, end: 50.01)]

    func testAClickOnABarFindsItsPoint() {
        let window = TimelineWindow()  // 0 to 100 s on 1000 px: 10 px per second
        XCTAssertEqual(window.pointIndex(atX: 150, width: 1000, points: points, total: 100), 0)
        XCTAssertEqual(window.pointIndex(atX: 250, width: 1000, points: points, total: 100), 1)
        XCTAssertNil(window.pointIndex(atX: 400, width: 1000, points: points, total: 100))  // between points
        XCTAssertEqual(window.pointIndex(atX: 98, width: 1000, points: points, total: 100), 0)  // inside the slop
        XCTAssertNil(window.pointIndex(atX: 96, width: 1000, points: points, total: 100))
        XCTAssertNil(window.pointIndex(atX: .nan, width: 1000, points: points, total: 100))
    }

    func testAShortPointStaysClickableAndTheNearestBarWins() {
        let window = TimelineWindow()
        // 0.01 s is 0.1 px, but its bar is drawn 3 px wide.
        XCTAssertEqual(window.pointIndex(atX: 502.5, width: 1000, points: points, total: 100), 2)
        let close = [Point(start: 10, end: 10.4), Point(start: 10.8, end: 12)]  // 4 px apart
        XCTAssertEqual(window.pointIndex(atX: 106.5, width: 1000, points: close, total: 100), 1)
        XCTAssertEqual(window.pointIndex(atX: 104.5, width: 1000, points: close, total: 100), 0)
    }

    func testHitsFollowTheZoomWindow() {
        var window = TimelineWindow()
        window.zoom(by: 10, anchor: 0, total: 100)  // 0 to 10 s: points 0 and 1 are off screen to the right
        XCTAssertNil(window.pointIndex(atX: 500, width: 1000, points: points, total: 100))
        window.pan(by: 15, total: 100)  // 15 to 25 s
        XCTAssertEqual(window.pointIndex(atX: 200, width: 1000, points: points, total: 100), 0)  // 17 s
        XCTAssertEqual(window.pointIndex(atX: 800, width: 1000, points: points, total: 100), 1)  // 23 s
    }
}
