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
}
