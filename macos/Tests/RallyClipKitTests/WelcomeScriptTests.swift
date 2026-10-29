import XCTest
@testable import RallyClipKit

final class WelcomeScriptTests: XCTestCase {
    func testStartsWithTheBrandLineAndEndsWithEveryLine() {
        let steps = WelcomeScript.steps
        XCTAssertEqual(steps.first, .init(wait: .zero, shown: [10, 0, 0], cursorLine: 1))
        XCTAssertEqual(steps.last?.shown, WelcomeScript.lines.map(\.count))
        XCTAssertEqual(steps.filter(\.done).count, 1)
        XCTAssertTrue(steps.last!.done)
    }

    func testTypesOneCharacterAtATimeOnTheLegacySchedule() {
        let steps = WelcomeScript.steps
        for (prev, next) in zip(steps, steps.dropFirst()) {
            XCTAssertLessThanOrEqual(zip(prev.shown, next.shown).map { $1 - $0 }.reduce(0, +), 1)
        }
        // 2 x (504 ms line start + 110 ms first char) + 34 x 89 ms (other chars and line ends).
        XCTAssertEqual(steps.reduce(Duration.zero) { $0 + $1.wait }, .milliseconds(4254))
    }
}
