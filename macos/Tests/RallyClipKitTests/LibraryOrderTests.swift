import XCTest
@testable import RallyClipKit

final class LibraryOrderTests: XCTestCase {
    func match(_ id: String, _ name: String, _ ts: Double?) -> Match {
        Match(id: id, name: name, duration_s: 60, point_duration_s: 30, n_segments: 3,
              created: "2026-10-0\(Int(ts ?? 0) % 10)T00:00:00", created_ts: ts, has_edits: false)
    }

    func testDateIsNewestFirstAndNameIsNaturalOrder() {
        let all = [match("a", "match 10", 1), match("b", "Match 2", 3), match("c", "Final", 2), match("d", "Match 2", 4)]
        XCTAssertEqual(LibraryOrder.sorted(all, by: .date, custom: []).map(\.id), ["d", "b", "c", "a"])
        // Numbers in numeric order, case ignored; equal names keep newest first.
        XCTAssertEqual(LibraryOrder.sorted(all, by: .name, custom: []).map(\.id), ["c", "d", "b", "a"])
    }

    func testCustomKeepsTheSavedOrderAndPutsNewMatchesFirst() {
        let all = [match("a", "A", 1), match("b", "B", 2), match("c", "C", 3), match("new", "N", 9)]
        // "gone" was deleted; "new" is not in the order yet.
        let order = ["b", "gone", "a", "c"]
        XCTAssertEqual(LibraryOrder.sorted(all, by: .custom, custom: order).map(\.id), ["new", "b", "a", "c"])
        XCTAssertEqual(LibraryOrder.sorted(all, by: .custom, custom: []).map(\.id), ["new", "c", "b", "a"])
    }

    func testMoveMatchesSwiftUIListDrops() {
        let ids = ["a", "b", "c", "d"]
        XCTAssertEqual(LibraryOrder.move(ids, from: [0], to: 3), ["b", "c", "a", "d"])  // drop before d
        XCTAssertEqual(LibraryOrder.move(ids, from: [3], to: 0), ["d", "a", "b", "c"])
        XCTAssertEqual(LibraryOrder.move(ids, from: [0, 2], to: 4), ["b", "d", "a", "c"])
        XCTAssertEqual(LibraryOrder.move(ids, from: [1], to: 1), ids)
    }

    func testOlderLibrariesWithoutCreatedTsStillDecode() throws {
        let json = #"{"id":"x","name":"Old","duration_s":1,"point_duration_s":0,"n_segments":0,"created":"2026-01-01T00:00:00","has_edits":false}"#
        let old = try JSONDecoder().decode(Match.self, from: Data(json.utf8))
        XCTAssertNil(old.created_ts)
    }
}
