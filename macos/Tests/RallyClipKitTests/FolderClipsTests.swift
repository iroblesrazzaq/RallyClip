import XCTest
@testable import RallyClipKit

final class FolderClipsTests: XCTestCase {
    func testSkipsHiddenTwinsAndUnplayableFormatsInNaturalOrder() {
        let names = ["GX100001.MP4", "._GX010001.MP4", "GX020001.mov", "GX010001.MP4", "notes.txt", "clip.mkv", ".DS_Store", "b.m4v"]
        let urls = names.map { URL(fileURLWithPath: "/card/DCIM/\($0)") }
        XCTAssertEqual(FolderClips.select(urls).map(\.lastPathComponent),
                       ["b.m4v", "GX010001.MP4", "GX020001.mov", "GX100001.MP4"])
    }
}
