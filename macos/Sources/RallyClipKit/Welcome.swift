/// First-launch intro: the legacy web app's typewriter, with the same lines and timing.
public enum WelcomeScript {
    public static let lines = ["RallyClip.", "AI Match Segmentation.", "Free, Forever."]
    /// Lines shown at once; the rest are typed.
    public static let immediateLines = 1
    public static let startDelay: Duration = .milliseconds(504)
    public static let lineDelay: Duration = .milliseconds(504)
    public static let firstCharDelay: Duration = .milliseconds(110)
    public static let charDelay: Duration = .milliseconds(89)

    /// One frame: wait this long, then show `shown[i]` characters of line i with
    /// the cursor on `cursorLine`. No cursor means typing is done.
    public struct Step: Equatable, Sendable {
        public let wait: Duration
        public let shown: [Int]
        public let cursorLine: Int?
        public var done: Bool { cursorLine == nil }
    }

    /// The first step is the start state (wait zero), the last is the finished headline.
    public static let steps: [Step] = {
        var shown = lines.indices.map { $0 < immediateLines ? lines[$0].count : 0 }
        var out = [Step(wait: .zero, shown: shown, cursorLine: immediateLines < lines.count ? immediateLines : nil)]
        var wait = startDelay
        for line in immediateLines..<lines.count {
            for index in 0..<lines[line].count {
                shown[line] = index + 1
                out.append(Step(wait: wait, shown: shown, cursorLine: line))
                wait = index == 0 ? firstCharDelay : charDelay
            }
            out.append(Step(wait: wait, shown: shown, cursorLine: line + 1 < lines.count ? line + 1 : nil))
            wait = lineDelay
        }
        return out
    }()
}
