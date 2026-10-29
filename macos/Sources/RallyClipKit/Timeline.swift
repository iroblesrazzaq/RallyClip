import Foundation

/// The part of the match timeline on screen: `span` seconds from `start`.
/// Every call takes the match length, because it is not known until the sources load.
public struct TimelineWindow: Equatable, Sendable {
    /// Seconds on screen at the deepest zoom.
    public static let minimumSpan = 1.0
    private var start = 0.0
    /// nil shows the whole match.
    private var span: Double?
    public init() {}

    public func visible(total: Double) -> (start: Double, span: Double) {
        let total = Self.length(total)
        let span = min(total, max(min(total, Self.minimumSpan), span ?? total))
        return (min(max(0, start), total - span), span)
    }
    public func zoom(total: Double) -> Double { Self.length(total) / visible(total: total).span }
    public static func maxZoom(total: Double) -> Double { max(1, length(total) / minimumSpan) }

    /// Zoom in (factor > 1) or out, and keep the time under `anchor` (0 to 1 across the width) in place.
    public mutating func zoom(by factor: Double, anchor: Double, total: Double) {
        guard factor.isFinite, factor > 0 else { return }
        let now = visible(total: total), length = Self.length(total)
        let anchor = anchor.isFinite ? min(1, max(0, anchor)) : 0.5
        let pinned = now.start + anchor * now.span
        let next = min(length, max(min(length, Self.minimumSpan), now.span / factor))
        start = min(max(0, pinned - anchor * next), length - next)
        span = next >= length ? nil : next
    }
    public mutating func pan(by seconds: Double, total: Double) {
        guard seconds.isFinite else { return }
        let now = visible(total: total)
        start = min(max(0, now.start + seconds), Self.length(total) - now.span)
    }
    /// Page the window so `time` is on screen. No change when it is already on screen.
    public mutating func reveal(_ time: Double, total: Double) {
        let now = visible(total: total)
        guard time.isFinite, time < now.start || time > now.start + now.span else { return }
        // A tenth in from the left, so playback runs a full page before the next one.
        start = min(max(0, time - now.span * 0.1), Self.length(total) - now.span)
    }
    public func time(at fraction: Double, total: Double) -> Double {
        let now = visible(total: total)
        return now.start + min(1, max(0, fraction)) * now.span
    }
    public func fraction(of time: Double, total: Double) -> Double {
        let now = visible(total: total)
        return (time - now.start) / now.span
    }
    /// The playhead x on a bar `width` wide, or nil when the time is off screen.
    public func playheadX(_ time: Double, total: Double, width: Double) -> Double? {
        guard time.isFinite, width > 0 else { return nil }
        let at = fraction(of: min(max(time, 0), Self.length(total)), total: total)
        guard (0...1).contains(at) else { return nil }
        return min(width - 1, max(1, at * width))
    }
    /// h:mm:ss.t (or m:ss.t under an hour), for a deep zoom.
    public static func fineLabel(_ seconds: Double) -> String {
        let tenths = seconds.isFinite ? Int((max(0, seconds) * 10).rounded()) : 0
        let (h, m, s, t) = (tenths / 36000, tenths / 600 % 60, tenths / 10 % 60, tenths % 10)
        return h > 0 ? String(format: "%d:%02d:%02d.%d", h, m, s, t) : String(format: "%d:%02d.%d", m, s, t)
    }
    private static func length(_ total: Double) -> Double { total.isFinite ? max(total, 0.001) : 0.001 }
}
