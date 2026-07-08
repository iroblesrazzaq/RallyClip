import Foundation
import AVFoundation
import Combine

/// Viewer state: point-only gapless playback + non-destructive edit mode + zoom.
/// Ports the scheduler (`playbackSegmentForSourceTime` / `advanceAfterActivePlaybackSegment`),
/// edit ops (add/delete/trim + autosave), and the timeline-zoom math from `script.js`.
/// The web dual-`<video>` gapless hack collapses to AVPlayer seeks here.
@MainActor
final class ViewerModel: ObservableObject {
    let match: MatchMeta
    let player: AVPlayer
    let duration: Double

    @Published var points: [Segment] = []
    @Published var currentTime: Double = 0
    @Published var isPlaying = false
    @Published var editing = false
    @Published var selectedIndex: Int = -1
    /// nil = full match; else the visible {start,end} window in source seconds.
    @Published var zoomWindow: (start: Double, end: Double)?
    @Published var sessionDirty = false

    let minPointSeconds = 0.5
    let addPointSeconds = 8.0
    let zoomMinSpan = 8.0

    private var timeObserver: Any?
    private var active: (end: Double, nextIndex: Int?)?
    private let store = MatchStore.shared
    private var onToast: (String, Toast.Tone) -> Void

    init(match: MatchMeta, onToast: @escaping (String, Toast.Tone) -> Void) {
        self.match = match
        self.duration = match.durationS
        self.onToast = onToast
        self.player = AVPlayer(url: store.sourceURL(match.id))
        self.points = store.segments(match.id)
        addObserver()
        let start = points.first?.start ?? 0
        seek(to: start, autoplay: true)
    }

    deinit {
        if let timeObserver { player.removeTimeObserver(timeObserver) }
    }

    private func addObserver() {
        timeObserver = player.addPeriodicTimeObserver(
            forInterval: CMTime(seconds: 0.05, preferredTimescale: 600), queue: .main) { [weak self] t in
            guard let self else { return }
            let s = CMTimeGetSeconds(t)
            self.currentTime = s
            self.isPlaying = self.player.timeControlStatus == .playing
            guard self.isPlaying, !self.editing, let active = self.active else { return }
            if s >= active.end - 0.05 { self.advance() }
        }
    }

    // MARK: - scheduling (ports playbackSegmentForSourceTime)

    private func segment(for t: Double) -> (end: Double, nextIndex: Int?) {
        guard !points.isEmpty else { return (duration, nil) }
        if let i = points.firstIndex(where: { t >= $0.start && t < $0.end }) {
            return (points[i].end, i + 1 < points.count ? i + 1 : nil)
        }
        if let j = points.firstIndex(where: { t < $0.start }) {
            return (points[j].end, j + 1 < points.count ? j + 1 : nil)
        }
        return (duration, nil)   // tail
    }

    private func advance() {
        guard let next = active?.nextIndex, next < points.count else {
            player.pause(); isPlaying = false; return
        }
        seek(to: points[next].start, autoplay: true)
    }

    // MARK: - transport

    func seek(to t: Double, autoplay: Bool) {
        let target = clamp(t)
        active = editing ? (duration, nil) : segment(for: target)
        player.seek(to: CMTime(seconds: target, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
        currentTime = target
        if autoplay { player.play(); isPlaying = true }
    }

    func togglePlay() {
        if isPlaying { player.pause(); isPlaying = false }
        else {
            if active == nil { active = editing ? (duration, nil) : segment(for: currentTime) }
            player.play(); isPlaying = true
        }
    }

    func skip(_ delta: Double) {
        let target = clamp(currentTime + delta)
        active = (duration, nil)   // manual: play straight through
        seek(to: target, autoplay: isPlaying)
    }

    func clamp(_ t: Double) -> Double {
        let v = max(0, t)
        return duration > 0 ? min(v, max(0, duration - 0.05)) : v
    }

    // MARK: - timeline window / zoom (ports timelineWindow / zoomEditTimeline)

    func window() -> (start: Double, end: Double) {
        guard editing, let w = zoomWindow, duration > 0 else { return (0, duration) }
        let s = max(0, min(w.start, duration))
        return (s, max(s, min(w.end, duration)))
    }

    func zoom(_ spanFactor: Double) {
        guard editing, duration > 0 else { return }
        let cur = window()
        let span = max(zoomMinSpan, (cur.end - cur.start) * spanFactor)
        if span >= duration { zoomWindow = nil; return }
        let focus = points.indices.contains(selectedIndex)
            ? (points[selectedIndex].start + points[selectedIndex].end) / 2
            : clamp(currentTime)
        let start = max(0, min(focus - span / 2, duration - span))
        zoomWindow = (start, start + span)
    }
    func fitZoom() { zoomWindow = nil }

    // MARK: - edit mode

    func enterEdit() {
        points = store.segments(match.id)
        editing = true
        zoomWindow = nil
        selectedIndex = points.isEmpty ? -1 : 0
        player.pause(); isPlaying = false
        active = (duration, nil)
    }
    func exitEdit() {
        editing = false; zoomWindow = nil
        selectedIndex = -1
        if sessionDirty { onToast("Point edits saved.", .success); sessionDirty = false }
        active = segment(for: currentTime)
    }

    func select(_ i: Int) {
        guard points.indices.contains(i) else { return }
        selectedIndex = i
        player.pause(); isPlaying = false
        seek(to: points[i].start, autoplay: false)
    }

    /// Drag a handle. edge: .start/.end. Clamps to neighbors + minPointSeconds.
    func trim(index: Int, edge: Edge, to t: Double) {
        guard points.indices.contains(index) else { return }
        var seg = points[index]
        if edge == .start {
            let lo = index > 0 ? points[index - 1].end : 0
            let hi = seg.end - minPointSeconds
            seg.start = (min(hi, max(lo, t)) * 1000).rounded() / 1000
        } else {
            let lo = seg.start + minPointSeconds
            let hi = index + 1 < points.count ? points[index + 1].start : duration
            seg.end = (min(hi, max(lo, t)) * 1000).rounded() / 1000
        }
        points[index] = seg
    }

    func addPointAtPlayhead() {
        guard editing, duration > 0 else { return }
        let t = clamp(currentTime)
        if points.contains(where: { t >= $0.start && t < $0.end }) {
            onToast("There is already a point here.", .error); return
        }
        let prevEnd = points.reduce(0.0) { $1.end <= t ? max($0, $1.end) : $0 }
        let nextStart = points.first(where: { t < $0.start })?.start ?? duration
        var start = max(t, prevEnd)
        let end = min(start + addPointSeconds, nextStart, duration)
        start = max(prevEnd, min(start, end - addPointSeconds))
        start = max(prevEnd, min(start, end - minPointSeconds))
        guard end - start >= minPointSeconds else { onToast("Not enough room for a new point here.", .error); return }
        let seg = Segment(start: (start * 1000).rounded() / 1000, end: (end * 1000).rounded() / 1000)
        points.append(seg)
        points.sort { $0.start < $1.start }
        selectedIndex = points.firstIndex(of: seg) ?? -1
        save()
    }

    func deleteSelected() {
        guard editing else { return }
        var i = selectedIndex
        if i < 0 { i = points.firstIndex(where: { currentTime >= $0.start && currentTime < $0.end }) ?? -1 }
        guard points.indices.contains(i) else { onToast("Select a point to delete.", .error); return }
        points.remove(at: i)
        selectedIndex = -1
        save()
    }

    func commitTrim() { save() }

    func resetEdits() {
        store.resetEdits(match.id)
        points = store.segments(match.id)
        selectedIndex = -1
        sessionDirty = false
        onToast("Points reset to the original.", .success)
    }

    func save() {
        sessionDirty = true
        store.saveEditedSegments(match.id, points)
    }

    enum Edge { case start, end }
}
