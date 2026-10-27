import AppKit
import AVFoundation
import Combine
import RallyClipKit
import UniformTypeIdentifiers

@MainActor
final class ViewerModel: ObservableObject {
    let match: Match
    let backend: Backend
    let player = AVPlayer()
    @Published var points: [Point] = []
    @Published var sources: [Source] = []
    @Published var sourceTime = 0.0
    @Published var playing = false
    @Published var pointOnly = true
    @Published var editing = false
    @Published var edited = false
    @Published var selected = 0
    @Published var loading = true
    @Published var saving = false
    @Published var error: String?
    @Published var exportProgress: Double?
    @Published var exportLabel = "Exporting video"
    @Published var selectedForExport: Set<Int> = []
    @Published var zoom = 1.0
    @Published var notice: String?
    private var savedPoints: [Point] = []
    private var playbackPoints: [Point] = []
    private var observer: Any?
    private var endObserver: NSObjectProtocol?
    private var statusObserver: NSKeyValueObservation?
    private var buildTask: Task<Void, Never>?
    private var exportTask: Task<Void, Never>?
    private let exporter = VideoExporter()
    private var generation = 0
    private var seekGeneration = 0
    var total: Double { sources.map(\.end).max() ?? match.duration_s }
    var pointDuration: Double { points.reduce(0) { $0 + $1.duration } }
    var currentPoint: Point? { points.indices.contains(selected) ? points[selected] : nil }
    var hasChanges: Bool { points != savedPoints }

    init(match: Match, backend: Backend) {
        self.match = match; self.backend = backend
        player.automaticallyWaitsToMinimizeStalling = true
        observer = player.addPeriodicTimeObserver(forInterval: CMTime(seconds: 0.05, preferredTimescale: 600), queue: .main) { [weak self] time in
            Task { @MainActor [weak self] in
                guard let self else { return }
                let time = time.seconds
                guard time.isFinite else { return }
                self.sourceTime = self.pointOnly && !self.editing
                    ? PointMath.sourceTime(at: time, points: self.playbackPoints) : time
                self.playing = self.player.rate > 0
                if !self.editing, let index = self.points.firstIndex(where: { self.sourceTime >= $0.start && self.sourceTime < $0.end }) {
                    self.selected = index
                }
            }
        }
        endObserver = NotificationCenter.default.addObserver(forName: .AVPlayerItemDidPlayToEndTime, object: nil, queue: .main) { [weak self] notification in
            Task { @MainActor [weak self] in
                guard let self, let item = notification.object as? AVPlayerItem, item === self.player.currentItem else { return }
                self.playing = false
            }
        }
    }

    func load() async {
        do {
            async let sourceResult = backend.sources(match.id)
            async let pointResult = backend.points(match.id)
            let (s, p) = try await (sourceResult, pointResult)
            sources = s; points = p.segments; savedPoints = p.segments; edited = p.edited
            await rebuild(at: points.first?.start ?? 0, autoplay: false)
        } catch { self.error = error.localizedDescription; loading = false }
    }
    func rebuild(at time: Double? = nil, autoplay: Bool? = nil) async {
        generation += 1; let ticket = generation
        let destination = time ?? sourceTime, resume = autoplay ?? playing
        player.pause(); loading = true
        do {
            let intervals = pointOnly && !editing ? points : [Point(start: 0, end: total)]
            if intervals.isEmpty { player.replaceCurrentItem(with: nil); loading = false; return }
            let composition = try await CompositionBuilder.build(sources: sources, points: intervals)
            guard ticket == generation, !Task.isCancelled else { return }
            playbackPoints = intervals
            let item = AVPlayerItem(asset: composition)
            item.preferredForwardBufferDuration = 3
            statusObserver = item.observe(\.status, options: [.new]) { [weak self] item, _ in
                if item.status == .failed {
                    Task { @MainActor [weak self] in self?.error = item.error?.localizedDescription ?? "Video playback failed." }
                }
            }
            player.replaceCurrentItem(with: item)
            seek(destination, autoplay: resume); loading = false
        } catch is CancellationError { }
        catch { if ticket == generation { self.error = error.localizedDescription; loading = false } }
    }
    /// exact: false while scrubbing, so each drag step decodes from a nearby keyframe.
    func seek(_ time: Double, autoplay: Bool? = nil, exact: Bool = true) {
        guard time.isFinite else { return }
        seekGeneration += 1; let ticket = seekGeneration
        let source = min(total, max(0, time))
        let target = pointOnly && !editing ? PointMath.playbackTime(at: source, points: playbackPoints) : source
        let resume = autoplay ?? playing
        if !resume { player.pause(); playing = false }
        let tolerance = exact ? CMTime.zero : CMTime(seconds: 0.5, preferredTimescale: 600)
        player.seek(to: CMTime(seconds: target, preferredTimescale: 60000), toleranceBefore: tolerance, toleranceAfter: tolerance) { [weak self] finished in
            Task { @MainActor [weak self] in
                guard let self, ticket == self.seekGeneration else { return }
                if finished && resume { self.player.play() }
            }
        }
        sourceTime = source
    }
    func togglePlay() {
        seekGeneration += 1
        if player.rate > 0 { player.pause(); playing = false }
        else {
            if let item = player.currentItem, player.currentTime() >= item.duration { seek(points.first?.start ?? 0, autoplay: true) }
            else { player.play() }
            playing = true
        }
    }
    func jump(_ index: Int) { guard points.indices.contains(index) else { return }; selected = index; seek(points[index].start) }
    func changeMode() { buildTask?.cancel(); buildTask = Task { await rebuild() } }
    func enterEdit() { player.pause(); playing = false; editing = true; changeMode() }
    func discardEdit() { points = savedPoints; selected = min(selected, max(0, points.count - 1)); editing = false; changeMode() }
    func saveEdit() async {
        saving = true; defer { saving = false }
        do {
            let valid = try PointMath.validate(points, duration: total)
            let response = try await backend.save(valid, id: match.id)
            points = response.segments; savedPoints = points; edited = response.edited
            editing = false; await rebuild(); notice = "Point edits saved"
        } catch { self.error = error.localizedDescription }
    }
    func reset() async {
        saving = true; defer { saving = false }
        do {
            let response = try await backend.reset(match.id)
            points = response.segments; savedPoints = points; edited = false; editing = false
            selected = 0; await rebuild(); notice = "Restored model predictions"
        } catch { self.error = error.localizedDescription }
    }
    func trim(start: Bool) {
        guard points.indices.contains(selected) else { return }
        var point = points[selected]
        if start { point.start = max(selected > 0 ? points[selected - 1].end : 0, min(sourceTime, point.end - 0.1)) }
        else { point.end = min(selected + 1 < points.count ? points[selected + 1].start : total, max(sourceTime, point.start + 0.1)) }
        points[selected] = point
    }
    func addPoint() {
        let start = max(0, min(sourceTime, total - 0.1))
        guard !points.contains(where: { start >= $0.start && start < $0.end }) else { error = "There is already a point here."; return }
        let next = points.first { $0.start > start }?.start ?? total
        points.append(Point(start: start, end: min(start + 8, next)))
        points.sort { $0.start < $1.start }; selected = points.firstIndex { $0.start == start } ?? 0
    }
    func deletePoint() {
        guard points.indices.contains(selected) else { return }
        points.remove(at: selected); selected = min(selected, max(0, points.count - 1)); selectedForExport = []
    }
    func exportCSV() {
        let panel = NSSavePanel(); panel.nameFieldStringValue = "\(match.name)_points.csv"; panel.allowedContentTypes = [.commaSeparatedText]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let text = "start_time,end_time\n" + points.map { String(format: "%.3f,%.3f", $0.start, $0.end) }.joined(separator: "\n") + "\n"
            try text.write(to: url, atomically: true, encoding: .utf8); notice = "CSV saved"
        } catch { self.error = error.localizedDescription }
    }
    func exportVideo(selectedOnly: Bool = false, individual: Bool = false) {
        guard exportTask == nil else { return }
        let chosen = selectedOnly ? points.enumerated().filter { selectedForExport.contains($0.offset) }.map(\.element) : points
        guard !chosen.isEmpty else { error = "Select at least one point."; return }
        let destination: URL
        if individual {
            let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.canCreateDirectories = true; panel.prompt = "Export points here"
            guard panel.runModal() == .OK, let url = panel.url else { return }; destination = url
        } else {
            let panel = NSSavePanel(); panel.allowedContentTypes = [.mpeg4Movie]; panel.nameFieldStringValue = "\(match.name)_\(selectedOnly ? "highlight" : "points").mp4"
            guard panel.runModal() == .OK, let url = panel.url else { return }; destination = url
        }
        exportProgress = 0; exportLabel = "Exporting video"
        exportTask = Task {
            do {
                if individual {
                    let staging = destination.appendingPathComponent(".rallyclip-\(UUID().uuidString)")
                    try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
                    defer { try? FileManager.default.removeItem(at: staging) }
                    for (index, point) in chosen.enumerated() {
                        exportLabel = "Exporting point \(index + 1) of \(chosen.count)"
                        try await exporter.export(sources: sources, points: [point], to: staging.appendingPathComponent(String(format: "point_%03d.mp4", index + 1))) { [weak self] progress in
                            Task { @MainActor [weak self] in self?.exportProgress = (Double(index) + progress) / Double(chosen.count) }
                        }
                    }
                    try Task.checkCancellation()
                    let output = destination.appendingPathComponent("RallyClip points \(UUID().uuidString.prefix(8))")
                    try FileManager.default.moveItem(at: staging, to: output)
                    NSWorkspace.shared.activateFileViewerSelecting([output])
                } else {
                    try await exporter.export(sources: sources, points: chosen, to: destination) { [weak self] progress in
                        Task { @MainActor [weak self] in self?.exportProgress = progress }
                    }
                    NSWorkspace.shared.activateFileViewerSelecting([destination])
                }
                notice = "Export complete"
            } catch is CancellationError { notice = "Export cancelled" }
            catch { self.error = error.localizedDescription }
            exportProgress = nil; exportTask = nil
        }
    }
    func cancelExport() { exportTask?.cancel(); Task { await exporter.cancel() } }
    func close() {
        generation += 1; seekGeneration += 1; buildTask?.cancel(); player.pause(); player.replaceCurrentItem(with: nil)
        if let observer { player.removeTimeObserver(observer); self.observer = nil }
        if let endObserver { NotificationCenter.default.removeObserver(endObserver); self.endObserver = nil }
        statusObserver = nil
        cancelExport()
    }
}
