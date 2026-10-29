import AppKit
import AVKit
import SwiftUI
import RallyClipKit

struct NativePlayer: NSViewRepresentable {
    let player: AVPlayer
    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.player = player; view.controlsStyle = .floating
        view.showsFullScreenToggleButton = true
        view.allowsPictureInPicturePlayback = true
        view.videoGravity = .resizeAspect
        view.setAccessibilityIdentifier("nativePlayer")
        return view
    }
    func updateNSView(_ view: AVPlayerView, context: Context) { view.player = player }
    static func dismantleNSView(_ view: AVPlayerView, coordinator: ()) { view.player = nil }
}

struct ViewerView: View {
    @EnvironmentObject var app: AppModel
    @StateObject private var vm: ViewerModel
    @State private var showReset = false
    /// The library's current copy, so a rename shows at once.
    let match: Match
    init(match: Match, backend: Backend) {
        self.match = match
        _vm = StateObject(wrappedValue: ViewerModel(match: match, backend: backend))
    }
    var body: some View {
        VStack(spacing: 0) {
            header.padding(20)
            Divider()
            HSplitView {
                VStack(spacing: 0) {
                    ZStack {
                        Color.black
                        NativePlayer(player: vm.player)
                        if vm.loading { ProgressView("Preparing playback…").padding(20).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12)) }
                    }.frame(minHeight: 260, maxHeight: .infinity)
                    transport.padding(14).background(.bar)
                    timeline.padding(16)
                    if vm.editing { editor.padding(.horizontal, 16).padding(.bottom, 16) }
                }.frame(minWidth: 550)
                pointList.frame(minWidth: 210, idealWidth: 245, maxWidth: 330)
            }
            if let progress = vm.exportProgress {
                Divider()
                HStack {
                    Text(vm.exportLabel).font(.callout)
                    ProgressView(value: progress).frame(maxWidth: 250)
                    Text("\(Int(progress * 100))%").monospacedDigit()
                    Spacer()
                    Button("Cancel") { vm.cancelExport() }
                }.padding(14)
            }
            if let notice = vm.notice {
                HStack { Image(systemName: "checkmark.circle.fill").foregroundStyle(.green); Text(notice); Spacer(); Button { vm.notice = nil } label: { Image(systemName: "xmark") }.buttonStyle(.plain) }
                    .font(.callout).padding(12).background(.quaternary)
            }
        }
        .task {
            app.canNavigate = {
                if vm.exportProgress != nil {
                    let alert = NSAlert(); alert.messageText = "Cancel the video export?"
                    alert.addButton(withTitle: "Cancel Export"); alert.addButton(withTitle: "Keep Exporting")
                    if alert.runModal() != .alertFirstButtonReturn { return false }
                    vm.cancelExport()
                }
                if vm.saving {  // the request would still save edits the user chose to discard
                    let alert = NSAlert(); alert.messageText = "Wait for the point edits to finish saving."
                    alert.runModal(); return false
                }
                if vm.hasChanges {
                    let alert = NSAlert(); alert.messageText = "Discard unsaved point edits?"
                    alert.addButton(withTitle: "Discard Edits"); alert.addButton(withTitle: "Keep Editing")
                    return alert.runModal() == .alertFirstButtonReturn
                }
                return true
            }
            await vm.load()
        }
        .onDisappear { vm.close(); app.canNavigate = nil }
        .onChange(of: match.name) { _, name in vm.name = name }
        .alert("RallyClip", isPresented: Binding(get: { vm.error != nil }, set: { if !$0 { vm.error = nil } })) {
            Button("OK", role: .cancel) { vm.error = nil }
        } message: { Text(vm.error ?? "") }
        .confirmationDialog("Restore the original model predictions?", isPresented: $showReset) {
            Button("Restore Original Points", role: .destructive) { Task { await vm.reset() } }
        } message: { Text("This discards your saved edits for this match.") }
    }
    var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(match.name).font(.title2.bold()).lineLimit(1)
                Text("\(vm.points.count) points · \(formatTime(vm.pointDuration)) of \(formatTime(vm.total))\(vm.edited ? " · Edited" : "")")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Spacer()
            if vm.editing {
                Button("Discard") { vm.discardEdit() }.disabled(vm.saving)
                Button(vm.saving ? "Saving…" : "Save Edits") { Task { await vm.saveEdit() } }
                    .buttonStyle(.borderedProminent).disabled(vm.saving).accessibilityIdentifier("saveEdits")
            } else {
                Button { vm.enterEdit() } label: { Label("Edit Points", systemImage: "slider.horizontal.3") }
                    .disabled(vm.loading || vm.exportProgress != nil).accessibilityIdentifier("editPoints")
                Menu {
                    Button("All Points — One Video…") { vm.exportVideo() }
                    Button("Selected Points — Highlight…") { vm.exportVideo(selectedOnly: true) }.disabled(vm.selectedForExport.isEmpty)
                    Button("Each Point — Separate Videos…") { vm.exportVideo(individual: true) }
                    Divider()
                    Button("Point Timestamps — CSV…") { vm.exportCSV() }
                } label: { Label("Export", systemImage: "square.and.arrow.up") }
                    .disabled(vm.loading || vm.exportProgress != nil).accessibilityIdentifier("exportMenu")
            }
        }
    }
    var transport: some View {
        HStack(spacing: 16) {
            Button { vm.jump(max(0, vm.selected - 1)) } label: { Image(systemName: "backward.end.fill") }.help("Previous point").accessibilityLabel("Previous point")
            Button { vm.seek(vm.sourceTime - 5) } label: { Image(systemName: "gobackward.5") }
            Button { vm.togglePlay() } label: { Image(systemName: vm.playing ? "pause.fill" : "play.fill").font(.title2).frame(width: 24) }
                .keyboardShortcut(.space, modifiers: []).accessibilityIdentifier("playPause")
            Button { vm.seek(vm.sourceTime + 5) } label: { Image(systemName: "goforward.5") }
            Button { vm.jump(min(vm.points.count - 1, vm.selected + 1)) } label: { Image(systemName: "forward.end.fill") }.help("Next point").accessibilityLabel("Next point")
            Text(formatTime(vm.sourceTime)).monospacedDigit().font(.callout)
            Spacer()
            Picker("Playback", selection: $vm.pointOnly) {
                Text("Points Only").tag(true); Text("Full Match").tag(false)
            }.pickerStyle(.segmented).labelsHidden().frame(width: 210).disabled(vm.editing)
                .onChange(of: vm.pointOnly) { _, _ in vm.changeMode() }
        }.buttonStyle(.plain).disabled(vm.loading)
    }
    var timeline: some View {
        let total = vm.total, shown = vm.window.visible(total: total), zoom = vm.window.zoom(total: total)
        return VStack(spacing: 10) {
            HStack {
                Text("MATCH TIMELINE").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                Button { vm.zoomTimeline(by: 0.5) } label: { Image(systemName: "minus.magnifyingglass") }
                    .buttonStyle(.plain).disabled(zoom <= 1.001).keyboardShortcut("-", modifiers: .command)
                    .help("Zoom out (⌘−). Pinch or ⌘-scroll on the timeline to zoom at the pointer.").accessibilityLabel("Zoom out")
                Text(String(format: zoom < 10 ? "%.1f×" : "%.0f×", zoom)).font(.caption).monospacedDigit().frame(minWidth: 40)
                Button { vm.zoomTimeline(by: 2) } label: { Image(systemName: "plus.magnifyingglass") }
                    .buttonStyle(.plain).disabled(zoom >= TimelineWindow.maxZoom(total: total) - 0.001).keyboardShortcut("=", modifiers: .command)
                    .help("Zoom in (⌘+). Pinch or ⌘-scroll on the timeline to zoom at the pointer.").accessibilityLabel("Zoom in")
            }
            GeometryReader { geo in
                let width = max(1, geo.size.width)
                ZStack {
                    TimelineCanvas(points: vm.points, selected: vm.selected, time: vm.sourceTime, total: total, window: vm.window)
                    TimelineInput(
                        scrub: { x, done in vm.seek(vm.window.time(at: x / width, total: total), autoplay: false, exact: done) },
                        zoom: { factor, x in vm.window.zoom(by: factor, anchor: x / width, total: total) },
                        pan: { dx in vm.window.pan(by: -dx / width * shown.span, total: total) })
                }
            }.frame(height: 48).accessibilityElement().accessibilityLabel("Match timeline")
                .accessibilityValue("\(formatTime(vm.sourceTime)) of \(formatTime(total))")
            HStack { Text(timelineLabel(shown.start, span: shown.span)); Spacer(); Text(timelineLabel(shown.start + shown.span, span: shown.span)) }
                .font(.caption).monospacedDigit().foregroundStyle(.secondary)
        }
    }
    /// Tenths of a second once less than 20 s is on screen.
    func timelineLabel(_ seconds: Double, span: Double) -> String {
        guard span < 20, seconds.isFinite else { return formatTime(seconds) }
        let tenths = Int((max(0, seconds) * 10).rounded())
        return String(format: "%d:%02d.%d", tenths / 600, tenths / 10 % 60, tenths % 10)
    }
    var pointList: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack { Text("Points").font(.headline); Spacer(); Text("\(vm.selectedForExport.count) selected").font(.caption).foregroundStyle(.secondary) }.padding(14)
            Divider()
            List {
                ForEach(Array(vm.points.enumerated()), id: \.offset) { index, point in
                    HStack(spacing: 8) {
                        Toggle("Select point \(index + 1)", isOn: Binding(get: { vm.selectedForExport.contains(index) }, set: { if $0 { vm.selectedForExport.insert(index) } else { vm.selectedForExport.remove(index) } }))
                            .labelsHidden().toggleStyle(.checkbox).disabled(vm.editing)
                        Button { vm.jump(index) } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Point \(index + 1)").fontWeight(index == vm.selected ? .bold : .regular)
                                    Text("\(formatTime(point.start)) – \(formatTime(point.end))").font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                Text(String(format: "%.1fs", point.duration)).font(.caption).foregroundStyle(.secondary)
                            }.contentShape(Rectangle())
                        }.buttonStyle(.plain)
                    }.padding(.vertical, 4).listRowBackground(index == vm.selected ? Color.rallyGreen.opacity(0.16) : Color.clear)
                }
            }.listStyle(.plain)
            if vm.points.isEmpty { ContentUnavailableView("No points", systemImage: "tennisball", description: Text("Use Edit Points to add an interval.")) }
        }
    }
    var editor: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Edit point boundaries", systemImage: "scissors").font(.headline)
                Spacer()
                Button("Add at Playhead") { vm.addPoint() }
                Button("Delete Point", role: .destructive) { vm.deletePoint() }.disabled(vm.currentPoint == nil)
            }
            if vm.currentPoint != nil {
                HStack {
                    Text("Start")
                    TextField("Start seconds", value: Binding(get: { vm.currentPoint?.start ?? 0 }, set: { if vm.points.indices.contains(vm.selected) { vm.points[vm.selected].start = $0 } }), format: .number.precision(.fractionLength(3)))
                        .frame(width: 95).textFieldStyle(.roundedBorder)
                    Button("Set to Playhead") { vm.trim(start: true) }
                    Spacer()
                    Text("End")
                    TextField("End seconds", value: Binding(get: { vm.currentPoint?.end ?? 0 }, set: { if vm.points.indices.contains(vm.selected) { vm.points[vm.selected].end = $0 } }), format: .number.precision(.fractionLength(3)))
                        .frame(width: 95).textFieldStyle(.roundedBorder)
                    Button("Set to Playhead") { vm.trim(start: false) }
                }.font(.callout)
            }
            HStack {
                Text("Scrub the full match, select a point, then set its start or end. Times are in source seconds.").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Restore Original…") { showReset = true }.disabled(vm.saving)
            }
        }.padding(14).background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
            .disabled(vm.saving)  // the save response replaces the points
    }
}

/// Draws only the part of the match on screen, so deep zoom costs no more than the full view.
private struct TimelineCanvas: View {
    let points: [Point]
    let selected: Int
    let time: Double
    let total: Double
    let window: TimelineWindow
    var body: some View {
        Canvas { context, size in
            let shown = window.visible(total: total), bar = CGRect(x: 0, y: 0, width: size.width, height: 38)
            let x = { (t: Double) in CGFloat((t - shown.start) / shown.span) * size.width }
            context.fill(Path(roundedRect: bar, cornerRadius: 6), with: .color(.secondary.opacity(0.18)))
            for (index, point) in points.enumerated() where point.end >= shown.start && point.start <= shown.start + shown.span {
                let left = max(-4, x(point.start)), right = min(size.width + 4, x(point.end))
                let rect = CGRect(x: left, y: 7, width: max(3, right - left), height: 24)
                context.fill(Path(roundedRect: rect, cornerRadius: 3),
                             with: .color(index == selected ? .rallyGreen : .rallyGreen.opacity(0.4)))
            }
            let head = min(size.width - 1, max(1, x(min(max(time, 0), max(total, 0)))))
            if head >= 0 && head <= size.width {
                context.fill(Path(CGRect(x: head - 2, y: 0, width: 4, height: 38)), with: .color(.black.opacity(0.45)))
                context.fill(Path(CGRect(x: head - 1, y: 0, width: 2, height: 38)), with: .color(.rallyBall))
            }
            // Where the shown part sits in the whole match, once zoomed in.
            if shown.span < total {
                let track = CGRect(x: 0, y: 43, width: size.width, height: 4)
                context.fill(Path(roundedRect: track, cornerRadius: 2), with: .color(.secondary.opacity(0.15)))
                let thumb = CGRect(x: CGFloat(shown.start / total) * size.width, y: 43,
                                   width: max(6, CGFloat(shown.span / total) * size.width), height: 4)
                context.fill(Path(roundedRect: thumb, cornerRadius: 2), with: .color(.secondary.opacity(0.6)))
            }
        }
    }
}

/// Mouse, trackpad and wheel input for the timeline. SwiftUI has no scroll-wheel
/// or pointer-anchored pinch event on macOS 14, so an AppKit view takes them.
private struct TimelineInput: NSViewRepresentable {
    /// x in points; `done` on mouse up, for an exact seek.
    var scrub: (CGFloat, Bool) -> Void
    /// factor > 1 zooms in, at x.
    var zoom: (Double, CGFloat) -> Void
    /// Content moves right by dx points.
    var pan: (CGFloat) -> Void
    func makeNSView(context: Context) -> InputView { InputView() }
    func updateNSView(_ view: InputView, context: Context) { view.scrub = scrub; view.zoom = zoom; view.pan = pan }

    final class InputView: NSView {
        var scrub: ((CGFloat, Bool) -> Void)?
        var zoom: ((Double, CGFloat) -> Void)?
        var pan: ((CGFloat) -> Void)?
        override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
        private func x(_ event: NSEvent) -> CGFloat { min(bounds.width, max(0, convert(event.locationInWindow, from: nil).x)) }
        override func mouseDown(with event: NSEvent) { scrub?(x(event), false) }
        override func mouseDragged(with event: NSEvent) { scrub?(x(event), false) }
        override func mouseUp(with event: NSEvent) { scrub?(x(event), true) }
        override func magnify(with event: NSEvent) { zoom?(1 + event.magnification, x(event)) }
        override func scrollWheel(with event: NSEvent) {
            let dx = event.scrollingDeltaX, dy = event.scrollingDeltaY
            // A trackpad reports points; a mouse wheel reports lines.
            let scale: CGFloat = event.hasPreciseScrollingDeltas ? 1 : 12
            if event.modifierFlags.contains(.command) {
                zoom?(exp(Double(abs(dy) >= abs(dx) ? dy : dx) * scale * 0.008), x(event))
            } else if abs(dx) > abs(dy) {
                pan?(dx * scale)
            } else if !event.hasPreciseScrollingDeltas && dy != 0 {
                pan?(dy * scale)  // a plain wheel scrolls along the timeline
            } else {
                super.scrollWheel(with: event)
            }
        }
    }
}
