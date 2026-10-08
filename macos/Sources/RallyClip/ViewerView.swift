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
    init(match: Match, backend: Backend) { _vm = StateObject(wrappedValue: ViewerModel(match: match, backend: backend)) }
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
                if vm.hasChanges {
                    let alert = NSAlert(); alert.messageText = "Discard unsaved point edits?"
                    alert.addButton(withTitle: "Discard Edits"); alert.addButton(withTitle: "Keep Editing")
                    return alert.runModal() == .alertFirstButtonReturn
                }
                return !vm.saving
            }
            await vm.load()
        }
        .onDisappear { vm.close(); app.canNavigate = nil }
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
                Text(vm.match.name).font(.title2.bold()).lineLimit(1)
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
        VStack(spacing: 10) {
            HStack {
                Text("MATCH TIMELINE").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                Button { vm.zoom = max(1, vm.zoom / 2) } label: { Image(systemName: "minus.magnifyingglass") }.buttonStyle(.plain)
                Text("\(Int(vm.zoom))×").font(.caption).monospacedDigit()
                Button { vm.zoom = min(16, vm.zoom * 2) } label: { Image(systemName: "plus.magnifyingglass") }.buttonStyle(.plain)
            }
            GeometryReader { outer in
                ScrollView(.horizontal) {
                    let width = outer.size.width * vm.zoom
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 6).fill(.quaternary)
                        ForEach(Array(vm.points.enumerated()), id: \.offset) { index, point in
                            RoundedRectangle(cornerRadius: 3).fill(index == vm.selected ? Color.accentColor : Color.accentColor.opacity(0.4))
                                .frame(width: max(3, width * point.duration / max(1, vm.total)), height: 24)
                                .offset(x: width * point.start / max(1, vm.total))
                        }
                        Rectangle().fill(.primary).frame(width: 2, height: 38)
                            .offset(x: min(width - 2, width * vm.sourceTime / max(1, vm.total)))
                    }.frame(width: width, height: 38)
                        .contentShape(Rectangle())
                        .gesture(DragGesture(minimumDistance: 0).onChanged { value in vm.seek(value.location.x / width * vm.total, autoplay: false) })
                }
            }.frame(height: 48).accessibilityLabel("Match timeline")
            HStack { Text("0:00"); Spacer(); Text(formatTime(vm.total)) }.font(.caption).foregroundStyle(.secondary)
        }
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
                    }.padding(.vertical, 4).listRowBackground(index == vm.selected ? Color.accentColor.opacity(0.14) : Color.clear)
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
    }
}
