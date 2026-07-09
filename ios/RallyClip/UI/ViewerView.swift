import SwiftUI
import AVFoundation

/// Match viewer. Mirrors `#viewerView`: header + actions (Edit / CSV / Export),
/// the video with an auto-hiding overlay (neon point-bar timeline, transport,
/// edit bar with zoom). Playback is point-only and gapless via `ViewerModel`.
struct ViewerView: View {
    @EnvironmentObject var app: AppModel
    @Environment(\.palette) private var palette
    @StateObject private var vm: ViewerModel
    @State private var controlsVisible = true
    @State private var hideTask: Task<Void, Never>?
    @State private var fullscreen = false
    @State private var share: ShareItem?

    init() {
        // `selected` is guaranteed set when this screen is shown.
        let model = AppModelHolder.current!
        _vm = StateObject(wrappedValue: ViewerModel(match: model.selected!, onToast: { m, t in model.toast(m, t) }))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if !fullscreen { header }
            videoArea
                .aspectRatio(fullscreen ? nil : 16.0/9.0, contentMode: .fit)
                .frame(maxWidth: fullscreen ? .infinity : nil, maxHeight: fullscreen ? .infinity : nil)
        }
        .padding(fullscreen ? 0 : 24)
        .frame(maxWidth: fullscreen ? .infinity : 1120)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(fullscreen ? Color.black.ignoresSafeArea() : nil)
        .sheet(item: $share) { ShareSheet(items: [$0.url]) }
        .onAppear { AppModelHolder.current = app; bumpControls() }
        .statusBarHidden(fullscreen)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button("← Saved matches") { app.showLibrary() }
                .buttonStyle(.plain).foregroundStyle(palette.ink).font(.serif(16))
            HStack(alignment: .bottom) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(vm.match.name).font(.serif(23, weight: .medium))
                    Text(vm.match.metaLine).font(.serif(14)).foregroundStyle(palette.inkSoft)
                }
                Spacer()
                if !vm.editing {
                    Button("Edit points") { vm.enterEdit() }
                        .buttonStyle(RCButtonStyle(kind: .secondary, palette: palette))
                }
                Button("CSV") { if let u = csvURL() { share = ShareItem(url: u) } }
                    .buttonStyle(RCButtonStyle(kind: .secondary, palette: palette))
                exportMenu
            }
        }
    }

    private var exportMenu: some View {
        Menu {
            Button("All points — one clip") {
                Task { if let u = await app.exportMatch(vm.match) { share = ShareItem(url: u) } }
            }
            Button("Selected points — highlight…") { vm.enterHighlight(); bumpControls() }
            Button("Each point separately — .zip") {
                Task { if let u = await app.exportPointsZip(vm.match) { share = ShareItem(url: u) } }
            }
        } label: {
            Text("Export ▾")
                .font(.serif(17, weight: .bold))
                .padding(.horizontal, 20).padding(.vertical, 12)
                .background(palette.accent).foregroundStyle(palette.onAccent)
                .overlay(Capsule().stroke(palette.ink, lineWidth: 1))
                .clipShape(Capsule())
        }
        .disabled(vm.editing)
    }

    private var videoArea: some View {
        ZStack {
            PlayerLayerView(player: vm.player)
                .background(Color.black)
                .onTapGesture { if !vm.editing && !vm.selectingHighlight { vm.togglePlay(); bumpControls() } }
            if controlsVisible || vm.editing || vm.selectingHighlight {
                overlay.transition(.opacity)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: fullscreen ? 0 : 8))
        .contentShape(Rectangle())
        .onHover { _ in bumpControls() }
    }

    private var overlay: some View {
        VStack {
            Spacer()
            VStack(spacing: 12) {
                TimelineBar(vm: vm, palette: palette)
                HStack {
                    Text("\(clock(vm.currentTime)) / \(clock(vm.duration))")
                        .font(.system(size: 15, weight: .bold)).monospacedDigit().foregroundStyle(.white)
                    controlButton("gobackward.5") { vm.skip(-5); bumpControls() }
                    controlButton(vm.isPlaying ? "pause.fill" : "play.fill", big: true) { vm.togglePlay(); bumpControls() }
                    controlButton("goforward.5") { vm.skip(5); bumpControls() }
                    Spacer()
                    controlButton(fullscreen ? "arrow.down.right.and.arrow.up.left" : "arrow.up.left.and.arrow.down.right") {
                        withAnimation { fullscreen.toggle() }; bumpControls()
                    }
                }
                if vm.editing { editBar }
                if vm.selectingHighlight { selectBar }
            }
            .padding(fullscreen ? 24 : 16)
            .background(
                LinearGradient(colors: [.black.opacity(0.78), .black.opacity(0.0)],
                               startPoint: .bottom, endPoint: .top))
        }
    }

    private var editBar: some View {
        HStack {
            Text(editHint).font(.system(size: 14, weight: .semibold)).foregroundStyle(.white.opacity(0.9)).lineLimit(1)
            Spacer()
            HStack(spacing: 6) {
                smallBtn("−") { vm.zoom(2) }
                smallBtn("+") { vm.zoom(0.5) }
                smallBtn("Fit") { vm.fitZoom() }.disabled(vm.zoomWindow == nil)
                smallBtn("Add") { vm.addPointAtPlayhead() }
                smallBtn("Delete") { vm.deleteSelected() }.disabled(vm.selectedIndex < 0)
                smallBtn("Reset") { vm.resetEdits() }.disabled(!vm.match.hasEdits && !vm.sessionDirty)
                Button("Done") { vm.exitEdit() }.buttonStyle(RCButtonStyle(kind: .primary, palette: palette, small: true))
            }
        }
    }

    private var selectBar: some View {
        HStack {
            Text(selectHint).font(.system(size: 14, weight: .semibold)).foregroundStyle(.white.opacity(0.9)).lineLimit(1)
            Spacer()
            HStack(spacing: 6) {
                smallBtn("Select all") { vm.selectAllHighlight() }
                smallBtn("Clear") { vm.clearHighlight() }
                smallBtn("Cancel") { vm.exitHighlight() }
                Button(vm.highlightSelection.isEmpty ? "Export highlight" : "Export highlight (\(vm.highlightSelection.count))") {
                    let indices = Array(vm.highlightSelection)
                    Task { if let u = await app.exportHighlight(vm.match, indices: indices) { share = ShareItem(url: u) } }
                    vm.exitHighlight()
                }
                .buttonStyle(RCButtonStyle(kind: .primary, palette: palette, small: true))
                .disabled(vm.highlightSelection.isEmpty)
            }
        }
    }

    private var selectHint: String {
        let n = vm.highlightSelection.count
        return n == 0 ? "Tap points to add them to the highlight."
                      : "\(n) point\(n == 1 ? "" : "s") selected — they’ll play back-to-back."
    }

    private var editHint: String {
        guard vm.points.indices.contains(vm.selectedIndex) else { return "Tap a point, then drag its ends to trim." }
        let s = vm.points[vm.selectedIndex]
        return "Point \(vm.selectedIndex + 1) of \(vm.points.count): \(clock(s.start)) – \(clock(s.end))"
    }

    private func controlButton(_ system: String, big: Bool = false, _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: system)
                .font(.system(size: big ? 26 : 18, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: big ? 44 : 34, height: big ? 44 : 34)
                .shadow(color: .black.opacity(0.6), radius: 4)
        }
    }
    private func smallBtn(_ label: String, _ action: @escaping () -> Void) -> some View {
        Button(label, action: action).buttonStyle(RCButtonStyle(kind: .secondary, palette: palette, small: true))
    }

    private func bumpControls() {
        controlsVisible = true
        hideTask?.cancel()
        hideTask = Task {
            try? await Task.sleep(nanoseconds: 2_600_000_000)
            if !vm.editing, vm.isPlaying { withAnimation { controlsVisible = false } }
        }
    }

    private func clock(_ s: Double) -> String {
        let safe = max(0, Int(s))
        let h = safe / 3600, m = (safe % 3600) / 60, sec = safe % 60
        return h > 0 ? String(format: "%d:%02d:%02d", h, m, sec) : String(format: "%d:%02d", m, sec)
    }

    private func csvURL() -> URL? {
        guard let data = MatchStore.shared.csvData(vm.match.id) else { return nil }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(vm.match.name)_segments.csv")
        try? data.write(to: url); return url
    }
}

/// AVPlayerLayer host (custom controls, unlike AVKit's VideoPlayer).
struct PlayerLayerView: UIViewRepresentable {
    let player: AVPlayer
    func makeUIView(context: Context) -> PlayerUIView { PlayerUIView(player: player) }
    func updateUIView(_ uiView: PlayerUIView, context: Context) {}
}
final class PlayerUIView: UIView {
    override class var layerClass: AnyClass { AVPlayerLayer.self }
    init(player: AVPlayer) {
        super.init(frame: .zero)
        (layer as! AVPlayerLayer).player = player
        (layer as! AVPlayerLayer).videoGravity = .resizeAspect
        backgroundColor = .black
    }
    required init?(coder: NSCoder) { fatalError() }
}

/// Weak holder so `ViewerModel` (built in the view's init) can reach the app model.
enum AppModelHolder { static weak var current: AppModel? }
