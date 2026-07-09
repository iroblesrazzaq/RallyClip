import SwiftUI

/// The overlay timeline: white progress track, neon point bars, a white
/// playhead thumb, and — in edit mode — draggable segments with end handles.
/// Mirrors the `.viewer-seek-wrap` stack + `renderViewerPointRanges` /
/// `renderEditTrack` positioning (everything routed through the zoom window).
struct TimelineBar: View {
    @ObservedObject var vm: ViewerModel
    let palette: Palette
    private let trackHeight: CGFloat = 5
    private let barHeight: CGFloat = 9

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let win = vm.window()
            let span = max(0.0001, win.end - win.start)
            let x: (Double) -> CGFloat = { t in CGFloat((min(max(t, win.start), win.end) - win.start) / span) * w }
            let time: (CGFloat) -> Double = { px in win.start + Double(max(0, min(1, px / w))) * span }
            let playX = x(vm.currentTime)

            ZStack(alignment: .leading) {
                // Track + progress underlay
                Capsule().fill(Color.white.opacity(0.3)).frame(height: trackHeight)
                Capsule().fill(Color.white.opacity(0.95)).frame(width: playX, height: trackHeight)

                // Neon point bars (hidden while editing / selecting; those modes
                // draw their own interactive bars instead).
                if !vm.editing && !vm.selectingHighlight {
                    ForEach(Array(vm.points.enumerated()), id: \.offset) { _, seg in
                        let sx = x(seg.start), ex = x(seg.end)
                        if ex > sx {
                            Capsule().fill(palette.tennis)
                                .overlay(Capsule().stroke(.black.opacity(0.85), lineWidth: 1))
                                .frame(width: max(2, ex - sx), height: barHeight)
                                .offset(x: sx)
                        }
                    }
                }

                // Highlight selection: tap a bar to toggle it into the export.
                if vm.selectingHighlight {
                    ForEach(Array(vm.points.enumerated()), id: \.offset) { idx, seg in
                        let sx = x(seg.start), ex = x(seg.end)
                        let on = vm.highlightSelection.contains(idx)
                        RoundedRectangle(cornerRadius: 5)
                            .fill(on ? palette.tennis.opacity(0.62) : Color.white.opacity(0.14))
                            .overlay(RoundedRectangle(cornerRadius: 5)
                                .stroke(on ? palette.tennis : .white.opacity(0.4), lineWidth: 1))
                            .frame(width: max(3, ex - sx), height: on ? 20 : 15)
                            .offset(x: sx)
                            .onTapGesture { vm.toggleHighlight(idx) }
                    }
                }

                // Edit segments + handles
                if vm.editing {
                    ForEach(Array(vm.points.enumerated()), id: \.offset) { idx, seg in
                        let sx = x(seg.start), ex = x(seg.end)
                        let selected = idx == vm.selectedIndex
                        let h: CGFloat = selected ? 20 : 15
                        RoundedRectangle(cornerRadius: 5)
                            .fill(palette.tennis.opacity(selected ? 0.62 : 0.42))
                            .overlay(RoundedRectangle(cornerRadius: 5).stroke(palette.tennis, lineWidth: 1))
                            .frame(width: max(3, ex - sx), height: h)
                            .offset(x: sx)
                            .onTapGesture { vm.select(idx) }
                        if selected {
                            handle(atX: sx, w: w) { vm.trim(index: idx, edge: .start, to: time($0)) }
                            handle(atX: ex, w: w) { vm.trim(index: idx, edge: .end, to: time($0)) }
                        }
                    }
                }

                // Playhead thumb
                Circle().fill(.white).frame(width: 16, height: 16)
                    .shadow(color: .black.opacity(0.7), radius: 3)
                    .offset(x: playX - 8)
            }
            .frame(height: 30)
            .frame(maxHeight: .infinity)
            .contentShape(Rectangle())
            .gesture(vm.editing || vm.selectingHighlight ? nil : DragGesture(minimumDistance: 0)
                .onChanged { g in vm.seek(to: time(g.location.x), autoplay: false) }
                .onEnded { g in vm.seek(to: time(g.location.x), autoplay: vm.isPlaying) })
        }
        .frame(height: 30)
    }

    private func handle(atX cx: CGFloat, w: CGFloat, onDrag: @escaping (CGFloat) -> Void) -> some View {
        RoundedRectangle(cornerRadius: 4)
            .fill(palette.tennis)
            .frame(width: 14, height: 28)
            .shadow(color: .black.opacity(0.7), radius: 3)
            .offset(x: cx - 7)
            .gesture(DragGesture(minimumDistance: 0)
                .onChanged { g in onDrag(g.location.x) }
                .onEnded { _ in vm.commitTrim() })
    }
}
