import SwiftUI

/// Welcome screen — the three-line editorial headline + "Get started".
/// Mirrors `WELCOME_LAYOUT` ("RallyClip." / "AI Match Segmentation." /
/// "Free, Forever.") with a light typewriter reveal.
struct WelcomeView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.palette) private var palette
    @State private var shown = 0

    private let lines: [(String, Bool)] = [
        ("RallyClip.", false),
        ("AI Match Segmentation.", false),
        ("Free, Forever.", true),   // accented
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 28) {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(0..<lines.count, id: \.self) { i in
                    Text(lines[i].0)
                        .font(.serif(46))
                        .foregroundStyle(lines[i].1 ? palette.success : palette.ink)
                        .italic(lines[i].1)
                        .opacity(shown > i ? 1 : 0)
                        .animation(.easeOut(duration: 0.35).delay(Double(i) * 0.45), value: shown)
                }
            }
            Button("Get started") { model.dismissWelcome() }
                .buttonStyle(RCButtonStyle(kind: .primary, palette: palette))
                .opacity(shown >= lines.count ? 1 : 0)
                .animation(.easeOut(duration: 0.3).delay(Double(lines.count) * 0.45), value: shown)
        }
        .frame(maxWidth: 640, alignment: .leading)
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear { shown = lines.count + 1 }
    }
}
