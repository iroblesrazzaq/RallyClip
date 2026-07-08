import SwiftUI

@main
struct RallyClipApp: App {
    @StateObject private var model = AppModel()
    var body: some Scene {
        WindowGroup { ContentView().environmentObject(model) }
    }
}

/// Root: palette by color scheme, paper background, screen switch + toast stack.
/// Mirrors the `.app-shell` / view-switch structure of `index.html` + `showView`.
struct ContentView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let palette = Palette.resolve(scheme)
        ZStack {
            palette.paper.ignoresSafeArea()
            Group {
                switch model.screen {
                case .welcome: WelcomeView()
                case .library: LibraryView()
                case .upload: UploadView()
                case .processing: AnalysisProgressView()
                case .viewer: ViewerView().id(model.selected?.id)
                }
            }
            ToastStack(toasts: model.toasts)
        }
        .environment(\.palette, palette)
        .tint(palette.ink)
    }
}

struct ToastStack: View {
    let toasts: [Toast]
    @Environment(\.palette) private var palette
    var body: some View {
        VStack(spacing: 8) {
            Spacer()
            ForEach(toasts) { t in
                Text(t.message)
                    .font(.serif(15))
                    .padding(.horizontal, 16).padding(.vertical, 12)
                    .frame(maxWidth: 320, alignment: .leading)
                    .background(background(for: t.tone))
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .padding(16)
        .animation(.easeInOut(duration: 0.2), value: toasts)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
    }
    private func background(for tone: Toast.Tone) -> Color {
        switch tone {
        case .info: return palette.ink
        case .success: return palette.success
        case .error: return Color(hex: 0x991B1B)
        }
    }
}
