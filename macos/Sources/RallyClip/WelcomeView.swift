import AppKit
import RallyClipKit
import SwiftUI

/// First-launch intro, after the legacy web app's welcome screen: same lines,
/// serif and typing, on the window's own background instead of the cream canvas.
struct WelcomeView: View {
    var onStart: () -> Void
    @State private var step = WelcomeScript.steps[0]
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    var body: some View {
        VStack(alignment: .leading, spacing: 30) {
            WelcomeHeadline(step: step)
            Button(action: onStart) { Text("Get started").frame(minWidth: 128) }
                .buttonStyle(.borderedProminent).controlSize(.large)
                .keyboardShortcut(.defaultAction).accessibilityIdentifier("welcomeStart")
                .opacity(step.done ? 1 : 0).disabled(!step.done)
                .animation(.easeInOut(duration: 0.25), value: step.done)
        }
        .frame(width: 800, alignment: .leading)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .windowBackgroundColor))
        .task {
            if reduceMotion { step = WelcomeScript.steps[WelcomeScript.steps.count - 1]; return }
            for next in WelcomeScript.steps {
                do { try await Task.sleep(for: next.wait) } catch { return }
                step = next
            }
        }
    }
}

struct WelcomeHeadline: View {
    let step: WelcomeScript.Step
    var size: CGFloat = 74
    var body: some View {
        VStack(alignment: .leading, spacing: size * 0.08) {
            ForEach(WelcomeScript.lines.indices, id: \.self) { line in
                HStack(spacing: 0) {
                    text(line)
                    if step.cursorLine == line { WelcomeCursor(size: size) }
                }.frame(height: size * 0.95, alignment: .leading)
            }
        }
        .font(.custom("Georgia", size: size)).fixedSize()
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(WelcomeScript.lines.joined(separator: " "))
    }

    private func text(_ line: Int) -> Text {
        let text = Text(String(WelcomeScript.lines[line].prefix(step.shown[line])))
        guard line == WelcomeScript.lines.count - 1 else { return text }
        return text.italic().foregroundColor(.rallyGreen)  // the legacy --free-accent green
    }
}

/// Thin bar that blinks every 0.45 s, like the legacy .typing-cursor.
private struct WelcomeCursor: View {
    let size: CGFloat
    @State private var visible = true
    var body: some View {
        Rectangle().frame(width: max(1, size * 0.024), height: size * 0.9)
            .opacity(visible ? 1 : 0).padding(.leading, size * 0.045)
            .task {
                while !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(450))
                    visible.toggle()
                }
            }
    }
}
