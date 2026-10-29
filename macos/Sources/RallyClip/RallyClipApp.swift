import AppKit
import SwiftUI

@main
struct RallyClipApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var model = AppModel()
    /// Set once the first-launch intro is dismissed; it never shows again.
    @AppStorage("welcomeSeen") private var welcomeSeen = false

    init() {
        // Packaging check for scripts/native/launch_smoke.sh: the window may not draw
        // the logo when it is off screen, so resolve it here and exit.
        if CommandLine.arguments.contains("--check-resources") {
            exit(logoURL.flatMap(NSImage.init(contentsOf:)) == nil ? 1 : 0)
        }
    }

    var body: some Scene {
        Window("RallyClip", id: "main") {
            Group {
                if welcomeSeen { ContentView() }
                else { WelcomeView { withAnimation(.easeInOut(duration: 0.3)) { welcomeSeen = true } } }
            }.environmentObject(model)
                .frame(minWidth: 1050, minHeight: 700)
                .tint(Color(red: 0.22, green: 0.68, blue: 0.52))
                .task {
                    delegate.model = model
                    NSApplication.shared.windows.first(where: { $0.identifier?.rawValue == "main" })?.delegate = delegate
                    await model.start()
                }
        }
        .defaultSize(width: 1360, height: 900)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Match…") { welcomeSeen = true; model.chooseImport(folder: false) }.keyboardShortcut("n")
                Button("Import Consecutive Clips…") { welcomeSeen = true; model.chooseImport(folder: true) }
            }
            CommandGroup(after: .appInfo) {
                Button("Check for Updates…") {
                    NSWorkspace.shared.open(URL(string: "https://github.com/iroblesrazzaq/RallyClip/releases")!)
                }
            }
            CommandGroup(replacing: .help) {
                Link("RallyClip Help", destination: URL(string: "https://github.com/iroblesrazzaq/RallyClip#readme")!)
            }
        }
        Settings { SettingsView() }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    weak var model: AppModel?
    private var closingApproved = false
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
    private func canQuit() -> Bool {
        if model?.busy == true {
            let alert = NSAlert(); alert.messageText = "Quit and cancel analysis?"
            alert.informativeText = "The current analysis will stop. Saved matches remain available."
            alert.addButton(withTitle: "Quit"); alert.addButton(withTitle: "Keep Analyzing")
            if alert.runModal() != .alertFirstButtonReturn { return false }
        }
        return model?.canNavigate?() != false
    }
    func windowShouldClose(_ sender: NSWindow) -> Bool {
        closingApproved = canQuit()
        return closingApproved
    }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if !closingApproved && !canQuit() { return .terminateCancel }
        model?.stop(); return .terminateNow
    }
}

struct SettingsView: View {
    var body: some View {
        Form {
            LabeledContent("Analysis model", value: "v0.6.1 · court lines + temporal convolution")
            LabeledContent("Processing", value: "On this Mac")
            LabeledContent("Playback", value: "Native AVFoundation")
            Button("Show Saved Matches in Finder") {
                NSWorkspace.shared.open(BackendProcess.dataRoot.appendingPathComponent("library"))
            }
            Text("Keep original camera footage available. Camera-folder imports reference those files for full-quality playback and export.")
                .font(.callout).foregroundStyle(.secondary)
        }.padding(24).frame(width: 510)
    }
}
