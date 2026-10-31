import AppKit
import Foundation
import RallyClipKit
import UniformTypeIdentifiers

@MainActor
final class AppModel: ObservableObject {
    @Published var matches: [Match] = []
    @Published var selected: String?
    @Published var starting = true
    @Published var error: String?
    @Published var job: Job?
    @Published var importing = false
    @Published var search = ""
    @Published var message: String?
    var backend: Backend?
    var canNavigate: (() -> Bool)?
    let service = BackendProcess()
    private var jobID: String?
    private var monitor: Task<Void, Never>?
    var busy: Bool { importing || jobID != nil }
    private var cancelRequested = false

    func navigate(to id: String?) {
        guard id != selected else { return }
        if canNavigate?() != false { selected = id }
    }

    func start() async {
        guard backend == nil else { return }
        starting = true
        do {
            let client = try await service.start()
            matches = try await client.library()
            backend = client
        }
        catch { service.stop(); backend = nil; self.error = error.localizedDescription }
        starting = false
    }
    func reload() async throws {
        guard let backend else { return }
        matches = try await backend.library()
    }
    func chooseImport(folder: Bool) {
        guard !busy, backend != nil else { return }
        let panel = NSOpenPanel()
        panel.canChooseDirectories = folder; panel.canChooseFiles = !folder
        panel.allowsMultipleSelection = false
        panel.title = folder ? "Choose consecutive camera clips" : "Choose a tennis match"
        panel.prompt = "Analyze"
        if !folder { panel.allowedContentTypes = [.movie, .mpeg4Movie, .quickTimeMovie] }
        guard panel.runModal() == .OK, let url = panel.url, !busy else { return }
        // Busy from here, so a second choice cannot start while the folder is listed.
        importing = true; cancelRequested = false
        Task {
            defer { importing = false }
            do {
                // List off the main actor: a card can hold tens of thousands of files.
                let urls = folder ? FolderClips.select(try await Task.detached { try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil) }.value) : [url]
                guard !urls.isEmpty else { throw RallyError.message("That folder contains no MP4, MOV or M4V clips.") }
                if cancelRequested { cancelRequested = false; return }  // Cancel pressed while listing: no job yet
                try await beginImport(urls, name: folder ? url.lastPathComponent : url.deletingPathExtension().lastPathComponent)
            } catch { self.error = error.localizedDescription }
        }
    }
    /// Call with `importing` already set.
    private func beginImport(_ urls: [URL], name: String) async throws {
        guard let backend else { return }
        jobID = try await backend.importMatch(paths: urls.map(\.path), name: name)
        job = nil
        // Cancel pressed during "Checking footage…": the job only has an id now.
        if cancelRequested, let id = jobID { cancelRequested = false; try await backend.cancel(id) }
        monitor = Task {
            do {
                while let id = jobID, !Task.isCancelled {
                    let status = try await backend.job(id); job = status
                    if status.status != "in_progress" {
                        jobID = nil
                        if status.status == "failed" { throw RallyError.message(status.error ?? "Analysis failed.") }
                        try await reload()
                        if let item = status.library_id { navigate(to: item) }
                        else if status.status == "completed" { message = "Analysis finished. No points were detected." }
                        break
                    }
                    try await Task.sleep(for: .milliseconds(500))
                }
            } catch is CancellationError { }
            catch { self.error = error.localizedDescription; jobID = nil }
        }
    }
    func cancel() {
        if importing && jobID == nil { cancelRequested = true; return }
        guard let id = jobID, let backend else { return }
        Task { do { try await backend.cancel(id) } catch { self.error = error.localizedDescription } }
    }
    func rename(_ match: Match) {
        let alert = NSAlert()
        alert.messageText = "Rename Match"
        let field = NSTextField(string: match.name)
        field.frame = NSRect(x: 0, y: 0, width: 300, height: 24)
        alert.accessoryView = field
        alert.addButton(withTitle: "Rename"); alert.addButton(withTitle: "Cancel")
        alert.window.initialFirstResponder = field
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        let name = field.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, name != match.name else { return }
        Task {
            do {
                try await backend?.rename(match.id, to: name)
                try await reload()
            } catch { self.error = error.localizedDescription }
        }
    }
    func delete(_ match: Match) {
        let alert = NSAlert()
        alert.messageText = "Delete “\(match.name)”?"
        alert.informativeText = "This removes the saved analysis and edits. Original camera files stay in place."
        alert.addButton(withTitle: "Delete"); alert.addButton(withTitle: "Cancel")
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        Task {
            do {
                try await backend?.delete(match.id)
                if selected == match.id { selected = nil }
                try await reload()
            } catch { self.error = error.localizedDescription }
        }
    }
    func stop() { monitor?.cancel(); service.stop() }
}
