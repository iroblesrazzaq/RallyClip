import AppKit
import SwiftUI
import RallyClipKit

struct ContentView: View {
    @EnvironmentObject var app: AppModel
    @AppStorage("librarySort") private var sort = LibrarySort.date
    /// Match ids in the custom order, one per line.
    @AppStorage("libraryOrder") private var customOrder = ""
    var ordered: [Match] { LibraryOrder.sorted(app.matches, by: sort, custom: customOrder.split(separator: "\n").map(String.init)) }
    var filtered: [Match] { ordered.filter { app.search.isEmpty || $0.name.localizedCaseInsensitiveContains(app.search) } }
    /// Drag to reorder only in the custom order, with every match shown.
    var canReorder: Bool { sort == .custom && app.search.isEmpty }
    var body: some View {
        NavigationSplitView {
            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: 12) {
                    RallyLogo(height: 28)
                    Text("RallyClip").font(.title2.bold())
                }.padding(20)
                Menu {
                    Button("Choose a Video…") { app.chooseImport(folder: false) }
                    Button("Choose Consecutive Clips…") { app.chooseImport(folder: true) }
                } label: { Label("New Match", systemImage: "plus").frame(maxWidth: .infinity).padding(.vertical, 6) }
                    .menuStyle(.borderlessButton).padding(10).background(.tint.opacity(0.15), in: RoundedRectangle(cornerRadius: 10))
                    .padding(.horizontal, 16).disabled(app.busy || app.starting).accessibilityIdentifier("newMatch")
                HStack(spacing: 8) {
                    TextField("Search matches", text: $app.search).textFieldStyle(.roundedBorder)
                    Menu {
                        Picker("Sort By", selection: $sort) {
                            ForEach(LibrarySort.allCases, id: \.self) { Text($0.label).tag($0) }
                        }.pickerStyle(.inline)
                    } label: { Image(systemName: "arrow.up.arrow.down") }
                        .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
                        .help(sort == .custom ? "Sort: custom order. Drag matches to reorder." : "Sort: \(sort.label.lowercased())")
                        .accessibilityLabel("Sort matches")
                }.padding(16)
                List(selection: Binding(get: { app.selected }, set: { app.navigate(to: $0) })) {
                    Section("SAVED MATCHES · \(app.matches.count)") {
                        ForEach(filtered) { match in
                            HStack(spacing: 10) {
                                Image(systemName: "play.rectangle.fill").font(.title2).foregroundStyle(.secondary)
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(match.name).font(.body.weight(.medium)).lineLimit(2)
                                    Text("\(match.n_segments) points · \(formatTime(match.duration_s))").font(.caption).foregroundStyle(.secondary)
                                }
                            }.padding(.vertical, 6).tag(match.id)
                                .contextMenu {
                                    Button("Rename…") { app.rename(match) }
                                    Divider()
                                    Button("Delete Match…", role: .destructive) { app.delete(match) }
                                }
                        }.onMove(perform: canReorder ? { source, destination in
                            customOrder = LibraryOrder.move(ordered.map(\.id), from: source, to: destination).joined(separator: "\n")
                        } : nil)
                    }
                }.listStyle(.sidebar)
                if app.busy || app.job != nil { AnalysisCard().padding(14) }
                HStack {
                    Image(systemName: "lock.shield"); Text("Local processing"); Spacer()
                }.font(.caption).foregroundStyle(.secondary).padding(18)
            }.navigationSplitViewColumnWidth(min: 260, ideal: 290, max: 360)
        } detail: {
            if app.starting {
                VStack(spacing: 16) { ProgressView(); Text("Starting RallyClip…").foregroundStyle(.secondary) }
            } else if let id = app.selected, let match = app.matches.first(where: { $0.id == id }), let backend = app.backend {
                ViewerView(match: match, backend: backend).id(id)
            } else { welcome }
        }
        .toolbar {
            ToolbarItem { Button { Task { do { try await app.reload() } catch { app.error = error.localizedDescription } } } label: { Label("Refresh Library", systemImage: "arrow.clockwise") } }
        }
        .alert("RallyClip", isPresented: Binding(get: { app.error != nil }, set: { if !$0 { app.error = nil } })) {
            if app.backend == nil { Button("Retry") { Task { await app.start() } } }
            Button("OK", role: .cancel) { app.error = nil }
        } message: { Text(app.error ?? "") }
        .alert("RallyClip", isPresented: Binding(get: { app.message != nil }, set: { if !$0 { app.message = nil } })) {
            Button("OK") { app.message = nil }
        } message: { Text(app.message ?? "") }
    }
    var welcome: some View {
        VStack(spacing: 22) {
            RallyLogo(height: 70)
            Text("Analyze a match").font(.largeTitle.weight(.semibold))
            Text("Choose a video or a folder of consecutive MP4 clips.").foregroundStyle(.secondary)
            HStack(spacing: 14) {
                Button { app.chooseImport(folder: false) } label: { Label("Choose a Video", systemImage: "film").padding(7) }
                    .buttonStyle(.borderedProminent).keyboardShortcut("o")
                Button { app.chooseImport(folder: true) } label: { Label("Camera Folder", systemImage: "folder").padding(7) }
                    .buttonStyle(.bordered)
            }.disabled(app.busy || app.backend == nil)
            Text("Saved analyses appear in the sidebar.").font(.callout).foregroundStyle(.secondary).padding(.top, 12)
        }.padding(40).frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct AnalysisCard: View {
    @EnvironmentObject var app: AppModel
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(app.busy ? "Analyzing match" : app.job?.status == "cancelled" ? "Analysis cancelled" : app.job?.status == "failed" ? "Analysis failed" : "Analysis finished").font(.headline)
                Spacer()
                if !app.busy { Button { app.job = nil } label: { Image(systemName: "xmark") }.buttonStyle(.plain) }
            }
            if app.importing { ProgressView(); Text("Checking footage…").font(.caption) }
            else if let job = app.job {
                ProgressView(value: job.progress)
                HStack { Text(stageLabel(job.stage)).lineLimit(1); Spacer(); Text("\(Int(job.progress * 100))%").monospacedDigit() }
                    .font(.caption).foregroundStyle(.secondary)
                if let eta = job.eta_seconds, eta > 0, app.busy { Text("About \(formatTime(eta)) remaining").font(.caption).foregroundStyle(.secondary) }
            } else { ProgressView() }
            if app.busy { Button("Cancel Analysis", role: .cancel) { app.cancel() }.font(.caption) }
        }.padding(14).background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 12))
    }
    func stageLabel(_ stage: String) -> String {
        ["proxy": "Preparing footage", "pose": "Finding player motion", "preprocess": "Selecting players",
         "feature": "Building features", "inference": "Detecting points", "output": "Saving match"][stage] ?? stage
    }
}

func formatTime(_ seconds: Double) -> String {
    guard seconds.isFinite else { return "0:00" }
    let s = max(0, Int(seconds))
    return s >= 3600 ? String(format: "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60) : String(format: "%d:%02d", s / 60, s % 60)
}
