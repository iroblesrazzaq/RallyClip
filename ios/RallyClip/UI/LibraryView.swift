import SwiftUI
import UIKit

/// Saved-matches grid. Mirrors `#libraryView`: logo + title, "Process a Match",
/// empty state, and cards (thumbnail, name, meta, Export / CSV / Delete).
struct LibraryView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.palette) private var palette
    @State private var share: ShareItem?
    @State private var pendingDelete: MatchMeta?

    private let columns = [GridItem(.adaptive(minimum: 260), spacing: 20)]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                HStack {
                    Text("Saved matches").font(.serif(34, weight: .bold)).foregroundStyle(palette.ink)
                    Spacer()
                    Button(model.library.isEmpty ? "Process a Match" : "New Match") { model.showUpload() }
                        .buttonStyle(RCButtonStyle(kind: .primary, palette: palette))
                }
                if model.library.isEmpty {
                    emptyState
                } else {
                    LazyVGrid(columns: columns, spacing: 20) {
                        ForEach(model.library) { item in card(item) }
                    }
                }
            }
            .padding(24)
            .frame(maxWidth: 1120)
            .frame(maxWidth: .infinity)
        }
        .sheet(item: $share) { ShareSheet(items: [$0.url]) }
        .confirmationDialog("Delete this match? This also deletes its CSV.",
                            isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }),
                            presenting: pendingDelete) { m in
            Button("Delete", role: .destructive) { model.deleteMatch(m); pendingDelete = nil }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Text("No matches yet").font(.serif(27, weight: .bold))
            Text("Drop in a full match and RallyClip saves just the points here.")
                .font(.serif(16)).foregroundStyle(palette.inkSoft)
        }
        .frame(maxWidth: .infinity).padding(48)
        .background(palette.canvas)
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(palette.hairline))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func card(_ item: MatchMeta) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            thumbnail(item)
            VStack(alignment: .leading, spacing: 4) {
                Text(item.name).font(.serif(17, weight: .semibold)).lineLimit(2)
                Text(item.metaLine).font(.serif(14)).foregroundStyle(palette.inkSoft)
            }.padding(.horizontal, 16).padding(.top, 14).padding(.bottom, 6)
            HStack(spacing: 8) {
                Button("Export video") { Task { if let u = await model.exportMatch(item) { share = ShareItem(url: u) } } }
                    .buttonStyle(RCButtonStyle(kind: .primary, palette: palette, small: true))
                Button("CSV") { if let u = csvTempURL(item) { share = ShareItem(url: u) } }
                    .buttonStyle(RCButtonStyle(kind: .secondary, palette: palette, small: true))
                Button("Delete") { pendingDelete = item }
                    .buttonStyle(RCButtonStyle(kind: .ghost, palette: palette, small: true))
                    .foregroundStyle(palette.danger)
            }.padding(16)
        }
        .background(palette.canvas)
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(palette.hairline))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .contentShape(Rectangle())
        .onTapGesture { model.openMatch(item) }
    }

    private func thumbnail(_ item: MatchMeta) -> some View {
        let url = MatchStore.shared.thumbURL(item.id)
        return Group {
            if let img = UIImage(contentsOfFile: url.path) {
                Image(uiImage: img).resizable().aspectRatio(16.0/9.0, contentMode: .fill)
            } else {
                Rectangle().fill(palette.hairline.opacity(0.4)).aspectRatio(16.0/9.0, contentMode: .fit)
            }
        }
        .frame(maxWidth: .infinity)
        .clipped()
    }

    private func csvTempURL(_ item: MatchMeta) -> URL? {
        guard let data = MatchStore.shared.csvData(item.id) else { return nil }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(item.name)_segments.csv")
        try? data.write(to: url)
        return url
    }
}

struct ShareItem: Identifiable { let id = UUID(); let url: URL }

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
