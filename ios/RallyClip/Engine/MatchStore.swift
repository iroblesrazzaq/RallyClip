import Foundation
import UIKit

/// On-device match library under Application Support/RallyClip/<id>/.
/// Mirrors the desktop library contract (`rallyclip_core/library.py`):
///   source.mov, segments.csv (model output, never rewritten),
///   segments_edited.csv (edits win), export.mp4 (lazy, invalidated on edit),
///   thumb.jpg, meta.json.
final class MatchStore {
    static let shared = MatchStore()
    private let root: URL
    private let fm = FileManager.default

    init() {
        let base = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        root = base.appendingPathComponent("RallyClip", isDirectory: true)
        try? fm.createDirectory(at: root, withIntermediateDirectories: true)
    }

    private func dir(_ id: String) -> URL { root.appendingPathComponent(id, isDirectory: true) }
    func sourceURL(_ id: String) -> URL { dir(id).appendingPathComponent("source.mov") }
    func exportURL(_ id: String) -> URL { dir(id).appendingPathComponent("export.mp4") }
    func highlightURL(_ id: String) -> URL { dir(id).appendingPathComponent("highlight.mp4") }
    private func pointsDir(_ id: String) -> URL { dir(id).appendingPathComponent("points", isDirectory: true) }
    func pointsZipURL(_ id: String) -> URL { dir(id).appendingPathComponent("points.zip") }
    private func metaURL(_ id: String) -> URL { dir(id).appendingPathComponent("meta.json") }
    private func csvURL(_ id: String) -> URL { dir(id).appendingPathComponent("segments.csv") }
    private func editedURL(_ id: String) -> URL { dir(id).appendingPathComponent("segments_edited.csv") }
    func thumbURL(_ id: String) -> URL { dir(id).appendingPathComponent("thumb.jpg") }

    // MARK: - list / read

    func list() -> [MatchMeta] {
        guard let ids = try? fm.contentsOfDirectory(atPath: root.path) else { return [] }
        return ids.compactMap { loadMeta($0) }.sorted { $0.createdISO > $1.createdISO }
    }

    func loadMeta(_ id: String) -> MatchMeta? {
        guard let data = try? Data(contentsOf: metaURL(id)) else { return nil }
        return try? JSONDecoder().decode(MatchMeta.self, from: data)
    }

    /// Effective segments: edited copy wins, else the original. Mirrors `resolve_segments`.
    func segments(_ id: String) -> [Segment] {
        if fm.fileExists(atPath: editedURL(id).path), let s = readCSV(editedURL(id)) { return s }
        return readCSV(csvURL(id)) ?? []
    }

    // MARK: - create

    /// Persist a finished analysis. Returns nil if there were no points (the
    /// desktop still shows "no points detected" and saves nothing).
    @discardableResult
    func create(sourceTempURL: URL, name: String, durationS: Double, segments: [Segment],
                thumbnail: UIImage?) -> MatchMeta? {
        guard !segments.isEmpty else { return nil }
        let id = UUID().uuidString
        let d = dir(id)
        try? fm.createDirectory(at: d, withIntermediateDirectories: true)
        try? fm.copyItem(at: sourceTempURL, to: sourceURL(id))
        writeCSV(segments, to: csvURL(id))
        if let thumbnail, let jpg = thumbnail.jpegData(compressionQuality: 0.8) { try? jpg.write(to: thumbURL(id)) }
        let meta = MatchMeta(id: id, name: name, createdISO: ISO8601DateFormatter().string(from: Date()),
                             durationS: durationS, sourceFilename: sourceTempURL.lastPathComponent,
                             nSegments: segments.count, pointDurationS: pointDuration(segments), hasEdits: false)
        writeMeta(meta)
        return meta
    }

    // MARK: - edits

    func saveEditedSegments(_ id: String, _ segments: [Segment]) {
        writeCSV(segments, to: editedURL(id))
        invalidateExports(id)   // all cut artifacts are now stale
        updateMeta(id) { $0.hasEdits = true; $0.nSegments = segments.count; $0.pointDurationS = pointDuration(segments) }
    }

    func resetEdits(_ id: String) {
        try? fm.removeItem(at: editedURL(id))
        invalidateExports(id)
        let original = readCSV(csvURL(id)) ?? []
        updateMeta(id) { $0.hasEdits = false; $0.nSegments = original.count; $0.pointDurationS = pointDuration(original) }
    }

    /// Drop every derived clip so the next export reflects the current segments.
    private func invalidateExports(_ id: String) {
        try? fm.removeItem(at: exportURL(id))
        try? fm.removeItem(at: highlightURL(id))
        try? fm.removeItem(at: pointsZipURL(id))
        try? fm.removeItem(at: pointsDir(id))
    }

    func delete(_ id: String) { try? fm.removeItem(at: dir(id)) }

    /// Lazily build (or reuse) the point-only export.
    func ensureExport(_ id: String) async throws -> URL {
        let url = exportURL(id)
        if fm.fileExists(atPath: url.path) { return url }
        try await ClipExporter.export(sourceURL: sourceURL(id), segments: segments(id), to: url)
        return url
    }

    /// Concatenate a user-selected subset of points into one highlight clip.
    /// `indices` index into `segments(id)` (the effective, edited-wins list).
    /// Rebuilt each call since the selection varies. Mirrors the desktop
    /// `/highlight?points=…` route.
    func buildHighlight(_ id: String, indices: [Int]) async throws -> URL {
        let all = segments(id)
        let picked = indices.sorted().filter { all.indices.contains($0) }.map { all[$0] }
        guard !picked.isEmpty else { throw PipelineError.export("no points selected") }
        let url = highlightURL(id)
        try await ClipExporter.export(sourceURL: sourceURL(id), segments: picked, to: url)
        return url
    }

    /// Each point as its own clip, zipped (`point_01.mp4`, …). Cached until an
    /// edit invalidates it. Mirrors the desktop `/points.zip` route.
    func ensurePointsZip(_ id: String) async throws -> URL {
        let zip = pointsZipURL(id)
        if fm.fileExists(atPath: zip.path) { return zip }
        try await ClipExporter.exportIndividual(sourceURL: sourceURL(id), segments: segments(id), to: pointsDir(id))
        try zipDirectory(pointsDir(id), to: zip)
        return zip
    }

    /// Zip a directory via NSFileCoordinator's `.forUploading` option (the
    /// system's built-in directory→.zip path; no third-party archiver).
    private func zipDirectory(_ directory: URL, to dest: URL) throws {
        let coordinator = NSFileCoordinator()
        var coordError: NSError?
        var copyError: Error?
        coordinator.coordinate(readingItemAt: directory, options: [.forUploading], error: &coordError) { tmpZip in
            do {
                try? fm.removeItem(at: dest)
                try fm.copyItem(at: tmpZip, to: dest)
            } catch { copyError = error }
        }
        if let coordError { throw coordError }
        if let copyError { throw copyError }
    }

    func csvData(_ id: String) -> Data? {
        let url = fm.fileExists(atPath: editedURL(id).path) ? editedURL(id) : csvURL(id)
        return try? Data(contentsOf: url)
    }

    // MARK: - helpers

    private func pointDuration(_ segs: [Segment]) -> Double { segs.reduce(0) { $0 + max(0, $1.end - $1.start) } }

    private func writeMeta(_ m: MatchMeta) {
        if let data = try? JSONEncoder().encode(m) { try? data.write(to: metaURL(m.id)) }
    }
    private func updateMeta(_ id: String, _ mutate: (inout MatchMeta) -> Void) {
        guard var m = loadMeta(id) else { return }
        mutate(&m); writeMeta(m)
    }

    /// CSV: `start_time,end_time` header + `%.3f` rows (matches write_segments_csv).
    private func writeCSV(_ segs: [Segment], to url: URL) {
        var out = "start_time,end_time\n"
        for s in segs { out += String(format: "%.3f,%.3f\n", s.start, s.end) }
        try? out.write(to: url, atomically: true, encoding: .utf8)
    }
    private func readCSV(_ url: URL) -> [Segment]? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var segs: [Segment] = []
        for (i, line) in text.split(separator: "\n").enumerated() {
            if i == 0 { continue }
            let parts = line.split(separator: ",")
            guard parts.count >= 2, let a = Double(parts[0]), let b = Double(parts[1]), b > a else { continue }
            segs.append(Segment(start: a, end: b))
        }
        return segs.sorted { $0.start < $1.start }
    }
}
