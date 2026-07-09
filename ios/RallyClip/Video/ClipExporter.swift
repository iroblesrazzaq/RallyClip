import Foundation
import AVFoundation

/// Concatenate the kept point intervals into one point-only .mp4.
/// Native analogue of `segmentation/segment.py`: merges overlapping/touching
/// intervals, then splices each into a single gap-free composition (video+audio)
/// re-timed sequentially, and exports H.264/AAC.
enum ClipExporter {

    static func export(sourceURL: URL, segments: [Segment], to outputURL: URL) async throws {
        guard !segments.isEmpty else { throw PipelineError.export("no points to export") }

        // Merge overlapping/touching intervals (matches segment_video's pre-merge).
        let eps = 1e-6
        let sorted = segments.sorted { $0.start < $1.start || ($0.start == $1.start && $0.end < $1.end) }
        var merged: [Segment] = []
        for s in sorted {
            if var last = merged.last, s.start <= last.end + eps {
                last.end = max(last.end, s.end); merged[merged.count - 1] = last
            } else { merged.append(s) }
        }

        let asset = AVURLAsset(url: sourceURL)
        let composition = AVMutableComposition()
        guard let srcVideo = try await asset.loadTracks(withMediaType: .video).first,
              let compVideo = composition.addMutableTrack(withMediaType: .video,
                                                          preferredTrackID: kCMPersistentTrackID_Invalid) else {
            throw PipelineError.export("no video track")
        }
        compVideo.preferredTransform = try await srcVideo.load(.preferredTransform)
        let srcAudio = try await asset.loadTracks(withMediaType: .audio).first
        let compAudio = srcAudio == nil ? nil
            : composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)

        var cursor = CMTime.zero
        for seg in merged {
            let start = CMTime(seconds: seg.start, preferredTimescale: 600)
            let dur = CMTime(seconds: max(0, seg.end - seg.start), preferredTimescale: 600)
            let range = CMTimeRange(start: start, duration: dur)
            try compVideo.insertTimeRange(range, of: srcVideo, at: cursor)
            if let srcAudio, let compAudio {
                try? compAudio.insertTimeRange(range, of: srcAudio, at: cursor)
            }
            cursor = cursor + dur
        }

        try? FileManager.default.removeItem(at: outputURL)
        guard let session = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetHighestQuality) else {
            throw PipelineError.export("could not create export session")
        }
        session.outputURL = outputURL
        session.outputFileType = .mp4
        session.shouldOptimizeForNetworkUse = true
        await session.export()
        if session.status != .completed {
            throw PipelineError.export(session.error?.localizedDescription ?? "export failed")
        }
    }

    /// Cut each point into its own clip (`point_01.mp4`, `point_02.mp4`, … in
    /// chronological order) inside `directory`, which is recreated empty first.
    /// Native analogue of the desktop `points.zip` build (one `segment_video`
    /// call per interval). Returns the written clip URLs.
    @discardableResult
    static func exportIndividual(sourceURL: URL, segments: [Segment], to directory: URL) async throws -> [URL] {
        guard !segments.isEmpty else { throw PipelineError.export("no points to export") }
        let fm = FileManager.default
        try? fm.removeItem(at: directory)
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let sorted = segments.sorted { $0.start < $1.start || ($0.start == $1.start && $0.end < $1.end) }
        var urls: [URL] = []
        for (i, seg) in sorted.enumerated() {
            let out = directory.appendingPathComponent(String(format: "point_%02d.mp4", i + 1))
            try await export(sourceURL: sourceURL, segments: [seg], to: out)
            urls.append(out)
        }
        return urls
    }
}
