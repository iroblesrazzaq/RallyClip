@preconcurrency import AVFoundation
import Foundation

/// One continuous media timeline. Point boundaries are composition edits, not
/// repeated seeks of an independently buffered HTTP video element.
public enum CompositionBuilder {
    public static func build(sources: [Source], points: [Point]) async throws -> AVMutableComposition {
        let composition = AVMutableComposition()
        guard let video = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid),
              let audio = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) else {
            throw RallyError.message("Could not create the playback timeline.")
        }
        let total = sources.map(\.end).max() ?? 0
        // validate allows an end up to 0.05 s past the footage; clip it so the coverage check holds.
        let intervals = try PointMath.validate(points, duration: total).map { Point(start: $0.start, end: min($0.end, total)) }
        // AVAssetTrack holds a weak reference to its asset. Keep the owning
        // assets alive until every source range has been copied.
        var assets: [AVURLAsset] = []
        defer { withExtendedLifetime(assets) {} }
        var loaded: [(Source, AVAssetTrack, AVAssetTrack?, CMTimeRange?)] = []
        // Open only the clips a point touches: one point of a long folder match needs one or two.
        for source in sources where intervals.contains(where: { min($0.end, source.end) > max($0.start, source.start_s) }) {
            try Task.checkCancellation()
            let url = URL(fileURLWithPath: source.path)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw RallyError.message("Reconnect the original footage: \(url.lastPathComponent)")
            }
            let asset = AVURLAsset(url: url)
            assets.append(asset)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                throw RallyError.message("No video track in \(url.lastPathComponent).")
            }
            let sound = try await asset.loadTracks(withMediaType: .audio).first
            let soundRange = try await sound?.load(.timeRange)
            loaded.append((source, track, sound, soundRange))
        }
        // One composition track has one transform. Folder checks compare size and
        // frame rate only, so reject clips that disagree on rotation.
        var transforms: [CGAffineTransform] = []
        for (_, track, _, _) in loaded { transforms.append(try await track.load(.preferredTransform)) }
        if let first = transforms.first {
            guard transforms.allSatisfy({ $0 == first }) else {
                throw RallyError.message("The clips in this match have different rotations. Import clips recorded in one orientation.")
            }
            video.preferredTransform = first
        }
        var cursor = CMTime.zero
        for point in intervals {
            var covered = 0.0
            for (source, track, sound, soundRange) in loaded {
                let lo = max(point.start, source.start_s), hi = min(point.end, source.end)
                guard hi > lo else { continue }
                let start = CMTime(seconds: lo - source.start_s, preferredTimescale: 60000)
                let duration = CMTime(seconds: hi - lo, preferredTimescale: 60000)
                let range = CMTimeRange(start: start, duration: duration)
                try video.insertTimeRange(range, of: track, at: cursor)
                if let sound, let soundRange {
                    let overlap = CMTimeRangeGetIntersection(range, otherRange: soundRange)
                    if overlap.duration > .zero {
                        let position = cursor + overlap.start - start
                        let end = CMTimeRangeGetEnd(audio.timeRange)
                        // Track insertion beyond its end does not create a
                        // reliable silent gap. Represent it explicitly so late
                        // audio remains aligned with its original video.
                        if end < position { audio.insertEmptyTimeRange(CMTimeRange(start: end, end: position)) }
                        try audio.insertTimeRange(overlap, of: sound, at: position)
                    }
                }
                cursor = cursor + duration
                covered += hi - lo
            }
            guard abs(covered - point.duration) < 0.05 else {
                throw RallyError.message("Original clips do not cover the selected point.")
            }
        }
        if composition.duration < cursor { composition.insertEmptyTimeRange(CMTimeRange(start: composition.duration, end: cursor)) }
        if !audio.segments.contains(where: { !$0.isEmpty }) { composition.removeTrack(audio) }
        else if CMTimeRangeGetEnd(audio.timeRange) < cursor {
            audio.insertEmptyTimeRange(CMTimeRange(start: CMTimeRangeGetEnd(audio.timeRange), end: cursor))
        }
        return composition
    }
}

public actor VideoExporter {
    private var active: AVAssetExportSession?
    public init() {}
    public func cancel() { active?.cancelExport() }
    public func export(sources: [Source], points: [Point], to destination: URL,
                       progress: @escaping @Sendable (Double) -> Void) async throws {
        guard !points.isEmpty else { throw RallyError.message("Select at least one point to export.") }
        let composition = try await CompositionBuilder.build(sources: sources, points: points)
        try Task.checkCancellation()
        guard let session = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetHighestQuality) else {
            throw RallyError.message("This footage cannot be exported.")
        }
        let temporary = destination.deletingLastPathComponent().appendingPathComponent(".rallyclip-\(UUID().uuidString).mp4")
        session.outputURL = temporary; session.outputFileType = .mp4
        session.shouldOptimizeForNetworkUse = true
        let audioTracks = composition.tracks(withMediaType: .audio)
        if !audioTracks.isEmpty {
            // Render silence into gaps rather than passing through an AAC edit
            // list that other players may collapse or repeat audio across.
            let mix = AVMutableAudioMix()
            mix.inputParameters = audioTracks.map { track in
                let parameters = AVMutableAudioMixInputParameters(track: track)
                parameters.setVolume(1, at: .zero)
                return parameters
            }
            session.audioMix = mix
        }
        active = session
        defer { active = nil; try? FileManager.default.removeItem(at: temporary) }
        let monitor = Task {
            while !Task.isCancelled {
                progress(Double(session.progress))
                try? await Task.sleep(for: .milliseconds(150))
            }
        }
        defer { monitor.cancel() }
        await withTaskCancellationHandler { await session.export() } onCancel: { Task { await self.cancel() } }
        try Task.checkCancellation()
        guard session.status == .completed else {
            throw session.error ?? RallyError.message("Video export was cancelled.")
        }
        if FileManager.default.fileExists(atPath: destination.path) {
            _ = try FileManager.default.replaceItemAt(destination, withItemAt: temporary)
        } else { try FileManager.default.moveItem(at: temporary, to: destination) }
        progress(1)
    }
}
