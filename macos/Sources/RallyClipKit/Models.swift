import Foundation

public struct Point: Codable, Equatable, Hashable, Sendable {
    public var start: Double
    public var end: Double
    public var duration: Double { end - start }
    public init(start: Double, end: Double) { self.start = start; self.end = end }
}

public struct Source: Codable, Equatable, Sendable {
    public var path: String
    public var start_s: Double
    public var duration_s: Double
    public var end: Double { start_s + duration_s }
    public init(path: String, start: Double, duration: Double) {
        self.path = path; start_s = start; duration_s = duration
    }
}

public struct Match: Codable, Identifiable, Hashable, Sendable {
    public let id: String
    public let name: String
    public let duration_s: Double
    public let point_duration_s: Double
    public let n_segments: Int
    public let created: String
    /// Seconds since 1970; older libraries may not have it.
    public let created_ts: Double?
    public let has_edits: Bool
}

public struct SegmentResponse: Codable, Sendable {
    public let segments: [Point]
    public let edited: Bool
}

public struct Job: Codable, Sendable {
    public struct Step: Codable, Sendable { public let status: String; public let progress: Double }
    public let status: String
    public let steps: [String: Step]
    public let error: String?
    public let eta_seconds: Double?
    public let library_id: String?
    public var progress: Double {
        guard !steps.isEmpty else { return 0 }
        return steps.values.reduce(0) { $0 + $1.progress } / Double(steps.count) / 100
    }
    public var stage: String {
        ["proxy", "pose", "preprocess", "feature", "inference", "output"]
            .first { steps[$0]?.status == "in_progress" } ?? "Preparing"
    }
}

public enum RallyError: LocalizedError {
    case message(String)
    public var errorDescription: String? { if case .message(let text) = self { return text }; return nil }
}

public enum PointMath {
    public static func validate(_ points: [Point], duration: Double) throws -> [Point] {
        guard points.count <= 2000 else { throw RallyError.message("A match can contain at most 2,000 points.") }
        let sorted = points.sorted { $0.start < $1.start }
        for (i, p) in sorted.enumerated() {
            guard p.start.isFinite, p.end.isFinite, p.start >= 0, p.end > p.start,
                  p.end <= duration + 0.05 else { throw RallyError.message("Point times must be inside the match, with the end after the start.") }
            if i > 0 && sorted[i - 1].end > p.start { throw RallyError.message("Points cannot overlap.") }
        }
        return sorted
    }

    public static func sourceTime(at playbackTime: Double, points: [Point]) -> Double {
        var cursor = 0.0
        for p in points {
            if playbackTime < cursor + p.duration { return p.start + max(0, playbackTime - cursor) }
            cursor += p.duration
        }
        return points.last?.end ?? 0
    }

    /// Where a Points Only seek to `sourceTime` lands: a time in a gap moves to the next point's start.
    public static func landedTime(at sourceTime: Double, points: [Point]) -> Double {
        self.sourceTime(at: playbackTime(at: sourceTime, points: points), points: points)
    }

    public static func playbackTime(at sourceTime: Double, points: [Point]) -> Double {
        var cursor = 0.0
        for p in points {
            if sourceTime <= p.end { return cursor + max(0, sourceTime - p.start) }
            cursor += p.duration
        }
        return cursor
    }
}

/// Camera clips in a chosen folder, in natural name order. Skips hidden files
/// (the ._ AppleDouble twins Finder writes on exFAT cards) and containers
/// AVFoundation cannot play (.mkv/.avi/.mts import through the web GUI only).
public enum FolderClips {
    public static let extensions: Set<String> = ["mp4", "mov", "m4v"]
    public static func select(_ urls: [URL]) -> [URL] {
        urls.filter { extensions.contains($0.pathExtension.lowercased()) && !$0.lastPathComponent.hasPrefix(".") }
            .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
    }
}
