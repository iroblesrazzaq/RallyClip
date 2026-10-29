import AVFoundation
import Foundation
import RallyClipKit

@main
struct Probe {
    struct Input: Decodable { let sources: [Source]; let segments: [Point] }
    @MainActor static func main() async throws {
        guard CommandLine.arguments.count == 3 else { throw RallyError.message("usage: RallyClipProbe input.json output.mp4") }
        let input = try JSONDecoder().decode(Input.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
        let destination = URL(fileURLWithPath: CommandLine.arguments[2])
        let composition = try await CompositionBuilder.build(sources: input.sources, points: input.segments)
        let player = AVPlayer(playerItem: AVPlayerItem(asset: composition))
        let output = AVPlayerItemVideoOutput(pixelBufferAttributes: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        player.currentItem!.add(output)
        player.play()
        let start = Date()
        var lastTime = 0.0, began = false, samples = 0, maxJump = 0.0, frames = 0
        var lastAdvance = Date(), maxStall = 0.0
        while Date().timeIntervalSince(start) < min(30, composition.duration.seconds + 12) {
            try await Task.sleep(for: .milliseconds(20))
            let time = player.currentTime().seconds
            guard time.isFinite else { continue }
            if time > 0.01 {
                if !began { began = true; lastAdvance = Date() }
                if time > lastTime + 0.001 {
                    maxJump = max(maxJump, time - lastTime)
                    maxStall = max(maxStall, Date().timeIntervalSince(lastAdvance))
                    lastAdvance = Date()
                }
                samples += 1
            }
            let stamp = player.currentTime()
            if output.hasNewPixelBuffer(forItemTime: stamp), output.copyPixelBuffer(forItemTime: stamp, itemTimeForDisplay: nil) != nil { frames += 1 }
            lastTime = time
            if time >= composition.duration.seconds - 0.05 { break }
        }
        player.pause()
        guard began, samples > 5, frames > 0, lastTime >= composition.duration.seconds - 0.1 else {
            throw RallyError.message("Native playback did not reach the end with decoded video frames (\(lastTime), \(frames)).")
        }
        try await VideoExporter().export(sources: input.sources, points: input.segments, to: destination) { _ in }
        let asset = AVURLAsset(url: destination)
        let duration = try await asset.load(.duration).seconds
        let report: [String: Any] = ["composition_duration": composition.duration.seconds, "export_duration": duration,
                                     "playback_samples": samples, "decoded_frames": frames,
                                     "largest_media_jump": maxJump, "longest_observed_stall": maxStall]
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        print(String(decoding: data, as: UTF8.self))
    }
}
