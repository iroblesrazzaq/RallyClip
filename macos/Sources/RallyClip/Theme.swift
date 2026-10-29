import AppKit
import SwiftUI

/// The SwiftPM resource bundle. Not Bundle.module: it looks only at the .app root
/// (codesign rejects files there, so build.sh puts it in Contents/Resources) and at
/// the absolute build path, and traps when neither exists, as on any Mac but the builder.
private let resourceBundle: Bundle? = {
    let name = "RallyClipNative_RallyClip.bundle"
    let dirs = [Bundle.main.resourceURL, Bundle.main.bundleURL]  // .app, then `swift run`
    return dirs.lazy.compactMap { $0.flatMap { Bundle(url: $0.appendingPathComponent(name)) } }.first
}()

/// The logo PNG; nil shows no logo. Also read by `RallyClip --check-resources`.
let logoURL = resourceBundle?.url(forResource: "rallyclip_logo_cropped", withExtension: "png")

/// Reuse the original logo in both system appearances.
struct RallyLogo: View {
    @Environment(\.colorScheme) private var scheme
    var height: CGFloat = 32
    var body: some View {
        if let url = logoURL, let image = NSImage(contentsOf: url) {
            Group {
                if scheme == .dark { Image(nsImage: image).resizable().scaledToFit().colorInvert() }
                else { Image(nsImage: image).resizable().scaledToFit() }
            }.frame(width: height * 1024 / 616, height: height).accessibilityLabel("RallyClip")
        }
    }
}

extension Color {
    /// The intro screen's tennis green: dark green in light mode, light green in dark mode.
    /// Packaged builds also set it as the app accent (packaging/macos/Assets.xcassets).
    static let rallyGreen = Color(nsColor: NSColor(name: "rallyGreen") { appearance in
        appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            ? NSColor(srgbRed: 0x7b / 255, green: 0xd6 / 255, blue: 0x3a / 255, alpha: 1)
            : NSColor(srgbRed: 0x2f / 255, green: 0x8f / 255, blue: 0x3a / 255, alpha: 1)
    })
    /// Tennis-ball green for the playhead; it shows on the green point bars in both modes.
    static let rallyBall = Color(red: 0xc8 / 255, green: 0xf0 / 255, blue: 0x3c / 255)
}
