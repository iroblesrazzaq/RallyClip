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
