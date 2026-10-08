import AppKit
import SwiftUI

/// Reuse the original logo in both system appearances.
struct RallyLogo: View {
    @Environment(\.colorScheme) private var scheme
    var height: CGFloat = 32
    var body: some View {
        if let url = Bundle.module.url(forResource: "rallyclip_logo_cropped", withExtension: "png"),
           let image = NSImage(contentsOf: url) {
            Group {
                if scheme == .dark { Image(nsImage: image).resizable().scaledToFit().colorInvert() }
                else { Image(nsImage: image).resizable().scaledToFit() }
            }.frame(width: height * 1024 / 616, height: height).accessibilityLabel("RallyClip")
        }
    }
}
