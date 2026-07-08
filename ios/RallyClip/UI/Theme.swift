import SwiftUI

extension Color {
    init(hex: UInt32) {
        self.init(.sRGB,
                  red: Double((hex >> 16) & 0xFF) / 255,
                  green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255,
                  opacity: 1)
    }
}

/// Design tokens ported from `frontend/styles.css` `:root` (+ dark variant).
/// Serif editorial look: cream paper, black ink, tennis-neon accent, pill buttons.
struct Palette {
    let paper: Color, canvas: Color, ink: Color, inkSoft: Color
    let hairline: Color, accent: Color, onAccent: Color
    let tennis: Color, danger: Color, success: Color, warning: Color

    static let light = Palette(
        paper: Color(hex: 0xFBF4E8), canvas: Color(hex: 0xFFF8EC), ink: Color(hex: 0x050505),
        inkSoft: Color(hex: 0x4C4C4C), hairline: Color(hex: 0xD9D7D1), accent: Color(hex: 0x050505),
        onAccent: .white, tennis: Color(hex: 0xCCFF00), danger: Color(hex: 0xF21D2F),
        success: Color(hex: 0x087F5B), warning: Color(hex: 0x9A6200))

    static let dark = Palette(
        paper: Color(hex: 0x050505), canvas: Color(hex: 0x0B0B0B), ink: Color(hex: 0xF7F6EF),
        inkSoft: Color(hex: 0xBFBDB5), hairline: Color(hex: 0x323232), accent: Color(hex: 0xF7F6EF),
        onAccent: Color(hex: 0x050505), tennis: Color(hex: 0xCCFF00), danger: Color(hex: 0xF21D2F),
        success: Color(hex: 0x7BD63A), warning: Color(hex: 0x9A6200))

    static func resolve(_ scheme: ColorScheme) -> Palette { scheme == .dark ? .dark : .light }
}

private struct PaletteKey: EnvironmentKey { static let defaultValue = Palette.light }
extension EnvironmentValues {
    var palette: Palette {
        get { self[PaletteKey.self] }
        set { self[PaletteKey.self] = newValue }
    }
}

extension Font {
    /// Georgia serif, matching the desktop headings/body.
    static func serif(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .custom("Georgia", size: size).weight(weight)
    }
}

/// Pill button styles matching `.btn` / `.btn-primary` / `.btn-secondary`.
struct RCButtonStyle: ButtonStyle {
    enum Kind { case primary, secondary, ghost }
    let kind: Kind
    let palette: Palette
    var small = false

    func makeBody(configuration: Configuration) -> some View {
        let bg: Color = {
            switch kind {
            case .primary: return palette.accent
            case .secondary: return palette.canvas
            case .ghost: return .clear
            }
        }()
        let fg: Color = kind == .primary ? palette.onAccent : palette.ink
        return configuration.label
            .font(.serif(small ? 15 : 17, weight: .bold))
            .padding(.horizontal, small ? 14 : 20)
            .padding(.vertical, small ? 7 : 12)
            .background(bg)
            .foregroundStyle(fg)
            .overlay(Capsule().stroke(kind == .ghost ? .clear : palette.ink, lineWidth: 1))
            .clipShape(Capsule())
            .opacity(configuration.isPressed ? 0.75 : 1)
            .scaleEffect(configuration.isPressed ? 0.99 : 1)
    }
}
