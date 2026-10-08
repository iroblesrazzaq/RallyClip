// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "RallyClipNative",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "RallyClip", targets: ["RallyClip"]),
               .executable(name: "RallyClipProbe", targets: ["RallyClipProbe"])],
    targets: [
        .target(name: "RallyClipKit"),
        .executableTarget(name: "RallyClip", dependencies: ["RallyClipKit"], resources: [.copy("Resources/rallyclip_logo_cropped.png")]),
        .executableTarget(name: "RallyClipProbe", dependencies: ["RallyClipKit"]),
        .testTarget(name: "RallyClipKitTests", dependencies: ["RallyClipKit"])
    ]
)
