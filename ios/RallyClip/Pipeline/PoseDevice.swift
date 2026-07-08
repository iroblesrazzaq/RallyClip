import Foundation

#if canImport(UIKit)
import UIKit
#endif

/// Which execution backend to run pose on. Mirrors `runtime/device.py`
/// (`_DEVICE_ORDER = cuda, coreml, mps, cpu`) reduced to the iOS-relevant set:
/// CoreML (Apple Neural Engine, static export) with a CPU parity fallback.
enum PoseDevice: String, CaseIterable, Identifiable {
    case coreml
    case cpu
    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .coreml: return "CoreML (fast, Apple)"
        case .cpu: return "CPU"
        }
    }

    /// CoreML EP is only usable on real Apple-silicon devices with a Neural
    /// Engine; the simulator (x86/arm mac host) has no ANE, so auto = CPU there.
    /// Mirrors `device.coreml_pose_available`.
    static var coremlAvailable: Bool {
        #if targetEnvironment(simulator)
        return false
        #elseif arch(arm64)
        return true
        #else
        return false
        #endif
    }

    /// Auto pick: CoreML when available, else CPU. Mirrors `resolve_auto_device`.
    static var auto: PoseDevice { coremlAvailable ? .coreml : .cpu }

    static var selectable: [PoseDevice] {
        coremlAvailable ? [.coreml, .cpu] : [.cpu]
    }
}
