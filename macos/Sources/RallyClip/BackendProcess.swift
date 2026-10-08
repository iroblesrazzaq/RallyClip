import AppKit
import Foundation
import RallyClipKit

@MainActor
final class BackendProcess {
    private var process: Process?
    private var readyDirectory: URL?
    private var logHandle: FileHandle?

    func start() async throws -> Backend {
        let environment = ProcessInfo.processInfo.environment
        let root = environment["RALLYCLIP_NATIVE_DATA_DIR"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("RallyClip")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("rallyclip-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                               attributes: [.posixPermissions: 0o700])
        readyDirectory = directory
        let ready = directory.appendingPathComponent("ready.json")
        let token = UUID().uuidString + UUID().uuidString
        let child = Process()
        let bundled = Bundle.main.resourceURL?.appendingPathComponent("Backend/RallyClipBackend")
        var env = environment
        if let bundled, FileManager.default.isExecutableFile(atPath: bundled.path) {
            child.executableURL = bundled
        } else if let checkout = environment["RALLYCLIP_NATIVE_REPO"], let python = environment["RALLYCLIP_NATIVE_PYTHON"] {
            child.executableURL = URL(fileURLWithPath: python)
            child.arguments = ["-m", "gui.native_backend"]
            child.currentDirectoryURL = URL(fileURLWithPath: checkout)
            env["PYTHONPATH"] = URL(fileURLWithPath: checkout).appendingPathComponent("src").path
        } else { throw RallyError.message("The bundled analysis engine is missing. Reinstall RallyClip or use scripts/native/run.sh from a checkout.") }
        env["RALLYCLIP_NATIVE_TOKEN"] = token
        env["RALLYCLIP_NATIVE_READY_FILE"] = ready.path
        env["RALLYCLIP_JOBS_DIR"] = root.appendingPathComponent("jobs").path
        env["RALLYCLIP_LIBRARY_DIR"] = root.appendingPathComponent("library").path
        env["RALLYCLIP_OUTPUT_DIR"] = root.appendingPathComponent("output_videos").path
        env["RALLYCLIP_CSV_DIR"] = root.appendingPathComponent("output_csvs").path
        env["RALLYCLIP_LOG_DIR"] = root.appendingPathComponent("logs").path
        env["PYTHONUNBUFFERED"] = "1"
        child.environment = env
        let log = root.appendingPathComponent("native-backend.log")
        FileManager.default.createFile(atPath: log.path, contents: nil)
        logHandle = try FileHandle(forWritingTo: log)
        child.standardOutput = logHandle; child.standardError = logHandle
        try child.run(); process = child
        do {
            for _ in 0..<400 {
                try Task.checkCancellation()
                if let data = try? Data(contentsOf: ready),
                   let object = try? JSONSerialization.jsonObject(with: data) as? [String: Int], let port = object["port"] {
                    let backend = Backend(port: port, token: token)
                    let _: [String: String] = try await backend.request("api/health")
                    return backend
                }
                guard child.isRunning else { throw RallyError.message("The analysis engine stopped during startup. See \(log.path).") }
                try await Task.sleep(for: .milliseconds(150))
            }
            throw RallyError.message("The analysis engine took too long to start. See \(log.path).")
        } catch { stop(); throw error }
    }

    func stop() {
        if let process, process.isRunning { process.terminate() }
        process = nil
        if let readyDirectory { try? FileManager.default.removeItem(at: readyDirectory) }
        try? logHandle?.close(); logHandle = nil
    }
}
