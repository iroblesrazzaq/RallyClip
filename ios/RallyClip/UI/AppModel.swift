import SwiftUI
import Combine

enum AppScreen { case welcome, library, upload, processing, viewer }

struct Toast: Identifiable, Equatable {
    enum Tone { case info, success, error }
    let id = UUID()
    let message: String
    let tone: Tone
}

/// App-wide state machine — the SwiftUI analogue of `RallyClipApp` in script.js
/// (welcome → library → upload → processing → viewer), plus the analysis job and
/// library CRUD. Everything the frontend did via `/api/*` happens here directly.
@MainActor
final class AppModel: ObservableObject {
    @Published var screen: AppScreen = .library
    @Published var library: [MatchMeta] = []
    @Published var selected: MatchMeta?
    @Published var toasts: [Toast] = []

    // Upload / config
    @Published var pickedURL: URL?
    @Published var pickedName: String = ""
    @Published var config: AnalysisConfig

    // Progress
    @Published var stages: [ProgressStage: ProgressEvent] = [:]
    @Published var overall: Double = 0
    @Published var stageText: String = "Ready"
    @Published var isProcessing = false

    let contract: ModelContract
    private var jobTask: Task<Void, Never>?
    private let welcomeKey = "rallyclip_welcome_seen"

    init() {
        let loaded = (try? ModelContract.loadBundled())
        self.contract = loaded ?? ModelContract(
            conf: 0.25, featureDim: 362, imgsz: 960, numKeypoints: 17, sampleFps: 5,
            screenWidth: 1280, screenHeight: 720, poseModelName: "yolov8n-pose-960-dynamic.onnx",
            inputName: "features", outputName: "logits", seqLen: 100, overlap: 50,
            low: 0.45, high: 0.7, minDurSec: 1.0, sigma: 1.0)
        self.config = AnalysisConfig.defaults(contract)
        let seen = UserDefaults.standard.bool(forKey: welcomeKey)
        screen = seen ? .library : .welcome
        refreshLibrary()
        // ViewerModel is built in ViewerView.init (before .onAppear), so it
        // needs to reach the app model via this holder from the start.
        AppModelHolder.current = self
    }

    // MARK: - navigation

    func dismissWelcome() {
        UserDefaults.standard.set(true, forKey: welcomeKey)
        screen = .library
    }
    func showLibrary() { selected = nil; refreshLibrary(); screen = .library }
    func showUpload() { resetConfig(); pickedURL = nil; pickedName = ""; screen = .upload }
    func resetConfig() { config = AnalysisConfig.defaults(contract) }

    func refreshLibrary() { library = MatchStore.shared.list() }

    // MARK: - toasts

    func toast(_ message: String, _ tone: Toast.Tone = .info) {
        let t = Toast(message: message, tone: tone)
        toasts.append(t)
        Task { try? await Task.sleep(nanoseconds: 4_000_000_000); toasts.removeAll { $0.id == t.id } }
    }

    // MARK: - analysis

    func startAnalysis() {
        guard let url = pickedURL, !isProcessing else { return }
        UserDefaults.standard.set(true, forKey: welcomeKey)
        resetProgress()
        isProcessing = true
        stageText = "Preparing"
        screen = .processing
        let job = AnalysisJob(sourceURL: url, displayName: pickedName, config: config, contract: contract)
        jobTask = Task {
            do {
                let meta = try await job.run { [weak self] ev in
                    Task { @MainActor in self?.apply(ev) }
                }
                await MainActor.run {
                    self.isProcessing = false
                    self.refreshLibrary()
                    self.screen = .library
                    if meta != nil { self.toast("Saved to your matches.", .success) }
                    else { self.toast("No tennis points detected in this video.", .info) }
                }
            } catch is CancellationError {
                await MainActor.run { self.finishCancelled() }
            } catch let e as PipelineError where e.errorDescription == PipelineError.cancelled.errorDescription {
                await MainActor.run { self.finishCancelled() }
            } catch {
                await MainActor.run {
                    self.isProcessing = false
                    self.toast(error.localizedDescription, .error)
                    self.screen = .upload
                }
            }
        }
    }

    func cancelAnalysis() { jobTask?.cancel() }

    private func finishCancelled() {
        isProcessing = false
        toast("Job cancelled.", .success)
        screen = .library
    }

    private func apply(_ ev: ProgressEvent) {
        stages[ev.stage] = ev
        let sum = ProgressStage.allCases.reduce(0.0) { $0 + Double(stages[$1]?.progress ?? 0) }
        overall = sum / Double(ProgressStage.allCases.count)
        stageText = currentStageLabel()
    }

    private func currentStageLabel() -> String {
        if let running = ProgressStage.allCases.first(where: { stages[$0]?.status == .inProgress }) {
            return running.activeLabel
        }
        if let waiting = ProgressStage.allCases.first(where: { (stages[$0]?.status ?? .waiting) == .waiting }) {
            return waiting == .pose ? "Starting" : waiting.activeLabel
        }
        return "Finishing"
    }

    private func resetProgress() {
        stages = [:]; overall = 0; stageText = "Ready"
    }

    // MARK: - library actions

    func openMatch(_ m: MatchMeta) { selected = m; screen = .viewer }
    func deleteMatch(_ m: MatchMeta) {
        MatchStore.shared.delete(m.id); refreshLibrary(); toast("Match deleted.", .success)
    }

    func exportMatch(_ m: MatchMeta) async -> URL? {
        do { return try await MatchStore.shared.ensureExport(m.id) }
        catch { toast("Could not export the clip.", .error); return nil }
    }
}
