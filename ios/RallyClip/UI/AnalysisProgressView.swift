import SwiftUI

/// Processing screen. Mirrors `#progress`: overall bar + stage headline, Cancel,
/// and an expandable per-stage breakdown (pose/preprocess/feature/inference/output).
struct AnalysisProgressView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.palette) private var palette
    @State private var showDetails = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text(model.stageText).font(.serif(17, weight: .bold))
                        Spacer()
                        Text("\(Int(model.overall.rounded()))%").font(.serif(16))
                    }
                    bar(model.overall / 100, color: palette.ink)
                }
                .padding(20)
                .background(palette.canvas)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(palette.hairline))
                .clipShape(RoundedRectangle(cornerRadius: 8))

                Button("Cancel") { model.cancelAnalysis() }
                    .buttonStyle(RCButtonStyle(kind: .secondary, palette: palette))

                DisclosureGroup("Details", isExpanded: $showDetails) {
                    VStack(spacing: 12) {
                        ForEach(ProgressStage.allCases, id: \.self) { stage in
                            let ev = model.stages[stage]
                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Text(stage.label).font(.serif(15))
                                    Spacer()
                                    Text(statusText(ev?.status ?? .waiting)).font(.serif(14)).foregroundStyle(palette.inkSoft)
                                }
                                bar(Double(ev?.progress ?? 0) / 100, color: palette.ink)
                            }
                        }
                    }.padding(.top, 12)
                }
                .font(.serif(16, weight: .bold))
                .tint(palette.ink)
            }
            .padding(24)
            .frame(maxWidth: 640)
            .frame(maxWidth: .infinity)
        }
    }

    private func bar(_ fraction: Double, color: Color) -> some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(palette.hairline.opacity(0.6))
                Capsule().fill(color).frame(width: geo.size.width * min(1, max(0, fraction)))
            }
        }
        .frame(height: 8)
        .animation(.easeInOut(duration: 0.2), value: fraction)
    }

    private func statusText(_ s: StageStatus) -> String {
        switch s {
        case .waiting: return "Waiting"
        case .inProgress: return "Running"
        case .completed: return "Done"
        case .failed: return "Failed"
        case .cancelled: return "Cancelled"
        }
    }
}
