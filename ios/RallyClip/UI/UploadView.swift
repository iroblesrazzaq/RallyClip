import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

/// Import + Advanced panel. Mirrors `#uploadView`: pick a match, show the file,
/// Start, and Advanced (device / hysteresis low / high / min segment length).
struct UploadView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.palette) private var palette
    @State private var pickerItem: PhotosPickerItem?
    @State private var showAdvanced = false
    @State private var importing = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Button("← Saved matches") { model.showLibrary() }
                    .buttonStyle(.plain).foregroundStyle(palette.ink).font(.serif(16))

                if model.pickedURL == nil {
                    PhotosPicker(selection: $pickerItem, matching: .videos) {
                        VStack(spacing: 8) {
                            Text(importing ? "Importing…" : "Drop a match video").font(.serif(34, weight: .bold))
                            Text("Browse videos").font(.serif(16)).foregroundStyle(palette.inkSoft)
                            Text("MP4 · up to 2GB").font(.serif(15)).foregroundStyle(palette.inkSoft)
                        }
                        .frame(maxWidth: .infinity, minHeight: 240)
                        .background(palette.canvas)
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(palette.ink))
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                } else {
                    HStack {
                        Text(model.pickedName).font(.serif(17, weight: .bold))
                        Spacer()
                        Button { model.pickedURL = nil; pickerItem = nil } label: { Image(systemName: "xmark") }
                            .foregroundStyle(palette.ink)
                    }
                    .padding(16)
                    .background(palette.canvas)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(palette.hairline))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }

                HStack {
                    Button("Start") { model.startAnalysis() }
                        .buttonStyle(RCButtonStyle(kind: .primary, palette: palette))
                        .disabled(model.pickedURL == nil)
                    Button(showAdvanced ? "Hide advanced" : "Advanced") { showAdvanced.toggle() }
                        .buttonStyle(RCButtonStyle(kind: .ghost, palette: palette))
                }

                if showAdvanced { advancedPanel }
            }
            .padding(24)
            .frame(maxWidth: 640)
            .frame(maxWidth: .infinity)
        }
        .onChange(of: pickerItem) { _, newValue in Task { await importVideo(newValue) } }
    }

    private var advancedPanel: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Advanced settings").font(.serif(19, weight: .bold))
            Text("Defaults follow the bundled model manifest. Change only if you know why.")
                .font(.serif(15)).foregroundStyle(palette.inkSoft)

            field("Match name") {
                TextField("Defaults to video name", text: Binding(
                    get: { model.config.outputName ?? "" },
                    set: { model.config.outputName = $0.isEmpty ? nil : $0 }))
                .textFieldStyle(.roundedBorder)
            }
            field("Device") {
                Picker("Device", selection: Binding(
                    get: { model.config.device },
                    set: { model.config.device = $0 })) {
                    Text("Auto (\(PoseDevice.auto.displayName))").tag(PoseDevice?.none)
                    ForEach(PoseDevice.selectable) { d in Text(d.displayName).tag(PoseDevice?.some(d)) }
                }.pickerStyle(.menu)
                Text(deviceNote).font(.serif(13)).foregroundStyle(palette.inkSoft)
            }
            numberField("Hysteresis low", $model.config.low)
            numberField("Hysteresis high", $model.config.high)
            numberField("Min segment length (sec)", $model.config.minDurSec)
        }
        .padding(20)
        .background(palette.canvas)
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(palette.hairline))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var deviceNote: String {
        switch model.config.device {
        case .none: return "Auto picks \(PoseDevice.auto.displayName) on this device."
        case .coreml: return "Runs pose on the Apple Neural Engine — much faster; results can differ minutely from CPU."
        case .cpu: return "Using CPU for pose extraction (byte-parity reference path)."
        }
    }

    private func field<C: View>(_ label: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 6) { Text(label).font(.serif(15)); content() }
    }
    private func numberField(_ label: String, _ value: Binding<Double>) -> some View {
        field(label) {
            TextField(label, value: value, format: .number).textFieldStyle(.roundedBorder).keyboardType(.decimalPad)
        }
    }

    private func importVideo(_ item: PhotosPickerItem?) async {
        guard let item else { return }
        importing = true; defer { importing = false }
        if let movie = try? await item.loadTransferable(type: VideoFile.self) {
            model.pickedURL = movie.url
            model.pickedName = movie.url.deletingPathExtension().lastPathComponent
        } else {
            model.toast("Could not import that video.", .error)
        }
    }
}

/// Copies a picked video out of the Photos sandbox into our temp dir.
struct VideoFile: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { SentTransferredFile($0.url) } importing: { received in
            let dst = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension(received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension)
            try? FileManager.default.removeItem(at: dst)
            try FileManager.default.copyItem(at: received.file, to: dst)
            return VideoFile(url: dst)
        }
    }
}
