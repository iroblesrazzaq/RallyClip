# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, copy_metadata

root = Path(SPECPATH).resolve().parents[1]
artifact = root / "models/rallyclip_v0.6.0"
a = Analysis(
    [str(root / "src/gui/native_backend.py")],
    pathex=[str(root / "src")],
    datas=[
        (str(artifact), "models/rallyclip_v0.6.0"),
        (str(root / "src/preprocessing/default_court_mask.png"), "preprocessing"),
        (str(root / "src/gui/frontend"), "gui/frontend"),
        (str(root / "docs/rallyclip_favicon_transparent2.png"), "docs"),
        (str(root / "docs/rallyclip_logo_cropped.png"), "docs"),
    ] + copy_metadata("rallyclip"),
    binaries=[],
    hiddenimports=["gui.app", "gui.analysis_worker", "cli.main", "runtime.assets", "runtime.device",
                   "runtime.defaults", "extraction.yolo_onnx_runner", "onnxruntime", "psutil"]
                  + collect_submodules("flask"),
    excludes=["torch", "torchvision", "ultralytics", "PySide6", "PyQt5", "PyQt6", "webview",
              "tensorflow", "keras", "matplotlib", "IPython", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="RallyClipBackend", console=True,
          strip=False, upx=False, argv_emulation=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="RallyClipBackend")
