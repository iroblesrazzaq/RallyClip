# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

_SPEC_DIR = Path(SPECPATH).resolve().parent
with (_SPEC_DIR / "pyproject.toml").open("rb") as _fh:
    _VERSION = tomllib.load(_fh)["project"]["version"]

# Keep this string in lockstep with runtime.defaults.DEFAULT_ARTIFACT_DIR
# (tests/test_release_packaging.py enforces that). Torch-free bundle: pose
# runs on the ONNX in this folder (extraction.yolo_onnx_runner + onnxruntime).
_DEFAULT_ARTIFACT_DIR = "models/rallyclip_v0.5.0"
_BUNDLE_IDENTIFIER = "com.iroblesrazzaq.rallyclip"

datas = [
    ("src/gui/frontend", "gui/frontend"),
    (_DEFAULT_ARTIFACT_DIR, _DEFAULT_ARTIFACT_DIR),
    ("src/preprocessing/default_court_mask.png", "preprocessing"),
    ("docs/rallyclip.icns", "docs"),
    ("docs/rallyclip_logo.svg", "docs"),
    ("docs/rallyclip_app_icon.svg", "docs"),
    ("docs/rallyclip_logo_cropped.png", "docs"),
    ("docs/rallyclip_favicon_transparent2.png", "docs"),
]
binaries = []
hiddenimports = [
    "gui.app",
    "gui.analysis_worker",
    "cli.main",
    "runtime.assets",
    "runtime.device",
    "runtime.defaults",
    "runtime.paths",
    "extraction.yolo_onnx_runner",
    "onnxruntime",
    "psutil",
    "webview",
]
hiddenimports += collect_submodules("flask")
tmp_ret = collect_all("psutil")
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]
# The in-app update check reads importlib.metadata.version("rallyclip");
# without the dist-info the frozen app falls back to a hardcoded 0.1.0 and
# nags about every release — including older ones.
datas += copy_metadata("rallyclip")


a = Analysis(
    ["src/gui/desktop.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt5",
        "PyQt6",
        "PySide2",
        "openvino",
        "torch",
        "torchvision",
        "ultralytics",
        "PySide6",
        "shiboken6",
        "tensorflow",
        "keras",
        "tf_keras",
        "tensorflow_hub",
        "tensorboard",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RallyClip",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file="packaging/macos/RallyClip.entitlements",
    icon=["docs/rallyclip.icns"],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="RallyClip",
)
app = BUNDLE(
    coll,
    name="RallyClip.app",
    icon="docs/rallyclip.icns",
    bundle_identifier=_BUNDLE_IDENTIFIER,
    entitlements_file="packaging/macos/RallyClip.entitlements",
    info_plist={
        "CFBundleName": "RallyClip",
        "CFBundleDisplayName": "RallyClip",
        "CFBundleIdentifier": _BUNDLE_IDENTIFIER,
        "CFBundleShortVersionString": _VERSION,
        "CFBundleVersion": _VERSION,
        "CFBundlePackageType": "APPL",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        "LSApplicationCategoryType": "public.app-category.sports",
    },
)
