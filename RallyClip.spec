# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

# Torch-free bundle: pose runs on the ONNX inside models/rallyclip_v0.3.1
# (extraction.yolo_onnx_runner + onnxruntime); no .pt weights, no ultralytics.
datas = [('src/gui/frontend', 'gui/frontend'), ('models/rallyclip_v0.3.1', 'models/rallyclip_v0.3.1'), ('src/preprocessing/default_court_mask.png', 'preprocessing'), ('docs/rallyclip.icns', 'docs'), ('docs/rallyclip_logo.svg', 'docs'), ('docs/rallyclip_app_icon.svg', 'docs'), ('docs/rallyclip_logo_cropped.png', 'docs'), ('docs/rallyclip_favicon_transparent2.png', 'docs')]
binaries = []
hiddenimports = ['gui.app', 'gui.analysis_worker', 'cli.main', 'runtime.assets', 'runtime.device', 'runtime.defaults', 'runtime.paths', 'extraction.yolo_onnx_runner', 'onnxruntime', 'psutil', 'webview']
hiddenimports += collect_submodules('flask')
tmp_ret = collect_all('psutil')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['src/gui/desktop.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6', 'PySide2', 'openvino', 'torch', 'torchvision', 'ultralytics', 'PySide6', 'shiboken6', 'tensorflow', 'keras', 'tf_keras', 'tensorflow_hub', 'tensorboard'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RallyClip',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file='packaging/macos/RallyClip.entitlements',
    icon=['docs/rallyclip.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RallyClip',
)
app = BUNDLE(
    coll,
    name='RallyClip.app',
    icon='docs/rallyclip.icns',
    bundle_identifier=None,
    entitlements_file='packaging/macos/RallyClip.entitlements',
)
