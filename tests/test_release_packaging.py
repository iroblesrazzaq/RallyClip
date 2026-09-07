from __future__ import annotations

import ast
import subprocess
import tomllib
from pathlib import Path

from runtime.defaults import DEFAULT_ARTIFACT_DIR

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "release"


def _pyproject_version() -> str:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(payload["project"]["version"])


def test_default_artifact_dir_has_shipped_weights():
    root = ROOT / DEFAULT_ARTIFACT_DIR
    for name in (
        "model.onnx",
        "scaler.json",
        "manifest.json",
        "yolov8n-pose-960-dynamic.onnx",
        "yolov8n-pose-544x960-static.onnx",
    ):
        assert (root / name).is_file(), f"missing {name} under {root}"


def test_pyinstaller_spec_bundles_default_artifact_dir():
    spec = (ROOT / "RallyClip.spec").read_text(encoding="utf-8")
    assert f'_DEFAULT_ARTIFACT_DIR = "{DEFAULT_ARTIFACT_DIR}"' in spec
    assert "_BUNDLE_IDENTIFIER = \"com.iroblesrazzaq.rallyclip\"" in spec
    assert "bundle_identifier=_BUNDLE_IDENTIFIER" in spec
    assert "CFBundleShortVersionString" in spec


def test_spec_bundle_identifier_is_set():
    tree = ast.parse((ROOT / "RallyClip.spec").read_text(encoding="utf-8"))
    assigned = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    ident = assigned["_BUNDLE_IDENTIFIER"]
    assert isinstance(ident, ast.Constant)
    assert ident.value == "com.iroblesrazzaq.rallyclip"


def test_release_scripts_are_valid_bash():
    scripts = sorted(SCRIPTS.glob("*.sh"))
    names = {path.name for path in scripts}
    assert names >= {
        "import_apple_cert.sh",
        "lib.sh",
        "make_macos_dmg.sh",
        "notarize_macos_dmg.sh",
        "package_macos.sh",
        "sign_macos_app.sh",
    }
    for path in scripts:
        result = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{path.name}: {result.stderr}"


def test_release_scripts_usage_without_args():
    for name in (
        "sign_macos_app.sh",
        "make_macos_dmg.sh",
        "notarize_macos_dmg.sh",
        "package_macos.sh",
    ):
        result = subprocess.run(
            ["bash", str(SCRIPTS / name)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2, name
        assert "Usage:" in result.stderr


def test_release_lib_project_version_matches_pyproject():
    script = r"""
set -euo pipefail
source scripts/release/lib.sh
release_project_version
"""
    result = subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.stdout.strip() == _pyproject_version()


def test_release_lib_dmg_basename():
    script = r"""
set -euo pipefail
source scripts/release/lib.sh
release_dmg_basename 0.3.0 arm64
"""
    result = subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.stdout.strip() == "RallyClip-0.3.0-macOS-arm64.dmg"


def test_release_workflow_uses_spec_and_signing_pipeline():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "pyinstaller --noconfirm RallyClip.spec" in workflow
    assert "scripts/release/import_apple_cert.sh" in workflow
    assert "scripts/release/package_macos.sh" in workflow
    assert "MACOS_CERTIFICATE_P12_BASE64" in workflow
    assert "APPSTORE_API_PRIVATE_KEY" in workflow
    assert "dist/RallyClip.app/Contents/MacOS/RallyClip" in workflow
    assert "models/rallyclip_v0.3.1" not in workflow
    assert "timeout-minutes: 180" in workflow


def test_notarize_script_submits_then_waits():
    script = (SCRIPTS / "notarize_macos_dmg.sh").read_text(encoding="utf-8")
    assert "notarytool submit" in script
    assert "notarytool wait" in script
    assert "RALLYCLIP_NOTARY_SUBMISSION_ID" in script
    assert "RALLYCLIP_NOTARY_TIMEOUT:-2h" in script
