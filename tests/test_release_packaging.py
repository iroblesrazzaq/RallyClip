from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

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
    assert "_spec_path.parent if _spec_path.is_file() else _spec_path" in spec


def _exec_spec_version_block(specpath: str) -> str:
    spec_lines = (ROOT / "RallyClip.spec").read_text(encoding="utf-8").splitlines()
    block: list[str] = []
    capturing = False
    for line in spec_lines:
        if line.startswith("_spec_path") or (
            not capturing and line.startswith("_SPEC_DIR")
        ):
            capturing = True
        if capturing:
            block.append(line)
            if "_VERSION" in line and "tomllib" in line:
                break
    ns = {"SPECPATH": specpath, "Path": Path, "tomllib": tomllib}
    exec("\n".join(block), ns)
    return str(ns["_VERSION"])


def test_spec_reads_pyproject_version_when_specpath_is_directory():
    # PyInstaller sets SPECPATH to the spec file's directory.
    assert _exec_spec_version_block(str(ROOT)) == _pyproject_version()


def test_spec_reads_pyproject_version_when_specpath_is_spec_file():
    assert _exec_spec_version_block(str(ROOT / "RallyClip.spec")) == _pyproject_version()


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
    if sys.platform == "win32":
        pytest.skip("macOS release scripts; Windows CI has no usable bash")
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
    if sys.platform == "win32":
        pytest.skip("macOS release scripts; Windows CI has no usable bash")
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
    if sys.platform == "win32":
        pytest.skip("macOS release scripts; Windows CI has no usable bash")
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


def test_release_lib_project_version_independent_of_cwd(tmp_path):
    if sys.platform == "win32":
        pytest.skip("macOS release scripts; Windows CI has no usable bash")
    script = f"""
set -euo pipefail
source "{SCRIPTS / "lib.sh"}"
release_project_version
"""
    result = subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.stdout.strip() == _pyproject_version()


def test_release_lib_dmg_basename():
    if sys.platform == "win32":
        pytest.skip("macOS release scripts; Windows CI has no usable bash")
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
    assert "Require Apple Silicon runner" in workflow
    assert "uname -m" in workflow
    assert "RALLYCLIP_SKIP_NOTARIZE=1" in workflow
    assert "Upload signed DMG artifact" in workflow
    assert "Notarize and staple DMG" in workflow
    assert "always() && steps.package.outputs.dmg_path != ''" in workflow
    signed_upload = workflow.index("Upload signed DMG artifact")
    notarize = workflow.index("Notarize and staple DMG")
    assert signed_upload < notarize


def test_package_script_records_dmg_before_notarize():
    script = (SCRIPTS / "package_macos.sh").read_text(encoding="utf-8")
    first_write = script.index("\nwrite_dmg_outputs\n")
    notarize = script.index("notarize_macos_dmg.sh")
    assert first_write < notarize
    assert script.count("write_dmg_outputs") >= 3


def test_release_workflow_keeps_signing_secrets_off_build_steps():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    _, rest = workflow.split("  build:\n", 1)
    header, steps = rest.split("    steps:\n", 1)
    secret_p12 = "MACOS_CERTIFICATE_P12_BASE64: ${{ secrets.MACOS_CERTIFICATE_P12_BASE64 }}"
    secret_p8 = "APPSTORE_API_PRIVATE_KEY: ${{ secrets.APPSTORE_API_PRIVATE_KEY }}"
    assert secret_p12 not in header
    assert secret_p8 not in header
    assert "RALLYCLIP_HAS_SIGNING_CERT" in header
    assert secret_p12 in steps
    assert secret_p8 in steps
    assert "Install build deps" in steps
    install_idx = steps.index("Install build deps")
    assert steps.index(secret_p12) > install_idx
    assert steps.index(secret_p8) > install_idx


def test_import_cert_writes_p12_into_private_tempdir():
    script = (SCRIPTS / "import_apple_cert.sh").read_text(encoding="utf-8")
    assert "mktemp -d" in script
    assert "umask 077" in script
    assert "chmod 600" in script
    assert "chmod 700" in script


def test_notarize_script_submits_then_waits():
    script = (SCRIPTS / "notarize_macos_dmg.sh").read_text(encoding="utf-8")
    assert "notarytool submit" in script
    assert "notarytool wait" in script
    assert "RALLYCLIP_NOTARY_SUBMISSION_ID" in script
    assert "RALLYCLIP_NOTARY_TIMEOUT:-2h" in script
    assert "mktemp -d" in script
    assert "umask 077" in script
    assert "chmod 600" in script
