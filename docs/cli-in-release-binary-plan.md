# Proposal: headless CLI mode in the shipped desktop binary

Status: implemented (2026-06-10) · Effort: small (~1–2 hrs incl. CI) · Owner: TBD

> 2026-07-04: code snippets below show the QtWebEngine-era desktop.py; the shell
> is now pywebview and the runtime is torch-free, but the --cli dispatch contract
> described here is unchanged.

> Implementation note: the smoke step in §2 was hardened beyond this plan.
> A bare `exit code == 1` check cannot distinguish the clean missing-video
> path from a broken-asset-resolution traceback (uncaught exceptions also
> exit 1), so the CI step additionally greps for `video file not found` and
> rejects `Traceback`. It also hides the repo's `models/` and runs from a
> neutral cwd, since `candidate_roots()` would otherwise satisfy resolution
> from the checkout instead of the bundle.

## 1. Problem

The release artifact (PyInstaller `--onedir` bundle built from `src/gui/desktop.py`,
see `.github/workflows/release.yml`) can only launch the GUI. There is no headless
path through the **shipped** binary, which blocks:

1. **Release e2e tests** — the planned eval harness (5 videos × 3 min, CSV vs GT,
   `compute_time_point_classification_metrics`) needs to run the *release binary*
   end-to-end and read its segments CSV. Today only the dev `rallyclip` CLI can do that.
2. **Frozen-path validation** — `build_run_config` / `resolve_asset` have never executed
   under `sys.frozen`. The GUI uses its own config path, so the manifest-driven CLI rails
   are untested in the bundle.
3. **Field debugging** — no way to exercise the pipeline on a user's machine without
   installing Python.

## 2. Design

### CLI contract

```bash
dist/RallyClip/RallyClip --cli --video match.mp4 --start-time 1240 --duration 180 \
  --write-csv --csv-output-dir /tmp/out --no-segment-video
```

- `--cli` must be **argv[1] exactly**. It is stripped and everything after it is handed
  to the existing `cli.main.main()` argparse untouched. No new flags, no duplicated parser.
- Exit code is whatever `cli.main.main()` returns. Without `--cli`, behavior is
  byte-identical to today (GUI launches).
- `RallyClip --cli --help` prints the normal `rallyclip` help.

### Code change — `src/gui/desktop.py`

Add a dispatch at the very top of `main()`, **before** any Qt import or
`QApplication(sys.argv)`:

```python
def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # Lazy import is deliberate (documented exception to the no-inline-imports
        # rule): cli.main pulls torch/ultralytics; importing it at module top would
        # add seconds to GUI startup before the splash screen can appear.
        from cli.main import main as cli_main

        sys.argv = [sys.argv[0], *sys.argv[2:]]
        return cli_main()

    try:
        from PySide6.QtCore import ...
```

Notes:

- `cli.main.main()` itself special-cases `argv[1] == "gui"`; we do not forward `--cli gui`
  support — `--cli` means headless, full stop. (If someone passes `--cli gui` they get the
  Flask `launch()`, which is harmless but undocumented; do not add handling for it.)
- Do NOT restructure `cli.main` — the dispatch is additive.

### Build change — `.github/workflows/release.yml`

1. Add `--hidden-import=cli.main` to the `pyinstaller` invocation. The lazy import in the
   dispatch branch is invisible to PyInstaller's static analysis (the existing
   `--hidden-import=gui.app` exists for the same reason). The transitive deps
   (torch, ultralytics, numpy, flask) are already collected via `gui.app` /
   `--collect-all=ultralytics`; `tomllib` is stdlib on the pinned Python 3.11.
2. Add a **packaging smoke step** after the build (all three OSes):

```bash
# 1) Frozen import chain + argparse works
dist/RallyClip/RallyClip --cli --help

# 2) Frozen asset resolution works: build_run_config must resolve the BUNDLED
#    model.onnx + manifest.json via _MEIPASS, then fail cleanly on the missing video.
set +e
dist/RallyClip/RallyClip --cli --video /nonexistent/missing.mp4 --no-segment-video
code=$?
set -e
test "$code" -eq 1
```

Check (2) is the valuable one: it executes `build_run_config` → `resolve_asset` →
`manifest_values` inside the frozen bundle (the `_MEIPASS` root in
`runtime.assets.candidate_roots()` is what's being proven), resolves the full contract,
and only then hits the `video file not found` early-return in `run_pipeline`
([src/cli/main.py:255-257](../src/cli/main.py#L255)) → exit code 1. If frozen asset
resolution is broken it raises `FileNotFoundError`/`SystemExit` with a different
code/traceback and the step fails. On Windows use the matching `bash` shell step
(the workflow already uses `shell: bash` everywhere).

## 3. Known frozen-CLI behaviors (accept, don't fix here)

- `run_pipeline` sets `models_dir = Path.cwd() / "models"` for YOLO weight downloads
  ([src/cli/main.py:271](../src/cli/main.py#L271)). In frozen mode weights are not
  bundled (release.yml comment, lines 39–41) so first run downloads to the user's
  cwd/Ultralytics cache. Fine for the eval harness; document in README.
- `--csv-output-dir` defaults to the video's directory; the harness always passes it
  explicitly, so no change to the frozen-GUI `~/RallyClip/...` redirect logic is needed.
- `config.toml` discovery is cwd-relative (`_load_config_dict`); headless invocations
  should pass flags explicitly and not rely on a config file next to the binary.

## 4. Tests

- **Unit** (`tests/test_desktop_dispatch.py`): monkeypatch `sys.argv` +
  `cli.main.main`, assert `desktop.main()` routes to the CLI with `--cli` stripped and
  returns its exit code; assert non-`--cli` argv does not import/call it (monkeypatch the
  import or assert Qt path is attempted). Keep it import-light: patch `cli.main.main`
  before calling.
- **CI smoke**: the two release.yml steps above (no videos, no GT needed).
- The full release e2e (5 videos × 3 min, metric floors) is a separate workstream
  (eval harness); this proposal only provides the entry point it shells out to.

## 5. Docs

- README: one short "Headless mode" subsection under the release/download section:
  `RallyClip --cli --video ... [same flags as rallyclip]`.

## 6. Acceptance criteria

1. `RallyClip` (no args) launches the GUI exactly as before.
2. `RallyClip --cli --help` exits 0 and prints the `rallyclip` help, from the built
   bundle on macOS, Windows, Linux (verified in release.yml).
3. `RallyClip --cli --video <missing>` exits 1 after resolving the bundled
   model/manifest (verified in release.yml smoke).
4. `RallyClip --cli --video <real.mp4> --start-time S --duration D --write-csv
   --csv-output-dir <tmp> --no-segment-video` produces `<stem>_segments.csv` with
   `start_time,end_time` rows (manual verification once, on a built bundle).
5. Unit dispatch test passes in the normal suite; no new `slow`/`e2e` markers needed.
6. `pytest -m "not slow and not e2e"` stays green; no behavior change for dev
   `rallyclip` / `rallyclip gui`.

## 7. Out of scope

- The eval harness itself (video subset manifest, GT clipping, metric floors).
- Bundling YOLO weights into the release.
- Any change to GUI config handling or `~/RallyClip` data roots.
