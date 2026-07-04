# RallyClip — E2E Test Plan (first release)

Status: L1 BUILT + green; CI gate wired. Scope = the `feat/first-release` desktop
app shipping `models/rallyclip_v0.3.1/` (hysteresis model). e2e v0.4.0 is out of scope.

## Build log

- **L1 implemented** in `tests/test_gui_e2e.py` (+ `tests/helpers/e2e_backend.py`):
  11 tests, real backend on a free port, real pipeline on the synthetic clip.
  `pytest -m e2e` → 11 passed (~16s, 2 real pose runs). Fast suite unchanged (96).
- **CI gate wired**: new `e2e` job in `.github/workflows/ci.yml` runs `pytest -m e2e`
  on PR/push (installs `libgl1` for cv2, fetches nano YOLO weights). `requests`
  added to the `dev` extra. `testpaths=["tests"]` added so collection ignores the
  local `dist_venv/`/`build_venv/` build dirs.
- **Two facts learned while building (corrections to the original L2/L3 ideas):**
  1. The "force a full-clip segment via low=high=0" trick is INVALID —
     `hysteresis_threshold` asserts `0.0 <= low < high <= 1.0`. And on a synthetic
     clip the model never fires on real thresholds (no humans → no points). So a
     real downloadable video CANNOT be produced end-to-end on synthetic input.
     → The video-download e2e instead asserts it *reflects reality*: a valid MP4
     if points were found, else a clean 404. The video *encode* itself is already
     covered directly by `tests/test_segment.py` (concat/audio/video-only/errors).
  2. For the same reason, extending the L3 frozen-bundle step to "also produce a
     segmented video" adds NO coverage (zero segments → `segment_video` never
     runs). That L3 add is dropped; it needs a real-match clip to be meaningful.
- **Golden clip fixture BUILT.** `scripts/make_e2e_clip.py` cuts a window out of a
  real training match (reusing `segment_video`) + offsets the annotations ->
  `data/e2e/<name>/{clip.mp4,golden.json}` (gitignored — private footage). First
  fixture: `aditi_5pts` (130s, 5 points). Golden tests in `tests/test_gui_e2e.py`
  upload it through the backend, score detected vs labels with a self-contained
  6-bin (`tests/helpers/golden_metrics.py`), and assert a loose floor
  (acceptable_frac >= 0.8, fp <= 1) + a real downloadable segmented MP4 — the
  ingest -> detect -> cut -> download path synthetic clips can't reach. Calibration
  run: 5/5 detected, 0 FP, 0 FN (good 2 / decent 3), acceptable 1.0. Optimistic
  because the clip is training data — fine for a regression fixture. Self-skips in
  CI (no footage); the synthetic L1 stays the CI gate.
- **L2 Playwright BUILT.** `tests/test_gui_playwright.py` drives the real frontend
  in headless Chromium: welcome -> pick file -> Start -> progress -> results +
  a real CSV download, plus cancel. Faithful proxy for the desktop webview (same
  HTML/JS/backend). Deps in the `e2e-ui` extra; CI e2e job installs them +
  `playwright install --with-deps chromium`. 3 tests, ~17s.
- **State**: fast suite 96 passed (unchanged); `pytest -m e2e` = 17 passed (11 L1
  + 3 golden + 3 L2) locally, golden self-skips in CI. L1+L2 run in the CI e2e job.

## 0. What we are actually shipping (the thing under test)

The release artifact is a **native desktop app** (PyInstaller onedir bundle), not a
browser app. At launch the Qt shell (`src/gui/desktop.py`) boots the Flask backend
(`src/gui/app.py`) in a thread on an ephemeral localhost port, then renders the
frontend (`src/gui/frontend/*`) inside an embedded Chromium webview
(`QWebEngineView`) pointed at `http://127.0.0.1:<port>/`. The webview talks to the
backend ONLY through 6 REST endpoints:

| endpoint | method | purpose |
| --- | --- | --- |
| `/` | GET | serve index.html |
| `/api/health` | GET | liveness |
| `/api/config/defaults` | GET | defaults + device list for the UI |
| `/api/upload-and-start` | POST | upload video + start a job |
| `/api/progress/<job_id>` | GET | poll job state/percent |
| `/api/cancel/<job_id>` | POST | cancel a running job |
| `/api/download/{video,csv}/<job_id>` | GET | fetch outputs |

Because the desktop webview uses exactly these endpoints, **driving the backend over
HTTP is a faithful reproduction of the desktop user journey** — minus webview
rendering and minus PyInstaller/Qt packaging. The plan layers tests to close those
two remaining gaps.

## 1. Goals / non-goals

Goals
- Release gate: a green e2e run proves the app ingests a video and produces point
  segments end to end, with every user-facing feature exercised.
- Cover each feature: upload, progress, cancel, download CSV, download video,
  config defaults, bad-input rejection.
- Cover the packaging path: the frozen bundle boots and runs (CI).
- Be the gate for the `/goal` release loop.

Non-goals
- Model accuracy / segment quality — that is the offline 6-bin eval. Synthetic
  clips contain no real points, so we assert structure, not segment values.
- Driving Qt widgets directly.
- A full cross-browser matrix (one engine is enough for a local desktop webview).

## 2. Layers (and what each catches)

| layer | tool | drives | catches | blind to |
| --- | --- | --- | --- | --- |
| **L1 backend journey** (PRIMARY gate) | pytest + httpx/requests, real server on a real port | the 6 REST endpoints | pipeline, job lifecycle, API contract, output files | frontend JS, packaging |
| **L2 browser UI** | Playwright (headless Chromium) vs the running backend | the real frontend the webview renders | welcome screen, file picker, progress bar, download/cancel buttons, JS wiring | packaging, Qt shell |
| **L3 frozen bundle** (CI extends `release.yml`) | the PyInstaller binary itself | `--cli` + GUI boot probe | missing hidden imports, asset resolution, Qt WebEngine helpers per-OS | nothing else runs the real artifact |
| **L4 Qt shell** (optional, low ROI) | boot `desktop.py` offscreen | the native shell + webview load | window/webview init | — mostly covered by L3 boot probe |

Recommendation: **L1 is the must-have gate**; L2 is a thin happy-path pass on top;
L3 is mostly already in `release.yml` and just needs extending. L4 is optional.

## 3. Test inventory

L1 — backend journey (pytest, marked `e2e`+`slow`)
- `happy_path_csv` — upload synthetic clip → poll progress to `done` → GET CSV;
  assert exit/state ok, CSV header `start_time,end_time`, rows sorted & within
  clip duration, segment count ≥ 0.
- `happy_path_video` — same job → GET segmented MP4; assert nonzero bytes, valid
  container (PyAV opens it), duration ≤ source.
- `progress_monotonic` — percent never decreases, terminal state reached, no stuck
  job (bounded poll timeout, no fixed sleeps).
- `cancel` — start job → cancel → state reflects cancellation, no output written.
- `bad_input_oversize` — upload > 2GB cap rejected (4xx, no job spawned).
- `bad_input_nonvideo` — non-video / empty file rejected cleanly (4xx, no traceback).
- `config_defaults_contract` — `/api/config/defaults` returns `fps=5.0`,
  `feature_set=v1`, `yolo_sizes`, `available_devices`, `auto_device`.
- `path_traversal` — already unit-covered; keep an e2e assertion that a crafted
  job id can't escape `JOBS_DIR`.
- (stretch) `cli_parity` — same clip via the CLI entrypoint yields the same
  segments as the GUI job (regression anchor across the two code paths).

L2 — browser UI (Playwright, marked `e2e`+`slow`, opt-in)
- `ui_happy_path` — load `/` → click *Get started* → set file input to the
  synthetic clip → assert progress bar advances → assert *Download* enabled →
  download triggers; capture screenshot/trace on failure.
- `ui_cancel` — start → click cancel → UI returns to idle/cancelled state.

L3 — frozen bundle (CI, extend `release.yml`)
- Keep: `--cli --help`, frozen asset-resolution, synthetic-clip pipeline → CSV,
  GUI boot probe (health + defaults).
- Add: in the synthetic-clip step also exercise `--write-csv` AND segmented video
  (`--segment-video`) so the video output path is covered in the frozen artifact.

## 4. Fixtures & infrastructure

- **Synthetic clip**: reuse `scripts/make_smoke_clip.py` (30s @ 10fps, mpeg4). One
  shared session fixture builds it once into a tmp dir.
- **YOLO weights**: download once into a cached `models/` (nano); skip the layer if
  ultralytics/weights unavailable (mirror `test_court_detection_e2e.py` gating).
- **Live backend fixture**: start the real server via `start_backend_thread()` on an
  ephemeral port (port 0 → OS-assigned), poll `/api/health` until up, yield base
  URL, tear down thread + temp jobs/output dirs. Reused by all L1 (and L2) tests.
- **Isolation**: per-test temp `JOBS_DIR` / output dirs via env + monkeypatch; no
  writes to the repo. Ephemeral port so parallel/repeat runs don't collide.
- **Determinism**: assert on structure (headers, state machine, exit codes, file
  validity, counts ≥ 0), never on exact segment timestamps.

## 5. Markers, execution, CI wiring

- Markers already exist: `slow`, `e2e`. Default `pytest` stays fast (these are
  deselected by default or via `-m "not e2e"`); run explicitly with `pytest -m e2e`.
- New CI job (separate from `release.yml`): on PR/push, install runtime deps + the
  cached nano weights, run `pytest -m e2e` for L1 (+ L2 if Playwright installed).
  Keep this off the tag-only release build; `release.yml` keeps the L3 frozen run.
- **`/goal` loop gate**: the loop is "done" when `pytest -m e2e` (L1) is green and
  the L3 CI steps pass.

## 6. Build order (proposed)

1. L1 live-backend fixture + `happy_path_csv` — smallest real end-to-end, the gate.
2. L1 `happy_path_video`, `progress_monotonic`, `cancel`.
3. L1 `bad_input_*`, `config_defaults_contract`, `path_traversal`.
4. Wire the new `pytest -m e2e` CI job.
5. L2 Playwright `ui_happy_path` (+ decide whether to add the Playwright dep now or
   defer to post-first-release).
6. L3: extend `release.yml` synthetic-clip step to also produce a segmented video.

Open questions for review
- Add the Playwright dependency for the first release, or ship on L1+L3 and add L2
  right after? (L1+L3 already covers pipeline + packaging; L2 adds frontend-JS
  confidence.)
- Is `cli_parity` worth it for v1, or defer until v0.4.0 introduces the second
  decode path?
- Do we want one "real" short clip with an actual point for a golden-ish assertion,
  or stay fully synthetic (probabilistic, structure-only) for the first release?
