# DECISIONS — append-only why-log

Format per entry: date — what / why / rejected alternative. Never rewrite old entries.

## 2026-07-03 — Doc harness created in this worktree (RallyClip-perf)

- **What:** AGENTS.md + features.json + docs/{REPO_MAP,PROGRESS,DECISIONS,ENVIRONMENT,testing}.md
  live at the root of this repo, committed on `refactor/runtime-api-engine`.
- **Why:** RallyClip-perf is where active work happens (latest commits, clean tree);
  the container dir (`rallyclip_container/`) is not a git repo, so the harness must
  live inside a repo to be versioned. Container-level context lives in
  docs/REPO_MAP.md's sibling table (no container-level doc file).
- **Rejected:** harness at container level (unversionable without a new git repo
  wrapping four checkouts — nested-repo mess); harness in `RallyClip/` (parked on
  `docs`, not where work happens; shares `.git` anyway).
- Structural choices made while scanning: replaced the old generic root `AGENTS.md`
  (repo-guidelines boilerplate, partly stale — e.g. claimed yolov8s default) and
  folded root `ENVIRONMENT.md` into `docs/ENVIRONMENT.md` so Tier-1 docs have one
  home. `TODO.md` kept as Tier-4 idea pile; current state lives in PROGRESS.md.
  Lint: no config exists and CI doesn't lint → documented scoped-lint convention
  (23 legacy ruff errors recorded in testing.md) instead of adding a lint config
  (that would be a behavior change beyond a docs harness).
- Test interpreter standardized on `../RallyClip/.venv-train/bin/python3` because it
  was verified in-session (212 passed); conda `tennis_env` kept as an alternative.

## 2026-07-04 — ONNX pose runner is the production path (PR #26)

- **What:** manifest points `feature_pipeline.yolo_model` at a bundled dynamic-axes
  960 ONNX; PoseExtractor and CourtDetector dispatch on the weights extension;
  torch/ultralytics moved to the `[train]` extra.
- **Why:** byte-equal segments on 17/17 sweep samples + golden clip; ~1.45× faster,
  ~40% less RSS; removes torch from install and bundle.
- **Rejected:** YOLO26 end-to-end export (NMS in graph — different output contract,
  raises typed error instead of silent mis-decode); keeping ultralytics as runtime
  fallback (would keep torch in the dependency closure).

## 2026-07-04 — System webview shell replaces QtWebEngine (PR #27)

- **What:** pywebview (WKWebView/WebView2) window over the unchanged Flask backend;
  deleted gui/native_player.py and the QWebChannel bridge.
- **Why:** the native Qt player existed only because Chromium-in-QtWebEngine ships no
  H.264/HEVC; the system webview plays both, and the frontend already had a complete
  HTML5 fallback. Bundle 765→266MB; ~1600 lines deleted; frontend unchanged.
- **Rejected:** pruning unused Qt modules only (~60-100MB, keeps Chromium + native
  player complexity); Tauri/Electron-style rewrite (new stack for no extra benefit).

## 2026-07-05 — CoreML EP + static-shape export wins the Apple-silicon spike (no MLX rewrite)

- **What:** benchmarked the shipped pose ONNX on this M-series Mac via onnxruntime
  execution providers (real frames from a saved match, 40-frame batches). Shipping
  config (dynamic-axes ONNX, CPU EP, rect 544x960): 15.6 fps. CoreML EP on the
  dynamic model: only ~1.2x — the Neural Engine rejects unbounded dims (E5RT
  "unbounded dimension"), so 110/380 nodes stay on CPU with 8+ partition round-trips.
  Re-exporting the same checkpoint with static shapes flips it: static rect
  544x960 + CoreML EP (MLProgram, MLComputeUnits=ALL) = **120.4 fps — ~7.7x the
  15.6 fps shipping path** (8.05x vs the same static model on CPU, 15.0 fps);
  max abs divergence on confident detections 1.2e-4. Static exports were made
  from models/yolov8n-pose.pt, the checkpoint the bundled dynamic ONNX came
  from; production must golden-verify the static export against the bundled
  ONNX before swapping. LSTM head: CoreML is *slower*
  (0.59s vs 0.48s/200 runs) — keep it on CPU. Scripts + JSON results committed in
  docs/perf/coreml-spike/.
- **Why it matters:** pose extraction is the pipeline bottleneck; ~7.7x there without
  new dependencies (CoreMLExecutionProvider ships in stock onnxruntime 1.24.4).
  Productionizing needs: (a) a static 544x960 export added to the model bundle,
  (b) an opt-in provider flag (CPU stays the parity default — 1e-4 divergence
  breaks byte-equal goldens), (c) a fallback for non-16:9 sources (letterbox pads
  to the static shape, as the spike did for square).
- **Rejected:** MLX rewrite (whole new inference stack for less gain than a
  re-export); NeuralNetwork-format CoreML (0.25 abs divergence — actually wrong);
  CPUAndNeuralEngine-only compute units (2x — ANE alone loses to ANE+GPU "ALL");
  accelerating the LSTM (measured slower on CoreML).

## 2026-07-05 — Frozen-app data lives in the OS app-data dir (PR #28)

- **What:** `gui.app._frozen_data_root()` puts packaged-build user data in
  `~/Library/Application Support/RallyClip` (macOS) / `%APPDATA%` (Windows) /
  XDG data home (Linux), with a one-time `shutil.move` migration from the
  v0.1.0 `~/RallyClip` location. Windows is selected via `sys.platform`.
- **Why:** dumping a data dir in `$HOME` violates platform conventions; the
  migration keeps v0.1.0 users' libraries. `sys.platform` (not `os.name`)
  because pathlib picks WindowsPath/PosixPath from `os.name` at instantiation —
  monkeypatching it breaks every `Path()` in tests on Windows.
- **Rejected:** `Path.rename` (EXDEV across filesystems — shutil.move falls
  back to copy+delete); zero-arg lru_cache memoization (leaks state across
  platform-monkeypatching tests for a handful of one-time stats).

## 2026-07-05 — Viewer streams the source file directly (PR #29)

- **What:** `/api/library/<id>/source` serves the saved match with
  `send_file(conditional=True)` (Range/206); the frontend models it as one
  full-length window so the existing source-time scheduler (seeks, point
  skips, timeline) is unchanged. WebM preview windows remain the automatic
  fallback (probe error or 10s timeout; probes are sequence-ticketed so stale
  callbacks are inert).
- **Why:** the stuck-at-first-8s bug: 8s VP8/WebM windows transcode at ~2.5×
  real time (17–22s per window, file-mtime evidence + live WebKit repro), so
  playback stalled at every boundary. The WebM pipeline only ever existed
  because QtWebEngine's Chromium lacked H.264 — the system webview (PR #27)
  decodes it natively.
- **Rejected:** speeding up the transcode (still a transcode; still burns CPU
  and disk); MSE path (permanently dormant, `canUseMsePreview()` false);
  deleting the window pipeline immediately (kept as codec-fallback until
  direct playback is QA-confirmed in the wild).

## 2026-07-05 — Segment edits are a shadow CSV, never the original (PR #31)

- **What:** viewer edit mode writes user point edits to `segments_edited.csv`;
  the model-produced `segments.csv` is never modified. The edited copy wins
  everywhere (`resolve_segments`: playback manifest, /segments, CSV download,
  lazy export — export.mp4 invalidated on edit/reset). Reset deletes the copy;
  it refuses if the original is missing (legacy items). Frontend autosaves are
  serialized and generation-guarded (a stale PUT can't resurrect a reset).
- **Why:** "Reset to original" must always be possible, so the original is a
  read-only contract; a shadow file with precedence is the smallest mechanism
  that gives every consumer the edited times without touching the analysis
  output. All behavior stays `/api/*` HTTP per the architecture invariant.
- **Rejected:** editing segments.csv in place with a backup copy (reversed
  precedence is easier to corrupt — an interrupted write loses the original);
  edits in meta.json (two sources of truth for point times); save-on-Done only
  (drag sessions lose work on crash; autosave matches the iPhone-Photos model).

## 2026-08-24 — Ship the champion TCN as default (v0.5.0); keep classic LSTM as fallback

- **What:** Export `TennisPointHeatmapTCN` (run `20260724_tcn64_cos1e4`) to
  `models/rallyclip_v0.5.0/` with three named logit outputs and
  `pipeline.id=frame_startend_heatmap`. Point CLI/GUI/packaging at that bundle.
  Keep `models/rallyclip_v0.4.0/` in-tree. Bake hybrid decode knobs into the
  ship manifest; dummy hysteresis keys (`sigma`/`low`/`high`/`min_dur_sec`) stay
  in `postprocess.params` so CLI/GUI `_resolve_mutable` / `build_gui_defaults`
  still resolve. GUI jobs resolve pipeline from the artifact unless the client
  explicitly sends `pipeline_id`.
- **Why:** Champion hybrid decode is ~43.9% test acceptable vs classic v0.4.0
  ~30.7% six-bin good. No extra training. Mac/CLI runtime already exists.
- **Rejected:** shipping heatmap LSTM (~31.5% ≈ classic); start/end-only or
  pair-DP as default decode; chasing bit-identical train-wt metrics (min-duration
  is applied at slightly different stages); `rallyclip serve` / Win-Linux freeze
  in the same change.

## 2026-09-07 — Desktop DMG is built/signed/notarized in GitHub Actions

- **What:** `release.yml` on `v*` tags runs tests, `pyinstaller RallyClip.spec`,
  Developer ID codesign (hardened runtime + timestamp), a drag-to-Applications
  DMG, `notarytool` submit/wait, stapler, and a draft GitHub Release asset
  named `RallyClip-<pyproject-version>-macOS-arm64.dmg`. The same scripts run
  locally (`scripts/release/package_macos.sh`). The spec bundles
  `runtime.defaults.DEFAULT_ARTIFACT_DIR` (currently `models/rallyclip_v0.5.0`)
  and sets `CFBundleIdentifier` `com.iroblesrazzaq.rallyclip`.
- **Why:** v0.1–v0.3 DMGs were assembled and notarized by hand; CI only uploaded
  an unsigned `.tar.gz`. Signing belongs in CI so a tag is the release
  artifact, not a leftover file on one Mac. GitHub-hosted `macos-latest`
  runners can codesign/notarize if the Developer ID `.p12` and App Store
  Connect API key are injected as secrets (no self-hosted Mac required).
- **Rejected:** keeping the ad-hoc `pyinstaller --hidden-import ...` CLI in
  `release.yml` (it had drifted from `RallyClip.spec`); `rcodesign` on Linux
  (extra toolchain, still need Apple notary); `apple-actions/import-codesign-certs`
  (another pin; a 40-line import script is enough); notarizing a zip of the
  `.app` then wrapping a DMG (one DMG submission matches the proven v0.1.0
  manual path); failing open on missing secrets for tags (that would re-ship
  unsigned builds). `workflow_dispatch` still wraps an unsigned DMG when
  secrets are absent.

## 2026-09-07 — PR review is Greptile only; Bugbot stays off

- **What:** Agents must not invoke Cursor Bugbot. PR review is Greptile
  (`@greptileai`, iterate to 5/5). Disable Bugbot for this repo in the Cursor
  dashboard so it does not spend tokens on every PR push.
- **Why:** Greptile is already the repo reviewer and is free here; Bugbot is a
  second paid review on the same diffs.
- **Rejected:** leaving Bugbot on "only when mentioned" (still easy to trip by
  commenting `cursor review`); uninstalling the whole Cursor GitHub App (that
  also breaks Cloud Agents / PR comments we still use).

## 2026-09-07 — ONNX off git; artifact zip; DMG still combines at build

- **What:** Plan (`docs/artifact-registry-plan.md`): git tracks manifests +
  `DEFAULT_ARTIFACT_DIR` pointer; weights live on GitHub Release
  `artifact-rallyclip_v0.5.0`. CI fetches the zip before PyInstaller. The
  notarized `.app` still contains `models/rallyclip_v0.5.0/` and does not
  download on launch. From-source clones run `scripts/fetch_artifact.py`.
- **Why:** Even `git clone --depth 1` currently pulls every ONNX on `main`
  (triplicated pose weights + old LSTM trees). Clone should be code; the DMG
  stays a single offline install.
- **Rejected:** Git LFS/DVC (extra quota/tooling); first-launch download inside
  the Mac app; putting weights in Application Support next to the library;
  rewriting git history in the same change.

## 2026-09-08 — Release probe boots the real GUI, Flask first

- **What:** Packaged release CI launches `"$BIN"` (no `--backend-only`). Flask
  starts and `/api/health` can succeed before pywebview/WKWebView is imported.
  After `create_window`, the binary prints `RallyClip desktop shell ready`; the
  probe requires that line plus health. `--backend-only` stays as a debug flag.
- **Why:** v0.5.0 release failed because importing WKWebView on the GitHub
  Mac runner stalled before Flask came up (30s). Switching the probe to
  `--backend-only` would not catch a broken pywebview bundle (Greptile P2;
  agreed).
- **Rejected:** Keep `--backend-only` as the release probe (misses desktop-shell
  regressions); importing webview before starting Flask (original hang).

## 2026-09-08 — Ready marker is `window.events.loaded`

- **What:** `RallyClip desktop shell ready` prints from `window.events.loaded`
  (DOM ready in WKWebView). Release probe also `kill -0`s the process after
  seeing the line. `create_window()` only allocates the Python Window object.
- **Why:** Greptile P1: a marker before `webview.start()` can pass while native
  GUI init still fails; CI then kills the process.
- **Rejected:** Printing after `create_window()` / before `webview.start()`.

## 2026-09-09 — Assisted DMG update in the packaged app (v0.5.1)

- **What:** Frozen Mac app downloads the Latest `v*` arm64 DMG into
  `~/Downloads`, checks SHA-256, and opens it. Localhost GUI opens that
  release page. App version is 0.5.1; artifact remains `rallyclip_v0.5.0`.
- **Why:** v0.5.0 only linked the Releases list. Users still replace the app
  in Applications themselves (running binary cannot overwrite itself).
- **Rejected:** Sparkle / in-app swap of `/Applications/RallyClip.app`;
  treating the ONNX zip as an app update; unifying web vs app library paths.

## 2026-09-09 — Update check skips artifact releases

- **What:** Status/download list GitHub `/releases` and pick the newest
  published `v*` tag. Checksum-verify a staging file before replacing any
  existing DMG in `~/Downloads`. Frozen update button becomes Cancel.
- **Why:** `/releases/latest` can be `artifact-rallyclip_*`; a failed repeat
  download must not delete a good installer; a 300s-per-file transfer must
  stay cancellable.
- **Rejected:** Trusting `/releases/latest` as the app channel; Sparkle;
  byte-progress UI (the server downloads, the browser waits on one POST).

## 2026-09-09 — Cancel stops the server-side DMG download

- **What:** `POST /api/update/cancel` sets an event the download loop checks
  between chunks. Sidecar checksums must name the selected DMG (no last-hash
  fallback). Staging files use a unique suffix so a retry cannot share paths.
- **Why:** Aborting the browser fetch left Flask downloading and `open`ing the
  DMG. An unnamed sidecar hash could verify the wrong bytes.
- **Rejected:** Client-only AbortController as the cancel mechanism.

