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

## 2026-09-20 — Local video uploads have no fixed byte-size ceiling

- **What:** Removed the browser's 2 GiB selection guard and Flask's matching
  `MAX_CONTENT_LENGTH`; the upload interface now states the 720p input-quality
  floor instead of advertising MP4-only / 2 GB restrictions.
- **Why:** Analysis is streaming and memory-bounded, and the app is a local-only
  service. The old cap was a legacy defensive limit rather than a model,
  decoder, or export constraint.
- **Rejected:** Raising the arbitrary ceiling to 4 or 8 GiB (the same problem at
  a different threshold); retaining a frontend-only warning that disagrees with
  the server. Available local disk space is now the practical upload bound.

## 2026-09-20 — Apple exports prefer VideoToolbox with libx264 fallback

- **What:** macOS exports try `h264_videotoolbox` first, using a resolution- and
  frame-rate-aware bitrate derived from the existing CRF preference. Input decode
  uses FFmpeg auto threading. Any VideoToolbox setup/runtime failure deletes the
  partial output and retries the same intervals with `libx264`.
- **Why:** The M1 Max media encoder was idle. On the user's real 14-cut 1080p60
  match, the prior software path took 159.16s; VideoToolbox plus parallel decode
  produced the same 165.48s, 1920x1080 H.264/AAC timeline in 32.09s (4.96x
  faster). Audio-sync and frame-duration regression tests pass on real hardware.
- **Rejected:** VideoToolbox decode (60s of the actual H.264 source took 19.43s
  versus 3.10s with FFmpeg software threading because frames must return to CPU
  memory); replacing `libx264` outright (hardware sessions/codecs can be absent);
  a fixed bitrate that would underserve 4K or bloat 720p.

## 2026-09-20 — Consecutive folders use a proxy timeline and external originals

- **What:** A macOS folder picker imports directly referenced, naturally sorted
  MP4 chunks as one continuous match. RallyClip creates one 1280x720-or-smaller,
  30 fps H.264/AAC proxy for analysis and review, saves exact per-file offsets in
  `sources.json`, and maps edited global point intervals back onto the original
  files for a single VideoToolbox export.
- **Why:** Multi-hour 4K matches should not be uploaded/copied one chunk at a time
  or analysed at 4K/60 when the model consumes 720p at 5 fps. A continuous proxy
  also allows model windows and points to cross camera-file boundaries while
  preserving the originals' resolution and audio in the final cut.
- **Rejected:** Concatenating or copying all 4K sources into the library
  (duplicates very large footage); analysing each chunk independently (loses
  boundary-spanning points and rejects short final chunks); exporting from the
  proxy (quality loss).
- **Tradeoff:** Originals are referenced in place and must remain available until
  export. Matching resolution/frame rate is required across consecutive chunks
  so one output stream can be encoded without scaling or cadence changes.

## 2026-09-20 — Consecutive chunks may report different frame rates

- **What:** Folder preflight requires matching dimensions but no longer requires
  equal average-frame-rate metadata. Multi-source export maps each decoded frame's
  source timestamp onto the output time base, dropping duplicate output ticks or
  leaving presentation gaps as needed while keeping every cut's real duration.
- **Why:** Cameras can report 59.94 versus 60.00 (or slightly different calculated
  averages) across otherwise compatible consecutive MP4 chunks. Rejecting these
  prevented valid matches; emitting every frame sequentially at the first clip's
  rate would instead introduce speed and audio-sync errors.
- **Rejected:** A wider arbitrary FPS tolerance (still rejects legitimate changes
  and leaves a cliff); ignoring the mismatch while using sequential output PTS
  (changes playback speed). Resolution changes remain rejected until export has
  an explicit output-resolution policy rather than silently scaling originals.

## 2026-09-20 — macOS proxies use AVFoundation hardware transcode plus packet remux

- **What:** For inputs above 720p on macOS, each source chunk is transcoded with
  `/usr/bin/avconvert` and `PresetAppleM4V720pHD` (960x720/30 for the user's 4:3
  DJI files). PyAV then joins the encoded H.264/AAC packets by rewriting PTS/DTS;
  it does not decode or encode them again. The prior PyAV proxy builder remains
  the fallback on other platforms or if native conversion fails.
- **Why:** The initial implementation used the M1 Max media encoder but software-
  decoded and CPU-scaled every 4K60 frame, measuring about 0.33x realtime on the
  live 102-minute match. Native AVFoundation processed 60 seconds of the exact
  source in 9.06s (6.6x realtime, 1.73s user CPU), and packet-remux joined two
  proxy minutes in 0.30s. This actually engages Apple's media decode path.
- **Rejected:** PyAV VideoToolbox decode plus CPU frame transfer (previous 1080p
  benchmark was slower than software decode); native transcode followed by a
  second proxy encode (unnecessary quality/time cost); a 400x300 cellular preset
  (too little pose detail). The 720p preset uses more proxy disk space, trading
  storage for much faster first-pass processing and smooth 30fps review.

## 2026-09-20 — Report frame-based progress for lazy exports

- **What:** Export status now includes a percentage calculated from emitted video frames, and the browser displays it on the export button.
- **Why:** Saved matches may contain thousands of seconds of source footage and require a full re-encode; a spinner-only state gave no indication whether the background job was advancing.
- **Rejected:** Estimating from output-file size, because the MP4 is written through an encoder and its size is not monotonic with frame completion.

## 2026-09-20 — Prefer keyframe-aligned stream copy for exports

- **What:** Saved-match exports first remux packets between nearby keyframes, padding point intervals by up to one second; the existing frame-accurate encoder remains the fallback.
- **Why:** Most exports can avoid decoding and re-encoding when the source contains usable keyframes, while the padding preserves the requested point context.
- **Rejected:** Always stream-copying, because files without a keyframe after an interval end cannot be clipped safely that way and must retain the exact encoder path.

## 2026-09-26 — Refuse unsafe stream-copy and fail mixed folder audio early

- **What:** Keyframe-padded export ranges that overlap or touch, and copied
  keyframe spans that run into the next cut, raise and fall back to the
  frame-accurate encoder. Folder preflight rejects a mix of audio and silent
  chunks before proxy generation. Pending folder selections keep only the
  latest token, and `/api/folder/start` drops tokens older than 30 minutes.
- **Why:** The remux fast path appended each padded window independently, so
  points less than two seconds apart (or a long GOP that overshoots the pad)
  repeated footage. Mixed-audio folders passed analysis and then failed every
  export. `folder_selections` stored every unused pick until process exit.
- **Rejected:** Merging overlapping pads into one remux (would keep the
  between-point gap inside the export); normalizing missing audio with silence
  during export (hides a bad folder after the expensive analysis); a TTL-only
  map with no cap (a burst of folder picks still grows without bound).

## 2026-09-26 — Stream copy stays inside the cut; folder and proxy guards

- **What:** A video-only remux that would include packets outside the selected
  interval falls back to frame-accurate encode. `/api/folder/select` uses only
  the native picker. Cancel signals the worker's process group. Native proxy
  chunks keep the original source duration instead of the packet-derived span.
  Proxy silence is emitted in frames no larger than one AAC frame. The upload
  Start button stays enabled when a folder is still selected.
- **Why:** Keyframe padding shipped extra footage. A JSON `folder_path` let any
  local caller list MP4 names. `terminate()` on the worker left `avconvert`
  running. Proxy packet durations drifted later-file export times. One silence
  allocation could be the size of a missing audio track. Folder preflight
  errors called `resetControls()`, which only re-enabled Start for a single file.
- **Rejected:** Deleting the remux path entirely (it is still valid when the
  keyframes already match the cut); accepting `folder_path` behind a header
  (the UI never sends a path).

## 2026-09-26 — Proxy join uses presentation time; folder picks stay bounded

- **What:** Native proxy remux measures each part from its earliest presentation
  timestamp and drops only packets that start after the original duration.
  Pending folder selections expire after 30 minutes. At most eight unused
  selections are kept; a further selection is refused instead of evicting
  another window's token. Cancel deletes leftover `.analysis-proxy-*.m4v`
  parts. Stream copy allows at most 1 ms outside the selected interval.
- **Why:** Using decode timestamps as the origin discarded the end of B-frame
  parts. Replacing every pending token broke a second window. Killing the worker
  skipped the converter's `finally` and left part files behind.
- **Rejected:** Keeping a single global folder token (breaks two windows);
  capping proxy packets by decode time (drops displayed frames).

## 2026-09-26 — Folder dismiss frees a slot; proxies stay on PyAV

- **What:** Removing a chosen folder, replacing it with a file, or choosing a
  different folder posts `/api/folder/dismiss` and drops that token. Analysis
  proxies are created only with PyAV (VideoToolbox, then libx264).
- **Why:** A cap of eight pending selections blocked a new import for 30 minutes
  once the UI forgot the tokens. `/usr/bin/avconvert` decoded and encoded outside
  PyAV, which is the runtime media rule.
- **Rejected:** Evicting the oldest token to make room (invalidates another
  window); keeping avconvert behind a macOS check (still leaves the runtime
  decode path).

## 2026-09-26 — Stream copy keeps the pad; proxy audio stays in place

- **What:** A video-only remux copies packets inside the keyframe pad and stops
  before the closing keyframe. Proxy audio that starts after the picture is
  preceded by silence. Cancel deletes `analysis_proxy.mp4`. A second export
  start does not spawn another thread once the item is already processing.
- **Why:** Comparing the remux to the exact cut rejected the pad, so every
  stream copy fell back to a re-encode. Late source audio was written at the
  start of the proxy chunk. Killing the worker left the partial proxy. Two
  export requests could both observe "missing" and each start work.
- **Rejected:** Turning stream copy off; padding the audio gap at the end of
  the chunk; letting the extra export thread wait on the encode lock.
