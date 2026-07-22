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

## 2026-07-19 — Training pose switches to YOLO26n ONNX; desktop stays v8 until retrain

- **What:** New training extracts will use **YOLO26n-pose ONNX** under a versioned
  contract (`models/pose/yolo26n/` + shared ORT runner with e2e `[1,300,57]`
  decode). Desktop v0.3.1 remains on **YOLOv8n** ONNX@960 (+ CoreML static
  sibling). Export/parity lab stays in sibling `YOLO-ONNX/`; bundled contract
  and training wire live in this repo. Re-export YOLO26 at training imgsz
  (960 rect / static 544×960) — do not use the experimental 640×640 square
  export for corpus work. Fingerprint pose HDF5 with model sha so v8/v26
  caches never mix.
- **Why:** YOLO26n is cheaper FLOPs for labeling/extract even if accuracy is
  similar; training and desktop were already misaligned (train Ultralytics
  `.pt` @ large imgsz vs desktop ONNX@960). A named ONNX contract is the path
  to one extract path for train (and later desktop after retrain).
- **Rejected:** Keep training on Ultralytics `.pt` forever (diverges from
  runtime, heavier); switch desktop to YOLO26 before LSTM retrain (feature
  distribution shift → silent quality drop); MLX rewrite (already rejected
  2026-07-05 — CoreML EP is the Apple path).

## 2026-07-19 — Normalize-to-720p@5fps before training YOLO (robust pipeline)

- **What:** Training pipeline gains an explicit normalize stage (1280×720 @ 5
  fps) before court/pose extract; `ignore_before_s` from new_data labels becomes
  target `-1` (dropped from loss); holdout frozen in `configs/train/holdout.yaml`;
  `train.head: classic | e2e_seg`. Implementation is in this worktree
  (uncommitted as of this entry); details in `../CHAT_HANDOFF.md` and
  `../TODO.md`.
- **Why:** Court detector is pixel-absolute (tuned at 720p); 5 fps is known
  sufficient; new_data warm-up must not be treated as negative; fixed ~5 h test set
  preferred over 5-fold for sweep cost.
- **Rejected:** Making court detector resolution-relative before retrain
  (larger change, riskier for first corpus); treating warm-up as negative
  class; leaving extract on full-fps YOLO then subsample at dataset build.

## 2026-07-19 (session 2)

- **YOLO26 training extraction runs CPU-only ORT for now.** CoreML deferred: the
  e2e export carries NMS in-graph, which risks bad CoreML EP partitioning; not
  worth engineering before the corpus is extracted once. Revisit if extract wall
  time hurts.
- **Pose models are manifest-defined backends** (models/pose/<name>/manifest.json:
  head family, input/letterbox contract, COCO-17 keypoint order, model sha256).
  Dataset identity tag = name@sha8; execution provider is provenance, not identity.
- **Unified corpus data root = container-level ../training_data/**, decoupled from
  both repos. Old RallyClip/data stays as-is for the v0.3.1 lineage.
- **c31e6888 excluded from corpus** (video present, zero export JSONs).
- **Flip augmentation deferred** — regenerate from normalized 720p@5fps videos
  later instead of carrying old flip files forward.

## 2026-07-19 (session 2, later) — YOLO26 → YOLOv8 reversal

- **Pose model = YOLOv8n everywhere (training extract, desktop, iOS). YOLO26 ditched.**
  Rationale: deployment target is Apple/CoreML, so ANE speed beats FLOPs. Measured
  (M2 Pro): v8-static-CoreML 90 fps + sub-pixel parity vs CPU (≤0.9 px); v26-CoreML
  41 fps with 30–180 px decode errors (e2e head fp16 candidate-selection flips —
  inherent to the head, would recur in native iOS CoreML); v26-CPU ~10–12 fps.
  Static-vs-dynamic v26 exports were bit-identical on CPU, isolating the EP as the
  cause. Accuracy delta v8n↔v26n judged marginal for 2-players-at-720p.
- **Corpus extraction runs provider=coreml** (static 544×960 sibling): ~1.5 h vs
  ~10–27 h CPU, and features come from the same runtime the app executes
  (train/serve consistency). EP stays provenance, not identity; parity gate in
  tests/test_pose_backend.py. CoreML is hard-refused for yolo26-e2e head family.
- YOLO26 bundle deleted from models/pose/; exports parked in ../YOLO-ONNX/exports/;
  e2e decoder + dispatch kept in the runner (cheap, tested, documents the contract).

## 2026-07-19 (session 2, data layout)

- **Provenance-nested data layout in ../training_data/**: `sources/` is the only
  physical home for video bytes (new_data session bundles moved whole; legacy_youtube
  for the 11 old originals + annotation provenance copies). Pipeline consumes
  flat interface layers only: `source_videos/` symlinks + flat `annotations/`
  JSONs (golden — edit sources and re-materialize, never the flat dir).
- **Normalize contract is a path tag**: `videos/norm=1280x720@5fps/` and
  `pose_data/norm=…/…` (paths.py norm_tag(); default threaded through
  raw_videos_dir/pose_*_dir so callers are unchanged). Future 1080p = sibling
  tree + court-detector retune, not an overwrite.
- **RallyClip/data deleted; RallyClip/raw_video dups deleted** (md5-verified vs
  canonical first; old annotations+runs archived in
  sources/legacy_youtube/legacy_rallyclip_data_archive.tar.gz). Old-model
  benchmarking uses the bundled models/rallyclip_v0.3.1 artifact.

## 2026-07-20 (session 3 — court fix, loader fix, court-vs-no-court results)

- **Court detection was silently failing corpus-wide** and nobody noticed: base.yaml
  points `yolo.model` at a manifest.json, `court.model_path: null` falls back to it,
  and `_load_yolo` handed the manifest to ultralytics `YOLO()` → "not a supported
  model format", caught → default/no mask. All 29→32 videos had been preprocessed
  with NO court filtering. **Fix:** `_load_yolo` now resolves a manifest/dir via
  `load_pose_backend(provider="cpu")` to the sha-verified ONNX on the CPU EP (court
  touches only a few frames; dynamic-CPU is faithful and avoids the static-letterbox
  concern). After fix: 30/32 detect; `abe49d12`, `e9a093a8` (both TRAIN videos) fail
  all anchors → preprocessed without a mask, flagged loudly.
- **Clean-frame extraction is now MIDDLE-anchored, outward-expanding** (user request).
  Anchor at video midpoint (not 60 s), mask occluding players, walk outward in ~10 s
  hops each way repainting still-occluded px from homography-aligned neighbours that
  have nobody over the same spot; stop a side when quad-IoU(neighbour→base) < 0.6
  (camera cut / pan-zoom) or video ends; stop entirely once no occluded px remain.
  `court.target_time: null` = midpoint (threaded Optional through pipeline →
  PreprocessConfig → fingerprint as "middle"); `CourtMaskCache` retries anchors
  0.5/0.4/0.6/0.3/0.7 of duration; preprocess prints a court-health summary.
  Inference path (`data_preprocessor`) made middle-first too.
- **HDF5 dataset loader now loads to RAM once** (`Hdf5SequenceDataset`, 6 GB guard).
  The dataset build writes features gzip-compressed with chunks (228,7,23) spanning
  228 sequences, so per-item DataLoader reads decompressed hugely-overlapping data:
  measured **~470 s/epoch** just in loading (800 random reads = 51.8 s) vs **2.35 s**
  to bulk-load the whole 1 GB array. In-memory index → epochs 8 min → ~30 s, bit-
  identical (verified). Root cause is the chunking; the in-mem load is the pragmatic
  fix (rechunking the build to (1,100,362) would fix the lazy path too — deferred).
- **Court filtering helps new_data, regresses legacy** (six-bin acceptable, each val-swept):
  classic no-court 31.1% new_data / 49.0% legacy → classic+court **37.1% new_data** / 31.3% legacy.
  Legacy loss is FN-driven. CONFOUNDED (court filtering AND +3 videos/retrain changed
  together); user notes the masks were legacy-tuned so the drop is likely training-mix
  + the middle-out frame shift, not mask over-aggression (Aditi masks more of frame
  than 9/5/15 yet scores fine). **Left unresolved per user** — new_data is the target domain,
  so this is a net win for the goal. Not committed to fixing the masks.
- **e2e head underperforms classic and overfits early; longer patience doesn't help.**
  e2e+court patience-5 (best ep2) 22.9% new_data / 34.0% legacy; patience-10 (best ep9,
  user-requested to avoid "cutting off too soon") 19.3% / 10.2% — WORSE. Val loss
  bottoms ~ep5 then climbs; the two runs differ mostly by MPS nondeterminism
  (best-val 13.7% vs 9.1%), which swamps the patience effect. Conclusion: the
  boundary-REGRESSION formulation is the ceiling (bad_seg still ~50%), not epoch
  count → motivates the Gaussian startness/endness heatmap head (TODO §3c).
- **Laptop sleep suspends training** (not kills): a run "died" 3× because macOS slept
  and suspended the python process; on wake, restarts stacked into 4 concurrent
  train.py clobbering the same run dir. Fix: wrap training in `caffeinate -ims`.

## 2026-07-20 (session 4) — Boundary-heatmap head (e2e_heatmap) beats classic+court on both domains

- **What:** added a third training head `train.head: e2e_heatmap` — twin per-frame
  startness/endness Gaussian heatmaps (BSN family) alongside the retained pointness
  head. New files only (no edits to seg_lstm/seg_loss/seg_loop/seg_evaluator):
  `models/heatmap_lstm.py` (TennisPointHeatmapLSTM, same BiLSTM backbone, 3 logits),
  `train/heatmap_loss.py` (E2EHeatmapLoss; soft Gaussian target from binary labels
  via `boundary_markers`+`_dist_to_nearest`, nearest-boundary == union-of-Gaussians),
  `eval/heatmap_evaluator.py`, `train/heatmap_loop.py`. Additive branches in
  pipeline.py `_run_train`/`_run_eval` + a knobs block in base.yaml. Reused
  `gt_segments_from_targets`, `compute_six_bin`, `_default_pos_weight`, the CPU-eval
  and verified-checkpoint patterns.
- **Result (val-swept, same 3 new_data + 3 legacy subsets as prior benches):**
  heatmap+court = **42.2% new_data / 46.3% legacy** — new best on new_data (vs classic+court
  37.1%) AND recovers most of the legacy regression (vs 31.3%). Only model strong on
  both domains at once. FN low (new_data 1.4%, legacy 13.9%) — recall preserved.
- **Why these choices (from a critical review of the first plan):**
  - **Hybrid decode, not pure peak-pair.** Pointness runs define segments (one per
    detected point → recall can't be lost to a missed boundary peak); start/end
    heatmaps only *refine* each edge via soft-argmax. The sweep picked hybrid over
    peakpair, confirming it. Pure peak-pick+greedy-pair was too fragile (a point
    needs both peaks AND a valid pairing — three conjunctive failure points, exactly
    the FN problem that sank e2e_seg). peakpair kept selectable for comparison.
  - **Balanced-BCE on the soft Gaussian as default**, CenterNet penalty-reduced
    focal as opt-in (`heatmap_loss: bce|focal`). BCE is simpler/harder to misconfig;
    focal has more knobs. Both implemented + tested.
  - Peak-NMS + final overlap-merge in the decoder (the first plan omitted both →
    would emit duplicate/overlapping phantom segments from jittered adjacent peaks).
- **Rejected / deferred:** soft-argmax *time* loss term (model is trained on heatmap
  SHAPE only, never directly on boundary-time error — obvious next lever for the
  still-~29% bad_seg); σ / focal hyperparameter sweep; production src/infer/ path
  (e2e_seg lacks it too). serve-convention label fix (TODO §3b-ii) still poisons
  boundary supervision for THIS head too — worth doing before trusting exact numbers.
- **Caveat:** single MPS-nondeterministic run, best epoch = 2 (overfits early like
  e2e_seg; patience-10 again burned 10 idle epochs). Ordering is clear + large but
  wants a couple of seeds before the exact numbers are final. Run:
  `runs/20260720_court_heatmap`; bench: `benchmarks/bench_court_heatmap.py`.

## 2026-07-21 (session 5) — Optimization-side levers all neutral-or-worse; +4 videos (corpus 37)

Batch of experiments on the leading heatmap head. **Meta-finding: every
optimization-side knob came back neutral or worse — the boundary/quality ceiling is
data/labels/metric, not optimization.** Do not re-run these:

- **Capacity is not a quality lever (3 confirmations).** GRU backbone (−24% params):
  underperforms LSTM (new_data 39.6 vs 42.2, legacy worse), and MPS has no fused GRU kernel
  (10x slower → train GRU on CPU, which is also deterministic). Hidden-size sweep
  (heatmap, matched decode): @128 42.2/48.3, @64 44.4/43.5, @32 41.1/40.1 — new_data flat
  across 85% param reduction, legacy declines gently, bad_seg never improves.
  **@64 is a free deployment win** (best new_data, 64% fewer params) but shrinking doesn't
  fix the early-overfit. Classic hidden-sweep inconclusive (postproc-operating-point
  confounded). GRU/backbone + hidden_size are config-switchable (heatmap_backbone,
  hidden_size); default lstm/128.
- **Soft-argmax time loss: NEGATIVE, shelved (opt-in, default off).** Added
  `heatmap_time_weight` — collapse each boundary's heatmap to a predicted time via
  soft-argmax, penalize squared time error (Gaussian NLL). It works on TRAIN (halved
  train time-error) but HURTS test monotonically: λ=1 new_data 42→36, λ=10 new_data 42→26 (good
  share collapsed 8.7→1.4%). Two reasons: (1) it optimizes the softmax MEAN while the
  hybrid decode reads the PEAK — moving the mean skews the bump and corrupts the peak;
  (2) boundary error is a generalization gap, not a train-fit gap. Kept in code
  (`heatmap_time_weight: 0.0`) for reproducibility; not shipped.
- **LR finder + cosine schedule: built, NEGATIVE for tuning.** New `lr_finder.py`
  (Smith/fastai LR range test: per-batch exp ramp, smoothed TRAIN loss, suggest
  min-loss-LR/10) + cosine scheduler (`lr_schedule: none|cosine`). Finder @128
  suggested 1.45e-3 ≈ current 1e-3 (validated the default). Cosine @128 (peak 1.45e-3):
  new_data 42→37, best ep1 (higher peak overfit faster). @64 finder suggested 6.28e-3 (flat
  min basin) → cosine @64 COLLAPSED (new_data 44→20). **Caveat: min/10 is unreliable when the
  finder curve has a flat minimum — it overshoots; use the steepest-descent elbow.**
  Cosine can't help when the model best-epochs at ep1-3 (LR hasn't decayed yet).
  Infra kept (default lr_schedule=none); not a lever here.
- **+4 new_data videos → corpus 37** (2026-07-21): 46cd→val (23min, shortest), 729→test
  (41min), f5f5+1e9d→train (42/76min, 2 longest). Split by duration (user). Court
  detection FAILED on 729 (angled) + 46cd → 5/37 maskless; 1e9d passed despite angle.
  Dataset `datasets/20260721_v37` (train25/val4/test8). Retrained the two leaders on
  it: heatmap@64 41.1 new_data / 47.6 leg (orig 3+3), classic@128 29.6/21.8; new 729 test
  video heatmap 35.7% vs classic 26.8%. Adding 2 train videos was within-noise neutral
  on the stable subset — data-quality-limited, not quantity-limited. heatmap@64 still
  leads. Runs: `20260721_heatmap64_v37`, `20260721_classic128_v37`.

**Next (not optimization): serve-convention label fix (§3b-ii) + absolute-time metric
(§3b-iii).** All the above dead ends point there.
