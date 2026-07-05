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
