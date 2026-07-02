# Runtime Config Refactor + Backwards Compat — Plan

**Status:** Historical planning note; superseded for active architecture work by
`docs/runtime-api-engine-refactor.md`.
**Branch:** Originally planned for `fix/runtime-pipeline-drift`; current active
runtime split is on `refactor/runtime-api-engine`.
**Owner:** ismael
**Last updated:** 2026-07-01

> 2026-07-01 note: this document remains useful for the older manifest/config
> contract analysis, but it is not the current handoff for the runtime/API/engine
> branch. The active branch now adds `rallyclip_core`, `rallyclip_engine`, and
> `rallyclip_api`, with model-object pipelines and shared playback contracts.
> Continue from `docs/runtime-api-engine-refactor.md` for current work.

---

## 1. Problem

The inference pipeline does not have a single source of truth for its parameters.
The same concepts have different values in different places, and the divergent values
are the **old v0.1 model's contract** masquerading as defaults.

Three sources of truth today:
- `models/rallyclip_v0.3.1/manifest.json` — correct v0.3.1 contract
- `config.toml` — correct v0.3.1 values
- Hardcoded literals in `RunConfig` + `build_run_config` — the **stale v0.1 contract**
  (`seq_len=300`, `overlap=150`, `sigma=1.5`, `high=0.8`, `min_dur_sec=0.5`, `fps=15`)

They agree only on the bundled ONNX happy path. The moment the manifest is missing or
`RunConfig` is constructed directly, the phantom v0.1 contract silently takes over.

### Why this is dangerous
Inference is silent. Extract at 15fps instead of 5, build 360 features instead of 362,
or window at 300 instead of 100, and the model emits wrong segments with no error.

### Codex audit findings (2026-06-03, verified)
Confirmed reachable issues:
- **P1** `RunConfig` dataclass defaults + `build_run_config` fallback literals encode v0.1
  ([src/cli/main.py:54-60](../src/cli/main.py#L54), [:240-246](../src/cli/main.py#L240)).
  Reachable via direct `RunConfig(...)` construction, or `--video` + a manifest-less model.
- **P1** `screen_width=1280`/`screen_height=720` hardcoded, affect feature values, not in any
  manifest/config ([src/features/feature_engineer.py:9](../src/features/feature_engineer.py#L9),
  [src/preprocessing/data_preprocessor.py:9](../src/preprocessing/data_preprocessor.py#L9)).
- **P2 behavioral bug** `write_csv` falls back to `False` when `--video` skips config.toml,
  even though config says `true` ([src/cli/main.py:193](../src/cli/main.py#L193)). `--video x.mp4`
  silently writes no CSV. Blocks the validation harness.
- **P2** pose extractor `imgsz=1920`/`target_fps=15` standalone defaults; court detector
  `conf=0.5` + hardcoded `models/yolov8s-pose.pt`; legacy `lstm_300`/`seq_len300` asset fallbacks.
- **Root cause** `--video` skips config.toml ([src/cli/main.py:169](../src/cli/main.py#L169)),
  which makes every fallback literal reachable in a normal run.
- **Downgraded** `input_size=360` ([src/infer/model.py:6](../src/infer/model.py#L6)) is
  self-correcting (overridden by checkpoint weight shape) and ONNX never hits it.

`conf` is a latent contract gap: manifest has `conf: null`, runtime uses `0.25`. Pose conf
filters detections that become features, so it belongs in the contract but was never pinned
at export. **Needs verification: what conf did training run at?**

---

## 2. Goals / Principles

1. **One guiding config for inference.** Every pipeline parameter resolves from manifest or
   config. No phantom literal defaults.
2. **Crash on missing, don't guess.** If a required contract/postprocess value is not present
   in manifest or config, fail loudly with a clear message. (Exempt: pure runtime/IO prefs.)
3. **Immutable vs mutable separation.** The model's identity is immutable and lives in the
   manifest. Tuning knobs are mutable and live in config.
4. **Models are self-describing.** Each model ships its own manifest; swapping models = swapping
   manifests. Backwards compat rides the same rails, no special-case legacy code.

---

## 3. Field Taxonomy

### Bucket 1 — Immutable (model identity). Source: **manifest only.** Crash if absent.
Changing these changes the inputs the model learned a function of → degraded/garbage output.

| Field | Notes |
|---|---|
| `feature_dim`, `feature_set` | Input tensor width. Already enforced via `FeatureSetV1`. |
| `seq_len` | ONNX input shape `[1, seq_len, feature_dim]`. Model rejects wrong value. |
| `target_fps` | Velocity/accel features scale by `dt`. Temporal dynamics assume this rate. |
| `imgsz` | YOLO resolution → keypoint coords → feature values. |
| `screen_width`, `screen_height` | Coordinate normalization. **Currently hardcoded, must move into manifest.** |
| `num_keypoints` (17) | Structural. |
| `conf` (pose) | Filters detections that become features. **Currently unpinned (manifest null) — verify.** |

CLI/config override of immutable fields: **allowed but warns loudly** ("overriding a
model-contract value, expect degraded perf"). Decision pending — see §7.

### Bucket 2 — Mutable (postprocessing). Source: **config override → manifest.postprocess → crash.**
Operate on model *output* probabilities. Tune freely; the model is untouched.

| Field | Tunes |
|---|---|
| `low`, `high` | Hysteresis sensitivity (precision/recall). |
| `sigma` | Probability-curve smoothing. |
| `min_dur_sec` | Minimum segment duration. |
| `threshold` | Base decision threshold. |
| `overlap` | Window stride/averaging (throughput vs smoothness). **Mutable** — model never sees it. |

### Bucket 3 — Runtime / IO. Source: **config → sane default OK.** Not a model lever.
`output_dir`, `output_name`, `csv_output_dir`, `write_csv`, `segment_video`, `yolo_device`,
`start_time`, `duration`. Defaults are fine here (crashing because `output_dir` is unset is just annoying).

---

## 4. Architecture

- **`manifest.json` = the model's birth certificate.** Immutable contract + training provenance
  (sha256, git commit, epoch) + recommended postprocess params. Never hand-edited. Missing → crash.
- **`config.toml` = the operator's control panel.** Mutable postprocess knobs + runtime IO.
  Ships with the discovered best-outcome defaults. Edit here to tune; never touch the model card.
- **`FeatureRegistry`** ([src/training/features/registry.py](../src/training/features/registry.py))
  routes `manifest.feature_set` → builder class. Currently registers only `"v1"`.

UX:
- Different **tuning** → edit `config.toml`.
- Different **model** → bring your own `model.onnx` + `manifest.json` (regenerated by training export).

---

## 5. Phase 1 — Config Refactor (DO FIRST)

Make the manifest the single source for immutable fields, config for mutable, no literals.

**Tasks:**
1. Remove stale literal defaults from `RunConfig` dataclass ([src/cli/main.py:54-60](../src/cli/main.py#L54)).
   Contract fields become required / sentinel, not v0.1 numbers.
2. Rewrite `build_run_config` resolution ([src/cli/main.py:229-251](../src/cli/main.py#L229)):
   - Immutable: manifest (or explicit override w/ warning). Missing → `SystemExit` with guidance.
   - Mutable: config → `manifest.postprocess` → crash. No literals.
   - IO: config → sane default.
3. Extend `_manifest_defaults_for_model` to also read `feature_set`, `screen_width`,
   `screen_height`, `conf`, `threshold` (add these to the manifest schema first).
4. Add `screen_width`/`screen_height` to the manifest and plumb them through
   `FeatureEngineer` / `DataPreprocessor` (remove the hardcoded 1280/720).
5. Fix the `write_csv` behavioral bug — `--video` must not silently drop config's `write_csv`.
   Likely: load config.toml even with `--video` for IO/mutable fields, OR make the IO defaults
   match config. (Revisit the "`--video` skips config.toml" decision — it's the root cause.)
6. Drop legacy asset fallbacks (`lstm_300_v0.1.pth`, `seq_len300`, `scaler_300`) from
   `_resolve_asset` relatives — legacy becomes a normal manifest-driven artifact (Phase 3).
7. Tests: manifest-required crash path, mutable override path, no-literal assertion,
   `write_csv` honored with `--video`, screen_w/h sourced from manifest.

**Out of scope for Phase 1:** GUI (deprecated + broken static path; see §8), `FeatureSetV0`.

---

## 6. Phase 2 — Validation Harness + v0.3.1 Baseline

Confirm the shipping model's perf isn't degraded, and build the tool Phase 3 needs.

**Data (in training repo `/Users/ismaelrobles-razzaq/cs_projects/RallyClip`):**
- Raw videos: `data/raw_videos/` (11 val-set base files)
- Ground truth: `data/annotations/*.json` (one per video)

**Tasks:**
1. Glue script: run CLI on a raw video → segment CSV → compare to `.json` GT →
   feed through [src/training/metrics/segment.py](../src/training/metrics/segment.py).
2. Report the 4 categories: false positive (`false_detected`), false negative (missed GT),
   good (sufficient `gt_coverage`), detected-but-insufficient-overlap.
3. Run across the val set, compare to manifest `metrics` (segment_f1 0.67, etc.).

This harness is reused as the v0 correctness check (Phase 3) and the Docker acceptance test.

---

## 7. v0 Backwards Compat (OPTIONAL — not committed, fully spec'd)

**Priority: low / nice-to-have.** Not required for shipping. The container only bundles
v0.3.1; the legacy model lives in the training repo. Phase 1's design *reserves the slot*
(registry + `feature_set` routing), so this can be picked up later with zero rework if ever
wanted. The spec below is preserved so the archaeology isn't lost — but don't treat it as
planned work.

If ever done: make the legacy v0.1 model runnable through the same manifest-driven rails.
Gated on the Phase 2 harness (can't ship a resurrected builder without proving byte-correctness
on a labeled video).

### Archaeology (done 2026-06-03, training repo git history)
- **Legacy is already an ONNX artifact:** `models/rallyclip_v0.1.0_legacy/` has
  `model.onnx` + `scaler.json` + `manifest.json`. Use the ONNX; retire the `.pth`.
- **Legacy contract** (from its manifest): `feature_dim=360`, `target_fps=15`, `conf=0.25`,
  `imgsz=null`, `input_shape=[1,300,360]`, `seq_len=300`, `overlap=150`,
  postprocess `high=0.8 low=0.45 min_dur=0.5 sigma=1.5`.
- **360→362 cutover commit: `6bc24e6` "finalize artifact cli runtime" (2026-03-30)** —
  appended `+ 1` / `box_conf` to the per-player formula.
- **Canonical v0 source: `12e5c9d:src/features/feature_engineer.py`** — last 360 version,
  byte-identical math to initial commit `600b388` (only diff is print→logging).
  `features_per_player = 1+4+2+2+2+1+1 + 17*3 + 17*2 + 17*2 + 17 + 17 + 14 = 180`, ×2 = **360**.

### Two landmines
1. **`feature_set` name is overloaded.** The legacy manifest says `feature_set: "v1"` but is
   360-dim; current `"v1"` is 362. **Do not route on the name alone.** Fix the legacy manifest
   to `feature_set: "v0"`, and/or also validate against `feature_dim`.
2. **Motion scaling differs (train/serve skew trap).** The v0 builder uses `dt = 1.0` (raw
   per-frame deltas). v1 uses `dt = 1.0 / target_fps` = 0.2 at 5fps. Running v0 through v1's
   scaling makes every velocity/accel **5× too large** → garbage. `FeatureSetV0` must bake in
   `dt=1.0` and ignore `target_fps`. **Port verbatim; do not derive v0 from v1.**
   - (Layout note: v1 = v0 with box_conf appended per player block, so v0 = v1 minus indices
     180 and 361 — not a simple tail-trim. Another reason to port the original code.)

### Tasks
1. Port `12e5c9d`'s `create_feature_vector` + `_calculate_velocity/_acceleration/_keypoint_velocity`
   (dt=1.0) + centroid/limb/normalization helpers into a `FeatureSetV0` class. `feature_vector_size=288`
   default is dead code — ignore it.
2. Register `"v0"` in `FeatureRegistry`.
3. Fix `models/rallyclip_v0.1.0_legacy/manifest.json` `feature_set` → `"v0"`.
4. Validate the legacy ONNX end-to-end on a labeled video using the Phase 2 harness.

---

## 8. Out of Scope

- **GUI** ([src/gui/app.py](../src/gui/app.py)) — deprecated (noted in README) and currently
  broken: `_find_static_dir()` resolves to a nonexistent `apps/gui/frontend` (actual bundle is
  `gui/frontend`), so it launches but serves no UI. Also builds config independently of
  `build_run_config` and never reads config.toml/manifest. Leave alone; revisit if revived.
- **Docker** — separate effort, comes after validation. The Phase 2 harness becomes its
  acceptance test (run same video in container, compare the 4 metrics).

---

## 9. Open Questions

1. **`conf` — RESOLVED, no skew.** Training built features at `conf=0.25` (on-disk pose data is
   namespaced `conf=0p25`; training scripts default 0.25), runtime uses 0.25. They match. The only
   gap is the manifest records `conf: null` — pin `conf: 0.25` in the manifest as immutable. Low priority.
2. **`yolo_model` — CONFIRMED SKEW: trained on NANO, CLI defaults to small (high priority).**
   Model was trained on **nano** (confirmed by user + on-disk evidence), but config.toml ships
   `yolo_model = "small"` ([../config.toml#L15](../config.toml#L15)) and manifest `yolo_model: null`.
   Evidence:
   - Run `prod_yolon960_fps5_seq20` = fps 5 / imgsz 960. **nano** has `imgsz=960/fps=5.0` with
     **22 files** (= manifest `video_count: 22`); **small** has *zero* fps=5 features (only fps=15.0).
   - Run name literally `yolo**n**960`.
   The shipping CLI default (`small`) runs a different pose model than training → **live pose
   train/serve skew** (small vs nano produce different keypoints, feeding the features).
   **Action:** set config.toml default `yolo_model = "nano"` and pin manifest
   `yolo_model: "yolov8n-pose"`. (Note: run `config.json` does not record the yolo model — that's
   why it was ambiguous; pinning it in the manifest fixes that permanently.)
3. **`--video` + config.toml — RESOLVED in principle.** The coarse "skip all of config.toml when
   `--video`" hack ([src/cli/main.py:169](../src/cli/main.py#L169)) becomes unnecessary once Phase 1
   does bucket-based resolution: manifest wins for immutable (so stale config can't override the
   contract), config always provides mutable + IO. **Delete the skip**; this also fixes the `write_csv`
   bug for free.
4. **Immutable override policy** — warn-and-allow vs hard-forbid CLI/config override of Bucket 1
   fields. Leaning warn-and-allow (you're the only one who'd experiment).

---

## 10. Sequencing Summary

1. Phase 1: config refactor (this is "codex fixes" + single-source design).
2. Phase 2: validation harness + v0.3.1 baseline.
3. (Separate) Docker, with Phase 2 harness as acceptance test.

Optional / not committed: v0 backwards compat (§7) — pick up only if the legacy model
ever needs to run through this CLI.
