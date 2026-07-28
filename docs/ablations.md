# Ablations

Planning doc for the next training round (~30 hrs labeled data, more incoming).

## Evaluation protocol (proposal — see discussion)

- **Metric:** grouped k-fold cross-validation, folds split **by video** (never within-video, to avoid leakage between overlapping windows).
- **k = 5** as the default. With ~24 videos that's ~5 videos / ~6 hrs held out per fold. LOSO is too expensive now that the video count is growing.
- Fold assignment is fixed once (seeded, stratified by video duration so folds are roughly balanced in hours) and reused across all ablations so results are comparable.
- **Decision metric:** the six-bin segment metric (`docs/segment_eval_metrics.md`, `src/training/metrics/segments6.py` on the `training/e2e-segment-model` branch): good / decent / bad segmentation / poor recognition / false positive / false negative, computed on held-out fold videos after each arch's full postprocessing (hysteresis for v0.3.1, decode for e2e).
- **Headline ranking scalar:** `2·share_good + share_decent` (the `good_weighted` score already used for checkpoint selection on the e2e branch). Full six-bin shares + median boundary errors reported alongside; tie-breaks follow the metric doc's optimization order (falses worst, then poor recognition, then bad segmentation, then decent→good).
- Report mean ± std across folds. An ablation "wins" only if it beats the baseline mean by more than the fold std.

## Ablation list

### 1. Architecture: v0.3.1 (canonical) vs e2e two-head

- **v0.3.1:** BiLSTM → per-frame in-play logit → hysteresis postprocessing (low/high thresholds + smoothing + min duration).
- **e2e:** same LSTM backbone, anchor-free per-frame regression heads (pointness + dist-to-start + dist-to-end), asymmetric two-knee boundary hinge loss, decode + cross-window offset aggregation. Lives on the `training/e2e-segment-model` branch (`models/seg_lstm.py`, `train/seg_loss.py`, `train/seg_loop.py`).
- Question: does learning the boundaries end-to-end beat hand-tuned hysteresis, especially on start/end boundary error (decent→good movement)?
- Note: branches need unifying — the ablation runner should be able to instantiate either arch from one config, on one branch.

### 2. Architecture: LSTM + attention

- Add self-attention on top of (or interleaved with) the BiLSTM. Variants to try:
  - BiLSTM → single multi-head self-attention layer → head(s)
  - Attention-only (small transformer encoder) as a stretch comparison
- Question: does global context within the window help, given seq_len is 10–20 s?

### 3. Hyperparameter sweep

Around the current baseline (hidden 128, 2 layers, dropout 0.2, lr 1e-3, pos_weight 3.0):

- hidden_size: 64 / 128 / 256
- num_layers: 1 / 2 / 3
- dropout: 0.1 / 0.2 / 0.4
- lr: 3e-4 / 1e-3 / 3e-3
- pos_weight: 1 / 3 / 5
- seq_len_seconds: 10 / 20 / 30
- target_fps: 5 / 10 / 15

Sweep 1-factor-at-a-time from the baseline first (cheap, ~19 runs × k folds), then a small random search over the interactions that mattered.

<!-- add more below -->

<!-- results below -->

## Results ledger

### Far-player crop (two-pass YOLO), 2026-07-25 — NEGATIVE

**Question.** The far player is the `-1` sentinel in 99.1% of frames (YOLO
letterboxes 1920→960, leaving them ~80px). Does a second pose pass over a fixed
960x540 top-half crop, fed as two extra slots, improve segmentation F1?

**Setup.** 1080p 26-video corpus (legacy 11 excluded: natively 720p, so a 960-wide
crop is pure upscaling). One factor: `feature_set: v2_nearfar` (290 dims) vs `v2`
(580 dims). Identical splits (7115/867/1388 sequences), identical TCN champion
config (c64 L5 k3 drop0.2, cosine lr 1e-4, 30 epochs). Decode tuned on val,
reported on test, ranked on F1 over (good+decent).

**Mechanism worked.** Crop slot population 70.9% vs the far slot's 0.5% — a ~140x
increase, essentially matching the near player (72.3%). The court-mask + feet
filter removed very little, so most crop detections were genuinely on-court.

**Result.**

| run | dims | params | val F1 | test F1 | prec | rec |
|---|---|---|---|---|---|---|
| `20260725_v2base_1080p` | 290 | 152,067 | 40.3% | **42.3%** | 40.0% | 44.8% |
| `20260725_v2crop_1080p` | 580 | 170,627 | **46.6%** | 39.0% | 36.6% | 41.8% |

**The crop lost 3.3 F1 on test while winning 6.3 on val.** Not overfitting: the
crop run has lower train loss (1.308 vs 1.398) AND lower val loss (1.168 vs 1.205)
AND higher val bal_acc. Not decode miscalibration either — the crop's val-chosen
config is also its test oracle (39.0% both ways).

**Per-video test deltas: -23.3, +1.9, +12.0, -3.7, -6.1.** No consistent direction.
The aggregate is variance across 5 test videos, not a systematic effect.

**Conclusion.** Populating the far block does not convert into F1 here. Consistent
with the earlier ablation that dropping all 181 far dims cost only 1.2 F1: far-player
pose appears to carry little segmentation signal beyond what the near player already
provides, whether present or absent. The bottleneck is elsewhere (bad_seg is ~36% of
GT in both arms — boundary precision, not player visibility).

**Caveats on this measurement.**
- Test split favours the treatment: its 5 videos average ~90% crop population vs
  70.9% corpus-wide, and every weak video (`6a3448` 0.0%, `ef87ad` 38.3%,
  `31980d` 23.7%) is in train. A fair split would likely be *worse* for the crop.
- Only ONE video (`6a3448`) has a truly empty crop, and it is in train — so this
  experiment cannot detect a regression on the empty-crop case.
- 3 val / 5 test videos: per-video variance (+/-23 F1) swamps the aggregate.
  Treat +/-3 F1 differences on this corpus as noise.

**Do not re-run as-is.** If revisited: cross-validate over folds rather than one
split, and target boundary precision (bad_seg) rather than player visibility.

**Side effects kept (both arms, so the comparison stays valid):**
- Training now falls back to `default_court_mask.png` when court detection fails
  (4/26 videos), matching runtime `data_preprocessor.compute_court_mask`. Previously
  training passed those frames through UNFILTERED. This closes a real training/runtime
  divergence and is worth keeping independent of this result.
- `scripts/extract_crop_poses.py` now uses the manifest pose backend
  (`load_pose_backend(..., provider="coreml")`), so both passes run identical weights.
  Verified parity vs ultralytics `.pt`: 17,591 vs 17,575 detections (0.09%).
