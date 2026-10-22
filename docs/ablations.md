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

### Boundary-only objective + DP pairing decode, 2026-07-29

**Question.** Train the TCN on startness/endness only (no pointness objective),
and pair start->end at decode. Two sub-questions turned out to be entangled.

**Setup.** 37-video 720p corpus, v1 362-dim features, TCN c64 L5 k3, cosine 1e-4.
`heatmap_cls_weight: 0.0` removes the pointness term (note: the loss reads
`heatmap_cls_weight`, NOT `cls_weight`). sigma 0.7s. selection_metric: f1 (added
this session -- `acceptable` is recall-only, which under peak-pairing rewards
emitting more peaks since nothing bounds the segment count).

**Results** (val-tuned decode, test-reported, ranked on F1 over good+decent):

| model | decoder | test F1 | prec | rec |
|---|---|---|---|---|
| champion (with pointness) | hybrid | **41.5%** | 38.8% | **44.6%** |
| boundary-only, DP-selected ckpt | DP lambda=3 | 33.6% | 38.6% | 29.8% |
| boundary-only, blind-selected ckpt | DP lambda=3 | 29.1% | 38.4% | 23.5% |
| champion | DP lambda=2 | 28.6% | 32.2% | 25.7% |
| boundary-only | greedy peakpair | 14.2% | 24.5% | 10.0% |
| champion | greedy peakpair | 6.9% | 14.3% | 4.6% |

**1. The decoder was worth +21.7 F1** (6.9 -> 28.6 on an identical checkpoint).
Greedy peakpair walks starts left to right and takes the first free end, so one
bad match cascades. `decode_pairdp` scores a segmentation by summed boundary
log-odds `logit(p_start) + logit(p_end) - lambda` and maximises exactly over all
valid segmentations (alternating, non-overlapping, duration-bounded) by DP.
The lambda term is half the win: at lambda=0 the DP over-segments (782 preds for
460 GT, 17.2% F1) because every positive-scoring pair gets admitted.

**2. Checkpoint selection through a broken decoder cost +4.5 F1** (29.1 -> 33.6).
The first run selected via greedy peakpair at peak_threshold 0.3, where 16% of
frames clear threshold against 0.9% true boundaries -- every epoch scored ~1%
good+decent, so "best.pth" was arbitrary. Wiring the tuned DP decode into
training fixed selection without touching the model.

**3. The objective comparison was unmeasurable until the decoder worked.**
Boundary-only vs pointness-trained read +5.9 F1 under greedy, +0.5 under DP
lambda=0, and +5.0 under tuned DP. Single split, and CV std on this corpus is
~10 F1. NOT settled -- do not cite a number for this.

**Where hybrid still wins: recall, not precision.** 38.6 vs 38.8 precision (a
dead heat) against 29.8 vs 44.6 recall. Two boundary heatmaps underdetermine HOW
MANY points exist; DP guesses with a scalar lambda while hybrid reads it off the
pointness track. Pointness earns its keep in the DECODER, not the loss.

**Next.** DP pairing with pointness informing the segment prior (e.g. add mean
pointness over [s,e] to the segment score in place of a constant lambda) --
keeps DP's boundary precision and hybrid's recall. Diagnostic supporting this:
the boundary-only model predicts 0.85 at true boundary frames vs 0.18
background, so boundary localisation is not the limitation.
