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
