# Segment-level evaluation metrics (end-objective eval)

Status: **spec only** — not implemented yet. Complements (does not replace) the existing
frame-level metrics (balanced accuracy, frame F1, AUROC) and the current segment F1/IoU,
which scale with point length and hide boundary quality.

## Motivation

The product objective is "good clips": a predicted point should start a little before the
actual point and end a little after it. We care about **absolute boundary distance in
seconds**, not IoU's relative scaling — a 0.5 s late start is equally bad on a 3 s point
and a 30 s point. IoU is used only for matching and as a coarse gate for the lower tiers.

## Definitions

- Ground-truth (GT) point: `[gt_start, gt_end]` in seconds.
- Predicted segment: `[pred_start, pred_end]` after postprocessing.
- Boundary errors for a matched pair:
  - `start_err = pred_start − gt_start` (negative ⇒ started **early** ⇒ extra footage ⇒ "outside")
  - `end_err   = pred_end − gt_end`     (positive ⇒ ended **late**  ⇒ extra footage ⇒ "outside")
- "Outside" tolerance = how much extra footage around the point is acceptable.
  "Inside" tolerance = how much of the actual point we may cut off (much stricter).

## Evaluation flow

### Step 1 — IoU cross-matrix

Compute IoU between every predicted segment and every GT point.

### Step 2 — Falses

- Predicted segment whose IoU with **every** GT point is **< 0.1** → **false positive**
  (a point was created where there shouldn't be one — negligible or no overlap).
- GT point whose IoU with **every** predicted segment is **< 0.1** → **false negative**
  (a real point the model missed).

These are removed from the pool before matching.

### Step 3 — Resolve to a 1-to-1 mapping

Remaining entities may form many-to-one mappings (two predictions overlapping one GT =
split error; one prediction spanning two GTs = merge error). When this happens the
outcome is definitely not a correct classification, regardless of how we resolve it.

**Heuristic:**

1. Take all (pred, GT) pairs with IoU ≥ 0.1 and match greedily by descending IoU,
   each prediction and each GT used at most once.
2. **Leftover predictions** (their only overlapping GT was claimed by a better
   prediction — duplicates/fragments from a split) → count as **false positive**.
   Product view: an extra clip the user has to delete.
3. **Leftover GT points** (their only overlapping prediction was claimed by another GT —
   the losing side of a merge) → count as **false negative**. Product view: that point
   doesn't get its own clip.
4. The *winning* pair of a split/merge is tiered normally in Step 4 — its boundary
   errors are large by construction (a merge winner ends far too late; a split winner
   cuts the point short), so the tolerances naturally demote it out of Good/Decent.

This keeps exactly six groups (no separate split/merge bucket) and pushes split/merge
errors into the falses, which is consistent with the optimization order below. If
splits/merges turn out to be frequent, add diagnostic counters (`n_split`, `n_merge`)
without changing the group taxonomy.

### Step 4 — Tier the 1-to-1 matched pairs

Each pair gets the best tier it satisfies:

**1. Good** — both boundaries within the tight tolerances:

| boundary | tolerance (outside / inside) | allowed range |
|---|---|---|
| start | 0.5 s early / 0.2 s late | `−0.5 ≤ start_err ≤ +0.2` |
| end   | 0.2 s early / 0.5 s late | `−0.2 ≤ end_err ≤ +0.5` |

**2. Decent** — not Good, both boundaries within the relaxed tolerances:

| boundary | tolerance (outside / inside) | allowed range |
|---|---|---|
| start | 1.5 s early / 0.5 s late | `−1.5 ≤ start_err ≤ +0.5` |
| end   | 0.5 s early / 1.5 s late | `−0.5 ≤ end_err ≤ +1.5` |

The remaining matched pairs (IoU ≥ 0.1 but outside Decent tolerances) divide into two:

**3. Point recognized, bad segmentation** — **IoU ≥ 0.5**. The point was properly
recognized, but a boundary is wrong somewhere (started/ended too early or too late).

**4. Poor point recognition** — **0.1 ≤ IoU < 0.5**. Some overlap, but bad enough that
this no longer counts as a recognized point.

Plus from Steps 2–3:

**5. False positive** · **6. False negative**

## Main description: the 6-group proportion

Every prediction and every GT point lands in exactly one group occurrence (a matched
pair counts once). The headline output is the share of each group over all events:

```
n_events = n_matched_pairs + n_false_positive + n_false_negative
share(g) = count(g) / n_events
```

Ideally everything is in Good, maybe Decent. **Optimization order** (worst first):

1. minimize **false positives / false negatives**
2. then **poor point recognition** (bad IoU)
3. then **recognized, bad segmentation**
4. then move **Decent → Good**

## Secondary metrics

- `false_positives_per_hour` (FPs have no GT denominator; normalize by video duration).
- Boundary-error distributions over matched pairs: mean / median / p90 of signed
  `start_err` and `end_err` — systematic early/late bias directly informs postprocess
  tuning (hysteresis thresholds, min-duration, smoothing sigma).
- Per-video breakdown alongside the aggregate, to spot footage-specific failure modes.

---

# Loss function (end-to-end segment prediction)

Status: **design spec** — for a future e2e model that predicts segments directly,
replacing hysteresis filtering. Kept strictly separate from the metric above:

- The **metric** is the 6-bin proportion — thresholded, interpretable, what we report.
- The **loss** is a continuous relaxation — differentiable, with gradients that always
  point toward the Good zone. It must **not** replicate the metric's bin thresholds:
  hard thresholds create zero-gradient plateaus inside bins and discontinuous cliffs at
  bin edges. The only intentional flat region is the Good zone itself (zero loss there
  by design). The 0.1 IoU cutoff exists **only in the metric**; in the loss, a 0.09-IoU
  near-miss naturally incurs less loss than a zero-overlap miss — no special casing.

## Prerequisite: the model must output segments

A segment loss needs segment predictions: tuples `(confidence, start, end)`. Two
candidate head designs:

1. **Set prediction (DETR-style):** K learnable queries → K candidate segments;
   Hungarian (or 1D-ordered) matching to GT during training. NMS-free. **Rejected for
   now**: training-time matching means a missed GT is handled by recruiting a slot
   ("mapping a false negative to a made-up positive"), matching is unstable early in
   training, and set prediction is data-hungry — wrong tool at ~10 h of footage.
2. **Anchor-free per-frame regression (ActionFormer-style) — CHOSEN, implemented in
   `training.models.seg_lstm` / `training.train.seg_loss`:** keep the BiLSTM; each
   frame predicts `(pointness_t, dist_to_start_t, dist_to_end_t)`; decode + merge at
   inference. No matching at train time at all: a missed point means every frame
   inside it carries pointness target 1 and contributes weighted loss directly (FN),
   hallucinated frames carry target 0 (FP). Matching survives only in the eval-time
   metric, where it belongs. Runs truncated by the window edge have the cut side
   masked out of the regression terms.

The loss below is written in the set-prediction form for generality; the implemented
per-frame translation applies the boundary hinge to the offset regressions of in-point
frames and replaces the confidence matching terms with per-frame weighted BCE.

## Matching (training-time)

Hungarian matching between the K predictions and the n GT points, with cost
`= −w_conf·confidence + boundary cost + IoU cost`. With `K ≥ n_gt`, **every GT point is
matched** — this is the mechanism that makes false negatives punishable: a missed point
is not a special loss term but a matched pair with large boundary/IoU loss plus a
confidence term pulling that slot up. The gradient always has a prediction to flow into.

## Components

### 1. Boundary loss — asymmetric two-knee hinge (per matched pair)

Zero loss inside the Good zone, slope λ₁ until the Decent edge, slope λ₂ beyond.
For one boundary with Good zone `[−g_out, +g_in]` and Decent zone `[−d_out, +d_in]`
(start boundary: g_out=0.5, g_in=0.2, d_out=1.5, d_in=0.5; end mirrored):

```
u = max(−e − g_out, 0)          # violation in the outside direction (too early/late-out)
v = max( e − g_in,  0)          # violation in the inside direction  (cut into the point)

L_b(e) = λ₁·min(u, d_out−g_out) + λ₂·max(u − (d_out−g_out), 0)      # outside branch
       + μ₁·min(v, d_in −g_in ) + μ₂·max(v − (d_in −g_in ), 0)      # inside branch
```

with `λ₁ < λ₂`, `μ₁ < μ₂`, and **μ ≥ λ** (cutting off part of the point is worse than
including extra footage). Continuous, convex, piecewise-linear; kinks are ReLU-like and
fine for SGD (optionally Huber-smooth them). `L_boundary = L_b(start_err) + L_b(end_err)`.

### 2. IoU component — 1D DIoU (per matched pair)

`L_iou = 1 − DIoU(pred, gt)` where `DIoU = IoU − (center_dist / enclosing_len)²`.

Plain `1 − IoU` has **zero gradient at zero overlap** (a pred 1 s away and 50 s away
both score IoU 0) — it stops pointing in the right direction exactly where we need it
most. The DIoU center-distance term keeps a nonzero gradient toward the GT at any
distance, and gives the graded 0-vs-0.09-IoU behavior automatically. Weight this term
small (β): the boundary hinges already carry the absolute-time objective; DIoU mainly
covers the far-miss regime and adds mild scale-relative pressure.

### 3. Confidence loss

Focal/BCE on each slot's confidence:
- **Matched slots** → target 1, weighted by `w_pos` (set high: an unconfident matched
  slot on a real point is an incipient **false negative** — the biggest sin).
- **Unmatched slots** → target 0, weighted by `w_neg`: this is the **false-positive
  penalty**. Unmatched predictions get no boundary gradient (nothing to regress toward);
  suppressing their confidence is the correct and only gradient.

## Total

```
L = Σ_matched [ α·L_boundary + β·L_iou + w_pos·FL(conf, 1) ]
  + Σ_unmatched [ w_neg·FL(conf, 0) ]
```

Weight ordering mirrors the metric's optimization order: `w_pos` (FN) ≥ `w_neg` (FP)
> β (poor IoU) with the hinge slopes (λ, μ) shaping the segmentation-quality tail.

## Gradient sanity checklist

- No zero-overlap dead zone: DIoU distance term + boundary hinges both give gradients
  at arbitrary distance. ✔
- No bin cliffs: all thresholds in the loss are hinge knees (continuous), never step
  functions. ✔
- Intentional flat region: exactly the Good zone, nowhere else (no pressure toward the
  GT center once inside Good — any position in the zone is equally acceptable). ✔
- FN gradient path: forced matching ⇒ every GT injects gradient into some slot. ✔
- FP gradient path: confidence suppression on unmatched slots. ✔

## Open implementation questions

- Head choice (set prediction vs per-frame regression) and K (max points per window).
- Window truncation: 20 s training sequences cut points at window edges — either train
  on full videos (BiLSTM allows it) or mask/clip boundary targets at window edges.
- Exact coefficients (λ₁, λ₂, μ₁, μ₂, α, β, w_pos, w_neg) — tune on the 6-bin metric.
- Huber-smoothing of hinge knees, focal-loss γ.
