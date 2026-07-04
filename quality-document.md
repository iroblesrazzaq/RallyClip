# Quality Document

## Product Domains

| Domain | Grade (A-D) | Evidence Status | Agent Legibility | Evidence Stability | Key Gaps |
| --- | --- | --- | --- | --- | --- |
| runtime | D | Not started | Feature intent is documented in `feature_list.json`. | Undefined | Input validation behavior needs evidence. |
| cli | D | Not started | Feature intent is documented in `feature_list.json`. | Undefined | Config and flag contract needs evidence. |
| desktop_gui | D | Not started | Feature intent is documented in `feature_list.json`. | Undefined | GUI flow needs end-to-end evidence. |
| segmentation | D | Not started | Feature intent is documented in `feature_list.json`. | Undefined | Interval decoding needs deterministic evidence. |
| training | D | Not started | Feature intent is documented in `feature_list.json`. | Undefined | Quality metrics need repeatable evidence. |

## Architectural Layers

| Layer | Grade (A-D) | Boundary Enforcement | Agent Legibility | Notes |
| --- | --- | --- | --- | --- |
| CLI and configuration | D | Undefined | Medium | Python 3.11 runtime; CLI behavior should remain isolated from GUI concerns. |
| Runtime assets and device selection | D | Undefined | Medium | Model assets, paths, and device fallback need clear contracts. |
| Preprocessing and pose extraction | D | Undefined | Medium | Video validation, court detection, and pose extraction should expose stable interfaces. |
| Inference and segmentation | D | Undefined | Medium | Frame probabilities and interval decoding should have clear feature-specific evidence. |
| GUI and desktop packaging | D | Undefined | Medium | Flask/browser UI and PySide desktop shell should share backend contracts. |
| Training and evaluation | D | Undefined | Medium | Dataset, features, model, and metric layers should keep reproducible evaluation boundaries. |

## Harness Simplification

When the harness becomes too heavy, use snapshot-remove-benchmark-compare:

1. Snapshot the current harness files and evidence records.
2. Remove one rule, checklist item, or document field that appears redundant.
3. Evaluate a representative feature handoff using that feature's evidence definition.
4. Compare whether correctness, evidence quality, scope discipline, or handoff readiness got worse.
5. Keep the removal only when the simpler harness preserves or improves the benchmark result.
