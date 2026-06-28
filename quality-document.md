# Quality Document

## Product Domains

| Domain | Grade (A-D) | Verification Status | Agent Legibility | Test Stability | Key Gaps |
| --- | --- | --- | --- | --- | --- |
| runtime | D | Not started | Feature intent is documented in `feature_list.json`. | Unverified | Input validation behavior needs evidence. |
| cli | D | Not started | Feature intent is documented in `feature_list.json`. | Unverified | Config and flag contract needs evidence. |
| desktop_gui | D | Not started | Feature intent is documented in `feature_list.json`. | Unverified | GUI flow needs end-to-end evidence. |
| segmentation | D | Not started | Feature intent is documented in `feature_list.json`. | Unverified | Interval decoding needs deterministic evidence. |
| training | D | Not started | Feature intent is documented in `feature_list.json`. | Unverified | Quality metrics need repeatable evidence. |

## Architectural Layers

| Layer | Grade (A-D) | Boundary Enforcement | Agent Legibility | Notes |
| --- | --- | --- | --- | --- |
| CLI and configuration | D | Unverified | Medium | Python 3.11 runtime; CLI behavior should remain isolated from GUI concerns. |
| Runtime assets and device selection | D | Unverified | Medium | Model assets, paths, and device fallback need clear contracts. |
| Preprocessing and pose extraction | D | Unverified | Medium | Video validation, court detection, and pose extraction should expose stable interfaces. |
| Inference and segmentation | D | Unverified | Medium | Frame probabilities and interval decoding should be independently testable. |
| GUI and desktop packaging | D | Unverified | Medium | Flask/browser UI and PySide desktop shell should share backend contracts. |
| Training and evaluation | D | Unverified | Medium | Dataset, features, model, and metric layers should keep reproducible evaluation boundaries. |

## Harness Simplification

When the harness becomes too heavy, use snapshot-remove-benchmark-compare:

1. Snapshot the current harness files and verification outputs.
2. Remove one rule, checklist item, or document field that appears redundant.
3. Run the standard verification command and evaluate a representative feature handoff.
4. Compare whether correctness, evidence quality, scope discipline, or handoff readiness got worse.
5. Keep the removal only when the simpler harness preserves or improves the benchmark result.
