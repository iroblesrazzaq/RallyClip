# Evaluator Rubric

Score each dimension from 0 to 2.

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Correctness | Implementation does not match target behavior. | Implementation partially matches target behavior with known gaps. | Implementation matches target behavior. |
| Evidence | Required evidence is missing. | Evidence is present but incomplete or indirect. | Feature-specific evidence is recorded. |
| Scope discipline | Work drifted outside the selected feature without justification. | Scope was mostly contained, with minor undocumented drift. | Agent stayed within the selected feature and documented any necessary exception. |
| Reliability | Result breaks after restart, rerun, or ordinary input variation. | Result usually survives rerun but has fragile assumptions. | Result survives restart, rerun, and expected input variation. |
| Maintainability | Code or docs are hard for the next session to understand. | Code or docs are understandable but leave avoidable ambiguity. | Code and docs are clear for the next session. |
| Handoff readiness | A new session cannot continue from repo artifacts alone. | A new session can continue after some inference or external context. | A new session can continue from repo artifacts alone. |

## Conclusion Options

- Accept: total score is 10-12 and no dimension scores 0.
- Revise: total score is 6-9 or any dimension needs focused follow-up.
- Block: total score is 0-5, evidence is absent, or the work cannot be evaluated safely.

## Tuning Notes

Expect 3-5 calibration rounds before this rubric fits RallyClip's actual workflow. Record every rubric change in a new decision entry or in `claude-progress.md`, including what evaluation failed, what changed, and why the new wording is more useful.
