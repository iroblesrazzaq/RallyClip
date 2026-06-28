# Current State

| Field | Value |
| --- | --- |
| Repository root directory | `/Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/RallyClip` |
| Standard startup path | `./init.sh` |
| Standard evidence path | Feature-specific; define before marking a feature `passing` |
| Highest priority unfinished feature | `runtime-video-validation` |
| Current blocker | none |

# Session Record

## Session 0 - 2026-06-28

Harness scaffolded. Feature-specific evidence definitions are intentionally deferred.

## Session 1 - 2026-06-28

Video UX work was moved away from the fragile browser/WebM chunk player for the packaged Mac app. The app now uses a native PySide6 QtMultimedia viewer with source-time playback, point-skip scheduling, hover controls, fullscreen, keyboard shortcuts, CSV/export actions, and a point-aware timeline overlay. Current manual testing shows the native viewer is working decently well for release use; the initial slow-mo behavior appears to have been buffering-related rather than a scheduler bug.

Still left: formal release-harness feature work has not started. The highest-priority tracked feature remains `runtime-video-validation`, and browser chunk playback remains a dev fallback rather than the production Mac playback path.
