# Session Handoff

## Current state

Harness environment loads through `init.sh`. Video UX for the Mac app has been reworked on `perf/streaming-pipeline` to use the native PySide6 QtMultimedia viewer rather than the browser chunk player.

## Changes this session

Native player work is functioning decently for release use: source-time playback, point skipping, gap bridge behavior, fullscreen, hover controls, keyboard shortcuts, CSV/export actions, and point-aware timeline overlay are in place. Saved-match card polish was also adjusted so titles are lighter and timestamps omit seconds.

## Still broken or unknown

Formal release-harness features are still not started. Browser chunk playback remains a fallback/dev path and is not the production Mac playback authority.

## Next best action

Pick feature with priority 1 from `feature_list.json`: `runtime-video-validation`.

## Commands

- Install: `: # local checkout uses the existing .venv-train environment`
- Start: `.venv-train/bin/rallyclip gui`
- Native playback tests: `python3 -m pytest -q tests/test_native_playback.py tests/test_desktop_dispatch.py`
- Evidence: feature-specific; define before marking a feature `passing`
