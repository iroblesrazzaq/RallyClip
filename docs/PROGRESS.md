# PROGRESS

_Updated 2026-10-07: native macOS implementation._

## Repo state

- Worktree `native-macos`, branch `feat/native-macos`, based on
  `feat/runtime-v060` with `fix/pr59-export-guards` merged (7dccd30).
- App/model version 0.6.0. Verified weights are published in GitHub release
  `artifact-rallyclip_v0.6.0`; no binaries added to git.
- Do not merge; the user merges. Runtime PR #62 is still a dependency.

## Implementation

- Native SwiftUI sidebar/player/editor with original RallyClip logo/icon,
  system font, light/dark appearance, green accent, factual UI text.
- Single continuous AVFoundation composition across points/camera files;
  full-match mode, fullscreen/PiP, seek, point selection, timeline zoom.
- Save/discard/reset boundary edits; add/delete intervals; guard navigation,
  window close, quitting, active exports and analysis.
- Native all-point/highlight/separate-video and CSV export, staging,
  cancellation, progress. Audio mix renders silence at delayed-audio gaps.
- Authenticated loopback helper bundles the proven v0.6 torch-free engine and
  learned court detector. Native imports reference originals and analyze a
  720p proxy. Existing Application Support saved matches are reused.
- Local build/run scripts, Swift tests/media probe, packaged import/edit/media
  smoke. Release workflow defaults to native; retains legacy manual option;
  existing Apple secrets/scripts sign, notarize, staple and upload the DMG.

## Verification this session

- Python gate: 483 passed, 7 skipped, 49 deselected; compileall and scoped ruff pass.
- Swift: 5 tests pass, including exported frame verification across a source boundary.
- Frozen backend analyzed the tennis fixture; edit/reset/delete passed. Native
  playback decoded 189 frames across three cuts; four-second export duration
  4.000067s, observed advancing-playback gaps about 23ms.
- Delayed audio boundary test verifies sound/silence/sound with decoded samples;
  native exporter now renders silence instead of passing AAC gap edit lists.
- Interactive import, saved edits and native MP4 export passed; output1280x720,
  17.466s as expected. Isolated library `/private/tmp/rallyclip-native-qa`.

## Remaining

- Run final packaged smoke and window-close guard QA on the final UI build.
- Push PR stacked on `feat/runtime-v060`; Greptile until 5/5.
- Run existing Release workflow on this branch; verify signed/notarized DMG,
  download it and test installed bundle on this Mac. Update evidence/results.
