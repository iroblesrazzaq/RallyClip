from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from training.eval.quality_harness import RunnerConfig, run_quality_eval


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RallyClip e2e quality evaluation and write a report.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tests/fixtures/quality/manifest.json"),
        help="Quality fixture manifest",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("tests/fixtures/quality/gt"),
        help="Directory containing GT JSON fixtures",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path(os.environ.get("RALLYCLIP_EVAL_VIDEO_DIR", "")),
        help="Directory containing source videos",
    )
    parser.add_argument("--output", type=Path, default=Path("quality_report.json"), help="Report JSON path")
    parser.add_argument("--runner", choices=["dev", "release"], default="dev")
    parser.add_argument("--release-bin", type=Path, default=_env_path("RALLYCLIP_RELEASE_BIN"))
    parser.add_argument(
        "--command",
        default=f"{sys.executable} -m cli.main",
        help="Dev runner command, split with shell-like quoting",
    )
    parser.add_argument("--artifact-dir", type=Path, help="Model artifact directory to pass through to rallyclip")
    args = parser.parse_args()

    if not args.video_dir or not args.video_dir.is_dir():
        raise SystemExit("Set --video-dir or RALLYCLIP_EVAL_VIDEO_DIR to the source video directory.")
    extra_args = ("--artifact-dir", str(args.artifact_dir)) if args.artifact_dir else ()
    if args.runner == "release":
        if args.release_bin is None:
            raise SystemExit("Set --release-bin or RALLYCLIP_RELEASE_BIN for release mode.")
        runner = RunnerConfig(mode="release", release_bin=args.release_bin, extra_args=extra_args)
    else:
        runner = RunnerConfig(mode="dev", command=tuple(shlex.split(args.command)), extra_args=extra_args)

    report = run_quality_eval(
        manifest_path=args.manifest,
        video_dir=args.video_dir,
        gt_dir=args.gt_dir,
        output_path=args.output,
        runner=runner,
    )
    print(json.dumps(report["pooled"], indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    return 0


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw) if raw else None


if __name__ == "__main__":
    raise SystemExit(main())
