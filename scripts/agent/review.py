#!/usr/bin/env python3
"""Headless, read-only code review of the current branch by Cursor and Greptile.

Claude Code authors every change; this script only collects second opinions.
It reviews committed changes (merge-base..HEAD) and never writes into the repo:
diff, raw reviewer output and normalized findings go to an out dir under
$TMPDIR/rallyclip-review/. Triage is the caller's job (see /agent-review).

Exit codes: 0 all reviewers ran, 1 usage/git error, 2 a reviewer failed
(partial findings still written), 3 a reviewer modified the working tree.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROMPT_FILE = Path(__file__).with_name("cursor_review_prompt.md")
DEFAULT_MODEL = "cursor-grok-4.6-high"
SEVERITIES = ("high", "medium", "low")
LENSES = {
    "correctness": (
        "Logic errors, wrong edge-case handling, off-by-one in seconds/frames/fps "
        "math, error handling that hides failures, resource leaks, races."
    ),
    "invariants": (
        "Violations of the project hard constraints below, broken contracts between "
        "CLI / /api/* / GUI, hardcoded model contract values, anything that changes "
        "golden CLI parity without saying so."
    ),
    "tests": (
        "Changed behavior with no test, tests that do not actually assert the "
        "change, skips/deselects/markers that hide a regression."
    ),
    "perf": (
        "Memory or IO blowups on multi-hour 1080p60 videos, accidental full-video "
        "loads, per-frame work that should be batched, blocking calls on the GUI path."
    ),
}


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def tree_snapshot(root: Path) -> str:
    """Hash of HEAD + status + uncommitted diff; changes if a reviewer edits anything."""
    h = hashlib.sha256()
    for args in (("rev-parse", "HEAD"), ("status", "--porcelain", "-uall"), ("diff", "HEAD")):
        h.update(git(root, *args).encode())
    return h.hexdigest()


def hard_constraints(root: Path) -> str:
    text = (root / "AGENTS.md").read_text(encoding="utf-8")
    m = re.search(r"^## Hard constraints.*?$(.*?)^## ", text, re.S | re.M)
    return m.group(1).strip() if m else "(AGENTS.md has no 'Hard constraints' section)"


def norm_severity(value: object) -> str:
    s = str(value or "").lower()
    if s in SEVERITIES:
        return s
    if s in ("critical", "p0", "p1", "error", "blocker"):
        return "high"
    if s in ("p2", "warning", "major"):
        return "medium"
    return "low"


def extract_findings(text: str) -> list[dict]:
    m = re.search(r"BEGIN_FINDINGS\s*(.*?)\s*END_FINDINGS", text, re.S)
    if not m:
        raise ValueError("no BEGIN_FINDINGS/END_FINDINGS block in reviewer output")
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", m.group(1).strip())
    data = json.loads(body or "[]")
    if not isinstance(data, list):
        raise ValueError("findings block is not a JSON array")
    return data


def run_cursor(root: Path, out: Path, ctx: dict, lens: str, model: str, timeout: int) -> list[dict]:
    template = PROMPT_FILE.read_text(encoding="utf-8")
    prompt = template.format(lens=lens, lens_desc=LENSES[lens], **ctx)
    cmd = [
        "agent", "-p", "--trust", "--mode", "ask", "--model", model,
        "--output-format", "json", "--workspace", str(root), "--add-dir", str(out), prompt,
    ]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout)
    (out / f"cursor-{lens}.raw.json").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"agent exited {proc.returncode}: {proc.stderr.strip()[:300]}")
    envelope = json.loads(proc.stdout)
    if envelope.get("is_error"):
        raise RuntimeError(f"agent reported error: {str(envelope.get('result'))[:300]}")
    return [
        {
            "reviewer": f"cursor:{model}",
            "lens": lens,
            "file": f.get("file"),
            "line": f.get("line"),
            "severity": norm_severity(f.get("severity")),
            "claim": f.get("claim", ""),
            "failure_scenario": f.get("failure_scenario", ""),
        }
        for f in extract_findings(envelope.get("result", ""))
    ]


def _greptile_items(data: object) -> list[dict]:
    """Find the list of comment objects in Greptile's JSON, whatever it is nested under."""
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        for key in ("comments", "findings", "issues", "results"):
            if isinstance(data.get(key), list):
                return _greptile_items(data[key])
        for value in data.values():
            if isinstance(value, (dict, list)):
                items = _greptile_items(value)
                if items:
                    return items
    return []


def run_greptile(root: Path, out: Path, base_branch: str, timeout: int) -> list[dict]:
    cmd = ["greptile", "review", "--branch", base_branch, "--json", "--no-color"]
    env = {**os.environ, "NO_COLOR": "1"}
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout, env=env)
    (out / "greptile.raw.json").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"greptile exited {proc.returncode}: {proc.stderr.strip()[:300]}")
    findings = []
    for c in _greptile_items(json.loads(proc.stdout)):
        findings.append({
            "reviewer": "greptile",
            "lens": "general",
            "file": c.get("file") or c.get("path") or c.get("filePath"),
            "line": c.get("line") or c.get("startLine") or c.get("start_line"),
            "severity": norm_severity(c.get("severity") or c.get("priority") or c.get("level")),
            "claim": c.get("body") or c.get("message") or c.get("comment") or c.get("title") or "",
            "failure_scenario": c.get("failure_scenario") or c.get("suggestion") or "",
        })
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main", help="base ref (default: origin/main)")
    ap.add_argument("--reviewers", default="cursor,greptile", help="comma list: cursor,greptile")
    ap.add_argument("--lenses", default="correctness,invariants,tests",
                    help=f"Cursor lenses, comma list of {','.join(LENSES)}")
    ap.add_argument("--model", default=os.environ.get("RALLYCLIP_REVIEW_MODEL", DEFAULT_MODEL),
                    help="Cursor model (env RALLYCLIP_REVIEW_MODEL; see `agent --list-models`)")
    ap.add_argument("--out", type=Path, help="output dir (default: $TMPDIR/rallyclip-review/<branch>-<ts>)")
    ap.add_argument("--timeout", type=int, default=900, help="per-reviewer timeout in seconds")
    args = ap.parse_args(argv)

    reviewers = [r for r in args.reviewers.split(",") if r]
    lenses = [lens for lens in args.lenses.split(",") if lens]
    bad = [r for r in reviewers if r not in ("cursor", "greptile")] + [x for x in lenses if x not in LENSES]
    if bad:
        print(f"error: unknown reviewer/lens: {', '.join(bad)}", file=sys.stderr)
        return 1
    for r, exe in (("cursor", "agent"), ("greptile", "greptile")):
        if r in reviewers and not shutil.which(exe):
            print(f"error: `{exe}` CLI not on PATH (needed for {r})", file=sys.stderr)
            return 1

    try:
        root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
        branch = git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
        merge_base = git(root, "merge-base", args.base, "HEAD").strip()
        head = git(root, "rev-parse", "HEAD").strip()
        diff = git(root, "diff", f"{merge_base}..HEAD")
        files = git(root, "diff", "--name-only", f"{merge_base}..HEAD").split()
    except subprocess.CalledProcessError as e:
        print(f"error: git {' '.join(e.cmd[1:])}: {e.stderr.strip()}", file=sys.stderr)
        return 1
    if not diff:
        print(f"nothing to review: HEAD has no commits beyond {args.base}")
        return 0
    if git(root, "status", "--porcelain").strip():
        print("warning: uncommitted changes are NOT reviewed (only merge-base..HEAD)", file=sys.stderr)

    out = args.out or Path(tempfile.gettempdir()) / "rallyclip-review" / (
        f"{re.sub(r'[^A-Za-z0-9._-]', '-', branch)}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    out = out.resolve()
    if out == root or root in out.parents:
        print("error: --out must be outside the repo", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    diff_path = out / "diff.patch"
    diff_path.write_text(diff, encoding="utf-8")
    ctx = {
        "diff_path": diff_path, "base": args.base, "merge_base": merge_base[:10],
        "head": head[:10], "files_changed": len(files), "constraints": hard_constraints(root),
    }

    before = tree_snapshot(root)
    jobs: dict[str, object] = {}
    with cf.ThreadPoolExecutor(max_workers=len(lenses) + 1) as pool:
        if "cursor" in reviewers:
            for lens in lenses:
                jobs[f"cursor:{lens}"] = pool.submit(run_cursor, root, out, ctx, lens, args.model, args.timeout)
        if "greptile" in reviewers:
            base_branch = args.base.split("/", 1)[1] if args.base.startswith("origin/") else args.base
            jobs["greptile"] = pool.submit(run_greptile, root, out, base_branch, args.timeout)
        findings, failures = [], {}
        for name, fut in jobs.items():
            try:
                findings += fut.result()
            except Exception as e:  # noqa: BLE001 - report every reviewer failure, keep the rest
                failures[name] = f"{type(e).__name__}: {e}"
    modified = tree_snapshot(root) != before

    findings.sort(key=lambda f: (SEVERITIES.index(f["severity"]), str(f["file"]), f["line"] or 0))
    for i, f in enumerate(findings, 1):
        f["id"] = i
    report = {
        "branch": branch, "base": args.base, "merge_base": merge_base, "head": head,
        "files_changed": len(files), "reviewers": list(jobs), "failures": failures,
        "tree_modified_by_reviewer": modified, "findings": findings,
    }
    (out / "findings.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    counts = {s: sum(f["severity"] == s for f in findings) for s in SEVERITIES}
    print(f"review {branch} vs {args.base} ({len(files)} files) -> {out / 'findings.json'}")
    print(f"findings: {counts['high']} high, {counts['medium']} medium, {counts['low']} low")
    for f in findings:
        loc = f"{f['file']}:{f['line']}" if f["line"] else str(f["file"])
        print(f"  #{f['id']} [{f['severity']}] {f['reviewer']}/{f['lens']} {loc} - {f['claim'][:160]}")
    for name, err in failures.items():
        print(f"FAILED {name}: {err}", file=sys.stderr)
    if modified:
        print("ERROR: a reviewer modified the working tree; inspect `git status` before continuing",
              file=sys.stderr)
        return 3
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
