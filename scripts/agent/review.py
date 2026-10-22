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
import signal
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


_LIVE: set[subprocess.Popen] = set()  # reviewer processes still running (killed on Ctrl-C)


def kill_tree(proc: subprocess.Popen) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # Windows: no process groups via start_new_session; kill the direct child
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


def run_capped(cmd: list[str], cwd: Path, timeout: int, env: dict | None = None) -> subprocess.CompletedProcess:
    """subprocess.run with a timeout that holds when the tool forks.

    `agent` is a bash wrapper around node; on timeout subprocess.run kills only the
    wrapper and then waits for the grandchildren's pipes, so the cap never fires.
    Run in a fresh process group and kill the whole group instead.
    """
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    _LIVE.add(proc)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        proc.communicate()
        raise
    finally:
        _LIVE.discard(proc)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def tree_snapshot(root: Path) -> str:
    """Fingerprint of HEAD, tracked edits, untracked file contents and ignored-file stats.

    Changes if a reviewer edits, creates or deletes anything in the checkout.
    Ignored files (weights, venvs, caches) are fingerprinted by size+mtime, not hashed.
    """
    h = hashlib.sha256()
    for args in (("rev-parse", "HEAD"), ("status", "--porcelain", "-uall"), ("diff", "HEAD")):
        h.update(git(root, *args).encode())
    for rel in git(root, "ls-files", "-z", "--others", "--exclude-standard").split("\0"):
        if rel and (root / rel).is_file():
            h.update(rel.encode() + hashlib.sha256((root / rel).read_bytes()).digest())
    for rel in git(root, "ls-files", "-z", "--others", "--ignored", "--exclude-standard").split("\0"):
        try:
            st = (root / rel).stat()
        except OSError:
            continue
        h.update(f"{rel}:{st.st_size}:{st.st_mtime_ns}".encode())
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


def norm_line(value: object) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def extract_findings(text: str) -> list[dict]:
    """Parse the LAST marker block; the model may quote the marker names or the prompt example first."""
    blocks = re.findall(r"^BEGIN_FINDINGS[ \t]*$(.*?)^END_FINDINGS[ \t]*$", text, re.S | re.M)
    if not blocks:
        raise ValueError("no BEGIN_FINDINGS/END_FINDINGS block in reviewer output")
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", blocks[-1].strip(), flags=re.I)
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
    for _attempt in range(2):  # one retry: the Cursor backend drops long streams ("Connection stalled")
        proc = run_capped(cmd, root, timeout)
        if proc.returncode == 0 or "RetriableError" not in proc.stderr:
            break
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
            "line": norm_line(f.get("line")),
            "severity": norm_severity(f.get("severity")),
            "claim": str(f.get("claim") or ""),
            "failure_scenario": str(f.get("failure_scenario") or ""),
        }
        for f in extract_findings(envelope.get("result", ""))
        if isinstance(f, dict)
    ]


def greptile_comments(data: object) -> list[dict]:
    """`greptile review --json` (v3) is {summary, confidence, ..., comments: [...]}."""
    if not isinstance(data, dict) or not isinstance(data.get("comments"), list):
        raise ValueError("unexpected greptile JSON: no top-level 'comments' list (CLI format changed?)")
    return [c for c in data["comments"] if isinstance(c, dict)]


def run_greptile(root: Path, out: Path, base: str, timeout: int, meta: dict) -> list[dict]:
    cmd = ["greptile", "review", "--branch", base, "--json", "--no-color"]
    env = {**os.environ, "NO_COLOR": "1"}
    proc = run_capped(cmd, root, timeout, env)
    (out / "greptile.raw.json").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"greptile exited {proc.returncode}: {proc.stderr.strip()[:300]}")
    data = json.loads(proc.stdout)
    comments = greptile_comments(data)
    meta["greptile_confidence"] = data.get("confidence")
    return [
        {
            "reviewer": "greptile",
            "lens": c.get("category") or "general",
            "file": c.get("path"),
            "line": norm_line(c.get("startLine")),
            "severity": norm_severity(c.get("severity")),
            "claim": str(c.get("body") or ""),
            "failure_scenario": str(c.get("suggestion") or ""),
        }
        for c in comments
    ]


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

    reviewers = [r.strip() for r in args.reviewers.split(",") if r.strip()]
    lenses = [x.strip() for x in args.lenses.split(",") if x.strip()]
    bad = [r for r in reviewers if r not in ("cursor", "greptile")] + [x for x in lenses if x not in LENSES]
    if bad:
        print(f"error: unknown reviewer/lens: {', '.join(bad)}", file=sys.stderr)
        return 1
    if not reviewers or ("cursor" in reviewers and not lenses):
        print("error: need at least one reviewer (and one lens when cursor is selected)", file=sys.stderr)
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

    if args.out:
        out = args.out.resolve()
        if out == root or root in out.parents:
            print("error: --out must be outside the repo", file=sys.stderr)
            return 1
        out.mkdir(parents=True, exist_ok=True)
    else:  # private, unpredictable dir: no pre-created folder or symlink can redirect writes
        parent = Path(tempfile.gettempdir()) / "rallyclip-review"
        parent.mkdir(mode=0o700, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9._-]", "-", branch)
        out = Path(tempfile.mkdtemp(prefix=f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}-", dir=parent))
    diff_path = out / "diff.patch"
    diff_path.write_text(diff, encoding="utf-8")
    ctx = {
        "diff_path": diff_path, "base": args.base, "merge_base": merge_base[:10],
        "head": head[:10], "files_changed": len(files), "constraints": hard_constraints(root),
    }

    before = tree_snapshot(root)
    meta: dict = {}
    jobs: dict[str, object] = {}
    with cf.ThreadPoolExecutor(max_workers=len(lenses) + 1) as pool:
        if "cursor" in reviewers:
            for lens in lenses:
                jobs[f"cursor:{lens}"] = pool.submit(run_cursor, root, out, ctx, lens, args.model, args.timeout)
        if "greptile" in reviewers:
            # Pass the ref as given: a local `main` can be far behind origin/main.
            jobs["greptile"] = pool.submit(run_greptile, root, out, args.base, args.timeout, meta)
        findings, failures = [], {}
        try:
            for name, fut in jobs.items():
                try:
                    findings += fut.result()
                except Exception as e:  # noqa: BLE001 - report every reviewer failure, keep the rest
                    failures[name] = f"{type(e).__name__}: {e}"
        except KeyboardInterrupt:
            # Reviewers run in their own sessions, so Ctrl-C does not reach them; kill them
            # here, before the pool's shutdown(wait=True) would block on their threads.
            for proc in list(_LIVE):
                kill_tree(proc)
            raise
    modified = tree_snapshot(root) != before

    findings.sort(key=lambda f: (SEVERITIES.index(f["severity"]), str(f["file"]), f["line"] or 0))
    for i, f in enumerate(findings, 1):
        f["id"] = i
    report = {
        "branch": branch, "base": args.base, "merge_base": merge_base, "head": head,
        "files_changed": len(files), "reviewers": list(jobs), "failures": failures,
        "tree_modified_by_reviewer": modified, **meta, "findings": findings,
    }
    (out / "findings.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    counts = {s: sum(f["severity"] == s for f in findings) for s in SEVERITIES}
    print(f"review {branch} vs {args.base} ({len(files)} files) -> {out / 'findings.json'}")
    conf = f", greptile confidence {meta['greptile_confidence']}/5" if "greptile_confidence" in meta else ""
    print(f"findings: {counts['high']} high, {counts['medium']} medium, {counts['low']} low{conf}")
    for f in findings:
        loc = f"{f['file']}:{f['line']}" if f["line"] else str(f["file"])
        claim = " ".join(f["claim"].split())[:160]
        print(f"  #{f['id']} [{f['severity']}] {f['reviewer']}/{f['lens']} {loc} - {claim}")
    for name, err in failures.items():
        print(f"FAILED {name}: {err}", file=sys.stderr)
    if modified:
        print("ERROR: a reviewer modified the working tree; inspect `git status` before continuing",
              file=sys.stderr)
        return 3
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
