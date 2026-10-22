"""scripts/agent/review.py: parsing, timeouts, retries and the main() exit contract (no reviewer CLIs run)."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("agent_review", _ROOT / "scripts" / "agent" / "review.py")
review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review)


def test_extract_findings_ignores_preamble_and_fences():
    text = 'I will review.\nBEGIN_FINDINGS\n```json\n[{"file": "a.py", "line": 3}]\n```\nEND_FINDINGS'
    assert review.extract_findings(text) == [{"file": "a.py", "line": 3}]


def test_extract_findings_takes_last_block_not_quoted_example():
    text = (
        "The prompt wants a BEGIN_FINDINGS/END_FINDINGS pair, like:\n"
        'BEGIN_FINDINGS\n[{"file": "src/path.py", "line": 123}]\nEND_FINDINGS\n'
        "Real answer:\n"
        'BEGIN_FINDINGS\n```JSON\n[{"file": "real.py", "line": 9}]\n```\nEND_FINDINGS\n'
    )
    assert review.extract_findings(text) == [{"file": "real.py", "line": 9}]


def test_extract_findings_empty_block():
    assert review.extract_findings("BEGIN_FINDINGS\n[]\nEND_FINDINGS") == []


@pytest.mark.parametrize("body", ["", "   \n", "```json\n```"])
def test_extract_findings_rejects_block_without_array(body):
    # Markers with no array is an unusable response, not "no findings".
    with pytest.raises(ValueError, match="empty findings block"):
        review.extract_findings(f"BEGIN_FINDINGS\n{body}\nEND_FINDINGS")


def test_extract_findings_requires_markers():
    with pytest.raises(ValueError):
        review.extract_findings('[{"file": "a.py"}]')


@pytest.mark.parametrize(
    "raw, expected",
    [("HIGH", "high"), ("critical", "high"), ("P2", "medium"), ("low", "low"), (None, "low"), ("nit", "low")],
)
def test_norm_severity(raw, expected):
    assert review.norm_severity(raw) == expected


@pytest.mark.parametrize("raw, expected", [(12, 12), ("12", 12), (None, None), ("", None), ("x", None)])
def test_norm_line(raw, expected):
    assert review.norm_line(raw) == expected


def test_greptile_comments_reads_v3_schema():
    data = {"summary": "s", "confidence": 3, "comments": [{"path": "b.py", "body": "bug"}, "junk"]}
    assert review.greptile_comments(data) == [{"path": "b.py", "body": "bug"}]


def test_greptile_comments_rejects_unknown_shape():
    with pytest.raises(ValueError):
        review.greptile_comments({"review": {"comments": []}})


def test_tree_snapshot_sees_untracked_and_ignored_edits(tmp_path):
    def run(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    run("init", "-q")
    (tmp_path / ".gitignore").write_text("*.onnx\n")
    (tmp_path / "a.py").write_text("x = 1\n")
    run("add", ".")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    (tmp_path / "notes.txt").write_text("one")
    (tmp_path / "w.onnx").write_text("weights")
    base = review.tree_snapshot(tmp_path)
    assert review.tree_snapshot(tmp_path) == base
    (tmp_path / "notes.txt").write_text("two")  # same name in `git status`, new content
    after_untracked = review.tree_snapshot(tmp_path)
    assert after_untracked != base
    (tmp_path / "w.onnx").write_text("weights-but-longer")  # ignored file
    after_ignored = review.tree_snapshot(tmp_path)
    assert after_ignored != after_untracked
    (tmp_path / "a.py").write_text("x = 2\n")  # tracked source edit
    assert review.tree_snapshot(tmp_path) != after_ignored


def test_main_rejects_empty_reviewer_selection(capsys):
    assert review.main(["--reviewers", ""]) == 1
    assert review.main(["--reviewers", "cursor", "--lenses", " "]) == 1


def test_hard_constraints_section_comes_from_agents_md():
    text = review.hard_constraints(_ROOT)
    assert "MUST NOT commit to `main`" in text
    assert "## Tier 2" not in text


def test_prompt_template_formats():
    keys = ("diff_path", "base", "merge_base", "head", "files_changed", "constraints")
    ctx = {k: f"<<{k}>>" for k in keys}
    prompt = review.PROMPT_FILE.read_text(encoding="utf-8").format(lens="<<lens>>", lens_desc="<<desc>>", **ctx)
    for value in [*ctx.values(), "<<lens>>", "<<desc>>"]:
        assert value in prompt, f"placeholder for {value} not substituted"
    assert "{diff_path}" not in prompt and "{constraints}" not in prompt
    assert "BEGIN_FINDINGS" in prompt and '"severity": "high"' in prompt


def test_run_cursor_tolerates_null_fields(monkeypatch, tmp_path):
    result = 'BEGIN_FINDINGS\n[{"file": "a.py", "line": "7", "claim": null, "failure_scenario": null}, "junk"]\nEND_FINDINGS'
    envelope = '{"type": "result", "is_error": false, "result": %s}' % json.dumps(result)

    calls = []

    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=envelope, stderr="")

    monkeypatch.setattr(review, "run_capped", fake_run)
    ctx = {k: "x" for k in ("diff_path", "base", "merge_base", "head", "files_changed", "constraints")}
    [f] = review.run_cursor(_ROOT, tmp_path, ctx, "tests", "m", 5)
    assert f == {"reviewer": "cursor:m", "lens": "tests", "file": "a.py", "line": 7,
                 "severity": "low", "claim": "", "failure_scenario": ""}
    [cmd] = calls
    assert cmd[:2] == ["agent", "-p"]
    assert cmd[cmd.index("--mode") + 1] == "ask"  # read-only: the reviewer must never get write mode
    assert cmd[cmd.index("--model") + 1] == "m"


def test_run_cursor_rejects_array_without_finding_objects(monkeypatch, tmp_path):
    result = 'BEGIN_FINDINGS\n["no issues found", 3]\nEND_FINDINGS'
    envelope = json.dumps({"type": "result", "is_error": False, "result": result})
    monkeypatch.setattr(review, "run_capped", lambda cmd, *a, **kw: subprocess.CompletedProcess(cmd, 0, envelope, ""))
    ctx = {k: "x" for k in ("diff_path", "base", "merge_base", "head", "files_changed", "constraints")}
    with pytest.raises(ValueError, match="no finding objects"):
        review.run_cursor(_ROOT, tmp_path, ctx, "tests", "m", 5)


def test_main_rejects_unknown_reviewer_or_lens(capsys):
    assert review.main(["--reviewers", "bugbot"]) == 1
    assert review.main(["--reviewers", "cursor", "--lenses", "style"]) == 1
    err = capsys.readouterr().err
    assert "unknown reviewer/lens: bugbot" in err and "unknown reviewer/lens: style" in err


def test_kill_tree_without_process_groups_kills_child(monkeypatch):
    monkeypatch.delattr(review.os, "killpg", raising=False)  # what Windows looks like
    killed = []

    class FakeProc:
        pid = 123

        def kill(self):
            killed.append(True)

    review.kill_tree(FakeProc())
    assert killed == [True]


def test_run_cursor_retries_once_on_retriable_error(monkeypatch, tmp_path):
    calls = []
    ok = '{"type": "result", "is_error": false, "result": "BEGIN_FINDINGS\\n[]\\nEND_FINDINGS"}'

    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        if len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="RetriableError: Connection stalled")
        return subprocess.CompletedProcess(cmd, 0, stdout=ok, stderr="")

    monkeypatch.setattr(review, "run_capped", fake_run)
    ctx = {k: "x" for k in ("diff_path", "base", "merge_base", "head", "files_changed", "constraints")}
    assert review.run_cursor(_ROOT, tmp_path, ctx, "tests", "m", 5) == []
    assert len(calls) == 2


@pytest.mark.parametrize(
    "stderr, expected_calls",
    [("boom: bad flag", 1), ("RetriableError: Connection stalled", 2)],  # no retry / retry capped at one
)
def test_run_cursor_retry_is_bounded(monkeypatch, tmp_path, stderr, expected_calls):
    calls = []

    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=stderr)

    monkeypatch.setattr(review, "run_capped", fake_run)
    ctx = {k: "x" for k in ("diff_path", "base", "merge_base", "head", "files_changed", "constraints")}
    with pytest.raises(RuntimeError, match="agent exited 1"):
        review.run_cursor(_ROOT, tmp_path, ctx, "tests", "m", 5)
    assert len(calls) == expected_calls


def test_run_greptile_maps_comment_fields(monkeypatch, tmp_path):
    data = {"confidence": 4, "comments": [
        {"path": "b.py", "startLine": "12", "severity": "P1", "category": "logic",
         "body": "off by one", "suggestion": "use <="},
        {"path": "c.py", "startLine": None, "severity": "P2", "body": None},
    ]}

    def fake_run(cmd, *a, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(data), stderr="")

    monkeypatch.setattr(review, "run_capped", fake_run)
    meta = {}
    first, second = review.run_greptile(_ROOT, tmp_path, "origin/main", 5, meta)
    assert meta == {"greptile_confidence": 4}
    assert first == {"reviewer": "greptile", "lens": "logic", "file": "b.py", "line": 12,
                     "severity": "high", "claim": "off by one", "failure_scenario": "use <="}
    assert (second["lens"], second["line"], second["severity"], second["claim"]) == ("general", None, "medium", "")


@pytest.fixture
def branch_repo(tmp_path, monkeypatch):
    """A git repo whose HEAD is one commit ahead of a local `base` branch; cwd set to it."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def run(*args):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
                       cwd=repo, check=True, capture_output=True)

    run("init", "-q")
    (repo / "AGENTS.md").write_text("## Hard constraints\n\n- be good\n\n## Tier 2\n")
    run("add", ".")
    run("commit", "-qm", "init")
    run("branch", "base")
    (repo / "a.py").write_text("x = 1\n")
    run("add", ".")
    run("commit", "-qm", "change")
    monkeypatch.chdir(repo)
    monkeypatch.setattr(review.shutil, "which", lambda exe: f"/usr/bin/{exe}")
    return repo


def _finding(file="a.py"):
    return {"reviewer": "greptile", "lens": "logic", "file": file, "line": 1,
            "severity": "medium", "claim": "c", "failure_scenario": "f"}


def test_main_exit_0_writes_findings(branch_repo, tmp_path, monkeypatch):
    monkeypatch.setattr(review, "run_greptile", lambda *a: [_finding()])
    out = tmp_path / "out"
    assert review.main(["--base", "base", "--reviewers", "greptile", "--out", str(out)]) == 0
    report = json.loads((out / "findings.json").read_text())
    assert report["failures"] == {} and report["tree_modified_by_reviewer"] is False
    assert [f["id"] for f in report["findings"]] == [1]


def test_main_exit_2_keeps_partial_results(branch_repo, tmp_path, monkeypatch):
    def boom(*a):
        raise RuntimeError("agent exited 1")

    monkeypatch.setattr(review, "run_cursor", boom)
    monkeypatch.setattr(review, "run_greptile", lambda *a: [_finding()])
    out = tmp_path / "out"
    argv = ["--base", "base", "--lenses", "tests", "--out", str(out)]
    assert review.main(argv) == 2
    report = json.loads((out / "findings.json").read_text())
    assert set(report["failures"]) == {"cursor:tests"} and len(report["findings"]) == 1


def test_main_exit_3_when_reviewer_edits_tree(branch_repo, tmp_path, monkeypatch):
    def edits(*a):
        (branch_repo / "a.py").write_text("x = 'reviewer was here'\n")
        return []

    monkeypatch.setattr(review, "run_greptile", edits)
    out = tmp_path / "out"
    assert review.main(["--base", "base", "--reviewers", "greptile", "--out", str(out)]) == 3
    assert json.loads((out / "findings.json").read_text())["tree_modified_by_reviewer"] is True


def test_main_refuses_tracked_edits_unless_allowed(branch_repo, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(review, "run_greptile", lambda *a: [])
    (branch_repo / "a.py").write_text("x = 'uncommitted'\n")
    argv = ["--base", "base", "--reviewers", "greptile", "--out", str(tmp_path / "out")]
    assert review.main(argv) == 1
    assert "uncommitted changes to tracked files" in capsys.readouterr().err
    assert review.main([*argv, "--allow-dirty"]) == 0


def test_main_rejects_out_inside_repo(branch_repo, monkeypatch, capsys):
    monkeypatch.setattr(review, "run_greptile", lambda *a: pytest.fail("reviewer must not run"))
    argv = ["--base", "base", "--reviewers", "greptile", "--out", str(branch_repo / "review-out")]
    assert review.main(argv) == 1
    assert "--out must be outside the repo" in capsys.readouterr().err
    assert not (branch_repo / "review-out").exists()


def test_default_out_dir_is_private_and_directly_in_tmp():
    out = review.default_out_dir("feat/x y")
    try:
        assert out.parent == Path(tempfile.gettempdir())  # no shared, pre-creatable parent
        assert out.name.startswith("rallyclip-review-feat-x-y-")
        if os.name == "posix":
            assert stat.S_IMODE(out.stat().st_mode) == 0o700
    finally:
        out.rmdir()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups; no `sh` on Windows runners")
def test_stop_reviewers_kills_running_and_blocks_new(tmp_path):
    errors = []

    def worker():
        try:
            review.run_capped(["sh", "-c", "sleep 30 & sleep 30"], tmp_path, 60)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=worker)
    start = time.monotonic()
    t.start()
    try:
        while not review._LIVE and time.monotonic() - start < 5:
            time.sleep(0.01)
        assert review._LIVE, "reviewer process never registered"
        review.stop_reviewers()
        t.join(10)
        assert not t.is_alive() and time.monotonic() - start < 10
        with pytest.raises(RuntimeError, match="review cancelled"):  # a retry after the stop
            review.run_capped(["sh", "-c", "true"], tmp_path, 5)
    finally:
        review._STOP.clear()
        t.join(30)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups; no `sh` on Windows runners")
def test_run_capped_kills_grandchildren_on_timeout(tmp_path):
    # `agent` is a bash wrapper whose node child holds the pipes; plain
    # subprocess.run(timeout=) waits for that child, so the cap never fires.
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        review.run_capped(["sh", "-c", "sleep 30 & sleep 30"], tmp_path, 1)
    assert time.monotonic() - start < 10
