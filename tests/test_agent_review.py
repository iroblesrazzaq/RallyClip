"""Parsing helpers of scripts/agent/review.py (no reviewer CLIs are invoked)."""

from __future__ import annotations

import importlib.util
import subprocess
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
    assert review.tree_snapshot(tmp_path) != after_untracked


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
    envelope = '{"type": "result", "is_error": false, "result": %s}' % __import__("json").dumps(result)

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=envelope, stderr="")

    monkeypatch.setattr(review.subprocess, "run", fake_run)
    ctx = {k: "x" for k in ("diff_path", "base", "merge_base", "head", "files_changed", "constraints")}
    [f] = review.run_cursor(_ROOT, tmp_path, ctx, "tests", "m", 5)
    assert (f["claim"], f["failure_scenario"], f["line"]) == ("", "", 7)


def test_run_cursor_retries_once_on_retriable_error(monkeypatch, tmp_path):
    calls = []
    ok = '{"type": "result", "is_error": false, "result": "BEGIN_FINDINGS\\n[]\\nEND_FINDINGS"}'

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="RetriableError: Connection stalled")
        return subprocess.CompletedProcess(cmd, 0, stdout=ok, stderr="")

    monkeypatch.setattr(review.subprocess, "run", fake_run)
    ctx = {k: "x" for k in ("diff_path", "base", "merge_base", "head", "files_changed", "constraints")}
    assert review.run_cursor(_ROOT, tmp_path, ctx, "tests", "m", 5) == []
    assert len(calls) == 2
