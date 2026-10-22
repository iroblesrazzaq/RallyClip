"""Parsing helpers of scripts/agent/review.py (no reviewer CLIs are invoked)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("agent_review", _ROOT / "scripts" / "agent" / "review.py")
review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review)


def test_extract_findings_ignores_preamble_and_fences():
    text = 'I will review.\nBEGIN_FINDINGS\n```json\n[{"file": "a.py", "line": 3}]\n```\nEND_FINDINGS'
    assert review.extract_findings(text) == [{"file": "a.py", "line": 3}]


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


def test_greptile_items_finds_nested_list():
    data = {"review": {"summary": "x", "comments": [{"path": "b.py", "body": "bug"}, "junk"]}}
    assert review._greptile_items(data) == [{"path": "b.py", "body": "bug"}]


def test_hard_constraints_section_comes_from_agents_md():
    text = review.hard_constraints(_ROOT)
    assert "MUST NOT commit to `main`" in text
    assert "## Tier 2" not in text


def test_prompt_template_formats():
    ctx = {k: "x" for k in ("diff_path", "base", "merge_base", "head", "files_changed", "constraints")}
    prompt = review.PROMPT_FILE.read_text(encoding="utf-8").format(lens="tests", lens_desc="d", **ctx)
    assert "BEGIN_FINDINGS" in prompt and '"severity": "high"' in prompt
