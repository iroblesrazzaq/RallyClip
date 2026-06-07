from __future__ import annotations

from pathlib import Path


def _candidate_roots() -> list[Path]:
    here = Path(__file__).resolve()
    roots = [Path.cwd()]
    for depth in (2, 3, 4, 5):
        try:
            roots.append(here.parents[depth])
        except IndexError:
            continue
    seen: list[Path] = []
    for root in roots:
        if root not in seen:
            seen.append(root)
    return seen


def resolve_frontend_dir() -> Path:
    """Locate bundled GUI static assets."""
    rel = Path("gui/frontend")
    for root in _candidate_roots():
        candidate = root / rel
        if candidate.is_dir() and (candidate / "index.html").exists():
            return candidate.resolve()
    fallback = Path(__file__).resolve().parents[2] / "gui" / "frontend"
    return fallback.resolve()
