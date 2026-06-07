from __future__ import annotations

import sys
from importlib.resources import files
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


def _frontend_from_package() -> Path | None:
    try:
        candidate = Path(str(files("gui") / "frontend"))
        if candidate.is_dir() and (candidate / "index.html").exists():
            return candidate.resolve()
    except Exception:
        pass
    return None


def resolve_frontend_dir() -> Path:
    """Locate bundled GUI static assets."""
    if getattr(sys, "frozen", False):
        bundle_root = Path(getattr(sys, "_MEIPASS", Path.cwd()))
        bundled = bundle_root / "gui" / "frontend"
        if bundled.is_dir() and (bundled / "index.html").exists():
            return bundled.resolve()

    packaged = _frontend_from_package()
    if packaged is not None:
        return packaged

    rel_paths = (Path("gui/frontend"), Path("src/gui/frontend"))
    for rel in rel_paths:
        for root in _candidate_roots():
            candidate = root / rel
            if candidate.is_dir() and (candidate / "index.html").exists():
                return candidate.resolve()

    fallback = Path(__file__).resolve().parents[1] / "gui" / "frontend"
    return fallback.resolve()
