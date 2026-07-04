"""Saved-match library storage resolution.

Pure path/contract logic for the on-disk library layout: one folder per saved
match containing ``source.mp4``, ``segments.csv``, ``meta.json``, and optional
thumbnail/export/cache files. No Flask, Qt, or heavy media imports.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

SOURCE_FILENAME = "source.mp4"
LEGACY_SOURCE_FILENAME = "video.mp4"
SEGMENTS_FILENAME = "segments.csv"
META_FILENAME = "meta.json"
THUMBNAIL_FILENAME = "thumb.jpg"
EXPORT_FILENAME = "export.mp4"


def new_item_id() -> str:
    """Sortable, unique library item id (timestamp + short random suffix)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


@dataclass(frozen=True)
class SavedMatchStore:
    """Resolves files inside a library root, rejecting ids that escape it."""

    root: Path

    def item_dir(self, item_id: str) -> Path:
        item_dir = (self.root / item_id).resolve()
        root = self.root.resolve()
        if root not in item_dir.parents:
            raise ValueError(f"Invalid library id: {item_id!r}")
        return item_dir

    def resolve_file(self, item_id: str, filename: str) -> Optional[Path]:
        """Resolve a file inside a library item, or None if the id/file is invalid."""
        try:
            item_dir = self.item_dir(item_id)
        except ValueError:
            return None
        path = item_dir / filename
        return path if path.exists() else None

    def resolve_source(self, item_id: str) -> Optional[Path]:
        """Resolve the full source video for a library item.

        ``source.mp4`` is the current storage contract. ``video.mp4`` is
        accepted only as a legacy fallback for items created before lazy
        export existed.
        """
        try:
            item_dir = self.item_dir(item_id)
        except ValueError:
            return None
        for filename in (SOURCE_FILENAME, LEGACY_SOURCE_FILENAME):
            path = item_dir / filename
            if path.exists():
                return path
        return None

    def read_meta(self, item_dir: Path) -> Dict[str, Any]:
        meta_path = item_dir / META_FILENAME
        if not meta_path.exists():
            return {}
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            logging.warning("Could not read library metadata from %s", meta_path, exc_info=True)
            return {}
        return meta if isinstance(meta, dict) else {}

    def list_items(self) -> List[Dict[str, Any]]:
        """List saved matches (newest first). An item needs meta.json + a source video."""
        items: List[Dict[str, Any]] = []
        if not self.root.exists():
            return items
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            meta_path = child / META_FILENAME
            has_source = (child / SOURCE_FILENAME).exists() or (child / LEGACY_SOURCE_FILENAME).exists()
            if not meta_path.exists() or not has_source:
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            meta["id"] = child.name  # trust the folder name, not the file contents
            meta["has_csv"] = (child / SEGMENTS_FILENAME).exists()
            meta["has_thumbnail"] = (child / THUMBNAIL_FILENAME).exists()
            meta["has_export"] = (child / EXPORT_FILENAME).exists()
            items.append(meta)
        items.sort(key=lambda m: m.get("created_ts", 0), reverse=True)
        return items
