"""Unit tests for the saved-match library store in rallyclip_core."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rallyclip_api import RallyClipServices
from rallyclip_core.library import SavedMatchStore, new_item_id


def _make_item(
    root: Path,
    item_id: str,
    *,
    created_ts: float = 0.0,
    with_csv: bool = True,
    source_name: str = "source.mp4",
) -> Path:
    item_dir = root / item_id
    item_dir.mkdir(parents=True)
    (item_dir / source_name).write_bytes(b"\x00")
    (item_dir / "meta.json").write_text(
        json.dumps({"id": "spoofed", "name": item_id, "created_ts": created_ts}),
        encoding="utf-8",
    )
    if with_csv:
        (item_dir / "segments.csv").write_text(
            "start_time,end_time\n1.0,2.0\n", encoding="utf-8"
        )
    return item_dir


def test_new_item_id_is_sortable_and_unique():
    first, second = new_item_id(), new_item_id()
    assert first != second
    assert len(first.rsplit("-", 1)[-1]) == 6


def test_item_dir_rejects_path_traversal(tmp_path):
    store = SavedMatchStore(root=tmp_path)
    with pytest.raises(ValueError):
        store.item_dir("../../etc/passwd")
    assert store.item_dir("ok-123").parent == tmp_path.resolve()


def test_resolve_file_returns_none_for_bad_id_or_missing_file(tmp_path):
    store = SavedMatchStore(root=tmp_path)
    _make_item(tmp_path, "item-a", with_csv=False)
    assert store.resolve_file("../escape", "segments.csv") is None
    assert store.resolve_file("item-a", "segments.csv") is None
    assert store.resolve_file("item-a", "meta.json") == tmp_path.resolve() / "item-a" / "meta.json"


def test_resolve_source_prefers_current_contract_over_legacy(tmp_path):
    store = SavedMatchStore(root=tmp_path)
    item_dir = _make_item(tmp_path, "item-a")
    (item_dir / "video.mp4").write_bytes(b"\x00")
    resolved = store.resolve_source("item-a")
    assert resolved is not None and resolved.name == "source.mp4"

    _make_item(tmp_path, "item-legacy", source_name="video.mp4")
    legacy = store.resolve_source("item-legacy")
    assert legacy is not None and legacy.name == "video.mp4"

    assert store.resolve_source("missing") is None
    assert store.resolve_source("../escape") is None


def test_list_items_newest_first_with_flags(tmp_path):
    store = SavedMatchStore(root=tmp_path)
    _make_item(tmp_path, "older", created_ts=100.0, with_csv=False)
    _make_item(tmp_path, "newer", created_ts=200.0)
    # Items without meta or source are skipped.
    (tmp_path / "no-meta").mkdir()
    (tmp_path / "no-meta" / "source.mp4").write_bytes(b"\x00")
    (tmp_path / "stray-file").write_text("not a dir", encoding="utf-8")

    items = store.list_items()
    assert [item["id"] for item in items] == ["newer", "older"]
    assert items[0]["has_csv"] is True
    assert items[1]["has_csv"] is False
    assert all(item["id"] != "spoofed" for item in items)  # folder name wins


def test_list_items_empty_when_root_missing(tmp_path):
    store = SavedMatchStore(root=tmp_path / "nonexistent")
    assert store.list_items() == []


def test_read_meta_tolerates_missing_and_invalid_json(tmp_path):
    store = SavedMatchStore(root=tmp_path)
    item_dir = tmp_path / "item"
    item_dir.mkdir()
    assert store.read_meta(item_dir) == {}
    (item_dir / "meta.json").write_text("{not json", encoding="utf-8")
    assert store.read_meta(item_dir) == {}
    (item_dir / "meta.json").write_text('["not a dict"]', encoding="utf-8")
    assert store.read_meta(item_dir) == {}


def test_services_list_library_falls_back_to_store(tmp_path):
    _make_item(tmp_path, "item-a", created_ts=1.0)
    services = RallyClipServices(
        defaults_provider=dict,
        runtime_status_provider=dict,
        runtime_warmup=lambda: None,
        saved_match_store=SavedMatchStore(root=tmp_path),
    )
    payload = services.list_library()
    assert [item["id"] for item in payload["items"]] == ["item-a"]
