"""Runtime v2 path: far-court crop pass, court line decode, net-line slots, v2 features."""

from types import SimpleNamespace

import numpy as np
import pytest

from extraction.crop_pass import crop_frame, crop_pixels, crop_to_frame, merge_detections
from preprocessing import court_lines
from preprocessing.court_lines import COURT_M, CourtLineModel, compute_court_geometry, decode, sample_frames
from preprocessing.data_preprocessor import DataPreprocessor
from preprocessing.netline_slots import CourtGeometry
from rallyclip_core.contracts import UnsupportedPipelineError
from rallyclip_engine.models import _iter_v2_features, build_feature_stream

# Straight-on synthetic court in 1920x1080 reference pixels: x = 960 + 50u,
# y = 600 - 20v (court meters), so the net (v = 0) sits at y = 600.
PX_TO_M = np.linalg.inv(np.array([[50.0, 0.0, 960.0], [0.0, -20.0, 600.0], [0.0, 0.0, 1.0]]))
GEOM = CourtGeometry(np.array([686.0, 600.0]), np.array([1234.0, 600.0]), PX_TO_M, "test")


def box_at(u, v, h=120.0):
    x, y = 960 + 50 * u, 600 - 20 * v
    return np.array([x - 25, y - h, x + 25, y], dtype=np.float32)


def detections(boxes, scale=1.0):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4) * scale
    n = len(boxes)
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    kps = np.repeat(np.stack([cx, cy], 1)[:, None, :], 17, axis=1)
    return {
        "boxes": boxes,
        "box_conf": np.full(n, 0.9, np.float32),
        "keypoints": kps.astype(np.float32),
        "conf": np.full((n, 17), 0.8, np.float32),
    }


# --- crop pass -------------------------------------------------------------------
def test_crop_window_is_the_training_crop_on_1080p():
    assert crop_pixels(1920, 1080) == (480, 0, 1440, 540)
    assert crop_frame(np.zeros((1080, 1920, 3), np.uint8)).shape == (540, 960, 3)


def test_crop_to_frame_maps_crop_pixels_back():
    boxes, kps = crop_to_frame(np.array([[0, 0, 480, 270]], np.float32), np.array([[[240, 135]]], np.float32),
                               width=3840, height=2160)
    np.testing.assert_allclose(boxes, [[960, 0, 1920, 540]])
    np.testing.assert_allclose(kps, [[[1440, 270]]])


def test_merge_detections_unions_overlapping_fragments():
    boxes = np.array([[0, 0, 10, 20], [0, 1, 10, 21], [50, 50, 60, 70]], np.float32)
    conf = np.array([0.5, 0.9, 0.7], np.float32)
    kps = np.arange(3 * 17 * 2, dtype=np.float32).reshape(3, 17, 2)
    kc = np.ones((3, 17), np.float32)
    ob, oc, ok, _ = merge_detections(boxes, conf, kps, kc)
    np.testing.assert_allclose(ob, [[0, 0, 10, 21], [50, 50, 60, 70]])
    np.testing.assert_allclose(oc, [0.9, 0.7])
    np.testing.assert_allclose(ok[0], kps[1])  # keypoints of the most confident member


# --- court line decode -----------------------------------------------------------
def test_decode_intersects_fitted_lines():
    probs = np.zeros((5, 72, 128), np.float32)
    probs[0, 60, 10:118] = 1.0      # near baseline
    probs[1, 20, 30:98] = 1.0       # far baseline
    probs[2, 15:66, 20] = 1.0       # left sideline
    probs[3, 15:66, 100] = 1.0      # right sideline
    probs[4, 40, 15:113] = 1.0      # net
    line_ok, points, point_ok = decode(probs)
    assert line_ok.all() and point_ok.all()
    # Pixel centers are at +0.5; points are normalized by the map size.
    expected = np.array([[20.5, 60.5], [100.5, 60.5], [20.5, 20.5], [100.5, 20.5], [20.5, 40.5], [100.5, 40.5]])
    np.testing.assert_allclose(points * [128, 72], expected, atol=1e-6)


def test_decode_reports_missing_line():
    probs = np.zeros((5, 72, 128), np.float32)
    probs[2, 10:60, 20] = 1.0
    probs[3, 10:60, 100] = 1.0
    line_ok, _points, point_ok = decode(probs)
    assert line_ok.tolist() == [False, False, True, True, False]
    assert not point_ok.any()


# --- slots + features ------------------------------------------------------------
def _pre():
    return DataPreprocessor(screen_width=1920, screen_height=1080)


def test_netline_slots_take_near_below_net_and_far_from_crop():
    near, far, spectator = box_at(0, -9), box_at(1, 11, h=50), box_at(-1, 3)
    frame = detections([near, spectator])
    frame["crop"] = detections([far])
    mask = np.zeros((1080, 1920), np.uint8)  # 0 = on court
    [(status, players)] = list(_pre().iter_slot_players([frame], mask, 1920, 1080, GEOM))
    assert status == 0
    np.testing.assert_allclose(players["near"]["box"], near)
    np.testing.assert_allclose(players["far"]["box"], far)


def test_slots_run_in_reference_pixels_for_other_resolutions():
    near, far = box_at(0, -9), box_at(1, 11, h=50)
    frame = detections([near], scale=2 / 3)  # 1280x720 source
    frame["crop"] = detections([far], scale=2 / 3)
    mask = np.zeros((720, 1280), np.uint8)
    [(_, players)] = list(_pre().iter_slot_players([frame], mask, 1280, 720, GEOM))
    np.testing.assert_allclose(players["near"]["box"], near, atol=1e-3)
    np.testing.assert_allclose(players["far"]["box"], far, atol=1e-3)


def test_classic_slots_without_geometry_and_skipped_frames():
    frame = detections([box_at(0, -9), box_at(1, 11, h=50)])
    mask = np.zeros((1080, 1920), np.uint8)
    stream = [frame, {"annotation_status": -1}]
    (status, players), skipped = list(_pre().iter_slot_players(stream, mask, 1920, 1080, None))
    assert status == 0 and players["near"] is not None
    assert skipped == (-1, None)


def test_v2_features_are_290_wide_and_drop_skipped_frames():
    frame = detections([box_at(0, -9)])
    frame["crop"] = detections([box_at(1, 11, h=50)])
    mask = np.zeros((1080, 1920), np.uint8)
    stream = _pre().iter_slot_players([frame, {"annotation_status": -1}, frame], mask, 1920, 1080, GEOM)
    out = list(_iter_v2_features(stream, "v2_nearfar", {}, 1920, 1080, 5.0))
    assert len(out) == 2
    assert all(vec.shape == (290,) and np.isfinite(vec).all() for vec, _ in out)
    # The crop player fills the far half: without it only the near half matches.
    no_far = {k: v for k, v in frame.items() if k != "crop"}
    stream = _pre().iter_slot_players([no_far, {"annotation_status": -1}, no_far], mask, 1920, 1080, GEOM)
    empty = list(_iter_v2_features(stream, "v2_nearfar", {}, 1920, 1080, 5.0))
    for (vec, _), (vec_empty, _) in zip(out, empty):
        np.testing.assert_array_equal(vec[:145], vec_empty[:145])
        assert not np.array_equal(vec[145:], vec_empty[145:])


def test_v2_features_honour_manifest_drops():
    frame = detections([box_at(0, -9)])
    mask = np.zeros((1080, 1920), np.uint8)
    stream = _pre().iter_slot_players([frame], mask, 1920, 1080, GEOM)
    [(vec, _)] = list(_iter_v2_features(stream, "v2_nearfar", {"drop_slots": ["far"]}, 1920, 1080, 5.0))
    assert vec.shape == (145,)


def test_runtime_rejects_a_feature_set_with_slots_it_does_not_fill(tmp_path):
    # v2 has crop0/crop1 slots; the runtime only fills near/far, so those would be zeros.
    request = SimpleNamespace(feature_set="v2", manifest_path=None, model_path=str(tmp_path / "model.onnx"))
    with pytest.raises(UnsupportedPipelineError, match="crop0"):
        build_feature_stream(request, None, None, None)


def test_sample_frames_keeps_the_first_frame_at_each_seek_target(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "court.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.options = {"qscale": "1", "g": "4"}
        for i in range(20):
            img = np.full((48, 64, 3), 10 * i, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)

    frames = sample_frames(str(path), (0.0, 0.5, 0.75))

    # 2 s at 10 fps: targets 0.0, 1.0 and 1.5 s -> frames 0, 10 and 15, not the keyframe before.
    assert [round(float(f.mean()) / 10) for f in frames] == [0, 10, 15]


def test_court_geometry_is_none_when_the_video_cannot_be_read(tmp_path):
    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"not a video")
    # A court failure must fall back to classic slots, not abort the run.
    assert compute_court_geometry(str(bad), str(tmp_path / "missing.onnx")) is None


def _court_px(uv):
    return np.array([960 + 50 * uv[0], 600 - 20 * uv[1]])


def test_court_scales_to_out_size_and_replaces_a_far_off_net_point(monkeypatch):
    # decode() returns the six points as fractions of the line map; court() owns the rest.
    points = np.array([_court_px(uv) / (1920, 1080) for uv in COURT_M])
    points[4, 1] += 0.1  # net_left 108 px off the corner-implied net at 1080p
    monkeypatch.setattr(court_lines, "decode", lambda probs: (np.ones(5, bool), points.copy(), np.ones(6, bool)))
    model = CourtLineModel.__new__(CourtLineModel)
    model.line_probs = lambda frames: None
    frames_4k = [np.zeros((2160, 3840, 3), np.uint8)]

    out = model.court(frames_4k, out_size=(1920, 1080))

    assert (out["width"], out["height"]) == (1920, 1080)
    np.testing.assert_allclose(out["points_px"]["near_left"], _court_px(COURT_M[0]), atol=0.1)
    assert out["net_source"] == ["corners", "model"]
    np.testing.assert_allclose(out["net_line_px"], [_court_px(COURT_M[4]), _court_px(COURT_M[5])], atol=0.01)
    hm = np.array(out["homography_px_to_m"])
    far_right = hm @ np.array([*_court_px(COURT_M[3]), 1.0])
    np.testing.assert_allclose(far_right[:2] / far_right[2], COURT_M[3], atol=1e-3)
    # Without out_size the points are in the frames' own pixels.
    np.testing.assert_allclose(model.court(frames_4k)["net_line_px"][1], 2 * _court_px(COURT_M[5]), atol=0.01)


def test_netline_manifest_wires_court_geometry_and_crop_pass(tmp_path, monkeypatch):
    import json

    (tmp_path / "court.onnx").write_bytes(b"")
    crop_window = [0.2, 0.0, 0.8, 0.5]
    (tmp_path / "manifest.json").write_text(json.dumps({"feature_pipeline": {
        "slot_mode": "netline", "court_model_file": "court.onnx", "crop_window": crop_window}}))
    hm = PX_TO_M.tolist()
    seen = {}

    class FakeCourt:
        frame_size = (1920, 1080)

        def geometry(self, out_size=None):
            seen["court_size"] = out_size
            return {"net_line_px": [[686.0, 600.0], [1234.0, 600.0]], "homography_px_to_m": hm}

    def fake_detect(video_path, model_path, providers=None):
        seen["court"] = model_path
        return FakeCourt()

    monkeypatch.setattr(court_lines, "detect_court", fake_detect)

    class Pre(DataPreprocessor):
        def __init__(self, **kwargs):
            super().__init__(screen_width=1920, screen_height=1080)

        def compute_court_mask(self, video_path, court=None):
            seen["mask_court"] = court
            return np.zeros((1080, 1920), np.uint8), None

        def _source_frame_shape(self, video_path):
            return 1080, 1920, 3

        def iter_slot_players(self, pose_data, court_mask, src_width, src_height, geometry=None):
            seen["geometry"] = geometry
            return super().iter_slot_players(pose_data, court_mask, src_width, src_height, geometry)

    class Extractor:
        def __init__(self, **kwargs):
            pass

        def iter_pose_frames(self, **kwargs):
            seen["crop_window"] = kwargs.get("crop_window")
            frame = detections([box_at(0, -9)])
            if kwargs.get("crop_window"):
                frame["crop"] = detections([box_at(1, 11, h=50)])
            yield frame

    request = SimpleNamespace(
        feature_set="v2_nearfar", manifest_path=str(tmp_path / "manifest.json"), model_path=str(tmp_path / "m.onnx"),
        yolo_weights=str(tmp_path / "pose.onnx"), yolo_device=None, models_dir=str(tmp_path), screen_width=1920,
        screen_height=1080, conf=0.25, video_path=str(tmp_path / "match.mp4"), start_time=0, duration=0, fps=5,
        imgsz=960)
    deps = SimpleNamespace(apply_pose_device=None, DataPreprocessor=Pre, PoseExtractor=Extractor, FeatureEngineer=None)

    [(vec, _)] = list(build_feature_stream(request, deps, None, None))

    assert seen["court"] == str(tmp_path / "court.onnx") and seen["court_size"] == (1920, 1080)
    assert seen["mask_court"] is not None  # one court pass feeds both the mask and the net line
    assert isinstance(seen["geometry"], CourtGeometry)
    np.testing.assert_allclose(seen["geometry"].px_to_m, PX_TO_M)
    assert seen["crop_window"] == tuple(crop_window)
    # The far slot comes from the crop pass, so the far half is a real player.
    stream = _pre().iter_slot_players([detections([box_at(0, -9)])], np.zeros((1080, 1920), np.uint8), 1920, 1080, GEOM)
    [(no_far, _)] = list(_iter_v2_features(stream, "v2_nearfar", {}, 1920, 1080, 5.0))
    assert vec.shape == (290,)
    np.testing.assert_array_equal(vec[:145], no_far[:145])
    assert not np.array_equal(vec[145:], no_far[145:])
