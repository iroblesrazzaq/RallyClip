from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("av")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from make_smoke_clip import make_clip  # noqa: E402

from preprocessing.data_preprocessor import rescale_player_to_reference  # noqa: E402
from runtime.video_validation import VideoValidationError, probe_video, validate_video  # noqa: E402

SEQ_LEN, FPS = 100, 5.0  # -> 20s minimum


# --------------------------------------------------------------------------- #
# validate_video
# --------------------------------------------------------------------------- #
def test_valid_720p_clip_passes(tmp_path):
    clip = tmp_path / "ok.mp4"
    make_clip(clip, duration_s=30.0)  # 1280x720, 30s
    info = validate_video(clip, seq_len=SEQ_LEN, fps=FPS)
    assert (info.width, info.height) == (1280, 720)
    assert info.duration_s >= 20.0


def test_unreadable_file_rejected(tmp_path):
    bad = tmp_path / "notvideo.mp4"
    bad.write_text("this is not a video", encoding="utf-8")
    with pytest.raises(VideoValidationError) as ei:
        validate_video(bad, seq_len=SEQ_LEN, fps=FPS)
    assert "could not be opened" in str(ei.value).lower()


def test_probe_video_wraps_metadata_errors(tmp_path, monkeypatch):
    import runtime.video_validation as vv

    class BadStream:
        type = "video"

        @property
        def codec_context(self):
            raise RuntimeError("codec metadata unavailable")

    class BadContainer:
        streams = [BadStream()]
        duration = None
        closed = False

        def close(self):
            self.closed = True

    container = BadContainer()
    clip = tmp_path / "bad-meta.mp4"
    clip.write_bytes(b"fake")
    monkeypatch.setattr(vv.av, "open", lambda _: container)

    with pytest.raises(VideoValidationError) as ei:
        probe_video(clip)

    assert "could not be read" in str(ei.value).lower()
    assert container.closed


def test_sub_720p_rejected(tmp_path):
    clip = tmp_path / "small.mp4"
    make_clip(clip, duration_s=30.0, width=640, height=360)
    with pytest.raises(VideoValidationError) as ei:
        validate_video(clip, seq_len=SEQ_LEN, fps=FPS)
    assert "720p" in str(ei.value)


def test_too_short_rejected(tmp_path):
    clip = tmp_path / "short.mp4"
    make_clip(clip, duration_s=5.0)  # 720p but < 20s
    with pytest.raises(VideoValidationError) as ei:
        validate_video(clip, seq_len=SEQ_LEN, fps=FPS)
    assert "too short" in str(ei.value).lower()


# --------------------------------------------------------------------------- #
# rescale_player_to_reference (resolution normalization)
# --------------------------------------------------------------------------- #
def _player(box, kp_val=100.0):
    return {
        "box": np.array(box, dtype=np.float32),
        "keypoints": np.full((17, 2), kp_val, dtype=np.float32),
        "conf": np.ones(17, dtype=np.float32),
        "box_conf": 0.9,
    }


def test_rescale_identity_at_reference():
    p = _player([100, 200, 300, 400])
    out = rescale_player_to_reference(p, 1280, 720, 1280, 720)
    assert out is p  # 720p input is a no-op


def test_rescale_none_passthrough():
    assert rescale_player_to_reference(None, 1920, 1080, 1280, 720) is None


def test_rescale_1080p_to_reference():
    # 16:9 source -> uniform 0.6667x scale into 1280x720 space.
    p = _player([192, 108, 960, 540], kp_val=192.0)
    out = rescale_player_to_reference(p, 1920, 1080, 1280, 720)
    assert out["box"][0] == pytest.approx(128.0, abs=1e-3)
    assert out["box"][1] == pytest.approx(72.0, abs=1e-3)
    assert out["box"][2] == pytest.approx(640.0, abs=1e-3)
    assert out["box"][3] == pytest.approx(360.0, abs=1e-3)
    assert out["keypoints"][0, 0] == pytest.approx(128.0, abs=1e-3)
    assert out["keypoints"][0, 1] == pytest.approx(128.0, abs=1e-3)
    # untouched fields preserved
    assert out["box_conf"] == 0.9
