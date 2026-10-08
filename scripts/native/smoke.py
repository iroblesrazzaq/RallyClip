"""Exercise the packaged backend and native AVFoundation player/exporter.

All data lives in a disposable directory. The user's saved matches are never
opened. Requires a completed scripts/native/build.sh build.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import secrets
import subprocess
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


def verify_audio(probe: Path, root: Path) -> dict:
    """Check silence and sync across a cut into a clip with delayed audio."""
    import av
    import numpy as np

    paths = [root / "tone-a.mp4", root / "tone-b.mp4"]
    rate = 48000
    for index, path in enumerate(paths):
        with av.open(str(path), "w") as container:
            video = container.add_stream("libx264", rate=30)
            video.width, video.height, video.pix_fmt = 320, 240, "yuv420p"
            audio = container.add_stream("aac", rate=rate)
            audio.layout = "mono"
            for frame_index in range(90):
                frame = av.VideoFrame.from_ndarray(np.full((240, 320, 3), 50 + index * 150, dtype=np.uint8), format="rgb24")
                for packet in video.encode(frame):
                    container.mux(packet)
            offset = rate if index else 0
            for sample in range(offset, rate * 3, 1024):
                count = min(1024, rate * 3 - sample)
                values = (np.sin(np.arange(sample, sample + count) * (440 + index * 440) * 2 * np.pi / rate) * 0.3).astype(np.float32)
                frame = av.AudioFrame.from_ndarray(values.reshape(1, -1), format="fltp", layout="mono")
                frame.sample_rate, frame.pts, frame.time_base = rate, sample, Fraction(1, rate)
                for packet in audio.encode(frame):
                    container.mux(packet)
            for stream in (video, audio):
                for packet in stream.encode():
                    container.mux(packet)
    descriptor = root / "audio-playback.json"
    descriptor.write_text(json.dumps({"sources": [
        {"path": str(path), "start_s": index * 3, "duration_s": 3} for index, path in enumerate(paths)
    ], "segments": [{"start": 2.5, "end": 3.8}, {"start": 4.2, "end": 5.5}]}))
    destination = root / "audio-export.mp4"
    report = json.loads(subprocess.check_output([str(probe.resolve()), str(descriptor), str(destination)], text=True, timeout=90))
    samples = np.zeros(rate * 3, dtype=np.float32)
    with av.open(str(destination)) as container:
        video, audio = container.streams.video[0], container.streams.audio[0]
        assert abs(float(video.duration * video.time_base) - 2.6) < 0.05
        assert abs(float(audio.duration * audio.time_base) - 2.6) < 0.05
        for frame in container.decode(audio):
            assert frame.sample_rate == rate
            values = frame.to_ndarray().mean(axis=0)
            start = round(float(frame.pts * frame.time_base) * rate)
            lo, hi = max(0, start), min(len(samples), start + len(values))
            if hi > lo:
                samples[lo:hi] = values[lo - start:hi - start]
    def rms(start, end):
        return float(np.sqrt(np.mean(samples[round(start * rate):round(end * rate)] ** 2)))
    assert rms(0.1, 0.35) > 0.1, "First clip's audio disappeared"
    assert rms(0.65, 1.15) < 0.01, "Late audio moved ahead of its source timeline"
    assert rms(1.6, 2.3) > 0.1, "Second clip's audio disappeared"
    return {"audio_sync": "passed", "late_audio_silence": "passed", "media": report}


def run(backend: Path, probe: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="rallyclip-native-smoke-") as raw:
        root = Path(raw)
        token = secrets.token_hex(32)
        ready = root / "ready.json"
        env = dict(os.environ, RALLYCLIP_NATIVE_TOKEN=token, RALLYCLIP_NATIVE_READY_FILE=str(ready),
                   RALLYCLIP_JOBS_DIR=str(root / "jobs"), RALLYCLIP_LIBRARY_DIR=str(root / "library"),
                   RALLYCLIP_OUTPUT_DIR=str(root / "output"), RALLYCLIP_CSV_DIR=str(root / "csv"),
                   RALLYCLIP_LOG_DIR=str(root / "logs"))
        with (root / "backend.log").open("w+") as log:
            process = subprocess.Popen([str(backend.resolve())], env=env, cwd=root, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 60
                while not ready.exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        raise RuntimeError("Packaged backend failed to boot:\n" + log.read()[-6000:])
                    time.sleep(0.1)
                port = json.loads(ready.read_text())["port"]
                base = f"http://127.0.0.1:{port}"

                def request(path, method="GET", body=None):
                    req = Request(base + path, method=method,
                                  data=json.dumps(body).encode() if body is not None else None,
                                  headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
                    with urlopen(req, timeout=120) as response:
                        return json.load(response)

                try:
                    urlopen(base + "/api/library", timeout=5)
                    raise AssertionError("Unauthenticated native API was accessible")
                except HTTPError as exc:
                    assert exc.code == 401
                assert request("/api/health")["status"] == "ok"
                clip = ROOT / "tests/fixtures/golden_cli/clip.mp4"
                job_id = request("/api/native/import", "POST", {"paths": [str(clip)], "name": "Native smoke"})["job_id"]
                deadline = time.monotonic() + 360
                while True:
                    status = request(f"/api/progress/{job_id}")
                    if status["status"] != "in_progress":
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("Native analysis did not finish")
                    time.sleep(0.5)
                if status["status"] != "completed":
                    log.seek(0)
                    raise AssertionError(f"{status}\n{log.read()[-6000:]}")
                item_id = status.get("library_id")
                assert item_id, "Golden match must create a saved library item"
                original = request(f"/api/library/{item_id}/segments")
                # Force three boundaries, including a gap, through the actual
                # native player regardless of the model's number of predictions.
                points = [{"start": 2.0, "end": 3.2}, {"start": 8.0, "end": 9.3}, {"start": 16.0, "end": 17.5}]
                edited = request(f"/api/library/{item_id}/segments", "PUT", {"segments": points})
                assert edited["edited"] and edited["segments"] == points
                sources = request(f"/api/native/library/{item_id}/sources")["sources"]
                descriptor = root / "playback.json"
                descriptor.write_text(json.dumps({"sources": sources, "segments": points}))
                exported = root / "native-export.mp4"
                report = subprocess.check_output([str(probe.resolve()), str(descriptor), str(exported)], text=True, timeout=90)
                media = json.loads(report)
                assert abs(media["export_duration"] - 4.0) < 0.1, media
                assert media["longest_observed_stall"] < 0.5, media
                restored = request(f"/api/library/{item_id}/segments/edits", "DELETE")
                assert restored["segments"] == original["segments"] and not restored["edited"]
                request(f"/api/library/{item_id}", "DELETE")
                assert not request("/api/library")["items"]
                audio_report = verify_audio(probe, root)
                print(json.dumps({"analysis": "passed", "edit_reset_delete": "passed", "native_media": media, "audio": audio_report}, indent=2))
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=ROOT / "build/native-swift/arm64-apple-macosx/release/RallyClipProbe")
    args = parser.parse_args()
    run(args.backend, args.probe)
