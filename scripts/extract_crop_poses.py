"""Side-car pose extraction on the fixed far-player crop.

The far player is detected in ~0.4% of frames by the full-frame pass: a 1920-wide
frame letterboxes to imgsz=960 (scale 0.5), leaving them at ~80px, near YOLO's
floor for pose. Running the same model on a fixed 960x540 window over the far
half feeds them in at native scale -- 2x the pixels -- which recovers detections
the full-frame pass misses entirely.

This writes a PARALLEL artifact rather than extending the raw pose h5: the
full-frame extraction is expensive, already done, and fingerprint-validated, so
re-running it to bolt on a second pass would be wasteful and risky. The
side-car mirrors the raw schema exactly (CSR: frame_offsets indexes into flat
detection arrays) and is keyed by the same 5 fps frame grid, so preprocess can
join on frame index.

Boxes and keypoints are remapped back to FULL-FRAME coordinates, so every
downstream consumer works in one coordinate system.

Usage:
  python scripts/extract_crop_poses.py [--data-root ...] [--limit N] [--overwrite]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_ROOT = Path("/Users/ismaelrobles-razzaq/2_cs_projects/rallyclip_container/training_data_1080p")
NORM = "norm=1280x720@5fps"          # path tag inherited from the pipeline contract
YOLO_TAG = "yolo=yolov8n-960@4a3fe0de"
CONF_TAG, IMGSZ_TAG = "conf=0p25", "imgsz=960"

W, H = 1920, 1080
CROP = (480, 0, 1440, 540)           # fixed 16:9 top-half window, frame-centred
FPS = 5.0
IMGSZ, CONF = 960, 0.25
MERGE_IOU = 0.6                      # matches PlayerAssigner.merge_iou_thresh
# Same manifest-backed backend the full-frame extraction uses, so both passes
# run identical weights and decode. provider=coreml uses the static-shape
# sibling on the ANE; the crop is 960x540, which letterboxes to exactly the
# static 544x960 input, so it is a natural fit (and far faster than torch/MPS).
MANIFEST = REPO / "models/pose/yolov8n/manifest.json"
PROVIDER = "coreml"


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge_detections(boxes, box_conf, kps, kp_conf, thresh=MERGE_IOU):
    """Union detections that overlap above `thresh` (single-linkage).

    A player clipped by the crop boundary routinely yields several partial boxes
    that survive YOLO's own NMS; without this they consume both crop slots and
    evict the far player. The union box is kept, along with the keypoints of the
    highest-confidence member of the cluster.
    """
    n = len(boxes)
    if n <= 1:
        return boxes, box_conf, kps, kp_conf
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _iou(boxes[i], boxes[j]) > thresh:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    ob, oc, ok, okc = [], [], [], []
    for members in groups.values():
        bs = boxes[members]
        best = members[int(np.argmax(box_conf[members]))]
        ob.append([bs[:, 0].min(), bs[:, 1].min(), bs[:, 2].max(), bs[:, 3].max()])
        oc.append(float(box_conf[best]))
        ok.append(kps[best])
        okc.append(kp_conf[best])
    order = np.argsort(-np.asarray(oc))
    return (np.asarray(ob, np.float32)[order], np.asarray(oc, np.float32)[order],
            np.asarray(ok, np.float32)[order], np.asarray(okc, np.float32)[order])


def _arrays(result):
    """(boxes, box_conf, keypoints, keypoint_conf) as numpy, or empties.

    The ONNX runner returns ultralytics-shaped results whose tensors may be
    numpy or torch depending on backend, so unwrap defensively -- same approach
    as Hdf5PoseExtractor._extract_arrays.
    """
    def np_(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    empty = (np.empty((0, 4), np.float32), np.empty((0,), np.float32),
             np.empty((0, 17, 2), np.float32), np.empty((0, 17), np.float32))
    if result is None or getattr(result, "boxes", None) is None:
        return empty
    try:
        b = np_(result.boxes.xyxy).astype(np.float32)
        bc = np_(result.boxes.conf).astype(np.float32)
    except Exception:
        return empty
    if len(b) == 0:
        return empty
    try:
        k = np_(result.keypoints.xy).astype(np.float32)
        kc = np_(result.keypoints.conf).astype(np.float32)
    except Exception:
        k = np.zeros((len(b), 17, 2), np.float32)
        kc = np.zeros((len(b), 17), np.float32)
    return b, bc, k, kc


def crop_raw_path(root: Path, stem: str) -> Path:
    return (root / "pose_data" / NORM / "crop_raw" / YOLO_TAG / CONF_TAG / IMGSZ_TAG
            / f"{stem}__start0__durfull.h5")


def raw_path(root: Path, stem: str) -> Path:
    return (root / "pose_data" / NORM / "raw" / YOLO_TAG / CONF_TAG / IMGSZ_TAG
            / f"{stem}__start0__durfull.h5")


def expected_frames(root: Path, stem: str):
    """Frame grid from the full-frame raw h5, so the side-car aligns exactly."""
    p = raw_path(root, stem)
    if not p.exists():
        return None
    with h5py.File(p, "r") as h:
        return h["frames"]["frame_index"][:], h["frames"]["timestamps"][:]


def extract_one(model, video: Path, out: Path, frame_index, timestamps, model_tag: str = "") -> None:
    x1, y1, x2, y2 = CROP
    n_expected = len(frame_index)
    boxes_all, bconf_all, kps_all, kconf_all = [], [], [], []
    offsets = [0]

    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "s.mp4"
        # One sequential decode at the target rate. Per-frame POS_MSEC seeking
        # re-decodes from a keyframe each time and is ~100x slower.
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video), "-vf", f"fps={FPS:g},scale={W}:{H},"
             f"crop={x2-x1}:{y2-y1}:{x1}:{y1}", "-an", "-loglevel", "error", str(clip)],
            check=True,
        )
        cap = cv2.VideoCapture(str(clip))
        batch, seen = [], 0
        BATCH = 16

        def flush(frames):
            nonlocal seen
            if not frames:
                return
            for r in model.predict(source=frames, imgsz=IMGSZ, conf=CONF, verbose=False):
                b, bc, k, kc = _arrays(r)
                if len(b):
                    b, bc, k, kc = merge_detections(b, bc, k, kc)
                    # crop -> full-frame coordinates
                    b = b + np.array([x1, y1, x1, y1], np.float32)
                    k = k + np.array([x1, y1], np.float32)
                    boxes_all.append(b); bconf_all.append(bc)
                    kps_all.append(k); kconf_all.append(kc)
                    offsets.append(offsets[-1] + len(b))
                else:
                    offsets.append(offsets[-1])
                seen += 1

        while seen + len(batch) < n_expected:
            ok, fr = cap.read()
            if not ok:
                break
            batch.append(fr)
            if len(batch) >= BATCH:
                flush(batch); batch = []
        flush(batch)
        cap.release()

    # Pad if the decoder yielded fewer frames than the raw grid, so frame_index
    # stays aligned (a short tail becomes "no detections", never a shift).
    while len(offsets) - 1 < n_expected:
        offsets.append(offsets[-1])

    cat = lambda xs, shape, dt: (np.concatenate(xs).astype(dt) if xs
                                 else np.empty(shape, dt))  # noqa: E731
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.h5")
    with h5py.File(tmp, "w") as h:
        h.attrs["video_path"] = str(video)
        h.attrs["crop"] = np.array(CROP, np.int64)
        h.attrs["imgsz"] = IMGSZ
        h.attrs["conf"] = CONF
        h.attrs["merge_iou"] = MERGE_IOU
        h.attrs["yolo_model"] = model_tag
        h.attrs["coords"] = "full_frame"      # boxes/keypoints already remapped
        h.attrs["width"], h.attrs["height"] = W, H
        f = h.create_group("frames")
        f.create_dataset("frame_index", data=np.asarray(frame_index, np.int64))
        f.create_dataset("timestamps", data=np.asarray(timestamps, np.float64))
        f.create_dataset("frame_offsets", data=np.asarray(offsets, np.int64))
        d = h.create_group("detections")
        d.create_dataset("boxes", data=cat(boxes_all, (0, 4), np.float32))
        d.create_dataset("box_conf", data=cat(bconf_all, (0,), np.float32))
        d.create_dataset("keypoints", data=cat(kps_all, (0, 17, 2), np.float32))
        d.create_dataset("keypoint_conf", data=cat(kconf_all, (0, 17), np.float32))
    tmp.replace(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--videos", nargs="*", default=None)
    args = ap.parse_args()

    from extraction.pose_backend import load_pose_backend

    root = args.data_root
    stems = args.videos or sorted(p.name[:-9] for p in (root / "annotations").glob("*.mp4.json"))
    if args.limit:
        stems = stems[: args.limit]
    model, meta = load_pose_backend(MANIFEST, provider=PROVIDER)
    print(f"pose backend: {meta.tag}  head={meta.head_family}  imgsz={meta.imgsz}  "
          f"provider={PROVIDER}", flush=True)

    for i, stem in enumerate(stems, 1):
        out = crop_raw_path(root, stem)
        if out.exists() and not args.overwrite:
            print(f"[{i}/{len(stems)}] {stem[:12]} exists, skip", flush=True)
            continue
        video = root / "source_videos" / f"{stem}.mp4"
        if not video.exists():
            print(f"[{i}/{len(stems)}] {stem[:12]} MISSING VIDEO", flush=True)
            continue
        grid = expected_frames(root, stem)
        if grid is None:
            print(f"[{i}/{len(stems)}] {stem[:12]} no raw h5 to align to, skip", flush=True)
            continue
        fi, ts = grid
        extract_one(model, video, out, fi, ts, meta.tag)
        with h5py.File(out, "r") as h:
            nd = len(h["detections"]["boxes"])
            nf = len(h["frames"]["frame_index"])
            hit = float(np.mean(np.diff(h["frames"]["frame_offsets"][:]) > 0)) * 100
        print(f"[{i}/{len(stems)}] {stem[:12]} frames={nf} dets={nd} "
              f"frames_with_detection={hit:.1f}%", flush=True)


if __name__ == "__main__":
    main()
