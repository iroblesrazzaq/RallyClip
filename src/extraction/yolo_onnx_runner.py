"""Torch-free, Ultralytics-shaped YOLO pose runner on ONNX Runtime.

Drop-in for the exact `ultralytics.YOLO` surface RallyClip consumes:

    model = YOLO("yolov8n-pose-960-dynamic.onnx")
    results = model.predict(source=[bgr, ...], conf=0.25, imgsz=960,
                            device="cpu", batch=1, verbose=False)
    results[0].boxes.xyxy.detach().cpu().numpy()      # float32 [N, 4]
    results[0].boxes.conf.detach().cpu().numpy()      # float32 [N]
    results[0].keypoints.xy.detach().cpu().numpy()    # float32 [N, 17, 2]
    results[0].keypoints.conf.detach().cpu().numpy()  # float32 [N, 17]
    for box in results[0].boxes:                       # court detector path
        box.cls.item(); box.conf.item(); box.xyxy.cpu().numpy()

Semantics mirror Ultralytics `.predict()` on a .pt model: rect letterbox
(auto=True, stride 32, pad 114, INTER_LINEAR), confidence filter, greedy NMS
at iou=0.7, max_det=300, conf-descending output, boxes/keypoints unletterboxed
and clipped to the original frame. Defaults (conf=0.25, imgsz=640) match
Ultralytics predict defaults; production pose callers pass imgsz per call
from the model manifest. Validated bitwise (preprocessing) and to <=3.1e-4 px
(decoded outputs) against Ultralytics — see YOLO-ONNX/scripts/parity_v8n_960.py.

Only the v8-pose raw head layout ([1, 56, N]: 4 xywh + 1 conf + 17*3 kpts) is
supported; anything else (e.g. YOLO26 end-to-end [1, 300, 57] exports, detect
or segment heads) raises UnsupportedOnnxOutputShapeError instead of silently
mis-decoding.

Dependencies: numpy, cv2 (image ops only), onnxruntime. No torch, no
ultralytics.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

DEFAULT_CONF = 0.25    # ultralytics predict defaults
DEFAULT_IMGSZ = 640
DEFAULT_IOU = 0.7
DEFAULT_MAX_DET = 300
STRIDE = 32


class UnsupportedOnnxOutputShapeError(ValueError):
    """The ONNX model's output is not a raw v8-pose head ([*, 56, N])."""

    def __init__(self, shape) -> None:
        super().__init__(
            f"Unsupported ONNX pose output shape {tuple(shape)}: expected a raw "
            "YOLOv8-pose head [1, 56, N] (4 xywh + 1 conf + 17*3 keypoints). "
            "End-to-end exports with NMS in the graph (e.g. [1, 300, 57]) and "
            "non-pose models are not supported by this runner."
        )


class _Tensor:
    """Numpy array with the .detach().cpu().numpy()/.item() chain callers expect."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def detach(self) -> "_Tensor":
        return self

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array

    def item(self):
        return self._array.item()

    def __array__(self, dtype=None):
        return self._array if dtype is None else self._array.astype(dtype)


class _Box:
    """One detection, shaped like an element of ultralytics Results.boxes."""

    def __init__(self, xyxy_row: np.ndarray, conf: np.ndarray) -> None:
        self.xyxy = _Tensor(xyxy_row.reshape(1, 4))
        self.conf = _Tensor(conf.reshape(()))
        # v8-pose is single-class (person); the raw head has no class column.
        self.cls = _Tensor(np.zeros((), dtype=np.float32))


class _Boxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray) -> None:
        self.xyxy = _Tensor(xyxy)
        self.conf = _Tensor(conf)
        self._xyxy = xyxy
        self._conf = conf

    def __len__(self) -> int:
        return int(self._xyxy.shape[0])

    def __iter__(self):
        for i in range(len(self)):
            yield _Box(self._xyxy[i], self._conf[i])


class _Keypoints:
    def __init__(self, xy: np.ndarray, conf: np.ndarray) -> None:
        self.xy = _Tensor(xy)
        self.conf = _Tensor(conf)


class Result:
    def __init__(self, boxes_xyxy, boxes_conf, kpt_xy, kpt_conf) -> None:
        self.boxes = _Boxes(boxes_xyxy, boxes_conf)
        self.keypoints = _Keypoints(kpt_xy, kpt_conf)


def letterbox(img: np.ndarray, new_shape: int, stride: int = STRIDE):
    """Rect letterbox exactly as Ultralytics .pt predict (auto=True).

    Returns (NCHW float32 RGB in [0,1], ratio, (pad_left, pad_top)).
    """
    shape = img.shape[:2]
    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape - new_unpad[0]) % stride / 2
    dh = (new_shape - new_unpad[1]) % stride / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = img[..., ::-1].transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor, dtype=np.float32) / 255.0, r, (left, top)


def letterbox_exact(img: np.ndarray, target_hw: tuple[int, int]):
    """Letterbox onto an exact (H, W) canvas (Ultralytics auto=False).

    Same resize + center-pad math as `letterbox`, but pads out to the full
    target instead of the minimal stride-32 rectangle, as a static-shape ONNX
    input requires. For sources whose stride-32 rect letterbox already equals
    the target — 16:9 at imgsz 960 -> 544x960 — the pixels are identical; any
    other aspect stays geometrically correct via extra pad (and, when the
    source is taller than the target aspect, a smaller scale).
    """
    th, tw = target_hw
    shape = img.shape[:2]
    r = min(th / shape[0], tw / shape[1])
    new_unpad = (min(tw, int(round(shape[1] * r))), min(th, int(round(shape[0] * r))))
    dw = (tw - new_unpad[0]) / 2
    dh = (th - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = img[..., ::-1].transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor, dtype=np.float32) / 255.0, r, (left, top)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS matching torchvision.ops.nms (suppress IoU > thr)."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / np.maximum(area_i + area_o - inter, 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def decode_v8_pose(
    pred: np.ndarray,
    ratio: float,
    pad: tuple[int, int],
    orig_hw: tuple[int, int],
    conf_thr: float,
    iou_thr: float = DEFAULT_IOU,
    max_det: int = DEFAULT_MAX_DET,
):
    """Decode raw v8-pose head output [56, N] (or [N, 56] / [1, 56, N])."""
    orig_shape = pred.shape
    if pred.ndim == 3:
        if pred.shape[0] != 1:
            raise UnsupportedOnnxOutputShapeError(orig_shape)
        pred = pred[0]
    if pred.ndim != 2:
        raise UnsupportedOnnxOutputShapeError(orig_shape)
    if pred.shape[0] != 56:
        pred = pred.T
    if pred.shape[0] != 56:
        raise UnsupportedOnnxOutputShapeError(orig_shape)
    xywh = pred[:4].T
    conf = pred[4]
    kpts = pred[5:].T.reshape(-1, 17, 3)

    mask = conf > conf_thr
    xywh, conf, kpts = xywh[mask], conf[mask], kpts[mask]
    empty = (
        np.empty((0, 4), np.float32),
        np.empty((0,), np.float32),
        np.empty((0, 17, 2), np.float32),
        np.empty((0, 17), np.float32),
    )
    if xywh.shape[0] == 0:
        return empty

    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    keep = _nms(xyxy, conf, iou_thr)[:max_det]
    xyxy, conf, kpts = xyxy[keep], conf[keep], kpts[keep]

    h, w = orig_hw
    pad_w, pad_h = pad
    xyxy[:, [0, 2]] = np.clip((xyxy[:, [0, 2]] - pad_w) / ratio, 0, w)
    xyxy[:, [1, 3]] = np.clip((xyxy[:, [1, 3]] - pad_h) / ratio, 0, h)
    kpt_xy = kpts[:, :, :2].copy()
    kpt_xy[:, :, 0] = np.clip((kpt_xy[:, :, 0] - pad_w) / ratio, 0, w)
    kpt_xy[:, :, 1] = np.clip((kpt_xy[:, :, 1] - pad_h) / ratio, 0, h)
    return (
        xyxy.astype(np.float32),
        conf.astype(np.float32),
        kpt_xy.astype(np.float32),
        kpts[:, :, 2].astype(np.float32),
    )


class YOLO:
    """Ultralytics-call-compatible ONNX pose runner (predict subset only)."""

    def __init__(self, model_path: str, *, providers: Optional[list] = None,
                 intra_op_threads: Optional[int] = None) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if intra_op_threads is not None:
            opts.intra_op_num_threads = intra_op_threads
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts,
            providers=providers or ["CPUExecutionProvider"],
        )
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        # Static-shape exports (the CoreML-friendly ones — dynamic axes block
        # the ANE) fix their own input size; letterbox to that exact canvas
        # and ignore the caller's imgsz.
        shape = model_input.shape
        self._static_hw = (
            (int(shape[2]), int(shape[3]))
            if len(shape) == 4 and all(isinstance(d, int) for d in shape[2:])
            else None
        )

    def to(self, device) -> "YOLO":  # parity with ultralytics API; CPU EP only
        return self

    def predict(
        self,
        source,
        conf: float = DEFAULT_CONF,
        imgsz: int = DEFAULT_IMGSZ,
        iou: float = DEFAULT_IOU,
        max_det: int = DEFAULT_MAX_DET,
        device: Optional[str] = None,   # accepted, CPU EP only for now
        batch: Optional[int] = None,    # accepted; inference is per-frame
        verbose: bool = False,          # accepted for signature parity
        **_ignored,
    ) -> list[Result]:
        if isinstance(source, np.ndarray) and source.ndim == 3:
            source = [source]
        results = []
        for img in source:
            if self._static_hw is not None:
                tensor, ratio, pad = letterbox_exact(img, self._static_hw)
            else:
                tensor, ratio, pad = letterbox(img, int(imgsz))
            pred = self._session.run(None, {self._input_name: tensor})[0]
            boxes, bconf, kpt_xy, kpt_conf = decode_v8_pose(
                pred, ratio, pad, img.shape[:2], float(conf), float(iou), int(max_det)
            )
            results.append(Result(boxes, bconf, kpt_xy, kpt_conf))
        return results

    # PoseExtractor calls model.predict(...); some callers use model(...) too.
    __call__ = predict
