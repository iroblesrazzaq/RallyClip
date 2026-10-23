"""Middle-anchored clean-frame court detection for the training court cache.

The v0.6 training corpus's court masks were built with this clean frame, so the
training cache keeps it; the runtime CourtDetector keeps its own +/-15 s
reference search (and the court regression goldens frozen from it).

The base frame sits at an anchor (the middle of the video by default). Players
occluding it are masked, then neighbours are taken OUTWARD in
CLEAN_FRAME_STEP_S hops each way, homography-aligned to the base, and used to
repaint occluded pixels nobody is standing over in that neighbour. A side stops
once its homography IoU with the base drops below CLEAN_FRAME_MIN_IOU (a camera
cut or big pan/zoom) or the video ends. The walk ends early once nothing is
occluded and never reaches past CLEAN_FRAME_MAX_OFFSET_S.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from preprocessing.court_detector_impl import CourtDetector


class MiddleAnchoredCourtDetector(CourtDetector):
    CLEAN_FRAME_STEP_S = 10.0
    CLEAN_FRAME_MIN_IOU = 0.6
    CLEAN_FRAME_MAX_OFFSET_S = 180.0
    CLEAN_FRAME_MIN_INLIERS = 10

    def process_video(self, video_path: str, target_time: Optional[float] = None):
        """As CourtDetector.process_video; target_time=None anchors at the middle."""
        return super().process_video(video_path, target_time=target_time)

    def _detect_person_boxes(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Person boxes (x, y, w, h) above ``self.conf`` in one frame.

        The pose backend is single-class (person), so every detection is a person;
        the ``cls == 0`` guard keeps parity if a multi-class detector is ever wired
        in. Empty list when no model is loaded."""
        if self.yolo_model is None:
            return []
        results = self.yolo_model.predict(source=frame, verbose=False)[0]
        boxes: List[Tuple[int, int, int, int]] = []
        for box in getattr(results, "boxes", []):
            try:
                if int(box.cls.item()) != 0:
                    continue
                if float(box.conf.item()) <= self.conf:
                    continue
                x0, y0, x1, y1 = [int(v) for v in box.xyxy.cpu().numpy().reshape(-1)]
                boxes.append((x0, y0, x1 - x0, y1 - y0))
            except Exception:
                continue
        return boxes

    @staticmethod
    def _boxes_to_mask(boxes: List[Tuple[int, int, int, int]], shape: Tuple[int, int], dilate_px: int = 5) -> np.ndarray:
        """Filled 255-on-0 occlusion mask for a list of (x, y, w, h) boxes."""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        for x, y, w, h in boxes:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        if dilate_px:
            mask = cv2.dilate(mask, np.ones((dilate_px, dilate_px), np.uint8), iterations=1)
        return mask

    @staticmethod
    def _quad_iou(M: np.ndarray, ref_shape: Tuple[int, int], base_shape: Tuple[int, int]) -> float:
        """IoU of the reference image quad (warped into base coords by M) with the
        base rectangle. ~1.0 for a static camera, falling as it pans/zooms — the
        signal used to decide a neighbour is too far to trust as a reference."""
        try:
            rh, rw = ref_shape[:2]
            bh, bw = base_shape[:2]
            ref_quad = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(ref_quad, M).reshape(-1, 2).astype(np.float32)
            if not np.all(np.isfinite(warped)):
                return 0.0
            base_quad = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
            inter, _ = cv2.intersectConvexConvex(warped, base_quad)
            area_ref = abs(cv2.contourArea(warped))
            union = area_ref + float(bw * bh) - inter
            return float(inter / union) if union > 0 else 0.0
        except cv2.error:
            return 0.0

    def _homography_to_base(
        self,
        orb,
        kp_base,
        des_base,
        reference_frame: np.ndarray,
        base_shape: Tuple[int, int],
    ) -> Tuple[Optional[np.ndarray], float, int]:
        """RANSAC homography mapping ``reference_frame`` -> base, with its coverage
        IoU and inlier count. M is None when it can't be estimated reliably."""
        kp_ref, des_ref = orb.detectAndCompute(reference_frame, None)
        if des_ref is None or des_base is None:
            return None, 0.0, 0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des_ref, des_base), key=lambda m: m.distance)
        good = matches[: min(100, len(matches))]
        if len(good) < self.CLEAN_FRAME_MIN_INLIERS:
            return None, 0.0, 0
        ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        base_pts = np.float32([kp_base[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, inliers = cv2.findHomography(ref_pts, base_pts, cv2.RANSAC, 5.0)
        if M is None:
            return None, 0.0, 0
        n_inliers = int(inliers.sum()) if inliers is not None else 0
        if n_inliers < self.CLEAN_FRAME_MIN_INLIERS:
            return None, 0.0, n_inliers
        return M, self._quad_iou(M, reference_frame.shape, base_shape), n_inliers

    def _repair_occlusions(self, reader, base_frame, occlusion, target_time, duration):
        """Walk outward from ``target_time`` repainting occluded base pixels.

        Returns ``(clean, remaining, n_refs)``: the repainted frame, the mask of
        pixels still occluded, and how many neighbours contributed pixels.
        """
        clean = base_frame.copy()
        remaining = occlusion.copy()
        orb = cv2.ORB_create(nfeatures=1000)
        kp_base, des_base = orb.detectAndCompute(base_frame, None)
        if des_base is None:
            logging.info("No features in base frame; cannot repair occlusions")
            return clean, remaining, 0

        h, w = base_frame.shape[:2]
        fps, total_frames = reader.fps, reader.total_frames
        step = max(1.0, self.CLEAN_FRAME_STEP_S)
        max_off = min(self.CLEAN_FRAME_MAX_OFFSET_S, max(duration - target_time, target_time))
        stopped = {1: False, -1: False}
        n_refs = 0
        offset = step
        while offset <= max_off and not (stopped[1] and stopped[-1]):
            for sign in (1, -1):
                if stopped[sign]:
                    continue
                t = target_time + sign * offset
                if t < 0 or t * fps >= total_frames:
                    stopped[sign] = True
                    continue
                ref = reader.read_frame_at_index(int(t * fps))
                if ref is None:
                    stopped[sign] = True
                    continue
                M, iou, n_in = self._homography_to_base(orb, kp_base, des_base, ref, base_frame.shape)
                if M is None or iou < self.CLEAN_FRAME_MIN_IOU:
                    logging.debug("Clean-frame ref t=%.1fs rejected (iou=%.2f inliers=%s)", t, iou, n_in)
                    stopped[sign] = True
                    continue
                warped = cv2.warpPerspective(ref, M, (w, h))
                coverage = cv2.warpPerspective(np.full(ref.shape[:2], 255, np.uint8), M, (w, h))
                ref_occ_warped = cv2.warpPerspective(
                    self._boxes_to_mask(self._detect_person_boxes(ref), ref.shape), M, (w, h)
                )
                # Repaint pixels that are still occluded in the base, are covered
                # by this warped neighbour, and are NOT under a player in it.
                fillable = (remaining > 0) & (coverage > 0) & (ref_occ_warped == 0)
                if np.any(fillable):
                    clean[fillable] = warped[fillable]
                    remaining[fillable] = 0
                    n_refs += 1
                if not np.any(remaining):
                    return clean, remaining, n_refs
            offset += step
        return clean, remaining, n_refs

    def extract_clean_frame(self, video_path: str, target_time: Optional[float] = None) -> np.ndarray:
        """Player-free frame anchored at ``target_time`` (the middle when None)."""
        from runtime.video_frames import VideoFrameReader

        reader = VideoFrameReader(video_path)
        fps = reader.fps
        total_frames = reader.total_frames
        duration = (total_frames / fps) if fps else 0.0

        try:
            # Anchor at the middle unless a caller pins a time (the court
            # regression goldens do). Mid-match is the most reliable place to find
            # in-play footage with a settled camera.
            if target_time is None:
                target_time = duration / 2.0 if duration > 0 else 60.0

            if self.yolo_model is None:
                logging.info("YOLO not available, using single frame at target time")
                frame = reader.read_frame_at_index(int(fps * target_time))
                if frame is None:
                    raise RuntimeError("Could not read frame at target time.")
                return frame

            base_frame = reader.read_frame_at_index(int(target_time * fps))
            if base_frame is None:
                raise RuntimeError("Could not read base frame at target time.")

            occlusion = self._boxes_to_mask(self._detect_person_boxes(base_frame), base_frame.shape)
            total_occ = int(np.count_nonzero(occlusion))
            if total_occ == 0:
                logging.info("No players occlude the base frame at %.1fs; using it as-is", target_time)
                return base_frame

            clean, remaining, n_refs = self._repair_occlusions(
                reader, base_frame, occlusion, target_time, duration
            )
            repaired = total_occ - int(np.count_nonzero(remaining))
            logging.info(
                "Clean frame @%.1fs: repaired %d/%d occluded px from %d reference frame(s)%s",
                target_time, repaired, total_occ, n_refs,
                "" if not np.any(remaining) else " (residual occlusion remains)",
            )
            return clean.astype(np.uint8)
        finally:
            reader.close()
