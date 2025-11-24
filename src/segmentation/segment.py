import os
from typing import List, Tuple

import av
import pandas as pd
import cv2


def load_intervals(csv_path: str) -> List[Tuple[float, float]]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'start_time' not in df.columns or 'end_time' not in df.columns:
        raise ValueError("CSV must contain 'start_time' and 'end_time' columns")
    df = df[['start_time', 'end_time']].dropna()
    df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_numeric(df['end_time'], errors='coerce')
    df = df.dropna().sort_values(['start_time', 'end_time']).reset_index(drop=True)
    return [(float(r.start_time), float(r.end_time)) for r in df.itertuples(index=False)]


def segment_video(input_video: str, intervals: List[Tuple[float, float]], output_path: str, eps: float = 1e-6) -> None:
    if not intervals:
        raise ValueError("No intervals provided")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    in_container = av.open(input_video)
    in_stream = next(s for s in in_container.streams if s.type == 'video')
    time_base = in_stream.time_base
    avg_rate = in_stream.average_rate or 30
    width = in_stream.codec_context.width
    height = in_stream.codec_context.height
    fps_out = float(avg_rate) if hasattr(avg_rate, '__float__') else float(avg_rate.numerator / avg_rate.denominator) if hasattr(avg_rate, 'numerator') else 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps_out, (width, height))
    if not out_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    kept_frames = 0
    idx = 0
    n = len(intervals)
    try:
        for frame in in_container.decode(in_stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * time_base)
            while idx < n and t > (intervals[idx][1] + eps):
                idx += 1
            if idx >= n:
                break
            start_t, end_t = intervals[idx]
            if (t + eps) >= start_t and (t - eps) <= end_t:
                kept_frames += 1
                bgr_frame = frame.to_ndarray(format='bgr24')
                out_writer.write(bgr_frame)
    finally:
        out_writer.release()
        in_container.close()


