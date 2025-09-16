#!/usr/bin/env python3
"""
Execute video segmentation based on annotation intervals.

Given an input MP4 and a CSV of intervals (columns: start_time,end_time),
this script iterates frames using PyAV and writes a new video containing only
frames whose timestamps fall within any annotated interval. Output is saved
to output_videos/ by default.
"""

import os
import argparse
from typing import List, Tuple

import av
import pandas as pd
import cv2


def load_intervals(csv_path: str) -> List[Tuple[float, float]]:
    """Load ordered [start_time, end_time] intervals (in seconds) from CSV."""
    df = pd.read_csv(csv_path)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    if 'start_time' not in df.columns or 'end_time' not in df.columns:
        raise ValueError("CSV must contain 'start_time' and 'end_time' columns")
    df = df[['start_time', 'end_time']].dropna()
    # Ensure numeric and ordered
    df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_numeric(df['end_time'], errors='coerce')
    df = df.dropna().sort_values(['start_time', 'end_time']).reset_index(drop=True)
    intervals = [(float(r.start_time), float(r.end_time)) for r in df.itertuples(index=False)]
    return intervals


def segment_video(
    input_video: str,
    intervals: List[Tuple[float, float]],
    output_path: str,
    eps: float = 1e-6,
) -> None:
    """Write a new video containing only frames within any of the intervals."""
    if not intervals:
        raise ValueError("No intervals provided")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Open input container and retrieve stream info
    in_container = av.open(input_video)
    in_stream = next(s for s in in_container.streams if s.type == 'video')
    time_base = in_stream.time_base
    avg_rate = in_stream.average_rate or 30
    width = in_stream.codec_context.width
    height = in_stream.codec_context.height

    # Use input video's frame rate for output to maintain original speed
    fps_out = float(avg_rate) if hasattr(avg_rate, '__float__') else float(avg_rate.numerator / avg_rate.denominator) if hasattr(avg_rate, 'numerator') else 30.0
    
    # Use OpenCV VideoWriter with original frame rate
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps_out, (width, height))
    
    if not out_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    kept_frames = 0
    total_frames = 0
    idx = 0
    n = len(intervals)

    try:
        for frame in in_container.decode(in_stream):
            total_frames += 1
            if frame.pts is None:
                continue
            t = float(frame.pts * time_base)

            # Advance current interval if we've passed its end
            while idx < n and t > (intervals[idx][1] + eps):
                idx += 1
            if idx >= n:
                break  # no more intervals

            start_t, end_t = intervals[idx]
            if (t + eps) >= start_t and (t - eps) <= end_t:
                # Keep this frame
                kept_frames += 1
                # Convert frame to BGR format for OpenCV
                bgr_frame = frame.to_ndarray(format='bgr24')
                out_writer.write(bgr_frame)

    finally:
        out_writer.release()
        in_container.close()

    print(f"Input frames: {total_frames}, Kept: {kept_frames}")
    print(f"✓ Segmented video written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Segment a video using start/end times from a CSV")
    parser.add_argument('--video', required=True, help='Path to input MP4 video')
    parser.add_argument('--annotations', required=True, help='Path to CSV with start_time,end_time')
    parser.add_argument('--output-dir', default='output_videos', help='Directory to save segmented video')
    parser.add_argument('--output-name', default=None, help='Optional output filename (e.g., video_segmented.mp4)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output if it exists')
    parser.add_argument('--eps', type=float, default=1e-6, help='Boundary tolerance in seconds')

    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.exists(args.annotations):
        raise FileNotFoundError(f"Annotations not found: {args.annotations}")

    intervals = load_intervals(args.annotations)

    base_name = os.path.splitext(os.path.basename(args.video))[0]
    output_name = args.output_name or f"{base_name}_segmented.mp4"
    output_path = os.path.join(args.output_dir, output_name)

    if os.path.exists(output_path) and not args.overwrite:
        print(f"✓ Output exists, skipping (use --overwrite to replace): {output_path}")
        return 0

    segment_video(args.video, intervals, output_path, eps=args.eps)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


