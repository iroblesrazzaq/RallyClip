from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .contracts import Interval, PlaybackManifest
from .intervals import point_duration


@dataclass(frozen=True)
class PlaybackPoint:
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class PlaybackSegment:
    start_ms: int
    end_ms: int
    point_index: Optional[int]
    next_point_index: Optional[int]
    mode: str


class SourceTimelineScheduler:
    """Pure source-time point-skip scheduler.

    Rendering stays platform-specific; this class only decides what source-time
    interval a player should play and where it should jump next.
    """

    def __init__(self, intervals: Iterable[dict[str, float] | Interval], duration_s: float | None):
        self.duration_ms = max(0, int(round(float(duration_s or 0) * 1000)))
        points: List[PlaybackPoint] = []
        for interval in intervals:
            if isinstance(interval, dict):
                start_s = float(interval.get("start", 0))
                end_s = float(interval.get("end", 0))
            else:
                start_s, end_s = interval
            start_ms = max(0, int(round(float(start_s) * 1000)))
            end_ms = max(start_ms, int(round(float(end_s) * 1000)))
            if end_ms > start_ms:
                points.append(PlaybackPoint(start_ms, end_ms))
        self.points = sorted(points, key=lambda point: (point.start_ms, point.end_ms))
        if self.points and self.duration_ms <= 0:
            self.duration_ms = max(point.end_ms for point in self.points)
        self.active_segment: Optional[PlaybackSegment] = None

    def default_start_ms(self) -> int:
        return self.points[0].start_ms if self.points else 0

    def clamp_ms(self, value_ms: int | float) -> int:
        value = max(0, int(round(float(value_ms))))
        if self.duration_ms > 0:
            return min(value, self.duration_ms)
        return value

    def classify(self, value_ms: int | float) -> PlaybackSegment:
        position_ms = self.clamp_ms(value_ms)
        if not self.points:
            return PlaybackSegment(
                start_ms=position_ms,
                end_ms=self.duration_ms,
                point_index=None,
                next_point_index=None,
                mode="continuous",
            )
        for index, point in enumerate(self.points):
            next_index = index + 1 if index + 1 < len(self.points) else None
            if position_ms < point.start_ms:
                return PlaybackSegment(
                    start_ms=position_ms,
                    end_ms=point.end_ms,
                    point_index=index,
                    next_point_index=next_index,
                    mode="gap_bridge",
                )
            if point.start_ms <= position_ms < point.end_ms:
                return PlaybackSegment(
                    start_ms=position_ms,
                    end_ms=point.end_ms,
                    point_index=index,
                    next_point_index=next_index,
                    mode="point",
                )
        return PlaybackSegment(
            start_ms=position_ms,
            end_ms=self.duration_ms,
            point_index=None,
            next_point_index=None,
            mode="tail",
        )

    def seek(self, value_ms: int | float) -> PlaybackSegment:
        self.active_segment = self.classify(value_ms)
        return self.active_segment

    def next_start_after_active(self) -> Optional[int]:
        if self.active_segment is None or self.active_segment.next_point_index is None:
            return None
        try:
            return self.points[self.active_segment.next_point_index].start_ms
        except IndexError:
            return None

    def tail_start_after_active(self) -> Optional[int]:
        if self.active_segment is None:
            return None
        if self.active_segment.next_point_index is not None:
            return None
        if self.active_segment.mode not in {"point", "gap_bridge"}:
            return None
        if self.duration_ms <= self.active_segment.end_ms:
            return None
        return self.active_segment.end_ms

    def should_advance(self, position_ms: int | float, tolerance_ms: int = 80) -> bool:
        if self.active_segment is None:
            return False
        return int(round(float(position_ms))) >= max(0, self.active_segment.end_ms - tolerance_ms)


def build_playback_manifest(
    *,
    source_duration_s: float,
    chunk_duration_s: float,
    point_intervals: Iterable[Interval],
) -> PlaybackManifest:
    return PlaybackManifest(
        source_duration_s=float(source_duration_s),
        chunk_duration_s=float(chunk_duration_s),
        point_intervals=sorted(
            [(float(start), float(end)) for start, end in point_intervals if float(end) > float(start)],
            key=lambda item: (item[0], item[1]),
        ),
    )


def playback_manifest_payload(manifest: PlaybackManifest) -> dict[str, object]:
    intervals = [{"start": start, "end": end} for start, end in manifest.point_intervals]
    return {
        "source_duration_s": round(manifest.source_duration_s, 3) if manifest.source_duration_s > 0 else None,
        "chunk_duration_s": manifest.chunk_duration_s,
        "segments": intervals,
        "point_intervals": intervals,
        "point_duration_s": round(point_duration(manifest.point_intervals), 3),
    }
