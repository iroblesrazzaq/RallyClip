from __future__ import annotations

import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Optional

try:  # pragma: no cover - availability depends on packaged runtime
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


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


class NativePlaybackScheduler:
    """Source-time point-skip scheduler for native playback."""

    def __init__(self, intervals: list[dict[str, float]] | list[tuple[float, float]], duration_s: float | None):
        self.duration_ms = max(0, int(round(float(duration_s or 0) * 1000)))
        points: list[PlaybackPoint] = []
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


def native_watchdog_reload_reason(
    *,
    playing: bool,
    position_ms: int,
    last_position_ms: int,
    seconds_since_frame: float,
    rss_mb: Optional[float],
    last_rss_mb: Optional[float],
) -> Optional[str]:
    if not playing:
        return None
    position_advanced = position_ms > last_position_ms + 250
    if position_advanced and seconds_since_frame > 5.0:
        return "video frames stopped while playback position advanced"
    if (
        rss_mb is not None
        and last_rss_mb is not None
        and rss_mb > 700.0
        and rss_mb > last_rss_mb + 10.0
    ):
        return f"memory rose to {rss_mb:.1f} MB"
    return None


def native_initial_media_for_descriptor(descriptor: dict[str, Any]) -> tuple[str, str]:
    proxy = descriptor.get("proxy") or {}
    proxy_path = proxy.get("path")
    if proxy.get("ready") and proxy_path:
        return "proxy", str(proxy_path)
    return "source", str(descriptor["source_path"])


def native_overlay_should_show(*, window_active: bool) -> bool:
    return bool(window_active)


def _native_playback_logger() -> logging.Logger:
    logger = logging.getLogger("rallyclip.native_playback")
    if getattr(logger, "_rallyclip_file_configured", False):
        return logger
    logger.setLevel(logging.INFO)
    try:
        log_dir = Path.home() / "RallyClip" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_dir / "native_playback.log",
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        setattr(logger, "_rallyclip_file_configured", True)
        logger.info("event=native_playback_log_ready path=%s", log_dir / "native_playback.log")
    except Exception:
        logger.debug("Could not configure native playback file logging.", exc_info=True)
        setattr(logger, "_rallyclip_file_configured", True)
    return logger


try:
    from PySide6.QtCore import (
        QEasingCurve,
        QEvent,
        QPointF,
        QPropertyAnimation,
        QRectF,
        QObject,
        Qt,
        QThread,
        QTimer,
        QUrl,
        Signal,
        Slot,
    )
    from PySide6.QtGui import QColor, QCursor, QPainter, QPen, QPolygonF
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
    from PySide6.QtWidgets import (
        QFileDialog,
        QFrame,
        QGridLayout,
        QGraphicsOpacityEffect,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )

    QT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on non-desktop installs
    QT_AVAILABLE = False


if QT_AVAILABLE:

    class PointTimelineSlider(QSlider):
        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(Qt.Orientation.Horizontal, parent)
            self._points: list[PlaybackPoint] = []
            self._duration_ms = 0
            self.setMinimumHeight(30)
            self.setMouseTracking(True)
            self.setStyleSheet("background: transparent;")

        def set_points(self, points: list[PlaybackPoint], duration_ms: int) -> None:
            self._points = points
            self._duration_ms = max(0, duration_ms)
            self.update()

        def paintEvent(self, event) -> None:  # type: ignore[override]
            del event
            duration_ms = max(self._duration_ms, self.maximum())
            if duration_ms <= 0:
                return

            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)

            pad = 8
            width = max(1, self.width() - (pad * 2))
            track_h = 6
            track_y = (self.height() - track_h) / 2
            track_radius = track_h / 2

            painter.setBrush(QColor(255, 255, 255, 48))
            painter.drawRoundedRect(pad, track_y, width, track_h, track_radius, track_radius)

            ratio = max(0.0, min(1.0, float(self.value()) / float(duration_ms)))
            played_width = width * ratio
            if played_width > 0:
                painter.setBrush(QColor(255, 255, 255, 210))
                painter.drawRoundedRect(pad, track_y, played_width, track_h, track_radius, track_radius)

            for point in self._points:
                left = pad + width * max(0, min(duration_ms, point.start_ms)) / duration_ms
                right = pad + width * max(0, min(duration_ms, point.end_ms)) / duration_ms
                point_width = max(2.0, right - left)
                painter.setBrush(QColor(0, 0, 0, 150))
                painter.drawRoundedRect(
                    left - 1.0,
                    track_y - 1.0,
                    point_width + 2.0,
                    track_h + 2.0,
                    track_radius + 1.0,
                    track_radius + 1.0,
                )
                painter.setBrush(QColor("#62d6a8"))
                painter.drawRoundedRect(left, track_y, point_width, track_h, track_radius, track_radius)

            cx = pad + width * ratio
            cy = self.height() / 2
            painter.setBrush(QColor(255, 255, 255, 34))
            painter.drawEllipse(cx - 15, cy - 15, 30, 30)
            painter.setBrush(QColor(255, 255, 255))
            painter.drawEllipse(cx - 8.0, cy - 8.0, 16, 16)
            painter.end()

        def mousePressEvent(self, event) -> None:  # type: ignore[override]
            self._set_value_from_event(event)
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
            if event.buttons() & Qt.MouseButton.LeftButton:
                self._set_value_from_event(event)
            super().mouseMoveEvent(event)

        def _set_value_from_event(self, event) -> None:
            max_value = self.maximum()
            if max_value <= 0:
                return
            pad = 8
            width = max(1, self.width() - (pad * 2))
            ratio = max(0.0, min(1.0, (event.position().x() - pad) / width))
            self.setValue(int(round(max_value * ratio)))


    class MediaControlButton(QPushButton):
        def __init__(self, icon_name: str, parent: Optional[QWidget] = None) -> None:
            super().__init__("", parent)
            self._icon_name = icon_name
            self.setFixedSize(48, 48)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        def set_icon(self, icon_name: str) -> None:
            if self._icon_name == icon_name:
                return
            self._icon_name = icon_name
            self.update()

        def paintEvent(self, event) -> None:  # type: ignore[override]
            del event
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            bounds = QRectF(3, 3, self.width() - 6, self.height() - 6)
            fill = QColor(12, 16, 24, 212)
            painter.setBrush(fill)
            painter.setPen(QPen(QColor(255, 255, 255, 44), 1))
            painter.drawEllipse(bounds)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255))

            cx = self.width() / 2
            cy = self.height() / 2
            if self._icon_name == "pause":
                bar_w = 5.0
                bar_h = 18.0
                gap = 4.8
                painter.drawRoundedRect(
                    QRectF(cx - gap - bar_w, cy - (bar_h / 2), bar_w, bar_h),
                    1.5,
                    1.5,
                )
                painter.drawRoundedRect(
                    QRectF(cx + gap, cy - (bar_h / 2), bar_w, bar_h),
                    1.5,
                    1.5,
                )
            else:
                triangle = QPolygonF(
                    [
                        QPointF(cx - 7, cy - 12),
                        QPointF(cx - 7, cy + 12),
                        QPointF(cx + 12, cy),
                    ]
                )
                painter.drawPolygon(triangle)
            painter.end()


    class ProxyWorker(QObject):
        finished = Signal(dict, str)

        def __init__(self, item_id: str) -> None:
            super().__init__()
            self.item_id = item_id

        @Slot()
        def run(self) -> None:
            try:
                from gui.app import ensure_native_playback_proxy

                self.finished.emit(ensure_native_playback_proxy(self.item_id), "")
            except Exception as exc:  # pragma: no cover - depends on ffmpeg/media failures
                self.finished.emit({}, str(exc))


    class NativeViewerBridge(QObject):
        def __init__(self, open_match: Callable[[str], bool]) -> None:
            super().__init__()
            self._open_match = open_match

        @Slot(str, result=bool)
        def openMatch(self, item_id: str) -> bool:
            try:
                return bool(self._open_match(str(item_id)))
            except Exception:
                return False


    class NativeViewerWidget(QWidget):
        backRequested = Signal()
        fullscreenRequested = Signal(bool)

        def __init__(self, port: int, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self.port = port
            self.item_id: Optional[str] = None
            self.descriptor: dict[str, Any] = {}
            self.scheduler = NativePlaybackScheduler([], 0)
            self._seeking = False
            self._pending_seek_ms: Optional[int] = None
            self._pending_autoplay = True
            self._last_autoplay_requested = True
            self._media_kind = "source"
            self._active_media_path: Optional[str] = None
            self._proxy_thread: Optional[QThread] = None
            self._proxy_worker: Optional[ProxyWorker] = None
            self._fullscreen = False
            self._media_ready = False
            self._pending_ready_attempts = 0
            self._last_overlay_show_monotonic = 0.0
            self._watchdog_reload_count = 0
            self._last_watchdog_position_ms = 0
            self._last_watchdog_rss_mb: Optional[float] = None
            self._last_frame_monotonic = time.monotonic()
            self._last_position_change_monotonic = time.monotonic()
            self._memory_process = psutil.Process(os.getpid()) if psutil is not None else None
            self.video_shell: Optional[QWidget] = None
            self.video_overlay: Optional[QWidget] = None
            self.control_tray: Optional[QFrame] = None
            self.title_overlay: Optional[QFrame] = None
            self._controls_effect: Optional[QGraphicsOpacityEffect] = None
            self._title_effect: Optional[QGraphicsOpacityEffect] = None
            self._controls_animation: Optional[QPropertyAnimation] = None
            self._title_animation: Optional[QPropertyAnimation] = None
            self._overlay_visible = True
            self._overlay_hide_timer = QTimer(self)
            self._overlay_hide_timer.setSingleShot(True)
            self._overlay_hide_timer.timeout.connect(self._fade_overlays)
            self._cursor_poll_timer = QTimer(self)
            self._cursor_poll_timer.setInterval(120)
            self._cursor_poll_timer.timeout.connect(self._poll_cursor_for_overlay)
            self._last_cursor_pos = QPointF(-1, -1)
            self._watchdog_timer = QTimer(self)
            self._watchdog_timer.setInterval(2000)
            self._watchdog_timer.timeout.connect(self._playback_watchdog_tick)

            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setMouseTracking(True)
            self.player = QMediaPlayer(self)
            self.audio_output = QAudioOutput(self)
            self.audio_output.setVolume(1.0)
            self.player.setAudioOutput(self.audio_output)

            self.video_widget = QVideoWidget(self)
            self.video_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.video_widget.setMouseTracking(True)
            self.video_widget.installEventFilter(self)
            self.player.setVideoOutput(self.video_widget)
            self.video_widget.videoSink().videoFrameChanged.connect(self._video_frame_changed)

            self.back_button = QPushButton("Back")
            self.play_button = MediaControlButton("play")
            self.rewind_button = QPushButton("↺")
            self.forward_button = QPushButton("↻")
            self.fullscreen_button = QPushButton("⛶")
            self.csv_button = QPushButton("CSV")
            self.export_button = QPushButton("Export")
            self.title_label = QLabel("Match")
            self.meta_label = QLabel("")
            self.status_label = QLabel("")
            self.status_label.setVisible(False)
            self.current_label = QLabel("0:00")
            self.duration_label = QLabel("0:00")
            self.slider = PointTimelineSlider(self)

            self._build_layout()
            self._connect_signals()

        def _build_layout(self) -> None:
            self.setStyleSheet(
                """
                QWidget { background: #101827; color: #f8fafc; font-size: 14px; }
                QLabel#title { font-size: 18px; font-weight: 750; }
                QLabel#meta, QLabel#status { color: #cbd5e1; }
                QPushButton {
                    background: #1f2a3d; border: 1px solid #334155; border-radius: 5px;
                    color: #f8fafc; padding: 8px 12px;
                }
                QPushButton:hover { background: #2c3a52; }
                QPushButton:disabled { color: #64748b; }
                QWidget#videoShell { background: #000000; }
                QWidget#videoOverlay { background: transparent; }
                QFrame#controlTray {
                    background: rgba(7, 11, 19, 215);
                    border: none;
                    border-radius: 0;
                }
                QFrame#titleOverlay {
                    background: rgba(7, 11, 19, 165);
                    border: none;
                    border-radius: 6px;
                }
                QFrame#titleOverlay QLabel {
                    background: transparent;
                    color: #f8fafc;
                    font-size: 18px;
                    font-weight: 750;
                    padding: 8px 12px;
                }
                QFrame#titleOverlay QPushButton {
                    background: rgba(12, 16, 24, 205);
                    border: 1px solid rgba(255, 255, 255, 36);
                    border-radius: 18px;
                    color: #ffffff;
                    min-height: 36px;
                    padding: 0 14px;
                    font-size: 14px;
                    font-weight: 700;
                }
                QFrame#titleOverlay QPushButton:hover { background: rgba(255, 255, 255, 28); }
                QFrame#controlTray QLabel {
                    background: transparent;
                    color: #f8fafc;
                    font-weight: 650;
                }
                QFrame#controlTray QPushButton {
                    background: transparent;
                    border: none;
                    border-radius: 20px;
                    color: #ffffff;
                    min-width: 40px;
                    min-height: 40px;
                    padding: 0;
                    font-size: 26px;
                    font-weight: 800;
                }
                QFrame#controlTray QPushButton:hover { background: rgba(255, 255, 255, 24); }
                QFrame#controlTray QPushButton#playButton {
                    background: transparent;
                    border: none;
                    min-width: 50px;
                    min-height: 50px;
                    border-radius: 25px;
                }
                QFrame#controlTray QPushButton#fullScreenButton {
                    min-width: 46px;
                    min-height: 46px;
                    border-radius: 23px;
                    font-size: 30px;
                }
                """
            )
            self.title_label.setObjectName("title")
            self.meta_label.setObjectName("meta")
            self.status_label.setObjectName("status")
            self.slider.setRange(0, 0)
            self.play_button.setObjectName("playButton")
            self.fullscreen_button.setObjectName("fullScreenButton")
            self.rewind_button.setToolTip("Back 5 seconds")
            self.forward_button.setToolTip("Forward 5 seconds")
            self.play_button.setToolTip("Play/Pause")
            self.fullscreen_button.setToolTip("Full screen")

            controls = QVBoxLayout()
            controls.setContentsMargins(34, 16, 34, 24)
            controls.setSpacing(10)
            controls.addWidget(self.slider)

            control_row = QHBoxLayout()
            control_row.setContentsMargins(0, 0, 0, 0)
            control_row.setSpacing(10)
            control_row.addWidget(self.rewind_button)
            control_row.addWidget(self.play_button)
            control_row.addWidget(self.forward_button)
            control_row.addSpacing(4)
            control_row.addWidget(self.current_label)
            separator_label = QLabel("/")
            separator_label.setObjectName("timeSeparator")
            control_row.addWidget(separator_label)
            control_row.addWidget(self.duration_label)
            control_row.addStretch(1)
            control_row.addWidget(self.fullscreen_button)
            controls.addLayout(control_row)

            overlay_flags = Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
            no_focus_flag = getattr(Qt.WindowType, "WindowDoesNotAcceptFocus", None)
            if no_focus_flag is not None:
                overlay_flags |= no_focus_flag
            self.control_tray = QFrame(None, overlay_flags)
            self.control_tray.setObjectName("controlTray")
            self.control_tray.setStyleSheet(
                """
                QFrame#controlTray {
                    background: rgba(7, 11, 19, 215);
                    border: none;
                    border-radius: 0;
                }
                QFrame#controlTray QLabel {
                    background: transparent;
                    color: #f8fafc;
                    font-weight: 650;
                }
                QFrame#controlTray QPushButton {
                    background: transparent;
                    border: none;
                    border-radius: 20px;
                    color: #ffffff;
                    min-width: 40px;
                    min-height: 40px;
                    padding: 0;
                    font-size: 26px;
                    font-weight: 800;
                }
                QFrame#controlTray QPushButton:hover { background: rgba(255, 255, 255, 24); }
                QFrame#controlTray QPushButton#playButton {
                    background: transparent;
                    border: none;
                    min-width: 50px;
                    min-height: 50px;
                    border-radius: 25px;
                }
                QFrame#controlTray QPushButton#fullScreenButton {
                    min-width: 46px;
                    min-height: 46px;
                    border-radius: 23px;
                    font-size: 30px;
                }
                """
            )
            self.control_tray.setLayout(controls)
            self.control_tray.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.control_tray.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            self.control_tray.setMouseTracking(True)
            self.control_tray.installEventFilter(self)
            self.control_tray.hide()

            self.title_overlay = QFrame(None, overlay_flags)
            self.title_overlay.setObjectName("titleOverlay")
            self.title_overlay.setStyleSheet(
                """
                QFrame#titleOverlay {
                    background: rgba(7, 11, 19, 165);
                    border: none;
                    border-radius: 6px;
                }
                QFrame#titleOverlay QLabel {
                    background: transparent;
                    color: #f8fafc;
                    font-size: 18px;
                    font-weight: 750;
                    padding: 8px 12px;
                }
                QFrame#titleOverlay QPushButton {
                    background: rgba(12, 16, 24, 205);
                    border: 1px solid rgba(255, 255, 255, 36);
                    border-radius: 18px;
                    color: #ffffff;
                    min-height: 36px;
                    padding: 0 14px;
                    font-size: 14px;
                    font-weight: 700;
                }
                QFrame#titleOverlay QPushButton:hover { background: rgba(255, 255, 255, 28); }
                """
            )
            self.title_overlay.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.title_overlay.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            self.title_overlay.setMouseTracking(True)
            self.title_overlay.installEventFilter(self)
            title_overlay_layout = QHBoxLayout(self.title_overlay)
            title_overlay_layout.setContentsMargins(8, 8, 8, 8)
            title_overlay_layout.setSpacing(10)
            title_overlay_layout.addWidget(self.back_button)
            title_overlay_layout.addWidget(self.title_label, 1)
            title_overlay_layout.addWidget(self.csv_button)
            title_overlay_layout.addWidget(self.export_button)
            self.title_overlay.hide()
            self.meta_label.hide()

            self._controls_effect = QGraphicsOpacityEffect(self.control_tray)
            self.control_tray.setGraphicsEffect(self._controls_effect)
            self._title_effect = QGraphicsOpacityEffect(self.title_overlay)
            self.title_overlay.setGraphicsEffect(self._title_effect)
            self._controls_animation = self._make_opacity_animation(self._controls_effect)
            self._title_animation = self._make_opacity_animation(self._title_effect)

            self.video_shell = QWidget(self)
            self.video_shell.setObjectName("videoShell")
            self.video_shell.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.video_shell.setMouseTracking(True)
            self.video_shell.installEventFilter(self)
            video_layout = QGridLayout(self.video_shell)
            video_layout.setContentsMargins(0, 0, 0, 0)
            video_layout.setSpacing(0)
            video_layout.addWidget(self.video_widget, 0, 0)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            layout.addWidget(self.video_shell, 1)
            layout.addWidget(self.status_label)
            self._install_overlay_activity_filters()

        def _connect_signals(self) -> None:
            self.back_button.clicked.connect(self._back)
            self.play_button.clicked.connect(self.toggle_playback)
            self.rewind_button.clicked.connect(lambda: self.skip_by_ms(-5000))
            self.forward_button.clicked.connect(lambda: self.skip_by_ms(5000))
            self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
            self.csv_button.clicked.connect(self.save_csv)
            self.export_button.clicked.connect(self.save_export)
            self.slider.sliderPressed.connect(lambda: setattr(self, "_seeking", True))
            self.slider.sliderReleased.connect(self._slider_released)
            self.slider.valueChanged.connect(self._slider_value_changed)
            self.player.positionChanged.connect(self._position_changed)
            self.player.durationChanged.connect(self._duration_changed)
            self.player.playbackStateChanged.connect(self._playback_state_changed)
            self.player.mediaStatusChanged.connect(self._media_status_changed)
            self.player.errorOccurred.connect(self._media_error)

        def _make_opacity_animation(self, effect: QGraphicsOpacityEffect) -> QPropertyAnimation:
            animation = QPropertyAnimation(effect, b"opacity", self)
            animation.setDuration(220)
            animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
            animation.finished.connect(self._hide_faded_overlays)
            return animation

        def _install_overlay_activity_filters(self) -> None:
            for widget in (
                self,
                self.video_shell,
                self.video_widget,
                self.control_tray,
                self.title_overlay,
                self.slider,
                self.rewind_button,
                self.play_button,
                self.forward_button,
                self.fullscreen_button,
                self.back_button,
                self.csv_button,
                self.export_button,
                self.title_label,
            ):
                if widget is not None:
                    widget.setMouseTracking(True)
                    widget.installEventFilter(self)

        def _animate_opacity(self, animation: Optional[QPropertyAnimation], target: float) -> None:
            if animation is None:
                return
            animation.stop()
            effect = animation.targetObject()
            start = float(effect.opacity()) if isinstance(effect, QGraphicsOpacityEffect) else target
            animation.setStartValue(start)
            animation.setEndValue(target)
            animation.start()

        def _window_allows_overlay_activity(self) -> bool:
            window = self.window()
            return native_overlay_should_show(window_active=bool(window and window.isActiveWindow()))

        def _show_overlays(self) -> None:
            if not self._window_allows_overlay_activity():
                return
            now = time.monotonic()
            if self._overlay_visible and now - self._last_overlay_show_monotonic < 0.35:
                self._overlay_hide_timer.start(3000)
                return
            self._last_overlay_show_monotonic = now
            self._overlay_visible = True
            if self.control_tray is not None:
                self.control_tray.show()
                self.control_tray.raise_()
            if self.title_overlay is not None:
                self.title_overlay.show()
                self.title_overlay.raise_()
            self._position_overlays()
            self._animate_opacity(self._controls_animation, 1.0)
            self._animate_opacity(self._title_animation, 1.0)
            self._overlay_hide_timer.start(3000)

        def _global_video_rect(self):
            if self.video_shell is None:
                return None
            video_rect = self.video_shell.rect()
            top_left = self.video_shell.mapToGlobal(video_rect.topLeft())
            return QRectF(top_left.x(), top_left.y(), video_rect.width(), video_rect.height())

        def _poll_cursor_for_overlay(self) -> None:
            if not self._window_allows_overlay_activity():
                self._last_cursor_pos = QPointF(-1, -1)
                return
            video_rect = self._global_video_rect()
            if video_rect is None:
                return
            pos = QCursor.pos()
            current_pos = QPointF(pos.x(), pos.y())
            if current_pos == self._last_cursor_pos:
                return
            self._last_cursor_pos = current_pos
            if video_rect.contains(current_pos):
                self._show_overlays()

        def _fade_overlays(self) -> None:
            self._overlay_visible = False
            self._animate_opacity(self._controls_animation, 0.0)
            self._animate_opacity(self._title_animation, 0.0)

        def _hide_faded_overlays(self) -> None:
            if self._overlay_visible:
                return
            if self.control_tray is not None:
                self.control_tray.hide()
            if self.title_overlay is not None:
                self.title_overlay.hide()

        def _position_overlays(self) -> None:
            if self.video_shell is None:
                return
            video_rect = self._global_video_rect()
            if video_rect is None:
                return
            video_x = int(video_rect.x())
            video_y = int(video_rect.y())
            video_w = int(video_rect.width())
            video_h = int(video_rect.height())
            if self.control_tray is not None:
                control_h = max(118, self.control_tray.sizeHint().height())
                self.control_tray.setGeometry(
                    video_x,
                    max(video_y, video_y + video_h - control_h),
                    video_w,
                    control_h,
                )
                self.control_tray.raise_()
            if self.title_overlay is not None:
                title_h = max(54, self.title_overlay.sizeHint().height())
                self.title_overlay.setGeometry(video_x + 16, video_y + 16, max(260, video_w - 32), title_h)
                self.title_overlay.raise_()

        def open_match(self, item_id: str) -> None:
            from gui.app import native_playback_descriptor

            self.stop()
            self.item_id = item_id
            self.descriptor = native_playback_descriptor(item_id)
            self.title_label.setText(str(self.descriptor.get("name") or item_id))
            self.meta_label.setText("")
            self._position_overlays()
            self._show_overlays()
            self._cursor_poll_timer.start()
            self._set_status("")
            self.csv_button.setEnabled(bool(self.descriptor.get("has_csv")))
            self.scheduler = NativePlaybackScheduler(
                self.descriptor.get("point_intervals") or [],
                self.descriptor.get("source_duration_s") or 0,
            )
            self.slider.setRange(0, max(0, self.scheduler.duration_ms))
            self.slider.set_points(self.scheduler.points, self.scheduler.duration_ms)
            self.duration_label.setText(self._format_ms(self.scheduler.duration_ms))
            self._media_kind, initial_media_path = native_initial_media_for_descriptor(self.descriptor)
            self._watchdog_reload_count = 0
            default_start = self.scheduler.default_start_ms()
            self._load_media(initial_media_path, default_start, True)
            self.setFocus(Qt.FocusReason.OtherFocusReason)

        def stop(self) -> None:
            self.player.stop()
            self.player.setSource(QUrl())
            self._pending_seek_ms = None
            self._seeking = False
            self._active_media_path = None
            self._cursor_poll_timer.stop()
            self._watchdog_timer.stop()

        def _load_media(self, path: str, seek_ms: int, autoplay: bool) -> None:
            self._media_ready = False
            self._pending_ready_attempts = 0
            self._pending_seek_ms = self.scheduler.clamp_ms(seek_ms)
            self._pending_autoplay = autoplay
            self._last_autoplay_requested = autoplay
            self._set_status("Loading video...")
            self._last_frame_monotonic = time.monotonic()
            self._last_position_change_monotonic = time.monotonic()
            self._last_watchdog_position_ms = self.scheduler.clamp_ms(seek_ms)
            self._last_watchdog_rss_mb = self._current_rss_mb()
            self._active_media_path = path
            _native_playback_logger().info(
                "event=native_playback_load item_id=%s media=%s path=%s seek_ms=%s autoplay=%s rss_mb=%s",
                self.item_id,
                self._media_kind,
                path,
                self._pending_seek_ms,
                autoplay,
                self._last_watchdog_rss_mb,
            )
            self.player.setSource(QUrl.fromLocalFile(path))
            self._watchdog_timer.start()
            QTimer.singleShot(80, self._apply_pending_seek_when_ready)

        def seek_to_ms(self, value_ms: int, autoplay: bool) -> None:
            target_ms = self.scheduler.clamp_ms(value_ms)
            self.scheduler.seek(target_ms)
            self._pending_seek_ms = target_ms
            self._pending_autoplay = autoplay
            self._last_autoplay_requested = autoplay
            self._apply_pending_seek()

        def skip_by_ms(self, delta_ms: int) -> None:
            was_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
            target_ms = self.scheduler.clamp_ms(self.player.position() + delta_ms)
            self.scheduler.active_segment = PlaybackSegment(
                start_ms=target_ms,
                end_ms=self.scheduler.duration_ms,
                point_index=None,
                next_point_index=None,
                mode="manual",
            )
            self._pending_seek_ms = target_ms
            self._pending_autoplay = was_playing
            self._last_autoplay_requested = was_playing
            self._apply_pending_seek(reclassify=False)
            self.setFocus(Qt.FocusReason.MouseFocusReason)

        def toggle_playback(self) -> None:
            if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.player.pause()
            else:
                self.scheduler.seek(self.player.position())
                self.player.play()
            self.setFocus(Qt.FocusReason.MouseFocusReason)

        def toggle_fullscreen(self) -> None:
            self._fullscreen = not self._fullscreen
            self.fullscreenRequested.emit(self._fullscreen)
            self.fullscreen_button.setText("×" if self._fullscreen else "⛶")
            self.setFocus(Qt.FocusReason.MouseFocusReason)

        def exit_fullscreen(self) -> None:
            if not self._fullscreen:
                return
            self._fullscreen = False
            self.fullscreenRequested.emit(False)
            self.fullscreen_button.setText("⛶")
            self.setFocus(Qt.FocusReason.OtherFocusReason)

        def save_csv(self) -> None:
            self._download_descriptor_url("csv_url", f"{self.item_id or 'match'}_segments.csv")

        def save_export(self) -> None:
            self._download_descriptor_url("export_url", f"{self.item_id or 'match'}_segmented.mp4")

        def _download_descriptor_url(self, key: str, suggested: str) -> None:
            endpoint = self.descriptor.get(key)
            if not endpoint:
                QMessageBox.warning(self, "RallyClip", "This file is not available.")
                return
            downloads = Path.home() / "Downloads"
            target, _ = QFileDialog.getSaveFileName(
                self,
                "Save",
                str((downloads if downloads.is_dir() else Path.home()) / suggested),
            )
            if not target:
                return
            url = f"http://127.0.0.1:{self.port}{endpoint}"
            try:
                with urllib.request.urlopen(url, timeout=300) as response:
                    Path(target).write_bytes(response.read())
            except (OSError, urllib.error.URLError) as exc:
                QMessageBox.warning(self, "RallyClip", f"Could not save file: {exc}")

        def _media_is_seek_ready(self) -> bool:
            if self._media_ready:
                return True
            if self.player.duration() > 0:
                return True
            return self.player.mediaStatus() in {
                QMediaPlayer.MediaStatus.LoadedMedia,
                QMediaPlayer.MediaStatus.BufferedMedia,
            }

        def _apply_pending_seek_when_ready(self) -> None:
            if self._pending_seek_ms is None:
                return
            if not self._media_is_seek_ready() and self._pending_ready_attempts < 25:
                self._pending_ready_attempts += 1
                QTimer.singleShot(80, self._apply_pending_seek_when_ready)
                return
            self._apply_pending_seek(settle_autoplay=True)

        def _apply_pending_seek(self, reclassify: bool = True, settle_autoplay: bool = False) -> None:
            if self._pending_seek_ms is None:
                return
            target_ms = self._pending_seek_ms
            autoplay = self._pending_autoplay
            if reclassify:
                self.scheduler.seek(target_ms)
            self.player.setPosition(target_ms)
            self.slider.setValue(target_ms)
            self.current_label.setText(self._format_ms(target_ms))
            self._pending_seek_ms = None
            self._set_status("")
            if autoplay:
                if settle_autoplay:
                    QTimer.singleShot(180, self.player.play)
                else:
                    self.player.play()

        def _slider_released(self) -> None:
            self._seeking = False
            was_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
            self.seek_to_ms(self.slider.value(), was_playing)

        def _slider_value_changed(self, value: int) -> None:
            if self._seeking:
                self.current_label.setText(self._format_ms(value))

        def _position_changed(self, position_ms: int) -> None:
            if abs(position_ms - self._last_watchdog_position_ms) >= 250:
                self._last_position_change_monotonic = time.monotonic()
            if not self._seeking:
                self.slider.setValue(position_ms)
                self.current_label.setText(self._format_ms(position_ms))
            if self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                return
            if not self.scheduler.should_advance(position_ms):
                return
            next_start = self.scheduler.next_start_after_active()
            if next_start is None:
                tail_start = self.scheduler.tail_start_after_active()
                if tail_start is not None:
                    self.scheduler.seek(max(position_ms, tail_start))
                    return
                self.player.pause()
                return
            self.seek_to_ms(next_start, True)

        def _duration_changed(self, duration_ms: int) -> None:
            if self.scheduler.duration_ms <= 0 and duration_ms > 0:
                self.scheduler.duration_ms = duration_ms
                self.slider.setRange(0, duration_ms)
                self.slider.set_points(self.scheduler.points, duration_ms)
                self.duration_label.setText(self._format_ms(duration_ms))
            self._apply_pending_seek_when_ready()

        def _playback_state_changed(self, state) -> None:
            if state == QMediaPlayer.PlaybackState.PlayingState:
                self.play_button.set_icon("pause")
            else:
                self.play_button.set_icon("play")

        def _media_status_changed(self, status) -> None:
            if status in {
                QMediaPlayer.MediaStatus.LoadedMedia,
                QMediaPlayer.MediaStatus.BufferedMedia,
                QMediaPlayer.MediaStatus.EndOfMedia,
            }:
                self._media_ready = True
                self._apply_pending_seek_when_ready()

        def _media_error(self, error, error_string: str = "") -> None:
            if error == QMediaPlayer.Error.NoError:
                return
            if self._media_kind == "source":
                self._start_proxy_fallback(error_string or "Source playback failed.")
                return
            self._set_status(error_string or "Could not play this video.")

        def _start_proxy_fallback(self, reason: str) -> None:
            if not self.item_id:
                self._set_status(reason)
                return
            self._media_kind = "proxy"
            seek_ms = self.player.position() or self.scheduler.default_start_ms()
            autoplay = self._last_autoplay_requested or self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
            self._set_status("Preparing compatible playback proxy...")
            self._proxy_thread = QThread(self)
            self._proxy_worker = ProxyWorker(self.item_id)
            self._proxy_worker.moveToThread(self._proxy_thread)
            self._proxy_thread.started.connect(self._proxy_worker.run)
            self._proxy_worker.finished.connect(
                lambda proxy, error: self._proxy_finished(proxy, error, seek_ms, autoplay)
            )
            self._proxy_worker.finished.connect(self._proxy_thread.quit)
            self._proxy_worker.finished.connect(self._proxy_worker.deleteLater)
            self._proxy_thread.finished.connect(self._proxy_thread.deleteLater)
            self._proxy_thread.start()

        def _proxy_finished(self, proxy: dict[str, Any], error: str, seek_ms: int, autoplay: bool) -> None:
            self._proxy_thread = None
            self._proxy_worker = None
            if error:
                self._set_status(error)
                return
            proxy_path = proxy.get("path")
            if not proxy_path:
                self._set_status("Could not prepare playback proxy.")
                return
            self._load_media(str(proxy_path), seek_ms, autoplay)

        def _reload_active_media_playback(self, reason: str) -> None:
            if self._watchdog_reload_count >= 1:
                self._watchdog_timer.stop()
                self.player.pause()
                self._set_status(f"Playback stalled: {reason}. Reopen this match to retry.")
                return
            media_path = self._active_media_path
            if not media_path:
                self._set_status(f"Playback stalled before video was ready: {reason}.")
                return
            self._watchdog_reload_count += 1
            position_ms = self.scheduler.clamp_ms(self.player.position() or self.scheduler.default_start_ms())
            autoplay = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
            _native_playback_logger().warning(
                "event=native_playback_reload reason=%s item_id=%s position_ms=%s rss_mb=%s",
                reason,
                self.item_id,
                position_ms,
                self._current_rss_mb(),
            )
            self.player.stop()
            self.player.setSource(QUrl())
            self._load_media(media_path, position_ms, autoplay)

        def _current_rss_mb(self) -> Optional[float]:
            if self._memory_process is None:
                return None
            try:
                return float(self._memory_process.memory_info().rss) / 1e6
            except Exception:
                return None

        def _video_frame_changed(self, frame) -> None:
            del frame
            self._last_frame_monotonic = time.monotonic()

        def _playback_watchdog_tick(self) -> None:
            now = time.monotonic()
            position_ms = int(self.player.position())
            rss_mb = self._current_rss_mb()
            status = self.player.mediaStatus()
            state = self.player.playbackState()
            _native_playback_logger().info(
                "event=native_playback_watchdog item_id=%s media=%s status=%s state=%s position_ms=%s "
                "buffer=%.3f rss_mb=%s seconds_since_frame=%.3f",
                self.item_id,
                self._media_kind,
                getattr(status, "name", str(status)),
                getattr(state, "name", str(state)),
                position_ms,
                float(self.player.bufferProgress()),
                f"{rss_mb:.1f}" if rss_mb is not None else "unknown",
                now - self._last_frame_monotonic,
            )
            if state != QMediaPlayer.PlaybackState.PlayingState:
                self._last_watchdog_position_ms = position_ms
                self._last_watchdog_rss_mb = rss_mb
                return

            reason = native_watchdog_reload_reason(
                playing=True,
                position_ms=position_ms,
                last_position_ms=self._last_watchdog_position_ms,
                seconds_since_frame=now - self._last_frame_monotonic,
                rss_mb=rss_mb,
                last_rss_mb=self._last_watchdog_rss_mb,
            )
            if reason:
                self._reload_active_media_playback(reason)
                return

            self._last_watchdog_position_ms = position_ms
            self._last_watchdog_rss_mb = rss_mb

        def _back(self) -> None:
            self._fullscreen = False
            if self.control_tray is not None:
                self.control_tray.hide()
            if self.title_overlay is not None:
                self.title_overlay.hide()
            self._cursor_poll_timer.stop()
            self.stop()
            self.backRequested.emit()

        def _set_status(self, message: str) -> None:
            self.status_label.setText(message)
            self.status_label.setVisible(bool(message))

        def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
            if event.type() in {
                QEvent.Type.MouseMove,
                QEvent.Type.Enter,
                QEvent.Type.HoverMove,
            } and self._window_allows_overlay_activity():
                self._show_overlays()
            video_targets = tuple(
                widget
                for widget in (
                    getattr(self, "video_widget", None),
                    getattr(self, "video_shell", None),
                )
                if widget is not None
            )
            if watched in video_targets:
                if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                    self.toggle_playback()
                    return True
            return super().eventFilter(watched, event)

        def keyPressEvent(self, event) -> None:  # type: ignore[override]
            key = event.key()
            if key == Qt.Key.Key_Space:
                self.toggle_playback()
                event.accept()
                return
            if key == Qt.Key.Key_F:
                self.toggle_fullscreen()
                event.accept()
                return
            if key == Qt.Key.Key_Escape:
                self.exit_fullscreen()
                event.accept()
                return
            if key == Qt.Key.Key_Left:
                self.skip_by_ms(-5000)
                event.accept()
                return
            if key == Qt.Key.Key_Right:
                self.skip_by_ms(5000)
                event.accept()
                return
            super().keyPressEvent(event)

        def resizeEvent(self, event) -> None:  # type: ignore[override]
            super().resizeEvent(event)
            self._position_overlays()

        def moveEvent(self, event) -> None:  # type: ignore[override]
            super().moveEvent(event)
            self._position_overlays()

        def hideEvent(self, event) -> None:  # type: ignore[override]
            super().hideEvent(event)
            if self.control_tray is not None:
                self.control_tray.hide()
            if self.title_overlay is not None:
                self.title_overlay.hide()

        def _meta_text(self) -> str:
            points = self.descriptor.get("point_intervals") or []
            duration = self.descriptor.get("source_duration_s")
            parts = [f"{len(points)} point{'s' if len(points) != 1 else ''}"]
            if duration:
                parts.append(f"{round(float(duration))}s video")
            source_name = self.descriptor.get("source_name")
            if source_name:
                parts.append(str(source_name))
            return " · ".join(parts)

        @staticmethod
        def _format_ms(value_ms: int | float) -> str:
            total = max(0, int(round(float(value_ms) / 1000)))
            minutes, seconds = divmod(total, 60)
            hours, minutes = divmod(minutes, 60)
            if hours:
                return f"{hours}:{minutes:02d}:{seconds:02d}"
            return f"{minutes}:{seconds:02d}"
