"""Animated splash screen with pulsing heart icon and progress bar."""

import math

from PySide6.QtWidgets import QWidget, QApplication, QGraphicsOpacityEffect
from PySide6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QRectF,
)
from PySide6.QtGui import (
    QPainter, QPainterPath, QColor, QLinearGradient, QRadialGradient,
    QFont, QPen, QBrush,
)


class SplashScreen(QWidget):

    WIDTH = 520
    HEIGHT = 380
    MIN_DISPLAY_MS = 2500
    FADE_DURATION_MS = 400
    TICK_MS = 30  # ~33 fps

    BG = QColor("#1F2937")
    BORDER = QColor("#374151")
    ACCENT = QColor("#2563EB")
    ACCENT_LIGHT = QColor("#3B82F6")
    TEXT = QColor("#FFFFFF")
    SUBTEXT = QColor("#9CA3AF")
    STATUS = QColor("#93C5FD")
    TRACK = QColor("#374151")
    HEART_DARK = QColor("#DC2626")
    HEART_LIGHT = QColor("#EF4444")
    MUTED = QColor("#6B7280")

    _STAGES = [
        (0, "Initializing...", 0.10),
        (500, "Loading UI components...", 0.35),
        (1200, "Preparing visualization...", 0.60),
        (1800, "Setting up workspace...", 0.85),
        (2200, "Ready!", 1.00),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(self.WIDTH, self.HEIGHT)

        screen = QApplication.primaryScreen().geometry()
        self.move(
            (screen.width() - self.WIDTH) // 2,
            (screen.height() - self.HEIGHT) // 2,
        )

        self._progress = 0.0
        self._target_progress = 0.0
        self._status_text = ""
        self._elapsed_ms = 0
        self._pulse_phase = 0.0
        self._stage_idx = 0

        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity)

        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._on_tick)

        self._main_window = None
        self._callback = None

    def start(self):
        self._status_text = self._STAGES[0][1]
        self._target_progress = self._STAGES[0][2]
        self.show()
        self._timer.start()

    def finish(self, main_window, callback=None):
        self._main_window = main_window
        self._callback = callback

        remaining = self.MIN_DISPLAY_MS - self._elapsed_ms
        if remaining > 0:
            QTimer.singleShot(remaining, self._begin_fade)
        else:
            self._begin_fade()

    def _on_tick(self):
        self._elapsed_ms += self.TICK_MS
        self._pulse_phase += 0.12  # heartbeat speed

        next_idx = self._stage_idx + 1
        if next_idx < len(self._STAGES):
            trigger_ms, msg, target = self._STAGES[next_idx]
            if self._elapsed_ms >= trigger_ms:
                self._stage_idx = next_idx
                self._status_text = msg
                self._target_progress = target

        self._progress += (self._target_progress - self._progress) * 0.08

        self.update()

    def _begin_fade(self):
        self._timer.stop()
        anim = QPropertyAnimation(self._opacity, b"opacity", self)
        anim.setDuration(self.FADE_DURATION_MS)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.InQuad)
        anim.finished.connect(self._on_fade_done)
        anim.start()
        self._fade_anim = anim  # prevent GC

    def _on_fade_done(self):
        if self._callback:
            self._callback()
        elif self._main_window:
            self._main_window.show()
        self.close()
        self.deleteLater()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)

        self._draw_background(p)
        self._draw_heart_icon(p)
        self._draw_text(p)
        self._draw_progress_bar(p)

        p.end()

    def _draw_background(self, p: QPainter):
        rect = QRectF(0, 0, self.WIDTH, self.HEIGHT)
        path = QPainterPath()
        path.addRoundedRect(rect, 14, 14)

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(self.BG))
        p.drawPath(path)

        p.setPen(QPen(self.BORDER, 1))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

    def _draw_heart_icon(self, p: QPainter):
        cx = self.WIDTH / 2
        cy = 90
        s = 38  # half-size

        glow_alpha = 0.12 + 0.08 * math.sin(self._pulse_phase)
        glow_radius = s * 1.8 + 4 * math.sin(self._pulse_phase * 0.7)
        grad = QRadialGradient(cx, cy, glow_radius)
        glow_color = QColor(self.HEART_LIGHT)
        glow_color.setAlphaF(glow_alpha)
        grad.setColorAt(0.0, glow_color)
        glow_color.setAlphaF(0.0)
        grad.setColorAt(1.0, glow_color)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(cx - glow_radius, cy - glow_radius,
                       glow_radius * 2, glow_radius * 2)

        heart = QPainterPath()
        heart.moveTo(cx, cy + s * 0.5)  # bottom tip
        # Left lobe
        heart.cubicTo(
            cx - s * 1.3, cy + s * 0.0,
            cx - s * 0.8, cy - s * 1.0,
            cx, cy - s * 0.35,
        )
        # Right lobe
        heart.cubicTo(
            cx + s * 0.8, cy - s * 1.0,
            cx + s * 1.3, cy + s * 0.0,
            cx, cy + s * 0.5,
        )

        heart_grad = QRadialGradient(cx, cy - s * 0.2, s * 1.2)
        heart_grad.setColorAt(0.0, self.HEART_LIGHT)
        heart_grad.setColorAt(1.0, self.HEART_DARK)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(heart_grad))
        p.drawPath(heart)

        artery = QPainterPath()
        artery.moveTo(cx - s * 0.5, cy - s * 0.6)
        artery.cubicTo(
            cx - s * 0.1, cy - s * 0.2,
            cx + s * 0.1, cy + s * 0.1,
            cx + s * 0.4, cy + s * 0.3,
        )
        pen = QPen(self.ACCENT, 2.5, Qt.SolidLine, Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(artery)

    def _draw_text(self, p: QPainter):
        p.setPen(self.TEXT)
        title_font = QFont("Segoe UI", 18, QFont.Bold)
        p.setFont(title_font)
        p.drawText(QRectF(0, 145, self.WIDTH, 35), Qt.AlignCenter,
                    "Stenosis Detection")

        p.setPen(self.SUBTEXT)
        sub_font = QFont("Segoe UI", 10)
        p.setFont(sub_font)
        p.drawText(QRectF(0, 180, self.WIDTH, 25), Qt.AlignCenter,
                    "University of Haifa & Ziv Medical Center")

        p.setPen(QPen(self.BORDER, 1))
        margin = 60
        y_sep = 215
        p.drawLine(margin, y_sep, self.WIDTH - margin, y_sep)

        p.setPen(self.STATUS)
        status_font = QFont("Segoe UI", 11)
        p.setFont(status_font)
        p.drawText(QRectF(0, 232, self.WIDTH, 25), Qt.AlignCenter,
                    self._status_text)

        p.setPen(self.MUTED)
        ver_font = QFont("Segoe UI", 9)
        p.setFont(ver_font)
        p.drawText(QRectF(0, 300, self.WIDTH, 20), Qt.AlignCenter,
                    "Research Demo v1.0")

    def _draw_progress_bar(self, p: QPainter):
        margin = 50
        bar_y = 268
        bar_h = 10
        bar_w = self.WIDTH - 2 * margin
        radius = bar_h / 2

        track_rect = QRectF(margin, bar_y, bar_w, bar_h)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(self.TRACK))
        p.drawRoundedRect(track_rect, radius, radius)

        fill_w = bar_w * self._progress
        if fill_w > 1:
            fill_rect = QRectF(margin, bar_y, fill_w, bar_h)
            grad = QLinearGradient(margin, 0, margin + fill_w, 0)
            grad.setColorAt(0.0, self.ACCENT)
            grad.setColorAt(1.0, self.ACCENT_LIGHT)
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(fill_rect, radius, radius)

            shine = QColor(255, 255, 255, 40)
            shine_rect = QRectF(margin, bar_y, fill_w, bar_h * 0.4)
            p.setBrush(QBrush(shine))
            p.drawRoundedRect(shine_rect, radius / 2, radius / 2)
