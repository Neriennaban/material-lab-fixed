from __future__ import annotations

import json
import os
import hashlib
import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import (
    QEasingCurve,
    QObject,
    QPropertyAnimation,
    QPointF,
    QRectF,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QImage,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsOpacityEffect,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSlider,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.imaging import simulate_microscope_view
from core.psf_engineering import DEFAULT_HYBRID_BALANCE, DEFAULT_SECTIONING_SHEAR_DEG
from core.virtual_slide import get_path_slide
from core.microscope_measurements import (
    derive_um_per_px_100x,
    format_metric_area_um2,
    format_metric_length_um,
    line_measurement,
    polygon_area_measurement,
    scale_audit_report,
)
from core.ui_v2_utils import (
    build_capture_metadata,
    choose_scale_bar,
    default_session_id,
    estimate_um_per_px,
    now_iso,
)
from export.export_images import save_image
from export.export_tables import save_json
from ui_qt.theme_mirea import (
    MICROSCOPE_THEME_PROFILE,
    build_qss,
    load_theme_mode,
    resolve_branding_logo,
    save_theme_mode,
    status_color,
)
from ui_qt.modern_widgets import FlexibleDoubleSpinBox, parse_flexible_float
from ui_qt.window_state_manager import WindowStateManager
from ui_qt.window_mode_mixin import WindowModeMixin

QDoubleSpinBox = FlexibleDoubleSpinBox

Image.MAX_IMAGE_PIXELS = None


@lru_cache(maxsize=8)
def _load_overlay_font(size: int) -> ImageFont.ImageFont:
    size_px = max(10, int(size))
    candidates: list[Path] = []
    win_dir = Path(os.environ.get("WINDIR", "C:/Windows"))
    fonts_dir = win_dir / "Fonts"
    for name in (
        "segoeui.ttf",
        "arial.ttf",
        "tahoma.ttf",
        "calibri.ttf",
        "DejaVuSans.ttf",
    ):
        candidates.append(fonts_dir / name)
        candidates.append(Path(name))
    for font_path in candidates:
        try:
            if font_path.exists():
                return ImageFont.truetype(str(font_path), size_px)
        except Exception:
            continue
    return ImageFont.load_default()


def _font_supports_text(font: ImageFont.ImageFont, text: str) -> bool:
    try:
        font.getmask(text)
        return True
    except Exception:
        return False


def _to_pixmap(image: np.ndarray) -> QPixmap:
    if image.ndim == 2:
        rgb = np.stack([image] * 3, axis=2).astype(np.uint8, copy=False)
    elif image.ndim == 3 and image.shape[2] >= 3:
        rgb = image[:, :, :3].astype(np.uint8, copy=False)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    arr = np.ascontiguousarray(rgb)
    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _safe_float(value: Any, default: float = 0.0) -> float:
    return parse_flexible_float(value, default)


class _AsyncRenderSignals(QObject):
    rendered = Signal(object)
    failed = Signal(object)


def _candidate_metadata_paths(image_path: Path) -> list[Path]:
    base = image_path.with_suffix("")
    return [
        image_path.with_suffix(".json"),
        base.with_name(base.name + "_student.json"),
        base.with_name(base.name + "_metadata.json"),
    ]


class ZoomView(QGraphicsView):
    measurementChanged = Signal(dict)
    measurementFinished = Signal(dict)
    measurementCleared = Signal()
    TOOL_OFF = "off"
    TOOL_LINE = "line"
    TOOL_POLYGON_AREA = "polygon_area"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMouseTracking(True)
        self._measurement_tool = self.TOOL_OFF
        self._measurement_active = False
        self._measure_start_scene: QPointF | None = None
        self._measure_end_scene: QPointF | None = None
        self._measure_um_per_px = 1.0
        self._polygon_vertices: list[QPointF] = []
        self._polygon_preview_scene: QPointF | None = None
        self._polygon_closed = False
        self._polygon_close_hot = False
        self._polygon_close_radius_px = 12.0

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)

    def set_measurement_tool(self, mode: str) -> None:
        tool = str(mode or self.TOOL_OFF).strip().lower()
        if tool not in {self.TOOL_OFF, self.TOOL_LINE, self.TOOL_POLYGON_AREA}:
            tool = self.TOOL_OFF
        self._measurement_tool = tool
        self._measurement_active = False
        self._reset_line_measurement_state()
        self._reset_polygon_measurement_state()
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag
            if self._measurement_tool != self.TOOL_OFF
            else QGraphicsView.DragMode.ScrollHandDrag
        )
        self.viewport().setCursor(
            Qt.CursorShape.CrossCursor
            if self._measurement_tool != self.TOOL_OFF
            else Qt.CursorShape.OpenHandCursor
        )
        self.viewport().update()

    def set_measurement_mode(self, enabled: bool) -> None:
        self.set_measurement_tool(self.TOOL_LINE if enabled else self.TOOL_OFF)

    def is_measurement_mode(self) -> bool:
        return bool(self._measurement_tool == self.TOOL_LINE)

    def measurement_tool(self) -> str:
        return str(self._measurement_tool)

    def set_measurement_um_per_px(self, value: float) -> None:
        self._measure_um_per_px = max(1e-9, float(value))
        self.viewport().update()

    def clear_measurement(self, *, emit_signal: bool = True) -> None:
        self._measurement_active = False
        self._reset_line_measurement_state()
        self._reset_polygon_measurement_state()
        self.viewport().update()
        if emit_signal:
            self.measurementCleared.emit()

    def current_measurement(self, *, finished: bool | None = None) -> dict[str, Any]:
        if self._measurement_tool == self.TOOL_POLYGON_AREA:
            return self._polygon_measurement_payload(finished=finished)
        if self._measurement_tool == self.TOOL_LINE:
            return self._line_measurement_payload(finished=finished)
        return {
            "valid": False,
            "kind": self.TOOL_OFF,
            "finished": bool(False if finished is None else finished),
        }

    def clamp_measurement_to_scene(self) -> None:
        if self._measure_start_scene is not None:
            self._measure_start_scene = self._clamp_scene_point(
                self._measure_start_scene
            )
        if self._measure_end_scene is not None:
            self._measure_end_scene = self._clamp_scene_point(self._measure_end_scene)
        if self._polygon_vertices:
            self._polygon_vertices = [
                self._clamp_scene_point(point) for point in self._polygon_vertices
            ]
        if self._polygon_preview_scene is not None:
            self._polygon_preview_scene = self._clamp_scene_point(
                self._polygon_preview_scene
            )
        self.viewport().update()

    def remove_last_polygon_vertex(self) -> bool:
        if (
            self._measurement_tool != self.TOOL_POLYGON_AREA
            or self._polygon_closed
            or not self._polygon_vertices
        ):
            return False
        self._polygon_vertices.pop()
        self._polygon_preview_scene = (
            self._polygon_vertices[-1] if self._polygon_vertices else None
        )
        self._polygon_close_hot = False
        self.viewport().update()
        if not self._polygon_vertices:
            self.measurementCleared.emit()
            return True
        self.measurementChanged.emit(self._polygon_measurement_payload(finished=False))
        return True

    def _reset_line_measurement_state(self) -> None:
        self._measure_start_scene = None
        self._measure_end_scene = None

    def _reset_polygon_measurement_state(self) -> None:
        self._polygon_vertices = []
        self._polygon_preview_scene = None
        self._polygon_closed = False
        self._polygon_close_hot = False

    def _clamp_scene_point(self, point: QPointF) -> QPointF:
        rect = self.sceneRect()
        if rect.isNull():
            return QPointF(point)
        x = min(max(float(point.x()), float(rect.left())), float(rect.right()))
        y = min(max(float(point.y()), float(rect.top())), float(rect.bottom()))
        return QPointF(x, y)

    def _lock_axis_if_needed(self, start: QPointF, end: QPointF) -> QPointF:
        modifiers = QApplication.keyboardModifiers()
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            dx = abs(float(end.x()) - float(start.x()))
            dy = abs(float(end.y()) - float(start.y()))
            if dx >= dy:
                return QPointF(float(end.x()), float(start.y()))
            return QPointF(float(start.x()), float(end.y()))
        return end

    def _line_measurement_payload(
        self, *, finished: bool | None = None
    ) -> dict[str, Any]:
        if self._measure_start_scene is None or self._measure_end_scene is None:
            return {
                "valid": False,
                "kind": self.TOOL_LINE,
                "finished": bool(False if finished is None else finished),
                "length_px": 0.0,
                "length_um": 0.0,
                "label": format_metric_length_um(0.0),
            }
        payload = line_measurement(
            (self._measure_start_scene.x(), self._measure_start_scene.y()),
            (self._measure_end_scene.x(), self._measure_end_scene.y()),
            self._measure_um_per_px,
        )
        payload["finished"] = bool(
            (not self._measurement_active) if finished is None else finished
        )
        return payload

    def _polygon_close_distance_met(self, viewport_point) -> bool:
        if len(self._polygon_vertices) < 3:
            return False
        start_view = self.mapFromScene(self._polygon_vertices[0])
        dx = float(viewport_point.x()) - float(start_view.x())
        dy = float(viewport_point.y()) - float(start_view.y())
        return bool((dx * dx) + (dy * dy) <= float(self._polygon_close_radius_px**2))

    def _polygon_measurement_payload(
        self, *, finished: bool | None = None
    ) -> dict[str, Any]:
        payload = polygon_area_measurement(
            [(float(point.x()), float(point.y())) for point in self._polygon_vertices],
            self._measure_um_per_px,
        )
        payload["finished"] = bool(
            self._polygon_closed if finished is None else finished
        )
        payload["closed"] = bool(self._polygon_closed)
        return payload

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if (
            self._measurement_tool == self.TOOL_LINE
            and event.button() == Qt.MouseButton.LeftButton
        ):
            scene_point = self._clamp_scene_point(
                self.mapToScene(event.position().toPoint())
            )
            self._measure_start_scene = scene_point
            self._measure_end_scene = scene_point
            self._measurement_active = True
            self.measurementChanged.emit(self._line_measurement_payload(finished=False))
            self.viewport().update()
            event.accept()
            return
        if (
            self._measurement_tool == self.TOOL_POLYGON_AREA
            and event.button() == Qt.MouseButton.LeftButton
        ):
            scene_point = self._clamp_scene_point(
                self.mapToScene(event.position().toPoint())
            )
            viewport_point = event.position().toPoint()
            if self._polygon_closed:
                self._reset_polygon_measurement_state()
            if self._polygon_vertices and self._polygon_close_distance_met(
                viewport_point
            ):
                self._polygon_closed = True
                self._polygon_preview_scene = None
                self._polygon_close_hot = False
                payload = self._polygon_measurement_payload(finished=True)
                self.measurementChanged.emit(payload)
                self.measurementFinished.emit(payload)
                self.viewport().update()
                event.accept()
                return
            self._polygon_vertices.append(scene_point)
            self._polygon_preview_scene = scene_point
            self._polygon_close_hot = self._polygon_close_distance_met(viewport_point)
            self.measurementChanged.emit(
                self._polygon_measurement_payload(finished=False)
            )
            self.viewport().update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if (
            self._measurement_tool == self.TOOL_LINE
            and self._measurement_active
            and self._measure_start_scene is not None
        ):
            scene_point = self._clamp_scene_point(
                self.mapToScene(event.position().toPoint())
            )
            self._measure_end_scene = self._lock_axis_if_needed(
                self._measure_start_scene, scene_point
            )
            self.measurementChanged.emit(self._line_measurement_payload(finished=False))
            self.viewport().update()
            event.accept()
            return
        if (
            self._measurement_tool == self.TOOL_POLYGON_AREA
            and self._polygon_vertices
            and not self._polygon_closed
        ):
            scene_point = self._clamp_scene_point(
                self.mapToScene(event.position().toPoint())
            )
            self._polygon_preview_scene = scene_point
            self._polygon_close_hot = self._polygon_close_distance_met(
                event.position().toPoint()
            )
            self.viewport().update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if (
            self._measurement_tool == self.TOOL_LINE
            and event.button() == Qt.MouseButton.LeftButton
            and self._measurement_active
        ):
            if self._measure_start_scene is not None:
                scene_point = self._clamp_scene_point(
                    self.mapToScene(event.position().toPoint())
                )
                self._measure_end_scene = self._lock_axis_if_needed(
                    self._measure_start_scene, scene_point
                )
            self._measurement_active = False
            payload = self._line_measurement_payload(finished=True)
            self.measurementChanged.emit(payload)
            self.measurementFinished.emit(payload)
            self.viewport().update()
            event.accept()
            return
        if (
            self._measurement_tool == self.TOOL_POLYGON_AREA
            and event.button() == Qt.MouseButton.LeftButton
        ):
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # type: ignore[override]
        super().drawForeground(painter, rect)
        painter.save()
        painter.resetTransform()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if self._measurement_tool == self.TOOL_LINE:
            hint_rect = QRectF(14.0, 14.0, 330.0, 28.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(10, 18, 28, 178))
            painter.drawRoundedRect(hint_rect, 8.0, 8.0)
            painter.setPen(QColor(240, 244, 248))
            painter.drawText(
                hint_rect.adjusted(10.0, 0.0, -10.0, 0.0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                "ЛКМ — измерить, Shift — зафиксировать ось",
            )
        elif self._measurement_tool == self.TOOL_POLYGON_AREA:
            hint_rect = QRectF(14.0, 14.0, 470.0, 28.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(10, 18, 28, 178))
            painter.drawRoundedRect(hint_rect, 8.0, 8.0)
            painter.setPen(QColor(240, 244, 248))
            painter.drawText(
                hint_rect.adjusted(10.0, 0.0, -10.0, 0.0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                "ЛКМ — вершины, клик по первой точке — замкнуть, Backspace — удалить вершину",
            )

        if (
            self._measure_start_scene is not None
            and self._measure_end_scene is not None
        ):
            p0 = self.mapFromScene(self._measure_start_scene)
            p1 = self.mapFromScene(self._measure_end_scene)
            line_pen = QPen(QColor(255, 205, 88), 2.0)
            line_pen.setCosmetic(True)
            painter.setPen(line_pen)
            painter.drawLine(p0, p1)

            painter.setPen(QPen(QColor(10, 18, 28), 1.0))
            painter.setBrush(QColor(255, 205, 88))
            painter.drawEllipse(
                QRectF(float(p0.x()) - 4.0, float(p0.y()) - 4.0, 8.0, 8.0)
            )
            painter.drawEllipse(
                QRectF(float(p1.x()) - 4.0, float(p1.y()) - 4.0, 8.0, 8.0)
            )

            payload = self._line_measurement_payload()
            if bool(payload.get("valid", False)):
                length_label = str(payload.get("label", "—"))
                px_label = f"{float(payload.get('length_px', 0.0)):.1f} px"
                angle_label = f"{float(payload.get('angle_deg', 0.0)):.1f}°"
                text = f"{length_label} | {px_label} | {angle_label}"
                fm = painter.fontMetrics()
                text_w = fm.horizontalAdvance(text) + 16
                text_h = fm.height() + 10
                mx = 0.5 * (float(p0.x()) + float(p1.x()))
                my = 0.5 * (float(p0.y()) + float(p1.y()))
                label_rect = QRectF(
                    mx - text_w / 2.0, my - text_h - 12.0, float(text_w), float(text_h)
                )
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(10, 18, 28, 210))
                painter.drawRoundedRect(label_rect, 8.0, 8.0)
                painter.setPen(QColor(248, 248, 248))
                painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, text)

        if self._polygon_vertices:
            points = [
                QPointF(
                    float(self.mapFromScene(point).x()),
                    float(self.mapFromScene(point).y()),
                )
                for point in self._polygon_vertices
            ]
            polygon_pen = QPen(QColor(103, 214, 255), 2.0)
            polygon_pen.setCosmetic(True)
            painter.setPen(polygon_pen)
            if self._polygon_closed and len(points) >= 3:
                polygon = QPolygonF(points)
                path = QPainterPath()
                path.addPolygon(polygon)
                painter.setBrush(QColor(103, 214, 255, 44))
                painter.drawPath(path)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPolygon(polygon)
            elif len(points) >= 2:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPolyline(QPolygonF(points))
            if (
                not self._polygon_closed
                and self._polygon_preview_scene is not None
                and points
            ):
                preview_point = self.mapFromScene(self._polygon_preview_scene)
                preview_pen = QPen(
                    QColor(103, 214, 255, 180), 1.5, Qt.PenStyle.DashLine
                )
                preview_pen.setCosmetic(True)
                painter.setPen(preview_pen)
                painter.drawLine(
                    points[-1],
                    QPointF(float(preview_point.x()), float(preview_point.y())),
                )
                if len(points) >= 3 and self._polygon_close_hot:
                    painter.drawLine(
                        QPointF(float(preview_point.x()), float(preview_point.y())),
                        points[0],
                    )

            vertex_pen = QPen(QColor(10, 18, 28), 1.0)
            vertex_pen.setCosmetic(True)
            for idx, point in enumerate(points):
                fill = QColor(103, 214, 255)
                if (
                    idx == 0
                    and len(points) >= 3
                    and self._polygon_close_hot
                    and not self._polygon_closed
                ):
                    fill = QColor(255, 205, 88)
                painter.setPen(vertex_pen)
                painter.setBrush(fill)
                painter.drawEllipse(
                    QRectF(float(point.x()) - 4.0, float(point.y()) - 4.0, 8.0, 8.0)
                )

            payload = self._polygon_measurement_payload()
            if bool(payload.get("valid", False)) and self._polygon_closed:
                text = f"{payload.get('label', '—')} | {int(payload.get('vertex_count', 0))} вершин"
                xs = [float(point.x()) for point in points]
                ys = [float(point.y()) for point in points]
                label_rect = QRectF(
                    (min(xs) + max(xs)) * 0.5 - 70.0,
                    (min(ys) + max(ys)) * 0.5 - 18.0,
                    140.0,
                    28.0,
                )
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(10, 18, 28, 210))
                painter.drawRoundedRect(label_rect, 8.0, 8.0)
                painter.setPen(QColor(248, 248, 248))
                painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, text)

        painter.restore()


class NavigatorLabel(QLabel):
    stageChanged = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(180, 180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._source_rgb: np.ndarray | None = None
        self._pan_x = 0.5
        self._pan_y = 0.5
        self._magnification = 200
        self._dragging = False

    def set_source(self, image_rgb: np.ndarray | None) -> None:
        self._source_rgb = image_rgb
        self._render()

    def set_view_state(self, pan_x: float, pan_y: float, magnification: int) -> None:
        self._pan_x = float(np.clip(pan_x, 0.0, 1.0))
        self._pan_y = float(np.clip(pan_y, 0.0, 1.0))
        self._magnification = int(max(100, magnification))
        self._render()

    def _render(self) -> None:
        if self._source_rgb is None:
            self.setText("Мини-карта")
            return
        base = Image.fromarray(self._source_rgb)
        base.thumbnail((320, 320), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(base)
        w, h = base.size
        ratio = np.clip(100.0 / float(self._magnification), 0.05, 1.0)
        rw = max(8, int(w * ratio))
        rh = max(8, int(h * ratio))
        cx = int(self._pan_x * (w - 1))
        cy = int(self._pan_y * (h - 1))
        x0 = int(np.clip(cx - rw // 2, 0, w - rw))
        y0 = int(np.clip(cy - rh // 2, 0, h - rh))
        x1 = x0 + rw
        y1 = y0 + rh
        draw.rectangle([x0, y0, x1, y1], outline=(255, 193, 87), width=2)
        arr = np.asarray(base, dtype=np.uint8)
        pix = _to_pixmap(arr)
        self.setPixmap(
            pix.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _emit_from_pos(self, x: int, y: int) -> None:
        pan_x = float(np.clip(x / max(1, self.width() - 1), 0.0, 1.0))
        pan_y = float(np.clip(y / max(1, self.height() - 1), 0.0, 1.0))
        self.stageChanged.emit(pan_x, pan_y)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._emit_from_pos(event.position().x(), event.position().y())

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._dragging:
            self._emit_from_pos(event.position().x(), event.position().y())

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False


class MicroscopeWindow(QMainWindow, WindowModeMixin):
    OBJECTIVES = list(range(100, 601, 10))
    DEFAULT_PROFILE_PATH = Path("profiles") / "microscope_profile_v2.json"
    OPTICAL_MODE_OPTIONS: list[tuple[str, str]] = [
        ("brightfield", "Светлое поле"),
        ("darkfield", "Тёмное поле"),
        ("polarized", "Поляризованный свет"),
        ("phase_contrast", "Фазовый контраст"),
        ("dic", "DIC"),
        ("magnetic_etching", "Магнитное травление"),
    ]
    PSF_PROFILE_OPTIONS: list[tuple[str, str]] = [
        ("standard", "Стандартный ТРФ"),
        ("bessel_extended_dof", "Бессель расш. ГРИП"),
        ("airy_push_pull", "Эйри двухполярный"),
        ("self_rotating", "Самовращающийся"),
        ("stir_sectioning", "СТИР секционирование"),
        ("lens_axicon_hybrid", "Линза-аксикон гибрид"),
    ]
    DEFAULT_MASK_RENDERING = {
        "enabled": True,
        "phase_strength": 1.0,
        "feature_strength": 0.95,
        "prep_strength": 0.9,
    }

    def __init__(self, samples_dir: str | Path | None = None) -> None:
        super().__init__()
        self.samples_dir = Path(samples_dir) if samples_dir else Path("examples") / "factory_v3_output"

        self.current_image_path: Path | None = None
        self.current_meta_path: Path | None = None
        self.current_source_gray: np.ndarray | None = None
        self.current_source_rgb: np.ndarray | None = None
        self.current_source_metadata: dict[str, Any] | None = None
        self.current_manifest_path: Path | None = None
        self.current_mask_entries: list[dict[str, Any]] = []
        self.current_mask_render_summary: dict[str, Any] = {
            "enabled": False,
            "mask_count_total": 0,
        }
        self._mask_fov_cache: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self._mask_fov_cache_max = 64
        self.current_capture: np.ndarray | None = None
        self.current_capture_meta: dict[str, Any] | None = None
        self.current_scale_bar: dict[str, Any] = {
            "enabled": False,
            "um_per_px": 1.0,
            "bar_um": 100.0,
            "bar_px": 100.0,
        }
        self.current_scale_source = "default.assumption"
        self.current_scale_audit: dict[str, Any] = {"ok": True}
        self.measurement_tool_mode = ZoomView.TOOL_OFF
        self.measurement_display_mode = ZoomView.TOOL_LINE
        self.line_measurement_history: list[dict[str, Any]] = []
        self.area_measurement_history: list[dict[str, Any]] = []
        self._line_measurement_counter = 0
        self._area_measurement_counter = 0

        self.session_id = default_session_id()
        self.capture_index = 0
        self.journal_records: list[dict[str, Any]] = []
        self.applied_heavy: dict[str, Any] = {
            "noise_sigma": 7.72,
            "vignette_strength": 0.30,
            "uneven_strength": 0.08,
            "add_dust": False,
            "add_scratches": False,
            "etch_uneven": 0.0,
        }
        self.heavy_dirty = False
        self._spin_slider_links: dict[int, QSlider] = {}
        self._animations: list[QPropertyAnimation] = []
        self._did_intro_animation = False
        self._syncing_optics_widgets = False
        self._view_needs_fit = True
        self._last_view_signature: tuple[Any, ...] | None = None
        self.focus_distance_mm = 18.0
        self.focus_user_configured = False
        self._view_update_timer = QTimer(self)
        self._view_update_timer.setSingleShot(True)
        self._view_update_timer.timeout.connect(self._update_view)
        self._objective_repeat_active = False
        self._render_signals = _AsyncRenderSignals(self)
        self._render_signals.rendered.connect(self._on_render_worker_result)
        self._render_signals.failed.connect(self._on_render_worker_error)
        self._render_lock = threading.Lock()
        self._render_wakeup = threading.Event()
        self._render_thread: threading.Thread | None = None
        self._render_shutdown = False
        self._render_pending_request: dict[str, Any] | None = None
        self._render_request_generation = 0
        self._render_requested_signature: tuple[Any, ...] | None = None

        self.ui_theme_profile_path = MICROSCOPE_THEME_PROFILE
        self.theme_mode = load_theme_mode(self.ui_theme_profile_path, default="light")

        self.setWindowTitle("Виртуальный микроскоп V3")
        self.setMinimumSize(1120, 760)
        self._build_ui()
        self._style(self.theme_mode)
        self._load_profile_on_start()
        self._reset_focus_to_unconfigured(trigger_update=False)
        self._scan_samples()

        # Инициализация управления окнами (вместо self.resize)
        window_state_manager = WindowStateManager("microscope")
        self.setup_window_modes(window_state_manager)

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        self._build_header(root_layout)

        self.main_split = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(self.main_split, stretch=1)

        left_scroll = QScrollArea()
        self.left_scroll = left_scroll
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(430)

        left = QWidget()
        left.setObjectName("leftNavCard")
        left_scroll.setWidget(left)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(620)
        right = QWidget()
        right.setObjectName("rightCard")
        right_scroll.setWidget(right)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(12)

        self.main_split.addWidget(left_scroll)
        self.main_split.addWidget(right_scroll)
        self.main_split.setChildrenCollapsible(False)
        self.main_split.setStretchFactor(0, 13)
        self.main_split.setStretchFactor(1, 24)
        self.main_split.setSizes([520, 960])

        self._build_instrument_bar(left_layout)
        self._build_library_box(left_layout)
        self._build_stage_box(left_layout)
        self._build_save_box(left_layout)
        left_layout.addStretch(1)

        self._build_view_box(right_layout)
        self._build_measurement_box(right_layout)
        self._build_journal_box(right_layout)
        self._build_info_box(right_layout)
        self._apply_responsive_layout()

    def _build_header(self, parent: QVBoxLayout) -> None:
        header = QWidget()
        header.setObjectName("microscopeHeader")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(10)

        self.header_logo_label = QLabel()
        self.header_logo_label.setObjectName("headerLogoLabel")
        self.header_logo_label.setMinimumWidth(54)
        self.header_logo_label.setMaximumWidth(70)
        self.header_logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_path = resolve_branding_logo()
        if logo_path is not None:
            try:
                logo = QPixmap(str(logo_path))
                if not logo.isNull():
                    self.header_logo_label.setPixmap(
                        logo.scaledToHeight(
                            44, Qt.TransformationMode.SmoothTransformation
                        )
                    )
                else:
                    self.header_logo_label.setText("РТУ")
            except Exception:
                self.header_logo_label.setText("РТУ")
        else:
            self.header_logo_label.setText("РТУ")
        layout.addWidget(self.header_logo_label)

        title_wrap = QVBoxLayout()
        title_wrap.setContentsMargins(0, 0, 0, 0)
        title_wrap.setSpacing(1)
        self.header_title_label = QLabel("Материаловедческий комплекс РТУ МИРЭА")
        self.header_title_label.setObjectName("headerBrandPrimary")
        self.header_subtitle_label = QLabel("Виртуальный микроскоп V3")
        self.header_subtitle_label.setObjectName("headerBrandSecondary")
        title_wrap.addWidget(self.header_title_label)
        title_wrap.addWidget(self.header_subtitle_label)
        title_widget = QWidget()
        title_widget.setObjectName("headerTitleWidget")
        title_widget.setLayout(title_wrap)
        layout.addWidget(title_widget, stretch=1)

        self.theme_mode_combo = QComboBox()
        self.theme_mode_combo.addItem("Светлая", "light")
        self.theme_mode_combo.addItem("Тёмная", "dark")
        t_idx = self.theme_mode_combo.findData(str(self.theme_mode))
        if t_idx >= 0:
            self.theme_mode_combo.setCurrentIndex(t_idx)
        self.theme_mode_combo.currentIndexChanged.connect(self._on_theme_mode_changed)
        theme_label = QLabel("Тема")
        theme_label.setObjectName("headerThemeLabel")
        layout.addWidget(theme_label)
        layout.addWidget(self.theme_mode_combo)

        parent.addWidget(header)

    def _bind_slider_to_spin(self, spin: QDoubleSpinBox, steps: int = 1000) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, steps)

        def spin_to_slider(value: float) -> None:
            low = float(spin.minimum())
            high = float(spin.maximum())
            if high <= low:
                return
            pos = int(round((float(value) - low) / (high - low) * steps))
            slider.blockSignals(True)
            slider.setValue(max(0, min(steps, pos)))
            slider.blockSignals(False)

        def slider_to_spin(pos: int) -> None:
            low = float(spin.minimum())
            high = float(spin.maximum())
            if high <= low:
                return
            value = low + (high - low) * (float(pos) / float(steps))
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
            spin.valueChanged.emit(spin.value())

        spin.valueChanged.connect(spin_to_slider)
        slider.valueChanged.connect(slider_to_spin)
        spin_to_slider(spin.value())
        self._spin_slider_links[id(spin)] = slider
        return slider

    def _slider_for_spin(self, spin: QDoubleSpinBox) -> QSlider | None:
        return self._spin_slider_links.get(id(spin))

    def _build_instrument_bar(self, parent: QVBoxLayout) -> None:
        box = QGroupBox("Приборная панель")
        box.setObjectName("instrumentCard")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)

        self.objective_down_btn = QPushButton("▼")
        self.objective_up_btn = QPushButton("▲")
        for btn in (self.objective_down_btn, self.objective_up_btn):
            btn.setFixedSize(28, 26)
            btn.setObjectName("compactSquareCta")
            btn.setAutoRepeat(True)
            btn.setAutoRepeatDelay(220)
            btn.setAutoRepeatInterval(45)
        self.objective_down_btn.pressed.connect(self._on_objective_button_press)
        self.objective_up_btn.pressed.connect(self._on_objective_button_press)
        self.objective_down_btn.released.connect(self._on_objective_button_release)
        self.objective_up_btn.released.connect(self._on_objective_button_release)
        self.objective_down_btn.clicked.connect(
            lambda: self._step_objective(
                -1, immediate=not self._objective_repeat_active
            )
        )
        self.objective_up_btn.clicked.connect(
            lambda: self._step_objective(1, immediate=not self._objective_repeat_active)
        )

        self.objective_spin = QSpinBox()
        self.objective_spin.setObjectName("instrumentSpin")
        self.objective_spin.setRange(self.OBJECTIVES[0], self.OBJECTIVES[-1])
        self.objective_spin.setSingleStep(10)
        self.objective_spin.setSuffix("x")
        self.objective_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.objective_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.objective_spin.setFixedWidth(132)
        self.objective_spin.setAccelerated(True)
        self.objective_spin.valueChanged.connect(self._on_objective_spin_changed)

        objective_row = QHBoxLayout()
        objective_row.setContentsMargins(0, 0, 0, 0)
        objective_row.setSpacing(4)
        objective_row.addWidget(self.objective_down_btn)
        objective_row.addWidget(self.objective_up_btn)
        objective_row.addWidget(self.objective_spin)
        objective_row.addStretch(1)
        grid.addWidget(QLabel("Приближение"), 0, 0)
        grid.addLayout(objective_row, 0, 1, 1, 2)

        self.focus_down_btn = QPushButton("▼")
        self.focus_up_btn = QPushButton("▲")
        for btn in (self.focus_down_btn, self.focus_up_btn):
            btn.setFixedSize(28, 26)
            btn.setObjectName("compactSquareCta")
            btn.setAutoRepeat(True)
            btn.setAutoRepeatDelay(220)
            btn.setAutoRepeatInterval(45)
        self.focus_down_btn.clicked.connect(lambda: self._step_focus_dial(-1))
        self.focus_up_btn.clicked.connect(lambda: self._step_focus_dial(1))

        self.focus_distance_spin = QDoubleSpinBox()
        self.focus_distance_spin.setObjectName("instrumentSpin")
        self.focus_distance_spin.setDecimals(2)
        self.focus_distance_spin.setSingleStep(0.05)
        self.focus_distance_spin.setSuffix(" mm")
        self.focus_distance_spin.setButtonSymbols(
            QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        self.focus_distance_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.focus_distance_spin.setFixedWidth(136)
        self.focus_distance_spin.setAccelerated(True)
        focus_tooltip = (
            "Базовое фокусное расстояние задаётся выбранным объективом. "
            "Целевой фокус = базовое расстояние + ((X - 0.5) × 1.4) + ((Y - 0.5) × -1.0) мм. "
            "Чем выше увеличение, тем меньше допустимая ошибка по фокусу."
        )
        self.focus_distance_spin.setToolTip(focus_tooltip)
        self.focus_distance_spin.valueChanged.connect(
            self._on_focus_distance_spin_changed
        )
        self.focus_help_btn = QPushButton("?")
        self.focus_help_btn.setFixedSize(24, 24)
        self.focus_help_btn.setObjectName("compactSquareCta")
        self.focus_help_btn.setToolTip(focus_tooltip)

        focus_row = QHBoxLayout()
        focus_row.setContentsMargins(0, 0, 0, 0)
        focus_row.setSpacing(4)
        focus_row.addWidget(self.focus_down_btn)
        focus_row.addWidget(self.focus_up_btn)
        focus_row.addWidget(self.focus_distance_spin)
        focus_row.addWidget(self.focus_help_btn)
        focus_row.addStretch(1)
        grid.addWidget(QLabel("Фокус"), 1, 0)
        grid.addLayout(focus_row, 1, 1, 1, 2)

        self.brightness = QDoubleSpinBox()
        self.brightness.setObjectName("instrumentSpin")
        self.brightness.setRange(0.5, 1.8)
        self.brightness.setSingleStep(0.01)
        self.brightness.setValue(1.0)
        self.brightness.valueChanged.connect(self._on_live_control_changed)
        brightness_slider = self._bind_slider_to_spin(self.brightness)
        self.brightness.setFixedWidth(104)
        self.brightness.setProperty("wheelEnabled", True)

        self.contrast = QDoubleSpinBox()
        self.contrast.setObjectName("instrumentSpin")
        self.contrast.setRange(0.5, 2.2)
        self.contrast.setSingleStep(0.01)
        self.contrast.setValue(1.1)
        self.contrast.valueChanged.connect(self._on_live_control_changed)
        contrast_slider = self._bind_slider_to_spin(self.contrast)
        self.contrast.setFixedWidth(104)
        self.contrast.setProperty("wheelEnabled", True)

        self.optical_mode_combo = QComboBox()
        for key, label in self.OPTICAL_MODE_OPTIONS:
            self.optical_mode_combo.addItem(label, key)
        self.optical_mode_combo.currentIndexChanged.connect(
            self._on_live_control_changed
        )
        self.darkfield_scatter_spin = QDoubleSpinBox()
        self.darkfield_scatter_spin.setObjectName("instrumentSpin")
        self.darkfield_scatter_spin.setRange(0.5, 2.0)
        self.darkfield_scatter_spin.setSingleStep(0.05)
        self.darkfield_scatter_spin.setValue(1.0)
        self.darkfield_scatter_spin.valueChanged.connect(self._on_live_control_changed)
        self.polarized_crossed_check = QCheckBox("Скрещённые поляризаторы")
        self.polarized_crossed_check.setChecked(True)
        self.polarized_crossed_check.toggled.connect(self._on_live_control_changed)
        self.phase_plate_combo = QComboBox()
        self.phase_plate_combo.addItem("Положительная пластина", "positive")
        self.phase_plate_combo.addItem("Отрицательная пластина", "negative")
        self.phase_plate_combo.currentIndexChanged.connect(
            self._on_live_control_changed
        )

        self.psf_profile_combo = QComboBox()
        for key, label in self.PSF_PROFILE_OPTIONS:
            self.psf_profile_combo.addItem(label, key)
        self.psf_profile_combo.currentIndexChanged.connect(
            self._on_live_control_changed
        )
        self.psf_strength_spin = QDoubleSpinBox()
        self.psf_strength_spin.setObjectName("instrumentSpin")
        self.psf_strength_spin.setRange(0.0, 1.0)
        self.psf_strength_spin.setSingleStep(0.05)
        self.psf_strength_spin.setValue(0.0)
        self.psf_strength_spin.valueChanged.connect(self._on_live_control_changed)
        self.sectioning_shear_spin = QDoubleSpinBox()
        self.sectioning_shear_spin.setObjectName("instrumentSpin")
        self.sectioning_shear_spin.setRange(0.0, 90.0)
        self.sectioning_shear_spin.setSingleStep(1.0)
        self.sectioning_shear_spin.setValue(DEFAULT_SECTIONING_SHEAR_DEG)
        self.sectioning_shear_spin.valueChanged.connect(self._on_live_control_changed)
        self.hybrid_balance_spin = QDoubleSpinBox()
        self.hybrid_balance_spin.setObjectName("instrumentSpin")
        self.hybrid_balance_spin.setRange(0.0, 1.0)
        self.hybrid_balance_spin.setSingleStep(0.05)
        self.hybrid_balance_spin.setValue(DEFAULT_HYBRID_BALANCE)
        self.hybrid_balance_spin.valueChanged.connect(self._on_live_control_changed)

        grid.addWidget(QLabel("Яркость"), 2, 0)
        grid.addWidget(self.brightness, 2, 1)
        grid.addWidget(brightness_slider, 2, 2)
        grid.addWidget(QLabel("Контраст"), 3, 0)
        grid.addWidget(self.contrast, 3, 1)
        grid.addWidget(contrast_slider, 3, 2)
        grid.addWidget(QLabel("Оптический режим"), 4, 0)
        grid.addWidget(self.optical_mode_combo, 4, 1, 1, 2)
        lbl_darkfield = QLabel("Чувствительность\nтёмного поля")
        lbl_darkfield.setWordWrap(True)
        grid.addWidget(lbl_darkfield, 5, 0)
        grid.addWidget(self.darkfield_scatter_spin, 5, 1, 1, 2)
        lbl_polarization = QLabel("Поляризация /\nфазовая пластина")
        lbl_polarization.setWordWrap(True)
        grid.addWidget(lbl_polarization, 6, 0)
        polarized_row = QHBoxLayout()
        polarized_row.setContentsMargins(0, 0, 0, 0)
        polarized_row.setSpacing(6)
        polarized_row.addWidget(self.polarized_crossed_check)
        polarized_row.addWidget(self.phase_plate_combo)
        polarized_row.addStretch(1)
        grid.addLayout(polarized_row, 6, 1, 1, 2)
        grid.addWidget(QLabel("Профиль ТРФ"), 7, 0)
        grid.addWidget(self.psf_profile_combo, 7, 1, 1, 2)
        grid.addWidget(QLabel("Сила ТРФ"), 8, 0)
        grid.addWidget(self.psf_strength_spin, 8, 1, 1, 2)
        lbl_shear = QLabel("Сдвиг\nсекционирования")
        lbl_shear.setWordWrap(True)
        grid.addWidget(lbl_shear, 9, 0)
        grid.addWidget(self.sectioning_shear_spin, 9, 1, 1, 2)
        grid.addWidget(QLabel("Гибридный баланс"), 10, 0)
        grid.addWidget(self.hybrid_balance_spin, 10, 1, 1, 2)

        self.capture_btn = QPushButton("Снимок (Пробел)")
        self.capture_btn.setObjectName("primaryCta")
        self.capture_btn.setMinimumHeight(30)
        self.capture_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        )
        self.capture_btn.clicked.connect(self._save_capture)
        self.heavy_state_label = QLabel("Эффекты: фиксированные")
        self.heavy_state_label.setObjectName("statusPill")
        grid.addWidget(self.capture_btn, 11, 0, 1, 3)
        grid.addWidget(self.heavy_state_label, 12, 0, 1, 3)

        self.status_objective_label = QLabel("Объектив: 200x")
        self.status_objective_label.setObjectName("statusPill")
        self.status_focus_label = QLabel("Фокус: не настроен")
        self.status_focus_label.setObjectName("statusPill")
        self.status_stage_label = QLabel("XY: 0.50, 0.50")
        self.status_stage_label.setObjectName("statusPill")
        self.status_psf_label = QLabel("ТРФ: стандартный")
        self.status_psf_label.setObjectName("statusPill")
        self.status_axial_profile_label = QLabel("Осевой: стандартный")
        self.status_axial_profile_label.setObjectName("statusPill")
        self.status_sectioning_label = QLabel("Секционирование: неактивно")
        self.status_sectioning_label.setObjectName("statusPill")
        self.status_dof_label = QLabel("Коэффициент ГРИП: 1.00")
        self.status_dof_label.setObjectName("statusPill")
        self.status_shift_label = QLabel("Осевой сдвиг: 0.000")
        self.status_shift_label.setObjectName("statusPill")
        grid.addWidget(self.status_objective_label, 13, 0, 1, 3)
        grid.addWidget(self.status_focus_label, 14, 0, 1, 3)
        grid.addWidget(self.status_stage_label, 15, 0, 1, 3)
        grid.addWidget(self.status_psf_label, 16, 0, 1, 3)
        grid.addWidget(self.status_axial_profile_label, 17, 0, 1, 3)
        grid.addWidget(self.status_sectioning_label, 18, 0, 1, 3)
        grid.addWidget(self.status_dof_label, 19, 0, 1, 3)
        grid.addWidget(self.status_shift_label, 20, 0, 1, 3)

        parent.addWidget(box)
        self._set_objective(200, immediate=False)
        self._reset_focus_to_unconfigured(trigger_update=False)

    def _build_library_box(self, parent: QVBoxLayout) -> None:
        box = QGroupBox("Библиотека образцов")
        layout = QVBoxLayout(box)
        top = QHBoxLayout()
        self.dir_edit = QLineEdit(str(self.samples_dir))
        browse_btn = QPushButton("Обзор")
        reload_btn = QPushButton("Обновить")
        browse_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        )
        reload_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        )
        browse_btn.setToolTip("Выбрать папку с образцами")
        reload_btn.setToolTip("Обновить список образцов")
        browse_btn.clicked.connect(self._browse_samples_dir)
        reload_btn.clicked.connect(self._scan_samples)
        top.addWidget(self.dir_edit)
        top.addWidget(browse_btn)
        top.addWidget(reload_btn)
        layout.addLayout(top)
        self.sample_list = QListWidget()
        self.sample_list.itemSelectionChanged.connect(self._on_select_sample)
        layout.addWidget(self.sample_list)
        parent.addWidget(box)

    def _build_stage_box(self, parent: QVBoxLayout) -> None:
        box = QGroupBox("Столик XY и мини-карта")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        self.navigator = NavigatorLabel()
        self.navigator.stageChanged.connect(self._set_stage_xy)
        top_row.addWidget(self.navigator, stretch=1)

        controls_wrap = QWidget()
        controls_layout = QVBoxLayout(controls_wrap)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        self.stage_x = QDoubleSpinBox()
        self.stage_x.setRange(0.0, 1.0)
        self.stage_x.setSingleStep(0.01)
        self.stage_x.setValue(0.5)
        self.stage_x.valueChanged.connect(self._on_live_control_changed)
        stage_x_slider = self._bind_slider_to_spin(self.stage_x)
        self.stage_y = QDoubleSpinBox()
        self.stage_y.setRange(0.0, 1.0)
        self.stage_y.setSingleStep(0.01)
        self.stage_y.setValue(0.5)
        self.stage_y.valueChanged.connect(self._on_live_control_changed)
        stage_y_slider = self._bind_slider_to_spin(self.stage_y)

        stage_grid = QGridLayout()
        stage_grid.setHorizontalSpacing(6)
        stage_grid.setVerticalSpacing(6)
        stage_grid.addWidget(QLabel("X"), 0, 0)
        stage_grid.addWidget(self.stage_x, 0, 1)
        stage_grid.addWidget(stage_x_slider, 0, 2)
        stage_grid.addWidget(QLabel("Y"), 1, 0)
        stage_grid.addWidget(self.stage_y, 1, 1)
        stage_grid.addWidget(stage_y_slider, 1, 2)
        controls_layout.addLayout(stage_grid)

        dpad = QGridLayout()
        dpad.setHorizontalSpacing(4)
        dpad.setVerticalSpacing(4)
        up_btn = QPushButton("↑")
        down_btn = QPushButton("↓")
        left_btn = QPushButton("←")
        right_btn = QPushButton("→")
        self.center_stage_btn = QPushButton("Центр")
        for btn in (up_btn, down_btn, left_btn, right_btn, self.center_stage_btn):
            btn.setObjectName("secondaryCta")
        up_btn.clicked.connect(
            lambda: self._move_stage(0.0, -self.stage_step_small.value())
        )
        down_btn.clicked.connect(
            lambda: self._move_stage(0.0, self.stage_step_small.value())
        )
        left_btn.clicked.connect(
            lambda: self._move_stage(-self.stage_step_small.value(), 0.0)
        )
        right_btn.clicked.connect(
            lambda: self._move_stage(self.stage_step_small.value(), 0.0)
        )
        self.center_stage_btn.clicked.connect(lambda: self._set_stage_xy(0.5, 0.5))
        dpad.addWidget(up_btn, 0, 1)
        dpad.addWidget(left_btn, 1, 0)
        dpad.addWidget(self.center_stage_btn, 1, 1)
        dpad.addWidget(right_btn, 1, 2)
        dpad.addWidget(down_btn, 2, 1)
        controls_layout.addLayout(dpad)

        self.stage_step_small = QDoubleSpinBox()
        self.stage_step_small.setRange(0.001, 0.2)
        self.stage_step_small.setSingleStep(0.001)
        self.stage_step_small.setValue(0.01)
        self.stage_step_large = QDoubleSpinBox()
        self.stage_step_large.setRange(0.005, 0.5)
        self.stage_step_large.setSingleStep(0.005)
        self.stage_step_large.setValue(0.05)

        steps_grid = QGridLayout()
        steps_grid.setHorizontalSpacing(6)
        steps_grid.setVerticalSpacing(6)
        steps_grid.addWidget(QLabel("Шаг малый"), 0, 0)
        steps_grid.addWidget(self.stage_step_small, 0, 1)
        steps_grid.addWidget(QLabel("Шаг крупный"), 1, 0)
        steps_grid.addWidget(self.stage_step_large, 1, 1)
        controls_layout.addLayout(steps_grid)
        top_row.addWidget(controls_wrap, stretch=1)

        layout.addLayout(top_row)
        hotkeys = QLabel(
            "Горячие клавиши: W/A/S/D — малый шаг, Shift+W/A/S/D — крупный шаг, +/- — приближение, Пробел — снимок"
        )
        hotkeys.setWordWrap(True)
        layout.addWidget(hotkeys)
        parent.addWidget(box)

    def _build_save_box(self, parent: QVBoxLayout) -> None:
        box = QGroupBox("Оверлеи и сохранение")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        self.reticle_check = QCheckBox("Прицельная метка")
        self.reticle_check.setChecked(True)
        self.reticle_check.toggled.connect(self._on_live_control_changed)
        self.scale_check = QCheckBox("Масштабная шкала")
        self.scale_check.setChecked(True)
        self.scale_check.toggled.connect(self._on_live_control_changed)
        overlay_row = QHBoxLayout()
        overlay_row.addWidget(self.reticle_check)
        overlay_row.addWidget(self.scale_check)
        overlay_row.addStretch(1)
        layout.addLayout(overlay_row)

        self.out_h = QSpinBox()
        self.out_h.setRange(256, 8192)
        self.out_h.setValue(1024)
        self.out_h.valueChanged.connect(self._on_live_control_changed)
        self.out_w = QSpinBox()
        self.out_w.setRange(256, 8192)
        self.out_w.setValue(1024)
        self.out_w.valueChanged.connect(self._on_live_control_changed)

        self.save_dir = QLineEdit(str(Path("examples") / "microscope_v2_captures"))
        browse_btn = QPushButton("Папка")
        browse_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        )
        browse_btn.setToolTip("Выбрать папку сохранения")
        browse_btn.clicked.connect(self._browse_save_dir)
        save_row = QHBoxLayout()
        save_row.addWidget(self.save_dir)
        save_row.addWidget(browse_btn)
        wrap = QWidget()
        wrap.setLayout(save_row)
        self.save_prefix = QLineEdit("capture_v2")

        self.profile_path_edit = QLineEdit(str(self.DEFAULT_PROFILE_PATH))
        profile_btns = QHBoxLayout()
        self.load_profile_btn = QPushButton("Загрузить профиль")
        self.save_profile_btn = QPushButton("Сохранить профиль")
        self.load_profile_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton)
        )
        self.save_profile_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        )
        self.load_profile_btn.clicked.connect(self._load_profile_from_ui)
        self.save_profile_btn.clicked.connect(self._save_profile_from_ui)
        profile_btns.addWidget(self.load_profile_btn)
        profile_btns.addWidget(self.save_profile_btn)
        profile_wrap = QWidget()
        profile_wrap.setLayout(profile_btns)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Выход (высота)", self.out_h)
        form.addRow("Выход (ширина)", self.out_w)
        form.addRow("Папка", wrap)
        form.addRow("Префикс", self.save_prefix)
        form.addRow("Файл профиля", self.profile_path_edit)
        layout.addLayout(form)
        layout.addWidget(profile_wrap)
        parent.addWidget(box)

    def _build_view_box(self, parent: QVBoxLayout) -> None:
        box = QGroupBox("Окуляр")
        box.setMinimumHeight(420)
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        self.fit_view_btn = QPushButton("Подогнать")
        self.fit_view_btn.setObjectName("secondaryCta")
        self.fit_view_btn.clicked.connect(self._fit_view_to_scene)
        self.reset_zoom_btn = QPushButton("1:1")
        self.reset_zoom_btn.setObjectName("secondaryCta")
        self.reset_zoom_btn.clicked.connect(self._reset_view_zoom)
        self.measure_mode_btn = QPushButton("Линейка")
        self.measure_mode_btn.setCheckable(True)
        self.measure_mode_btn.setObjectName("secondaryCta")
        self.measure_mode_btn.toggled.connect(self._on_measurement_mode_toggled)
        self.area_measure_mode_btn = QPushButton("Автолиния")
        self.area_measure_mode_btn.setCheckable(True)
        self.area_measure_mode_btn.setObjectName("secondaryCta")
        self.area_measure_mode_btn.toggled.connect(
            self._on_area_measurement_mode_toggled
        )
        self.scale_toolbar_label = QLabel("—")
        self.scale_toolbar_label.setObjectName("statusPill")
        toolbar.addWidget(self.fit_view_btn)
        toolbar.addWidget(self.reset_zoom_btn)
        toolbar.addWidget(self.measure_mode_btn)
        toolbar.addWidget(self.area_measure_mode_btn)
        toolbar.addStretch(1)
        toolbar.addWidget(QLabel("Масштаб"))
        toolbar.addWidget(self.scale_toolbar_label)
        layout.addLayout(toolbar)

        self.view_scene = QGraphicsScene(self)
        self.view = ZoomView()
        self.view.setScene(self.view_scene)
        self.view.setMinimumHeight(360)
        self.view.measurementChanged.connect(self._on_measurement_changed)
        self.view.measurementFinished.connect(self._on_measurement_finished)
        self.view.measurementCleared.connect(self._on_measurement_cleared)
        layout.addWidget(self.view)
        parent.addWidget(box, stretch=10)

    def _build_measurement_box(self, parent: QVBoxLayout) -> None:
        box = QGroupBox("Измерения и масштаб")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        self.measure_mode_status_label = QLabel("Инструмент: выкл")
        self.measure_mode_status_label.setObjectName("statusPill")
        self.scale_validation_label = QLabel("Масштаб: —")
        self.scale_validation_label.setObjectName("statusPill")
        status_row.addWidget(self.measure_mode_status_label)
        status_row.addWidget(self.scale_validation_label)
        status_row.addStretch(1)
        layout.addLayout(status_row)

        self.measurement_current_label = QLabel("Текущее измерение: —")
        self.measurement_current_label.setWordWrap(True)
        self.measurement_average_label = QLabel("Среднее: —")
        self.measurement_average_label.setWordWrap(True)
        self.measurement_scale_source_label = QLabel("Калибровка: —")
        self.measurement_scale_source_label.setWordWrap(True)
        layout.addWidget(self.measurement_current_label)
        layout.addWidget(self.measurement_average_label)
        layout.addWidget(self.measurement_scale_source_label)

        hint = QLabel(
            "ЛКМ по изображению — линейка или контур площади. Для площади замкните контур кликом по первой точке, Backspace удаляет последнюю вершину."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        controls_row = QHBoxLayout()
        layout.addLayout(controls_row)

        line_row = QHBoxLayout()
        line_row.setSpacing(6)
        line_row.addWidget(QLabel("Линейка"))
        line_row.addStretch(1)
        self.measure_clear_line_btn = QPushButton("Очистить линейку")
        self.measure_clear_line_btn.setObjectName("secondaryCta")
        self.measure_clear_line_btn.clicked.connect(self._clear_line_measurements)
        line_row.addWidget(self.measure_clear_line_btn)
        layout.addLayout(line_row)

        self.line_measurement_table = QTableWidget(0, 4)
        self.line_measurement_table.setMinimumHeight(110)
        self.line_measurement_table.setMaximumHeight(170)
        layout.addWidget(self.line_measurement_table)

        area_row = QHBoxLayout()
        area_row.setSpacing(6)
        area_row.addWidget(QLabel("Автолиния"))
        area_row.addStretch(1)
        self.measure_clear_area_btn = QPushButton("Очистить автолинию")
        self.measure_clear_area_btn.setObjectName("secondaryCta")
        self.measure_clear_area_btn.clicked.connect(self._clear_area_measurements)
        area_row.addWidget(self.measure_clear_area_btn)
        layout.addLayout(area_row)

        self.area_measurement_table = QTableWidget(0, 4)
        self.area_measurement_table.setMinimumHeight(110)
        self.area_measurement_table.setMaximumHeight(170)
        layout.addWidget(self.area_measurement_table)
        self._configure_measurement_tables()
        parent.addWidget(box, stretch=1)

    def _build_journal_box(self, parent: QVBoxLayout) -> None:
        box = QGroupBox("Журнал сессии")
        layout = QVBoxLayout(box)
        self.journal = QTableWidget(0, 7)
        self.journal.setHorizontalHeaderLabels(
            ["Время", "Образец", "Объектив", "Фокус, мм", "X/Y", "Стадия", "Файл"]
        )
        self.journal.verticalHeader().setVisible(False)
        self.journal.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.journal.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.journal.horizontalHeader().setSectionResizeMode(
            6, QHeaderView.ResizeMode.Stretch
        )
        self.journal.horizontalHeader().setMinimumSectionSize(90)
        self.journal.setMinimumHeight(120)
        self.journal.cellDoubleClicked.connect(self._restore_journal_row)
        layout.addWidget(self.journal)
        parent.addWidget(box, stretch=1)

    def _build_info_box(self, parent: QVBoxLayout) -> None:
        box = QGroupBox("Карточка кадра")
        layout = QVBoxLayout(box)
        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setMinimumHeight(330)
        layout.addWidget(self.info)
        parent.addWidget(box, stretch=1)

    def _fit_view_to_scene(self) -> None:
        if not hasattr(self, "view") or not hasattr(self, "view_scene"):
            return
        if self.view_scene.sceneRect().isNull():
            return
        self.view.resetTransform()
        self.view.fitInView(
            self.view_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

    def _reset_view_zoom(self) -> None:
        if hasattr(self, "view"):
            self.view.resetTransform()

    def _set_measurement_tool_mode(self, tool: str) -> None:
        mode = str(tool or ZoomView.TOOL_OFF).strip().lower()
        if mode not in {
            ZoomView.TOOL_OFF,
            ZoomView.TOOL_LINE,
            ZoomView.TOOL_POLYGON_AREA,
        }:
            mode = ZoomView.TOOL_OFF
        self.measurement_tool_mode = mode
        if mode in {ZoomView.TOOL_LINE, ZoomView.TOOL_POLYGON_AREA}:
            self.measurement_display_mode = mode
        self._sync_measurement_table_compat()
        if hasattr(self, "view"):
            self.view.set_measurement_tool(mode)
        if hasattr(self, "measure_mode_btn"):
            self.measure_mode_btn.blockSignals(True)
            self.measure_mode_btn.setChecked(mode == ZoomView.TOOL_LINE)
            self.measure_mode_btn.blockSignals(False)
        if hasattr(self, "area_measure_mode_btn"):
            self.area_measure_mode_btn.blockSignals(True)
            self.area_measure_mode_btn.setChecked(mode == ZoomView.TOOL_POLYGON_AREA)
            self.area_measure_mode_btn.blockSignals(False)
        if hasattr(self, "measure_mode_status_label"):
            mode_label = {
                ZoomView.TOOL_OFF: "Инструмент: выкл",
                ZoomView.TOOL_LINE: "Инструмент: линейка",
                ZoomView.TOOL_POLYGON_AREA: "Инструмент: площадь",
            }
            self.measure_mode_status_label.setText(
                mode_label.get(mode, "Инструмент: выкл")
            )
        self._refresh_measurement_tables()
        self._on_measurement_cleared()

    def _on_measurement_mode_toggled(self, checked: bool) -> None:
        if checked:
            self._set_measurement_tool_mode(ZoomView.TOOL_LINE)
        elif self.measurement_tool_mode == ZoomView.TOOL_LINE:
            self._set_measurement_tool_mode(ZoomView.TOOL_OFF)

    def _on_area_measurement_mode_toggled(self, checked: bool) -> None:
        if checked:
            self._set_measurement_tool_mode(ZoomView.TOOL_POLYGON_AREA)
        elif self.measurement_tool_mode == ZoomView.TOOL_POLYGON_AREA:
            self._set_measurement_tool_mode(ZoomView.TOOL_OFF)

    def _active_measurement_history(self) -> list[dict[str, Any]]:
        if self.measurement_display_mode == ZoomView.TOOL_POLYGON_AREA:
            return self.area_measurement_history
        if self.measurement_display_mode == ZoomView.TOOL_LINE:
            return self.line_measurement_history
        return self.line_measurement_history

    def _sync_measurement_table_compat(self) -> None:
        if self.measurement_display_mode == ZoomView.TOOL_POLYGON_AREA and hasattr(
            self, "area_measurement_table"
        ):
            self.measurement_table = self.area_measurement_table
            return
        if hasattr(self, "line_measurement_table"):
            self.measurement_table = self.line_measurement_table

    def _configure_measurement_table(
        self,
        table: QTableWidget,
        headers: list[str],
    ) -> None:
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )

    def _configure_measurement_tables(self) -> None:
        if not hasattr(self, "line_measurement_table") or not hasattr(
            self, "area_measurement_table"
        ):
            return
        self._configure_measurement_table(
            self.line_measurement_table, ["#", "Длина", "px", "Угол"]
        )
        self._configure_measurement_table(
            self.area_measurement_table, ["#", "Площадь", "px²", "Вершины"]
        )
        self._sync_measurement_table_compat()
        self._refresh_measurement_tables()

    def _configure_measurement_table_for_active_tool(self) -> None:
        self._sync_measurement_table_compat()
        self._refresh_measurement_tables()

    def _refresh_line_measurement_table(self) -> None:
        if not hasattr(self, "line_measurement_table"):
            return
        history = self.line_measurement_history
        self.line_measurement_table.setRowCount(len(history))
        for row, item in enumerate(history):
            self.line_measurement_table.setItem(
                row, 0, QTableWidgetItem(str(item["index"]))
            )
            self.line_measurement_table.setItem(
                row, 1, QTableWidgetItem(str(item["label"]))
            )
            self.line_measurement_table.setItem(
                row, 2, QTableWidgetItem(f"{float(item.get('length_px', 0.0)):.1f}")
            )
            self.line_measurement_table.setItem(
                row, 3, QTableWidgetItem(f"{float(item.get('angle_deg', 0.0)):.1f}°")
            )
        self.line_measurement_table.scrollToBottom()

    def _refresh_area_measurement_table(self) -> None:
        if not hasattr(self, "area_measurement_table"):
            return
        history = self.area_measurement_history
        self.area_measurement_table.setRowCount(len(history))
        for row, item in enumerate(history):
            self.area_measurement_table.setItem(
                row, 0, QTableWidgetItem(str(item["index"]))
            )
            self.area_measurement_table.setItem(
                row, 1, QTableWidgetItem(str(item["label"]))
            )
            self.area_measurement_table.setItem(
                row, 2, QTableWidgetItem(f"{float(item.get('area_px2', 0.0)):.1f}")
            )
            self.area_measurement_table.setItem(
                row, 3, QTableWidgetItem(str(int(item.get("vertex_count", 0))))
            )
        self.area_measurement_table.scrollToBottom()

    def _refresh_measurement_tables(self) -> None:
        self._sync_measurement_table_compat()
        self._refresh_line_measurement_table()
        self._refresh_area_measurement_table()

    def _refresh_measurement_table(self) -> None:
        self._sync_measurement_table_compat()
        self._refresh_measurement_tables()

    def _clear_all_measurements(self) -> None:
        self.line_measurement_history = []
        self.area_measurement_history = []
        self._line_measurement_counter = 0
        self._area_measurement_counter = 0
        if hasattr(self, "line_measurement_table"):
            self.line_measurement_table.setRowCount(0)
        if hasattr(self, "area_measurement_table"):
            self.area_measurement_table.setRowCount(0)
        if hasattr(self, "view"):
            self.view.clear_measurement(emit_signal=False)
        self._on_measurement_cleared()

    def _clear_line_measurements(self) -> None:
        self.line_measurement_history = []
        self._line_measurement_counter = 0
        if hasattr(self, "line_measurement_table"):
            self.line_measurement_table.setRowCount(0)
        if hasattr(self, "view") and self.measurement_tool_mode == ZoomView.TOOL_LINE:
            self.view.clear_measurement(emit_signal=False)
        self._on_measurement_cleared()

    def _clear_area_measurements(self) -> None:
        self.area_measurement_history = []
        self._area_measurement_counter = 0
        if hasattr(self, "area_measurement_table"):
            self.area_measurement_table.setRowCount(0)
        if (
            hasattr(self, "view")
            and self.measurement_tool_mode == ZoomView.TOOL_POLYGON_AREA
        ):
            self.view.clear_measurement(emit_signal=False)
        self._on_measurement_cleared()

    def _clear_measurements(self) -> None:
        if self.measurement_display_mode == ZoomView.TOOL_POLYGON_AREA:
            self._clear_area_measurements()
            return
        self._clear_line_measurements()

    def _on_measurement_changed(self, payload: dict[str, Any]) -> None:
        if not hasattr(self, "measurement_current_label"):
            return
        kind = str(payload.get("kind", self.measurement_tool_mode))
        if kind == ZoomView.TOOL_POLYGON_AREA:
            vertex_count = int(payload.get("vertex_count", 0))
            if bool(payload.get("valid", False)):
                self.measurement_current_label.setText(
                    f"Текущая площадь: {payload.get('label', '—')} | {vertex_count} вершин"
                )
                return
            if vertex_count > 0:
                self.measurement_current_label.setText(
                    f"Текущий контур: {vertex_count} вершин"
                )
                return
            self.measurement_current_label.setText("Текущая площадь: —")
            return
        if not bool(payload.get("valid", False)):
            self.measurement_current_label.setText("Текущий отрезок: —")
            return
        self.measurement_current_label.setText(
            f"Текущий отрезок: {payload.get('label', '—')} | {float(payload.get('length_px', 0.0)):.1f} px | "
            f"{float(payload.get('angle_deg', 0.0)):.1f}°"
        )

    def _append_line_measurement_history(self, payload: dict[str, Any]) -> None:
        self._line_measurement_counter += 1
        entry = {
            "index": int(self._line_measurement_counter),
            "label": str(payload.get("label", "—")),
            "length_px": float(payload.get("length_px", 0.0)),
            "length_um": float(payload.get("length_um", 0.0)),
            "angle_deg": float(payload.get("angle_deg", 0.0)),
            "x0_px": float(payload.get("x0_px", 0.0)),
            "y0_px": float(payload.get("y0_px", 0.0)),
            "x1_px": float(payload.get("x1_px", 0.0)),
            "y1_px": float(payload.get("y1_px", 0.0)),
        }
        self.line_measurement_history.append(entry)
        if len(self.line_measurement_history) > 20:
            self.line_measurement_history = self.line_measurement_history[-20:]

    def _append_area_measurement_history(self, payload: dict[str, Any]) -> None:
        self._area_measurement_counter += 1
        entry = {
            "index": int(self._area_measurement_counter),
            "label": str(payload.get("label", "—")),
            "area_px2": float(payload.get("area_px2", 0.0)),
            "area_um2": float(payload.get("area_um2", 0.0)),
            "perimeter_px": float(payload.get("perimeter_px", 0.0)),
            "perimeter_um": float(payload.get("perimeter_um", 0.0)),
            "vertex_count": int(payload.get("vertex_count", 0)),
            "vertices_px": list(payload.get("vertices_px", [])),
        }
        self.area_measurement_history.append(entry)
        if len(self.area_measurement_history) > 20:
            self.area_measurement_history = self.area_measurement_history[-20:]

    def _update_measurement_average(self) -> None:
        if not hasattr(self, "measurement_average_label"):
            return
        line_count = len(self.line_measurement_history)
        area_count = len(self.area_measurement_history)
        if line_count == 0 and area_count == 0:
            self.measurement_average_label.setText("Среднее: —")
            return
        lines: list[str] = []
        if line_count:
            avg_um = sum(
                float(item.get("length_um", 0.0))
                for item in self.line_measurement_history
            ) / max(1, line_count)
            lines.append(
                f"Линейка: {line_count} изм., среднее {format_metric_length_um(avg_um)}"
            )
        else:
            lines.append("Линейка: —")
        if area_count:
            avg_area_um2 = sum(
                float(item.get("area_um2", 0.0))
                for item in self.area_measurement_history
            ) / max(1, area_count)
            lines.append(
                f"Автолиния: {area_count} изм., средняя площадь {format_metric_area_um2(avg_area_um2)}"
            )
        else:
            lines.append("Автолиния: —")
        self.measurement_average_label.setText("\n".join(lines))

    def _on_measurement_finished(self, payload: dict[str, Any]) -> None:
        self._on_measurement_changed(payload)
        if not bool(payload.get("valid", False)):
            return
        kind = str(payload.get("kind", ""))
        if kind == ZoomView.TOOL_POLYGON_AREA:
            if float(payload.get("area_px2", 0.0)) > 0.0:
                self._append_area_measurement_history(payload)
        elif float(payload.get("length_px", 0.0)) >= 1.0:
            self._append_line_measurement_history(payload)
        self._refresh_measurement_tables()
        self._update_measurement_average()

    def _on_measurement_cleared(self) -> None:
        if hasattr(self, "measurement_current_label"):
            if self.measurement_tool_mode == ZoomView.TOOL_POLYGON_AREA:
                self.measurement_current_label.setText("Текущая площадь: —")
            elif self.measurement_tool_mode == ZoomView.TOOL_LINE:
                self.measurement_current_label.setText("Текущий отрезок: —")
            else:
                self.measurement_current_label.setText("Текущее измерение: —")
        self._refresh_measurement_tables()
        self._update_measurement_average()

    def _measurement_snapshot(self) -> dict[str, Any]:
        current = (
            self.view.current_measurement()
            if hasattr(self, "view")
            else {"valid": False}
        )
        if isinstance(current, dict):
            current = {
                key: value
                for key, value in current.items()
                if key
                in {
                    "valid",
                    "kind",
                    "x0_px",
                    "y0_px",
                    "x1_px",
                    "y1_px",
                    "dx_px",
                    "dy_px",
                    "length_px",
                    "length_um",
                    "angle_deg",
                    "um_per_px",
                    "label",
                    "finished",
                    "area_px2",
                    "area_um2",
                    "perimeter_px",
                    "perimeter_um",
                    "vertex_count",
                    "vertices_px",
                    "closed",
                }
            }
        active_history = self._active_measurement_history()
        return {
            "tool_mode": str(self.measurement_tool_mode),
            "mode_enabled": bool(self.measurement_tool_mode != ZoomView.TOOL_OFF),
            "current": current,
            "history": list(active_history),
            "history_count": int(len(active_history)),
            "line_history": list(self.line_measurement_history),
            "area_history": list(self.area_measurement_history),
            "average_length_um": (
                float(
                    sum(
                        float(item.get("length_um", 0.0))
                        for item in self.line_measurement_history
                    )
                    / len(self.line_measurement_history)
                )
                if self.line_measurement_history
                else 0.0
            ),
            "average_area_um2": (
                float(
                    sum(
                        float(item.get("area_um2", 0.0))
                        for item in self.area_measurement_history
                    )
                    / len(self.area_measurement_history)
                )
                if self.area_measurement_history
                else 0.0
            ),
            "scale_source": str(self.current_scale_source),
            "scale_audit": dict(self.current_scale_audit),
        }

    def _update_scale_status_widgets(
        self, *, um_per_px: float, audit: dict[str, Any], source: str
    ) -> None:
        scale_text = f"{float(um_per_px):.4f} мкм/px"
        if hasattr(self, "scale_toolbar_label"):
            self.scale_toolbar_label.setText(scale_text)
        if hasattr(self, "scale_validation_label"):
            if str(source) == "default.assumption":
                self.scale_validation_label.setText("Масштаб: по умолчанию")
            else:
                self.scale_validation_label.setText(
                    "Масштаб: OK"
                    if bool(audit.get("ok", False))
                    else "Масштаб: проверь"
                )
        if hasattr(self, "measurement_scale_source_label"):
            self.measurement_scale_source_label.setText(
                f"Калибровка: {source} | ожидаемо {float(audit.get('expected_um_per_px', um_per_px)):.4f} мкм/px"
            )

    def _on_theme_mode_changed(self, *_args: Any) -> None:
        mode = str(self.theme_mode_combo.currentData() or "light").strip().lower()
        self._style(mode)
        try:
            save_theme_mode(self.ui_theme_profile_path, self.theme_mode)
        except Exception:
            pass

    def _set_heavy_state_label(self, dirty: bool) -> None:
        self.heavy_state_label.setText("Эффекты: фиксированные")
        self.heavy_state_label.setStyleSheet(
            f"color: {status_color(self.theme_mode, 'text_secondary')}; font-weight: 700;"
        )
        self._pulse_widget(self.heavy_state_label, duration=180, start=0.45)

    def _update_dashboard_status(
        self, objective: int | None = None, focus_distance_mm: float | None = None
    ) -> None:
        obj = int(objective if objective is not None else self._current_objective())
        fval = float(
            focus_distance_mm
            if focus_distance_mm is not None
            else self._current_focus_distance_mm()
        )
        x = float(self.stage_x.value()) if hasattr(self, "stage_x") else 0.5
        y = float(self.stage_y.value()) if hasattr(self, "stage_y") else 0.5
        if hasattr(self, "status_objective_label"):
            new_text = f"Объектив: {obj}x"
            if self.status_objective_label.text() != new_text:
                self.status_objective_label.setText(new_text)
                self._pulse_widget(
                    self.status_objective_label, duration=150, start=0.65
                )
        if hasattr(self, "status_focus_label"):
            new_text = (
                f"Фокус: {fval:.1f} mm"
                if self.focus_user_configured
                else f"Фокус: не настроен ({fval:.1f} mm)"
            )
            if self.status_focus_label.text() != new_text:
                self.status_focus_label.setText(new_text)
                self._pulse_widget(self.status_focus_label, duration=150, start=0.65)
        if hasattr(self, "status_stage_label"):
            new_text = f"XY: {x:.2f}, {y:.2f}"
            if self.status_stage_label.text() != new_text:
                self.status_stage_label.setText(new_text)
                self._pulse_widget(self.status_stage_label, duration=150, start=0.65)
        if hasattr(self, "status_psf_label"):
            profile = (
                str(self.psf_profile_combo.currentData() or "standard")
                if hasattr(self, "psf_profile_combo")
                else "standard"
            )
            new_text = f"ТРФ: {profile}"
            if self.status_psf_label.text() != new_text:
                self.status_psf_label.setText(new_text)
                self._pulse_widget(self.status_psf_label, duration=150, start=0.65)

    def _set_status_pill_text(self, attr: str, text: str) -> None:
        if not hasattr(self, attr):
            return
        widget = getattr(self, attr)
        if widget.text() != text:
            widget.setText(text)
            self._pulse_widget(widget, duration=150, start=0.65)

    def _update_psf_dashboard(self, fov_meta: dict[str, Any]) -> None:
        profile_mode = str(
            fov_meta.get(
                "focus_profile_mode", fov_meta.get("axial_profile_mode", "standard")
            )
        )
        sectioning_active = bool(fov_meta.get("sectioning_active", False))
        suppression = float(fov_meta.get("sectioning_suppression_score", 0.0) or 0.0)
        dof_factor = float(fov_meta.get("effective_dof_factor", 1.0) or 1.0)
        shift_sig = float(fov_meta.get("axial_shift_signature", 0.0) or 0.0)
        self._set_status_pill_text(
            "status_axial_profile_label", f"Осевой: {profile_mode}"
        )
        self._set_status_pill_text(
            "status_sectioning_label",
            f"Секционирование: {'активно' if sectioning_active else 'неактивно'} ({suppression:.3f})",
        )
        self._set_status_pill_text("status_dof_label", f"Коэффициент ГРИП: {dof_factor:.2f}")
        self._set_status_pill_text(
            "status_shift_label", f"Осевой сдвиг: {shift_sig:.3f}"
        )

    def _style(self, mode: str | None = None) -> None:
        if mode:
            self.theme_mode = str(mode).strip().lower() or "light"
        self.setStyleSheet(build_qss(self.theme_mode))
        if hasattr(self, "navigator"):
            self.navigator.setStyleSheet(
                "background: transparent; "
                f"border: 1px solid {status_color(self.theme_mode, 'border')}; border-radius: 6px;"
            )
        if hasattr(self, "heavy_state_label"):
            self._set_heavy_state_label(bool(self.heavy_dirty))

    def _apply_responsive_layout(self) -> None:
        if not hasattr(self, "main_split"):
            return
        if hasattr(self, "view"):
            split_h = max(720, int(self.main_split.height() or self.height()))
            view_min = int(max(360, min(760, split_h * 0.48)))
            view_max = int(max(view_min + 80, min(1200, split_h * 0.80)))
            self.view.setMinimumHeight(view_min)
            self.view.setMaximumHeight(view_max)
            if hasattr(self, "journal"):
                journal_min = int(max(100, min(180, split_h * 0.14)))
                journal_max = int(max(journal_min + 30, min(280, split_h * 0.22)))
                self.journal.setMinimumHeight(journal_min)
                self.journal.setMaximumHeight(journal_max)
            if hasattr(self, "line_measurement_table") or hasattr(
                self, "area_measurement_table"
            ):
                measurement_min = int(max(110, min(160, split_h * 0.14)))
                measurement_max = int(
                    max(measurement_min + 24, min(210, split_h * 0.20))
                )
                if hasattr(self, "line_measurement_table"):
                    self.line_measurement_table.setMinimumHeight(measurement_min)
                    self.line_measurement_table.setMaximumHeight(measurement_max)
                if hasattr(self, "area_measurement_table"):
                    self.area_measurement_table.setMinimumHeight(measurement_min)
                    self.area_measurement_table.setMaximumHeight(measurement_max)
            if hasattr(self, "info"):
                info_min = int(max(270, min(450, split_h * 0.36)))
                info_max = int(max(info_min + 90, min(720, split_h * 0.54)))
                self.info.setMinimumHeight(info_min)
                self.info.setMaximumHeight(info_max)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_responsive_layout()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._apply_responsive_layout()
        if self._did_intro_animation:
            return
        self._did_intro_animation = True
        try:
            self.setWindowOpacity(0.0)
            anim = QPropertyAnimation(self, b"windowOpacity", self)
            anim.setDuration(260)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            anim.start()
            self._animations.append(anim)
        except Exception:
            pass

    def _pulse_widget(
        self, widget: QWidget, duration: int = 160, start: float = 0.65
    ) -> None:
        if widget is None:
            return
        try:
            effect = widget.graphicsEffect()
            if not isinstance(effect, QGraphicsOpacityEffect):
                effect = QGraphicsOpacityEffect(widget)
                widget.setGraphicsEffect(effect)
            anim = QPropertyAnimation(effect, b"opacity", widget)
            anim.setDuration(max(80, int(duration)))
            anim.setStartValue(max(0.1, float(start)))
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

            def _cleanup() -> None:
                try:
                    self._animations.remove(anim)
                except Exception:
                    pass

            anim.finished.connect(_cleanup)
            anim.start()
            self._animations.append(anim)
        except Exception:
            return

    def _scan_samples(self) -> None:
        directory = Path(self.dir_edit.text().strip())
        self.samples_dir = directory
        self.sample_list.clear()
        if not directory.exists():
            return
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        files = sorted(
            [
                p
                for p in directory.rglob("*")
                if p.is_file() and p.suffix.lower() in exts
            ]
        )
        filtered: list[Path] = []
        for path in files:
            stem = path.stem.lower()
            # Exclude auxiliary render assets (masks/diagram) and keep primary sample images.
            if any(
                token in stem
                for token in ("_phase_", "_feature_", "_prep_", "_diagram")
            ):
                continue
            filtered.append(path)
        files = filtered
        for path in files:
            try:
                label = str(path.relative_to(directory))
            except Exception:
                label = path.name
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.sample_list.addItem(item)
        pass  # Do not auto-select; student opens files manually

    def _clear_mask_cache(self) -> None:
        self._mask_fov_cache.clear()

    def _cache_get_mask(self, key: tuple[Any, ...]) -> np.ndarray | None:
        cached = self._mask_fov_cache.get(key)
        if cached is None:
            return None
        self._mask_fov_cache.move_to_end(key)
        return cached

    def _cache_put_mask(self, key: tuple[Any, ...], value: np.ndarray) -> np.ndarray:
        self._mask_fov_cache[key] = value
        self._mask_fov_cache.move_to_end(key)
        while len(self._mask_fov_cache) > self._mask_fov_cache_max:
            self._mask_fov_cache.popitem(last=False)
        return value

    def _parse_mask_entry(self, path: Path) -> dict[str, Any]:
        stem = path.stem
        stem_low = stem.lower()
        if "_phase_" in stem_low:
            split_idx = stem_low.rfind("_phase_")
            name = stem[split_idx + len("_phase_") :]
            return {"category": "phase", "name": name, "path": path}
        if "_feature_" in stem_low:
            split_idx = stem_low.rfind("_feature_")
            name = stem[split_idx + len("_feature_") :]
            return {"category": "feature", "name": name, "path": path}
        if "_prep_" in stem_low:
            split_idx = stem_low.rfind("_prep_")
            name = stem[split_idx + len("_prep_") :]
            return {"category": "prep", "name": name, "path": path}
        return {"category": "unknown", "name": stem, "path": path}

    def _collect_manifest_entries(
        self, manifest_payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for key in ("phase_and_feature_masks", "prep_maps"):
            raw = manifest_payload.get(key, [])
            if not isinstance(raw, list):
                continue
            for item in raw:
                try:
                    path = Path(str(item))
                except Exception:
                    continue
                if not path.exists():
                    continue
                entries.append(self._parse_mask_entry(path))
        return entries

    def _find_manifest_for_image(
        self, image_path: Path
    ) -> tuple[Path | None, dict[str, Any] | None]:
        direct = image_path.with_name(f"{image_path.stem}_manifest.json")
        if direct.exists():
            try:
                payload = json.loads(direct.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return direct, payload
            except Exception:
                pass

        for candidate in sorted(image_path.parent.glob("*_manifest.json")):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            manifest_image = payload.get("image")
            if not isinstance(manifest_image, str):
                continue
            try:
                manifest_image_path = Path(manifest_image).resolve()
                if manifest_image_path == image_path.resolve():
                    return candidate, payload
            except Exception:
                if Path(manifest_image).name == image_path.name:
                    return candidate, payload
        return None, None

    def _discover_mask_entries(
        self, image_path: Path
    ) -> tuple[Path | None, list[dict[str, Any]]]:
        manifest_path, manifest_payload = self._find_manifest_for_image(image_path)
        entries: list[dict[str, Any]] = []
        if isinstance(manifest_payload, dict):
            entries.extend(self._collect_manifest_entries(manifest_payload))

        if not entries:
            for pattern in (
                f"{image_path.stem}_phase_*.png",
                f"{image_path.stem}_feature_*.png",
                f"{image_path.stem}_prep_*.png",
            ):
                for path in sorted(image_path.parent.glob(pattern)):
                    entries.append(self._parse_mask_entry(path))
        uniq: dict[str, dict[str, Any]] = {}
        for entry in entries:
            p = entry.get("path")
            if not isinstance(p, Path):
                continue
            uniq[str(p.resolve())] = entry
        return manifest_path, list(uniq.values())

    def _load_mask_fov(
        self,
        entry: dict[str, Any],
        *,
        fov_meta: dict[str, Any],
        output_size: tuple[int, int],
    ) -> np.ndarray | None:
        path = entry.get("path")
        if not isinstance(path, Path) or not path.exists():
            return None
        if self.current_source_gray is None:
            return None
        origin = fov_meta.get("crop_origin_px")
        crop_size = fov_meta.get("crop_size_px")
        if not (
            isinstance(origin, (list, tuple))
            and isinstance(crop_size, (list, tuple))
            and len(origin) == 2
            and len(crop_size) == 2
        ):
            return None
        src_h, src_w = self.current_source_gray.shape
        y0_src = max(0, int(origin[0]))
        x0_src = max(0, int(origin[1]))
        crop_h_src = max(1, int(crop_size[0]))
        crop_w_src = max(1, int(crop_size[1]))
        out_h = max(1, int(output_size[0]))
        out_w = max(1, int(output_size[1]))
        key = (str(path), y0_src, x0_src, crop_h_src, crop_w_src, out_h, out_w)
        cached = self._cache_get_mask(key)
        if cached is not None:
            return cached
        try:
            slide = get_path_slide(path)
            mask_h, mask_w = slide.base_shape
            if src_h <= 0 or src_w <= 0 or mask_h <= 0 or mask_w <= 0:
                return None
            y0 = int(round(float(y0_src) * float(mask_h) / float(src_h)))
            x0 = int(round(float(x0_src) * float(mask_w) / float(src_w)))
            ch = max(1, int(round(float(crop_h_src) * float(mask_h) / float(src_h))))
            cw = max(1, int(round(float(crop_w_src) * float(mask_w) / float(src_w))))
            crop, _meta = slide.extract_pixels(
                origin_px=(y0, x0),
                crop_size_px=(ch, cw),
                output_size=(out_h, out_w),
            )
            arr = crop.astype(np.float32, copy=False) / 255.0
            return self._cache_put_mask(key, arr)
        except Exception:
            return None

    def _phase_tone_delta(self, name: str) -> float:
        n = str(name).upper()
        if "FERRITE" in n or "ALPHA" in n or "MATRIX" in n:
            return 12.0
        if "PEARLITE" in n or "BETA" in n:
            return -8.0
        if "CEMENTITE" in n or "CARBIDE" in n or "SI" in n:
            return -18.0
        if "MARTENSITE" in n or "BAINITE" in n:
            return -12.0
        if "LIQUID" in n or n == "L":
            return 10.0
        h = (sum(ord(ch) for ch in n) % 17) - 8
        return float(h)

    def _feature_delta(self, name: str, mask: np.ndarray) -> np.ndarray:
        n = str(name).lower()
        if "phase_bound" in n or "boundary" in n:
            return mask * 22.0
        if "high_contrast" in n:
            return mask * 11.0
        if "low_contrast" in n:
            return -mask * 11.0
        h = ((sum(ord(ch) for ch in n) % 13) - 6) * 1.6
        return mask * float(h)

    def _apply_masks_to_view(
        self, view_gray: np.ndarray, fov_meta: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not bool(self.DEFAULT_MASK_RENDERING.get("enabled", True)):
            return view_gray, {
                "enabled": False,
                "mask_count_total": len(self.current_mask_entries),
            }
        if not self.current_mask_entries:
            return view_gray, {"enabled": True, "mask_count_total": 0}

        phase_gain = float(self.DEFAULT_MASK_RENDERING.get("phase_strength", 1.0))
        feature_gain = float(self.DEFAULT_MASK_RENDERING.get("feature_strength", 0.95))
        prep_gain = float(self.DEFAULT_MASK_RENDERING.get("prep_strength", 0.9))
        out_h, out_w = view_gray.shape

        additive = np.zeros((out_h, out_w), dtype=np.float32)
        multiplicative = np.ones((out_h, out_w), dtype=np.float32)
        used_phase = 0
        used_feature = 0
        used_prep = 0

        for entry in self.current_mask_entries:
            category = str(entry.get("category", "unknown"))
            name = str(entry.get("name", ""))
            mask = self._load_mask_fov(
                entry, fov_meta=fov_meta, output_size=(out_h, out_w)
            )
            if mask is None:
                continue
            if category == "phase":
                additive += mask * self._phase_tone_delta(name) * phase_gain
                used_phase += 1
            elif category == "feature":
                additive += self._feature_delta(name, mask) * feature_gain
                used_feature += 1
            elif category == "prep":
                n = name.lower()
                centered = mask - 0.5
                if "topography" in n:
                    multiplicative *= 1.0 + centered * (0.22 * prep_gain)
                elif "scratch" in n:
                    additive -= mask * (16.0 * prep_gain)
                elif "contamination" in n:
                    additive -= mask * (18.0 * prep_gain)
                elif "deformation" in n:
                    additive += centered * (14.0 * prep_gain)
                elif "smear" in n:
                    additive -= mask * (9.0 * prep_gain)
                elif "etch_rate" in n:
                    additive += centered * (20.0 * prep_gain)
                else:
                    additive += centered * (8.0 * prep_gain)
                used_prep += 1
            else:
                additive += (mask - 0.5) * 6.0

        combined = view_gray.astype(np.float32) * multiplicative + additive
        combined = np.clip(combined, 0.0, 255.0).astype(np.uint8)
        summary = {
            "enabled": True,
            "mask_count_total": int(len(self.current_mask_entries)),
            "mask_count_phase": int(used_phase),
            "mask_count_feature": int(used_feature),
            "mask_count_prep": int(used_prep),
            "phase_strength": float(phase_gain),
            "feature_strength": float(feature_gain),
            "prep_strength": float(prep_gain),
        }
        return combined, summary

    def _on_select_sample(self) -> None:
        items = self.sample_list.selectedItems()
        if not items:
            return
        path = Path(items[0].data(Qt.ItemDataRole.UserRole))
        self.current_image_path = path
        self.current_meta_path = next(
            (
                candidate
                for candidate in _candidate_metadata_paths(path)
                if candidate.exists()
            ),
            None,
        )
        self.current_source_metadata = None
        self.current_manifest_path = None
        self.current_mask_entries = []
        self.current_mask_render_summary = {"enabled": False, "mask_count_total": 0}
        self._clear_mask_cache()
        self._clear_all_measurements()
        self._last_view_signature = None
        self._view_update_timer.stop()

        try:
            with Image.open(path) as img:
                gray = np.asarray(img.convert("L"), dtype=np.uint8)
                nav = img.copy()
                nav.thumbnail((2048, 2048), Image.Resampling.BILINEAR)
                rgb = np.asarray(nav.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            QMessageBox.critical(self, "Образец", f"Ошибка чтения изображения:\n{exc}")
            return

        self.current_source_gray = gray
        self.current_source_rgb = rgb
        self.navigator.set_source(rgb)
        if self.current_meta_path and self.current_meta_path.exists():
            try:
                payload = json.loads(self.current_meta_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    self.current_source_metadata = payload
            except Exception:
                self.current_source_metadata = None

        manifest_path, mask_entries = self._discover_mask_entries(path)
        self.current_manifest_path = manifest_path
        self.current_mask_entries = mask_entries
        self._reset_focus_to_unconfigured(trigger_update=False)
        self._view_needs_fit = True
        self._update_view()

    def _set_objective(self, objective: int, *, immediate: bool = True) -> None:
        idx = self._objective_index_for_value(int(objective))
        current_focus = (
            self._current_focus_distance_mm()
            if hasattr(self, "focus_distance_spin")
            else None
        )
        self._syncing_optics_widgets = True
        try:
            if hasattr(self, "objective_spin"):
                self.objective_spin.blockSignals(True)
                self.objective_spin.setValue(int(self.OBJECTIVES[idx]))
                self.objective_spin.blockSignals(False)
            self._update_objective_dial_label()
            self._update_focus_distance_range()
            if current_focus is not None:
                low, high = self._focus_distance_limits_mm(int(self.OBJECTIVES[idx]))
                self._update_focus_spin_value(float(np.clip(current_focus, low, high)))
            self._update_focus_distance_label()
        finally:
            self._syncing_optics_widgets = False
        self._update_dashboard_status(
            objective=self._current_objective(),
            focus_distance_mm=self._current_focus_distance_mm(),
        )
        if immediate:
            self._queue_view_update(immediate=True)

    def _current_objective(self) -> int:
        if hasattr(self, "objective_spin"):
            return int(self.objective_spin.value())
        if 200 in self.OBJECTIVES:
            return 200
        return int(self.OBJECTIVES[0])

    def _objective_index_for_value(self, objective: int) -> int:
        mags = [int(v) for v in self.OBJECTIVES]
        if not mags:
            return 0
        if objective in mags:
            return mags.index(objective)
        diffs = [abs(m - int(objective)) for m in mags]
        return int(np.argmin(np.asarray(diffs, dtype=np.int64)))

    def _on_objective_button_press(self) -> None:
        self._objective_repeat_active = True

    def _on_objective_button_release(self) -> None:
        if self._objective_repeat_active:
            self._objective_repeat_active = False
            self._queue_view_update(immediate=True)

    def _step_objective(self, direction: int, *, immediate: bool = True) -> None:
        current_idx = self._objective_index_for_value(self._current_objective())
        next_idx = int(
            np.clip(current_idx + int(direction), 0, len(self.OBJECTIVES) - 1)
        )
        self._set_objective(int(self.OBJECTIVES[next_idx]), immediate=immediate)

    def _update_objective_dial_label(self) -> None:
        if not hasattr(self, "objective_value_label"):
            return
        self.objective_value_label.setText(f"{self._current_objective()}x")

    def _on_objective_spin_changed(self, value: int) -> None:
        if self._syncing_optics_widgets:
            return
        snapped = int(self.OBJECTIVES[self._objective_index_for_value(int(value))])
        self._set_objective(snapped)

    def _focus_distance_limits_mm(
        self, objective: int | None = None
    ) -> tuple[float, float]:
        nominal = self._objective_nominal_distance_mm(objective)
        return nominal - 4.0, nominal + 4.0

    def _update_focus_distance_range(self) -> None:
        if not hasattr(self, "focus_distance_spin"):
            return
        low, high = self._focus_distance_limits_mm()
        self.focus_distance_spin.blockSignals(True)
        self.focus_distance_spin.setRange(low, high)
        self.focus_distance_spin.blockSignals(False)

    def _on_focus_distance_spin_changed(self, value: float) -> None:
        if self._syncing_optics_widgets:
            return
        self._set_focus_distance_mm(float(value), user_configured=True)

    def _focus_indicator_text(self, value_mm: float) -> str:
        return f"{value_mm:.2f} mm"

    def _update_focus_spin_value(self, value_mm: float) -> None:
        if not hasattr(self, "focus_distance_spin"):
            return
        self.focus_distance_spin.blockSignals(True)
        self.focus_distance_spin.setValue(float(value_mm))
        self.focus_distance_spin.blockSignals(False)

    def _objective_nominal_distance_mm(self, objective: int | None = None) -> float:
        mag = int(objective if objective is not None else self._current_objective())
        return float(np.clip(36.0 - 0.045 * mag, 9.0, 32.0))

    def _current_focus_distance_mm(self) -> float:
        if hasattr(self, "focus_distance_spin"):
            return float(self.focus_distance_spin.value())
        low, _high = self._focus_distance_limits_mm()
        return float(low)

    def _update_focus_distance_label(self) -> None:
        self.focus_distance_mm = self._current_focus_distance_mm()
        if hasattr(self, "focus_value_label"):
            if self.focus_user_configured:
                self.focus_value_label.setText(
                    self._focus_indicator_text(self.focus_distance_mm)
                )
            else:
                self.focus_value_label.setText("не настроен")

    def _set_focus_distance_mm(
        self, distance_mm: float, *, user_configured: bool = False
    ) -> None:
        low, high = self._focus_distance_limits_mm()
        clamped = float(np.clip(distance_mm, low, high))
        self.focus_user_configured = bool(user_configured)
        self._update_focus_spin_value(clamped)
        self._update_focus_distance_label()
        self._update_dashboard_status(
            objective=self._current_objective(), focus_distance_mm=clamped
        )
        self._on_live_control_changed()

    def _step_focus_dial(self, direction: int) -> None:
        if not hasattr(self, "focus_distance_spin"):
            return
        next_value = float(self.focus_distance_spin.value()) + float(direction) * float(
            self.focus_distance_spin.singleStep()
        )
        self._set_focus_distance_mm(next_value, user_configured=True)

    def _reset_focus_to_unconfigured(self, *, trigger_update: bool = True) -> None:
        low, _high = self._focus_distance_limits_mm()
        self.focus_user_configured = False
        self._update_focus_spin_value(low)
        self._update_focus_distance_label()
        self._update_dashboard_status(
            objective=self._current_objective(), focus_distance_mm=low
        )
        if trigger_update:
            self._on_live_control_changed()

    def _focus_target_mm(self, objective: int, pan_x: float, pan_y: float) -> float:
        stage_bias = ((pan_x - 0.5) * 1.4) + ((pan_y - 0.5) * -1.0)
        return float(self._objective_nominal_distance_mm(objective) + stage_bias)

    def _focus_quality_from_error(self, objective: int, focus_error_mm: float) -> float:
        tolerance_mm = max(0.18, 0.75 - 0.0009 * float(objective))
        return float(np.clip(1.0 - float(focus_error_mm) / tolerance_mm, 0.0, 1.0))

    def _focus_status_text(self, focus_quality: float) -> str:
        if focus_quality >= 0.85:
            return "в фокусе"
        if focus_quality >= 0.45:
            return "близко к фокусу"
        return "вне фокуса"

    @staticmethod
    def _q(value: Any, precision: int = 4) -> int:
        return int(round(float(value) * (10**precision)))

    def _view_signature(
        self,
        *,
        objective: int,
        focus_distance_mm: float,
        pan_x: float,
        pan_y: float,
        focus_target_mm: float,
    ) -> tuple[Any, ...]:
        source = self.current_source_gray
        if source is None:
            return ()
        return (
            source.__array_interface__["data"][0],
            int(source.shape[0]),
            int(source.shape[1]),
            str(self.current_image_path or ""),
            int(objective),
            self._q(focus_distance_mm),
            self._q(pan_x),
            self._q(pan_y),
            self._q(focus_target_mm),
            int(self.out_h.value()),
            int(self.out_w.value()),
            self._q(self.brightness.value(), precision=3),
            self._q(self.contrast.value(), precision=3),
            str(self._current_optical_mode()),
            tuple(sorted(self._current_optical_mode_parameters().items())),
            self._q(self.applied_heavy.get("vignette_strength", 0.30)),
            self._q(self.applied_heavy.get("uneven_strength", 0.08)),
            self._q(self.applied_heavy.get("noise_sigma", 7.72)),
            self._q(self.applied_heavy.get("etch_uneven", 0.0)),
            str(self.psf_profile_combo.currentData() or "standard"),
            self._q(self.psf_strength_spin.value()),
            self._q(self.sectioning_shear_spin.value()),
            self._q(self.hybrid_balance_spin.value()),
            1 if bool(self.applied_heavy.get("add_dust", False)) else 0,
            1 if bool(self.applied_heavy.get("add_scratches", False)) else 0,
            1 if self.reticle_check.isChecked() else 0,
            1 if self.scale_check.isChecked() else 0,
            int(len(self.current_mask_entries)),
        )

    def _current_optical_mode(self) -> str:
        if hasattr(self, "optical_mode_combo"):
            return str(self.optical_mode_combo.currentData() or "brightfield")
        return "brightfield"

    def _current_optical_mode_parameters(self) -> dict[str, Any]:
        mode = self._current_optical_mode()
        if mode == "darkfield":
            return {"scatter_sensitivity": float(self.darkfield_scatter_spin.value())}
        if mode == "polarized":
            crossed = bool(self.polarized_crossed_check.isChecked())
            return {
                "crossed_polars": crossed,
                "polarizer_angle_deg": 0.0,
                "analyzer_angle_deg": 90.0 if crossed else 0.0,
            }
        if mode == "phase_contrast":
            return {
                "phase_plate_type": str(
                    self.phase_plate_combo.currentData() or "positive"
                )
            }
        return {}

    def _set_stage_xy(self, x: float, y: float) -> None:
        self.stage_x.blockSignals(True)
        self.stage_y.blockSignals(True)
        try:
            self.stage_x.setValue(float(np.clip(x, 0.0, 1.0)))
            self.stage_y.setValue(float(np.clip(y, 0.0, 1.0)))
        finally:
            self.stage_x.blockSignals(False)
            self.stage_y.blockSignals(False)
        self._on_live_control_changed()

    def _move_stage(self, dx: float, dy: float) -> None:
        self._set_stage_xy(self.stage_x.value() + dx, self.stage_y.value() + dy)

    def _queue_view_update(self, *, immediate: bool = False) -> None:
        if self.current_source_gray is None:
            return
        if immediate:
            self._view_update_timer.stop()
            self._update_view()
            return
        self._view_update_timer.start(18)

    def _on_live_control_changed(self) -> None:
        self._queue_view_update()

    def _ensure_render_worker(self) -> None:
        if self._render_thread is not None and self._render_thread.is_alive():
            return
        self._render_shutdown = False
        self._render_thread = threading.Thread(
            target=self._render_worker_loop,
            name="MicroscopeRenderWorker",
            daemon=True,
        )
        self._render_thread.start()

    def _render_worker_loop(self) -> None:
        while True:
            self._render_wakeup.wait()
            self._render_wakeup.clear()
            while True:
                with self._render_lock:
                    if self._render_shutdown:
                        return
                    request = self._render_pending_request
                    self._render_pending_request = None
                if request is None:
                    break
                try:
                    view, fov_meta = simulate_microscope_view(
                        sample=request["sample"],
                        magnification=request["objective"],
                        pan_x=request["pan_x"],
                        pan_y=request["pan_y"],
                        output_size=request["output_size"],
                        focus_distance_mm=request["focus_distance_mm"],
                        focus_target_mm=request["focus_target_mm"],
                        um_per_px_100x=request["um_per_px_100x"],
                        brightness=request["brightness"],
                        contrast=request["contrast"],
                        optical_mode=request["optical_mode"],
                        optical_mode_parameters=request["optical_mode_parameters"],
                        vignette_strength=request["vignette_strength"],
                        uneven_strength=request["uneven_strength"],
                        noise_sigma=request["noise_sigma"],
                        add_dust=request["add_dust"],
                        add_scratches=request["add_scratches"],
                        etch_uneven=request["etch_uneven"],
                        psf_profile=request["psf_profile"],
                        psf_strength=request["psf_strength"],
                        sectioning_shear_deg=request["sectioning_shear_deg"],
                        hybrid_balance=request["hybrid_balance"],
                        seed=request["seed"],
                    )
                    self._render_signals.rendered.emit(
                        {
                            "generation": request["generation"],
                            "signature": request["signature"],
                            "request": request,
                            "view": view,
                            "fov_meta": fov_meta,
                        }
                    )
                except RuntimeError:
                    return
                except Exception as exc:
                    try:
                        self._render_signals.failed.emit(
                            {
                                "generation": request["generation"],
                                "signature": request["signature"],
                                "error": str(exc),
                            }
                        )
                    except RuntimeError:
                        return

    def _build_render_request(self) -> dict[str, Any] | None:
        if self.current_source_gray is None:
            return None

        objective = self._current_objective()
        focus_distance_mm = self._current_focus_distance_mm()
        pan_x = float(self.stage_x.value())
        pan_y = float(self.stage_y.value())
        focus_target_mm = self._focus_target_mm(objective, pan_x, pan_y)
        focus_quality = self._focus_quality_from_error(
            objective, abs(focus_distance_mm - focus_target_mm)
        )
        base_um_per_px_100x = self._um_per_px_100x()
        view_signature = self._view_signature(
            objective=objective,
            focus_distance_mm=focus_distance_mm,
            pan_x=pan_x,
            pan_y=pan_y,
            focus_target_mm=focus_target_mm,
        )
        if (
            not self._view_needs_fit
            and view_signature == self._last_view_signature
            and self.current_capture is not None
        ):
            return None
        if view_signature == self._render_requested_signature:
            return None
        self._render_request_generation += 1
        self._render_requested_signature = view_signature
        return {
            "generation": self._render_request_generation,
            "signature": view_signature,
            "sample": self.current_source_gray,
            "objective": objective,
            "focus_distance_mm": focus_distance_mm,
            "pan_x": pan_x,
            "pan_y": pan_y,
            "focus_target_mm": focus_target_mm,
            "focus_quality": focus_quality,
            "um_per_px_100x": base_um_per_px_100x,
            "brightness": float(self.brightness.value()),
            "contrast": float(self.contrast.value()),
            "optical_mode": self._current_optical_mode(),
            "optical_mode_parameters": self._current_optical_mode_parameters(),
            "output_size": (int(self.out_h.value()), int(self.out_w.value())),
            "vignette_strength": float(
                self.applied_heavy.get("vignette_strength", 0.30)
            ),
            "uneven_strength": float(self.applied_heavy.get("uneven_strength", 0.08)),
            "noise_sigma": float(self.applied_heavy.get("noise_sigma", 7.72)),
            "add_dust": bool(self.applied_heavy.get("add_dust", False)),
            "add_scratches": bool(self.applied_heavy.get("add_scratches", False)),
            "etch_uneven": float(self.applied_heavy.get("etch_uneven", 0.0)),
            "psf_profile": str(self.psf_profile_combo.currentData() or "standard"),
            "psf_strength": float(self.psf_strength_spin.value()),
            "sectioning_shear_deg": float(self.sectioning_shear_spin.value()),
            "hybrid_balance": float(self.hybrid_balance_spin.value()),
            "seed": 1234,
        }

    def _submit_render_request(self, request: dict[str, Any]) -> None:
        self._ensure_render_worker()
        with self._render_lock:
            self._render_pending_request = request
        self._render_wakeup.set()

    def _on_render_worker_result(self, payload: dict[str, Any]) -> None:
        generation = int(payload.get("generation", -1))
        signature = payload.get("signature")
        if (
            generation != self._render_request_generation
            or signature != self._render_requested_signature
        ):
            return
        self._apply_render_result(
            payload["request"],
            np.asarray(payload["view"]),
            dict(payload["fov_meta"]),
        )

    def _on_render_worker_error(self, payload: dict[str, Any]) -> None:
        generation = int(payload.get("generation", -1))
        signature = payload.get("signature")
        if (
            generation != self._render_request_generation
            or signature != self._render_requested_signature
        ):
            return
        self._render_requested_signature = None

    def closeEvent(self, event) -> None:
        if hasattr(self, "state_manager"):
            self.state_manager.save_state(self)
        self._shutdown_render_worker()
        super().closeEvent(event)

    def _shutdown_render_worker(self) -> None:
        self._render_shutdown = True
        self._render_wakeup.set()
        thread = self._render_thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=1.0)
        self._render_thread = None

    def _apply_heavy(self) -> None:
        current = dict(self.applied_heavy)
        self.applied_heavy = {
            "noise_sigma": 7.72,
            "vignette_strength": 0.30,
            "uneven_strength": float(current.get("uneven_strength", 0.08)),
            "add_dust": bool(current.get("add_dust", False)),
            "add_scratches": bool(current.get("add_scratches", False)),
            "etch_uneven": float(current.get("etch_uneven", 0.0)),
        }
        self.heavy_dirty = False
        self._set_heavy_state_label(False)
        self._update_view()

    def _route_summary(self) -> dict[str, Any]:
        route_summary: dict[str, Any] = {}
        metadata_payload = self.current_source_metadata or {}
        timeline = metadata_payload.get("route_timeline")
        if not isinstance(timeline, list):
            timeline = metadata_payload.get("process_timeline")
        process_route = metadata_payload.get("process_route")
        if not isinstance(process_route, dict):
            req_v3 = metadata_payload.get("request_v3")
            if isinstance(req_v3, dict):
                process_route = req_v3.get("process_route")
        req = metadata_payload.get("request")
        req_v3 = metadata_payload.get("request_v3")
        if isinstance(timeline, list):
            route_summary["step_count"] = len(timeline)
        if isinstance(process_route, dict):
            route_summary["route_name"] = str(process_route.get("route_name", ""))
        if isinstance(req, dict):
            route_summary["step_index"] = req.get("preview_step_index")
            route_summary["route_policy"] = str(req.get("route_policy", ""))
        if isinstance(req_v3, dict):
            route_summary["sample_id"] = str(req_v3.get("sample_id", ""))
        stage = metadata_payload.get("final_stage") or metadata_payload.get("stage")
        if stage is not None:
            route_summary["final_stage"] = str(stage)
        props = metadata_payload.get("property_indicators")
        if isinstance(props, dict):
            route_summary["property_indicators"] = props
        if isinstance(metadata_payload.get("prep_timeline"), list):
            route_summary["prep_step_count"] = len(
                metadata_payload.get("prep_timeline", [])
            )
        etch_summary = metadata_payload.get("etch_summary")
        if isinstance(etch_summary, dict):
            route_summary["etch_reagent"] = str(etch_summary.get("reagent", ""))
            route_summary["etch_time_s"] = float(etch_summary.get("time_s", 0.0))
        return route_summary

    def _um_per_px_100x(self) -> float:
        value, source = derive_um_per_px_100x(
            self.current_source_metadata or {}, default=1.0
        )
        self.current_scale_source = str(source)
        return float(value)

    def _source_generator_version(self) -> str:
        meta = self.current_source_metadata or {}
        version = meta.get("generator_version")
        if isinstance(version, str):
            return version
        request_v3 = meta.get("request_v3")
        if isinstance(request_v3, dict):
            req_ver = request_v3.get("generator_version")
            if isinstance(req_ver, str):
                return req_ver
        return ""

    def _source_prep_signature(self) -> dict[str, Any]:
        meta = self.current_source_metadata or {}
        prep_summary = meta.get("prep_summary")
        if isinstance(prep_summary, dict):
            return dict(prep_summary)
        request_v3 = meta.get("request_v3")
        if isinstance(request_v3, dict):
            prep_route = request_v3.get("prep_route")
            if isinstance(prep_route, dict):
                return {
                    "roughness_target_um": prep_route.get("roughness_target_um"),
                    "relief_mode": prep_route.get("relief_mode"),
                    "contamination_level": prep_route.get("contamination_level"),
                    "step_count": len(prep_route.get("steps", []))
                    if isinstance(prep_route.get("steps"), list)
                    else 0,
                }
        return {}

    def _source_etch_signature(self) -> dict[str, Any]:
        meta = self.current_source_metadata or {}
        etch_summary = meta.get("etch_summary")
        if isinstance(etch_summary, dict):
            return dict(etch_summary)
        request_v3 = meta.get("request_v3")
        if isinstance(request_v3, dict):
            etch = request_v3.get("etch_profile")
            if isinstance(etch, dict):
                return dict(etch)
        return {}

    def _source_quality_metrics(self) -> dict[str, Any]:
        meta = self.current_source_metadata or {}
        quality = meta.get("quality_metrics")
        return dict(quality) if isinstance(quality, dict) else {}

    def _apply_overlays(
        self, image: np.ndarray, fov_meta: dict[str, Any]
    ) -> np.ndarray:
        rgb = image if image.ndim == 3 else np.stack([image] * 3, axis=2)
        pil = Image.fromarray(rgb.astype(np.uint8, copy=False))
        draw = ImageDraw.Draw(pil)
        w, h = pil.size

        if self.reticle_check.isChecked():
            cx = w // 2
            cy = h // 2
            draw.line((cx - 40, cy, cx + 40, cy), fill=(235, 90, 80), width=1)
            draw.line((cx, cy - 40, cx, cy + 40), fill=(235, 90, 80), width=1)
            draw.ellipse(
                (cx - 3, cy - 3, cx + 3, cy + 3), outline=(235, 90, 80), width=1
            )

        scale_info = {
            "enabled": False,
            "um_per_px": 1.0,
            "bar_um": 100.0,
            "bar_nm": 100000.0,
            "bar_px": 100.0,
        }
        if self.scale_check.isChecked():
            um_per_px = float(fov_meta.get("um_per_px") or 0.0)
            if um_per_px <= 0.0:
                um_per_px = estimate_um_per_px(
                    um_per_px_100x=self._um_per_px_100x(),
                    crop_size_px=fov_meta.get("crop_size_px"),
                    output_size_px=[h, w],
                )
            scale_info = choose_scale_bar(um_per_px)
            bar_px = int(scale_info["bar_px"])
            y = h - 28
            x0 = max(20, w - bar_px - 28)
            x1 = x0 + bar_px
            draw.line((x0, y, x1, y), fill=(245, 245, 245), width=3)
            font = _load_overlay_font(max(11, int(h * 0.018)))
            unit = "нм" if _font_supports_text(font, "нм") else "nm"
            label = f"{int(round(float(scale_info['bar_nm'])))} {unit}"
            try:
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_h = max(1, int(text_bbox[3] - text_bbox[1]))
            except Exception:
                text_h = max(12, int(h * 0.03))
            text_y = max(4, y - text_h - 8)
            draw.text((x0, text_y), label, fill=(245, 245, 245), font=font)

        self.current_scale_bar = scale_info
        return np.asarray(pil, dtype=np.uint8)

    def _apply_measurement_overlay_to_capture(self, image: np.ndarray) -> np.ndarray:
        payload = (
            self.view.current_measurement()
            if hasattr(self, "view")
            else {"valid": False}
        )
        if not bool(payload.get("valid", False)):
            return image
        rgb = image if image.ndim == 3 else np.stack([image] * 3, axis=2)
        pil = Image.fromarray(rgb.astype(np.uint8, copy=False))
        draw = ImageDraw.Draw(pil)
        if str(payload.get("kind", "")) == ZoomView.TOOL_POLYGON_AREA:
            vertices = payload.get("vertices_px", [])
            if not isinstance(vertices, list) or len(vertices) < 3:
                return image
            polygon = [
                (float(item[0]), float(item[1]))
                for item in vertices
                if isinstance(item, list) and len(item) == 2
            ]
            if len(polygon) < 3:
                return image
            draw.polygon(polygon, outline=(103, 214, 255), fill=(44, 96, 118))
            for px, py in polygon:
                draw.ellipse(
                    (px - 4.0, py - 4.0, px + 4.0, py + 4.0),
                    fill=(103, 214, 255),
                    outline=(10, 18, 28),
                    width=1,
                )
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            label = f"{payload.get('label', '—')} | {int(payload.get('vertex_count', 0))} вершин"
            font = _load_overlay_font(max(11, int(pil.size[1] * 0.018)))
            cx = 0.5 * (min(xs) + max(xs))
            cy = 0.5 * (min(ys) + max(ys))
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                tw = max(1, int(bbox[2] - bbox[0]))
                th = max(1, int(bbox[3] - bbox[1]))
            except Exception:
                tw = max(90, int(pil.size[0] * 0.16))
                th = max(18, int(pil.size[1] * 0.03))
            rx0 = max(6, int(cx - tw / 2.0 - 6.0))
            ry0 = max(6, int(cy - th / 2.0 - 4.0))
            rx1 = min(pil.size[0] - 6, rx0 + tw + 12)
            ry1 = min(pil.size[1] - 6, ry0 + th + 8)
            draw.rounded_rectangle(
                (rx0, ry0, rx1, ry1),
                radius=6,
                fill=(10, 18, 28),
                outline=(103, 214, 255),
                width=1,
            )
            draw.text((rx0 + 6, ry0 + 4), label, fill=(248, 248, 248), font=font)
            return np.asarray(pil, dtype=np.uint8)
        x0 = float(payload.get("x0_px", 0.0))
        y0 = float(payload.get("y0_px", 0.0))
        x1 = float(payload.get("x1_px", 0.0))
        y1 = float(payload.get("y1_px", 0.0))
        draw.line((x0, y0, x1, y1), fill=(255, 205, 88), width=2)
        for cx, cy in ((x0, y0), (x1, y1)):
            draw.ellipse(
                (cx - 4.0, cy - 4.0, cx + 4.0, cy + 4.0),
                fill=(255, 205, 88),
                outline=(10, 18, 28),
                width=1,
            )
        label = f"{payload.get('label', '—')} | {float(payload.get('length_px', 0.0)):.1f} px"
        font = _load_overlay_font(max(11, int(pil.size[1] * 0.018)))
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = max(1, int(bbox[2] - bbox[0]))
            th = max(1, int(bbox[3] - bbox[1]))
        except Exception:
            tw = max(80, int(pil.size[0] * 0.14))
            th = max(18, int(pil.size[1] * 0.03))
        mx = int(round((x0 + x1) * 0.5))
        my = int(round((y0 + y1) * 0.5))
        rx0 = max(6, mx - tw // 2 - 6)
        ry0 = max(6, my - th - 18)
        rx1 = min(pil.size[0] - 6, rx0 + tw + 12)
        ry1 = min(pil.size[1] - 6, ry0 + th + 8)
        draw.rounded_rectangle(
            (rx0, ry0, rx1, ry1),
            radius=6,
            fill=(10, 18, 28),
            outline=(255, 205, 88),
            width=1,
        )
        draw.text((rx0 + 6, ry0 + 4), label, fill=(248, 248, 248), font=font)
        return np.asarray(pil, dtype=np.uint8)

    def _update_view(self) -> None:
        request = self._build_render_request()
        if request is None:
            return
        self._submit_render_request(request)

    def _apply_render_result(
        self, request: dict[str, Any], view: np.ndarray, fov_meta: dict[str, Any]
    ) -> None:
        objective = int(request["objective"])
        focus_distance_mm = float(request["focus_distance_mm"])
        pan_x = float(request["pan_x"])
        pan_y = float(request["pan_y"])
        focus_quality = float(request["focus_quality"])
        base_um_per_px_100x = float(request["um_per_px_100x"])
        view_signature = request["signature"]

        view, mask_summary = self._apply_masks_to_view(view, fov_meta)
        self.current_mask_render_summary = dict(mask_summary)
        current_um_per_px = float(
            fov_meta.get("um_per_px")
            or estimate_um_per_px(
                um_per_px_100x=base_um_per_px_100x,
                crop_size_px=fov_meta.get("crop_size_px"),
                output_size_px=view.shape,
            )
        )
        self.current_scale_audit = scale_audit_report(
            objective=objective,
            source_size_px=self.current_source_gray.shape,
            crop_size_px=fov_meta.get("crop_size_px"),
            output_size_px=view.shape,
            um_per_px_100x=base_um_per_px_100x,
            actual_um_per_px=current_um_per_px,
        )
        if hasattr(self, "view"):
            self.view.set_measurement_um_per_px(current_um_per_px)
        self._update_scale_status_widgets(
            um_per_px=current_um_per_px,
            audit=self.current_scale_audit,
            source=self.current_scale_source,
        )
        rgb = np.stack([view] * 3, axis=2).astype(np.uint8)
        rgb = self._apply_overlays(rgb, fov_meta)

        route_summary = self._route_summary()
        if self.current_mask_entries:
            route_summary["mask_count_total"] = int(len(self.current_mask_entries))
        if self.current_manifest_path is not None:
            route_summary["manifest_path"] = str(self.current_manifest_path)
        controls_state = {
            "objective": objective,
            "focus_distance_mm": focus_distance_mm,
            "stage_x": pan_x,
            "stage_y": pan_y,
            "optical_mode": self._current_optical_mode(),
            "optical_mode_parameters": dict(self._current_optical_mode_parameters()),
            "psf_profile": str(self.psf_profile_combo.currentData() or "standard"),
            "psf_strength": float(self.psf_strength_spin.value()),
            "sectioning_shear_deg": float(self.sectioning_shear_spin.value()),
            "hybrid_balance": float(self.hybrid_balance_spin.value()),
        }

        microscope_params = {
            "magnification": objective,
            "focus": focus_quality,
            "focus_distance_mm": focus_distance_mm,
            "brightness": float(self.brightness.value()),
            "contrast": float(self.contrast.value()),
            "optical_mode": self._current_optical_mode(),
            "optical_mode_parameters": dict(self._current_optical_mode_parameters()),
            "psf_profile": str(self.psf_profile_combo.currentData() or "standard"),
            "psf_strength": float(self.psf_strength_spin.value()),
            "sectioning_shear_deg": float(self.sectioning_shear_spin.value()),
            "hybrid_balance": float(self.hybrid_balance_spin.value()),
            "output_size": [int(self.out_h.value()), int(self.out_w.value())],
            **self.applied_heavy,
        }

        self.current_capture = rgb
        self.current_capture_meta = build_capture_metadata(
            source_image=str(self.current_image_path)
            if self.current_image_path
            else "",
            source_metadata=str(self.current_meta_path)
            if self.current_meta_path
            else "",
            microscope_params=microscope_params,
            view_meta=fov_meta,
            route_summary=route_summary,
            session_id=self.session_id,
            capture_index=int(self.capture_index + 1),
            reticle_enabled=bool(self.reticle_check.isChecked()),
            scale_bar=dict(self.current_scale_bar),
            controls_state=controls_state,
            source_generator_version=self._source_generator_version(),
            prep_signature=self._source_prep_signature(),
            etch_signature=self._source_etch_signature(),
            quality_metrics=self._source_quality_metrics(),
            mask_rendering=dict(mask_summary),
        )
        if self.current_manifest_path is not None:
            self.current_capture_meta["source_manifest"] = str(
                self.current_manifest_path
            )
        self.current_capture_meta["available_masks"] = [
            {
                "category": str(item.get("category", "")),
                "name": str(item.get("name", "")),
                "path": str(item.get("path", "")),
            }
            for item in self.current_mask_entries
        ]
        self.current_capture_meta["measurement_tool"] = self._measurement_snapshot()
        self.current_capture_meta["scale_audit"] = dict(self.current_scale_audit)
        self.current_capture_meta["scale_source"] = str(self.current_scale_source)

        current_transform = self.view.transform()
        self.view_scene.clear()
        pix = _to_pixmap(rgb)
        self.view_scene.addPixmap(pix)
        self.view_scene.setSceneRect(pix.rect())
        if hasattr(self, "view"):
            self.view.clamp_measurement_to_scene()
        if self._view_needs_fit:
            self.view.resetTransform()
            self.view.fitInView(
                self.view_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
            )
            self._view_needs_fit = False
        else:
            self.view.setTransform(current_transform)

        if self.current_source_rgb is not None:
            self.navigator.set_view_state(pan_x, pan_y, objective)
        self._update_dashboard_status(
            objective=objective, focus_distance_mm=focus_distance_mm
        )
        self._update_psf_dashboard(fov_meta)
        self._last_view_signature = view_signature

        info_lines = [
            f"Сессия: {self.session_id}",
            f"Образец: {self.current_image_path.name if self.current_image_path else '-'}",
            f"Объектив: {objective}x",
            f"Фокусное расстояние: {focus_distance_mm:.2f} mm",
            f"Настройка фокуса: {'выполнена' if self.focus_user_configured else 'не настроен пользователем'}",
            f"Состояние фокуса: {self._focus_status_text(focus_quality)}",
            f"Столик XY: ({pan_x:.3f}, {pan_y:.3f})",
            f"Начало кадра: {fov_meta.get('crop_origin_px')}",
            f"Размер crop: {fov_meta.get('crop_size_px')}",
            f"Шкала: {int(round(float(self.current_scale_bar.get('bar_nm', 0))))} нм / {self.current_scale_bar.get('bar_px', 0)} px",
            f"Масштаб кадра: {current_um_per_px:.4f} мкм/px ({self.current_scale_source})",
            f"Проверка масштаба: {'OK' if bool(self.current_scale_audit.get('ok', False)) else 'проверь metadata'}",
            f"Оптический режим: {str(fov_meta.get('optical_mode', 'brightfield'))}",
            f"Профиль ТРФ: {str(fov_meta.get('psf_profile', 'standard'))}",
            f"Режим осевого профиля: {str(fov_meta.get('focus_profile_mode', fov_meta.get('axial_profile_mode', 'standard')))}",
            f"Эффективный коэффициент ГРИП: {float(fov_meta.get('effective_dof_factor', 1.0) or 1.0):.3f}",
        ]
        if bool(fov_meta.get("pure_iron_baseline_applied", False)):
            info_lines.extend(
                [
                    "Профиль чистого железа: активен",
                    f"Индекс чистоты: {float(fov_meta.get('pure_iron_cleanliness_score', 0.0) or 0.0):.3f}",
                    f"Подавление тёмных дефектов: {float(fov_meta.get('pure_iron_dark_defect_suppression', 0.0) or 0.0):.3f}",
                    f"Видимость границ: {float(fov_meta.get('pure_iron_boundary_visibility_score', 0.0) or 0.0):.3f}",
                    f"Коэффициент гашения в поляризованном свете: {float(fov_meta.get('pure_iron_polarized_extinction_score', 0.0) or 0.0):.3f}",
                ]
            )
            recommendation = dict(fov_meta.get("pure_iron_optical_recommendation", {}))
            if recommendation:
                info_lines.append(
                    "Рекомендация для чистого железа: "
                    f"основной={recommendation.get('default_mode', 'brightfield')}, "
                    f"диагностический={recommendation.get('diagnostic_mode', 'darkfield')}, "
                    f"дополнительный={recommendation.get('secondary_mode', 'phase_contrast')}"
                )
            if bool(fov_meta.get("pure_iron_dark_defect_warning", False)):
                info_lines.append(
                    "Предупреждение: тёмные дефекты слишком сильны для чистого ферритного железа."
                )
        if str(fov_meta.get("optical_mode", "")) == "magnetic_etching":
            info_lines.extend(
                [
                    f"Магнитное поле активно: {'да' if bool(fov_meta.get('magnetic_field_active', False)) else 'нет'}",
                    f"Доля ферромагнитной области: {float(fov_meta.get('ferromagnetic_fraction', 0.0) or 0.0):.3f}",
                    f"Доля магнитного сигнала: {float(fov_meta.get('magnetic_signal_fraction', 0.0) or 0.0):.3f}",
                ]
            )
        if str(fov_meta.get("psf_profile", "standard")) != "standard":
            info_lines.append("Режим: исследовательская оптика")
        if bool(fov_meta.get("sectioning_active", False)):
            info_lines.extend(
                [
                    f"Секционирование: активно",
                    f"Коэффициент подавления: {float(fov_meta.get('sectioning_suppression_score', 0.0) or 0.0):.3f}",
                    f"Сигнатура осевого сдвига: {float(fov_meta.get('axial_shift_signature', 0.0) or 0.0):.3f}",
                ]
            )
        info_lines.append(
            "Маски: "
            + (
                "выкл"
                if not bool(mask_summary.get("enabled", False))
                else f"всего={mask_summary.get('mask_count_total', 0)}, "
                f"фазы={mask_summary.get('mask_count_phase', 0)}, "
                f"признаки={mask_summary.get('mask_count_feature', 0)}, "
                f"подготовка={mask_summary.get('mask_count_prep', 0)}"
            )
        )
        if route_summary.get("final_stage"):
            info_lines.append(f"Стадия: {route_summary.get('final_stage')}")
        if route_summary.get("route_name"):
            info_lines.append(f"Маршрут: {route_summary.get('route_name')}")
        if route_summary.get("step_count") is not None:
            info_lines.append(f"Шагов маршрута: {route_summary.get('step_count')}")
        if route_summary.get("prep_step_count") is not None:
            info_lines.append(
                f"Шагов подготовки: {route_summary.get('prep_step_count')}"
            )
        if route_summary.get("etch_reagent"):
            info_lines.append(
                f"Травление: {route_summary.get('etch_reagent')} / {float(route_summary.get('etch_time_s', 0.0)):.1f} c"
            )
        generator_version = self._source_generator_version()
        if generator_version:
            info_lines.append(f"Источник: {generator_version}")
        if self.current_manifest_path is not None:
            info_lines.append(f"Манифест: {self.current_manifest_path.name}")
        self.info.setPlainText("\n".join(info_lines))

    def _browse_samples_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Папка с образцами")
        if path:
            self.dir_edit.setText(path)
            self._scan_samples()

    def _browse_save_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Папка сохранения")
        if path:
            self.save_dir.setText(path)

    def _profile_path(self) -> Path:
        raw = self.profile_path_edit.text().strip()
        return Path(raw) if raw else self.DEFAULT_PROFILE_PATH

    def _legacy_focus_distance_mm(
        self, coarse: Any, fine: Any, objective: int | None = None
    ) -> float:
        mag = int(objective if objective is not None else self._current_objective())
        base = self._objective_nominal_distance_mm(mag)
        coarse_val = _safe_float(coarse, 0.9)
        fine_val = _safe_float(fine, 0.0)
        normalized = float(np.clip(coarse_val + fine_val, 0.0, 1.0))
        return float(base - 4.0 + normalized * 8.0)

    def _profile_payload_from_controls(self) -> dict[str, Any]:
        focus_distance_mm = self._current_focus_distance_mm()
        return {
            "version": 1,
            "heavy": dict(self.applied_heavy),
            "optics": {
                "focus_distance_mm": focus_distance_mm,
                "focus_user_configured": bool(self.focus_user_configured),
                "focus_coarse": float(
                    np.clip(
                        (
                            focus_distance_mm
                            - (self._objective_nominal_distance_mm() - 4.0)
                        )
                        / 8.0,
                        0.0,
                        1.0,
                    )
                ),
                "focus_fine": 0.0,
                "brightness": float(self.brightness.value()),
                "contrast": float(self.contrast.value()),
                "optical_mode": self._current_optical_mode(),
                "optical_mode_parameters": dict(
                    self._current_optical_mode_parameters()
                ),
                "psf_profile": str(self.psf_profile_combo.currentData() or "standard"),
                "psf_strength": float(self.psf_strength_spin.value()),
                "sectioning_shear_deg": float(self.sectioning_shear_spin.value()),
                "hybrid_balance": float(self.hybrid_balance_spin.value()),
            },
            "stage": {
                "step_small": float(self.stage_step_small.value()),
                "step_large": float(self.stage_step_large.value()),
            },
            "overlays": {
                "reticle_enabled": bool(self.reticle_check.isChecked()),
                "scale_bar_enabled": bool(self.scale_check.isChecked()),
                "measurement_tool_mode": str(self.measurement_tool_mode),
                "measurement_ruler_enabled": bool(
                    self.measurement_tool_mode == ZoomView.TOOL_LINE
                ),
            },
            "mask_rendering": {
                "enabled": bool(self.DEFAULT_MASK_RENDERING.get("enabled", True)),
                "phase_strength": float(
                    self.DEFAULT_MASK_RENDERING.get("phase_strength", 1.0)
                ),
                "feature_strength": float(
                    self.DEFAULT_MASK_RENDERING.get("feature_strength", 0.95)
                ),
                "prep_strength": float(
                    self.DEFAULT_MASK_RENDERING.get("prep_strength", 0.9)
                ),
            },
        }

    def _apply_profile_payload(self, payload: dict[str, Any]) -> None:
        heavy = payload.get("heavy", {})
        if isinstance(heavy, dict):
            self.applied_heavy["uneven_strength"] = float(
                heavy.get(
                    "uneven_strength", self.applied_heavy.get("uneven_strength", 0.08)
                )
            )
            self.applied_heavy["add_dust"] = bool(
                heavy.get("add_dust", self.applied_heavy.get("add_dust", False))
            )
            self.applied_heavy["add_scratches"] = bool(
                heavy.get(
                    "add_scratches", self.applied_heavy.get("add_scratches", False)
                )
            )
            self.applied_heavy["etch_uneven"] = float(
                heavy.get("etch_uneven", self.applied_heavy.get("etch_uneven", 0.0))
            )

        optics = payload.get("optics", {})
        if isinstance(optics, dict):
            focus_distance_mm = optics.get("focus_distance_mm")
            if isinstance(focus_distance_mm, (int, float)):
                self._set_focus_distance_mm(
                    float(focus_distance_mm),
                    user_configured=bool(optics.get("focus_user_configured", False)),
                )
            else:
                self._set_focus_distance_mm(
                    self._legacy_focus_distance_mm(
                        optics.get("focus_coarse", 0.9),
                        optics.get("focus_fine", 0.0),
                    ),
                    user_configured=bool(optics.get("focus_user_configured", False)),
                )
            self.brightness.setValue(
                float(optics.get("brightness", self.brightness.value()))
            )
            self.contrast.setValue(float(optics.get("contrast", self.contrast.value())))
            optical_idx = self.optical_mode_combo.findData(
                str(optics.get("optical_mode", "brightfield"))
            )
            if optical_idx >= 0:
                self.optical_mode_combo.setCurrentIndex(optical_idx)
            optical_params = optics.get("optical_mode_parameters", {})
            if isinstance(optical_params, dict):
                self.darkfield_scatter_spin.setValue(
                    float(
                        optical_params.get(
                            "scatter_sensitivity", self.darkfield_scatter_spin.value()
                        )
                    )
                )
                self.polarized_crossed_check.setChecked(
                    bool(
                        optical_params.get(
                            "crossed_polars", self.polarized_crossed_check.isChecked()
                        )
                    )
                )
                phase_idx = self.phase_plate_combo.findData(
                    str(
                        optical_params.get(
                            "phase_plate_type",
                            self.phase_plate_combo.currentData() or "positive",
                        )
                    )
                )
                if phase_idx >= 0:
                    self.phase_plate_combo.setCurrentIndex(phase_idx)
            psf_idx = self.psf_profile_combo.findData(
                str(optics.get("psf_profile", "standard"))
            )
            if psf_idx >= 0:
                self.psf_profile_combo.setCurrentIndex(psf_idx)
            self.psf_strength_spin.setValue(
                float(optics.get("psf_strength", self.psf_strength_spin.value()))
            )
            self.sectioning_shear_spin.setValue(
                float(
                    optics.get(
                        "sectioning_shear_deg", self.sectioning_shear_spin.value()
                    )
                )
            )
            self.hybrid_balance_spin.setValue(
                float(optics.get("hybrid_balance", self.hybrid_balance_spin.value()))
            )

        stage = payload.get("stage", {})
        if isinstance(stage, dict):
            self.stage_step_small.setValue(
                float(stage.get("step_small", self.stage_step_small.value()))
            )
            self.stage_step_large.setValue(
                float(stage.get("step_large", self.stage_step_large.value()))
            )

        overlays = payload.get("overlays", {})
        if isinstance(overlays, dict):
            self.reticle_check.setChecked(
                bool(overlays.get("reticle_enabled", self.reticle_check.isChecked()))
            )
            self.scale_check.setChecked(
                bool(overlays.get("scale_bar_enabled", self.scale_check.isChecked()))
            )
            tool_mode = overlays.get("measurement_tool_mode")
            if not isinstance(tool_mode, str):
                tool_mode = (
                    ZoomView.TOOL_LINE
                    if bool(overlays.get("measurement_ruler_enabled", False))
                    else ZoomView.TOOL_OFF
                )
            self._set_measurement_tool_mode(str(tool_mode))

        # UI-настройки масок удалены: используется фиксированный набор DEFAULT_MASK_RENDERING.

        self._apply_heavy()

    def _load_profile_on_start(self) -> None:
        path = self._profile_path()
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            save_json(self._profile_payload_from_controls(), path)
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._apply_profile_payload(payload)
        except Exception:
            return

    def _load_profile_from_ui(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить профиль микроскопа",
            str(self._profile_path()),
            "JSON (*.json)",
        )
        if not path:
            return
        self.profile_path_edit.setText(path)
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Ожидается JSON-объект")
            self._apply_profile_payload(payload)
            QMessageBox.information(self, "Профиль", f"Профиль загружен:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Профиль", f"Ошибка загрузки профиля: {exc}")

    def _save_profile_from_ui(self) -> None:
        default = str(self._profile_path())
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить профиль микроскопа", default, "JSON (*.json)"
        )
        if not path:
            return
        self.profile_path_edit.setText(path)
        payload = self._profile_payload_from_controls()
        save_json(payload, path)
        QMessageBox.information(self, "Профиль", f"Профиль сохранен:\n{path}")

    def _save_capture(self) -> None:
        if self.current_capture is None or self.current_capture_meta is None:
            QMessageBox.warning(self, "Сохранение", "Нет данных для сохранения")
            return
        out = Path(self.save_dir.text().strip())
        out.mkdir(parents=True, exist_ok=True)
        prefix = self.save_prefix.text().strip() or "capture_v2"
        self.capture_index += 1
        image_path = out / f"{prefix}_{self.capture_index:03d}.png"
        meta_path = out / f"{prefix}_{self.capture_index:03d}.json"

        self.current_capture_meta["capture_index"] = int(self.capture_index)
        self.current_capture_meta["measurement_tool"] = self._measurement_snapshot()
        if isinstance(self.current_capture_meta.get("measurement_tool"), dict):
            self.current_capture_meta["measurement_tool"]["rendered_on_capture"] = bool(
                self.current_capture_meta["measurement_tool"]
                .get("current", {})
                .get("valid", False)
            )
        capture_to_save = self._apply_measurement_overlay_to_capture(
            self.current_capture
        )
        save_image(capture_to_save, image_path)
        save_json(self.current_capture_meta, meta_path)

        self._append_journal_row(image_path=image_path, meta_path=meta_path)
        QMessageBox.information(
            self, "Сохранение", f"Сохранено:\n{image_path}\n{meta_path}"
        )

    def _append_journal_row(self, image_path: Path, meta_path: Path) -> None:
        meta = self.current_capture_meta or {}
        controls = meta.get("controls_state", {})
        route = (
            meta.get("route_summary", {})
            if isinstance(meta.get("route_summary"), dict)
            else {}
        )
        row = self.journal.rowCount()
        self.journal.insertRow(row)
        self.journal.setItem(row, 0, QTableWidgetItem(now_iso()))
        self.journal.setItem(
            row,
            1,
            QTableWidgetItem(
                self.current_image_path.name if self.current_image_path else "-"
            ),
        )
        self.journal.setItem(
            row, 2, QTableWidgetItem(str(controls.get("objective", "-")))
        )
        focus_val = float(
            controls.get("focus_distance_mm", self._current_focus_distance_mm())
        )
        self.journal.setItem(row, 3, QTableWidgetItem(f"{focus_val:.2f} mm"))
        xy = f"({float(controls.get('stage_x', 0.5)):.3f}, {float(controls.get('stage_y', 0.5)):.3f})"
        self.journal.setItem(row, 4, QTableWidgetItem(xy))
        stage_text = str(route.get("final_stage", "-"))
        psf_profile = str(controls.get("psf_profile", "standard"))
        if psf_profile != "standard":
            stage_text = f"{stage_text} | research:{psf_profile}"
        self.journal.setItem(row, 5, QTableWidgetItem(stage_text))
        self.journal.setItem(row, 6, QTableWidgetItem(str(image_path)))

        self.journal_records.append(
            {
                "image": str(image_path),
                "meta": str(meta_path),
                "controls": dict(controls),
                "source_image": str(self.current_image_path)
                if self.current_image_path
                else "",
            }
        )

    def _restore_journal_row(self, row: int, _column: int) -> None:
        if row < 0 or row >= len(self.journal_records):
            return
        rec = self.journal_records[row]
        controls = rec.get("controls", {})
        obj = int(controls.get("objective", 200))
        self._set_objective(obj)
        if "focus_distance_mm" in controls:
            self._set_focus_distance_mm(
                float(
                    controls.get("focus_distance_mm", self._current_focus_distance_mm())
                )
            )
        else:
            self._set_focus_distance_mm(
                self._legacy_focus_distance_mm(
                    controls.get("focus_coarse", 0.9),
                    controls.get("focus_fine", 0.0),
                    objective=obj,
                )
            )
        self._set_stage_xy(
            float(controls.get("stage_x", 0.5)), float(controls.get("stage_y", 0.5))
        )
        optical_idx = self.optical_mode_combo.findData(
            str(controls.get("optical_mode", "brightfield"))
        )
        if optical_idx >= 0:
            self.optical_mode_combo.setCurrentIndex(optical_idx)
        optical_params = controls.get("optical_mode_parameters", {})
        if isinstance(optical_params, dict):
            self.darkfield_scatter_spin.setValue(
                float(
                    optical_params.get(
                        "scatter_sensitivity", self.darkfield_scatter_spin.value()
                    )
                )
            )
            self.polarized_crossed_check.setChecked(
                bool(
                    optical_params.get(
                        "crossed_polars", self.polarized_crossed_check.isChecked()
                    )
                )
            )
            phase_idx = self.phase_plate_combo.findData(
                str(
                    optical_params.get(
                        "phase_plate_type",
                        self.phase_plate_combo.currentData() or "positive",
                    )
                )
            )
            if phase_idx >= 0:
                self.phase_plate_combo.setCurrentIndex(phase_idx)
        psf_idx = self.psf_profile_combo.findData(
            str(controls.get("psf_profile", "standard"))
        )
        if psf_idx >= 0:
            self.psf_profile_combo.setCurrentIndex(psf_idx)
        if "psf_strength" in controls:
            self.psf_strength_spin.setValue(float(controls.get("psf_strength", 0.0)))
        if "sectioning_shear_deg" in controls:
            self.sectioning_shear_spin.setValue(
                float(
                    controls.get("sectioning_shear_deg", DEFAULT_SECTIONING_SHEAR_DEG)
                )
            )
        if "hybrid_balance" in controls:
            self.hybrid_balance_spin.setValue(
                float(controls.get("hybrid_balance", DEFAULT_HYBRID_BALANCE))
            )
        self._update_view()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        key = event.key()
        step = (
            float(self.stage_step_large.value())
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            else float(self.stage_step_small.value())
        )

        if key == Qt.Key.Key_W:
            self._move_stage(0.0, -step)
            return
        if key == Qt.Key.Key_S:
            self._move_stage(0.0, step)
            return
        if key == Qt.Key.Key_A:
            self._move_stage(-step, 0.0)
            return
        if key == Qt.Key.Key_D:
            self._move_stage(step, 0.0)
            return
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            mags = self.OBJECTIVES
            cur = self._current_objective()
            idx = max(0, min(len(mags) - 1, mags.index(cur) + 1))
            self._set_objective(mags[idx])
            return
        if key == Qt.Key.Key_Minus:
            mags = self.OBJECTIVES
            cur = self._current_objective()
            idx = max(0, min(len(mags) - 1, mags.index(cur) - 1))
            self._set_objective(mags[idx])
            return
        if key == Qt.Key.Key_Space:
            self._save_capture()
            return
        if key == Qt.Key.Key_Backspace:
            if hasattr(self, "view") and self.view.remove_last_polygon_vertex():
                return

        super().keyPressEvent(event)


def launch_microscope_app(samples_dir: str | Path | None = None) -> None:
    from ui_qt.spinbox_wheel_filter import SpinBoxWheelFilter

    app = QApplication.instance() or QApplication([])

    # Install global wheel event filter for spinboxes
    wheel_filter = SpinBoxWheelFilter(app)
    app.installEventFilter(wheel_filter)

    win = MicroscopeWindow(samples_dir=samples_dir)
    win.show()
    app.exec()
