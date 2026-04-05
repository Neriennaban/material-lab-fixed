from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPainter, QPen, QValidator
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def normalize_decimal_input(value: Any) -> str:
    text = str(value or "").strip()
    return text.replace("\u00a0", " ").replace(" ", "").replace(",", ".")


def parse_flexible_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    normalized = normalize_decimal_input(value)
    if not normalized:
        return float(default)
    try:
        return float(normalized)
    except Exception:
        return float(default)


class FlexibleDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that accepts both comma and dot as decimal separators."""

    def _strip_affixes(self, text: str) -> str:
        raw = str(text or "").strip()
        prefix = self.prefix()
        suffix = self.suffix()
        if prefix and raw.startswith(prefix):
            raw = raw[len(prefix) :]
        if suffix and raw.endswith(suffix):
            raw = raw[: -len(suffix)]
        return raw.strip()

    def valueFromText(self, text: str) -> float:  # type: ignore[override]
        return parse_flexible_float(self._strip_affixes(text), float(self.value()))

    def validate(self, text: str, pos: int):  # type: ignore[override]
        normalized = normalize_decimal_input(self._strip_affixes(text))
        if normalized in {"", "-", "+", ".", "-.", "+."}:
            return (QValidator.State.Intermediate, text, pos)
        try:
            float(normalized)
        except Exception:
            return (QValidator.State.Invalid, text, pos)
        return (QValidator.State.Acceptable, text, pos)


class ModernCard(QFrame):
    """Modern card widget with shadow and rounded corners"""

    def __init__(self, title: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("modernCard")
        self._setup_ui(title)

    def _setup_ui(self, title: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        if title:
            title_label = QLabel(title)
            title_label.setObjectName("cardTitle")
            font = title_label.font()
            font.setPointSize(15)
            font.setWeight(QFont.Weight.Bold)
            title_label.setFont(font)
            layout.addWidget(title_label)

        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(8)
        layout.addLayout(self.content_layout)

    def add_widget(self, widget: QWidget) -> None:
        """Add widget to card content"""
        self.content_layout.addWidget(widget)


class StatusBadge(QLabel):
    """Colored status badge widget"""

    def __init__(self, text: str = "", status_type: str = "info", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setObjectName("statusBadge")
        self.status_type = status_type
        self._update_style()

    def set_status(self, text: str, status_type: str = "info") -> None:
        """Update badge text and status type"""
        self.setText(text)
        self.status_type = status_type
        self._update_style()

    def _update_style(self) -> None:
        """Apply status-specific styling"""
        colors = {
            "success": ("#059669", "#D1FAE5"),
            "warning": ("#D97706", "#FEF3C7"),
            "error": ("#DC2626", "#FEE2E2"),
            "info": ("#3B82F6", "#DBEAFE"),
            "default": ("#64748B", "#F1F5F9"),
        }
        
        fg, bg = colors.get(self.status_type, colors["default"])
        self.setStyleSheet(f"""
            QLabel#statusBadge {{
                background: {bg};
                color: {fg};
                border: 1px solid {fg};
                border-radius: 12px;
                padding: 6px 14px;
                font-weight: 600;
                font-size: 13px;
            }}
        """)


class IconButton(QPushButton):
    """Modern icon button with hover effects"""

    def __init__(self, text: str = "", icon_text: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("iconButton")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        if icon_text:
            icon_label = QLabel(icon_text)
            icon_label.setObjectName("buttonIcon")
            layout.addWidget(icon_label)

        if text:
            text_label = QLabel(text)
            text_label.setObjectName("buttonText")
            layout.addWidget(text_label)


class ProgressIndicator(QWidget):
    """Modern progress indicator with percentage"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._progress = 0
        self.setMinimumHeight(8)
        self.setMaximumHeight(8)

    def set_progress(self, value: int) -> None:
        """Set progress value (0-100)"""
        self._progress = max(0, min(100, value))
        self.update()

    def paintEvent(self, event) -> None:
        """Custom paint for smooth progress bar"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(Qt.GlobalColor.lightGray)
        painter.drawRoundedRect(self.rect(), 4, 4)

        # Progress
        if self._progress > 0:
            progress_width = int(self.width() * (self._progress / 100))
            progress_rect = self.rect()
            progress_rect.setWidth(progress_width)
            
            painter.setBrush(Qt.GlobalColor.blue)
            painter.drawRoundedRect(progress_rect, 4, 4)


class Divider(QFrame):
    """Horizontal or vertical divider line"""

    def __init__(self, orientation: Qt.Orientation = Qt.Orientation.Horizontal, parent: QWidget | None = None):
        super().__init__(parent)
        
        if orientation == Qt.Orientation.Horizontal:
            self.setFrameShape(QFrame.Shape.HLine)
            self.setMaximumHeight(1)
        else:
            self.setFrameShape(QFrame.Shape.VLine)
            self.setMaximumWidth(1)
        
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setObjectName("divider")


class CollapsibleSection(QWidget):
    """Collapsible section widget"""

    toggled = Signal(bool)

    def __init__(self, title: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._is_expanded = True
        self._setup_ui(title)

    def _setup_ui(self, title: str) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        # Header
        self.toggle_button = QPushButton(f"▼ {title}")
        self.toggle_button.setObjectName("sectionToggle")
        self.toggle_button.clicked.connect(self._toggle)
        main_layout.addWidget(self.toggle_button)

        # Content
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(16, 8, 8, 8)
        main_layout.addWidget(self.content_widget)

    def _toggle(self) -> None:
        """Toggle section visibility"""
        self._is_expanded = not self._is_expanded
        self.content_widget.setVisible(self._is_expanded)
        
        arrow = "▼" if self._is_expanded else "▶"
        current_text = self.toggle_button.text()
        new_text = f"{arrow} {current_text[2:]}"
        self.toggle_button.setText(new_text)
        
        self.toggled.emit(self._is_expanded)

    def add_widget(self, widget: QWidget) -> None:
        """Add widget to collapsible content"""
        self.content_layout.addWidget(widget)

    def is_expanded(self) -> bool:
        """Check if section is expanded"""
        return self._is_expanded
