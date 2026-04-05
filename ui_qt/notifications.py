from __future__ import annotations

from PySide6.QtCore import QPropertyAnimation, QRect, Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class NotificationWidget(QFrame):
    """Modern notification toast widget"""

    def __init__(
        self,
        message: str,
        notification_type: str = "info",
        duration: int = 3000,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("notificationToast")
        self.notification_type = notification_type
        self._setup_ui(message)
        self._apply_style()

        if duration > 0:
            QTimer.singleShot(duration, self.fade_out)

    def _setup_ui(self, message: str) -> None:
        """Setup notification UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        # Icon
        icon_label = QLabel(self._get_icon())
        icon_label.setObjectName("notificationIcon")
        layout.addWidget(icon_label)

        # Message
        message_label = QLabel(message)
        message_label.setObjectName("notificationMessage")
        message_label.setWordWrap(True)
        layout.addWidget(message_label, 1)

        # Close button
        close_btn = QPushButton("✕")
        close_btn.setObjectName("notificationClose")
        close_btn.setFixedSize(24, 24)
        close_btn.clicked.connect(self.fade_out)
        layout.addWidget(close_btn)

        self.setFixedHeight(60)
        self.setMinimumWidth(300)
        self.setMaximumWidth(500)

    def _get_icon(self) -> str:
        """Get icon for notification type"""
        icons = {
            "success": "✓",
            "warning": "⚠",
            "error": "✕",
            "info": "ℹ",
        }
        return icons.get(self.notification_type, "ℹ")

    def _apply_style(self) -> None:
        """Apply notification type styling"""
        colors = {
            "success": ("#059669", "#D1FAE5", "#047857"),
            "warning": ("#D97706", "#FEF3C7", "#B45309"),
            "error": ("#DC2626", "#FEE2E2", "#B91C1C"),
            "info": ("#3B82F6", "#DBEAFE", "#2563EB"),
        }

        fg, bg, border = colors.get(self.notification_type, colors["info"])

        self.setStyleSheet(f"""
            QFrame#notificationToast {{
                background: {bg};
                border: 2px solid {border};
                border-radius: 12px;
            }}
            QLabel#notificationIcon {{
                color: {fg};
                font-size: 20px;
                font-weight: bold;
            }}
            QLabel#notificationMessage {{
                color: {fg};
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton#notificationClose {{
                background: transparent;
                color: {fg};
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton#notificationClose:hover {{
                background: {fg};
                color: {bg};
            }}
        """)

    def fade_out(self) -> None:
        """Fade out and close notification"""
        self._fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self._fade_animation.setDuration(300)
        self._fade_animation.setStartValue(1.0)
        self._fade_animation.setEndValue(0.0)
        self._fade_animation.finished.connect(self.deleteLater)
        self._fade_animation.start()


class NotificationManager:
    """Manager for displaying notifications"""

    def __init__(self, parent: QWidget):
        self.parent = parent
        self.notifications: list[NotificationWidget] = []
        self.container = QWidget(parent)
        self.container.setObjectName("notificationContainer")

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        layout.addStretch()

        self.container.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, False
        )
        self.container.raise_()

    def show_notification(
        self,
        message: str,
        notification_type: str = "info",
        duration: int = 3000,
    ) -> None:
        """Show a notification toast"""
        notification = NotificationWidget(
            message, notification_type, duration, self.container
        )

        layout = self.container.layout()
        layout.insertWidget(layout.count() - 1, notification)

        self.notifications.append(notification)
        notification.destroyed.connect(lambda: self._remove_notification(notification))

        self._update_positions()

    def _remove_notification(self, notification: NotificationWidget) -> None:
        """Remove notification from list"""
        if notification in self.notifications:
            self.notifications.remove(notification)
        self._update_positions()

    def _update_positions(self) -> None:
        """Update notification positions"""
        parent_width = self.parent.width()
        parent_height = self.parent.height()

        self.container.setGeometry(0, 0, parent_width, parent_height)
        self.container.raise_()

    def success(self, message: str, duration: int = 3000) -> None:
        """Show success notification"""
        self.show_notification(message, "success", duration)

    def warning(self, message: str, duration: int = 4000) -> None:
        """Show warning notification"""
        self.show_notification(message, "warning", duration)

    def error(self, message: str, duration: int = 5000) -> None:
        """Show error notification"""
        self.show_notification(message, "error", duration)

    def info(self, message: str, duration: int = 3000) -> None:
        """Show info notification"""
        self.show_notification(message, "info", duration)
