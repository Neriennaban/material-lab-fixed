"""
Spinbox wheel event filter for improved UX.

This module provides an event filter that prevents QSpinBox and QDoubleSpinBox
widgets from responding to mouse wheel events when they don't have keyboard focus.
This prevents accidental value changes while scrolling and improves the user experience
in forms with many numeric input fields.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox


class SpinBoxWheelFilter(QObject):
    """
    Event filter to prevent spinboxes from responding to wheel events.
    This prevents accidental value changes while scrolling and improves UX.

    Usage:
        app = QApplication([])
        wheel_filter = SpinBoxWheelFilter(app)
        app.installEventFilter(wheel_filter)

    The filter blocks ALL wheel events on QSpinBox and QDoubleSpinBox widgets.
    When blocked, the event propagates to the parent widget (e.g., scroll area)
    for normal scrolling behavior. Users must use keyboard arrows or manual
    input to change spinbox values.
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """
        Filter wheel events for all spinboxes.

        Args:
            obj: The object receiving the event
            event: The event to filter

        Returns:
            True if event should be blocked, False to pass through
        """
        if event.type() == QEvent.Type.Wheel:
            if isinstance(obj, (QSpinBox, QDoubleSpinBox)):
                # Allow wheel events on spinboxes explicitly marked as wheel-enabled
                if obj.property("wheelEnabled"):
                    return False
                # Block wheel events on all other spinboxes
                event.ignore()
                return True  # Event handled (blocked)

        return super().eventFilter(obj, event)
