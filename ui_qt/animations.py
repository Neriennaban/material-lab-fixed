from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRect, QSize, Qt
from PySide6.QtWidgets import QGraphicsOpacityEffect, QWidget


class AnimationHelper:
    """Helper class for creating smooth UI animations"""

    @staticmethod
    def fade_in(widget: QWidget, duration: int = 300) -> QPropertyAnimation:
        """Create fade-in animation for widget"""
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(duration)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        return animation

    @staticmethod
    def fade_out(widget: QWidget, duration: int = 300) -> QPropertyAnimation:
        """Create fade-out animation for widget"""
        effect = widget.graphicsEffect()
        if not effect:
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
        
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(duration)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.Type.InCubic)
        
        return animation

    @staticmethod
    def slide_in(widget: QWidget, direction: str = "left", duration: int = 400) -> QPropertyAnimation:
        """Create slide-in animation for widget
        
        Args:
            widget: Widget to animate
            direction: Direction to slide from ("left", "right", "top", "bottom")
            duration: Animation duration in milliseconds
        """
        animation = QPropertyAnimation(widget, b"geometry")
        animation.setDuration(duration)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        current_geo = widget.geometry()
        start_geo = QRect(current_geo)
        
        if direction == "left":
            start_geo.moveLeft(-current_geo.width())
        elif direction == "right":
            start_geo.moveLeft(widget.parent().width() if widget.parent() else 800)
        elif direction == "top":
            start_geo.moveTop(-current_geo.height())
        elif direction == "bottom":
            start_geo.moveTop(widget.parent().height() if widget.parent() else 600)
        
        animation.setStartValue(start_geo)
        animation.setEndValue(current_geo)
        
        return animation

    @staticmethod
    def expand_height(widget: QWidget, target_height: int, duration: int = 300) -> QPropertyAnimation:
        """Animate widget height expansion"""
        animation = QPropertyAnimation(widget, b"maximumHeight")
        animation.setDuration(duration)
        animation.setStartValue(0)
        animation.setEndValue(target_height)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        return animation

    @staticmethod
    def collapse_height(widget: QWidget, duration: int = 300) -> QPropertyAnimation:
        """Animate widget height collapse"""
        animation = QPropertyAnimation(widget, b"maximumHeight")
        animation.setDuration(duration)
        animation.setStartValue(widget.height())
        animation.setEndValue(0)
        animation.setEasingCurve(QEasingCurve.Type.InCubic)
        
        return animation

    @staticmethod
    def pulse(widget: QWidget, scale_factor: float = 1.05, duration: int = 200) -> QPropertyAnimation:
        """Create pulse animation effect"""
        animation = QPropertyAnimation(widget, b"size")
        animation.setDuration(duration)
        
        current_size = widget.size()
        target_size = QSize(
            int(current_size.width() * scale_factor),
            int(current_size.height() * scale_factor)
        )
        
        animation.setStartValue(current_size)
        animation.setEndValue(target_size)
        animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        return animation
