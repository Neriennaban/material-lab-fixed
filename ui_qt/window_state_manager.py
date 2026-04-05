"""
Window State Manager - управление сохранением и восстановлением состояния окон.

Использует QStandardPaths для кросс-платформенного хранения конфигурации.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtCore import QStandardPaths, QRect, QPoint, QSize, Qt
from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtGui import QScreen


class WindowStateManager:
    """Управление состоянием окон через JSON конфигурацию."""

    SCHEMA_VERSION = "1.0"

    def __init__(self, app_name: str):
        """
        Инициализация менеджера состояния окна.

        Args:
            app_name: Имя приложения для идентификации в конфиге (например, "generator" или "microscope")
        """
        self.app_name = app_name
        self.config_path = self._get_config_path()
        self.config_data = self._load_config()

    def _get_config_path(self) -> Path:
        """Получить путь к файлу конфигурации (кросс-платформенный)."""
        # Используем AppDataLocation для хранения конфигурации
        app_data = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)

        if not app_data:
            # Fallback на домашнюю директорию
            app_data = str(Path.home() / ".material_lab")

        config_dir = Path(app_data)
        config_dir.mkdir(parents=True, exist_ok=True)

        return config_dir / "window_state.json"

    def _load_config(self) -> Dict[str, Any]:
        """Загрузить конфигурацию из файла."""
        if not self.config_path.exists():
            return self._get_default_config()

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Проверка версии схемы
            if data.get("schema_version") != self.SCHEMA_VERSION:
                print(f"Warning: Config schema version mismatch. Using defaults.")
                return self._get_default_config()

            return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading config: {e}. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Получить конфигурацию по умолчанию."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "generator_window": {
                "geometry": {"x": 100, "y": 100, "width": 1920, "height": 1080},
                "state": "maximized",
                "kiosk_mode": False
            },
            "microscope_window": {
                "geometry": {"x": 100, "y": 100, "width": 1920, "height": 1080},
                "state": "maximized",
                "kiosk_mode": False
            }
        }

    def _save_config(self) -> None:
        """Сохранить конфигурацию в файл."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving config: {e}")

    def save_state(self, window: QMainWindow) -> None:
        """
        Сохранить текущее состояние окна.

        Args:
            window: Окно для сохранения состояния
        """
        window_key = f"{self.app_name}_window"

        # Получаем геометрию окна
        geometry = window.geometry()

        # Определяем состояние окна
        if window.isFullScreen():
            state = "fullscreen"
        elif window.isMaximized():
            state = "maximized"
        else:
            state = "normal"

        # Проверяем kiosk mode (если есть атрибут)
        kiosk_mode = getattr(window, '_kiosk_mode', False)

        # Сохраняем в конфиг
        self.config_data[window_key] = {
            "geometry": {
                "x": geometry.x(),
                "y": geometry.y(),
                "width": geometry.width(),
                "height": geometry.height()
            },
            "state": state,
            "kiosk_mode": kiosk_mode
        }

        self._save_config()

    def restore_state(self, window: QMainWindow) -> None:
        """
        Восстановить сохраненное состояние окна.

        Args:
            window: Окно для восстановления состояния
        """
        window_key = f"{self.app_name}_window"

        if window_key not in self.config_data:
            # Нет сохраненного состояния - применяем дефолт (maximized)
            self._apply_default_state(window)
            return

        window_state = self.config_data[window_key]

        # Восстанавливаем геометрию
        geometry_data = window_state.get("geometry", {})
        geometry = QRect(
            geometry_data.get("x", 100),
            geometry_data.get("y", 100),
            geometry_data.get("width", 1920),
            geometry_data.get("height", 1080)
        )

        # Проверяем, что окно не за пределами экрана
        geometry = self._ensure_on_screen(geometry)

        window.setGeometry(geometry)

        # Восстанавливаем состояние окна
        state = window_state.get("state", "maximized")

        if state == "maximized":
            window.showMaximized()
        elif state == "fullscreen":
            window.showFullScreen()
        else:
            window.showNormal()

        # Восстанавливаем kiosk mode флаг (но не активируем его сразу)
        kiosk_mode = window_state.get("kiosk_mode", False)
        setattr(window, '_kiosk_mode', kiosk_mode)

    def _apply_default_state(self, window: QMainWindow) -> None:
        """Применить состояние по умолчанию (maximized)."""
        window.showMaximized()
        setattr(window, '_kiosk_mode', False)

    def _ensure_on_screen(self, geometry: QRect) -> QRect:
        """
        Убедиться, что окно находится на видимом экране.

        Args:
            geometry: Геометрия окна

        Returns:
            Скорректированная геометрия
        """
        app = QApplication.instance()
        if not app:
            return geometry

        # Получаем все доступные экраны
        screens = app.screens()
        if not screens:
            return geometry

        # Проверяем, пересекается ли окно хотя бы с одним экраном
        window_center = geometry.center()

        for screen in screens:
            screen_geometry = screen.geometry()
            if screen_geometry.contains(window_center):
                # Окно на видимом экране
                return geometry

        # Окно за пределами всех экранов - центрируем на primary screen
        primary_screen = app.primaryScreen()
        if not primary_screen:
            return geometry

        screen_geometry = primary_screen.availableGeometry()

        # Масштабируем окно до 90% экрана, если оно больше
        new_width = min(geometry.width(), int(screen_geometry.width() * 0.9))
        new_height = min(geometry.height(), int(screen_geometry.height() * 0.9))

        # Центрируем на экране
        new_x = screen_geometry.x() + (screen_geometry.width() - new_width) // 2
        new_y = screen_geometry.y() + (screen_geometry.height() - new_height) // 2

        return QRect(new_x, new_y, new_width, new_height)
