"""
Window Mode Mixin - добавление функциональности управления режимами окна.

Предоставляет методы для переключения между обычным, fullscreen и kiosk режимами.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow
from PySide6.QtGui import QShortcut, QKeySequence

from ui_qt.window_state_manager import WindowStateManager


class WindowModeMixin:
    """
    Mixin для добавления управления режимами окна в QMainWindow.

    Предоставляет:
    - Kiosk mode (F11, Shift+F11): fullscreen + скрытие menu/status bar
    - Exit kiosk mode (Esc): выход из kiosk mode
    - Fullscreen toggle (Alt+Enter): обычный fullscreen без скрытия UI
    """

    def setup_window_modes(self, state_manager: WindowStateManager) -> None:
        """
        Инициализация управления режимами окна.

        Args:
            state_manager: Менеджер состояния окна для сохранения/загрузки
        """
        if not isinstance(self, QMainWindow):
            raise TypeError("WindowModeMixin can only be used with QMainWindow subclasses")

        self.state_manager = state_manager
        self._kiosk_mode = False
        self._menu_bar_visible = True
        self._status_bar_visible = True

        # Создаем горячие клавиши
        self._setup_shortcuts()

        # Восстанавливаем сохраненное состояние
        state_manager.restore_state(self)

    def _setup_shortcuts(self) -> None:
        """Создать горячие клавиши для управления режимами окна."""
        # F11 - toggle kiosk mode
        self.shortcut_kiosk_f11 = QShortcut(QKeySequence(Qt.Key.Key_F11), self)
        self.shortcut_kiosk_f11.activated.connect(self.toggle_kiosk_mode)

        # Shift+F11 - альтернативная комбинация для kiosk mode
        self.shortcut_kiosk_shift_f11 = QShortcut(
            QKeySequence(Qt.Modifier.SHIFT | Qt.Key.Key_F11), self
        )
        self.shortcut_kiosk_shift_f11.activated.connect(self.toggle_kiosk_mode)

        # Esc - выход из kiosk mode (только если активен)
        self.shortcut_exit_kiosk = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.shortcut_exit_kiosk.activated.connect(self.exit_kiosk_mode)

        # Alt+Enter - обычный fullscreen (опционально)
        self.shortcut_fullscreen = QShortcut(
            QKeySequence(Qt.Modifier.ALT | Qt.Key.Key_Return), self
        )
        self.shortcut_fullscreen.activated.connect(self.toggle_fullscreen)

    def toggle_kiosk_mode(self) -> None:
        """
        Переключить kiosk mode.

        Kiosk mode = fullscreen + скрытие menu bar и status bar.
        """
        if self._kiosk_mode:
            self._exit_kiosk_mode_impl()
        else:
            self._enter_kiosk_mode_impl()

    def _enter_kiosk_mode_impl(self) -> None:
        """Войти в kiosk mode."""
        self._kiosk_mode = True

        # Сохраняем текущее состояние видимости UI элементов
        menu_bar = self.menuBar()
        status_bar = self.statusBar()

        if menu_bar:
            self._menu_bar_visible = menu_bar.isVisible()
            menu_bar.hide()

        if status_bar:
            self._status_bar_visible = status_bar.isVisible()
            status_bar.hide()

        # Переходим в fullscreen
        self.showFullScreen()

    def _exit_kiosk_mode_impl(self) -> None:
        """Выйти из kiosk mode."""
        self._kiosk_mode = False

        # Восстанавливаем видимость UI элементов
        menu_bar = self.menuBar()
        status_bar = self.statusBar()

        if menu_bar and self._menu_bar_visible:
            menu_bar.show()

        if status_bar and self._status_bar_visible:
            status_bar.show()

        # Выходим из fullscreen
        self.showNormal()

    def exit_kiosk_mode(self) -> None:
        """
        Выйти из kiosk mode (только если активен).

        Используется для обработки Esc - выходит только если в kiosk mode.
        """
        if self._kiosk_mode:
            self._exit_kiosk_mode_impl()

    def toggle_fullscreen(self) -> None:
        """
        Переключить обычный fullscreen (без скрытия UI).

        Это НЕ kiosk mode - menu bar и status bar остаются видимыми.
        """
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def closeEvent(self, event) -> None:
        """
        Обработка закрытия окна - сохранение состояния.

        Note: Этот метод должен быть вызван через super() в подклассе,
        если подкласс переопределяет closeEvent.
        """
        if hasattr(self, 'state_manager'):
            self.state_manager.save_state(self)

        # Вызываем родительский closeEvent (QMainWindow всегда имеет этот метод)
        super().closeEvent(event)
