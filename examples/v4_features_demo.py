"""
Демонстрация новых возможностей V4
"""

from __future__ import annotations

import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.cache_manager import AdvancedCache
from core.logger import setup_logger
from core.performance import get_monitor, measure
from core.validators import RangeValidator, ValidationResult
from ui_qt.animations import AnimationHelper
from ui_qt.modern_widgets import (
    CollapsibleSection,
    Divider,
    ModernCard,
    ProgressIndicator,
    StatusBadge,
)
from ui_qt.notifications import NotificationManager
from ui_qt.theme_mirea import build_qss, load_theme_mode


class V4DemoWindow(QMainWindow):
    """Демонстрационное окно с новыми возможностями V4"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Демонстрация возможностей V4")
        self.setMinimumSize(1000, 700)

        # Настройка логгера
        self.logger = setup_logger("v4_demo", log_dir="logs")
        self.logger.info("Запуск демонстрационного приложения V4")

        # Кэш
        self.cache = AdvancedCache(cache_dir=".cache/demo", default_ttl=300)

        # Применение темы
        theme_mode = "light"
        self.setStyleSheet(build_qss(theme_mode))

        # Создание UI
        self._setup_ui()

        # Менеджер уведомлений
        self.notifications = NotificationManager(self)

        self.logger.info("Приложение инициализировано")

    def _setup_ui(self):
        """Настройка интерфейса"""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Заголовок
        title = QLabel("🚀 Новые возможности V4")
        title.setObjectName("sectionHeader")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Горизонтальный контейнер для карточек
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(16)

        # Карточка 1: Компоненты UI
        ui_card = self._create_ui_components_card()
        cards_layout.addWidget(ui_card)

        # Карточка 2: Backend возможности
        backend_card = self._create_backend_card()
        cards_layout.addWidget(backend_card)

        layout.addLayout(cards_layout)

        # Разделитель
        layout.addWidget(Divider())

        # Сворачиваемая секция
        collapsible = CollapsibleSection("Дополнительная информация")
        info_label = QLabel(
            "V4 включает множество улучшений:\n"
            "• Современный дизайн с улучшенной типографикой\n"
            "• Система анимаций для плавных переходов\n"
            "• Продвинутое кэширование с TTL и LRU\n"
            "• Мониторинг производительности\n"
            "• Валидация данных\n"
            "• Улучшенное логирование"
        )
        collapsible.add_widget(info_label)
        layout.addWidget(collapsible)

        layout.addStretch()

    def _create_ui_components_card(self) -> ModernCard:
        """Создание карточки с UI компонентами"""
        card = ModernCard("UI Компоненты")

        # Статус-бейджи
        badges_layout = QHBoxLayout()
        badges_layout.addWidget(StatusBadge("Успех", "success"))
        badges_layout.addWidget(StatusBadge("Предупреждение", "warning"))
        badges_layout.addWidget(StatusBadge("Ошибка", "error"))
        badges_layout.addWidget(StatusBadge("Инфо", "info"))
        badges_layout.addStretch()

        badges_widget = QWidget()
        badges_widget.setLayout(badges_layout)
        card.add_widget(badges_widget)

        # Индикатор прогресса
        progress_label = QLabel("Индикатор прогресса:")
        card.add_widget(progress_label)

        self.progress = ProgressIndicator()
        self.progress.set_progress(65)
        card.add_widget(self.progress)

        # Кнопки для демонстрации
        buttons_layout = QHBoxLayout()

        btn_animate = QPushButton("Анимация")
        btn_animate.setObjectName("primaryCta")
        btn_animate.clicked.connect(self._demo_animation)
        buttons_layout.addWidget(btn_animate)

        btn_notify = QPushButton("Уведомления")
        btn_notify.setObjectName("secondaryCta")
        btn_notify.clicked.connect(self._demo_notifications)
        buttons_layout.addWidget(btn_notify)

        buttons_widget = QWidget()
        buttons_widget.setLayout(buttons_layout)
        card.add_widget(buttons_widget)

        return card

    def _create_backend_card(self) -> ModernCard:
        """Создание карточки с backend возможностями"""
        card = ModernCard("Backend возможности")

        # Кнопка для демонстрации кэша
        btn_cache = QPushButton("Тест кэша")
        btn_cache.clicked.connect(self._demo_cache)
        card.add_widget(btn_cache)

        # Кнопка для демонстрации производительности
        btn_perf = QPushButton("Тест производительности")
        btn_perf.clicked.connect(self._demo_performance)
        card.add_widget(btn_perf)

        # Кнопка для демонстрации валидации
        btn_validate = QPushButton("Тест валидации")
        btn_validate.clicked.connect(self._demo_validation)
        card.add_widget(btn_validate)

        # Кнопка для показа статистики
        btn_stats = QPushButton("Показать статистику")
        btn_stats.setObjectName("accentCta")
        btn_stats.clicked.connect(self._show_stats)
        card.add_widget(btn_stats)

        return card

    def _demo_animation(self):
        """Демонстрация анимаций"""
        self.logger.info("Демонстрация анимаций")

        # Анимация прогресс-бара
        for i in range(0, 101, 5):
            self.progress.set_progress(i)
            QApplication.processEvents()

        self.notifications.success("Анимация завершена!")

    def _demo_notifications(self):
        """Демонстрация уведомлений"""
        self.logger.info("Демонстрация уведомлений")

        self.notifications.info("Это информационное уведомление")
        self.notifications.success("Операция выполнена успешно!")
        self.notifications.warning("Внимание: проверьте данные")
        self.notifications.error("Произошла ошибка")

    def _demo_cache(self):
        """Демонстрация кэширования"""
        self.logger.info("Демонстрация кэширования")

        with measure("cache_demo"):
            # Первый вызов - вычисление
            key = {"operation": "demo", "value": 42}
            result = self.cache.get_or_compute(
                key, lambda: sum(range(1000000)), ttl=60
            )

            # Второй вызов - из кэша
            cached_result = self.cache.get(key)

        stats = self.cache.stats()
        self.notifications.success(
            f"Кэш работает! Hit rate: {stats['hit_rate']:.1%}"
        )

    def _demo_performance(self):
        """Демонстрация мониторинга производительности"""
        self.logger.info("Демонстрация мониторинга производительности")

        with measure("performance_demo"):
            # Имитация работы
            total = sum(range(1000000))

        monitor = get_monitor()
        stats = monitor.get_stats("performance_demo")

        if stats:
            self.notifications.info(
                f"Операция выполнена за {stats['mean']:.3f}s"
            )

    def _demo_validation(self):
        """Демонстрация валидации"""
        self.logger.info("Демонстрация валидации")

        result = ValidationResult()

        # Валидация температуры
        validator = RangeValidator("temperature", min_value=0, max_value=1500)
        validator.validate(850, result)

        # Валидация некорректного значения
        validator.validate(2000, result)

        if result.is_valid:
            self.notifications.success("Все данные валидны")
        else:
            error_msg = "\n".join(
                f"{e.field}: {e.message}" for e in result.errors
            )
            self.notifications.warning(f"Ошибки валидации:\n{error_msg}")

    def _show_stats(self):
        """Показать статистику"""
        self.logger.info("Показ статистики")

        # Статистика кэша
        cache_stats = self.cache.stats()

        # Статистика производительности
        perf_monitor = get_monitor()
        perf_report = perf_monitor.report()

        # Вывод в консоль
        print("\n" + "=" * 50)
        print("СТАТИСТИКА КЭША:")
        print(f"  Запросов: {cache_stats['total_requests']}")
        print(f"  Попаданий: {cache_stats['hits']}")
        print(f"  Промахов: {cache_stats['misses']}")
        print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
        print(f"  Элементов в памяти: {cache_stats['memory_items']}")
        print("\n" + perf_report)
        print("=" * 50 + "\n")

        self.notifications.info("Статистика выведена в консоль")

    def resizeEvent(self, event):
        """Обработка изменения размера окна"""
        super().resizeEvent(event)
        if hasattr(self, "notifications"):
            self.notifications._update_positions()


def main():
    """Главная функция"""
    app = QApplication(sys.argv)

    window = V4DemoWindow()
    window.show()

    # Приветственное уведомление
    window.notifications.success("Добро пожаловать в V4!")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
