from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFontMetrics, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.alloy_validation import format_validation_report, validate_alloy
from core.cooling_modes import (
    canonicalize_cooling_mode,
    cooling_mode_label_ru,
    cooling_mode_options_ru,
)
from core.cooling_curve import normalize_cooling_curve_points
from core.contracts_v2 import (
    GenerationOutputV2,
    GenerationRequestV2,
    ProcessRoute,
    ProcessingOperation,
    ProcessingState,
    ThermoBackendConfig,
)
from core.diagram_engine import (
    available_diagram_systems,
    render_diagram_snapshot,
    save_diagram_png,
)
from core.pipeline_v2 import GenerationPipelineV2
from core.route_validation import (
    available_route_methods,
    route_templates,
    validate_process_route,
)
from core.ui_v2_utils import normalize_compare_mode
from export.export_images import save_image
from export.export_tables import save_json
from ui_qt.modern_widgets import FlexibleDoubleSpinBox, parse_flexible_float

QDoubleSpinBox = FlexibleDoubleSpinBox


@dataclass(slots=True)
class FactoryState:
    sample_name: str = "sample_v2"
    composition_mode: str = "elements"
    composition: dict[str, float] = field(
        default_factory=lambda: {"Fe": 99.2, "C": 0.8}
    )
    processing: ProcessingState = field(
        default_factory=lambda: ProcessingState(cooling_mode="auto")
    )
    route_policy: str = "route_driven"
    route_name: str = "Маршрут"
    route_operations: list[ProcessingOperation] = field(default_factory=list)
    step_preview_index: int | None = None
    step_series_enabled: bool = True
    cooling_curve_enabled: bool = False
    cooling_curve_mode: str = "per_degree"
    cooling_curve_degree_step: float = 1.0
    cooling_curve_max_points: int = 220
    cooling_curve_points: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"time_min": 0.0, "temperature_c": 20.0},
            {"time_min": 10.0, "temperature_c": 20.0},
        ]
    )
    generator: str = "auto"
    generator_params_text: str = "{}"
    seed: int = 4200
    resolution: tuple[int, int] = (1024, 1024)
    microscope_params: dict[str, Any] = field(
        default_factory=lambda: {
            "magnification": 200,
            "focus": 0.95,
            "brightness": 1.0,
            "contrast": 1.1,
            "noise_sigma": 2.2,
            "vignette_strength": 0.12,
            "uneven_strength": 0.08,
            "add_dust": False,
            "add_scratches": False,
            "etch_uneven": 0.0,
            "output_size": [1024, 1024],
        }
    )
    auto_normalize: bool = True
    strict_validation: bool = True
    thermo: ThermoBackendConfig = field(default_factory=ThermoBackendConfig)
    compare_mode: str = "single"
    batch_output_dir: str = str(Path("examples") / "factory_v2_output")
    batch_prefix: str = "sample_v2"


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=2).astype(np.uint8, copy=False)
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3].astype(np.uint8, copy=False)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _to_pixmap(image: np.ndarray) -> QPixmap:
    rgb = _to_rgb(image)
    arr = np.ascontiguousarray(rgb)
    h, w, _ = arr.shape
    qimage = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


def _safe_float(text: str, default: float = 0.0) -> float:
    return parse_flexible_float(text, default)


def _hconcat(images: list[np.ndarray], gap: int = 8, bg: int = 16) -> np.ndarray:
    prepared = [_to_rgb(img) for img in images]
    max_h = max(img.shape[0] for img in prepared)
    total_w = sum(img.shape[1] for img in prepared) + gap * (len(prepared) - 1)
    canvas = np.full((max_h, total_w, 3), bg, dtype=np.uint8)
    x = 0
    for img in prepared:
        h, w = img.shape[:2]
        y = (max_h - h) // 2
        canvas[y : y + h, x : x + w] = img
        x += w + gap
    return canvas


def _diff_map(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    b = _to_rgb(before).mean(axis=2).astype(np.int16)
    a = _to_rgb(after).mean(axis=2).astype(np.int16)
    d = np.abs(a - b).astype(np.uint8)
    r = np.clip(d.astype(np.int16) * 2, 0, 255).astype(np.uint8)
    g = np.clip(255 - d.astype(np.int16), 0, 255).astype(np.uint8)
    bl = np.clip(30 + d.astype(np.int16) // 2, 0, 255).astype(np.uint8)
    return np.stack([r, g, bl], axis=2)


class ZoomView(QGraphicsView):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        allow_drag: bool = True,
        on_resized: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_resized = on_resized
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if allow_drag
            else QGraphicsView.DragMode.NoDrag
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        factor = 1.16 if event.angleDelta().y() > 0 else 1.0 / 1.16
        self.scale(factor, factor)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._on_resized is not None:
            self._on_resized()


class WidePopupCombo(QComboBox):
    def __init__(
        self, popup_min_width: int = 300, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.popup_min_width = int(popup_min_width)
        self.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.setMinimumContentsLength(14)
        self.view().setTextElideMode(Qt.TextElideMode.ElideNone)
        self.view().setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def showPopup(self) -> None:  # type: ignore[override]
        fm = QFontMetrics(self.view().font())
        content_w = 0
        for i in range(self.count()):
            content_w = max(content_w, fm.horizontalAdvance(self.itemText(i)))
        popup_width = max(self.popup_min_width, self.width(), content_w + 64)
        self.view().setMinimumWidth(popup_width)
        super().showPopup()


class SampleFactoryWindow(QMainWindow):
    STEP_TITLES = [
        "Состав",
        "Технологический маршрут",
        "Генерация",
        "Предпросмотр и экспорт",
    ]
    ROUTE_COLUMNS = [
        "Метод",
        "T, C",
        "Время, мин",
        "Охлаждение",
        "Деф., %",
        "Старение, ч",
        "T стар., C",
        "Примечание",
    ]
    CURVE_COLUMNS = ["Время, мин", "Температура, C"]
    COOLING_MODE_OPTIONS = cooling_mode_options_ru(include_auto=True)
    PHASE_COLUMNS = ["Фаза", "Доля, %"]
    FE_C_PHASES = [
        ("Ferrite", "Феррит"),
        ("Pearlite", "Перлит"),
        ("Cementite", "Цементит"),
        ("Austenite", "Аустенит"),
        ("Martensite", "Мартенсит"),
        ("Bainite", "Бейнит"),
    ]
    METHOD_LABELS = {
        "anneal_full": "Полный отжиг",
        "anneal_recrystallization": "Рекристаллизационный отжиг",
        "normalize": "Нормализация",
        "quench_water": "Закалка в воде",
        "quench_oil": "Закалка в масле",
        "temper_low": "Отпуск низкий",
        "temper_medium": "Отпуск средний",
        "temper_high": "Отпуск высокий",
        "solution_treat": "Растворный отжиг",
        "age_natural": "Естественное старение",
        "age_artificial": "Искусственное старение",
        "overage": "Перестаривание",
        "cold_roll": "Холодная прокатка",
        "hot_roll": "Горячая прокатка",
        "forging": "Ковка",
        "drawing": "Волочение",
        "cast_slow": "Литье (медленное)",
        "cast_fast": "Литье (быстрое)",
        "directional_solidification": "Направленная кристаллизация",
        "stress_relief": "Снятие напряжений",
        "homogenize": "Гомогенизация",
    }
    GENERATOR_LABELS = {
        "auto": "Авто (универсальный)",
        "grains": "Зеренная структура",
        "pearlite": "Перлит",
        "phase_map": "Фазовая карта",
        "legacy_phase_map": "Фазовая карта (legacy)",
        "dendritic_cast": "Дендритный литейный",
        "crm_fe_c": "Fe-C (CRM)",
        "dislocations": "Дислокационные ямки",
        "eutectic": "Эвтектика",
        "calphad_phase": "CALPHAD-фазовый",
        "martensite": "Мартенсит",
        "tempered": "Отпущенная структура",
        "aged_al": "Состаренный Al-сплав",
    }
    SYSTEM_LABELS = {
        "fe-c": "Fe-C",
        "fe-si": "Fe-Si",
        "al-si": "Al-Si",
        "cu-zn": "Cu-Zn",
        "al-cu-mg": "Al-Cu-Mg",
        "custom-multicomponent": "Пользовательская многокомпонентная",
    }
    ROUTE_POLICY_LABELS = {
        "single_state": "Одиночное состояние",
        "route_driven": "Маршрут обработки",
    }
    ROUTE_TEMPLATE_LABELS = {
        "al_cu_mg_solution_age": "Al-Cu-Mg: растворный отжиг и старение",
        "cold_work_recrystallize": "Холодная деформация и рекристаллизация",
        "fe-c_quench_temper": "Fe-C: закалка и отпуск",
    }

    def __init__(
        self,
        presets_dir: str | Path | None = None,
        calphad_profile_path: str | Path | None = None,
        calphad_tdb_dir: str | Path | None = None,
        calphad_cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.pipeline = GenerationPipelineV2(
            presets_dir=presets_dir,
            calphad_profile_path=calphad_profile_path,
            calphad_tdb_dir=calphad_tdb_dir,
            calphad_cache_dir=calphad_cache_dir,
        )
        self.route_methods = available_route_methods()
        self.route_templates = route_templates()
        self.state = FactoryState()
        self.state.thermo.db_profile_path = str(self.pipeline.calphad_profile_path)
        if self.pipeline.calphad_cache_dir is not None:
            self.state.thermo.cache_dir = str(self.pipeline.calphad_cache_dir)
        self.step_status: list[str] = ["warn", "warn", "warn", "warn"]
        self.current_request: GenerationRequestV2 | None = None
        self.current_output: GenerationOutputV2 | None = None
        self.current_display: np.ndarray | None = None
        self.current_curve_outputs: list[GenerationOutputV2] = []
        self.current_diagram_snapshot: dict[str, Any] | None = None
        self.batch_queue: list[GenerationRequestV2] = []
        self._live_enabled = True
        self._heavy_dirty = False
        self._before_cache_key: str = ""
        self._before_cache_output: GenerationOutputV2 | None = None
        self._step_cache_key: str = ""
        self._step_cache_outputs: list[GenerationOutputV2] = []
        self._curve_cache_key: str = ""
        self._curve_cache_outputs: list[GenerationOutputV2] = []
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(110)
        self._preview_timer.timeout.connect(
            lambda: self._run_preview(use_heavy=False, silent=True)
        )
        self._diagram_resize_timer = QTimer(self)
        self._diagram_resize_timer.setSingleShot(True)
        self._diagram_resize_timer.setInterval(100)
        self._diagram_resize_timer.timeout.connect(self._render_diagram)

        self.setWindowTitle("Генератор образцов V2")
        self.resize(1920, 1080)
        self._build_ui()
        self._style()
        self._load_preset_names()
        self._seed_default_route()
        self._sync_widgets_from_state()
        self._validate_all_steps()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)

        split = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(split)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        steps_box = QGroupBox("Шаги мастера")
        steps_layout = QVBoxLayout(steps_box)
        self.steps_list = QListWidget()
        self.steps_list.setMinimumWidth(300)
        self.steps_list.currentRowChanged.connect(self._set_step)
        for i, title in enumerate(self.STEP_TITLES, start=1):
            self.steps_list.addItem(f"! {i}. {title}")
        steps_layout.addWidget(self.steps_list)
        self.step_help = QLabel(
            "Заполняйте шаги по порядку. Блокирующие ошибки отмечены как X."
        )
        self.step_help.setWordWrap(True)
        steps_layout.addWidget(self.step_help)
        left_layout.addWidget(steps_box)

        quick_box = QGroupBox("Быстрые действия")
        quick_grid = QGridLayout(quick_box)
        quick_preview = QPushButton("Предпросмотр")
        quick_preview.clicked.connect(
            lambda: self._run_preview(use_heavy=False, silent=False)
        )
        quick_apply = QPushButton("Применить тяжелые эффекты")
        quick_apply.clicked.connect(self._apply_heavy_and_preview)
        quick_add = QPushButton("Добавить в пакет")
        quick_add.clicked.connect(self._add_to_batch)
        quick_run = QPushButton("Запустить пакет")
        quick_run.clicked.connect(self._run_batch)
        quick_grid.addWidget(quick_preview, 0, 0)
        quick_grid.addWidget(quick_apply, 0, 1)
        quick_grid.addWidget(quick_add, 1, 0)
        quick_grid.addWidget(quick_run, 1, 1)
        left_layout.addWidget(quick_box)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        self.stack = QStackedWidget()
        right_layout.addWidget(self.stack, stretch=1)
        self._build_step_1()
        self._build_step_2()
        self._build_step_3()
        self._build_step_4()
        self.steps_list.setCurrentRow(0)
        self._set_step(0)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("Назад")
        self.next_btn = QPushButton("Далее")
        self.prev_btn.clicked.connect(self._go_prev)
        self.next_btn.clicked.connect(self._go_next)
        nav.addWidget(self.prev_btn)
        nav.addStretch(1)
        nav.addWidget(self.next_btn)
        right_layout.addLayout(nav)
        self._set_step(self.steps_list.currentRow())

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([340, 1580])

    def _build_step_1(self) -> None:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(6, 6, 6, 6)
        page_layout.setSpacing(8)

        src_box = QGroupBox("Пресет и имя образца")
        src_form = QFormLayout(src_box)
        self.preset_combo = QComboBox()
        load_btn = QPushButton("Загрузить пресет")
        load_btn.clicked.connect(self._load_selected_preset)
        row = QHBoxLayout()
        row.addWidget(self.preset_combo, 1)
        row.addWidget(load_btn)
        row_wrap = QWidget()
        row_wrap.setLayout(row)
        self.sample_name_edit = QLineEdit(self.state.sample_name)
        self.sample_name_edit.textChanged.connect(self._on_state_changed)
        self.composition_mode_combo = QComboBox()
        self.composition_mode_combo.addItem("По элементам", "elements")
        self.composition_mode_combo.addItem("По фазам (Fe-C)", "phase_fe_c")
        self.composition_mode_combo.currentIndexChanged.connect(
            self._on_composition_mode_changed
        )
        src_form.addRow("Пресет", row_wrap)
        src_form.addRow("Имя образца", self.sample_name_edit)
        src_form.addRow("Режим состава", self.composition_mode_combo)
        page_layout.addWidget(src_box)

        self.comp_box = QGroupBox("Состав сплава, мас.%")
        comp_layout = QVBoxLayout(self.comp_box)
        self.comp_table = QTableWidget(0, 2)
        self.comp_table.setHorizontalHeaderLabels(["Элемент", "мас.%"])
        self.comp_table.verticalHeader().setVisible(False)
        self.comp_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Interactive
        )
        self.comp_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.comp_table.horizontalHeader().setMinimumSectionSize(120)
        self.comp_table.setMinimumHeight(280)
        self.comp_table.itemChanged.connect(
            lambda _item: self._update_step1_validation()
        )
        comp_layout.addWidget(self.comp_table)

        btn_grid = QGridLayout()
        add_btn = QPushButton("Добавить")
        rem_btn = QPushButton("Удалить")
        norm_btn = QPushButton("Нормализовать до 100%")
        imp_btn = QPushButton("Импорт JSON")
        exp_btn = QPushButton("Экспорт JSON")
        add_btn.clicked.connect(self._comp_add)
        rem_btn.clicked.connect(self._comp_remove)
        norm_btn.clicked.connect(self._comp_normalize)
        imp_btn.clicked.connect(self._comp_import)
        exp_btn.clicked.connect(self._comp_export)
        btn_grid.addWidget(add_btn, 0, 0)
        btn_grid.addWidget(rem_btn, 0, 1)
        btn_grid.addWidget(norm_btn, 0, 2)
        btn_grid.addWidget(imp_btn, 1, 0)
        btn_grid.addWidget(exp_btn, 1, 1)
        comp_layout.addLayout(btn_grid)

        self.comp_sum_label = QLabel("Сумма: 0.000 мас.%")
        self.comp_validation_hint = QLabel("")
        self.comp_validation_hint.setWordWrap(True)
        comp_layout.addWidget(self.comp_sum_label)
        comp_layout.addWidget(self.comp_validation_hint)
        page_layout.addWidget(self.comp_box)

        self.phase_box = QGroupBox("Фазовый состав Fe-C, %")
        phase_layout = QVBoxLayout(self.phase_box)
        self.phase_table = QTableWidget(0, 2)
        self.phase_table.setHorizontalHeaderLabels(self.PHASE_COLUMNS)
        self.phase_table.verticalHeader().setVisible(False)
        self.phase_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.phase_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.phase_table.setMinimumHeight(220)
        self.phase_table.itemChanged.connect(
            lambda _item: self._update_step1_validation()
        )
        phase_layout.addWidget(self.phase_table)
        phase_btns = QHBoxLayout()
        phase_norm_btn = QPushButton("Нормализовать фазы до 100%")
        phase_norm_btn.clicked.connect(self._phase_normalize)
        phase_btns.addWidget(phase_norm_btn)
        phase_layout.addLayout(phase_btns)
        self.phase_sum_label = QLabel("Сумма фаз: 0.000 %")
        self.phase_hint = QLabel(
            "Фазовый режим используется как учебная аппроксимация для Fe-C."
        )
        self.phase_hint.setWordWrap(True)
        phase_layout.addWidget(self.phase_sum_label)
        phase_layout.addWidget(self.phase_hint)
        page_layout.addWidget(self.phase_box)
        page_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.stack.addWidget(scroll)

    def _build_step_2(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        state_box = QGroupBox("Режим и базовое состояние")
        state_form = QFormLayout(state_box)
        self.route_policy_combo = QComboBox()
        self.route_policy_combo.addItem("Одиночное состояние", "single_state")
        self.route_policy_combo.addItem("Маршрут обработки", "route_driven")
        self.route_policy_combo.currentIndexChanged.connect(
            self._on_route_policy_changed
        )
        self.route_name_edit = QLineEdit(self.state.route_name)
        self.route_name_edit.textChanged.connect(self._update_step2_validation)
        self.preview_step_spin = QSpinBox()
        self.preview_step_spin.setRange(-1, 999)
        self.preview_step_spin.setValue(-1)
        self.preview_step_spin.valueChanged.connect(self._update_step2_validation)
        self.step_series_check = QCheckBox("Экспортировать пошаговую серию")
        self.step_series_check.setChecked(True)
        self.step_series_check.toggled.connect(self._update_step2_validation)

        self.proc_temp = QDoubleSpinBox()
        self.proc_temp.setRange(-273.15, 2500.0)
        self.proc_temp.setValue(20.0)
        self.proc_temp.valueChanged.connect(self._update_step2_validation)
        self.proc_cooling = QComboBox()
        for code, label in self.COOLING_MODE_OPTIONS:
            self.proc_cooling.addItem(label, code)
        self.proc_cooling.setCurrentIndex(self.proc_cooling.findData("auto"))
        self.proc_cooling.currentIndexChanged.connect(self._update_step2_validation)
        self.proc_deform = QDoubleSpinBox()
        self.proc_deform.setRange(0.0, 95.0)
        self.proc_deform.setValue(0.0)
        self.proc_deform.valueChanged.connect(self._update_step2_validation)
        self.proc_aging_h = QDoubleSpinBox()
        self.proc_aging_h.setRange(0.0, 500.0)
        self.proc_aging_h.setValue(0.0)
        self.proc_aging_h.valueChanged.connect(self._update_step2_validation)
        self.proc_aging_t = QDoubleSpinBox()
        self.proc_aging_t.setRange(-273.15, 1000.0)
        self.proc_aging_t.setValue(20.0)
        self.proc_aging_t.valueChanged.connect(self._update_step2_validation)
        self.proc_pressure = QDoubleSpinBox()
        self.proc_pressure.setRange(0.0, 5000.0)
        self.proc_pressure.setValue(0.0)
        self.proc_pressure.valueChanged.connect(self._update_step2_validation)
        self.proc_note = QLineEdit("")
        self.proc_note.textChanged.connect(self._update_step2_validation)

        state_form.addRow("Режим", self.route_policy_combo)
        state_form.addRow("Имя маршрута", self.route_name_edit)
        state_form.addRow("Шаг предпросмотра (-1 = финал)", self.preview_step_spin)
        state_form.addRow(self.step_series_check)
        state_form.addRow("Температура, C", self.proc_temp)
        state_form.addRow("Охлаждение", self.proc_cooling)
        state_form.addRow("Деформация, %", self.proc_deform)
        state_form.addRow("Старение, ч", self.proc_aging_h)
        state_form.addRow("Температура старения, C", self.proc_aging_t)
        state_form.addRow("Давление, МПа", self.proc_pressure)
        state_form.addRow("Примечание", self.proc_note)
        layout.addWidget(state_box)

        route_box = QGroupBox("Маршрут операций")
        route_layout = QVBoxLayout(route_box)
        self.route_table = QTableWidget(0, len(self.ROUTE_COLUMNS))
        self.route_table.setHorizontalHeaderLabels(self.ROUTE_COLUMNS)
        self.route_table.verticalHeader().setVisible(False)
        self.route_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.route_table.horizontalHeader().setSectionResizeMode(
            len(self.ROUTE_COLUMNS) - 1, QHeaderView.ResizeMode.Stretch
        )
        self.route_table.horizontalHeader().setMinimumSectionSize(105)
        self.route_table.setColumnWidth(0, 320)
        self.route_table.setColumnWidth(1, 95)
        self.route_table.setColumnWidth(2, 95)
        self.route_table.setColumnWidth(3, 170)
        self.route_table.setColumnWidth(4, 90)
        self.route_table.setColumnWidth(5, 105)
        self.route_table.setColumnWidth(6, 110)
        self.route_table.setWordWrap(False)
        self.route_table.verticalHeader().setDefaultSectionSize(30)
        self.route_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.route_table.setMinimumHeight(280)
        self.route_table.itemChanged.connect(
            lambda _item: self._update_step2_validation()
        )
        route_layout.addWidget(self.route_table)

        route_btn1 = QGridLayout()
        add_btn = QPushButton("Добавить")
        rem_btn = QPushButton("Удалить")
        up_btn = QPushButton("Вверх")
        down_btn = QPushButton("Вниз")
        add_btn.clicked.connect(self._route_add)
        rem_btn.clicked.connect(self._route_remove)
        up_btn.clicked.connect(self._route_up)
        down_btn.clicked.connect(self._route_down)
        route_btn1.addWidget(add_btn, 0, 0)
        route_btn1.addWidget(rem_btn, 0, 1)
        route_btn1.addWidget(up_btn, 0, 2)
        route_btn1.addWidget(down_btn, 0, 3)
        route_layout.addLayout(route_btn1)

        route_btn2 = QGridLayout()
        tmpl_btn = QPushButton("Загрузить шаблон")
        imp_btn = QPushButton("Импорт маршрута")
        exp_btn = QPushButton("Экспорт маршрута")
        tmpl_btn.clicked.connect(self._route_load_template)
        imp_btn.clicked.connect(self._route_import)
        exp_btn.clicked.connect(self._route_export)
        route_btn2.addWidget(tmpl_btn, 0, 0)
        route_btn2.addWidget(imp_btn, 0, 1)
        route_btn2.addWidget(exp_btn, 0, 2)
        route_layout.addLayout(route_btn2)
        layout.addWidget(route_box)

        val_box = QGroupBox("Проверка маршрута")
        val_layout = QVBoxLayout(val_box)
        self.route_validation_text = QPlainTextEdit()
        self.route_validation_text.setReadOnly(True)
        self.route_validation_text.setMinimumHeight(180)
        val_layout.addWidget(self.route_validation_text)
        layout.addWidget(val_box)

        curve_box = QGroupBox("Кривая охлаждения и фазовый переход")
        curve_layout = QVBoxLayout(curve_box)
        curve_top = QGridLayout()
        self.curve_enable_check = QCheckBox("Использовать кривую охлаждения")
        self.curve_enable_check.setChecked(False)
        self.curve_enable_check.toggled.connect(self._update_step2_validation)
        self.curve_mode_combo = QComboBox()
        self.curve_mode_combo.addItem("Поградусно (интерполяция)", "per_degree")
        self.curve_mode_combo.addItem("Только опорные точки", "points")
        self.curve_mode_combo.currentIndexChanged.connect(self._update_step2_validation)
        self.curve_degree_step_spin = QDoubleSpinBox()
        self.curve_degree_step_spin.setRange(0.1, 25.0)
        self.curve_degree_step_spin.setSingleStep(0.1)
        self.curve_degree_step_spin.setValue(1.0)
        self.curve_degree_step_spin.valueChanged.connect(self._update_step2_validation)
        self.curve_max_points_spin = QSpinBox()
        self.curve_max_points_spin.setRange(5, 2000)
        self.curve_max_points_spin.setValue(220)
        self.curve_max_points_spin.valueChanged.connect(self._update_step2_validation)

        curve_top.addWidget(self.curve_enable_check, 0, 0, 1, 2)
        curve_top.addWidget(QLabel("Режим дискретизации"), 1, 0)
        curve_top.addWidget(self.curve_mode_combo, 1, 1)
        curve_top.addWidget(QLabel("Шаг по температуре, C"), 2, 0)
        curve_top.addWidget(self.curve_degree_step_spin, 2, 1)
        curve_top.addWidget(QLabel("Макс. кадров"), 2, 2)
        curve_top.addWidget(self.curve_max_points_spin, 2, 3)
        curve_layout.addLayout(curve_top)

        self.curve_table = QTableWidget(0, len(self.CURVE_COLUMNS))
        self.curve_table.setHorizontalHeaderLabels(self.CURVE_COLUMNS)
        self.curve_table.verticalHeader().setVisible(False)
        self.curve_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.curve_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.curve_table.setMinimumHeight(170)
        self.curve_table.itemChanged.connect(
            lambda _item: self._update_step2_validation()
        )
        curve_layout.addWidget(self.curve_table)

        curve_btns = QGridLayout()
        curve_add_btn = QPushButton("Добавить точку")
        curve_add_btn.clicked.connect(self._curve_add_point)
        curve_remove_btn = QPushButton("Удалить точку")
        curve_remove_btn.clicked.connect(self._curve_remove_point)
        curve_seed_btn = QPushButton("Из режима обработки")
        curve_seed_btn.clicked.connect(self._curve_seed_from_processing)
        curve_sort_btn = QPushButton("Сортировать по времени")
        curve_sort_btn.clicked.connect(self._curve_sort_points)
        curve_btns.addWidget(curve_add_btn, 0, 0)
        curve_btns.addWidget(curve_remove_btn, 0, 1)
        curve_btns.addWidget(curve_seed_btn, 0, 2)
        curve_btns.addWidget(curve_sort_btn, 0, 3)
        curve_layout.addLayout(curve_btns)
        layout.addWidget(curve_box)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.stack.addWidget(scroll)

    def _build_step_3(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        basic_box = QGroupBox("Базовые параметры генерации")
        basic_form = QFormLayout(basic_box)
        self.generator_combo = QComboBox()
        for generator_code in self.pipeline.generator_registry.available_generators():
            generator_label = self.GENERATOR_LABELS.get(generator_code, generator_code)
            self.generator_combo.addItem(generator_label, generator_code)
        default_idx = self.generator_combo.findData("auto")
        if default_idx >= 0:
            self.generator_combo.setCurrentIndex(default_idx)
        else:
            cast_idx = self.generator_combo.findData("dendritic_cast")
            if cast_idx >= 0:
                self.generator_combo.setCurrentIndex(cast_idx)
        self.generator_combo.currentIndexChanged.connect(self._on_state_changed)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        self.seed_spin.setValue(4200)
        self.seed_spin.valueChanged.connect(self._on_state_changed)
        self.res_h_spin = QSpinBox()
        self.res_h_spin.setRange(128, 4096)
        self.res_h_spin.setValue(1024)
        self.res_h_spin.valueChanged.connect(self._on_basic_live_changed)
        self.res_w_spin = QSpinBox()
        self.res_w_spin.setRange(128, 4096)
        self.res_w_spin.setValue(1024)
        self.res_w_spin.valueChanged.connect(self._on_basic_live_changed)
        self.mag_combo = QComboBox()
        self.mag_combo.addItems(["100", "200", "400", "600"])
        self.mag_combo.setCurrentText("200")
        self.mag_combo.currentTextChanged.connect(self._on_basic_live_changed)
        self.focus_spin = QDoubleSpinBox()
        self.focus_spin.setRange(0.0, 1.0)
        self.focus_spin.setSingleStep(0.01)
        self.focus_spin.setValue(0.95)
        self.focus_spin.valueChanged.connect(self._on_basic_live_changed)
        self.brightness_spin = QDoubleSpinBox()
        self.brightness_spin.setRange(0.5, 1.8)
        self.brightness_spin.setValue(1.0)
        self.brightness_spin.valueChanged.connect(self._on_basic_live_changed)
        self.contrast_spin = QDoubleSpinBox()
        self.contrast_spin.setRange(0.5, 2.2)
        self.contrast_spin.setValue(1.1)
        self.contrast_spin.valueChanged.connect(self._on_basic_live_changed)
        basic_form.addRow("Генератор", self.generator_combo)
        basic_form.addRow("Сид (seed)", self.seed_spin)
        basic_form.addRow("Разрешение H", self.res_h_spin)
        basic_form.addRow("Разрешение W", self.res_w_spin)
        basic_form.addRow("Увеличение", self.mag_combo)
        basic_form.addRow("Фокус", self.focus_spin)
        basic_form.addRow("Яркость", self.brightness_spin)
        basic_form.addRow("Контраст", self.contrast_spin)
        layout.addWidget(basic_box)

        self.expert_box = QGroupBox("Экспертные параметры")
        self.expert_box.setCheckable(True)
        self.expert_box.setChecked(False)
        exp_form = QFormLayout(self.expert_box)
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 30.0)
        self.noise_spin.setValue(2.2)
        self.noise_spin.valueChanged.connect(self._on_heavy_changed)
        self.vignette_spin = QDoubleSpinBox()
        self.vignette_spin.setRange(0.0, 0.8)
        self.vignette_spin.setValue(0.12)
        self.vignette_spin.valueChanged.connect(self._on_heavy_changed)
        self.uneven_spin = QDoubleSpinBox()
        self.uneven_spin.setRange(0.0, 0.6)
        self.uneven_spin.setValue(0.08)
        self.uneven_spin.valueChanged.connect(self._on_heavy_changed)
        self.dust_check = QCheckBox("Пыль")
        self.dust_check.toggled.connect(self._on_heavy_changed)
        self.scratch_check = QCheckBox("Царапины")
        self.scratch_check.toggled.connect(self._on_heavy_changed)
        self.etch_spin = QDoubleSpinBox()
        self.etch_spin.setRange(0.0, 1.0)
        self.etch_spin.setSingleStep(0.01)
        self.etch_spin.setValue(0.0)
        self.etch_spin.valueChanged.connect(self._on_heavy_changed)
        self.generator_params_edit = QPlainTextEdit("{}")
        self.generator_params_edit.setFixedHeight(130)
        self.generator_params_edit.textChanged.connect(self._on_state_changed)
        exp_form.addRow("Шум", self.noise_spin)
        exp_form.addRow("Виньетирование", self.vignette_spin)
        exp_form.addRow("Неравномерность света", self.uneven_spin)
        exp_form.addRow(self.dust_check)
        exp_form.addRow(self.scratch_check)
        exp_form.addRow("Неравномерное травление", self.etch_spin)
        exp_form.addRow("Параметры генератора (JSON)", self.generator_params_edit)
        apply_heavy_btn = QPushButton("Применить тяжелые эффекты")
        apply_heavy_btn.clicked.connect(self._apply_heavy_and_preview)
        exp_form.addRow(apply_heavy_btn)
        layout.addWidget(self.expert_box)

        val_box = QGroupBox("Валидация")
        val_layout = QVBoxLayout(val_box)
        self.auto_norm_check = QCheckBox("Автонормализация состава")
        self.auto_norm_check.setChecked(True)
        self.auto_norm_check.toggled.connect(self._on_state_changed)
        self.strict_check = QCheckBox("Строгая валидация (hard-stop)")
        self.strict_check.setChecked(True)
        self.strict_check.toggled.connect(self._on_state_changed)
        self.validation_text = QPlainTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setMinimumHeight(170)
        actions = QHBoxLayout()
        validate_btn = QPushButton("Проверить")
        validate_btn.clicked.connect(self._validate_clicked)
        autofix_btn = QPushButton("Автоисправление")
        autofix_btn.clicked.connect(self._auto_fix)
        actions.addWidget(validate_btn)
        actions.addWidget(autofix_btn)
        val_layout.addWidget(self.auto_norm_check)
        val_layout.addWidget(self.strict_check)
        val_layout.addLayout(actions)
        val_layout.addWidget(self.validation_text)
        layout.addWidget(val_box)

        calphad_box = QGroupBox("CALPHAD")
        calphad_layout = QVBoxLayout(calphad_box)
        calphad_form = QFormLayout()
        self.calphad_backend_label = QLabel("PyCalphad")
        self.calphad_backend_label.setStyleSheet("font-weight: 600;")
        self.calphad_strict_check = QCheckBox("Строгий режим CALPHAD-only")
        self.calphad_strict_check.setChecked(True)
        self.calphad_strict_check.setEnabled(False)
        self.calphad_profile_edit = QLineEdit(str(self.pipeline.calphad_profile_path))
        self.calphad_profile_edit.textChanged.connect(self._on_state_changed)
        self.calphad_pressure_spin = QDoubleSpinBox()
        self.calphad_pressure_spin.setRange(10_000.0, 5_000_000.0)
        self.calphad_pressure_spin.setDecimals(1)
        self.calphad_pressure_spin.setSingleStep(10_000.0)
        self.calphad_pressure_spin.setValue(float(self.state.thermo.pressure_pa))
        self.calphad_pressure_spin.valueChanged.connect(self._on_state_changed)
        self.calphad_tgrid_spin = QDoubleSpinBox()
        self.calphad_tgrid_spin.setRange(0.5, 100.0)
        self.calphad_tgrid_spin.setValue(float(self.state.thermo.t_grid_step_c))
        self.calphad_tgrid_spin.valueChanged.connect(self._on_state_changed)
        self.calphad_scheil_dt_spin = QDoubleSpinBox()
        self.calphad_scheil_dt_spin.setRange(0.1, 100.0)
        self.calphad_scheil_dt_spin.setValue(float(self.state.thermo.scheil_dt_c))
        self.calphad_scheil_dt_spin.valueChanged.connect(self._on_state_changed)
        self.calphad_scheil_check = QCheckBox("Scheil включен")
        self.calphad_scheil_check.setChecked(bool(self.state.thermo.scheil_enabled))
        self.calphad_scheil_check.toggled.connect(self._on_state_changed)
        self.calphad_kinetics_check = QCheckBox("Kinetics (JMAK+LSW) включен")
        self.calphad_kinetics_check.setChecked(bool(self.state.thermo.kinetics_enabled))
        self.calphad_kinetics_check.toggled.connect(self._on_state_changed)
        self.calphad_db_table = QTableWidget(0, 2)
        self.calphad_db_table.setHorizontalHeaderLabels(["Система", "Путь TDB"])
        self.calphad_db_table.verticalHeader().setVisible(False)
        self.calphad_db_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.calphad_db_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.calphad_db_table.setMinimumHeight(145)
        self.calphad_db_table.itemChanged.connect(
            lambda _item: self._on_state_changed()
        )

        calphad_form.addRow("Backend", self.calphad_backend_label)
        calphad_form.addRow(self.calphad_strict_check)
        calphad_form.addRow("Профиль CALPHAD", self.calphad_profile_edit)
        calphad_form.addRow("Давление, Па", self.calphad_pressure_spin)
        calphad_form.addRow("Шаг T-сетки, °C", self.calphad_tgrid_spin)
        calphad_form.addRow("Шаг Scheil dT, °C", self.calphad_scheil_dt_spin)
        calphad_form.addRow(self.calphad_scheil_check)
        calphad_form.addRow(self.calphad_kinetics_check)
        calphad_layout.addLayout(calphad_form)
        calphad_layout.addWidget(self.calphad_db_table)

        calphad_actions = QHBoxLayout()
        self.calphad_check_btn = QPushButton("Проверить базы")
        self.calphad_check_btn.clicked.connect(self._check_calphad_setup_clicked)
        calphad_actions.addWidget(self.calphad_check_btn)
        calphad_actions.addStretch(1)
        calphad_layout.addLayout(calphad_actions)

        self.calphad_status_text = QPlainTextEdit()
        self.calphad_status_text.setReadOnly(True)
        self.calphad_status_text.setMinimumHeight(90)
        calphad_layout.addWidget(self.calphad_status_text)
        layout.addWidget(calphad_box)

        self._fill_calphad_db_table(self.state.thermo.db_overrides)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.stack.addWidget(scroll)

    def _build_step_4(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        top_box = QGroupBox("Сравнение и экспорт")
        top_grid = QGridLayout(top_box)
        self.compare_mode_combo = QComboBox()
        self.compare_mode_combo.addItem("Один кадр", "single")
        self.compare_mode_combo.addItem("До/После", "before_after")
        self.compare_mode_combo.addItem("Пошагово", "step_by_step")
        self.compare_mode_combo.addItem(
            "Фазовый переход (кривая)", "phase_transition_curve"
        )
        self.compare_mode_combo.addItem("Карта отличий", "diff_map")
        self.compare_mode_combo.currentIndexChanged.connect(self._update_compare_mode)
        self.diagram_system_combo = QComboBox()
        self.diagram_system_combo.addItem("Авто", "")
        for system_name in available_diagram_systems():
            self.diagram_system_combo.addItem(
                self.SYSTEM_LABELS.get(system_name, system_name), system_name
            )
        self.layer_axes = QCheckBox("Оси")
        self.layer_axes.setChecked(True)
        self.layer_grid = QCheckBox("Сетка")
        self.layer_grid.setChecked(True)
        self.layer_lines = QCheckBox("Линии")
        self.layer_lines.setChecked(True)
        self.layer_inv = QCheckBox("Инвариантные точки")
        self.layer_inv.setChecked(True)
        self.layer_regions = QCheckBox("Фазовые поля")
        self.layer_regions.setChecked(True)
        self.layer_marker = QCheckBox("Маркер текущей точки")
        self.layer_marker.setChecked(True)
        preview_btn = QPushButton("Обновить предпросмотр")
        preview_btn.clicked.connect(
            lambda: self._run_preview(use_heavy=False, silent=False)
        )
        apply_btn = QPushButton("Применить тяжелые эффекты")
        apply_btn.clicked.connect(self._apply_heavy_and_preview)
        save_img_btn = QPushButton("Сохранить изображение")
        save_img_btn.clicked.connect(self._save_current_image)
        save_meta_btn = QPushButton("Сохранить метаданные")
        save_meta_btn.clicked.connect(self._save_current_metadata)
        save_diag_btn = QPushButton("Сохранить диаграмму")
        save_diag_btn.clicked.connect(self._save_current_diagram)
        add_batch_btn = QPushButton("Добавить в пакет")
        add_batch_btn.clicked.connect(self._add_to_batch)
        run_batch_btn = QPushButton("Запустить пакет")
        run_batch_btn.clicked.connect(self._run_batch)

        top_grid.addWidget(QLabel("Режим сравнения"), 0, 0)
        top_grid.addWidget(self.compare_mode_combo, 0, 1)
        top_grid.addWidget(QLabel("Система диаграммы"), 0, 2)
        top_grid.addWidget(self.diagram_system_combo, 0, 3)
        top_grid.addWidget(self.layer_axes, 1, 0)
        top_grid.addWidget(self.layer_grid, 1, 1)
        top_grid.addWidget(self.layer_lines, 1, 2)
        top_grid.addWidget(self.layer_inv, 1, 3)
        top_grid.addWidget(self.layer_regions, 2, 0)
        top_grid.addWidget(self.layer_marker, 2, 1)
        top_grid.addWidget(preview_btn, 3, 0)
        top_grid.addWidget(apply_btn, 3, 1)
        top_grid.addWidget(add_batch_btn, 3, 2)
        top_grid.addWidget(run_batch_btn, 3, 3)
        top_grid.addWidget(save_img_btn, 4, 0)
        top_grid.addWidget(save_meta_btn, 4, 1)
        top_grid.addWidget(save_diag_btn, 4, 2)
        for col in range(4):
            top_grid.setColumnStretch(col, 1)
        layout.addWidget(top_box)

        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        preview_box = QGroupBox("Предпросмотр")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_scene = QGraphicsScene(self)
        self.preview_view = ZoomView()
        self.preview_view.setScene(self.preview_scene)
        self.preview_view.setMinimumHeight(520)
        preview_layout.addWidget(self.preview_view)
        left_layout.addWidget(preview_box)
        self.preview_info = QPlainTextEdit()
        self.preview_info.setReadOnly(True)
        self.preview_info.setMinimumHeight(160)
        left_layout.addWidget(self.preview_info)

        diagram_box = QGroupBox("Диаграмма состояния")
        diagram_layout = QVBoxLayout(diagram_box)
        self.diagram_scene = QGraphicsScene(self)
        self.diagram_view = ZoomView(
            allow_drag=False, on_resized=self._on_diagram_view_resized
        )
        self.diagram_view.setScene(self.diagram_scene)
        self.diagram_view.setMinimumHeight(340)
        diagram_layout.addWidget(self.diagram_view)

        batch_box = QGroupBox("Пакетная очередь")
        batch_layout = QVBoxLayout(batch_box)
        self.batch_table = QTableWidget(0, 6)
        self.batch_table.setHorizontalHeaderLabels(
            ["#", "Имя", "Генератор", "Режим", "Сид", "Разрешение"]
        )
        self.batch_table.verticalHeader().setVisible(False)
        self.batch_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.batch_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.batch_table.horizontalHeader().setMinimumSectionSize(100)
        self.batch_table.setMinimumHeight(220)
        self.batch_dir_edit = QLineEdit(self.state.batch_output_dir)
        browse_batch = QPushButton("Папка")
        browse_batch.clicked.connect(self._browse_batch_dir)
        self.batch_prefix_edit = QLineEdit(self.state.batch_prefix)
        row = QGridLayout()
        row.addWidget(QLabel("Папка"), 0, 0)
        row.addWidget(self.batch_dir_edit, 0, 1)
        row.addWidget(browse_batch, 0, 2)
        row.addWidget(QLabel("Префикс"), 1, 0)
        row.addWidget(self.batch_prefix_edit, 1, 1)
        self.batch_log = QPlainTextEdit()
        self.batch_log.setReadOnly(True)
        self.batch_log.setMinimumHeight(120)
        batch_layout.addWidget(self.batch_table)
        batch_layout.addLayout(row)
        batch_layout.addWidget(self.batch_log)

        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.addWidget(diagram_box)
        right_split.addWidget(batch_box)
        right_split.setStretchFactor(0, 2)
        right_split.setStretchFactor(1, 1)
        right_split.setSizes([650, 350])

        split.addWidget(left)
        split.addWidget(right_split)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 1)
        split.setSizes([1280, 640])
        layout.addWidget(split, stretch=1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.stack.addWidget(scroll)

    def _style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #121821; color: #e9eff7; font-size: 13px; }
            QGroupBox {
                border: 1px solid #35475d; border-radius: 8px; margin-top: 8px;
                font-weight: 600; padding-top: 8px;
            }
            QGroupBox::title { left: 12px; color: #dce6f2; }
            QListWidget, QTableWidget, QLineEdit, QComboBox, QPlainTextEdit, QSpinBox, QDoubleSpinBox {
                background: #1f2a38; border: 1px solid #475d77;
            }
            QPushButton {
                background: #3a5774; border: 1px solid #4f6f8f;
                border-radius: 6px; padding: 6px 10px; min-height: 28px;
            }
            QPushButton:hover { background: #46698d; }
            QGraphicsView { background: #090d15; border: 1px solid #3f5873; border-radius: 6px; }
            QHeaderView::section { background: #243246; color: #d7e1ee; border: 1px solid #3b516a; padding: 4px; }
            QScrollArea { border: none; }
            """
        )

    def _set_step(self, index: int) -> None:
        if not hasattr(self, "stack"):
            return
        if index < 0:
            index = 0
        if index >= self.stack.count():
            index = self.stack.count() - 1
        self.stack.setCurrentIndex(index)
        if hasattr(self, "steps_list") and self.steps_list.currentRow() != index:
            self.steps_list.blockSignals(True)
            self.steps_list.setCurrentRow(index)
            self.steps_list.blockSignals(False)
        if hasattr(self, "prev_btn"):
            self.prev_btn.setEnabled(index > 0)
        if hasattr(self, "next_btn"):
            self.next_btn.setEnabled(index < self.stack.count() - 1)
        hints = {
            0: "Шаг 1: задайте состав и проверьте сумму.",
            1: "Шаг 2: задайте маршрут или одиночное состояние.",
            2: "Шаг 3: настройте генератор и параметры.",
            3: "Шаг 4: сравнение, диаграмма, экспорт, пакет.",
        }
        if hasattr(self, "step_help"):
            self.step_help.setText(hints.get(index, ""))

    def _go_prev(self) -> None:
        self._set_step(self.stack.currentIndex() - 1)

    def _go_next(self) -> None:
        self._validate_all_steps()
        if self.step_status[self.stack.currentIndex()] == "error":
            QMessageBox.warning(
                self, "Переход к шагу", "Исправьте блокирующие ошибки перед переходом."
            )
            return
        self._set_step(self.stack.currentIndex() + 1)

    def _set_step_status(self, index: int, status: str, message: str = "") -> None:
        self.step_status[index] = status
        prefix = {"ok": "OK", "warn": "!", "error": "X"}.get(status, "!")
        self.steps_list.item(index).setText(
            f"{prefix} {index + 1}. {self.STEP_TITLES[index]}"
        )
        self.steps_list.item(index).setToolTip(message or self.STEP_TITLES[index])

    def _validate_all_steps(self) -> None:
        self._update_step1_validation()
        self._update_step2_validation()
        self._update_step3_validation()
        self._update_step4_status()

    def _item_text(self, row: int, col: int) -> str:
        widget = self.route_table.cellWidget(row, col)
        if isinstance(widget, QComboBox):
            return str(widget.currentData() or widget.currentText()).strip()
        item = self.route_table.item(row, col)
        if item is None:
            return ""
        return item.text().strip()

    def _composition_mode(self) -> str:
        return str(self.composition_mode_combo.currentData() or "elements")

    def _selected_generator(self) -> str:
        return str(
            self.generator_combo.currentData() or self.generator_combo.currentText()
        ).strip()

    def _phase_table_values(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in range(self.phase_table.rowCount()):
            phase_item = self.phase_table.item(row, 0)
            pct_item = self.phase_table.item(row, 1)
            if phase_item is None:
                continue
            phase = phase_item.text().strip()
            if not phase:
                continue
            pct = 0.0 if pct_item is None else _safe_float(pct_item.text(), 0.0)
            out[phase] = max(0.0, float(pct))
        return out

    def _phase_to_composition(
        self, phases_pct: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, Any]]:
        total = float(sum(phases_pct.values()))
        if total <= 0.0:
            return {"Fe": 99.8, "C": 0.2}, {
                "dominant_phase": "Ferrite",
                "normalized_phases": {"Ferrite": 100.0},
            }

        normalized = {k: (100.0 * v / total) for k, v in phases_pct.items()}
        carbon_model = {
            "Ferrite": 0.02,
            "Pearlite": 0.80,
            "Cementite": 6.67,
            "Austenite": 0.80,
            "Martensite": 1.00,
            "Bainite": 0.50,
        }
        carbon = 0.0
        for phase_name, pct in normalized.items():
            carbon += float(carbon_model.get(phase_name, 0.1)) * float(pct / 100.0)
        carbon = float(np.clip(carbon, 0.02, 6.67))
        composition = {"Fe": float(max(0.0, 100.0 - carbon)), "C": carbon}
        dominant_phase = max(normalized, key=normalized.get)
        return composition, {
            "dominant_phase": dominant_phase,
            "normalized_phases": normalized,
        }

    def _current_processing(self) -> ProcessingState:
        pressure = float(self.proc_pressure.value())
        return ProcessingState(
            temperature_c=float(self.proc_temp.value()),
            cooling_mode=str(self.proc_cooling.currentData() or "equilibrium"),
            deformation_pct=float(self.proc_deform.value()),
            aging_hours=float(self.proc_aging_h.value()),
            aging_temperature_c=float(self.proc_aging_t.value()),
            pressure_mpa=pressure if pressure > 0.0 else None,
            note=self.proc_note.text().strip(),
        )

    def _element_table_composition(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in range(self.comp_table.rowCount()):
            sym_item = self.comp_table.item(row, 0)
            wt_item = self.comp_table.item(row, 1)
            if sym_item is None:
                continue
            symbol = sym_item.text().strip()
            if not symbol:
                continue
            wt = 0.0 if wt_item is None else _safe_float(wt_item.text(), 0.0)
            out[symbol] = out.get(symbol, 0.0) + wt
        return out

    def _table_composition(self) -> dict[str, float]:
        if self._composition_mode() == "phase_fe_c":
            comp, _ = self._phase_to_composition(self._phase_table_values())
            return comp
        return self._element_table_composition()

    def _route_from_table(self) -> list[ProcessingOperation]:
        ops: list[ProcessingOperation] = []
        for row in range(self.route_table.rowCount()):
            method = self._item_text(row, 0)
            if not method:
                continue
            ops.append(
                ProcessingOperation(
                    method=method,
                    temperature_c=_safe_float(self._item_text(row, 1), 20.0),
                    duration_min=_safe_float(self._item_text(row, 2), 0.0),
                    cooling_mode=self._item_text(row, 3) or "equilibrium",
                    deformation_pct=_safe_float(self._item_text(row, 4), 0.0),
                    aging_hours=_safe_float(self._item_text(row, 5), 0.0),
                    aging_temperature_c=_safe_float(self._item_text(row, 6), 20.0),
                    note=self._item_text(row, 7),
                )
            )
        return ops

    def _curve_points_from_table(self) -> list[dict[str, float]]:
        points: list[dict[str, float]] = []
        for row in range(self.curve_table.rowCount()):
            time_item = self.curve_table.item(row, 0)
            temp_item = self.curve_table.item(row, 1)
            if time_item is None and temp_item is None:
                continue
            time_val = 0.0 if time_item is None else _safe_float(time_item.text(), 0.0)
            temp_val = (
                float(self.proc_temp.value())
                if temp_item is None
                else _safe_float(temp_item.text(), float(self.proc_temp.value()))
            )
            points.append(
                {
                    "time_min": float(max(0.0, time_val)),
                    "temperature_c": float(temp_val),
                }
            )
        return normalize_cooling_curve_points(
            points, fallback_temperature_c=float(self.proc_temp.value())
        )

    def _fill_curve_table(self, points: list[dict[str, float]] | None = None) -> None:
        data = normalize_cooling_curve_points(
            points or [], fallback_temperature_c=float(self.proc_temp.value())
        )
        self.curve_table.blockSignals(True)
        try:
            self.curve_table.setRowCount(0)
            for point in data:
                row = self.curve_table.rowCount()
                self.curve_table.insertRow(row)
                self.curve_table.setItem(
                    row, 0, QTableWidgetItem(f"{float(point['time_min']):.6g}")
                )
                self.curve_table.setItem(
                    row, 1, QTableWidgetItem(f"{float(point['temperature_c']):.6g}")
                )
        finally:
            self.curve_table.blockSignals(False)

    def _curve_add_point(self) -> None:
        points = self._curve_points_from_table()
        if points:
            t = float(points[-1]["time_min"]) + 5.0
            temp = float(points[-1]["temperature_c"])
        else:
            t = 0.0
            temp = float(self.proc_temp.value())
        points.append({"time_min": t, "temperature_c": temp})
        self._fill_curve_table(points)
        self._update_step2_validation()

    def _curve_remove_point(self) -> None:
        rows = sorted(
            {idx.row() for idx in self.curve_table.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self.curve_table.removeRow(row)
        if self.curve_table.rowCount() < 2:
            self._curve_seed_from_processing()
        else:
            self._update_step2_validation()

    def _curve_seed_from_processing(self) -> None:
        temp = float(self.proc_temp.value())
        target = 20.0 if temp >= 20.0 else temp
        points = [
            {"time_min": 0.0, "temperature_c": temp},
            {"time_min": 6.0, "temperature_c": float((temp + target) * 0.55)},
            {"time_min": 14.0, "temperature_c": target},
        ]
        self._fill_curve_table(points)
        self._update_step2_validation()

    def _curve_sort_points(self) -> None:
        points = self._curve_points_from_table()
        points.sort(key=lambda x: x["time_min"])
        self._fill_curve_table(points)
        self._update_step2_validation()

    def _route_policy(self) -> str:
        return str(self.route_policy_combo.currentData() or "single_state")

    def _generator_params(self) -> dict[str, Any]:
        raw = self.generator_params_edit.toPlainText().strip() or "{}"
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Параметры генератора должны быть JSON-объектом")
        return parsed

    def _fill_calphad_db_table(self, overrides: dict[str, str]) -> None:
        rows = ["fe-c", "fe-si", "al-si", "cu-zn", "al-cu-mg"]
        self.calphad_db_table.blockSignals(True)
        try:
            self.calphad_db_table.setRowCount(0)
            for system in rows:
                row = self.calphad_db_table.rowCount()
                self.calphad_db_table.insertRow(row)
                system_item = QTableWidgetItem(system)
                system_item.setFlags(system_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.calphad_db_table.setItem(row, 0, system_item)
                self.calphad_db_table.setItem(
                    row, 1, QTableWidgetItem(str(overrides.get(system, "")))
                )
        finally:
            self.calphad_db_table.blockSignals(False)

    def _calphad_db_overrides_from_table(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for row in range(self.calphad_db_table.rowCount()):
            system_item = self.calphad_db_table.item(row, 0)
            path_item = self.calphad_db_table.item(row, 1)
            system = "" if system_item is None else system_item.text().strip().lower()
            path = "" if path_item is None else path_item.text().strip()
            if system and path:
                out[system] = path
        return out

    def _thermo_config_from_widgets(self) -> ThermoBackendConfig:
        return ThermoBackendConfig(
            backend="calphad_py",
            strict_mode=True,
            db_profile_path=(self.calphad_profile_edit.text().strip() or None),
            db_overrides=self._calphad_db_overrides_from_table(),
            cache_dir=self.state.thermo.cache_dir,
            cache_policy=self.state.thermo.cache_policy,
            equilibrium_model=self.state.thermo.equilibrium_model,
            scheil_enabled=bool(self.calphad_scheil_check.isChecked()),
            kinetics_enabled=bool(self.calphad_kinetics_check.isChecked()),
            pressure_pa=float(self.calphad_pressure_spin.value()),
            t_grid_step_c=float(self.calphad_tgrid_spin.value()),
            scheil_dt_c=float(self.calphad_scheil_dt_spin.value()),
        )

    def _check_calphad_setup_clicked(self) -> None:
        thermo = self._thermo_config_from_widgets()
        try:
            report = self.pipeline.validate_calphad_setup(thermo=thermo)
        except Exception as exc:
            self.calphad_status_text.setPlainText(f"Ошибка проверки CALPHAD: {exc}")
            return

        lines = [
            f"Backend: {report.get('backend')}",
            f"Strict: {report.get('strict_mode')}",
            "",
        ]
        systems = report.get("systems", {})
        for system in ["fe-c", "fe-si", "al-si", "cu-zn", "al-cu-mg"]:
            item = systems.get(system, {}) if isinstance(systems, dict) else {}
            if bool(item.get("ok")):
                lines.append(f"[OK] {system}: {item.get('path')}")
            else:
                lines.append(f"[ERR] {system}: {item.get('error')}")
        lines.append("")
        lines.append(f"Итог: {'готово' if report.get('is_valid') else 'есть ошибки'}")
        self.calphad_status_text.setPlainText("\n".join(lines))

    def _microscope_params(self, use_heavy: bool) -> dict[str, Any]:
        params = {
            "magnification": int(self.mag_combo.currentText()),
            "focus": float(self.focus_spin.value()),
            "brightness": float(self.brightness_spin.value()),
            "contrast": float(self.contrast_spin.value()),
            "output_size": [int(self.res_h_spin.value()), int(self.res_w_spin.value())],
        }
        if use_heavy and self.expert_box.isChecked():
            params.update(
                {
                    "noise_sigma": float(self.noise_spin.value()),
                    "vignette_strength": float(self.vignette_spin.value()),
                    "uneven_strength": float(self.uneven_spin.value()),
                    "add_dust": bool(self.dust_check.isChecked()),
                    "add_scratches": bool(self.scratch_check.isChecked()),
                    "etch_uneven": float(self.etch_spin.value()),
                }
            )
        else:
            params.update(
                {
                    "noise_sigma": 0.0,
                    "vignette_strength": 0.0,
                    "uneven_strength": 0.0,
                    "add_dust": False,
                    "add_scratches": False,
                    "etch_uneven": 0.0,
                }
            )
        return params

    def _compose_request(self, use_heavy: bool) -> GenerationRequestV2:
        route_ops = self._route_from_table()
        route: ProcessRoute | None = None
        if route_ops:
            route = ProcessRoute(
                operations=route_ops,
                route_name=self.route_name_edit.text().strip() or "Маршрут",
                route_notes="",
                step_preview_enabled=bool(self.step_series_check.isChecked()),
            )
        step_idx = int(self.preview_step_spin.value())
        preview_step_index = None if step_idx < 0 else step_idx
        composition = self._table_composition()
        generator_params = self._generator_params()

        if self._composition_mode() == "phase_fe_c":
            phases = self._phase_table_values()
            phase_comp, phase_meta = self._phase_to_composition(phases)
            composition = phase_comp
            stage_map = {
                "Ferrite": "ferrite",
                "Pearlite": "pearlite",
                "Cementite": "pearlite_cementite",
                "Austenite": "austenite",
                "Martensite": "martensite",
                "Bainite": "bainite",
            }
            dominant_phase = str(phase_meta.get("dominant_phase", "Ferrite"))
            generator_params.setdefault(
                "phase_input", phase_meta.get("normalized_phases", phases)
            )
            if self._selected_generator() in {"phase_map", "auto"}:
                generator_params.setdefault("system", "fe-c")
                generator_params.setdefault(
                    "stage", stage_map.get(dominant_phase, "auto")
                )

        if bool(self.curve_enable_check.isChecked()):
            generator_params["cooling_curve_enabled"] = True
            generator_params["cooling_curve_mode"] = str(
                self.curve_mode_combo.currentData() or "per_degree"
            )
            generator_params["cooling_curve_degree_step"] = float(
                self.curve_degree_step_spin.value()
            )
            generator_params["cooling_curve_max_points"] = int(
                self.curve_max_points_spin.value()
            )
            generator_params["cooling_curve"] = self._curve_points_from_table()
        else:
            generator_params.pop("cooling_curve_enabled", None)
            generator_params.pop("cooling_curve_mode", None)
            generator_params.pop("cooling_curve_degree_step", None)
            generator_params.pop("cooling_curve_max_points", None)
            generator_params.pop("cooling_curve", None)

        return GenerationRequestV2(
            mode="direct",
            composition=composition,
            processing=self._current_processing(),
            generator=self._selected_generator(),
            generator_params=generator_params,
            seed=int(self.seed_spin.value()),
            resolution=(int(self.res_h_spin.value()), int(self.res_w_spin.value())),
            microscope_params=self._microscope_params(use_heavy=use_heavy),
            preset_name=self.preset_combo.currentText().strip() or None,
            auto_normalize=bool(self.auto_norm_check.isChecked()),
            strict_validation=bool(self.strict_check.isChecked()),
            thermo=self._thermo_config_from_widgets(),
            process_route=route,
            route_policy=self._route_policy(),
            preview_step_index=preview_step_index,
        )

    def _sync_state_from_widgets(self) -> None:
        self.state.sample_name = self.sample_name_edit.text().strip() or "sample_v2"
        self.state.composition_mode = self._composition_mode()
        self.state.composition = self._table_composition()
        self.state.processing = self._current_processing()
        self.state.route_policy = self._route_policy()
        self.state.route_name = self.route_name_edit.text().strip() or "Маршрут"
        self.state.route_operations = self._route_from_table()
        step_idx = int(self.preview_step_spin.value())
        self.state.step_preview_index = None if step_idx < 0 else step_idx
        self.state.step_series_enabled = bool(self.step_series_check.isChecked())
        self.state.cooling_curve_enabled = bool(self.curve_enable_check.isChecked())
        self.state.cooling_curve_mode = str(
            self.curve_mode_combo.currentData() or "per_degree"
        )
        self.state.cooling_curve_degree_step = float(
            self.curve_degree_step_spin.value()
        )
        self.state.cooling_curve_max_points = int(self.curve_max_points_spin.value())
        self.state.cooling_curve_points = self._curve_points_from_table()
        self.state.generator = self._selected_generator()
        self.state.generator_params_text = (
            self.generator_params_edit.toPlainText().strip() or "{}"
        )
        self.state.seed = int(self.seed_spin.value())
        self.state.resolution = (
            int(self.res_h_spin.value()),
            int(self.res_w_spin.value()),
        )
        self.state.microscope_params = self._microscope_params(use_heavy=True)
        self.state.auto_normalize = bool(self.auto_norm_check.isChecked())
        self.state.strict_validation = bool(self.strict_check.isChecked())
        self.state.thermo = self._thermo_config_from_widgets()
        self.state.compare_mode = normalize_compare_mode(
            self.compare_mode_combo.currentData()
        )
        self.state.batch_output_dir = (
            self.batch_dir_edit.text().strip() or self.state.batch_output_dir
        )
        self.state.batch_prefix = (
            self.batch_prefix_edit.text().strip() or self.state.batch_prefix
        )

    def _sync_widgets_from_state(self) -> None:
        self._live_enabled = False
        try:
            self.sample_name_edit.setText(self.state.sample_name)
            self.composition_mode_combo.setCurrentIndex(
                1 if self.state.composition_mode == "phase_fe_c" else 0
            )
            self._fill_composition_table(self.state.composition)
            self.route_name_edit.setText(self.state.route_name)
            self._fill_route_table(self.state.route_operations)
            self.route_policy_combo.setCurrentIndex(
                1 if self.state.route_policy == "route_driven" else 0
            )
            self.preview_step_spin.setValue(
                -1
                if self.state.step_preview_index is None
                else int(self.state.step_preview_index)
            )
            self.step_series_check.setChecked(self.state.step_series_enabled)
            self.curve_enable_check.setChecked(bool(self.state.cooling_curve_enabled))
            curve_mode_idx = self.curve_mode_combo.findData(
                self.state.cooling_curve_mode
            )
            self.curve_mode_combo.setCurrentIndex(
                0 if curve_mode_idx < 0 else curve_mode_idx
            )
            self.curve_degree_step_spin.setValue(
                float(self.state.cooling_curve_degree_step)
            )
            self.curve_max_points_spin.setValue(
                int(self.state.cooling_curve_max_points)
            )
            self.proc_temp.setValue(float(self.state.processing.temperature_c))
            self._fill_curve_table(self.state.cooling_curve_points)
            cooling_idx = self.proc_cooling.findData(self.state.processing.cooling_mode)
            self.proc_cooling.setCurrentIndex(cooling_idx if cooling_idx >= 0 else 0)
            self.proc_deform.setValue(float(self.state.processing.deformation_pct))
            self.proc_aging_h.setValue(float(self.state.processing.aging_hours))
            self.proc_aging_t.setValue(float(self.state.processing.aging_temperature_c))
            self.proc_pressure.setValue(
                float(self.state.processing.pressure_mpa or 0.0)
            )
            self.proc_note.setText(self.state.processing.note)
            generator_idx = self.generator_combo.findData(self.state.generator)
            if generator_idx >= 0:
                self.generator_combo.setCurrentIndex(generator_idx)
            self.seed_spin.setValue(int(self.state.seed))
            self.res_h_spin.setValue(int(self.state.resolution[0]))
            self.res_w_spin.setValue(int(self.state.resolution[1]))
            mp = self.state.microscope_params
            self.mag_combo.setCurrentText(str(int(mp.get("magnification", 200))))
            self.focus_spin.setValue(float(mp.get("focus", 0.95)))
            self.brightness_spin.setValue(float(mp.get("brightness", 1.0)))
            self.contrast_spin.setValue(float(mp.get("contrast", 1.1)))
            self.noise_spin.setValue(float(mp.get("noise_sigma", 2.2)))
            self.vignette_spin.setValue(float(mp.get("vignette_strength", 0.12)))
            self.uneven_spin.setValue(float(mp.get("uneven_strength", 0.08)))
            self.dust_check.setChecked(bool(mp.get("add_dust", False)))
            self.scratch_check.setChecked(bool(mp.get("add_scratches", False)))
            self.etch_spin.setValue(float(mp.get("etch_uneven", 0.0)))
            self.generator_params_edit.setPlainText(self.state.generator_params_text)
            self.auto_norm_check.setChecked(self.state.auto_normalize)
            self.strict_check.setChecked(self.state.strict_validation)
            thermo = self.state.thermo
            self.calphad_profile_edit.setText(
                str(thermo.db_profile_path or self.pipeline.calphad_profile_path)
            )
            self.calphad_pressure_spin.setValue(float(thermo.pressure_pa))
            self.calphad_tgrid_spin.setValue(float(thermo.t_grid_step_c))
            self.calphad_scheil_dt_spin.setValue(float(thermo.scheil_dt_c))
            self.calphad_scheil_check.setChecked(bool(thermo.scheil_enabled))
            self.calphad_kinetics_check.setChecked(bool(thermo.kinetics_enabled))
            self._fill_calphad_db_table(dict(thermo.db_overrides))
            self.batch_dir_edit.setText(self.state.batch_output_dir)
            self.batch_prefix_edit.setText(self.state.batch_prefix)
            self._set_compare_mode(self.state.compare_mode)
            self._on_composition_mode_changed()
            self._on_route_policy_changed()
        finally:
            self._live_enabled = True

    def _fill_composition_table(self, composition: dict[str, float]) -> None:
        self.comp_table.blockSignals(True)
        try:
            self.comp_table.setRowCount(0)
            for symbol, value in composition.items():
                row = self.comp_table.rowCount()
                self.comp_table.insertRow(row)
                self.comp_table.setItem(row, 0, QTableWidgetItem(str(symbol)))
                self.comp_table.setItem(row, 1, QTableWidgetItem(f"{float(value):.6g}"))
        finally:
            self.comp_table.blockSignals(False)
        self._update_step1_validation()

    def _fill_phase_table(self, phases: dict[str, float] | None = None) -> None:
        phase_values = phases or {"Ferrite": 50.0, "Pearlite": 50.0}
        self.phase_table.blockSignals(True)
        try:
            self.phase_table.setRowCount(0)
            for key, label in self.FE_C_PHASES:
                row = self.phase_table.rowCount()
                self.phase_table.insertRow(row)
                self.phase_table.setItem(row, 0, QTableWidgetItem(key))
                self.phase_table.setItem(
                    row, 1, QTableWidgetItem(f"{float(phase_values.get(key, 0.0)):.6g}")
                )
                phase_item = self.phase_table.item(row, 0)
                if phase_item is not None:
                    phase_item.setToolTip(label)
                    phase_item.setFlags(
                        phase_item.flags() & ~Qt.ItemFlag.ItemIsEditable
                    )
        finally:
            self.phase_table.blockSignals(False)

    def _phase_normalize(self) -> None:
        phases = self._phase_table_values()
        total = sum(phases.values())
        if total <= 0:
            QMessageBox.warning(self, "Фазы", "Сумма фаз <= 0.")
            return
        normalized = {k: (100.0 * v / total) for k, v in phases.items()}
        self._fill_phase_table(normalized)
        self._update_step1_validation()

    def _on_composition_mode_changed(self) -> None:
        phase_mode = self._composition_mode() == "phase_fe_c"
        self.comp_box.setVisible(not phase_mode)
        self.phase_box.setVisible(phase_mode)
        if phase_mode and self.phase_table.rowCount() == 0:
            self._fill_phase_table()
        self._update_step1_validation()

    def _comp_add(self) -> None:
        row = self.comp_table.rowCount()
        self.comp_table.insertRow(row)
        self.comp_table.setItem(row, 0, QTableWidgetItem("Fe"))
        self.comp_table.setItem(row, 1, QTableWidgetItem("0.0"))
        self._update_step1_validation()

    def _comp_remove(self) -> None:
        rows = sorted(
            {idx.row() for idx in self.comp_table.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self.comp_table.removeRow(row)
        self._update_step1_validation()

    def _comp_normalize(self) -> None:
        composition = self._element_table_composition()
        total = sum(composition.values())
        if total <= 0.0:
            QMessageBox.warning(self, "Нормализация", "Сумма состава <= 0.")
            return
        normalized = {k: v * (100.0 / total) for k, v in composition.items()}
        self._fill_composition_table(normalized)

    def _comp_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Импорт состава", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(
                payload.get("composition"), dict
            ):
                payload = payload["composition"]
            composition = {str(k): float(v) for k, v in dict(payload).items()}
        except Exception as exc:
            QMessageBox.critical(self, "Импорт", f"Ошибка импорта состава: {exc}")
            return
        self._fill_composition_table(composition)

    def _comp_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт состава", "composition.json", "JSON (*.json)"
        )
        if path:
            save_json({"composition": self._table_composition()}, path)

    def _update_step1_validation(self) -> None:
        composition = self._table_composition()
        total = sum(composition.values())
        self.comp_sum_label.setText(f"Сумма: {total:.4f} мас.%")
        if self._composition_mode() == "phase_fe_c":
            phase_total = sum(self._phase_table_values().values())
            self.phase_sum_label.setText(f"Сумма фаз: {phase_total:.4f} %")
            comp_ph, phase_meta = self._phase_to_composition(self._phase_table_values())
            self.phase_hint.setText(
                f"Эквивалентный состав: Fe={comp_ph.get('Fe', 0.0):.3f}%, C={comp_ph.get('C', 0.0):.3f}%."
                f" Доминирующая фаза: {phase_meta.get('dominant_phase', '-')}"
            )
        report = validate_alloy(
            composition=composition,
            processing=self._current_processing(),
            auto_normalize=bool(self.auto_norm_check.isChecked())
            if hasattr(self, "auto_norm_check")
            else True,
            strict_custom_limits=bool(self.strict_check.isChecked())
            if hasattr(self, "strict_check")
            else True,
        )
        if report.is_valid:
            system_label = self.SYSTEM_LABELS.get(
                report.inferred_system, report.inferred_system
            )
            msg = f"Система: {system_label} (достоверность={report.confidence:.2f})"
            if report.warnings:
                self._set_step_status(0, "warn", report.warnings[0])
                msg += f"; предупреждений: {len(report.warnings)}"
            else:
                self._set_step_status(0, "ok", msg)
            self.comp_validation_hint.setText(msg)
        else:
            err = report.errors[0] if report.errors else "Ошибка состава"
            self.comp_validation_hint.setText("Ошибка: " + err)
            self._set_step_status(0, "error", err)
        self._sync_state_from_widgets()

    def _fill_route_table(self, operations: list[ProcessingOperation]) -> None:
        self.route_table.blockSignals(True)
        try:
            self.route_table.setRowCount(0)
            for op in operations:
                self._append_route_row(op)
        finally:
            self.route_table.blockSignals(False)

    def _append_route_row(self, op: ProcessingOperation) -> None:
        row = self.route_table.rowCount()
        self.route_table.insertRow(row)
        method_combo = WidePopupCombo(popup_min_width=520)
        for method in self.route_methods:
            label = self.METHOD_LABELS.get(method, method)
            method_combo.addItem(label, method)
        method_idx = method_combo.findData(op.method)
        if method_idx < 0 and op.method:
            method_combo.insertItem(0, f"{op.method} (custom)", op.method)
            method_idx = 0
        method_combo.setCurrentIndex(method_idx if method_idx >= 0 else 0)
        method_combo.currentIndexChanged.connect(self._update_step2_validation)
        self.route_table.setCellWidget(row, 0, method_combo)
        self.route_table.setItem(row, 1, QTableWidgetItem(f"{op.temperature_c:.6g}"))
        self.route_table.setItem(row, 2, QTableWidgetItem(f"{op.duration_min:.6g}"))
        cooling_combo = WidePopupCombo(popup_min_width=320)
        for code, label in self.COOLING_MODE_OPTIONS:
            cooling_combo.addItem(label, code)
        cooling_code = canonicalize_cooling_mode(op.cooling_mode)
        cooling_idx = cooling_combo.findData(cooling_code)
        if cooling_idx < 0 and op.cooling_mode:
            label = cooling_mode_label_ru(op.cooling_mode)
            if label == canonicalize_cooling_mode(op.cooling_mode):
                label = f"{op.cooling_mode} (custom)"
            cooling_combo.insertItem(
                0, label, canonicalize_cooling_mode(op.cooling_mode)
            )
            cooling_idx = 0
        cooling_combo.setCurrentIndex(cooling_idx if cooling_idx >= 0 else 0)
        cooling_combo.currentIndexChanged.connect(self._update_step2_validation)
        self.route_table.setCellWidget(row, 3, cooling_combo)
        self.route_table.setItem(row, 4, QTableWidgetItem(f"{op.deformation_pct:.6g}"))
        self.route_table.setItem(row, 5, QTableWidgetItem(f"{op.aging_hours:.6g}"))
        self.route_table.setItem(
            row, 6, QTableWidgetItem(f"{op.aging_temperature_c:.6g}")
        )
        self.route_table.setItem(row, 7, QTableWidgetItem(op.note))

    def _route_add(self) -> None:
        methods = [self.METHOD_LABELS.get(name, name) for name in self.route_methods]
        choice, ok = QInputDialog.getItem(
            self, "Добавить операцию", "Метод", methods, 0, False
        )
        if not ok or not choice:
            return
        reverse_map = {
            self.METHOD_LABELS.get(name, name): name for name in self.route_methods
        }
        method = reverse_map.get(
            choice, self.route_methods[0] if self.route_methods else ""
        )
        if not method:
            return
        self._append_route_row(ProcessingOperation(method=method))
        self._update_step2_validation()

    def _route_remove(self) -> None:
        rows = sorted(
            {idx.row() for idx in self.route_table.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self.route_table.removeRow(row)
        self._update_step2_validation()

    def _swap_route_rows(self, a: int, b: int) -> None:
        ops = self._route_from_table()
        if a < 0 or b < 0 or a >= len(ops) or b >= len(ops):
            return
        ops[a], ops[b] = ops[b], ops[a]
        self._fill_route_table(ops)

    def _route_up(self) -> None:
        row = self.route_table.currentRow()
        if row <= 0:
            return
        self._swap_route_rows(row, row - 1)
        self.route_table.selectRow(row - 1)
        self._update_step2_validation()

    def _route_down(self) -> None:
        row = self.route_table.currentRow()
        if row < 0 or row >= self.route_table.rowCount() - 1:
            return
        self._swap_route_rows(row, row + 1)
        self.route_table.selectRow(row + 1)
        self._update_step2_validation()

    def _route_load_template(self) -> None:
        keys = sorted(self.route_templates.keys())
        if not keys:
            QMessageBox.information(self, "Шаблоны", "Шаблоны не найдены.")
            return
        labels = [self.ROUTE_TEMPLATE_LABELS.get(k, k) for k in keys]
        label, ok = QInputDialog.getItem(
            self, "Шаблон маршрута", "Шаблон", labels, 0, False
        )
        if not ok or not label:
            return
        label_to_key = {self.ROUTE_TEMPLATE_LABELS.get(k, k): k for k in keys}
        key = label_to_key.get(label, keys[0])
        ops = [ProcessingOperation.from_dict(x) for x in self.route_templates[key]]
        self.route_name_edit.setText(self.ROUTE_TEMPLATE_LABELS.get(key, key))
        self._fill_route_table(ops)
        self._update_step2_validation()

    def _route_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Импорт маршрута", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            route = ProcessRoute.from_dict(
                json.loads(Path(path).read_text(encoding="utf-8"))
            )
        except Exception as exc:
            QMessageBox.critical(self, "Импорт", f"Ошибка импорта маршрута: {exc}")
            return
        self.route_name_edit.setText(route.route_name)
        self.step_series_check.setChecked(route.step_preview_enabled)
        self._fill_route_table(route.operations)
        self._update_step2_validation()

    def _route_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт маршрута", "route.json", "JSON (*.json)"
        )
        if not path:
            return
        route = ProcessRoute(
            operations=self._route_from_table(),
            route_name=self.route_name_edit.text().strip() or "Маршрут",
            route_notes="",
            step_preview_enabled=bool(self.step_series_check.isChecked()),
        )
        save_json(route.to_dict(), path)

    def _on_route_policy_changed(self) -> None:
        enabled = self._route_policy() == "route_driven"
        self.route_table.setEnabled(enabled)
        self.preview_step_spin.setEnabled(enabled)
        self.step_series_check.setEnabled(enabled)
        self._update_step2_validation()

    def _update_step2_validation(self) -> None:
        inferred = validate_alloy(
            composition=self._table_composition(),
            processing=self._current_processing(),
            auto_normalize=True,
            strict_custom_limits=False,
        ).inferred_system
        policy = self._route_policy()
        ops = self._route_from_table()
        route_errors: list[str] = []
        route_warnings: list[str] = []
        route_valid = True
        if policy == "route_driven":
            route = ProcessRoute(
                operations=ops,
                route_name=self.route_name_edit.text().strip() or "Маршрут",
                route_notes="",
                step_preview_enabled=bool(self.step_series_check.isChecked()),
            )
            rv = validate_process_route(
                route=route,
                inferred_system=inferred,
                processing_context=self._current_processing(),
            )
            route_valid = bool(rv.is_valid)
            route_errors = list(rv.errors)
            route_warnings = list(rv.warnings)

        curve_enabled = bool(self.curve_enable_check.isChecked())
        curve_errors: list[str] = []
        curve_warnings: list[str] = []
        curve_points: list[dict[str, float]] = []
        if curve_enabled:
            raw_rows: list[tuple[float, float]] = []
            for row in range(self.curve_table.rowCount()):
                t_item = self.curve_table.item(row, 0)
                temp_item = self.curve_table.item(row, 1)
                t_val = _safe_float(
                    "" if t_item is None else t_item.text(), float("nan")
                )
                temp_val = _safe_float(
                    "" if temp_item is None else temp_item.text(), float("nan")
                )
                if np.isnan(t_val) or np.isnan(temp_val):
                    curve_errors.append(
                        f"Кривая: строка {row + 1} содержит нечисловые значения"
                    )
                    continue
                raw_rows.append((float(t_val), float(temp_val)))
            if len(raw_rows) < 2:
                curve_errors.append("Кривая: требуется минимум 2 точки")
            for idx in range(1, len(raw_rows)):
                if raw_rows[idx][0] <= raw_rows[idx - 1][0]:
                    curve_errors.append("Кривая: время должно строго расти по строкам")
                    break
            if float(self.curve_degree_step_spin.value()) <= 0.0:
                curve_errors.append("Кривая: шаг по температуре должен быть > 0")
            curve_points = normalize_cooling_curve_points(
                [{"time_min": t, "temperature_c": temp} for t, temp in raw_rows],
                fallback_temperature_c=float(self.proc_temp.value()),
            )
            if len(curve_points) >= 2:
                total_drop = (
                    curve_points[0]["temperature_c"] - curve_points[-1]["temperature_c"]
                )
                if abs(total_drop) < 0.1:
                    curve_warnings.append(
                        "Кривая: перепад температуры близок к нулю, фазовый переход может быть слабо выражен"
                    )
                if len(curve_points) > int(self.curve_max_points_spin.value()):
                    curve_warnings.append(
                        "Кривая: число опорных точек превышает лимит кадров, серия будет прорежена"
                    )

        lines = [
            f"Система: {self.SYSTEM_LABELS.get(inferred, inferred)}",
            f"Режим: {self.ROUTE_POLICY_LABELS.get(policy, policy)}",
        ]
        if policy == "single_state":
            lines.append("Маршрут: не используется")
        else:
            lines.append(f"Операций маршрута: {len(ops)}")
            lines.append(
                f"Валидность маршрута: {'Корректен' if route_valid else 'Ошибка'}"
            )
            if route_errors:
                lines.append("Ошибки маршрута:")
                lines.extend([f"- {e}" for e in route_errors])
            if route_warnings:
                lines.append("Предупреждения маршрута:")
                lines.extend([f"- {w}" for w in route_warnings])

        if curve_enabled:
            lines.append(f"Кривая охлаждения: включена, точек={len(curve_points)}")
            lines.append(
                f"Режим кривой: {self.curve_mode_combo.currentData()}, шаг={self.curve_degree_step_spin.value():.3g} C"
            )
            if curve_errors:
                lines.append("Ошибки кривой:")
                lines.extend([f"- {e}" for e in curve_errors])
            if curve_warnings:
                lines.append("Предупреждения кривой:")
                lines.extend([f"- {w}" for w in curve_warnings])
        else:
            lines.append("Кривая охлаждения: выключена")

        self.route_validation_text.setPlainText("\n".join(lines))

        status = "ok"
        message = "Маршрут/кривая валидны"
        if policy == "route_driven" and not ops:
            status = "warn"
            message = "Маршрут пуст"
        if policy == "route_driven" and route_errors:
            status = "error"
            message = route_errors[0]
        if curve_enabled and curve_errors:
            status = "error"
            message = curve_errors[0]
        elif status != "error" and (route_warnings or curve_warnings):
            status = "warn"
            message = (route_warnings + curve_warnings)[0]
        self._set_step_status(1, status, message)

        self.preview_step_spin.setMaximum(max(-1, len(ops) - 1))
        self._sync_state_from_widgets()

    def _on_state_changed(self) -> None:
        self._sync_state_from_widgets()
        self._update_step3_validation()

    def _on_basic_live_changed(self) -> None:
        self._on_state_changed()
        if self._live_enabled and self.stack.currentIndex() == 3:
            self._preview_timer.start()

    def _on_heavy_changed(self) -> None:
        self._heavy_dirty = True
        self._on_state_changed()

    def _update_step3_validation(self) -> None:
        try:
            _ = self._generator_params()
        except Exception as exc:
            self.validation_text.setPlainText(str(exc))
            self._set_step_status(2, "error", str(exc))
            return
        report = validate_alloy(
            composition=self._table_composition(),
            processing=self._current_processing(),
            auto_normalize=bool(self.auto_norm_check.isChecked()),
            strict_custom_limits=bool(self.strict_check.isChecked()),
        )
        self.validation_text.setPlainText(format_validation_report(report))
        if report.is_valid and not report.warnings:
            self._set_step_status(2, "ok", "Параметры валидны")
        elif report.is_valid:
            self._set_step_status(2, "warn", report.warnings[0])
        else:
            self._set_step_status(2, "error", report.errors[0])

    def _validate_clicked(self) -> None:
        self._validate_all_steps()

    def _auto_fix(self) -> None:
        if self._composition_mode() == "phase_fe_c":
            phases = self._phase_table_values()
            total = float(sum(phases.values()))
            if total <= 0.0:
                QMessageBox.warning(
                    self, "Автоисправление", "Исправить автоматически не удалось."
                )
                return
            normalized = {key: (100.0 * value / total) for key, value in phases.items()}
            self._fill_phase_table(normalized)
            self._update_step1_validation()
            QMessageBox.information(
                self, "Автоисправление", "Фазовый состав нормализован."
            )
            return

        report = validate_alloy(
            composition=self._table_composition(),
            processing=self._current_processing(),
            auto_normalize=True,
            strict_custom_limits=False,
        )
        if report.normalized_composition:
            self._fill_composition_table(report.normalized_composition)
            QMessageBox.information(self, "Автоисправление", "Состав нормализован.")
        else:
            QMessageBox.warning(
                self, "Автоисправление", "Исправить автоматически не удалось."
            )

    def _set_compare_mode(self, key: str) -> None:
        mode = normalize_compare_mode(key)
        idx = self.compare_mode_combo.findData(mode)
        self.compare_mode_combo.setCurrentIndex(0 if idx < 0 else idx)

    def _update_compare_mode(self) -> None:
        self.state.compare_mode = normalize_compare_mode(
            self.compare_mode_combo.currentData()
        )
        if self.current_output is not None:
            self._render_compare_display()
        self._update_step4_status()

    def _update_step4_status(self) -> None:
        if self.current_output is None:
            self._set_step_status(3, "warn", "Предпросмотр не построен")
        elif (
            normalize_compare_mode(self.compare_mode_combo.currentData())
            == "phase_transition_curve"
            and not self.curve_enable_check.isChecked()
        ):
            self._set_step_status(
                3, "warn", "Для фазового перехода включите кривую охлаждения"
            )
        elif self._heavy_dirty:
            self._set_step_status(3, "warn", "Есть непримененные тяжелые параметры")
        else:
            self._set_step_status(3, "ok", "Готово к экспорту")

    def _on_diagram_view_resized(self) -> None:
        if self.current_request is None or self.current_output is None:
            return
        self._diagram_resize_timer.start()

    def _cache_signature(self, tag: str, *, include_layers: bool = False) -> str:
        if self.current_request is None:
            return ""
        payload: dict[str, Any] = {
            "tag": tag,
            "request": self.current_request.to_dict(),
            "compare_mode": normalize_compare_mode(
                self.compare_mode_combo.currentData()
            ),
        }
        if include_layers:
            payload["layers"] = self._diagram_layers()
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _apply_heavy_and_preview(self) -> None:
        self._run_preview(use_heavy=True, silent=False)

    def _run_preview(self, use_heavy: bool, silent: bool) -> None:
        self._validate_all_steps()
        if any(status == "error" for status in self.step_status[:3]):
            if not silent:
                QMessageBox.warning(
                    self, "Предпросмотр", "Есть блокирующие ошибки в шагах 1-3."
                )
            return
        try:
            request = self._compose_request(use_heavy=use_heavy)
            output = self.pipeline.generate(request)
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "Предпросмотр", f"Ошибка генерации: {exc}")
            return
        self.current_request = request
        self.current_output = output
        self.current_curve_outputs = []
        if use_heavy:
            self._heavy_dirty = False
        self._render_compare_display()
        self._render_diagram()
        self._update_step4_status()

    def _generate_before_output(self) -> GenerationOutputV2 | None:
        if self.current_request is None:
            return None
        key = self._cache_signature("before", include_layers=True)
        if (
            key
            and key == self._before_cache_key
            and self._before_cache_output is not None
        ):
            return self._before_cache_output
        req = GenerationRequestV2.from_dict(self.current_request.to_dict())
        req.route_policy = "single_state"
        req.process_route = None
        req.preview_step_index = None
        try:
            out = self.pipeline.generate(req)
            self._before_cache_key = key
            self._before_cache_output = out
            return out
        except Exception:
            self._before_cache_key = ""
            self._before_cache_output = None
            return None

    def _step_series_outputs(self) -> list[GenerationOutputV2]:
        if self.current_request is None or self.current_request.process_route is None:
            return []
        key = self._cache_signature("step_series", include_layers=True)
        if key and key == self._step_cache_key and self._step_cache_outputs:
            return self._step_cache_outputs
        total = len(self.current_request.process_route.operations)
        if total <= 0:
            return []
        picks = sorted(set([0, max(0, total // 2), total - 1]))
        out: list[GenerationOutputV2] = []
        for idx in picks:
            req = GenerationRequestV2.from_dict(self.current_request.to_dict())
            req.preview_step_index = idx
            try:
                out.append(self.pipeline.generate(req))
            except Exception:
                continue
        self._step_cache_key = key
        self._step_cache_outputs = out
        return out

    def _cooling_curve_series_outputs(self) -> list[GenerationOutputV2]:
        if self.current_request is None:
            return []
        key = self._cache_signature("cooling_curve_series", include_layers=True)
        if key and key == self._curve_cache_key and self._curve_cache_outputs:
            return self._curve_cache_outputs
        try:
            out = self.pipeline.generate_cooling_curve_series(self.current_request)
            self._curve_cache_key = key
            self._curve_cache_outputs = out
            return out
        except Exception:
            self._curve_cache_key = ""
            self._curve_cache_outputs = []
            return []

    def _render_compare_display(self) -> None:
        if self.current_output is None:
            return
        mode = normalize_compare_mode(self.compare_mode_combo.currentData())
        after = self.current_output.image_rgb
        display = after
        curve_track: list[dict[str, Any]] = []
        curve_events: list[dict[str, Any]] = []
        if mode == "before_after":
            before = self._generate_before_output()
            if before is not None:
                display = _hconcat([before.image_rgb, after])
        elif mode == "step_by_step":
            frames = self._step_series_outputs()
            if frames:
                display = _hconcat([f.image_rgb for f in frames])
        elif mode == "phase_transition_curve":
            frames = self._cooling_curve_series_outputs()
            self.current_curve_outputs = frames
            if frames:
                picks_count = min(6, len(frames))
                idxs = np.linspace(0, len(frames) - 1, num=picks_count, dtype=int)
                picked = [frames[int(i)] for i in idxs]
                display = _hconcat([f.image_rgb for f in picked])
                track_raw = frames[-1].metadata.get("phase_transition_track")
                if isinstance(track_raw, list):
                    curve_track = [x for x in track_raw if isinstance(x, dict)]
                events_raw = frames[-1].metadata.get("phase_transition_events")
                if isinstance(events_raw, list):
                    curve_events = [x for x in events_raw if isinstance(x, dict)]
        elif mode == "diff_map":
            before = self._generate_before_output()
            if before is not None:
                display = _diff_map(before.image_rgb, after)

        self.current_display = display
        self.preview_scene.clear()
        pix = _to_pixmap(display)
        self.preview_scene.addPixmap(pix)
        self.preview_scene.setSceneRect(pix.rect())
        self.preview_view.resetTransform()
        self.preview_view.fitInView(
            self.preview_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )
        self.preview_view.centerOn(self.preview_scene.sceneRect().center())

        meta = self.current_output.metadata
        lines = [
            f"Образец: {self.sample_name_edit.text().strip() or '-'}",
            f"Режим сравнения: {mode}",
            f"Система: {meta.get('inferred_system', '-')}",
            f"Стадия: {meta.get('final_stage', meta.get('stage', '-'))}",
            f"Генератор: {self.current_request.generator if self.current_request else '-'}",
        ]
        if mode == "phase_transition_curve":
            lines.append(f"Кадров перехода: {len(self.current_curve_outputs)}")
            if curve_track:
                stages = [str(x.get("stage", "")) for x in curve_track]
                unique_stages: list[str] = []
                for st in stages:
                    if st and (not unique_stages or unique_stages[-1] != st):
                        unique_stages.append(st)
                lines.append("Траектория стадий:")
                lines.append("- " + " -> ".join(unique_stages[:12]))
                start = curve_track[0]
                end = curve_track[-1]
                lines.append(
                    f"Диапазон: {start.get('temperature_c', '-')}C -> {end.get('temperature_c', '-')}C, "
                    f"{start.get('time_min', '-')} -> {end.get('time_min', '-')} мин"
                )
                lines.append(
                    f"Жидкая фаза: {float(start.get('liquid_fraction', 0.0)):.2f} -> "
                    f"{float(end.get('liquid_fraction', 0.0)):.2f}"
                )
            if curve_events:
                lines.append("События переходов:")
                for ev in curve_events[:6]:
                    lines.append(
                        f"- {ev.get('transition_kind', 'none')}:{ev.get('event', 'point')} "
                        f"(T={ev.get('temperature_c', '-')}, L={ev.get('liquid_fraction', '-')})"
                    )
        current_transition = meta.get("phase_transition_state")
        if isinstance(current_transition, dict):
            lines.append("Текущий переход:")
            lines.append(f"- Тип: {current_transition.get('transition_kind', 'none')}")
            lines.append(
                f"- L/S: {float(current_transition.get('liquid_fraction', 0.0)):.2f} / "
                f"{float(current_transition.get('solid_fraction', 1.0)):.2f}"
            )
            lines.append(
                f"- Направление: {current_transition.get('thermal_direction', 'steady')}"
            )
        calphad = meta.get("calphad")
        if isinstance(calphad, dict):
            eq = calphad.get("equilibrium_result", {})
            if isinstance(eq, dict):
                phases = eq.get("stable_phases", {})
                lines.append("CALPHAD:")
                lines.append(
                    f"- Solver: {calphad.get('solver_status', '-')}, "
                    f"{float(calphad.get('compute_time_ms', 0.0)):.2f} ms"
                )
                lines.append(f"- База: {calphad.get('database_used', '-')}")
                if isinstance(phases, dict) and phases:
                    top = sorted(
                        phases.items(), key=lambda x: float(x[1]), reverse=True
                    )[:4]
                    lines.append(
                        "- Фазы: "
                        + ", ".join([f"{name}:{float(frac):.3f}" for name, frac in top])
                    )
        props = meta.get("property_indicators")
        if isinstance(props, dict):
            lines.append("Свойства (оценочно):")
            lines.append(f"- HV: {props.get('hv_estimate', '-')}")
            lines.append(f"- UTS (MPa): {props.get('uts_estimate_mpa', '-')}")
            lines.append(f"- Пластичность: {props.get('ductility_class', '-')}")
        self.preview_info.setPlainText("\n".join(lines))

    def _diagram_layers(self) -> dict[str, bool]:
        return {
            "axes": bool(self.layer_axes.isChecked()),
            "grid": bool(self.layer_grid.isChecked()),
            "lines": bool(self.layer_lines.isChecked()),
            "invariants": bool(self.layer_inv.isChecked()),
            "regions": bool(self.layer_regions.isChecked()),
            "marker": bool(self.layer_marker.isChecked()),
        }

    def _render_diagram(self) -> None:
        if self.current_request is None or self.current_output is None:
            return
        viewport = self.diagram_view.viewport()
        view_w = max(1, int(viewport.width()) - 8)
        view_h = max(1, int(viewport.height()) - 8)
        render_w = max(640, view_w)
        render_h = max(420, view_h)
        requested = self.diagram_system_combo.currentData() or None
        snapshot = render_diagram_snapshot(
            composition=self.current_request.composition,
            processing=self.current_request.processing,
            requested_system=requested,
            inferred_system=self.current_output.validation_report.inferred_system,
            confidence=self.current_output.validation_report.confidence,
            layers=self._diagram_layers(),
            size=(render_w, render_h),
        )
        self.current_diagram_snapshot = snapshot
        diagram = np.asarray(snapshot["image"], dtype=np.uint8)
        self.diagram_scene.clear()
        pix = _to_pixmap(diagram)
        self.diagram_scene.addPixmap(pix)
        self.diagram_scene.setSceneRect(pix.rect())
        self.diagram_view.resetTransform()
        self.diagram_view.fitInView(
            self.diagram_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )
        self.diagram_view.centerOn(self.diagram_scene.sceneRect().center())

    def _save_current_image(self) -> None:
        if self.current_display is None:
            QMessageBox.warning(self, "Сохранение", "Сначала выполните предпросмотр.")
            return
        default = (self.sample_name_edit.text().strip() or "sample_v2") + ".png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить изображение",
            default,
            "PNG (*.png);;JPG (*.jpg);;TIFF (*.tiff)",
        )
        if not path:
            return
        save_image(self.current_display, path)
        QMessageBox.information(self, "Сохранение", f"Сохранено:\n{path}")

    def _save_current_metadata(self) -> None:
        if self.current_output is None or self.current_request is None:
            QMessageBox.warning(self, "Сохранение", "Нет метаданных для сохранения.")
            return
        default = (self.sample_name_edit.text().strip() or "sample_v2") + ".json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить метаданные", default, "JSON (*.json)"
        )
        if not path:
            return
        payload = self.current_output.metadata_json_safe()
        payload["ui_compare_mode"] = normalize_compare_mode(
            self.compare_mode_combo.currentData()
        )
        payload["sample_name"] = self.sample_name_edit.text().strip() or "sample_v2"
        payload["request"] = self.current_request.to_dict()
        if self.current_curve_outputs:
            payload["cooling_curve_preview_frames"] = len(self.current_curve_outputs)
            track = self.current_curve_outputs[-1].metadata.get(
                "phase_transition_track"
            )
            if isinstance(track, list):
                payload["cooling_curve_preview_track"] = track
        save_json(payload, path)
        QMessageBox.information(self, "Сохранение", f"Сохранено:\n{path}")

    def _save_current_diagram(self) -> None:
        if not self.current_diagram_snapshot:
            QMessageBox.warning(
                self, "Сохранение", "Нет диаграммы. Выполните предпросмотр."
            )
            return
        default = (self.sample_name_edit.text().strip() or "sample_v2") + "_diagram.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить диаграмму", default, "PNG (*.png)"
        )
        if not path:
            return
        save_diagram_png(self.current_diagram_snapshot, path)
        QMessageBox.information(self, "Сохранение", f"Сохранено:\n{path}")

    def _browse_batch_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Папка пакетной генерации")
        if path:
            self.batch_dir_edit.setText(path)

    def _add_to_batch(self) -> None:
        try:
            request = self._compose_request(use_heavy=True)
        except Exception as exc:
            QMessageBox.critical(self, "Пакет", f"Ошибка сборки запроса: {exc}")
            return
        self.batch_queue.append(request)
        row = self.batch_table.rowCount()
        self.batch_table.insertRow(row)
        self.batch_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        self.batch_table.setItem(
            row,
            1,
            QTableWidgetItem(
                self.sample_name_edit.text().strip() or f"sample_{row + 1}"
            ),
        )
        self.batch_table.setItem(
            row,
            2,
            QTableWidgetItem(
                self.GENERATOR_LABELS.get(request.generator, request.generator)
            ),
        )
        self.batch_table.setItem(
            row,
            3,
            QTableWidgetItem(
                self.ROUTE_POLICY_LABELS.get(request.route_policy, request.route_policy)
            ),
        )
        self.batch_table.setItem(row, 4, QTableWidgetItem(str(request.seed)))
        self.batch_table.setItem(
            row, 5, QTableWidgetItem(f"{request.resolution[0]}x{request.resolution[1]}")
        )

    def _run_batch(self) -> None:
        if not self.batch_queue:
            QMessageBox.information(self, "Пакет", "Очередь пуста.")
            return
        out_dir = Path(
            self.batch_dir_edit.text().strip() or self.state.batch_output_dir
        )
        prefix = self.batch_prefix_edit.text().strip() or "sample_v2"
        try:
            result = self.pipeline.generate_batch(
                self.batch_queue, output_dir=out_dir, file_prefix=prefix
            )
        except Exception as exc:
            QMessageBox.critical(self, "Пакет", f"Ошибка пакетной генерации: {exc}")
            return
        ok_count = sum(1 for row in result.rows if not row.get("error"))
        fail_count = len(result.rows) - ok_count
        lines = [
            f"Пакетная генерация завершена: успешно={ok_count}, ошибок={fail_count}",
            f"CSV-индекс: {result.csv_index_path}",
        ]
        for row in result.rows:
            if row.get("error"):
                lines.append(f"- {row.get('sample_id')}: {row.get('error')}")
        self.batch_log.setPlainText("\n".join(lines))
        QMessageBox.information(self, "Пакет", "Генерация завершена.")

    def _load_preset_names(self) -> None:
        self.preset_combo.clear()
        for path in self.pipeline.list_preset_paths():
            self.preset_combo.addItem(path.stem, str(path))

    def _seed_default_route(self) -> None:
        if not self.state.route_operations and self.route_templates:
            first = sorted(self.route_templates.keys())[0]
            self.state.route_name = self.ROUTE_TEMPLATE_LABELS.get(first, first)
            self.state.route_operations = [
                ProcessingOperation.from_dict(x) for x in self.route_templates[first]
            ]

    def _load_selected_preset(self) -> None:
        path = self.preset_combo.currentData()
        if not path:
            return
        try:
            preset = self.pipeline.load_preset(path)
            req = self.pipeline.request_from_preset(preset)
        except Exception as exc:
            QMessageBox.critical(self, "Пресет", f"Ошибка загрузки пресета: {exc}")
            return
        self.state.sample_name = req.preset_name or preset.name
        self.state.composition_mode = "elements"
        self.state.composition = dict(req.composition)
        self.state.processing = req.processing
        self.state.generator = req.generator
        self.state.seed = req.seed
        self.state.resolution = req.resolution
        self.state.generator_params_text = json.dumps(
            req.generator_params, ensure_ascii=False, indent=2
        )
        self.state.cooling_curve_enabled = bool(
            req.generator_params.get("cooling_curve_enabled", False)
        )
        self.state.cooling_curve_mode = str(
            req.generator_params.get("cooling_curve_mode", "per_degree")
        )
        self.state.cooling_curve_degree_step = float(
            req.generator_params.get("cooling_curve_degree_step", 1.0)
        )
        self.state.cooling_curve_max_points = int(
            req.generator_params.get("cooling_curve_max_points", 220)
        )
        self.state.cooling_curve_points = normalize_cooling_curve_points(
            req.generator_params.get("cooling_curve", []),
            fallback_temperature_c=float(req.processing.temperature_c),
        )
        self.state.microscope_params = dict(req.microscope_params)
        self.state.auto_normalize = req.auto_normalize
        self.state.strict_validation = req.strict_validation
        self.state.thermo = (
            ThermoBackendConfig.from_dict(req.thermo.to_dict())
            if req.thermo is not None
            else ThermoBackendConfig(
                db_profile_path=str(self.pipeline.calphad_profile_path)
            )
        )
        self.state.route_policy = req.route_policy
        self.state.step_preview_index = req.preview_step_index
        if req.process_route is not None:
            self.state.route_name = req.process_route.route_name
            self.state.route_operations = list(req.process_route.operations)
            self.state.step_series_enabled = req.process_route.step_preview_enabled
        else:
            self.state.route_name = "Маршрут"
            self.state.route_operations = []
            self.state.step_series_enabled = True
        self._sync_widgets_from_state()
        self._validate_all_steps()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._preview_timer.stop()
        super().closeEvent(event)


def launch_sample_factory_app(
    presets_dir: str | Path | None = None,
    calphad_profile_path: str | Path | None = None,
    calphad_tdb_dir: str | Path | None = None,
    calphad_cache_dir: str | Path | None = None,
) -> None:
    app = QApplication.instance() or QApplication([])
    win = SampleFactoryWindow(
        presets_dir=presets_dir,
        calphad_profile_path=calphad_profile_path,
        calphad_tdb_dir=calphad_tdb_dir,
        calphad_cache_dir=calphad_cache_dir,
    )
    win.show()
    app.exec()
