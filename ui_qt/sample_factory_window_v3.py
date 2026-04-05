from __future__ import annotations

import json
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
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
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSlider,
    QSplitter,
    QStackedWidget,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.contracts_v3 import (
    EtchProfileV3,
    GenerationOutputV3,
    MetallographyRequestV3,
    PhaseModelConfigV3,
    PrepOperationV3,
    QuenchSettingsV3,
    SamplePrepRouteV3,
    SynthesisProfileV3,
    ThermalTransitionV3,
    ThermalPointV3,
    ThermalProgramV3,
)
from core.metallography_v3.phase_orchestrator import infer_training_system
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3
from core.metallography_v3.hardness_brinell import hbw_from_indent
from core.metallography_v3.quench_media_v3 import canonicalize_quench_medium_code
from core.metallography_v3.thermal_program_v3 import (
    effective_processing_from_thermal,
    sample_thermal_program,
    summarize_thermal_program,
    validate_thermal_program,
)
from core.security import (
    activate_teacher_mode_with_prompt,
    resolve_teacher_key_path,
)
from ui_qt.theme_mirea import (
    GENERATOR_THEME_PROFILE,
    build_qss,
    load_theme_mode,
    resolve_branding_logo,
    save_theme_mode,
    status_color,
)
from ui_qt.modern_widgets import FlexibleDoubleSpinBox, parse_flexible_float
from ui_qt.window_state_manager import WindowStateManager
from ui_qt.window_mode_mixin import WindowModeMixin
from export.export_images import save_image
from export.export_tables import save_json

logger = logging.getLogger(__name__)

QDoubleSpinBox = FlexibleDoubleSpinBox

try:
    import pyqtgraph as pg  # type: ignore
except ImportError:  # pragma: no cover
    pg = None
    logger.warning("pyqtgraph not available - thermal plots will be disabled")

SENSITIVITY_OPTIONS: list[tuple[str, str]] = [
    ("realistic", "Реалистичный"),
    ("educational", "Учебный усиленный"),
    ("high_contrast", "Высокий контраст"),
]

GENERATION_MODE_OPTIONS: list[tuple[str, str]] = [
    ("edu_engineering", "Учебно-инженерный"),
    ("realistic_visual", "Реалистичный визуальный"),
    ("pro_realistic", "Pro realistic"),
]

OPTICAL_MODE_OPTIONS: list[tuple[str, str]] = [
    ("brightfield", "Светлое поле"),
    ("darkfield", "Тёмное поле"),
    ("polarized", "Поляризованный свет"),
    ("phase_contrast", "Фазовый контраст"),
    ("dic", "DIC / Nomarski"),
    ("magnetic_etching", "Магнитное травление"),
]

PSF_PROFILE_OPTIONS: list[tuple[str, str]] = [
    ("standard", "Стандартный PSF"),
    ("bessel_extended_dof", "Bessel extended DOF"),
    ("airy_push_pull", "Airy push/pull"),
    ("self_rotating", "Self-rotating"),
    ("stir_sectioning", "STIR sectioning"),
    ("lens_axicon_hybrid", "Lens-axicon hybrid"),
]

PHASE_EMPHASIS_OPTIONS: list[tuple[str, str]] = [
    ("contrast_texture", "Контраст + текстура"),
    ("max_contrast", "Максимальный контраст"),
    ("morphology_only", "Только морфология"),
]

TOPOLOGY_OPTIONS: list[tuple[str, str]] = [
    ("auto", "Авто"),
    ("equiaxed", "Равноосная"),
    ("columnar", "Столбчатая"),
    ("lamellar", "Ламеллярная"),
    ("lath", "Реечная"),
    ("dendritic", "Дендритная"),
    ("networked precipitates", "Сетчатые выделения"),
]

SYSTEM_GENERATOR_OPTIONS: list[tuple[str, str]] = [
    ("system_auto", "Авто (по системе)"),
    ("system_fe_c", "Система Fe-C"),
    ("system_fe_si", "Система Fe-Si"),
    ("system_al_si", "Система Al-Si"),
    ("system_cu_zn", "Система Cu-Zn"),
    ("system_al_cu_mg", "Система Al-Cu-Mg"),
    ("system_custom", "Пользовательская (резерв)"),
]

ETCH_REAGENTS: list[str] = ["nital_2", "picral", "keller", "fry", "custom"]

AGITATION_OPTIONS: list[tuple[str, str]] = [
    ("none", "Без перемешивания"),
    ("gentle", "Умеренное"),
    ("active", "Активное"),
]

PREP_METHODS: list[str] = [
    "grinding_600",
    "grinding_800",
    "grinding_1200",
    "polishing_3um",
    "polishing_1um",
    "electropolish",
    "cleaning",
    "mounting",
]

COOLANT_OPTIONS: list[str] = [
    "water",
    "alcohol",
    "oil",
    "electrolyte",
    "ultrasonic",
    "none",
]

RESOLUTION_OPTIONS: list[tuple[str, tuple[int, int]]] = [
    ("1024 x 1024 (черновой)", (1024, 1024)),
    ("1536 x 1536", (1536, 1536)),
    ("2048 x 2048 (финальный)", (2048, 2048)),
    ("4096 x 4096 (детальный)", (4096, 4096)),
    ("8192 x 8192 (ультра)", (8192, 8192)),
    ("16384 x 16384 (16K)", (16384, 16384)),
]

STEP_NAMES: list[str] = [
    "1. Система и состав",
    "2. Термопрограмма",
    "3. Подготовка шлифа",
    "4. Травление",
    "5. Синтез и фазовая модель",
    "6. Контроль качества и экспорт",
]

QUENCH_MEDIA_OPTIONS: list[tuple[str, str]] = [
    ("water_20", "Вода 20°C"),
    ("water_100", "Вода 100°C"),
    ("brine_20_30", "Солёная вода 20-30°C"),
    ("oil_20_80", "Масло 20-80°C"),
    ("polymer", "Полимерный раствор"),
    ("air", "Воздух"),
    ("furnace", "Печь"),
    ("custom", "Пользовательская среда"),
]

QUENCH_MEDIUM_DEFAULT_BATH: dict[str, float] = {
    "water_20": 20.0,
    "water_100": 100.0,
    "brine_20_30": 25.0,
    "oil_20_80": 60.0,
    "polymer": 40.0,
    "air": 25.0,
    "furnace": 25.0,
    "custom": 20.0,
}
QUENCH_MEDIUM_LABELS_RU: dict[str, str] = {
    code: label for code, label in QUENCH_MEDIA_OPTIONS
}
SEGMENT_MEDIUM_OPTIONS: list[tuple[str, str]] = [("inherit", "Наследовать")] + list(
    QUENCH_MEDIA_OPTIONS
)
SEGMENT_MEDIUM_LABELS_RU: dict[str, str] = {
    code: label for code, label in SEGMENT_MEDIUM_OPTIONS
}

THERMAL_TRANSITION_MODEL_OPTIONS: list[tuple[str, str]] = [
    ("auto", "Авто по среде"),
    ("linear", "Линейный"),
    ("sigmoid", "Сигмоида"),
    ("power", "Степенной"),
    ("cosine", "Косинусный"),
]
THERMAL_TRANSITION_MODEL_LABELS_RU: dict[str, str] = {
    code: label for code, label in THERMAL_TRANSITION_MODEL_OPTIONS
}

TRAINING_SYSTEM_OPTIONS: tuple[str, ...] = (
    "fe-c",
    "fe-si",
    "al-si",
    "cu-zn",
    "al-cu-mg",
)

TEXTBOOK_PROFILE_BY_SYSTEM: dict[str, str] = {
    "fe-c": "textbook_steel_bw",
    "fe-si": "textbook_steel_bw",
    "al-si": "textbook_alsi_bw",
    "cu-zn": "textbook_brass_bw",
    "al-cu-mg": "textbook_heat_treatment_bw",
}

LAB_TEMPLATE_PRESETS: list[tuple[str, str]] = [
    ("ЛР1: Доэвтектоидная сталь Fe-C", "fe_c_hypoeutectoid_textbook"),
    ("ЛР1: Эвтектоидная сталь Fe-C", "fe_c_eutectoid_textbook"),
    ("ЛР1: Заэвтектоидная сталь Fe-C", "fe_c_hypereutectoid_textbook"),
    ("ЛР1 Pro: Доэвтектоидная сталь Fe-C", "fe_c_hypoeutectoid_pro_realistic"),
    ("ЛР1 Pro: Эвтектоидная сталь Fe-C", "fe_c_eutectoid_pro_realistic"),
    ("ЛР1 Pro: Заэвтектоидная сталь Fe-C", "fe_c_hypereutectoid_pro_realistic"),
    ("ЛР1: Al-Si эвтектика", "alsi_eutectic_textbook"),
    ("ЛР1: Латунь α/β", "brass_alpha_beta_textbook"),
    ("ЛР4: Закаленная сталь", "steel_quenched_textbook"),
    ("ЛР4: Отпуск 200°C", "steel_tempered_200_textbook"),
    ("ЛР4: Отпуск 400°C", "steel_tempered_400_textbook"),
    ("ЛР4: Отпуск 600°C", "steel_tempered_600_textbook"),
]

LR1_SAMPLE_LIBRARY_PRESETS: list[tuple[str, str]] = [
    ("ЛР1 ASTM 5", "LR1_ASTM5"),
    ("ЛР1 ASTM 6", "LR1_ASTM6"),
    ("ЛР1 ASTM 7", "LR1_ASTM7"),
    ("ЛР1 ASTM 8", "LR1_ASTM8"),
]

STEEL_SAMPLE_LIBRARY_PRESETS: list[tuple[str, str]] = [
    ("Ст45 норм.", "steel45_normalized_textbook"),
    ("Ст45 улучш.", "steel45_improved_textbook"),
    ("У8 зак.+отп.", "steel_u8_tool_textbook"),
]

CAST_IRON_SAMPLE_LIBRARY_PRESETS: list[tuple[str, str]] = [
    ("СЧ20", "cast_iron_grey_textbook"),
]

LAB_RESEARCH_TEMPLATE_PRESETS: list[tuple[str, str]] = [
    (
        "Исследовательская оптика: Fe-C Bessel DOF",
        "fe_c_eutectoid_research_optics_bessel",
    ),
    (
        "Исследовательская оптика: Fe-C STIR sectioning",
        "fe_c_eutectoid_research_optics_stir",
    ),
    (
        "Исследовательская оптика: Fe-C lens-axicon",
        "fe_c_eutectoid_research_optics_hybrid",
    ),
]

SYSTEM_LABELS_RU: dict[str, str] = {
    "fe-c": "Fe-C (железо-углерод)",
    "fe-si": "Fe-Si (электротехническая сталь)",
    "al-si": "Al-Si (литейный алюминиевый)",
    "cu-zn": "Cu-Zn (латунь)",
    "al-cu-mg": "Al-Cu-Mg (дюралюмины)",
    "custom-multicomponent": "Пользовательская многокомпонентная",
}

PREP_METHOD_LABELS_RU: dict[str, str] = {
    "grinding_600": "Шлифовка P600",
    "grinding_800": "Шлифовка P800",
    "grinding_1200": "Шлифовка P1200",
    "polishing_3um": "Полировка 3 мкм",
    "polishing_1um": "Полировка 1 мкм",
    "electropolish": "Электрополировка",
    "cleaning": "Очистка",
    "mounting": "Монтаж в оправку",
}

COOLANT_LABELS_RU: dict[str, str] = {
    "water": "Вода",
    "alcohol": "Спирт",
    "oil": "Масло",
    "electrolyte": "Электролит",
    "ultrasonic": "Ультразвуковая ванна",
    "none": "Без охлаждения",
}

ETCH_REAGENT_LABELS_RU: dict[str, str] = {
    "nital_2": "Ниталь 2%",
    "picral": "Пикраль",
    "keller": "Реактив Келлера",
    "fry": "Реактив Фрая",
    "custom": "Пользовательский реактив",
}

PREP_TEMPLATE_LABELS_RU: dict[str, str] = {
    "al_cast_polish": "Литой Al-Si: шлифовка и полировка",
    "si_etch_pit": "Si: подготовка под ямки травления",
    "steel_bright_field": "Сталь: светлое поле",
    "textbook_alsi_bw": "Учебник (ч/б): Al-Si",
    "textbook_brass_bw": "Учебник (ч/б): латунь",
    "textbook_heat_treatment_bw": "Учебник (ч/б): термообработка",
    "textbook_steel_bw": "Учебник (ч/б): сталь",
}

ETCH_PROFILE_LABELS_RU: dict[str, str] = {
    "fry": "Фрай (базовый)",
    "keller": "Келлер (базовый)",
    "nital_2": "Ниталь 2% (базовый)",
    "picral": "Пикраль (базовый)",
    "textbook_alsi_bw": "Учебник (ч/б): Al-Si",
    "textbook_brass_bw": "Учебник (ч/б): латунь",
    "textbook_heat_treatment_bw": "Учебник (ч/б): термообработка",
    "textbook_steel_bw": "Учебник (ч/б): сталь",
}

SYNTH_PROFILE_LABELS_RU: dict[str, str] = {
    "textbook_steel_bw": "Учебник (ч/б): сталь",
    "textbook_alsi_bw": "Учебник (ч/б): Al-Si",
    "textbook_brass_bw": "Учебник (ч/б): латунь",
    "textbook_heat_treatment_bw": "Учебник (ч/б): термообработка",
}

GENERATION_MODE_LABELS_RU: dict[str, str] = {
    key: label for key, label in GENERATION_MODE_OPTIONS
}
OPTICAL_MODE_LABELS_RU: dict[str, str] = {
    key: label for key, label in OPTICAL_MODE_OPTIONS
}
PSF_PROFILE_LABELS_RU: dict[str, str] = {
    key: label for key, label in PSF_PROFILE_OPTIONS
}
PHASE_EMPHASIS_LABELS_RU: dict[str, str] = {
    key: label for key, label in PHASE_EMPHASIS_OPTIONS
}
SENSITIVITY_LABELS_RU: dict[str, str] = {
    key: label for key, label in SENSITIVITY_OPTIONS
}
PHASE_CONTROL_MODE_LABELS_RU: dict[str, str] = {
    "auto_with_override": "Авто + ручная коррекция",
    "auto_only": "Только авто",
    "manual_only": "Только ручной",
}
SYSTEM_GENERATOR_MODE_LABELS_RU: dict[str, str] = {
    key: label for key, label in SYSTEM_GENERATOR_OPTIONS
}
STRUCTURE_STAGE_LABELS_RU: dict[str, str] = {
    "alpha_pearlite": "Феррит-перлит",
    "pearlite": "Перлит",
    "pearlite_cementite": "Перлит-цементит",
    "martensite": "Мартенсит",
    "martensite_tetragonal": "Мартенсит",
    "martensite_cubic": "Мартенсит",
    "troostite_quench": "Троостит закалки",
    "troostite_temper": "Троостит отпуска",
    "sorbite_quench": "Сорбит закалки",
    "sorbite_temper": "Сорбит отпуска",
    "tempered_low": "Отпущенный мартенсит",
    "tempered_medium": "Троостит отпуска",
    "tempered_high": "Сорбит отпуска",
    "bainite": "Бейнит",
    "ledeburite": "Ледебурит",
    "ferrite": "Феррит",
    "austenite": "Аустенит",
}
PHASE_NAME_LABELS_RU: dict[str, str] = {
    "FERRITE": "Феррит",
    "PEARLITE": "Перлит",
    "CEMENTITE": "Цементит",
    "AUSTENITE": "Аустенит",
    "MARTENSITE": "Мартенсит",
    "MARTENSITE_TETRAGONAL": "Мартенсит",
    "MARTENSITE_CUBIC": "Мартенсит",
    "TROOSTITE": "Троостит",
    "SORBITE": "Сорбит",
    "BAINITE": "Бейнит",
    "LEDEBURITE": "Ледебурит",
    "TEMPERED_MATRIX": "Отпущенная матрица",
}

LR1_TEMPLATE_GROUP_LABEL = "──── ЛР1 / ASTM E112 ────"
CURRICULUM_TEMPLATE_GROUP_LABEL = "──── Учебные шаблоны ЛР ────"
STEEL_LIBRARY_GROUP_LABEL = "──── Учебные стали и термообработка ────"
CAST_IRON_LIBRARY_GROUP_LABEL = "──── Учебные чугуны ────"
RESEARCH_OPTICS_GROUP_LABEL = "──── Исследовательская оптика ────"

PRESET_DISPLAY_LABELS_RU: dict[str, str] = {
    "LR1_ASTM5": "ЛР1 ASTM 5",
    "LR1_ASTM6": "ЛР1 ASTM 6",
    "LR1_ASTM7": "ЛР1 ASTM 7",
    "LR1_ASTM8": "ЛР1 ASTM 8",
    "fe_c_hypoeutectoid_textbook": "ЛР1 доэвтект.",
    "fe_c_eutectoid_textbook": "ЛР1 эвтектоид.",
    "fe_c_hypereutectoid_textbook": "ЛР1 заэвтект.",
    "fe_c_hypoeutectoid_pro_realistic": "ЛР1 Pro доэвт.",
    "fe_c_eutectoid_pro_realistic": "ЛР1 Pro эвтект.",
    "fe_c_hypereutectoid_pro_realistic": "ЛР1 Pro заэвт.",
    "alsi_eutectic_textbook": "ЛР1 Al-Si",
    "brass_alpha_beta_textbook": "ЛР1 латунь α/β",
    "steel_quenched_textbook": "ЛР4 закалка",
    "steel_tempered_200_textbook": "ЛР4 отпуск 200°",
    "steel_tempered_400_textbook": "ЛР4 отпуск 400°",
    "steel_tempered_600_textbook": "ЛР4 отпуск 600°",
    "steel45_normalized_textbook": "Ст45 норм.",
    "steel45_improved_textbook": "Ст45 улучш.",
    "steel_u8_tool_textbook": "У8 зак.+отп.",
    "cast_iron_grey_textbook": "СЧ20",
    "fe_c_eutectoid_research_optics_bessel": "Иссл. оптика Bessel",
    "fe_c_eutectoid_research_optics_stir": "Иссл. оптика STIR",
    "fe_c_eutectoid_research_optics_hybrid": "Иссл. оптика hybrid",
}

PRESET_DISPLAY_TOOLTIPS_RU: dict[str, str] = {
    "LR1_ASTM5": "ЛР1 ASTM 5: крупное зерно, ~90 мкм, 100×.",
    "LR1_ASTM6": "ЛР1 ASTM 6: среднее зерно, ~65 мкм, 100×.",
    "LR1_ASTM7": "ЛР1 ASTM 7: мелкое зерно, ~45 мкм, 100×.",
    "LR1_ASTM8": "ЛР1 ASTM 8: очень мелкое зерно, ~32 мкм, 100×.",
    "fe_c_hypoeutectoid_textbook": "ЛР1: доэвтектоидная сталь Fe-C.",
    "fe_c_eutectoid_textbook": "ЛР1: эвтектоидная сталь Fe-C.",
    "fe_c_hypereutectoid_textbook": "ЛР1: заэвтектоидная сталь Fe-C.",
    "fe_c_hypoeutectoid_pro_realistic": "ЛР1 Pro: доэвтектоидная сталь Fe-C.",
    "fe_c_eutectoid_pro_realistic": "ЛР1 Pro: эвтектоидная сталь Fe-C.",
    "fe_c_hypereutectoid_pro_realistic": "ЛР1 Pro: заэвтектоидная сталь Fe-C.",
    "alsi_eutectic_textbook": "ЛР1: Al-Si эвтектика.",
    "brass_alpha_beta_textbook": "ЛР1: латунь α/β.",
    "steel_quenched_textbook": "ЛР4: закаленная сталь.",
    "steel_tempered_200_textbook": "ЛР4: отпуск 200°C.",
    "steel_tempered_400_textbook": "ЛР4: отпуск 400°C.",
    "steel_tempered_600_textbook": "ЛР4: отпуск 600°C.",
    "steel45_normalized_textbook": "Ст45: нормализация, феррит + перлит.",
    "steel45_improved_textbook": "Ст45: улучшение, закалка + высокий отпуск.",
    "steel_u8_tool_textbook": "У8: инструментальная сталь, закалка и низкий отпуск.",
    "cast_iron_grey_textbook": "СЧ20: серый чугун с пластинчатым графитом.",
    "fe_c_eutectoid_research_optics_bessel": "Исследовательская оптика: Fe-C Bessel DOF.",
    "fe_c_eutectoid_research_optics_stir": "Исследовательская оптика: Fe-C STIR sectioning.",
    "fe_c_eutectoid_research_optics_hybrid": "Исследовательская оптика: Fe-C lens-axicon.",
}


def _json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    return parse_flexible_float(value, default)


def _label_ru(code: str, mapping: dict[str, str]) -> str:
    return mapping.get(code, code)


def _yes_no(value: Any) -> str:
    return "да" if bool(value) else "нет"


def _preset_visible_label(stem: str) -> str:
    return PRESET_DISPLAY_LABELS_RU.get(str(stem), str(stem))


def _preset_tooltip(stem: str) -> str:
    token = str(stem)
    return PRESET_DISPLAY_TOOLTIPS_RU.get(token, token)


def _add_combo_item_with_tooltip(
    combo: QComboBox, label: str, data: Any, tooltip: str | None = None
) -> int:
    combo.addItem(label, data)
    index = combo.count() - 1
    if tooltip:
        combo.setItemData(index, tooltip, Qt.ItemDataRole.ToolTipRole)
    return index


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


def _pil_to_pixmap(image: Image.Image) -> QPixmap:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return _to_pixmap(arr)


def _preview_image_rgb(image_rgb: np.ndarray, max_side: int = 3072) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        raise ValueError(f"Unsupported preview shape: {image_rgb.shape}")
    h, w = int(image_rgb.shape[0]), int(image_rgb.shape[1])
    if max(h, w) <= int(max_side):
        return image_rgb[:, :, :3]
    scale = float(max(h, w)) / float(max(256, int(max_side)))
    out_h = max(256, int(round(h / scale)))
    out_w = max(256, int(round(w / scale)))
    pil = Image.fromarray(image_rgb[:, :, :3], mode="RGB")
    resized = pil.resize((out_w, out_h), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


class ZoomView(QGraphicsView):
    resized = Signal()

    def __init__(self, parent: QWidget | None = None, allow_drag: bool = True) -> None:
        super().__init__(parent)
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if allow_drag
            else QGraphicsView.DragMode.NoDrag
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.resized.emit()


class SampleFactoryWindowV3(QMainWindow, WindowModeMixin):
    def __init__(
        self,
        presets_dir: str | Path | None = None,
        profiles_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.presets_dir = (
            Path(presets_dir) if presets_dir is not None else Path("presets_v3")
        )
        self.profiles_dir = (
            Path(profiles_dir) if profiles_dir is not None else Path("profiles_v3")
        )

        self.pipeline = MetallographyPipelineV3(
            presets_dir=self.presets_dir,
            profiles_dir=self.profiles_dir,
        )
        self._etch_profiles = _json_load(self.profiles_dir / "etch_profiles.json").get(
            "profiles", {}
        )
        self._prep_templates = _json_load(
            self.profiles_dir / "prep_templates.json"
        ).get("templates", {})
        self._synth_profiles = _json_load(
            self.profiles_dir / "metallography_profiles.json"
        ).get("profiles", {})
        self._reference_profiles = self._find_reference_profiles()

        self.current_request: MetallographyRequestV3 | None = None
        self.current_output: GenerationOutputV3 | None = None
        self.is_final_render: bool = False
        self.current_mode = "student"
        self.teacher_private_key_path: Path | None = None
        self.ui_theme_profile_path = GENERATOR_THEME_PROFILE
        self.theme_mode = load_theme_mode(self.ui_theme_profile_path, default="light")
        self._did_intro_animation = False
        self._animations: list[QPropertyAnimation] = []
        self._is_validating_composition = False
        self._loaded_preset_context: dict[str, Any] = {}

        self.setWindowTitle("Материаловедческий генератор шлифов V3")
        self._build_ui()
        self._apply_style(self.theme_mode)
        self._load_presets()
        self._init_defaults()

        # Инициализация управления окнами (вместо self.resize)
        window_state_manager = WindowStateManager("generator")
        self.setup_window_modes(window_state_manager)

    def closeEvent(self, event) -> None:
        if hasattr(self, "state_manager"):
            self.state_manager.save_state(self)
        super().closeEvent(event)

    def _find_reference_profiles(self) -> list[str]:
        root = self.profiles_dir / "reference_profiles"
        if not root.exists():
            return []
        items: list[str] = []
        for path in sorted(root.glob("*.json")):
            items.append(path.stem)
        return items

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QWidget()
        header.setObjectName("appHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 8, 10, 8)
        header_layout.setSpacing(12)
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
        header_layout.addWidget(self.header_logo_label)
        self.brand_title_label = QLabel("Материаловедческий комплекс РТУ МИРЭА")
        self.brand_title_label.setObjectName("headerBrandPrimary")
        self.brand_subtitle_label = QLabel("Генератор виртуальных шлифов V3")
        self.brand_subtitle_label.setObjectName("headerBrandSecondary")
        title_wrap = QVBoxLayout()
        title_wrap.setContentsMargins(0, 0, 0, 0)
        title_wrap.setSpacing(1)
        title_wrap.addWidget(self.brand_title_label)
        title_wrap.addWidget(self.brand_subtitle_label)
        title_widget = QWidget()
        title_widget.setObjectName("headerTitleWidget")
        title_widget.setLayout(title_wrap)
        header_layout.addWidget(title_widget, stretch=1)
        self.theme_mode_combo = QComboBox()
        self.theme_mode_combo.addItem("Светлая", "light")
        self.theme_mode_combo.addItem("Тёмная", "dark")
        theme_idx = self.theme_mode_combo.findData(str(self.theme_mode))
        if theme_idx >= 0:
            self.theme_mode_combo.setCurrentIndex(theme_idx)
        self.theme_mode_combo.currentIndexChanged.connect(self._on_theme_mode_changed)
        theme_label = QLabel("Тема")
        theme_label.setObjectName("headerThemeLabel")
        self.mode_label = QLabel("👨‍🎓 СТУДЕНТ")
        self.mode_label.setObjectName("statusPill")
        self.mode_switch_btn = QPushButton("Режим преподавателя")
        self.mode_switch_btn.setObjectName("secondaryCta")
        self.mode_switch_btn.clicked.connect(self._switch_mode)
        header_layout.addWidget(theme_label)
        header_layout.addWidget(self.theme_mode_combo)
        header_layout.addWidget(self.mode_label)
        header_layout.addWidget(self.mode_switch_btn)
        layout.addWidget(header)

        self.main_split = QSplitter(Qt.Orientation.Horizontal)
        self.main_split.setHandleWidth(8)
        layout.addWidget(self.main_split, stretch=1)

        left_nav = QWidget()
        left_nav.setObjectName("leftNavCard")
        left_nav.setMinimumWidth(220)
        left_nav.setMaximumWidth(300)
        left_nav_layout = QVBoxLayout(left_nav)
        left_nav_layout.setContentsMargins(10, 10, 10, 10)
        left_nav_layout.setSpacing(8)
        left_nav_layout.addWidget(QLabel("Стадии генерации"))
        self.step_list = QListWidget()
        self.step_list.setObjectName("stepList")
        self.step_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for name in STEP_NAMES:
            self.step_list.addItem(QListWidgetItem(name))
        self.step_list.setCurrentRow(0)
        self.step_list.currentRowChanged.connect(self._on_step_changed)
        left_nav_layout.addWidget(self.step_list, stretch=1)
        self.step_status_label = QLabel("Шаг 1 из 6")
        self.step_status_label.setObjectName("stepStatusLabel")
        left_nav_layout.addWidget(self.step_status_label)
        self.step_progress_bar = QProgressBar()
        self.step_progress_bar.setObjectName("stepProgressBar")
        self.step_progress_bar.setTextVisible(True)
        self.step_progress_bar.setMinimum(1)
        self.step_progress_bar.setValue(1)
        left_nav_layout.addWidget(self.step_progress_bar)
        self.main_split.addWidget(left_nav)

        center = QWidget()
        center.setObjectName("centerCard")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(10, 10, 10, 10)
        center_layout.setSpacing(10)
        self.step_stack = QStackedWidget()
        center_layout.addWidget(self.step_stack, stretch=1)
        center_layout.addWidget(self._build_footer_nav())
        self.main_split.addWidget(center)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(360)
        right = QWidget()
        right.setObjectName("rightCard")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        self.preview_scene = QGraphicsScene(self)
        self.preview_view = ZoomView(allow_drag=True)
        self.preview_view.setScene(self.preview_scene)
        self.preview_view.setMinimumHeight(300)
        preview_box = QGroupBox("Предпросмотр структуры")
        preview_box.setObjectName("previewCard")
        preview_box.setMinimumHeight(360)
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.addWidget(self.preview_view, stretch=1)
        right_layout.addWidget(preview_box, stretch=8)

        # Галерея промежуточных рендеров
        self.intermediate_renders_group = QGroupBox(
            "Промежуточные рендеры (эволюция структуры)"
        )
        self.intermediate_renders_group.setVisible(False)
        self.intermediate_renders_group.setMinimumHeight(200)
        self.intermediate_renders_group.setMaximumHeight(240)
        intermediate_layout = QVBoxLayout(self.intermediate_renders_group)
        self.intermediate_scroll = QScrollArea()
        self.intermediate_scroll.setWidgetResizable(True)
        self.intermediate_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.intermediate_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.intermediate_container = QWidget()
        self.intermediate_container_layout = QHBoxLayout(self.intermediate_container)
        self.intermediate_container_layout.setSpacing(8)
        self.intermediate_container_layout.setContentsMargins(4, 4, 4, 4)
        self.intermediate_scroll.setWidget(self.intermediate_container)
        intermediate_layout.addWidget(self.intermediate_scroll)
        right_layout.addWidget(self.intermediate_renders_group, stretch=2)

        info_box = QGroupBox("Информация по кадру")
        info_box.setObjectName("infoCard")
        info_box.setMinimumHeight(180)
        info_layout = QVBoxLayout(info_box)
        self.info_text = QPlainTextEdit()
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        right_layout.addWidget(info_box, stretch=4)

        right_scroll.setWidget(right)
        self.main_split.addWidget(right_scroll)
        self.main_split.setStretchFactor(0, 0)
        self.main_split.setStretchFactor(1, 2)
        self.main_split.setStretchFactor(2, 3)
        self._apply_responsive_layout()

        self.step_stack.addWidget(self._wrap_scroll(self._build_step_composition()))
        self.step_stack.addWidget(self._wrap_scroll(self._build_step_process()))
        self.step_stack.addWidget(self._wrap_scroll(self._build_step_prep()))
        self.step_stack.addWidget(self._wrap_scroll(self._build_step_etch()))
        self.step_stack.addWidget(self._wrap_scroll(self._build_step_synthesis()))
        self.step_stack.addWidget(self._wrap_scroll(self._build_step_qc_export()))
        self.step_progress_bar.setMaximum(max(1, self.step_stack.count()))
        self._update_step_status()

    def _wrap_scroll(self, content: QWidget) -> QWidget:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(content)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        return area

    def _section_header(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionHeader")
        return label

    def _build_footer_nav(self) -> QWidget:
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.btn_prev_step = QPushButton("Назад")
        self.btn_next_step = QPushButton("Далее")
        self.btn_preview = QPushButton("Предпросмотр")
        self.btn_render_final = QPushButton("Финальный рендер")
        self.btn_export_package = QPushButton("Экспорт пакета ЛР")
        self.btn_prev_step.setObjectName("secondaryCta")
        self.btn_next_step.setObjectName("secondaryCta")
        self.btn_preview.setObjectName("secondaryCta")
        self.btn_render_final.setObjectName("primaryCta")
        self.btn_export_package.setObjectName("accentCta")
        self.btn_preview.setMinimumHeight(34)
        self.btn_render_final.setMinimumHeight(34)
        self.btn_export_package.setMinimumHeight(34)
        self.btn_export_package.setEnabled(False)
        style = self.style()
        self.btn_prev_step.setIcon(
            style.standardIcon(QStyle.StandardPixmap.SP_ArrowLeft)
        )
        self.btn_next_step.setIcon(
            style.standardIcon(QStyle.StandardPixmap.SP_ArrowRight)
        )
        self.btn_preview.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.btn_render_final.setIcon(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        )
        self.btn_export_package.setIcon(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        )

        self.btn_prev_step.clicked.connect(self._go_prev_step)
        self.btn_next_step.clicked.connect(self._go_next_step)
        self.btn_preview.clicked.connect(self._generate_preview)
        self.btn_render_final.clicked.connect(self._generate_final)
        self.btn_export_package.clicked.connect(self._export_lab_package)

        lay.addWidget(self.btn_prev_step)
        lay.addWidget(self.btn_next_step)
        lay.addStretch(1)
        lay.addWidget(self.btn_preview)
        lay.addWidget(self.btn_render_final)
        lay.addWidget(self.btn_export_package)
        return bar

    def _build_step_composition(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)
        layout.addWidget(self._section_header("Исходные параметры образца"))

        top_grid = QGridLayout()
        self.sample_id_edit = QLineEdit("sample_v3")
        self.preset_combo = QComboBox()
        self.preset_combo.view().setTextElideMode(Qt.TextElideMode.ElideRight)
        self.btn_load_preset = QPushButton("Загрузить пресет")
        self.btn_load_preset.clicked.connect(self._load_selected_preset)
        self.strict_validation_check = QCheckBox("Строгая валидация (strict)")
        self.strict_validation_check.setChecked(True)
        self.system_hint_combo = QComboBox()
        self.system_hint_combo.addItem("Автоопределение", "")
        for sys_name in TRAINING_SYSTEM_OPTIONS:
            self.system_hint_combo.addItem(
                _label_ru(sys_name, SYSTEM_LABELS_RU), sys_name
            )

        top_grid.addWidget(QLabel("ID образца"), 0, 0)
        top_grid.addWidget(self.sample_id_edit, 0, 1)
        top_grid.addWidget(QLabel("Пресет V3"), 1, 0)
        top_grid.addWidget(self.preset_combo, 1, 1)
        top_grid.addWidget(self.btn_load_preset, 1, 2)
        top_grid.addWidget(QLabel("Подсказка системы"), 2, 0)
        top_grid.addWidget(self.system_hint_combo, 2, 1)
        top_grid.addWidget(self.strict_validation_check, 2, 2)
        layout.addLayout(top_grid)

        comp_box = QGroupBox("Состав по элементам, мас.%")
        comp_layout = QVBoxLayout(comp_box)
        self.composition_table = QTableWidget(0, 2)
        self.composition_table.setHorizontalHeaderLabels(["Элемент", "мас.%"])
        head = self.composition_table.horizontalHeader()
        head.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        head.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.composition_table.verticalHeader().setVisible(False)
        self.composition_table.itemChanged.connect(self._validate_composition_realtime)
        comp_layout.addWidget(self.composition_table)

        comp_actions = QGridLayout()
        self.btn_comp_add = QPushButton("Добавить элемент")
        self.btn_comp_remove = QPushButton("Удалить строку")
        self.btn_comp_normalize = QPushButton("Нормализовать до 100%")
        self.btn_comp_import = QPushButton("Импорт JSON")
        self.btn_comp_export = QPushButton("Экспорт JSON")
        self.btn_comp_add.clicked.connect(self._add_composition_row)
        self.btn_comp_remove.clicked.connect(self._remove_composition_row)
        self.btn_comp_normalize.clicked.connect(self._normalize_composition_rows)
        self.btn_comp_import.clicked.connect(self._import_composition_json)
        self.btn_comp_export.clicked.connect(self._export_composition_json)
        comp_actions.addWidget(self.btn_comp_add, 0, 0)
        comp_actions.addWidget(self.btn_comp_remove, 0, 1)
        comp_actions.addWidget(self.btn_comp_normalize, 0, 2)
        comp_actions.addWidget(self.btn_comp_import, 1, 0)
        comp_actions.addWidget(self.btn_comp_export, 1, 1)
        comp_layout.addLayout(comp_actions)
        layout.addWidget(comp_box)

        coverage_box = QGroupBox("Определение системы")
        coverage_layout = QHBoxLayout(coverage_box)
        self.coverage_label = QLabel("Статус не проверен")
        self.btn_check_coverage = QPushButton("Проверить систему")
        self.btn_check_coverage.clicked.connect(self._check_phase_model_resolution)
        coverage_layout.addWidget(self.coverage_label, stretch=1)
        coverage_layout.addWidget(self.btn_check_coverage)
        layout.addWidget(coverage_box)

        layout.addStretch(1)
        return page

    def _build_step_process(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)
        layout.addWidget(self._section_header("Термопрограмма и режимы охлаждения"))
        layout.addWidget(self._build_thermal_widget(), stretch=1)

        val_box = QGroupBox("Проверка термопрограммы")
        val_box.setMinimumHeight(220)
        val_layout = QVBoxLayout(val_box)
        self.route_validation_text = QPlainTextEdit()
        self.route_validation_text.setReadOnly(True)
        self.route_validation_text.setMinimumHeight(140)
        self.btn_validate_route = QPushButton("Проверить термопрограмму")
        self.btn_validate_route.clicked.connect(self._validate_thermal_ui)
        val_layout.addWidget(self.route_validation_text)
        val_layout.addWidget(self.btn_validate_route)
        layout.addWidget(val_box, stretch=1)
        return page

    def _build_thermal_widget(self) -> QWidget:
        box = QGroupBox("Термопрограмма: точки нагрев-охлаждение")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        top = QGridLayout()
        self.thermal_name_edit = QLineEdit("Термопрограмма")
        self.thermal_notes_edit = QLineEdit("")
        top.addWidget(QLabel("Имя"), 0, 0)
        top.addWidget(self.thermal_name_edit, 0, 1)
        top.addWidget(QLabel("Примечание"), 1, 0)
        top.addWidget(self.thermal_notes_edit, 1, 1)
        layout.addLayout(top)

        self.thermal_points_table = QTableWidget(0, 7)
        self.thermal_points_table.setHorizontalHeaderLabels(
            [
                "Время, с",
                "Температура, °C",
                "Метка",
                "Фикс.",
                "Переход",
                "Кривизна",
                "Среда сегмента",
            ]
        )
        self.thermal_points_table.setMinimumHeight(190)
        t_head = self.thermal_points_table.horizontalHeader()
        t_head.setMinimumSectionSize(74)
        t_head.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        t_head.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        t_head.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        t_head.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        t_head.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)
        t_head.setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
        t_head.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.thermal_points_table.setColumnWidth(0, 120)
        self.thermal_points_table.setColumnWidth(1, 130)
        self.thermal_points_table.setColumnWidth(3, 70)
        self.thermal_points_table.setColumnWidth(4, 125)
        self.thermal_points_table.setColumnWidth(5, 110)
        self.thermal_points_table.verticalHeader().setVisible(False)
        self.thermal_points_table.itemChanged.connect(self._refresh_thermal_plot)
        layout.addWidget(self.thermal_points_table, stretch=1)

        transition_defaults = QGroupBox("Переходы по умолчанию для новых сегментов")
        transition_defaults_form = QFormLayout(transition_defaults)
        self.default_transition_model_combo = QComboBox()
        for code, label in THERMAL_TRANSITION_MODEL_OPTIONS:
            self.default_transition_model_combo.addItem(label, code)
        self.default_transition_curvature_spin = QDoubleSpinBox()
        self.default_transition_curvature_spin.setRange(0.15, 12.0)
        self.default_transition_curvature_spin.setSingleStep(0.1)
        self.default_transition_curvature_spin.setValue(1.0)
        self.default_transition_curvature_spin.setToolTip(
            "Кривизна перехода между точками термопрограммы.\n"
            "1.0 = линейный переход\n"
            ">1.0 = выпуклая кривая (быстрое начало, медленный конец)\n"
            "<1.0 = вогнутая кривая (медленное начало, быстрый конец)\n"
            "Типичный диапазон: 0.5-2.0"
        )
        self.default_transition_segment_medium_combo = QComboBox()
        for code, label in SEGMENT_MEDIUM_OPTIONS:
            self.default_transition_segment_medium_combo.addItem(label, code)
        self.auto_transition_by_medium_check = QCheckBox("Авто от среды")
        self.auto_transition_by_medium_check.setChecked(True)
        self.default_transition_model_combo.currentIndexChanged.connect(
            self._on_transition_defaults_changed
        )
        self.default_transition_curvature_spin.valueChanged.connect(
            self._on_transition_defaults_changed
        )
        self.default_transition_segment_medium_combo.currentIndexChanged.connect(
            self._on_transition_defaults_changed
        )
        self.auto_transition_by_medium_check.stateChanged.connect(
            self._on_transition_defaults_changed
        )
        transition_defaults_form.addRow(
            "Модель перехода", self.default_transition_model_combo
        )
        transition_defaults_form.addRow(
            "Кривизна", self.default_transition_curvature_spin
        )
        transition_defaults_form.addRow(
            "Среда сегмента", self.default_transition_segment_medium_combo
        )
        transition_defaults_form.addRow("", self.auto_transition_by_medium_check)
        layout.addWidget(transition_defaults)

        actions = QGridLayout()
        self.btn_thermal_add = QPushButton("Добавить точку")
        self.btn_thermal_remove = QPushButton("Удалить точку")
        self.btn_thermal_default = QPushButton("Сбросить шаблон")
        self.btn_curve_png = QPushButton("Сохранить график PNG")
        self.btn_curve_png_hd = QPushButton("Сохранить график HD")
        self.btn_curve_csv = QPushButton("Сохранить точки CSV")
        self.btn_thermal_add.clicked.connect(lambda: self._add_thermal_point_row(None))
        self.btn_thermal_remove.clicked.connect(self._remove_thermal_point_row)
        self.btn_thermal_default.clicked.connect(self._set_default_thermal_program)
        self.btn_curve_png.clicked.connect(self._export_thermal_curve_png)
        self.btn_curve_png_hd.clicked.connect(self._export_thermal_curve_png_hd)
        self.btn_curve_csv.clicked.connect(self._export_thermal_curve_csv)
        self.btn_curve_png_hd.setToolTip(
            "Экспорт графика в высоком разрешении (300 DPI, 3000x2000 px)"
        )
        actions.addWidget(self.btn_thermal_add, 0, 0)
        actions.addWidget(self.btn_thermal_remove, 0, 1)
        actions.addWidget(self.btn_thermal_default, 0, 2)
        actions.addWidget(self.btn_curve_png, 1, 0)
        actions.addWidget(self.btn_curve_png_hd, 1, 1)
        actions.addWidget(self.btn_curve_csv, 1, 2)
        layout.addLayout(actions)

        plot_box = QGroupBox("График T(t)")
        plot_box.setMinimumHeight(380)
        plot_layout = QVBoxLayout(plot_box)
        if pg is not None:
            self.thermal_plot_widget = pg.PlotWidget()
            self.thermal_plot_widget.setMinimumHeight(320)
            self.thermal_plot_widget.setLabel("left", "Температура, °C")
            self.thermal_plot_widget.setLabel("bottom", "Время, c")
            self.thermal_plot_widget.showGrid(x=True, y=True, alpha=0.25)
            plot_layout.addWidget(self.thermal_plot_widget, stretch=1)
        else:
            self.thermal_plot_widget = None
            self.thermal_plot_placeholder = QLabel(
                "График термообработки недоступен.\n\n"
                "Данные можно экспортировать в CSV для анализа\n"
                "в других программах (Excel, Origin, etc.)."
            )
            self.thermal_plot_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.thermal_plot_placeholder.setStyleSheet(
                f"color: {status_color(self.theme_mode, 'warning')}; font-weight: 600;"
            )
            plot_layout.addWidget(self.thermal_plot_placeholder)
        layout.addWidget(plot_box, stretch=1)

        # Чекбокс для генерации промежуточных рендеров
        self.generate_intermediate_renders_check = QCheckBox(
            "Генерировать промежуточные рендеры для каждой точки"
        )
        self.generate_intermediate_renders_check.setChecked(False)
        self.generate_intermediate_renders_check.setToolTip(
            "Создать рендер микроструктуры для каждой точки термопрограммы.\n"
            "Полезно для ЛР3 (закалка) и ЛР4 (отпуск) для анализа эволюции структуры.\n"
            "ВНИМАНИЕ: Увеличивает время генерации пропорционально количеству точек."
        )
        layout.addWidget(self.generate_intermediate_renders_check)

        quench_box = QGroupBox("Параметры закалки")
        quench_form = QFormLayout(quench_box)
        self.quench_medium_combo = QComboBox()
        for code, label in QUENCH_MEDIA_OPTIONS:
            self.quench_medium_combo.addItem(label, code)
        self.quench_medium_combo.currentIndexChanged.connect(
            self._on_quench_medium_changed
        )
        self.quench_time_spin = QDoubleSpinBox()
        self.quench_time_spin.setRange(0.0, 36000.0)
        self.quench_time_spin.setValue(0.0)
        self.quench_time_spin.setSuffix(" c")
        self.quench_time_spin.valueChanged.connect(self._refresh_thermal_plot)
        self.quench_bath_temp_spin = QDoubleSpinBox()
        self.quench_bath_temp_spin.setRange(-50.0, 300.0)
        self.quench_bath_temp_spin.setValue(20.0)
        self.quench_bath_temp_spin.valueChanged.connect(self._refresh_thermal_plot)
        self.quench_sample_temp_spin = QDoubleSpinBox()
        self.quench_sample_temp_spin.setRange(20.0, 1800.0)
        self.quench_sample_temp_spin.setValue(840.0)
        self.quench_sample_temp_spin.valueChanged.connect(self._refresh_thermal_plot)
        self.quench_custom_name = QLineEdit("")
        self.quench_severity_spin = QDoubleSpinBox()
        self.quench_severity_spin.setRange(0.05, 3.0)
        self.quench_severity_spin.setSingleStep(0.05)
        self.quench_severity_spin.setValue(1.0)
        self.quench_severity_spin.setToolTip(
            "Фактор интенсивности закалки (H-фактор Гроссмана).\n"
            "Характеризует скорость теплоотвода от поверхности образца.\n\n"
            "Типичные значения:\n"
            "0.1-0.3 = медленное охлаждение (воздух, печь)\n"
            "0.5-1.0 = умеренное охлаждение (масло)\n"
            "1.0-2.0 = интенсивное охлаждение (вода)\n"
            "2.0-3.0 = очень интенсивное (рассол, полимеры)\n\n"
            "Влияет на глубину прокаливаемости и структуру."
        )
        quench_form.addRow("Среда", self.quench_medium_combo)
        quench_form.addRow("Время закалки", self.quench_time_spin)
        quench_form.addRow("Температура среды", self.quench_bath_temp_spin)
        quench_form.addRow("Температура образца", self.quench_sample_temp_spin)
        quench_form.addRow("Пользовательская среда", self.quench_custom_name)
        quench_form.addRow("Фактор интенсивности", self.quench_severity_spin)
        layout.addWidget(quench_box)

        sampling_box = QGroupBox("Дискретизация кривой")
        sampling_form = QFormLayout(sampling_box)
        self.thermal_sampling_combo = QComboBox()
        self.thermal_sampling_combo.addItem("Поградусно", "per_degree")
        self.thermal_sampling_combo.addItem("Только опорные точки", "points")
        self.thermal_sampling_combo.currentIndexChanged.connect(
            self._refresh_thermal_plot
        )
        self.thermal_degree_step_spin = QDoubleSpinBox()
        self.thermal_degree_step_spin.setRange(0.1, 50.0)
        self.thermal_degree_step_spin.setValue(1.0)
        self.thermal_degree_step_spin.valueChanged.connect(self._refresh_thermal_plot)
        self.thermal_max_frames_spin = QSpinBox()
        self.thermal_max_frames_spin.setRange(2, 4000)
        self.thermal_max_frames_spin.setValue(320)
        self.thermal_max_frames_spin.valueChanged.connect(self._refresh_thermal_plot)
        sampling_form.addRow("Режим", self.thermal_sampling_combo)
        sampling_form.addRow("Шаг по температуре, °C", self.thermal_degree_step_spin)
        sampling_form.addRow("Макс. кадров", self.thermal_max_frames_spin)
        layout.addWidget(sampling_box)
        return box

    def _build_step_prep(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)
        layout.addWidget(self._section_header("Маршрут подготовки шлифа"))

        top = QHBoxLayout()
        self.prep_template_combo = QComboBox()
        self.prep_template_combo.addItem("Выберите шаблон подготовки", "")
        for name in sorted(self._prep_templates.keys()):
            self.prep_template_combo.addItem(
                _label_ru(name, PREP_TEMPLATE_LABELS_RU), name
            )
        self.btn_apply_prep_template = QPushButton("Применить шаблон")
        self.btn_apply_prep_template.clicked.connect(self._apply_prep_template)
        top.addWidget(self.prep_template_combo)
        top.addWidget(self.btn_apply_prep_template)
        layout.addLayout(top)

        self.prep_table = QTableWidget(0, 14)
        self.prep_table.setHorizontalHeaderLabels(
            [
                "Метод",
                "Время, c",
                "Абразив, мкм",
                "Нагрузка, N",
                "RPM",
                "Охлаждение",
                "Угол, °",
                "Профиль нагрузки",
                "Ткань",
                "Суспензия",
                "Смазка, мл/мин",
                "Осцил., Гц",
                "Траектория",
                "Комментарий",
            ]
        )
        prep_head = self.prep_table.horizontalHeader()
        for col in range(13):
            prep_head.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        prep_head.setSectionResizeMode(13, QHeaderView.ResizeMode.Stretch)
        self.prep_table.verticalHeader().setVisible(False)
        layout.addWidget(self.prep_table, stretch=1)

        prep_actions = QGridLayout()
        self.btn_prep_add = QPushButton("Добавить шаг")
        self.btn_prep_remove = QPushButton("Удалить шаг")
        self.btn_prep_up = QPushButton("Вверх")
        self.btn_prep_down = QPushButton("Вниз")
        self.btn_prep_add.clicked.connect(lambda: self._add_prep_row(None))
        self.btn_prep_remove.clicked.connect(self._remove_prep_row)
        self.btn_prep_up.clicked.connect(lambda: self._move_prep_row(-1))
        self.btn_prep_down.clicked.connect(lambda: self._move_prep_row(1))
        prep_actions.addWidget(self.btn_prep_add, 0, 0)
        prep_actions.addWidget(self.btn_prep_remove, 0, 1)
        prep_actions.addWidget(self.btn_prep_up, 0, 2)
        prep_actions.addWidget(self.btn_prep_down, 0, 3)
        layout.addLayout(prep_actions)

        prefs_box = QGroupBox("Итоговые параметры подготовки")
        prefs_form = QFormLayout(prefs_box)
        self.prep_rough_spin = QDoubleSpinBox()
        self.prep_rough_spin.setRange(0.001, 10.0)
        self.prep_rough_spin.setDecimals(4)
        self.prep_rough_spin.setValue(0.05)
        self.prep_rough_spin.setToolTip(
            "Целевая шероховатость поверхности после подготовки (Ra, мкм).\n\n"
            "Типичные значения:\n"
            "0.01-0.05 мкм = зеркальная полировка (финишная)\n"
            "0.05-0.2 мкм = тонкая полировка\n"
            "0.2-1.0 мкм = грубая полировка\n"
            ">1.0 мкм = шлифовка\n\n"
            "Влияет на качество выявления структуры при травлении."
        )
        self.prep_relief_combo = QComboBox()
        self.prep_relief_combo.addItem("Связь с твердостью", "hardness_coupled")
        self.prep_relief_combo.addItem("Связь с фазами", "phase_coupled")
        self.prep_contam_spin = QDoubleSpinBox()
        self.prep_contam_spin.setRange(0.0, 1.0)
        self.prep_contam_spin.setSingleStep(0.01)
        self.prep_contam_spin.setValue(0.02)
        self.prep_contam_spin.setToolTip(
            "Уровень контаминации поверхности (0.0-1.0).\n\n"
            "Загрязнения от абразивов, смазок, окислов.\n"
            "0.0 = идеально чистая поверхность\n"
            "0.01-0.05 = типичная лабораторная подготовка\n"
            "0.1-0.3 = повышенная контаминация\n"
            ">0.5 = сильное загрязнение\n\n"
            "Влияет на равномерность травления и артефакты."
        )
        prefs_form.addRow("Целевая шероховатость, мкм", self.prep_rough_spin)
        prefs_form.addRow("Режим рельефа", self.prep_relief_combo)
        prefs_form.addRow("Контаминация", self.prep_contam_spin)
        layout.addWidget(prefs_box)
        return page

    def _build_step_etch(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)
        layout.addWidget(self._section_header("Параметры травления и концентрации"))

        top = QHBoxLayout()
        self.etch_profile_combo = QComboBox()
        self.etch_profile_combo.addItem("Выберите профиль травления", "")
        for name in sorted(self._etch_profiles.keys()):
            self.etch_profile_combo.addItem(
                _label_ru(name, ETCH_PROFILE_LABELS_RU), name
            )
        self.btn_apply_etch_profile = QPushButton("Применить профиль")
        self.btn_apply_etch_profile.clicked.connect(self._apply_etch_profile)
        top.addWidget(self.etch_profile_combo)
        top.addWidget(self.btn_apply_etch_profile)
        layout.addLayout(top)

        form_box = QGroupBox("Параметры травления")
        form = QFormLayout(form_box)
        self.etch_reagent_combo = QComboBox()
        for reagent in ETCH_REAGENTS:
            self.etch_reagent_combo.addItem(
                _label_ru(reagent, ETCH_REAGENT_LABELS_RU), reagent
            )
        self.etch_time_spin = QDoubleSpinBox()
        self.etch_time_spin.setRange(0.1, 600.0)
        self.etch_time_spin.setValue(8.0)
        self.etch_time_spin.valueChanged.connect(self._update_etch_risk)
        self.etch_temp_spin = QDoubleSpinBox()
        self.etch_temp_spin.setRange(-20.0, 120.0)
        self.etch_temp_spin.setValue(22.0)
        self.etch_temp_spin.valueChanged.connect(self._update_etch_risk)
        self.etch_agitation_combo = QComboBox()
        for code, label in AGITATION_OPTIONS:
            self.etch_agitation_combo.addItem(label, code)
        self.etch_overetch_spin = QDoubleSpinBox()
        self.etch_overetch_spin.setRange(0.2, 3.0)
        self.etch_overetch_spin.setSingleStep(0.05)
        self.etch_overetch_spin.setValue(1.0)
        self.etch_overetch_spin.valueChanged.connect(self._update_etch_risk)
        self.etch_conc_unit_combo = QComboBox()
        self.etch_conc_unit_combo.addItem("wt.%", "wt_pct")
        self.etch_conc_unit_combo.addItem("mol/L", "mol_l")
        self.etch_conc_value_spin = QDoubleSpinBox()
        self.etch_conc_value_spin.setRange(0.0001, 100.0)
        self.etch_conc_value_spin.setDecimals(4)
        self.etch_conc_value_spin.setValue(2.0)
        self.etch_conc_value_spin.valueChanged.connect(self._update_etch_risk)
        self.etch_conc_value_spin.setToolTip(
            "Концентрация травителя (wt.% или mol/L).\n\n"
            "Влияет на скорость и селективность травления.\n"
            "Низкая концентрация = медленное, контролируемое травление\n"
            "Высокая концентрация = быстрое, агрессивное травление\n\n"
            "Типичные диапазоны зависят от реактива:\n"
            "Ниталь (HNO₃): 2-5 wt.%\n"
            "Пикраль: 4-10 wt.%\n\n"
            "Проверяйте статус диапазона ниже."
        )
        self.etch_conc_equiv_label = QLabel("Эквивалент: 0.000 mol/L")
        self.etch_conc_status_label = QLabel("Диапазон: не проверено")
        self.etch_conc_unit_combo.currentIndexChanged.connect(self._update_etch_risk)
        form.addRow("Реактив", self.etch_reagent_combo)
        form.addRow("Время, с", self.etch_time_spin)
        form.addRow("Температура, °C", self.etch_temp_spin)
        form.addRow("Перемешивание", self.etch_agitation_combo)
        form.addRow("Коэф. перетрава", self.etch_overetch_spin)
        form.addRow("Единица концентрации", self.etch_conc_unit_combo)
        form.addRow("Концентрация", self.etch_conc_value_spin)
        form.addRow("Эквивалент", self.etch_conc_equiv_label)
        form.addRow("Статус диапазона", self.etch_conc_status_label)
        layout.addWidget(form_box)

        self.etch_risk_label = QLabel("Риск травления: в норме")
        layout.addWidget(self.etch_risk_label)
        layout.addStretch(1)
        return page

    def _build_step_synthesis(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)
        layout.addWidget(
            self._section_header("Синтез микроструктуры и фазовый контроль")
        )

        synth_box = QGroupBox("Параметры синтеза микроструктуры")
        synth_grid = QGridLayout(synth_box)
        self.visual_standard_combo = QComboBox()
        self.visual_standard_combo.addItem("Учебник (ч/б)", "textbook_bw")
        self.visual_standard_combo.addItem("Пользовательский", "custom")
        self.visual_standard_combo.currentIndexChanged.connect(
            self._on_visual_standard_changed
        )
        self.lab_template_combo = QComboBox()
        self.lab_template_combo.view().setTextElideMode(Qt.TextElideMode.ElideRight)
        self.lab_template_combo.addItem("Выберите шаблон ЛР", "")
        self.lab_template_combo.addItem(LR1_TEMPLATE_GROUP_LABEL, "")
        for label, preset_key in LR1_SAMPLE_LIBRARY_PRESETS:
            _add_combo_item_with_tooltip(
                self.lab_template_combo,
                label,
                preset_key,
                _preset_tooltip(preset_key),
            )
        self.lab_template_combo.addItem(CURRICULUM_TEMPLATE_GROUP_LABEL, "")
        for label, preset_key in LAB_TEMPLATE_PRESETS:
            _add_combo_item_with_tooltip(
                self.lab_template_combo,
                _preset_visible_label(preset_key),
                preset_key,
                _preset_tooltip(preset_key),
            )
        self.lab_template_combo.addItem(STEEL_LIBRARY_GROUP_LABEL, "")
        for label, preset_key in STEEL_SAMPLE_LIBRARY_PRESETS:
            _add_combo_item_with_tooltip(
                self.lab_template_combo,
                _preset_visible_label(preset_key),
                preset_key,
                _preset_tooltip(preset_key),
            )
        self.lab_template_combo.addItem(CAST_IRON_LIBRARY_GROUP_LABEL, "")
        for label, preset_key in CAST_IRON_SAMPLE_LIBRARY_PRESETS:
            _add_combo_item_with_tooltip(
                self.lab_template_combo,
                _preset_visible_label(preset_key),
                preset_key,
                _preset_tooltip(preset_key),
            )
        self.lab_template_combo.addItem(RESEARCH_OPTICS_GROUP_LABEL, "")
        for label, preset_key in LAB_RESEARCH_TEMPLATE_PRESETS:
            _add_combo_item_with_tooltip(
                self.lab_template_combo,
                label,
                preset_key,
                _preset_tooltip(preset_key),
            )
        self.btn_apply_lab_template = QPushButton("Применить шаблон ЛР")
        self.btn_apply_lab_template.clicked.connect(self._apply_lab_template)
        self.synth_profile_combo = QComboBox()
        for profile in sorted(self._synth_profiles.keys()):
            self.synth_profile_combo.addItem(
                _label_ru(profile, SYNTH_PROFILE_LABELS_RU), profile
            )
        if self.synth_profile_combo.count() == 0:
            self.synth_profile_combo.addItem(
                _label_ru("textbook_steel_bw", SYNTH_PROFILE_LABELS_RU),
                "textbook_steel_bw",
            )

        self.synth_topology_combo = QComboBox()
        for key, label in TOPOLOGY_OPTIONS:
            self.synth_topology_combo.addItem(label, key)
        self.synth_topology_combo.currentIndexChanged.connect(
            self._on_topology_mode_changed
        )
        self.synth_system_generator_combo = QComboBox()
        for key, label in SYSTEM_GENERATOR_OPTIONS:
            self.synth_system_generator_combo.addItem(label, key)
        self.synth_contrast_spin = QDoubleSpinBox()
        self.synth_contrast_spin.setRange(0.5, 2.5)
        self.synth_contrast_spin.setSingleStep(0.05)
        self.synth_contrast_spin.setValue(1.2)
        self.synth_contrast_spin.setToolTip(
            "Целевой контраст изображения микроструктуры.\n\n"
            "0.5-0.8 = низкий контраст (слабое травление)\n"
            "1.0 = нормальный контраст\n"
            "1.2-1.5 = повышенный контраст (учебный режим)\n"
            "1.5-2.5 = высокий контраст (максимальная видимость фаз)\n\n"
            "Влияет на различимость границ зерен и фаз."
        )
        self.synth_sharp_spin = QDoubleSpinBox()
        self.synth_sharp_spin.setRange(0.4, 2.5)
        self.synth_sharp_spin.setSingleStep(0.05)
        self.synth_sharp_spin.setValue(1.2)
        self.synth_sharp_spin.setToolTip(
            "Резкость границ между фазами и зернами.\n\n"
            "0.4-0.8 = размытые границы (диффузионные переходы)\n"
            "1.0 = нормальная резкость\n"
            "1.2-1.8 = четкие границы (учебный режим)\n"
            "1.8-2.5 = очень резкие границы (идеализированная структура)\n\n"
            "Влияет на читаемость микроструктуры."
        )
        self.synth_artifact_spin = QDoubleSpinBox()
        self.synth_artifact_spin.setRange(0.0, 1.0)
        self.synth_artifact_spin.setSingleStep(0.01)
        self.synth_artifact_spin.setValue(0.2)
        self.synth_artifact_spin.setToolTip(
            "Уровень артефактов подготовки и травления (0.0-1.0).\n\n"
            "0.0 = идеальное изображение без артефактов\n"
            "0.1-0.3 = реалистичные артефакты (царапины, неравномерность)\n"
            "0.3-0.6 = заметные дефекты подготовки\n"
            "0.6-1.0 = сильные артефакты (плохая подготовка)\n\n"
            "Имитирует реальные условия лабораторной практики."
        )
        self.synth_sensitivity_combo = QComboBox()
        for key, label in SENSITIVITY_OPTIONS:
            self.synth_sensitivity_combo.addItem(label, key)
        self.synth_generation_mode_combo = QComboBox()
        for key, label in GENERATION_MODE_OPTIONS:
            self.synth_generation_mode_combo.addItem(label, key)
        self.synth_phase_emphasis_combo = QComboBox()
        for key, label in PHASE_EMPHASIS_OPTIONS:
            self.synth_phase_emphasis_combo.addItem(label, key)
        self.synth_phase_tolerance_spin = QDoubleSpinBox()
        self.synth_phase_tolerance_spin.setRange(0.0, 50.0)
        self.synth_phase_tolerance_spin.setSingleStep(1.0)
        self.synth_phase_tolerance_spin.setValue(20.0)
        self.synth_phase_tolerance_spin.setSuffix(" %")

        synth_grid.addWidget(QLabel("Визуальный стандарт"), 0, 0)
        synth_grid.addWidget(self.visual_standard_combo, 0, 1)
        synth_grid.addWidget(QLabel("Шаблон ЛР"), 1, 0)
        synth_grid.addWidget(self.lab_template_combo, 1, 1)
        synth_grid.addWidget(self.btn_apply_lab_template, 1, 2)
        synth_grid.addWidget(QLabel("Профиль синтеза"), 2, 0)
        synth_grid.addWidget(self.synth_profile_combo, 2, 1)
        synth_grid.addWidget(QLabel("Топология"), 3, 0)
        synth_grid.addWidget(self.synth_topology_combo, 3, 1)
        synth_grid.addWidget(QLabel("Системный генератор"), 4, 0)
        synth_grid.addWidget(self.synth_system_generator_combo, 4, 1)
        synth_grid.addWidget(QLabel("Контраст"), 5, 0)
        synth_grid.addWidget(self.synth_contrast_spin, 5, 1)
        synth_grid.addWidget(QLabel("Резкость границ"), 6, 0)
        synth_grid.addWidget(self.synth_sharp_spin, 6, 1)
        synth_grid.addWidget(QLabel("Артефакты"), 7, 0)
        synth_grid.addWidget(self.synth_artifact_spin, 7, 1)
        synth_grid.addWidget(QLabel("Чувствительность к составу"), 8, 0)
        synth_grid.addWidget(self.synth_sensitivity_combo, 8, 1)
        synth_grid.addWidget(QLabel("Режим генерации"), 9, 0)
        synth_grid.addWidget(self.synth_generation_mode_combo, 9, 1)
        synth_grid.addWidget(QLabel("Выраженность фаз"), 10, 0)
        synth_grid.addWidget(self.synth_phase_emphasis_combo, 10, 1)
        synth_grid.addWidget(QLabel("Допуск отклонения фаз"), 11, 0)
        synth_grid.addWidget(self.synth_phase_tolerance_spin, 11, 1)
        layout.addWidget(synth_box)

        render_box = QGroupBox("Рендер и виртуальный микроскоп")
        render_grid = QGridLayout(render_box)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        self.seed_spin.setValue(42)
        self.resolution_combo = QComboBox()
        for label, value in RESOLUTION_OPTIONS:
            self.resolution_combo.addItem(label, value)
        self.reference_combo = QComboBox()
        self.reference_combo.addItem("Не использовать", "")
        for rid in self._reference_profiles:
            self.reference_combo.addItem(rid.replace("_", " "), rid)
        self.ms_simulate_check = QCheckBox(
            "Применять микроскоп-эффекты к предпросмотру"
        )
        self.ms_magnification_combo = QComboBox()
        for mag in (100, 200, 400, 600):
            self.ms_magnification_combo.addItem(f"{mag}x", mag)
        self.ms_optical_mode_combo = QComboBox()
        for key, label in OPTICAL_MODE_OPTIONS:
            self.ms_optical_mode_combo.addItem(label, key)
        self.ms_psf_profile_combo = QComboBox()
        for key, label in PSF_PROFILE_OPTIONS:
            self.ms_psf_profile_combo.addItem(label, key)
        self.ms_psf_strength_spin = QDoubleSpinBox()
        self.ms_psf_strength_spin.setRange(0.0, 1.0)
        self.ms_psf_strength_spin.setSingleStep(0.05)
        self.ms_psf_strength_spin.setValue(0.0)
        self.ms_sectioning_shear_spin = QDoubleSpinBox()
        self.ms_sectioning_shear_spin.setRange(0.0, 90.0)
        self.ms_sectioning_shear_spin.setSingleStep(1.0)
        self.ms_sectioning_shear_spin.setValue(35.0)
        self.ms_hybrid_balance_spin = QDoubleSpinBox()
        self.ms_hybrid_balance_spin.setRange(0.0, 1.0)
        self.ms_hybrid_balance_spin.setSingleStep(0.05)
        self.ms_hybrid_balance_spin.setValue(0.5)
        self.ms_focus_spin = QDoubleSpinBox()
        self.ms_focus_spin.setRange(0.2, 1.6)
        self.ms_focus_spin.setSingleStep(0.02)
        self.ms_focus_spin.setValue(0.95)
        self.ms_brightness_spin = QDoubleSpinBox()
        self.ms_brightness_spin.setRange(0.2, 2.5)
        self.ms_brightness_spin.setValue(1.0)
        self.ms_contrast_spin = QDoubleSpinBox()
        self.ms_contrast_spin.setRange(0.2, 2.5)
        self.ms_contrast_spin.setValue(1.1)
        self.ms_noise_spin = QDoubleSpinBox()
        self.ms_noise_spin.setRange(0.0, 20.0)
        self.ms_noise_spin.setValue(1.5)
        self.ms_vignette_spin = QDoubleSpinBox()
        self.ms_vignette_spin.setRange(0.0, 1.0)
        self.ms_vignette_spin.setValue(0.12)
        self.ms_uneven_spin = QDoubleSpinBox()
        self.ms_uneven_spin.setRange(0.0, 1.0)
        self.ms_uneven_spin.setValue(0.08)
        self.ms_dust_check = QCheckBox("Пыль")
        self.ms_scratch_check = QCheckBox("Царапины")

        render_grid.addWidget(QLabel("Сид (seed)"), 0, 0)
        render_grid.addWidget(self.seed_spin, 0, 1)
        render_grid.addWidget(QLabel("Разрешение"), 1, 0)
        render_grid.addWidget(self.resolution_combo, 1, 1)
        render_grid.addWidget(QLabel("Референс-профиль"), 2, 0)
        render_grid.addWidget(self.reference_combo, 2, 1)
        render_grid.addWidget(self.ms_simulate_check, 3, 0, 1, 2)
        render_grid.addWidget(QLabel("Увеличение"), 4, 0)
        render_grid.addWidget(self.ms_magnification_combo, 4, 1)
        render_grid.addWidget(QLabel("Оптический режим"), 5, 0)
        render_grid.addWidget(self.ms_optical_mode_combo, 5, 1)
        render_grid.addWidget(QLabel("PSF профиль"), 6, 0)
        render_grid.addWidget(self.ms_psf_profile_combo, 6, 1)
        render_grid.addWidget(QLabel("PSF сила"), 7, 0)
        render_grid.addWidget(self.ms_psf_strength_spin, 7, 1)
        render_grid.addWidget(QLabel("Sectioning shear"), 8, 0)
        render_grid.addWidget(self.ms_sectioning_shear_spin, 8, 1)
        render_grid.addWidget(QLabel("Hybrid balance"), 9, 0)
        render_grid.addWidget(self.ms_hybrid_balance_spin, 9, 1)
        render_grid.addWidget(QLabel("Фокус"), 10, 0)
        render_grid.addWidget(self.ms_focus_spin, 10, 1)
        render_grid.addWidget(QLabel("Яркость"), 11, 0)
        render_grid.addWidget(self.ms_brightness_spin, 11, 1)
        render_grid.addWidget(QLabel("Контраст"), 12, 0)
        render_grid.addWidget(self.ms_contrast_spin, 12, 1)
        render_grid.addWidget(QLabel("Шум"), 13, 0)
        render_grid.addWidget(self.ms_noise_spin, 13, 1)
        render_grid.addWidget(QLabel("Виньетка"), 14, 0)
        render_grid.addWidget(self.ms_vignette_spin, 14, 1)
        render_grid.addWidget(QLabel("Неравномерность"), 15, 0)
        render_grid.addWidget(self.ms_uneven_spin, 15, 1)
        render_grid.addWidget(self.ms_dust_check, 16, 0)
        render_grid.addWidget(self.ms_scratch_check, 16, 1)
        layout.addWidget(render_box)

        brinell_box = QGroupBox("Твердость по Бринеллю")
        brinell_form = QFormLayout(brinell_box)
        self.brinell_mode_combo = QComboBox()
        self.brinell_mode_combo.addItem("Оценка из структуры", "estimated")
        self.brinell_mode_combo.addItem("Прямой расчет (P, D, d)", "direct")
        self.brinell_p_spin = QDoubleSpinBox()
        self.brinell_p_spin.setRange(1.0, 5000.0)
        self.brinell_p_spin.setValue(187.5)
        self.brinell_d_ball_spin = QDoubleSpinBox()
        self.brinell_d_ball_spin.setRange(0.5, 20.0)
        self.brinell_d_ball_spin.setValue(2.5)
        self.brinell_d_indent_spin = QDoubleSpinBox()
        self.brinell_d_indent_spin.setRange(0.01, 20.0)
        self.brinell_d_indent_spin.setValue(0.9)
        self.btn_brinell_direct = QPushButton("Рассчитать HBW")
        self.btn_brinell_direct.clicked.connect(self._compute_brinell_direct_ui)
        self.brinell_direct_label = QLabel("HBW: -")
        brinell_form.addRow("Режим", self.brinell_mode_combo)
        brinell_form.addRow("P, кгс", self.brinell_p_spin)
        brinell_form.addRow("D, мм", self.brinell_d_ball_spin)
        brinell_form.addRow("d, мм", self.brinell_d_indent_spin)
        brinell_form.addRow(self.btn_brinell_direct, self.brinell_direct_label)
        layout.addWidget(brinell_box)

        phase_box = QGroupBox("Фазовый контроль")
        phase_layout = QVBoxLayout(phase_box)
        phase_form = QFormLayout()
        self.phase_control_mode_combo = QComboBox()
        self.phase_control_mode_combo.addItem(
            "Авто + ручная коррекция", "auto_with_override"
        )
        self.phase_control_mode_combo.addItem("Только авто", "auto_only")
        self.phase_control_mode_combo.addItem("Только ручной", "manual_only")
        phase_form.addRow("Режим фаз", self.phase_control_mode_combo)
        phase_layout.addLayout(phase_form)

        self.manual_phase_table = QTableWidget(0, 2)
        self.manual_phase_table.setHorizontalHeaderLabels(["Фаза", "Доля"])
        phase_head = self.manual_phase_table.horizontalHeader()
        phase_head.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        phase_head.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.manual_phase_table.verticalHeader().setVisible(False)
        self.manual_phase_table.setMinimumHeight(140)
        phase_layout.addWidget(self.manual_phase_table)

        phase_actions = QGridLayout()
        self.btn_phase_add = QPushButton("Добавить фазу")
        self.btn_phase_remove = QPushButton("Удалить фазу")
        self.btn_phase_normalize = QPushButton("Нормализовать")
        self.btn_phase_add.clicked.connect(lambda: self._add_manual_phase_row("", 0.0))
        self.btn_phase_remove.clicked.connect(self._remove_manual_phase_row)
        self.btn_phase_normalize.clicked.connect(self._normalize_manual_phase_rows)
        phase_actions.addWidget(self.btn_phase_add, 0, 0)
        phase_actions.addWidget(self.btn_phase_remove, 0, 1)
        phase_actions.addWidget(self.btn_phase_normalize, 0, 2)
        phase_layout.addLayout(phase_actions)

        blend_wrap = QWidget()
        blend_layout = QHBoxLayout(blend_wrap)
        blend_layout.setContentsMargins(0, 0, 0, 0)
        self.phase_override_slider = QSlider(Qt.Orientation.Horizontal)
        self.phase_override_slider.setRange(0, 100)
        self.phase_override_slider.setValue(35)
        self.phase_override_slider.valueChanged.connect(
            self._sync_phase_override_from_slider
        )
        self.phase_override_spin = QDoubleSpinBox()
        self.phase_override_spin.setRange(0.0, 1.0)
        self.phase_override_spin.setSingleStep(0.01)
        self.phase_override_spin.setValue(0.35)
        self.phase_override_spin.valueChanged.connect(
            self._sync_phase_override_from_spin
        )
        blend_layout.addWidget(self.phase_override_slider, stretch=1)
        blend_layout.addWidget(self.phase_override_spin)
        phase_form.addRow("Вес ручной коррекции", blend_wrap)
        self.allow_custom_fallback_check = QCheckBox(
            "Разрешить резервный режим для пользовательских многокомпонентных"
        )
        self.allow_custom_fallback_check.setChecked(True)
        phase_form.addRow(self.allow_custom_fallback_check)
        self.phase_balance_tolerance_spin = QDoubleSpinBox()
        self.phase_balance_tolerance_spin.setRange(0.0, 50.0)
        self.phase_balance_tolerance_spin.setSingleStep(1.0)
        self.phase_balance_tolerance_spin.setValue(20.0)
        self.phase_balance_tolerance_spin.setSuffix(" %")
        phase_form.addRow("Допуск баланса фаз", self.phase_balance_tolerance_spin)
        layout.addWidget(phase_box)
        layout.addStretch(1)
        return page

    def _build_step_qc_export(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)
        layout.addWidget(self._section_header("Проверка качества и экспорт пакета ЛР"))

        exp_box = QGroupBox("Экспорт пакета виртуальной лабораторной")
        exp_grid = QGridLayout(exp_box)
        self.export_dir_edit = QLineEdit(
            str((Path.cwd() / "examples" / "factory_v3_output").resolve())
        )
        self.btn_export_dir = QPushButton("...")
        self.btn_export_dir.setFixedWidth(36)
        self.btn_export_dir.clicked.connect(self._browse_export_dir)
        self.export_prefix_edit = QLineEdit("lab_sample")
        self.export_masks_check = QCheckBox("Сохранять маски фаз/признаков")
        self.export_masks_check.setChecked(True)
        self.export_prep_maps_check = QCheckBox("Сохранять карты подготовки")
        self.export_prep_maps_check.setChecked(True)
        self.btn_save_request = QPushButton("Сохранить текущий запрос V3")
        self.btn_save_request.clicked.connect(self._save_request_json)
        self.btn_load_request = QPushButton("Загрузить запрос V3")
        self.btn_load_request.clicked.connect(self._load_request_json)

        exp_grid.addWidget(QLabel("Папка экспорта"), 0, 0)
        exp_grid.addWidget(self.export_dir_edit, 0, 1)
        exp_grid.addWidget(self.btn_export_dir, 0, 2)
        exp_grid.addWidget(QLabel("Префикс файлов"), 1, 0)
        exp_grid.addWidget(self.export_prefix_edit, 1, 1)
        exp_grid.addWidget(self.export_masks_check, 2, 0, 1, 2)
        exp_grid.addWidget(self.export_prep_maps_check, 3, 0, 1, 2)
        exp_grid.addWidget(self.btn_save_request, 4, 0)
        exp_grid.addWidget(self.btn_load_request, 4, 1)
        layout.addWidget(exp_box)

        qc_box = QGroupBox("Контроль качества и метаданные")
        qc_layout = QVBoxLayout(qc_box)
        self.textbook_readability_label = QLabel("Учебная читаемость фаз: нет данных")
        self.textbook_readability_label.setStyleSheet(
            f"color: {status_color(self.theme_mode, 'text_secondary')}; font-weight: 600;"
        )
        qc_layout.addWidget(self.textbook_readability_label)
        self.qc_text = QPlainTextEdit()
        self.qc_text.setReadOnly(True)
        qc_layout.addWidget(self.qc_text)
        layout.addWidget(qc_box, stretch=1)
        return page

    def _on_theme_mode_changed(self, *_args: Any) -> None:
        mode = str(self.theme_mode_combo.currentData() or "light").strip().lower()
        self._apply_style(mode)
        try:
            save_theme_mode(self.ui_theme_profile_path, self.theme_mode)
        except Exception:
            pass

    def _switch_mode(self) -> None:
        if self.current_mode == "student":
            self._activate_teacher_mode()
            return
        self._activate_student_mode()

    def _activate_teacher_mode(self) -> None:
        try:
            resolved = activate_teacher_mode_with_prompt(self)
        except Exception as exc:
            QMessageBox.critical(
                self, "Ошибка", f"Не удалось загрузить приватный ключ:\n{exc}"
            )
            return
        if resolved is None:
            return
        self.teacher_private_key_path = resolved
        self.current_mode = "teacher"
        self.mode_label.setText("👨‍🏫 ПРЕПОДАВАТЕЛЬ")
        self.mode_switch_btn.setText("Режим студента")
        if self.current_output is not None:
            self._show_output(self.current_output)

    def _activate_student_mode(self) -> None:
        self.current_mode = "student"
        self.mode_label.setText("👨‍🎓 СТУДЕНТ")
        self.mode_switch_btn.setText("Режим преподавателя")
        if self.current_output is not None:
            self._show_output(self.current_output)

    def _apply_style(self, mode: str | None = None) -> None:
        if mode:
            self.theme_mode = str(mode).strip().lower() or "light"
        self.setStyleSheet(build_qss(self.theme_mode) + self._generator_extra_qss())
        if hasattr(self, "etch_conc_status_label"):
            self._update_etch_risk()
        if hasattr(self, "textbook_readability_label"):
            txt = self.textbook_readability_label.text().lower()
            if "в норме" in txt:
                color = status_color(self.theme_mode, "success")
                weight = 700
            elif "ниже цели" in txt:
                color = status_color(self.theme_mode, "warning")
                weight = 700
            else:
                color = status_color(self.theme_mode, "text_secondary")
                weight = 600
            self.textbook_readability_label.setStyleSheet(
                f"color: {color}; font-weight: {weight};"
            )
        if hasattr(self, "thermal_plot_placeholder"):
            self.thermal_plot_placeholder.setStyleSheet(
                f"color: {status_color(self.theme_mode, 'warning')}; font-weight: 600;"
            )

    def _generator_extra_qss(self) -> str:
        bg = status_color(self.theme_mode, "bg_surface")
        fg = status_color(self.theme_mode, "text_primary")
        border = status_color(self.theme_mode, "border")
        sel = status_color(self.theme_mode, "primary")
        return f"""
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
    background: {bg};
    color: {fg};
    padding: 4px 8px;
}}
QSpinBox, QDoubleSpinBox {{
    padding-right: 24px;
}}
QAbstractItemView {{
    background: {bg};
    color: {fg};
    selection-background-color: {sel};
}}
QTableWidget {{
    background: {bg};
    color: {fg};
    gridline-color: {border};
}}
QTableWidget::item {{
    background: {bg};
    color: {fg};
    padding: 2px 4px;
}}
QTableWidget QLineEdit, QTableWidget QSpinBox, QTableWidget QDoubleSpinBox, QTableWidget QComboBox {{
    background: {bg};
    color: {fg};
    border: 1px solid {border};
    selection-background-color: {sel};
}}
"""

    def _on_transition_defaults_changed(self, *_args: Any) -> None:
        self._sync_transition_cells(apply_defaults=True)
        self._refresh_thermal_plot()

    def _init_defaults(self) -> None:
        self._set_composition({"Fe": 99.2, "C": 0.8})
        self._set_default_thermal_program()
        self._add_prep_row(
            PrepOperationV3(
                method="grinding_800",
                duration_s=90.0,
                abrasive_um=18.0,
                load_n=22.0,
                rpm=180.0,
                coolant="water",
                note="",
            )
        )
        self._add_prep_row(
            PrepOperationV3(
                method="polishing_1um",
                duration_s=100.0,
                abrasive_um=1.0,
                load_n=10.0,
                rpm=120.0,
                coolant="alcohol",
                note="",
            )
        )
        self._update_etch_risk()
        self._apply_visual_standard(force=True)
        self._update_step_status()
        self._render_placeholder("Нажмите «Предпросмотр», чтобы построить структуру.")
        self._check_phase_model_resolution()

    def _load_presets(self) -> None:
        self.preset_combo.clear()
        self._preset_path_by_stem: dict[str, str] = {}
        paths = self.pipeline.list_preset_paths()
        self.preset_combo.addItem("Выберите пресет", "")
        lr1_paths: list[Path] = []
        steel_paths: list[Path] = []
        cast_iron_paths: list[Path] = []
        research_paths: list[Path] = []
        other_paths: list[Path] = []
        for path in paths:
            self._preset_path_by_stem[path.stem] = str(path)
            if path.stem in {
                preset_key for _, preset_key in LR1_SAMPLE_LIBRARY_PRESETS
            }:
                lr1_paths.append(path)
            elif path.stem in {
                preset_key for _, preset_key in STEEL_SAMPLE_LIBRARY_PRESETS
            }:
                steel_paths.append(path)
            elif path.stem in {
                preset_key for _, preset_key in CAST_IRON_SAMPLE_LIBRARY_PRESETS
            }:
                cast_iron_paths.append(path)
            elif path.stem in {
                preset_key for _, preset_key in LAB_RESEARCH_TEMPLATE_PRESETS
            }:
                research_paths.append(path)
            else:
                other_paths.append(path)

        if lr1_paths:
            self.preset_combo.addItem(LR1_TEMPLATE_GROUP_LABEL, "")
            for path in lr1_paths:
                _add_combo_item_with_tooltip(
                    self.preset_combo,
                    _preset_visible_label(path.stem),
                    str(path),
                    _preset_tooltip(path.stem),
                )
        if steel_paths:
            self.preset_combo.addItem(STEEL_LIBRARY_GROUP_LABEL, "")
            for path in steel_paths:
                _add_combo_item_with_tooltip(
                    self.preset_combo,
                    _preset_visible_label(path.stem),
                    str(path),
                    _preset_tooltip(path.stem),
                )
        if cast_iron_paths:
            self.preset_combo.addItem(CAST_IRON_LIBRARY_GROUP_LABEL, "")
            for path in cast_iron_paths:
                _add_combo_item_with_tooltip(
                    self.preset_combo,
                    _preset_visible_label(path.stem),
                    str(path),
                    _preset_tooltip(path.stem),
                )
        if other_paths:
            self.preset_combo.addItem(CURRICULUM_TEMPLATE_GROUP_LABEL, "")
            for path in other_paths:
                _add_combo_item_with_tooltip(
                    self.preset_combo,
                    _preset_visible_label(path.stem),
                    str(path),
                    _preset_tooltip(path.stem),
                )
        if research_paths:
            self.preset_combo.addItem(RESEARCH_OPTICS_GROUP_LABEL, "")
            for path in research_paths:
                _add_combo_item_with_tooltip(
                    self.preset_combo,
                    _preset_visible_label(path.stem),
                    str(path),
                    _preset_tooltip(path.stem),
                )

    def _make_transition_model_combo(self, model_code: str) -> QComboBox:
        combo = QComboBox(self.thermal_points_table)
        for code, label in THERMAL_TRANSITION_MODEL_OPTIONS:
            combo.addItem(label, code)
        code_norm = str(model_code or "").strip().lower()
        if not code_norm:
            code_norm = (
                "auto" if self.auto_transition_by_medium_check.isChecked() else "linear"
            )
        idx = combo.findData(code_norm)
        if idx < 0:
            idx = combo.findData(
                "auto" if self.auto_transition_by_medium_check.isChecked() else "linear"
            )
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.currentIndexChanged.connect(self._refresh_thermal_plot)
        return combo

    def _make_segment_medium_combo(self, medium_code: str) -> QComboBox:
        combo = QComboBox(self.thermal_points_table)
        for code, label in SEGMENT_MEDIUM_OPTIONS:
            combo.addItem(label, code)
        code_norm = str(medium_code or "inherit").strip().lower() or "inherit"
        idx = combo.findData(code_norm)
        if idx < 0:
            idx = combo.findData("inherit")
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.currentIndexChanged.connect(self._refresh_thermal_plot)
        return combo

    def _add_thermal_point_row(self, point: ThermalPointV3 | None) -> None:
        p = point or ThermalPointV3(
            time_s=0.0, temperature_c=20.0, label="", locked=False
        )
        transition = ThermalTransitionV3.from_dict(
            getattr(p, "transition_to_next", None).to_dict()
            if isinstance(getattr(p, "transition_to_next", None), ThermalTransitionV3)
            else {}
        )
        if point is None:
            transition.model = str(
                self.default_transition_model_combo.currentData() or "linear"
            )
            transition.curvature = float(self.default_transition_curvature_spin.value())
            transition.segment_medium_code = str(
                self.default_transition_segment_medium_combo.currentData() or "inherit"
            )
        row = self.thermal_points_table.rowCount()
        self.thermal_points_table.insertRow(row)
        self.thermal_points_table.setItem(
            row, 0, QTableWidgetItem(f"{float(p.time_s):.3f}")
        )
        self.thermal_points_table.setItem(
            row, 1, QTableWidgetItem(f"{float(p.temperature_c):.3f}")
        )
        self.thermal_points_table.setItem(row, 2, QTableWidgetItem(str(p.label or "")))
        lock_item = QTableWidgetItem("1" if bool(p.locked) else "0")
        self.thermal_points_table.setItem(row, 3, lock_item)
        self.thermal_points_table.setCellWidget(
            row, 4, self._make_transition_model_combo(str(transition.model))
        )
        self.thermal_points_table.setItem(
            row, 5, QTableWidgetItem(f"{float(transition.curvature):.3f}")
        )
        self.thermal_points_table.setCellWidget(
            row, 6, self._make_segment_medium_combo(str(transition.segment_medium_code))
        )
        self._sync_transition_cells()

    def _remove_thermal_point_row(self) -> None:
        row = self.thermal_points_table.currentRow()
        if row >= 0:
            self.thermal_points_table.removeRow(row)
        self._sync_transition_cells()
        self._refresh_thermal_plot()

    def _sync_transition_cells(self, apply_defaults: bool = False) -> None:
        rows = self.thermal_points_table.rowCount()
        selected_default_model = (
            str(self.default_transition_model_combo.currentData() or "").strip().lower()
        )
        if selected_default_model:
            default_model = selected_default_model
        else:
            default_model = (
                "auto" if self.auto_transition_by_medium_check.isChecked() else "linear"
            )
        default_medium = str(
            self.default_transition_segment_medium_combo.currentData() or "inherit"
        )
        default_curvature_text = (
            f"{float(self.default_transition_curvature_spin.value()):.3f}"
        )
        for row in range(rows):
            is_last = row == rows - 1
            model_combo = self.thermal_points_table.cellWidget(row, 4)
            if isinstance(model_combo, QComboBox):
                if is_last:
                    model_combo.setEnabled(True)
                    model_combo.setToolTip(
                        "Переход последней точки не используется при расчете"
                    )
                else:
                    model_combo.setEnabled(True)
                    if apply_defaults:
                        target = default_model
                        idx = model_combo.findData(target)
                        if idx >= 0 and model_combo.currentIndex() != idx:
                            model_combo.blockSignals(True)
                            model_combo.setCurrentIndex(idx)
                            model_combo.blockSignals(False)
                    elif model_combo.currentIndex() < 0:
                        target = default_model
                        idx = model_combo.findData(target)
                        if idx >= 0:
                            model_combo.setCurrentIndex(idx)

            medium_combo = self.thermal_points_table.cellWidget(row, 6)
            if isinstance(medium_combo, QComboBox):
                if is_last:
                    medium_combo.setEnabled(True)
                    medium_combo.setToolTip(
                        "Переход последней точки не используется при расчете"
                    )
                else:
                    medium_combo.setEnabled(True)
                    if apply_defaults:
                        idx = medium_combo.findData(default_medium)
                        if idx >= 0 and medium_combo.currentIndex() != idx:
                            medium_combo.blockSignals(True)
                            medium_combo.setCurrentIndex(idx)
                            medium_combo.blockSignals(False)
                    elif medium_combo.currentIndex() < 0:
                        idx = medium_combo.findData(default_medium)
                        if idx >= 0:
                            medium_combo.setCurrentIndex(idx)

            curv_item = self.thermal_points_table.item(row, 5)
            if curv_item is None:
                curv_item = QTableWidgetItem("")
                self.thermal_points_table.setItem(row, 5, curv_item)
            flags = (
                Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsEditable
            )
            if is_last:
                curv_item.setText("—")
                flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
            else:
                if apply_defaults:
                    curv_item.setText(default_curvature_text)
                elif not curv_item.text().strip() or curv_item.text().strip() == "—":
                    curv_item.setText(default_curvature_text)
            curv_item.setFlags(flags)

    def _set_default_thermal_program(self) -> None:
        self.thermal_points_table.setRowCount(0)
        defaults = [
            ThermalPointV3(
                time_s=0.0,
                temperature_c=20.0,
                label="Старт",
                locked=True,
                transition_to_next=ThermalTransitionV3(
                    model="auto", curvature=1.6, segment_medium_code="air"
                ),
            ),
            ThermalPointV3(
                time_s=600.0,
                temperature_c=840.0,
                label="Нагрев",
                transition_to_next=ThermalTransitionV3(
                    model="auto", curvature=1.0, segment_medium_code="air"
                ),
            ),
            ThermalPointV3(
                time_s=720.0,
                temperature_c=840.0,
                label="Выдержка",
                transition_to_next=ThermalTransitionV3(
                    model="auto", curvature=2.0, segment_medium_code="inherit"
                ),
            ),
            ThermalPointV3(
                time_s=900.0,
                temperature_c=20.0,
                label="Охлаждение",
                transition_to_next=ThermalTransitionV3(
                    model="linear", curvature=1.0, segment_medium_code="inherit"
                ),
            ),
        ]
        for p in defaults:
            self._add_thermal_point_row(p)
        midx = self.quench_medium_combo.findData("air")
        if midx >= 0:
            self.quench_medium_combo.setCurrentIndex(midx)
        self.quench_time_spin.setValue(0.0)
        self.quench_bath_temp_spin.setValue(25.0)
        self.quench_sample_temp_spin.setValue(840.0)
        self.quench_custom_name.setText("")
        self.quench_severity_spin.setValue(1.0)
        m_idx = self.default_transition_model_combo.findData("linear")
        if m_idx >= 0:
            self.default_transition_model_combo.setCurrentIndex(m_idx)
        self.default_transition_curvature_spin.setValue(1.0)
        med_idx = self.default_transition_segment_medium_combo.findData("inherit")
        if med_idx >= 0:
            self.default_transition_segment_medium_combo.setCurrentIndex(med_idx)
        self.auto_transition_by_medium_check.setChecked(True)
        sidx = self.thermal_sampling_combo.findData("per_degree")
        if sidx >= 0:
            self.thermal_sampling_combo.setCurrentIndex(sidx)
        self.thermal_degree_step_spin.setValue(1.0)
        self.thermal_max_frames_spin.setValue(320)
        self._sync_transition_cells()
        self._refresh_thermal_plot()

    def _collect_thermal_points(self) -> list[ThermalPointV3]:
        points: list[ThermalPointV3] = []
        rows = self.thermal_points_table.rowCount()
        auto_by_medium = bool(self.auto_transition_by_medium_check.isChecked())
        for row in range(rows):
            t_item = self.thermal_points_table.item(row, 0)
            temp_item = self.thermal_points_table.item(row, 1)
            label_item = self.thermal_points_table.item(row, 2)
            lock_item = self.thermal_points_table.item(row, 3)
            model_widget = self.thermal_points_table.cellWidget(row, 4)
            curvature_item = self.thermal_points_table.item(row, 5)
            medium_widget = self.thermal_points_table.cellWidget(row, 6)
            model_raw = ""
            if isinstance(model_widget, QComboBox):
                model_raw = str(model_widget.currentData() or "").strip().lower()
            if row >= rows - 1:
                model_raw = "linear"
            if auto_by_medium and model_raw in {"", "—", "auto"}:
                model_value = ""
                curvature_value = 0.0
            else:
                model_value = model_raw or "linear"
                curvature_value = _safe_float(
                    curvature_item.text() if curvature_item else 1.0, 1.0
                )
            medium_value = "inherit"
            if isinstance(medium_widget, QComboBox):
                medium_value = (
                    str(medium_widget.currentData() or "inherit").strip().lower()
                    or "inherit"
                )
            if row >= rows - 1:
                medium_value = "inherit"
            points.append(
                ThermalPointV3(
                    time_s=_safe_float(t_item.text() if t_item else 0.0, 0.0),
                    temperature_c=_safe_float(
                        temp_item.text() if temp_item else 20.0, 20.0
                    ),
                    label=(label_item.text() if label_item else ""),
                    locked=str(lock_item.text() if lock_item else "0").strip()
                    in {"1", "true", "True", "да", "yes"},
                    transition_to_next=ThermalTransitionV3(
                        model=model_value,
                        curvature=float(curvature_value),
                        segment_medium_code=medium_value,
                    ),
                )
            )
        return points

    def _collect_thermal_program(self) -> ThermalProgramV3:
        quench = QuenchSettingsV3(
            medium_code=str(self.quench_medium_combo.currentData() or "air"),
            quench_time_s=float(self.quench_time_spin.value()),
            bath_temperature_c=float(self.quench_bath_temp_spin.value()),
            sample_temperature_c=float(self.quench_sample_temp_spin.value()),
            custom_medium_name=self.quench_custom_name.text().strip(),
            custom_severity_factor=float(self.quench_severity_spin.value()),
        )
        return ThermalProgramV3(
            points=self._collect_thermal_points(),
            quench=quench,
            sampling_mode=str(
                self.thermal_sampling_combo.currentData() or "per_degree"
            ),
            degree_step_c=float(self.thermal_degree_step_spin.value()),
            max_frames=int(self.thermal_max_frames_spin.value()),
        )

    def _on_quench_medium_changed(self, *_args: Any) -> None:
        code = str(self.quench_medium_combo.currentData() or "air")
        target_bath = QUENCH_MEDIUM_DEFAULT_BATH.get(code)
        if target_bath is not None:
            self.quench_bath_temp_spin.blockSignals(True)
            self.quench_bath_temp_spin.setValue(float(target_bath))
            self.quench_bath_temp_spin.blockSignals(False)
        self._refresh_thermal_plot()

    @staticmethod
    def _curve_point_code(index: int) -> str:
        # Excel-like sequence: A..Z, AA..ZZ, AAA...
        n = int(index) + 1
        if n <= 0:
            return "A"
        chars: list[str] = []
        while n > 0:
            n, rem = divmod(n - 1, 26)
            chars.append(chr(ord("A") + rem))
        return "".join(reversed(chars))

    def _refresh_thermal_plot(self, *_args: Any) -> None:
        if (
            pg is None
            or not hasattr(self, "thermal_plot_widget")
            or self.thermal_plot_widget is None
        ):
            return
        try:
            program = self._collect_thermal_program()
            points = list(program.points or [])
            points.sort(key=lambda p: float(p.time_s))
            xs = [float(p.time_s) for p in points]
            ys = [float(p.temperature_c) for p in points]
            if len(points) < 2:
                self.thermal_plot_widget.clear()
                if xs and ys:
                    self.thermal_plot_widget.plot(
                        xs,
                        ys,
                        pen=None,
                        symbol="o",
                        symbolSize=6,
                        symbolBrush=(180, 225, 255),
                    )
                return

            # Always draw a dense preview curve so nonlinear transitions are visible
            # even when export/discretization mode is "points only".
            preview_step = max(0.5, min(2.0, float(program.degree_step_c)))
            estimated_samples = 0
            for i in range(len(points) - 1):
                dt = abs(
                    float(points[i + 1].temperature_c) - float(points[i].temperature_c)
                )
                estimated_samples += max(4, int(dt / preview_step) + 4)
            preview_max_frames = int(min(200000, max(2000, estimated_samples + 32)))
            preview_program = ThermalProgramV3(
                points=list(program.points or []),
                quench=program.quench,
                sampling_mode="per_degree",
                degree_step_c=preview_step,
                max_frames=preview_max_frames,
            )
            sampled = sample_thermal_program(preview_program)
            line_x = (
                [float(row.get("time_s", 0.0)) for row in sampled] if sampled else xs
            )
            line_y = (
                [float(row.get("temperature_c", 0.0)) for row in sampled]
                if sampled
                else ys
            )
            if xs and ys and line_x and line_y:
                if abs(line_x[0] - xs[0]) > 1e-6 or abs(line_y[0] - ys[0]) > 1e-6:
                    line_x.insert(0, xs[0])
                    line_y.insert(0, ys[0])
                if abs(line_x[-1] - xs[-1]) > 1e-6 or abs(line_y[-1] - ys[-1]) > 1e-6:
                    line_x.append(xs[-1])
                    line_y.append(ys[-1])
            self.thermal_plot_widget.clear()
            pen_curve = pg.mkPen(color=(110, 190, 255), width=3)
            self.thermal_plot_widget.plot(line_x, line_y, pen=pen_curve)
            pen_points = pg.mkPen(color=(210, 230, 255), width=1)
            self.thermal_plot_widget.plot(
                xs,
                ys,
                pen=None,
                symbol="o",
                symbolPen=pen_points,
                symbolSize=6,
                symbolBrush=(180, 225, 255),
            )
            if xs and ys and line_y:
                y_min = float(min(ys))
                y_max = float(max(ys))
                y_span = max(1.0, y_max - y_min)
                y_offset = max(4.0, y_span * 0.03)
                for idx, (x_val, y_val) in enumerate(zip(xs, ys, strict=True)):
                    code = self._curve_point_code(idx)
                    dy = y_offset if (idx % 2 == 0) else -y_offset
                    anchor = (0.5, 1.0) if dy > 0 else (0.5, 0.0)
                    text_item = pg.TextItem(
                        text=code, color=(225, 235, 250), anchor=anchor
                    )
                    text_item.setPos(float(x_val), float(y_val + dy))
                    self.thermal_plot_widget.addItem(text_item)
        except Exception:
            pass

    def _export_thermal_curve_png(self) -> None:
        if (
            pg is None
            or not hasattr(self, "thermal_plot_widget")
            or self.thermal_plot_widget is None
        ):
            QMessageBox.warning(self, "Экспорт графика", "pyqtgraph не установлен.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график термопрограммы", "thermal_curve.png", "PNG (*.png)"
        )
        if not path:
            return
        try:
            from pyqtgraph.exporters import ImageExporter  # type: ignore

            exporter = ImageExporter(self.thermal_plot_widget.plotItem)
            exporter.export(path)
            self._set_status(f"График сохранен: {Path(path).name}")
        except Exception as exc:
            QMessageBox.warning(self, "Экспорт графика", str(exc))

    def _export_thermal_curve_png_hd(self) -> None:
        """Экспорт графика термопрограммы в высоком разрешении (300 DPI)."""
        if (
            pg is None
            or not hasattr(self, "thermal_plot_widget")
            or self.thermal_plot_widget is None
        ):
            QMessageBox.warning(self, "Экспорт графика HD", "pyqtgraph не установлен.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить график термопрограммы (HD)",
            "thermal_curve_hd.png",
            "PNG (*.png)",
        )
        if not path:
            return
        try:
            from pyqtgraph.exporters import ImageExporter  # type: ignore

            exporter = ImageExporter(self.thermal_plot_widget.plotItem)
            # Устанавливаем высокое разрешение: 3000x2000 пикселей (~10x6.7 дюймов при 300 DPI)
            exporter.parameters()["width"] = 3000
            exporter.export(path)
            self._set_status(f"График HD сохранен: {Path(path).name}")
        except Exception as exc:
            QMessageBox.warning(self, "Экспорт графика HD", str(exc))

    def _export_thermal_curve_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить точки термопрограммы", "thermal_curve.csv", "CSV (*.csv)"
        )
        if not path:
            return
        program = self._collect_thermal_program()
        points = list(program.points or [])
        points.sort(key=lambda p: float(p.time_s))
        with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(
                [
                    "time_s",
                    "temperature_c",
                    "label",
                    "locked",
                    "transition_model",
                    "curvature",
                    "segment_medium_code",
                    "segment_medium_factor",
                ]
            )
            for idx, p in enumerate(points):
                tr = getattr(p, "transition_to_next", None)
                model = ""
                curvature = ""
                medium = ""
                factor = ""
                if idx < len(points) - 1 and isinstance(tr, ThermalTransitionV3):
                    model = str(tr.model)
                    curvature = f"{float(tr.curvature):.6f}"
                    medium = str(tr.segment_medium_code)
                    factor = (
                        ""
                        if tr.segment_medium_factor is None
                        else f"{float(tr.segment_medium_factor):.6f}"
                    )
                writer.writerow(
                    [
                        f"{float(p.time_s):.6f}",
                        f"{float(p.temperature_c):.6f}",
                        str(p.label),
                        int(bool(p.locked)),
                        model,
                        curvature,
                        medium,
                        factor,
                    ]
                )
        self._set_status(f"CSV сохранен: {Path(path).name}")

    def _add_composition_row(self, symbol: str = "", value: float = 0.0) -> None:
        row = self.composition_table.rowCount()
        self.composition_table.insertRow(row)
        self.composition_table.setItem(row, 0, QTableWidgetItem(symbol))
        self.composition_table.setItem(
            row, 1, QTableWidgetItem(f"{value:.4f}" if value > 0 else "0.0")
        )

    def _remove_composition_row(self) -> None:
        row = self.composition_table.currentRow()
        if row >= 0:
            self.composition_table.removeRow(row)

    def _set_composition(self, composition: dict[str, float]) -> None:
        self.composition_table.setRowCount(0)
        for symbol, value in composition.items():
            self._add_composition_row(str(symbol), float(value))

    def _collect_composition(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in range(self.composition_table.rowCount()):
            symbol_item = self.composition_table.item(row, 0)
            wt_item = self.composition_table.item(row, 1)
            symbol = "" if symbol_item is None else symbol_item.text().strip()
            if not symbol:
                continue
            wt = _safe_float(wt_item.text() if wt_item else 0.0, 0.0)
            if wt > 0:
                out[symbol] = wt
        return out

    def _normalize_composition_rows(self) -> None:
        comp = self._collect_composition()
        total = float(sum(comp.values()))
        if total <= 1e-9:
            return
        norm = {k: v * 100.0 / total for k, v in comp.items()}
        self._set_composition(norm)

    def _validate_composition_realtime(
        self, _item: QTableWidgetItem | None = None
    ) -> None:
        """Validate composition in real-time and highlight errors."""
        if self._is_validating_composition:
            return
        self._is_validating_composition = True
        from PySide6.QtGui import QColor

        try:
            with QSignalBlocker(self.composition_table):
                # Collect composition
                comp = self._collect_composition()
                total = sum(comp.values())

                # Validate each row
                for row in range(self.composition_table.rowCount()):
                    symbol_item = self.composition_table.item(row, 0)
                    wt_item = self.composition_table.item(row, 1)

                    if symbol_item is None or wt_item is None:
                        continue

                    symbol = symbol_item.text().strip()
                    wt_text = wt_item.text().strip()

                    # Reset background
                    symbol_item.setBackground(QColor(255, 255, 255))
                    wt_item.setBackground(QColor(255, 255, 255))
                    symbol_item.setToolTip("")
                    wt_item.setToolTip("")

                    # Check for empty symbol
                    if not symbol:
                        symbol_item.setBackground(QColor(255, 200, 200))
                        symbol_item.setToolTip("Элемент не указан")
                        continue

                    # Check for invalid weight
                    try:
                        wt = float(wt_text) if wt_text else 0.0
                    except ValueError:
                        wt_item.setBackground(QColor(255, 200, 200))
                        wt_item.setToolTip("Некорректное числовое значение")
                        continue

                    # Check for negative or zero weight
                    if wt <= 0:
                        wt_item.setBackground(QColor(255, 230, 200))
                        wt_item.setToolTip("Содержание должно быть > 0")

                    # Check for unrealistic values
                    if wt > 100:
                        wt_item.setBackground(QColor(255, 200, 200))
                        wt_item.setToolTip("Содержание не может превышать 100%")

                # Check total sum
                if abs(total - 100.0) > 0.1 and total > 0:
                    # Highlight all weight cells if sum is not 100%
                    color = (
                        QColor(255, 255, 200)
                        if abs(total - 100.0) < 5.0
                        else QColor(255, 230, 200)
                    )
                    tooltip = f"Сумма: {total:.2f}% (должна быть 100%)"

                    for row in range(self.composition_table.rowCount()):
                        wt_item = self.composition_table.item(row, 1)
                        if wt_item is not None:
                            current_bg = wt_item.background().color()
                            # Only override if not already marked as error
                            if current_bg == QColor(255, 255, 255):
                                wt_item.setBackground(color)
                                wt_item.setToolTip(tooltip)
        finally:
            self._is_validating_composition = False

    def _import_composition_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Импорт состава", "", "JSON (*.json)"
        )
        if not path:
            return
        payload = _json_load(Path(path))
        comp = payload.get("composition", payload)
        if not isinstance(comp, dict):
            QMessageBox.warning(self, "Импорт", "Формат JSON не распознан.")
            return
        parsed: dict[str, float] = {}
        for k, v in comp.items():
            val = _safe_float(v, -1.0)
            if val > 0.0:
                parsed[str(k)] = val
        if not parsed:
            QMessageBox.warning(self, "Импорт", "В файле нет валидного состава.")
            return
        self._set_composition(parsed)

    def _export_composition_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт состава", "composition_v3.json", "JSON (*.json)"
        )
        if not path:
            return
        save_json({"composition": self._collect_composition()}, path)

    def _recommended_textbook_profile_id(self) -> str:
        composition = self._collect_composition()
        hint = str(self.system_hint_combo.currentData() or "")
        system, _, _ = infer_training_system(composition=composition, system_hint=hint)
        return TEXTBOOK_PROFILE_BY_SYSTEM.get(system, "textbook_steel_bw")

    def _on_visual_standard_changed(self) -> None:
        self._apply_visual_standard(force=False)

    def _apply_visual_standard(self, force: bool) -> None:
        mode = str(self.visual_standard_combo.currentData() or "textbook_bw")
        if mode != "textbook_bw":
            return
        composition = self._collect_composition()
        hint = str(self.system_hint_combo.currentData() or "")
        system, _, _ = infer_training_system(composition=composition, system_hint=hint)
        profile_id = self._recommended_textbook_profile_id()
        pidx = self.synth_profile_combo.findData(profile_id)
        if pidx >= 0:
            self.synth_profile_combo.setCurrentIndex(pidx)
        elif force:
            self.synth_profile_combo.addItem(
                _label_ru(profile_id, SYNTH_PROFILE_LABELS_RU), profile_id
            )
            self.synth_profile_combo.setCurrentIndex(
                self.synth_profile_combo.count() - 1
            )
        prep_profile = {
            "fe-c": "textbook_steel_bw",
            "fe-si": "textbook_steel_bw",
            "al-si": "textbook_alsi_bw",
            "cu-zn": "textbook_brass_bw",
            "al-cu-mg": "textbook_heat_treatment_bw",
        }.get(system, "textbook_steel_bw")
        if force:
            prep_idx = self.prep_template_combo.findData(prep_profile)
            if prep_idx >= 0:
                self.prep_template_combo.setCurrentIndex(prep_idx)
                self._apply_prep_template()
            etch_idx = self.etch_profile_combo.findData(prep_profile)
            if etch_idx >= 0:
                self.etch_profile_combo.setCurrentIndex(etch_idx)
                self._apply_etch_profile()

        self.synth_generation_mode_combo.setCurrentIndex(
            max(0, self.synth_generation_mode_combo.findData("edu_engineering"))
        )
        self.synth_phase_emphasis_combo.setCurrentIndex(
            max(0, self.synth_phase_emphasis_combo.findData("contrast_texture"))
        )
        self.synth_sensitivity_combo.setCurrentIndex(
            max(0, self.synth_sensitivity_combo.findData("educational"))
        )
        self.synth_phase_tolerance_spin.setValue(20.0)
        self.synth_contrast_spin.setValue(max(1.2, self.synth_contrast_spin.value()))
        self.synth_sharp_spin.setValue(max(1.2, self.synth_sharp_spin.value()))
        self.synth_artifact_spin.setValue(min(self.synth_artifact_spin.value(), 0.22))

    def _apply_lab_template(self) -> None:
        preset_stem = str(self.lab_template_combo.currentData() or "")
        if not preset_stem:
            return
        path = self._preset_path_by_stem.get(preset_stem, "")
        if not path:
            QMessageBox.warning(
                self,
                "Шаблон ЛР",
                f"Не найден пресет: {_preset_visible_label(preset_stem)}",
            )
            return
        try:
            payload = self.pipeline.load_preset(path)
            self._remember_preset_context(payload)
            req = MetallographyRequestV3.from_dict(payload)
            self._apply_request_to_ui(req)
            self.visual_standard_combo.setCurrentIndex(
                max(0, self.visual_standard_combo.findData("textbook_bw"))
            )
            self._apply_visual_standard(force=True)
            self._set_status(
                f"Шаблон ЛР применен: {_preset_visible_label(preset_stem)}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Шаблон ЛР", str(exc))

    def _add_manual_phase_row(self, phase_name: str, fraction: float) -> None:
        row = self.manual_phase_table.rowCount()
        self.manual_phase_table.insertRow(row)
        self.manual_phase_table.setItem(row, 0, QTableWidgetItem(str(phase_name)))
        self.manual_phase_table.setItem(
            row,
            1,
            QTableWidgetItem(f"{float(fraction):.4f}" if fraction > 0.0 else "0.0"),
        )

    def _remove_manual_phase_row(self) -> None:
        row = self.manual_phase_table.currentRow()
        if row >= 0:
            self.manual_phase_table.removeRow(row)

    def _collect_manual_phase_fractions(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in range(self.manual_phase_table.rowCount()):
            phase_item = self.manual_phase_table.item(row, 0)
            frac_item = self.manual_phase_table.item(row, 1)
            phase = "" if phase_item is None else phase_item.text().strip()
            if not phase:
                continue
            frac = _safe_float(frac_item.text() if frac_item else 0.0, 0.0)
            if frac > 0.0:
                out[phase] = frac
        return out

    def _normalize_manual_phase_rows(self) -> None:
        vals = self._collect_manual_phase_fractions()
        total = float(sum(vals.values()))
        if total <= 1e-12:
            return
        self.manual_phase_table.setRowCount(0)
        for phase, value in vals.items():
            self._add_manual_phase_row(phase, value / total)

    def _sync_phase_override_from_slider(self, raw: int) -> None:
        val = float(raw) / 100.0
        self.phase_override_spin.blockSignals(True)
        self.phase_override_spin.setValue(val)
        self.phase_override_spin.blockSignals(False)

    def _sync_phase_override_from_spin(self, value: float) -> None:
        raw = int(round(float(value) * 100.0))
        self.phase_override_slider.blockSignals(True)
        self.phase_override_slider.setValue(raw)
        self.phase_override_slider.blockSignals(False)

    def _make_float_spin(
        self,
        value: float,
        low: float,
        high: float,
        step: float = 1.0,
        decimals: int = 2,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _add_prep_row(self, op: PrepOperationV3 | None) -> None:
        row = self.prep_table.rowCount()
        self.prep_table.insertRow(row)
        operation = op or PrepOperationV3(
            method="grinding_800",
            duration_s=90.0,
            abrasive_um=18.0,
            load_n=20.0,
            rpm=170.0,
        )

        method_combo = QComboBox()
        for method in PREP_METHODS:
            method_combo.addItem(_label_ru(method, PREP_METHOD_LABELS_RU), method)
        idx = method_combo.findData(operation.method)
        if idx >= 0:
            method_combo.setCurrentIndex(idx)
        self.prep_table.setCellWidget(row, 0, method_combo)
        self.prep_table.setCellWidget(
            row, 1, self._make_float_spin(operation.duration_s, 0.0, 36000.0, 1.0)
        )
        self.prep_table.setCellWidget(
            row,
            2,
            self._make_float_spin(
                float(operation.abrasive_um or 0.0), 0.0, 1000.0, 0.1
            ),
        )
        self.prep_table.setCellWidget(
            row,
            3,
            self._make_float_spin(float(operation.load_n or 0.0), 0.0, 1000.0, 0.1),
        )
        self.prep_table.setCellWidget(
            row, 4, self._make_float_spin(float(operation.rpm or 0.0), 0.0, 5000.0, 1.0)
        )
        coolant_combo = QComboBox()
        for coolant in COOLANT_OPTIONS:
            coolant_combo.addItem(_label_ru(coolant, COOLANT_LABELS_RU), coolant)
        cidx = coolant_combo.findData(operation.coolant or "none")
        if cidx >= 0:
            coolant_combo.setCurrentIndex(cidx)
        self.prep_table.setCellWidget(row, 5, coolant_combo)
        self.prep_table.setCellWidget(
            row,
            6,
            self._make_float_spin(float(operation.direction_deg), 0.0, 360.0, 1.0),
        )
        load_profile_combo = QComboBox()
        for code, label in [
            ("constant", "Постоянный"),
            ("ramp_up", "Рост"),
            ("ramp_down", "Спад"),
            ("pulse", "Импульс"),
        ]:
            load_profile_combo.addItem(label, code)
        lp_idx = load_profile_combo.findData(operation.load_profile)
        if lp_idx >= 0:
            load_profile_combo.setCurrentIndex(lp_idx)
        self.prep_table.setCellWidget(row, 7, load_profile_combo)
        cloth_combo = QComboBox()
        for code, label in [
            ("standard", "Стандарт"),
            ("hard", "Жесткая"),
            ("soft", "Мягкая"),
            ("final", "Финишная"),
        ]:
            cloth_combo.addItem(label, code)
        cl_idx = cloth_combo.findData(operation.cloth_type)
        if cl_idx >= 0:
            cloth_combo.setCurrentIndex(cl_idx)
        self.prep_table.setCellWidget(row, 8, cloth_combo)
        slurry_combo = QComboBox()
        for code, label in [
            ("diamond", "Алмаз"),
            ("alumina", "Оксид Al"),
            ("silica", "Кремнезем"),
            ("custom", "Пользовательская"),
        ]:
            slurry_combo.addItem(label, code)
        sl_idx = slurry_combo.findData(operation.slurry_type)
        if sl_idx >= 0:
            slurry_combo.setCurrentIndex(sl_idx)
        self.prep_table.setCellWidget(row, 9, slurry_combo)
        self.prep_table.setCellWidget(
            row,
            10,
            self._make_float_spin(
                float(operation.lubricant_flow_ml_min), 0.0, 500.0, 0.1
            ),
        )
        self.prep_table.setCellWidget(
            row,
            11,
            self._make_float_spin(float(operation.oscillation_hz), 0.0, 30.0, 0.1),
        )
        path_combo = QComboBox()
        for code, label in [
            ("linear", "Линейная"),
            ("circular", "Круговая"),
            ("figure8", "Восьмерка"),
            ("random", "Случайная"),
        ]:
            path_combo.addItem(label, code)
        pa_idx = path_combo.findData(operation.path_pattern)
        if pa_idx >= 0:
            path_combo.setCurrentIndex(pa_idx)
        self.prep_table.setCellWidget(row, 12, path_combo)
        self.prep_table.setItem(row, 13, QTableWidgetItem(operation.note or ""))

    def _remove_prep_row(self) -> None:
        row = self.prep_table.currentRow()
        if row >= 0:
            self.prep_table.removeRow(row)

    def _prep_row_to_operation(self, row: int) -> PrepOperationV3:
        method_combo = self.prep_table.cellWidget(row, 0)
        dur_spin = self.prep_table.cellWidget(row, 1)
        abrasive_spin = self.prep_table.cellWidget(row, 2)
        load_spin = self.prep_table.cellWidget(row, 3)
        rpm_spin = self.prep_table.cellWidget(row, 4)
        coolant_combo = self.prep_table.cellWidget(row, 5)
        direction_spin = self.prep_table.cellWidget(row, 6)
        load_profile_combo = self.prep_table.cellWidget(row, 7)
        cloth_combo = self.prep_table.cellWidget(row, 8)
        slurry_combo = self.prep_table.cellWidget(row, 9)
        lube_spin = self.prep_table.cellWidget(row, 10)
        oscill_spin = self.prep_table.cellWidget(row, 11)
        path_combo = self.prep_table.cellWidget(row, 12)
        note_item = self.prep_table.item(row, 13)
        return PrepOperationV3(
            method=str(method_combo.currentData())
            if isinstance(method_combo, QComboBox)
            else "grinding_800",
            duration_s=float(dur_spin.value())
            if isinstance(dur_spin, QDoubleSpinBox)
            else 0.0,
            abrasive_um=float(abrasive_spin.value())
            if isinstance(abrasive_spin, QDoubleSpinBox)
            else 0.0,
            load_n=float(load_spin.value())
            if isinstance(load_spin, QDoubleSpinBox)
            else 0.0,
            rpm=float(rpm_spin.value())
            if isinstance(rpm_spin, QDoubleSpinBox)
            else 0.0,
            coolant=str(coolant_combo.currentData())
            if isinstance(coolant_combo, QComboBox)
            else None,
            direction_deg=float(direction_spin.value())
            if isinstance(direction_spin, QDoubleSpinBox)
            else 0.0,
            load_profile=str(load_profile_combo.currentData())
            if isinstance(load_profile_combo, QComboBox)
            else "constant",
            cloth_type=str(cloth_combo.currentData())
            if isinstance(cloth_combo, QComboBox)
            else "standard",
            slurry_type=str(slurry_combo.currentData())
            if isinstance(slurry_combo, QComboBox)
            else "diamond",
            lubricant_flow_ml_min=float(lube_spin.value())
            if isinstance(lube_spin, QDoubleSpinBox)
            else 0.0,
            cleaning_between_steps=False,
            oscillation_hz=float(oscill_spin.value())
            if isinstance(oscill_spin, QDoubleSpinBox)
            else 0.0,
            path_pattern=str(path_combo.currentData())
            if isinstance(path_combo, QComboBox)
            else "linear",
            note="" if note_item is None else note_item.text().strip(),
        )

    def _set_prep_row(self, row: int, op: PrepOperationV3) -> None:
        if row < 0 or row >= self.prep_table.rowCount():
            return
        self.prep_table.removeRow(row)
        self.prep_table.insertRow(row)

        method_combo = QComboBox()
        for method in PREP_METHODS:
            method_combo.addItem(_label_ru(method, PREP_METHOD_LABELS_RU), method)
        idx = method_combo.findData(op.method)
        if idx >= 0:
            method_combo.setCurrentIndex(idx)
        self.prep_table.setCellWidget(row, 0, method_combo)
        self.prep_table.setCellWidget(
            row, 1, self._make_float_spin(op.duration_s, 0.0, 36000.0, 1.0)
        )
        self.prep_table.setCellWidget(
            row,
            2,
            self._make_float_spin(float(op.abrasive_um or 0.0), 0.0, 1000.0, 0.1),
        )
        self.prep_table.setCellWidget(
            row, 3, self._make_float_spin(float(op.load_n or 0.0), 0.0, 1000.0, 0.1)
        )
        self.prep_table.setCellWidget(
            row, 4, self._make_float_spin(float(op.rpm or 0.0), 0.0, 5000.0, 1.0)
        )
        coolant_combo = QComboBox()
        for coolant in COOLANT_OPTIONS:
            coolant_combo.addItem(_label_ru(coolant, COOLANT_LABELS_RU), coolant)
        cidx = coolant_combo.findData(op.coolant or "none")
        if cidx >= 0:
            coolant_combo.setCurrentIndex(cidx)
        self.prep_table.setCellWidget(row, 5, coolant_combo)
        self.prep_table.setCellWidget(
            row, 6, self._make_float_spin(float(op.direction_deg), 0.0, 360.0, 1.0)
        )

        load_profile_combo = QComboBox()
        for code, label in [
            ("constant", "Постоянный"),
            ("ramp_up", "Рост"),
            ("ramp_down", "Спад"),
            ("pulse", "Импульс"),
        ]:
            load_profile_combo.addItem(label, code)
        lp_idx = load_profile_combo.findData(op.load_profile)
        if lp_idx >= 0:
            load_profile_combo.setCurrentIndex(lp_idx)
        self.prep_table.setCellWidget(row, 7, load_profile_combo)

        cloth_combo = QComboBox()
        for code, label in [
            ("standard", "Стандарт"),
            ("hard", "Жесткая"),
            ("soft", "Мягкая"),
            ("final", "Финишная"),
        ]:
            cloth_combo.addItem(label, code)
        cl_idx = cloth_combo.findData(op.cloth_type)
        if cl_idx >= 0:
            cloth_combo.setCurrentIndex(cl_idx)
        self.prep_table.setCellWidget(row, 8, cloth_combo)

        slurry_combo = QComboBox()
        for code, label in [
            ("diamond", "Алмаз"),
            ("alumina", "Оксид Al"),
            ("silica", "Кремнезем"),
            ("custom", "Пользовательская"),
        ]:
            slurry_combo.addItem(label, code)
        sl_idx = slurry_combo.findData(op.slurry_type)
        if sl_idx >= 0:
            slurry_combo.setCurrentIndex(sl_idx)
        self.prep_table.setCellWidget(row, 9, slurry_combo)
        self.prep_table.setCellWidget(
            row,
            10,
            self._make_float_spin(float(op.lubricant_flow_ml_min), 0.0, 500.0, 0.1),
        )
        self.prep_table.setCellWidget(
            row, 11, self._make_float_spin(float(op.oscillation_hz), 0.0, 30.0, 0.1)
        )
        path_combo = QComboBox()
        for code, label in [
            ("linear", "Линейная"),
            ("circular", "Круговая"),
            ("figure8", "Восьмерка"),
            ("random", "Случайная"),
        ]:
            path_combo.addItem(label, code)
        pa_idx = path_combo.findData(op.path_pattern)
        if pa_idx >= 0:
            path_combo.setCurrentIndex(pa_idx)
        self.prep_table.setCellWidget(row, 12, path_combo)
        self.prep_table.setItem(row, 13, QTableWidgetItem(op.note or ""))

    def _move_prep_row(self, direction: int) -> None:
        row = self.prep_table.currentRow()
        target = row + direction
        if row < 0 or target < 0 or target >= self.prep_table.rowCount():
            return
        src = self._prep_row_to_operation(row)
        dst = self._prep_row_to_operation(target)
        self.prep_table.removeRow(max(row, target))
        self.prep_table.removeRow(min(row, target))
        if direction < 0:
            self.prep_table.insertRow(target)
            self.prep_table.insertRow(row)
            self._set_prep_row(target, src)
            self._set_prep_row(row, dst)
            self.prep_table.selectRow(target)
        else:
            self.prep_table.insertRow(row)
            self.prep_table.insertRow(target)
            self._set_prep_row(row, dst)
            self._set_prep_row(target, src)
            self.prep_table.selectRow(target)

    def _apply_prep_template(self) -> None:
        key = str(self.prep_template_combo.currentData() or "")
        if not key:
            return
        payload = self._prep_templates.get(key, {})
        if not isinstance(payload, dict):
            return
        self.prep_table.setRowCount(0)
        for item in payload.get("steps", []):
            if not isinstance(item, dict):
                continue
            self._add_prep_row(PrepOperationV3.from_dict(item))
        self.prep_rough_spin.setValue(
            _safe_float(payload.get("roughness_target_um", 0.05), 0.05)
        )
        ridx = self.prep_relief_combo.findData(
            str(payload.get("relief_mode", "hardness_coupled"))
        )
        if ridx >= 0:
            self.prep_relief_combo.setCurrentIndex(ridx)
        self.prep_contam_spin.setValue(
            _safe_float(payload.get("contamination_level", 0.02), 0.02)
        )

    def _apply_etch_profile(self) -> None:
        key = str(self.etch_profile_combo.currentData() or "")
        if not key:
            return
        payload = self._etch_profiles.get(key, {})
        if not isinstance(payload, dict):
            return
        ridx = self.etch_reagent_combo.findData(str(payload.get("reagent", "nital_2")))
        if ridx >= 0:
            self.etch_reagent_combo.setCurrentIndex(ridx)
        self.etch_time_spin.setValue(_safe_float(payload.get("time_s", 8.0), 8.0))
        self.etch_temp_spin.setValue(
            _safe_float(payload.get("temperature_c", 22.0), 22.0)
        )
        aidx = self.etch_agitation_combo.findData(
            str(payload.get("agitation", "gentle"))
        )
        if aidx >= 0:
            self.etch_agitation_combo.setCurrentIndex(aidx)
        self.etch_overetch_spin.setValue(
            _safe_float(payload.get("overetch_factor", 1.0), 1.0)
        )
        unit = str(payload.get("concentration_unit", "wt_pct"))
        cidx = self.etch_conc_unit_combo.findData(unit)
        if cidx >= 0:
            self.etch_conc_unit_combo.setCurrentIndex(cidx)
        conc_val = _safe_float(
            payload.get(
                "concentration_value", payload.get("concentration_wt_pct", 2.0)
            ),
            2.0,
        )
        self.etch_conc_value_spin.setValue(conc_val)
        self._update_etch_risk()

    def _update_etch_risk(self) -> None:
        time_s = float(self.etch_time_spin.value())
        temp_c = float(self.etch_temp_spin.value())
        overetch = float(self.etch_overetch_spin.value())
        conc_unit = str(self.etch_conc_unit_combo.currentData() or "wt_pct")
        conc_value = float(self.etch_conc_value_spin.value())
        if conc_unit == "wt_pct":
            wt = conc_value
            mol = wt * 10.0 / 63.0
        else:
            mol = conc_value
            wt = mol * 63.0 / 10.0
        self.etch_conc_equiv_label.setText(
            f"Эквивалент: {wt:.3f} wt.% / {mol:.3f} mol/L"
        )
        if 0.1 <= wt <= 20.0 and 0.01 <= mol <= 5.0:
            self.etch_conc_status_label.setText("Диапазон: в норме")
            self.etch_conc_status_label.setStyleSheet(
                f"color: {status_color(self.theme_mode, 'success')}; font-weight: 600;"
            )
        else:
            self.etch_conc_status_label.setText("Диапазон: вне рекомендованного")
            self.etch_conc_status_label.setStyleSheet(
                f"color: {status_color(self.theme_mode, 'warning')}; font-weight: 600;"
            )
        score = (
            (time_s / 10.0)
            + max(0.0, (temp_c - 22.0) / 20.0)
            + max(0.0, overetch - 1.0) * 1.4
        )
        score += max(0.0, wt - 2.0) * 0.08
        if score < 1.2:
            text = "Риск травления: низкий (возможен недотрав)"
        elif score < 2.8:
            text = "Риск травления: рабочий диапазон"
        else:
            text = "Риск травления: высокий (возможен перетрав)"
        self.etch_risk_label.setText(text)

    def _compute_brinell_direct_ui(self) -> None:
        try:
            payload = hbw_from_indent(
                load_kgf=float(self.brinell_p_spin.value()),
                ball_d_mm=float(self.brinell_d_ball_spin.value()),
                indent_d_mm=float(self.brinell_d_indent_spin.value()),
            )
            self.brinell_direct_label.setText(f"HBW: {float(payload['HBW']):.1f}")
        except Exception as exc:
            self.brinell_direct_label.setText(f"Ошибка: {exc}")

    def _browse_export_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Выберите папку экспорта", self.export_dir_edit.text().strip()
        )
        if path:
            self.export_dir_edit.setText(path)

    def _save_request_json(self) -> None:
        try:
            request = self._collect_request(final_render=False, for_preview_only=False)
        except Exception as exc:
            QMessageBox.warning(self, "Сохранение запроса", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить запрос V3", "request_v3.json", "JSON (*.json)"
        )
        if not path:
            return
        save_json(request.to_dict(), path)

    def _load_request_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить запрос V3", "", "JSON (*.json)"
        )
        if not path:
            return
        payload = _json_load(Path(path))
        if not isinstance(payload, dict):
            QMessageBox.warning(self, "Загрузка", "Неверный формат JSON.")
            return
        try:
            req = MetallographyRequestV3.from_dict(payload)
            self._apply_request_to_ui(req)
            self._set_status(f"Запрос загружен: {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Загрузка запроса", str(exc))

    def _collect_prep_route(self) -> SamplePrepRouteV3:
        steps = [
            self._prep_row_to_operation(r) for r in range(self.prep_table.rowCount())
        ]
        return SamplePrepRouteV3(
            steps=steps,
            roughness_target_um=float(self.prep_rough_spin.value()),
            relief_mode=str(self.prep_relief_combo.currentData() or "hardness_coupled"),
            contamination_level=float(self.prep_contam_spin.value()),
        )

    def _collect_etch_profile(self) -> EtchProfileV3:
        unit = str(self.etch_conc_unit_combo.currentData() or "wt_pct")
        value = float(self.etch_conc_value_spin.value())
        if unit == "wt_pct":
            wt = value
            mol = wt * 10.0 / 63.0
        else:
            mol = value
            wt = mol * 63.0 / 10.0
        return EtchProfileV3(
            reagent=str(self.etch_reagent_combo.currentData() or "nital_2"),
            time_s=float(self.etch_time_spin.value()),
            temperature_c=float(self.etch_temp_spin.value()),
            agitation=str(self.etch_agitation_combo.currentData() or "gentle"),
            overetch_factor=float(self.etch_overetch_spin.value()),
            concentration_value=float(value),
            concentration_unit=unit,
            concentration_wt_pct=float(wt),
            concentration_mol_l=float(mol),
        )

    def _collect_synthesis_profile(self) -> SynthesisProfileV3:
        return SynthesisProfileV3(
            profile_id=str(
                self.synth_profile_combo.currentData() or "textbook_steel_bw"
            ),
            phase_topology_mode=str(self.synth_topology_combo.currentData() or "auto"),
            system_generator_mode=str(
                self.synth_system_generator_combo.currentData() or "system_auto"
            ),
            contrast_target=float(self.synth_contrast_spin.value()),
            boundary_sharpness=float(self.synth_sharp_spin.value()),
            artifact_level=float(self.synth_artifact_spin.value()),
            composition_sensitivity_mode=str(
                self.synth_sensitivity_combo.currentData() or "realistic"
            ),
            generation_mode=str(
                self.synth_generation_mode_combo.currentData() or "edu_engineering"
            ),
            phase_emphasis_style=str(
                self.synth_phase_emphasis_combo.currentData() or "contrast_texture"
            ),
            phase_fraction_tolerance_pct=float(self.synth_phase_tolerance_spin.value()),
        )

    def _collect_phase_model(self) -> PhaseModelConfigV3:
        return PhaseModelConfigV3(
            engine="explicit_rules_v3",
            phase_control_mode=str(
                self.phase_control_mode_combo.currentData() or "auto_with_override"
            ),
            manual_phase_fractions=self._collect_manual_phase_fractions(),
            manual_override_weight=float(self.phase_override_spin.value()),
            allow_custom_fallback=bool(self.allow_custom_fallback_check.isChecked()),
            phase_balance_tolerance_pct=float(
                self.phase_balance_tolerance_spin.value()
            ),
        )

    def _collect_microscope_profile(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "simulate_preview": bool(self.ms_simulate_check.isChecked()),
            "magnification": int(self.ms_magnification_combo.currentData() or 200),
            "optical_mode": str(
                self.ms_optical_mode_combo.currentData() or "brightfield"
            ),
            "psf_profile": str(self.ms_psf_profile_combo.currentData() or "standard"),
            "psf_strength": float(self.ms_psf_strength_spin.value()),
            "sectioning_shear_deg": float(self.ms_sectioning_shear_spin.value()),
            "hybrid_balance": float(self.ms_hybrid_balance_spin.value()),
            "focus": float(self.ms_focus_spin.value()),
            "brightness": float(self.ms_brightness_spin.value()),
            "contrast": float(self.ms_contrast_spin.value()),
            "noise_sigma": float(self.ms_noise_spin.value()),
            "vignette_strength": float(self.ms_vignette_spin.value()),
            "uneven_strength": float(self.ms_uneven_spin.value()),
            "add_dust": bool(self.ms_dust_check.isChecked()),
            "add_scratches": bool(self.ms_scratch_check.isChecked()),
            "etch_uneven": 0.0,
        }
        if str(self.brinell_mode_combo.currentData() or "estimated") == "direct":
            payload["brinell_direct"] = {
                "P_kgf": float(self.brinell_p_spin.value()),
                "D_mm": float(self.brinell_d_ball_spin.value()),
                "d_mm": float(self.brinell_d_indent_spin.value()),
            }
        return payload

    def _selected_resolution(self) -> tuple[int, int]:
        data = self.resolution_combo.currentData()
        if isinstance(data, tuple) and len(data) == 2:
            return int(data[0]), int(data[1])
        return 1024, 1024

    def _collect_request(
        self, final_render: bool, for_preview_only: bool
    ) -> MetallographyRequestV3:
        composition = self._collect_composition()
        if not composition:
            raise ValueError("Состав пуст. Добавьте хотя бы один элемент.")
        thermal_program = self._collect_thermal_program()
        thermal_check = validate_thermal_program(thermal_program)
        if not bool(thermal_check.get("is_valid", False)):
            raise ValueError(
                "; ".join([str(x) for x in thermal_check.get("errors", [])])
            )
        request = MetallographyRequestV3(
            sample_id=self.sample_id_edit.text().strip() or "sample_v3",
            composition_wt=composition,
            system_hint=(
                None
                if not self.system_hint_combo.currentData()
                else str(self.system_hint_combo.currentData())
            ),
            material_grade=(
                None
                if not self._loaded_preset_context.get("material_grade")
                else str(self._loaded_preset_context.get("material_grade"))
            ),
            material_class_ru=(
                None
                if not self._loaded_preset_context.get("material_class_ru")
                else str(self._loaded_preset_context.get("material_class_ru"))
            ),
            lab_work=(
                None
                if not self._loaded_preset_context.get("lab_work")
                else str(self._loaded_preset_context.get("lab_work"))
            ),
            target_astm_grain_size=(
                None
                if self._loaded_preset_context.get("target_astm_grain_size")
                in (None, "")
                else float(self._loaded_preset_context.get("target_astm_grain_size"))
            ),
            mean_grain_diameter_um=(
                None
                if self._loaded_preset_context.get("mean_grain_diameter_um")
                in (None, "")
                else float(self._loaded_preset_context.get("mean_grain_diameter_um"))
            ),
            expected_properties=dict(
                self._loaded_preset_context.get("expected_properties", {})
            ),
            preset_metadata=dict(
                self._loaded_preset_context.get("preset_metadata", {})
            ),
            thermal_program=thermal_program,
            prep_route=self._collect_prep_route(),
            etch_profile=self._collect_etch_profile(),
            synthesis_profile=self._collect_synthesis_profile(),
            phase_model=self._collect_phase_model(),
            microscope_profile=self._collect_microscope_profile(),
            seed=int(self.seed_spin.value()),
            resolution=self._selected_resolution(),
            strict_validation=bool(self.strict_validation_check.isChecked()),
            reference_profile_id=(
                None
                if not self.reference_combo.currentData()
                else str(self.reference_combo.currentData())
            ),
            generate_intermediate_renders=bool(
                self.generate_intermediate_renders_check.isChecked()
            ),
        )
        if for_preview_only and not final_render:
            h, w = request.resolution
            request.resolution = (min(h, 1024), min(w, 1024))
        return request

    def _validate_route_ui(self) -> None:
        self._validate_thermal_ui()

    def _validate_thermal_ui(self) -> None:
        try:
            program = self._collect_thermal_program()
            validation = validate_thermal_program(program)
            summary = summarize_thermal_program(program)
            _, runtime_summary, quench_summary = effective_processing_from_thermal(
                program
            )
            op_inf = dict(runtime_summary.get("operation_inference", {}))
            medium_code = str(
                quench_summary.get(
                    "medium_code_resolved", quench_summary.get("medium_code", "")
                )
            )
            shift_c = float(op_inf.get("recommended_temper_shift_c", 0.0))
            has_quench_curve = bool(op_inf.get("has_quench", False))
            quench_effect_applied = bool(
                runtime_summary.get("quench_effect_applied", has_quench_curve)
            )
            quench_effect_reason = str(runtime_summary.get("quench_effect_reason", ""))
            quench_profile_media = {
                "water_20",
                "water_100",
                "brine_20_30",
                "oil_20_80",
                "polymer",
                "custom",
            }
            lines = [
                f"Точки: {summary.get('point_count', 0)}",
                f"Длительность: {summary.get('duration_s', 0.0):.1f} c",
                f"Tmin/Tmax: {summary.get('temperature_min_c', 0.0):.1f} / {summary.get('temperature_max_c', 0.0):.1f} °C",
                f"Скорость охлаждения (макс): {summary.get('max_cooling_rate_c_per_s', 0.0):.3f} °C/c",
                f"Среда закалки: {_label_ru(medium_code, QUENCH_MEDIUM_LABELS_RU)}",
                f"Прогноз HRC после закалки: {quench_summary.get('hardness_hrc_as_quenched_range', '')}",
                f"Остаточный аустенит (оценка): {quench_summary.get('as_quenched_prediction', {}).get('retained_austenite_fraction_est', '')}",
                f"Рекомендованный сдвиг отпуска: +{shift_c:.1f} °C",
                f"Закалка по кривой: {'да' if has_quench_curve else 'нет'}",
                f"Влияние среды закалки: {'применяется' if quench_effect_applied else 'не применяется'} ({quench_effect_reason})",
                f"Валидность: {'OK' if validation.get('is_valid', False) else 'ОШИБКА'}",
                "",
                "Ошибки:",
            ]
            if (not quench_effect_applied) and medium_code in quench_profile_media:
                lines.extend(
                    [
                        "",
                        "ПРЕДУПРЕЖДЕНИЕ:",
                        "Среда закалки выбрана, но на кривой нет закалочного сегмента — влияние среды не применяется.",
                    ]
                )
            errs = list(validation.get("errors", []))
            warns = list(validation.get("warnings", []))
            lines.extend([f"- {e}" for e in errs] if errs else ["- нет"])
            lines.append("")
            lines.append("Предупреждения:")
            lines.extend([f"- {w}" for w in warns] if warns else ["- нет"])
            self.route_validation_text.setPlainText("\n".join(lines))
        except Exception as exc:
            self.route_validation_text.setPlainText(
                f"Ошибка проверки термопрограммы:\n{exc}"
            )

    def _check_phase_model_resolution(self) -> None:
        composition = self._collect_composition()
        if not composition:
            self.coverage_label.setText("Состав пуст")
            return
        system, confidence, is_fallback = infer_training_system(
            composition=composition,
            system_hint=str(self.system_hint_combo.currentData() or ""),
        )
        text = (
            f"Система: {_label_ru(system, SYSTEM_LABELS_RU)} | достоверность={confidence:.2f}"
            + (" | резервный режим" if is_fallback else "")
        )
        self.coverage_label.setText(text)
        self.route_validation_text.setPlainText(text)

    def _on_topology_mode_changed(self) -> None:
        _ = str(self.synth_topology_combo.currentData() or "auto")

    def _on_step_changed(self, index: int) -> None:
        if index < 0:
            return
        self.step_stack.setCurrentIndex(index)
        self._animate_step_transition(index)
        self._update_step_status()

    def _update_step_status(self) -> None:
        current = self.step_stack.currentIndex() + 1
        total = self.step_stack.count()
        self.step_status_label.setText(f"Шаг {current} из {total}")
        if hasattr(self, "step_progress_bar"):
            self.step_progress_bar.setMaximum(max(1, total))
            self.step_progress_bar.setValue(max(1, current))
            self.step_progress_bar.setFormat(f"{current}/{total}")
        self.btn_prev_step.setEnabled(current > 1)
        self.btn_next_step.setEnabled(current < total)

    def _apply_responsive_layout(self) -> None:
        if not hasattr(self, "main_split"):
            return
        total_w = max(1040, int(self.width()))
        left_w = int(max(210, min(300, total_w * 0.18)))
        right_w = int(max(360, min(520, total_w * 0.30)))
        center_w = max(400, total_w - left_w - right_w - 32)
        self.main_split.setSizes([left_w, center_w, right_w])

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_responsive_layout()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
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

    def _animate_step_transition(self, index: int) -> None:
        if index < 0 or index >= self.step_stack.count():
            return
        page = self.step_stack.widget(index)
        if page is None:
            return
        effect = QGraphicsOpacityEffect(page)
        page.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", page)
        anim.setDuration(180)
        anim.setStartValue(0.18)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        def _cleanup() -> None:
            try:
                page.setGraphicsEffect(None)
            except Exception:
                pass
            try:
                self._animations.remove(anim)
            except Exception:
                pass

        anim.finished.connect(_cleanup)
        anim.start()
        self._animations.append(anim)

    def _go_prev_step(self) -> None:
        cur = self.step_stack.currentIndex()
        if cur > 0:
            self.step_list.setCurrentRow(cur - 1)

    def _go_next_step(self) -> None:
        cur = self.step_stack.currentIndex()
        if cur + 1 < self.step_stack.count():
            self.step_list.setCurrentRow(cur + 1)

    def _set_status(self, text: str) -> None:
        self.coverage_label.setText(text)

    def _render_placeholder(self, text: str) -> None:
        self.preview_scene.clear()
        lbl = self.preview_scene.addText(text)
        lbl.setDefaultTextColor(Qt.GlobalColor.lightGray)
        self.info_text.setPlainText("")
        self.qc_text.setPlainText("")
        self.textbook_readability_label.setText("Учебная читаемость фаз: нет данных")
        self.textbook_readability_label.setStyleSheet(
            f"color: {status_color(self.theme_mode, 'text_secondary')}; font-weight: 600;"
        )

    def _generate_preview(self) -> None:
        self._run_generation(final_render=False)

    def _generate_final(self) -> None:
        self._run_generation(final_render=True)

    def _run_generation(self, final_render: bool) -> None:
        try:
            self.setCursor(Qt.CursorShape.WaitCursor)
            request = self._collect_request(
                final_render=final_render, for_preview_only=True
            )
            self.current_request = request
            output = self.pipeline.generate(request)
            self.current_output = output
            self.is_final_render = final_render  # Store for _show_output
            self._show_output(output)
            self.btn_export_package.setEnabled(True)
            render_mode = "финальный рендер" if final_render else "предпросмотр"
            inferred = _label_ru(
                str(output.metadata.get("inferred_system", "")), SYSTEM_LABELS_RU
            )
            self._set_status(f"Готово: {render_mode}, {inferred}")
        except Exception as exc:
            QMessageBox.critical(self, "Генерация", str(exc))
            self._set_status(f"Ошибка: {exc}")
        finally:
            self.unsetCursor()

    def _show_output(self, output: GenerationOutputV3) -> None:
        self.preview_scene.clear()
        preview_rgb = _preview_image_rgb(output.image_rgb, max_side=3072)
        pix = _to_pixmap(preview_rgb)
        self.preview_scene.addPixmap(pix)
        self.preview_scene.setSceneRect(pix.rect())
        self.preview_view.resetTransform()
        self.preview_view.fitInView(
            self.preview_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

        meta = dict(output.metadata)

        # Simplified view for final render in student mode only.
        if (
            hasattr(self, "is_final_render")
            and self.is_final_render
            and self.current_mode == "student"
        ):
            self._show_intermediate_renders([])
            self._show_simplified_output(output)
            return

        # Отображение промежуточных рендеров
        self._show_intermediate_renders(output.intermediate_renders)

        effect = meta.get("composition_effect", {})
        phase_model = meta.get("phase_model", {})
        phase_model_report = meta.get("phase_model_report", {})
        system_resolution = meta.get("system_resolution", {})
        system_generator = meta.get("system_generator", {})
        textbook_profile = meta.get("textbook_profile", {})
        stable = {}
        if isinstance(phase_model_report, dict):
            stable = phase_model_report.get("blended_phase_fractions", {})
        quality = meta.get("quality_metrics", {})
        prep_summary = meta.get("prep_summary", {})
        props = meta.get("property_indicators", {})
        lines = [
            f"Образец: {meta.get('sample_id', '')}",
            f"Система: {_label_ru(str(meta.get('inferred_system', '')), SYSTEM_LABELS_RU)}",
            f"Финальная стадия: {meta.get('final_stage', '')}",
        ]
        hi = meta.get("high_resolution_render", {})
        if isinstance(hi, dict):
            req_res = hi.get("requested_resolution")
            int_res = hi.get("internal_resolution")
            if isinstance(req_res, (list, tuple)) and len(req_res) == 2:
                lines.append(f"Разрешение (запрошено): {req_res[0]} x {req_res[1]}")
            if isinstance(int_res, (list, tuple)) and len(int_res) == 2:
                lines.append(
                    f"Разрешение (внутренний рендер): {int_res[0]} x {int_res[1]}"
                )
            if bool(hi.get("enabled", False)):
                lines.append(f"Режим high-res: апскейл {hi.get('upscale_method', '')}")
        lines.extend(
            [
                "",
                "Состав-зависимый эффект:",
                f"- режим: {_label_ru(str(effect.get('mode', '')), SENSITIVITY_LABELS_RU)}",
                f"- индекс легирования: {effect.get('solute_index', '')}",
                f"- хеш состава: {effect.get('composition_hash', '')}",
                f"- сдвиг seed: {effect.get('seed_offset', '')}",
                f"- компенсация однофазного режима: {_yes_no(effect.get('single_phase_compensation', False))}",
                "",
                "Фазовая модель:",
                f"- движок: {phase_model.get('engine', '')}",
                f"- режим фаз: {_label_ru(str(phase_model.get('phase_control_mode', '')), PHASE_CONTROL_MODE_LABELS_RU)}",
                f"- вес ручной коррекции: {phase_model.get('override_weight', '')}",
                f"- разрешен резервный режим: {_yes_no(phase_model.get('allow_custom_fallback', ''))}",
                f"- стадия: {system_resolution.get('stage', meta.get('final_stage', ''))}",
                f"- достоверность: {system_resolution.get('confidence', '')}",
                "- устойчивые фазы:",
            ]
        )
        if isinstance(system_generator, dict):
            lines.extend(
                [
                    "",
                    "Системный генератор:",
                    f"- запрошенный режим: {_label_ru(str(system_generator.get('requested_mode', '')), SYSTEM_GENERATOR_MODE_LABELS_RU)}",
                    f"- выбранный режим: {_label_ru(str(system_generator.get('resolved_mode', '')), SYSTEM_GENERATOR_MODE_LABELS_RU)}",
                    f"- выбранная система: {_label_ru(str(system_generator.get('resolved_system', '')), SYSTEM_LABELS_RU)}",
                    f"- выбранная стадия: {system_generator.get('resolved_stage', '')}",
                    f"- использован резервный режим: {_yes_no(system_generator.get('fallback_used', False))}",
                    f"- причина выбора: {system_generator.get('selection_reason', '')}",
                    f"- достоверность: {system_generator.get('confidence', '')}",
                ]
            )
            fe_c_unified = system_generator.get("fe_c_unified", {})
            if isinstance(fe_c_unified, dict) and fe_c_unified:
                lines.extend(
                    [
                        "- Fe-C unified:",
                        f"  - включен: {_yes_no(fe_c_unified.get('enabled', False))}",
                        f"  - покрытие стадий: {_yes_no(fe_c_unified.get('stage_coverage_pass', False))}",
                        f"  - стадия: {fe_c_unified.get('resolved_stage', '')}",
                        f"  - режим смешивания: {fe_c_unified.get('blending_mode', '')}",
                        f"  - причина fallback: {fe_c_unified.get('fallback_reason', '')}",
                    ]
                )
        if isinstance(stable, dict) and stable:
            for name, frac in sorted(
                stable.items(), key=lambda item: item[1], reverse=True
            ):
                lines.append(f"  - {name}: {float(frac):.4f}")
        else:
            lines.append("  - нет данных")
        fe_c_render = meta.get("fe_c_phase_render", {})
        if isinstance(fe_c_render, dict) and fe_c_render:
            rendered_layers = fe_c_render.get("rendered_phase_layers", [])
            if isinstance(rendered_layers, list) and rendered_layers:
                lines.append("- реально отрисованные фазы:")
                for layer in rendered_layers:
                    lines.append(f"  - {layer}")
        lines.extend(
            [
                "",
                "Оценочные свойства:",
                f"- HV: {props.get('hv_estimate', props.get('hardness_hv_est', ''))}",
                f"- UTS, MPa: {props.get('uts_estimate_mpa', props.get('uts_mpa_est', ''))}",
                f"- Пластичность: {props.get('ductility_class', '')}",
                "",
                "Контроль качества:",
                f"- пройдено: {_yes_no(quality.get('passed', ''))}",
                f"- динамический диапазон p05-p95: {quality.get('dynamic_range_p05_p95', '')}",
            ]
        )
        if isinstance(prep_summary, dict) and bool(
            prep_summary.get("implicit_baseline_route_applied", False)
        ):
            lines.append(
                "- базовый маршрут подготовки применен автоматически для realistic-рендера"
            )
        if isinstance(textbook_profile, dict):
            readability_targets = textbook_profile.get("readability_targets", {})
            achieved_readability = textbook_profile.get("achieved_readability", {})
            target_const = textbook_profile.get("target_microconstituents", [])
            if isinstance(target_const, list):
                target_const_str = ", ".join(str(x) for x in target_const)
            else:
                target_const_str = str(target_const)
            lines.extend(
                [
                    "",
                    "Учебниковый профиль:",
                    f"- профиль: {_label_ru(str(textbook_profile.get('profile_id', '')), SYNTH_PROFILE_LABELS_RU)}",
                    f"- соответствие: {_yes_no(textbook_profile.get('pass', False))}",
                    f"- целевые структурные составляющие: {target_const_str}",
                    f"- цель по разделимости >= {readability_targets.get('separability_score_min', '')}",
                    f"- цель по динамическому диапазону >= {readability_targets.get('dynamic_range_min', '')}",
                    f"- достигнутая разделимость: {achieved_readability.get('separability_score', '')}",
                    f"- достигнутый динамический диапазон: {achieved_readability.get('dynamic_range_p05_p95', '')}",
                ]
            )
        visibility = meta.get("phase_visibility_report", {})
        trace = meta.get("engineering_trace", {})
        if isinstance(trace, dict):
            lines.extend(
                [
                    "",
                    "Параметры режима генерации:",
                    f"- режим: {_label_ru(str(trace.get('generation_mode', '')), GENERATION_MODE_LABELS_RU)}",
                    f"- стиль выраженности фаз: {_label_ru(str(trace.get('phase_emphasis_style', '')), PHASE_EMPHASIS_LABELS_RU)}",
                    f"- допуск долей фаз, %: {trace.get('phase_fraction_tolerance_pct', '')}",
                ]
            )
        continuous_state = meta.get("continuous_transformation_state", {})
        if isinstance(continuous_state, dict) and continuous_state:
            lines.extend(
                [
                    "",
                    "Continuous transformation state:",
                    f"- family: {continuous_state.get('transformation_family', '')}",
                    f"- growth mode: {continuous_state.get('growth_mode', '')}",
                    f"- partitioning: {continuous_state.get('partitioning_mode', '')}",
                    f"- prior-austenite grain, мкм: {_safe_float(continuous_state.get('prior_austenite_grain_size_um', 0.0), 0.0):.3f}",
                    f"- colony size mean, мкм: {_safe_float(continuous_state.get('colony_size_um_mean', 0.0), 0.0):.3f}",
                    f"- interlamellar spacing mean, мкм: {_safe_float(continuous_state.get('interlamellar_spacing_um_mean', 0.0), 0.0):.3f}",
                    f"- martensite packet, мкм: {_safe_float(continuous_state.get('martensite_packet_size_um', 0.0), 0.0):.3f}",
                    f"- bainite sheaf length, мкм: {_safe_float(continuous_state.get('bainite_sheaf_length_um', 0.0), 0.0):.3f}",
                    f"- recovery level: {_safe_float(continuous_state.get('recovery_level', 0.0), 0.0):.3f}",
                    f"- ferrite effective exposure, c: {_safe_float(continuous_state.get('ferrite_effective_exposure_s', 0.0), 0.0):.1f}",
                    f"- pearlite effective exposure, c: {_safe_float(continuous_state.get('pearlite_effective_exposure_s', 0.0), 0.0):.1f}",
                    f"- bainite effective exposure, c: {_safe_float(continuous_state.get('bainite_effective_exposure_s', 0.0), 0.0):.1f}",
                    f"- martensite effective exposure, c: {_safe_float(continuous_state.get('martensite_effective_exposure_s', 0.0), 0.0):.1f}",
                    f"- ferrite progress: {_safe_float(continuous_state.get('ferrite_progress', 0.0), 0.0):.3f}",
                    f"- pearlite progress: {_safe_float(continuous_state.get('pearlite_progress', 0.0), 0.0):.3f}",
                    f"- ferrite/pearlite competition: {_safe_float(continuous_state.get('ferrite_pearlite_competition_index', 0.0), 0.0):.3f}",
                    f"- bainite activation: {_safe_float(continuous_state.get('bainite_activation_progress', 0.0), 0.0):.3f}",
                    f"- martensite conversion: {_safe_float(continuous_state.get('martensite_conversion_progress', 0.0), 0.0):.3f}",
                ]
            )
        validation_pro = meta.get("validation_pro", {})
        if isinstance(validation_pro, dict) and validation_pro:
            lines.extend(
                [
                    "",
                    "Pro validation:",
                    f"- ASTM grain number proxy: {_safe_float(validation_pro.get('grain_size_astm_number_proxy', 0.0), 0.0):.3f}",
                    f"- mean lineal intercept X, мкм: {_safe_float(validation_pro.get('mean_lineal_intercept_um_x', 0.0), 0.0):.3f}",
                    f"- mean lineal intercept Y, мкм: {_safe_float(validation_pro.get('mean_lineal_intercept_um_y', 0.0), 0.0):.3f}",
                    f"- 2-point correlation score: {_safe_float(validation_pro.get('two_point_corr_score', 0.0), 0.0):.3f}",
                    f"- PSD score: {_safe_float(validation_pro.get('psd_score', 0.0), 0.0):.3f}",
                    f"- directional artifact anisotropy: {_safe_float(validation_pro.get('directional_artifact_anisotropy_score', 0.0), 0.0):.3f}",
                    f"- scratch revelation risk: {_safe_float(validation_pro.get('scratch_trace_revelation_risk', 0.0), 0.0):.3f}",
                    f"- false porosity pull-out risk: {_safe_float(validation_pro.get('false_porosity_pullout_risk', 0.0), 0.0):.3f}",
                    f"- surface Ra, мкм: {_safe_float(validation_pro.get('surface_roughness_ra_um', 0.0), 0.0):.4f}",
                ]
            )
            diagnostics = validation_pro.get("artifact_diagnostics", [])
            if isinstance(diagnostics, list) and diagnostics:
                lines.extend(["", "Artifact diagnostics:"])
                for item in diagnostics[:6]:
                    lines.append(f"- {item}")
            artifact_bundle = validation_pro.get("artifact_risk_scores", {})
            if isinstance(artifact_bundle, dict) and artifact_bundle:
                lines.extend(
                    [
                        "",
                        "Artifact risk bundle:",
                        f"- dominant driver: {artifact_bundle.get('dominant_driver', '')}",
                        f"- dominant trigger ratio: {_safe_float(artifact_bundle.get('dominant_trigger_ratio', 0.0), 0.0):.3f}",
                        f"- triggered count: {int(artifact_bundle.get('triggered_count', 0) or 0)}",
                    ]
                )
        reflected_light = meta.get("reflected_light_model", {})
        if isinstance(reflected_light, dict) and reflected_light:
            lines.extend(
                [
                    "",
                    "Reflected-light model:",
                    f"- optical mode: {_label_ru(str(reflected_light.get('optical_mode', '')), OPTICAL_MODE_LABELS_RU)}",
                    f"- PSF profile: {_label_ru(str(reflected_light.get('psf_profile', 'standard')), PSF_PROFILE_LABELS_RU)}",
                    f"- PSF strength: {_safe_float(reflected_light.get('psf_strength', 0.0), 0.0):.3f}",
                    f"- effective DOF factor: {_safe_float(reflected_light.get('effective_dof_factor', 1.0), 1.0):.3f}",
                    f"- lamella modulation: {'да' if bool(reflected_light.get('lamella_modulation_applied', False)) else 'нет'}",
                    f"- bainite modulation: {'да' if bool(reflected_light.get('bainite_modulation_applied', False)) else 'нет'}",
                    f"- widmanstätten modulation: {'да' if bool(reflected_light.get('widmanstatten_modulation_applied', False)) else 'нет'}",
                    f"- packet modulation: {'да' if bool(reflected_light.get('packet_modulation_applied', False)) else 'нет'}",
                ]
            )
            if bool(reflected_light.get("sectioning_active", False)):
                lines.append(
                    f"- sectioning suppression: {_safe_float(reflected_light.get('sectioning_suppression_score', 0.0), 0.0):.3f}"
                )
            if str(reflected_light.get("optical_mode", "")) == "dic":
                lines.extend(
                    [
                        f"- DIC shear axis, deg: {_safe_float(reflected_light.get('dic_shear_axis_deg', 0.0), 0.0):.1f}",
                        f"- DIC signal std: {_safe_float(reflected_light.get('dic_signal_std', 0.0), 0.0):.4f}",
                    ]
                )
            if str(reflected_light.get("optical_mode", "")) == "magnetic_etching":
                lines.extend(
                    [
                        f"- magnetic field active: {_yes_no(reflected_light.get('magnetic_field_active', False))}",
                        f"- ferromagnetic fraction: {_safe_float(reflected_light.get('ferromagnetic_fraction', 0.0), 0.0):.3f}",
                        f"- magnetic signal fraction: {_safe_float(reflected_light.get('magnetic_signal_fraction', 0.0), 0.0):.3f}",
                    ]
                )
        electron_guidance = meta.get("electron_microscopy_guidance", {})
        if isinstance(electron_guidance, dict) and electron_guidance:
            sem = dict(electron_guidance.get("sem_guidance", {}))
            tem = dict(electron_guidance.get("tem_guidance", {}))
            why = list(electron_guidance.get("why", []))
            lines.extend(
                [
                    "",
                    "Electron microscopy guidance:",
                    f"- primary: {electron_guidance.get('primary_recommendation', '')}",
                    f"- SEM mode: {sem.get('preferred_mode', 'none')}",
                    f"- TEM candidate: {'yes' if bool(tem.get('recommended', False)) else 'no'}",
                ]
            )
            for reason in why[:3]:
                lines.append(f"- why: {reason}")
            if bool(sem.get("avoid_etching_for_material_contrast", False)):
                lines.append(
                    "- warning: for BSE material contrast etching is better avoided"
                )
            if bool(sem.get("deformation_free_flat_surface_required", False)):
                lines.append(
                    "- warning: channeling/orientation-sensitive work needs a deformation-free flat surface"
                )
            if bool(sem.get("coating_may_help", False)):
                lines.append(
                    "- warning: coating may help for charging/low-yield or semiconductive cases"
                )
        if isinstance(phase_model_report, dict):
            auto = phase_model_report.get("auto_phase_fractions", {})
            manual = phase_model_report.get("manual_phase_fractions", {})
            blended = phase_model_report.get("blended_phase_fractions", {})
            calibration_mode = str(phase_model_report.get("calibration_mode", ""))
            calibration_profile = str(phase_model_report.get("calibration_profile", ""))
            calibration_source = str(phase_model_report.get("calibration_source", ""))
            table_match_error_pct = _safe_float(
                phase_model_report.get("table_match_error_pct", 0.0), 0.0
            )
            lines.extend(
                [
                    "",
                    "Фазовая коррекция:",
                    f"- коррекция применена: {_yes_no(phase_model_report.get('blend_applied', False))}",
                    f"- в пределах допуска: {_yes_no(phase_model_report.get('within_tolerance', True))}",
                    f"- допуск баланса фаз, %: {phase_model_report.get('phase_balance_tolerance_pct', '')}",
                    f"- использован резервный режим: {_yes_no(phase_model_report.get('fallback_used', False))}",
                    f"- причина резерва: {phase_model_report.get('fallback_reason', '')}",
                ]
            )
            if calibration_mode:
                lines.extend(
                    [
                        "",
                        "Калибровка Fe-C:",
                        f"- профиль калибровки: {calibration_profile or 'не задан'}",
                        f"- режим: {calibration_mode}",
                        f"- источник: {calibration_source or 'не задан'}",
                        f"- отклонение от таблицы, %: {table_match_error_pct:.2f}",
                    ]
                )
            if isinstance(auto, dict) and auto:
                lines.append("- авто:")
                for phase_name, val in sorted(
                    auto.items(), key=lambda item: item[1], reverse=True
                ):
                    lines.append(f"  - {phase_name}: {float(val):.4f}")
            if isinstance(manual, dict) and manual:
                lines.append("- ручной:")
                for phase_name, val in sorted(
                    manual.items(), key=lambda item: item[1], reverse=True
                ):
                    lines.append(f"  - {phase_name}: {float(val):.4f}")
            if isinstance(blended, dict) and blended:
                lines.append("- итоговый:")
                for phase_name, val in sorted(
                    blended.items(), key=lambda item: item[1], reverse=True
                ):
                    lines.append(f"  - {phase_name}: {float(val):.4f}")
        if isinstance(visibility, dict):
            target = visibility.get("target_phase_fractions", {})
            actual = visibility.get("achieved_phase_fractions", {})
            err = visibility.get("fraction_error_pct", {})
            lines.extend(
                [
                    "",
                    "Отчет читаемости фаз:",
                    f"- в пределах допуска: {_yes_no(visibility.get('within_tolerance', ''))}",
                    f"- индекс разделимости: {visibility.get('separability_score', '')}",
                ]
            )
            if isinstance(target, dict) and target:
                lines.append("- целевые доли:")
                for phase_name, val in sorted(
                    target.items(), key=lambda item: item[1], reverse=True
                ):
                    lines.append(f"  - {phase_name}: {float(val):.4f}")
            if isinstance(actual, dict) and actual:
                lines.append("- фактические доли:")
                for phase_name, val in sorted(
                    actual.items(), key=lambda item: item[1], reverse=True
                ):
                    lines.append(f"  - {phase_name}: {float(val):.4f}")
            if isinstance(err, dict) and err:
                lines.append("- ошибка, %:")
                for phase_name, val in sorted(
                    err.items(), key=lambda item: item[1], reverse=True
                ):
                    lines.append(f"  - {phase_name}: {float(val):.2f}")
        thermal = meta.get("thermal_program_summary", {})
        quench = meta.get("quench_summary", {})
        quench_profile = meta.get("quench_medium_profile", {})
        temper_adjustment = meta.get("temper_adjustment", {})
        as_quenched_pred = meta.get("as_quenched_prediction", {})
        operation_guidance = meta.get("operation_guidance", {})
        inferred_ops = meta.get("operations_from_curve", {})
        brinell = meta.get("brinell", {})
        if isinstance(thermal, dict):
            lines.extend(
                [
                    "",
                    "Термопрограмма:",
                    f"- точек: {thermal.get('point_count', '')}",
                    f"- длительность, c: {thermal.get('duration_s', '')}",
                    f"- Tmin/Tmax, °C: {thermal.get('temperature_min_c', '')} / {thermal.get('temperature_max_c', '')}",
                    f"- макс. скорость охлаждения, °C/c: {thermal.get('max_effective_cooling_rate_c_per_s', thermal.get('max_cooling_rate_c_per_s', ''))}",
                    f"- закалка применена: {_yes_no(thermal.get('quench_effect_applied', False))}",
                    f"- причина: {thermal.get('quench_effect_reason', '')}",
                ]
            )
        if isinstance(quench, dict):
            medium_code = str(
                quench.get("medium_code_resolved", quench.get("medium_code", ""))
            )
            lines.extend(
                [
                    "",
                    "Закалка:",
                    f"- среда: {_label_ru(medium_code, QUENCH_MEDIUM_LABELS_RU)} ({medium_code})",
                    f"- интенсивность: {quench.get('severity_effective', '')}",
                    f"- T ванны / образца: {quench.get('bath_temperature_c', '')} / {quench.get('sample_temperature_c', '')}",
                    f"- влияние применено: {_yes_no(quench.get('effect_applied', False))}",
                    f"- причина: {quench.get('effect_reason', '')}",
                ]
            )
            if not bool(quench.get("effect_applied", False)):
                lines.append(
                    "  - предупреждение: среда закалки игнорируется, т.к. на кривой нет quench-сегмента"
                )
        if isinstance(quench_profile, dict) and quench_profile:
            lines.extend(
                [
                    "",
                    "Прогноз закалки по среде:",
                    f"- HRC после закалки: {quench_profile.get('hardness_hrc_as_quenched_range', '')}",
                    f"- напряжения, МПа: {quench_profile.get('stress_mpa_range', '')}",
                    f"- глубина прокаливания, мм: {quench_profile.get('harden_depth_mm_range', '')}",
                    f"- скорость 800-400°C, °C/с: {quench_profile.get('cooling_rate_band_800_400', '')}",
                    f"- риск дефектов: {quench_profile.get('defect_risk', '')}",
                ]
            )
        if isinstance(as_quenched_pred, dict) and as_quenched_pred:
            lines.extend(
                [
                    f"- мартенсит (оценка): {as_quenched_pred.get('martensite_fraction_est', '')}",
                    f"- остаточный аустенит (оценка): {as_quenched_pred.get('retained_austenite_fraction_est', '')}",
                    f"- тип мартенсита: {as_quenched_pred.get('martensite_type', '')}",
                ]
            )
        if isinstance(temper_adjustment, dict) and temper_adjustment:
            lines.extend(
                [
                    "",
                    "Корректировка отпусков по среде:",
                    f"- примененный сдвиг, °C: {temper_adjustment.get('shift_c', 0.0)}",
                    f"- профиль среды: {temper_adjustment.get('source_medium', '')}",
                    f"- карта сдвигов (low/medium/high): {temper_adjustment.get('applied_to_ranges', {})}",
                ]
            )
        if isinstance(operation_guidance, dict) and operation_guidance:
            lines.extend(
                [
                    "",
                    "Рекомендации:",
                    f"- низкий отпуск обязателен: {_yes_no(operation_guidance.get('low_temper_required', False))}",
                    f"- рекомендуемая выдержка отпуска, с: {operation_guidance.get('recommended_hold_s', '')}",
                ]
            )
        if isinstance(inferred_ops, dict):
            inferred_list = inferred_ops.get("operations", [])
            inferred_summary = inferred_ops.get("summary", {})
            inferred_summary_dict = (
                inferred_summary if isinstance(inferred_summary, dict) else {}
            )
            lines.extend(
                [
                    "",
                    "Операции, определенные по кривой:",
                    f"- всего: {inferred_summary_dict.get('count', 0)}",
                    f"- есть аустенизация: {_yes_no(inferred_summary_dict.get('has_austenitization', False))}",
                    f"- есть закалка: {_yes_no(inferred_summary_dict.get('has_quench', False))}",
                    f"- есть отпуск: {_yes_no(inferred_summary_dict.get('has_temper', False))}",
                ]
            )
            if isinstance(inferred_list, list) and inferred_list:
                max_rows = min(8, len(inferred_list))
                lines.append("- список:")
                for op in inferred_list[:max_rows]:
                    if not isinstance(op, dict):
                        continue
                    seg_raw = op.get("segment", {})
                    seg = seg_raw if isinstance(seg_raw, dict) else {}
                    lines.append(
                        "  - "
                        + f"{op.get('label_ru', op.get('code', ''))}: "
                        + f"t={seg.get('t0_s', '')}-{seg.get('t1_s', '')} c, "
                        + f"T={seg.get('temp0_c', '')}->{seg.get('temp1_c', '')} °C"
                    )
        if isinstance(brinell, dict):
            est = brinell.get("estimated", {})
            direct = brinell.get("direct", {})
            lines.extend(["", "Бринелль (HBW):"])
            if isinstance(est, dict):
                lines.append(f"- оценочно из структуры: {est.get('HBW', '')}")
            if isinstance(direct, dict) and direct:
                lines.append(
                    f"- прямой расчет: {direct.get('HBW', direct.get('error', ''))}"
                )
        self.info_text.setPlainText("\n".join(lines))
        self.qc_text.setPlainText(
            json.dumps(output.metadata_json_safe(), ensure_ascii=False, indent=2)
        )
        if isinstance(textbook_profile, dict):
            passed = bool(textbook_profile.get("pass", False))
            achieved = textbook_profile.get("achieved_readability", {})
            sep = _safe_float(achieved.get("separability_score", 0.0), 0.0)
            dyn = _safe_float(achieved.get("dynamic_range_p05_p95", 0.0), 0.0)
            self.textbook_readability_label.setText(
                f"Учебная читаемость фаз: {'В норме' if passed else 'Ниже цели'} | разделимость={sep:.3f}, диапазон={dyn:.1f}"
            )
            self.textbook_readability_label.setStyleSheet(
                f"color: {status_color(self.theme_mode, 'success')}; font-weight: 700;"
                if passed
                else f"color: {status_color(self.theme_mode, 'warning')}; font-weight: 700;"
            )
        else:
            self.textbook_readability_label.setText(
                "Учебная читаемость фаз: нет данных"
            )
            self.textbook_readability_label.setStyleSheet(
                f"color: {status_color(self.theme_mode, 'text_secondary')}; font-weight: 600;"
            )

    def _format_structure_stage_label(self, stage: str) -> str:
        code = str(stage or "").strip().lower()
        if not code:
            return ""
        return STRUCTURE_STAGE_LABELS_RU.get(code, code.replace("_", "-"))

    def _resolve_student_structure_name(self, output: GenerationOutputV3) -> str:
        meta = dict(output.metadata)
        candidates = [
            meta.get("final_stage"),
            meta.get("resolved_stage"),
            (meta.get("system_generator", {}) or {}).get("resolved_stage")
            if isinstance(meta.get("system_generator"), dict)
            else None,
        ]
        for candidate in candidates:
            label = self._format_structure_stage_label(str(candidate or ""))
            if label:
                return label

        phase_model_report = meta.get("phase_model_report", {})
        phase_fractions = (
            phase_model_report.get("blended_phase_fractions", {})
            if isinstance(phase_model_report, dict)
            else {}
        )
        if isinstance(phase_fractions, dict):
            dominant: list[str] = []
            for phase_name, fraction in sorted(
                phase_fractions.items(), key=lambda item: float(item[1]), reverse=True
            ):
                frac = _safe_float(fraction, 0.0)
                if frac < 0.12:
                    continue
                label = PHASE_NAME_LABELS_RU.get(str(phase_name).upper())
                if label and label not in dominant:
                    dominant.append(label)
                if len(dominant) == 2:
                    break
            if dominant:
                if len(dominant) == 1:
                    return dominant[0]
                return f"{dominant[0]}-{dominant[1].lower()}"
        return "не определено"

    def _show_simplified_output(self, output: GenerationOutputV3) -> None:
        """Упрощенный вывод для студентов после финального рендера."""
        meta = dict(output.metadata)

        # Разрешение
        hi = meta.get("high_resolution_render", {})
        resolution_str = "не указано"
        if isinstance(hi, dict):
            req_res = hi.get("requested_resolution")
            if isinstance(req_res, (list, tuple)) and len(req_res) == 2:
                resolution_str = f"{req_res[0]} x {req_res[1]}"

        structure_name = self._resolve_student_structure_name(output)

        # Твердость
        hardness_str = "не определена"
        props = meta.get("property_indicators", {})
        brinell = meta.get("brinell", {})

        # Приоритет: HV из property_indicators, затем HBW из brinell
        hv_value = props.get("hv_estimate", props.get("hardness_hv_est", ""))
        if hv_value:
            hardness_str = f"HV {hv_value}"
        else:
            est = brinell.get("estimated", {})
            if isinstance(est, dict):
                hbw_value = est.get("HBW", "")
                if hbw_value:
                    hardness_str = f"HBW {hbw_value}"

        lines = [
            f"Название: {structure_name}",
            f"Разрешение: {resolution_str}",
            f"Твердость: {hardness_str}",
        ]

        self.info_text.setPlainText("\n".join(lines))
        self.qc_text.setPlainText("Служебные данные скрыты в режиме студента.")

    def _show_intermediate_renders(self, intermediate_renders: list) -> None:
        """Отображает промежуточные рендеры в горизонтальной галерее."""
        # Очищаем предыдущие рендеры
        while self.intermediate_container_layout.count():
            item = self.intermediate_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not intermediate_renders:
            self.intermediate_renders_group.setVisible(False)
            return

        self.intermediate_renders_group.setVisible(True)

        for render in intermediate_renders:
            # Создаем карточку для каждого промежуточного рендера
            card = QFrame()
            card.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
            card.setLineWidth(1)
            card.setMinimumWidth(150)
            card.setMaximumWidth(200)
            card.setCursor(Qt.CursorShape.PointingHandCursor)

            card_layout = QVBoxLayout(card)
            card_layout.setSpacing(4)
            card_layout.setContentsMargins(4, 4, 4, 4)

            # Превью изображения
            thumbnail_size = 140
            thumbnail_rgb = _preview_image_rgb(
                render.image_rgb, max_side=thumbnail_size
            )
            thumbnail_pix = _to_pixmap(thumbnail_rgb)
            thumbnail_label = QLabel()
            thumbnail_label.setPixmap(thumbnail_pix)
            thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumbnail_label.setFixedSize(thumbnail_size, thumbnail_size)
            thumbnail_label.setScaledContents(False)
            card_layout.addWidget(thumbnail_label)

            # Информация о точке
            info_label = QLabel(
                f"<b>{render.label}</b><br>"
                f"T: {render.temperature_c:.0f}°C<br>"
                f"t: {render.time_s:.0f}с<br>"
                f"{render.phase_info.get('stage', '')}"
            )
            info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            info_label.setWordWrap(True)
            card_layout.addWidget(info_label)

            # Обработчик клика для отображения в основном превью
            def make_click_handler(r):
                def handler(event):
                    self._show_intermediate_render_in_preview(r)

                return handler

            card.mousePressEvent = make_click_handler(render)

            self.intermediate_container_layout.addWidget(card)

        # Добавляем растягивающийся элемент в конец
        self.intermediate_container_layout.addStretch()

    def _show_intermediate_render_in_preview(self, render) -> None:
        """Отображает выбранный промежуточный рендер в основном превью."""
        self.preview_scene.clear()
        preview_rgb = _preview_image_rgb(render.image_rgb, max_side=3072)
        pix = _to_pixmap(preview_rgb)
        self.preview_scene.addPixmap(pix)
        self.preview_scene.setSceneRect(pix.rect())
        self.preview_view.resetTransform()
        self.preview_view.fitInView(
            self.preview_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

        # Обновляем информационную панель
        phase_fractions = render.phase_info.get("phase_fractions", {})
        phase_lines = [
            f"  - {name}: {frac:.4f}"
            for name, frac in sorted(
                phase_fractions.items(), key=lambda x: x[1], reverse=True
            )
        ]

        info_lines = [
            f"Промежуточный рендер: {render.label}",
            f"Точка {render.point_index + 1}",
            f"Температура: {render.temperature_c:.1f}°C",
            f"Время: {render.time_s:.1f} с",
            f"Система: {render.phase_info.get('system', '')}",
            f"Стадия: {render.phase_info.get('stage', '')}",
            "",
            "Фазовые доли:",
        ] + phase_lines

        self.info_text.setPlainText("\n".join(info_lines))

    def _save_mask_dict(
        self, masks: dict[str, np.ndarray] | None, out_dir: Path, prefix: str
    ) -> list[Path]:
        paths: list[Path] = []
        if not isinstance(masks, dict):
            return paths
        for name, mask in masks.items():
            if not isinstance(mask, np.ndarray):
                continue
            arr = (mask > 0).astype(np.uint8) * 255
            path = out_dir / f"{prefix}_{name}.png"
            save_image(arr, path)
            paths.append(path)
        return paths

    def _calculate_carbon_content(self) -> float | None:
        """Calculate carbon content from phase fractions for Fe-C system."""
        if not self.current_output or not self.current_output.metadata:
            return None

        system = (
            str(self.current_output.metadata.get("inferred_system", "")).strip().lower()
        )
        if system != "fe-c":
            return None

        phase_report = self.current_output.metadata.get("phase_model_report", {})
        phase_fractions = phase_report.get("blended_phase_fractions", {})

        # For hypoeutectoid steels: C = pearlite_fraction * 0.8
        pearlite = phase_fractions.get("PEARLITE", 0.0)
        if pearlite > 0:
            return round(pearlite * 0.8, 4)

        return None

    def _calculate_steel_grade(self) -> str | None:
        """Calculate steel grade from carbon content."""
        carbon = self._calculate_carbon_content()
        if carbon is None:
            return None

        # Round to nearest standard grade
        grade_number = round(carbon * 100)

        # Standard grades: 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85
        standard_grades = [10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
        closest_grade = min(standard_grades, key=lambda x: abs(x - grade_number))

        return str(closest_grade)

    def _export_lab_package(self) -> None:
        """Export complete lab package with security features."""
        import json
        from core.security import compute_image_hash, sign_data, encrypt_answers

        if self.current_output is None or self.current_output.image_rgb is None:
            QMessageBox.warning(self, "Ошибка", "Нет изображения для экспорта")
            return

        # Get export directory
        base_dir = Path(
            self.export_dir_edit.text().strip() or "examples/factory_v3_output"
        )
        base_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped subdirectory
        prefix = (
            self.export_prefix_edit.text().strip()
            or self.current_request.sample_id
            or "lab_sample"
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_dir = base_dir / f"{prefix}_{stamp}"
        package_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Save image
            image_path = package_dir / f"{prefix}.png"
            save_image(self.current_output.image_rgb, image_path)

            # 2. Compute image hash
            image_hash = compute_image_hash(image_path)

            # 3. Load public key (embedded in application)
            from core.security.crypto_manager import get_embedded_public_key

            try:
                public_key = get_embedded_public_key()
            except FileNotFoundError as e:
                QMessageBox.critical(
                    self, "Ошибка", f"Публичный ключ не найден:\n{str(e)}"
                )
                return

            # 4. Create student data
            from core.contracts_v3 import StudentDataV3

            student_data = StudentDataV3(
                sample_id=self.current_request.sample_id,
                timestamp=datetime.now().isoformat(),
                composition_wt={},
                thermal_program=self.current_request.thermal_program.to_dict(),
                prep_route=self.current_request.prep_route.to_dict(),
                etch_profile=self.current_request.etch_profile.to_dict(),
                seed=self.current_request.seed,
                resolution=self.current_request.resolution,
                image_sha256=image_hash,
            )

            student_dict = student_data.to_dict()
            student_dict["composition_masked"] = True

            # 5. Sign student data (optional - use shared teacher key resolution)
            print("Checking for private key...")
            private_key_path = (
                self.teacher_private_key_path
                or resolve_teacher_key_path(prefer_saved=True)
            )
            print(f"Private key path: {private_key_path}")
            if private_key_path is not None and private_key_path.exists():
                print("Private key found, signing student data...")
                private_key = private_key_path.read_bytes()
                signature = sign_data(student_dict, private_key)
                student_dict["digital_signature"] = signature
                print("Student data signed")
            else:
                print(
                    "WARNING: Private key not found - package will be created without digital signature"
                )
                student_dict["digital_signature"] = None

            # 6. Save student.json
            student_json_path = package_dir / f"{prefix}_student.json"
            with open(student_json_path, "w", encoding="utf-8") as f:
                json.dump(student_dict, f, indent=2, ensure_ascii=False)

            # 7. Create teacher answers
            from core.contracts_v3 import TeacherAnswersV3

            phase_report = self.current_output.metadata.get("phase_model_report", {})
            phase_fractions = phase_report.get("blended_phase_fractions", {})

            answers = TeacherAnswersV3(
                sample_id=self.current_request.sample_id,
                image_sha256=image_hash,
                phase_fractions=phase_fractions,
                inferred_system=self.current_output.metadata.get("inferred_system"),
                steel_grade=self._calculate_steel_grade(),
                carbon_content_calculated=self._calculate_carbon_content(),
                verification={
                    "seed": self.current_request.seed,
                    "timestamp": datetime.now().isoformat(),
                    "generator_version": "v3.0.0",
                },
            )

            # 8. Encrypt and save answers
            encrypted_answers = encrypt_answers(answers.to_dict(), public_key)
            answers_path = package_dir / f"{prefix}_answers.enc"
            answers_path.write_bytes(encrypted_answers)

            # 9. Save full metadata (for reference, not for students)
            meta_path = package_dir / f"{prefix}_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.current_output.metadata_json_safe(),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            # 10. Create manifest
            manifest = {
                "package_version": "3.0.0",
                "created_at": datetime.now().isoformat(),
                "diagram": None,
                "diagram_deprecated": True,
                "files": {
                    "image": f"{prefix}.png",
                    "student_data": f"{prefix}_student.json",
                    "teacher_answers": f"{prefix}_answers.enc",
                    "metadata": f"{prefix}_metadata.json",
                },
                "security": {
                    "encryption": "RSA-2048 + AES-256-GCM",
                    "signature": "RSA-PSS with SHA256",
                    "image_hash_algorithm": "SHA256",
                },
            }

            manifest_path = package_dir / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            prefixed_manifest_path = package_dir / f"{prefix}_manifest.json"
            with open(prefixed_manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            self._set_status(f"Пакет ЛР сохранен: {package_dir}")
            QMessageBox.information(
                self,
                "Успех",
                f"Пакет ЛР экспортирован:\n{package_dir}\n\n"
                f"Файлы:\n"
                f"• {prefix}.png - изображение\n"
                f"• {prefix}_student.json - данные для студента\n"
                f"• {prefix}_answers.enc - ответы (зашифрованы)\n"
                f"• {prefix}_metadata.json - полные метаданные\n"
                f"• manifest.json - описание пакета",
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка экспорта", f"Не удалось экспортировать пакет:\n{str(e)}"
            )

    def _remember_preset_context(self, payload: dict[str, Any]) -> None:
        metadata = dict(payload.get("metadata", {}))
        self._loaded_preset_context = {
            "material_grade": payload.get("material_grade"),
            "material_class_ru": payload.get("material_class_ru"),
            "lab_work": payload.get("lab_work"),
            "target_astm_grain_size": payload.get("target_astm_grain_size"),
            "mean_grain_diameter_um": payload.get("mean_grain_diameter_um"),
            "expected_properties": dict(payload.get("expected_properties", {})),
            "preset_metadata": metadata,
        }

    def _load_selected_preset(self) -> None:
        path = str(self.preset_combo.currentData() or "")
        if not path:
            return
        try:
            payload = self.pipeline.load_preset(path)
            self._remember_preset_context(payload)
            req = MetallographyRequestV3.from_dict(payload)
            self._apply_request_to_ui(req)
            self._set_status(
                f"Загружен пресет: {_preset_visible_label(Path(path).stem)}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Пресет", str(exc))

    def _apply_request_to_ui(self, req: MetallographyRequestV3) -> None:
        self.sample_id_edit.setText(req.sample_id)
        self._set_composition(req.composition_wt)
        hint_idx = self.system_hint_combo.findData(req.system_hint or "")
        if hint_idx >= 0:
            self.system_hint_combo.setCurrentIndex(hint_idx)
        self.strict_validation_check.setChecked(bool(req.strict_validation))

        self.thermal_name_edit.setText("Термопрограмма")
        self.thermal_notes_edit.setText("")
        tp = req.thermal_program
        self.thermal_points_table.setRowCount(0)
        for point in tp.points:
            self._add_thermal_point_row(point)
        q = tp.quench
        resolved_medium = str(q.medium_code)
        if self.quench_medium_combo.findData(resolved_medium) < 0:
            canonical = canonicalize_quench_medium_code(
                medium_code=str(q.medium_code),
                bath_temperature_c=float(q.bath_temperature_c),
            )
            resolved_medium = str(canonical.get("resolved_code", resolved_medium))
        qidx = self.quench_medium_combo.findData(resolved_medium)
        if qidx >= 0:
            self.quench_medium_combo.setCurrentIndex(qidx)
        self.quench_time_spin.setValue(float(q.quench_time_s))
        self.quench_bath_temp_spin.setValue(float(q.bath_temperature_c))
        self.quench_sample_temp_spin.setValue(float(q.sample_temperature_c))
        self.quench_custom_name.setText(str(q.custom_medium_name))
        self.quench_severity_spin.setValue(float(q.custom_severity_factor))
        sidx = self.thermal_sampling_combo.findData(str(tp.sampling_mode))
        if sidx >= 0:
            self.thermal_sampling_combo.setCurrentIndex(sidx)
        self.thermal_degree_step_spin.setValue(float(tp.degree_step_c))
        self.thermal_max_frames_spin.setValue(int(tp.max_frames))
        self._sync_transition_cells()
        self._refresh_thermal_plot()

        self.prep_table.setRowCount(0)
        for op in req.prep_route.steps:
            self._add_prep_row(op)
        self.prep_rough_spin.setValue(req.prep_route.roughness_target_um)
        ridx = self.prep_relief_combo.findData(req.prep_route.relief_mode)
        if ridx >= 0:
            self.prep_relief_combo.setCurrentIndex(ridx)
        self.prep_contam_spin.setValue(req.prep_route.contamination_level)

        ridx = self.etch_reagent_combo.findData(req.etch_profile.reagent)
        if ridx >= 0:
            self.etch_reagent_combo.setCurrentIndex(ridx)
        self.etch_time_spin.setValue(req.etch_profile.time_s)
        self.etch_temp_spin.setValue(req.etch_profile.temperature_c)
        aidx = self.etch_agitation_combo.findData(req.etch_profile.agitation)
        if aidx >= 0:
            self.etch_agitation_combo.setCurrentIndex(aidx)
        self.etch_overetch_spin.setValue(req.etch_profile.overetch_factor)
        cidx = self.etch_conc_unit_combo.findData(req.etch_profile.concentration_unit)
        if cidx >= 0:
            self.etch_conc_unit_combo.setCurrentIndex(cidx)
        self.etch_conc_value_spin.setValue(float(req.etch_profile.concentration_value))

        pidx = self.synth_profile_combo.findData(req.synthesis_profile.profile_id)
        if pidx >= 0:
            self.synth_profile_combo.setCurrentIndex(pidx)
        if str(req.synthesis_profile.profile_id).startswith("textbook_"):
            self.visual_standard_combo.setCurrentIndex(
                max(0, self.visual_standard_combo.findData("textbook_bw"))
            )
        else:
            self.visual_standard_combo.setCurrentIndex(
                max(0, self.visual_standard_combo.findData("custom"))
            )
        tidx = self.synth_topology_combo.findData(
            req.synthesis_profile.phase_topology_mode
        )
        if tidx >= 0:
            self.synth_topology_combo.setCurrentIndex(tidx)
        else:
            raise ValueError(
                "LEGACY_FIELD_REMOVED: значение 'synthesis_profile.phase_topology_mode' не поддерживается в V3."
            )
        sgidx = self.synth_system_generator_combo.findData(
            getattr(req.synthesis_profile, "system_generator_mode", "system_auto")
        )
        if sgidx >= 0:
            self.synth_system_generator_combo.setCurrentIndex(sgidx)
        self.synth_contrast_spin.setValue(req.synthesis_profile.contrast_target)
        self.synth_sharp_spin.setValue(req.synthesis_profile.boundary_sharpness)
        self.synth_artifact_spin.setValue(req.synthesis_profile.artifact_level)
        sidx = self.synth_sensitivity_combo.findData(
            req.synthesis_profile.composition_sensitivity_mode
        )
        if sidx >= 0:
            self.synth_sensitivity_combo.setCurrentIndex(sidx)
        gidx = self.synth_generation_mode_combo.findData(
            req.synthesis_profile.generation_mode
        )
        if gidx >= 0:
            self.synth_generation_mode_combo.setCurrentIndex(gidx)
        eidx = self.synth_phase_emphasis_combo.findData(
            req.synthesis_profile.phase_emphasis_style
        )
        if eidx >= 0:
            self.synth_phase_emphasis_combo.setCurrentIndex(eidx)
        self.synth_phase_tolerance_spin.setValue(
            float(req.synthesis_profile.phase_fraction_tolerance_pct)
        )
        pm = req.phase_model
        pm_idx = self.phase_control_mode_combo.findData(pm.phase_control_mode)
        if pm_idx >= 0:
            self.phase_control_mode_combo.setCurrentIndex(pm_idx)
        self.manual_phase_table.setRowCount(0)
        for phase_name, val in pm.manual_phase_fractions.items():
            self._add_manual_phase_row(str(phase_name), float(val))
        self.phase_override_slider.blockSignals(True)
        self.phase_override_spin.blockSignals(True)
        self.phase_override_slider.setValue(
            int(round(float(pm.manual_override_weight) * 100.0))
        )
        self.phase_override_spin.setValue(float(pm.manual_override_weight))
        self.phase_override_slider.blockSignals(False)
        self.phase_override_spin.blockSignals(False)
        self.allow_custom_fallback_check.setChecked(bool(pm.allow_custom_fallback))
        self.phase_balance_tolerance_spin.setValue(
            float(pm.phase_balance_tolerance_pct)
        )

        self.seed_spin.setValue(req.seed)
        res_idx = -1
        for idx in range(self.resolution_combo.count()):
            data = self.resolution_combo.itemData(idx)
            if isinstance(data, tuple) and tuple(data) == tuple(req.resolution):
                res_idx = idx
                break
        if res_idx >= 0:
            self.resolution_combo.setCurrentIndex(res_idx)

        ref_idx = self.reference_combo.findData(req.reference_profile_id or "")
        if ref_idx >= 0:
            self.reference_combo.setCurrentIndex(ref_idx)

        ms = req.microscope_profile
        self.ms_simulate_check.setChecked(bool(ms.get("simulate_preview", False)))
        mag_idx = self.ms_magnification_combo.findData(
            int(ms.get("magnification", 200))
        )
        if mag_idx >= 0:
            self.ms_magnification_combo.setCurrentIndex(mag_idx)
        optical_idx = self.ms_optical_mode_combo.findData(
            str(ms.get("optical_mode", "brightfield"))
        )
        if optical_idx >= 0:
            self.ms_optical_mode_combo.setCurrentIndex(optical_idx)
        psf_idx = self.ms_psf_profile_combo.findData(
            str(ms.get("psf_profile", "standard"))
        )
        if psf_idx >= 0:
            self.ms_psf_profile_combo.setCurrentIndex(psf_idx)
        self.ms_psf_strength_spin.setValue(
            _safe_float(ms.get("psf_strength", 0.0), 0.0)
        )
        self.ms_sectioning_shear_spin.setValue(
            _safe_float(ms.get("sectioning_shear_deg", 35.0), 35.0)
        )
        self.ms_hybrid_balance_spin.setValue(
            _safe_float(ms.get("hybrid_balance", 0.5), 0.5)
        )
        self.ms_focus_spin.setValue(_safe_float(ms.get("focus", 0.95), 0.95))
        self.ms_brightness_spin.setValue(_safe_float(ms.get("brightness", 1.0), 1.0))
        self.ms_contrast_spin.setValue(_safe_float(ms.get("contrast", 1.1), 1.1))
        self.ms_noise_spin.setValue(_safe_float(ms.get("noise_sigma", 1.5), 1.5))
        self.ms_vignette_spin.setValue(
            _safe_float(ms.get("vignette_strength", 0.12), 0.12)
        )
        self.ms_uneven_spin.setValue(_safe_float(ms.get("uneven_strength", 0.08), 0.08))
        self.ms_dust_check.setChecked(bool(ms.get("add_dust", False)))
        self.ms_scratch_check.setChecked(bool(ms.get("add_scratches", False)))
        direct_raw = ms.get("brinell_direct", {})
        direct = direct_raw if isinstance(direct_raw, dict) else {}
        if direct:
            idx = self.brinell_mode_combo.findData("direct")
            if idx >= 0:
                self.brinell_mode_combo.setCurrentIndex(idx)
            self.brinell_p_spin.setValue(_safe_float(direct.get("P_kgf", 187.5), 187.5))
            self.brinell_d_ball_spin.setValue(_safe_float(direct.get("D_mm", 2.5), 2.5))
            self.brinell_d_indent_spin.setValue(
                _safe_float(direct.get("d_mm", 0.9), 0.9)
            )
            self._compute_brinell_direct_ui()
        else:
            idx = self.brinell_mode_combo.findData("estimated")
            if idx >= 0:
                self.brinell_mode_combo.setCurrentIndex(idx)
            self.brinell_direct_label.setText("HBW: -")

        self._validate_route_ui()
        self._update_etch_risk()
        self._check_phase_model_resolution()


def launch_sample_factory_app_v3(
    presets_dir: str | Path | None = None,
    profiles_dir: str | Path | None = None,
) -> int:
    from ui_qt.spinbox_wheel_filter import SpinBoxWheelFilter

    app = QApplication.instance() or QApplication([])

    # Install global wheel event filter for spinboxes
    wheel_filter = SpinBoxWheelFilter(app)
    app.installEventFilter(wheel_filter)

    window = SampleFactoryWindowV3(
        presets_dir=presets_dir,
        profiles_dir=profiles_dir,
    )
    window.show()
    return app.exec()
