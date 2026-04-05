from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from export.export_tables import save_json

THEME_MODES = ("light", "dark")
STYLE_PROFILES = ("mirea_web_v1",)
DEFAULT_STYLE_PROFILE = "mirea_web_v1"
THEME_SCHEMA_VERSION = 2

GENERATOR_THEME_PROFILE = Path("profiles") / "ui_theme_generator_v3.json"
MICROSCOPE_THEME_PROFILE = Path("profiles") / "ui_theme_microscope_v3.json"

BRANDING_DIR = Path("ui_qt") / "assets" / "branding"
DEFAULT_BRANDING_LOGO = BRANDING_DIR / "mirea_gerb_colour.png"

THEME_TOKENS: dict[str, dict[str, str]] = {
    "light": {
        "bg_base": "#F8FAFC",
        "bg_surface": "#FFFFFF",
        "bg_canvas": "#FAFBFC",
        "bg_elevated": "#FFFFFF",
        "text_primary": "#0F172A",
        "text_secondary": "#475569",
        "text_tertiary": "#94A3B8",
        "border": "#E2E8F0",
        "border_strong": "#CBD5E1",
        "border_focus": "#0EA5E9",
        "primary": "#0369A1",
        "primary_hover": "#0284C7",
        "primary_pressed": "#075985",
        "primary_light": "#E0F2FE",
        "accent": "#0EA5E9",
        "accent_hover": "#38BDF8",
        "accent_soft": "#E0F2FE",
        "success": "#059669",
        "success_light": "#D1FAE5",
        "warning": "#D97706",
        "warning_light": "#FEF3C7",
        "error": "#DC2626",
        "error_light": "#FEE2E2",
        "info": "#3B82F6",
        "info_light": "#DBEAFE",
        "focus": "#0EA5E9",
        "shadow_sm": "rgba(15, 23, 42, 0.05)",
        "shadow_md": "rgba(15, 23, 42, 0.1)",
        "shadow_lg": "rgba(15, 23, 42, 0.15)",
        "header_grad_start": "#0369A1",
        "header_grad_end": "#0EA5E9",
        "header_text": "#FFFFFF",
        "header_border": "#0284C7",
        "btn_text": "#0F172A",
        "btn_text_hover": "#0369A1",
        "btn_text_pressed": "#075985",
        "btn_text_disabled": "#94A3B8",
        "btn_primary_bg": "#0369A1",
        "btn_primary_bg_hover": "#0284C7",
        "btn_primary_text": "#FFFFFF",
        "btn_primary_text_hover": "#FFFFFF",
        "section_header_text": "#0F172A",
        "overlay": "rgba(15, 23, 42, 0.5)",
    },
    "dark": {
        "bg_base": "#0F172A",
        "bg_surface": "#1E293B",
        "bg_canvas": "#0F172A",
        "bg_elevated": "#334155",
        "text_primary": "#F1F5F9",
        "text_secondary": "#CBD5E1",
        "text_tertiary": "#64748B",
        "border": "#334155",
        "border_strong": "#475569",
        "border_focus": "#38BDF8",
        "primary": "#0EA5E9",
        "primary_hover": "#38BDF8",
        "primary_pressed": "#0284C7",
        "primary_light": "#1E3A5F",
        "accent": "#38BDF8",
        "accent_hover": "#7DD3FC",
        "accent_soft": "#1E3A5F",
        "success": "#10B981",
        "success_light": "#064E3B",
        "warning": "#F59E0B",
        "warning_light": "#78350F",
        "error": "#EF4444",
        "error_light": "#7F1D1D",
        "info": "#60A5FA",
        "info_light": "#1E3A8A",
        "focus": "#38BDF8",
        "shadow_sm": "rgba(0, 0, 0, 0.2)",
        "shadow_md": "rgba(0, 0, 0, 0.3)",
        "shadow_lg": "rgba(0, 0, 0, 0.4)",
        "header_grad_start": "#0C4A6E",
        "header_grad_end": "#0369A1",
        "header_text": "#F1F5F9",
        "header_border": "#1E293B",
        "btn_text": "#F1F5F9",
        "btn_text_hover": "#38BDF8",
        "btn_text_pressed": "#0EA5E9",
        "btn_text_disabled": "#475569",
        "btn_primary_bg": "#0EA5E9",
        "btn_primary_bg_hover": "#38BDF8",
        "btn_primary_text": "#0F172A",
        "btn_primary_text_hover": "#0F172A",
        "section_header_text": "#F1F5F9",
        "overlay": "rgba(0, 0, 0, 0.7)",
    },
}


def _normalize_mode(value: str | None, default: str = "light") -> str:
    mode = str(value or "").strip().lower()
    if mode in THEME_MODES:
        return mode
    return default if default in THEME_MODES else "light"


def _normalize_style_profile(value: str | None, default: str = DEFAULT_STYLE_PROFILE) -> str:
    profile = str(value or "").strip().lower()
    if profile in STYLE_PROFILES:
        return profile
    return default if default in STYLE_PROFILES else DEFAULT_STYLE_PROFILE


def load_theme_settings(
    path: str | Path,
    default_mode: str = "light",
    default_style_profile: str = DEFAULT_STYLE_PROFILE,
) -> dict[str, Any]:
    p = Path(path)
    mode = _normalize_mode(default_mode)
    style_profile = _normalize_style_profile(default_style_profile)
    schema_version = THEME_SCHEMA_VERSION
    migrate = False

    try:
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            mode = _normalize_mode(str(data.get("theme_mode", mode)), default=mode)
            style_profile = _normalize_style_profile(
                str(data.get("style_profile", style_profile)),
                default=style_profile,
            )
            try:
                src_schema = int(data.get("schema_version", 0) or 0)
            except Exception:
                src_schema = 0
            migrate = src_schema < THEME_SCHEMA_VERSION or "style_profile" not in data
    except Exception:
        pass

    if migrate:
        try:
            save_theme_mode(p, mode=mode, style_profile=style_profile)
        except Exception:
            pass

    return {
        "schema_version": schema_version,
        "theme_mode": mode,
        "style_profile": style_profile,
    }


def load_theme_mode(path: str | Path, default: str = "light") -> str:
    return str(load_theme_settings(path, default_mode=default).get("theme_mode", _normalize_mode(default)))


def save_theme_mode(path: str | Path, mode: str, style_profile: str = DEFAULT_STYLE_PROFILE) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": THEME_SCHEMA_VERSION,
        "theme_mode": _normalize_mode(mode),
        "style_profile": _normalize_style_profile(style_profile),
        "last_updated": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(payload, p)


def resolve_branding_logo(preferred: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if preferred:
        candidates.append(Path(preferred))
    candidates.extend(
        [
            DEFAULT_BRANDING_LOGO,
            BRANDING_DIR / "mirea_logo_iri.png",
            BRANDING_DIR / "mirea_logo_pish.png",
        ]
    )
    if BRANDING_DIR.exists():
        candidates.extend(sorted(BRANDING_DIR.glob("*.png")))
        candidates.extend(sorted(BRANDING_DIR.glob("*.jpg")))
        candidates.extend(sorted(BRANDING_DIR.glob("*.jpeg")))
    for path in candidates:
        try:
            if path.exists() and path.is_file():
                return path
        except Exception:
            continue
    return None


def theme_tokens(mode: str) -> dict[str, str]:
    return dict(THEME_TOKENS[_normalize_mode(mode)])


def status_color(mode: str, key: str, fallback: str = "#B8C5DA") -> str:
    return str(theme_tokens(mode).get(key, fallback))


def build_qss(mode: str) -> str:
    t = theme_tokens(mode)
    return f"""
QWidget {{
    background: transparent;
    color: {t["text_primary"]};
    font-family: "Segoe UI", "Tahoma", "Arial", sans-serif;
    font-size: 15px;
    font-weight: 500;
}}
QMainWindow {{
    background: {t["bg_base"]};
}}
QWidget#appHeader, QWidget#microscopeHeader {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {t["header_grad_start"]}, stop:1 {t["header_grad_end"]});
    border: none;
    border-radius: 16px;
    padding: 4px;
}}
QWidget#appHeader *, QWidget#microscopeHeader * {{
    background: transparent;
}}
QWidget#headerTitleWidget {{
    background: transparent;
    border: none;
}}
QLabel#headerLogoLabel {{
    background: transparent;
    border: none;
    padding: 8px;
}}
QWidget#leftNavCard, QWidget#centerCard, QWidget#rightCard {{
    background: {t["bg_surface"]};
    border: 1px solid {t["border"]};
    border-radius: 16px;
    padding: 4px;
}}
QGroupBox#previewCard, QGroupBox#infoCard, QGroupBox#instrumentCard {{
    border: 1px solid {t["border"]};
    background: {t["bg_surface"]};
}}
QGroupBox {{
    border: 1px solid {t["border"]};
    border-radius: 12px;
    margin-top: 12px;
    font-weight: 700;
    font-size: 16px;
    padding-top: 12px;
    background: {t["bg_surface"]};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 4px 10px;
    color: {t["text_primary"]};
    background: {t["bg_elevated"]};
    border: 1px solid {t["border"]};
    border-radius: 6px;
}}
QLabel {{
    color: {t["text_primary"]};
    background: transparent;
}}
QLabel#headerBrandPrimary {{
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: {t["header_text"]};
}}
QLabel#headerBrandSecondary {{
    font-size: 15px;
    font-weight: 600;
    color: {t["header_text"]};
    opacity: 0.9;
}}
QLabel#headerThemeLabel {{
    color: {t["header_text"]};
    font-weight: 600;
    font-size: 14px;
}}
QWidget#appHeader QComboBox, QWidget#microscopeHeader QComboBox {{
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.55);
    border-radius: 8px;
    color: {t["header_text"]};
    padding: 3px 8px;
}}
QWidget#appHeader QComboBox::drop-down, QWidget#microscopeHeader QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QWidget#appHeader QComboBox QAbstractItemView, QWidget#microscopeHeader QComboBox QAbstractItemView {{
    background: {t["bg_surface"]};
    color: {t["text_primary"]};
    border: 1px solid {t["border"]};
    selection-background-color: {t["primary"]};
}}
QLabel#stepStatusLabel {{
    font-weight: 600;
    color: {t["text_secondary"]};
}}
QLabel#statusPill {{
    border: 1px solid {t["border"]};
    border-radius: 12px;
    padding: 6px 14px;
    background: {t["bg_elevated"]};
    color: {t["text_primary"]};
    font-weight: 700;
    font-size: 14px;
}}
QLabel#sectionHeader {{
    color: {t["section_header_text"]};
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.3px;
    background: transparent;
    border: none;
    border-bottom: 3px solid {t["accent"]};
    padding-bottom: 6px;
    padding-top: 4px;
}}
QLabel#studentSectionLabel {{
    font-weight: 700;
    color: {t["text_primary"]};
    margin-bottom: 4px;
}}
QLabel#studentFormulaLabel {{
    color: {t["accent"]};
    background: {t["bg_elevated"]};
    border: 1px solid {t["border"]};
    border-radius: 8px;
    padding: 8px 10px;
    font-family: Consolas, "Segoe UI", monospace;
    font-weight: 600;
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QPlainTextEdit, QTableWidget {{
    background: {t["bg_surface"]};
    border: 1.5px solid {t["border"]};
    border-radius: 10px;
    selection-background-color: {t["primary"]};
    padding: 8px 12px;
    font-size: 15px;
    font-weight: 500;
}}
QSpinBox, QDoubleSpinBox {{
    padding-right: 24px;
    min-height: 28px;
}}
QSpinBox#instrumentSpin, QDoubleSpinBox#instrumentSpin {{
    padding: 6px 10px;
    padding-right: 12px;
    min-height: 24px;
}}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    background: transparent;
    border: none;
    width: 18px;
}}
QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {{
    width: 10px;
    height: 10px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus, QTableWidget:focus {{
    border-color: {t["border_focus"]};
    border-width: 2px;
}}
QComboBox QAbstractItemView {{
    background: {t["bg_surface"]};
    color: {t["text_primary"]};
    border: 1px solid {t["border"]};
    selection-background-color: {t["primary"]};
}}
QComboBox::drop-down {{
    background: transparent;
    border: none;
    width: 18px;
}}
QListWidget#stepList {{
    border: 1px solid {t["border"]};
    border-radius: 12px;
    background: transparent;
    outline: 0;
    padding: 4px;
}}
QListWidget#stepList::item {{
    min-height: 36px;
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid transparent;
    margin: 3px 4px;
    font-weight: 500;
}}
QListWidget#stepList::item:selected {{
    background: {t["primary"]};
    color: {t["btn_primary_text"]};
    border: 1px solid transparent;
    font-weight: 600;
}}
QListWidget#stepList::item:selected:active, QListWidget#stepList::item:selected:!active {{
    border: 1px solid transparent;
}}
QListWidget#stepList::item:hover {{
    background: {t["primary_light"]};
}}
QHeaderView::section {{
    background: {t["bg_surface"]};
    color: {t["text_primary"]};
    border: 1px solid {t["border"]};
    padding: 6px;
    font-weight: 600;
}}
QPushButton {{
    background: {t["bg_surface"]};
    color: {t["btn_text"]};
    border: 1.5px solid {t["border"]};
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 600;
    font-size: 15px;
}}
QPushButton:hover {{
    background: {t["bg_elevated"]};
    border-color: {t["primary"]};
    color: {t["btn_text_hover"]};
}}
QPushButton:pressed {{
    background: {t["primary_light"]};
    border-color: {t["primary_pressed"]};
    color: {t["btn_text_pressed"]};
}}
QPushButton:checked {{
    background: {t["accent_soft"]};
    color: {t["accent"]};
    border-color: {t["accent"]};
}}
QPushButton:disabled {{
    background: {t["bg_surface"]};
    color: {t["btn_text_disabled"]};
    border-color: {t["border"]};
    opacity: 0.5;
}}
QPushButton#primaryCta {{
    background: {t["btn_primary_bg"]};
    border: none;
    color: {t["btn_primary_text"]};
    font-weight: 700;
    padding: 12px 24px;
}}
QPushButton#primaryCta:disabled {{
    background: {t["border"]};
    color: {t["btn_text_disabled"]};
    opacity: 0.6;
}}
QPushButton#primaryCta:hover {{
    background: {t["btn_primary_bg_hover"]};
    color: {t["btn_primary_text_hover"]};
}}
QPushButton#primaryCta:pressed {{
    background: {t["primary_pressed"]};
    color: {t["btn_primary_text"]};
}}
QPushButton#secondaryCta {{
    background: transparent;
    color: {t["primary"]};
    border: 1.5px solid {t["primary"]};
}}
QPushButton#secondaryCta:disabled {{
    color: {t["btn_text_disabled"]};
    border-color: {t["border"]};
    opacity: 0.5;
}}
QPushButton#secondaryCta:hover {{
    background: {t["primary_light"]};
    color: {t["primary_hover"]};
    border-color: {t["primary_hover"]};
}}
QPushButton#secondaryCta:pressed {{
    background: {t["primary_light"]};
    color: {t["primary_pressed"]};
    border-color: {t["primary_pressed"]};
}}
QPushButton#compactSquareCta {{
    background: transparent;
    color: {t["primary"]};
    border: 1.5px solid {t["primary"]};
    border-radius: 8px;
    padding: 0px;
    min-width: 24px;
    min-height: 24px;
    font-weight: 700;
}}
QPushButton#compactSquareCta:hover {{
    background: {t["primary_light"]};
    color: {t["primary_hover"]};
    border-color: {t["primary_hover"]};
}}
QPushButton#compactSquareCta:pressed {{
    background: {t["primary_light"]};
    color: {t["primary_pressed"]};
    border-color: {t["primary_pressed"]};
}}
QPushButton#studentExportButton {{
    background: {t["success"]};
    color: {t["btn_primary_text"]};
    border: 1px solid {t["success"]};
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 700;
    margin-top: 8px;
}}
QPushButton#studentExportButton:hover {{
    background: {t["success_light"]};
    color: {t["btn_primary_text"]};
    border-color: {t["success"]};
}}
QPushButton#studentExportButton:disabled {{
    background: {t["bg_elevated"]};
    color: {t["btn_text_disabled"]};
    border-color: {t["border"]};
}}
QPushButton#accentCta {{
    background: transparent;
    color: {t["accent"]};
    border: 1.5px solid {t["accent"]};
}}
QPushButton#accentCta:disabled {{
    color: {t["btn_text_disabled"]};
    border-color: {t["border"]};
    opacity: 0.5;
}}
QPushButton#accentCta:hover {{
    background: {t["accent_soft"]};
    color: {t["accent_hover"]};
    border-color: {t["accent_hover"]};
}}
QPushButton#accentCta:pressed {{
    background: {t["accent_soft"]};
    color: {t["primary"]};
    border-color: {t["primary"]};
}}
QGraphicsView {{
    background: {t["bg_canvas"]};
    border: 1px solid {t["border"]};
    border-radius: 12px;
}}
QScrollArea {{
    border: 0px;
    background: transparent;
}}
QProgressBar#stepProgressBar {{
    border: 1px solid {t["border"]};
    border-radius: 10px;
    background: {t["bg_canvas"]};
    text-align: center;
    min-height: 22px;
    color: {t["text_primary"]};
    font-weight: 600;
}}
QProgressBar#stepProgressBar::chunk {{
    border-radius: 9px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t["header_grad_start"]}, stop:1 {t["header_grad_end"]});
}}
QSplitter::handle {{
    background: {t["border"]};
    border-radius: 3px;
}}
QSplitter::handle:hover {{
    background: {t["border_strong"]};
}}
QScrollBar:vertical {{
    width: 12px;
    background: transparent;
    margin: 3px;
}}
QScrollBar::handle:vertical {{
    background: {t["border"]};
    border-radius: 6px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {t["border_strong"]};
}}
QScrollBar:horizontal {{
    height: 12px;
    background: transparent;
    margin: 3px;
}}
QScrollBar::handle:horizontal {{
    background: {t["border"]};
    border-radius: 6px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {t["border_strong"]};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0px;
    height: 0px;
}}
QToolTip {{
    background: {t["bg_elevated"]};
    color: {t["text_primary"]};
    border: 1px solid {t["border"]};
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 14px;
}}
"""
