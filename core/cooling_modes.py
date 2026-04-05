from __future__ import annotations

from typing import Any

CANONICAL_COOLING_MODES: tuple[str, ...] = (
    "equilibrium",
    "slow_cool",
    "normalized",
    "quenched",
    "tempered",
    "aged",
    "overaged",
    "natural_aged",
    "solutionized",
    "cold_worked",
)

_COOLING_MODE_LABELS_RU: dict[str, str] = {
    "auto": "Автовыбор",
    "equilibrium": "Равновесное охлаждение",
    "slow_cool": "Медленное охлаждение",
    "normalized": "Нормализация (воздух)",
    "quenched": "Закалка",
    "tempered": "Отпуск",
    "aged": "Искусственное старение",
    "overaged": "Перестаривание",
    "natural_aged": "Естественное старение",
    "solutionized": "Растворный отжиг",
    "cold_worked": "Холодная деформация",
}

_COOLING_MODE_ALIASES: dict[str, str] = {
    "": "equilibrium",
    "eq": "equilibrium",
    "equilibrium": "equilibrium",
    "balance": "equilibrium",
    "slow": "slow_cool",
    "slowcool": "slow_cool",
    "slow_cool": "slow_cool",
    "slow-cool": "slow_cool",
    "normalized": "normalized",
    "normalize": "normalized",
    "normalizing": "normalized",
    "norm": "normalized",
    "quench": "quenched",
    "quenched": "quenched",
    "water_quench": "quenched",
    "oil_quench": "quenched",
    "water-quench": "quenched",
    "oil-quench": "quenched",
    "temper": "tempered",
    "tempered": "tempered",
    "age": "aged",
    "aged": "aged",
    "overage": "overaged",
    "overaged": "overaged",
    "natural_aged": "natural_aged",
    "natural-aged": "natural_aged",
    "naturally_aged": "natural_aged",
    "solution": "solutionized",
    "solutionized": "solutionized",
    "solution_treat": "solutionized",
    "solutionized_treat": "solutionized",
    "cold": "cold_worked",
    "cold_worked": "cold_worked",
    "cold-worked": "cold_worked",
    "cold_work": "cold_worked",
    "cold-work": "cold_worked",
    "auto": "auto",
    "automatic": "auto",
    "autoselect": "auto",
}


def canonicalize_cooling_mode(raw: Any) -> str:
    value = "" if raw is None else str(raw).strip().lower()
    value = value.replace(" ", "_")
    return _COOLING_MODE_ALIASES.get(value, value or "equilibrium")


def cooling_mode_label_ru(code: str) -> str:
    canonical = canonicalize_cooling_mode(code)
    return _COOLING_MODE_LABELS_RU.get(canonical, canonical)


def cooling_mode_options_ru(include_auto: bool = True) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    if include_auto:
        options.append(("auto", _COOLING_MODE_LABELS_RU["auto"]))
    for code in CANONICAL_COOLING_MODES:
        options.append((code, _COOLING_MODE_LABELS_RU.get(code, code)))
    return options


def _processing_value(processing: Any, name: str, default: float = 0.0) -> float:
    if isinstance(processing, dict):
        value = processing.get(name, default)
    else:
        value = getattr(processing, name, default)
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return float(default)


def resolve_auto_cooling_mode(inferred_system: str, processing: Any) -> str:
    system = str(inferred_system or "custom-multicomponent").strip().lower()
    temperature = _processing_value(processing, "temperature_c", 20.0)
    deformation = _processing_value(processing, "deformation_pct", 0.0)
    aging_hours = _processing_value(processing, "aging_hours", 0.0)

    if system == "fe-c":
        if 120.0 <= temperature <= 700.0 and aging_hours > 0.0:
            return "tempered"
        if temperature >= 780.0:
            return "normalized"
        return "equilibrium"

    if system == "fe-si":
        if deformation > 1.0:
            return "cold_worked"
        if temperature >= 850.0:
            return "normalized"
        return "equilibrium"

    if system == "al-si":
        if aging_hours > 0.2:
            return "aged"
        if temperature >= 600.0:
            return "slow_cool"
        return "equilibrium"

    if system == "cu-zn":
        if deformation > 1.0:
            return "cold_worked"
        return "equilibrium"

    if system == "al-cu-mg":
        if aging_hours > 0.2:
            return "aged"
        if temperature >= 480.0:
            return "solutionized"
        return "quenched"

    if deformation > 1.0:
        return "cold_worked"
    if temperature >= 550.0:
        return "slow_cool"
    return "equilibrium"
