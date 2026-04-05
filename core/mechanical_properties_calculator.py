"""
Калькулятор механических свойств по микроструктуре.
На основе данных учебника Братковский_Шевченко_Материаловедение_2017.

Расчёт свойств углеродистых сталей на основе фазового состава.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass
class PhaseProperties:
    """Механические свойства фазы."""
    name: str
    hardness_hb: float
    tensile_strength_mpa: float
    elongation_pct: float
    young_modulus_gpa: float = 210.0
    density_g_cm3: float = 7.85


_RULEBOOK_DIR = Path(__file__).resolve().parent / "rulebook"
_TEXTBOOK_PROPERTIES_PATH = _RULEBOOK_DIR / "textbook_material_properties.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _typical_value(payload: Any, default: float = 0.0) -> float:
    if isinstance(payload, dict):
        if "typical" in payload:
            return _safe_float(payload.get("typical"), default)
        if "value" in payload:
            return _safe_float(payload.get("value"), default)
        if "min" in payload and "max" in payload:
            return (_safe_float(payload.get("min"), default) + _safe_float(payload.get("max"), default)) / 2.0
    return _safe_float(payload, default)


@lru_cache(maxsize=1)
def _textbook_reference() -> dict[str, Any]:
    if not _TEXTBOOK_PROPERTIES_PATH.exists():
        return {}
    try:
        return json.loads(_TEXTBOOK_PROPERTIES_PATH.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


# Свойства фаз из учебника (табличные данные)
PHASE_PROPERTIES: dict[str, PhaseProperties] = {
    "ferrite": PhaseProperties(
        name="Феррит",
        hardness_hb=90,
        tensile_strength_mpa=275,
        elongation_pct=45,
        young_modulus_gpa=210,
        density_g_cm3=7.87
    ),
    "pearlite": PhaseProperties(
        name="Перлит",
        hardness_hb=200,
        tensile_strength_mpa=850,
        elongation_pct=12,
        young_modulus_gpa=210,
        density_g_cm3=7.84
    ),
    "pearlite_fine": PhaseProperties(
        name="Тонкий перлит",
        hardness_hb=240,
        tensile_strength_mpa=920,
        elongation_pct=10
    ),
    "pearlite_coarse": PhaseProperties(
        name="Крупный перлит",
        hardness_hb=180,
        tensile_strength_mpa=780,
        elongation_pct=15
    ),
    "cementite": PhaseProperties(
        name="Цементит",
        hardness_hb=800,
        tensile_strength_mpa=250,
        elongation_pct=0,
        young_modulus_gpa=180,
        density_g_cm3=7.69
    ),
    "austenite": PhaseProperties(
        name="Аустенит",
        hardness_hb=170,
        tensile_strength_mpa=600,
        elongation_pct=35,
        young_modulus_gpa=200,
        density_g_cm3=7.84
    ),
    "martensite": PhaseProperties(
        name="Мартенсит",
        hardness_hb=650,
        tensile_strength_mpa=2100,
        elongation_pct=2,
        young_modulus_gpa=210
    ),
    "martensite_tempered_low": PhaseProperties(
        name="Мартенсит отпуска (низкий)",
        hardness_hb=600,
        tensile_strength_mpa=1900,
        elongation_pct=3
    ),
    "martensite_tempered_medium": PhaseProperties(
        name="Троостит отпуска",
        hardness_hb=450,
        tensile_strength_mpa=1400,
        elongation_pct=6
    ),
    "martensite_tempered_high": PhaseProperties(
        name="Сорбит отпуска",
        hardness_hb=280,
        tensile_strength_mpa=850,
        elongation_pct=15
    ),
    "bainite_upper": PhaseProperties(
        name="Верхний бейнит",
        hardness_hb=400,
        tensile_strength_mpa=1300,
        elongation_pct=10
    ),
    "bainite_lower": PhaseProperties(
        name="Нижний бейнит",
        hardness_hb=500,
        tensile_strength_mpa=1600,
        elongation_pct=6
    ),
    "ledeburite": PhaseProperties(
        name="Ледебурит",
        hardness_hb=550,
        tensile_strength_mpa=400,
        elongation_pct=1
    ),
    "graphite_flake": PhaseProperties(
        name="Графит пластинчатый",
        hardness_hb=5,
        tensile_strength_mpa=20,
        elongation_pct=0,
        young_modulus_gpa=10,
        density_g_cm3=2.2
    ),
    "graphite_spheroidal": PhaseProperties(
        name="Графит шаровидный",
        hardness_hb=5,
        tensile_strength_mpa=20,
        elongation_pct=0,
        young_modulus_gpa=10,
        density_g_cm3=2.2
    )
}


def _phase_properties_from_reference() -> dict[str, PhaseProperties]:
    reference = _textbook_reference()
    phases = dict(reference.get("fe_c_phases", {})) if isinstance(reference, dict) else {}
    if not phases:
        return {}

    def _entry(name: str, payload: dict[str, Any], *, default_modulus: float = 210.0, default_density: float = 7.85) -> PhaseProperties:
        return PhaseProperties(
            name=str(payload.get("name_ru", name)),
            hardness_hb=_typical_value(payload.get("hardness_hb"), PHASE_PROPERTIES.get(name, PhaseProperties(name, 0, 0, 0)).hardness_hb),
            tensile_strength_mpa=_typical_value(payload.get("tensile_strength_mpa"), PHASE_PROPERTIES.get(name, PhaseProperties(name, 0, 0, 0)).tensile_strength_mpa),
            elongation_pct=_typical_value(payload.get("elongation_pct"), PHASE_PROPERTIES.get(name, PhaseProperties(name, 0, 0, 0)).elongation_pct),
            young_modulus_gpa=_safe_float(payload.get("young_modulus_gpa"), default_modulus),
            density_g_cm3=_safe_float(payload.get("density_g_cm3"), default_density),
        )

    out: dict[str, PhaseProperties] = {
        "ferrite": _entry("ferrite", dict(phases.get("ferrite", {}))),
        "pearlite": _entry("pearlite", dict(phases.get("pearlite", {}))),
        "cementite": _entry("cementite", dict(phases.get("cementite", {})), default_modulus=180.0, default_density=7.69),
        "austenite": _entry("austenite", dict(phases.get("austenite", {})), default_modulus=200.0, default_density=7.84),
        "martensite": _entry("martensite", dict(phases.get("martensite", {}))),
        "ledeburite": _entry("ledeburite", dict(phases.get("ledeburite", {})), default_modulus=180.0, default_density=7.69),
    }
    pearlite_payload = dict(phases.get("pearlite", {}))
    lamellar = dict(pearlite_payload.get("lamellar_spacing_um", {}))
    out["pearlite_coarse"] = PhaseProperties(
        name="Крупный перлит",
        hardness_hb=max(150.0, out["pearlite"].hardness_hb - 20.0),
        tensile_strength_mpa=max(600.0, out["pearlite"].tensile_strength_mpa - 70.0),
        elongation_pct=min(20.0, out["pearlite"].elongation_pct + 3.0),
    )
    out["pearlite_fine"] = PhaseProperties(
        name="Тонкий перлит",
        hardness_hb=out["pearlite"].hardness_hb + 40.0,
        tensile_strength_mpa=out["pearlite"].tensile_strength_mpa + 70.0,
        elongation_pct=max(4.0, out["pearlite"].elongation_pct - 2.0),
    )
    bainite_payload = dict(phases.get("bainite", {}))
    bainite_hardness = dict(bainite_payload.get("hardness_hb", {}))
    out["bainite_upper"] = PhaseProperties(
        name="Верхний бейнит",
        hardness_hb=_typical_value(dict(bainite_hardness.get("upper", {})), PHASE_PROPERTIES["bainite_upper"].hardness_hb),
        tensile_strength_mpa=_typical_value(bainite_payload.get("tensile_strength_mpa"), PHASE_PROPERTIES["bainite_upper"].tensile_strength_mpa) - 150.0,
        elongation_pct=max(3.0, _typical_value(bainite_payload.get("elongation_pct"), PHASE_PROPERTIES["bainite_upper"].elongation_pct)),
    )
    out["bainite_lower"] = PhaseProperties(
        name="Нижний бейнит",
        hardness_hb=_typical_value(dict(bainite_hardness.get("lower", {})), PHASE_PROPERTIES["bainite_lower"].hardness_hb),
        tensile_strength_mpa=_typical_value(bainite_payload.get("tensile_strength_mpa"), PHASE_PROPERTIES["bainite_lower"].tensile_strength_mpa),
        elongation_pct=max(2.0, _typical_value(bainite_payload.get("elongation_pct"), PHASE_PROPERTIES["bainite_lower"].elongation_pct) - 2.0),
    )
    out["martensite_tempered_low"] = PhaseProperties(name="Мартенсит отпуска (низкий)", hardness_hb=600, tensile_strength_mpa=1900, elongation_pct=3)
    out["martensite_tempered_medium"] = PhaseProperties(name="Троостит отпуска", hardness_hb=450, tensile_strength_mpa=1400, elongation_pct=6)
    out["martensite_tempered_high"] = PhaseProperties(name="Сорбит отпуска", hardness_hb=280, tensile_strength_mpa=850, elongation_pct=15)
    out["graphite_flake"] = PHASE_PROPERTIES["graphite_flake"]
    out["graphite_spheroidal"] = PHASE_PROPERTIES["graphite_spheroidal"]
    return out


_PHASE_OVERRIDES = _phase_properties_from_reference()
if _PHASE_OVERRIDES:
    PHASE_PROPERTIES.update(_PHASE_OVERRIDES)


def calculate_rule_of_mixtures(
    phase_fractions: dict[str, float],
    property_name: str
) -> float:
    """
    Расчёт свойства по правилу смесей.

    Args:
        phase_fractions: Доли фаз (должны суммироваться в 1.0)
        property_name: Название свойства ('hardness_hb', 'tensile_strength_mpa', etc.)

    Returns:
        Рассчитанное свойство
    """
    result = 0.0
    total_fraction = sum(phase_fractions.values())

    if abs(total_fraction - 1.0) > 0.01:
        # Нормализация
        phase_fractions = {k: v / total_fraction for k, v in phase_fractions.items()}

    for phase_name, fraction in phase_fractions.items():
        phase_key = phase_name.lower()
        if phase_key in PHASE_PROPERTIES:
            props = PHASE_PROPERTIES[phase_key]
            value = getattr(props, property_name, 0.0)
            result += fraction * value

    return result


def calculate_properties_from_microstructure(
    phase_fractions: dict[str, float],
    carbon_pct: float = 0.0,
    pearlite_spacing: str = "medium"
) -> dict[str, Any]:
    """
    Расчёт комплекса механических свойств по микроструктуре.

    Args:
        phase_fractions: Доли фаз
        carbon_pct: Содержание углерода (для поправок)
        pearlite_spacing: Дисперсность перлита ('coarse', 'medium', 'fine')

    Returns:
        Словарь с рассчитанными свойствами
    """
    # Поправка на дисперсность перлита
    if "pearlite" in phase_fractions:
        spacing_factor = {"coarse": 0.9, "medium": 1.0, "fine": 1.15}.get(pearlite_spacing, 1.0)
        if pearlite_spacing == "fine":
            # Заменяем обычный перлит на тонкий
            phase_fractions["pearlite_fine"] = phase_fractions.get("pearlite_fine", 0) + phase_fractions["pearlite"] * 0.5
            phase_fractions["pearlite_coarse"] = phase_fractions.get("pearlite_coarse", 0) + phase_fractions["pearlite"] * 0.5
            del phase_fractions["pearlite"]

    # Расчёт свойств
    hardness_hb = calculate_rule_of_mixtures(phase_fractions, "hardness_hb")
    uts_mpa = calculate_rule_of_mixtures(phase_fractions, "tensile_strength_mpa")
    elongation_pct = calculate_rule_of_mixtures(phase_fractions, "elongation_pct")

    # Поправка на углерод в твёрдом растворе (упрощённо)
    if carbon_pct > 0.02:
        ss_hardening = 50.0 * min(carbon_pct, 0.8)
        hardness_hb += ss_hardening
        uts_mpa += 100.0 * min(carbon_pct, 0.8)

    # Пластичность сильно снижается с увеличением твёрдости
    if hardness_hb > 400:
        elongation_pct *= 0.5
    elif hardness_hb > 250:
        elongation_pct *= 0.75

    # Модуль упругости слабо зависит от микроструктуры
    young_modulus = 210.0  # ГПа для сталей

    # Плотность по правилу смесей
    density = calculate_rule_of_mixtures(phase_fractions, "density_g_cm3")
    if density < 7.0:
        density = 7.0  # Минимальная для сталей

    # Относительное сужение площади (эмпирическая связь с пластичностью)
    reduction_of_area = min(80, elongation_pct * 2.5)

    # Ударная вязкость (очень упрощённо)
    if hardness_hb < 200:
        impact_toughness = 80 + (200 - hardness_hb) * 0.5
    elif hardness_hb < 350:
        impact_toughness = 40 + (350 - hardness_hb) * 0.3
    else:
        impact_toughness = max(5, 40 - (hardness_hb - 350) * 0.1)

    # Предел текучести (эмпирическое соотношение)
    yield_strength = uts_mpa * 0.6
    if hardness_hb > 300:
        yield_strength = uts_mpa * 0.75
    elif hardness_hb < 150:
        yield_strength = uts_mpa * 0.5

    return {
        "hardness": {
            "hb": round(hardness_hb, 0),
            "hrc_estimate": round(max(0, (hardness_hb - 100) / 10), 0) if hardness_hb > 100 else None
        },
        "tensile": {
            "uts_mpa": round(uts_mpa, 0),
            "yield_strength_mpa": round(yield_strength, 0),
            "uniform_elongation_pct": round(elongation_pct * 0.6, 1),
            "total_elongation_pct": round(elongation_pct, 1),
            "reduction_of_area_pct": round(reduction_of_area, 0)
        },
        "elastic": {
            "young_modulus_gpa": young_modulus,
            "shear_modulus_gpa": round(young_modulus / 2.6, 0),
            "bulk_modulus_gpa": round(young_modulus / 3, 0),
            "poisson_ratio": 0.30
        },
        "toughness": {
            "impact_j_cm2": round(impact_toughness, 0),
            "fracture_toughness_mpa_m05": round(max(20, 150 - hardness_hb * 0.15), 0)
        },
        "physical": {
            "density_g_cm3": round(density, 2)
        },
        "phase_fractions": {k: round(v, 3) for k, v in phase_fractions.items()},
        "notes": "Расчёт по правилу смесей на основе данных Братковский_Шевченко_2017"
    }


def estimate_properties_from_carbon(carbon_pct: float, treatment: str = "normalized") -> dict[str, Any]:
    """
    Оценка свойств стали только по содержанию углерода.
    Для нормализованного состояния.

    Args:
        carbon_pct: Содержание углерода в %
        treatment: Состояние ('normalized', 'quenched', 'annealed')

    Returns:
        Словарь со свойствами
    """
    c = carbon_pct

    if treatment == "normalized":
        # Данные для нормализованной стали (таблица из учебника)
        if c <= 0.1:
            ferrite = 0.87
            pearlite = 0.13
        elif c <= 0.2:
            ferrite = 0.75
            pearlite = 0.25
        elif c <= 0.4:
            ferrite = 0.50
            pearlite = 0.50
        elif c <= 0.6:
            ferrite = 0.25
            pearlite = 0.75
        elif c <= 0.8:
            ferrite = 0.0
            pearlite = 1.0
        else:
            pearlite = (6.67 - c) / (6.67 - 0.8)
            cementite = 1 - pearlite
            return calculate_properties_from_microstructure({
                "pearlite": pearlite,
                "cementite": cementite
            }, c)

        return calculate_properties_from_microstructure({
            "ferrite": ferrite,
            "pearlite": pearlite
        }, c)

    elif treatment == "quenched":
        # Закалённая сталь (мартенсит)
        hardness = 650 + 500 * min(c, 0.8)
        return {
            "hardness": {"hb": round(hardness, 0)},
            "tensile": {
                "uts_mpa": round(2000 + 200 * c * 100, 0),
                "yield_strength_mpa": round(1800 + 150 * c * 100, 0),
                "total_elongation_pct": round(max(1, 3 - 3 * c), 1)
            },
            "microstructure": "Мартенсит",
            "notes": "Закалённое состояние (без отпуска)"
        }

    elif treatment == "annealed":
        # Отожжённая сталь (более мягкая)
        base = estimate_properties_from_carbon(c, "normalized")
        base["hardness"]["hb"] = round(base["hardness"]["hb"] * 0.85, 0)
        base["tensile"]["uts_mpa"] = round(base["tensile"]["uts_mpa"] * 0.85, 0)
        base["tensile"]["total_elongation_pct"] = round(base["tensile"]["total_elongation_pct"] * 1.2, 1)
        base["notes"] = "Отожжённое состояние"
        return base

    return {}


def get_material_grade_properties(grade: str) -> dict[str, Any]:
    """
    Возвращает свойства для стандартных марок сталей и чугунов.
    """
    reference = _textbook_reference()
    cast_iron = dict(reference.get("cast_iron", {})) if isinstance(reference, dict) else {}
    grades_db = {
        # Углеродистые стали
        "ст10": {"C": 0.10, "treatment": "normalized", "uts_mpa": 320, "hb": 105},
        "ст20": {"C": 0.20, "treatment": "normalized", "uts_mpa": 420, "hb": 125},
        "ст35": {"C": 0.35, "treatment": "normalized", "uts_mpa": 550, "hb": 170},
        "ст45": {"C": 0.45, "treatment": "normalized", "uts_mpa": 650, "hb": 200},
        "ст45_improved": {"C": 0.45, "treatment": "improved", "uts_mpa": 800, "hb": 280},
        "ст60": {"C": 0.60, "treatment": "normalized", "uts_mpa": 780, "hb": 220},
        "ст65Г": {"C": 0.65, "Mn": 1.0, "treatment": "quenched_medium_temp", "uts_mpa": 1000, "hb": 300},

        # Инструментальные
        "у7": {"C": 0.70, "treatment": "quenched_low_temp", "hb": 600},
        "у8": {"C": 0.80, "treatment": "quenched_low_temp", "hb": 650},
        "у10": {"C": 1.00, "treatment": "quenched_low_temp", "hb": 700},
        "у12": {"C": 1.20, "treatment": "quenched_low_temp", "hb": 750},

        # Чугуны
        "сч15": {"C": 3.0, "Si": 1.8, "type": "grey_iron", "uts_mpa": 150, "hb": 170},
        "сч20": {"C": 3.2, "Si": 2.0, "type": "grey_iron", "uts_mpa": 200, "hb": 195},
        "сч25": {"C": 3.4, "Si": 2.2, "type": "grey_iron", "uts_mpa": 250, "hb": 210},
        "вч40": {"C": 3.5, "Si": 2.5, "Mg": 0.05, "type": "ductile_iron", "uts_mpa": 400, "hb": 180},
        "вч50": {"C": 3.6, "Si": 2.6, "Mg": 0.05, "type": "ductile_iron", "uts_mpa": 500, "hb": 200},
        "вч60": {"C": 3.7, "Si": 2.7, "Mg": 0.05, "type": "ductile_iron", "uts_mpa": 600, "hb": 220},
    }

    if cast_iron:
        grey = dict(cast_iron.get("grey", {}))
        if grey:
            grades_db["СЃС‡20"] = {
                "C": 3.2,
                "Si": 2.0,
                "type": "grey_iron",
                "uts_mpa": _typical_value(grey.get("tensile_strength_mpa"), 200),
                "hb": _typical_value(grey.get("hardness_hb"), 195),
            }
        ductile = dict(cast_iron.get("ductile", {}))
        if ductile:
            grades_db["РІС‡50"] = {
                "C": 3.6,
                "Si": 2.6,
                "Mg": 0.05,
                "type": "ductile_iron",
                "uts_mpa": _typical_value(ductile.get("tensile_strength_mpa"), 500),
                "hb": _typical_value(ductile.get("hardness_hb"), 200),
            }

    if cast_iron:
        grey = dict(cast_iron.get("grey", {}))
        if grey:
            grades_db["\u0441\u044720"] = {
                "C": 3.2,
                "Si": 2.0,
                "type": "grey_iron",
                "uts_mpa": _typical_value(grey.get("tensile_strength_mpa"), 200),
                "hb": _typical_value(grey.get("hardness_hb"), 195),
            }
        ductile = dict(cast_iron.get("ductile", {}))
        if ductile:
            grades_db["\u0432\u044750"] = {
                "C": 3.6,
                "Si": 2.6,
                "Mg": 0.05,
                "type": "ductile_iron",
                "uts_mpa": _typical_value(ductile.get("tensile_strength_mpa"), 500),
                "hb": _typical_value(ductile.get("hardness_hb"), 200),
            }

    grade_lower = grade.lower().strip()
    if grade_lower not in grades_db:
        return {"error": f"Марка {grade} не найдена в базе"}

    data = grades_db[grade_lower]

    if data.get("type") == "grey_iron":
        return {
            "grade": grade,
            "type": "Серый чугун",
            "composition": {"C": data["C"], "Si": data.get("Si", 2.0)},
            "properties": {
                "tensile_strength_mpa": data["uts_mpa"],
                "hardness_hb": data["hb"],
                "compressive_strength_mpa": data["uts_mpa"] * 4,
                "elongation_pct": 0.7
            }
        }
    elif data.get("type") == "ductile_iron":
        return {
            "grade": grade,
            "type": "Высокопрочный чугун",
            "composition": {"C": data["C"], "Si": data.get("Si", 2.5), "Mg": data.get("Mg", 0.05)},
            "properties": {
                "tensile_strength_mpa": data["uts_mpa"],
                "hardness_hb": data["hb"],
                "elongation_pct": 5 if data["uts_mpa"] < 500 else 2
            }
        }
    else:
        # Сталь
        treatment = data.get("treatment", "normalized")
        base_props = estimate_properties_from_carbon(data["C"], treatment)

        # Переопределяем табличными значениями
        if "uts_mpa" in data:
            base_props["tensile"]["uts_mpa"] = data["uts_mpa"]
        if "hb" in data:
            base_props["hardness"]["hb"] = data["hb"]

        return {
            "grade": grade,
            "type": "Сталь",
            "composition": {"C": data["C"], "Mn": data.get("Mn", 0.5)},
            "treatment": treatment,
            "properties": base_props
        }


# Коэффициенты Холла-Петча из учебника Братковский_Шевченко
HALL_PETCH_COEFFICIENTS: dict[str, dict[str, float]] = {
    "ferrite": {"sigma_0_mpa": 50.0, "k_y": 0.50},      # α-Fe
    "austenite": {"sigma_0_mpa": 70.0, "k_y": 0.35},    # γ-Fe
    "pearlite": {"sigma_0_mpa": 120.0, "k_y": 0.80},    # эвтектоидная смесь
}

_HP_REFERENCE = dict(_textbook_reference().get("hall_petch_coefficients", {})) if isinstance(_textbook_reference(), dict) else {}
for _phase_name in ("ferrite", "austenite", "pearlite"):
    _payload = dict(_HP_REFERENCE.get(_phase_name, {}))
    if _payload:
        HALL_PETCH_COEFFICIENTS[_phase_name] = {
            "sigma_0_mpa": _safe_float(_payload.get("sigma_0_mpa"), HALL_PETCH_COEFFICIENTS[_phase_name]["sigma_0_mpa"]),
            "k_y": _safe_float(_payload.get("k_y"), HALL_PETCH_COEFFICIENTS[_phase_name]["k_y"]),
        }


def calculate_yield_strength_hall_petch(
    grain_size_um: float,
    material: str = "ferrite"
) -> dict[str, Any]:
    """
    Расчёт предела текучести по формуле Холла-Петча.

    σ_y = σ_0 + k_y × d^(-1/2)

    где:
    - σ_y — предел текучести (МПа)
    - σ_0 — сопротивление решётки дислокациям (МПа)
    - k_y — коэффициент Холла-Петча (МПа·м^0.5)
    - d — средний размер зерна (м)

    Args:
        grain_size_um: Средний размер зерна в микрометрах (мкм)
        material: Тип материала:
            - "ferrite" — феррит (α-Fe), σ_0=50 МПа, k_y=0.50 МПа·м^0.5
            - "austenite" — аустенит (γ-Fe), σ_0=70 МПа, k_y=0.35 МПа·м^0.5
            - "pearlite" — перлит, σ_0=120 МПа, k_y=0.80 МПа·м^0.5

    Returns:
        Словарь с результатами расчёта:
        - grain_size_um: размер зерна (мкм)
        - grain_size_m: размер зерна (м)
        - material: тип материала
        - sigma_0_mpa: сопротивление решётки (МПа)
        - k_y: коэффициент Холла-Петча (МПа·м^0.5)
        - yield_strength_mpa: предел текучести (МПа)
        - formula: строка с формулой

    Примеры:
        >>> calculate_yield_strength_hall_petch(25, "ferrite")
        {'yield_strength_mpa': 350.0, ...}

        >>> calculate_yield_strength_hall_petch(10, "ferrite")
        {'yield_strength_mpa': 408.0, ...}
    """
    if grain_size_um <= 0:
        raise ValueError("Размер зерна должен быть положительным числом")

    # Получаем коэффициенты для материала
    if material.lower() not in HALL_PETCH_COEFFICIENTS:
        available = ", ".join(HALL_PETCH_COEFFICIENTS.keys())
        raise ValueError(f"Неизвестный материал '{material}'. Доступные: {available}")

    coeffs = HALL_PETCH_COEFFICIENTS[material.lower()]
    sigma_0 = coeffs["sigma_0_mpa"]
    k_y = coeffs["k_y"]

    # Конвертируем размер зерна из мкм в метры
    grain_size_m = grain_size_um * 1e-6

    # Расчёт по формуле Холла-Петча
    d_inverse_sqrt = 1.0 / (grain_size_m ** 0.5)
    yield_strength = sigma_0 + k_y * d_inverse_sqrt

    # Дополнительный расчёт для разных размеров зерна (сравнение)
    reference_sizes = {
        "ASTM_5 (65 мкм)": 65,
        "ASTM_6 (45 мкм)": 45,
        "ASTM_7 (32 мкм)": 32,
        "ASTM_8 (22 мкм)": 22,
        "ASTM_10 (11 мкм)": 11,
        "ASTM_12 (5.5 мкм)": 5.5,
    }

    comparison = {}
    for name, size_um in reference_sizes.items():
        size_m = size_um * 1e-6
        ys = sigma_0 + k_y / (size_m ** 0.5)
        comparison[name] = round(ys, 0)

    return {
        "grain_size_um": grain_size_um,
        "grain_size_m": grain_size_m,
        "material": material.lower(),
        "sigma_0_mpa": sigma_0,
        "k_y_mpa_m05": k_y,
        "yield_strength_mpa": round(yield_strength, 1),
        "formula": f"σ_y = σ_0 + k_y × d^(-1/2) = {sigma_0} + {k_y} × ({grain_size_m:.2e})^(-0.5)",
        "comparison_table": comparison,
        "notes": "Формула Холла-Петча: предел текучести растёт с уменьшением размера зерна"
    }


def calculate_astm_grain_size_number(
    grains_per_mm2: float
) -> dict[str, Any]:
    """
    Расчёт номера зерна по ASTM E112.

    Формула: n = 2^(G-1), где:
    - n — количество зёрен на дюйм² при увеличении 100×
    - G — номер зерна по ASTM

    Пересчёт: 1 дюйм² = 645.16 мм²

    Args:
        grains_per_mm2: Количество зёрен на мм²

    Returns:
        Словарь с результатами:
        - grains_per_mm2: зёрен на мм²
        - grains_per_inch2: зёрен на дюйм² (при 100×)
        - astm_grain_size_number: номер зерна ASTM
        - mean_grain_diameter_um: средний диаметр зерна (мкм)
    """
    if grains_per_mm2 <= 0:
        raise ValueError("Плотность зёрен должна быть положительным числом")

    # Пересчёт на дюйм² (1 дюйм = 25.4 мм, 1 дюйм² = 645.16 мм²)
    grains_per_inch2 = grains_per_mm2 * 645.16

    # Расчёт номера ASTM: G = 1 + log2(n)
    import math
    astm_number = 1.0 + math.log2(grains_per_inch2)

    # Средний диаметр зерна (приближённо)
    mean_diameter_um = math.sqrt(1.0 / grains_per_mm2) * 1000

    # Диапазон ASTM номеров
    astm_ranges = {
        "ASTM 0": {"min": 0, "max": 1, "grain_size_um": "500+"},
        "ASTM 1-2": {"min": 1, "max": 3, "grain_size_um": "250-500"},
        "ASTM 3-4": {"min": 3, "max": 5, "grain_size_um": "125-250"},
        "ASTM 5-6": {"min": 5, "max": 7, "grain_size_um": "65-125"},
        "ASTM 7-8": {"min": 7, "max": 9, "grain_size_um": "32-65"},
        "ASTM 9-10": {"min": 9, "max": 11, "grain_size_um": "16-32"},
        "ASTM 11-12": {"min": 11, "max": 13, "grain_size_um": "8-16"},
        "ASTM 13+": {"min": 13, "max": 20, "grain_size_um": "<8"},
    }

    # Определение диапазона
    astm_range = None
    for name, range_data in astm_ranges.items():
        if range_data["min"] <= astm_number < range_data["max"]:
            astm_range = name
            break

    if astm_range is None:
        astm_range = "ASTM 13+" if astm_number >= 13 else "ASTM 0"

    return {
        "grains_per_mm2": round(grains_per_mm2, 2),
        "grains_per_inch2": round(grains_per_inch2, 1),
        "astm_grain_size_number": round(astm_number, 2),
        "mean_grain_diameter_um": round(mean_diameter_um, 1),
        "astm_range": astm_range,
        "formula": f"G = 1 + log₂(n) = 1 + log₂({grains_per_inch2:.1f}) = {astm_number:.2f}",
        "notes": "ASTM E112: чем больше номер, тем мельче зерно"
    }
