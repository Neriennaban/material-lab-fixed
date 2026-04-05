"""
Калькулятор режимов термической обработки по данным учебника Братковский_Шевченко_2017.

Расчёт критических температур, режимов закалки и отпуска для углеродистых и легированных сталей.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass
class SteelComposition:
    """Химический состав стали в масс.%"""
    C: float = 0.0
    Mn: float = 0.0
    Si: float = 0.0
    Cr: float = 0.0
    Ni: float = 0.0
    Mo: float = 0.0
    V: float = 0.0
    Cu: float = 0.0
    Al: float = 0.0
    S: float = 0.0
    P: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "SteelComposition":
        return cls(
            C=float(data.get("C", 0.0)),
            Mn=float(data.get("Mn", 0.0)),
            Si=float(data.get("Si", 0.0)),
            Cr=float(data.get("Cr", 0.0)),
            Ni=float(data.get("Ni", 0.0)),
            Mo=float(data.get("Mo", 0.0)),
            V=float(data.get("V", 0.0)),
            Cu=float(data.get("Cu", 0.0)),
            Al=float(data.get("Al", 0.0)),
            S=float(data.get("S", 0.0)),
            P=float(data.get("P", 0.0)),
        )


_RULEBOOK_DIR = Path(__file__).resolve().parent / "rulebook"
_TEXTBOOK_PROPERTIES_PATH = _RULEBOOK_DIR / "textbook_material_properties.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@lru_cache(maxsize=1)
def _textbook_reference() -> dict[str, Any]:
    if not _TEXTBOOK_PROPERTIES_PATH.exists():
        return {}
    try:
        return json.loads(_TEXTBOOK_PROPERTIES_PATH.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def calculate_ac1(composition: SteelComposition) -> float:
    """
    Расчёт температуры Ac1 (начало аустенитизации при нагреве).
    Эмпирическая формула для углеродистых и низколегированных сталей.
    """
    ac1 = 723.0 - 10.7 * composition.Mn - 16.9 * composition.Ni + 29.1 * composition.Si + 16.9 * composition.Cr
    ac1 += 290.0 * composition.As if hasattr(composition, 'As') else 0.0  # мышьяк (редко)
    return max(680.0, min(780.0, ac1))  # разумные пределы


def calculate_ac3(composition: SteelComposition) -> float:
    """
    Расчёт температуры Ac3 (конец аустенитизации для доэвтектоидных сталей).
    Формула учитывает влияние легирующих элементов.
    """
    c = composition.C
    mn = composition.Mn
    si = composition.Si
    cr = composition.Cr
    ni = composition.Ni
    mo = composition.Mo

    # Базовая температура для чистого Fe-C
    ac3 = 910.0 - 203.0 * (c ** 0.5)

    # Поправки на легирующие элементы
    ac3 -= 15.2 * mn
    ac3 += 44.7 * si
    ac3 += 31.5 * mo
    ac3 += 10.4 * v_ if (v_ := composition.V) > 0 else 0.0
    ac3 -= 17.9 * ni

    # Для заэвтектоидных сталей (C > 0.8%) Ac3 не применяется
    if c > 0.8:
        return None  # Используется Ac1 + (30-50)°C

    return max(727.0, min(950.0, ac3))


def calculate_ms_temperature(composition: SteelComposition) -> float:
    """
    Расчёт температуры начала мартенситного превращения (Ms).
    Формула Эндрюса с поправками.
    """
    c = composition.C
    mn = composition.Mn
    cr = composition.Cr
    ni = composition.Ni
    mo = composition.Mo

    ms = 539.0 - 423.0 * c - 30.4 * mn - 12.1 * cr - 17.7 * ni - 7.5 * mo

    # Поправка на молибден (при содержании > 0.5%)
    if mo > 0.5:
        ms -= 15.0 * (mo - 0.5)

    return max(20.0, min(500.0, ms))


def calculate_mf_temperature(composition: SteelComposition) -> float:
    """
    Расчёт температуры окончания мартенситного превращения (Mf).
    Mf ≈ Ms - 215°C (эмпирическое соотношение)
    """
    ms = calculate_ms_temperature(composition)
    mf = ms - 215.0
    return max(-100.0, mf)


def get_quench_temperature(composition: SteelComposition, steel_type: str = "hypoeutectoid") -> dict[str, Any]:
    """
    Расчёт температуры закалки.

    Args:
        composition: Химический состав стали
        steel_type: Тип стали:
            - "hypoeutectoid" - доэвтектоидная (C < 0.8%)
            - "eutectoid" - эвтектоидная (C ≈ 0.8%)
            - "hypereutectoid" - заэвтектоидная (C > 0.8%)

    Returns:
        Словарь с рекомендуемой температурой и обоснованием
    """
    c = composition.C

    if steel_type == "hypoeutectoid":
        ac3 = calculate_ac3(composition)
        if ac3 is None:
            ac3 = 780.0  # fallback
        quench_temp = ac3 + 40.0  # Ac3 + (30-50)°C
        rationale = f"Полная аустенитизация: Ac3 = {ac3:.0f}°C + 40°C"
        microstructure_result = "Мартенсит + остаточный аустенит"

    elif steel_type == "eutectoid":
        ac1 = calculate_ac1(composition)
        quench_temp = ac1 + 40.0  # Ac1 + (30-50)°C
        rationale = f"Аустенитизация эвтектоидной стали: Ac1 = {ac1:.0f}°C + 40°C"
        microstructure_result = "Мартенсит + остаточный аустенит (10-15%)"

    else:  # hypereutectoid
        ac1 = calculate_ac1(composition)
        quench_temp = ac1 + 40.0  # Ac1 + (30-50)°C
        rationale = f"Неполная аустенитизация: Ac1 = {ac1:.0f}°C + 40°C (для сохранения цементита)"
        microstructure_result = "Мартенсит + цементит вторичный + остаточный аустенит"

    return {
        "quench_temperature_c": round(quench_temp, 0),
        "rationale": rationale,
        "resulting_microstructure": microstructure_result,
        "ac1": round(calculate_ac1(composition), 0),
        "ac3": round(calculate_ac3(composition), 0) if steel_type == "hypoeutectoid" else None,
        "ms_temperature": round(calculate_ms_temperature(composition), 0),
        "mf_temperature": round(calculate_mf_temperature(composition), 0),
    }


def get_tempering_temperature(hardness_target: str, composition: SteelComposition) -> dict[str, Any]:
    """
    Расчёт температуры отпуска по целевой твёрдости.

    Args:
        hardness_target: Целевая твёрдость:
            - "high" - высокая (режущий инструмент)
            - "medium" - средняя (пружины, рессоры)
            - "low" - низкая (конструкционная прочность)
        composition: Химический состав

    Returns:
        Рекомендуемый режим отпуска
    """
    regimes = {
        "high": {
            "name_ru": "Низкий отпуск",
            "temperature_range": (150, 250),
            "typical": 200,
            "result": "Мартенсит отпуска (ε-карбиды)",
            "hardness_hb_range": (550, 650),
            "applications": "Режущий инструмент, подшипники, цементированные детали",
            "ductility": "низкая",
            "toughness": "низкая"
        },
        "medium": {
            "name_ru": "Средний отпуск",
            "temperature_range": (350, 450),
            "typical": 400,
            "result": "Троостит отпуска",
            "hardness_hb_range": (400, 500),
            "applications": "Пружины, рессоры, штампы",
            "ductility": "средняя",
            "toughness": "средняя"
        },
        "low": {
            "name_ru": "Высокий отпуск",
            "temperature_range": (550, 650),
            "typical": 600,
            "result": "Сорбит отпуска",
            "hardness_hb_range": (250, 350),
            "applications": "Валы, шестерни, оси (улучшение)",
            "ductility": "высокая",
            "toughness": "высокая"
        }
    }
    reference = _textbook_reference()
    tempering = dict(reference.get("heat_treatment_temperatures", {}).get("tempering", {})) if isinstance(reference, dict) else {}
    if tempering:
        mapping = {"high": "low", "medium": "medium", "low": "high"}
        for target_key, ref_key in mapping.items():
            payload = dict(tempering.get(ref_key, {}))
            if not payload:
                continue
            temp_range = dict(payload.get("temperature_range", {}))
            hb_range = dict(payload.get("hardness_hb", {}))
            regimes[target_key].update(
                {
                    "name_ru": str(payload.get("name_ru", regimes[target_key]["name_ru"])),
                    "temperature_range": (
                        int(round(_safe_float(temp_range.get("min"), regimes[target_key]["temperature_range"][0]))),
                        int(round(_safe_float(temp_range.get("max"), regimes[target_key]["temperature_range"][1]))),
                    ),
                    "typical": int(round(_safe_float(payload.get("typical"), regimes[target_key]["typical"]))),
                    "result": str(payload.get("result", regimes[target_key]["result"])),
                    "hardness_hb_range": (
                        int(round(_safe_float(hb_range.get("min"), regimes[target_key]["hardness_hb_range"][0]))),
                        int(round(_safe_float(hb_range.get("max"), regimes[target_key]["hardness_hb_range"][1]))),
                    ),
                    "applications": str(payload.get("application", regimes[target_key]["applications"])),
                }
            )

    regime = regimes.get(hardness_target, regimes["low"])

    # Поправка на легирование (Cr, Mo, V повышают температуру отпуска)
    temp_correction = 10.0 * (composition.Cr + composition.Mo + composition.V)
    typical_temp = regime["typical"] + min(50, temp_correction)

    return {
        "name_ru": regime["name_ru"],
        "temperature_range_c": regime["temperature_range"],
        "recommended_temperature_c": round(typical_temp, 0),
        "result_microstructure": regime["result"],
        "expected_hardness_hb": regime["hardness_hb_range"],
        "applications": regime["applications"],
        "ductility": regime["ductility"],
        "toughness": regime["toughness"]
    }


def estimate_hardenability(composition: SteelComposition) -> dict[str, Any]:
    """
    Оценка прокаливаемости стали по химическому составу.
    Упрощённый расчёт идеального критического диаметра (Dci).

    Возвращает оценку способности стали закаливаться на заданную глубину.
    """
    c = composition.C

    # Базовая прокаливаемость для углеродистой стали
    if c < 0.3:
        base_dci = 15.0  # мм
    elif c < 0.5:
        base_dci = 25.0
    elif c < 0.7:
        base_dci = 35.0
    else:
        base_dci = 45.0

    # Поправки на легирующие элементы (коэффициенты Grossmann)
    mn_factor = 1.0 + 0.7 * composition.Mn
    si_factor = 1.0 + 0.3 * composition.Si
    cr_factor = 1.0 + 2.0 * composition.Cr
    ni_factor = 1.0 + 0.4 * composition.Ni
    mo_factor = 1.0 + 1.5 * composition.Mo

    total_factor = mn_factor * si_factor * cr_factor * ni_factor * mo_factor
    dci = base_dci * total_factor

    # Оценка
    if dci < 20:
        hardenability = "низкая"
        quench_medium = "Вода"
    elif dci < 40:
        hardenability = "средняя"
        quench_medium = "Вода или масло"
    elif dci < 60:
        hardenability = "хорошая"
        quench_medium = "Масло"
    else:
        hardenability = "высокая"
        quench_medium = "Масло или воздух"

    return {
        "ideal_critical_diameter_mm": round(dci, 1),
        "hardenability_rating": hardenability,
        "recommended_quench_medium": quench_medium,
        "factors": {
            "carbon_contribution": base_dci,
            "manganese_factor": round(mn_factor, 2),
            "silicon_factor": round(si_factor, 2),
            "chromium_factor": round(cr_factor, 2),
            "nickel_factor": round(ni_factor, 2),
            "molybdenum_factor": round(mo_factor, 2),
            "total_factor": round(total_factor, 2)
        }
    }


def calculate_phase_fractions_fe_c(carbon_pct: float, temperature_c: float) -> dict[str, float]:
    """
    Расчёт количественного содержания фаз в сплавах Fe-C по правилу рычага.

    Args:
        carbon_pct: Содержание углерода в масс.%
        temperature_c: Температура в °C

    Returns:
        Словарь с долями фаз (от 0 до 1)
    """
    c = carbon_pct
    t = temperature_c

    phases = {
        "ferrite": 0.0,
        "austenite": 0.0,
        "cementite": 0.0,
        "pearlite": 0.0,
        "ledeburite": 0.0,
        "liquid": 0.0
    }

    # Выше ликвидуса
    if t > 1539 and c < 2.14:
        phases["liquid"] = 1.0
        return phases

    if t > 1147 and c > 2.14:
        # Область L + γ или L + Fe3C
        if c < 4.3:
            # L + γ
            liquidus_c = 2.14 + (4.3 - 2.14) * (1539 - t) / (1539 - 1147)
            solidus_c = 2.14
            if liquidus_c > solidus_c:
                phases["liquid"] = max(0, min(1, (c - solidus_c) / (liquidus_c - solidus_c)))
                phases["austenite"] = 1 - phases["liquid"]
        else:
            # L + Fe3C
            phases["liquid"] = max(0, min(1, (6.67 - c) / (6.67 - 4.3)))
            phases["cementite"] = 1 - phases["liquid"]
        return phases

    # Эвтектическая реакция
    if abs(t - 1147) < 1 and c > 2.14:
        if c < 4.3:
            phases["austenite"] = (4.3 - c) / (4.3 - 2.14)
            phases["ledeburite"] = 1 - phases["austenite"]
        else:
            phases["cementite"] = (c - 4.3) / (6.67 - 4.3)
            phases["ledeburite"] = 1 - phases["cementite"]
        return phases

    # Ниже эвтектики, выше эвтектоида
    if 727 < t < 1147:
        if c < 0.8:
            # α + γ
            gamma_c = 0.8  # упрощённо
            phases["austenite"] = max(0, min(1, (c - 0.02) / (gamma_c - 0.02)))
            phases["ferrite"] = 1 - phases["austenite"]
        elif c < 2.14:
            # γ + Fe3C
            phases["austenite"] = max(0, min(1, (6.67 - c) / (6.67 - 0.8)))
            phases["cementite"] = 1 - phases["austenite"]
        elif c < 4.3:
            phases["austenite"] = (4.3 - c) / (4.3 - 2.14)
            phases["cementite"] = 1 - phases["austenite"]
        else:
            phases["ledeburite"] = (6.67 - c) / (6.67 - 4.3)
            phases["cementite"] = 1 - phases["ledeburite"]
        return phases

    # Эвтектоидная реакция
    if abs(t - 727) < 1:
        if c < 0.8:
            phases["ferrite"] = (0.8 - c) / (0.8 - 0.02)
            phases["pearlite"] = 1 - phases["ferrite"]
        elif c < 2.14:
            phases["cementite"] = (c - 0.8) / (6.67 - 0.8)
            phases["pearlite"] = 1 - phases["cementite"]
        return phases

    # Ниже эвтектоида
    if t < 727:
        if c < 0.02:
            phases["ferrite"] = 1.0
        elif c < 0.8:
            phases["ferrite"] = (0.8 - c) / (0.8 - 0.02)
            phases["pearlite"] = 1 - phases["ferrite"]
        elif c < 2.14:
            phases["pearlite"] = (6.67 - c) / (6.67 - 0.8)
            phases["cementite"] = 1 - phases["pearlite"]
        elif c < 4.3:
            phases["pearlite"] = (6.67 - c) / (6.67 - 0.8) * 0.52
            phases["ledeburite"] = (6.67 - c) / (6.67 - 4.3) * 0.48
            phases["cementite"] = 1 - phases["pearlite"] - phases["ledeburite"]
        else:
            phases["ledeburite"] = (6.67 - c) / (6.67 - 4.3)
            phases["cementite"] = 1 - phases["ledeburite"]

    return {k: round(v, 4) for k, v in phases.items()}


def get_heat_treatment_recommendations(composition: SteelComposition) -> dict[str, Any]:
    """
    Полные рекомендации по термической обработке стали.

    Возвращает комплексные рекомендации на основе химического состава.
    """
    c = composition.C

    # Определение типа стали
    if c < 0.25:
        steel_class = "low_carbon"
        steel_class_ru = "Низкоуглеродистая"
    elif c < 0.60:
        steel_class = "medium_carbon"
        steel_class_ru = "Среднеуглеродистая"
    else:
        steel_class = "high_carbon"
        steel_class_ru = "Высокоуглеродистая"

    # Режим закалки
    if c < 0.8:
        quench_info = get_quench_temperature(composition, "hypoeutectoid")
    elif abs(c - 0.8) < 0.05:
        quench_info = get_quench_temperature(composition, "eutectoid")
    else:
        quench_info = get_quench_temperature(composition, "hypereutectoid")

    # Прокаливаемость
    hardenability = estimate_hardenability(composition)

    # Рекомендации по отпуску для разных применений
    tempering_recommendations = {
        "cutting_tool": get_tempering_temperature("high", composition),
        "spring": get_tempering_temperature("medium", composition),
        "structural": get_tempering_temperature("low", composition)
    }

    # Фазы при комнатной температуре
    phases_rt = calculate_phase_fractions_fe_c(c, 20)

    return {
        "steel_classification": {
            "class": steel_class,
            "class_ru": steel_class_ru,
            "carbon_content_pct": round(c * 100, 2)
        },
        "quenching": quench_info,
        "hardenability": hardenability,
        "tempering_options": tempering_recommendations,
        "room_temperature_phases": phases_rt,
        "critical_temperatures": {
            "ac1": round(calculate_ac1(composition), 0),
            "ac3": round(calculate_ac3(composition), 0) if c < 0.8 else None,
            "ms": round(calculate_ms_temperature(composition), 0),
            "mf": round(calculate_mf_temperature(composition), 0)
        }
    }
