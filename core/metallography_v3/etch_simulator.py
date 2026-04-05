from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from core.contracts_v3 import EtchProfileV3
from core.metallography_v3.realism_utils import clamp, low_frequency_field, normalize01


ETCH_PRESETS: dict[str, dict[str, float]] = {
    "nital_2": {"base": 1.0, "boundary_boost": 0.34, "phase_contrast": 0.28},
    "picral": {"base": 0.88, "boundary_boost": 0.31, "phase_contrast": 0.24},
    "keller": {"base": 0.95, "boundary_boost": 0.26, "phase_contrast": 0.22},
    "fry": {"base": 1.08, "boundary_boost": 0.38, "phase_contrast": 0.32},
    "custom": {"base": 1.0, "boundary_boost": 0.25, "phase_contrast": 0.2},
}

_ELECTROLYTIC_PROFILE_DEFAULTS: dict[str, dict[str, float]] = {
    "pure_iron_electropolish": {"voltage_v": 30.0, "temperature_c": 20.0},
    "steel_electropolish": {"voltage_v": 35.0, "temperature_c": 20.0},
    "stainless_electropolish": {"voltage_v": 35.0, "temperature_c": 20.0},
    "copper_alloy_electropolish": {"voltage_v": 5.0, "temperature_c": 20.0},
    "aluminum_electropolish": {"voltage_v": 35.0, "temperature_c": 20.0},
}

_RULES_PATH = (
    Path(__file__).resolve().parents[1]
    / "rulebook"
    / "etch_concentration_rules_v3.json"
)


def _load_concentration_rules() -> dict[str, Any]:
    if not _RULES_PATH.exists():
        return {}
    try:
        return json.loads(_RULES_PATH.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


_CONC_RULES = _load_concentration_rules()


def _phase_rank(phase_name: str) -> float:
    key = str(phase_name).upper()
    if "LIQUID" in key:
        return 0.2
    if any(
        tok in key
        for tok in ("CEMENT", "CARB", "THETA", "ETA", "KAPPA", "S_PHASE", "QPHASE")
    ):
        return 1.25
    if any(tok in key for tok in ("MARTENSITE", "BAINITE", "TROOSTITE", "SORBITE")):
        return 1.10
    if any(tok in key for tok in ("BCC", "FCC", "ALPHA", "BETA", "GAMMA")):
        return 1.0
    if any(tok in key for tok in ("SI", "INTERMET", "PRECIP")):
        return 1.15
    return 0.95


def _phase_selectivity(*, phase_name: str, reagent: str, system: str | None) -> float:
    key = str(phase_name).upper()
    reagent_l = str(reagent or "custom").strip().lower()
    system_l = str(system or "").strip().lower()
    base = _phase_rank(key)

    if reagent_l == "nital_2":
        if any(tok in key for tok in ("MARTENSITE", "BAINITE", "TROOSTITE", "SORBITE")):
            base += 0.12
        elif any(tok in key for tok in ("CEMENTITE", "CARB")):
            base -= 0.08
        elif "PEARLITE" in key:
            base += 0.06
    elif reagent_l == "picral":
        if any(tok in key for tok in ("CEMENTITE", "CARB", "PEARLITE")):
            base += 0.18
        elif any(
            tok in key for tok in ("FERRITE", "AUSTENITE", "ALPHA", "FCC_A1", "BCC")
        ):
            base -= 0.06
    elif reagent_l == "keller":
        if system_l.startswith("al"):
            if any(
                tok in key for tok in ("SI", "THETA", "S_PHASE", "QPHASE", "PRECIP")
            ):
                base += 0.10
            elif any(tok in key for tok in ("FCC_A1", "ALPHA")):
                base -= 0.04
    elif reagent_l == "fry":
        if any(tok in key for tok in ("AUSTENITE", "MARTENSITE")):
            base += 0.12
    return float(base)


def _composition_fraction(composition_wt: dict[str, float] | None, key: str) -> float:
    if not isinstance(composition_wt, dict):
        return 0.0
    total = 0.0
    cleaned: dict[str, float] = {}
    for name, value in composition_wt.items():
        try:
            vv = float(value)
        except Exception:
            continue
        if vv <= 0.0:
            continue
        cleaned[str(name).strip()] = vv
        total += vv
    if total <= 1e-12:
        return 0.0
    return float(cleaned.get(key, 0.0) / total * 100.0)


def _phase_coverage_fraction(
    phase_masks: dict[str, np.ndarray] | None, candidates: set[str]
) -> float:
    if not isinstance(phase_masks, dict):
        return 0.0
    total = 0.0
    for name, mask in phase_masks.items():
        if not isinstance(mask, np.ndarray):
            continue
        if str(name).upper() in candidates:
            total += float((mask > 0).mean())
    return float(total)


def _is_pure_iron_like(
    *,
    system: str | None,
    composition_wt: dict[str, float] | None,
    phase_masks: dict[str, np.ndarray] | None,
) -> bool:
    sys_name = str(system or "").strip().lower()
    if sys_name not in {"fe-si", "fe-c"}:
        return False
    fe_pct = _composition_fraction(composition_wt, "Fe")
    c_pct = _composition_fraction(composition_wt, "C")
    si_pct = _composition_fraction(composition_wt, "Si")
    if isinstance(phase_masks, dict) and set(str(k) for k in phase_masks.keys()) == {
        "solid"
    }:
        ferritic_cov = 1.0
        dark_cov = 0.0
    else:
        ferritic_cov = _phase_coverage_fraction(
            phase_masks, {"BCC_B2", "FERRITE", "DELTA_FERRITE"}
        )
        dark_cov = _phase_coverage_fraction(
            phase_masks,
            {
                "PEARLITE",
                "CEMENTITE",
                "MARTENSITE",
                "BAINITE",
                "FESI_INTERMETALLIC",
                "THETA",
                "S_PHASE",
                "QPHASE",
            },
        )
    return bool(
        fe_pct >= 99.8
        and c_pct <= 0.03
        and si_pct <= 0.25
        and ferritic_cov >= 0.92
        and dark_cov <= 0.08
    )


def _resolve_concentration(etch_profile: EtchProfileV3) -> dict[str, Any]:
    reag = str(etch_profile.reagent or "custom")
    rules = (
        dict(_CONC_RULES.get("reagents", {})).get(reag, {})
        if isinstance(_CONC_RULES, dict)
        else {}
    )
    unit = str(etch_profile.concentration_unit or "wt_pct").strip().lower()
    if unit not in {"wt_pct", "mol_l"}:
        unit = "wt_pct"

    val = float(etch_profile.concentration_value)
    wt = float(etch_profile.concentration_wt_pct)
    mol = float(etch_profile.concentration_mol_l)

    basis = dict(rules.get("conversion_basis", {})) if isinstance(rules, dict) else {}
    molar_mass = float(basis.get("molar_mass_g_mol", 63.0))
    density = float(basis.get("density_g_ml", 1.0))

    if unit == "wt_pct":
        wt = float(val)
        mol = (wt * max(0.1, density) * 10.0) / max(1e-6, molar_mass)
    else:
        mol = float(val)
        wt = (mol * max(1e-6, molar_mass)) / (max(0.1, density) * 10.0)

    wt_range = (
        list(rules.get("wt_pct_range", [0.1, 20.0]))
        if isinstance(rules, dict)
        else [0.1, 20.0]
    )
    mol_range = (
        list(rules.get("mol_l_range", [0.01, 5.0]))
        if isinstance(rules, dict)
        else [0.01, 5.0]
    )
    wt_min, wt_max = float(wt_range[0]), float(wt_range[1])
    mol_min, mol_max = float(mol_range[0]), float(mol_range[1])

    warnings: list[str] = []
    if wt < wt_min or wt > wt_max:
        warnings.append(
            f"Концентрация wt.% вне рекомендованного диапазона [{wt_min}; {wt_max}]."
        )
    if mol < mol_min or mol > mol_max:
        warnings.append(
            f"Концентрация mol/L вне рекомендованного диапазона [{mol_min}; {mol_max}]."
        )

    wt = clamp(wt, 0.01, 100.0)
    mol = clamp(mol, 1e-4, 20.0)

    return {
        "reagent": reag,
        "unit": unit,
        "input_value": float(val),
        "wt_pct": float(wt),
        "mol_l": float(mol),
        "wt_pct_range": [wt_min, wt_max],
        "mol_l_range": [mol_min, mol_max],
        "within_range": bool(not warnings),
        "warnings": warnings,
    }


def apply_etch(
    *,
    image_gray: np.ndarray,
    phase_masks: dict[str, np.ndarray],
    etch_profile: EtchProfileV3,
    seed: int,
    prep_maps: dict[str, np.ndarray] | None = None,
    system: str | None = None,
    composition_wt: dict[str, float] | None = None,
    effect_vector: dict[str, float] | None = None,
) -> dict[str, Any]:
    arr = image_gray.astype(np.float32).copy()
    h, w = arr.shape
    rng = np.random.default_rng(seed + 911)

    preset = ETCH_PRESETS.get(etch_profile.reagent, ETCH_PRESETS["custom"])
    base = float(preset["base"])
    boundary_boost = float(preset["boundary_boost"])
    phase_contrast = float(preset["phase_contrast"])

    concentration = _resolve_concentration(etch_profile)
    wt_scale = clamp(float(concentration["wt_pct"]) / 2.0, 0.4, 2.2)
    pure_iron_like = _is_pure_iron_like(
        system=system, composition_wt=composition_wt, phase_masks=phase_masks
    )
    etch_mode = (
        str(getattr(etch_profile, "etch_mode", "chemical") or "chemical")
        .strip()
        .lower()
    )
    electrolyte_code = (
        str(getattr(etch_profile, "electrolyte_code", "") or "").strip().lower()
    )
    voltage_v = float(getattr(etch_profile, "voltage_v", 0.0) or 0.0)
    voltage_ratio_to_polish = float(
        getattr(etch_profile, "voltage_ratio_to_polish", 0.0) or 0.0
    )
    current_density = float(getattr(etch_profile, "current_density_a_cm2", 0.0) or 0.0)
    area_mode = (
        str(getattr(etch_profile, "area_mode", "global") or "global").strip().lower()
    )

    time_factor = float(max(0.1, etch_profile.time_s / 8.0))
    temp_factor = float(
        max(0.5, min(1.8, (etch_profile.temperature_c + 273.15) / (22.0 + 273.15)))
    )
    overetch = float(max(0.5, min(2.2, etch_profile.overetch_factor)))
    agitation = str(etch_profile.agitation).lower().strip()
    agitation_factor = {"none": 0.9, "gentle": 1.0, "active": 1.12}.get(agitation, 1.0)

    etch_rate = np.full(
        (h, w),
        base * time_factor * temp_factor * agitation_factor * overetch * wt_scale,
        dtype=np.float32,
    )
    selectivity_field = np.zeros((h, w), dtype=np.float32)

    if phase_masks:
        for idx, (phase, mask) in enumerate(phase_masks.items()):
            zone = (mask > 0).astype(np.float32)
            if zone.max() <= 0.0:
                continue
            selectivity = _phase_selectivity(
                phase_name=phase, reagent=etch_profile.reagent, system=system
            )
            etch_rate += zone * ((selectivity - 1.0) * phase_contrast)
            selectivity_field += zone * selectivity
            if ndimage is not None:
                border = ndimage.binary_dilation(zone > 0, iterations=1) ^ (zone > 0)
                etch_rate += (
                    border.astype(np.float32) * boundary_boost * (0.66 + 0.08 * idx)
                )

    prep_coupling_applied = False
    topography = np.zeros((h, w), dtype=np.float32)
    scratch = np.zeros((h, w), dtype=np.float32)
    deformation = np.zeros((h, w), dtype=np.float32)
    smear = np.zeros((h, w), dtype=np.float32)
    contamination = np.zeros((h, w), dtype=np.float32)
    pullout = np.zeros((h, w), dtype=np.float32)
    relief = np.zeros((h, w), dtype=np.float32)
    electropolish_local_mask = np.zeros((h, w), dtype=np.float32)
    if isinstance(prep_maps, dict) and prep_maps:
        prep_coupling_applied = True
        topography = (
            prep_maps.get("topography", topography).astype(np.float32) / 255.0
            if isinstance(prep_maps.get("topography"), np.ndarray)
            else topography
        )
        scratch = (
            prep_maps.get("scratch", scratch).astype(np.float32) / 255.0
            if isinstance(prep_maps.get("scratch"), np.ndarray)
            else scratch
        )
        deformation = (
            prep_maps.get("deformation_layer", deformation).astype(np.float32) / 255.0
            if isinstance(prep_maps.get("deformation_layer"), np.ndarray)
            else deformation
        )
        smear = (
            prep_maps.get("smear", smear).astype(np.float32) / 255.0
            if isinstance(prep_maps.get("smear"), np.ndarray)
            else smear
        )
        contamination = (
            prep_maps.get("contamination", contamination).astype(np.float32) / 255.0
            if isinstance(prep_maps.get("contamination"), np.ndarray)
            else contamination
        )
        pullout = (
            prep_maps.get("pullout", pullout).astype(np.float32) / 255.0
            if isinstance(prep_maps.get("pullout"), np.ndarray)
            else pullout
        )
        relief = (
            prep_maps.get("relief", relief).astype(np.float32) / 255.0
            if isinstance(prep_maps.get("relief"), np.ndarray)
            else relief
        )
        electropolish_local_mask = (
            prep_maps.get("electropolish_local_mask", electropolish_local_mask).astype(
                np.float32
            )
            / 255.0
            if isinstance(prep_maps.get("electropolish_local_mask"), np.ndarray)
            else electropolish_local_mask
        )

    effect_vector = dict(effect_vector or {})
    dislocation = float(max(0.0, min(1.0, effect_vector.get("dislocation_proxy", 0.0))))
    low_etch = low_frequency_field((h, w), seed=seed + 37, sigma=18.0)
    preferential_attack = (
        deformation * 0.24 + scratch * 0.12 + pullout * 0.20 + dislocation * 0.08
    )
    shielding = smear * 0.18 + contamination * 0.14
    stain_map = normalize01(contamination * 0.55 + smear * 0.25 + low_etch * 0.20)
    relief_map = normalize01(
        (topography - 0.5) * 0.55 + (relief - 0.5) * 0.30 + pullout * 0.15
    )
    relief_edges = normalize01(
        np.hypot(np.gradient(relief_map, axis=1), np.gradient(relief_map, axis=0))
    )
    phase_edges = normalize01(selectivity_field)

    pure_iron_dark_defect_suppression = 0.0
    pure_iron_cleanliness_score = 0.0
    pure_iron_boundary_visibility_score = 0.0
    aggressive_etch = False
    if pure_iron_like:
        aggressive_etch = bool(
            etch_profile.time_s > 10.0
            or float(etch_profile.overetch_factor) > 1.15
            or str(etch_profile.agitation).strip().lower() == "active"
            or float(concentration["wt_pct"]) > 3.0
        )
        attack_scale = 0.28 if not aggressive_etch else 0.72
        stain_scale = 0.10 if not aggressive_etch else 0.45
        relief_scale = 0.58 if not aggressive_etch else 0.88
        preferential_attack *= attack_scale
        stain_map *= stain_scale
        relief_map = normalize01((relief_map - 0.5) * relief_scale + 0.5)
        low_etch = (low_etch - 0.5) * (0.34 if not aggressive_etch else 0.78) + 0.5
        boundary_boost *= 1.06 if not aggressive_etch else 1.0
        phase_contrast *= 0.62 if not aggressive_etch else 0.92
        pure_iron_dark_defect_suppression = float(
            clamp(1.0 - 0.5 * (attack_scale + stain_scale), 0.0, 1.0)
        )

    electrolytic_followup = False
    if etch_mode == "electrolytic":
        electrolytic_followup = True
        base_profile = dict(_ELECTROLYTIC_PROFILE_DEFAULTS.get(electrolyte_code, {}))
        if voltage_v <= 0.0:
            voltage_v = float(base_profile.get("voltage_v", 0.0))
        if voltage_ratio_to_polish <= 0.0:
            voltage_ratio_to_polish = 0.1
        local_mask = np.ones((h, w), dtype=np.float32)
        if area_mode == "local" and np.any(electropolish_local_mask > 0):
            local_mask = np.clip(electropolish_local_mask, 0.0, 1.0)
        phase_relief_risk = float(
            np.clip(
                (0.20 + 0.70 * float(np.std(selectivity_field)))
                * float(local_mask.mean()),
                0.0,
                1.0,
            )
        )
        furrowing_risk = float(
            np.clip(
                (0.12 + 0.35 * float(np.mean(np.abs(np.gradient(local_mask, axis=1)))))
                * max(0.0, voltage_ratio_to_polish),
                0.0,
                1.0,
            )
        )
        pitting_risk = float(
            np.clip(
                (0.08 + 0.18 * max(0.0, voltage_ratio_to_polish - 0.12))
                * (1.15 if etch_profile.agitation == "active" else 1.0),
                0.0,
                1.0,
            )
        )
        edge_effect_risk = float(
            np.clip(
                (1.0 - float(local_mask.mean())) * 0.4 + phase_relief_risk * 0.2,
                0.0,
                1.0,
            )
        )
        passivation_risk = float(
            np.clip(
                0.10 + 0.25 * float(smear.mean()) + (0.10 if pure_iron_like else 0.0),
                0.0,
                1.0,
            )
        )
        electrolytic_weight = np.clip(
            local_mask * (0.75 + 0.25 * min(1.0, voltage_ratio_to_polish / 0.1)),
            0.0,
            1.0,
        )
        etch_rate = (
            etch_rate * (1.0 - 0.32 * electrolytic_weight)
            + (0.88 + 0.18 * relief_edges) * electrolytic_weight
        )
        stain_map = stain_map * (1.0 - 0.68 * electrolytic_weight)
        relief_map = normalize01(
            relief_map * (1.0 + 0.30 * electrolytic_weight)
            + phase_edges * 0.12 * electrolytic_weight
        )
        preferential_attack *= 1.0 - 0.45 * electrolytic_weight
    else:
        phase_relief_risk = 0.0
        furrowing_risk = 0.0
        pitting_risk = 0.0
        edge_effect_risk = 0.0
        passivation_risk = 0.0
    etch_rate += preferential_attack
    etch_rate -= shielding
    etch_rate += (low_etch - 0.5) * 0.12 * max(0.8, overetch)

    noise = rng.normal(0.0, 0.07, size=(h, w)).astype(np.float32)
    if ndimage is not None:
        noise = ndimage.gaussian_filter(noise, sigma=2.4)
    if pure_iron_like:
        noise *= 0.55
    etch_rate = np.clip(etch_rate + noise, 0.22, 3.2)

    attack_gain = 28.0
    relief_gain = 6.0 + 10.0 * (overetch - 0.5)
    pullout_gain = 10.0
    stain_gain = 4.0 + 10.0 * max(0.0, overetch - 1.0)
    if pure_iron_like:
        attack_gain = 9.0 if not aggressive_etch else 14.0
        relief_gain *= 0.56 if not aggressive_etch else 0.72
        pullout_gain *= 0.22 if not aggressive_etch else 0.40
        stain_gain *= 0.12 if not aggressive_etch else 0.22
    arr = arr - (etch_rate - 1.0) * attack_gain
    arr -= relief_map * relief_gain
    arr -= pullout * pullout_gain
    arr -= stain_map * stain_gain
    if ndimage is not None:
        low = ndimage.gaussian_filter(arr, sigma=1.5)
        unsharp = 0.22 * min(1.6, overetch)
        if pure_iron_like:
            unsharp *= 0.55
        arr = arr + (arr - low) * unsharp
    if pure_iron_like:
        dark_floor = float(np.quantile(arr, 0.08))
        target_floor = 136.0 if not aggressive_etch else 122.0
        arr += max(0.0, target_floor - dark_floor)
        pure_iron_cleanliness_score = float(
            clamp(
                1.0
                - (
                    0.52 * float(stain_map.mean())
                    + 0.28 * float(contamination.mean())
                    + 0.20 * float(scratch.mean())
                ),
                0.0,
                1.0,
            )
        )
        pure_iron_boundary_visibility_score = float(
            clamp(
                0.55 * (1.0 - abs(float(relief_map.mean()) - 0.5) * 2.0)
                + 0.45 * (1.0 - float(stain_map.mean())),
                0.0,
                1.0,
            )
        )
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    etch_maps = {
        "etch_rate": np.clip(etch_rate / 3.2 * 255.0, 0, 255).astype(np.uint8),
        "stain": np.clip(stain_map * 255.0, 0, 255).astype(np.uint8),
        "relief_shading": np.clip(relief_map * 255.0, 0, 255).astype(np.uint8),
        "selectivity": np.clip(normalize01(selectivity_field) * 255.0, 0, 255).astype(
            np.uint8
        ),
    }

    return {
        "image_gray": arr,
        "etch_rate_map": etch_maps["etch_rate"],
        "etch_maps": etch_maps,
        "etch_summary": {
            "reagent": etch_profile.reagent,
            "time_s": float(etch_profile.time_s),
            "temperature_c": float(etch_profile.temperature_c),
            "agitation": etch_profile.agitation,
            "overetch_factor": float(etch_profile.overetch_factor),
            "etch_rate_mean": float(etch_rate.mean()),
            "etch_rate_std": float(etch_rate.std()),
            "concentration": concentration,
            "phase_selectivity_mode": f"{str(system or '').lower() or 'generic'}:{str(etch_profile.reagent).lower()}",
            "prep_coupling_applied": bool(prep_coupling_applied),
            "stain_level_mean": float(stain_map.mean()),
            "relief_shading_mean": float(relief_map.mean()),
            "etch_mode": etch_mode,
            "electrolyte_code": electrolyte_code,
            "voltage_v": float(voltage_v),
            "voltage_ratio_to_polish": float(voltage_ratio_to_polish),
            "current_density_a_cm2": float(current_density),
            "local_area_fraction": float(electropolish_local_mask.mean())
            if area_mode == "local"
            else 1.0,
            "phase_relief_risk": float(phase_relief_risk),
            "furrowing_risk": float(furrowing_risk),
            "pitting_risk": float(pitting_risk),
            "edge_effect_risk": float(edge_effect_risk),
            "passivation_risk": float(passivation_risk),
            "post_electropolish_electroetch_used": bool(electrolytic_followup),
            "post_electropolish_chemical_etch_used": bool(
                not electrolytic_followup
                and bool(getattr(etch_profile, "requires_prior_electropolish", False))
            ),
            "pure_iron_baseline_applied": bool(pure_iron_like),
            "pure_iron_cleanliness_score": float(pure_iron_cleanliness_score),
            "pure_iron_dark_defect_suppression": float(
                pure_iron_dark_defect_suppression
            ),
            "pure_iron_boundary_visibility_score": float(
                pure_iron_boundary_visibility_score
            ),
        },
        "etch_concentration": concentration,
    }
