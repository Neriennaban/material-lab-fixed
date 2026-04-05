from __future__ import annotations

from typing import Any

import numpy as np

from .materials import MaterialPreset


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def canonicalize_composition(composition: dict[str, Any] | None) -> dict[str, float]:
    if not composition:
        return {}

    aliases = {
        "c": "C",
        "carbon": "C",
        "si": "Si",
        "silicon": "Si",
        "zn": "Zn",
        "zinc": "Zn",
        "cu": "Cu",
        "copper": "Cu",
        "al": "Al",
        "aluminum": "Al",
        "aluminium": "Al",
        "mg": "Mg",
        "magnesium": "Mg",
        "mn": "Mn",
        "ni": "Ni",
        "cr": "Cr",
        "fe": "Fe",
    }

    out: dict[str, float] = {}
    for raw_key, raw_value in composition.items():
        key = str(raw_key).strip().lower().replace("%", "").replace("_wt", "").replace("wt", "")
        mapped = aliases.get(key, str(raw_key).strip())
        try:
            out[mapped] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return out


def merge_composition(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, float]:
    merged = canonicalize_composition(base)
    if override:
        for key, value in canonicalize_composition(override).items():
            merged[key] = float(value)
    return merged


def apply_composition_rules(
    preset: MaterialPreset,
    generation_params: dict[str, Any],
    composition: dict[str, float],
) -> tuple[dict[str, Any], list[str]]:
    """
    Translate composition (wt.%) into generator parameters.

    The mapping is educational and deterministic, focused on visual realism.
    """

    params = dict(generation_params)
    notes: list[str] = []

    material = preset.material.lower()
    generator = preset.generator.lower().strip()
    c = float(composition.get("C", 0.0))
    si = float(composition.get("Si", 0.0))
    zn = float(composition.get("Zn", 0.0))
    cu = float(composition.get("Cu", 0.0))
    al = float(composition.get("Al", 0.0))
    mg = float(composition.get("Mg", 0.0))

    # Fe-C steels / pearlite response.
    if c > 0.0 and (generator in {"pearlite", "tempered", "martensite"} or "steel" in material):
        pearlite_fraction = _clamp(c / 0.82, 0.08, 0.96)
        params["pearlite_fraction"] = pearlite_fraction
        params["lamella_period_px"] = _clamp(8.8 - 3.8 * min(c, 1.2), 3.6, 9.2)
        params["ferrite_brightness"] = int(_clamp(198 - 72 * min(c, 1.1), 110, 205))
        params["cementite_brightness"] = int(_clamp(78 - 18 * min(c, 1.2), 40, 84))
        notes.append(f"C={c:.3f}% -> pearlite_fraction={pearlite_fraction:.2f}")

    # Fe-Si steels.
    if si > 0.0 and ("fe-si" in material or ("steel" in material and generator == "grains")):
        base_size = float(params.get("mean_grain_size_px", 48.0))
        params["mean_grain_size_px"] = _clamp(base_size * (1.0 + 0.045 * (si - 1.4)), 26.0, 96.0)
        params["boundary_contrast"] = _clamp(
            float(params.get("boundary_contrast", 0.5)) + 0.03 * (si - 1.4),
            0.28,
            0.85,
        )
        notes.append(f"Si={si:.2f}% -> grain_size={params['mean_grain_size_px']:.1f}")

    # Cu-Zn brasses (alpha-brass educational model).
    if zn > 0.0 and ("brass" in material or generator == "grains"):
        base_size = float(params.get("mean_grain_size_px", 54.0))
        params["mean_grain_size_px"] = _clamp(base_size * (1.0 + 0.008 * (zn - 32.0)), 28.0, 92.0)
        params["boundary_contrast"] = _clamp(
            float(params.get("boundary_contrast", 0.52)) + 0.005 * (zn - 32.0),
            0.32,
            0.86,
        )
        params["inclusion_fraction"] = _clamp(
            float(params.get("inclusion_fraction", 0.0012)) + 0.00003 * abs(zn - 32.0),
            0.0,
            0.008,
        )
        notes.append(f"Zn={zn:.1f}% -> alpha/brass contrast tuned")

    # Al-Si cast alloys.
    if si > 0.0 and (generator == "eutectic" or "al-si" in material):
        si_fraction = _clamp(0.06 + 0.022 * si, 0.12, 0.68)
        params["si_phase_fraction"] = si_fraction
        params["eutectic_scale_px"] = _clamp(9.2 - 0.28 * si, 3.8, 10.0)
        if si >= 11.0:
            params["morphology"] = "needle"
        elif si >= 7.0:
            params["morphology"] = "branched"
        else:
            params["morphology"] = "network"
        notes.append(f"Si={si:.1f}% -> si_phase_fraction={si_fraction:.2f}")

    # Duralumin / age hardenable Al-Cu-Mg alloys.
    if generator == "aged_al" or ("dural" in material or ("al" in material and cu > 0.0)):
        if cu <= 0.0:
            cu = 4.2
        if mg <= 0.0:
            mg = 1.4
        precip = _clamp(0.028 + 0.013 * cu + 0.011 * mg, 0.04, 0.2)
        params["precipitate_fraction"] = precip
        params["precipitate_scale_px"] = _clamp(2.8 - 0.22 * mg, 1.2, 3.4)
        notes.append(f"Cu={cu:.2f}% Mg={mg:.2f}% -> precipitate_fraction={precip:.2f}")

    # Pure copper visual tuning.
    if cu > 0.0 and "copper" in material and generator == "grains":
        params["mean_grain_size_px"] = _clamp(
            float(params.get("mean_grain_size_px", 52.0)) * (1.0 + 0.002 * (cu - 99.0)),
            20.0,
            110.0,
        )
        notes.append(f"Cu={cu:.2f}% -> conductivity-grade grain tuning")

    # Store normalized sum for visibility in metadata/UI.
    if composition:
        total = float(np.sum(list(composition.values())))
        notes.append(f"composition_total={total:.2f} wt.%")

    return params, notes

