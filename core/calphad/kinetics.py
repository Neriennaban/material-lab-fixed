from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

_RULEBOOK_PATH = Path(__file__).resolve().parents[1] / "rulebook" / "kinetics_rules.json"


def _load_rules() -> dict[str, Any]:
    try:
        return json.loads(_RULEBOOK_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"defaults": {}, "systems": {}}


_RULES = _load_rules()


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def run_jmak_lsw(
    *,
    system: str,
    temperature_c: float,
    aging_hours: float,
    base_phase_fractions: dict[str, float] | None = None,
) -> dict[str, Any]:
    defaults = _RULES.get("defaults", {}) if isinstance(_RULES.get("defaults"), dict) else {}
    sys_map = _RULES.get("systems", {}) if isinstance(_RULES.get("systems"), dict) else {}
    params = dict(defaults)
    params.update(dict(sys_map.get(system, {})))

    k0 = float(params.get("k0", 0.02))
    n = float(params.get("n", 1.6))
    q_j_mol = float(params.get("q_j_mol", 85_000.0))
    r_gas = 8.314
    temp_k = max(1.0, float(temperature_c) + 273.15)
    hours = max(0.0, float(aging_hours))
    t_seconds = hours * 3600.0
    k_t = k0 * math.exp(-q_j_mol / (r_gas * temp_k))
    transformed = 1.0 - math.exp(-((k_t * t_seconds) ** max(0.2, n)))
    transformed = _clamp(transformed, 0.0, 1.0)

    r0_nm = float(params.get("r0_nm", 2.0))
    k_lsw = float(params.get("k_lsw_nm3_s", 0.005))
    r_nm = (max(0.0, r0_nm**3 + k_lsw * t_seconds)) ** (1.0 / 3.0)

    base_precip = 0.0
    if isinstance(base_phase_fractions, dict):
        for name, frac in base_phase_fractions.items():
            if any(tag in str(name).upper() for tag in ("THETA", "ETA", "KAPPA", "CARBIDE", "PRECIP")):
                base_precip += max(0.0, float(frac))
    precipitate_fraction = _clamp(base_precip + transformed * float(params.get("precip_scale", 0.18)), 0.0, 0.45)

    state = "underaged"
    if transformed > 0.65:
        state = "peak_aged"
    if transformed > 0.88:
        state = "overaged"

    return {
        "enabled": True,
        "model": "JMAK+LSW",
        "system": str(system),
        "temperature_c": float(temperature_c),
        "aging_hours": float(hours),
        "jmak_fraction": float(round(transformed, 6)),
        "precipitate_fraction": float(round(precipitate_fraction, 6)),
        "mean_radius_nm": float(round(r_nm, 4)),
        "coarsening_state": state,
    }

