from __future__ import annotations

from typing import Any

import numpy as np

from .cache import CalphadCache
from .db_manager import CalphadDBReference
from .engine_pycalphad import run_equilibrium_grid


def run_scheil(
    *,
    db_ref: CalphadDBReference,
    system: str,
    composition: dict[str, float],
    t_start_c: float,
    t_end_c: float,
    d_t_c: float = 5.0,
    pressure_pa: float = 101325.0,
    equilibrium_model: str = "global_min",
    cache: CalphadCache | None = None,
) -> dict[str, Any]:
    step = max(0.1, abs(float(d_t_c)))
    t0 = float(t_start_c)
    t1 = float(t_end_c)

    if t0 >= t1:
        temps = np.arange(t0, t1 - 1e-9, -step, dtype=float).tolist()
    else:
        temps = np.arange(t0, t1 + 1e-9, step, dtype=float).tolist()
    if not temps:
        temps = [t0]

    points = run_equilibrium_grid(
        db_ref=db_ref,
        system=system,
        composition=composition,
        temperatures_c=temps,
        pressure_pa=float(pressure_pa),
        equilibrium_model=equilibrium_model,
        cache=cache,
    )

    trajectory: list[dict[str, Any]] = []
    solid_progress = 0.0
    for idx, item in enumerate(points):
        liq = float(item.liquid_fraction)
        sol = float(max(0.0, min(1.0, 1.0 - liq)))
        # Enforce Scheil-like monotonic increase of solid fraction during cooling.
        if t0 >= t1:
            solid_progress = max(solid_progress, sol)
        else:
            solid_progress = sol
        liq_progress = float(max(0.0, min(1.0, 1.0 - solid_progress)))
        trajectory.append(
            {
                "index": idx,
                "temperature_c": float(item.temperature_c),
                "f_liquid": liq_progress,
                "f_solid": solid_progress,
                "stable_phases": dict(item.stable_phases),
                "solver_status": item.solver_status,
            }
        )

    return {
        "enabled": True,
        "mode": "scheil_gulliver",
        "t_start_c": float(t0),
        "t_end_c": float(t1),
        "d_t_c": float(step),
        "point_count": len(trajectory),
        "trajectory": trajectory,
    }

