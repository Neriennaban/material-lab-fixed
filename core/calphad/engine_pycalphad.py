from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cache import CalphadCache
from .db_manager import CalphadDBReference

_DB_OBJECTS: dict[str, Any] = {}
_DEFAULT_CACHE = CalphadCache(
    cache_dir=Path(__file__).resolve().parents[2] / ".cache" / "calphad",
    policy="hybrid",
)

_SYSTEM_COMPONENTS: dict[str, tuple[str, ...]] = {
    "fe-c": ("FE", "C", "VA"),
    "fe-si": ("FE", "SI", "VA"),
    "al-si": ("AL", "SI", "VA"),
    "cu-zn": ("CU", "ZN", "VA"),
    "al-cu-mg": ("AL", "CU", "MG", "VA"),
}


@dataclass(slots=True)
class CalphadEquilibriumResult:
    system: str
    stable_phases: dict[str, float]
    liquid_fraction: float
    solid_fraction: float
    chemical_potentials: dict[str, float]
    solver_status: str
    compute_time_ms: float
    temperature_c: float
    pressure_pa: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_database(path: Path) -> Any:
    key = str(path.resolve())
    if key in _DB_OBJECTS:
        return _DB_OBJECTS[key]
    from pycalphad import Database

    db = Database(str(path))
    _DB_OBJECTS[key] = db
    return db


def _normalize_composition(
    system: str, composition: dict[str, float]
) -> dict[str, float]:
    comps = _SYSTEM_COMPONENTS[system]
    elements = [c for c in comps if c != "VA"]
    values = []
    for el in elements:
        values.append(
            max(0.0, float(composition.get(el.capitalize(), composition.get(el, 0.0))))
        )
    total = max(1e-12, float(sum(values)))
    norm = {el: val / total for el, val in zip(elements, values, strict=True)}
    return norm


def _dominant_component(system: str, composition: dict[str, float]) -> str:
    comp_norm = _normalize_composition(system, composition)
    if not comp_norm:
        return "SOLID"
    return max(comp_norm, key=comp_norm.get)


def _build_conditions(
    system: str, composition: dict[str, float], temperature_c: float, pressure_pa: float
) -> dict[Any, float]:
    from pycalphad import variables as v

    comp_norm = _normalize_composition(system, composition)
    elements = [c for c in _SYSTEM_COMPONENTS[system] if c != "VA"]
    if not elements:
        raise ValueError(f"SOLVER_FAIL: no components for system {system}")
    ref = elements[0]
    cond: dict[Any, float] = {
        v.T: float(temperature_c) + 273.15,
        v.P: float(pressure_pa),
    }
    for el in elements[1:]:
        key = getattr(v, "X")(el)
        cond[key] = float(comp_norm.get(el, 0.0))
    return cond


def _extract_stable_phases(eq: Any) -> dict[str, float]:
    phases = np.asarray(eq.Phase.values, dtype=object).ravel()
    if not hasattr(eq, "NP"):
        found = [str(x) for x in phases if str(x) and str(x).lower() != "nan"]
        if not found:
            return {}
        uniq = sorted(set(found))
        w = 1.0 / max(1, len(uniq))
        return {p: w for p in uniq}

    npv = np.asarray(eq.NP.values, dtype=float).ravel()
    acc: dict[str, float] = {}
    for p, n in zip(phases, npv, strict=False):
        name = str(p).strip()
        if not name or name.lower() == "nan":
            continue
        frac = float(n) if np.isfinite(n) and n > 0 else 0.0
        if frac <= 0:
            continue
        acc[name] = acc.get(name, 0.0) + frac
    total = float(sum(acc.values()))
    if total <= 1e-12:
        found = [
            str(x).strip()
            for x in phases
            if str(x).strip() and str(x).strip().lower() != "nan"
        ]
        if not found:
            return {}
        uniq = sorted(set(found))
        w = 1.0 / max(1, len(uniq))
        return {p: w for p in uniq}
    return {k: float(v / total) for k, v in acc.items()}


def _extract_mu(eq: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    mu = getattr(eq, "MU", None)
    if mu is None:
        return out
    try:
        values = np.asarray(mu.values, dtype=float).ravel()
        comps = [str(x) for x in mu.coords.get("component", [])]
        for idx, comp in enumerate(comps):
            if idx < len(values) and np.isfinite(values[idx]):
                out[str(comp)] = float(values[idx])
    except Exception:
        return {}
    return out


def run_equilibrium(
    *,
    db_ref: CalphadDBReference,
    system: str,
    composition: dict[str, float],
    temperature_c: float,
    pressure_pa: float,
    equilibrium_model: str = "global_min",
    cache: CalphadCache | None = None,
) -> CalphadEquilibriumResult:
    system_name = str(system).strip().lower()
    if system_name not in _SYSTEM_COMPONENTS:
        raise ValueError(f"SYSTEM_UNSUPPORTED: {system}")

    cache = _DEFAULT_CACHE if cache is None else cache
    cache_key = None
    if cache is not None:
        cache_payload = {
            "kind": "equilibrium",
            "db_path": str(db_ref.path),
            "system": system_name,
            "composition": {str(k): float(v) for k, v in dict(composition).items()},
            "temperature_c": float(temperature_c),
            "pressure_pa": float(pressure_pa),
            "equilibrium_model": str(equilibrium_model),
        }
        cache_key = cache.make_key(cache_payload)
        cached = cache.get(cache_key)
        if isinstance(cached, dict) and cached.get("stable_phases"):
            return CalphadEquilibriumResult(**cached)

    try:
        from pycalphad import equilibrium
    except Exception as exc:  # pragma: no cover - env dependent
        raise ValueError(f"CALPHAD_UNAVAILABLE: {exc}") from exc

    db = _get_database(db_ref.path)
    comps = list(_SYSTEM_COMPONENTS[system_name])
    cond = _build_conditions(
        system=system_name,
        composition=composition,
        temperature_c=temperature_c,
        pressure_pa=pressure_pa,
    )
    phases = list(db.phases.keys())

    started = time.perf_counter()
    try:
        eq = equilibrium(db, comps, phases, cond)
    except Exception as exc:
        raise ValueError(f"SOLVER_FAIL: {exc}") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    stable = _extract_stable_phases(eq)
    solver_status = "ok"
    if not stable:
        dominant = _dominant_component(system_name, composition)
        stable = {f"{dominant}_SOLID": 1.0}
        solver_status = "approx_fallback"
    liquid = 0.0
    for name, frac in stable.items():
        if "LIQUID" in name.upper():
            liquid += float(frac)
    liquid = float(max(0.0, min(1.0, liquid)))
    solid = float(max(0.0, min(1.0, 1.0 - liquid)))
    mu = _extract_mu(eq)

    result = CalphadEquilibriumResult(
        system=system_name,
        stable_phases=stable,
        liquid_fraction=liquid,
        solid_fraction=solid,
        chemical_potentials=mu,
        solver_status=solver_status,
        compute_time_ms=float(round(elapsed_ms, 3)),
        temperature_c=float(temperature_c),
        pressure_pa=float(pressure_pa),
    )
    if cache is not None and cache_key is not None:
        cache.set(cache_key, result.to_dict())
    return result


def run_equilibrium_grid(
    *,
    db_ref: CalphadDBReference,
    system: str,
    composition: dict[str, float],
    temperatures_c: list[float],
    pressure_pa: float,
    equilibrium_model: str = "global_min",
    cache: CalphadCache | None = None,
) -> list[CalphadEquilibriumResult]:
    out: list[CalphadEquilibriumResult] = []
    for temp in temperatures_c:
        cache_key = None
        if cache is not None:
            key_payload = {
                "db_hash": db_ref.sha256,
                "system": system,
                "composition": {
                    str(k): float(v)
                    for k, v in sorted(composition.items(), key=lambda x: x[0])
                },
                "temperature_c": float(temp),
                "pressure_pa": float(pressure_pa),
                "equilibrium_model": equilibrium_model,
            }
            cache_key = cache.make_key(key_payload)
            found = cache.get(cache_key)
            if isinstance(found, dict):
                out.append(CalphadEquilibriumResult(**found))
                continue

        result = run_equilibrium(
            db_ref=db_ref,
            system=system,
            composition=composition,
            temperature_c=float(temp),
            pressure_pa=float(pressure_pa),
            equilibrium_model=equilibrium_model,
        )
        if cache is not None and cache_key is not None:
            cache.set(cache_key, result.to_dict())
        out.append(result)
    return out
