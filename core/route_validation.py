from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cooling_modes import canonicalize_cooling_mode, resolve_auto_cooling_mode
from .contracts_v2 import ProcessRoute, ProcessingOperation, ProcessingState

_RULEBOOK_DIR = Path(__file__).resolve().parent / "rulebook"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_TECH_RULES = _load_json(_RULEBOOK_DIR / "technology_rules.json")


@dataclass(slots=True)
class RouteValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    normalized_operations: list[ProcessingOperation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "normalized_operations": [op.to_dict() for op in self.normalized_operations],
        }


def available_route_methods() -> list[str]:
    return sorted(_TECH_RULES.get("operations", {}).keys())


def route_templates() -> dict[str, list[dict[str, Any]]]:
    defaults = _TECH_RULES.get("route_defaults", {})
    out: dict[str, list[dict[str, Any]]] = {}
    for key, value in defaults.items():
        if isinstance(value, list):
            casted: list[dict[str, Any]] = []
            for item in value:
                if isinstance(item, dict):
                    casted.append(dict(item))
            out[str(key)] = casted
    return out


def _apply_defaults(operation: ProcessingOperation) -> ProcessingOperation:
    config = _TECH_RULES.get("operations", {}).get(operation.method)
    if not isinstance(config, dict):
        return operation
    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        return operation
    data = operation.to_dict()
    for key, value in defaults.items():
        if key == "method":
            continue
        current = data.get(key)
        if isinstance(current, (int, float)) and float(current) == 0.0:
            data[key] = value
        elif isinstance(current, str) and not current.strip():
            data[key] = value
    return ProcessingOperation.from_dict(data)


def _global_bounds_check(operation: ProcessingOperation, errors: list[str]) -> None:
    gl = _TECH_RULES.get("global_limits", {})
    bounds_map = {
        "temperature_c": operation.temperature_c,
        "duration_min": operation.duration_min,
        "deformation_pct": operation.deformation_pct,
        "aging_hours": operation.aging_hours,
        "pressure_mpa": 0.0 if operation.pressure_mpa is None else operation.pressure_mpa,
    }
    for key, value in bounds_map.items():
        bounds = gl.get(key)
        if not isinstance(bounds, list) or len(bounds) != 2:
            continue
        low, high = float(bounds[0]), float(bounds[1])
        if float(value) < low or float(value) > high:
            errors.append(f"{operation.method}: {key}={value} outside [{low}, {high}]")


def _system_sequence_checks(
    inferred_system: str,
    normalized: list[ProcessingOperation],
    errors: list[str],
) -> None:
    compatibility = _TECH_RULES.get("compatibility", {}).get(inferred_system, {})
    if not isinstance(compatibility, dict):
        return

    requirements = compatibility.get("forbid_without_previous", {})
    if isinstance(requirements, dict):
        for idx, op in enumerate(normalized):
            required = requirements.get(op.method)
            if not required:
                continue
            if not isinstance(required, list):
                continue
            previous_methods = {o.method for o in normalized[:idx]}
            if not any(m in previous_methods for m in required):
                errors.append(
                    f"{op.method} at step {idx + 1} requires one of previous operations: {required}"
                )


def _mutually_exclusive_checks(normalized: list[ProcessingOperation], errors: list[str]) -> None:
    pairs = _TECH_RULES.get("mutually_exclusive_cooling_modes", [])
    if not isinstance(pairs, list):
        return
    for idx, op in enumerate(normalized):
        mode = op.cooling_mode.strip().lower()
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            a = str(pair[0]).strip().lower()
            b = str(pair[1]).strip().lower()
            if mode in {a, b} and op.method in {"overage", "quench_water", "quench_oil"}:
                # Explicit same-step conflict check with method semantics.
                if mode == "quenched" and op.method == "overage":
                    errors.append(f"Step {idx + 1}: cooling_mode 'quenched' conflicts with method 'overage'")
                if mode == "overaged" and op.method in {"quench_water", "quench_oil"}:
                    errors.append(f"Step {idx + 1}: cooling_mode 'overaged' conflicts with quench method")


def _operation_processing_conflict_check(
    operation: ProcessingOperation,
    errors: list[str],
    warnings: list[str],
) -> None:
    if operation.method.startswith("quench") and operation.cooling_mode not in {"quenched", "quench", "water_quench", "oil_quench"}:
        warnings.append(
            f"{operation.method}: cooling_mode '{operation.cooling_mode}' is unusual for quench operation"
        )
    if operation.method.startswith("temper") and operation.cooling_mode != "tempered":
        warnings.append(
            f"{operation.method}: recommended cooling_mode is 'tempered'"
        )
    if operation.method.startswith("age") and operation.aging_hours <= 0.0:
        errors.append(f"{operation.method}: aging_hours must be > 0")


def _resolve_operation_cooling_mode(
    operation: ProcessingOperation,
    inferred_system: str,
    processing_context: ProcessingState | None,
    warnings: list[str],
) -> str:
    canonical = canonicalize_cooling_mode(operation.cooling_mode)
    if canonical != "auto":
        return canonical

    cfg = _TECH_RULES.get("operations", {}).get(operation.method, {})
    defaults = cfg.get("defaults", {}) if isinstance(cfg, dict) else {}
    default_mode = canonicalize_cooling_mode(defaults.get("cooling_mode", ""))
    if default_mode and default_mode != "auto":
        warnings.append(f"{operation.method}: cooling_mode 'auto' resolved to '{default_mode}'")
        return default_mode

    base = processing_context if processing_context is not None else ProcessingState()
    proxy = ProcessingState(
        temperature_c=float(operation.temperature_c),
        cooling_mode="auto",
        deformation_pct=float(operation.deformation_pct),
        aging_hours=float(operation.aging_hours),
        aging_temperature_c=float(operation.aging_temperature_c),
        pressure_mpa=base.pressure_mpa,
        note=base.note,
    )
    resolved = resolve_auto_cooling_mode(inferred_system=inferred_system, processing=proxy)
    warnings.append(f"{operation.method}: cooling_mode 'auto' resolved to '{resolved}'")
    return resolved


def validate_process_route(
    route: ProcessRoute | dict[str, Any] | None,
    inferred_system: str,
    processing_context: ProcessingState | None = None,
) -> RouteValidationResult:
    if route is None:
        return RouteValidationResult(is_valid=True, normalized_operations=[])

    route_obj = route if isinstance(route, ProcessRoute) else ProcessRoute.from_dict(route)
    errors: list[str] = []
    warnings: list[str] = []
    normalized: list[ProcessingOperation] = []
    available = set(available_route_methods())

    for idx, operation in enumerate(route_obj.operations):
        if not operation.method:
            errors.append(f"Step {idx + 1}: method is empty")
            continue
        if operation.method not in available:
            errors.append(f"Step {idx + 1}: unknown method '{operation.method}'")
            continue

        op = _apply_defaults(operation)
        resolved_mode = _resolve_operation_cooling_mode(
            operation=op,
            inferred_system=inferred_system,
            processing_context=processing_context,
            warnings=warnings,
        )
        if resolved_mode != op.cooling_mode:
            op = ProcessingOperation.from_dict({**op.to_dict(), "cooling_mode": resolved_mode})
        _global_bounds_check(op, errors)
        _operation_processing_conflict_check(op, errors, warnings)

        if processing_context is not None and op.pressure_mpa is None and processing_context.pressure_mpa is not None:
            op = ProcessingOperation.from_dict({**op.to_dict(), "pressure_mpa": processing_context.pressure_mpa})

        normalized.append(op)

    _system_sequence_checks(inferred_system=inferred_system, normalized=normalized, errors=errors)
    _mutually_exclusive_checks(normalized=normalized, errors=errors)

    if not normalized and route_obj.operations:
        errors.append("Route contains no valid operations")

    if inferred_system == "custom-multicomponent" and normalized:
        warnings.append("Custom system route interpretation is approximate")

    return RouteValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        normalized_operations=normalized,
    )
