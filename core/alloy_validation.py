from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .calphad.db_manager import CALPHAD_SUPPORTED_SYSTEMS
from .cooling_modes import canonicalize_cooling_mode, resolve_auto_cooling_mode
from .contracts_v2 import AlloyComposition, ProcessingState, ValidationReport


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_RULEBOOK_ROOT = Path(__file__).resolve().parent / "rulebook"
_ELEMENTS_RULES = _load_json(_RULEBOOK_ROOT / "elements_limits.json")
_SYSTEMS_RULES = _load_json(_RULEBOOK_ROOT / "systems_rules.json")
_PROCESSING_RULES = _load_json(_RULEBOOK_ROOT / "processing_rules.json")


PERIODIC_SYMBOLS = set(_ELEMENTS_RULES.get("elements", {}).keys())

SYMBOL_ALIASES = {
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
    "iron": "Fe",
    "fe": "Fe",
}


def _title_symbol(symbol: str) -> str:
    s = symbol.strip()
    if not s:
        return s
    if len(s) == 1:
        return s.upper()
    return s[0].upper() + s[1:].lower()


def normalize_composition_keys(composition: dict[str, Any]) -> tuple[AlloyComposition, list[str]]:
    normalized: AlloyComposition = {}
    errors: list[str] = []
    for raw_key, raw_value in composition.items():
        key = str(raw_key).strip().replace("%", "")
        alias_key = SYMBOL_ALIASES.get(key.lower(), key)
        symbol = _title_symbol(alias_key)
        if symbol not in PERIODIC_SYMBOLS:
            errors.append(f"Unknown element symbol: {raw_key}")
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors.append(f"Invalid concentration for {symbol}: {raw_value}")
            continue
        normalized[symbol] = normalized.get(symbol, 0.0) + value
    return normalized, errors


def infer_system(composition: AlloyComposition) -> tuple[str, float]:
    comp = {k: float(v) for k, v in composition.items() if float(v) > 0.0}
    if not comp:
        return "custom-multicomponent", 0.0

    inference_map: dict[str, list[str]] = _SYSTEMS_RULES.get("system_inference", {})
    systems_cfg: dict[str, Any] = _SYSTEMS_RULES.get("systems", {})
    scores: dict[str, float] = {}
    for system, markers in inference_map.items():
        cfg = systems_cfg.get(system, {})
        required_any = cfg.get("required_any", []) if isinstance(cfg, dict) else []
        if required_any:
            has_required = False
            for group in required_any:
                if isinstance(group, list) and all(comp.get(str(symbol), 0.0) > 0.0 for symbol in group):
                    has_required = True
                    break
            if not has_required:
                continue
        marker_sum = sum(comp.get(m, 0.0) for m in markers)
        if marker_sum <= 0.0:
            continue
        coverage = marker_sum / max(sum(comp.values()), 1e-9)
        simplicity = min(1.0, len(markers) / max(len(comp), 1))
        scores[system] = 0.75 * coverage + 0.25 * simplicity

    if not scores:
        total = max(1e-9, sum(comp.values()))
        max_share = max(comp.values()) / total
        complexity_penalty = min(0.35, max(0.0, (len(comp) - 1) * 0.06))
        confidence = float(max(0.2, min(0.9, 0.25 + 0.7 * max_share - complexity_penalty)))
        return "custom-multicomponent", confidence

    best = max(scores, key=scores.get)
    confidence = float(max(0.2, min(1.0, scores[best])))
    return best, confidence


def infer_calphad_system(inferred_system: str, composition: AlloyComposition) -> str:
    system = str(inferred_system or "").strip().lower()
    if system in set(CALPHAD_SUPPORTED_SYSTEMS):
        return system
    comp = {str(k): max(0.0, float(v)) for k, v in composition.items()}
    if not comp:
        return "custom-multicomponent"
    total = max(1e-9, float(sum(comp.values())))
    if total <= 0.0:
        return "custom-multicomponent"

    si = float(comp.get("Si", 0.0))
    fe = float(comp.get("Fe", 0.0))
    al = float(comp.get("Al", 0.0))
    cu = float(comp.get("Cu", 0.0))
    zn = float(comp.get("Zn", 0.0))
    mg = float(comp.get("Mg", 0.0))
    c = float(comp.get("C", 0.0))

    scores = {
        "fe-c": (fe + 2.4 * c) if (fe > 0.0 or c > 0.0) else 0.0,
        "fe-si": (fe + 1.5 * si) if (fe > 0.0 or si > 0.0) else 0.0,
        "al-si": (al + 1.6 * si) if (al > 0.0 or si > 0.0) else 0.0,
        "cu-zn": (cu + 1.5 * zn) if (cu > 0.0 or zn > 0.0) else 0.0,
        "al-cu-mg": (al + 1.2 * cu + 1.1 * mg) if (al > 0.0 and (cu > 0.0 or mg > 0.0)) else 0.0,
    }
    best = max(scores, key=scores.get)
    if float(scores[best]) / total < 0.35:
        return "custom-multicomponent"
    return best


def _validate_sum(
    composition: AlloyComposition,
    auto_normalize: bool,
    errors: list[str],
    warnings: list[str],
) -> tuple[AlloyComposition, float, float]:
    raw_sum = float(sum(composition.values()))
    tolerance = float(_ELEMENTS_RULES.get("sum_tolerance_wt", 1.0))
    if raw_sum <= 0.0:
        errors.append("Composition sum must be > 0")
        return composition, raw_sum, raw_sum

    if abs(raw_sum - 100.0) <= tolerance:
        return composition, raw_sum, raw_sum

    if not auto_normalize:
        errors.append(f"Composition sum must be 100 +/- {tolerance} wt.% (got {raw_sum:.4f})")
        return composition, raw_sum, raw_sum

    normalized = {k: float(v) * (100.0 / raw_sum) for k, v in composition.items()}
    warnings.append(f"Composition was auto-normalized from sum={raw_sum:.4f} wt.%")
    return normalized, raw_sum, float(sum(normalized.values()))


def _validate_element_ranges(
    composition: AlloyComposition,
    errors: list[str],
    warnings: list[str],
) -> None:
    default_range = _ELEMENTS_RULES.get("default_range_wt", [0.0, 100.0])
    per_element = _ELEMENTS_RULES.get("elements", {})
    trace = float(_ELEMENTS_RULES.get("trace_threshold_wt", 1e-4))

    for symbol, value in composition.items():
        limits = per_element.get(symbol, default_range)
        low = float(limits[0])
        high = float(limits[1])
        if value < low or value > high:
            errors.append(f"{symbol} concentration {value:.6f} out of [{low}, {high}]")
        if 0.0 < value < trace:
            warnings.append(f"{symbol} concentration {value:.6f} below trace threshold {trace}")


def _validate_system_ranges(
    inferred_system: str,
    composition: AlloyComposition,
    errors: list[str],
    warnings: list[str],
) -> None:
    systems = _SYSTEMS_RULES.get("systems", {})
    config = systems.get(inferred_system)
    if not isinstance(config, dict):
        warnings.append("No system-specific rules found; custom validation only.")
        return

    required_any = config.get("required_any", [])
    if required_any:
        satisfied = False
        for group in required_any:
            if all(float(composition.get(symbol, 0.0)) > 0.0 for symbol in group):
                satisfied = True
                break
        if not satisfied:
            errors.append(f"Composition does not satisfy required markers for {inferred_system}: {required_any}")

    ranges_wt: dict[str, list[float]] = config.get("ranges_wt", {})
    for symbol, bounds in ranges_wt.items():
        value = float(composition.get(symbol, 0.0))
        low = float(bounds[0])
        high = float(bounds[1])
        if value < low or value > high:
            errors.append(f"{inferred_system}: {symbol}={value:.4f} outside [{low}, {high}]")


def _validate_processing(
    inferred_system: str,
    processing: ProcessingState,
    errors: list[str],
    warnings: list[str],
) -> None:
    canonical_mode = canonicalize_cooling_mode(processing.cooling_mode)
    if canonical_mode == "auto":
        resolved = resolve_auto_cooling_mode(inferred_system=inferred_system, processing=processing)
        warnings.append(f"cooling_mode 'auto' resolved to '{resolved}' for validation")
        canonical_mode = resolved

    proc = ProcessingState(
        temperature_c=processing.temperature_c,
        cooling_mode=canonical_mode,
        deformation_pct=processing.deformation_pct,
        aging_hours=processing.aging_hours,
        aging_temperature_c=processing.aging_temperature_c,
        pressure_mpa=processing.pressure_mpa,
        note=processing.note,
    )

    global_rules: dict[str, list[float]] = _PROCESSING_RULES.get("global", {})
    for field_name, bounds in global_rules.items():
        value = getattr(proc, field_name)
        if value is None:
            continue
        low = float(bounds[0])
        high = float(bounds[1])
        if float(value) < low or float(value) > high:
            errors.append(f"Processing {field_name}={value} out of [{low}, {high}]")

    system_rules = _PROCESSING_RULES.get("systems", {}).get(inferred_system)
    if not isinstance(system_rules, dict):
        warnings.append("No system-specific processing rules; global-only validation applied.")
        return

    temp_bounds = system_rules.get("temperature_c")
    if temp_bounds:
        low = float(temp_bounds[0])
        high = float(temp_bounds[1])
        if proc.temperature_c < low or proc.temperature_c > high:
            errors.append(
                f"{inferred_system}: temperature {proc.temperature_c} outside [{low}, {high}]"
            )

    allowed_modes = system_rules.get("cooling_modes", [])
    if allowed_modes and proc.cooling_mode not in allowed_modes:
        errors.append(f"{inferred_system}: cooling_mode '{proc.cooling_mode}' not in {allowed_modes}")

    checks = system_rules.get("checks", [])
    for check in checks:
        if not isinstance(check, dict):
            continue
        check_if = check.get("if", {})
        active = True
        for condition_key, condition_value in check_if.items():
            if condition_key == "cooling_mode":
                active = active and proc.cooling_mode == condition_value
            elif condition_key == "cooling_mode_in":
                active = active and proc.cooling_mode in list(condition_value)
        if not active:
            continue

        required = check.get("require", {})
        for field_name, bounds in required.items():
            val = getattr(proc, field_name)
            low = float(bounds[0])
            high = float(bounds[1])
            if val < low or val > high:
                errors.append(
                    f"{inferred_system}: check '{check.get('name', 'unnamed')}' failed for "
                    f"{field_name}={val}; expected [{low}, {high}]"
                )


def validate_alloy(
    composition: dict[str, Any],
    processing: ProcessingState | dict[str, Any] | None = None,
    auto_normalize: bool = True,
    strict_custom_limits: bool = True,
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    normalized, parse_errors = normalize_composition_keys(composition)
    errors.extend(parse_errors)

    normalized, raw_sum, norm_sum = _validate_sum(
        composition=normalized,
        auto_normalize=auto_normalize,
        errors=errors,
        warnings=warnings,
    )
    _validate_element_ranges(normalized, errors, warnings)

    inferred_system, confidence = infer_system(normalized)
    _validate_system_ranges(inferred_system, normalized, errors, warnings)

    calphad_system = infer_calphad_system(inferred_system, normalized)
    if calphad_system in set(CALPHAD_SUPPORTED_SYSTEMS):
        if calphad_system != inferred_system:
            warnings.append(
                f"CALPHAD coverage mapped inferred system '{inferred_system}' -> '{calphad_system}'"
            )
    else:
        warnings.append("CALPHAD coverage: no supported base system match; strict CALPHAD mode may block generation.")

    proc = processing if isinstance(processing, ProcessingState) else ProcessingState.from_dict(processing)
    _validate_processing(inferred_system, proc, errors, warnings)

    if inferred_system == "custom-multicomponent" and strict_custom_limits and confidence < 0.2:
        errors.append(
            "Custom multicomponent alloy has low system confidence; provide clearer major elements "
            "or switch to known alloy family."
        )

    return ValidationReport(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        normalized_composition=normalized,
        inferred_system=inferred_system,
        confidence=confidence,
        raw_sum_wt=raw_sum,
        normalized_sum_wt=norm_sum,
    )


def available_rulebook_paths() -> dict[str, str]:
    return {
        "elements_limits": str((_RULEBOOK_ROOT / "elements_limits.json").resolve()),
        "systems_rules": str((_RULEBOOK_ROOT / "systems_rules.json").resolve()),
        "processing_rules": str((_RULEBOOK_ROOT / "processing_rules.json").resolve()),
        "diagram_lines": str((_RULEBOOK_ROOT / "diagram_lines.json").resolve()),
    }


def format_validation_report(report: ValidationReport) -> str:
    lines = [
        f"Valid: {'yes' if report.is_valid else 'no'}",
        f"Inferred system: {report.inferred_system}",
        f"Confidence: {report.confidence:.2f}",
        f"Sum raw/normalized: {report.raw_sum_wt:.4f} / {report.normalized_sum_wt:.4f} wt.%",
    ]
    if report.errors:
        lines.append("Errors:")
        lines.extend([f"- {msg}" for msg in report.errors])
    if report.warnings:
        lines.append("Warnings:")
        lines.extend([f"- {msg}" for msg in report.warnings])
    if report.normalized_composition:
        lines.append("Normalized composition (wt.%):")
        for element, value in sorted(report.normalized_composition.items()):
            lines.append(f"- {element}: {value:.6g}")
    return "\n".join(lines)
