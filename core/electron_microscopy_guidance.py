from __future__ import annotations

from typing import Any


def _phase_value(phase_fractions: dict[str, float], *names: str) -> float:
    total = 0.0
    for name in names:
        total += float(phase_fractions.get(name, 0.0) or 0.0)
    return float(total)


def _level(score: float, *, low: float, high: float) -> str:
    if score >= high:
        return "high"
    if score >= low:
        return "medium"
    return "low"


def build_electron_microscopy_guidance(
    *,
    system: str,
    stage: str,
    composition_wt: dict[str, float] | None,
    phase_fractions: dict[str, float] | None,
    prep_summary: dict[str, Any] | None,
    etch_summary: dict[str, Any] | None,
    preset_metadata: dict[str, Any] | None = None,
    precipitation_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    system_l = str(system or "").strip().lower()
    stage_l = str(stage or "").strip().lower()
    phases = {str(k).upper(): float(v) for k, v in dict(phase_fractions or {}).items()}
    prep = dict(prep_summary or {})
    preset = dict(preset_metadata or {})
    precipitation = dict(precipitation_state or {})

    rough_like = bool(
        float(prep.get("relief_mean", 0.0) or 0.0) >= 0.56
        or float(prep.get("false_porosity_from_chipping_risk", 0.0) or 0.0) >= 0.40
        or float(prep.get("outer_fragmented_layer_risk", 0.0) or 0.0) >= 0.55
    )
    porous_like = bool(
        float(prep.get("pullout_mean", 0.0) or 0.0) >= 0.18
        or float(prep.get("false_porosity_from_chipping_risk", 0.0) or 0.0) >= 0.35
    )
    fracture_like = "fracture" in stage_l or "fractograph" in str(preset.get("target_microstructure", "")).lower()

    intermetallic_fraction = _phase_value(
        phases,
        "SI",
        "PRECIPITATE",
        "THETA",
        "S_PHASE",
        "QPHASE",
        "BETA",
        "FESI_INTERMETALLIC",
    )
    composition_contrast_case = bool(system_l in {"al-si", "al-cu-mg", "cu-zn"} or intermetallic_fraction >= 0.12)
    delta_austenite_case = bool(_phase_value(phases, "DELTA_FERRITE") >= 0.05 and _phase_value(phases, "AUSTENITE") >= 0.05)

    aged_precipitation_case = bool(
        system_l == "al-cu-mg"
        and (
            stage_l in {"natural_aged", "artificial_aged", "aged", "overaged"}
            or any(str(key).lower().startswith("precip") for key in precipitation.keys())
            or "aged" in str(preset.get("sample_id", "")).lower()
            or "aged" in str(preset.get("target_microstructure", "")).lower()
        )
    )

    why: list[str] = []
    recommended_modalities = ["optical"]
    primary_recommendation = "optical"
    sem_recommended = False
    sem_mode = "none"
    sem_avoid_etching = False
    sem_coating = False
    sem_charging_score = 0.0
    sem_outgassing_score = 0.0
    sem_deformation_free = False
    sem_conductive_mounting = False

    if rough_like or porous_like or fracture_like:
        sem_recommended = True
        sem_mode = "secondary_electron"
        primary_recommendation = "sem_se"
        recommended_modalities.insert(0, "sem_se")
        why.append("SEM-SE лучше подходит для morphology/topography, rough and pore-sensitive surfaces")
        sem_conductive_mounting = porous_like
    elif composition_contrast_case:
        sem_recommended = True
        sem_mode = "backscattered_electron"
        primary_recommendation = "sem_bse"
        recommended_modalities.insert(0, "sem_bse")
        sem_avoid_etching = True
        why.append("SEM-BSE полезен для material/atomic-number contrast on polished unetched sections")

    if delta_austenite_case:
        sem_recommended = True
        sem_deformation_free = True
        why.append("delta-ferrite/austenite separation benefits from deformation-free flat preparation")

    if sem_mode == "secondary_electron":
        sem_charging_score += 0.20
        sem_outgassing_score += 0.15
    if porous_like:
        sem_charging_score += 0.20
        sem_outgassing_score += 0.15
    if float(prep.get("contamination_level", 0.0) or 0.0) >= 0.02:
        sem_outgassing_score += 0.20
    if float(prep.get("contamination_level", 0.0) or 0.0) >= 0.03:
        sem_coating = True
    if system_l not in {"fe-c", "fe-si", "al-si", "cu-zn", "al-cu-mg"}:
        sem_coating = True
        sem_charging_score += 0.25

    tem_recommended = aged_precipitation_case
    tem_target_thickness = "few_hundred_nm" if tem_recommended else "not_applicable"
    tem_parallel_faces = bool(tem_recommended)
    tem_window_or_jet = bool(tem_recommended and system_l in {"fe-c", "fe-si", "al-si", "al-cu-mg", "cu-zn"})
    tem_ion_beam = bool(tem_recommended)
    if tem_recommended:
        if "tem" not in recommended_modalities:
            recommended_modalities.append("tem")
        why.append("TEM is a candidate for very fine precipitate / thin-foil questions")

    if primary_recommendation == "optical" and not why:
        why.append("Optical microscopy remains the first-line method for this case")

    seen: set[str] = set()
    ordered_modalities: list[str] = []
    for item in recommended_modalities:
        if item not in seen:
            ordered_modalities.append(item)
            seen.add(item)

    return {
        "advisory_only": True,
        "source_pages": "ASM_Vol9_141_160",
        "primary_recommendation": primary_recommendation,
        "recommended_modalities": ordered_modalities,
        "sem_guidance": {
            "recommended": bool(sem_recommended),
            "preferred_mode": sem_mode,
            "avoid_etching_for_material_contrast": bool(sem_avoid_etching),
            "coating_may_help": bool(sem_coating),
            "charging_risk": _level(sem_charging_score, low=0.25, high=0.55),
            "outgassing_risk": _level(sem_outgassing_score, low=0.20, high=0.45),
            "deformation_free_flat_surface_required": bool(sem_deformation_free),
            "conductive_mounting_or_adhesive_helpful": bool(sem_conductive_mounting or sem_coating),
        },
        "tem_guidance": {
            "recommended": bool(tem_recommended),
            "target_thickness_guidance": tem_target_thickness,
            "parallel_face_thinning_required": bool(tem_parallel_faces),
            "window_or_jet_electropolish_candidate": bool(tem_window_or_jet),
            "ion_beam_final_thinning_candidate": bool(tem_ion_beam),
        },
        "why": why[:3],
    }
