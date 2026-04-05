from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None


def _phase_separability_score(
    image_gray: np.ndarray, phase_masks: dict[str, np.ndarray] | None
) -> float:
    if not isinstance(phase_masks, dict):
        return 0.0

    means: list[float] = []
    stds: list[float] = []
    for key, mask in phase_masks.items():
        if str(key) in {"L", "solid"}:
            continue
        if not isinstance(mask, np.ndarray):
            continue
        pix = image_gray[mask > 0]
        if pix.size < 32:
            continue
        means.append(float(np.mean(pix)))
        stds.append(float(np.std(pix)) + 1e-6)

    if len(means) < 2:
        return 0.0

    ratios: list[float] = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            ratios.append(abs(means[i] - means[j]) / (stds[i] + stds[j]))

    if not ratios:
        return 0.0

    raw = float(np.mean(ratios))
    score = float(1.0 - np.exp(-0.9 * raw))
    return float(np.clip(score, 0.0, 1.0))


def _unexpected_dark_spot_metrics(
    image_gray: np.ndarray, *, threshold: float = 36.0
) -> dict[str, float]:
    arr = image_gray.astype(np.float32, copy=False)
    mask = arr < float(threshold)
    fraction = float(mask.mean()) if mask.size else 0.0
    if not np.any(mask):
        return {
            "fraction": fraction,
            "count": 0.0,
            "largest_component_px": 0.0,
        }
    if ndimage is None:
        return {
            "fraction": fraction,
            "count": float(1.0 if fraction > 0.0 else 0.0),
            "largest_component_px": float(mask.sum()),
        }
    labels, count = ndimage.label(mask.astype(np.uint8))
    if int(count) <= 0:
        return {
            "fraction": fraction,
            "count": 0.0,
            "largest_component_px": 0.0,
        }
    sizes = np.bincount(labels.ravel())
    largest = float(np.max(sizes[1:])) if sizes.size > 1 else 0.0
    return {
        "fraction": fraction,
        "count": float(int(count)),
        "largest_component_px": largest,
    }


def run_quality_checks(
    *,
    image_gray: np.ndarray,
    phase_masks: dict[str, np.ndarray] | None,
    feature_masks: dict[str, np.ndarray] | None,
    prep_maps: dict[str, np.ndarray] | None,
    phase_visibility_report: dict[str, Any] | None = None,
    generation_mode: str = "",
    profile_id: str = "",
    pure_iron_baseline_applied: bool = False,
) -> dict[str, Any]:
    arr = image_gray.astype(np.float32)
    p05 = float(np.quantile(arr, 0.05))
    p95 = float(np.quantile(arr, 0.95))
    dyn = float(max(0.0, p95 - p05))
    std = float(arr.std())

    warnings: list[str] = []
    if dyn < 42.0 and not bool(pure_iron_baseline_applied):
        warnings.append("Низкий динамический диапазон, фазы могут читаться слабо.")
    if std < 18.0 and not bool(pure_iron_baseline_applied):
        warnings.append("Низкий локальный контраст изображения.")

    phase_coverage: dict[str, float] = {}
    if isinstance(phase_masks, dict):
        for key, mask in phase_masks.items():
            if isinstance(mask, np.ndarray):
                phase_coverage[str(key)] = float((mask > 0).mean())
    if phase_coverage and abs(sum(phase_coverage.values()) - 1.0) > 0.25:
        warnings.append("Суммарная площадь фазовых масок выглядит несогласованной.")

    feature_density: dict[str, float] = {}
    if isinstance(feature_masks, dict):
        for key, mask in feature_masks.items():
            if isinstance(mask, np.ndarray):
                feature_density[str(key)] = float((mask > 0).mean())

    prep_stats: dict[str, float] = {}
    if isinstance(prep_maps, dict):
        for key in ("scratch", "contamination", "topography"):
            layer = prep_maps.get(key)
            if isinstance(layer, np.ndarray):
                prep_stats[f"{key}_mean"] = float(
                    layer.astype(np.float32).mean() / 255.0
                )

    separability_score = _phase_separability_score(
        image_gray=image_gray, phase_masks=phase_masks
    )
    if separability_score < 0.18 and not bool(pure_iron_baseline_applied):
        warnings.append(
            "Недостаточная читаемость фаз: низкая разделимость по тону/контрасту."
        )

    educational_textbook_steel = (
        str(generation_mode or "").strip().lower() == "edu_engineering"
        and str(profile_id or "").strip().lower() == "textbook_steel_bw"
    )
    dark_spot_metrics = {"fraction": 0.0, "count": 0.0, "largest_component_px": 0.0}
    if educational_textbook_steel or bool(pure_iron_baseline_applied):
        dark_spot_metrics = _unexpected_dark_spot_metrics(image_gray, threshold=36.0)
        largest = float(dark_spot_metrics["largest_component_px"])
        if largest >= max(28.0, float(image_gray.size) / 25000.0):
            warnings.append(
                "Обнаружены нефизичные черные пятна для brightfield-учебного рендера."
            )

    visibility_report = dict(phase_visibility_report or {})
    if visibility_report:
        within_tol = bool(visibility_report.get("within_tolerance", True))
        if not within_tol:
            warnings.append("Отклонение долей фаз превышает выбранный допуск.")

    passed = len(warnings) == 0
    return {
        "passed": bool(passed),
        "warnings": warnings,
        "dynamic_range_p05_p95": dyn,
        "std_contrast": std,
        "phase_coverage": phase_coverage,
        "feature_density": feature_density,
        "prep_stats": prep_stats,
        "phase_separability_score": separability_score,
        "unexpected_dark_spot_fraction": float(dark_spot_metrics["fraction"]),
        "unexpected_dark_spot_count": float(dark_spot_metrics["count"]),
        "unexpected_dark_spot_largest_component_px": float(
            dark_spot_metrics["largest_component_px"]
        ),
        "phase_visibility_report": visibility_report,
    }
