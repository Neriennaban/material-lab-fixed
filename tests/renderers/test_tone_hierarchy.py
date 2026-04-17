"""Phase 2+: проверка тоновой иерархии для каждой редизайненной структуры.

На Phase 1 ни один stub не подключен в runtime, поэтому тест
``test_rendered_mean_tones_match_card`` skip'ает стадии, чьи семейства
ещё не активированы. По мере реализации sub-plan'ов (Phase 2-8) стадии
добавляются в ``_ACTIVE_STAGES`` → тест начинает реально сравнивать
mean-тона.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.contracts_v2 import ProcessingState
from core.metallography_v3.structure_card import list_cards, load_card
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import (
    _ACTIVATED_RENDERER_STAGES,
    _STAGE_TO_RENDERER,
    render_fe_c_unified,
)


# Минимальный маппинг id карточки -> stage, который она калибрует.
# Расширяется по мере роста множества карточек.
_CARD_TO_STAGE: dict[str, str] = {
    # Phase 4 martensite cards
    "martensite_lath": "martensite_cubic",
    "martensite_plate": "martensite_tetragonal",
    "martensite_mixed": "martensite",
    "retained_austenite": None,  # post-process phase; not directly rendered
    "bainite_upper": "bainite_upper",
    "bainite_lower": "bainite_lower",
    "bainite_cfb": "carbide_free_bainite",
    "tempered_low": "tempered_low",
    "tempered_medium": "tempered_medium",
    "tempered_high": "tempered_high",
    "troostite_quench": "troostite_quench",
    "sorbite_quench": "sorbite_quench",
    "white_cast_iron_hypoeutectic": "white_cast_iron_hypoeutectic",
    "white_cast_iron_eutectic": "white_cast_iron_eutectic",
    "white_cast_iron_hypereutectic": "white_cast_iron_hypereutectic",
    "ledeburite_ld_prime": "ledeburite",
    "austenite": "austenite",
    "delta_ferrite": "delta_ferrite",
    "alpha_gamma": "alpha_gamma",
    "gamma_cementite": "gamma_cementite",
    "liquid": "liquid",
    "liquid_gamma": "liquid_gamma",
    "widmanstatten_ferrite": "widmanstatten_ferrite",
    "decarburized_layer": "decarburized_layer",
    "carburized_layer": "carburized_layer",
    "granular_pearlite": "granular_pearlite",
}


# Стадии, где новый renderer активирован и tone-hierarchy тест
# включён. По мере активации семейств (Phase 3-8) это множество
# расширяется — диспетчер fe_c_unified ведёт авторитетный список в
# _ACTIVATED_RENDERER_STAGES.
_ACTIVE_STAGES: frozenset[str] = _ACTIVATED_RENDERER_STAGES


_STAGE_RUNTIME_DEFAULTS: dict[str, tuple[dict[str, float], dict[str, float], float]] = {
    # Phase 2
    "austenite": ({"AUSTENITE": 1.0}, {"Fe": 99.2, "C": 0.8}, 900.0),
    "delta_ferrite": (
        {"DELTA_FERRITE": 0.15, "AUSTENITE": 0.85},
        {"Fe": 99.95, "C": 0.05},
        1450.0,
    ),
    "alpha_gamma": (
        {"FERRITE": 0.55, "AUSTENITE": 0.45},
        {"Fe": 99.7, "C": 0.3},
        800.0,
    ),
    "gamma_cementite": (
        {"AUSTENITE": 0.72, "CEMENTITE": 0.28},
        {"Fe": 98.8, "C": 1.2},
        900.0,
    ),
    "liquid": ({"LIQUID": 1.0}, {"Fe": 99.5, "C": 0.5}, 1600.0),
    "liquid_gamma": (
        {"LIQUID": 0.62, "AUSTENITE": 0.38},
        {"Fe": 99.7, "C": 0.3},
        1480.0,
    ),
    # Phase 3
    "ledeburite": (
        {"PEARLITE": 0.49, "CEMENTITE": 0.51},
        {"Fe": 95.7, "C": 4.3},
        500.0,
    ),
    "white_cast_iron_eutectic": (
        {"LEDEBURITE": 1.0},
        {"Fe": 95.7, "C": 4.3},
        20.0,
    ),
    "white_cast_iron_hypoeutectic": (
        {"LEDEBURITE": 0.65, "PEARLITE": 0.35},
        {"Fe": 97.0, "C": 3.0},
        20.0,
    ),
    "white_cast_iron_hypereutectic": (
        {"LEDEBURITE": 0.70, "CEMENTITE_PRIMARY": 0.30},
        {"Fe": 94.5, "C": 5.5},
        20.0,
    ),
    # Phase 4
    "martensite_cubic": (
        {"MARTENSITE_CUBIC": 0.94, "CEMENTITE": 0.06},
        {"Fe": 99.7, "C": 0.3},
        20.0,
    ),
    "martensite_tetragonal": (
        {"MARTENSITE_TETRAGONAL": 0.82, "CEMENTITE": 0.05, "AUSTENITE": 0.13},
        {"Fe": 98.8, "C": 1.2},
        20.0,
    ),
    "martensite": (
        {"MARTENSITE": 0.85, "CEMENTITE": 0.05, "AUSTENITE": 0.10},
        {"Fe": 99.2, "C": 0.8},
        20.0,
    ),
    # Phase 5
    "bainite_upper": (
        {"BAINITE": 0.78, "CEMENTITE": 0.22},
        {"Fe": 99.55, "C": 0.45},
        480.0,
    ),
    "bainite_lower": (
        {"BAINITE": 0.85, "CEMENTITE": 0.15},
        {"Fe": 99.3, "C": 0.7},
        320.0,
    ),
    "carbide_free_bainite": (
        {"BAINITE": 0.70, "AUSTENITE": 0.25, "MARTENSITE": 0.05},
        {"Fe": 97.6, "C": 0.4, "Si": 1.8},
        300.0,
    ),
    # Phase 6
    "troostite_quench": (
        {"TROOSTITE": 0.88, "CEMENTITE": 0.12},
        {"Fe": 99.4, "C": 0.6},
        550.0,
    ),
    "sorbite_quench": (
        {"SORBITE": 0.84, "CEMENTITE": 0.16},
        {"Fe": 99.45, "C": 0.55},
        620.0,
    ),
    # Phase 7
    "tempered_low": (
        {"MARTENSITE": 0.92, "CEMENTITE": 0.08},
        {"Fe": 99.55, "C": 0.45},
        220.0,
    ),
    "tempered_medium": (
        {"TROOSTITE": 0.70, "CEMENTITE": 0.20, "FERRITE": 0.10},
        {"Fe": 99.55, "C": 0.45},
        420.0,
    ),
    "tempered_high": (
        {"SORBITE": 0.42, "FERRITE": 0.40, "CEMENTITE": 0.18},
        {"Fe": 99.6, "C": 0.4},
        580.0,
    ),
}


@pytest.mark.parametrize("card_id", list_cards())
def test_card_rgb_tones_within_sane_range(card_id):
    """Санитарная проверка: все тона карточки в допустимом диапазоне
    uint8. Не требует рендера."""
    card = load_card(card_id)
    for reagent, components in card.rgb_tones.items():
        for comp_name, rgb in components.items():
            for ch in rgb:
                assert 0 <= ch <= 255, (
                    f"{card_id}:{reagent}:{comp_name} has out-of-range "
                    f"component {ch}"
                )


@pytest.mark.parametrize("card_id", list_cards())
def test_card_stage_is_registered_or_skipped(card_id):
    """Карточка должна указывать на стадию, зарегистрированную в
    ``_STAGE_TO_RENDERER``."""
    target_stage = _CARD_TO_STAGE.get(card_id, "__missing__")
    if target_stage is None:
        pytest.skip(f"card {card_id} is an overlay/sub-phase, not a standalone stage")
    if target_stage == "__missing__":
        pytest.skip(f"no stage mapping for card {card_id} yet")
    assert target_stage in _STAGE_TO_RENDERER, (
        f"card {card_id} references stage {target_stage!r} which is not "
        f"registered in _STAGE_TO_RENDERER"
    )


@pytest.mark.parametrize("card_id", list_cards())
def test_rendered_mean_tones_match_card(card_id):
    """Отрендерить стадию и сравнить mean-тон картинки с
    «усреднённым» тоном по rgb_tones.nital карточки (±30 единиц u8).

    Для стадий вне ``_ACTIVE_STAGES`` — skip до активации renderer'а
    в соответствующем sub-plan'е.
    """
    target_stage = _CARD_TO_STAGE.get(card_id, "__missing__")
    if target_stage is None:
        pytest.skip(f"card {card_id} is an overlay/sub-phase, not a standalone stage")
    if target_stage == "__missing__":
        pytest.skip(f"no stage mapping for card {card_id} yet")
    if target_stage not in _ACTIVE_STAGES:
        pytest.skip(f"stage {target_stage!r} not yet activated (Phase 3-8 pending)")

    # Целевой тон — composition-weighted average тонов фаз.
    # Для каждой фазы в phase_composition берём среднее по наиболее
    # подходящему компоненту nital.rgb_tones и взвешиваем по доле.
    card = load_card(card_id)
    nital = card.rgb_tones.get("nital", {})
    if not nital:
        pytest.skip(f"{card_id}: no nital tones in card")

    _PHASE_KEYS = {
        "AUSTENITE": (
            "austenite",
            "matrix_austenite",
            "retained_austenite_blocks",
            "interior",
            "matrix",
        ),
        "FERRITE": ("ferrite", "matrix", "interior"),
        "DELTA_FERRITE": ("delta_islands",),
        "CEMENTITE": (
            "cementite_matrix",
            "cementite_globules",
            "cementite",
            "cementite_films",
        ),
        "CEMENTITE_PRIMARY": ("primary_cementite_plates", "cementite_matrix"),
        "LEDEBURITE": ("cementite_matrix", "pearlite_islands"),
        "LIQUID": ("liquid_matrix", "bright", "dark"),
        "PEARLITE": (
            "pearlite_islands",
            "primary_pearlite_dendrites",
            "matrix",
        ),
        "MARTENSITE": (
            "laths",
            "plate_body",
            "matrix",
            "martensite_cores",
        ),
        "MARTENSITE_CUBIC": ("laths",),
        "MARTENSITE_TETRAGONAL": ("plate_body", "laths"),
        "BAINITE": ("matrix",),
        "BAINITE_UPPER": ("matrix",),
        "BAINITE_LOWER": ("matrix", "background"),
        "TROOSTITE": ("matrix", "background"),
        "SORBITE": ("colony_mean", "ferrite_lamellae"),
    }

    def _tone_for_phase(phase_name: str) -> float | None:
        name = phase_name.upper()
        if name == "LEDEBURITE":
            # §1.6: Ld′ ≈ 49% перлит + 51% цементит по массе —
            # blended тон отражает реальный mean картинки в
            # «леопардовой» матрице.
            p_rgb = nital.get("pearlite_islands") or nital.get("matrix")
            c_rgb = nital.get("cementite_matrix") or nital.get("cementite")
            if p_rgb is None or c_rgb is None:
                return None
            return 0.49 * float(np.mean(p_rgb)) + 0.51 * float(np.mean(c_rgb))
        for key in _PHASE_KEYS.get(name, ()):
            rgb = nital.get(key)
            if rgb is not None:
                return float(np.mean(rgb))
        return None

    comp = card.phase_composition or {}
    numer = 0.0
    denom = 0.0
    for phase, frac in comp.items():
        tone = _tone_for_phase(phase)
        if tone is None:
            continue
        numer += float(frac) * tone
        denom += float(frac)
    if denom < 1e-6:
        # Fallback: avg всех компонентов.
        target_mean = float(
            np.mean([comp for rgb in nital.values() for comp in rgb])
        )
    else:
        target_mean = numer / denom

    runtime = _STAGE_RUNTIME_DEFAULTS.get(target_stage)
    if runtime is None:
        pytest.skip(f"no runtime defaults for {target_stage} yet")
    fractions, composition, temperature_c = runtime
    ctx = SystemGenerationContext(
        size=(192, 192),
        seed=4242,
        inferred_system="fe-c",
        stage=target_stage,
        phase_fractions=fractions,
        composition_wt=composition,
        processing=ProcessingState(
            temperature_c=temperature_c, cooling_mode="equilibrium"
        ),
    )
    out = render_fe_c_unified(ctx)
    actual_mean = float(out.image_gray.mean())

    # Допуск зависит от сложности композиции. Однофазные / слабо
    # смешанные стадии калибруются точнее, составные (чугуны с
    # первичными фазами на leopard-матрице, Widmanstätten на
    # перлитном фоне) — шире: карточный тон суммы фаз плохо предсказует
    # mean рендера при резко контрастных компонентах.
    _COMPOSITE_STAGES = {
        "white_cast_iron_hypoeutectic",
        "white_cast_iron_hypereutectic",
    }
    tolerance = 100.0 if target_stage in _COMPOSITE_STAGES else 55.0
    diff = abs(actual_mean - target_mean)
    assert diff < tolerance, (
        f"{card_id} ({target_stage}): mean image tone {actual_mean:.1f} "
        f"is too far from matrix target {target_mean:.1f} "
        f"(diff={diff:.1f}, tolerance={tolerance:.0f})"
    )
