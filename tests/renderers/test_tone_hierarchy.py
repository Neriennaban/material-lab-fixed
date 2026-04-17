"""Phase 2+: проверка тоновой иерархии для каждой редизайненной структуры.

На Phase 1 ни один stub не подключен в runtime, поэтому все проверки
skip'ают сами себя. Тест активируется в sub-plan'е каждой фазы по мере
подключения соответствующего renderer'а.
"""
from __future__ import annotations

import pytest

from core.metallography_v3.structure_card import list_cards, load_card
from core.metallography_v3.system_generators.fe_c_unified import (
    _STAGE_TO_RENDERER,
)


# Минимальный маппинг id карточки -> stage, который она калибрует.
# Расширяется по мере роста множества карточек.
_CARD_TO_STAGE: dict[str, str] = {
    "martensite_lath": "martensite_cubic",
    "martensite_plate": "martensite_tetragonal",
    "martensite_mixed": "martensite",
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
    "widmanstatten_ferrite": "widmanstatten_ferrite",
    "decarburized_layer": "decarburized_layer",
    "carburized_layer": "carburized_layer",
    "granular_pearlite": "granular_pearlite",
}


@pytest.mark.parametrize("card_id", list_cards())
def test_card_rgb_tones_within_sane_range(card_id):
    """Санитарная проверка: все тона карточки в допустимом диапазоне
    uint8. Не требует рендера — работает на Phase 1."""
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
    """Если карточка указывает на стадию — она должна быть в
    диспетчере. Для карточек без маппинга (пока не все заведены) — skip."""
    target_stage = _CARD_TO_STAGE.get(card_id)
    if target_stage is None:
        pytest.skip(f"no stage mapping for card {card_id} yet")
    assert target_stage in _STAGE_TO_RENDERER, (
        f"card {card_id} references stage {target_stage!r} which is not "
        f"registered in _STAGE_TO_RENDERER"
    )


@pytest.mark.parametrize("card_id", list_cards())
def test_rendered_mean_tones_match_card(card_id):
    """Phase 2+: отрендерить стадию и сравнить mean-тона фаз с
    rgb_tones из карточки в допуске 20%.

    На Phase 1 все модули-семейства бросают NotImplementedError →
    тест автоматически skip'ается.
    """
    target_stage = _CARD_TO_STAGE.get(card_id)
    if target_stage is None:
        pytest.skip(f"no stage mapping for card {card_id} yet")
    pytest.skip(
        "activated per-family in Phase 2-8 sub-plans "
        "(current renderers are Phase 1 stubs)"
    )
