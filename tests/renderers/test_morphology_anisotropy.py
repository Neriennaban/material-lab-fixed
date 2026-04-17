"""Phase 2+: проверка морфологии (анизотропия, углы 60°/120°) для
иглolchatых структур — мартенсит (plate), Widmanstätten, нижний бейнит.

На Phase 1 тест skip'ается: в карточках ещё нет поля
``morphology.needle_angles_deg`` у большинства структур, и рендереры —
stub'ы. Активируется по семействам в sub-plan'ах Phase 4, 5, 8.
"""
from __future__ import annotations

import pytest

from core.metallography_v3.structure_card import list_cards, load_card


@pytest.mark.parametrize("card_id", list_cards())
def test_needle_angles_declared_where_applicable(card_id):
    """Санитарная: если карточка объявляет needle_angles_deg —
    значения в диапазоне [0, 180]."""
    card = load_card(card_id)
    angles = card.morphology.get("needle_angles_deg")
    if angles is None:
        pytest.skip(f"{card_id} doesn't declare needle_angles_deg")
    assert isinstance(angles, list) and angles
    for a in angles:
        assert isinstance(a, (int, float))
        assert 0.0 <= float(a) <= 180.0, (
            f"{card_id}: needle angle {a} out of [0, 180]"
        )


@pytest.mark.parametrize("card_id", list_cards())
def test_rendered_morphology_matches_card_angles(card_id):
    """Phase 4/5/8: Radon/Hough-анализ выводов рендера и сравнение с
    card.morphology.needle_angles_deg. На Phase 1 skip."""
    pytest.skip(
        "activated per-family in Phase 2-8 sub-plans "
        "(current renderers are Phase 1 stubs)"
    )
