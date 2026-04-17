"""Phase 2+: SSIM-сравнение рендера с принятым научным эталоном.

На Phase 1 все SSIM-проверки skip'аются (рендереры — stub'ы; эталонные
PNG ещё не собраны). Активируется per-family в sub-plan'ах Phase 2-8.

Этот модуль также проверяет, что пути в card.reference_images[] можно
резолвить (хотя бы для тех карточек, где эталонный PNG уже собран).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.metallography_v3.structure_card import list_cards, load_card

SSIM_THRESHOLD: float = 0.95


@pytest.mark.parametrize("card_id", list_cards())
def test_reference_image_paths_resolvable(card_id):
    """Если эталон есть физически — путь в карточке должен на него
    указывать. Если эталон ещё не собран (Phase 1) — skip."""
    card = load_card(card_id)
    if not card.reference_images:
        pytest.skip(f"{card_id} declares no reference images")

    repo_root = Path(__file__).resolve().parents[2]
    missing: list[str] = []
    for ref in card.reference_images:
        path = repo_root / ref.path
        if not path.is_file():
            missing.append(str(path))
    if missing:
        pytest.skip(
            f"{card_id} reference images not yet collected: "
            f"{len(missing)} missing (e.g. {missing[0]})"
        )
    # If all present — trivial pass; real SSIM comparison activates when
    # corresponding renderer is implemented (Phase 2+).


@pytest.mark.parametrize("card_id", list_cards())
def test_rendered_matches_reference_ssim(card_id):
    """Phase 2+: генерировать пресет для карточки, сравнить с эталоном
    через SSIM. На Phase 1 skip."""
    pytest.skip(
        "activated per-family in Phase 2-8 sub-plans "
        "(current renderers are Phase 1 stubs)"
    )
