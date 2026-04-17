"""Phase 1 guard: все JSON-карточки в datasets/structure_cards/
загружаются валидно + соответствуют файловой схеме.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from core.metallography_v3.structure_card import (
    STRUCTURE_CARDS_DIR,
    StructureCardError,
    list_cards,
    load_card,
)


class StructureCardTests(unittest.TestCase):
    def test_cards_dir_exists(self) -> None:
        self.assertTrue(
            STRUCTURE_CARDS_DIR.is_dir(),
            f"structure_cards dir missing: {STRUCTURE_CARDS_DIR}",
        )

    def test_schema_file_present(self) -> None:
        schema = STRUCTURE_CARDS_DIR / "_schema.json"
        self.assertTrue(schema.is_file(), "_schema.json missing")

    def test_all_cards_load(self) -> None:
        cards = list_cards()
        self.assertGreaterEqual(
            len(cards),
            2,
            f"expected at least 2 smoke cards, got {cards}",
        )
        for cid in cards:
            with self.subTest(card=cid):
                card = load_card(cid)
                self.assertEqual(card.id, cid)
                self.assertTrue(card.name_ru)
                self.assertTrue(card.reference_section.startswith("§"))
                self.assertIn("nital", card.rgb_tones)
                self.assertGreater(len(card.rgb_tones["nital"]), 0)
                # Все RGB нормализованы как tuple[int,int,int] в [0,255].
                for reagent, components in card.rgb_tones.items():
                    for comp_name, rgb in components.items():
                        self.assertEqual(
                            len(rgb),
                            3,
                            f"{cid}:{reagent}:{comp_name} not RGB triple",
                        )
                        for ch in rgb:
                            self.assertGreaterEqual(ch, 0)
                            self.assertLessEqual(ch, 255)

    def test_smoke_cards_present(self) -> None:
        cards = set(list_cards())
        self.assertIn("martensite_lath", cards)
        self.assertIn("bainite_upper", cards)

    def test_malformed_card_raises(self) -> None:
        bad_path = STRUCTURE_CARDS_DIR / "_tmp_malformed.json"
        bad_path.write_text('{"id": "x"}', encoding="utf-8")
        try:
            with self.assertRaises(StructureCardError):
                load_card("_tmp_malformed")
        finally:
            bad_path.unlink(missing_ok=True)
            load_card.cache_clear()


if __name__ == "__main__":
    unittest.main()
