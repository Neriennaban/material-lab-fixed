from __future__ import annotations

import unittest

from core.contracts_v2 import ProcessingOperation, ProcessingState
from core.cooling_modes import (
    canonicalize_cooling_mode,
    cooling_mode_label_ru,
    cooling_mode_options_ru,
    resolve_auto_cooling_mode,
)


class CoolingModesV2Tests(unittest.TestCase):
    def test_alias_canonicalization(self) -> None:
        self.assertEqual(canonicalize_cooling_mode("quench"), "quenched")
        self.assertEqual(canonicalize_cooling_mode("slow-cool"), "slow_cool")
        self.assertEqual(canonicalize_cooling_mode("AUTO"), "auto")

    def test_ru_labels_and_options(self) -> None:
        self.assertEqual(cooling_mode_label_ru("tempered"), "Отпуск")
        options = cooling_mode_options_ru(include_auto=True)
        self.assertTrue(any(code == "auto" for code, _ in options))
        self.assertTrue(any(label == "Равновесное охлаждение" for _, label in options))

    def test_auto_resolution_by_system(self) -> None:
        al_proc = ProcessingState(temperature_c=180.0, cooling_mode="auto", aging_hours=8.0)
        self.assertEqual(resolve_auto_cooling_mode("al-cu-mg", al_proc), "aged")

        fe_proc = ProcessingState(temperature_c=900.0, cooling_mode="auto", aging_hours=0.0)
        self.assertEqual(resolve_auto_cooling_mode("fe-c", fe_proc), "normalized")

    def test_contracts_canonicalize_operation_mode(self) -> None:
        op = ProcessingOperation(method="quench_water", cooling_mode="quench")
        self.assertEqual(op.cooling_mode, "quenched")


if __name__ == "__main__":
    unittest.main()
