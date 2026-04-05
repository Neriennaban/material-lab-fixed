from __future__ import annotations

import unittest

from core.alloy_validation import validate_alloy
from core.contracts_v2 import ProcessingState


class AlloyValidationV2Tests(unittest.TestCase):
    def test_valid_fe_c_temper_case(self) -> None:
        report = validate_alloy(
            composition={"Fe": 99.2, "C": 0.8},
            processing=ProcessingState(temperature_c=400.0, cooling_mode="tempered"),
            auto_normalize=True,
            strict_custom_limits=True,
        )
        self.assertTrue(report.is_valid)
        self.assertEqual(report.inferred_system, "fe-c")

    def test_unknown_element_symbol(self) -> None:
        report = validate_alloy(
            composition={"Fe": 98.0, "Xx": 2.0},
            processing=ProcessingState(),
            auto_normalize=True,
            strict_custom_limits=True,
        )
        self.assertFalse(report.is_valid)
        self.assertTrue(any("Unknown element symbol" in e for e in report.errors))

    def test_sum_without_auto_normalization_fails(self) -> None:
        report = validate_alloy(
            composition={"Fe": 95.0, "C": 0.3},
            processing=ProcessingState(),
            auto_normalize=False,
            strict_custom_limits=True,
        )
        self.assertFalse(report.is_valid)
        self.assertTrue(any("100 +/-" in e for e in report.errors))

    def test_sum_with_auto_normalization_passes(self) -> None:
        report = validate_alloy(
            composition={"Fe": 95.0, "C": 0.3},
            processing=ProcessingState(),
            auto_normalize=True,
            strict_custom_limits=True,
        )
        self.assertTrue(report.is_valid)
        self.assertAlmostEqual(report.normalized_sum_wt, 100.0, places=5)
        self.assertTrue(any("auto-normalized" in w for w in report.warnings))

    def test_processing_conflict_detected(self) -> None:
        report = validate_alloy(
            composition={"Fe": 99.2, "C": 0.8},
            processing=ProcessingState(temperature_c=80.0, cooling_mode="tempered"),
            auto_normalize=True,
            strict_custom_limits=True,
        )
        self.assertFalse(report.is_valid)
        self.assertTrue(any("tempering_window" in e for e in report.errors))

    def test_pure_fe_is_valid_in_strict_mode(self) -> None:
        report = validate_alloy(
            composition={"Fe": 99.95},
            processing=ProcessingState(temperature_c=25.0, cooling_mode="equilibrium"),
            auto_normalize=True,
            strict_custom_limits=True,
        )
        self.assertTrue(report.is_valid)

    def test_pure_cu_is_valid_in_strict_mode(self) -> None:
        report = validate_alloy(
            composition={"Cu": 99.9},
            processing=ProcessingState(temperature_c=25.0, cooling_mode="equilibrium"),
            auto_normalize=True,
            strict_custom_limits=True,
        )
        self.assertTrue(report.is_valid)

    def test_pure_si_is_valid_in_strict_mode(self) -> None:
        report = validate_alloy(
            composition={"Si": 99.999},
            processing=ProcessingState(temperature_c=25.0, cooling_mode="equilibrium"),
            auto_normalize=True,
            strict_custom_limits=True,
        )
        self.assertTrue(report.is_valid)


if __name__ == "__main__":
    unittest.main()
