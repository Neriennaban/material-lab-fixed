from __future__ import annotations

import unittest

from core.contracts_v2 import ProcessingState
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.registry import SystemGeneratorRegistryV3


class SystemGeneratorRegistryV3Tests(unittest.TestCase):
    def test_resolve_auto_known_system(self) -> None:
        reg = SystemGeneratorRegistryV3()
        mode, fallback, reason = reg.resolve_mode("system_auto", "fe-c")
        self.assertEqual(mode, "system_fe_c")
        self.assertFalse(fallback)
        self.assertIn("auto_by_inferred_system", reason)

    def test_resolve_auto_unknown_system_to_custom(self) -> None:
        reg = SystemGeneratorRegistryV3()
        mode, fallback, reason = reg.resolve_mode("system_auto", "x-y-z")
        self.assertEqual(mode, "system_custom")
        self.assertTrue(fallback)
        self.assertIn("auto_fallback_custom", reason)

    def test_manual_override_kept(self) -> None:
        reg = SystemGeneratorRegistryV3()
        mode, fallback, reason = reg.resolve_mode("system_al_si", "fe-c")
        self.assertEqual(mode, "system_al_si")
        self.assertFalse(fallback)
        self.assertEqual(reason, "manual_override")

    def test_generate_returns_selection(self) -> None:
        reg = SystemGeneratorRegistryV3()
        ctx = SystemGenerationContext(
            size=(64, 64),
            seed=1,
            inferred_system="fe-c",
            stage="pearlite",
            phase_fractions={"FERRITE": 0.2, "PEARLITE": 0.8},
            composition_wt={"Fe": 99.2, "C": 0.8},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="normalized"),
            confidence=0.91,
        )
        result, selection = reg.generate(context=ctx, requested_mode="system_auto")
        self.assertEqual(result.image_gray.shape, (64, 64))
        self.assertIn("PEARLITE", result.phase_masks)
        self.assertEqual(selection.resolved_mode, "system_fe_c")
        self.assertEqual(selection.resolved_stage, "pearlite")
        self.assertAlmostEqual(selection.confidence, 0.91, places=6)


if __name__ == "__main__":
    unittest.main()
