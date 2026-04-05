from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v2 import ProcessingState
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.registry import SystemGeneratorRegistryV3


class SystemGeneratorsV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = SystemGeneratorRegistryV3()

    def _run_case(self, mode: str, ctx: SystemGenerationContext) -> None:
        out1, sel1 = self.registry.generate(context=ctx, requested_mode=mode)
        out2, sel2 = self.registry.generate(context=ctx, requested_mode=mode)
        self.assertEqual(out1.image_gray.shape, ctx.size)
        self.assertEqual(out1.image_gray.dtype, np.uint8)
        self.assertIsInstance(out1.phase_masks, dict)
        self.assertEqual(sel1.resolved_mode, mode)
        self.assertEqual(sel2.resolved_mode, mode)
        self.assertTrue(np.array_equal(out1.image_gray, out2.image_gray))

    def test_fe_c_generator(self) -> None:
        ctx = SystemGenerationContext(
            size=(96, 96),
            seed=101,
            inferred_system="fe-c",
            stage="pearlite",
            phase_fractions={"FERRITE": 0.25, "PEARLITE": 0.75},
            composition_wt={"Fe": 99.2, "C": 0.8},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="normalized"),
        )
        self._run_case("system_fe_c", ctx)

    def test_fe_si_generator(self) -> None:
        ctx = SystemGenerationContext(
            size=(96, 96),
            seed=102,
            inferred_system="fe-si",
            stage="recrystallized_ferrite",
            phase_fractions={"BCC_B2": 0.95, "FESI_INTERMETALLIC": 0.05},
            composition_wt={"Fe": 98.6, "Si": 1.4},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
        )
        self._run_case("system_fe_si", ctx)

    def test_al_si_generator(self) -> None:
        ctx = SystemGenerationContext(
            size=(96, 96),
            seed=103,
            inferred_system="al-si",
            stage="eutectic",
            phase_fractions={"FCC_A1": 0.2, "EUTECTIC_ALSI": 0.7, "SI": 0.1},
            composition_wt={"Al": 87.4, "Si": 12.6},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
        )
        self._run_case("system_al_si", ctx)

    def test_cu_zn_generator(self) -> None:
        ctx = SystemGenerationContext(
            size=(96, 96),
            seed=104,
            inferred_system="cu-zn",
            stage="alpha_beta",
            phase_fractions={"ALPHA": 0.65, "BETA": 0.35},
            composition_wt={"Cu": 60.0, "Zn": 40.0},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
        )
        self._run_case("system_cu_zn", ctx)

    def test_al_cu_mg_generator(self) -> None:
        ctx = SystemGenerationContext(
            size=(96, 96),
            seed=105,
            inferred_system="al-cu-mg",
            stage="artificial_aged",
            phase_fractions={"FCC_A1": 0.86, "THETA": 0.08, "S_PHASE": 0.06},
            composition_wt={"Al": 94.1, "Cu": 4.4, "Mg": 1.5},
            processing=ProcessingState(temperature_c=180.0, cooling_mode="aged", aging_hours=12.0, aging_temperature_c=180.0),
        )
        self._run_case("system_al_cu_mg", ctx)

    def test_custom_generator(self) -> None:
        ctx = SystemGenerationContext(
            size=(96, 96),
            seed=106,
            inferred_system="custom-multicomponent",
            stage="custom_equilibrium",
            phase_fractions={"MATRIX": 0.72, "SECONDARY": 0.18, "PRECIPITATES": 0.10},
            composition_wt={"Ni": 62.0, "Cr": 24.0, "Mo": 9.0, "W": 5.0},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
        )
        self._run_case("system_custom", ctx)


if __name__ == "__main__":
    unittest.main()
