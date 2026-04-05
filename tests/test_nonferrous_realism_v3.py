from __future__ import annotations

import unittest

from core.contracts_v2 import ProcessingState
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.generator_al_cu_mg import generate_al_cu_mg
from core.metallography_v3.system_generators.generator_al_si import generate_al_si
from core.metallography_v3.system_generators.generator_cu_zn import generate_cu_zn


class NonFerrousRealismV3Tests(unittest.TestCase):
    def test_al_si_cast_refines_with_faster_cooling(self) -> None:
        slow = generate_al_si(
            SystemGenerationContext(
                size=(112, 112),
                seed=601,
                inferred_system="al-si",
                stage="eutectic",
                phase_fractions={"FCC_A1": 0.20, "EUTECTIC_ALSI": 0.70, "SI": 0.10},
                composition_wt={"Al": 87.4, "Si": 12.6},
                processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            )
        )
        fast = generate_al_si(
            SystemGenerationContext(
                size=(112, 112),
                seed=601,
                inferred_system="al-si",
                stage="eutectic",
                phase_fractions={"FCC_A1": 0.20, "EUTECTIC_ALSI": 0.70, "SI": 0.10},
                composition_wt={"Al": 87.4, "Si": 12.6},
                processing=ProcessingState(temperature_c=20.0, cooling_mode="water_quench"),
            )
        )

        slow_meta = dict(slow.metadata.get("system_generator_extra", {}).get("al_si_morphology", {}))
        fast_meta = dict(fast.metadata.get("system_generator_extra", {}).get("al_si_morphology", {}))

        self.assertGreater(float(slow_meta.get("dendrite_arm_spacing_px", 0.0)), float(fast_meta.get("dendrite_arm_spacing_px", 0.0)))
        self.assertGreater(float(slow_meta.get("eutectic_scale_px", 0.0)), float(fast_meta.get("eutectic_scale_px", 0.0)))

    def test_al_si_hypereutectic_contains_primary_si(self) -> None:
        out = generate_al_si(
            SystemGenerationContext(
                size=(112, 112),
                seed=602,
                inferred_system="al-si",
                stage="primary_si_eutectic",
                phase_fractions={"SI": 0.24, "EUTECTIC_ALSI": 0.56, "FCC_A1": 0.20},
                composition_wt={"Al": 75.0, "Si": 25.0},
                processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            )
        )
        meta = dict(out.metadata.get("system_generator_extra", {}).get("al_si_morphology", {}))
        self.assertGreater(float(meta.get("primary_si_fraction_visual", 0.0)), 0.10)
        self.assertGreater(float((out.phase_masks["SI"] > 0).mean()), 0.12)

    def test_cu_zn_alpha_beta_has_twins_and_boundary_beta(self) -> None:
        out = generate_cu_zn(
            SystemGenerationContext(
                size=(112, 112),
                seed=603,
                inferred_system="cu-zn",
                stage="alpha_beta",
                phase_fractions={"ALPHA": 0.65, "BETA": 0.35},
                composition_wt={"Cu": 60.0, "Zn": 40.0},
                processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            )
        )
        meta = dict(out.metadata.get("system_generator_extra", {}).get("cu_zn_morphology", {}))
        self.assertGreater(float(meta.get("alpha_twins_density", 0.0)), 0.10)
        self.assertGreater(float(meta.get("beta_boundary_bias", 0.0)), 0.60)

    def test_al_cu_mg_overaged_is_coarser_than_artificially_aged(self) -> None:
        artificial = generate_al_cu_mg(
            SystemGenerationContext(
                size=(112, 112),
                seed=604,
                inferred_system="al-cu-mg",
                stage="artificial_aged",
                phase_fractions={"FCC_A1": 0.84, "THETA": 0.10, "S_PHASE": 0.06},
                composition_wt={"Al": 94.1, "Cu": 4.4, "Mg": 1.5},
                processing=ProcessingState(temperature_c=180.0, cooling_mode="aged", aging_hours=12.0, aging_temperature_c=180.0),
            )
        )
        overaged = generate_al_cu_mg(
            SystemGenerationContext(
                size=(112, 112),
                seed=604,
                inferred_system="al-cu-mg",
                stage="overaged",
                phase_fractions={"FCC_A1": 0.78, "THETA": 0.12, "S_PHASE": 0.07, "QPHASE": 0.03},
                composition_wt={"Al": 93.0, "Cu": 5.0, "Mg": 1.5, "Si": 0.5},
                processing=ProcessingState(temperature_c=220.0, cooling_mode="aged", aging_hours=36.0, aging_temperature_c=220.0),
            )
        )

        artificial_meta = dict(artificial.metadata.get("system_generator_extra", {}).get("al_cu_mg_morphology", {}))
        overaged_meta = dict(overaged.metadata.get("system_generator_extra", {}).get("al_cu_mg_morphology", {}))

        self.assertGreater(float(overaged_meta.get("precipitate_scale_px", 0.0)), float(artificial_meta.get("precipitate_scale_px", 0.0)))
        self.assertGreaterEqual(float(overaged_meta.get("pfz_width_px", 0.0)), float(artificial_meta.get("pfz_width_px", 0.0)))


if __name__ == "__main__":
    unittest.main()
