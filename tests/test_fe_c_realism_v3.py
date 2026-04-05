from __future__ import annotations

import unittest

from core.contracts_v2 import ProcessingState
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import render_fe_c_unified


class FeCRealismV3Tests(unittest.TestCase):
    def _martensite_ctx(self, *, c_wt: float, seed: int = 111) -> SystemGenerationContext:
        carbides = 0.06 if c_wt < 0.35 else 0.12
        return SystemGenerationContext(
            size=(160, 160),
            seed=seed,
            inferred_system="fe-c",
            stage="martensite",
            phase_fractions={"MARTENSITE": 1.0 - carbides, "CEMENTITE": carbides},
            composition_wt={"Fe": max(0.0, 100.0 - c_wt), "C": c_wt},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="water_quench"),
        )

    def test_martensite_style_tracks_carbon_content(self) -> None:
        low = render_fe_c_unified(self._martensite_ctx(c_wt=0.20, seed=501))
        high = render_fe_c_unified(self._martensite_ctx(c_wt=0.90, seed=501))

        low_trace = dict(low.metadata.get("fe_c_phase_render", {}).get("morphology_trace", {}))
        high_trace = dict(high.metadata.get("fe_c_phase_render", {}).get("morphology_trace", {}))

        self.assertEqual(low_trace.get("martensite_style"), "lath_dominant")
        self.assertEqual(high_trace.get("martensite_style"), "plate_dominant")
        self.assertLess(float(low_trace.get("band_spacing_px", 0.0)), float(high_trace.get("band_spacing_px", 0.0)))

    def test_faster_cooling_refines_pearlite_spacing(self) -> None:
        slow = render_fe_c_unified(
            SystemGenerationContext(
                size=(160, 160),
                seed=701,
                inferred_system="fe-c",
                stage="pearlite",
                phase_fractions={"PEARLITE": 0.78, "FERRITE": 0.22},
                composition_wt={"Fe": 99.2, "C": 0.8},
                processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            )
        )
        fast = render_fe_c_unified(
            SystemGenerationContext(
                size=(160, 160),
                seed=701,
                inferred_system="fe-c",
                stage="pearlite",
                phase_fractions={"PEARLITE": 0.78, "FERRITE": 0.22},
                composition_wt={"Fe": 99.2, "C": 0.8},
                processing=ProcessingState(temperature_c=20.0, cooling_mode="normalized"),
            )
        )

        slow_trace = dict(slow.metadata.get("fe_c_phase_render", {}).get("morphology_trace", {}))
        fast_trace = dict(fast.metadata.get("fe_c_phase_render", {}).get("morphology_trace", {}))

        self.assertGreater(float(slow_trace.get("interlamellar_spacing_px", 0.0)), float(fast_trace.get("interlamellar_spacing_px", 0.0)))
        self.assertGreater(float((slow.phase_masks["PEARLITE"] > 0).mean()), 0.60)
        self.assertGreater(float((fast.phase_masks["PEARLITE"] > 0).mean()), 0.60)

    def test_proeutectoid_phases_are_boundary_biased(self) -> None:
        hypo = render_fe_c_unified(
            SystemGenerationContext(
                size=(160, 160),
                seed=901,
                inferred_system="fe-c",
                stage="alpha_pearlite",
                phase_fractions={"FERRITE": 0.32, "PEARLITE": 0.68},
                composition_wt={"Fe": 99.4, "C": 0.6},
                processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            )
        )
        hyper = render_fe_c_unified(
            SystemGenerationContext(
                size=(160, 160),
                seed=902,
                inferred_system="fe-c",
                stage="pearlite_cementite",
                phase_fractions={"PEARLITE": 0.72, "CEMENTITE": 0.28},
                composition_wt={"Fe": 98.9, "C": 1.1},
                processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            )
        )

        hypo_trace = dict(hypo.metadata.get("fe_c_phase_render", {}).get("morphology_trace", {}))
        hyper_trace = dict(hyper.metadata.get("fe_c_phase_render", {}).get("morphology_trace", {}))

        self.assertEqual(hypo_trace.get("proeutectoid_phase"), "FERRITE")
        self.assertEqual(hyper_trace.get("proeutectoid_phase"), "CEMENTITE")
        self.assertGreater(float(hypo_trace.get("boundary_phase_bias", 0.0)), 0.60)
        self.assertGreater(float(hyper_trace.get("boundary_phase_bias", 0.0)), 0.60)
        self.assertGreater(float((hypo.phase_masks["FERRITE"] > 0).mean()), 0.10)
        self.assertGreater(float((hyper.phase_masks["CEMENTITE"] > 0).mean()), 0.08)


if __name__ == "__main__":
    unittest.main()
