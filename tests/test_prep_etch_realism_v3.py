from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v3 import EtchProfileV3, MetallographyRequestV3, PrepOperationV3, SamplePrepRouteV3, SynthesisProfileV3, ThermalPointV3
from core.metallography_v3.etch_simulator import apply_etch
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3
from core.metallography_v3.prep_simulator import apply_prep_route


class PrepEtchRealismV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((128, 128), 128, dtype=np.uint8)
        self.soft = np.zeros((128, 128), dtype=np.uint8)
        self.soft[:, :64] = 1
        self.brittle = np.zeros((128, 128), dtype=np.uint8)
        self.brittle[:, 64:] = 1
        self.route = SamplePrepRouteV3(
            steps=[
                PrepOperationV3(method="grinding", duration_s=120.0, abrasive_um=15.0, load_n=30.0, rpm=240.0, direction_deg=18.0),
                PrepOperationV3(method="polishing", duration_s=90.0, abrasive_um=3.0, load_n=18.0, rpm=160.0, cloth_type="soft", slurry_type="diamond"),
            ],
            roughness_target_um=0.05,
        )

    def test_brittle_phase_gets_more_pullout_than_soft_matrix(self) -> None:
        prep = apply_prep_route(
            image_gray=self.image,
            prep_route=self.route,
            seed=33,
            phase_masks={"FCC_A1": self.soft, "SI": self.brittle},
            system="al-si",
        )

        pullout = prep["prep_maps"]["pullout"].astype(np.float32)
        left_mean = float(pullout[:, :64].mean())
        right_mean = float(pullout[:, 64:].mean())

        self.assertTrue(bool(prep["prep_summary"].get("phase_coupling_applied")))
        self.assertGreater(right_mean, left_mean * 5.0)
        self.assertGreater(float(prep["prep_summary"].get("pullout_mean", 0.0)), 0.0)

    def test_etch_uses_prep_maps_and_returns_selectivity_layers(self) -> None:
        prep = apply_prep_route(
            image_gray=self.image,
            prep_route=self.route,
            seed=33,
            phase_masks={"FCC_A1": self.soft, "SI": self.brittle},
            system="al-si",
        )
        etched = apply_etch(
            image_gray=prep["image_gray"],
            phase_masks={"FCC_A1": self.soft, "SI": self.brittle},
            etch_profile=EtchProfileV3(reagent="keller"),
            seed=44,
            prep_maps=prep["prep_maps"],
            system="al-si",
        )

        self.assertTrue(bool(etched["etch_summary"].get("prep_coupling_applied")))
        self.assertIn("etch_maps", etched)
        self.assertIn("selectivity", etched["etch_maps"])
        self.assertIn("stain", etched["etch_maps"])
        etch_rate = etched["etch_maps"]["etch_rate"].astype(np.float32)
        self.assertGreater(float(etch_rate[:, 64:].mean()), float(etch_rate[:, :64].mean()))

    def test_pipeline_propagates_prep_and_etch_coupling(self) -> None:
        pipeline = MetallographyPipelineV3()
        req = MetallographyRequestV3(
            sample_id="prep_etch_pipe",
            system_hint="al-si",
            composition_wt={"Al": 87.4, "Si": 12.6},
            resolution=(96, 96),
            seed=2468,
            synthesis_profile=SynthesisProfileV3(generation_mode="realistic_visual", system_generator_mode="system_auto"),
            prep_route=self.route,
            etch_profile=EtchProfileV3(reagent="keller"),
        )
        req.thermal_program.points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=200.0, temperature_c=730.0),
            ThermalPointV3(time_s=280.0, temperature_c=730.0),
            ThermalPointV3(time_s=920.0, temperature_c=20.0),
        ]

        out = pipeline.generate(req)

        self.assertTrue(bool(out.metadata.get("prep_summary", {}).get("phase_coupling_applied")))
        self.assertTrue(bool(out.metadata.get("etch_summary", {}).get("prep_coupling_applied")))
        self.assertIn("stain_level_mean", out.metadata.get("etch_summary", {}))
        self.assertIn("pullout_mean", out.metadata.get("prep_summary", {}))

    def test_pure_ferrite_baseline_is_cleaner_than_aggressive_route(self) -> None:
        ferrite = np.ones((128, 128), dtype=np.uint8)
        gentle_route = SamplePrepRouteV3(
            steps=[
                PrepOperationV3(method="grinding_800", duration_s=75.0, abrasive_um=18.0, load_n=18.0, rpm=150.0, coolant="alcohol"),
                PrepOperationV3(method="polishing_1um", duration_s=90.0, abrasive_um=1.0, load_n=8.0, rpm=105.0, cloth_type="napless", slurry_type="colloidal_silica"),
            ],
            roughness_target_um=0.035,
            contamination_level=0.006,
        )
        aggressive_route = SamplePrepRouteV3(
            steps=[
                PrepOperationV3(method="grinding_600", duration_s=180.0, abrasive_um=28.0, load_n=32.0, rpm=280.0, coolant="none"),
                PrepOperationV3(method="polishing_1um", duration_s=210.0, abrasive_um=1.0, load_n=20.0, rpm=160.0, cloth_type="long_nap", slurry_type="diamond"),
            ],
            roughness_target_um=0.07,
            contamination_level=0.04,
        )

        gentle_prep = apply_prep_route(
            image_gray=self.image,
            prep_route=gentle_route,
            seed=90,
            phase_masks={"BCC_B2": ferrite},
            system="fe-si",
            composition_wt={"Fe": 99.95},
        )
        aggressive_prep = apply_prep_route(
            image_gray=self.image,
            prep_route=aggressive_route,
            seed=91,
            phase_masks={"BCC_B2": ferrite},
            system="fe-si",
            composition_wt={"Fe": 99.95},
        )
        self.assertTrue(bool(gentle_prep["prep_summary"].get("pure_iron_baseline_applied")))
        self.assertLess(float(gentle_prep["prep_summary"]["scratch_mean"]), float(aggressive_prep["prep_summary"]["scratch_mean"]))
        self.assertLess(float(gentle_prep["prep_summary"]["smear_mean"]), float(aggressive_prep["prep_summary"]["smear_mean"]))
        self.assertLess(
            float(gentle_prep["prep_summary"]["false_porosity_from_chipping_risk"]),
            float(aggressive_prep["prep_summary"]["false_porosity_from_chipping_risk"]),
        )

        gentle_etch = apply_etch(
            image_gray=gentle_prep["image_gray"],
            phase_masks={"BCC_B2": ferrite},
            etch_profile=EtchProfileV3(reagent="nital_2"),
            seed=92,
            prep_maps=gentle_prep["prep_maps"],
            system="fe-si",
            composition_wt={"Fe": 99.95},
        )
        aggressive_etch = apply_etch(
            image_gray=aggressive_prep["image_gray"],
            phase_masks={"BCC_B2": ferrite},
            etch_profile=EtchProfileV3(reagent="nital_2", time_s=12.0, overetch_factor=1.2, agitation="active"),
            seed=93,
            prep_maps=aggressive_prep["prep_maps"],
            system="fe-si",
            composition_wt={"Fe": 99.95},
        )
        self.assertTrue(bool(gentle_etch["etch_summary"].get("pure_iron_baseline_applied")))
        self.assertGreater(
            float(gentle_etch["etch_summary"]["pure_iron_cleanliness_score"]),
            float(aggressive_etch["etch_summary"]["pure_iron_cleanliness_score"]),
        )
        self.assertLess(
            float(gentle_etch["etch_summary"]["stain_level_mean"]),
            float(aggressive_etch["etch_summary"]["stain_level_mean"]),
        )


if __name__ == "__main__":
    unittest.main()
