from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from core.contracts_v3 import EtchProfileV3, PrepOperationV3, SamplePrepRouteV3
from core.metallography_v3.etch_simulator import apply_etch
from core.metallography_v3.prep_simulator import apply_prep_route


class ElectropolishProfilesV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = np.full((128, 128), 160, dtype=np.uint8)
        self.ferrite = np.ones((128, 128), dtype=np.uint8)
        self.duplex_left = np.zeros((128, 128), dtype=np.uint8)
        self.duplex_left[:, :64] = 1
        self.duplex_right = np.zeros((128, 128), dtype=np.uint8)
        self.duplex_right[:, 64:] = 1
        self.profile_path = Path("profiles_v3") / "electropolish_profiles.json"

    def test_curated_profiles_roundtrip_through_contracts(self) -> None:
        payload = json.loads(self.profile_path.read_text(encoding="utf-8"))
        profiles = dict(payload.get("profiles", {}))
        self.assertIn("pure_iron_electropolish", profiles)
        self.assertIn("steel_electropolish", profiles)
        self.assertIn("stainless_electropolish", profiles)
        self.assertIn("copper_alloy_electropolish", profiles)
        self.assertIn("aluminum_electropolish", profiles)

        pure_step = PrepOperationV3.from_dict(
            {
                "method": profiles["pure_iron_electropolish"]["prep_method"],
                "electrolyte_code": profiles["pure_iron_electropolish"]["electrolyte_code"],
                "voltage_v": profiles["pure_iron_electropolish"]["voltage_v"],
                "electrolyte_temperature_c": profiles["pure_iron_electropolish"]["electrolyte_temperature_c"],
                "post_polish_followup": profiles["pure_iron_electropolish"]["post_polish_followup"],
            }
        )
        copper_step = PrepOperationV3.from_dict(
            {
                "method": profiles["copper_alloy_electropolish"]["prep_method"],
                "electrolyte_code": profiles["copper_alloy_electropolish"]["electrolyte_code"],
                "voltage_v": profiles["copper_alloy_electropolish"]["voltage_v"],
                "electrolyte_temperature_c": profiles["copper_alloy_electropolish"]["electrolyte_temperature_c"],
                "spot_diameter_mm": profiles["copper_alloy_electropolish"]["spot_diameter_mm"],
                "probe_tip_radius_mm": profiles["copper_alloy_electropolish"]["probe_tip_radius_mm"],
                "movement_pattern": profiles["copper_alloy_electropolish"]["movement_pattern"],
                "electrolyte_refresh_interval_s": profiles["copper_alloy_electropolish"]["electrolyte_refresh_interval_s"],
            }
        )

        self.assertEqual(str(pure_step.method), "electropolish_bath")
        self.assertEqual(str(pure_step.electrolyte_code), "II-1")
        self.assertEqual(str(copper_step.method), "local_electropolish_tampon")
        self.assertGreater(float(copper_step.spot_diameter_mm or 0.0), 0.0)

    def test_electropolish_bath_reduces_damage_for_pure_ferrite(self) -> None:
        mechanical = apply_prep_route(
            image_gray=self.sample,
            prep_route=SamplePrepRouteV3(
                steps=[
                    PrepOperationV3(method="grinding_800", duration_s=90.0, abrasive_um=16.0, load_n=18.0, rpm=180.0),
                    PrepOperationV3(method="polishing_1um", duration_s=120.0, abrasive_um=1.0, load_n=10.0, rpm=140.0),
                ],
                roughness_target_um=0.04,
                contamination_level=0.01,
            ),
            seed=510,
            phase_masks={"BCC_B2": self.ferrite},
            system="fe-si",
            composition_wt={"Fe": 99.95},
        )
        electropolished = apply_prep_route(
            image_gray=self.sample,
            prep_route=SamplePrepRouteV3(
                steps=[
                    PrepOperationV3(method="grinding_800", duration_s=90.0, abrasive_um=16.0, load_n=18.0, rpm=180.0),
                    PrepOperationV3(
                        method="electropolish_bath",
                        duration_s=75.0,
                        voltage_v=30.0,
                        electrolyte_code="II-1",
                        electrolyte_temperature_c=20.0,
                        post_polish_followup="electrolytic_etch",
                    ),
                ],
                roughness_target_um=0.03,
                contamination_level=0.01,
            ),
            seed=511,
            phase_masks={"BCC_B2": self.ferrite},
            system="fe-si",
            composition_wt={"Fe": 99.95},
        )

        mech_summary = dict(mechanical["prep_summary"])
        electro_summary = dict(electropolished["prep_summary"])
        self.assertEqual(str(electro_summary.get("electropolish_mode", "")), "bath")
        self.assertEqual(str(electro_summary.get("electropolish_profile_id", "")), "ii-1")
        self.assertLess(float(electro_summary.get("scratch_mean", 1.0)), float(mech_summary.get("scratch_mean", 0.0)))
        self.assertLess(float(electro_summary.get("smear_mean", 1.0)), float(mech_summary.get("smear_mean", 0.0)))
        self.assertGreater(float(electro_summary.get("pure_iron_cleanliness_score", 0.0)), 0.55)
        self.assertTrue(bool(electro_summary.get("post_electropolish_electroetch_used", False)))

    def test_local_electropolish_and_electrolytic_etch_remain_local(self) -> None:
        prep = apply_prep_route(
            image_gray=self.sample,
            prep_route=SamplePrepRouteV3(
                steps=[
                    PrepOperationV3(method="grinding", duration_s=120.0, abrasive_um=15.0, load_n=24.0, rpm=220.0),
                    PrepOperationV3(
                        method="local_electropolish_tampon",
                        duration_s=60.0,
                        electrolyte_code="table3_copper_alloy",
                        voltage_v=5.0,
                        electrolyte_temperature_c=10.0,
                        spot_diameter_mm=7.0,
                        probe_tip_radius_mm=1.6,
                        movement_pattern="circular",
                        electrolyte_refresh_interval_s=60.0,
                    ),
                ],
                roughness_target_um=0.04,
                contamination_level=0.02,
            ),
            seed=512,
            phase_masks={"ALPHA": self.duplex_left, "BETA": self.duplex_right},
            system="cu-zn",
            composition_wt={"Cu": 60.0, "Zn": 40.0},
        )
        etched = apply_etch(
            image_gray=prep["image_gray"],
            phase_masks={"ALPHA": self.duplex_left, "BETA": self.duplex_right},
            etch_profile=EtchProfileV3(
                reagent="custom",
                etch_mode="electrolytic",
                electrolyte_code="table3_copper_alloy",
                voltage_ratio_to_polish=0.1,
                area_mode="local",
                requires_prior_electropolish=True,
            ),
            seed=513,
            prep_maps=prep["prep_maps"],
            system="cu-zn",
            composition_wt={"Cu": 60.0, "Zn": 40.0},
        )

        prep_summary = dict(prep["prep_summary"])
        etch_summary = dict(etched["etch_summary"])
        self.assertEqual(str(prep_summary.get("electropolish_mode", "")), "tampon")
        self.assertGreater(float(prep_summary.get("local_area_fraction", 0.0)), 0.0)
        self.assertLess(float(prep_summary.get("local_area_fraction", 1.0)), 1.0)
        self.assertEqual(str(etch_summary.get("etch_mode", "")), "electrolytic")
        self.assertTrue(bool(etch_summary.get("post_electropolish_electroetch_used", False)))
        self.assertGreater(float(etch_summary.get("local_area_fraction", 0.0)), 0.0)
        self.assertLess(float(etch_summary.get("local_area_fraction", 1.0)), 1.0)
        self.assertGreater(float(etch_summary.get("phase_relief_risk", 0.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
