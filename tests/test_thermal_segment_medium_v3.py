from __future__ import annotations

import unittest

from core.contracts_v3 import ThermalPointV3, ThermalProgramV3, ThermalTransitionV3
from core.metallography_v3.thermal_program_v3 import sample_thermal_program, summarize_thermal_program


class ThermalSegmentMediumV3Tests(unittest.TestCase):
    def test_inherit_uses_quench_medium_on_cooling(self) -> None:
        program = ThermalProgramV3(
            points=[
                ThermalPointV3(time_s=0.0, temperature_c=840.0, transition_to_next=ThermalTransitionV3(model="auto", segment_medium_code="inherit")),
                ThermalPointV3(time_s=40.0, temperature_c=120.0),
            ],
            sampling_mode="per_degree",
            degree_step_c=20.0,
            max_frames=120,
        )
        program.quench.medium_code = "brine_20_30"
        rows = sample_thermal_program(program)
        self.assertTrue(rows)
        self.assertTrue(all(str(r.get("segment_medium_code", "")) == "brine_20_30" for r in rows))

    def test_effective_cooling_order_matches_medium_intensity(self) -> None:
        def _max_effective_for(medium: str) -> float:
            p = ThermalProgramV3(
                points=[
                    ThermalPointV3(
                        time_s=0.0,
                        temperature_c=860.0,
                        transition_to_next=ThermalTransitionV3(model="auto", segment_medium_code=medium),
                    ),
                    ThermalPointV3(time_s=80.0, temperature_c=40.0),
                ],
                sampling_mode="per_degree",
                degree_step_c=10.0,
                max_frames=150,
            )
            p.quench.medium_code = medium
            summary = summarize_thermal_program(p)
            return abs(float(summary.get("max_effective_cooling_rate_c_per_s", 0.0)))

        brine = _max_effective_for("brine_20_30")
        water20 = _max_effective_for("water_20")
        water100 = _max_effective_for("water_100")
        oil = _max_effective_for("oil_20_80")
        self.assertGreater(brine, water20)
        self.assertGreater(water20, water100)
        self.assertGreater(water100, oil)


if __name__ == "__main__":
    unittest.main()

