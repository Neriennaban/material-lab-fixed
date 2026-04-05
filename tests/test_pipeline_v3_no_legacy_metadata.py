from __future__ import annotations

import unittest

from core.contracts_v3 import MetallographyRequestV3, ThermalPointV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


class PipelineV3NoLegacyMetadataTests(unittest.TestCase):
    def test_output_has_no_legacy_metadata_blocks(self) -> None:
        pipeline = MetallographyPipelineV3()
        req = MetallographyRequestV3(
            sample_id="strict_meta",
            composition_wt={"Fe": 99.2, "C": 0.8},
            system_hint="fe-c",
            resolution=(96, 96),
            seed=777,
        )
        req.thermal_program.points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=260.0, temperature_c=840.0),
            ThermalPointV3(time_s=420.0, temperature_c=840.0),
            ThermalPointV3(time_s=520.0, temperature_c=40.0),
        ]
        out = pipeline.generate(req)
        meta = dict(out.metadata)
        self.assertNotIn("calphad", meta)
        thermal = dict(meta.get("thermal_program_summary", {}))
        self.assertNotIn("legacy_route_converted", thermal)


if __name__ == "__main__":
    unittest.main()

