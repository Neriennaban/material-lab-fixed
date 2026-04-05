from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v2 import ProcessingState
from core.generator_phase_map import SYSTEM_STAGE_ORDER
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import render_fe_c_unified


class FeCUnifiedStageCoverageTests(unittest.TestCase):
    def test_all_fe_c_stages_render(self) -> None:
        stages = list(SYSTEM_STAGE_ORDER.get("fe-c", []))
        self.assertTrue(stages)
        for idx, stage in enumerate(stages):
            ctx = SystemGenerationContext(
                size=(96, 96),
                seed=1000 + idx,
                inferred_system="fe-c",
                stage=stage,
                phase_fractions={},
                composition_wt={"Fe": 99.2, "C": 0.8},
                processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            )
            out = render_fe_c_unified(ctx)
            self.assertEqual(out.image_gray.shape, ctx.size, msg=stage)
            self.assertEqual(out.image_gray.dtype, np.uint8, msg=stage)
            self.assertTrue(bool(out.phase_masks), msg=stage)
            meta = dict(out.metadata)
            self.assertIn("fe_c_phase_render", meta, msg=stage)
            self.assertIn("system_generator_extra", meta, msg=stage)
            fe_c_unified = dict(meta.get("system_generator_extra", {}).get("fe_c_unified", {}))
            self.assertTrue(bool(fe_c_unified.get("stage_coverage_pass", False)), msg=stage)


if __name__ == "__main__":
    unittest.main()
