from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v2 import ProcessingState
from core.metallography_v3.realism_utils import boundary_mask_from_labels
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import (
    _grain_map,
    render_fe_c_unified,
)


class HypoeutectoidFerriteBoundaryTests(unittest.TestCase):
    def test_ferrite_grain_boundaries_stay_visibly_darker(self) -> None:
        out = render_fe_c_unified(
            SystemGenerationContext(
                size=(160, 160),
                seed=903,
                inferred_system="fe-c",
                stage="alpha_pearlite",
                phase_fractions={"FERRITE": 0.34, "PEARLITE": 0.66},
                composition_wt={"Fe": 99.38, "C": 0.62},
                processing=ProcessingState(
                    temperature_c=20.0,
                    cooling_mode="equilibrium",
                ),
            )
        )

        trace = dict(
            out.metadata.get("fe_c_phase_render", {}).get("morphology_trace", {})
        )
        grain_scale = float(trace.get("prior_austenite_grain_size_px", 110.0))
        labels = _grain_map(
            size=(160, 160),
            seed=903 + 1001,
            mean_grain_size_px=grain_scale,
        )["labels"]
        ferrite_mask = out.phase_masks["FERRITE"] > 0
        boundary_core = (boundary_mask_from_labels(labels, width=1) > 0) & ferrite_mask
        boundary_band = (boundary_mask_from_labels(labels, width=2) > 0) & ferrite_mask
        ferrite_interior = ferrite_mask & (~boundary_band)

        self.assertGreater(int(boundary_core.sum()), 40)
        self.assertGreater(int(ferrite_interior.sum()), 200)

        image = out.image_gray.astype(np.float32)
        boundary_mean = float(image[boundary_core].mean())
        interior_mean = float(image[ferrite_interior].mean())
        self.assertGreater(
            interior_mean - boundary_mean,
            8.0,
            (
                "ferrite boundary contrast too weak: "
                f"interior={interior_mean:.2f}, boundary={boundary_mean:.2f}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
