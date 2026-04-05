from __future__ import annotations

import unittest

from core.contracts_v2 import GenerationRequestV2, ProcessingState
from core.pipeline_v2 import GenerationPipelineV2


class PipelineV2CoolingCurveSeriesTests(unittest.TestCase):
    def test_generate_cooling_curve_series(self) -> None:
        pipeline = GenerationPipelineV2()
        request = GenerationRequestV2(
            mode="direct",
            composition={"Fe": 99.2, "C": 0.8},
            processing=ProcessingState(temperature_c=1600.0, cooling_mode="equilibrium"),
            generator="auto",
            generator_params={
                "cooling_curve_enabled": True,
                "cooling_curve_mode": "per_degree",
                "cooling_curve_degree_step": 20.0,
                "cooling_curve_max_points": 80,
                "cooling_curve": [
                    {"time_min": 0.0, "temperature_c": 1600.0},
                    {"time_min": 6.0, "temperature_c": 20.0},
                ],
            },
            seed=515,
            resolution=(96, 96),
            strict_validation=True,
        )

        series = pipeline.generate_cooling_curve_series(request)
        self.assertGreaterEqual(len(series), 2)
        self.assertTrue(all("cooling_curve_point" in frame.metadata for frame in series))
        self.assertTrue(all("phase_transition_track" in frame.metadata for frame in series))
        self.assertTrue(all("phase_transition_state" in frame.metadata for frame in series))
        self.assertTrue(all("phase_transition_events" in frame.metadata for frame in series))
        self.assertTrue(all(frame.metadata.get("cooling_curve_series", {}).get("enabled") for frame in series))

        summary = series[-1].metadata.get("cooling_curve_series", {}).get("summary", {})
        render_cache = summary.get("render_cache", {})
        self.assertLessEqual(int(render_cache.get("unique_renders", len(series))), len(series))
        self.assertGreaterEqual(int(render_cache.get("cache_hits", 0)), 0)
        self.assertTrue(all("calphad" in frame.metadata for frame in series))
        self.assertTrue(
            all(
                isinstance(frame.metadata.get("calphad", {}).get("equilibrium_result", {}), dict)
                for frame in series
            )
        )


if __name__ == "__main__":
    unittest.main()
