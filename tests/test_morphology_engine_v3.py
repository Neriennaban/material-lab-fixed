from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v2 import ProcessingState
from core.contracts_v3 import SynthesisProfileV3, ThermalPointV3, ThermalProgramV3
from core.metallography_v3.microstructure_state import build_microstructure_state
from core.metallography_v3.morphology_engine import generate_phase_topology
from core.metallography_v3.phase_orchestrator import PhaseBundleV3
from core.metallography_v3.thermal_program_v3 import (
    effective_processing_from_thermal,
    infer_operations_from_thermal_program,
    summarize_thermal_program,
)


def _micro_state(seed: int) -> tuple[ProcessingState, dict, dict, object]:
    program = ThermalProgramV3(
        points=[
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=200.0, temperature_c=780.0),
            ThermalPointV3(time_s=320.0, temperature_c=780.0),
            ThermalPointV3(time_s=620.0, temperature_c=30.0),
        ]
    )
    processing, thermal_summary, quench_summary = effective_processing_from_thermal(program)
    ops = infer_operations_from_thermal_program(program, summary=summarize_thermal_program(program), quench_summary=quench_summary)
    micro_state = build_microstructure_state(
        composition={"Fe": 99.2, "C": 0.8},
        inferred_system="fe-c",
        processing=processing,
        thermal_summary=thermal_summary,
        operations_from_curve=ops,
        quench_summary=quench_summary,
        seed=seed,
    )
    return processing, thermal_summary, quench_summary, micro_state


class MorphologyEngineV3Tests(unittest.TestCase):
    def test_generate_system_generator_path(self) -> None:
        _, thermal_summary, quench_summary, micro_state = _micro_state(seed=123)
        phase_bundle = PhaseBundleV3(
            system="fe-c",
            stage="pearlite",
            phase_fractions={"FERRITE": 0.2, "PEARLITE": 0.8},
            phase_model_report={
                "auto_phase_fractions": {"FERRITE": 0.2, "PEARLITE": 0.8},
                "manual_phase_fractions": {},
                "blended_phase_fractions": {"FERRITE": 0.2, "PEARLITE": 0.8},
                "blend_applied": False,
                "fallback_used": False,
                "fallback_reason": "",
            },
            confidence=0.9,
        )
        out = generate_phase_topology(
            size=(96, 96),
            seed=123,
            phase_bundle=phase_bundle,
            micro_state=micro_state,
            synthesis_profile=SynthesisProfileV3(),
            reference_style=None,
            composition_wt={"Fe": 99.2, "C": 0.8},
            composition_sensitivity_mode="realistic",
            generation_mode="edu_engineering",
            phase_emphasis_style="contrast_texture",
            phase_fraction_tolerance_pct=20.0,
            thermal_summary=thermal_summary,
            quench_summary=quench_summary,
        )
        self.assertIn("image_gray", out)
        self.assertEqual(out["image_gray"].shape, (96, 96))
        self.assertEqual(out["image_gray"].dtype, np.uint8)
        self.assertIn("phase_masks", out)
        self.assertIsInstance(out["phase_masks"], dict)
        self.assertIn("system_generator", out)
        sysgen = dict(out["system_generator"])
        self.assertEqual(sysgen.get("resolved_mode"), "system_fe_c")

    def test_reject_v2_compat_mode(self) -> None:
        _, thermal_summary, quench_summary, micro_state = _micro_state(seed=456)
        phase_bundle = PhaseBundleV3(
            system="fe-c",
            stage="pearlite",
            phase_fractions={"FERRITE": 0.2, "PEARLITE": 0.8},
            phase_model_report={
                "auto_phase_fractions": {"FERRITE": 0.2, "PEARLITE": 0.8},
                "manual_phase_fractions": {},
                "blended_phase_fractions": {"FERRITE": 0.2, "PEARLITE": 0.8},
                "blend_applied": False,
                "fallback_used": False,
                "fallback_reason": "",
            },
            confidence=0.9,
        )
        profile = SynthesisProfileV3(phase_topology_mode="v2_pearlite")
        with self.assertRaises(ValueError):
            generate_phase_topology(
                size=(96, 96),
                seed=456,
                phase_bundle=phase_bundle,
                micro_state=micro_state,
                synthesis_profile=profile,
                reference_style=None,
                composition_wt={"Fe": 99.2, "C": 0.8},
                composition_sensitivity_mode="realistic",
                generation_mode="edu_engineering",
                phase_emphasis_style="contrast_texture",
                phase_fraction_tolerance_pct=20.0,
                thermal_summary=thermal_summary,
                quench_summary=quench_summary,
            )


if __name__ == "__main__":
    unittest.main()

