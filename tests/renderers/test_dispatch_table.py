"""Phase 1 guard: _STAGE_TO_RENDERER покрывает все ожидаемые стадии
редизайна и каждый модуль-семейство регистрирует ровно свои стадии.
"""
from __future__ import annotations

import unittest


class DispatchTableTests(unittest.TestCase):
    def test_dispatch_table_covers_target_stages(self) -> None:
        from core.metallography_v3.system_generators.fe_c_unified import (
            _STAGE_TO_RENDERER,
        )

        expected = {
            # martensite family
            "martensite",
            "martensite_tetragonal",
            "martensite_cubic",
            # bainite family
            "bainite_upper",
            "bainite_lower",
            "carbide_free_bainite",
            # tempered family
            "tempered_low",
            "tempered_medium",
            "tempered_high",
            "troostite_temper",
            "sorbite_temper",
            # quench products
            "troostite_quench",
            "sorbite_quench",
            # white cast iron
            "ledeburite",
            "white_cast_iron_hypoeutectic",
            "white_cast_iron_eutectic",
            "white_cast_iron_hypereutectic",
            # high-temperature phases
            "austenite",
            "delta_ferrite",
            "alpha_gamma",
            "gamma_cementite",
            "liquid",
            "liquid_gamma",
            # widmanstatten
            "widmanstatten_ferrite",
            # surface layers
            "decarburized_layer",
            "carburized_layer",
            # granular pearlite
            "granular_pearlite",
        }
        missing = expected - set(_STAGE_TO_RENDERER.keys())
        self.assertFalse(
            missing,
            f"_STAGE_TO_RENDERER missing expected stages: {sorted(missing)}",
        )

    def test_each_module_exports_renderer_contract(self) -> None:
        from core.metallography_v3.renderers import (
            bainite,
            granular_pearlite,
            high_temp_phases,
            martensite,
            quench_products,
            surface_layers,
            tempered,
            white_cast_iron,
            widmanstatten,
        )

        modules = (
            bainite,
            granular_pearlite,
            high_temp_phases,
            martensite,
            quench_products,
            surface_layers,
            tempered,
            white_cast_iron,
            widmanstatten,
        )
        for mod in modules:
            self.assertTrue(
                hasattr(mod, "HANDLES_STAGES"),
                f"{mod.__name__} missing HANDLES_STAGES",
            )
            self.assertIsInstance(mod.HANDLES_STAGES, frozenset)
            self.assertGreater(
                len(mod.HANDLES_STAGES),
                0,
                f"{mod.__name__} HANDLES_STAGES is empty",
            )
            self.assertTrue(
                hasattr(mod, "render"),
                f"{mod.__name__} missing render()",
            )
            self.assertTrue(
                callable(mod.render),
                f"{mod.__name__}.render is not callable",
            )

    def test_stage_mapping_is_unique(self) -> None:
        """Каждая стадия должна принадлежать ровно одному модулю."""
        from core.metallography_v3.system_generators.fe_c_unified import (
            _RENDERER_MODULES,
        )

        all_stages: list[str] = []
        for mod in _RENDERER_MODULES:
            all_stages.extend(mod.HANDLES_STAGES)
        self.assertEqual(
            len(all_stages),
            len(set(all_stages)),
            "Stage appears in more than one renderer module: "
            f"{sorted(s for s in set(all_stages) if all_stages.count(s) > 1)}",
        )

    def test_unimplemented_stubs_raise_not_implemented(self) -> None:
        """Модули-семейства, ещё не реализованные в Phase 2-4, должны
        бросать NotImplementedError при вызове render() (заглушка).

        По мере активации Phase 5-8 этот тест будет переключаться на
        следующий stub-модуль; когда все реализованы — тест удаляется.
        """
        # Phase 5-8 остаётся: bainite, tempered, quench_products,
        # widmanstatten, surface_layers, granular_pearlite.
        # Здесь тестируем widmanstatten (Phase 8, самый отложенный).
        from core.metallography_v3.renderers import widmanstatten
        from core.metallography_v3.system_generators.base import (
            SystemGenerationContext,
        )
        from core.contracts_v2 import ProcessingState

        ctx = SystemGenerationContext(
            size=(64, 64),
            seed=1,
            inferred_system="fe-c",
            stage="widmanstatten_ferrite",
            phase_fractions={"FERRITE": 0.5, "PEARLITE": 0.5},
            composition_wt={"Fe": 99.7, "C": 0.3},
            processing=ProcessingState(
                temperature_c=800.0,
                cooling_mode="air_cool",
            ),
        )
        with self.assertRaises(NotImplementedError):
            widmanstatten.render(
                context=ctx,
                stage="widmanstatten_ferrite",
                phase_fractions={"FERRITE": 0.5, "PEARLITE": 0.5},
                seed_split={
                    "seed_topology": 1,
                    "seed_boundary": 2,
                    "seed_particles": 3,
                    "seed_lamella": 4,
                    "seed_noise": 5,
                },
            )


if __name__ == "__main__":
    unittest.main()
