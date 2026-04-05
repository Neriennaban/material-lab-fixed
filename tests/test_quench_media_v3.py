from __future__ import annotations

import unittest

from core.metallography_v3.quench_media_v3 import (
    canonicalize_quench_medium_code,
    defaults_quench,
    list_quench_media,
    resolve_quench_medium,
)


class QuenchMediaV3Tests(unittest.TestCase):
    def test_catalog_and_defaults(self) -> None:
        media = list_quench_media()
        self.assertTrue(media)
        codes = {str(item.get("code", "")) for item in media}
        for code in ("water_20", "water_100", "brine_20_30", "oil_20_80"):
            self.assertIn(code, codes)
        defaults = defaults_quench()
        self.assertIn("medium_code", defaults)
        self.assertEqual(str(defaults["medium_code"]), "water_20")

    def test_custom_severity_applied(self) -> None:
        base = resolve_quench_medium(
            "oil_20_80",
            quench_time_s=40.0,
            bath_temperature_c=20.0,
            sample_temperature_c=840.0,
            custom_severity_factor=1.0,
        )
        weak = resolve_quench_medium(
            "oil_20_80",
            quench_time_s=40.0,
            bath_temperature_c=20.0,
            sample_temperature_c=840.0,
            custom_severity_factor=0.4,
        )
        self.assertLess(float(weak["severity_effective"]), float(base["severity_effective"]))

    def test_intensity_order_for_key_media(self) -> None:
        brine = resolve_quench_medium(
            "brine_20_30",
            quench_time_s=30.0,
            bath_temperature_c=25.0,
            sample_temperature_c=840.0,
        )
        water20 = resolve_quench_medium(
            "water_20",
            quench_time_s=30.0,
            bath_temperature_c=20.0,
            sample_temperature_c=840.0,
        )
        water100 = resolve_quench_medium(
            "water_100",
            quench_time_s=30.0,
            bath_temperature_c=100.0,
            sample_temperature_c=840.0,
        )
        oil = resolve_quench_medium(
            "oil_20_80",
            quench_time_s=30.0,
            bath_temperature_c=60.0,
            sample_temperature_c=840.0,
        )
        s_brine = float(brine["severity_effective"])
        s_w20 = float(water20["severity_effective"])
        s_w100 = float(water100["severity_effective"])
        s_oil = float(oil["severity_effective"])
        self.assertGreater(s_brine, s_w20)
        self.assertGreater(s_w20, s_w100)
        self.assertGreater(s_w100, s_oil)

    def test_legacy_water_mapping(self) -> None:
        low = canonicalize_quench_medium_code("water", bath_temperature_c=20.0)
        high = canonicalize_quench_medium_code("water", bath_temperature_c=95.0)
        self.assertEqual(str(low.get("resolved_code", "")), "water_20")
        self.assertEqual(str(high.get("resolved_code", "")), "water_100")


if __name__ == "__main__":
    unittest.main()
