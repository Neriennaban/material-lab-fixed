from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v3 import EtchProfileV3
from core.metallography_v3.etch_simulator import apply_etch


class EtchConcentrationV3Tests(unittest.TestCase):
    def test_concentration_wt_and_mol(self) -> None:
        img = np.full((64, 64), 128, dtype=np.uint8)
        phase = {"FERRITE": np.ones((64, 64), dtype=np.uint8)}

        out_wt = apply_etch(
            image_gray=img,
            phase_masks=phase,
            etch_profile=EtchProfileV3(
                reagent="nital_2",
                concentration_unit="wt_pct",
                concentration_value=2.0,
                concentration_wt_pct=2.0,
                concentration_mol_l=0.3,
            ),
            seed=42,
        )
        out_mol = apply_etch(
            image_gray=img,
            phase_masks=phase,
            etch_profile=EtchProfileV3(
                reagent="nital_2",
                concentration_unit="mol_l",
                concentration_value=0.32,
                concentration_wt_pct=2.0,
                concentration_mol_l=0.32,
            ),
            seed=42,
        )

        self.assertIn("etch_concentration", out_wt)
        self.assertIn("etch_concentration", out_mol)
        self.assertGreater(float(out_wt["etch_concentration"]["wt_pct"]), 0.0)
        self.assertGreater(float(out_mol["etch_concentration"]["mol_l"]), 0.0)


if __name__ == "__main__":
    unittest.main()
