"""C3 — generate the ferro-micro atlas of 18+ Fe-C microstructures.

The TZ §15 asks for an atlas covering the carbon range from pure
ferrite (0 % C) to hypereutectic white cast iron (>4.3 % C). This
script renders a fixed list of 18 reference compositions plus a
JSON manifest with per-sample phase fractions and metric scores
(when ``--with-metrics`` is passed).

Example
-------
python -m scripts.generate_ferro_micro_atlas --output-dir examples/ferro_micro_atlas/
python -m scripts.generate_ferro_micro_atlas --output-dir /tmp/atlas/ --with-metrics --width 512 --height 512
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.metallography_v3 import ferro_micro_api as fm  # noqa: E402
from core.metallography_v3.quality_metrics_ferro import (  # noqa: E402
    phase_fraction_error,
)


# 18 reference compositions covering the full Fe-C diagram.
ATLAS_RECIPES = [
    {"label": "armco_iron", "carbon": 0.02, "magnification": 100},
    {"label": "steel_08", "carbon": 0.08, "magnification": 200},
    {"label": "steel_10", "carbon": 0.10, "magnification": 200},
    {"label": "steel_20", "carbon": 0.20, "magnification": 400},
    {"label": "aisi_1030", "carbon": 0.30, "magnification": 400},
    {"label": "steel_45", "carbon": 0.45, "magnification": 400},
    {"label": "aisi_1050", "carbon": 0.50, "magnification": 400},
    {"label": "steel_60", "carbon": 0.60, "magnification": 400},
    {"label": "eutectoid_077", "carbon": 0.77, "magnification": 500},
    {"label": "steel_u8", "carbon": 0.80, "magnification": 500},
    {"label": "steel_u10", "carbon": 1.00, "magnification": 500},
    {"label": "steel_u12", "carbon": 1.20, "magnification": 500},
    {"label": "steel_u13", "carbon": 1.30, "magnification": 500},
    {"label": "fe_2pct", "carbon": 2.00, "magnification": 500},
    {"label": "white_cast_iron_3pct", "carbon": 3.00, "magnification": 200},
    {"label": "white_cast_iron_4pct", "carbon": 4.00, "magnification": 200},
    {"label": "white_cast_iron_eutectic", "carbon": 4.30, "magnification": 200},
    {"label": "white_cast_iron_5_5pct", "carbon": 5.50, "magnification": 200},
]


def _expected_fractions_for_carbon(c_wt: float) -> dict[str, float]:
    """Lever-rule expectation for the simple Fe-C cases used by the
    atlas — used by the ``--with-metrics`` mode."""
    eutectoid = 0.77
    solubility = 0.02
    if c_wt < solubility:
        return {"FERRITE": 1.0}
    if c_wt < eutectoid:
        pearlite = (c_wt - solubility) / (eutectoid - solubility)
        return {
            "FERRITE": float(max(0.0, 1.0 - pearlite)),
            "PEARLITE": float(min(1.0, pearlite)),
        }
    if c_wt <= 2.14:
        cementite = (c_wt - eutectoid) / (6.67 - eutectoid)
        return {
            "PEARLITE": float(max(0.0, 1.0 - cementite)),
            "CEMENTITE": float(min(1.0, cementite)),
        }
    return {"LEDEBURITE": 1.0}


def _save_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        Image.fromarray(image, mode="L").save(path)
    else:
        Image.fromarray(image, mode="RGB").save(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate-ferro-micro-atlas",
        description="Generate the ferro-micro atlas of 18 Fe-C microstructures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the PNG sequence and manifest.",
    )
    parser.add_argument("--width", type=int, default=512, help="Image width, px.")
    parser.add_argument("--height", type=int, default=512, help="Image height, px.")
    parser.add_argument(
        "--color-mode",
        default="grayscale_nital",
        choices=(
            "grayscale_nital",
            "nital_warm",
            "dic_polarized",
            "tint_etch_blue_yellow",
        ),
        help="Output colour palette for every sample.",
    )
    parser.add_argument("--seed", type=int, default=4242, help="Base random seed.")
    parser.add_argument(
        "--with-metrics",
        action="store_true",
        help="Compute lever-rule phase-fraction error per sample.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    for index, recipe in enumerate(ATLAS_RECIPES):
        out_path = args.output_dir / f"{index:02d}_{recipe['label']}.png"
        sample = fm.generate(
            carbon=recipe["carbon"],
            width=args.width,
            height=args.height,
            magnification=recipe["magnification"],
            color_mode=args.color_mode,
            seed=args.seed + index,
            return_info=args.with_metrics,
        )
        _save_image(sample.image, out_path)
        sample_record: dict = {
            "index": index,
            "label": recipe["label"],
            "carbon_wt": recipe["carbon"],
            "magnification": recipe["magnification"],
            "file": out_path.name,
        }
        if args.with_metrics and sample.info is not None:
            expected = _expected_fractions_for_carbon(recipe["carbon"])
            err = phase_fraction_error(
                phase_masks=sample.phase_masks,
                expected_fractions=expected,
            )
            sample_record["phase_fractions"] = sample.info["phases"]
            sample_record["expected_fractions"] = expected
            sample_record["max_relative_error_pct"] = err["max_relative_error_pct"]
        samples.append(sample_record)
        print(
            f"  [{index + 1}/{len(ATLAS_RECIPES)}] {out_path.name}"
            f" (C={recipe['carbon']:.2f})"
        )

    manifest = {
        "version": 1,
        "color_mode": args.color_mode,
        "image_size": [args.height, args.width],
        "samples": samples,
    }
    manifest_path = args.output_dir / "atlas_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
