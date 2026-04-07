"""ferro-micro command-line interface (TZ §9).

Thin argparse wrapper around ``core.metallography_v3.ferro_micro_api``
that exposes the user-facing parameter names from §9 of the TZ:
``--carbon``, ``--cooling-rate``, ``--magnification``, ``--etchant``,
``-o``, plus an ``--atlas`` mode for batch generation across a carbon
range.

Examples
--------

# Equilibrium structure, 0.4 % C
python -m scripts.ferro_micro_cli --carbon 0.4 --width 1024 --height 1024 -o steel_040.png

# Quench: 0.45 % C, water, 500x
python -m scripts.ferro_micro_cli --carbon 0.45 --austenitization-temp 860 \\
    --cooling-rate 200 --magnification 500 --etchant nital -o quenched.png

# Hypoeutectic cast iron, 100x with surface defects
python -m scripts.ferro_micro_cli --carbon 3.5 --magnification 100 \\
    --color-mode grayscale_nital -o cast_iron.png

# Batch generation across the carbon range
python -m scripts.ferro_micro_cli --atlas --carbon-range 0.0 6.67 0.5 \\
    --output-dir atlas/

# Custom thermal program from a JSON file
python -m scripts.ferro_micro_cli --carbon 0.8 --thermal-program program.json \\
    -o annealed.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

# When invoked as ``python -m scripts.ferro_micro_cli`` the package
# import path already contains the repo root. When invoked as a script
# (``python scripts/ferro_micro_cli.py``) we add the repo root manually.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.metallography_v3 import ferro_micro_api as fm  # noqa: E402


def _save_image(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        Image.fromarray(image, mode="L").save(output_path)
    elif image.ndim == 3 and image.shape[2] == 3:
        Image.fromarray(image, mode="RGB").save(output_path)
    else:
        raise ValueError(f"unsupported image shape: {image.shape}")


def _load_thermal_program(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "points" in payload:
        return list(payload["points"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"thermal program file must be a list or contain 'points': {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ferro-micro",
        description="Generate Fe-C metallography microstructures (TZ §9).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--carbon",
        type=float,
        help="Carbon content, weight percent (0..6.67). Required unless --atlas.",
    )
    parser.add_argument("--width", type=int, default=1024, help="Image width, px.")
    parser.add_argument("--height", type=int, default=1024, help="Image height, px.")
    parser.add_argument(
        "--cooling-rate",
        type=float,
        default=1.0,
        help="Cooling rate °C/s. Drives the auto thermal program.",
    )
    parser.add_argument(
        "--austenitization-temp",
        type=float,
        default=None,
        help="Austenitisation temperature, °C (default 870).",
    )
    parser.add_argument(
        "--holding-time",
        type=float,
        default=60.0,
        help="Hold time at austenitisation, minutes.",
    )
    parser.add_argument(
        "--magnification",
        type=int,
        default=200,
        help="Microscope magnification (100 / 200 / 500 / 1000).",
    )
    parser.add_argument(
        "--etchant",
        default="nital",
        help="Etchant: nital, picral, klemm_1, le_perra, beraha_iii, …",
    )
    parser.add_argument(
        "--color-mode",
        default="grayscale_nital",
        choices=(
            "grayscale_nital",
            "nital_warm",
            "dic_polarized",
            "tint_etch_blue_yellow",
        ),
        help="Output colour palette.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--thermal-program",
        type=Path,
        default=None,
        help="Path to a JSON file with a custom thermal program (overrides --cooling-rate).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Required unless --atlas.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print phase fractions, ASTM grain size and hardness alongside the render.",
    )

    parser.add_argument(
        "--atlas",
        action="store_true",
        help="Batch mode: render a sequence of carbons.",
    )
    parser.add_argument(
        "--carbon-range",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "STEP"),
        default=None,
        help="With --atlas: sweep range, e.g. 0.0 6.67 0.5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="With --atlas: directory to write the PNG sequence.",
    )
    return parser


def _run_atlas(args: argparse.Namespace) -> int:
    if args.carbon_range is None:
        print("--atlas requires --carbon-range START STOP STEP", file=sys.stderr)
        return 2
    if args.output_dir is None:
        print("--atlas requires --output-dir", file=sys.stderr)
        return 2
    start, stop, step = args.carbon_range
    if step <= 0.0:
        print("--carbon-range STEP must be > 0", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    carbons: list[float] = []
    c = float(start)
    while c <= float(stop) + 1e-9:
        carbons.append(round(c, 4))
        c += float(step)

    manifest: list[dict] = []
    for idx, carbon in enumerate(carbons):
        out_path = args.output_dir / f"sample_{idx:03d}_C{int(round(carbon * 100)):04d}.png"
        sample = fm.generate(
            carbon=carbon,
            width=args.width,
            height=args.height,
            cooling_rate=args.cooling_rate,
            austenitization_temp=args.austenitization_temp,
            holding_time=args.holding_time,
            magnification=args.magnification,
            etchant=args.etchant,
            color_mode=args.color_mode,
            seed=args.seed + idx,
            return_info=True,
        )
        _save_image(sample.image, out_path)
        manifest.append(
            {
                "index": idx,
                "carbon_wt": carbon,
                "path": out_path.name,
                "phases": sample.info["phases"],
            }
        )
        print(f"  [{idx + 1}/{len(carbons)}] {out_path.name} (C={carbon:.2f})")

    manifest_path = args.output_dir / "atlas.json"
    manifest_path.write_text(
        json.dumps(
            {
                "carbon_range": list(args.carbon_range),
                "magnification": args.magnification,
                "color_mode": args.color_mode,
                "samples": manifest,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Atlas manifest written to {manifest_path}")
    return 0


def _run_single(args: argparse.Namespace) -> int:
    if args.carbon is None:
        print("--carbon is required (use --atlas for batch mode)", file=sys.stderr)
        return 2
    if args.output is None:
        print("-o/--output is required for single-render mode", file=sys.stderr)
        return 2

    thermal_program = (
        _load_thermal_program(args.thermal_program)
        if args.thermal_program is not None
        else None
    )

    sample = fm.generate(
        carbon=args.carbon,
        width=args.width,
        height=args.height,
        cooling_rate=args.cooling_rate,
        austenitization_temp=args.austenitization_temp,
        holding_time=args.holding_time,
        magnification=args.magnification,
        etchant=args.etchant,
        color_mode=args.color_mode,
        seed=args.seed,
        thermal_program=thermal_program,
        return_info=args.info,
    )
    _save_image(sample.image, args.output)
    print(f"Wrote {args.output} ({sample.image.shape[1]}x{sample.image.shape[0]} px)")
    if args.info and sample.info is not None:
        print("Phases:")
        for phase, fraction in sample.info["phases"].items():
            print(f"  {phase}: {fraction:.3f}")
        if sample.info.get("grain_size_astm") is not None:
            print(f"ASTM grain size: {sample.info['grain_size_astm']}")
        if sample.info.get("hardness_hv") is not None:
            print(f"Hardness HV: {sample.info['hardness_hv']:.0f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.atlas:
        return _run_atlas(args)
    return _run_single(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
