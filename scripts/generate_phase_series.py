from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.generator_phase_map import generate_phase_stage_structure, supported_stages
from export.export_images import save_image
from export.export_tables import save_json


def parse_composition(raw: str) -> dict[str, float]:
    text = raw.strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return {str(k): float(v) for k, v in dict(payload).items()}
    except json.JSONDecodeError:
        out: dict[str, float] = {}
        for part in text.split(","):
            token = part.strip()
            if not token or "=" not in token:
                continue
            key, value = token.split("=", maxsplit=1)
            out[key.strip()] = float(value.strip())
        return out


def generate_series(
    output_dir: Path,
    system: str,
    composition: dict[str, float],
    size: tuple[int, int],
    seed: int,
    temperature_c: float,
    cooling_mode: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = supported_stages(system)
    if not stages:
        raise ValueError(f"Unsupported system: {system}")

    saved: list[Path] = []
    for idx, stage in enumerate(stages):
        result = generate_phase_stage_structure(
            size=size,
            seed=seed + idx,
            system=system,
            composition=composition,
            stage=stage,
            temperature_c=temperature_c,
            cooling_mode=cooling_mode,
        )
        image_path = output_dir / f"{system}_{stage}.png"
        save_image(result["image"], image_path)
        saved.append(image_path)

        meta_path = output_dir / f"{system}_{stage}.json"
        save_json(result["metadata"], meta_path)
        saved.append(meta_path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate full stage series for alloy phase systems")
    parser.add_argument("--system", required=True, help="fe-c | fe-si | al-si | cu-zn | al-cu-mg")
    parser.add_argument("--composition", default="{}", help='JSON, e.g. {"C":0.8,"Fe":99.2}')
    parser.add_argument("--output-dir", type=Path, default=Path("examples") / "phase_series")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=4000)
    parser.add_argument("--temperature-c", type=float, default=20.0)
    parser.add_argument("--cooling-mode", default="equilibrium")
    args = parser.parse_args()

    composition = parse_composition(args.composition)
    files = generate_series(
        output_dir=args.output_dir,
        system=args.system,
        composition=composition,
        size=(args.height, args.width),
        seed=args.seed,
        temperature_c=args.temperature_c,
        cooling_mode=args.cooling_mode,
    )
    print(f"Generated {len(files)} files in {args.output_dir}")


if __name__ == "__main__":
    main()
