"""C5 — blind evaluation script for ferro-micro.

Generates a shuffled mix of synthetic samples (rendered via the
ferro-micro API) and reference photographs (from a directory the
expert points at), drops them into an output folder with anonymised
filenames, and writes a JSON answer key. The expert grades each image
without knowing which is real and which is generated; the answer key
lets the operator score the run afterwards.

Usage
-----
python -m scripts.blind_eval_ferro_micro --references datasets/fe_c_references/ \\
    --output-dir /tmp/blind_eval/ --count 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import shutil
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.metallography_v3 import ferro_micro_api as fm  # noqa: E402

_ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _collect_reference_images(reference_dir: Path) -> list[Path]:
    return sorted(
        [p for p in reference_dir.rglob("*") if p.suffix.lower() in _ALLOWED_EXT]
    )


def _generate_synthetic_samples(
    *,
    count: int,
    width: int,
    height: int,
    color_mode: str,
    seed: int,
) -> list[tuple[str, np.ndarray]]:
    """Render ``count`` synthetic samples spanning the carbon range."""
    rng = np.random.default_rng(int(seed))
    samples: list[tuple[str, np.ndarray]] = []
    for index in range(int(count)):
        carbon = float(rng.uniform(0.05, 5.5))
        magnification = int(rng.choice([100, 200, 400, 500]))
        sample = fm.generate(
            carbon=carbon,
            width=width,
            height=height,
            magnification=magnification,
            color_mode=color_mode,
            seed=seed + 91 * index + 17,
        )
        label = f"synth_C{int(round(carbon * 100)):04d}_mag{magnification}"
        samples.append((label, sample.image))
    return samples


def _shuffled_anonymised_pairs(
    items: list[tuple[str, str, np.ndarray | None, Path | None]],
    *,
    seed: int,
) -> list[dict]:
    """Shuffle ``items`` deterministically and assign anonymised IDs.

    Each tuple is ``(category, label, image_array_or_none, source_path_or_none)``.
    Either ``image_array`` or ``source_path`` must be provided.
    """
    rng = np.random.default_rng(int(seed))
    indices = list(range(len(items)))
    rng.shuffle(indices)
    pairs: list[dict] = []
    for new_index, old_index in enumerate(indices):
        category, label, array, source = items[old_index]
        anon_id = secrets.token_hex(4)
        pairs.append(
            {
                "anon_id": anon_id,
                "order_index": new_index,
                "category": category,
                "label": label,
                "image": array,
                "source": source,
            }
        )
    return pairs


def _save_pair(pair: dict, output_dir: Path) -> Path:
    out_path = output_dir / f"{pair['order_index']:03d}_{pair['anon_id']}.png"
    if pair["image"] is not None:
        arr = pair["image"]
        if arr.ndim == 2:
            Image.fromarray(arr, mode="L").save(out_path)
        else:
            Image.fromarray(arr, mode="RGB").save(out_path)
    elif pair["source"] is not None:
        shutil.copyfile(pair["source"], out_path.with_suffix(pair["source"].suffix))
        return out_path.with_suffix(pair["source"].suffix)
    else:
        raise ValueError("pair has no image or source")
    return out_path


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blind-eval-ferro-micro",
        description="Build a shuffled real/synthetic blind-eval set.",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=None,
        help="Directory of real reference micrographs (recursive).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to drop the shuffled images and answer key.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of synthetic samples to generate.",
    )
    parser.add_argument(
        "--width", type=int, default=512, help="Synthetic image width, px."
    )
    parser.add_argument(
        "--height", type=int, default=512, help="Synthetic image height, px."
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
    )
    parser.add_argument("--seed", type=int, default=20260408)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items: list[tuple[str, str, np.ndarray | None, Path | None]] = []

    # Real reference images.
    if args.references is not None and args.references.exists():
        reference_paths = _collect_reference_images(args.references)
        for path in reference_paths:
            items.append(("real", path.stem, None, path))
        print(f"Collected {len(reference_paths)} real reference images.")
    else:
        print("No --references directory provided; generating synthetic-only set.")

    # Synthetic samples.
    synthetic = _generate_synthetic_samples(
        count=args.count,
        width=args.width,
        height=args.height,
        color_mode=args.color_mode,
        seed=args.seed,
    )
    for label, array in synthetic:
        items.append(("synthetic", label, array, None))

    if not items:
        print("Nothing to evaluate — provide --references or non-zero --count.")
        return 2

    pairs = _shuffled_anonymised_pairs(items, seed=args.seed)
    answer_key: list[dict] = []
    for pair in pairs:
        saved_path = _save_pair(pair, args.output_dir)
        answer_key.append(
            {
                "anon_id": pair["anon_id"],
                "order_index": pair["order_index"],
                "category": pair["category"],
                "label": pair["label"],
                "file": saved_path.name,
                "sha256": _hash_file(saved_path),
            }
        )

    answer_path = args.output_dir / "answer_key.json"
    answer_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "color_mode": args.color_mode,
                "image_size": [args.height, args.width],
                "items": answer_key,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(answer_key)} samples to {args.output_dir}")
    print(f"Answer key: {answer_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
