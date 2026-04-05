from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ui_qt.sample_factory_window_v3 import launch_sample_factory_app_v3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metallography generator V3 launcher")
    parser.add_argument(
        "--presets-dir",
        type=Path,
        default=Path("presets_v3"),
        help="Directory with V3 presets",
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=Path("profiles_v3"),
        help="Directory with V3 profiles",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(
        launch_sample_factory_app_v3(
            presets_dir=args.presets_dir,
            profiles_dir=args.profiles_dir,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
