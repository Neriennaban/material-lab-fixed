from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from PySide6.QtWidgets import QApplication

from runtime_patches import apply_runtime_patches, apply_ui_runtime_patches

apply_runtime_patches()
apply_ui_runtime_patches()

from ui_qt.microscope_window import MicroscopeWindow
from ui_qt.spinbox_wheel_filter import SpinBoxWheelFilter


DEFAULT_SAMPLES_DIR = Path("")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Virtual microscope V3 launcher")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        help="Directory with exported lab sample packages",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    app = QApplication.instance() or QApplication(
        sys.argv if argv is None else list(argv)
    )
    wheel_filter = SpinBoxWheelFilter(app)
    app.installEventFilter(wheel_filter)
    window = MicroscopeWindow(samples_dir=args.samples_dir)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
