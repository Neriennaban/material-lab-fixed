from __future__ import annotations

import sys
from pathlib import Path


def get_app_base_dir() -> Path:
    """Return the application base directory for dev and frozen builds."""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(str(meipass)).resolve()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent
