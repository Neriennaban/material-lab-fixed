from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.calphad.db_manager import CALPHAD_SUPPORTED_SYSTEMS, resolve_database_reference, validate_database_reference
from core.contracts_v2 import ThermoBackendConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CALPHAD setup for V2")
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "profiles" / "calphad_profile_v2.json",
        help="Path to CALPHAD profile JSON",
    )
    parser.add_argument("--tdb-dir", type=Path, default=None, help="Optional override dir with TDB files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thermo = ThermoBackendConfig(db_profile_path=str(args.profile), strict_mode=True)
    ok = True
    for system in CALPHAD_SUPPORTED_SYSTEMS:
        try:
            ref = resolve_database_reference(system=system, thermo=thermo, profile_path=args.profile, tdb_dir=args.tdb_dir)
            validate_database_reference(ref)
            print(f"[OK] {system}: {ref.path}")
        except Exception as exc:
            ok = False
            print(f"[FAIL] {system}: {exc}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
