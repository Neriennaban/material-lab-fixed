# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def existing_data_dirs() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for name in (
        "presets_v3",
        "profiles_v3",
        "presets",
        "profiles",
        "datasets",
        "docs",
        "examples",
        "export",
        "keys",
    ):
        path = ROOT / name
        if path.exists():
            out.append((str(path), name))
    return out


a = Analysis(
    [str(ROOT / "run_app_v2.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=existing_data_dirs(),
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VirtualMicroscopeV2",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="VirtualMicroscopeV2",
)
