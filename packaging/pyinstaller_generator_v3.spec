# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

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


datas = existing_data_dirs()
binaries = []
hiddenimports = []
tmp_ret = collect_all("pyqtgraph")
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]


a = Analysis(
    [str(ROOT / "run_generator_app_v3.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    name="MetallographyGeneratorV3",
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
    name="MetallographyGeneratorV3",
)
